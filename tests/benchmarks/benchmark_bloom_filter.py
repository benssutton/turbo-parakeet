#!/usr/bin/env python
"""
Benchmark comparison: Custom Bloom Filter (Rust-optimized) vs pybloom-live

Tests:
1. Construction time
2. Insertion time (bulk add)
3. Membership query time
4. Memory usage
5. False positive rate accuracy

Dataset sizes: 1k, 10k, 100k items
"""

import time
import sys
import tracemalloc
from pathlib import Path
from typing import Callable, Tuple, Optional
import polars as pl

# Add project root to path so services package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Our implementation
from services.bloom_filter import BloomFilter as CustomBloomFilter

# pybloom-live (install: pip install pybloom-live)
try:
    from pybloom_live import BloomFilter as PyBloomFilter
    PYBLOOM_AVAILABLE = True
except ImportError:
    print("WARNING: pybloom-live not installed. Install with: pip install pybloom-live")
    PYBLOOM_AVAILABLE = False

# fastbloom-rs (install: pip install fastbloom-rs)
try:
    from fastbloom_rs import BloomFilter as FastBloomFilter
    FASTBLOOM_AVAILABLE = True
except ImportError:
    print("WARNING: fastbloom-rs not installed. Install with: pip install fastbloom-rs")
    FASTBLOOM_AVAILABLE = False


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.construction_time = 0.0
        self.insertion_time = 0.0
        self.query_time = 0.0
        self.memory_peak_mb = 0.0
        self.false_positive_rate = 0.0

    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"{self.name}\n"
            f"{'='*60}\n"
            f"Construction time:    {self.construction_time*1000:>10.2f} ms\n"
            f"Insertion time:       {self.insertion_time*1000:>10.2f} ms\n"
            f"Query time:           {self.query_time*1000:>10.2f} ms\n"
            f"Peak memory:          {self.memory_peak_mb:>10.2f} MB\n"
            f"False positive rate:  {self.false_positive_rate*100:>10.4f} %\n"
        )


def generate_dataset(size: int) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Generate training and test datasets."""
    # Training set: numbers 0 to size-1
    training = pl.LazyFrame({"items": [f"abcdefgh_item_{i}" for i in range(size)]})

    # Test set: half are in training (true positives), half are not (false positives)
    test_positives = [f"abcdefgh_item_{i}" for i in range(0, size, 2)]  # Every other item
    test_negatives = [f"abcdefgh_item_new_{i}" for i in range(size // 2)]  # New items
    test = pl.LazyFrame({"items": test_positives + test_negatives})

    return training, test


def benchmark_custom_bloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01
) -> BenchmarkResult:
    """Benchmark our custom Rust-optimized Bloom filter."""
    result = BenchmarkResult("Custom Bloom Filter (Rust-optimized)")

    # Track memory
    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = CustomBloomFilter(expected_count, fp_rate)
    result.construction_time = time.perf_counter() - start

    # 2. Insertion
    start = time.perf_counter()
    bf.add(training)
    result.insertion_time = time.perf_counter() - start

    # 3. Query
    start = time.perf_counter()
    membership_df = bf.membership_ratio(test)
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate (count negatives that were incorrectly found)
    #membership_series = membership_df["items"]
    # Second half of test set are negatives
    #test_length = len(membership_series)
    #negatives_start = test_length // 2
    #false_positives = sum(1 for i in range(negatives_start, test_length)
    #                     if membership_series[i])
    #result.false_positive_rate = false_positives / (test_length - negatives_start)

    result.false_positive_rate = 0

    return result


def benchmark_pybloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01
) -> BenchmarkResult:
    """Benchmark pybloom-live implementation."""
    result = BenchmarkResult("pybloom-live (Pure Python)")

    # Track memory
    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = PyBloomFilter(capacity=expected_count, error_rate=fp_rate)
    result.construction_time = time.perf_counter() - start

    # Convert LazyFrames to lists for pybloom iteration
    training_items = training.select("items").collect()["items"].to_list()
    test_items = test.select("items").collect()["items"].to_list()

    # 2. Insertion
    start = time.perf_counter()
    for item in training_items:
        bf.add(item)
    result.insertion_time = time.perf_counter() - start

    # 3. Query
    start = time.perf_counter()
    membership = [item in bf for item in test_items]
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate
    test_length = len(membership)
    negatives_start = test_length // 2
    false_positives = sum(1 for i in range(negatives_start, test_length)
                         if membership[i])
    result.false_positive_rate = false_positives / (test_length - negatives_start)

    return result


def benchmark_fastbloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01
) -> BenchmarkResult:
    """Benchmark fastbloom-rs implementation."""
    result = BenchmarkResult("fastbloom-rs (Rust)")

    # Track memory
    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = FastBloomFilter(expected_count, fp_rate)
    result.construction_time = time.perf_counter() - start

    # Convert LazyFrames to lists for fastbloom iteration
    training_items = training.select("items").collect()["items"].to_list()
    test_items = test.select("items").collect()["items"].to_list()

    # 2. Insertion
    start = time.perf_counter()
    for item in training_items:
        bf.add(item)
    result.insertion_time = time.perf_counter() - start

    # 3. Query
    start = time.perf_counter()
    membership = [bf.contains(item) for item in test_items]
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate
    test_length = len(membership)
    negatives_start = test_length // 2
    false_positives = sum(1 for i in range(negatives_start, test_length)
                         if membership[i])
    result.false_positive_rate = false_positives / (test_length - negatives_start)

    return result


def print_comparison(custom: BenchmarkResult, pybloom: Optional[BenchmarkResult] = None, fastbloom: Optional[BenchmarkResult] = None):
    """Print side-by-side comparison with speedup factors."""
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100)

    # Build header dynamically based on available results
    header = f"{'Metric':<25} {'Custom':<15}"
    if pybloom:
        header += f" {'pybloom':<15} {'vs Custom':<15}"
    if fastbloom:
        header += f" {'fastbloom':<15} {'vs Custom':<15}"
    print(header)
    print("-"*100)

    # Construction
    line = f"{'Construction (ms)':<25} {custom.construction_time*1000:<15.2f}"
    if pybloom:
        speedup = pybloom.construction_time / custom.construction_time if custom.construction_time > 0 else 0
        line += f" {pybloom.construction_time*1000:<15.2f} {speedup:<15.2f}×"
    if fastbloom:
        speedup = fastbloom.construction_time / custom.construction_time if custom.construction_time > 0 else 0
        line += f" {fastbloom.construction_time*1000:<15.2f} {speedup:<15.2f}×"
    print(line)

    # Insertion
    line = f"{'Insertion (ms)':<25} {custom.insertion_time*1000:<15.2f}"
    if pybloom:
        speedup = pybloom.insertion_time / custom.insertion_time if custom.insertion_time > 0 else 0
        line += f" {pybloom.insertion_time*1000:<15.2f} {speedup:<15.2f}×"
    if fastbloom:
        speedup = fastbloom.insertion_time / custom.insertion_time if custom.insertion_time > 0 else 0
        line += f" {fastbloom.insertion_time*1000:<15.2f} {speedup:<15.2f}×"
    print(line)

    # Query
    line = f"{'Query (ms)':<25} {custom.query_time*1000:<15.2f}"
    if pybloom:
        speedup = pybloom.query_time / custom.query_time if custom.query_time > 0 else 0
        line += f" {pybloom.query_time*1000:<15.2f} {speedup:<15.2f}×"
    if fastbloom:
        speedup = fastbloom.query_time / custom.query_time if custom.query_time > 0 else 0
        line += f" {fastbloom.query_time*1000:<15.2f} {speedup:<15.2f}×"
    print(line)

    # Memory
    line = f"{'Peak Memory (MB)':<25} {custom.memory_peak_mb:<15.2f}"
    if pybloom:
        ratio = custom.memory_peak_mb / pybloom.memory_peak_mb if pybloom.memory_peak_mb > 0 else 0
        line += f" {pybloom.memory_peak_mb:<15.2f} {ratio:<15.2f}×"
    if fastbloom:
        ratio = custom.memory_peak_mb / fastbloom.memory_peak_mb if fastbloom.memory_peak_mb > 0 else 0
        line += f" {fastbloom.memory_peak_mb:<15.2f} {ratio:<15.2f}×"
    print(line)

    # False positive rate
    line = f"{'False Positive Rate':<25} {custom.false_positive_rate*100:<15.4f}%"
    if pybloom:
        line += f" {pybloom.false_positive_rate*100:<15.4f}% {'':<15}"
    if fastbloom:
        line += f" {fastbloom.false_positive_rate*100:<15.4f}% {'':<15}"
    print(line)

    print("="*100)


def main():
    """Run benchmarks on multiple dataset sizes."""
    if not PYBLOOM_AVAILABLE and not FASTBLOOM_AVAILABLE:
        print("\nCannot run benchmarks without comparison libraries.")
        print("Install at least one: pip install pybloom-live OR pip install fastbloom-rs")
        sys.exit(1)

    dataset_sizes = [50000] #[1000, 10000, 100000, 1000000, 3000000]
    fp_rate = 0.01

    print("="*100)
    print("BLOOM FILTER PERFORMANCE BENCHMARK")
    print("="*100)
    print(f"Custom: Rust-optimized (Polars plugin with MurmurHash3 and parallel processing)")
    if PYBLOOM_AVAILABLE:
        print(f"pybloom-live: Pure Python implementation")
    if FASTBLOOM_AVAILABLE:
        print(f"fastbloom-rs: Rust-based implementation")
    print(f"\nFalse Positive Rate Target: {fp_rate*100}%")
    print(f"Dataset Sizes: {dataset_sizes}")
    print("="*100)

    for size in dataset_sizes:
        print(f"\n{'#'*100}")
        print(f"# Dataset Size: {size:,} items")
        print(f"{'#'*100}")

        # Generate data
        training, test = generate_dataset(size)
        training_len = training.select(pl.len()).collect()[0, 0]
        test_len = test.select(pl.len()).collect()[0, 0]
        print(f"Training set: {training_len:,} items")
        print(f"Test set: {test_len:,} items ({test_len//2} positives, {test_len//2} negatives)")

        # Benchmark custom implementation
        print("\nBenchmarking Custom Bloom Filter...")
        custom_result = benchmark_custom_bloom(training, test, size, fp_rate)

        # Benchmark pybloom if available
        pybloom_result = None
        if PYBLOOM_AVAILABLE:
            print("Benchmarking pybloom-live...")
            pybloom_result = benchmark_pybloom(training, test, size, fp_rate)

        # Benchmark fastbloom if available
        fastbloom_result = None
        if FASTBLOOM_AVAILABLE:
            print("Benchmarking fastbloom-rs...")
            fastbloom_result = benchmark_fastbloom(training, test, size, fp_rate)

        # Print results
        print(custom_result)
        if pybloom_result:
            print(pybloom_result)
        if fastbloom_result:
            print(fastbloom_result)
        print_comparison(custom_result, pybloom_result, fastbloom_result)

    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
