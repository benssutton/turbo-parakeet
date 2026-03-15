#!/usr/bin/env python
"""
Benchmark comparison: Custom Bloom Filter (Rust-optimized) vs pybloom-live

Tests:
1. Construction time
2. Insertion time (bulk add)
3. Membership query time across n columns
4. Memory usage
5. False positive rate accuracy

Dataset sizes: configurable
n columns: all checked against one bloom filter in a single call (custom) vs column loop (others)
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
PyBloomFilter = None
try:
    from pybloom_live import BloomFilter as PyBloomFilter
    PYBLOOM_AVAILABLE = True
except ImportError:
    print("WARNING: pybloom-live not installed. Install with: pip install pybloom-live")
    PYBLOOM_AVAILABLE = False

# fastbloom-rs (install: pip install fastbloom-rs)
FastBloomFilter = None
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


def generate_dataset(size: int, n: int) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Generate training and test datasets.

    Training: single column 'items' used to build the bloom filter.
    Test: n columns, each with size//2 true positives and size//2 negatives,
          all checked against the same bloom filter.
    """
    training = pl.LazyFrame({"items": [f"abcdefgh_item_{i}" for i in range(size)]})

    test_data = {}
    for col_idx in range(n):
        test_positives = [f"abcdefgh_item_{i}" for i in range(0, size, 2)]
        test_negatives = [f"abcdefgh_item_new_{col_idx}_{i}" for i in range(size // 2)]
        test_data[f"col_{col_idx}"] = test_positives + test_negatives

    test = pl.LazyFrame(test_data)
    return training, test


def benchmark_custom_bloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01,
) -> BenchmarkResult:
    """
    Benchmark our custom Rust-optimized Bloom filter.
    Passes all n test columns in a single membership_ratio call.
    """
    result = BenchmarkResult("Custom Bloom Filter (Rust-optimized, batch n columns)")

    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = CustomBloomFilter(expected_count, fp_rate)
    result.construction_time = time.perf_counter() - start

    # 2. Insertion (train on first/only training column)
    start = time.perf_counter()
    bf.add(training)
    result.insertion_time = time.perf_counter() - start

    # 3. Query — all n columns in a single call
    test_df = test.collect()
    start = time.perf_counter()
    ratio_df = bf.membership_ratio(test_df)
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate — average across columns (negatives are second half of each col)
    col_len = len(test_df)

    fp_rates = []
    for row in ratio_df.iter_rows(named=True):
        # ratio_all = found / total; back out found count
        found_all = round(row["ratio_all"] * col_len)
        # negatives occupy second half — estimate via total found minus known positives
        # positives = first half (size//2 items, every other from range(size))
        n_positives = col_len // 2
        n_negatives = col_len - n_positives
        false_pos = max(0, found_all - n_positives)
        fp_rates.append(false_pos / n_negatives if n_negatives > 0 else 0.0)

    result.false_positive_rate = sum(fp_rates) / len(fp_rates) if fp_rates else 0.0

    return result


def benchmark_pybloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01,
) -> BenchmarkResult:
    """Benchmark pybloom-live implementation, iterating n columns one by one."""
    result = BenchmarkResult("pybloom-live (Pure Python, column loop)")

    assert PyBloomFilter is not None
    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = PyBloomFilter(capacity=expected_count, error_rate=fp_rate)
    result.construction_time = time.perf_counter() - start

    # Convert training to list for pybloom
    training_items = training.select("items").collect()["items"].to_list()
    test_df = test.collect()

    # 2. Insertion
    start = time.perf_counter()
    for item in training_items:
        bf.add(item)
    result.insertion_time = time.perf_counter() - start

    # 3. Query — iterate each column separately
    start = time.perf_counter()
    memberships = {}
    for col in test_df.columns:
        col_items = test_df[col].to_list()
        memberships[col] = [item in bf for item in col_items]
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate — average across columns
    col_len = len(test_df)
    n_positives = col_len // 2
    n_negatives = col_len - n_positives
    fp_rates = []
    for col, membership in memberships.items():
        false_pos = sum(1 for i in range(n_positives, col_len) if membership[i])
        fp_rates.append(false_pos / n_negatives if n_negatives > 0 else 0.0)
    result.false_positive_rate = sum(fp_rates) / len(fp_rates) if fp_rates else 0.0

    return result


def benchmark_fastbloom(
    training: pl.LazyFrame,
    test: pl.LazyFrame,
    expected_count: int,
    fp_rate: float = 0.01,
) -> BenchmarkResult:
    """Benchmark fastbloom-rs implementation, iterating n columns one by one."""
    result = BenchmarkResult("fastbloom-rs (Rust, column loop)")

    assert FastBloomFilter is not None
    tracemalloc.start()

    # 1. Construction
    start = time.perf_counter()
    bf = FastBloomFilter(expected_count, fp_rate)
    result.construction_time = time.perf_counter() - start

    training_items = training.select("items").collect()["items"].to_list()
    test_df = test.collect()

    # 2. Insertion
    start = time.perf_counter()
    for item in training_items:
        bf.add(item)
    result.insertion_time = time.perf_counter() - start

    # 3. Query — iterate each column separately
    start = time.perf_counter()
    memberships = {}
    for col in test_df.columns:
        col_items = test_df[col].to_list()
        memberships[col] = [bf.contains(item) for item in col_items]
    result.query_time = time.perf_counter() - start

    # 4. Memory
    _, peak = tracemalloc.get_traced_memory()
    result.memory_peak_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # 5. False positive rate — average across columns
    col_len = len(test_df)
    n_positives = col_len // 2
    n_negatives = col_len - n_positives
    fp_rates = []
    for col, membership in memberships.items():
        false_pos = sum(1 for i in range(n_positives, col_len) if membership[i])
        fp_rates.append(false_pos / n_negatives if n_negatives > 0 else 0.0)
    result.false_positive_rate = sum(fp_rates) / len(fp_rates) if fp_rates else 0.0

    return result


def print_comparison(
    custom: BenchmarkResult,
    pybloom: Optional[BenchmarkResult] = None,
    fastbloom: Optional[BenchmarkResult] = None,
):
    """Print side-by-side comparison with speedup factors."""
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100)

    header = f"{'Metric':<25} {'Custom':<15}"
    if pybloom:
        header += f" {'pybloom':<15} {'vs Custom':<15}"
    if fastbloom:
        header += f" {'fastbloom':<15} {'vs Custom':<15}"
    print(header)
    print("-"*100)

    def row(label, attr):
        val = getattr(custom, attr)
        line = f"{label:<25} {val*1000:<15.2f}"
        if pybloom:
            pv = getattr(pybloom, attr)
            speedup = pv / val if val > 0 else 0
            line += f" {pv*1000:<15.2f} {speedup:<15.2f}×"
        if fastbloom:
            fv = getattr(fastbloom, attr)
            speedup = fv / val if val > 0 else 0
            line += f" {fv*1000:<15.2f} {speedup:<15.2f}×"
        return line

    print(row("Construction (ms)", "construction_time"))
    print(row("Insertion (ms)", "insertion_time"))
    print(row("Query (ms)", "query_time"))

    # Memory (lower custom = smaller ratio shown as custom/other)
    line = f"{'Peak Memory (MB)':<25} {custom.memory_peak_mb:<15.2f}"
    if pybloom:
        ratio = custom.memory_peak_mb / pybloom.memory_peak_mb if pybloom.memory_peak_mb > 0 else 0
        line += f" {pybloom.memory_peak_mb:<15.2f} {ratio:<15.2f}×"
    if fastbloom:
        ratio = custom.memory_peak_mb / fastbloom.memory_peak_mb if fastbloom.memory_peak_mb > 0 else 0
        line += f" {fastbloom.memory_peak_mb:<15.2f} {ratio:<15.2f}×"
    print(line)

    line = f"{'False Positive Rate':<25} {custom.false_positive_rate*100:<15.4f}%"
    if pybloom:
        line += f" {pybloom.false_positive_rate*100:<15.4f}% {'':<15}"
    if fastbloom:
        line += f" {fastbloom.false_positive_rate*100:<15.4f}% {'':<15}"
    print(line)

    print("="*100)


def main():
    """Run benchmarks on multiple dataset sizes."""
    n_columns = 50  # number of test columns to check per bloom filter

    if not PYBLOOM_AVAILABLE and not FASTBLOOM_AVAILABLE:
        print("\nCannot run benchmarks without comparison libraries.")
        print("Install at least one: pip install pybloom-live OR pip install fastbloom-rs")
        sys.exit(1)

    dataset_sizes = [50000]
    fp_rate = 0.01

    print("="*100)
    print("BLOOM FILTER PERFORMANCE BENCHMARK")
    print("="*100)
    print(f"Custom: Rust-optimized (batch {n_columns}-column membership_ratio call)")
    if PYBLOOM_AVAILABLE:
        print(f"pybloom-live: Pure Python (column-by-column loop)")
    if FASTBLOOM_AVAILABLE:
        print(f"fastbloom-rs: Rust-based (column-by-column loop)")
    print(f"\nTest columns (n): {n_columns}")
    print(f"False Positive Rate Target: {fp_rate*100}%")
    print(f"Dataset Sizes: {dataset_sizes}")
    print("="*100)

    for size in dataset_sizes:
        print(f"\n{'#'*100}")
        print(f"# Dataset Size: {size:,} items  |  {n_columns} test columns")
        print(f"{'#'*100}")

        training, test = generate_dataset(size, n_columns)
        training_len = training.select(pl.len()).collect()[0, 0]
        test_len = test.select(pl.len()).collect()[0, 0]
        print(f"Training set: {training_len:,} items (1 column)")
        print(f"Test set:     {test_len:,} rows × {n_columns} columns "
              f"({test_len//2:,} positives + {test_len - test_len//2:,} negatives per column)")

        print("\nBenchmarking Custom Bloom Filter...")
        custom_result = benchmark_custom_bloom(training, test, size, fp_rate)

        pybloom_result = None
        if PYBLOOM_AVAILABLE:
            print("Benchmarking pybloom-live...")
            pybloom_result = benchmark_pybloom(training, test, size, fp_rate)

        fastbloom_result = None
        if FASTBLOOM_AVAILABLE:
            print("Benchmarking fastbloom-rs...")
            fastbloom_result = benchmark_fastbloom(training, test, size, fp_rate)

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
