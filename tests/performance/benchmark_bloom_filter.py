#!/usr/bin/env python
"""
Benchmark comparison: Custom Bloom Filter (Rust-optimized) vs fastbloom-rs

Tests:
1. Construction time
2. Insertion time (bulk add)
3. Membership query time across n columns
4. Memory usage
5. False positive rate accuracy

Dataset sizes: configurable
n columns: all checked against one bloom filter in a single call (custom) vs column loop (fastbloom-rs)
"""

import time
import sys
import tracemalloc
from pathlib import Path
from typing import Tuple, Optional
import polars as pl

# Add project root to path so services package is importable
sys.path.insert(0, str(Path(__file__).parents[2]))

# Our implementation
from services.analytics.bloom_filter import BloomFilter as CustomBloomFilter

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
    fastbloom: Optional[BenchmarkResult],
    fp_rate: float,
    n_items: int,
    n_columns: int,
) -> None:
    W = 28
    print()
    print("BLOOM FILTER BENCHMARK")
    print(f"{n_items:,} items | {n_columns} columns | FP target: {fp_rate*100:.2f}%")
    print()
    print(f"{'Method':<{W}}  {'Construct':>10}  {'Insert':>10}  {'Query':>10}  {'Memory':>10}")
    print("-" * (W + 48))
    print(
        f"{'Custom (Rust, batch)':<{W}}"
        f"  {custom.construction_time*1000:>9.1f}ms"
        f"  {custom.insertion_time*1000:>9.1f}ms"
        f"  {custom.query_time*1000:>9.1f}ms"
        f"  {custom.memory_peak_mb:>8.1f} MB"
    )
    if fastbloom:
        print(
            f"{'fastbloom-rs (loop)':<{W}}"
            f"  {fastbloom.construction_time*1000:>9.1f}ms"
            f"  {fastbloom.insertion_time*1000:>9.1f}ms"
            f"  {fastbloom.query_time*1000:>9.1f}ms"
            f"  {fastbloom.memory_peak_mb:>8.1f} MB"
        )

        def _speedup(a: float, b: float) -> str:
            return f"{b/a:.1f}x" if a > 0 else "—"

        print(
            f"{'Speedup (Custom)':<{W}}"
            f"  {_speedup(custom.construction_time, fastbloom.construction_time):>10}"
            f"  {_speedup(custom.insertion_time, fastbloom.insertion_time):>10}"
            f"  {_speedup(custom.query_time, fastbloom.query_time):>10}"
            f"  {'n/a':>10}"
        )

    print()
    fp_tol = fp_rate * 3.0
    custom_ok = custom.false_positive_rate <= fp_tol
    if fastbloom:
        fast_ok = fastbloom.false_positive_rate <= fp_tol
        print(f"FP rate within tolerance (<={fp_tol*100:.2f}%): {'yes' if (custom_ok and fast_ok) else 'no'}")
        print(f"  Custom:       {custom.false_positive_rate*100:.3f}%")
        print(f"  fastbloom-rs: {fastbloom.false_positive_rate*100:.3f}%")
    else:
        print(f"FP rate within tolerance (<={fp_tol*100:.2f}%): {'yes' if custom_ok else 'no'}")
        print(f"  Custom: {custom.false_positive_rate*100:.3f}%")
    print()


def main():
    """Run benchmarks on multiple dataset sizes."""
    n_columns = 50
    dataset_sizes = [50000]
    fp_rate = 0.01

    for size in dataset_sizes:
        training, test = generate_dataset(size, n_columns)

        print(f"Benchmarking custom filter ({size:,} items, {n_columns} columns)...")
        custom_result = benchmark_custom_bloom(training, test, size, fp_rate)

        fastbloom_result = None
        if FASTBLOOM_AVAILABLE:
            print("Benchmarking fastbloom-rs...")
            fastbloom_result = benchmark_fastbloom(training, test, size, fp_rate)

        print_comparison(custom_result, fastbloom_result, fp_rate, size, n_columns)


if __name__ == "__main__":
    main()
