"""
Benchmark script comparing custom plugin vs native Polars for joint entropy calculation.

This script:
1. Loads large_dataset.arrow as LazyFrame (zero-copy)
2. Calculates pairwise joint entropy for all column pairs
3. Compares plugin and native Polars group_by approach
4. Reports speedup and validates entropy values match
"""

import polars as pl
import time
from pathlib import Path
from itertools import combinations
import sys

# Add parent directory to path to import plugin
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics import (
    pairwise_joint_entropy,
    threeway_joint_entropy,
)

# Data file path
DATA_PATH = Path(__file__).parent.parent / "data/large_dataset.arrow"


def load_data() -> pl.LazyFrame:
    """Load Arrow file as LazyFrame for zero-copy operations."""
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure large_dataset.arrow exists in the project root.")
        sys.exit(1)
    return pl.scan_ipc(DATA_PATH)


def benchmark_native_polars(lf: pl.LazyFrame, col_a: str, col_b: str) -> tuple[float, float]:
    """
    Benchmark native Polars method using group_by + aggregation.

    Returns:
        tuple[float, float]: (entropy_value, duration_seconds)
    """
    start = time.perf_counter()

    # Native Polars approach: group_by + count + entropy calculation
    result = (
        lf
        .group_by([col_a, col_b])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("p")
        )
        .select(
            (-pl.col("p") * pl.col("p").log(base=2.0)).sum().alias("entropy")
        )
        .collect()
    )

    duration = time.perf_counter() - start
    entropy_value = result["entropy"][0]
    return entropy_value, duration


def validate_entropy_match(e1: float, e2: float, rtol: float = 1e-5) -> bool:
    """Check if two entropy values match within relative tolerance."""
    import math
    return math.isclose(e1, e2, rel_tol=rtol)


def _run_batch_plugin(plugin_fn, lf: pl.LazyFrame) -> tuple[dict, float]:
    """Run a pairwise plugin function and return (entropy_dict, duration)."""
    start = time.perf_counter()
    result = plugin_fn(lf.collect())
    duration = time.perf_counter() - start

    unnested = result.unnest(result.columns[0])
    entropy_dict = {
        (row["col_a"], row["col_b"]): row["entropy"]
        for row in unnested.iter_rows(named=True)
    }
    return entropy_dict, duration


def _run_threeway_plugin(plugin_fn, lf: pl.LazyFrame) -> tuple[dict, float]:
    """Run a threeway plugin function and return (entropy_dict, duration)."""
    start = time.perf_counter()
    result = plugin_fn(lf.collect())
    duration = time.perf_counter() - start

    unnested = result.unnest(result.columns[0])
    entropy_dict = {
        (row["col_a"], row["col_b"], row["col_c"]): row["entropy"]
        for row in unnested.iter_rows(named=True)
    }
    return entropy_dict, duration


def benchmark_threeway_native(lf: pl.LazyFrame, col_a: str, col_b: str, col_c: str) -> tuple[float, float]:
    """
    Benchmark native Polars 3-way entropy using group_by + aggregation.

    Returns:
        tuple[float, float]: (entropy_value, duration_seconds)
    """
    start = time.perf_counter()

    # Native Polars approach: group_by + count + entropy calculation
    result = (
        lf
        .group_by([col_a, col_b, col_c])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("p")
        )
        .select(
            (-pl.col("p") * pl.col("p").log(base=2.0)).sum().alias("entropy")
        )
        .collect()
    )

    duration = time.perf_counter() - start
    entropy_value = result["entropy"][0]
    return entropy_value, duration


def main():
    print("=" * 70)
    print("PAIRWISE & 3-WAY JOINT ENTROPY BENCHMARK")
    print()
    print("PAIRWISE (2-way):")
    print("  1. Native Polars group_by")
    print("  2. Batch Plugin (pairwise_joint_entropy)")
    print()
    print("3-WAY:")
    print("  3. Batch Plugin (threeway_joint_entropy)")
    print("  4. Native Polars group_by (for comparison)")
    print("=" * 70)
    print()

    # Load dataset
    print("Loading dataset...")
    lf = load_data()

    # Get column names (requires minimal collect to inspect schema)
    schema = lf.schema
    columns = list(schema.keys())

    print(f"Dataset: {len(columns)} columns")
    print(f"Columns: {', '.join(columns)}")

    # Generate all column pairs
    column_pairs = list(combinations(columns, 2))
    print(f"Total pairs: {len(column_pairs)}")
    print()

    # ========================================================================
    # PAIRWISE BENCHMARKS
    # ========================================================================

    # Batch plugin (average of 3 runs)
    print("Running Batch Plugin (all pairs at once, 3 runs)...")
    entropy_plugin: dict = {}
    plugin_times = []
    for run in range(3):
        entropy_plugin, t = _run_batch_plugin(pairwise_joint_entropy, lf)
        plugin_times.append(t)
        print(f"  Run {run + 1}: {t*1000:.2f}ms")
    time_plugin = sum(plugin_times) / len(plugin_times)
    print(f"  Average: {time_plugin*1000:.2f}ms")
    print()

    # Native Polars per-pair
    results = []
    for idx, (col_a, col_b) in enumerate(column_pairs, 1):
        try:
            entropy_native, time_native = benchmark_native_polars(lf, col_a, col_b)
        except Exception:
            continue

        entropy_plugin_pair = entropy_plugin.get((col_a, col_b))

        # Validate plugin vs native
        if entropy_plugin_pair is not None:
            if not validate_entropy_match(entropy_native, entropy_plugin_pair):
                diff = abs(entropy_native - entropy_plugin_pair)
                print(f"  Mismatch ({col_a}, {col_b}): diff = {diff:.6f}")

        results.append({
            "col_a": col_a,
            "col_b": col_b,
            "entropy_native": entropy_native,
            "entropy_plugin": entropy_plugin_pair,
            "time_native": time_native,
        })

    # Pairwise summary
    if results:
        print()
        print("=" * 70)
        print("PAIRWISE SUMMARY")
        print("=" * 70)
        print()

        total_time_native = sum(r["time_native"] for r in results)

        print(f"Tested pairs:                          {len(results)}")
        print()
        print(f"Total time (native Polars):            {total_time_native*1000:.2f}ms")
        print(f"Total time (plugin batch):             {time_plugin*1000:.2f}ms")
        print()

        speedup = total_time_native / time_plugin if time_plugin > 0 else float('inf')

        print(f"Plugin speedup vs native:              {speedup:.2f}x")
        print()

        mismatches = sum(
            1 for r in results
            if r["entropy_plugin"] is not None
            and not validate_entropy_match(r["entropy_native"], r["entropy_plugin"])
        )

        if mismatches == 0:
            print("Plugin vs native: All entropy values matched")
        else:
            print(f"Plugin vs native: {mismatches} pair(s) mismatched")

        print()
        print("=" * 70)
    else:
        print("No results to display.")

    # ========================================================================
    # 3-WAY ENTROPY BENCHMARKS
    # ========================================================================

    print()
    print()
    print("=" * 70)
    print("3-WAY JOINT ENTROPY BENCHMARK")
    print("=" * 70)
    print()

    # Generate triplet combinations (will be limited to 5000 by plugin)
    all_triplets = list(combinations(columns, 3))
    total_possible_triplets = len(all_triplets)

    print(f"Total possible triplets: {len(columns)} choose 3 = {total_possible_triplets}")
    print(f"Benchmark limit: 5000 triplets")
    print()

    # 3-way plugin (average of 3 runs)
    print("Running 3-Way Plugin (batch - up to 5000 triplets, 3 runs)...")
    entropy_3way: dict = {}
    threeway_times = []
    for run in range(3):
        entropy_3way, t = _run_threeway_plugin(threeway_joint_entropy, lf)
        threeway_times.append(t)
        print(f"  Run {run + 1}: {t*1000:.2f}ms")
    time_3way = sum(threeway_times) / len(threeway_times)
    actual_triplets = len(entropy_3way)
    print(f"  Average over 3 runs: {time_3way*1000:.2f}ms ({actual_triplets} triplets)")
    print(f"  Average per triplet: {(time_3way*1000)/actual_triplets:.3f}ms")
    print()

    # Native Polars per-triplet
    print(f"Running Native Polars for {actual_triplets} triplets (for comparison)...")
    triplets_to_benchmark = list(entropy_3way.keys())

    threeway_results = []
    total_time_3way_native = 0.0
    mismatches_3way = 0

    for idx, (col_a, col_b, col_c) in enumerate(triplets_to_benchmark, 1):
        if idx % 500 == 0 or idx == actual_triplets:
            print(f"  Progress: {idx}/{actual_triplets} triplets...")

        try:
            entropy_native, time_native = benchmark_threeway_native(lf, col_a, col_b, col_c)
            total_time_3way_native += time_native

            entropy_plugin_val = entropy_3way.get((col_a, col_b, col_c))

            if entropy_plugin_val is not None:
                if not validate_entropy_match(entropy_plugin_val, entropy_native):
                    diff = abs(entropy_plugin_val - entropy_native)
                    print(f"  Mismatch ({col_a}, {col_b}, {col_c}): diff = {diff:.6f}")
                    mismatches_3way += 1

            threeway_results.append({
                "cols": (col_a, col_b, col_c),
                "entropy_plugin": entropy_plugin_val,
                "entropy_native": entropy_native,
                "time_native": time_native,
            })

        except Exception as e:
            print(f"  Error for ({col_a}, {col_b}, {col_c}): {e}")
            continue

    print()

    # 3-way summary
    if threeway_results:
        print("=" * 70)
        print("3-WAY SUMMARY")
        print("=" * 70)
        print()

        print(f"Tested triplets:                       {len(threeway_results)}")
        print()
        print(f"Total time (native Polars):            {total_time_3way_native*1000:.2f}ms")
        print(f"Total time (plugin batch):             {time_3way*1000:.2f}ms")
        print()

        speedup_3way = total_time_3way_native / time_3way if time_3way > 0 else float('inf')

        print(f"Plugin speedup vs native:              {speedup_3way:.2f}x")
        print()

        # Average times
        avg_plugin = (time_3way * 1000) / len(threeway_results)
        avg_native = (total_time_3way_native * 1000) / len(threeway_results)

        print(f"Average time per triplet:")
        print(f"  Plugin:    {avg_plugin:.3f}ms")
        print(f"  Native:    {avg_native:.3f}ms")
        print()

        if mismatches_3way == 0:
            print("Plugin vs native: All entropy values matched")
        else:
            print(f"Plugin vs native: {mismatches_3way} triplet(s) mismatched")

        print()
        print("=" * 70)


if __name__ == "__main__":
    main()
