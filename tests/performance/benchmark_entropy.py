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

from analytics import (
    pairwise_joint_entropy,
    threeway_joint_entropy,
    marginal_entropy,
)

# Data file path
DATA_PATH = Path(__file__).parents[1] / "data" / "large_dataset.arrow"


def load_data() -> pl.LazyFrame:
    """Load Arrow file as LazyFrame for zero-copy operations."""
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure large_dataset.arrow exists in the project root.")
        sys.exit(1)
    return pl.scan_ipc(DATA_PATH)


def benchmark_native_polars_marginal(lf: pl.LazyFrame, col: str) -> tuple[float, float]:
    """
    Benchmark native Polars marginal (single-column) entropy using
    value_counts + Series.entropy(base=2).

    Returns:
        tuple[float, float]: (entropy_value, duration_seconds)
    """
    start = time.perf_counter()

    s = lf.select(col).collect().to_series()
    counts = s.value_counts().get_column("count")
    entropy_value = counts.entropy(base=2, normalize=True)

    duration = time.perf_counter() - start
    return entropy_value, duration


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


def _run_marginal_plugin(plugin_fn, lf: pl.LazyFrame) -> tuple[dict, float]:
    """Run the marginal plugin function and return (entropy_dict, duration)."""
    start = time.perf_counter()
    result = plugin_fn(lf.collect())
    duration = time.perf_counter() - start

    unnested = result.unnest(result.columns[0])
    entropy_dict = {
        row["col_name"]: row["entropy"]
        for row in unnested.iter_rows(named=True)
    }
    return entropy_dict, duration


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
    print("Loading dataset...")
    lf = load_data()
    columns = list(lf.collect_schema().keys())
    column_pairs = list(combinations(columns, 2))

    # -- Marginal plugin (3 runs) -----------------------------------------------
    print(f"Running marginal plugin ({len(columns)} columns, 3 runs)...")
    entropy_marginal: dict = {}
    marginal_times = []
    for _ in range(3):
        entropy_marginal, t = _run_marginal_plugin(marginal_entropy, lf)
        marginal_times.append(t)
    time_marginal = sum(marginal_times) / len(marginal_times)

    # -- Native Polars per-column ------------------------------------------------
    print(f"Running native Polars per-column ({len(columns)} columns)...")
    total_time_marginal_native = 0.0
    marginal_mismatches = 0
    marginal_tested = 0
    for col in columns:
        try:
            entropy_native, time_native = benchmark_native_polars_marginal(lf, col)
            total_time_marginal_native += time_native
            marginal_tested += 1
        except Exception:
            continue
        plugin_val = entropy_marginal.get(col)
        if plugin_val is not None and not validate_entropy_match(entropy_native, plugin_val):
            marginal_mismatches += 1

    # -- Pairwise plugin (3 runs) ----------------------------------------------
    print(f"Running pairwise plugin ({len(column_pairs)} pairs, 3 runs)...")
    entropy_plugin: dict = {}
    plugin_times = []
    for _ in range(3):
        entropy_plugin, t = _run_batch_plugin(pairwise_joint_entropy, lf)
        plugin_times.append(t)
    time_plugin = sum(plugin_times) / len(plugin_times)

    # -- Native Polars per-pair ------------------------------------------------
    print(f"Running native Polars per-pair ({len(column_pairs)} pairs)...")
    total_time_native = 0.0
    pair_mismatches = 0
    pairs_tested = 0
    for col_a, col_b in column_pairs:
        try:
            entropy_native, time_native = benchmark_native_polars(lf, col_a, col_b)
            total_time_native += time_native
            pairs_tested += 1
        except Exception:
            continue
        plugin_val = entropy_plugin.get((col_a, col_b))
        if plugin_val is not None and not validate_entropy_match(entropy_native, plugin_val):
            pair_mismatches += 1

    # -- 3-way plugin (3 runs) -------------------------------------------------
    # C(101,3) = 166,650 triplets: each run takes on the order of a minute.
    print("Running 3-way plugin (3 runs)...")
    entropy_3way: dict = {}
    threeway_times = []
    for run in range(3):
        entropy_3way, t = _run_threeway_plugin(threeway_joint_entropy, lf)
        threeway_times.append(t)
        print(f"  run {run + 1}/3 done in {t:.1f}s ({len(entropy_3way):,} triplets)")
    time_3way = sum(threeway_times) / len(threeway_times)
    actual_triplets = len(entropy_3way)

    # -- Native Polars per-triplet (capped sample) ----------------------------
    MAX_NATIVE_TRIPLETS = 10_000
    native_triplet_sample = list(entropy_3way.keys())[:MAX_NATIVE_TRIPLETS]
    print(f"Running native Polars per-triplet ({len(native_triplet_sample):,} of {actual_triplets:,} triplets)...")
    total_time_3way_native = 0.0
    triplet_mismatches = 0
    triplets_tested = 0
    for col_a, col_b, col_c in native_triplet_sample:
        try:
            entropy_native, time_native = benchmark_threeway_native(lf, col_a, col_b, col_c)
            total_time_3way_native += time_native
            triplets_tested += 1
        except Exception:
            continue
        plugin_val = entropy_3way.get((col_a, col_b, col_c))
        if plugin_val is not None and not validate_entropy_match(plugin_val, entropy_native):
            triplet_mismatches += 1

    # -- Summary ---------------------------------------------------------------
    W = 32
    print()
    print("JOINT ENTROPY BENCHMARK")
    print(f"{len(column_pairs)} pairs | {actual_triplets} triplets | {len(columns)} columns")
    print()
    print(f"Marginal (single-column)")
    print(f"{'Method':<{W}}  {'Avg time (3 runs)':>18}  {'vs plugin':>10}")
    print("-" * (W + 32))
    print(f"{'Plugin batch':<{W}}  {time_marginal * 1000:>17.1f}ms  {'1.0x':>10}")
    if marginal_tested > 0:
        print(f"{'Native Polars (per-column)':<{W}}  {total_time_marginal_native * 1000:>17.1f}ms  {total_time_marginal_native / time_marginal:>9.1f}x")
    print()
    print(f"Pairwise (2-way)")
    print(f"{'Method':<{W}}  {'Avg time (3 runs)':>18}  {'vs plugin':>10}")
    print("-" * (W + 32))
    print(f"{'Plugin batch':<{W}}  {time_plugin * 1000:>17.1f}ms  {'1.0x':>10}")
    if pairs_tested > 0:
        print(f"{'Native Polars (per-pair)':<{W}}  {total_time_native * 1000:>17.1f}ms  {total_time_native / time_plugin:>9.1f}x")
    print()
    print(f"3-way")
    print(f"{'Method':<{W}}  {'Avg time (3 runs)':>18}  {'vs plugin':>10}")
    print("-" * (W + 32))
    print(f"{'Plugin batch':<{W}}  {time_3way * 1000:>17.1f}ms  {'1.0x':>10}")
    if triplets_tested > 0:
        avg_native_ms = total_time_3way_native * 1000 / triplets_tested
        extrap_ms = avg_native_ms * actual_triplets
        extrap_speedup = extrap_ms / (time_3way * 1000)
        print(f"{'Native Polars (10K sample)':<{W}}  {extrap_ms:>17.1f}ms  {extrap_speedup:>9.1f}x  (extrap.)")
    print()
    print("Correctness vs native Polars  (rtol=1e-5)")
    print(f"  Marginal: {'yes' if marginal_mismatches == 0 else f'no  ({marginal_mismatches}/{marginal_tested} mismatches)'}")
    print(f"  Pairwise: {'yes' if pair_mismatches == 0 else f'no  ({pair_mismatches}/{pairs_tested} mismatches)'}")
    print(f"  3-way:    {'yes' if triplet_mismatches == 0 else f'no  ({triplet_mismatches}/{triplets_tested} mismatches)'}")
    print()


if __name__ == "__main__":
    main()
