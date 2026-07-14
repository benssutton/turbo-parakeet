"""
Benchmark comparing Rust plugin vs polars-ds vs scipy for pairwise chi-squared
independence testing.

This script:
1. Loads large_dataset.arrow as a LazyFrame (zero-copy)
2. Identifies suitable columns: Boolean, String, Categorical, Enum, and
   integer columns (include_integer=True) with at most MAX_UNIQUE unique values
3. Runs Rust plugin pairwise_chi_squared — 3 runs averaged
4. Runs polars-ds pairwise_chi_squared (Python-loop baseline) — 3 runs averaged
5. Runs scipy.stats.chi2_contingency per-pair for correctness validation
6. Reports timing and validates chi2 statistics match across all three

Note: scipy is called with correction=False to match polars-ds / Rust behaviour
(no Yates' continuity correction).
"""

import math
import sys
import time
from itertools import combinations
from pathlib import Path

import polars as pl

_ANALYTICS_ROOT = Path(__file__).parent.parent / "services" / "analytics"
sys.path.insert(0, str(_ANALYTICS_ROOT))

from chi_squared import _get_suitable_columns, pairwise_chi_squared as pairwise_chi_squared_pds
from analytics import pairwise_chi_squared as pairwise_chi_squared_rust

DATA_PATH = Path(__file__).parent / "data" / "large_dataset.arrow"

# Upper cardinality bound passed to pairwise_chi_squared and used to filter
# columns. Chi-squared on near-unique columns is statistically meaningless and
# causes memory exhaustion when building contingency tables.
_MAX_UNIQUE = 1000


def load_data() -> pl.LazyFrame:
    """Load Arrow file as LazyFrame for zero-copy operations."""
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure large_dataset.arrow exists in tests/data/.")
        sys.exit(1)
    return pl.scan_ipc(DATA_PATH)


def _run_batch_pds(df: pl.DataFrame) -> tuple[dict, float]:
    """
    Run polars-ds pairwise_chi_squared and return (result_dict, duration_seconds).
    result_dict maps (col_a, col_b) -> {chi2_stat, p_value, cramers_v}.
    """
    start = time.perf_counter()
    result = pairwise_chi_squared_pds(df, include_integer=True, min_unique=2, max_unique=_MAX_UNIQUE)
    duration = time.perf_counter() - start

    unnested = result.unnest("pairwise_chi_squared")
    result_dict = {
        (row["col_a"], row["col_b"]): {
            "chi2_stat": row["chi2_stat"],
            "p_value": row["p_value"],
            "cramers_v": row["cramers_v"],
        }
        for row in unnested.iter_rows(named=True)
    }
    return result_dict, duration


def _run_batch_rust(df: pl.DataFrame, pairs: list[tuple[str, str]]) -> tuple[dict, float]:
    """
    Run Rust pairwise_chi_squared and return (result_dict, duration_seconds).
    result_dict maps (col_a, col_b) -> {chi2_stat, p_value, cramers_v}.
    """
    start = time.perf_counter()
    result = pairwise_chi_squared_rust(df, pairs=pairs)
    duration = time.perf_counter() - start

    unnested = result.unnest("pairwise_chi_squared")
    result_dict = {
        (row["col_a"], row["col_b"]): {
            "chi2_stat": row["chi2_stat"],
            "p_value": row["p_value"],
            "cramers_v": row["cramers_v"],
        }
        for row in unnested.iter_rows(named=True)
    }
    return result_dict, duration


def benchmark_scipy_pair(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
) -> tuple[float, float, float]:
    """
    Compute chi-squared for one pair using scipy.

    Builds the contingency table via Polars group_by + pivot, then calls
    scipy.stats.chi2_contingency(correction=False) to match polars-ds and Rust.

    Returns
    -------
    tuple[float, float, float]
        (chi2_stat, p_value, duration_seconds)
    """
    from scipy.stats import chi2_contingency

    start = time.perf_counter()

    pair_df = df.select([col_a, col_b]).drop_nulls()

    contingency = (
        pair_df
        .group_by([col_a, col_b])
        .agg(pl.len().alias("count"))
        .pivot(index=col_a, on=col_b, values="count")
        .fill_null(0)
    )

    value_cols = [c for c in contingency.columns if c != col_a]
    table = contingency.select(value_cols).to_numpy()

    chi2_result = chi2_contingency(table, correction=False)
    duration = time.perf_counter() - start

    return chi2_result.statistic, chi2_result.pvalue, duration


def validate_chi2_match(v1: float, v2: float, rtol: float = 1e-4) -> bool:
    """Return True if two chi2 values agree within relative tolerance."""
    if math.isnan(v1) or math.isnan(v2):
        return math.isnan(v1) and math.isnan(v2)
    return math.isclose(v1, v2, rel_tol=rtol)


def main() -> None:
    print("=" * 70)
    print("PAIRWISE CHI-SQUARED INDEPENDENCE TEST BENCHMARK")
    print()
    print("  1. Rust plugin     (pairwise_chi_squared — analytics)")
    print("  2. polars-ds batch (pairwise_chi_squared — chi_squared service)")
    print("  3. scipy per-pair  (chi2_contingency, correction=False)")
    print("=" * 70)
    print()

    print("Loading dataset...")
    lf = load_data()
    df = lf.collect()

    all_cols = list(df.schema.keys())
    suitable_cols = _get_suitable_columns(df, include_integer=True, min_unique=2, max_unique=_MAX_UNIQUE)
    all_pairs = list(combinations(suitable_cols, 2))

    print(f"Dataset columns (total):                   {len(all_cols)}")
    print(f"Suitable columns (max_unique={_MAX_UNIQUE}):         {len(suitable_cols)}")
    print(f"  {', '.join(suitable_cols)}")
    print(f"Total pairs:                               {len(all_pairs)}")
    print()

    if not all_pairs:
        print("No suitable column pairs found. Exiting.")
        return

    # =========================================================================
    # RUST PLUGIN
    # =========================================================================
    print("Running Rust plugin (3 runs)...")
    rust_result: dict = {}
    rust_times: list[float] = []
    for run in range(3):
        rust_result, t = _run_batch_rust(df, all_pairs)
        rust_times.append(t)
        print(f"  Run {run + 1}: {t * 1000:.2f}ms")

    time_rust = sum(rust_times) / len(rust_times)
    print(f"  Average: {time_rust * 1000:.2f}ms")
    print()

    # =========================================================================
    # POLARS-DS BATCH
    # =========================================================================
    print("Running polars-ds batch (3 runs)...")
    pds_result: dict = {}
    pds_times: list[float] = []
    for run in range(3):
        pds_result, t = _run_batch_pds(df)
        pds_times.append(t)
        print(f"  Run {run + 1}: {t * 1000:.2f}ms")

    time_pds = sum(pds_times) / len(pds_times)
    print(f"  Average: {time_pds * 1000:.2f}ms")
    print()

    # =========================================================================
    # SCIPY PER-PAIR
    # =========================================================================
    scipy_results: list[dict] = []
    total_time_scipy = 0.0
    rust_mismatches = 0
    pds_mismatches = 0
    skipped = 0

    print(f"Running scipy per-pair ({len(all_pairs)} pairs)...")
    for idx, (col_a, col_b) in enumerate(all_pairs, 1):
        if idx % 10 == 0 or idx == len(all_pairs):
            print(f"  Progress: {idx}/{len(all_pairs)} pairs...")

        try:
            chi2_scipy, pval_scipy, t = benchmark_scipy_pair(df, col_a, col_b)
            total_time_scipy += t
        except Exception as e:
            print(f"  scipy error for ({col_a}, {col_b}): {e}")
            skipped += 1
            continue

        rust_val = rust_result.get((col_a, col_b))
        if rust_val is not None:
            if not validate_chi2_match(chi2_scipy, rust_val["chi2_stat"]):
                diff = abs(chi2_scipy - rust_val["chi2_stat"])
                print(f"  Rust mismatch ({col_a}, {col_b}): "
                      f"scipy={chi2_scipy:.6f}  rust={rust_val['chi2_stat']:.6f}  "
                      f"diff={diff:.6f}")
                rust_mismatches += 1

        pds_val = pds_result.get((col_a, col_b))
        if pds_val is not None:
            if not validate_chi2_match(chi2_scipy, pds_val["chi2_stat"]):
                diff = abs(chi2_scipy - pds_val["chi2_stat"])
                print(f"  polars-ds mismatch ({col_a}, {col_b}): "
                      f"scipy={chi2_scipy:.6f}  pds={pds_val['chi2_stat']:.6f}  "
                      f"diff={diff:.6f}")
                pds_mismatches += 1

        scipy_results.append({
            "col_a": col_a,
            "col_b": col_b,
            "chi2_scipy": chi2_scipy,
            "pval_scipy": pval_scipy,
            "time": t,
        })

    tested = len(scipy_results)
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total pairs (suitable columns):            {len(all_pairs)}")
    print(f"Pairs compared (scipy):                    {tested}")
    if skipped:
        print(f"Pairs skipped (errors):                    {skipped}")
    print()
    print(f"Total time (Rust plugin, avg 3 runs):      {time_rust * 1000:.2f}ms")
    print(f"Total time (polars-ds batch, avg 3 runs):  {time_pds * 1000:.2f}ms")
    if tested > 0:
        print(f"Total time (scipy per-pair):               {total_time_scipy * 1000:.2f}ms")
    print()

    if tested > 0:
        if time_rust > 0:
            print(f"Speedup Rust vs polars-ds:                 {time_pds / time_rust:.2f}x")
            print(f"Speedup Rust vs scipy:                     {total_time_scipy / time_rust:.2f}x")
        print()

        if rust_mismatches == 0:
            print("Rust vs scipy:      All chi2 values matched  (rtol=1e-4)")
        else:
            print(f"Rust vs scipy:      {rust_mismatches} pair(s) mismatched")

        if pds_mismatches == 0:
            print("polars-ds vs scipy: All chi2 values matched  (rtol=1e-4)")
        else:
            print(f"polars-ds vs scipy: {pds_mismatches} pair(s) mismatched")

    print()

    # Sample of results (Rust)
    print("Sample results — Rust (first 5 pairs):")
    sample_pairs = list(rust_result.items())[:5]
    for (ca, cb), vals in sample_pairs:
        print(f"  ({ca}, {cb}): "
              f"chi2={vals['chi2_stat']:.4f}  "
              f"p={vals['p_value']:.4f}  "
              f"V={vals['cramers_v']:.4f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
