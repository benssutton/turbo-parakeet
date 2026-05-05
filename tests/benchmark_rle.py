"""
Benchmark: pure-Python run-length encoding analysis (Approach A, Polars rle).

This script:
1. Loads large_dataset.arrow as a DataFrame
2. Runs column_run_stats over all columns - 3 runs averaged
3. Validates each column's histogram against n_rows and n_runs
4. Reports per-column run statistics, REE compression distribution, and
   sample histograms for the most-compressible columns
"""

import sys
import time
from pathlib import Path

import polars as pl

_ANALYTICS_ROOT = Path(__file__).parent.parent / "services" / "analytics"
sys.path.insert(0, str(_ANALYTICS_ROOT))

from run_length import column_run_stats

DATA_PATH = Path(__file__).parent / "data" / "large_dataset.arrow"


def load_data() -> pl.LazyFrame:
    """Load Arrow file as LazyFrame for zero-copy operations."""
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure large_dataset.arrow exists in tests/data/.")
        sys.exit(1)
    return pl.scan_ipc(DATA_PATH)


def _validate(result: pl.DataFrame) -> int:
    """
    Sanity-check each column's histogram:
      sum(size * count) must equal n_rows
      sum(count)        must equal n_runs

    Returns number of mismatched columns (0 = all good).
    """
    mismatches = 0
    for row in result.iter_rows(named=True):
        hist = row["run_length_histogram"]
        if not hist:
            continue
        sum_sizes = sum(h["size"] * h["count"] for h in hist)
        sum_counts = sum(h["count"] for h in hist)
        if sum_sizes != row["n_rows"]:
            print(
                f"  MISMATCH ({row['col_name']}): "
                f"sum(size*count)={sum_sizes} != n_rows={row['n_rows']}"
            )
            mismatches += 1
        if sum_counts != row["n_runs"]:
            print(
                f"  MISMATCH ({row['col_name']}): "
                f"sum(count)={sum_counts} != n_runs={row['n_runs']}"
            )
            mismatches += 1
    return mismatches


def main() -> None:
    print("=" * 72)
    print("RUN-LENGTH ENCODING ANALYSIS BENCHMARK")
    print()
    print("  Approach A: pure Python using polars.Expr.rle()")
    print("=" * 72)
    print()

    print("Loading dataset...")
    lf = load_data()
    df = lf.collect()
    print(f"Dataset: {df.height:,} rows x {df.width} columns")
    print()

    # =========================================================================
    # TIMING - 3 runs averaged
    # =========================================================================
    print("Running column_run_stats (3 runs)...")
    times: list[float] = []
    result = column_run_stats(df)  # warm-up not timed
    for run in range(3):
        start = time.perf_counter()
        result = column_run_stats(df)
        duration = time.perf_counter() - start
        times.append(duration)
        print(f"  Run {run + 1}: {duration * 1000:.2f}ms")
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time * 1000:.2f}ms")
    print(f"  Per column avg: {avg_time * 1000 / df.width:.2f}ms")
    print()

    # =========================================================================
    # VALIDATION
    # =========================================================================
    print("Validating histograms...")
    mismatches = _validate(result)
    if mismatches == 0:
        print(f"  All {result.height} columns: histogram sums match n_rows and n_runs")
    else:
        print(f"  {mismatches} mismatch(es) detected")
    print()

    # =========================================================================
    # PER-COLUMN STATS
    # =========================================================================
    sorted_result = result.sort("compression_ratio", descending=True)

    print("=" * 72)
    print("PER-COLUMN RUN STATS (sorted by compression ratio, descending)")
    print("=" * 72)
    print(f"{'Column':<20} {'n_runs':>10} {'comp.':>8} {'mean':>10} {'max':>10}")
    print("-" * 72)
    for row in sorted_result.iter_rows(named=True):
        print(
            f"{row['col_name']:<20} "
            f"{row['n_runs']:>10,} "
            f"{row['compression_ratio']:>8.2f} "
            f"{row['mean_run_length']:>10.2f} "
            f"{row['max_run_length']:>10,}"
        )
    print()

    # =========================================================================
    # COMPRESSION DISTRIBUTION
    # =========================================================================
    print("=" * 72)
    print("REE COMPRESSION DISTRIBUTION")
    print("=" * 72)
    n_total = result.height
    buckets = [
        (1.0, 1.1, "no benefit  (<= 1.1x)"),
        (1.1, 1.5, "marginal    (1.1 - 1.5x)"),
        (1.5, 2.0, "modest      (1.5 - 2x)"),
        (2.0, 5.0, "good        (2 - 5x)"),
        (5.0, 10.0, "strong      (5 - 10x)"),
        (10.0, float("inf"), "excellent   (>= 10x)"),
    ]
    for low, high, label in buckets:
        count = result.filter(
            (pl.col("compression_ratio") >= low)
            & (pl.col("compression_ratio") < high)
        ).height
        bar = "#" * int(40 * count / n_total) if n_total > 0 else ""
        print(f"  {label:<26} {count:>4} cols  {bar}")
    print()

    # =========================================================================
    # SAMPLE HISTOGRAMS - top 5 most-compressible columns
    # =========================================================================
    print("=" * 72)
    print("TOP 5 - RUN LENGTH HISTOGRAM (top 5 most-frequent run sizes)")
    print("=" * 72)
    for row in sorted_result.head(5).iter_rows(named=True):
        hist = sorted(row["run_length_histogram"], key=lambda h: -h["count"])[:5]
        hist_str = ", ".join(f"{h['size']}x{h['count']}" for h in hist)
        print(
            f"  {row['col_name']:<20} "
            f"comp={row['compression_ratio']:>6.2f}x  "
            f"runs={row['n_runs']:>6,}  "
            f"[{hist_str}]"
        )
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Dataset:                     {df.height:,} rows x {df.width} columns")
    print(f"  column_run_stats avg:        {avg_time * 1000:.2f}ms (3 runs)")
    print(f"  Per-column avg:              {avg_time * 1000 / df.width:.2f}ms")
    n_beneficial = result.filter(pl.col("compression_ratio") >= 1.5).height
    print(f"  Cols with REE benefit >=1.5x: {n_beneficial} / {n_total}")
    print("=" * 72)


if __name__ == "__main__":
    main()
