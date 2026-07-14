"""
Benchmark comparing the Rust plugin vs scikit-learn for pairwise Adjusted
Rand Index.

This script:
1. Loads large_dataset.arrow (50K rows, 101 columns; String columns are
   near-unique, so scores there are ~1 by construction — still valid timing)
2. Runs Rust plugin pairwise_adjusted_rand over all C(101,2) = 5050 pairs —
   3 runs averaged
3. Runs sklearn.metrics.adjusted_rand_score per pair with the same drop-null
   preprocessing (preprocessing time included — it is part of the sklearn
   workflow) — 3 runs averaged
4. Reports timing and validates ARI values match (rel_tol=1e-9, abs_tol=1e-12)
"""

import math
import sys
import time
from pathlib import Path

import polars as pl
from sklearn.metrics import adjusted_rand_score

_ANALYTICS_ROOT = Path(__file__).parent.parent / "services" / "analytics"
sys.path.insert(0, str(_ANALYTICS_ROOT))

from analytics import pairwise_adjusted_rand

DATA_PATH = Path(__file__).parent / "data" / "large_dataset.arrow"


def load_data() -> pl.DataFrame:
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    return pl.read_ipc(DATA_PATH)


def _run_plugin(df: pl.DataFrame) -> tuple[dict, float]:
    """Run the Rust plugin; returns ({(a, b): (ari, n_valid)}, seconds)."""
    start = time.perf_counter()
    result = pairwise_adjusted_rand(df)
    duration = time.perf_counter() - start
    rows = result.unnest(result.columns[0])
    out = {
        (row["col_a"], row["col_b"]): (row["ari"], row["n_valid"])
        for row in rows.iter_rows(named=True)
    }
    return out, duration


def _sklearn_pair(df: pl.DataFrame, col_a: str, col_b: str) -> float:
    """sklearn ARI with the plugin's drop-null policy (preprocessing included)."""
    sub = df.select([col_a, col_b]).drop_nulls()
    if sub.height == 0:
        return float("nan")
    # to_physical: Date -> Int32 etc.; String stays String. sklearn accepts both.
    x = sub[col_a].to_physical().to_numpy()
    y = sub[col_b].to_physical().to_numpy()
    return adjusted_rand_score(x, y)


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"{df.width} columns -> {df.width * (df.width - 1) // 2} pairs")

    # -- Rust plugin (3 runs) --------------------------------------------------
    print("Running Rust plugin (3 runs)...")
    plugin_result: dict = {}
    plugin_times = []
    for run in range(3):
        plugin_result, t = _run_plugin(df)
        plugin_times.append(t)
        print(f"  run {run + 1}/3 done in {t:.2f}s")
    time_plugin = sum(plugin_times) / len(plugin_times)

    # -- sklearn per-pair (3 runs) ----------------------------------------------
    print("Running sklearn per-pair (3 runs)...")
    sklearn_result: dict = {}
    sklearn_times = []
    for run in range(3):
        start = time.perf_counter()
        sklearn_result = {
            pair: _sklearn_pair(df, pair[0], pair[1]) for pair in plugin_result
        }
        t = time.perf_counter() - start
        sklearn_times.append(t)
        print(f"  run {run + 1}/3 done in {t:.1f}s")
    time_sklearn = sum(sklearn_times) / len(sklearn_times)

    # -- Correctness -------------------------------------------------------------
    mismatches = 0
    for pair, (ari, _n_valid) in plugin_result.items():
        sk = sklearn_result[pair]
        if math.isnan(ari) and math.isnan(sk):
            continue
        if math.isnan(ari) != math.isnan(sk) or not math.isclose(
            ari, sk, rel_tol=1e-9, abs_tol=1e-12
        ):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH {pair}: plugin={ari} sklearn={sk}")

    # -- Summary -------------------------------------------------------------------
    W = 32
    print()
    print("ADJUSTED RAND INDEX BENCHMARK")
    print(f"{len(plugin_result)} pairs | {df.height} rows")
    print()
    print(f"{'Method':<{W}}  {'Avg time (3 runs)':>18}  {'vs plugin':>10}")
    print("-" * (W + 32))
    print(f"{'Rust plugin batch':<{W}}  {time_plugin * 1000:>17.1f}ms  {'1.0x':>10}")
    print(f"{'sklearn (per-pair loop)':<{W}}  {time_sklearn * 1000:>17.1f}ms  {time_sklearn / time_plugin:>9.1f}x")
    print()
    print(f"Correctness vs sklearn (rel_tol=1e-9): {'yes' if mismatches == 0 else f'no ({mismatches} mismatches)'}")
    print()


if __name__ == "__main__":
    main()
