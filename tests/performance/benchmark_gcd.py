"""
Benchmark: whole-column GCD — Rust plugin vs numpy.gcd.reduce vs math.gcd.

Performance only; correctness is gated by tests/test_gcd.py. Results are
cross-checked here purely as a sanity guard.

Shapes:
 1. 10M rows × 4 Int64 columns of k·g, no nulls  — narrow & long (intra-column parallelism)
 2. 1M rows × 100 Int64 columns of k·g          — wide (inter-column parallelism)
 3. tests/data/large_dataset.arrow (50K × 101)  — realistic mix; adds a math.gcd baseline
 4. 10M rows × 4 random Int64 columns (GCD 1)    — early exit at the minimum increment

Shapes 1–2 have GCD 3600 > 1, so the plugin must scan every value (full scan).
Shape 4 shows the early exit; numpy.gcd.reduce has none and always scans in full.

Baselines receive pre-extracted numpy arrays / Python lists (extraction is
not timed); the plugin is timed end-to-end from a DataFrame. RUNS runs averaged.

Run: python tests/performance/benchmark_gcd.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parents[2] / "services" / "analytics"))

from analytics import column_gcd

DATA_PATH = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
RUNS = 3
SEED = 42
G = 3_600


def timed(fn):
    times = []
    result = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, sum(times) / RUNS


def plugin(df: pl.DataFrame) -> dict[str, int | None]:
    out = column_gcd(df).unnest("column_gcd")
    return dict(zip(out["column"].to_list(), out["gcd"].to_list()))


def integer_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in df.schema.items() if dt.is_integer() or dt.is_temporal()]


def make_multiples(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    return pl.DataFrame(
        {f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * G for i in range(n_cols)}
    )


def make_random(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    return pl.DataFrame(
        {f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64) for i in range(n_cols)}
    )


def run_shape(name: str, df: pl.DataFrame, with_math: bool) -> None:
    cols = integer_columns(df)
    arrays = {c: df[c].to_physical().drop_nulls().to_numpy() for c in cols}

    got, t_plugin = timed(lambda: plugin(df))
    exp_np, t_np = timed(lambda: {c: int(np.gcd.reduce(a)) for c, a in arrays.items()})

    print(f"\n{name}: {df.height:,} rows × {df.width} cols ({len(cols)} integer-backed)")
    print(f"  plugin (column_gcd)  {t_plugin * 1e3:10.2f} ms")
    print(f"  numpy.gcd.reduce     {t_np * 1e3:10.2f} ms   speedup {t_np / t_plugin:6.1f}x")
    mismatches = [c for c in cols if got[c] != exp_np[c]]

    if with_math:
        lists = {c: a.tolist() for c, a in arrays.items()}
        exp_math, t_math = timed(lambda: {c: math.gcd(*v) for c, v in lists.items()})
        print(f"  math.gcd(*col)       {t_math * 1e3:10.2f} ms   speedup {t_math / t_plugin:6.1f}x")
        mismatches += [c for c in cols if got[c] != exp_math[c]]

    if mismatches:
        raise SystemExit(f"  RESULT MISMATCH in {sorted(set(mismatches))}")
    print("  results match")


def main() -> None:
    print(f"Whole-column GCD benchmark ({RUNS} runs averaged)")
    run_shape("Shape 1 — narrow/long", make_multiples(10_000_000, 4), with_math=False)
    run_shape("Shape 2 — wide", make_multiples(1_000_000, 100), with_math=False)
    run_shape("Shape 3 — large_dataset.arrow", pl.read_ipc(DATA_PATH), with_math=True)
    run_shape("Shape 4 — early exit (GCD 1)", make_random(10_000_000, 4), with_math=False)


if __name__ == "__main__":
    main()
