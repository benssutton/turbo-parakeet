"""Planted codec and skip-index frames: GCD multiples, Delta-friendly sequences and skip-index
layouts (the ClickHouse calibration frames; same seeds and draw order, so at 1 000 000 rows the
values equal the former calibration scripts')."""

import numpy as np
import polars as pl


def gcd_frame(n_rows: int = 1_000_000) -> pl.DataFrame:
    """Multiples of 1000, of 2 and of 3, and second-resolution microsecond timestamps."""
    rng = np.random.default_rng(0)
    multiples = rng.integers(0, 10**9 + 1, n_rows, dtype=np.uint64) * np.uint64(1000)
    even = rng.integers(2**31, 2**32, n_rows, dtype=np.uint64) * np.uint64(2)
    three = rng.integers(0, 2**40 // 3, n_rows, dtype=np.int64) * 3
    secs = rng.integers(0, 2 * 10**9, n_rows, dtype=np.int64)
    ts = pl.Series("ts_seconds", secs * 1_000_000).cast(pl.Datetime("us"))
    return pl.DataFrame(
        {"gcd_1000": multiples, "gcd_even": even, "gcd_three": three}
    ).with_columns(ts)


def delta_frame(n_rows: int = 1_000_000) -> pl.DataFrame:
    """Regular and jittered millisecond timestamps, random walks (Int64 / Int16 / Int32), sorted
    ids, and random Int64 (where every transform is pruned)."""
    rng = np.random.default_rng(1)
    step = np.full(n_rows, 1000, dtype=np.int64)
    jit = step + rng.integers(-5, 6, n_rows)
    gaps = rng.random(n_rows) < 0.01
    jit[gaps] = rng.integers(1000, 60_000, int(gaps.sum()))
    t0 = 1_700_000_000_000
    walk = np.cumsum(rng.integers(-1000, 1001, n_rows), dtype=np.int64)
    ids = np.cumsum(rng.integers(1, 11, n_rows), dtype=np.uint64)
    w16 = np.clip(np.cumsum(rng.integers(-3, 4, n_rows)), -32768, 32767).astype(
        np.int16
    )
    w32 = np.cumsum(rng.integers(-100, 101, n_rows)).astype(np.int32)
    random_ints = rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64)

    def ts(name: str, d: np.ndarray) -> pl.Series:
        return pl.Series(name, t0 + np.cumsum(d), dtype=pl.Int64).cast(
            pl.Datetime("ms")
        )

    return pl.DataFrame(
        [
            ts("ts_regular", step),
            ts("ts_jitter", jit),
            pl.Series("walk", walk),
            pl.Series("ids_sorted", ids),
            pl.Series("walk_i16", w16),
            pl.Series("walk_i32", w32),
            pl.Series("random_ints", random_ints),
        ]
    )


def skip_frame(n_rows: int = 1_000_000) -> pl.DataFrame:
    """Skip-index layouts in stored order: sorted ids, 1000 clustered values (one run each),
    50 random values, and runs of 10 random hashes. `n_rows` must be a multiple of 1000.
    """
    if n_rows % 1000:
        raise ValueError(f"skip_frame rows must be a multiple of 1000, got {n_rows}")
    rng = np.random.default_rng(1)
    clustered = np.repeat(rng.permutation(1000) * 1_000_003 % 2**31, n_rows // 1000)
    grouped = np.repeat(np.random.default_rng(3).integers(0, 2**62, n_rows // 10), 10)
    return pl.DataFrame(
        {
            "sorted_ids": np.arange(n_rows),
            "clustered": clustered,
            "random_50": np.random.default_rng(2).integers(0, 50, n_rows),
            "grouped_hashes": grouped,
        }
    )
