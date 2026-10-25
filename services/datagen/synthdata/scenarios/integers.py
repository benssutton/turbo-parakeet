"""Integer columns with planted GCDs, random values and planted runs."""

from __future__ import annotations

import numpy as np
import polars as pl


def integer_multiples(n_rows: int, n_cols: int, g: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… whose values are k·g, k uniform in [-2**40, 2**40)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * g
            for i in range(n_cols)
        }
    )


def integer_random(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… of random values; whole-column GCD is 1 almost surely (early exit)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64)
            for i in range(n_cols)
        }
    )


def _run_lengths(n_rows: int, mean_run: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric run lengths with mean `mean_run`, summing to at least `n_rows`."""
    lengths = rng.geometric(1 / mean_run, size=int(n_rows // mean_run) + 64)
    while lengths.sum() < n_rows:
        more = rng.geometric(1 / mean_run, size=int(n_rows // mean_run) + 64)
        lengths = np.concatenate([lengths, more])
    return lengths


def runs_int(
    n_rows: int,
    n_cols: int,
    mean_run: int,
    cardinality: int,
    prefix: str = "c",
    seed: int = 42,
) -> pl.DataFrame:
    """Int64 columns {prefix}0… made of runs: geometric run lengths with mean `mean_run`
    (1: every row its own draw), each run's value uniform in [0, cardinality). Adjacent
    runs drawing the same value merge (probability 1 / cardinality)."""
    rng = np.random.default_rng(seed)
    columns = {}
    for i in range(n_cols):
        lengths = _run_lengths(n_rows, mean_run, rng)
        values = rng.integers(0, cardinality, lengths.size, dtype=np.int64)
        columns[f"{prefix}{i}"] = np.repeat(values, lengths)[:n_rows]
    return pl.DataFrame(columns)


def runs_str(
    n_rows: int,
    n_cols: int,
    mean_run: int,
    cardinality: int,
    prefix: str = "s",
    seed: int = 42,
) -> pl.DataFrame:
    """runs_int's runs as String values "v<k>"."""
    df = runs_int(n_rows, n_cols, mean_run, cardinality, prefix, seed)
    return df.with_columns(
        ("v" + pl.col(c).cast(pl.String)).alias(c) for c in df.columns
    )
