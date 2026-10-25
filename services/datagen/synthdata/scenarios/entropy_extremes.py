"""Near-unique and near-constant frames and their token generators."""

from __future__ import annotations

import numpy as np
import polars as pl


def _skewed_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """30 tokens drawn with power-law weights (heavy skew, ~30 distinct values)."""
    k = np.arange(1, 31)
    w = 1.0 / k**0.8
    w /= w.sum()
    return rng.choice(30, size=n, p=w)


def _high_unique_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """All n tokens distinct (random permutation)."""
    return rng.permutation(n)


def _sparse_tokens(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """200 unique tokens at random positions; remaining positions forced null.
    Returns (tokens, null_mask) — tokens[i] is only valid when null_mask[i]=False.
    """
    tokens = np.empty(n, dtype=np.int64)
    positions = rng.choice(n, size=min(200, n), replace=False)
    tokens[positions] = np.arange(len(positions))
    null_mask = np.ones(n, dtype=bool)
    null_mask[positions] = False
    return tokens, null_mask


def near_unique(n_rows: int = 1_000_000, seed: int = 42) -> pl.DataFrame:
    """Int64 columns that stress entropy sketches and the exact kernels' hash-map path:
    u0–u2 near-unique (a permutation with ~0.1% of rows copying another row's value), a
    genuine composite key key_hi = i // 1000, key_lo = i % 1000 over shuffled i (unique
    together), `skewed` (95% one value, the rest over 50,000 others), `mid` / `low`
    with ~1,000 / 10 values, and `flag`, a near-constant 0/1 column with one 1 per
    10,000 rows."""
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(3):
        values = rng.permutation(n_rows)
        dup = rng.choice(n_rows, n_rows // 1000, replace=False)
        values[dup] = values[rng.integers(0, n_rows, dup.size)]
        cols[f"u{i}"] = values
    perm = rng.permutation(n_rows)
    cols["key_hi"] = perm // 1000
    cols["key_lo"] = perm % 1000
    cols["skewed"] = np.where(
        rng.random(n_rows) < 0.95, 0, rng.integers(1, 50_001, n_rows)
    )
    cols["mid"] = rng.integers(0, 1_000, n_rows)
    cols["low"] = rng.integers(0, 10, n_rows)
    cols["flag"] = (np.arange(n_rows) % 10_000 == 7).astype(np.int64)
    return pl.DataFrame({k: v.astype(np.int64) for k, v in cols.items()})


def near_constant(n_rows: int = 1_000_000, n_cols: int = 3) -> pl.DataFrame:
    """Int64 columns f0... where column i is 1 on rows with row % 10_000 == 7 + 1_000 * i,
    else 0: deterministic, and every pair and triplet is a near-constant joint stream.
    """
    rows = np.arange(n_rows)
    return pl.DataFrame(
        {
            f"f{i}": (rows % 10_000 == 7 + 1_000 * i).astype(np.int64)
            for i in range(n_cols)
        }
    )
