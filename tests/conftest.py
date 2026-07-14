import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analytics"))

import numpy as np
import polars as pl
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow — deselect with '-m \"not slow\"'",
    )


N_ROWS: int = 1_000  # Change here to scale the shared test dataset


def _skewed_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """30 tokens drawn with power-law weights (heavy skew, ~30 distinct values)."""
    k = np.arange(1, 31)
    w = 1.0 / k ** 0.8
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
    positions = rng.choice(n, size=200, replace=False)
    tokens[positions] = np.arange(200)
    null_mask = np.ones(n, dtype=bool)
    null_mask[positions] = False
    return tokens, null_mask


def make_column(
    name: str,
    dtype_spec: str,
    shape: str,
    n_rows: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Build one pl.Series with the given dtype and distribution shape.

    dtype_spec : "float64" | "uint32" | "boolean" | "categorical" | "list" | "arr"
    shape      : "skewed" | "high_unique" | "sparse"
    """
    if shape == "sparse":
        tokens, null_mask = _sparse_tokens(n_rows, rng)
    else:
        tokens = _skewed_tokens(n_rows, rng) if shape == "skewed" else _high_unique_tokens(n_rows, rng)
        null_rate = 0.10 if shape == "skewed" else 0.05
        null_mask = rng.random(n_rows) < null_rate

    # Verify cardinality for high_unique (boolean only has 2 values — skip it)
    if shape == "high_unique" and dtype_spec != "boolean":
        non_null = tokens[~null_mask]
        n_distinct = len(np.unique(non_null))
        assert n_distinct / n_rows >= 0.85, (
            f"{name}: expected ≥85% distinct values, got {n_distinct}/{n_rows}"
        )

    values: list = []
    for i in range(n_rows):
        if null_mask[i]:
            values.append(None)
            continue
        t = int(tokens[i])
        if dtype_spec == "float64":
            values.append(float(t))
        elif dtype_spec == "uint32":
            values.append(t)
        elif dtype_spec == "boolean":
            values.append(bool(t % 2))
        elif dtype_spec == "categorical":
            values.append(f"cat_{t}")
        elif dtype_spec == "list":
            values.append([t, t + 1])
        else:  # arr
            values.append([t, t + 1, t + 2])

    if dtype_spec == "float64":
        return pl.Series(name, values, dtype=pl.Float64)
    if dtype_spec == "uint32":
        return pl.Series(name, values, dtype=pl.UInt32)
    if dtype_spec == "boolean":
        return pl.Series(name, values, dtype=pl.Boolean)
    if dtype_spec == "categorical":
        return pl.Series(name, values, dtype=pl.String).cast(pl.Categorical)
    if dtype_spec == "list":
        return pl.Series(name, values, dtype=pl.List(pl.Int32))
    # arr
    return pl.Series(name, values, dtype=pl.Array(pl.Int32, 3))


def make_dataset(n_rows: int = N_ROWS, seed: int = 42) -> pl.DataFrame:
    """Build the shared 18-column test DataFrame (6 dtypes × 3 shapes).
       Columns are named {dtype_spec}_{shape}, e.g. float64_skewed, cat_sparse.
    """
    rng = np.random.default_rng(seed)
    cols = []
    for dtype_spec in ("float64", "uint32", "boolean", "categorical", "list", "arr"):
        for shape in ("skewed", "high_unique", "sparse"):
            cols.append(make_column(f"{dtype_spec}_{shape}", dtype_spec, shape, n_rows, rng))
    return pl.DataFrame(cols)


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """18-column DataFrame shared across all test modules in the session."""
    return make_dataset()
