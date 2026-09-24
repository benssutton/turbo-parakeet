"""
Seeded toy-data generators shared by accuracy tests (small sizes) and speed
benchmarks (large sizes). Every generator is deterministic for a given seed.

Ordered techniques:   mixed_dtypes, low_cardinality
Multi-set techniques: related_frames, similar_frames
Per-column (GCD):     integer_multiples, integer_random
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa


def _with_nulls(name: str, values: np.ndarray, null_mask: np.ndarray) -> pl.Series:
    return pl.from_arrow(pa.array(values, mask=null_mask)).alias(name)


# ── ordered techniques ────────────────────────────────────────────────────────

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
    positions = rng.choice(n, size=min(200, n), replace=False)
    tokens[positions] = np.arange(len(positions))
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
    return pl.Series(name, values, dtype=pl.Array(pl.Int32, 3))


def mixed_dtypes(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame:
    """18 columns (6 dtypes × 3 shapes) named {dtype_spec}_{shape}, e.g. categorical_sparse."""
    rng = np.random.default_rng(seed)
    cols = []
    for dtype_spec in ("float64", "uint32", "boolean", "categorical", "list", "arr"):
        for shape in ("skewed", "high_unique", "sparse"):
            cols.append(make_column(f"{dtype_spec}_{shape}", dtype_spec, shape, n_rows, rng))
    return pl.DataFrame(cols)


def low_cardinality(
    n_rows: int, n_cols: int, cardinality: int = 20, null_rate: float = 0.05, seed: int = 42
) -> pl.DataFrame:
    """Int64 columns c000… with `cardinality` values each. Every 5th column is a noisy
    copy of the previous one, so some pairs are genuinely associated."""
    rng = np.random.default_rng(seed)
    cols, prev = [], None
    for i in range(n_cols):
        if prev is not None and i % 5 == 4:
            values = (prev + rng.integers(0, 2, n_rows)) % cardinality
        else:
            values = rng.integers(0, cardinality, n_rows)
        prev = values
        cols.append(_with_nulls(f"c{i:03d}", values, rng.random(n_rows) < null_rate))
    return pl.DataFrame(cols)


# ── multi-set techniques ──────────────────────────────────────────────────────

def related_frames(n_rows: int = 1_000, seed: int = 42) -> dict[str, pl.DataFrame]:
    """Three frames with planted relationships (enumeration order customers, orders, archive):

        customers.id     ~ orders.customer_id    pk_fk   (FK, 10% null)
        customers.id     ~ archive.id            pk_pk   (same id set, shuffled)
        customers.name   ~ orders.customer_name  pk_fk
        customers.name   ~ archive.name          ~93% shared - similar, not contained
        customers.region ~ orders.region         b_in_a  (orders uses 2 of 4 regions)
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_rows) + 1_000_000
    names = [f"cust_{i}" for i in ids]
    regions = np.array(["north", "south", "east", "west"])
    customers = pl.DataFrame(
        {
            "id": ids,
            "name": names,
            "region": rng.choice(regions, n_rows),
            "score": rng.integers(0, 100, n_rows),
        }
    )
    pick = rng.integers(0, n_rows, n_rows)
    orders = pl.DataFrame(
        [
            pl.Series("order_id", np.arange(n_rows) + 5_000_000),
            _with_nulls("customer_id", ids[pick], rng.random(n_rows) < 0.10),
            pl.Series("customer_name", [names[i] for i in pick]),
            pl.Series("region", rng.choice(regions[:2], n_rows)),
            pl.Series("amount", rng.integers(1, 500, n_rows)),
        ]
    )
    archive_names = [names[i] for i in rng.permutation(n_rows)]
    for i in np.flatnonzero(rng.random(n_rows) < 0.07):
        archive_names[i] = f"new_{i}"
    archive = pl.DataFrame({"id": rng.permutation(ids), "name": archive_names})
    return {"customers": customers, "orders": orders, "archive": archive}


def containment_pairs(
    n_pairs: int = 20, small_size: int = 5, large_size: int = 200, seed: int = 42
) -> dict[str, pl.DataFrame]:
    """`n_pairs` column pairs (small.small_i, large.large_i) where the small column's
    values are a subset of the large column's — overlap == 1.0 (fully contained) but
    Jaccard = small_size / large_size (~0.025 at the defaults), far below the MinHash
    LSH candidate threshold (~0.43). Each pair draws from its own disjoint numeric
    range, so off-diagonal pairs (small_i vs large_j, i != j; small_i vs small_j) are
    ~disjoint and contribute no accidental truth positives.

    Exercises the documented containment-recall gap on MinHashRust/MinHashDatasketch
    (CLAUDE.md Next Steps / I2): the LSH candidate threshold is Jaccard-calibrated, so
    a small-in-large containment pair with low Jaccard is pruned before verification
    ever runs, even though its Overlap Coefficient is 1.0.
    """
    rng = np.random.default_rng(seed)
    span = large_size * 10
    small_cols, large_cols = {}, {}
    for i in range(n_pairs):
        base = i * span
        pool = rng.choice(np.arange(base, base + span), size=large_size, replace=False)
        small = rng.choice(pool, size=small_size, replace=False)
        small_cols[f"small_{i:02d}"] = small.tolist()
        large_cols[f"large_{i:02d}"] = pool.tolist()
    return {"small": pl.DataFrame(small_cols), "large": pl.DataFrame(large_cols)}


def similar_frames(
    n_similar: int = 8,
    n_independent: int = 12,
    col_size: int = 200,
    n_elements: int = 2_000,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Two frames each with (n_similar + n_independent) columns. Each of the n_similar
    cross-frame pairs sim_ii shares >= 93% of elements (OC above 0.9). Independent
    columns come from disjoint regions of the element space to minimise accidental
    similarity. Shorter columns are null-padded."""
    rng = np.random.default_rng(seed)
    df0: dict[str, list] = {}
    df1: dict[str, list] = {}
    for i in range(n_similar):
        base = rng.choice(n_elements, size=col_size, replace=False).tolist()
        shared = int(col_size * 0.93)
        extra = rng.integers(0, n_elements, size=col_size - shared).tolist()
        df0[f"sim_{i:02d}"] = base
        df1[f"sim_{i:02d}"] = base[:shared] + extra
    region = n_elements // max(n_independent, 1)
    for i in range(n_independent):
        lo = (i * region) % n_elements
        pool = list(range(lo, min(lo + region, n_elements))) or list(range(n_elements))
        size = int(min(len(pool), rng.integers(50, col_size)))
        df0[f"ind0_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()
        df1[f"ind1_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()

    def to_frame(cols: dict[str, list]) -> pl.DataFrame:
        longest = max(len(v) for v in cols.values())
        return pl.DataFrame({k: v + [None] * (longest - len(v)) for k, v in cols.items()})

    return {"df0": to_frame(df0), "df1": to_frame(df1)}


# ── per-column (GCD) ──────────────────────────────────────────────────────────

def integer_multiples(n_rows: int, n_cols: int, g: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… whose values are k·g, k uniform in [-2**40, 2**40)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * g for i in range(n_cols)})


def integer_random(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… of random values; whole-column GCD is 1 almost surely (early exit)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64) for i in range(n_cols)})
