"""Frame collections with planted overlap, containment and similarity."""

from __future__ import annotations

import numpy as np
import polars as pl

from synthdata.scenarios.zoo import _with_nulls


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
    (ToDo 12): the LSH candidate threshold is Jaccard-calibrated, so
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
        return pl.DataFrame(
            {k: v + [None] * (longest - len(v)) for k, v in cols.items()}
        )

    return {"df0": to_frame(df0), "df1": to_frame(df1)}
