"""Skip-index recommendations (spec 2026-10-20): keyword validation and one-shot known answers."""

import numpy as np
import polars as pl
import pytest

from analytics.recommend import OneShotRecommender, StreamingRecommender

BAD = [
    {"skip_granularities": ()},
    {"skip_granularities": (1, 1)},
    {"skip_granularities": (0,)},
    {"set_max_sizes": (0,)},
    {"bloom_fprs": (0.0,)},
    {"bloom_fprs": (1.0,)},
    {"min_skip": 1.5},
    {"max_overflow": -0.1},
]


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
@pytest.mark.parametrize(
    "kwargs", BAD, ids=lambda k: next(iter(k)) + str(next(iter(k.values())))
)
def test_keywords_are_validated(cls, kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        cls(**kwargs)


N = 200_000


def recommend(values, dtype=None, **kw):
    s = pl.Series("x", values)
    if dtype is not None:
        s = s.cast(dtype)
    row = OneShotRecommender(**kw).add(pl.DataFrame([s])).result().row(0, named=True)
    return row["clickhouse_skip_index"], row["clickhouse_skip_indexes"]


def show(rows):
    return "\n".join(str(r) for r in rows)


def sorted_ids():
    return np.arange(N)


def clustered():
    rng = np.random.default_rng(1)
    return np.repeat(rng.permutation(200) * 1_000_003 % 2**31, 1_000)


def random_50():
    return np.random.default_rng(2).integers(0, 50, N)


def grouped_hashes(bits=62):
    return np.repeat(np.random.default_rng(3).integers(0, 2**bits, N // 10), 10)


def test_sorted_unique_column_gets_minmax():
    index, rows = recommend(sorted_ids())
    assert index is not None and index.startswith("minmax"), show(rows)


def test_clustered_high_cardinality_gets_set():
    index, rows = recommend(clustered())
    assert index is not None and index.startswith("set(125)"), show(rows)


def test_random_order_gets_no_index():
    index, rows = recommend(random_50())
    assert index is None, show(rows)
    assert rows and all(r["outcome"] in ("rejected", "pruned") for r in rows), show(
        rows
    )


def test_grouped_random_hashes_get_bloom():
    index, rows = recommend(grouped_hashes())
    assert index is not None and index.startswith("bloom_filter(0.025)"), show(rows)


def test_decimal_column_has_no_bloom_candidate():
    # scale 2: a scale-0 Decimal is narrowed to UInt64 by the type rules, which does take a bloom filter
    values = pl.Series("x", (grouped_hashes(56) % 10**17) / 100.0).cast(
        pl.Decimal(18, 2)
    )
    index, rows = recommend(values)
    bloom = [r for r in rows if r["kind"] == "bloom_filter"]
    assert all(r["outcome"] == "pruned" for r in bloom), show(rows)
    assert bloom and all("R36" in r["evidence"] for r in bloom), show(rows)
    assert index is None or not index.startswith("bloom_filter")


def test_min_skip_moves_the_boundary():
    index, rows = recommend(sorted_ids(), min_skip=0.99)
    assert index is None, show(rows)


def test_skip_granularities_limit_the_rows():
    _, rows = recommend(sorted_ids(), skip_granularities=(1,))
    measured = [r for r in rows if r["skip_rows"] is not None]
    assert measured and all(r["granularity"] == 1 for r in measured), show(rows)


def test_set_max_sizes_above_the_window_distinct_leave_the_gate():
    # ~820 distinct per window: set(1000) does not overflow, so the gate (not the overflow rule) decides
    _, rows = recommend(
        grouped_hashes(), skip_granularities=(1,), set_max_sizes=(1000,)
    )
    sets = [r for r in rows if r["kind"] == "set"]
    assert sets and any(r["outcome"] != "pruned" for r in sets), show(rows)
