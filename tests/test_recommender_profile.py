"""Entropy and exemplar columns of both recommenders (spec 2026-10-18 §3, §4)."""

import math
from datetime import date

import polars as pl
import pytest

from analytics import _plugin
from analytics.recommend import OneShotRecommender, StreamingRecommender
from datagen import describe_mixed, large_dataset, large_dataset_mixed

FRAMES = [
    pytest.param(lambda: describe_mixed(2_000), id="describe_mixed"),
    pytest.param(large_dataset, id="large"),
    pytest.param(large_dataset_mixed, id="mixed"),
]


def one_shot(frame: pl.DataFrame, **params) -> dict[str, dict]:
    out = OneShotRecommender(**params).add(frame).result()
    return {r["column"]: r for r in out.iter_rows(named=True)}


@pytest.mark.parametrize("make", FRAMES)
def test_one_shot_entropy_matches_marginal_entropy(make):
    frame = make()
    rows = one_shot(frame)
    computed = [c for c, r in rows.items() if r["status"] == "computed"]
    want = dict(_plugin.marginal_entropy(frame.select(computed)).iter_rows())
    for c in computed:
        r = rows[c]
        assert r["entropy"] == pytest.approx(want[c], rel=1e-12, abs=1e-12), c
        assert (r["entropy_low"], r["entropy_high"]) == (r["entropy"], r["entropy"])
        assert r["entropy_method"] == "exact"


def test_one_shot_entropy_known_answers():
    frame = pl.DataFrame(
        {
            "constant": [7] * 8,
            "distinct": list(range(8)),
            "half_null": [1, 1, 1, 1, None, None, None, None],
        }
    )
    rows = one_shot(frame)
    assert rows["constant"]["entropy"] == 0.0
    assert rows["distinct"]["entropy"] == pytest.approx(3.0)
    assert rows["half_null"]["entropy"] == pytest.approx(1.0)


def test_one_shot_exemplars_known_answers():
    frame = pl.DataFrame(
        {
            "a": [None, 5, None, 6],
            "s": ["x", "y", "z", "w"],
            "l": [[1, 2], None, [None], [3]],
            "n": pl.Series([None, None, None, None], dtype=pl.Int64),
        }
    )
    rows = one_shot(frame)
    assert (rows["a"]["exemplar_row"], rows["a"]["exemplar_value"]) == (1, "5")
    assert rows["a"]["null_exemplar_row"] == 0
    assert (rows["s"]["exemplar_row"], rows["s"]["null_exemplar_row"]) == (0, None)
    assert rows["l"]["exemplar_value"] == "[1, 2]"
    assert rows["l"]["null_exemplar_row"] == 1  # top-level nulls only
    assert (rows["n"]["exemplar_row"], rows["n"]["exemplar_value"]) == (None, None)
    assert rows["n"]["null_exemplar_row"] == 0


def test_exemplar_value_matches_min_max_rendering_for_scalars():
    frame = pl.DataFrame({"x": [2.5], "d": [date(2024, 1, 2)], "s": ["hi"]})
    for r in one_shot(frame).values():
        assert r["exemplar_value"] == r["min"], r["column"]


@pytest.mark.parametrize("batch_rows", [1, 3, 1_000])
def test_streaming_exemplars_and_exact_entropy_equal_one_shot(batch_rows):
    frame = pl.DataFrame(
        {
            "a": [None, None, None, 5, None, 6] * 3,
            "s": ["x", "y", None, "z", "w", "v"] * 3,
            "l": [[1], None, [2, None], [3], [], None] * 3,
        }
    )
    want = one_shot(frame)
    rec = StreamingRecommender()
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    for r in rec.result().iter_rows(named=True):
        o = want[r["column"]]
        for k in [
            "exemplar_row",
            "exemplar_value",
            "null_exemplar_row",
            "entropy_method",
        ]:
            assert r[k] == o[k], (r["column"], k)
        assert r["entropy"] == pytest.approx(o["entropy"], rel=1e-12, abs=1e-12)


def test_streaming_late_and_missing_columns_count_as_null():
    rec = StreamingRecommender()
    rec.add(pl.DataFrame({"a": [1, 2]}))
    rec.add(pl.DataFrame({"a": [3], "b": ["x"]}))
    rows = {r["column"]: r for r in rec.result().iter_rows(named=True)}
    assert (rows["b"]["exemplar_row"], rows["b"]["null_exemplar_row"]) == (2, 0)
    assert rows["b"]["entropy"] == pytest.approx(
        -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    )
