"""Entropy and exemplar columns of both recommenders (spec 2026-10-18 §3, §4)."""

from datetime import date

import polars as pl
import pytest

from analytics import _plugin
from analytics.recommend import OneShotRecommender
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
