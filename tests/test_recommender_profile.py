"""Entropy and exemplar columns of both recommenders."""

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from turbo_parakeet import _plugin
from turbo_parakeet.recommenders import OneShotRecommender, StreamingRecommender
from synthdata.synth import synth_small, synth_windows
from synthdata.scenarios import describe_mixed

FRAMES = [
    pytest.param(lambda: describe_mixed(2_000), id="describe_mixed"),
    pytest.param(synth_small, id="cover"),
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


def zipf_frame(n: int, d: int, alpha: float, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, d + 1) ** alpha
    return pl.DataFrame({"x": rng.choice(d, size=n, p=w / w.sum())})


STREAMS = [
    pytest.param(
        lambda: pl.DataFrame(
            {"x": np.random.default_rng(0).integers(0, 200_000, 400_000)}
        ),
        id="uniform",
    ),
    pytest.param(lambda: zipf_frame(400_000, 200_000, 1.1), id="zipf1.1"),
]


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["sample", "ams"])
@pytest.mark.parametrize("make", STREAMS)
def test_streaming_entropy_after_overflow(make, estimator):
    frame = make()
    exact = one_shot(frame)["x"]["entropy"]
    rec = StreamingRecommender(entropy_estimator=estimator)
    for off in range(0, frame.height, 50_000):
        rec.add(frame.slice(off, 50_000))
    r = rec.result().row(0, named=True)
    assert r["entropy_method"] == estimator
    # The sample estimator is weak on skewed columns (heavy values outside the
    # sample); only its interval coverage is asserted there (the AMS estimator must meet the 5% bar).
    if not (estimator == "sample" and make is STREAMS[1].values[0]):
        assert r["entropy"] == pytest.approx(exact, rel=0.05)
    assert r["entropy_low"] <= exact <= r["entropy_high"]


def test_entropy_estimator_is_validated():
    with pytest.raises(ValueError, match="entropy_estimator"):
        StreamingRecommender(entropy_estimator="exact")


@pytest.mark.slow
def test_ams_streaming_entropy_on_cover_windows():
    frame = synth_windows()
    exact = {
        r["column"]: r
        for r in OneShotRecommender().add(frame).result().iter_rows(named=True)
    }
    rec = StreamingRecommender(entropy_estimator="ams")
    for off in range(0, frame.height, 10_000):
        rec.add(frame.slice(off, 10_000))
    seen = set()
    for r in rec.result().iter_rows(named=True):
        if r["status"] != "computed":
            continue
        seen.add(r["entropy_method"])
        h = exact[r["column"]]["entropy"]
        assert r["entropy_method"] in ("exact", "ams"), r["column"]
        assert r["entropy_low"] - 1e-9 <= h <= r["entropy_high"] + 1e-9, r["column"]
        assert r["entropy"] == pytest.approx(h, rel=0.05, abs=1e-9), r["column"]
    assert "ams" in seen  # the float / string columns overflow the 10 000-value sample


def test_exemplar_ids_keep_their_types_and_nulls():
    frame = pl.DataFrame(
        {
            "order": [10, 11, None, 13],
            "line": ["a", "b", "c", "d"],
            "price": [None, 1.5, None, 2.5],
        }
    )
    for cls in (OneShotRecommender, StreamingRecommender):
        out = cls(id_columns=["order", "line"]).add(frame).result()
        assert out.schema["exemplar_id"] == pl.Struct(
            {"order": pl.Int64, "line": pl.String}
        )
        price = out.filter(pl.col("column") == "price").row(0, named=True)
        assert price["exemplar_id"] == {"order": 11, "line": "b"}
        assert price["null_exemplar_id"] == {"order": 10, "line": "a"}
        order = out.filter(pl.col("column") == "order").row(0, named=True)
        assert order["null_exemplar_id"] == {"order": None, "line": "c"}
        line = out.filter(pl.col("column") == "line").row(0, named=True)
        assert line["null_exemplar_id"] is None


def test_without_id_columns_the_id_columns_are_null_typed():
    out = OneShotRecommender().add(pl.DataFrame({"a": [1, None]})).result()
    assert out.schema["exemplar_id"] == pl.Null
    assert out.schema["null_exemplar_id"] == pl.Null


@pytest.mark.parametrize("batch_rows", [1, 2, 5])
def test_streaming_ids_equal_one_shot(batch_rows):
    frame = pl.DataFrame(
        {
            "k": list(range(10)),
            "a": [None, None, 1, None, 2] * 2,
            "s": ["x"] * 3 + [None] * 7,
        }
    )
    want = {
        r["column"]: r
        for r in OneShotRecommender(id_columns=["k"])
        .add(frame)
        .result()
        .iter_rows(named=True)
    }
    rec = StreamingRecommender(id_columns=["k"])
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    for r in rec.result().iter_rows(named=True):
        for k in ["exemplar_id", "null_exemplar_id"]:
            assert r[k] == want[r["column"]][k], (r["column"], k)


def test_streaming_late_column_null_exemplar_takes_row_zero_ids():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [100, 101]}))
    rec.add(pl.DataFrame({"k": [102], "b": ["x"]}))
    b = rec.result().filter(pl.col("column") == "b").row(0, named=True)
    assert (b["exemplar_id"], b["null_exemplar_id"]) == ({"k": 102}, {"k": 100})


def test_streaming_absent_column_null_exemplar_takes_the_batch_start_ids():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [100, 101], "a": [1, 2]}))
    rec.add(pl.DataFrame({"k": [102, 103]}))
    a = rec.result().filter(pl.col("column") == "a").row(0, named=True)
    assert (a["exemplar_id"], a["null_exemplar_id"]) == ({"k": 100}, {"k": 102})


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
def test_id_column_errors(cls):
    rec = cls(id_columns=["missing"])
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="not in the input"):
        rec.add(frame)
    with pytest.raises(ValueError, match="duplicate id column"):
        cls(id_columns=["a", "a"])


def test_streaming_id_column_missing_from_a_later_batch():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [1], "a": [2]}))
    later = pl.DataFrame({"a": [3]})
    with pytest.raises(ValueError, match="not in the input"):
        rec.add(later)


def test_streaming_failed_add_leaves_the_result_unchanged():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [1, 2], "a": [None, 2]}))
    before = rec.result()
    bad = pl.DataFrame({"a": [3, None]})
    with pytest.raises(ValueError):
        rec.add(bad)
    assert rec.result().equals(before)
    rec.add(pl.DataFrame({"k": [3], "a": [4]}))
    assert rec.result().row(0, named=True)["n_rows"] == 3


def test_streaming_id_column_type_change_is_rejected():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [1], "a": [2]}))
    changed = pl.DataFrame({"k": ["x"], "a": [3]})
    with pytest.raises(ValueError, match="type changed"):
        rec.add(changed)


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
def test_ineligible_id_column_is_rejected(cls):
    frame = pl.DataFrame({"k": pl.Series([1, 2], dtype=pl.Int128), "a": [1, 2]})
    rec = cls(id_columns=["k"])
    with pytest.raises(ValueError, match="ineligible"):
        rec.add(frame)
