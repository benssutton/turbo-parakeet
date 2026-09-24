"""
Set-similarity (Jaccard / Overlap Coefficient) accuracy tests — every implementation
in analytics.similarity.

Reference: SimilarityExactLRU (brute force over every pair; distinct-value sets
memoised by a per-instance LRU cache). MinHash implementations are probabilistic
(EXACT=False): pairs they evaluate carry exact metrics (verification is exact), the
rest are "pruned", and recall of passing pairs must be >= Similarity.MIN_RECALL.

Run slow tests:   pytest -m slow
"""

import gc
import math
import weakref

import polars as pl
import pytest

from analytics.similarity import Similarity, SimilarityExactLRU
from datagen import related_frames, similar_frames
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.similarity"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")

SMALL = {
    "df1": pl.DataFrame({"A": list(range(1, 11)), "B": list(range(3, 13)), "C": list(range(10, 20))}),
    "df2": pl.DataFrame({"A": list(range(1, 6)), "B": list(range(4, 9)), "C": list(range(10, 15))}),
}


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("make", [pytest.param(lambda: SMALL, id="small"), pytest.param(lambda: related_frames(200), id="related")])
def test_contract(impl, make):
    frames = make()
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("make", [pytest.param(lambda: SMALL, id="small"), pytest.param(lambda: related_frames(1_000), id="related")])
def test_agrees_with_reference(impl, make):
    frames = make()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.slow
@pytest.mark.parametrize("impl", OTHERS)
def test_recall_medium_scale(impl):
    frames = similar_frames()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def test_reference_known_pairs():
    out = run(SimilarityExactLRU, SMALL, overlap_threshold=0.9)
    passing = out.filter(pl.col("passes_jaccard") | pl.col("passes_overlap"))
    assert set(passing.select("df_a", "col_a", "df_b", "col_b").rows()) == {
        ("df1", "A", "df1", "B"),  # Jaccard 8/12
        ("df1", "A", "df2", "A"),  # df2.A ⊂ df1.A
        ("df1", "A", "df2", "B"),
        ("df1", "B", "df2", "B"),
        ("df1", "C", "df2", "C"),
    }
    aa = out.filter((pl.col("col_a") == "A") & (pl.col("df_b") == "df2") & (pl.col("col_b") == "A")).row(0, named=True)
    assert (aa["jaccard"], aa["overlap"]) == (0.5, 1.0)


@pytest.mark.parametrize("impl", ALL)
def test_value_families(impl):
    df = pl.DataFrame({"i32": pl.Series([1, 2], dtype=pl.Int32), "i64": pl.Series([1, 2], dtype=pl.Int64), "s": ["1", "2"]})
    out = run(load(impl), {"t": df})
    assert out.select("col_a", "col_b", "status").rows() == [
        ("i32", "i64", "computed"),
        ("i32", "s", "ineligible"),
        ("i64", "s", "ineligible"),
    ]
    assert out.row(0, named=True)["jaccard"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_pipe_and_shared_names(impl):
    frames = {"x|y": pl.DataFrame({"a|b": [1, 2, 3]}), "z": pl.DataFrame({"a|b": [1, 2, 3]})}
    r = run(load(impl), frames).row(0, named=True)
    assert (r["df_a"], r["col_a"], r["df_b"], r["col_b"]) == ("x|y", "a|b", "z", "a|b")
    assert r["status"] == "computed" and (r["jaccard"], r["overlap"]) == (1.0, 1.0)


@pytest.mark.parametrize("impl", ALL)
def test_empty_column(impl):
    df = pl.DataFrame(
        {"e": pl.Series([None, None], dtype=pl.Int64), "v": [1, 2], "e2": pl.Series([None, None], dtype=pl.Int64)}
    )
    rows = {(r["col_a"], r["col_b"]): r for r in run(load(impl), {"t": df}).iter_rows(named=True)}
    ev, ee = rows[("e", "v")], rows[("e", "e2")]
    assert ev["status"] == ee["status"] == "computed"
    assert ev["jaccard"] == 0.0 and math.isnan(ev["overlap"])
    assert math.isnan(ee["jaccard"]) and math.isnan(ee["overlap"])
    assert not any(r["passes_jaccard"] or r["passes_overlap"] for r in rows.values())


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(Similarity, jaccard=[0.6, 0.59, NAN], overlap=[0.5, 0.95, NAN])
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = Fixed().add({"t": df}).result()
    assert out["passes_jaccard"].to_list() == [True, False, False]
    assert out["passes_overlap"].to_list() == [False, True, False]
    out = Fixed(jaccard_threshold=0.5, overlap_threshold=0.99).add({"t": df}).result()
    assert out["passes_jaccard"].to_list() == [True, True, False]
    assert out["passes_overlap"].to_list() == [False, False, False]


# ── the deliberate LRU cache ────────────────────────────────────────────────

def test_lru_is_per_instance_and_cleared_on_add():
    a, b = SimilarityExactLRU(), SimilarityExactLRU()
    assert a._distinct is not b._distinct
    a.add(SMALL).result()
    info = a._distinct.cache_info()
    assert info.currsize == 6 and info.hits == 24  # 15 pairs × 2 lookups, 6 misses
    a.add({"df3": pl.DataFrame({"A": [1]})})
    assert a._distinct.cache_info().currsize == 0


def test_cache_size_bounds_the_lru():
    t = SimilarityExactLRU(cache_size=2)
    t.add(SMALL).result()
    assert t._distinct.cache_info().maxsize == 2


def test_instances_are_not_kept_alive_by_the_cache():
    t = SimilarityExactLRU()
    t.add(SMALL).result()
    ref = weakref.ref(t)
    del t
    gc.collect()
    assert ref() is None
