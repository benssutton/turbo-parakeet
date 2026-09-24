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
import subprocess
import sys
import textwrap
import weakref

import polars as pl
import pytest

from analytics.similarity import Similarity, SimilarityExactLRU
from datagen import containment_pairs, related_frames, similar_frames
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I2 / known limitation (CLAUDE.md Next Steps): the LSH candidate threshold is "
        "calibrated against Jaccard, so a containment pair (overlap == 1.0) with low "
        "Jaccard — small set fully inside a much larger one — is pruned before "
        "verification ever runs. The threshold heuristic is intentionally unchanged "
        "(controller ruling: behaviour-preserving refactor); this test documents the "
        "gap so a future recall fix makes it start passing (and stops being xfail)."
    ),
)
@pytest.mark.parametrize("impl", OTHERS)
def test_recall_on_containment_pairs(impl):
    frames = containment_pairs()
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


# ── optional-dependency import isolation ────────────────────────────────────

def test_import_and_use_without_datasketch():
    """analytics.similarity must import, and MinHashRust/SimilarityExactLRU must
    work, when datasketch cannot be imported (it's an optional reference impl,
    lazily imported by MinHashDatasketch only). Run in a subprocess so faking the
    ImportError can't leak into other tests via sys.modules."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["datasketch"] = None  # forces ImportError on `import datasketch`
        import polars as pl
        import analytics.similarity as sim

        frames = {"x": pl.DataFrame({"a": [1, 2, 3]}), "y": pl.DataFrame({"a": [1, 2, 4]})}
        assert sim.MinHashRust().add(frames).result().height == 1
        assert sim.SimilarityExactLRU().add({n: f for n, f in frames.items()}).result().height == 1

        try:
            sim.MinHashDatasketch
        except ImportError:
            pass
        else:
            raise SystemExit("MinHashDatasketch should have raised ImportError")
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ── the deliberate LRU cache ────────────────────────────────────────────────

def test_lru_is_per_instance_and_cleared_on_add():
    a, b = SimilarityExactLRU(), SimilarityExactLRU()
    assert a._distinct is not b._distinct
    a.add(SMALL).result()
    # result() clears the cache once it's done (see test_lru_cleared_after_result), so
    # hits/misses from the run just finished are observed via the snapshot it leaves.
    info = a.last_cache_info
    assert info.currsize == 6 and info.hits == 24  # 15 pairs × 2 lookups, 6 misses
    assert a._distinct.cache_info().currsize == 0
    a.add({"df3": pl.DataFrame({"A": [1]})})
    assert a._distinct.cache_info().currsize == 0


def test_lru_cleared_after_result():
    t = SimilarityExactLRU()
    t.add(SMALL).result()
    assert t._distinct.cache_info().currsize == 0
    assert t._distinct.cache_info().hits == 0 and t._distinct.cache_info().misses == 0
    # a second result() must not see a warm/stale cache from the first run
    t.result()
    assert t.last_cache_info.hits == 24 and t.last_cache_info.misses == 6
    assert t._distinct.cache_info().currsize == 0


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
