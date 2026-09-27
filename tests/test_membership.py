"""
Set-membership / containment accuracy tests — every implementation in analytics.membership.

Reference: MembershipExact (Python sets of distinct non-null values). Bloom
implementations are probabilistic (EXACT=False): no false negatives, exact
distinct/non-null counts, and an aggregate false-positive rate within
FP_TOLERANCE × fp_rate of exact containment.
"""

import math
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from analytics import _plugin
from analytics.membership import BloomMembership, Membership
from analytics.membership.rust import bloom_geometry
from datagen import mixed_dtypes, related_frames
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.membership"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = related_frames(200)
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: related_frames(1_000), id="related"),
        pytest.param(lambda: {"x": mixed_dtypes(300), "y": mixed_dtypes(300, seed=7)}, id="mixed"),
    ],
)
def test_agrees_with_reference(impl, make):
    frames = make()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def test_reference_known_containment():
    frames = {"p": pl.DataFrame({"id": [1, 2, 3, 4]}), "c": pl.DataFrame({"pid": [1, 1, 2, None]})}
    r = run(reference(PKG), frames).row(0, named=True)
    assert (r["ratio_a_in_b"], r["ratio_b_in_a"]) == (0.5, 1.0)
    assert (r["n_distinct_a"], r["n_distinct_b"], r["n_non_null_a"], r["n_non_null_b"]) == (4, 2, 4, 3)
    assert (r["unique_a"], r["unique_b"], r["relationship"]) == (True, False, "pk_fk")


@pytest.mark.parametrize("impl", ALL)
def test_planted_relationships(impl):
    out = run(load(impl), related_frames(1_000))
    rel = {(r["df_a"], r["col_a"], r["df_b"], r["col_b"]): r["relationship"] for r in out.iter_rows(named=True)}
    assert rel[("customers", "id", "orders", "customer_id")] == "pk_fk"
    assert rel[("customers", "id", "archive", "id")] == "pk_pk"
    assert rel[("customers", "name", "orders", "customer_name")] == "pk_fk"
    assert rel[("customers", "region", "orders", "region")] == "b_in_a"
    assert rel[("customers", "name", "archive", "name")] == "none"  # ~93% shared: similar, not contained
    assert rel[("customers", "id", "customers", "name")] is None  # Int64 vs String: ineligible


@pytest.mark.parametrize("impl", ALL)
def test_value_families(impl):
    df = pl.DataFrame(
        {
            "i32": pl.Series([1, 2], dtype=pl.Int32),
            "i64": pl.Series([1, 2], dtype=pl.Int64),
            "d": pl.Series([1, 2], dtype=pl.Int32).cast(pl.Date),
            "s": ["1", "2"],
            "f": [1.0, 2.0],
        }
    )
    out = run(load(impl), {"t": df})
    done = out.filter(pl.col("status") == "computed")
    assert done.select("col_a", "col_b").rows() == [("i32", "i64")]
    assert done.row(0, named=True)["ratio_a_in_b"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_empty_column(impl):
    df = pl.DataFrame({"e": pl.Series([None, None], dtype=pl.Int64), "v": [1, 2]})
    r = run(load(impl), {"t": df}).row(0, named=True)
    assert r["status"] == "computed"
    assert math.isnan(r["ratio_a_in_b"]) and r["ratio_b_in_a"] == 0.0
    assert r["unique_a"] is False and r["relationship"] == "none"


@pytest.mark.parametrize("impl", ALL)
def test_categorical_values_match_by_label_across_frames(impl):
    shared = [f"tok_{i}" for i in range(50)]
    pad = [f"pad_{i}" for i in range(50)]
    frames = {
        "f": pl.DataFrame({"c": pl.Series(shared).cast(pl.Categorical)}),
        "q": pl.DataFrame({"c": pl.Series(pad + shared).cast(pl.Categorical)}),
    }
    assert run(load(impl), frames).row(0, named=True)["ratio_a_in_b"] == 1.0


@pytest.mark.parametrize("impl", OTHERS)
def test_false_positive_rate_on_disjoint_sets(impl):
    frames = {"a": pl.DataFrame({"v": list(range(2_000))}), "b": pl.DataFrame({"w": list(range(10_000, 12_000))})}
    r = run(load(impl), frames).row(0, named=True)
    assert r["ratio_a_in_b"] <= 0.03 and r["ratio_b_in_a"] <= 0.03


@pytest.mark.parametrize("impl", OTHERS)
def test_fp_rate_validation(impl):
    with pytest.raises(ValueError):
        load(impl)(fp_rate=0.0)


# ── optional-dependency import isolation ────────────────────────────────────

def test_import_and_use_without_fastbloom_rs():
    """analytics.membership must import, and BloomRust/MembershipExact must work,
    when fastbloom_rs cannot be imported (it's an optional reference impl, lazily
    imported by BloomFastbloom only). Run in a subprocess so faking the ImportError
    can't leak into other tests via sys.modules."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["fastbloom_rs"] = None  # forces ImportError on `import fastbloom_rs`
        import polars as pl
        import analytics.membership as mem

        frames = {"x": pl.DataFrame({"a": [1, 2, 3]}), "y": pl.DataFrame({"a": [1, 2, 4]})}
        assert mem.BloomRust().add(frames).result().height == 1
        assert mem.MembershipExact().add({n: f for n, f in frames.items()}).result().height == 1

        try:
            mem.BloomFastbloom
        except ImportError:
            pass
        else:
            raise SystemExit("BloomFastbloom should have raised ImportError")
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ── BloomMembership.agreement FP-rate bound ─────────────────────────────────

class _ConcreteBloom(BloomMembership):
    """BloomMembership is abstract (_compute); agreement() never calls it."""

    def _compute(self, frames, combos):
        raise NotImplementedError


def _bloom_frames(ratio: float, n: int):
    Fixed = with_metrics(
        BloomMembership,
        ratio_a_in_b=[ratio],
        ratio_b_in_a=[ratio],
        n_distinct_a=[n],
        n_distinct_b=[n],
        n_non_null_a=[n],
        n_non_null_b=[n],
    )
    return Fixed().add({"t": pl.DataFrame({"a": [1], "b": [1]})}).result()


def test_bloom_agreement_skips_rate_check_below_min_negatives():
    # n=14, 1 false positive per side (13/14 exact vs 14/14 observed): the aggregate
    # rate is 1/14 ~ 0.071, way above FP_TOLERANCE(3) x fp_rate(0.01) = 0.03, but the
    # tiny sample (well under 100 negatives) makes that bound meaningless.
    result, reference = _bloom_frames(1.0, 14), _bloom_frames(13 / 14, 14)
    assert _ConcreteBloom().agreement(result, reference) == []


def test_bloom_agreement_rate_check_still_fires_above_min_negatives():
    # n=1000, ratio 0.9 -> 1.0: 100 negatives per side, aggregate FP rate 1.0 far
    # above the 0.03 bound - large enough that the check must still fire.
    result, reference = _bloom_frames(1.0, 1000), _bloom_frames(0.9, 1000)
    problems = _ConcreteBloom().agreement(result, reference)
    assert any("false-positive rate" in p for p in problems)


# ── binding-level regression: bloom_filter_bits on a multi-chunk Series ─────

def test_bloom_filter_bits_multi_chunk_matches_rechunked():
    """A multi-chunk Series must give the same Bloom bits as its rechunked form:
    bloom_filter_bits concatenates the stream into one batch before hashing, so
    chunking never changes which rows are counted."""
    a = pl.Series("v", list(range(0, 500)))
    b = pl.Series("v", list(range(500, 1000)))
    multi = pl.concat([a, b], rechunk=False)
    assert multi.n_chunks() > 1
    rechunked = multi.rechunk()
    assert rechunked.n_chunks() == 1
    m, k = bloom_geometry(multi.len(), 0.01)
    assert _plugin.bloom_filter_bits(multi, k=k, m=m) == _plugin.bloom_filter_bits(rechunked, k=k, m=m)


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        Membership,
        ratio_a_in_b=[1.0, 1.0, 0.4, 1.0, 0.96, 0.5, 0.5, NAN, 0.0, 0.0],
        ratio_b_in_a=[1.0, 0.4, 1.0, 1.0, 0.5, 0.95, 0.5, NAN, 0.0, 0.0],
        n_distinct_a=[5, 3, 5, 3, 3, 3, 3, 0, 1, 1],
        n_distinct_b=[5, 5, 3, 3, 5, 3, 3, 0, 1, 1],
        n_non_null_a=[5, 6, 5, 6, 6, 6, 3, 0, 1, 1],
        n_non_null_b=[5, 5, 6, 6, 6, 6, 3, 0, 1, 1],
    )
    df = pl.DataFrame({c: [1] for c in "abcde"})  # C(5,2) = 10 pairs
    out = Fixed().add({"t": df}).result()
    assert out["relationship"].to_list() == [
        "pk_pk", "fk_pk", "pk_fk", "mutual", "a_in_b", "b_in_a", "none", "none", "none", "none"
    ]
    assert out["unique_a"].to_list() == [True, False, True, False, False, False, True, False, True, True]
    assert out["unique_b"].to_list() == [True, True, False, False, False, False, True, False, True, True]
    relaxed = Fixed(containment_threshold=0.5).add({"t": df}).result()
    assert relaxed["relationship"][6] == "pk_pk"
