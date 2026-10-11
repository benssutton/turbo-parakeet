"""
Pairwise joint entropy / mutual information accuracy tests — every implementation
in analytics.pairwise_entropy.

Reference: PairwiseEntropyPolars (value_counts → entropy(base=2)); null is its own
category. mixed_dtypes: all 18 columns eligible → C(18,2) = 153 pairs.
"""

import math

import polars as pl
import pytest

from analytics.pairwise_entropy import PairwiseEntropy
from datagen import mixed_dtypes
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
    with_metrics,
)

PKG = "analytics.pairwise_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


def _sketch(impl) -> bool:
    return impl.endswith("AmsRust")


def _tol(impl) -> dict:
    """Approx kwargs: sketches use their own agreement tolerance, exact kernels pytest's."""
    cls = load(impl)
    return {"rel": cls.RTOL, "abs": cls.ATOL} if _sketch(impl) else {}


def _mi_tol(impl, k: int) -> float:
    """Tolerance around MI / NMI = 0 or 1: sketches carry k entropies' RTOL errors."""
    return load(impl).RTOL * k + 1e-12 if _sketch(impl) else 1e-12


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 153


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def _row(impl, a, b):
    return run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_independent_columns(impl):
    r = _row(impl, [0, 0, 1, 1], [0, 1, 0, 1])
    tol = _tol(impl)
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 2.0), **tol)
    # MI and NMI carry the entropies' errors: RTOL · (h_a + h_b + h_ab).
    mi_tol = _mi_tol(impl, 4)
    assert r["mi"] == pytest.approx(0.0, abs=mi_tol)
    assert r["nmi"] == pytest.approx(0.0, abs=mi_tol)
    assert r["redundant"] is False
    # 4 rows, 4 distinct pairs
    assert r["near_unique"] is True
    assert r["n_rows"] == 4


@pytest.mark.parametrize("impl", ALL)
def test_redundant_columns_and_null_category(impl):
    r = _row(
        impl, [None, None, 1, 1], ["x", "x", "y", "y"]
    )  # null is a category: a determines b
    tol = _tol(impl)
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 1.0), **tol)
    assert r["nmi"] == pytest.approx(1.0, abs=_mi_tol(impl, 3))
    if not _sketch(impl):  # exact kernels; an estimate may land either side of 0.9
        assert r["redundant"] is True
    assert r["near_unique"] is False


@pytest.mark.parametrize("impl", ALL)
def test_constant_column_nmi_is_nan(impl):
    r = _row(impl, [1, 1, 1, 1], [0, 1, 0, 1])
    assert r["h_a"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(r["nmi"])
    assert r["redundant"] is False


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        PairwiseEntropy,
        h_a=[1.0] * 3,
        h_b=[1.0] * 3,
        h_ab=[1.95, 1.5, 1.0],
        mi=[0.05, 0.5, 1.0],
        nmi=[0.95, 0.9, NAN],
        n_rows=[4, 4, 1],
    )
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = Fixed().add({"t": df}).result()
    assert out["redundant"].to_list() == [True, True, False]
    assert out["near_unique"].to_list() == [
        True,
        False,
        False,
    ]  # log2(4) - 0.1 = 1.9; n_rows 1 → False
    assert Fixed(nmi_threshold=0.92).add({"t": df}).result()["redundant"].to_list() == [
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame(
        {"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.String)}
    )
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]


@pytest.mark.parametrize("impl", ALL)
def test_unique_constant_and_composite_key(impl):
    n = 1_000
    df = pl.DataFrame(
        {
            "hi": [i // 10 for i in range(n)],
            "lo": [i % 10 for i in range(n)],
            "u": list(range(n)),
            "k": [7] * n,
        }
    )
    rows = {
        (r["col_a"], r["col_b"]): r
        for r in run(load(impl), {"t": df}).iter_rows(named=True)
    }
    log_n = math.log2(n)
    assert rows[("hi", "lo")]["h_ab"] == pytest.approx(log_n, rel=1e-9)  # composite key
    assert rows[("hi", "lo")]["near_unique"] is True
    assert rows[("u", "k")]["h_a"] == pytest.approx(log_n, rel=1e-9)  # unique column
    assert rows[("u", "k")]["h_b"] == pytest.approx(0.0, abs=1e-12)  # constant column
    assert rows[("u", "k")]["h_ab"] == pytest.approx(log_n, rel=1e-9)


def test_ams_seed_is_reproducible_and_num_samples_checked():
    from polars.testing import assert_frame_equal

    from analytics.pairwise_entropy import PairwiseEntropyAmsRust

    frames = {"t": mixed_dtypes(300)}
    first = run(PairwiseEntropyAmsRust, frames)
    assert_frame_equal(first, run(PairwiseEntropyAmsRust, frames))
    assert not first["h_ab"].equals(run(PairwiseEntropyAmsRust, frames, seed=1)["h_ab"])
    with pytest.raises(ValueError, match="num_samples"):
        PairwiseEntropyAmsRust(num_samples=0)
    with pytest.raises(ValueError, match="num_samples"):
        PairwiseEntropyAmsRust(num_samples=2**20 + 1)
    PairwiseEntropyAmsRust(num_samples=2**20)
    with pytest.raises(ValueError, match="seed"):
        PairwiseEntropyAmsRust(seed=-1)
