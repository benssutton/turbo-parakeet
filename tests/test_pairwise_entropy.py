"""
Pairwise joint entropy / mutual information accuracy tests — every implementation
in analytics.pairwise_entropy.

Reference: PairwiseEntropyPolars (value_counts → entropy(base=2)); null is its own
category. mixed_dtypes: all 18 columns eligible → C(18,2) = 153 pairs.
"""

import polars as pl
import pytest

from analytics.pairwise_entropy import PairwiseEntropy
from datagen import mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.pairwise_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


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
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 2.0))
    assert r["mi"] == pytest.approx(0.0, abs=1e-12) and r["nmi"] == pytest.approx(0.0, abs=1e-12)
    assert r["redundant"] is False and r["near_unique"] is True  # 4 rows, 4 distinct pairs
    assert r["n_rows"] == 4


@pytest.mark.parametrize("impl", ALL)
def test_redundant_columns_and_null_category(impl):
    r = _row(impl, [None, None, 1, 1], ["x", "x", "y", "y"])  # null is a category: a determines b
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 1.0))
    assert r["nmi"] == pytest.approx(1.0) and r["redundant"] is True and r["near_unique"] is False


@pytest.mark.parametrize("impl", ALL)
def test_constant_column_nmi_is_nan(impl):
    r = _row(impl, [1, 1, 1, 1], [0, 1, 0, 1])
    assert r["h_a"] == pytest.approx(0.0, abs=1e-12)
    assert r["nmi"] != r["nmi"] and r["redundant"] is False  # NaN


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
    assert out["near_unique"].to_list() == [True, False, False]  # log2(4) - 0.1 = 1.9; n_rows 1 → False
    assert Fixed(nmi_threshold=0.92).add({"t": df}).result()["redundant"].to_list() == [True, False, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.String)})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
