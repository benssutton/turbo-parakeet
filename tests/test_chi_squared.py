"""
Pairwise chi-squared accuracy tests — every implementation in analytics.chi_squared.

Reference: ChiSquaredScipy (scipy.stats.chi2_contingency, correction=False, after
dropping rows where either column is null). Eligible columns in mixed_dtypes:
boolean_*, uint32_*, categorical_* (9) → C(9,2) = 36 computed pairs; float64_*,
list_*, arr_* are ineligible.
"""

import math

import polars as pl
import pytest

from analytics.chi_squared import ChiSquared
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

PKG = "analytics.chi_squared"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


# 1. Contract
@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 36


# 2. Reference agreement
@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers
@pytest.mark.parametrize("impl", ALL)
def test_known_2x2_table(impl):
    # [[20, 30], [30, 20]], N=100: every expected count is 25 → χ² = 4 × (5²/25) = 4.
    a = ["x"] * 50 + ["y"] * 50
    b = ["p"] * 20 + ["q"] * 30 + ["p"] * 30 + ["q"] * 20
    row = run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)
    assert row["chi2_stat"] == pytest.approx(4.0)
    assert row["p_value"] == pytest.approx(0.04550026389635842)
    assert row["cramers_v"] == pytest.approx(0.2)
    assert row["low_expected_count"] is False
    assert row["n_valid"] == 100


@pytest.mark.parametrize("impl", ALL)
def test_nulls_dropped_and_constant_column_undefined(impl):
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "y", None],
            "b": ["p", "p", "q", "q", "p"],
            "c": ["k"] * 5,
        }
    )
    rows = {
        (r["col_a"], r["col_b"]): r
        for r in run(load(impl), {"t": df}).iter_rows(named=True)
    }
    assert rows[("a", "b")]["n_valid"] == 4
    assert rows[("a", "b")]["low_expected_count"] is True
    for pair in [("a", "c"), ("b", "c")]:
        r = rows[pair]
        assert math.isnan(r["chi2_stat"])
        assert math.isnan(r["p_value"])
        assert math.isnan(r["cramers_v"])
        assert r["low_expected_count"] is False
        assert r["associated"] is False
    assert rows[("a", "c")]["n_valid"] == 4


# 4. Conclusions
def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        ChiSquared,
        chi2_stat=[1.0] * 3,
        p_value=[0.5] * 3,
        cramers_v=[0.29, 0.3, NAN],
        low_expected_count=[False] * 3,
        n_valid=[10] * 3,
    )
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    assert Fixed().add({"t": df}).result()["associated"].to_list() == [
        False,
        True,
        False,
    ]
    assert Fixed(cramers_v_threshold=0.2).add({"t": df}).result()[
        "associated"
    ].to_list() == [True, True, False]


def test_threshold_validation():
    Fixed = with_metrics(ChiSquared)
    with pytest.raises(ValueError):
        Fixed(cramers_v_threshold=1.5)
    with pytest.raises(ValueError):
        Fixed(max_unique=1)


# Eligibility
@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame(
        {"a": pl.Series([], dtype=pl.String), "b": pl.Series([], dtype=pl.Int64)}
    )
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]


@pytest.mark.parametrize("impl", ALL)
def test_cardinality_cap_and_dtype_eligibility(impl):
    df = pl.DataFrame({"a": [1, 2, 3, 1], "b": [1, 1, 2, 2], "f": [1.0, 2.0, 3.0, 4.0]})
    out = run(load(impl), {"t": df}, max_unique=2)
    assert out.select("col_a", "col_b", "status").rows() == [
        ("a", "b", "ineligible"),  # a has 3 distinct values > max_unique
        ("a", "f", "ineligible"),
        ("b", "f", "ineligible"),  # Float64 is not categorical
    ]
