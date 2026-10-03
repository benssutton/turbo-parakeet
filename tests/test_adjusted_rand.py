"""
Pairwise Adjusted Rand Index accuracy tests — every implementation in analytics.adjusted_rand.

Reference: AdjustedRandSklearn (sklearn.metrics.adjusted_rand_score on string
labels after dropping rows where either column is null). Eligible columns in
mixed_dtypes: everything except list_* / arr_* (12) → C(12,2) = 66 computed pairs.
"""

import math

import polars as pl
import pytest

from analytics.adjusted_rand import AdjustedRand
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

PKG = "analytics.adjusted_rand"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 66


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def _ari(impl, a, b):
    return run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_known_answers(impl):
    assert _ari(impl, [0, 0, 1, 1], [5, 5, 7, 7])["ari"] == pytest.approx(
        1.0
    )  # identical partitions, relabelled
    assert _ari(impl, [0, 0, 1, 1], [0, 1, 0, 1])["ari"] == pytest.approx(
        -0.5
    )  # maximally crossed
    assert _ari(impl, [1, 1, 1, 1], [2, 2, 2, 2])["ari"] == pytest.approx(
        1.0
    )  # both constant (sklearn convention)


@pytest.mark.parametrize("impl", ALL)
def test_no_overlap_is_nan(impl):
    row = _ari(impl, [1, 2, None, None], [None, None, 1, 2])
    assert math.isnan(row["ari"])
    assert row["n_valid"] == 0
    assert row["same_partition"] is False


@pytest.mark.parametrize("impl", ALL)
def test_negative_zero_and_zero_are_one_label(impl):
    row = _ari(impl, [-0.0, 0.0, 1.0, 1.0], [3, 3, 4, 4])
    assert row["ari"] == pytest.approx(1.0)


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(AdjustedRand, ari=[0.89, 0.9, float("nan")], n_valid=[4] * 3)
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    assert Fixed().add({"t": df}).result()["same_partition"].to_list() == [
        False,
        True,
        False,
    ]
    assert Fixed(ari_threshold=0.5).add({"t": df}).result()[
        "same_partition"
    ].to_list() == [True, True, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame(
        {"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Int64)}
    )
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]


@pytest.mark.parametrize("impl", ALL)
def test_nested_columns_are_ineligible(impl):
    df = pl.DataFrame({"a": [1, 2], "l": [[1], [2]]})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
