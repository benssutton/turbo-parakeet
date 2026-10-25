"""
Pairwise Adjusted Rand Index accuracy tests — every implementation in turbo_parakeet.techniques.adjusted_rand.

Reference: AdjustedRandSklearn (sklearn.metrics.adjusted_rand_score on string
labels after dropping rows where either column is null). spread(cover_small(), 14) — 14,
not 18, keeps the pair count and the sklearn reference's time below the old fixture's;
that spread has no Duration column (tested separately below).
Eligible: everything except binary and struct (11) → C(11,2) = 55 computed pairs.
"""

import math
from datetime import time, timedelta

import polars as pl
import pytest

from turbo_parakeet.techniques.adjusted_rand import AdjustedRand
from synthdata.cover import cover_small, spread
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
    with_metrics,
)

PKG = "turbo_parakeet.techniques.adjusted_rand"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": spread(cover_small(), 14).head(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 55


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": spread(cover_small(), 14).head(1_000)}
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


@pytest.mark.parametrize("impl", ALL)
def test_time_labels_keep_sub_second_precision(impl):
    micros = [0, 1, 500, 999_999, 250_000]
    codes = [i % 5 for i in range(1_000)]
    times = [time(0, 0, 1, micros[c]) for c in codes]
    row = _ari(impl, times, codes)
    assert row["ari"] == pytest.approx(1.0)
    assert row["n_valid"] == 1_000


@pytest.mark.parametrize("impl", ALL)
def test_duration_column_computes(impl):
    codes = [i % 4 for i in range(400)]
    durations = [timedelta(microseconds=c * 7 + 1) for c in codes]
    row = _ari(impl, durations, codes)
    assert row["ari"] == pytest.approx(1.0)
    assert row["n_valid"] == 400


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
