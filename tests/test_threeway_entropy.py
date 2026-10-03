"""
Three-way joint entropy accuracy tests — every implementation in analytics.threeway_entropy.

Reference: ThreewayEntropyPolars. mixed_dtypes: 18 eligible columns → C(18,3) = 816 triplets.
"""

import polars as pl
import pytest

from analytics.threeway_entropy import ThreewayEntropy
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

PKG = "analytics.threeway_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert out.height == 816


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", ALL)
def test_known_answers(impl):
    df = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "c": [0, 0, 0, None]})
    r = run(load(impl), {"t": df}).row(0, named=True)
    assert r["h_abc"] == pytest.approx(2.0)  # 4 distinct triples
    assert r["n_rows"] == 4
    assert r["near_unique"] is True


def test_conclusions():
    Fixed = with_metrics(
        ThreewayEntropy, h_abc=[1.95, 1.5, 0.0, 1.95], n_rows=[4, 4, 1, 4]
    )
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})  # C(4,3) = 4 triplets
    assert Fixed().add({"t": df}).result()["near_unique"].to_list() == [
        True,
        False,
        False,
        True,
    ]
    assert Fixed(near_unique_margin=0.01).add({"t": df}).result()[
        "near_unique"
    ].to_list() == [False, False, False, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({c: pl.Series([], dtype=pl.Int64) for c in "abc"})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
