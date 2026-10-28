"""
Marginal-entropy accuracy tests — every implementation in
turbo_parakeet.techniques.marginal_entropy. Oracles: hand-worked entropies and
MarginalEntropyPolars (the reference). Null is its own category.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math

import polars as pl
import pytest

from synthdata.synth import synth_small, spread
from synthdata.scenarios import low_cardinality, near_constant
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.marginal_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def h(impl: str, s: pl.Series) -> float:
    return run(load(impl), {"t": s.to_frame()})["h"][0]


# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "synth_frame": spread(synth_small(), 18).head(300),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [
        synth_small,
        lambda: low_cardinality(50_000, 6),
    ],
    ids=["synth_frame", "lowcard"],
)
def test_agrees_with_reference(impl, make):
    frames = {"t": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers (exact implementations; AMS within its RTOL)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("k", [1, 2, 4, 16])
def test_uniform_is_log2_k(impl, k):
    got = h(impl, pl.Series("x", list(range(k)) * 250))
    assert got == pytest.approx(math.log2(k), rel=load(impl).RTOL, abs=1e-9)


@pytest.mark.parametrize("impl", ALL)
def test_null_is_one_category(impl):
    got = h(impl, pl.Series("x", [1, None, 1, None] * 100))
    assert got == pytest.approx(1.0, rel=load(impl).RTOL, abs=1e-9)


@pytest.mark.parametrize("impl", ALL)
def test_near_constant(impl):
    frames = {"t": near_constant(100_000, 1)}
    want = run(reference(PKG), frames)["h"][0]
    assert run(load(impl), frames)["h"][0] == pytest.approx(
        want, rel=load(impl).RTOL, abs=1e-9
    )


def test_ams_is_an_estimator():
    from turbo_parakeet.techniques.marginal_entropy import MarginalEntropyAmsRust

    assert MarginalEntropyAmsRust.EXACT is False
    with pytest.raises(ValueError):
        MarginalEntropyAmsRust(num_samples=0)


@pytest.mark.parametrize("got, ok", [(0.94, True), (1.09, True), (0.85, False)])
def test_ams_agreement_is_relative_or_absolute_in_bits(got, ok):
    """A 1-bit column: 0.06 bits off exceeds RTOL (5%) but not ATOL (0.1 bits)."""
    from turbo_parakeet.techniques.marginal_entropy import (
        MarginalEntropyAmsRust,
        MarginalEntropyPolars,
    )

    frames = {"t": pl.DataFrame({"x": [1, 2] * 50})}
    want = MarginalEntropyPolars().add(frames).result()
    assert want["h"][0] == pytest.approx(1.0)
    problems = MarginalEntropyAmsRust().agreement(
        want.with_columns(h=pl.lit(got)), want
    )
    assert (problems == []) is ok
