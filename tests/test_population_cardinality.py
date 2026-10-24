"""
Population-cardinality accuracy tests — every implementation in
turbo_parakeet.techniques.population_cardinality. Oracles: hand-worked counts and
PopulationCardinalityPolars (Polars group_by + NumPy SplitMix64 split). Counts must match
exactly and estimates within RTOL 1e-9; the split is pinned to the Rust one by a
hand-computed vector.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math

import numpy as np
import polars as pl
import pytest

from datagen import large_dataset, mixed_dtypes, population_samples
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.population_cardinality"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def stats(impl, s, **params):
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", [0, 7])
def test_agrees_with_reference(impl, seed):
    sample, _ = population_samples(
        50_000,
        [(1_000, "uniform"), (20_000, "uniform"), (20_000, ("zipf", 1.1))],
        seed=seed,
    )
    for frames in (
        {"large": large_dataset()},
        {"mixed": mixed_dtypes(2_000)},
        {"s": sample},
    ):
        cls = load(impl)
        assert_agrees(
            cls(seed=seed),
            run(cls, frames, seed=seed),
            run(reference(PKG), frames, seed=seed),
        )


# 3. Known answers


@pytest.mark.parametrize("impl", ALL)
def test_counts_and_chao1_by_hand(impl):
    r = stats(impl, pl.Series("x", ["a", "a", "a", "b", "b", "c", "d", None, None]))
    assert (r["n_non_null"], r["n_distinct"], r["f1"], r["f2"]) == (7, 4, 2, 1)
    assert r["chao1"] == pytest.approx(4.5)
    assert math.isnan(r["schnabel"])  # d/n = 4/7 >= 0.5


@pytest.mark.parametrize("impl", ALL)
def test_schnabel_defined_on_a_dense_sample(impl):
    r = stats(impl, pl.Series("x", np.arange(100_000) % 1_000))  # every value seen 100x
    assert r["n_distinct"] == 1_000
    assert r["schnabel"] == pytest.approx(1_000, rel=0.02)
    assert r["schnabel_low"] <= r["schnabel"] <= r["schnabel_high"]
    assert r["chao1"] == 1_000.0  # no singletons


def test_split_subsets_match_the_rust_splitmix():
    from turbo_parakeet.techniques.population_cardinality.polars import subsets

    # The Rust `subset(0, row)` for rows 0..8, computed with Python ints from the SplitMix64
    # finaliser in describe/frequencies.rs (independently of the code under test).
    assert subsets(0, 8).tolist() == [1, 2, 1, 0, 1, 2, 2, 0]


def test_truth_figures_by_hand():
    from turbo_parakeet.techniques.population_cardinality import (
        PopulationCardinalityPolars,
    )

    frames = {"t": pl.DataFrame({"x": np.arange(100_000) % 1_000})}
    result = run(PopulationCardinalityPolars, frames)
    out = PopulationCardinalityPolars().accuracy(
        frames, result, result, truth={("t", "x"): 2_000}
    )
    assert out["observed_rel_err"] == pytest.approx(0.5)  # d = 1 000 of D = 2 000
    assert out["chao1_rel_err"] == pytest.approx(0.5)
    assert out["chao1_coverage"] == 0.0
    assert out["schnabel_n_undefined"] == 0
    assert "max_rel_err" in out  # the default agreement figures are kept


def test_truth_figures_over_several_columns():
    from turbo_parakeet.techniques.population_cardinality import (
        PopulationCardinalityPolars,
    )

    frames = {
        "t": pl.DataFrame(
            {"dense": np.arange(100_000) % 1_000, "distinct": np.arange(100_000)}
        )
    }
    result = run(PopulationCardinalityPolars, frames)
    truth = {("t", "dense"): 1_000, ("t", "distinct"): 100_000}
    out = PopulationCardinalityPolars().accuracy(frames, result, result, truth=truth)
    assert out["schnabel_n_undefined"] == 1  # d/n = 1 on the distinct column
    assert out["schnabel_coverage"] == 1.0  # the dense interval contains D, one defined
    assert (
        out["chao1_coverage"] == 0.5
    )  # dense contains D, distinct (Chao1 5e9) does not
    dense = result.filter(pl.col("col_a") == "dense").row(0, named=True)
    assert out["schnabel_rel_err"] == pytest.approx(
        abs(dense["schnabel"] - 1_000) / 1_000
    )


def test_truth_for_no_computed_column_returns_only_the_defaults():
    from turbo_parakeet.techniques.population_cardinality import (
        PopulationCardinalityPolars,
    )

    frames = {"t": pl.DataFrame({"x": np.arange(1_000) % 10})}
    result = run(PopulationCardinalityPolars, frames)
    out = PopulationCardinalityPolars().accuracy(
        frames, result, result, truth={("t", "absent"): 10}
    )
    assert "observed_rel_err" not in out and "schnabel_coverage" not in out
    assert "max_rel_err" in out


# 4. Null alignment and seeds


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", [0, 5])
def test_nulls_keep_their_original_row_index(impl, seed):
    s = pl.Series("x", [None, "a", None, "a", "b", None, "c", "c", "c", "d"] * 50)
    frames = {"t": s.to_frame()}
    cls = load(impl)
    assert_agrees(
        cls(seed=seed),
        run(cls, frames, seed=seed),
        run(reference(PKG), frames, seed=seed),
    )


@pytest.mark.parametrize("impl", ALL)
def test_seed_changes_the_split(impl):
    s = pl.Series("x", np.random.default_rng(0).integers(0, 3_000, 20_000))
    r0, r1 = (stats(impl, s, seed=k) for k in (0, 1))
    assert r0["schnabel"] != r1["schnabel"]
    assert r0["n_distinct"] == r1["n_distinct"]


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", [0, 1])
def test_seeded_estimates_equal_the_reference(impl, seed):
    s = pl.Series("x", np.random.default_rng(0).integers(0, 3_000, 20_000))
    got = stats(impl, s, seed=seed)
    ref = run(reference(PKG), {"t": s.to_frame()}, seed=seed).row(0, named=True)
    for k in ("schnabel", "schnabel_low", "schnabel_high", "chao1"):
        assert got[k] == pytest.approx(ref[k], rel=1e-9, nan_ok=True), k
