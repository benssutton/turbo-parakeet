"""Stratified selection of a technique's combinations (benchmarks)."""

import polars as pl

from turbo_parakeet.techniques._select import strata, stratified_combinations
from turbo_parakeet.techniques.pairwise_entropy import PairwiseEntropyRust

N = 2_000
FRAMES = {
    "t": pl.DataFrame(
        {
            "two": [i % 2 for i in range(N)],
            "low": [i % 10 for i in range(N)],
            "uniq": list(range(N)),
            "uniq_s": [f"s{i}" for i in range(N)],
            "low_s": [f"v{i % 7}" for i in range(N)],
            "uniq2": [N - i for i in range(N)],
            # ineligible in every technique (no Arrow 128-bit integer)
            "wide": pl.Series(range(N), dtype=pl.Int128),
        }
    )
}


def test_strata_split_families_and_regimes():
    s = strata(PairwiseEntropyRust(), FRAMES)
    keys = set(s)
    assert (("int", "int"), "flat") in keys  # two x low
    assert (("int", "int"), "hash") in keys  # uniq x uniq2: 4e6 > 2**20
    assert (("int", "string"), "flat") in keys
    assert all(("t", "wide") not in k for group in s.values() for k in group)


def test_every_stratum_represented_and_deterministic():
    s = strata(PairwiseEntropyRust(), FRAMES)
    picked = stratified_combinations(PairwiseEntropyRust(), FRAMES, n=len(s), seed=3)
    assert len(picked) == len(s)
    assert {next(key for key, g in s.items() if k in g) for k in picked} == set(s)
    assert picked == stratified_combinations(
        PairwiseEntropyRust(), FRAMES, n=len(s), seed=3
    )


def test_at_most_n_and_all_when_n_is_large():
    assert len(stratified_combinations(PairwiseEntropyRust(), FRAMES, n=4)) == 4
    every = sum(len(g) for g in strata(PairwiseEntropyRust(), FRAMES).values())
    assert (
        len(stratified_combinations(PairwiseEntropyRust(), FRAMES, n=10_000)) == every
    )


def test_picked_combinations_compute():
    picked = stratified_combinations(PairwiseEntropyRust(), FRAMES, n=5)
    out = PairwiseEntropyRust().add(FRAMES).result(combinations=picked)
    assert out.height == 5 and set(out["status"]) == {"computed"}
