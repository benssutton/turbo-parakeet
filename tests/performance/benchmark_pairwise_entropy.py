"""
Pairwise joint entropy benchmark (speed, peak memory, accuracy; harness.py): PairwiseEntropyRust
and PairwiseEntropyAmsRust (estimator, also scored against PairwiseEntropyRust) vs
PairwiseEntropyPolars (reference).

Run: python tests/performance/benchmark_pairwise_entropy.py
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.pairwise_entropy",
        [
            cover_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
    )
