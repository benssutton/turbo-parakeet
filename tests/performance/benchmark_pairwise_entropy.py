"""
Pairwise joint entropy benchmark (speed, peak memory, accuracy; harness.py): PairwiseEntropyRust
and PairwiseEntropyAmsRust (estimator, also scored against PairwiseEntropyRust) vs
PairwiseEntropyPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_pairwise_entropy.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.pairwise_entropy",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
    )
