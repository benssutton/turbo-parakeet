"""
Marginal entropy benchmark (speed, peak memory, accuracy; harness.py): MarginalEntropyRust
and MarginalEntropyAmsRust (estimator, also scored against MarginalEntropyRust) vs
MarginalEntropyPolars (reference). The pairwise-entropy shapes.

Run: python tests/performance/benchmark_marginal_entropy.py
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.marginal_entropy",
        [
            cover_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
    )
