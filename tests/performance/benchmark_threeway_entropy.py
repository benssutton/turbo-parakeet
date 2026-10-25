"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust and ThreewayEntropyAmsRust vs
ThreewayEntropyPolars (reference).

Run: python tests/performance/benchmark_threeway_entropy.py

Each dataset computes 50 stratified triplets (400 on the wide shape).
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.threeway_entropy",
        [
            cover_dataset(),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            # Many small combinations: per-combination overhead.
            Dataset(
                "wide 20K x 30",
                lambda: {"t": low_cardinality(20_000, 30)},
                combinations=400,
            ),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
    )
