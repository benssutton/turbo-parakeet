"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust and ThreewayEntropyAmsRust vs
ThreewayEntropyPolars (reference).

Run: python tests/performance/benchmark_threeway_entropy.py

cover is cut to its first 40 columns (9,880 triplets): all 77 columns give 73,150
triplets, far past the reference's budget.
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.threeway_entropy",
        [
            cover_dataset(columns=40),
            # All 77 columns -> C(77, 3) = 73,150 triplets. Neither the one-triplet-at-a-
            # time Polars reference nor a 4096-estimator sketch per triplet finishes within
            # budget, so only the exact Rust implementation is timed here.
            cover_dataset(exclude=("ThreewayEntropyPolars", "ThreewayEntropyAmsRust")),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
        runs=3,
    )
