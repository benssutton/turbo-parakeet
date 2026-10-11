"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust and ThreewayEntropyAmsRust vs
ThreewayEntropyPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_threeway_entropy.py

large_dataset.arrow is cut to its first 40 columns (9,880 triplets): all 101 columns
give 166,650 triplets (~65 s per Rust run), far past the reference's budget.
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "analytics.threeway_entropy",
        [
            large_dataset(columns=40),
            # All 101 columns -> C(101, 3) = 166,650 triplets. Neither the one-triplet-at-a-
            # time Polars reference nor a 4096-estimator sketch per triplet finishes within
            # budget, so only the exact Rust implementation is timed here.
            large_dataset(exclude=("ThreewayEntropyPolars", "ThreewayEntropyAmsRust")),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
        runs=3,
    )
