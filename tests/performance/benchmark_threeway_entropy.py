"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust vs ThreewayEntropyPolars (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_threeway_entropy.py

large_dataset.arrow is cut to its first 40 columns (9,880 triplets): all 101 columns
give 166,650 triplets (~65 s per Rust run), far past the reference's budget.
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.threeway_entropy",
        [
            large_dataset(columns=40),
            # All 101 columns -> C(101, 3) = 166,650 triplets. ThreewayEntropyPolars
            # cannot finish that many one-triplet-at-a-time value_counts calls within
            # budget, so it's excluded; only the Rust implementation is timed here.
            large_dataset(exclude=("ThreewayEntropyPolars",)),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
        ],
        runs=3,
    )
