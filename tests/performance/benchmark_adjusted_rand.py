"""
Adjusted Rand Index speed benchmark: AdjustedRandRust vs AdjustedRandSklearn (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_adjusted_rand.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.adjusted_rand",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        ],
    )
