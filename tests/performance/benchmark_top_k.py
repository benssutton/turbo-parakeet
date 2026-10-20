"""
Top-k speed benchmark: TopKHeavyKeeperRust vs TopKSpaceSavingRust vs TopKPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_top_k.py

Shapes: large_dataset (realistic mix); narrow / long (parallelism inside a column:
SpaceSaving merges 2^20-row chunks, HeavyKeeper cannot); wide (parallelism across columns);
zipf 10M x 4 (stream-like skew, 1M possible values). `agrees` on large_dataset can be
false from ties at the k-th count (agreement precision is a lower bound there).
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, top_k_streams

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.top_k",
        [
            large_dataset(),
            Dataset(
                "narrow 10M x 4",
                lambda: {"t": low_cardinality(10_000_000, 4, cardinality=1_000)},
            ),
            Dataset(
                "wide 200K x 100",
                lambda: {"t": low_cardinality(200_000, 100, cardinality=1_000)},
            ),
            Dataset(
                "zipf 10M x 4",
                lambda: {
                    "t": top_k_streams(
                        10_000_000, columns=["zipf_15", "zipf_11", "zipf_08", "flat"]
                    )
                },
            ),
        ],
    )
