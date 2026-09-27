"""
recommend speed benchmark: RecommendRust only — there is no Python implementation,
so the report shows the parallel speedup (Rust@1 thread ÷ Rust@N threads).

Run: /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_recommend.py

Shapes as benchmark_describe.py: large_dataset.arrow; narrow 10M x 4; wide 1M x 100; nested 1M x 3.
"""

from harness import Dataset, large_dataset, run

from datagen import describe_narrow, describe_nested, describe_wide

if __name__ == "__main__":
    run(
        "analytics.recommend",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", lambda: {"t": describe_narrow(10_000_000)}),
            Dataset("wide 1M x 100", lambda: {"t": describe_wide(1_000_000, 100)}),
            Dataset("nested 1M x 3", lambda: {"t": describe_nested(1_000_000)}),
        ],
    )
