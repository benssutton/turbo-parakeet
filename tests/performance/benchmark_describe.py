"""
describe speed benchmark: DescribeRust vs DescribeDataFusion vs DescribePolars (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_describe.py

Shapes: large_dataset.arrow (realistic mix); narrow 10M x 4 (parallelism inside a
column — chunked frequency maps and scanners); wide 1M x 100 (parallelism across
columns); nested 1M x 3 (inner values, row-encoded extremes, struct hashing).
"""

from harness import Dataset, large_dataset, run

from datagen import describe_narrow, describe_nested, describe_wide

if __name__ == "__main__":
    run(
        "analytics.describe",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", lambda: {"t": describe_narrow(10_000_000)}),
            Dataset("wide 1M x 100", lambda: {"t": describe_wide(1_000_000, 100)}),
            Dataset("nested 1M x 3", lambda: {"t": describe_nested(1_000_000)}),
        ],
    )
