"""
MinMax skip-index speed benchmark: MinMaxIndexRust vs MinMaxIndexPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_minmax_index.py

Default window (8 192 rows). Shapes: large_dataset.arrow (realistic mix); clustered 10M x 4
(runs of ~1 000 rows, cardinality 1 000, int + string: index-friendly); random 10M x 4
(cardinality 100 000: wide window ranges); wide 1M x 100 (mean runs 1 / 10 / 100 / 1 000;
parallelism across columns). Results: spec 2026-10-16-skip-index-techniques §10.
"""

import polars as pl
from harness import Dataset, large_dataset, run

from datagen import runs_int, runs_str


def clustered() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.concat(
            [
                runs_int(10_000_000, 2, 1_000, 1_000),
                runs_str(10_000_000, 2, 1_000, 1_000),
            ],
            how="horizontal",
        )
    }


def wide() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.concat(
            [
                runs_int(1_000_000, 25, m, 10_000, prefix=f"m{m}_", seed=m)
                for m in (1, 10, 100, 1_000)
            ],
            how="horizontal",
        )
    }


DATASETS = [
    large_dataset(),
    Dataset("clustered 10M x 4", clustered),
    Dataset("random 10M x 4", lambda: {"t": runs_int(10_000_000, 4, 1, 100_000)}),
    Dataset("wide 1M x 100", wide),
]

if __name__ == "__main__":
    run("analytics.minmax_index", DATASETS)
