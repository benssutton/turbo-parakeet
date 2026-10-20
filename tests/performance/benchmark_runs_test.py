"""
Runs-test speed benchmark: RunsTestRust vs RunsTestPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_runs_test.py

Shapes: large_dataset (realistic mix); long runs 10M x 4 (REE-friendly, mean run
1 000, int + string: the kernel hashes once per run, so the per-row scan dominates);
random 10M x 4 (R ≈ n, one hash insert per row — the worst case); wide 1M x 100 (mean runs
1 / 10 / 100 / 1 000; parallelism across columns). Results: spec 2026-10-13-runs-test §10.
"""

import polars as pl
from harness import Dataset, large_dataset, run

from datagen import runs_int, runs_str


def long_runs() -> dict[str, pl.DataFrame]:
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


if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.runs_test",
        [
            large_dataset(),
            Dataset("long runs 10M x 4", long_runs),
            Dataset(
                "random 10M x 4", lambda: {"t": runs_int(10_000_000, 4, 1, 10_000_000)}
            ),
            Dataset("wide 1M x 100", wide),
        ],
    )
