"""
Runs-test speed benchmark: RunsTestRust vs RunsTestPolars (reference).

Run: python tests/performance/benchmark_runs_test.py

Shapes: synth_frame (50 000 rows, the covering mix); long runs 10M x 4 (REE-friendly, mean run
1 000, int + string: the kernel hashes once per run, so the per-row scan dominates);
random 10M x 4 (R ≈ n, one hash insert per row — the worst case); wide 1M x 100 (mean runs
1 / 10 / 100 / 1 000; parallelism across columns). Results (on the former 50K x 101 mix): Rust total 16.9x,
3.9x wide, 3.0x long runs, 1.2x random.
"""

import polars as pl
from harness import Dataset, cover_dataset, run

from synthdata.scenarios import runs_int, runs_str


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
            cover_dataset(),
            Dataset("long runs 10M x 4", long_runs),
            Dataset(
                "random 10M x 4", lambda: {"t": runs_int(10_000_000, 4, 1, 10_000_000)}
            ),
            Dataset("wide 1M x 100", wide),
        ],
    )
