"""
Delta speed benchmark: DeltaRust vs DeltaPolars (Python-int reference).

Run: python tests/performance/benchmark_delta.py

Shapes: synth_frame (50 000 rows, the covering mix); timestamps 1M x 4 (regular step + jitter, ms);
wide 20K x 100 (random walks of varying step size). The reference works in Python ints,
so the shapes are smaller than the numpy-fast techniques'.
"""

import numpy as np
import polars as pl
from harness import Dataset, synth_dataset, run


def timestamps() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(1)
    n = 1_000_000
    return {
        "t": pl.DataFrame(
            {
                f"ts{i}": np.cumsum(rng.integers(995, 1_006, n) + 1_000 * i).astype(
                    "datetime64[ms]"
                )
                for i in range(4)
            }
        )
    }


def wide() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(2)
    n = 20_000
    return {
        "t": pl.DataFrame(
            {
                f"w{i}": np.cumsum(rng.integers(-(10 ** (i % 9)), 10 ** (i % 9) + 1, n))
                for i in range(100)
            }
        )
    }


if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.delta",
        [
            synth_dataset(),
            Dataset("timestamps 1M x 4", timestamps),
            Dataset("wide 20K x 100", wide),
        ],
    )
