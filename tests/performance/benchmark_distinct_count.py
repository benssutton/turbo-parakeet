"""
Distinct-count benchmark (speed, peak memory, accuracy; harness.py): DistinctCountHllRust
(HyperLogLog, estimator, one variant per precision) and DistinctCountPolarsApprox (which
ignores `precision`, so its variants repeat) vs the exact Polars reference. Shapes: the
generated large dataset; 10M x 4 Int64 columns of 10^3 .. 10^7 distinct values; wide 1M x 100
(Int64 / Float64 / String / Date); 10M x 2 strings. Uniform draws of 10^7 labels over 10^7
rows see about 63% of them; the reference is exact either way.

Run: python tests/performance/benchmark_distinct_count.py
"""

import numpy as np
import polars as pl
from harness import Dataset, Variant, large_dataset, run

from datagen import describe_wide


def narrow() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.DataFrame(
            {
                f"d{e}": np.random.default_rng(e).integers(0, 10**e, 10_000_000)
                for e in (3, 5, 6, 7)
            }
        )
    }


def strings() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(0)
    return {
        "t": pl.DataFrame(
            {
                "items_1e6": pl.Series(rng.integers(0, 10**6, 10_000_000)).cast(
                    pl.String
                ),
                "ids_1e7": pl.Series(rng.integers(0, 10**7, 10_000_000)).cast(
                    pl.String
                ),
            }
        ).with_columns(items_1e6="item_" + pl.col("items_1e6"))
    }


if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.distinct_count",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", narrow),
            Dataset("wide 1M x 100", lambda: {"t": describe_wide(1_000_000, 100)}),
            Dataset("strings 10M x 2", strings),
        ],
        variants=[
            Variant(f"p={p}", estimator_params={"precision": p})
            for p in (10, 12, 14, 16)
        ],
    )
