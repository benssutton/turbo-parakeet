"""ClickHouse set(max_size) skip-index suitability per column."""

import polars as pl

from turbo_parakeet.techniques._skip_index import COMMON_METRICS, SkipIndex


class SetIndex(SkipIndex):
    """Windows and metrics as SkipIndex. A window stores the set of its entries — its
    distinct non-null values, plus NULL when it holds a null (ClickHouse stores NULL for
    Nullable columns). A set of more than `max_size` entries is stored empty and skips
    nothing (0 = unlimited); otherwise the window is skipped for `col = v` iff v is not in
    it. NULL counts toward max_size only: it never helps a `col = v` query.

    Also reported: `window_distinct_mean` / `window_distinct_max` (entries per window) and
    `overflow_fraction` (windows whose set overflowed), with their random-order
    expectations `expected_window_distinct` (exact) and `expected_overflow_fraction`. The
    skip baseline is exact when max_size = 0; otherwise the entry count is approximated as
    normal with its exact mean, variance and conditional means (within ~0.01 of Monte Carlo).
    """

    METRICS = {
        **COMMON_METRICS,
        "window_distinct_mean": pl.Float64,
        "window_distinct_max": pl.UInt64,
        "overflow_fraction": pl.Float64,
        "expected_window_distinct": pl.Float64,
        "expected_overflow_fraction": pl.Float64,
    }

    def __init__(
        self,
        *,
        index_granularity: int = 8192,
        granularity: int = 1,
        max_size: int = 100,
    ):
        super().__init__(index_granularity=index_granularity, granularity=granularity)
        if type(max_size) is not int or max_size < 0:
            raise ValueError(f"max_size must be an integer >= 0, got {max_size!r}")
        self.max_size = max_size
