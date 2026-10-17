"""Shared base of the skip-index techniques — analytics.minmax_index and analytics.set_index
(spec docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md) — and the arithmetic
their Polars references share (they share none with the Rust kernels)."""

import math

import numpy as np
import polars as pl

from analytics._dtypes import ORDERED_SCALARS
from analytics.base import Technique, metric_mismatches

COMMON_METRICS = {
    "n_rows": pl.UInt64,
    "n_non_null": pl.UInt64,
    "n_distinct": pl.UInt64,
    "window_rows": pl.UInt64,
    "n_windows": pl.UInt64,
    "skip_rows_mean": pl.Float64,
    "skip_rows_std": pl.Float64,
    "skip_values_mean": pl.Float64,
    "skip_values_std": pl.Float64,
    "expected_skip_rows": pl.Float64,
    "expected_skip_values": pl.Float64,
}


def positive(name: str, value) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


class SkipIndex(Technique):
    """How well a column, in its row order, suits a ClickHouse data-skipping index.

    A window of w = index_granularity × granularity rows — one index block — slides one row
    at a time over the column (a column shorter than w is one window), so every granule
    alignment counts once. For a point query `col = v`, a window's skip is the share of
    queries it lets ClickHouse skip, v drawn from the non-null values either by row
    frequency (`skip_rows_*`: frequent values are queried more) or uniformly over the
    distinct values (`skip_values_*`: rare-value search). `*_mean` / `*_std` are the mean
    and population std over windows: the expected skip under a random alignment, and its
    spread.

    `expected_skip_rows` / `expected_skip_values` are the same expectations had the rows
    arrived in a uniformly random order (closed form, from the value counts): observed minus
    expected is the gain due to ordering; expected itself is the gain due to rarity, which
    any order keeps.

    Metrics only — no conclusions: choosing an index, max_size or GRANULARITY is the
    recommenders' decision. Eligible: integers (≤ 64-bit), Boolean, Float, Decimal,
    temporal, String, Binary, Categorical and Enum columns with at least one row; a column
    without non-null values has NaN skips (no query value exists). `dtype` (Python's
    str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = COMMON_METRICS
    RTOL = 1e-9
    ATOL = 1e-12
    # The baselines sum lgamma differences (~10⁸ at 10⁷ rows) from two libms.
    BASELINE_RTOL = 1e-6

    def __init__(self, *, index_granularity: int = 8192, granularity: int = 1):
        super().__init__()
        positive("index_granularity", index_granularity)
        positive("granularity", granularity)
        if index_granularity * granularity > 2**64 - 1:
            raise ValueError("index_granularity × granularity must fit in 64 bits")
        self.index_granularity = index_granularity
        self.granularity = granularity

    @property
    def window_rows(self) -> int:
        return self.index_granularity * self.granularity

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, ORDERED_SCALARS) and series.len() > 0

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def agreement(self, result, reference):
        keys = self.key_columns()
        baselines = [m for m in self.METRICS if m.startswith("expected_")]
        observed = [m for m in self.METRICS if m not in baselines]
        problems = metric_mismatches(
            result, reference, keys, observed, self.RTOL, self.ATOL
        ) + metric_mismatches(
            result, reference, keys, baselines, self.BASELINE_RTOL, self.ATOL
        )
        return list(dict.fromkeys(problems))  # both calls report status mismatches


# ── the references' arithmetic ─────────────────────────────────────────────────


def ranked(s: pl.Series) -> tuple[pl.Series, np.ndarray]:
    """(ids, counts): each row's rank among the distinct non-null values (UInt32; null for a
    null row) and the rows per rank. Order as the kernels: Categorical by its string, Enum by
    category order, floats canonical (−0.0 = 0.0, one NaN, sorted last)."""
    if s.dtype == pl.Categorical:
        s = s.cast(pl.String)
    elif isinstance(s.dtype, pl.Enum):
        s = s.to_physical()
    elif s.dtype in (pl.Float32, pl.Float64):
        s = s.set((s == 0).fill_null(False), 0.0).fill_nan(float("nan"))
    distinct = s.drop_nulls().unique().sort()
    ids = pl.DataFrame({"v": s}).join(
        distinct.to_frame("v").with_row_index("id"),
        on="v",
        how="left",
        maintain_order="left",
    )["id"]
    counts = np.bincount(ids.drop_nulls().to_numpy(), minlength=distinct.len())
    return ids, counts.astype(np.int64)


def windows(n: int, window_rows: int) -> tuple[int, int]:
    """(w, number of windows) for a non-empty column."""
    w = min(n, window_rows)
    return w, n - w + 1


def mean_std(a: np.ndarray, den: int) -> tuple[float, float]:
    """Mean (exact, Python ints) and population std (two-pass) of a / den; NaN without a
    denominator."""
    if den == 0 or a.size == 0:
        return math.nan, math.nan
    return int(a.sum()) / (a.size * den), float(np.std(a / den))


def avoid(n: int, w: int, m: int) -> float:
    """P(a w-row window of a random order of n rows holds none of m given rows)."""
    if m == 0:
        return 1.0
    if n - m < w:
        return 0.0
    lg = math.lgamma
    return math.exp(lg(n - m + 1) - lg(n - m - w + 1) - lg(n + 1) + lg(n - w + 1))


def phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2))


def common_metrics(
    n: int, counts: np.ndarray, w: int, rows: np.ndarray, values: np.ndarray, expected
) -> dict:
    """COMMON_METRICS from the per-window numerators and (expected rows, expected values)."""
    nn, d = int(counts.sum()), int(counts.size)
    (rm, rs), (vm, vs) = mean_std(rows, nn), mean_std(values, d)
    return {
        "n_rows": n,
        "n_non_null": nn,
        "n_distinct": d,
        "window_rows": w,
        "n_windows": n - w + 1,
        "skip_rows_mean": rm,
        "skip_rows_std": rs,
        "skip_values_mean": vm,
        "skip_values_std": vs,
        "expected_skip_rows": expected[0],
        "expected_skip_values": expected[1],
    }
