"""DescribePolars ★ — the accuracy reference: Polars expressions per column
(group_by frequency table, sort for extremes, str.* for the scanners) and pyarrow
for sizes. Columns are processed one after another."""

from __future__ import annotations

import numpy as np
import polars as pl

from analytics.describe._sizes import column_sizes
from analytics.describe._values import FLOATS, STRING_LIKE, flatten, frac_digits, frequency_summary, n_midnight, subsets
from analytics.describe.base import GROUP_B, GROUP_C, VALUE_METRICS, Describe


class DescribePolars(Describe):
    """Reference: one Polars pipeline per column; pyarrow IPC writer for sizes."""

    def _compute(self, frames, combos):
        rows = [self._row(frames[n][c]) for ((n, c),) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})

    def _row(self, s: pl.Series) -> dict:
        row = {
            "n_rows": s.len(), "n_null": s.null_count(), **profile(s, self.seed),
            "n_midnight": n_midnight(s), **column_sizes(s, self.zstd_level),
        }
        inner = flatten(s) if isinstance(s.dtype, (pl.List, pl.Array)) else None
        row["inner_n_values"] = None if inner is None else inner.len()
        row["inner_n_null"] = None if inner is None else inner.null_count()
        inner_profile = dict.fromkeys(VALUE_METRICS) if inner is None else profile(inner, self.seed)
        return row | {f"inner_{k}": v for k, v in inner_profile.items()}


def profile(s: pl.Series, seed: int) -> dict:
    """Every VALUE_METRICS entry for one series (outer column or flattened inner values)."""
    freq = frequencies(s, seed)
    summary = frequency_summary(freq["count"].to_numpy(), freq["first"].to_numpy(), freq["mask"].to_numpy(), s.len(), s.null_count())
    return {**summary, **extremes(s, freq), **lengths(s), **float_stats(s), **string_stats(s)}


def frequencies(s: pl.Series, seed: int) -> pl.DataFrame:
    """Distinct non-null values with count, first row index and OR of 1 << split subset.
    Polars groups -0.0 with 0.0 and all NaNs together."""
    return (
        pl.DataFrame({"v": s, "i": np.arange(s.len(), dtype=np.uint64), "m": (1 << subsets(s.len(), seed)).astype(np.uint8)})
        .filter(pl.col("v").is_not_null())
        .group_by("v")
        .agg(pl.len().cast(pl.UInt64).alias("count"), pl.col("i").min().alias("first"), pl.col("m").bitwise_or().alias("mask"))
    )


def extremes(s: pl.Series, freq: pl.DataFrame) -> dict:
    """First occurrence of the min and max: sort the distinct values (Polars order;
    NaN excluded) and take their first-occurrence indices."""
    values = freq.select("v", "first")
    if isinstance(s.dtype, FLOATS):
        values = values.filter(pl.col("v").is_not_nan())
    if values.height == 0:
        return {"argmin": None, "argmax": None}
    values = values.sort("v")
    return {"argmin": values["first"][0], "argmax": values["first"][-1]}


def lengths(s: pl.Series) -> dict:
    dtype = s.dtype
    if isinstance(dtype, STRING_LIKE):
        lens = s.cast(pl.String).str.len_bytes()
    elif dtype == pl.Binary:
        lens = s.bin.size()
    elif isinstance(dtype, pl.List):
        lens = s.list.len()
    elif isinstance(dtype, pl.Array):
        lens = s.arr.len()
    else:
        return {"min_len": None, "max_len": None}
    return {"min_len": lens.min(), "max_len": lens.max()}


def float_stats(s: pl.Series) -> dict:
    if not isinstance(s.dtype, FLOATS):
        return dict.fromkeys(GROUP_B)
    v = s.drop_nulls()
    finite = v.filter(v.is_finite())
    return {
        "n_nan": int(v.is_nan().sum()),
        "n_inf": int(v.is_infinite().sum()),
        "n_fractional": int((finite != finite.floor()).sum()),
        # Shortest round-trip strings in the column's own width (f32 digits for Float32).
        "max_frac_digits": frac_digits(finite.unique().cast(pl.String)),
        "n_f32_inexact": None if s.dtype == pl.Float32 else int((finite.cast(pl.Float32).cast(pl.Float64) != finite).sum()),
    }


def string_stats(s: pl.Series) -> dict:
    """Group C — the numeric-string and ISO scanners (implemented in Task 6)."""
    return dict.fromkeys(GROUP_C)
