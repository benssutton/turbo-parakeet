"""DescribePolars ★ — the accuracy reference: Polars expressions per column
(group_by frequency table, sort for extremes, str.* for the scanners) and pyarrow
for sizes. Columns are processed one after another."""

from __future__ import annotations

import numpy as np
import polars as pl

from analytics.describe._sizes import column_sizes
from analytics.describe._values import (
    FLOATS,
    FRAC_DIGITS,
    INT_DIGITS,
    ISO_DATE,
    ISO_DATETIME,
    ISO_DATETIME_TZ,
    ISO_FRACTION,
    ISO_MIDNIGHT,
    ISO_OFFSET,
    ISO_TIME,
    LEADING_ZERO,
    NESTED,
    NUMERIC,
    NUMERIC_INT,
    STRING_LIKE,
    byte_lengths,
    flatten,
    frac_digits,
    frequency_summary,
    n_midnight,
    sig_digits,
    subsets,
)
from analytics.describe.base import (
    GROUP_B,
    GROUP_C,
    LEVEL_INPUTS,
    VALUE_METRICS,
    Describe,
)
from analytics.gcd.base import INTEGER_BACKED
from analytics.gcd.math import math_gcd


class DescribePolars(Describe):
    """Reference: one Polars pipeline per column; pyarrow IPC writer for sizes."""

    def _compute(self, frames, combos):
        rows = [self._row(frames[n][c]) for ((n, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in {**self.METRICS, **self.INPUTS}}
        )

    def _row(self, s: pl.Series) -> dict:
        row = {
            "n_rows": s.len(),
            "n_null": s.null_count(),
            **profile(s, self.seed),
            "n_midnight": n_midnight(s),
            **column_sizes(s, self.zstd_level),
        }
        inner = flatten(s) if isinstance(s.dtype, (pl.List, pl.Array)) else None
        row["inner_n_values"] = None if inner is None else inner.len()
        row["inner_n_null"] = None if inner is None else inner.null_count()
        inner_profile = (
            dict.fromkeys({**VALUE_METRICS, **LEVEL_INPUTS})
            if inner is None
            else profile(inner, self.seed)
        )
        return row | {f"inner_{k}": v for k, v in inner_profile.items()}


def profile(s: pl.Series, seed: int) -> dict:
    """Every VALUE_METRICS and LEVEL_INPUTS entry for one series (outer column or
    flattened inner values)."""
    freq = frequencies(s, seed)
    summary = frequency_summary(freq["count"].to_numpy(), freq["mask"].to_numpy())
    return {
        **summary,
        **extremes(s, freq),
        **lengths(s),
        **totals(s, freq),
        **float_stats(s),
        **string_stats(s),
    }


def totals(s: pl.Series, freq: pl.DataFrame) -> dict:
    """gcd of the physical values (as the Gcd technique) and the byte totals of all /
    distinct string or binary values."""
    lens = byte_lengths(s)
    return {
        "gcd": math_gcd(s) if isinstance(s.dtype, INTEGER_BACKED) else None,
        "sum_len": None if lens is None else int(lens.sum()),
        "sum_len_unique": None if lens is None else int(byte_lengths(freq["v"]).sum()),
    }


def frequencies(s: pl.Series, seed: int) -> pl.DataFrame:
    """Distinct non-null values with count, first row index and OR of 1 << split subset.
    Polars groups -0.0 with 0.0 and all NaNs together."""
    return (
        pl.DataFrame(
            {
                "v": s,
                "i": np.arange(s.len(), dtype=np.uint64),
                "m": (1 << subsets(s.len(), seed)).astype(np.uint8),
            }
        )
        .filter(pl.col("v").is_not_null())
        .group_by("v")
        .agg(
            pl.len().cast(pl.UInt64).alias("count"),
            pl.col("i").min().alias("first"),
            pl.col("m").bitwise_or().alias("mask"),
        )
    )


def extremes(s: pl.Series, freq: pl.DataFrame) -> dict:
    """First occurrence of the min and max: sort the distinct values (Polars order;
    NaN excluded) and take their first-occurrence indices."""
    if isinstance(s.dtype, NESTED):  # no extremes (spec 2026-10-01 §13.3)
        return {"argmin": None, "argmax": None}
    values = freq.select("v", "first")
    if isinstance(s.dtype, FLOATS):
        values = values.filter(pl.col("v").is_not_nan())
    if values.height == 0:
        return {"argmin": None, "argmax": None}
    values = values.sort("v")
    return {"argmin": values["first"][0], "argmax": values["first"][-1]}


def lengths(s: pl.Series) -> dict:
    dtype = s.dtype
    lens = byte_lengths(s)
    if lens is None:
        if isinstance(dtype, pl.List):
            lens = s.list.len()
        elif isinstance(dtype, pl.Array):
            lens = s.arr.len()
        else:
            return {"min_len": None, "max_len": None}
    # drop_nulls first: Polars flags arr.len() as sorted even when a null row sits in
    # the middle, and max() of a "sorted" series returns its last element (None).
    lens = lens.drop_nulls()
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
        "n_f32_inexact": (
            None
            if s.dtype == pl.Float32
            else int((finite.cast(pl.Float32).cast(pl.Float64) != finite).sum())
        ),
    }


def string_stats(s: pl.Series) -> dict:
    """Group C: numeric-string and ISO 8601 counts over non-null values."""
    if not isinstance(s.dtype, STRING_LIKE):
        return dict.fromkeys(GROUP_C)
    v = s.cast(pl.String).drop_nulls()
    numeric = v.filter(v.str.contains(NUMERIC))
    ints = v.filter(v.str.contains(NUMERIC_INT))
    int_digits = ints.str.extract(INT_DIGITS, 1).str.len_bytes()
    in_range = ints.len() > 0 and int_digits.max() <= 38
    parsed = ints.str.to_integer(dtype=pl.Int128) if in_range else None

    date_ok = v.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False).is_not_null()
    is_date = v.str.contains(ISO_DATE) & date_ok
    is_time = v.str.contains(ISO_TIME)
    is_dt = v.str.contains(ISO_DATETIME) & date_ok
    is_tz = v.str.contains(ISO_DATETIME_TZ) & date_ok
    timed = v.filter(is_time | is_dt | is_tz)
    offsets = (
        v.filter(is_tz)
        .str.extract(ISO_OFFSET, 1)
        .replace({"Z": "+00:00", "-00:00": "+00:00"})
    )
    stamped = v.filter(is_dt | is_tz)

    return {
        "n_numeric": numeric.len(),
        "n_numeric_int": ints.len(),
        "n_leading_zero": int(v.str.contains(LEADING_ZERO).sum()),
        "numeric_int_min": parsed.min() if parsed is not None else None,
        "numeric_int_max": parsed.max() if parsed is not None else None,
        "numeric_max_int_digits": numeric.str.extract(INT_DIGITS, 1)
        .str.len_bytes()
        .max(),
        "numeric_max_frac_digits": numeric.str.extract(FRAC_DIGITS, 1)
        .str.len_bytes()
        .fill_null(0)
        .max(),
        "numeric_min_frac_digits": numeric.str.extract(FRAC_DIGITS, 1)
        .str.len_bytes()
        .fill_null(0)
        .min(),
        "numeric_max_sig_digits": sig_digits(numeric).max(),
        "n_iso_date": int(is_date.sum()),
        "n_iso_time": int(is_time.sum()),
        "n_iso_datetime": int(is_dt.sum()),
        "n_iso_datetime_tz": int(is_tz.sum()),
        "iso_max_frac_digits": timed.str.extract(ISO_FRACTION, 1)
        .str.len_bytes()
        .fill_null(0)
        .max(),
        "iso_max_sig_frac_digits": timed.str.extract(ISO_FRACTION, 1)
        .str.strip_chars_end("0")
        .str.len_bytes()
        .fill_null(0)
        .max(),
        "iso_n_offsets": offsets.n_unique(),
        "iso_n_midnight": int(stamped.str.contains(ISO_MIDNIGHT).sum()),
    }
