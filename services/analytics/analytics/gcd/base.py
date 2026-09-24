"""Whole-column GCD — the quantity ClickHouse's GCD codec divides by."""

import polars as pl

from analytics.base import Technique, computed

I128_LIMIT = 2**127
INTEGER_BACKED = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Decimal, pl.Date, pl.Datetime, pl.Duration, pl.Time,
)


class Gcd(Technique):
    """GCD of the magnitudes of each column's raw physical integer values.

    Results are in physical units: Decimal → unscaled integer, Date → days,
    Datetime/Duration → their time unit, Time → ns. Nulls are skipped; all-null,
    all-zero and zero-row columns → 0; a magnitude of 2**127 (only i128::MIN
    values) is not representable as Int128 → null. Other dtypes — including
    Categorical/Enum and UInt128, which the plugin's Rust polars cannot receive —
    are ineligible. `dtype` (Python's str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {"gcd": pl.Int128}
    CONCLUSIONS = {"gcd_compressible": pl.Boolean}

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, INTEGER_BACKED)

    def describe(self, frames, combos):
        return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(gcd_compressible=computed(pl.col("gcd") > 1))
