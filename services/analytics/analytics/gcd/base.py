"""Whole-column GCD — the quantity ClickHouse's GCD codec divides by."""

import polars as pl

from analytics.base import Technique, computed

INTEGER_BACKED = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Decimal,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
)


class Gcd(Technique):
    """GCD of the magnitudes of each column's raw physical integer values.

    Results are in physical units: Decimal → unscaled integer, Date → days,
    Datetime/Duration → their time unit, Time → ns. Nulls are skipped; all-null,
    all-zero and zero-row columns → 0. `gcd` is Decimal(38, 0): Arrow has no plain
    128-bit integer, and a Decimal's unscaled values fit in 38 digits. Other dtypes
    — including Categorical/Enum and 128-bit integers (analytics._dtypes.WIDE_INTEGERS)
    — are ineligible. `dtype` (Python's str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {"gcd": pl.Decimal(38, 0)}
    CONCLUSIONS = {"gcd_compressible": pl.Boolean}

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, INTEGER_BACKED)

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(gcd_compressible=computed(pl.col("gcd") > 1))
