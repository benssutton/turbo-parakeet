import math

import polars as pl

from analytics.gcd.base import I128_LIMIT, Gcd


def math_gcd(series: pl.Series) -> int | None:
    g = math.gcd(*series.to_physical().drop_nulls().to_list())
    return None if g >= I128_LIMIT else g


class GcdMath(Gcd):
    """Reference: math.gcd over each column's physical values (arbitrary precision,
    so Int128, wide Decimal and MIN magnitudes are exact). Single core."""

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, {"gcd": [math_gcd(frames[n][c]) for ((n, c),) in combos]})
