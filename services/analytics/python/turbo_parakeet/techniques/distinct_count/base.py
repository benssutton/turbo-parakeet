"""How many distinct non-null values each column holds (per-column scope)."""

import polars as pl

from turbo_parakeet._dtypes import encodable, is_nested
from turbo_parakeet.techniques.base import Technique, metric_mismatches

FLOATS = (pl.Float32, pl.Float64)


def canonical(s: pl.Series) -> pl.Series:
    """Values as the Rust encoder keys them: -0.0 = 0.0, one NaN, Categorical / Enum by
    string; nulls dropped."""
    s = s.drop_nulls()
    if s.dtype in FLOATS:
        return (s + 0.0).fill_nan(float("nan"))
    if isinstance(s.dtype, (pl.Categorical, pl.Enum)):
        return s.cast(pl.String)
    return s


class DistinctCount(Technique):
    """`n_distinct`: the (estimated) number of distinct non-null values, with a +-3 sigma
    interval [`low`, `high`] (floored at 0, capped at `n_non_null`), the relative
    `std_error` (0 when exact, null when not exposed) and the sketch's `sketch_bytes`.
    Floats canonical, Categorical / Enum by string. Eligible: the Rust encoder's scalar
    dtypes in a non-empty column; nested, Binary, Struct, Null, Object and 128-bit
    integers are not. Metrics only."""

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_non_null": pl.UInt64,
        "n_distinct": pl.Float64,
        "low": pl.Float64,
        "high": pl.Float64,
        "std_error": pl.Float64,
        "sketch_bytes": pl.UInt64,
    }
    ACCURACY_METRICS = ("n_distinct",)

    def eligible(self, series: pl.Series) -> bool:
        return (
            series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)
        )

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def interval_rows(self, combos, n_non_null, n_distinct, std_error, sketch_bytes):
        """metrics_frame with the +-3 sigma interval derived once (std_error null: no
        interval)."""
        low, high = [], []
        for n, d, se in zip(n_non_null, n_distinct, std_error):
            spread = 3 * d * (se or 0.0)
            low.append(max(d - spread, 0.0) if se is not None else None)
            high.append(min(d + spread, float(n)) if se is not None else None)
        return self.metrics_frame(
            combos,
            {
                "n_non_null": n_non_null,
                "n_distinct": n_distinct,
                "low": low,
                "high": high,
                "std_error": std_error,
                "sketch_bytes": sketch_bytes,
            },
        )

    def agreement(self, result, reference):
        """`n_non_null` exact; `n_distinct` within RTOL (the estimators' class / instance
        tolerance); interval columns are not compared."""
        keys = self.key_columns()
        return metric_mismatches(
            result, reference, keys, ["n_non_null"], 0.0, 0.0
        ) + metric_mismatches(
            result, reference, keys, ["n_distinct"], self.RTOL, self.ATOL
        )
