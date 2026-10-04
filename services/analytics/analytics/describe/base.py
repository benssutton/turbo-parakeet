"""Per-column profile for choosing narrower / more compressible Arrow types (Spec A:
docs/superpowers/specs/2026-09-26-describe-technique-design.md).

Implementations fill METRICS and the private INPUTS (first-occurrence argmin / argmax,
f1, f2, capture history); the base renders min / max (arrow-rs text, `_plugin.render`),
picks the estimate and classifies each column.
"""

from __future__ import annotations

import polars as pl

from analytics._dtypes import WIDE_INTEGERS
from analytics.base import Technique, metric_mismatches
from analytics.describe import estimators
from analytics.describe._values import FLOATS, INTEGERS, NESTED, STRING_LIKE, flatten

U64, U32, F64 = pl.UInt64, pl.UInt32, pl.Float64
D38 = pl.Decimal(
    38, 0
)  # Arrow has no plain 128-bit integer; values are capped at 38 digits
LU64 = pl.List(pl.UInt64)

GROUP_A = {  # whole values — every eligible dtype
    "n_unique": U64,
    "min_len": U64,
    "max_len": U64,
    "gcd": D38,
    "sum_len": U64,
    "sum_len_unique": U64,
}
GROUP_B = {  # Float32 / Float64 only
    "n_nan": U64,
    "n_inf": U64,
    "n_fractional": U64,
    "max_frac_digits": U32,
    "n_f32_inexact": U64,
}
GROUP_C = {  # String / Categorical / Enum only
    "n_numeric": U64,
    "n_numeric_int": U64,
    "n_leading_zero": U64,
    "numeric_int_min": D38,
    "numeric_int_max": D38,
    "numeric_max_int_digits": U32,
    "numeric_max_frac_digits": U32,
    "numeric_min_frac_digits": U32,
    "numeric_max_sig_digits": U32,
    "n_iso_date": U64,
    "n_iso_time": U64,
    "n_iso_datetime": U64,
    "n_iso_datetime_tz": U64,
    "iso_max_frac_digits": U32,
    "iso_max_sig_frac_digits": U32,
    "iso_n_offsets": U64,
    "iso_n_midnight": U64,
}
VALUE_METRICS = {
    **GROUP_A,
    **GROUP_B,
    **GROUP_C,
}  # computed on outer values and on inner values
SIZE_METRICS = {
    "size_bytes": U64,
    "size_zstd_bytes": U64,
    "size_polars_bytes": U64,
    "size_polars_zstd_bytes": U64,
}
METRICS = {
    "n_rows": U64,
    "n_null": U64,
    **VALUE_METRICS,
    "n_midnight": U64,
    **SIZE_METRICS,
    "inner_n_values": U64,
    "inner_n_null": U64,
    **{f"inner_{k}": v for k, v in VALUE_METRICS.items()},
}

# Private estimator inputs (never in the result): first-occurrence row indices of the
# extremes, singletons / doubletons and the 3-way split's capture history.
LEVEL_INPUTS = {
    "argmin": U64,
    "argmax": U64,
    "f1": U64,
    "f2": U64,
    "capture_history": LU64,
}
INPUTS = {**LEVEL_INPUTS, **{f"inner_{k}": v for k, v in LEVEL_INPUTS.items()}}

ESTIMATES = {
    "unique": pl.Boolean,
    "est_cardinality": F64,
    "est_low": F64,
    "est_high": F64,
    "est_method": pl.String,
    "estimates_agree": pl.Boolean,
}
_LEVEL = {"min": pl.String, "max": pl.String, **ESTIMATES, "class": pl.String}
CONCLUSIONS = {**_LEVEL, **{f"inner_{k}": v for k, v in _LEVEL.items()}}

# Agreement (spec 2026-10-01 §13.8): exact unless listed. est_method and
# estimates_agree depend on each implementation's own seeded split: not compared.
EXACT_CONCLUSIONS = [
    c
    for c in CONCLUSIONS
    if c.removeprefix("inner_") in ("min", "max", "unique", "class")
]
ESTIMATE_CONCLUSIONS = [
    c
    for c in CONCLUSIONS
    if c.removeprefix("inner_") in ("est_cardinality", "est_low", "est_high")
]
TOLERANCES = {"size_zstd_bytes": 0.01, "size_polars_zstd_bytes": 0.01}
SCHNABEL_RTOL = 0.10


def _unsupported(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Object, pl.Null)) or dtype in WIDE_INTEGERS:
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _unsupported(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_unsupported(f.dtype) for f in dtype.fields)
    return False


class Describe(Technique):
    """Per-column profile: counts, cardinality estimates, extremes, lengths,
    float / numeric-string / ISO-datetime scanners, Arrow and Polars sizes, the same
    for list inner values, and a classification (first match wins):
    null → constant → boolean → ordinal → categorical → discrete.

    `seed` fixes the 3-way split behind the Schnabel estimate.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = METRICS
    INPUTS = INPUTS
    CONCLUSIONS = CONCLUSIONS

    def __init__(
        self,
        *,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if categorical_threshold < 0:
            raise ValueError(
                f"categorical_threshold must be >= 0, got {categorical_threshold}"
            )
        if not 1 <= zstd_level <= 22:
            raise ValueError(f"zstd_level must be in 1..22, got {zstd_level}")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed}")
        self.categorical_threshold = categorical_threshold
        self.zstd_level = zstd_level
        self.seed = seed

    # ── technique hooks ───────────────────────────────────────────────────────

    def eligible(self, series: pl.Series) -> bool:
        return not _unsupported(series.dtype)

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        columns: dict[str, list] = {k: [] for k in self.CONCLUSIONS}
        for row in out.iter_rows(named=True):
            values = self._conclusions(row) if row["status"] == "computed" else {}
            for k, v in columns.items():
                v.append(values.get(k))
        return out.with_columns(
            pl.Series(k, v, dtype=self.CONCLUSIONS[k]) for k, v in columns.items()
        )

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        # base.same_value only applies rtol/atol when at least one side is already
        # a Python float; two ints (e.g. UInt64 size_zstd_bytes) compare by ==
        # otherwise. Cast every tolerance-bearing metric to Float64 first so the
        # float branch always triggers, regardless of the metric's own dtype.
        keys = self.key_columns()
        exact = [m for m in self.METRICS if m not in TOLERANCES] + EXACT_CONCLUSIONS
        problems = metric_mismatches(result, reference, keys, exact, 0.0, 0.0)

        def as_float(df, cols):
            return df.with_columns(pl.col(c).cast(F64) for c in cols)

        for metric, rtol in TOLERANCES.items():
            problems += metric_mismatches(
                as_float(result, [metric]),
                as_float(reference, [metric]),
                keys,
                [metric],
                rtol,
                0.0,
            )
        problems += metric_mismatches(
            result, reference, keys, ESTIMATE_CONCLUSIONS, SCHNABEL_RTOL, 0.0
        )
        return list(dict.fromkeys(problems))

    # ── conclusions for one computed row ─────────────────────────────────────

    def _conclusions(self, r: dict) -> dict:
        s = self._collected[r["df_a"]][r["col_a"]]
        out = self._one_level(s, r, "", r["n_rows"])
        if r["inner_n_values"] is not None:
            out |= self._one_level(flatten(s), r, "inner_", r["inner_n_values"])
        return out

    def _one_level(self, s: pl.Series, r: dict, p: str, n_values: int) -> dict:
        n = n_values - r[f"{p}n_null"]
        est = estimators.estimate(
            r[f"{p}n_unique"], n, r[f"{p}f1"], r[f"{p}f2"], r[f"{p}capture_history"]
        )
        lo, hi = _render(s, r[f"{p}argmin"], r[f"{p}argmax"])
        return {
            f"{p}min": lo,
            f"{p}max": hi,
            **{f"{p}{k}": v for k, v in est.items()},
            f"{p}class": self._classify(s, r, p, n_values, n, est["est_cardinality"]),
        }

    def _classify(
        self,
        s: pl.Series,
        r: dict,
        p: str,
        n_values: int,
        n: int,
        est: float,
    ) -> str:
        if r[f"{p}n_null"] == n_values:
            return "null"
        if r[f"{p}n_unique"] == 1:
            return "constant"
        if r[f"{p}n_unique"] == 2:
            return "boolean"
        bounds = _whole_range(s, r, p, n)
        if bounds is not None and 0 <= bounds[0] and bounds[1] <= 2 * n_values:
            return "ordinal"
        if est <= self.categorical_threshold:
            return "categorical"
        return "discrete"


def _render(
    s: pl.Series, lo: int | None, hi: int | None
) -> tuple[str | None, str | None]:
    """min / max as arrow-rs text (spec 2026-10-01 §13.1); None for nested dtypes."""
    if lo is None or isinstance(s.dtype, NESTED):
        return None, None
    from analytics import _plugin

    lo_text, hi_text = _plugin.render(s.gather([lo, hi]))
    return lo_text, hi_text


def _whole_range(s: pl.Series, r: dict, p: str, n: int) -> tuple[float, float] | None:
    """(min, max) when every non-null value is a whole number, else None. Temporal
    dtypes never qualify (their physical integers are not quantities)."""
    lo, hi = r[f"{p}argmin"], r[f"{p}argmax"]
    if lo is None:
        return None
    dtype = s.dtype
    if isinstance(dtype, INTEGERS) or (
        isinstance(dtype, pl.Decimal) and dtype.scale == 0
    ):
        return int(s[lo]), int(s[hi])
    if (
        isinstance(dtype, FLOATS)
        and r[f"{p}n_fractional"] == r[f"{p}n_nan"] == r[f"{p}n_inf"] == 0
    ):
        return s[lo], s[hi]
    if (
        isinstance(dtype, STRING_LIKE)
        and r[f"{p}n_numeric_int"] == n
        and r[f"{p}n_leading_zero"] == 0
        and r[f"{p}numeric_int_min"] is not None
    ):
        return r[f"{p}numeric_int_min"], r[f"{p}numeric_int_max"]
    return None
