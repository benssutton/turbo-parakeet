"""Per-column profile for choosing narrower / more compressible Arrow types (Spec A:
docs/superpowers/specs/2026-09-26-describe-technique-design.md).

Implementations fill METRICS. Min, max and top-5 values are reported as row indices
of their first occurrence; this base renders them, estimates cardinality and
classifies each column identically for every implementation.
"""

from __future__ import annotations

import polars as pl

from analytics.base import Technique, metric_mismatches
from analytics.describe import estimators
from analytics.describe._values import FLOATS, INTEGERS, STRING_LIKE, flatten

U64, U32, I128, F64 = pl.UInt64, pl.UInt32, pl.Int128, pl.Float64
LU64 = pl.List(pl.UInt64)

GROUP_A = {  # whole values — every eligible dtype
    "n_unique": U64, "entropy": F64, "f1": U64, "f2": U64, "argmin": U64, "argmax": U64,
    "min_len": U64, "max_len": U64, "top5_idx": LU64, "top5_count": LU64, "capture_history": LU64,
}
GROUP_B = {  # Float32 / Float64 only
    "n_nan": U64, "n_inf": U64, "n_fractional": U64, "max_frac_digits": U32, "n_f32_inexact": U64,
}
GROUP_C = {  # String / Categorical / Enum only
    "n_numeric": U64, "n_numeric_int": U64, "n_leading_zero": U64,
    "numeric_int_min": I128, "numeric_int_max": I128,
    "numeric_max_int_digits": U32, "numeric_max_frac_digits": U32,
    "n_iso_date": U64, "n_iso_time": U64, "n_iso_datetime": U64, "n_iso_datetime_tz": U64,
    "iso_max_frac_digits": U32, "iso_n_offsets": U64, "iso_n_midnight": U64,
}
VALUE_METRICS = {**GROUP_A, **GROUP_B, **GROUP_C}  # computed on outer values and on inner values
SIZE_METRICS = {"size_bytes": U64, "size_zstd_bytes": U64, "size_polars_bytes": U64, "size_polars_zstd_bytes": U64}
METRICS = {
    "n_rows": U64, "n_null": U64, **VALUE_METRICS, "n_midnight": U64, **SIZE_METRICS,
    "inner_n_values": U64, "inner_n_null": U64, **{f"inner_{k}": v for k, v in VALUE_METRICS.items()},
}

CLASS = pl.Enum(["null", "constant", "boolean", "ordinal", "categorical", "discrete"])
METHOD = pl.Enum(["exact", "duj1", "schnabel", "chao1"])
TOP5 = pl.List(pl.Struct({"value": pl.String, "count": pl.UInt64}))
ESTIMATES = {
    "unique": pl.Boolean, "chao1": F64, "chao1_low": F64, "chao1_high": F64,
    "schnabel": F64, "schnabel_low": F64, "schnabel_high": F64,
    "est_cardinality": F64, "est_method": METHOD, "est_low": F64, "est_high": F64, "estimates_agree": pl.Boolean,
}
_RENDERED = {"min": pl.String, "max": pl.String, "top5": TOP5}
CONCLUSIONS = {
    **_RENDERED, **ESTIMATES, "class": CLASS,
    **{f"inner_{k}": v for k, v in _RENDERED.items()}, **{f"inner_{k}": v for k, v in ESTIMATES.items()}, "inner_class": CLASS,
}

# Agreement: exact unless listed. capture_history depends on each implementation's
# own seeded split, so it is compared through the Schnabel estimate instead.
TOLERANCES = {"entropy": 1e-9, "inner_entropy": 1e-9, "size_zstd_bytes": 0.01, "size_polars_zstd_bytes": 0.01}
SPLIT_DEPENDENT = ("capture_history", "inner_capture_history")
SCHNABEL_RTOL = 0.10

_UINT128 = getattr(pl, "UInt128", None)


def _unsupported(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Object, pl.Null)) or (_UINT128 is not None and dtype == _UINT128):
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _unsupported(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_unsupported(f.dtype) for f in dtype.fields)
    return False


class Describe(Technique):
    """Per-column profile: counts, entropy, cardinality estimates, extremes, lengths,
    float / numeric-string / ISO-datetime scanners, Arrow and Polars sizes, the same
    for list inner values, and a classification (first match wins):
    null → constant → boolean → ordinal → categorical → discrete.

    `population_rows` (int, or dict frame → int) is the size of the population the
    frame samples; it selects the estimator (exact / Duj1) and N in the ordinal rule
    (0 ≤ min, max ≤ 2N). `seed` fixes the 3-way split behind the Schnabel estimate.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = METRICS
    CONCLUSIONS = CONCLUSIONS

    def __init__(
        self,
        *,
        population_rows: int | dict[str, int] | None = None,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        pops = [] if population_rows is None else population_rows.values() if isinstance(population_rows, dict) else [population_rows]
        if any(not isinstance(p, int) or p < 0 for p in pops):
            raise ValueError(f"population_rows must be non-negative ints, got {population_rows!r}")
        if categorical_threshold < 0:
            raise ValueError(f"categorical_threshold must be >= 0, got {categorical_threshold}")
        if not 1 <= zstd_level <= 22:
            raise ValueError(f"zstd_level must be in 1..22, got {zstd_level}")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed}")
        self.population_rows = population_rows
        self.categorical_threshold = categorical_threshold
        self.zstd_level = zstd_level
        self.seed = seed

    def add(self, frames):
        """Rebuild every column that holds an Array or Struct (at any depth) before it
        is registered. py-polars 1.41 exports a *sliced* Array/Struct with nulls as
        inconsistent Arrow (Array: slice offset applied twice; Struct: short child),
        which aborts the Rust plugin's process and makes pyarrow/DataFusion raise.
        A gather over every row produces a fresh, consistent buffer. Lazy-safe."""
        return super().add({n: _normalise(f) for n, f in frames.items()})

    def _population(self, frame: str) -> int | None:
        if isinstance(self.population_rows, dict):
            return self.population_rows.get(frame)
        return self.population_rows

    # ── technique hooks ───────────────────────────────────────────────────────

    def eligible(self, series: pl.Series) -> bool:
        return not _unsupported(series.dtype)

    def describe(self, frames, combos):
        dtypes = {(n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()}
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        columns: dict[str, list] = {k: [] for k in self.CONCLUSIONS}
        for row in out.iter_rows(named=True):
            values = self._conclusions(row) if row["status"] == "computed" else {}
            for k, v in columns.items():
                v.append(values.get(k))
        return out.with_columns(pl.Series(k, v, dtype=self.CONCLUSIONS[k]) for k, v in columns.items())

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        # base.same_value only applies rtol/atol when at least one side is already
        # a Python float; two ints (e.g. UInt64 size_zstd_bytes) compare by ==
        # otherwise. Cast every tolerance-bearing metric to Float64 first so the
        # float branch always triggers, regardless of the metric's own dtype.
        keys = self.key_columns()
        exact = [m for m in self.METRICS if m not in TOLERANCES and m not in SPLIT_DEPENDENT]
        problems = metric_mismatches(result, reference, keys, exact, 0.0, 0.0)
        as_float = lambda df, cols: df.with_columns(pl.col(c).cast(F64) for c in cols)
        for metric, rtol in TOLERANCES.items():
            problems += metric_mismatches(
                as_float(result, [metric]), as_float(reference, [metric]), keys, [metric], rtol, 0.0
            )
        schnabel_cols = ["schnabel", "inner_schnabel"]
        problems += metric_mismatches(
            as_float(result, schnabel_cols), as_float(reference, schnabel_cols), keys, schnabel_cols, SCHNABEL_RTOL, 0.0
        )
        return list(dict.fromkeys(problems))

    # ── conclusions for one computed row ─────────────────────────────────────

    def _conclusions(self, r: dict) -> dict:
        s = self._collected[r["df_a"]][r["col_a"]]
        n_rows = r["n_rows"]
        pop = self._population(r["df_a"])
        if pop is not None and pop < n_rows:
            raise ValueError(f"population_rows {pop} < {n_rows} rows in frame {r['df_a']!r}")
        q = None if pop is None else 1.0 if pop == n_rows else n_rows / pop
        out = self._one_level(s, r, "", n_rows, q, pop if pop is not None else n_rows)
        if r["inner_n_values"] is not None:
            inner_n = r["inner_n_values"]
            out |= self._one_level(flatten(s), r, "inner_", inner_n, q, inner_n / q if q else inner_n)
        return out

    def _one_level(self, s: pl.Series, r: dict, p: str, n_values: int, q: float | None, big_n: float) -> dict:
        n = n_values - r[f"{p}n_null"]
        est = estimators.estimate(r[f"{p}n_unique"], n, r[f"{p}f1"], r[f"{p}f2"], r[f"{p}capture_history"], q)
        at = lambda i: None if i is None else str(s[i])
        return {
            f"{p}min": at(r[f"{p}argmin"]),
            f"{p}max": at(r[f"{p}argmax"]),
            f"{p}top5": [{"value": str(s[i]), "count": c} for i, c in zip(r[f"{p}top5_idx"], r[f"{p}top5_count"])],
            **{f"{p}{k}": v for k, v in est.items()},
            f"{p}class": self._classify(s, r, p, n_values, n, big_n, est["est_cardinality"]),
        }

    def _classify(self, s: pl.Series, r: dict, p: str, n_values: int, n: int, big_n: float, est: float) -> str:
        if r[f"{p}n_null"] == n_values:
            return "null"
        if r[f"{p}n_unique"] == 1:
            return "constant"
        if r[f"{p}n_unique"] == 2:
            return "boolean"
        bounds = _whole_range(s, r, p, n)
        if bounds is not None and 0 <= bounds[0] and bounds[1] <= 2 * big_n:
            return "ordinal"
        if est <= self.categorical_threshold:
            return "categorical"
        return "discrete"


def _holds_array_or_struct(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Array, pl.Struct)):
        return True
    if isinstance(dtype, pl.List):
        return _holds_array_or_struct(dtype.inner)
    return False


def _normalise(frame):
    if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
        return frame  # Technique.add raises the TypeError
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    cols = [c for c, dt in schema.items() if _holds_array_or_struct(dt)]
    if not cols:
        return frame
    return frame.with_columns(pl.col(c).gather(pl.int_range(pl.len())) for c in cols)


def _whole_range(s: pl.Series, r: dict, p: str, n: int) -> tuple[float, float] | None:
    """(min, max) when every non-null value is a whole number, else None. Temporal
    dtypes never qualify (their physical integers are not quantities)."""
    lo, hi = r[f"{p}argmin"], r[f"{p}argmax"]
    if lo is None:
        return None
    dtype = s.dtype
    if isinstance(dtype, INTEGERS) or (isinstance(dtype, pl.Decimal) and dtype.scale == 0):
        return int(s[lo]), int(s[hi])
    if isinstance(dtype, FLOATS) and r[f"{p}n_fractional"] == r[f"{p}n_nan"] == r[f"{p}n_inf"] == 0:
        return s[lo], s[hi]
    if (
        isinstance(dtype, STRING_LIKE)
        and r[f"{p}n_numeric_int"] == n
        and r[f"{p}n_leading_zero"] == 0
        and r[f"{p}numeric_int_min"] is not None
    ):
        return r[f"{p}numeric_int_min"], r[f"{p}numeric_int_max"]
    return None
