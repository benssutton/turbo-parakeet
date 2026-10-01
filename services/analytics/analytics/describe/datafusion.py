"""DescribeDataFusion — Apache DataFusion SQL over each column's Arrow data."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from datafusion import SessionContext

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
    NUMERIC,
    NUMERIC_INT,
    STRING_LIKE,
    flatten,
    frac_digits,
    frequency_summary,
    n_midnight,
    subsets,
)
from analytics.describe.base import (
    GROUP_B,
    GROUP_C,
    LEVEL_INPUTS,
    VALUE_METRICS,
    Describe,
)
from analytics.describe.polars import profile as polars_profile
from analytics.gcd.base import INTEGER_BACKED
from analytics.gcd.math import math_gcd


class DescribeDataFusion(Describe):
    """Each column (and its flattened inner values) is registered as an Arrow table
    `t(row, sub, v, o)` and profiled with DataFusion SQL: GROUP BY for frequencies
    (COUNT, MIN(row), BIT_OR of the split subset), ORDER BY … LIMIT 1 for
    first-occurrence extremes, regexp_like (Rust regex: linear time) and
    try_cast(… AS DATE) for the string scanners. Enum columns order by their
    physical codes (`o`); Categorical/Enum values are scanned as strings (`v`).

    Computed outside SQL — exactly, never approximated:
      - sizes: the shared pyarrow helper (_sizes.column_sizes);
      - inner values: the shared Polars `flatten` (DataFusion's unnest drops null elements);
      - n_midnight: Polars dt.time() (timezone-aware date_trunc is not relied on);
      - f1 / f2 / capture history: numpy over the GROUP BY result;
      - min_len/max_len of Binary: pyarrow binary_length (octet_length takes only strings);
      - every group A metric of List/Array/Struct columns whose values hold floats,
        Enum or Categorical: the Polars reference helpers (SQL cannot key nested
        -0.0/NaN, list-of-dictionary children or Enum order exactly);
      - max_frac_digits: the shared parser over CAST(v AS VARCHAR) of distinct finite values;
      - gcd: Python math.gcd over the physical values (as GcdMath; SQL has no exact GCD aggregate);
      - sum_len / sum_len_unique of Binary: pyarrow binary_length.
    """

    def _compute(self, frames, combos):
        ctx = SessionContext()
        rows = [self._row(ctx, frames[n][c]) for ((n, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in {**self.METRICS, **self.INPUTS}}
        )

    def _row(self, ctx: SessionContext, s: pl.Series) -> dict:
        row = {
            "n_rows": s.len(),
            "n_null": s.null_count(),
            **self._profile(ctx, s),
            "n_midnight": n_midnight(s),
            **column_sizes(s, self.zstd_level),
        }
        inner = flatten(s) if isinstance(s.dtype, (pl.List, pl.Array)) else None
        row["inner_n_values"] = None if inner is None else inner.len()
        row["inner_n_null"] = None if inner is None else inner.null_count()
        inner_profile = (
            dict.fromkeys({**VALUE_METRICS, **LEVEL_INPUTS})
            if inner is None
            else self._profile(ctx, inner)
        )
        return row | {f"inner_{k}": v for k, v in inner_profile.items()}

    def _profile(self, ctx: SessionContext, s: pl.Series) -> dict:
        if _needs_polars(s.dtype):
            return polars_profile(s, self.seed)
        n = s.len()
        v = s.cast(pl.String) if isinstance(s.dtype, (pl.Categorical, pl.Enum)) else s
        o = s.to_physical() if isinstance(s.dtype, pl.Enum) else v
        table = pa.table(
            {
                "row": np.arange(n, dtype=np.uint64),
                "sub": subsets(n, self.seed),
                "v": _arrow(v),
                "o": _arrow(o),
            }
        )
        ctx.register_record_batches(
            "t",
            [
                table.to_batches()
                or [pa.RecordBatch.from_pylist([], schema=table.schema)]
            ],
        )
        try:
            return {
                **_frequencies(ctx, s),
                **_extremes(ctx, s),
                **_lengths(ctx, s),
                **_totals(ctx, s),
                **_floats(ctx, s),
                **_strings(ctx, s),
            }
        finally:
            ctx.deregister_table("t")


_INEXACT_IN_SQL = (pl.Float32, pl.Float64, pl.Enum, pl.Categorical)


def _needs_polars(dtype: pl.DataType, nested: bool = False) -> bool:
    """True for List/Array/Struct values that hold floats, Enum or Categorical:
    DataFusion cannot key them exactly (-0.0 and NaN payloads inside nested values,
    dictionary children in lists, Enum category order)."""
    if isinstance(dtype, (pl.List, pl.Array)):
        return _needs_polars(dtype.inner, True)
    if isinstance(dtype, pl.Struct):
        return any(_needs_polars(f.dtype, True) for f in dtype.fields)
    return nested and isinstance(dtype, _INEXACT_IN_SQL)


def _arrow(s: pl.Series) -> pa.Array:
    return s.rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())


def _one(ctx: SessionContext, sql: str) -> dict:
    return ctx.sql(sql).to_arrow_table().to_pylist()[0]


def _key(dtype: pl.DataType, col: str) -> str:
    """Canonical grouping/ordering key: one NaN, and -0.0 → 0.0 for floats.

    DataFusion compares floats by total order (-0.0 = 0 is false), so the sign of
    zero is dropped arithmetically: IEEE -0.0 + 0.0 = +0.0."""
    if not isinstance(dtype, FLOATS):
        return col
    t = "REAL" if dtype == pl.Float32 else "DOUBLE"
    return f"CASE WHEN isnan({col}) THEN CAST('NaN' AS {t}) ELSE {col} + CAST(0 AS {t}) END"


def _frequencies(ctx: SessionContext, s: pl.Series) -> dict:
    key = _key(s.dtype, "v")
    freq = ctx.sql(
        f"SELECT {key} AS k, COUNT(*) AS c, BIT_OR(CAST(1 AS BIGINT) << sub) AS m "
        f"FROM t WHERE v IS NOT NULL GROUP BY {key}"
    ).to_arrow_table()
    return frequency_summary(freq["c"].to_numpy(), freq["m"].to_numpy())


def _extremes(ctx: SessionContext, s: pl.Series) -> dict:
    ok = "o IS NOT NULL" + (" AND NOT isnan(o)" if isinstance(s.dtype, FLOATS) else "")
    key = _key(s.dtype, "o")
    r = _one(
        ctx,
        f"SELECT (SELECT row FROM t WHERE {ok} ORDER BY {key} ASC NULLS FIRST, row ASC LIMIT 1) AS lo, "
        f"(SELECT row FROM t WHERE {ok} ORDER BY {key} DESC NULLS LAST, row ASC LIMIT 1) AS hi",
    )
    return {"argmin": r["lo"], "argmax": r["hi"]}


def _lengths(ctx: SessionContext, s: pl.Series) -> dict:
    if s.dtype == pl.Binary:  # octet_length() accepts only strings in DataFusion SQL
        lens = pc.binary_length(_arrow(s))
        return {"min_len": pc.min(lens).as_py(), "max_len": pc.max(lens).as_py()}
    if isinstance(s.dtype, STRING_LIKE):
        expr = "octet_length(v)"
    elif isinstance(s.dtype, (pl.List, pl.Array)):
        expr = "array_length(v)"
    else:
        return {"min_len": None, "max_len": None}
    r = _one(
        ctx, f"SELECT MIN({expr}) AS lo, MAX({expr}) AS hi FROM t WHERE v IS NOT NULL"
    )
    return {"min_len": r["lo"], "max_len": r["hi"]}


def _totals(ctx: SessionContext, s: pl.Series) -> dict:
    out = {
        "gcd": math_gcd(s) if isinstance(s.dtype, INTEGER_BACKED) else None,
        "sum_len": None,
        "sum_len_unique": None,
    }
    if s.dtype == pl.Binary:  # octet_length() accepts only strings in DataFusion SQL
        arr = _arrow(s)
        out["sum_len"] = pc.sum(pc.binary_length(arr)).as_py() or 0
        out["sum_len_unique"] = (
            pc.sum(pc.binary_length(pc.unique(arr.drop_null()))).as_py() or 0
        )
    elif isinstance(s.dtype, STRING_LIKE):
        r = _one(
            ctx,
            "SELECT SUM(octet_length(v)) AS total, "
            "(SELECT SUM(octet_length(u.v)) FROM (SELECT DISTINCT v FROM t WHERE v IS NOT NULL) u) AS uniq FROM t",
        )
        out["sum_len"], out["sum_len_unique"] = r["total"] or 0, r["uniq"] or 0
    return out


def _floats(ctx: SessionContext, s: pl.Series) -> dict:
    if not isinstance(s.dtype, FLOATS):
        return dict.fromkeys(GROUP_B)
    t = "REAL" if s.dtype == pl.Float32 else "DOUBLE"
    finite = f"(NOT isnan(v) AND abs(v) <> CAST('Infinity' AS {t}))"
    r = _one(
        ctx,
        f"""SELECT
              SUM(CASE WHEN isnan(v) THEN 1 ELSE 0 END) AS nan,
              SUM(CASE WHEN NOT isnan(v) AND abs(v) = CAST('Infinity' AS {t}) THEN 1 ELSE 0 END) AS inf,
              SUM(CASE WHEN {finite} AND (v + CAST(0 AS {t})) <> trunc(v + CAST(0 AS {t})) THEN 1 ELSE 0 END) AS frac,
              SUM(CASE WHEN {finite} AND CAST(CAST(v AS REAL) AS DOUBLE) <> CAST(v AS DOUBLE) THEN 1 ELSE 0 END) AS f32
            FROM t WHERE v IS NOT NULL""",
    )
    reprs = ctx.sql(
        f"SELECT DISTINCT CAST(v AS VARCHAR) AS r FROM t WHERE v IS NOT NULL AND {finite}"
    ).to_arrow_table()
    return {
        "n_nan": r["nan"] or 0,
        "n_inf": r["inf"] or 0,
        "n_fractional": r["frac"] or 0,
        "max_frac_digits": frac_digits(
            pl.Series(reprs["r"].to_pylist(), dtype=pl.String)
        ),
        "n_f32_inexact": None if s.dtype == pl.Float32 else r["f32"] or 0,
    }


def _strings(ctx: SessionContext, s: pl.Series) -> dict:
    if not isinstance(s.dtype, STRING_LIKE):
        return dict.fromkeys(GROUP_C)
    sig = "length(regexp_replace(v, '^-?0*', ''))"
    off = f"regexp_match(v, '{ISO_OFFSET}')[1]"
    r = _one(
        ctx,
        f"""WITH s AS (
              SELECT v,
                regexp_like(v, '{NUMERIC}') AS num, regexp_like(v, '{NUMERIC_INT}') AS nint,
                try_cast(substr(v, 1, 10) AS DATE) IS NOT NULL AS dok,
                regexp_like(v, '{ISO_DATE}') AS d, regexp_like(v, '{ISO_TIME}') AS tm,
                regexp_like(v, '{ISO_DATETIME}') AS dt, regexp_like(v, '{ISO_DATETIME_TZ}') AS tz
              FROM t WHERE v IS NOT NULL)
            SELECT
              SUM(CAST(num AS BIGINT)) AS n_numeric,
              SUM(CAST(nint AS BIGINT)) AS n_numeric_int,
              SUM(CAST(regexp_like(v, '{LEADING_ZERO}') AS BIGINT)) AS n_leading_zero,
              MAX(CASE WHEN nint THEN {sig} END) AS int_sig,
              MIN(CASE WHEN nint AND {sig} <= 38 THEN CAST(v AS DECIMAL(38, 0)) END) AS int_min,
              MAX(CASE WHEN nint AND {sig} <= 38 THEN CAST(v AS DECIMAL(38, 0)) END) AS int_max,
              MAX(CASE WHEN num THEN length(regexp_match(v, '{INT_DIGITS}')[1]) END) AS int_digits,
              MAX(CASE WHEN num THEN COALESCE(length(regexp_match(v, '{FRAC_DIGITS}')[1]), 0) END) AS frac_digits,
              MIN(CASE WHEN num THEN COALESCE(length(regexp_match(v, '{FRAC_DIGITS}')[1]), 0) END) AS min_frac_digits,
              MAX(CASE WHEN num THEN length(ltrim(concat(COALESCE(regexp_match(v, '{INT_DIGITS}')[1], ''), COALESCE(regexp_match(v, '{FRAC_DIGITS}')[1], '')), '0')) END) AS sig_digits,
              SUM(CAST(d AND dok AS BIGINT)) AS n_iso_date,
              SUM(CAST(tm AS BIGINT)) AS n_iso_time,
              SUM(CAST(dt AND dok AS BIGINT)) AS n_iso_datetime,
              SUM(CAST(tz AND dok AS BIGINT)) AS n_iso_datetime_tz,
              MAX(CASE WHEN tm OR ((dt OR tz) AND dok) THEN COALESCE(length(regexp_match(v, '{ISO_FRACTION}')[1]), 0) END) AS iso_frac,
              MAX(CASE WHEN tm OR ((dt OR tz) AND dok) THEN COALESCE(length(rtrim(regexp_match(v, '{ISO_FRACTION}')[1], '0')), 0) END) AS iso_sig,
              COUNT(DISTINCT CASE WHEN tz AND dok THEN (CASE WHEN {off} IN ('Z', '-00:00') THEN '+00:00' ELSE {off} END) END) AS iso_n_offsets,
              SUM(CAST((dt OR tz) AND dok AND regexp_like(v, '{ISO_MIDNIGHT}') AS BIGINT)) AS iso_n_midnight
            FROM s""",
    )
    in_range = r["int_sig"] is not None and r["int_sig"] <= 38
    return {
        "n_numeric": r["n_numeric"] or 0,
        "n_numeric_int": r["n_numeric_int"] or 0,
        "n_leading_zero": r["n_leading_zero"] or 0,
        "numeric_int_min": int(r["int_min"]) if in_range else None,
        "numeric_int_max": int(r["int_max"]) if in_range else None,
        "numeric_max_int_digits": r["int_digits"],
        "numeric_max_frac_digits": r["frac_digits"],
        "numeric_min_frac_digits": r["min_frac_digits"],
        "numeric_max_sig_digits": r["sig_digits"],
        "n_iso_date": r["n_iso_date"] or 0,
        "n_iso_time": r["n_iso_time"] or 0,
        "n_iso_datetime": r["n_iso_datetime"] or 0,
        "n_iso_datetime_tz": r["n_iso_datetime_tz"] or 0,
        "iso_max_frac_digits": r["iso_frac"],
        "iso_max_sig_frac_digits": r["iso_sig"],
        "iso_n_offsets": r["iso_n_offsets"] or 0,
        "iso_n_midnight": r["iso_n_midnight"] or 0,
    }
