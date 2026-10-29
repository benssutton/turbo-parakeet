"""Build query variants and the SQL predicates that target a selectivity.

The two predicate kinds:

- `range`: a contiguous interval of the *sorted* values covering the target
  fraction, starting at a random offset so it is not always the tail of the
  distribution.
- `in`: the K most frequent values whose combined frequency is closest to
  the target fraction (see `in_predicate`'s docstring for the exact
  algorithm).

Both predicate builders record the *actual* achieved selectivity, computed
from the data, never assumed from the request: duplicate values straddling a
range boundary, or a target fraction that lands between two IN-list
candidates, both mean the achieved fraction can differ slightly from the
target, and the report needs the real number.

`range_predicate` and `in_predicate` take the dataset's materialised numpy
values plus an optional `ch_type` (the variant's column type). Rendering
**must** follow `ch_type`, not the value's runtime type: the harness's
generator produces plain numbers even for `Decimal`/`Date`/`DateTime64`/
`String` columns (the loader casts them at insert time, per
`chbench.harness.loader.cast_column`), so the same raw float or int has to
render differently depending on which column it is destined for. Two
failures this must avoid, both confirmed against a live server:

- A bare decimal literal such as `v IN (-0.54929124)` is parsed by
  ClickHouse as `Float64`, and the `IN` set is converted to the column type
  through float arithmetic, which truncates — `materialize(toDecimal32(
  '-0.54929124', 8)) IN (-0.54929124)` returns 0 there, while the quoted
  string form `IN ('-0.54929124')` returns 1. So `Decimal*` literals render
  as a quoted decimal string at the type's own scale.
- `materialize('1234567') BETWEEN 1000000 AND 2000000` errors (code 386, no
  common type) against a `String` column: the value must be quoted, and
  stringified first if it is not already a string (matching how the loader
  casts a numeric dataset into a `String` column).

`ch_type` is keyword-only and defaults to `None`, so that a caller with no
`TableVariant` in hand (as the original test suite for this module has)
falls back to dispatching on the value's own runtime type instead: `str` /
`numpy.str_` quoted and escaped, `numpy.datetime64` / `datetime.date` /
`datetime.datetime` quoted ISO, everything else (including
`decimal.Decimal`, which already carries its own scale) bare. `expand_queries`
always has a `TableVariant` and always passes `variant.ch_type` through.

Both predicate builders also strip null representations (`NaN` for a
float64 array — which is what a nullable Arrow column becomes once
materialised to numpy — `NaT` for a datetime64 array, `None` inside an
object array) before sorting or counting: a null can never satisfy a SQL
predicate, so it must never end up chosen as an IN value or as one end of a
BETWEEN range, and the *achieved* fraction is still divided by the full row
count (nulls included), since that is the population the target fraction is
relative to.

`full_scan_sql`, which is handed a `TableVariant` directly, decides its
numeric-vs-string branch from `ch_type` the same way.
"""

from __future__ import annotations

import datetime
import decimal
import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from chbench.harness.loader import arrow_type_for, base_type
from chbench.harness.suite import QueryAxes
from chbench.harness.validate import TableVariant

_DECIMAL_SCALE_RE = re.compile(r"^Decimal\d+\((\d+)\)$")
_TIME64_SCALE_RE = re.compile(r"^Time64\((\d)\)$")
_UNIT_PRECISION = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def query_values(array: pa.Array) -> np.ndarray:
    """The dataset column as the numpy values the predicate builders sort and
    count. `to_numpy(zero_copy_only=False)` except where that loses the value:
    a time32 / time64 column becomes its integer ticks since midnight (numpy
    has no such type, and pyarrow refuses nanoseconds), and a 64-bit integer
    column with nulls stays exact as Python ints (the float64 numpy would
    widen it to rounds above 2**53)."""
    t = array.type
    if pa.types.is_time32(t):
        array = array.cast(pa.int32())
    elif pa.types.is_time64(t):
        array = array.cast(pa.int64())
    elif pa.types.is_integer(t) and t.bit_width == 64 and array.null_count:
        return np.array(array.to_pylist(), dtype=object)
    return array.to_numpy(zero_copy_only=False)


@dataclass(frozen=True)
class QueryVariant:
    kind: str
    predicate_kind: str | None
    target_fraction: float | None
    values_n: int | None
    predicate_sql: str | None
    actual_fraction: float | None


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _render_literal_runtime(value) -> str:
    """Render one value as a ClickHouse SQL literal, dispatching on the
    value's own runtime type. Used only when no `ch_type` is available; see
    `_render_literal`."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (str, np.str_)):
        return _quote(str(value))
    if isinstance(value, np.datetime64):
        return _quote(str(np.datetime_as_string(value)).replace("T", " "))
    if isinstance(value, (datetime.date, datetime.datetime)):
        return _quote(value.isoformat().replace("T", " "))
    return str(value)


def _render_literal_typed(value, ch_type: str) -> str:
    """Render one value as a ClickHouse SQL literal for the given column
    type (unwrapping `Nullable`/`LowCardinality`). See the module docstring
    for why this must key off `ch_type` and not the value's runtime type."""
    t = base_type(ch_type)
    if t == "String":
        if isinstance(value, (str, np.str_)):
            text = str(value)
        else:
            # Match the loader's own cast exactly (`cast_column` ->
            # `pc.cast`), rather than Python's `str()`: pyarrow's numeric
            # cast to string does not agree with `str()` in general — e.g.
            # `3.0` -> `'3'` (not `'3.0'`) and `1e10` -> `'1e+10'` (not
            # `'10000000000.0'`). A literal built with `str()` would then
            # never match the stored row for a whole-number value.
            text = pa.array([value]).cast(pa.string())[0].as_py()
        return _quote(text)
    if t.startswith("Decimal"):
        m = _DECIMAL_SCALE_RE.match(t)
        places = int(m.group(1)) if m else 0
        if isinstance(value, decimal.Decimal):
            return _quote(f"{value:.{places}f}")
        return _quote(f"{float(value):.{places}f}")
    if t.startswith("Enum"):
        return _quote(str(value))
    if t.startswith("FixedString"):
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value).rstrip(b"\0")
            try:
                return _quote(data.decode("utf-8"))
            except UnicodeDecodeError:
                return f"unhex('{data.hex()}')"
        return _quote(str(value))
    if t == "Bool":
        return "true" if bool(value) else "false"
    m = _TIME64_SCALE_RE.match(t)
    if m:
        # `value` is the integer ticks since midnight in the dataset's unit,
        # which is the column's own precision (`resolve_type`).
        places = int(m.group(1))
        seconds, ticks = divmod(int(value), 10**places)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return _quote(text + (f".{ticks:0{places}d}" if places else ""))
    if t in ("Date", "Date32") or t.startswith("DateTime"):
        # The raw value is whatever numeric/date-like scalar the generator
        # produced; convert it the way the loader does (`cast_column` ->
        # `pc.cast`, from an Arrow array of the dataset's own type), so the
        # rendered literal names the exact date the loaded row actually
        # has. It goes through numpy, not a Python list: on pyarrow 24
        # `pa.array([np.datetime64('1970-01-02', 'D')])`, with or without a
        # `type=`, raises `TypeError: int() argument must be ... not
        # 'datetime.date'`, which is what crashed every `Date` filter query,
        # while the numpy path infers `date32` and casts cleanly.
        # A nullable integer dataset reaches here as float64 (numpy has no
        # nullable int, so `to_numpy` widens and puts NaN in the holes), and
        # Arrow has no double -> date cast; the loader's array is still
        # integer-typed, so recover the integer before casting.
        target = arrow_type_for(ch_type)
        raw = np.asarray([value])
        if raw.dtype.kind == "f":
            raw = raw.astype(np.int32 if pa.types.is_date32(target) else np.int64)
        casted = pa.array(raw).cast(target)
        if not pa.types.is_timestamp(target) or (not target.tz and target.unit != "ns"):
            return _quote(casted[0].as_py().isoformat().replace("T", " "))
        # A nanosecond or zoned timestamp: Python's datetime has microseconds
        # and local-time zones, so spell the UTC instant at the column's
        # precision through numpy, and for a zoned column name the zone the
        # string is in -- a bare string would be read in the column's own zone.
        naive = casted.cast(pa.timestamp(target.unit)) if target.tz else casted
        text = _quote(
            np.datetime_as_string(
                naive.to_numpy(zero_copy_only=False)[0], unit=target.unit
            ).replace("T", " ")
        )
        if not target.tz:
            return text
        if t.startswith("DateTime64"):
            return f"toDateTime64({text}, {_UNIT_PRECISION[target.unit]}, 'UTC')"
        return f"toDateTime({text}, 'UTC')"
    return str(value)


def _render_literal(value, ch_type: str | None = None) -> str:
    """Render one value as a ClickHouse SQL literal for `ch_type` (a
    `TableVariant.ch_type`), or, when `ch_type` is `None`, by dispatching on
    the value's own runtime type. See the module docstring."""
    if ch_type is not None:
        return _render_literal_typed(value, ch_type)
    return _render_literal_runtime(value)


def _non_null_mask(v: np.ndarray) -> np.ndarray:
    """True for elements that are not a null representation: `NaN` for a
    float array (what a nullable Arrow column becomes once materialised to
    numpy), `NaT` for a datetime64 array, `None`/`NaN` inside an object
    array. Every other dtype has no null representation here and is all
    True."""
    kind = v.dtype.kind
    if kind == "f":
        return ~np.isnan(v)
    if kind == "M":
        return ~np.isnat(v)
    if kind == "O":

        def _is_null(x):
            return x is None or (isinstance(x, float) and math.isnan(x))

        return np.array([not _is_null(x) for x in v], dtype=bool)
    return np.ones(len(v), dtype=bool)


def range_predicate(
    values: np.ndarray,
    target_fraction: float,
    rng: np.random.Generator,
    *,
    ch_type: str | None = None,
) -> tuple[str, float]:
    """A `v BETWEEN a AND b` predicate covering `target_fraction` of `values`.

    Null representations (see `_non_null_mask`) are stripped before sorting:
    a null can never satisfy `BETWEEN`, so it must never become one of the
    window's endpoints. The window is a contiguous slice of the *non-null*
    sorted values, of length `round(target_fraction * n)` (`n` including
    nulls, then clamped to `[1, count of non-null values]`), starting at a
    uniformly random offset among the offsets that keep the window in
    bounds — so the interval is not always anchored at the tail. The
    achieved fraction is recomputed from the non-null values themselves
    (`a <= v <= b`) and divided by `n` including nulls, so duplicate values
    straddling either boundary are correctly counted in or out and the
    fraction stays relative to every row, not just the non-null ones.

    `ch_type` (a `TableVariant.ch_type`) decides literal rendering; see the
    module docstring for why it must, and `_render_literal` for the
    fallback when it is omitted.
    """
    v = np.asarray(values)
    n = len(v)
    clean = v[_non_null_mask(v)]
    n_clean = len(clean)
    if n_clean == 0:
        raise ValueError(
            "range_predicate: values are all null; there is no "
            "non-null value to build a range from"
        )
    sorted_v = np.sort(clean)
    window = min(n_clean, max(1, round(target_fraction * n)))
    max_start = n_clean - window
    start = int(rng.integers(0, max_start + 1))
    lo = sorted_v[start]
    hi = sorted_v[start + window - 1]
    matched = np.count_nonzero((clean >= lo) & (clean <= hi))
    actual = float(matched) / n
    sql = f"v BETWEEN {_render_literal(lo, ch_type)} AND {_render_literal(hi, ch_type)}"
    return sql, actual


def in_predicate(
    values: np.ndarray,
    target_fraction: float,
    n_values: int,
    *,
    ch_type: str | None = None,
) -> tuple[str, float]:
    """A `v IN (...)` predicate whose combined frequency is closest to
    `target_fraction` among sets of `n_values` distinct values.

    Algorithm: a K-step greedy search (K = `n_values`, or the number of
    distinct values if fewer) over the *non-null* values (see
    `_non_null_mask` — a null can never satisfy `IN`, so it must never be
    chosen as one of the values). At each step, from the values not yet
    chosen, pick whichever one's addition to the running combined frequency
    brings that running frequency, divided by the full row count including
    nulls, closest to `target_fraction`; ties are broken toward the more
    frequent value, and any further tie toward the value that sorts first
    (for full determinism, e.g. when two values are exactly equally
    frequent). Repeat until K values are chosen. This is a greedy
    approximation, not an exhaustive search over all size-K subsets, but
    for K = 1 it is exact: this module's K=1 test case
    (frequencies 50/30/10/5/2/1/1/1 out of 100, target 0.20) has 30/100 and
    10/100 tied at distance 0.10 from the target; the tie-break picks the
    more frequent value (30/100).

    `ch_type` (a `TableVariant.ch_type`) decides literal rendering; see the
    module docstring for why it must, and `_render_literal` for the
    fallback when it is omitted.
    """
    v = np.asarray(values)
    n = len(v)
    clean = v[_non_null_mask(v)]
    uniq, counts = np.unique(clean, return_counts=True)
    if len(uniq) == 0:
        raise ValueError(
            "in_predicate: values are all null; there is no "
            "non-null value to build an IN list from"
        )
    k = min(n_values, len(uniq))
    remaining = list(range(len(uniq)))
    chosen: list[int] = []
    running = 0
    for _ in range(k):
        best = min(
            remaining,
            key=lambda idx: (
                abs((running + counts[idx]) / n - target_fraction),
                -counts[idx],
                idx,
            ),
        )
        chosen.append(best)
        running += counts[best]
        remaining.remove(best)
    literals = ", ".join(_render_literal(uniq[i], ch_type) for i in chosen)
    sql = f"v IN ({literals})"
    return sql, running / n


def _aggregate_select(variant: TableVariant) -> str:
    """`count(), sum(toFloat64(v))` for numeric types (including decimals,
    dates, timestamps, Bool and Time64), `count(), sum(length(v))` for
    String and FixedString, `count(), sum(length(toString(v)))` for the enums
    (which have no numeric sum a reader would recognise): every value is read
    but nothing is shipped to the client."""
    base = base_type(variant.ch_type)
    if base == "String" or base.startswith("FixedString"):
        return "count(), sum(length(v))"
    if base.startswith("Enum"):
        return "count(), sum(length(toString(v)))"
    return "count(), sum(toFloat64(v))"


def full_scan_sql(name: str, variant: TableVariant) -> str:
    """`SELECT count(), sum(toFloat64(v))` for numeric types (including
    decimals, dates and timestamps), `SELECT count(), sum(length(v))` for
    strings — every value is read but nothing is shipped to the client."""
    return f"SELECT {_aggregate_select(variant)} FROM {name}"


def filter_sql(name: str, variant: TableVariant, predicate_sql: str) -> str:
    """The full-scan aggregate (`_aggregate_select`) with `WHERE
    predicate_sql` appended, so the filter reads the matched column values.

    A bare `SELECT count() FROM t WHERE <primary-key range>` does not: on
    ClickHouse 26.8 it is answered through the exact-count projection
    (`EXPLAIN` shows `AggregatingProjection` over `ReadFromPreparedSource
    (_exact_count_projection)`), which counts fully-covered granules from
    the primary index and reads only the boundary granules. Measured on a
    200 000-row `Float64` table sorted by `v` with a 10 % range filter
    (measured in an earlier run): `EXPLAIN indexes = 1` selected `Granules: 4/25`, yet
    `SelectedMarks = 2`, `SelectedRanges = 2` and `read_rows = 16385` for
    20 000 matched rows. That measures index lookup, not column
    decompression, and flatters every primary-key variant; the aggregate
    over `v` forces every matched granule to be read.
    """
    return f"SELECT {_aggregate_select(variant)} FROM {name} WHERE {predicate_sql}"


def expand_queries(
    axes: QueryAxes, values: np.ndarray, variant: TableVariant, rng: np.random.Generator
) -> list[QueryVariant]:
    """One `QueryVariant` per `axes.kinds` entry, in order (`axes.samples` is the repeat count,
    not an axis). A `range` predicate consumes `rng`; an `in` predicate is deterministic.
    """
    variants: list[QueryVariant] = []
    for kind in axes.kinds:
        predicate_kind: str | None = None
        predicate_sql: str | None = None
        actual_fraction: float | None = None
        values_n: int | None = None

        if kind.kind == "filter":
            predicate_kind = kind.predicate
            values_n = kind.values
            if predicate_kind == "range":
                predicate_sql, actual_fraction = range_predicate(
                    values, kind.target_fraction, rng, ch_type=variant.ch_type
                )
            elif predicate_kind == "in":
                predicate_sql, actual_fraction = in_predicate(
                    values, kind.target_fraction, kind.values, ch_type=variant.ch_type
                )
            else:
                raise ValueError(f"unknown predicate kind {predicate_kind!r}")

        variants.append(
            QueryVariant(
                kind=kind.kind,
                predicate_kind=predicate_kind,
                target_fraction=kind.target_fraction,
                values_n=values_n,
                predicate_sql=predicate_sql,
                actual_fraction=actual_fraction,
            )
        )
    return variants


def query_id(table_id: str, variant: QueryVariant) -> str:
    """A 16-hex hash of `table_id` and every field of `variant` that affects
    the query. Stable across calls with the same inputs; changes if any
    field changes."""
    key = "\x1f".join(
        [
            table_id,
            variant.kind,
            str(variant.predicate_kind),
            str(variant.target_fraction),
            str(variant.values_n),
            str(variant.predicate_sql),
            str(variant.actual_fraction),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
