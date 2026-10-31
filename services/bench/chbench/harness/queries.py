"""Build query variants and the SQL predicates that target a selectivity.

The two predicate kinds:

- `range`: a contiguous interval of the *sorted* values covering the target
  fraction, starting at a random offset so it is not always the tail of the
  distribution.
- `in`: the K most frequent values whose combined frequency is closest to
  the target fraction (see `in_predicate`'s docstring for the exact
  algorithm).

A filter is built in two steps. The engine-neutral one (`choose_range`, `choose_in`, and
`choose_queries` for a suite's query axes) chooses the values from the data and computes the
achieved fraction; one choice serves every engine, so all of them query the same values. The
renderers write it for one engine: `render_clickhouse` (the ClickHouse SQL this module always
wrote, byte for byte), `render_datafusion` (typed DataFusion SQL literals) and
`render_polars_text` (the stored text of the Polars expression the worker builds). The
in-process engines' answers are checked against `reference_result`, computed with pyarrow.

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
import pyarrow.compute as pc

from chbench.harness.inproc_types import arrow_family, parse_datafusion_type
from chbench.harness.loader import arrow_type_for, base_type
from chbench.harness.suite import QueryAxes
from chbench.harness.validate import TableVariant

_DECIMAL_SCALE_RE = re.compile(r"^Decimal\d+\((\d+)\)$")
_TIME64_SCALE_RE = re.compile(r"^Time64\((\d)\)$")
_INTEGER_RE = re.compile(r"^U?Int(8|16|32|64|128|256)$")
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
        if isinstance(value, (bytes, bytearray)):
            # a binary path dataset, loaded as it is (`cast_column`)
            try:
                text = bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                return f"unhex('{bytes(value).hex()}')"
        elif isinstance(value, (str, np.str_)):
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
    if _INTEGER_RE.match(t) and isinstance(value, (float, np.floating)):
        # a nullable Int8-Int32 column is float64 here (exact: below 2**53); 64-bit ones stay
        # Python ints (`query_values`)
        return str(int(value))
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


@dataclass(frozen=True)
class ValueChoice:
    """The engine-neutral half of a filter: the values its predicate names, chosen from the data,
    and the achieved fraction. `range`: `values` is `[lo, hi]`, both bounds inclusive; `in`: the chosen
    values in choice order. The values are elements of the numpy array `query_values` gave.
    """

    predicate_kind: str
    values: np.ndarray
    actual_fraction: float


def choose_range(
    values: np.ndarray, target_fraction: float, rng: np.random.Generator
) -> ValueChoice:
    """The window of `range_predicate` (see its docstring) as a `ValueChoice`."""
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
    return ValueChoice(
        "range", sorted_v[[start, start + window - 1]], float(matched) / n
    )


def choose_in(values: np.ndarray, target_fraction: float, n_values: int) -> ValueChoice:
    """The values of `in_predicate` (see its docstring) as a `ValueChoice`, in choice order."""
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
    return ValueChoice("in", uniq[chosen], running / n)


def render_clickhouse(choice: ValueChoice, ch_type: str | None = None) -> str:
    """The ClickHouse predicate of `choice` for a column of `ch_type` (`_render_literal`)."""
    if choice.predicate_kind == "range":
        lo, hi = choice.values
        return f"v BETWEEN {_render_literal(lo, ch_type)} AND {_render_literal(hi, ch_type)}"
    literals = ", ".join(_render_literal(x, ch_type) for x in choice.values)
    return f"v IN ({literals})"


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
    fallback when it is omitted. `choose_range` + `render_clickhouse`.
    """
    choice = choose_range(values, target_fraction, rng)
    return render_clickhouse(choice, ch_type), choice.actual_fraction


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
    fallback when it is omitted. `choose_in` + `render_clickhouse`.
    """
    choice = choose_in(values, target_fraction, n_values)
    return render_clickhouse(choice, ch_type), choice.actual_fraction


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
    strings — every value is read but nothing is shipped to the client. A String
    scan disables `optimize_functions_to_subcolumns`, which would answer
    `length(v)` from the size subcolumn alone (FINDINGS §12)."""
    sql = f"SELECT {_aggregate_select(variant)} FROM {name}"
    if base_type(variant.ch_type) == "String":
        sql += " SETTINGS optimize_functions_to_subcolumns = 0"
    return sql


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


def choose_queries(
    axes: QueryAxes, values: np.ndarray, rng: np.random.Generator
) -> list[ValueChoice | None]:
    """One entry per `axes.kinds` entry, in order: None for a full scan, the `ValueChoice` of a
    filter. A `range` consumes `rng`; an `in` is deterministic. Engine-neutral: every engine's
    queries of a suite are rendered from one call."""
    out: list[ValueChoice | None] = []
    for kind in axes.kinds:
        if kind.kind != "filter":
            out.append(None)
        elif kind.predicate == "range":
            out.append(choose_range(values, kind.target_fraction, rng))
        elif kind.predicate == "in":
            out.append(choose_in(values, kind.target_fraction, kind.values))
        else:
            raise ValueError(f"unknown predicate kind {kind.predicate!r}")
    return out


def render_predicate(
    choice: ValueChoice, variant: TableVariant, dataset_type: pa.DataType | None = None
) -> str:
    """`choice` written for `variant`'s engine: ClickHouse SQL (`render_clickhouse`), DataFusion
    SQL (`render_datafusion`; needs the dataset's Arrow type) or the Polars expression's text
    (`render_polars_text`)."""
    engine = variant.engine
    if engine == "clickhouse":
        return render_clickhouse(choice, variant.ch_type)
    if engine not in ("datafusion", "polars"):
        raise ValueError(f"no predicate renderer for engine {engine!r}")
    values = choice_values_arrow(choice, dataset_type)
    if engine == "datafusion":
        return render_datafusion(
            choice.predicate_kind, values, parse_datafusion_type(variant.type_spec)
        )
    return render_polars_text(choice.predicate_kind, values)


def render_queries(
    axes: QueryAxes,
    choices: list[ValueChoice | None],
    variant: TableVariant,
    dataset_type: pa.DataType | None = None,
) -> list[QueryVariant]:
    """One `QueryVariant` per `axes.kinds` entry and its choice (`choose_queries`), the predicate
    written for `variant`'s engine (`render_predicate`)."""
    out: list[QueryVariant] = []
    for kind, choice in zip(axes.kinds, choices, strict=True):
        is_filter = kind.kind == "filter"
        out.append(
            QueryVariant(
                kind=kind.kind,
                predicate_kind=kind.predicate if is_filter else None,
                target_fraction=kind.target_fraction,
                values_n=kind.values if is_filter else None,
                predicate_sql=(
                    None
                    if choice is None
                    else render_predicate(choice, variant, dataset_type)
                ),
                actual_fraction=None if choice is None else choice.actual_fraction,
            )
        )
    return out


def expand_queries(
    axes: QueryAxes, values: np.ndarray, variant: TableVariant, rng: np.random.Generator
) -> list[QueryVariant]:
    """One `QueryVariant` per `axes.kinds` entry, in order (`axes.samples` is the repeat count,
    not an axis). A `range` predicate consumes `rng`; an `in` predicate is deterministic.
    `choose_queries` + `render_queries`."""
    return render_queries(axes, choose_queries(axes, values, rng), variant)


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


_DF_UNIT = {"s": "Second", "ms": "Millisecond", "us": "Microsecond", "ns": "Nanosecond"}


def _is_text(t: pa.DataType) -> bool:
    return (
        pa.types.is_string(t)
        or pa.types.is_large_string(t)
        or pa.types.is_string_view(t)
    )


def _is_bytes(t: pa.DataType) -> bool:
    return (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_binary_view(t)
        or pa.types.is_fixed_size_binary(t)
    )


def _value_type(t: pa.DataType) -> pa.DataType:
    """A dictionary column is queried as its value type."""
    return t.value_type if pa.types.is_dictionary(t) else t


def choice_values_arrow(choice: ValueChoice, dtype: pa.DataType) -> pa.Array:
    """`choice.values` as an Arrow array of the dataset's type `dtype`. The numpy values are what
    `query_values` made of the column: time ticks as integers, a nullable integer widened to
    float64 (exact below 2**53), 64-bit integers with nulls as Python ints."""
    if pa.types.is_time(dtype):
        base = pa.int32() if pa.types.is_time32(dtype) else pa.int64()
        return (
            pa.array(np.asarray(choice.values, dtype=np.int64)).cast(base).cast(dtype)
        )
    if pa.types.is_integer(dtype) and choice.values.dtype == object:
        # Python ints: a uint64 above 2**63 overflows pyarrow's int64 inference
        return pa.array(choice.values, type=dtype)
    arr = pa.array(choice.values)
    return arr if arr.type == dtype else arr.cast(dtype)


def _clock(ticks: int, unit: str) -> str:
    """`HH:MM:SS[.fraction]` of `ticks` since midnight in `unit`."""
    places = _UNIT_PRECISION[unit]
    seconds, fraction = divmod(int(ticks), 10**places)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return text + (f".{fraction:0{places}d}" if places else "")


def _datafusion_literal(scalar: pa.Scalar) -> str:
    """One value as a DataFusion SQL literal of its own Arrow type: quoted strings (`'` doubled),
    `X'..'` binaries, `CAST('..' AS DECIMAL(p, s))`, `DATE '..'`, timestamps and times through
    `arrow_cast` at their unit (a zoned timestamp spelled as its UTC instant, `Z`)."""
    t = scalar.type
    if _is_text(t):
        return "'" + scalar.as_py().replace("'", "''") + "'"
    if _is_bytes(t):
        return f"X'{scalar.as_py().hex()}'"
    if pa.types.is_boolean(t):
        return "true" if scalar.as_py() else "false"
    if pa.types.is_integer(t):
        return str(scalar.as_py())
    if pa.types.is_floating(t):
        sql = "FLOAT" if t.bit_width == 32 else "DOUBLE"
        x = float(scalar.as_py())
        text = repr(x)
        return (
            f"CAST({text} AS {sql})" if math.isfinite(x) else f"CAST('{text}' AS {sql})"
        )
    if pa.types.is_decimal(t):
        text = format(scalar.as_py(), "f")
        return f"CAST('{text}' AS DECIMAL({t.precision}, {t.scale}))"
    if pa.types.is_date32(t):
        return f"DATE '{scalar.as_py().isoformat()}'"
    if pa.types.is_timestamp(t):
        ticks = scalar.cast(pa.int64()).as_py()
        iso = str(np.datetime_as_string(np.datetime64(ticks, t.unit), unit=t.unit))
        zone = f'Some("{t.tz}")' if t.tz else "None"
        suffix = "Z" if t.tz else ""
        return f"arrow_cast('{iso}{suffix}', 'Timestamp({_DF_UNIT[t.unit]}, {zone})')"
    if pa.types.is_time(t):
        width = "Time32" if pa.types.is_time32(t) else "Time64"
        ticks = scalar.cast(pa.int32() if width == "Time32" else pa.int64()).as_py()
        return f"arrow_cast('{_clock(ticks, t.unit)}', '{width}({_DF_UNIT[t.unit]})')"
    raise ValueError(f"no DataFusion literal for type {t}")


def render_datafusion(
    predicate_kind: str, values: pa.Array, column_type: pa.DataType
) -> str:
    """The DataFusion predicate over `v` naming `values` (the dataset's type), cast to the column's
    (value) type first, as the worker casts the column."""
    t = _value_type(column_type)
    literals = [_datafusion_literal(s) for s in values.cast(t)]
    if predicate_kind == "range":
        return f"v BETWEEN {literals[0]} AND {literals[1]}"
    return f"v IN ({', '.join(literals)})"


def datafusion_total(column_type: pa.DataType) -> str:
    """The aggregate beside `count(*)`, mirroring ClickHouse's `_aggregate_select` so every value
    is read: bytes for strings and binaries (dictionaries as their value type), days for dates,
    seconds for timestamps and times, the value as DOUBLE otherwise."""
    t = _value_type(column_type)
    if _is_text(t):
        return "sum(octet_length(v))"
    if _is_bytes(t):
        return "sum(octet_length(encode(v, 'hex')) / 2)"
    if pa.types.is_date32(t):
        return "sum(CAST(arrow_cast(v, 'Int32') AS DOUBLE))"
    if pa.types.is_timestamp(t):
        scale = 10 ** _UNIT_PRECISION[t.unit]
        return f"sum(CAST(arrow_cast(v, 'Int64') AS DOUBLE) / {scale}.0)"
    if pa.types.is_time(t):
        width = "Int32" if pa.types.is_time32(t) else "Int64"
        scale = 10 ** _UNIT_PRECISION[t.unit]
        return f"sum(CAST(arrow_cast(v, '{width}') AS DOUBLE) / {scale}.0)"
    return "sum(CAST(v AS DOUBLE))"


def datafusion_table(table_id: str) -> str:
    """The name an in-process DataFusion table is registered under."""
    return f"t_{table_id}"


def datafusion_sql(
    table: str, column_type: pa.DataType, predicate_sql: str | None
) -> str:
    """`SELECT count(*), <datafusion_total> FROM table [WHERE predicate_sql]`."""
    sql = f"SELECT count(*), {datafusion_total(column_type)} FROM {table}"
    return sql if predicate_sql is None else f"{sql} WHERE {predicate_sql}"


def render_polars_text(predicate_kind: str, values: pa.Array) -> str:
    """The stored text of the Polars predicate the worker builds from `values`
    (`polars_engine.polars_predicate`): `col("v").is_in([...])` / `col("v").is_between(lo, hi)`.
    """
    items = [repr(x) for x in values.to_pylist()]
    if predicate_kind == "range":
        if _is_text(values.type):  # Polars reads a bare string bound as a column name
            items = [f"lit({x})" for x in items]
        return f'col("v").is_between({items[0]}, {items[1]})'
    return f'col("v").is_in([{", ".join(items)}])'


@dataclass(frozen=True)
class Reference:
    """What every in-process engine must answer for one query: the row count and the total (bytes
    as an int; days, seconds or values as a float; None when the column holds the dataset's values
    as engine-formatted text). `magnitude`, the total of absolute values, scales the float
    tolerance."""

    count: int
    total: int | float | None
    magnitude: float


def _widen_decimal(array: pa.Array) -> pa.Array:
    """decimal32 / decimal64 as decimal128 (pyarrow has no `is_in` / comparison kernels for the
    narrow widths); anything else unchanged."""
    t = array.type
    if pa.types.is_decimal32(t) or pa.types.is_decimal64(t):
        return array.cast(pa.decimal128(t.precision, t.scale))
    return array


def _as_double(rows: pa.Array) -> pa.Array:
    """`rows` as float64, rounding integers beyond 2**53 (hash / id columns, nanosecond
    timestamps) as an engine's CAST(v AS DOUBLE) does instead of refusing them."""
    return pc.cast(rows, pa.float64(), safe=False)


def _total(rows: pa.Array) -> tuple[int | float, float]:
    t = rows.type
    if pa.types.is_string_view(t):
        rows, t = rows.cast(pa.large_string()), pa.large_string()
    if pa.types.is_binary_view(t):
        rows, t = rows.cast(pa.large_binary()), pa.large_binary()
    if _is_text(t) or _is_bytes(t):
        n = int(pc.sum(pc.binary_length(rows)).as_py() or 0)
        return n, float(n)
    if pa.types.is_date32(t):
        x = _as_double(rows.cast(pa.int32()))
    elif pa.types.is_timestamp(t):
        x = pc.divide(
            _as_double(rows.cast(pa.int64())),
            float(10 ** _UNIT_PRECISION[t.unit]),
        )
    elif pa.types.is_time(t):
        base = pa.int32() if pa.types.is_time32(t) else pa.int64()
        x = pc.divide(_as_double(rows.cast(base)), float(10 ** _UNIT_PRECISION[t.unit]))
    else:
        x = _as_double(rows)
    return float(pc.sum(x).as_py() or 0.0), float(pc.sum(pc.abs(x)).as_py() or 0.0)


def reference_result(
    array: pa.Array,
    predicate_kind: str | None,
    values: pa.Array | None,
    family: str | None = None,
) -> Reference:
    """The pyarrow answer to one query over the dataset column `array`: every row for a full scan
    (`predicate_kind` None), the rows whose value is in `values` (`in`) or between `values[0]` and
    `values[1]` (`range`). `family` (`inproc_types.column_family` of the queried column; None: the
    dataset's own) says what the total means: a date dataset in a timestamp column totals seconds,
    a timestamp dataset in a date column days, and a text column of a non-text dataset has no
    reference total (each engine formats the text itself)."""
    array = _widen_decimal(array)
    if values is not None:
        values = _widen_decimal(values)
    if predicate_kind is None:
        rows = array
    elif predicate_kind == "in":
        rows = array.filter(pc.is_in(array, value_set=values))
    else:
        rows = array.filter(
            pc.and_(pc.greater_equal(array, values[0]), pc.less_equal(array, values[1]))
        )
    source = arrow_family(array.type)
    target = family or source
    if target in ("string", "categorical", "binary") and source not in (
        "string",
        "binary",
    ):
        return Reference(count=len(rows), total=None, magnitude=0.0)
    if target == "timestamp" and source == "date":
        rows = rows.cast(pa.timestamp("s"))
    elif target == "date" and source == "timestamp":
        rows = pc.cast(rows, pa.date32(), safe=False)
    total, magnitude = _total(rows)
    return Reference(count=len(rows), total=total, magnitude=magnitude)


def results_agree(reference: Reference, count, total) -> bool:
    """`count` exactly; an int total exactly (an integral number equal to it: 7.9 does not match
    7); a float total within 1e-9 of the magnitude summed (`Reference.magnitude`, which absorbs
    the rounding of integers beyond 2**53 cast to double); no total compared when the reference
    has none. A NULL total (an empty SQL sum) reads as 0.

    Non-finite totals: a finite reference needs a finite result. A non-finite reference (NaN,
    inf or -inf) accepts any non-finite result, because once the data holds a NaN, an infinity
    or values whose sum overflows, summation order alone picks the outcome: inf + -inf is NaN
    in one order and an infinity in another, and 1e308, 1e308, -1e308, -1e308 overflows to inf
    or to -inf. A NaN in the data makes every order NaN, so a stricter rule would need to know
    which case holds; the count still guards the rows selected."""
    if count is None or int(count) != reference.count:
        return False
    if reference.total is None:
        return True
    got = 0 if total is None else total
    if isinstance(reference.total, int):
        if isinstance(got, float):
            return got.is_integer() and int(got) == reference.total
        return isinstance(got, (int, np.integer)) and int(got) == reference.total
    got = float(got)
    if not math.isfinite(reference.total):
        return not math.isfinite(got)
    if not math.isfinite(got):
        return False
    scale = max(reference.magnitude, abs(reference.total))
    return abs(got - reference.total) <= 1e-9 * scale
