"""The in-process engines' suite vocabulary: the `type`, `storage` and `compression` values of a
`polars:` or `datafusion:` block, and which dataset types can feed which column types.

Pure text and pyarrow: nothing here imports Polars or DataFusion, so parsing a suite never starts
an engine (the worker processes do, with their thread count already set).

- Polars type names are what the recommenders emit in `polars_type` (Polars' `str(dtype)`, e.g.
  `Categorical(Categories(name="v", namespace="", physical=pl.UInt8))`,
  `Decimal(precision=5, scale=2)`, `Datetime(time_unit='us', time_zone=None)`) or the short forms
  `Categorical(UInt8)`, `Decimal(5, 2)`, `Datetime(us)`, `Datetime(us, UTC)`. Each parses to a
  `PolarsType` whose `label` (the short form) is what the store records as the table's type. Bare
  `Categorical` is `Categorical(UInt32)`; every width is cast to named categories in the worker,
  so no variant touches Polars' process-global mapping (ToDo 11). `Enum` takes its categories from
  the data (first-seen order), `Enum(categories=[...])` names them; both are labelled `Enum`.
- DataFusion type names are pyarrow type strings exactly as pyarrow prints them (`str(pa_type)`),
  dictionaries of string or binary values included. Run-end encoding is refused: DataFusion 54 and
  55 reject MIN / MAX over it (CLAUDE.md, Rules › Targets; ToDo 6).
- Compression: `none`, `LZ4_FRAME`, `ZSTD` (level 1) or `ZSTD(N)`, N 1-22, stored as `none`,
  `LZ4_FRAME` or `ZSTD(N)`. Storage: `memory` or `ipc`. IPC batching (`ipc_batch_rows`, storage
  `ipc` only): rows per record batch, stored as text, by default `122880` (Polars' own chunking:
  realistic writers produce batched files, and one whole-file batch was 3-4x slower to query in
  both engines), or `whole` (the file is one record batch).
- Families (`numeric`, `boolean`, `string`, `categorical`, `binary`, `date`, `timestamp`, `time`)
  say which dataset types can feed which column types (`FEEDS`, checked at parse time) and what a
  query's total means (`queries.reference_result`).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pyarrow as pa

STORAGES = ("memory", "ipc")
DEFAULT_IPC_BATCH_ROWS = 122_880  # rows per record batch; stored as "122880"

_ZSTD_RE = re.compile(r"ZSTD\(([0-9]+)\)")
_COMPRESSION_HELP = "none, LZ4_FRAME, ZSTD or ZSTD(N) for N from 1 to 22"


def normalise_ipc_compression(value) -> str:
    """The stored spelling of an IPC compression axis value (`ZSTD` becomes `ZSTD(1)`);
    ValueError for anything else."""
    if value in ("none", "LZ4_FRAME"):
        return value
    if value == "ZSTD":
        return "ZSTD(1)"
    m = _ZSTD_RE.fullmatch(value) if isinstance(value, str) else None
    if m and 1 <= int(m.group(1)) <= 22:
        return f"ZSTD({int(m.group(1))})"
    raise ValueError(f"IPC compression {value!r} is not one of {_COMPRESSION_HELP}")


def normalise_ipc_batch_rows(value) -> str:
    """An `ipc_batch_rows` axis value in its stored spelling: a positive integer as text (rows
    per record batch; `DEFAULT_IPC_BATCH_ROWS` by default) or `whole` (the file is one record
    batch, the layout the recommenders' size predictions model); ValueError otherwise.
    """
    if value == "whole":
        return "whole"
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    raise ValueError(
        f"ipc_batch_rows value {value!r} is not 'whole' or a positive integer (rows per batch)"
    )


def ipc_batch_rows(value: str) -> int | None:
    """The writer's `max_chunksize` for a stored `ipc_batch_rows` value: None for `whole`."""
    return None if value == "whole" else int(value)


def ipc_codec(compression: str) -> pa.Codec | None:
    """The pyarrow IPC codec of a stored compression: None, LZ4 frame, or ZSTD at its level."""
    if compression == "none":
        return None
    if compression == "LZ4_FRAME":
        return pa.Codec("lz4")
    return pa.Codec("zstd", int(_ZSTD_RE.fullmatch(compression).group(1)))


@dataclass(frozen=True)
class PolarsType:
    """A parsed Polars type name. `kind` is the scalar name (`Int16`, `String`, ...) or one of
    `Decimal`, `Datetime`, `Categorical`, `Enum`; the parameters belong to those four.
    """

    label: str
    kind: str
    precision: int | None = None
    scale: int | None = None
    unit: str | None = None
    tz: str | None = None
    width: str | None = None
    categories: tuple[str, ...] | None = None


_POLARS_SCALARS = frozenset(
    {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Boolean",
        "Date",
        "Time",
        "String",
        "Binary",
    }
)
_WIDTH = r"(UInt8|UInt16|UInt32)"
_CAT_SHORT = re.compile(rf"Categorical(?:\({_WIDTH}\))?")
_CAT_LONG = re.compile(
    rf'Categorical\(Categories\(name="[^"]*", namespace="[^"]*", physical=pl\.{_WIDTH}\)\)'
)
_CAT_NAMED = re.compile(r'Categorical\(Categories\("[^"]*"\)\)')
_DECIMAL = re.compile(r"Decimal\((?:precision=)?([0-9]+), (?:scale=)?([0-9]+)\)")
_DATETIME_SHORT = re.compile(r"Datetime\((ms|us|ns)(?:, ([^)]+))?\)")
_DATETIME_LONG = re.compile(
    r"Datetime\(time_unit='(ms|us|ns)', time_zone=(None|'[^']+')\)"
)
_ENUM = re.compile(r"Enum(?:\(categories=(\[.*\])\))?", re.DOTALL)
_POLARS_HELP = (
    "Int8-Int64, UInt8-UInt64, Float32, Float64, Boolean, Date, Time, String, Binary, "
    "Decimal(p, s), Datetime(ms|us|ns[, zone]), Categorical[(UInt8|UInt16|UInt32)], Enum, "
    "or Polars' own str(dtype) of one of these"
)


def _datetime(unit: str, tz: str | None) -> PolarsType:
    label = f"Datetime({unit}, {tz})" if tz else f"Datetime({unit})"
    return PolarsType(label, "Datetime", unit=unit, tz=tz)


def _enum(name: str, listed: str | None) -> PolarsType:
    if listed is None:
        return PolarsType("Enum", "Enum")
    try:
        cats = ast.literal_eval(listed)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"{name!r} is not a polars type the bench reads: {e}") from e
    if (
        not cats
        or not all(isinstance(c, str) for c in cats)
        or len(set(cats)) != len(cats)
    ):
        raise ValueError(
            f"{name!r} is not a polars type the bench reads: Enum categories must be distinct "
            f"strings, at least one"
        )
    return PolarsType("Enum", "Enum", categories=tuple(cats))


def parse_polars_type(name) -> PolarsType:
    """A Polars type name (short form or Polars' `str(dtype)`) parsed; ValueError naming the
    vocabulary otherwise."""
    text = str(name).strip()
    if text in _POLARS_SCALARS:
        return PolarsType(text, text)
    m = _CAT_SHORT.fullmatch(text) or _CAT_LONG.fullmatch(text)
    if m:
        width = m.group(1) or "UInt32"
        return PolarsType(f"Categorical({width})", "Categorical", width=width)
    if _CAT_NAMED.fullmatch(text):
        return PolarsType("Categorical(UInt32)", "Categorical", width="UInt32")
    m = _DECIMAL.fullmatch(text)
    if m:
        p, s = int(m.group(1)), int(m.group(2))
        if not (1 <= p <= 38 and 0 <= s <= p):
            raise ValueError(
                f"{name!r} is not a polars type the bench reads: Decimal precision must be "
                f"1-38 and scale 0 to the precision"
            )
        return PolarsType(f"Decimal({p}, {s})", "Decimal", precision=p, scale=s)
    m = _DATETIME_SHORT.fullmatch(text)
    if m:
        return _datetime(m.group(1), m.group(2))
    m = _DATETIME_LONG.fullmatch(text)
    if m:
        return _datetime(m.group(1), None if m.group(2) == "None" else m.group(2)[1:-1])
    m = _ENUM.fullmatch(text)
    if m:
        return _enum(text, m.group(1))
    raise ValueError(f"{name!r} is not a polars type the bench reads ({_POLARS_HELP})")


_DICT_RE = re.compile(r"dictionary<values=(\w+), indices=(\w+), ordered=([01])>")
_DEC_RE = re.compile(r"decimal(32|64|128)\(([0-9]+), ([0-9]+)\)")
_TS_RE = re.compile(r"timestamp\[(s|ms|us|ns)(?:, tz=([^\]]+))?\]")
_TIME_RE = re.compile(r"(time32|time64)\[(s|ms|us|ns)\]")
_FSB_RE = re.compile(r"fixed_size_binary\[([0-9]+)\]")
_ARROW_SCALARS = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "double",
        "string",
        "large_string",
        "string_view",
        "binary",
        "large_binary",
        "binary_view",
        "date32[day]",
    }
)
_DICT_VALUES = frozenset({"string", "large_string", "binary", "large_binary"})
_DICT_INDICES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_DATAFUSION_HELP = (
    "pyarrow type strings as pyarrow prints them: bool, int8-int64, uint8-uint64, float, "
    "double, string, large_string, string_view, binary, large_binary, binary_view, "
    "date32[day], decimal32|64|128(p, s), timestamp[unit(, tz=zone)], time32[s|ms], "
    "time64[us|ns], fixed_size_binary[n], dictionary<values=string|large_string|binary|"
    "large_binary, indices=<integer>, ordered=0|1>"
)


def _datafusion(text: str) -> pa.DataType:
    m = _DICT_RE.fullmatch(text)
    if m:
        values, indices, ordered = m.groups()
        if values not in _DICT_VALUES or indices not in _DICT_INDICES:
            raise ValueError(
                "a dictionary holds string or binary values on integer indices"
            )
        return pa.dictionary(
            pa.type_for_alias(indices),
            pa.type_for_alias(values),
            ordered=ordered == "1",
        )
    m = _DEC_RE.fullmatch(text)
    if m:
        return getattr(pa, f"decimal{m.group(1)}")(int(m.group(2)), int(m.group(3)))
    m = _TS_RE.fullmatch(text)
    if m:
        return pa.timestamp(m.group(1), m.group(2))
    m = _TIME_RE.fullmatch(text)
    if m:
        return getattr(pa, m.group(1))(m.group(2))
    m = _FSB_RE.fullmatch(text)
    if m:
        return pa.binary(int(m.group(1)))
    if text in _ARROW_SCALARS:
        return pa.type_for_alias(text)
    try:
        alias = pa.type_for_alias(text)
    except (ValueError, KeyError):
        alias = None
    if alias is not None and str(alias) != text:
        raise ValueError(f"spell it as pyarrow prints it, {str(alias)!r}")
    raise ValueError("not in the vocabulary")


def parse_datafusion_type(name) -> pa.DataType:
    """A pyarrow type string parsed; ValueError naming the vocabulary otherwise (and the reason
    for run-end encoding)."""
    text = str(name).strip()
    if text.startswith("run_end_encoded"):
        raise ValueError(
            f"{name!r} is not a datafusion type the bench measures: run-end encoding is not "
            f"recommended, DataFusion 54 and 55 reject MIN / MAX over it (ToDo 6)"
        )
    try:
        t = _datafusion(text)
    except ValueError as e:
        raise ValueError(
            f"{name!r} is not a datafusion type the bench measures ({e}; {_DATAFUSION_HELP})"
        ) from e
    if str(t) != text:
        raise ValueError(
            f"{name!r} is not a datafusion type the bench measures as written: spell it as "
            f"pyarrow prints it, {str(t)!r}"
        )
    return t


def type_label(engine: str, name) -> str:
    """The label the store records for a `type` axis value of `engine`."""
    if engine == "polars":
        return parse_polars_type(name).label
    return str(parse_datafusion_type(name))


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


def arrow_family(t: pa.DataType) -> str | None:
    """The family of an Arrow type; None for one no column of the bench holds (lists, ...)."""
    if pa.types.is_dictionary(t):
        values = arrow_family(t.value_type)
        return "categorical" if values == "string" else values
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "numeric"
    if pa.types.is_boolean(t):
        return "boolean"
    if _is_text(t):
        return "string"
    if _is_bytes(t):
        return "binary"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_time(t):
        return "time"
    return None


_POLARS_FAMILY = {
    "Boolean": "boolean",
    "Date": "date",
    "Time": "time",
    "String": "string",
    "Binary": "binary",
    "Datetime": "timestamp",
    "Categorical": "categorical",
    "Enum": "categorical",
}


def polars_family(t: PolarsType) -> str:
    return _POLARS_FAMILY.get(t.kind, "numeric")


def column_family(engine: str, name) -> str | None:
    """The family of the column a `type` axis value of `engine` makes."""
    if engine == "polars":
        return polars_family(parse_polars_type(name))
    return arrow_family(parse_datafusion_type(name))


# Which column families a dataset of each family can feed. Anything else no cast could make;
# a cast these allow that still fails or loses a value is a skipped cell at run time. A boolean
# dataset feeds numbers because `to_chbench` writes a recommended Boolean column of 0 / 1 integers
# as a boolean dataset beside its original integer type.
FEEDS = {
    "numeric": frozenset({"numeric", "string"}),
    "boolean": frozenset({"boolean", "numeric", "string"}),
    "string": frozenset({"string", "categorical", "binary"}),
    "binary": frozenset({"binary", "string"}),
    "date": frozenset({"date", "timestamp", "string"}),
    "timestamp": frozenset({"timestamp", "date", "string"}),
    "time": frozenset({"time", "string"}),
}


def feed_error(dataset_type: pa.DataType, engine: str, name) -> str | None:
    """None when a dataset of `dataset_type` can feed the column `name` of `engine`; else why
    not, naming both sides."""
    source = arrow_family(dataset_type)
    if source is None:
        return (
            f"dataset.arrow_type {dataset_type} cannot feed any {engine} column; the harness "
            f"benchmarks one scalar column per table"
        )
    target = column_family(engine, name)
    if target in FEEDS[source]:
        return None
    return (
        f"{engine}.type value {name!r} is a {target} column, which a {source} dataset "
        f"(dataset.arrow_type {dataset_type}) cannot feed; a {source} dataset feeds "
        f"{sorted(FEEDS[source])}"
    )
