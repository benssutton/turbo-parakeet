"""Expand a suite's ClickHouse axes into the cartesian product of table
variants, and validate each cell against the measured applicability rules.

The rules and skip reasons are the matrices measured against a live
ClickHouse server, recorded in FINDINGS §5 and FINDINGS §9.
The rule is skip and record, never coerce: an invalid cell is never
silently adjusted into something ClickHouse would accept, it is marked
`skipped` with the server's own rejection message so the report can show why.

Applicability keys off the resolved *base* type carried on the `type` axis
(e.g. `Int32`, `String`, `Decimal32`): `nullable` and the `LowCardinality`
codec are separate axes that wrap the base type in the rendered DDL, but they
do not change which codecs or skip indexes the base type accepts, so no
separate "strip the wrapper" step is needed — the checks simply never see the
wrapper in the first place.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from synthdata import DatasetSpec

if TYPE_CHECKING:
    # Deferred: chbench.harness.suite imports the vocabulary constants below,
    # so importing Suite here at runtime would be circular. The annotations
    # below are strings (`from __future__ import annotations`), so this
    # import is only ever needed by a type checker.
    from chbench.harness.suite import Suite

# The type axis vocabulary: the base types measured against a live server
# (`spikes/spike_ch_roundtrip.py` for the first eight,
# `spikes/spike_ch_vocabulary.py` for the rest). Anything else is
# unmeasured, not merely "probably fine by analogy" -- the project's rule is to
# record only what it measured, so an out-of-vocabulary type is a hard error,
# not a guess. `suite.py` imports this to reject it at parse time;
# `_build_variant` below keeps a backstop raise for callers that build a
# `ClickHouseAxes` directly. The axis value is the base name; the parameters
# (`Decimal*` scale, `DateTime64` precision and zone, `FixedString` width,
# `Enum*` values, `Time64` precision) come from the dataset (`resolve_type`).
KNOWN_TYPES = frozenset(
    {
        "Float64",
        "Float32",
        "Decimal32",
        "Decimal64",
        "Decimal128",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Bool",
        "String",
        "FixedString",
        "Enum8",
        "Enum16",
        "Date",
        "Date32",
        "DateTime",
        "DateTime64",
        "Time64",
    }
)

# The codec axis vocabulary (the codecs measured in FINDINGS §5 plus `none` and
# the `LowCardinality` type wrapper; `GCD` from FINDINGS §9). `suite.py` imports
# this too.
KNOWN_CODECS = frozenset(
    {
        "none",
        "Delta",
        "DoubleDelta",
        "Gorilla",
        "T64",
        "LowCardinality",
        "FPC",
        "GCD",
    }
)

_SET_INDEX_RE = re.compile(r"^set\((\d+)\)$")
_ZSTD_LEVEL_RE = re.compile(r"^ZSTD\((\d+)\)$")


SKIP_GRANULARITIES = (1, 2, 4, 8)
BLOOM_FPRS = ("0.001", "0.01", "0.025", "0.05")
_BLOOM_FPR_RE = re.compile(r"^bloom_filter\((\d+\.\d+)\)$")


def split_skip_index(value: str) -> tuple[str, int]:
    """`<index>[:<G>]` -> (index, G); no suffix means GRANULARITY 1."""
    index, sep, g = value.partition(":")
    if not sep:
        return value, 1
    return index, int(g) if g.isascii() and g.isdigit() else -1


def is_known_skip_index(value: str) -> bool:
    """`none`, `minmax`, `bloom_filter`, `bloom_filter(F)` for F in the measured
    false-positive rates, or `set(N)` for a positive integer N; any but `none`
    optionally followed by `:G`, G in {1, 2, 4, 8}."""
    index, g = split_skip_index(value)
    if index == "none":
        return value == "none"
    if g not in SKIP_GRANULARITIES:
        return False
    if index in ("minmax", "bloom_filter"):
        return True
    m = _BLOOM_FPR_RE.match(index)
    if m is not None:
        return m.group(1) in BLOOM_FPRS
    m = _SET_INDEX_RE.match(index)
    return m is not None and int(m.group(1)) > 0


def is_known_compression(value: str) -> bool:
    """`LZ4`, `ZSTD`, or `ZSTD(N)` for N from 1 to 22."""
    if value in ("LZ4", "ZSTD"):
        return True
    m = _ZSTD_LEVEL_RE.match(value)
    return m is not None and 1 <= int(m.group(1)) <= 22


# Storage width in bytes: the `N` of the Gorilla rejection message
# `Gorilla(N), <compression>` and the size test of Delta / DoubleDelta /
# Gorilla (the server accepts widths 1, 2, 4 and 8 only). Measured values from
# FINDINGS §5 and FINDINGS §9; `FixedString(n)` is n bytes (`_width`).
_STORAGE_WIDTH = {
    "Int8": 1,
    "UInt8": 1,
    "Bool": 1,
    "Enum8": 1,
    "Int16": 2,
    "UInt16": 2,
    "Enum16": 2,
    "Date": 2,
    "Int32": 4,
    "UInt32": 4,
    "Float32": 4,
    "Decimal32": 4,
    "Date32": 4,
    "DateTime": 4,
    "Int64": 8,
    "UInt64": 8,
    "Float64": 8,
    "Decimal64": 8,
    "DateTime64": 8,
    "Time64": 8,
    "Decimal128": 16,
}
_CODEC_WIDTHS = (1, 2, 4, 8)
_FLOATS = frozenset({"Float32", "Float64"})

# Precision ClickHouse reports a Decimal(P, S) type as, keyed by the DDL
# width name.
_DECIMAL_PRECISION = {
    "Decimal32": 9,
    "Decimal64": 18,
    "Decimal128": 38,
}

# Inclusive bounds of the integer axis types.
_INTEGER_BOUNDS = {
    "Int8": (-(2**7), 2**7 - 1),
    "Int16": (-(2**15), 2**15 - 1),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "UInt8": (0, 2**8 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}

# Types the GCD codec accepts (FINDINGS §9): Int*, UInt*, Decimal*, Date*,
# DateTime* -- and Bool.
_GCD_TYPES = (
    frozenset(_INTEGER_BOUNDS)
    | frozenset(_DECIMAL_PRECISION)
    | {"Bool", "Date", "Date32", "DateTime", "DateTime64"}
)

# Types T64 accepts (FINDINGS §5, FINDINGS §9).
_T64_TYPES = frozenset(_INTEGER_BOUNDS) | {
    "Decimal32",
    "Decimal64",
    "Bool",
    "Date",
    "Date32",
    "DateTime",
    "DateTime64",
    "Enum8",
    "Enum16",
    "Time64",
}

# Types the `bloom_filter` index rejects (FINDINGS §5, FINDINGS §9).
_NO_BLOOM = frozenset(_DECIMAL_PRECISION) | {"Date32", "Time64"}

# Types the LowCardinality wrapper rejects with code 43 rather than 455.
_LC_CODE_43 = frozenset(_DECIMAL_PRECISION) | {
    "DateTime64",
    "Enum8",
    "Enum16",
    "Time64",
}

_UNIT_PRECISION = {"s": 0, "ms": 3, "us": 6, "ns": 9}
_ENUM_LIMIT = {"Enum8": 127, "Enum16": 32767}
_DECIMAL_RESOLVED_RE = re.compile(r"^Decimal\d+\((\d+)\)$")
_FIXEDSTRING_RE = re.compile(r"^FixedString\((\d+)\)$")


@dataclass(frozen=True)
class TableVariant:
    ch_type: str
    nullable: bool
    primary_key: bool
    projection: bool
    codec: str
    skip_index: str
    compression: str
    status: str
    skip_reason: str | None
    warnings: tuple[str, ...]


class TypeResolutionError(ValueError):
    """A `type` axis value the dataset cannot parameterise (an `Enum8` over
    more than 127 values, an all-null `Enum`, a `Time64` over a non-time
    dataset). `expand_grid` records it as a skipped cell carrying the message."""


def _quote_name(name: str) -> str:
    return "'" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"


def resolve_type(
    axis_type: str, spec: DatasetSpec, array: pa.Array | None = None
) -> str:
    """Map a `type` axis value to its DDL spelling, taking the parameters from
    the dataset.

    - `Decimal32` / `Decimal64` / `Decimal128` take `decimal_places` as scale.
    - `DateTime64(p[, 'tz'])`: p from a timestamp dataset's unit (s / ms / us /
      ns -> 0 / 3 / 6 / 9), the zone from its `tz`; any other dataset (the
      generated integer datasets) gets `DateTime64(3)`, as before.
    - `DateTime[('tz')]` likewise (the zone only).
    - `Time64(p)` from a time32 / time64 dataset's unit (the same map).
    - `FixedString(n)`: n of a `fixed_size_binary(n)` dataset, else the longest
      value of a string dataset in bytes (needs `array`).
    - `Enum8` / `Enum16`: the dataset's distinct non-null strings in sorted
      order, codes 1..n (needs `array`); `Enum8` holds at most 127 values with
      those codes, `Enum16` 32767.

    Everything else is passed through verbatim. `TypeResolutionError` when the
    dataset cannot parameterise the type; `ValueError` when `array` is needed
    and missing.
    """
    if axis_type in _DECIMAL_PRECISION:
        return f"{axis_type}({spec.decimal_places})"
    dtype = spec.arrow_dtype
    if axis_type == "DateTime64":
        if not pa.types.is_timestamp(dtype):
            return "DateTime64(3)"
        p = _UNIT_PRECISION[dtype.unit]
        return f"DateTime64({p}, '{dtype.tz}')" if dtype.tz else f"DateTime64({p})"
    if axis_type == "DateTime":
        zone = dtype.tz if pa.types.is_timestamp(dtype) else None
        return f"DateTime('{zone}')" if zone else "DateTime"
    if axis_type == "Time64":
        if not (pa.types.is_time32(dtype) or pa.types.is_time64(dtype)):
            raise TypeResolutionError(
                f"Time64 needs a time32 or time64 dataset, but dataset.arrow_type is "
                f"{spec.arrow_type!r}"
            )
        return f"Time64({_UNIT_PRECISION[dtype.unit]})"
    if axis_type == "FixedString":
        return _resolve_fixed_string(dtype, array)
    if axis_type in _ENUM_LIMIT:
        return _resolve_enum(axis_type, array)
    return axis_type


def _resolve_fixed_string(dtype: pa.DataType, array: pa.Array | None) -> str:
    if pa.types.is_fixed_size_binary(dtype):
        return f"FixedString({dtype.byte_width})"
    if array is None:
        raise ValueError(
            "resolving FixedString over a string dataset needs the dataset "
            "array (the longest value's byte length)"
        )
    longest = pc.max(pc.binary_length(array)).as_py()
    if not longest:
        raise TypeResolutionError(
            "FixedString(n) needs n >= 1, but the dataset has no non-empty value"
        )
    return f"FixedString({longest})"


def _resolve_enum(axis_type: str, array: pa.Array | None) -> str:
    if array is None:
        raise ValueError(
            f"resolving {axis_type} needs the dataset array (its distinct " "values)"
        )
    names = sorted(pc.unique(array.drop_null()).to_pylist())
    limit = _ENUM_LIMIT[axis_type]
    if not names:
        raise TypeResolutionError(
            f"{axis_type} needs values, but the dataset has no " "non-null value"
        )
    if len(names) > limit:
        raise TypeResolutionError(
            f"{axis_type} holds at most {limit} values (codes 1..n), but the dataset has "
            f"{len(names)} distinct values"
        )
    return (
        f"{axis_type}("
        + ", ".join(f"{_quote_name(n)} = {i}" for i, n in enumerate(names, 1))
        + ")"
    )


def _label(axis_type: str, resolved: str) -> str:
    """The type as the server spells it in its own messages: decimals as
    `Decimal(P, S)`, everything else as resolved."""
    if axis_type in _DECIMAL_PRECISION:
        m = _DECIMAL_RESOLVED_RE.match(resolved)
        return f"Decimal({_DECIMAL_PRECISION[axis_type]}, {m.group(1)})"
    return resolved


def _width(axis_type: str, resolved: str) -> int | None:
    """Storage width in bytes; None for String."""
    m = _FIXEDSTRING_RE.match(resolved)
    if m:
        return int(m.group(1))
    return _STORAGE_WIDTH.get(axis_type)


def _low_cardinality_skip_reason(
    axis_type: str, resolved: str, nullable: bool
) -> str | None:
    """None for String and FixedString: ClickHouse 26.8 accepts
    `LowCardinality` on those (and their `Nullable` forms) and rejects it on
    every other type, with one of two messages measured by executing the DDL
    (FINDINGS §5, FINDINGS §9):

    - numbers, Bool, Date, Date32, DateTime: code 455
      `SUSPICIOUS_TYPE_FOR_LOW_CARDINALITY`, naming the full column type
      including any `Nullable(...)` wrapper;
    - Decimal, DateTime64, Enum, Time64: code 43 `ILLEGAL_TYPE_OF_ARGUMENT`,
      naming the resolved type, again inside `Nullable(...)` when nullable.
    """
    if axis_type in ("String", "FixedString"):
        return None
    inner = _label(axis_type, resolved)
    if nullable:
        inner = f"Nullable({inner})"
    if axis_type in _LC_CODE_43:
        return (
            "DataTypeLowCardinality is supported only for numbers, strings, "
            f"Date or DateTime, but got {inner}."
        )
    return (
        f"Creating columns of type LowCardinality({inner}) is prohibited by "
        "default due to expected negative impact on performance. It can be "
        "enabled with the `allow_suspicious_low_cardinality_types` setting."
    )


def _size_reason(codec: str, label: str) -> str:
    return (
        f"Codec {codec} is only applicable for data types of size 1, 2, 4, 8 bytes. "
        f"Given type {label}."
    )


def _codec_skip_reason(
    axis_type: str, codec: str, resolved: str, compression: str, nullable: bool = False
) -> str | None:
    """None if `codec` is accepted on `axis_type`; else ClickHouse's own
    rejection message, verbatim, from the codec matrices measured in
    FINDINGS §5 and FINDINGS §9. `resolved` is the DDL type (`resolve_type`);
    `nullable` only matters to the LowCardinality rule, whose message names the
    wrapped type."""
    if codec == "none":
        return None

    if codec == "LowCardinality":
        # A type wrapper, not a CODEC clause.
        return _low_cardinality_skip_reason(axis_type, resolved, nullable)

    label = _label(axis_type, resolved)
    width = _width(axis_type, resolved)

    if codec == "GCD":
        if axis_type in _GCD_TYPES:
            return None
        return (
            f"Codec GCD cannot be applied to column {label} because it can only be "
            "used with Int*, UInt*, Decimal*, Date* or DateTime* types."
        )

    if codec == "Delta":
        if axis_type == "String":
            return (
                "Codec Delta is not applicable for String because the data type is "
                "not of fixed size."
            )
        if width in _CODEC_WIDTHS:
            return None
        return _size_reason("Delta", label)

    if codec == "DoubleDelta":
        if axis_type in ("String", "FixedString"):
            return (
                f"Codec DoubleDelta is not applicable for {label} "
                f"because the data type is not numeric."
            )
        if width in _CODEC_WIDTHS:
            return None
        return _size_reason("DoubleDelta", label)

    if codec == "Gorilla":
        if axis_type in _FLOATS:
            return None
        if axis_type == "String":
            # Non-fixed-size: Gorilla gives Delta's clause under its own name.
            return (
                "Codec Gorilla is not applicable for String because the data type "
                "is not of fixed size."
            )
        if width not in _CODEC_WIDTHS:
            return _size_reason("Gorilla", label)
        # The `compression` axis also allows ZSTD, and the DDL always applies
        # CODEC(Gorilla, <compression>), so the server names the actual pair.
        return (
            f"The combination of compression codecs Gorilla({width}), {compression} "
            "is meaningless, because it does not make sense to apply a "
            "floating-point time series codec to non-floating-point columns "
            "(Note: you can enable setting 'allow_suspicious_codecs' to skip "
            "this check)."
        )

    if codec == "T64":
        if axis_type in _T64_TYPES:
            return None
        return f"T64 codec is not supported for specified type {label}."

    if codec == "FPC":
        if axis_type in _FLOATS:
            return None
        return (
            f"Codec 'FPC' is not applicable for {label} because the data "
            f"type is not Float*."
        )

    raise ValueError(f"unknown codec {codec!r}")


def _skip_index_skip_reason(
    axis_type: str, skip_index: str, resolved: str, nullable: bool = False
) -> str | None:
    """None if `skip_index` is accepted on `axis_type`; else ClickHouse's own
    rejection message from the measured skip-index matrices. `minmax` and
    `set(N)` were accepted on every type tested; `bloom_filter` is rejected on
    the Decimals, Date32 and Time64, naming the column type inside
    `Nullable(...)` when nullable."""
    if (
        split_skip_index(skip_index)[0].startswith("bloom_filter")
        and axis_type in _NO_BLOOM
    ):
        inner = _label(axis_type, resolved)
        if nullable:
            inner = f"Nullable({inner})"
        return (
            f"Unexpected type {inner} of bloom filter index: When validating "
            f"secondary index `ix`."
        )
    return None


def _range_skip_reason(
    axis_type: str, resolved: str, spec: DatasetSpec, value_min, value_max
) -> str | None:
    """None if the achieved range fits `axis_type`'s representable range (at
    the dataset's scale for decimals); else a reason naming the type, its
    bound and the offending value (the value range must fit the target type
    given scale, checked against the achieved/rounded data, not the
    requested `value_range`).

    `Decimal32` has 9 total digits, `Decimal64` 18, `Decimal128` 38; at scale
    S the representable magnitude is `10**(precision - S) - 10**(-S)` (e.g.
    Decimal32(8) -> 9.99999999, matching the bound measured in FINDINGS §5).
    The
    integer types are checked against their two's-complement bounds. Other
    types have no width-dependent range check here.
    """
    if axis_type in _INTEGER_BOUNDS:
        lo, hi = _INTEGER_BOUNDS[axis_type]
        if value_min is not None and value_min < lo:
            return (
                f"{axis_type} can represent {lo} to {hi}, but the achieved minimum "
                f"is {value_min}."
            )
        if value_max is not None and value_max > hi:
            return (
                f"{axis_type} can represent {lo} to {hi}, but the achieved maximum "
                f"is {value_max}."
            )
        return None
    if axis_type not in _DECIMAL_PRECISION:
        return None
    precision = _DECIMAL_PRECISION[axis_type]
    scale = spec.decimal_places
    bound = 10 ** (precision - scale) - 10 ** (-scale)
    for label, value in (("minimum", value_min), ("maximum", value_max)):
        if abs(value) > bound:
            return (
                f"{resolved} can represent magnitude up to {bound} at scale "
                f"{scale}, but the achieved {label} is {value}, which overflows."
            )
    return None


def _build_variant(
    dataset: DatasetSpec,
    axis_type: str,
    nullable: bool,
    primary_key: bool,
    projection: bool,
    codec: str,
    skip_index: str,
    compression: str,
    null_count: int,
    value_min,
    value_max,
    resolved: str | TypeResolutionError,
) -> TableVariant:
    if axis_type not in KNOWN_TYPES:
        # Backstop for callers that build a `ClickHouseAxes` directly instead
        # of going through `parse_suite` (which rejects this at parse time).
        # An unmeasured type must never silently fall through the applicability
        # checks below and be bucketed with String's (or anyone else's) skip
        # reason: that would be a fabricated ClickHouse message the server
        # never produced.
        raise ValueError(
            f"type {axis_type!r} is not in the measured type vocabulary "
            f"{sorted(KNOWN_TYPES)!r}; measure it in spikes/spike_ch_vocabulary.py "
            f"before adding it."
        )

    status = "ok"
    skip_reason: str | None = None
    warnings: list[str] = []

    # Rule order (first skip wins): the dataset cannot parameterise the type;
    # nulls require nullable; unsigned requires non-negative min; nullable
    # sorting key; range fit (decimals, integers); codec applicability;
    # skip-index applicability; projection-with-primary-key warning.
    if isinstance(resolved, TypeResolutionError):
        status = "skipped"
        skip_reason = str(resolved)
        resolved = axis_type

    if status == "ok" and null_count > 0 and not nullable:
        status = "skipped"
        skip_reason = (
            f"Dataset contains {null_count} null value(s) but the "
            f"column is not nullable."
        )

    if (
        status == "ok"
        and axis_type.startswith("UInt")
        and value_min is not None
        and value_min < 0
    ):
        status = "skipped"
        skip_reason = (
            f"Unsigned type {axis_type} cannot represent the achieved "
            f"minimum value {value_min}, which is negative."
        )

    # Measured on ClickHouse 26.8.2.7, 2026-09-16, by executing the rendered
    # DDL: a nullable column cannot be the sorting key unless the MergeTree
    # setting `allow_nullable_key` is enabled, which the design never decided
    # to turn on, so this is a skip rather than a DDL setting change.
    if status == "ok" and nullable and primary_key:
        status = "skipped"
        skip_reason = (
            "Sorting key contains nullable columns, but merge tree "
            "setting `allow_nullable_key` is disabled"
        )

    if status == "ok":
        reason = _range_skip_reason(axis_type, resolved, dataset, value_min, value_max)
        if reason is not None:
            status = "skipped"
            skip_reason = reason

    if status == "ok":
        reason = _codec_skip_reason(
            axis_type, codec, resolved, compression, nullable=nullable
        )
        if reason is not None:
            status = "skipped"
            skip_reason = reason

    if status == "ok":
        reason = _skip_index_skip_reason(axis_type, skip_index, resolved, nullable)
        if reason is not None:
            status = "skipped"
            skip_reason = reason

    if status == "ok" and projection and primary_key:
        warnings.append(
            "projection is redundant with primary_key: both order the table by v"
        )

    ch_type = resolved
    if nullable:
        # Nullable wraps first: Nullable(LowCardinality(X)) is illegal in
        # ClickHouse (code 43, "Nested type LowCardinality(X) cannot be
        # inside Nullable type"); only LowCardinality(Nullable(X)) is
        # accepted, so LowCardinality must be the outer wrapper.
        ch_type = f"Nullable({ch_type})"
    if codec == "LowCardinality":
        ch_type = f"LowCardinality({ch_type})"

    return TableVariant(
        ch_type=ch_type,
        nullable=nullable,
        primary_key=primary_key,
        projection=projection,
        codec=codec,
        skip_index=skip_index,
        compression=compression,
        status=status,
        skip_reason=skip_reason,
        warnings=tuple(warnings),
    )


def expand_grid(
    suite: Suite,
    *,
    null_count: int,
    value_min,
    value_max,
    clickhouse_version: str,
    array: pa.Array | None = None,
) -> list[TableVariant]:
    """Expand `suite.clickhouse`'s axes into the cartesian product of table
    variants, validating each cell against the measured applicability rules.

    `null_count`, `value_min` and `value_max` are the dataset's achieved
    values (from the generator's manifest), not the requested spec: the
    unsigned-min and nulls-require-nullable rules must run against what the
    data actually contains. `clickhouse_version` is accepted here (and
    forwarded to `table_id` by the orchestrator) because the applicability
    matrices are themselves per-version; the current matrices, measured
    against 26.8.2.7 and 26.8.31.2, do not yet vary by version.

    `array` is the loaded dataset column: the types whose parameters are its
    values (`FixedString` over strings, `Enum8` / `Enum16`) are resolved from
    it, once per type (`resolve_type`); a type the dataset cannot parameterise
    makes its cells `skipped` with the reason. Without it those types raise
    `ValueError`.
    """
    axes = suite.clickhouse
    resolved = {}
    for axis_type in axes.type:
        try:
            resolved[axis_type] = resolve_type(axis_type, suite.dataset, array)
        except TypeResolutionError as e:
            resolved[axis_type] = e
    variants = []
    for (
        axis_type,
        nullable,
        primary_key,
        projection,
        codec,
        skip_index,
        compression,
    ) in itertools.product(
        axes.type,
        axes.nullable,
        axes.primary_key,
        axes.projection,
        axes.codec,
        axes.skip_index,
        axes.compression,
    ):
        variants.append(
            _build_variant(
                suite.dataset,
                axis_type,
                nullable,
                primary_key,
                projection,
                codec,
                skip_index,
                compression,
                null_count,
                value_min,
                value_max,
                resolved[axis_type],
            )
        )
    return variants


def table_id(variant: TableVariant, dataset_id: str, clickhouse_version: str) -> str:
    """A 16-hex hash of the variant's seven axis values (ch_type, nullable,
    primary_key, projection, codec, skip_index, compression) plus the
    dataset id and server version. Stable across calls with the same inputs;
    changes if any input changes."""
    key = "\x1f".join(
        [
            variant.ch_type,
            str(variant.nullable),
            str(variant.primary_key),
            str(variant.projection),
            variant.codec,
            variant.skip_index,
            variant.compression,
            dataset_id,
            clickhouse_version,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
