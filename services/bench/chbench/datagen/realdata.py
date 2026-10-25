"""Real-data datasets (spec section 4.6, ruling R35).

A `DatasetSpec` with `path` / `column` describes an existing Arrow IPC file
column instead of a generator recipe. Its `rows`, `arrow_type`, `null_count`,
`decimal_places`, `value_range` and `content_hash` are measured from the file
by `path_spec`; the generation targets do not apply. Raises `SpecError` only,
like the rest of the spec layer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from chbench.datagen.spec import _SCALARS, DatasetSpec, SpecError, parse_arrow_type

__all__ = ["arrow_type_string", "file_digest", "read_column", "path_spec"]


def arrow_type_string(dtype: pa.DataType) -> str:
    """The spec section 4.1 spelling of `dtype`; SpecError naming the type when
    chbench's vocabulary cannot express it (large_string, decimal256, list,
    ...)."""
    for name, ctor in _SCALARS.items():
        if dtype == ctor():
            return name
    s = None
    if pa.types.is_decimal32(dtype):
        s = f"decimal32({dtype.precision},{dtype.scale})"
    elif pa.types.is_decimal64(dtype):
        s = f"decimal64({dtype.precision},{dtype.scale})"
    elif pa.types.is_decimal128(dtype):
        s = f"decimal128({dtype.precision},{dtype.scale})"
    elif pa.types.is_timestamp(dtype):
        s = (
            f"timestamp[{dtype.unit}]"
            if dtype.tz is None
            else f"timestamp[{dtype.unit}, tz={dtype.tz}]"
        )
    elif pa.types.is_fixed_size_binary(dtype):
        s = f"fixed_size_binary({dtype.byte_width})"
    elif pa.types.is_time32(dtype):
        s = f"time32[{dtype.unit}]"
    elif pa.types.is_time64(dtype):
        s = f"time64[{dtype.unit}]"
    if s is None or parse_arrow_type(s) != dtype:
        raise SpecError(
            f"arrow type {dtype} cannot be expressed in chbench's arrow_type "
            "vocabulary (spec section 4.1)"
        )
    return s


def file_digest(path: str | Path, column: str) -> str:
    """16 hex characters of SHA-256 over the file's bytes and the column name:
    a changed file, or another column of it, is another dataset."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update(b"\0" + column.encode("utf-8"))
    return h.hexdigest()[:16]


def read_column(path: str | Path, column: str) -> pa.Array:
    """The column of an Arrow IPC file (Feather v2) as one array, copied out of
    the file so no handle stays open (as `cache._load`). SpecError naming the
    path (and column) when the file, its format or the column is missing."""
    try:
        with pa.OSFile(str(path), "rb") as source:
            table = pa.ipc.open_file(source).read_all()
    except (OSError, pa.ArrowInvalid) as e:
        raise SpecError(
            f"dataset path {str(path)!r} is not a readable Arrow IPC " f"file: {e}"
        ) from e
    if column not in table.schema.names:
        raise SpecError(
            f"dataset column {column!r} is not in {str(path)!r}; it "
            f"has {table.schema.names!r}"
        )
    chunked = table.column(column)
    return chunked.chunk(0) if chunked.num_chunks == 1 else chunked.combine_chunks()


def widen_decimal(array: pa.Array) -> pa.Array:
    """`array`, with a decimal32 / decimal64 column cast to decimal128 of the same
    precision and scale (several Arrow kernels, e.g. dictionary_encode, have none
    for the narrow decimals); any other array unchanged."""
    t = array.type
    if pa.types.is_decimal32(t) or pa.types.is_decimal64(t):
        return array.cast(pa.decimal128(t.precision, t.scale))
    return array


def _value_range(array: pa.Array) -> tuple[float, float] | None:
    """Min and max of the non-null values of a numeric, temporal or boolean
    column, in the generator's units (temporals as integers in their unit's
    resolution); None for strings or when every value is null."""
    t = array.type
    if (
        pa.types.is_string(t)
        or pa.types.is_fixed_size_binary(t)
        or array.null_count == len(array)
    ):
        return None
    array = widen_decimal(array)
    if pa.types.is_date32(t):
        array = array.cast(pa.int32())
    elif pa.types.is_timestamp(t) or pa.types.is_time64(t):
        array = array.cast(pa.int64())
    elif pa.types.is_time32(t):
        array = array.cast(pa.int32())
    elif pa.types.is_boolean(t):
        array = array.cast(pa.int8())
    bounds = pc.min_max(array)
    return float(bounds["min"].as_py()), float(bounds["max"].as_py())


def path_spec(
    path: str | Path, column: str, *, seed: int = 0, decimal_places: int | None = None
) -> DatasetSpec:
    """Measure `column` of the Arrow IPC file at `path` into a validated
    `DatasetSpec`. A decimal column's `decimal_places` is its scale (an explicit
    value must equal it); for a float column it is the explicit value, else
    None. The path is made absolute."""
    path = os.path.abspath(path)
    array = read_column(path, column)
    dtype = array.type
    if pa.types.is_decimal(dtype) and decimal_places is None:
        decimal_places = dtype.scale
    spec = DatasetSpec(
        rows=len(array),
        arrow_type=arrow_type_string(dtype),
        null_count=array.null_count,
        seed=seed,
        value_range=_value_range(array),
        decimal_places=decimal_places,
        path=path,
        column=column,
        content_hash=file_digest(path, column),
    )
    spec.validate()
    return spec
