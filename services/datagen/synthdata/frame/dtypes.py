"""Base arrays (int64 / float64 / string) to the column's final type."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

DECIMAL_PRECISION, DECIMAL_SCALE = 12, 2


def to_type(base: pa.Array, type_: str) -> pa.Array:
    if type_ in ("int64", "float64", "string"):
        return base
    if type_ in ("categorical", "enum"):
        return pc.dictionary_encode(base)
    if type_ == "binary":
        return base.cast(pa.binary())
    if type_ == "bool":
        return base.cast(pa.bool_())
    if type_ == "uint32":
        return base.cast(pa.uint32())
    if type_ == "date32":
        return base.cast(pa.int32()).cast(pa.date32())
    if type_ == "timestamp":
        return base.cast(pa.timestamp("us"))
    if type_ == "duration":
        return base.cast(pa.duration("us"))
    if type_ == "time64":
        return base.cast(pa.time64("us"))
    if type_ == "decimal":
        return _decimal_from_unscaled(base)
    raise ValueError(f"unknown type {type_!r}")


def _decimal_from_unscaled(base: pa.Array) -> pa.Array:
    """Decimal128(12, 2) whose unscaled values are the int64 base (two's-complement limbs)."""
    base = base.combine_chunks() if isinstance(base, pa.ChunkedArray) else base
    if base.offset:  # the validity bitmap below is read from bit 0
        base = pa.concat_arrays([base])
    v = base.fill_null(0).to_numpy().astype(np.int64)
    limbs = np.empty((v.size, 2), np.int64)
    limbs[:, 0], limbs[:, 1] = v, v >> 63
    validity = base.buffers()[0] if base.null_count else None
    return pa.Array.from_buffers(
        pa.decimal128(DECIMAL_PRECISION, DECIMAL_SCALE),
        len(base),
        [validity, pa.py_buffer(limbs.tobytes())],
        null_count=base.null_count,
    )


def to_polars(table: pa.Table, types: dict[str, str]) -> pl.DataFrame:
    """The frame in Polars; dictionary columns of type "enum" become Enum (sorted categories)."""
    df = pl.from_arrow(table)
    enums = {}
    for name, t in types.items():
        if t == "enum":
            cats = sorted(df[name].cast(pl.String).drop_nulls().unique().to_list())
            enums[name] = df[name].cast(pl.String).cast(pl.Enum(cats))
    return df.with_columns(**enums) if enums else df
