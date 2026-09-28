"""Dtype groupings shared by the technique bases."""

import polars as pl

INTEGERS_64 = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)
STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)
NESTED = (pl.List, pl.Array)

# Arrow has no 128-bit integer type. Polars exports Int128 / UInt128 in its private
# formats `_pli128` / `_plu128`, which the Rust extension's Arrow boundary (and every
# non-Polars Arrow consumer) rejects, so columns holding them — at any nesting depth —
# are ineligible in every technique. Cast to Decimal(38, 0) or Int64 to analyse them.
WIDE_INTEGERS = tuple(
    t for t in (pl.Int128, getattr(pl, "UInt128", None)) if t is not None
)


def holds_wide_integer(dtype: pl.DataType) -> bool:
    if dtype in WIDE_INTEGERS:
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return holds_wide_integer(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(holds_wide_integer(f.dtype) for f in dtype.fields)
    return False


# Dtypes the Rust encoder (src/shared.rs::encode_series) accepts. Anything else
# (Struct, Binary, Null, Object, 128-bit integers) makes the extension raise.
ENCODABLE = (
    *INTEGERS_64,
    pl.Boolean,
    pl.Float32,
    pl.Float64,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    *STRING_LIKE,
    pl.Decimal,
    *NESTED,
)


def encodable(dtype: pl.DataType) -> bool:
    return isinstance(dtype, ENCODABLE) and not holds_wide_integer(dtype)


def is_nested(dtype: pl.DataType) -> bool:
    return isinstance(dtype, NESTED)


def value_family(dtype: pl.DataType) -> str:
    """Values of two columns are only comparable (multi-set techniques) within one family.

    Integers up to 64 bits share a family because the Rust encoder widens them to
    the same u64 key; String/Categorical/Enum share one because categoricals hash
    by label. Everything else must match its dtype exactly: a Date 19000 and an
    Int32 19000 share a physical key in Rust but are different values.
    """
    if isinstance(dtype, INTEGERS_64):
        return "int"
    if isinstance(dtype, STRING_LIKE):
        return "str"
    return str(dtype)
