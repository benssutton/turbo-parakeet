"""Dtype groupings shared by the technique bases."""

import polars as pl

INTEGERS_64 = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)
NESTED = (pl.List, pl.Array)

# Dtypes the Rust encoder (src/shared.rs::encode_series) accepts. Anything else
# (Struct, Binary, Null, Object, UInt128) makes the plugin raise.
ENCODABLE = (
    *INTEGERS_64, pl.Int128, pl.Boolean, pl.Float32, pl.Float64,
    pl.Date, pl.Datetime, pl.Duration, pl.Time, *STRING_LIKE, pl.Decimal, *NESTED,
)


def encodable(dtype: pl.DataType) -> bool:
    return isinstance(dtype, ENCODABLE)


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
