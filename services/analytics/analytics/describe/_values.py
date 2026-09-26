"""Helpers shared by Describe's base and its Python implementations."""

from __future__ import annotations

import polars as pl

FLOATS = (pl.Float32, pl.Float64)
STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)
INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def flatten(s: pl.Series) -> pl.Series:
    """Values one nesting level down, in order, skipping null lists.

    Element i of the result is what `inner_argmin` / `inner_top5_idx` index into.
    Empty lists are filtered before exploding because explode turns them into a
    null row. The Rust kernel (describe/mod.rs::flatten) uses the same definition.
    """
    valid = s.drop_nulls()
    lengths = valid.list.len() if isinstance(s.dtype, pl.List) else valid.arr.len()
    return valid.filter(lengths > 0).explode()
