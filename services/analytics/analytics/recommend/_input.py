"""Input preparation shared by the recommenders (spec
docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md §5.2)."""

from __future__ import annotations

import polars as pl

from analytics._dtypes import holds_wide_integer
from analytics.base import _normalise


def holds_nested_null(dtype: pl.DataType) -> bool:
    """Null below the top level: ineligible (py-polars also exports such a Null level
    with a buffer, which the Arrow boundary refuses)."""
    if isinstance(dtype, (pl.List, pl.Array)):
        return dtype.inner == pl.Null or holds_nested_null(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(
            f.dtype == pl.Null or holds_nested_null(f.dtype) for f in dtype.fields
        )
    return False


def boolean_pairs(pairs) -> tuple[tuple[str, str], ...]:
    """`pairs` as tuples of two strings, else ValueError (Rust checks the values)."""
    out = tuple(tuple(p) for p in pairs)
    if any(len(p) != 2 or not all(isinstance(v, str) for v in p) for p in out):
        raise ValueError(f"boolean_pairs must be pairs of two strings, got {pairs!r}")
    return out


def top_k(k) -> int | None:
    """`k` checked: a non-negative int, or None for every ranked value."""
    if k is None:
        return None
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"top_k must be a non-negative integer or None, got {k!r}")
    return k


def prepare(frame) -> tuple[object, list[tuple[str, str]]]:
    """`frame` as the Rust side reads it, and (name, dtype) of the columns dropped
    from it as ineligible: Int128 / UInt128, Object and nested-Null. Arrow objects
    pass through unchanged."""
    ineligible: list[tuple[str, str]] = []
    if isinstance(frame, pl.DataFrame):
        ineligible = [
            (name, str(dtype))
            for name, dtype in frame.schema.items()
            if holds_wide_integer(dtype)
            or isinstance(dtype, pl.Object)
            or holds_nested_null(dtype)
        ]
        if len(ineligible) == frame.width:
            # Dropping every column would lose the height; the rows still count.
            frame = pl.DataFrame(height=frame.height)
        else:
            # A sliced Array / Struct with nulls exports as invalid Arrow (see
            # analytics.base._normalise): rebuild those columns.
            frame = _normalise(frame.drop([name for name, _ in ineligible]))
        if any(dtype == pl.Null for dtype in frame.schema.values()):
            # py-polars exports a Null column with one buffer, which arrow-rs
            # rejects (the Null type has none); pyarrow re-exports it with none.
            frame = frame.to_arrow()
    return frame, ineligible
