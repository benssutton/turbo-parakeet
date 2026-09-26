"""Helpers shared by Describe's base and its Python implementations."""

from __future__ import annotations

from datetime import time

import numpy as np
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


def subsets(n: int, seed: int) -> np.ndarray:
    """Seeded 3-way split for the Schnabel estimate (Python implementations).
    Rust uses splitmix64(seed + row) % 3, so capture histories differ between them
    and are compared through the Schnabel estimate (within 10%)."""
    return np.random.default_rng(seed).integers(0, 3, n)


def frequency_summary(count, first, mask, n_rows: int, n_null: int) -> dict:
    """Group A metrics from a frequency table of distinct non-null values: their
    counts, first-occurrence row indices and OR-ed split masks (numpy arrays)."""
    count = np.asarray(count, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    cats = np.append(count, n_null) if n_null else count
    p = cats / n_rows if n_rows else cats.astype(float)
    top = np.lexsort((first, -count))[:5]  # count desc, then first occurrence asc
    return {
        "n_unique": len(count),
        "entropy": float(-(p * np.log2(p)).sum()) + 0.0 if n_rows else float("nan"),
        "f1": int((count == 1).sum()),
        "f2": int((count == 2).sum()),
        "top5_idx": first[top].tolist(),
        "top5_count": count[top].tolist(),
        "capture_history": np.bincount(np.asarray(mask, dtype=np.int64), minlength=8)[1:8].tolist(),
    }


def frac_digits(reprs: pl.Series) -> int | None:
    """Max decimal places over shortest round-trip float strings ("0.1", "1e-7",
    "1.5e+20", "3.0"): max(0, fraction digits without trailing zeros − exponent).
    None when there are none."""
    if reprs.len() == 0:
        return None
    parts = reprs.str.extract_groups(r"^-?[0-9]+(?:\.([0-9]*?)0*)?(?:[eE]\+?(-?[0-9]+))?$")
    frac = parts.struct.field("1").str.len_bytes().fill_null(0).cast(pl.Int64)
    exp = parts.struct.field("2").cast(pl.Int64).fill_null(0)
    return int((frac - exp).clip(lower_bound=0).max())


def n_midnight(s: pl.Series) -> int | None:
    """Datetime values at exactly 00:00:00 local time (column time zone, else naive)."""
    if not isinstance(s.dtype, pl.Datetime):
        return None
    return int((s.dt.time() == time(0)).sum())
