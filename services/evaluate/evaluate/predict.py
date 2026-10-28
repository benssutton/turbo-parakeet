"""Predictions keyed by cell. Size predictions are written before the run (suites.py,
predicted.json); skip predictions need the values chbench queried, so they are computed after it:
per (skip index, value) the share of marks the index can skip, exactly, from the dataset in
stored order (the former skip calibration's block statistics). Bloom: the expectation x (1 - fpr).
"""

from __future__ import annotations

import calendar
import datetime
import decimal
import math
import re

import numpy as np
import polars as pl

MARK_ROWS = 8192
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'", re.DOTALL)
_WRAPPED = re.compile(
    r"toDateTime(?:64)?\(('(?:[^'\\]|\\.)*')(?:, \d+)?, 'UTC'\)", re.DOTALL
)


def parse_skip(skip: str) -> tuple[str, float | None, int]:
    m = re.fullmatch(r"(minmax|set|bloom_filter)(?:\(([\d.]+)\))?:(\d+)", skip)
    if m is None:
        raise ValueError(f"not a chbench skip value: {skip!r}")
    return (
        m.group(1),
        (None if m.group(2) is None else float(m.group(2))),
        int(m.group(3)),
    )


_NS_PER_UNIT = {"ns": 1, "us": 1_000, "ms": 1_000_000}


def _temporal(text: str, dtype: pl.DataType) -> int:
    """A quoted date / datetime / time as the column's physical integer (days since the epoch;
    epoch ticks in the datetime's unit; nanoseconds since midnight). chbench writes datetimes as
    UTC wall-clock, which is what polars stores for zoned columns too."""
    if dtype == pl.Date:
        return (datetime.date.fromisoformat(text) - datetime.date(1970, 1, 1)).days
    whole, _, frac = text.partition(".")
    ns = int(frac.ljust(9, "0")[:9]) if frac else 0
    if dtype == pl.Time:
        t = datetime.time.fromisoformat(whole)
        return ((t.hour * 60 + t.minute) * 60 + t.second) * 1_000_000_000 + ns
    secs = calendar.timegm(datetime.datetime.fromisoformat(whole).timetuple())
    return (secs * 1_000_000_000 + ns) // _NS_PER_UNIT[dtype.time_unit]


def parse_literal(lit: str, dtype: pl.DataType):
    """The Python value of a ClickHouse literal for a column of `dtype`; None when not parsed
    (a list, durations, ...). chbench renders by the ClickHouse type, so strings, dates,
    datetimes, times and decimals arrive quoted; a quoted number for a numeric dtype is not parsed.
    """
    lit = lit.strip()
    # toDateTime64('..', p, 'UTC') / toDateTime('..', 'UTC')
    wrapped = _WRAPPED.fullmatch(lit)
    if wrapped:
        lit = wrapped.group(1)
    try:
        if _STRING.fullmatch(lit):
            text = re.sub(r"\\(.)", r"\1", lit[1:-1], flags=re.DOTALL)
            if dtype in (pl.String, pl.Categorical) or isinstance(
                dtype, (pl.Enum, pl.Categorical)
            ):
                return text
            if dtype in (pl.Date, pl.Time) or isinstance(dtype, pl.Datetime):
                return _temporal(text, dtype)
            if isinstance(dtype, pl.Decimal):
                return decimal.Decimal(text)
            return None
        if lit.lower() in ("true", "false"):
            return int(lit.lower() == "true")
        if dtype.is_integer():
            return int(lit)
        if dtype.is_float():
            return float(lit)
        if dtype == pl.Boolean:
            return int(lit.lower() in ("true", "1"))
    except (ValueError, decimal.InvalidOperation):
        pass
    return None


def value_of(predicate_sql: str | None, dtype: pl.DataType):
    """The single value of chbench's point predicate `v IN (<literal>)`, or None."""
    m = re.fullmatch(r"v IN \((.*)\)", predicate_sql or "", re.DOTALL)
    return parse_literal(m.group(1), dtype) if m else None


class _Blocks:
    """Per index block (G marks) of a column in stored order."""

    def __init__(self, col: pl.Series, g: int, kind: str):
        self.kind = kind
        rows_per_block = g * MARK_ROWS
        n = len(col)
        self.n_blocks = math.ceil(n / rows_per_block)
        self.marks = np.full(self.n_blocks, g, dtype=np.int64)
        self.marks[-1] = math.ceil(n / MARK_ROWS) - g * (self.n_blocks - 1)
        self.df = pl.DataFrame({"v": col}).with_columns(
            b=(pl.int_range(pl.len()) // rows_per_block).cast(pl.Int64)
        )
        if kind == "minmax":
            agg = (
                self.df.group_by("b")
                .agg(lo=pl.col("v").min(), hi=pl.col("v").max())
                .sort("b")
            )
            self.lo, self.hi = agg["lo"], agg["hi"]
        elif kind == "set":
            # n_unique includes the NULL entry
            self.distinct = (
                self.df.group_by("b")
                .agg(d=pl.col("v").n_unique())
                .sort("b")["d"]
                .to_numpy()
            )

    def skipped_marks(self, v, max_size: int | None) -> float:
        """Marks in blocks the index must be able to skip (exact set semantics)."""
        if self.kind == "minmax":
            out = ((self.lo > v) | (self.hi < v)).fill_null(True).to_numpy()
        else:
            has = np.zeros(self.n_blocks, dtype=bool)
            has[self.df.filter(pl.col("v") == v)["b"].unique().to_numpy()] = True
            out = ~has
            if self.kind == "set" and max_size:
                out &= self.distinct <= max_size
        return float(self.marks[out].sum())


class SkipPredictor:
    """Predicted skip shares for one column (in stored order)."""

    def __init__(self, col: pl.Series):
        self.dtype = col.dtype  # the logical dtype, for parsing literals
        self.col = col.cast(pl.UInt8) if col.dtype == pl.Boolean else col
        # temporal columns are compared as integers: no unit or zone mismatch
        if col.dtype.is_temporal():
            self.col = col.to_physical()
        self.total_marks = math.ceil(len(col) / MARK_ROWS)
        self._blocks: dict[tuple[str, int], _Blocks] = {}

    def skip(self, skip: str, v) -> float:
        if skip == "none":
            return 0.0
        kind, arg, g = parse_skip(skip)
        if (kind, g) not in self._blocks:
            self._blocks[kind, g] = _Blocks(self.col, g, kind)
        share = (
            self._blocks[kind, g].skipped_marks(v, int(arg) if kind == "set" else None)
            / self.total_marks
        )
        return share * (1 - arg) if kind == "bloom_filter" else share

    def rows_of(self, v) -> int:
        return int((self.col == v).sum())
