"""Measured column statistics back into factor levels, and the coverage they add up to."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import pyarrow as pa

from synthdata.cover.factors import FACTORS, MAX_STEP, STRUCTURED
from synthdata.cover.pairwise import allowed_pairs, pairs
from synthdata.frame.measure import column_stats

UNKNOWN = "?"


def _nulls(share: float) -> str:
    if share == 0:
        return "none"
    if share == 1:
        return "all"
    if share <= 0.2:
        return "sparse"
    return "heavy" if 0.3 <= share <= 0.7 else UNKNOWN


def _cardinality(d: int, m: int) -> str:
    if d == 1:
        return "constant"
    if d == 2:
        return "binary"
    if d <= 20:
        return "low"
    if d >= 0.9 * m:
        return "near_unique"
    return "mid" if d <= 0.5 * m else UNKNOWN


# rho of a clustered column is bounded by that of its sorted arrangement: a 90 / 10
# binary column reaches sqrt(3 * 0.1 * 0.9) ≈ 0.52 sorted, ≈ 0.47 at 10% disorder
# (measured 0.40 - 0.51 at 500 rows). Runs of mean 8 leave |rho| as noise of the run
# count: up to 0.26 at 2 000 rows, 0.43 at 500 (a 500-row tail lands in "?").
CLUSTERED_RHO, UNORDERED_RHO = 0.38, 0.35


def _order(stats: dict) -> str:
    rho, ratio = stats["rho"], stats["run_ratio"]
    if stats["descents"] == 0:
        return "sorted"
    if rho >= CLUSTERED_RHO:
        return "clustered"
    if abs(rho) < UNORDERED_RHO and ratio >= 3:
        return "runs"
    if abs(rho) < UNORDERED_RHO and ratio < 1.5:
        return "random"
    return UNKNOWN


def classify(stats: dict, family: str) -> dict:
    """Levels from statistics; "?" where the statistics fall between two levels."""
    rows, nulls = stats["rows"], stats["null_count"]
    out = {f: None for f in FACTORS}
    out["family"] = family
    out["nulls"] = _nulls(nulls / rows if rows else 0.0)
    if out["nulls"] == "all":
        return out
    m, d = rows - nulls, stats["distinct"]
    walk = (
        stats["max_step"] is not None
        and stats["max_step"] <= MAX_STEP
        and stats["descents"] > 0
    )
    if (
        family in STRUCTURED and d >= 3 and walk
    ):  # a sorted dense column: small steps too
        out["structure"] = "walk"
        return out
    out["cardinality"] = _cardinality(d, m)
    if family in STRUCTURED and d >= 3:
        out["structure"] = "multiples" if (stats["gcd"] or 1) > 1 else "plain"
    if d >= 2 and out["cardinality"] != "near_unique":
        e = stats["entropy_norm"]
        out["skew"] = "uniform" if e >= 0.9 else "heavy" if e <= 0.6 else UNKNOWN
    if d >= 2:
        out["order"] = _order(stats)
    return out


@dataclass
class Coverage:
    measured: dict  # column -> levels
    # column -> {factor: (intended, measured)}
    mismatches: dict = field(default_factory=dict)
    covered: set = field(default_factory=set)
    missing: set = field(default_factory=set)


def _report(measured: dict, intended: dict | None) -> Coverage:
    cov = Coverage(measured)
    for name, lv in measured.items():
        cov.covered |= pairs(lv)
        if intended and name in intended:
            want = intended[name]
            diff = {f: (want[f], lv[f]) for f in FACTORS if want[f] != lv[f]}
            if diff:
                cov.mismatches[name] = diff
    cov.missing = allowed_pairs() - cov.covered
    return cov


def coverage(manifest, intended: dict | None = None) -> Coverage:
    """Coverage of a generated frame; `intended` (column -> levels) lists mismatches."""
    measured = {}
    for name, col in manifest.columns.items():
        fam = _TYPE_FAMILY.get(col["type"])
        if fam is not None and (intended is None or name in intended):
            measured[name] = classify(col, fam)
    return _report(measured, intended)


_TYPE_FAMILY = {
    "int64": "int",
    "uint32": "uint",
    "float64": "float",
    "decimal": "decimal",
    "string": "string",
    "categorical": "categorical",
    "enum": "categorical",
    "bool": "boolean",
    "binary": "binary",
    "date32": "temporal",
    "timestamp": "temporal",
    "duration": "temporal",
    "time64": "temporal",
}


def family_of(dtype) -> str | None:
    """The factor family of a Polars dtype (None: nested or unsupported)."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        return "int"
    if dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "uint"
    if dtype in (pl.Float32, pl.Float64):
        return "float"
    if isinstance(dtype, pl.Decimal):
        return "decimal"
    if dtype == pl.String:
        return "string"
    if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
        return "categorical"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype == pl.Binary:
        return "binary"
    if dtype in (pl.Date, pl.Time) or isinstance(dtype, (pl.Datetime, pl.Duration)):
        return "temporal"
    return None


def _base(s: pl.Series) -> tuple[pa.Array, bool]:
    """A column's non-null base values for column_stats: (array, integer base?)."""
    s = s.drop_nulls()
    dt = s.dtype
    if isinstance(dt, pl.Decimal):
        unscaled = (s.cast(pl.Float64) * 10**dt.scale).round().cast(pl.Int64)
        return unscaled.to_arrow(), True
    temporal = dt in (pl.Date, pl.Time) or isinstance(dt, (pl.Datetime, pl.Duration))
    if dt == pl.Boolean or temporal:
        return s.to_physical().cast(pl.Int64).to_arrow(), True
    if dt == pl.Categorical or isinstance(dt, pl.Enum):
        return s.cast(pl.String).to_arrow(), False
    if dt in (pl.Float32, pl.Float64):
        return s.cast(pl.Float64).to_arrow(), False
    if dt.is_integer():
        return s.cast(pl.Int64).to_arrow(), True
    return s.to_arrow(), False


def coverage_of(frame: pl.DataFrame) -> Coverage:
    """Coverage of any frame (e.g. a fixture not built by synthdata)."""
    measured = {}
    for name in frame.columns:
        fam = family_of(frame.schema[name])
        if fam is None:
            continue
        values, integer = _base(frame[name])
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        measured[name] = classify(column_stats(values, frame.height, integer), fam)
    return _report(measured, None)
