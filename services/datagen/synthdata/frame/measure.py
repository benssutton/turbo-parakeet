"""Statistics measured on a generated column (after the final row order) and between a
derived column and its parent."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from synthdata.frame.values import dense_ids

NAN = float("nan")


def column_stats(values: pa.Array, rows: int, integer: bool) -> dict:
    """`values` are the column's non-null base values in row order.

    distinct, entropy_norm (H / log2 d), rho (correlation of position with value rank),
    run_ratio (expected runs under a random order / observed runs of equal values),
    descents (adjacent pairs that decrease), gcd and max_step (integer bases only)."""
    m = len(values)
    out = {
        "rows": rows,
        "null_count": rows - m,
        "distinct": 0,
        "entropy_norm": NAN,
        "rho": NAN,
        "run_ratio": NAN,
        "descents": 0,
        "gcd": None,
        "max_step": None,
    }
    if m == 0:
        return out
    ids = dense_ids(values)
    counts = np.bincount(ids)
    d = counts.size
    out["distinct"] = int(d)
    if d >= 2:
        p = counts / m
        out["entropy_norm"] = float(-(p * np.log2(p)).sum() / math.log2(d))
        diff = np.diff(ids)
        out["descents"] = int((diff < 0).sum())
        out["rho"] = float(np.corrcoef(np.arange(m), ids)[0, 1])
        runs = 1 + np.count_nonzero(diff)
        expected = 1 + (m * m - (counts.astype(np.float64) ** 2).sum()) / m
        out["run_ratio"] = float(expected / runs)
    if integer:
        v = values.to_numpy().astype(np.int64)
        out["gcd"] = int(np.gcd.reduce(np.abs(v)))
        out["max_step"] = int(np.abs(np.diff(v)).max()) if m > 1 else 0
    return out


def containment(child: pa.Array, parent: pa.Array) -> float:
    """Share of the child's distinct non-null values that are among the parent's."""
    c = pc.unique(child.drop_null())
    if len(c) == 0:
        return NAN
    inside = pc.is_in(c, value_set=pc.unique(parent.drop_null()))
    return pc.sum(inside).as_py() / len(c)


def _ids_with_null(a: pa.Array) -> np.ndarray:
    enc = pc.dictionary_encode(a)
    return enc.indices.fill_null(len(enc.dictionary)).to_numpy().astype(np.int64)


def _entropy(ids: np.ndarray) -> float:
    p = np.bincount(ids) / ids.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def nmi(a: pa.Array, b: pa.Array) -> float:
    """(H(a) + H(b) - H(a, b)) / min(H(a), H(b)), null one category."""
    ia, ib = _ids_with_null(a), _ids_with_null(b)
    ha, hb = _entropy(ia), _entropy(ib)
    hab = _entropy(ia * (ib.max() + 1) + ib)
    lo = min(ha, hb)
    return NAN if lo == 0 else (ha + hb - hab) / lo


def joint_distinct(a: pa.Array, b: pa.Array) -> int:
    """Distinct (a, b) pairs over the rows where both are non-null."""
    both = pc.and_(pc.is_valid(a), pc.is_valid(b))
    ia = _ids_with_null(a.filter(both))
    ib = _ids_with_null(b.filter(both))
    return int(np.unique(ia * (ib.max(initial=0) + 1) + ib).size)
