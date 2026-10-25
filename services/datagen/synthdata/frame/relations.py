"""Columns derived from a parent column's base values (int64 / float64 / string)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from synthdata.frame.spec import CompositeKey, ForeignKey, NoisyCopy, Relabel, Subset
from synthdata.frame.values import dense_ids, null_mask, scatter


def child_type(relation, parent_type: str) -> str:
    if isinstance(relation, Relabel):
        return relation.type
    if isinstance(relation, CompositeKey):
        return "int64"
    return parent_type


def derive(relation, parent: pa.Array, rng: np.random.Generator) -> pa.Array:
    n = len(parent)
    distinct = pc.unique(parent.drop_null())
    if isinstance(relation, NoisyCopy):
        k = round(relation.p * n)
        idx = np.arange(n)
        pos = rng.choice(n, k, replace=False)
        idx[pos] = n + rng.integers(0, len(distinct), k)
        return pa.concat_arrays([parent, distinct]).take(pa.array(idx))
    if isinstance(relation, Relabel):
        valid = pc.is_valid(parent).to_numpy(zero_copy_only=False)
        ids = np.zeros(n, np.int64)
        ids[valid] = dense_ids(parent.drop_null())
        new = rng.permutation(len(distinct))[ids]
        if relation.type == "string":
            return pa.array(np.char.add("r", new.astype(str)), pa.string(), mask=~valid)
        return pa.array(new, pa.int64(), mask=~valid)
    if isinstance(relation, ForeignKey):
        d = len(distinct)
        w = None
        if relation.zipf_exponent is not None:
            w = rng.permutation(
                np.arange(1, d + 1, dtype=np.float64) ** -relation.zipf_exponent
            )
            w /= w.sum()
        k = round(relation.null_share * n)
        values = distinct.take(pa.array(rng.choice(d, n - k, p=w)))
        return scatter(values, null_mask(n, k, None, rng))
    if isinstance(relation, Subset):
        d = len(distinct)
        keep = round(relation.share * d)
        kept = distinct.take(pa.array(rng.choice(d, keep, replace=False)))
        pool = pa.concat_arrays([kept, _fresh(distinct, d - keep)])
        pick = rng.integers(0, len(pool), n)
        pick[: min(n, len(pool))] = np.arange(
            min(n, len(pool))
        )  # every pool value appears
        return pool.take(pa.array(rng.permutation(pick)))
    if isinstance(relation, CompositeKey):
        ids = dense_ids(parent)
        by_value = np.argsort(ids, kind="stable")
        sorted_ids = ids[by_value]
        starts = np.flatnonzero(np.r_[True, np.diff(sorted_ids) != 0])
        group_start = np.repeat(starts, np.diff(np.r_[starts, n]))
        out = np.empty(n, np.int64)
        out[by_value] = np.arange(n) - group_start
        return pa.array(out)
    raise TypeError(f"unknown relation {relation!r}")


def _fresh(distinct: pa.Array, k: int) -> pa.Array:
    """k values of the parent's base type that the parent does not hold."""
    if pa.types.is_string(distinct.type):
        return pa.array(
            [f"new_{i}" for i in range(k)], pa.string()
        )  # generated strings are alphanumeric
    top = pc.max(distinct).as_py() if len(distinct) else 0
    return pa.array(np.arange(1, k + 1, dtype=np.int64) + int(top), distinct.type)
