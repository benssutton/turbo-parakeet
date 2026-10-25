"""A few representative combinations of a technique instead of all of them (benchmarks).

Eligible, compatible combinations are grouped into strata — the sorted dtype families of
their columns and a cardinality regime from the product of their distinct counts (null
counted as a value): `flat` <= 2**20, `hash` <= 2**64, `u128` above, the joint kernels'
counting paths — and taken one stratum at a time, round-robin, each stratum in a seeded
order."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from turbo_parakeet.techniques.base import Combo, Technique

FLAT, HASH = 2**20, 2**64


def _family(dtype: pl.DataType) -> str:
    if dtype.is_signed_integer():
        return "int"
    if dtype.is_unsigned_integer():
        return "uint"
    if dtype.is_float():
        return "float"
    if isinstance(dtype, pl.Decimal):
        return "decimal"
    if dtype == pl.String:
        return "string"
    if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
        return "categorical"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_temporal():
        return "temporal"
    return str(dtype.base_type())


def _collected(frames: dict) -> dict[str, pl.DataFrame]:
    return {
        n: f.collect() if isinstance(f, pl.LazyFrame) else f for n, f in frames.items()
    }


def strata(technique: Technique, frames: dict) -> dict[tuple, list[Combo]]:
    """Eligible, compatible combinations by (families, regime), canonical order within each."""
    frames = _collected(frames)
    schemas = {n: f.schema for n, f in frames.items()}  # once per frame
    ok = {
        (n, c): technique.eligible(f[c]) for n, f in frames.items() for c in f.columns
    }
    candidates = [
        k
        for k in type(technique).enumerate(frames)
        if all(ok[col] for col in k)
        and technique.compatible([schemas[n][c] for n, c in k])
    ]
    distinct = {
        col: frames[col[0]][col[1]].n_unique()
        for col in dict.fromkeys(col for k in candidates for col in k)
    }
    out: dict[tuple, list[Combo]] = {}
    for k in candidates:
        product = math.prod(distinct[col] for col in k)
        regime = "flat" if product <= FLAT else "hash" if product <= HASH else "u128"
        families = tuple(sorted(_family(schemas[n][c]) for n, c in k))
        out.setdefault((families, regime), []).append(k)
    return out


def stratified_combinations(
    technique: Technique, frames: dict, n: int = 50, seed: int = 0
) -> list[Combo]:
    """At most `n` combinations, one stratum at a time (all of them when n is large)."""
    groups = strata(technique, frames)
    total = sum(len(g) for g in groups.values())
    if n >= total:  # every candidate, in canonical order
        chosen = {k for g in groups.values() for k in g}
        return [k for k in type(technique).enumerate(_collected(frames)) if k in chosen]
    rng = np.random.default_rng(seed)
    queues = [
        [g[i] for i in rng.permutation(len(g))] for _, g in sorted(groups.items())
    ]
    out: list[Combo] = []
    depth = 0
    while len(out) < n:
        for q in queues:
            if depth < len(q):
                out.append(q[depth])
                if len(out) == n:
                    break
        depth += 1
    return out
