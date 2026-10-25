"""A column's non-null base values (integer, float or string), their order, and the null
mask they are scattered around."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from synthdata.column.spec import DatasetSpec
from synthdata.frame.spec import (
    STRING_TYPES,
    Clustered,
    ColumnSpec,
    Multiples,
    Random,
    Runs,
    Sorted,
    Walk,
)
from synthdata.generate import generate

# Integer base ranges per type (the value, or unscaled / physical value), wide enough for
# near-unique columns at benchmark scale.
INT_RANGE = {
    "int64": (-(10**9), 10**9),
    "uint32": (0, 4 * 10**9),
    "decimal": (-(10**11), 10**11),
    "bool": (0, 1),
    "date32": (0, 600_000),  # near-unique multiples of 7 at 73 728 rows fit
    "timestamp": (1_500_000_000_000_000, 1_800_000_000_000_000),
    "duration": (0, 10**12),
    "time64": (0, 86_400 * 10**6 - 1),
}
FLOAT_RANGE, FLOAT_PLACES = (-1e4, 1e4), 3
STRING_LENGTH = 8


def base_kind(type_: str) -> str:
    """ "string", "float" or "int": the base the column pipeline generates."""
    if type_ in STRING_TYPES:
        return "string"
    return "float" if type_ == "float64" else "int"


def distinct_count(spec: ColumnSpec, m: int) -> int:
    d = spec.distinct if spec.distinct is not None else round(spec.distinct_share * m)
    return max(1, min(int(d), m))


def dataset_spec(spec: ColumnSpec, m: int, seed: int) -> DatasetSpec:
    d = distinct_count(spec, m)
    kw = dict(rows=m, null_count=0, seed=seed, cardinality_ratio=d / m)
    if spec.entropy is not None and 2 <= d < m:
        kw["entropy_bits"] = spec.entropy * math.log2(d)
    kind = base_kind(spec.type)
    if kind == "string":
        return DatasetSpec(arrow_type="string", string_length=STRING_LENGTH, **kw)
    if kind == "float":
        return DatasetSpec(
            arrow_type="float64",
            value_range=FLOAT_RANGE,
            decimal_places=FLOAT_PLACES,
            **kw,
        )
    lo, hi = INT_RANGE[spec.type]
    if isinstance(spec.structure, Multiples):
        lo, hi = -((-lo) // spec.structure.g), hi // spec.structure.g
    return DatasetSpec(arrow_type="int64", value_range=(lo, hi), **kw)


def base_values(
    spec: ColumnSpec, m: int, seq: np.random.SeedSequence
) -> tuple[pa.Array, list[str]]:
    """m non-null base values in the column's order, and the column pipeline's warnings."""
    kind = base_kind(spec.type)
    empty = {"string": pa.string(), "float": pa.float64(), "int": pa.int64()}[kind]
    if m == 0:
        return pa.array([], empty), []
    order_seq, gen_seq = seq.spawn(2)
    rng = np.random.default_rng(order_seq)
    if isinstance(spec.structure, Walk):
        lo, hi = INT_RANGE[spec.type]
        s = spec.structure.max_step
        steps = rng.integers(-s, s + 1, m - 1)
        start = (lo + hi) // 2
        return pa.array(start + np.concatenate([[0], np.cumsum(steps)]), pa.int64()), []
    dataset = generate(dataset_spec(spec, m, int(gen_seq.generate_state(1)[0])))
    values = dataset.array
    if isinstance(spec.structure, Multiples):
        values = pc.multiply(values, pa.scalar(spec.structure.g, pa.int64()))
    return apply_order(values, spec.order, rng), list(dataset.manifest.warnings)


def dense_ids(values: pa.Array) -> np.ndarray:
    """0-based dense ranks of non-null values (equal values share an id)."""
    ranks = pc.rank(values, sort_keys="ascending", tiebreaker="dense")
    return ranks.to_numpy().astype(np.int64) - 1


def apply_order(values: pa.Array, order, rng: np.random.Generator) -> pa.Array:
    n = len(values)
    if n < 2:
        return values
    if isinstance(order, Random):
        return values.take(pa.array(rng.permutation(n)))
    ids = dense_ids(values)
    perm = np.argsort(ids, kind="stable")
    if isinstance(order, Clustered):
        moved = rng.choice(n, round(order.disorder * n), replace=False)
        perm[moved] = perm[rng.permutation(moved)]
    elif isinstance(order, Runs):
        perm = _runs_permutation(ids, perm, order.mean_run, rng)
    elif not isinstance(order, Sorted):
        raise TypeError(f"unknown order {order!r}")
    return values.take(pa.array(perm))


def _runs_permutation(ids, by_value, mean_run, rng) -> np.ndarray:
    """Each value's positions cut into runs of geometric length, the runs shuffled."""
    bounds = np.flatnonzero(np.diff(ids[by_value])) + 1
    chunks = []
    for group in np.split(by_value, bounds):
        start = 0
        while start < group.size:
            length = int(rng.geometric(1 / mean_run))
            chunks.append(group[start : start + length])
            start += length
    return np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])


def null_mask(
    n: int, k: int, null_run: float | None, rng: np.random.Generator
) -> np.ndarray:
    """A boolean mask with k nulls: at random, or in runs of mean length `null_run`."""
    mask = np.zeros(n, bool)
    if k == 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    if null_run is None:
        mask[rng.choice(n, k, replace=False)] = True
        return mask
    runs = max(1, min(k, n - k + 1, round(k / null_run)))
    lengths = 1 + rng.multinomial(k - runs, np.full(runs, 1 / runs))
    gaps = rng.multinomial(n - k - (runs - 1), np.full(runs + 1, 1 / (runs + 1)))
    gaps[1:-1] += 1  # at least one valid row between null runs, so they do not merge
    pos = 0
    for gap, length in zip(gaps, lengths):
        pos += gap
        mask[pos : pos + length] = True
        pos += length
    return mask


def scatter(values: pa.Array, mask: np.ndarray) -> pa.Array:
    """`values` in order at the non-null positions of `mask`; null slots hold no bytes."""
    if mask.all():
        return pa.nulls(mask.size, values.type)
    idx = np.zeros(mask.size, np.int64)
    idx[~mask] = np.arange(len(values))
    return values.take(pa.array(idx, mask=mask))
