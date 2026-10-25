"""Generate a frame from a FrameSpec and measure it."""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from synthdata.frame.dtypes import to_polars, to_type
from synthdata.frame.measure import column_stats, containment, joint_distinct, nmi
from synthdata.frame.relations import child_type, derive
from synthdata.frame.spec import (
    INTEGER_TYPES,
    ColumnSpec,
    CompositeKey,
    ForeignKey,
    FrameSpec,
    NoisyCopy,
    Relabel,
    Subset,
    Walk,
    frame_id,
    spec_json,
)
from synthdata.frame.values import base_values, distinct_count, null_mask, scatter


class TargetMissed(Exception):
    """Strict mode: a column or relation did not get what its spec asked for."""

    def __init__(self, column: str, target: str, wanted, achieved):
        self.column, self.target, self.wanted, self.achieved = (
            column,
            target,
            wanted,
            achieved,
        )
        super().__init__(
            f"{column}: {target}: wanted {wanted!r}, achieved {achieved!r}"
        )


@dataclass(eq=False)
class FrameManifest:
    frame_id: str
    rows: int
    seed: int
    numpy_version: str
    columns: dict  # name -> {"type", "spec", **column_stats}
    relations: dict = field(default_factory=dict)  # child -> metrics
    warnings: list = field(default_factory=list)
    generation_seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps(_nan_to_none(asdict(self)), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FrameManifest":
        return cls(**_none_to_nan(json.loads(text)))

    def __eq__(self, other) -> bool:  # NaN-aware: compare the JSON forms
        return isinstance(other, FrameManifest) and self.to_json() == other.to_json()


_NAN_KEYS = ("entropy_norm", "rho", "run_ratio", "containment", "nmi")


def _nan_to_none(o):
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return None if isinstance(o, float) and math.isnan(o) else o


def _none_to_nan(o, key=None):
    if isinstance(o, dict):
        return {k: _none_to_nan(v, k) for k, v in o.items()}
    return float("nan") if o is None and key in _NAN_KEYS else o


def _column_seed(seed: int, name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([seed, zlib.crc32(name.encode())])


def generate_frame(spec: FrameSpec, strict: bool = False):
    """(polars DataFrame, FrameManifest). Strict: a missed target raises TargetMissed."""
    spec.validate()
    started = time.perf_counter()
    bases: dict[str, pa.Array] = {}
    types: dict[str, str] = {}
    warnings: list[str] = []
    for name, col in spec.columns.items():
        values_seq, nulls_seq = _column_seed(spec.seed, name).spawn(2)
        rng = np.random.default_rng(nulls_seq)
        if isinstance(col, ColumnSpec):
            k = round(col.null_share * spec.rows)
            values, w = base_values(col, spec.rows - k, values_seq)
            warnings += [f"{name}: {x}" for x in w]
            bases[name] = scatter(values, null_mask(spec.rows, k, col.null_run, rng))
            types[name] = col.type
        else:
            bases[name] = derive(col, bases[col.parent], rng)
            types[name] = child_type(col, types[col.parent])
    if spec.order_by:
        bases = _frame_order(spec, bases)
    columns = {
        name: {
            "type": types[name],
            "spec": spec_json(col),
            **column_stats(
                bases[name].drop_null(), spec.rows, types[name] in INTEGER_TYPES
            ),
        }
        for name, col in spec.columns.items()
    }
    relations = {
        name: _relation_metrics(col, bases[name], bases[col.parent])
        for name, col in spec.columns.items()
        if not isinstance(col, ColumnSpec)
    }
    manifest = FrameManifest(
        frame_id(spec),
        spec.rows,
        spec.seed,
        np.__version__,
        columns,
        relations,
        warnings,
        time.perf_counter() - started,
    )
    if strict:
        check(spec, manifest)
    table = pa.table({name: to_type(b, types[name]) for name, b in bases.items()})
    return to_polars(table, types), manifest


def _frame_order(spec: FrameSpec, bases: dict) -> dict:
    keys = pa.table({c: bases[c] for c in spec.order_by})
    # Nulls sort last (the default placement).
    sort_keys = [(c, "ascending") for c in spec.order_by]
    perm = pc.sort_indices(keys, sort_keys=sort_keys).to_numpy()
    if spec.order_disorder > 0:
        rng = np.random.default_rng(_column_seed(spec.seed, "__order__"))
        moved = rng.choice(
            perm.size, round(spec.order_disorder * perm.size), replace=False
        )
        perm[moved] = perm[rng.permutation(moved)]
    take = pa.array(perm)
    return {k: v.take(take) for k, v in bases.items()}


def _relation_metrics(rel, child: pa.Array, parent: pa.Array) -> dict:
    out = {
        "kind": type(rel).__name__,
        "parent": rel.parent,
        "containment": containment(child, parent),
        "nmi": nmi(child, parent),
    }
    if isinstance(rel, CompositeKey):
        out["joint_distinct"] = joint_distinct(parent, child)
    return out


def check(spec: FrameSpec, manifest: FrameManifest) -> None:
    """Raise TargetMissed for the first column or relation off its spec."""
    for w in manifest.warnings:
        name, _, text = w.partition(": ")
        if text.startswith(("miss:", "clamp:")):
            metric = text.split(": ")[1] if ": " in text else text
            raise TargetMissed(name, metric, "within tolerance", text)
    for name, col in spec.columns.items():
        got = manifest.columns[name]
        if isinstance(col, ColumnSpec):
            m = spec.rows - round(col.null_share * spec.rows)
            if m and not isinstance(col.structure, Walk):
                want = distinct_count(col, m)
                if abs(got["distinct"] - want) > max(1, 0.01 * want):
                    raise TargetMissed(name, "distinct", want, got["distinct"])
            continue
        rel = manifest.relations[name]
        if isinstance(col, (ForeignKey, NoisyCopy)) and rel["containment"] != 1.0:
            raise TargetMissed(name, "containment", 1.0, rel["containment"])
        if isinstance(col, Relabel) and not rel["nmi"] > 1 - 1e-9:
            raise TargetMissed(name, "nmi", 1.0, rel["nmi"])
        if isinstance(col, Subset) and abs(
            rel["containment"] - col.share
        ) > 0.02 + 1 / max(1, got["distinct"]):
            raise TargetMissed(name, "containment", col.share, rel["containment"])
        if isinstance(col, CompositeKey) and rel["joint_distinct"] != spec.rows:
            raise TargetMissed(name, "joint_distinct", spec.rows, rel["joint_distinct"])
