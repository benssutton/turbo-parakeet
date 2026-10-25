"""Frame specs: the columns, the relations between them and the row order of a frame."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from synthdata.column.spec import SpecError

# Bump when generation changes the values a spec produces (invalidates the frame cache).
FRAME_VERSION = 1

TYPES = (
    "int64",
    "uint32",
    "float64",
    "decimal",
    "string",
    "categorical",
    "enum",
    "bool",
    "date32",
    "timestamp",
    "duration",
    "time64",
    "binary",
)
STRING_TYPES = frozenset({"string", "categorical", "enum", "binary"})
INTEGER_TYPES = frozenset(
    {"int64", "uint32", "decimal", "bool", "date32", "timestamp", "duration", "time64"}
)
STRUCTURED_TYPES = frozenset(
    {"int64", "uint32", "decimal", "date32", "timestamp", "duration", "time64"}
)


@dataclass(frozen=True)
class Random:
    """The order the values were drawn in (a uniform shuffle)."""


@dataclass(frozen=True)
class Runs:
    """Equal values in runs of geometric length with this mean."""

    mean_run: float = 8.0


@dataclass(frozen=True)
class Clustered:
    """Sorted, then a share `disorder` of the positions shuffled among themselves."""

    disorder: float = 0.1


@dataclass(frozen=True)
class Sorted:
    """Ascending."""


@dataclass(frozen=True)
class Plain:
    """No integer structure."""


@dataclass(frozen=True)
class Multiples:
    """Every value a multiple of g (the ClickHouse GCD codec)."""

    g: int


@dataclass(frozen=True)
class Walk:
    """A random walk with steps in [-max_step, max_step] (Delta / DoubleDelta codecs)."""

    max_step: int = 8


@dataclass(frozen=True)
class ColumnSpec:
    """One generated column. `distinct` (a count) or `distinct_share` (of the non-null
    rows) sets the distinct non-null values; `entropy` is H / log2(distinct) in (0, 1]
    (None: uniform frequencies)."""

    type: str
    distinct: int | None = None
    distinct_share: float | None = None
    entropy: float | None = None
    null_share: float = 0.0
    null_run: float | None = None
    order: Random | Runs | Clustered | Sorted = Random()
    structure: Plain | Multiples | Walk = Plain()


@dataclass(frozen=True)
class NoisyCopy:
    """The parent with a share p of rows redrawn from its distinct values."""

    parent: str
    p: float


@dataclass(frozen=True)
class Relabel:
    """A random bijection of the parent's distinct values, as `type` ("int64" or "string")."""

    parent: str
    type: str = "string"


@dataclass(frozen=True)
class ForeignKey:
    """Drawn from the parent's distinct values (uniform, or Zipf with this exponent)."""

    parent: str
    null_share: float = 0.0
    zipf_exponent: float | None = None


@dataclass(frozen=True)
class Subset:
    """A share of the parent's distinct values plus fresh values for the rest."""

    parent: str
    share: float


@dataclass(frozen=True)
class CompositeKey:
    """With its (null-free, non-unique) parent, unique together: the occurrence index of
    each row within its parent value's group."""

    parent: str


RELATIONS = (NoisyCopy, Relabel, ForeignKey, Subset, CompositeKey)


@dataclass(frozen=True)
class FrameSpec:
    rows: int
    columns: dict = field(
        default_factory=dict
    )  # name -> ColumnSpec | relation, in order
    order_by: tuple[str, ...] = ()  # frame-level order: sorted by these columns
    order_disorder: float = (
        0.0  # with order_by: a share of positions shuffled after sorting
    )
    seed: int = 0

    def validate(self) -> None:
        if self.rows < 0:
            raise SpecError(f"rows must be >= 0, got {self.rows}")
        seen: dict[str, object] = {}
        for name, col in self.columns.items():
            if isinstance(col, ColumnSpec):
                _validate_column(name, col, self.rows)
            elif isinstance(col, RELATIONS):
                _validate_relation(name, col, seen)
            else:
                raise SpecError(f"{name}: not a ColumnSpec or a relation: {col!r}")
            seen[name] = col
        for name in self.order_by:
            if name not in self.columns:
                raise SpecError(f"order_by names unknown column {name!r}")
        if not 0 <= self.order_disorder <= 1:
            raise SpecError(
                f"order_disorder must be in [0, 1], got {self.order_disorder}"
            )


def _validate_column(name: str, c: ColumnSpec, rows: int) -> None:
    if c.type not in TYPES:
        raise SpecError(f"{name}: unknown type {c.type!r}; one of {', '.join(TYPES)}")
    if not 0 <= c.null_share <= 1:
        raise SpecError(f"{name}: null_share must be in [0, 1], got {c.null_share}")
    if c.null_run is not None and c.null_run < 1:
        raise SpecError(f"{name}: null_run must be >= 1, got {c.null_run}")
    walk = isinstance(c.structure, Walk)
    if not walk and c.null_share < 1:
        if (c.distinct is None) == (c.distinct_share is None):
            raise SpecError(
                f"{name}: exactly one of distinct or distinct_share must be set"
            )
        if c.distinct is not None and c.distinct < 1:
            raise SpecError(f"{name}: distinct must be >= 1, got {c.distinct}")
        if c.distinct_share is not None and not 0 < c.distinct_share <= 1:
            raise SpecError(
                f"{name}: distinct_share must be in (0, 1], got {c.distinct_share}"
            )
    if c.entropy is not None and not 0 < c.entropy <= 1:
        raise SpecError(
            f"{name}: entropy is H / log2(distinct), in (0, 1]; got {c.entropy}"
        )
    if not isinstance(c.structure, Plain):
        if c.type not in STRUCTURED_TYPES:
            raise SpecError(
                f"{name}: structure {c.structure} needs an integer-backed type"
            )
        if walk and not isinstance(c.order, Random):
            raise SpecError(f"{name}: a walk fixes the order; order must be Random()")
        if isinstance(c.structure, Multiples) and c.structure.g < 2:
            raise SpecError(f"{name}: Multiples(g) needs g >= 2")
    if isinstance(c.order, Runs) and c.order.mean_run < 1:
        raise SpecError(f"{name}: Runs.mean_run must be >= 1")
    if isinstance(c.order, Clustered) and not 0 <= c.order.disorder <= 1:
        raise SpecError(f"{name}: Clustered.disorder must be in [0, 1]")


def _validate_relation(name: str, r, seen: dict) -> None:
    if r.parent not in seen:
        raise SpecError(f"{name}: parent {r.parent!r} must be an earlier column")
    if isinstance(r, NoisyCopy) and not 0 <= r.p <= 1:
        raise SpecError(f"{name}: NoisyCopy.p must be in [0, 1]")
    if isinstance(r, Relabel) and r.type not in ("int64", "string"):
        raise SpecError(f"{name}: Relabel.type must be int64 or string")
    if isinstance(r, ForeignKey) and not 0 <= r.null_share <= 1:
        raise SpecError(f"{name}: ForeignKey.null_share must be in [0, 1]")
    if isinstance(r, Subset) and not 0 <= r.share <= 1:
        raise SpecError(f"{name}: Subset.share must be in [0, 1]")
    if isinstance(r, CompositeKey):
        parent = seen[r.parent]
        if not isinstance(parent, ColumnSpec) or parent.null_share > 0:
            raise SpecError(
                f"{name}: CompositeKey needs a generated parent without nulls"
            )


def spec_json(obj):
    """A JSON-ready, order-stable rendering of a spec (dataclasses named by their class)."""
    if dataclasses.is_dataclass(obj):
        return {
            "kind": type(obj).__name__,
            **{
                f.name: spec_json(getattr(obj, f.name)) for f in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, dict):
        return {k: spec_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [spec_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    return obj


def frame_id(spec: FrameSpec) -> str:
    payload = {
        "spec": spec_json(spec),
        "numpy": np.__version__,
        "version": FRAME_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
