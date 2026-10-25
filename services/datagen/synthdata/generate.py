"""The public entry point and the manifest.

`generate(spec)` runs the whole pipeline: validate, build the multiset (stage
1), lay out the sequence (stage 2), refine the inter-arrival CV when it is
the only ordering target, measure every metric on the final values, wrap to
Arrow, and assemble the manifest.

The manifest is where the best-effort contract (a missed target warns, never
fails) becomes visible. For every metric the DatasetSpec targeted it records
the target, the value achieved on the final array, whether that landed inside tolerance,
and the status and message of any solve. A missed tolerance is a warning,
never an exception; a clamp is recorded distinctly from a near miss because
the solve's `status` says so (`clamped_low` or `clamped_high`) and its
message names the reachable bound. Warnings follow the fixed vocabulary in
`WARNING_PREFIXES` (defined in `solve.py`, re-exported here): every one
starts with `clamp:`, `miss:`, `skipped:` or `undefined:` and then the
metric's name.

After `spec.validate()` nothing here raises deliberately: validation rejects
every value_range or category whose rounded bounds overflow the declared
type, so `wrap`'s own range check is defence in depth and raises
`AssertionError` if the generator ever produces a value outside the
DatasetSpec's bounds.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa

from synthdata.column import metrics
from synthdata.column.multiset import build_multiset
from synthdata.column.refine import refine_interarrival
from synthdata.column.sequence import dichotomy_kind_for, layout_sequence
from synthdata.column.solve import WARNING_PREFIXES, SolveResult, SolveStatus
from synthdata.column.spec import DatasetSpec
from synthdata.column.wrap import wrap

__all__ = [
    "COLUMN_NAME",
    "WARNING_PREFIXES",
    "SolveRecord",
    "Manifest",
    "Dataset",
    "generate",
]

COLUMN_NAME = "value"

# The metrics a spec can target, in manifest order. `cardinality_ratio` and
# `entropy_bits` are always measured, whether or not they were targeted.
_TARGETABLE = (
    "cardinality_ratio",
    "entropy_bits",
    "index_of_dispersion",
    "runs_z",
    "interarrival_cv",
)


@dataclass(frozen=True)
class SolveRecord:
    """How the solve (or layout) that chose a metric's parameter ended.

    status:  one of the `SolveStatus` value strings: "ok", "clamped_low",
             "clamped_high", "invalid_bracket" or "not_converged". A harness
             reads this; it never parses `message`.
    message: the solve's own account of the outcome, for a human reader.
    """

    status: str
    message: str

    @classmethod
    def from_result(cls, result: SolveResult) -> "SolveRecord":
        return cls(status=result.status.value, message=result.message)

    @classmethod
    def from_status(cls, status: SolveStatus, message: str) -> "SolveRecord":
        return cls(status=status.value, message=message)


@dataclass
class Manifest:
    """What one dataset was asked for and what it got.

    Vocabulary. Two fields are machine-readable and a harness must branch on
    them rather than on prose. `solves[name].status` is one of the
    `SolveStatus` value strings: "ok", "clamped_low", "clamped_high",
    "invalid_bracket", "not_converged". Every entry of `warnings` is
    "<prefix> <metric> ..." with the prefix one of `WARNING_PREFIXES`
    ("clamp:", "miss:", "skipped:", "undefined:") and the metric one of
    cardinality_ratio, entropy_bits, index_of_dispersion, runs_z,
    interarrival_cv, so `warning.split()[:2]` classifies it. A clamp means
    the target lay outside the reachable band and the nearest bound was
    honoured; a miss means a reachable target was not hit (a tolerance miss,
    a solve that did not converge, a support shortfall); skipped means a
    step was not run so a target was not pursued; undefined means the metric
    has no value for this data.

    achieved:          metric name -> value measured on the final values, before
                       wrapping. Always holds cardinality_ratio and entropy_bits,
                       plus every metric the DatasetSpec targeted. `nan` when the metric
                       is undefined for the column (e.g. runs_z on a constant).
    targets:           metric name -> the DatasetSpec's target, for targeted
                       metrics only.
    within_tolerance:  metric name -> whether |achieved - target| is inside the
                       DatasetSpec's tolerance (relative for interarrival_cv,
                       absolute otherwise). `None` for interarrival_cv when
                       runs_z is also targeted: the settled combined-target
                       behaviour records the achieved CV without applying its
                       tolerance.
    solves:            metric name -> `SolveRecord` for the solve that chose the
                       parameter for it (entropy_bits, index_of_dispersion,
                       interarrival_cv), or for the runs layout (runs_z: a
                       clamp against the band reachable from the measured
                       split, else ok). Only metrics that had a solve or a
                       runs layout; a target dropped as undefined has none.
    warnings:          every warning the pipeline emitted, in pipeline order,
                       followed by the tolerance misses. See the vocabulary
                       above.
    clip_collapsed:    support values lost to clip collapse (draws past a
                       value_range bound merging onto it).
    round_collapsed:   support values lost to rounding collisions that the
                       bounded redraw could not recover. Both are 0 when no
                       cardinality_ratio was set.
    """

    dataset_id: str
    spec: DatasetSpec
    achieved: dict[str, float]
    targets: dict[str, float]
    within_tolerance: dict[str, bool | None]
    solves: dict[str, SolveRecord]
    warnings: list[str] = field(default_factory=list)
    clip_collapsed: int = 0
    round_collapsed: int = 0
    generated_at: str = ""
    generation_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec"] = json.loads(self.spec.to_json())
        # nan is not JSON; write null so other readers can parse the file.
        d["achieved"] = {
            k: (None if _is_nan(v) else v) for k, v in self.achieved.items()
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        d = dict(d)
        d["spec"] = DatasetSpec.from_dict(d["spec"])
        d["achieved"] = {
            k: (float("nan") if v is None else v) for k, v in d["achieved"].items()
        }
        d["solves"] = {
            k: (v if isinstance(v, SolveRecord) else SolveRecord(**v))
            for k, v in d["solves"].items()
        }
        return cls(**d)


@dataclass
class Dataset:
    array: pa.Array
    manifest: Manifest


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _inner_row_count(spec: DatasetSpec) -> int:
    """How many values the multiset and sequence stages produce.

    For list<T> the DatasetSpec's `rows` counts arrays, and the flat value count
    is rows * round(mean(array_len_range)); `wrap` then chunks
    that many values into `rows` arrays. Every other type generates `rows`.
    """
    if pa.types.is_list(spec.arrow_dtype) and spec.array_len_range is not None:
        lo, hi = spec.array_len_range
        return spec.rows * int(round((lo + hi) / 2))
    return spec.rows


def _measure(values: np.ndarray, spec: DatasetSpec) -> dict[str, float]:
    kind = dichotomy_kind_for(spec)
    achieved = {
        "cardinality_ratio": metrics.cardinality_ratio(values),
        "entropy_bits": metrics.entropy_bits(values),
    }
    if spec.index_of_dispersion is not None:
        achieved["index_of_dispersion"] = metrics.index_of_dispersion(values)
    if spec.runs_z is not None:
        achieved["runs_z"] = metrics.runs_z(values, kind)
    if spec.interarrival_cv is not None:
        achieved["interarrival_cv"] = metrics.interarrival_cv(
            values, decimal_places=spec.decimal_places
        )
    return {k: float(v) for k, v in achieved.items()}


def _check_tolerances(
    spec: DatasetSpec, targets: dict[str, float], achieved: dict[str, float]
) -> tuple[dict[str, bool | None], list[str]]:
    within: dict[str, bool | None] = {}
    warnings: list[str] = []
    tol = spec.tolerances
    for name, target in targets.items():
        got = achieved[name]
        if name == "interarrival_cv":
            if spec.runs_z is not None:
                # Settled combined-target behaviour: honour runs_z, record the
                # achieved CV, do not apply its tolerance.
                within[name] = None
                continue
            limit = tol.interarrival_cv_relative * abs(target)
            desc = f"{tol.interarrival_cv_relative:.0%} relative"
        else:
            limit = getattr(tol, name)
            desc = f"+/-{limit:g}"
        if _is_nan(got):
            # A constant column has no runs standard deviation; fewer than
            # three repeat-runs give no gap CV. That is not a miss against the
            # tolerance, it is the metric having no value for this data.
            within[name] = False
            warnings.append(
                f"undefined: {name} is undefined for this data (achieved nan "
                f"against target {target:g}), so the tolerance cannot apply"
            )
            continue
        ok = abs(got - target) <= limit
        within[name] = bool(ok)
        if not ok:
            warnings.append(
                f"miss: {name}: achieved {got:.6g} is outside the {desc} "
                f"tolerance of target {target:g}"
            )
    return within, warnings


def generate(spec: DatasetSpec) -> Dataset:
    """Run the whole pipeline for `spec` and return the array with its manifest.

    Raises SpecError from `spec.validate()`; after that every miss is a
    manifest warning. The one generator is created from `spec.seed` and
    `build_multiset` is its first consumer, so the dispersion solve's trial
    builds (which reseed from `spec.seed` themselves) see the same draw the
    final multiset makes.
    """
    spec.validate()
    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(spec.seed)

    inner_rows = _inner_row_count(spec)
    stage_spec = spec if inner_rows == spec.rows else replace(spec, rows=inner_rows)

    warnings: list[str] = []
    solves: dict[str, SolveRecord] = {}

    # Stage 1: which values, how often. First consumer of rng.
    multiset = build_multiset(stage_spec, rng)
    warnings.extend(multiset.warnings)
    solves.update({k: SolveRecord.from_result(v) for k, v in multiset.solves.items()})

    # Stage 2: their order. The runs layout is exact by construction, so its
    # only outcome worth a record is whether the target sat inside the band.
    laid = layout_sequence(multiset.support, multiset.counts, stage_spec, rng)
    warnings.extend(laid.warnings)
    values = laid.values
    if laid.plan is not None:
        solves["runs_z"] = SolveRecord.from_status(laid.plan.status, laid.plan.message)

    # Bounded refinement of the inter-arrival CV. When it is
    # skipped -- runs_z is also targeted -- its `values` is None and the
    # combined layout from layout_sequence is the correct one to keep.
    if spec.interarrival_cv is not None:
        refined = refine_interarrival(
            multiset.support, multiset.counts, stage_spec, rng
        )
        if refined.skipped:
            warnings.append(f"skipped: interarrival_cv refinement: {refined.reason}")
        else:
            values = refined.values
            solves["interarrival_cv"] = SolveRecord.from_result(refined.solve)
            if not refined.solve.ok:
                warnings.append(
                    refined.solve.warning("interarrival_cv", spec.interarrival_cv)
                )

    # Measure on the final values, before wrapping.
    achieved = _measure(values, spec)
    targets = {
        name: float(getattr(spec, name))
        for name in _TARGETABLE
        if getattr(spec, name) is not None
    }
    within, tolerance_warnings = _check_tolerances(spec, targets, achieved)
    warnings.extend(tolerance_warnings)

    array = wrap(values, spec, rng)

    manifest = Manifest(
        dataset_id=spec.dataset_id,
        spec=spec,
        achieved=achieved,
        targets=targets,
        within_tolerance=within,
        solves=solves,
        warnings=warnings,
        clip_collapsed=multiset.clip_collapsed,
        round_collapsed=multiset.round_collapsed,
        generated_at=generated_at,
        generation_seconds=time.perf_counter() - started,
    )
    return Dataset(array=array, manifest=manifest)
