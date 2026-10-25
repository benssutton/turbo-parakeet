"""Stage 2: ordering the multiset.

Stage one (multiset.py) decides which values appear and how often. Stage two
decides their order. It is only allowed to permute: every stage one metric
(entropy, cardinality ratio, ...) must survive exactly. This module owns both
order metrics -- the Wald-Wolfowitz runs z-score (2a) and the coefficient of
variation of the gaps between repeat-runs (2b) -- and the dispatcher
`layout_sequence` that picks a layout from whichever targets a spec sets.

Runs method (spec 4.3 step 1): dichotomise the multiset, invert the runs-test
mean/sd for the group sizes to get a target run count R, then lay out R
alternating blocks whose lengths are random compositions of n1 and n2.

Inter-arrival method (spec 4.3 step 2): every value with count >= 2 forms one
run; singles fill the segments between runs. Start-to-start gaps are drawn from
a gamma of shape 1/CV^2, the preceding run's own length is subtracted and the
remainder clipped at zero, and the singles are allocated across the segments
by largest remainder. See `layout_interarrival` for the two structural floors.

Combined targets (settled by the project owner, spec 4.3/4.4): lay out once,
runs pattern first, and record the achieved CV without its tolerance. See
`layout_sequence`.

`target_run_count`, `composition` and `layout_runs` are lifted verbatim from
the phase one spike (`spikes/spike_runs.py`), which verified by construction
that the block-count swap below (needed when the starting group cannot supply
enough blocks) is necessary and correct, and that the layout preserves the
multiset exactly. Do not simplify that swap.

`layout_runs` never inspects which dichotomy produced its boolean mask -- it
only permutes and reslices `x[b]` and `x[~b]` -- so it is dichotomy-agnostic
(spikes/FINDINGS.md section 7). What differs by dichotomy is the *reachable
band*: a skewed categorical's `topfreq` split can have a far lower positive
ceiling than a balanced numeric split, so `runs_plan` reports the band from
the measured `n1, n2` of the actual column rather than assuming a 50/50 split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from chbench.datagen.metrics import (
    dichotomise,
    largest_remainder,
    runs_expected,
    runs_z_bounds,
)
from chbench.datagen.solve import SolveStatus
from chbench.datagen.spec import DatasetSpec

# interarrival_cv is the CV of the gaps between repeat-run starts; two gaps need
# three runs, so below this the metric is undefined (design 4.3 step 2: "if
# there are not enough duplicates to form runs, warn").
MIN_REPEAT_RUNS = 3


def target_run_count(n1: int, n2: int, z: float) -> int:
    mu, sd = runs_expected(n1, n2)
    r_max = 2 * min(n1, n2) + (0 if n1 == n2 else 1)
    return int(np.clip(round(mu + z * sd), 2, r_max))


def composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Split `total` into `parts` positive integers, uniformly at random."""
    if parts == 1:
        return np.array([total], dtype=np.int64)
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total])))


def layout_runs(
    x: np.ndarray, z: float, rng: np.random.Generator, kind: str = "median"
) -> np.ndarray:
    x = np.asarray(x)
    if z is None:
        return rng.permutation(x)
    b = dichotomise(x, kind)
    a_vals, b_vals = rng.permutation(x[b]), rng.permutation(x[~b])
    n1, n2 = len(a_vals), len(b_vals)
    R = target_run_count(n1, n2, z)
    ra, rb = (R + 1) // 2, R // 2  # the starting group gets the extra block
    start_with_a = True
    if ra > n1:  # A cannot supply that many blocks
        ra, rb, start_with_a = rb, ra, False
    la, lb = composition(n1, ra, rng), composition(n2, rb, rng)
    pools = (
        [(a_vals, la), (b_vals, lb)] if start_with_a else [(b_vals, lb), (a_vals, la)]
    )
    out, pos = [], [0, 0]
    for i in range(R):
        g = i % 2
        vals, lens = pools[g]
        length = int(lens[i // 2])
        out.append(vals[pos[g] : pos[g] + length])
        pos[g] += length
    return np.concatenate(out)


@dataclass(frozen=True)
class RunsPlan:
    """The reachable runs-z band for one column's measured split.

    Everything here is computed from the *measured* n1, n2 -- never an
    assumed even split -- so the caller can record a clamp rather than
    silently accepting one when the requested `z` falls outside [z_lo, z_hi].

    `status` uses the solve vocabulary so the manifest can record the runs
    layout under `solves["runs_z"]` exactly like a bisection: CLAMPED_LOW
    below the band, CLAMPED_HIGH above it, OK inside. `message` is the
    matching status message.
    """

    n1: int
    n2: int
    mu: float
    sd: float
    r_max: int
    z_lo: float
    z_hi: float
    target_R: int
    clamped: bool
    z: float
    status: SolveStatus

    @property
    def message(self) -> str:
        if self.status is SolveStatus.OK:
            return "ok"
        bound = self.z_lo if self.status is SolveStatus.CLAMPED_LOW else self.z_hi
        side = "min" if self.status is SolveStatus.CLAMPED_LOW else "max"
        return (
            f"clamped: {side} reachable {bound:.6g} for the measured split "
            f"n1={self.n1}, n2={self.n2}; laid out {self.target_R} runs"
        )


def runs_plan(x: np.ndarray, z: float, kind: str) -> RunsPlan:
    x = np.asarray(x)
    b = dichotomise(x, kind)
    n1, n2 = int(b.sum()), int((~b).sum())
    mu, sd = runs_expected(n1, n2)
    r_max = 2 * min(n1, n2) + (0 if n1 == n2 else 1)
    z_lo, z_hi = runs_z_bounds(n1, n2)
    if n1 == 0 or n2 == 0:
        # Single-group split: the run count is deterministically 1, the band is
        # undefined (nan) and there is nothing to clamp. `layout_sequence` reads
        # n1, n2 off the plan to drop the runs target with a warning.
        target_R, status = 1, SolveStatus.OK
    else:
        target_R = target_run_count(n1, n2, z)
        if z < z_lo:
            status = SolveStatus.CLAMPED_LOW
        elif z > z_hi:
            status = SolveStatus.CLAMPED_HIGH
        else:
            status = SolveStatus.OK
    return RunsPlan(
        n1=n1,
        n2=n2,
        mu=mu,
        sd=sd,
        r_max=r_max,
        z_lo=z_lo,
        z_hi=z_hi,
        target_R=target_R,
        clamped=status is not SolveStatus.OK,
        z=z,
        status=status,
    )


def dichotomy_kind_for(spec: DatasetSpec) -> str:
    """ "topfreq" for string, categorical and boolean specs, "median" otherwise.

    Categorical is identified by `spec.categories`, not by arrow_type alone:
    a numeric column can also carry a fixed support set. Unwrap list<T> to
    its element type first -- array columns dichotomise on their elements.
    """
    dtype = spec.arrow_dtype
    if pa.types.is_list(dtype):
        dtype = dtype.value_type
    if (
        spec.categories is not None
        or pa.types.is_string(dtype)
        or pa.types.is_boolean(dtype)
    ):
        return "topfreq"
    return "median"


def layout_interarrival(
    support: np.ndarray, counts: np.ndarray, cv: float, rng: np.random.Generator
) -> np.ndarray:
    """Order the multiset so the gaps between repeat-runs have roughly the CV `cv`.

    Lifted from `spikes/spike_interarrival.py` keeping only the start-to-start
    gap mode (5.9 to 33.6 percent relative error); the free-gap mode measured
    42 to 45 percent and is rejected. The draw undershoots: Task 10's refine
    pass bisects on the input CV, which is why the achieved CV must be
    monotone in it.

    Two floors are structural, not bugs. The mean gap is fixed by the multiset
    at rows / runs, and every gap is an integer >= 2, so a CV near zero is
    unreachable by construction (alternating gaps of 4 and 5 already give
    about 0.11).

    Segments: `m` runs leave `m + 1` segments for the singles. The `m - 1`
    interior segments come from start-to-start gaps with the preceding run's
    length subtracted. The leading and trailing segments have no preceding
    run to subtract, so they take their own draws from the same gamma with
    no subtraction. (The spike drew one gap too many and reused it for the
    trailing segment, subtracting the last run's length for a segment no run
    follows, and averaged a vector still holding a zero placeholder for the
    leading one; this handles both ends explicitly instead.)

    The assembly (from `free_int` on) was a Python loop appending one
    `np.repeat` and two slices per repeat-run; replaced per FINDINGS §8 with
    the numpy-only scatter from `spikes/spike_vectorise_interarrival.py`,
    verified there to produce the byte-identical array 11.5x faster.
    """
    support, counts = np.asarray(support), np.asarray(counts)
    is_run = counts >= 2
    run_vals, run_len = support[is_run], counts[is_run]
    singles = rng.permutation(support[~is_run])
    m, S = len(run_vals), len(singles)
    if m == 0:  # nothing to gap: a plain shuffle
        return singles
    order = rng.permutation(m)
    run_vals, run_len = run_vals[order], run_len[order]
    shape = 1.0 / cv**2
    # Expected start-to-start gap: the singles spread over m + 1 segments plus
    # the run the gap starts with. Only its ratio to `run_len` matters for the
    # clipping; largest_remainder rescales the weights to exactly S singles.
    mean_gap = S / (m + 1) + (run_len[:-1].mean() if m > 1 else 0.0)
    draws = rng.gamma(shape, mean_gap / shape, size=m + 1)
    interior = np.maximum(draws[: m - 1] - run_len[:-1], 0.0)
    free = np.concatenate(([draws[m - 1]], interior, [draws[m]]))
    free_int = largest_remainder(free, S)

    run_before = np.concatenate(([0], np.cumsum(run_len[:-1])))
    run_start = np.cumsum(free_int[:m]) + run_before
    total_run = int(run_len.sum())
    n = S + total_run
    within = np.arange(total_run) - np.repeat(run_before, run_len)
    run_pos = np.repeat(run_start, run_len) + within

    out = np.empty(n, dtype=support.dtype)
    is_run_pos = np.zeros(n, dtype=bool)
    is_run_pos[run_pos] = True
    out[run_pos] = np.repeat(run_vals, run_len)
    out[~is_run_pos] = singles
    return out


@dataclass(frozen=True)
class SequenceResult:
    """`plan` is the runs plan the layout followed, so the caller can record
    its status (a clamp, or ok) under the manifest's solves; None when no
    runs_z was targeted or the target was dropped as undefined."""

    values: np.ndarray
    warnings: list[str]
    combined: bool
    plan: RunsPlan | None = None


def _combined_warnings(plan: RunsPlan, counts: np.ndarray) -> list[str]:
    """The settled combined-target warnings.

    The runs block pattern is fixed before any CV randomness is drawn, so the
    runs target survives intact; the cost lands entirely on the CV. Filling
    the scattered block positions with one contiguous inter-arrival sequence
    per group splits a repeat-run wherever a block boundary falls inside one,
    so the distortion is governed by mean block length (rows / R) against the
    multiset's mean gap between runs (rows / runs). Near-random z gives blocks
    of about 2 rows, already below the roughly 4.7-row mean gap at
    cardinality 0.75: 42 to 46 percent of runs are destroyed and the achieved
    CV is pulled toward 1 from whichever side the target sat (measured 0.5 to
    0.76 and 2.0 to 1.10). That is the ordinary case. It worsens as z rises
    through zero and only eases once z is negative enough for blocks to
    outgrow the mean gap.
    """
    n = int(counts.sum())
    runs = int(np.count_nonzero(counts >= 2))
    block = n / plan.target_R
    gap = n / runs if runs else float("inf")
    out = [
        "skipped: interarrival_cv tolerance, because runs_z and "
        "interarrival_cv trade off: the layout honours runs_z and records "
        "the achieved interarrival_cv without applying its tolerance (mean "
        f"block length {block:.2f} rows against a mean run gap of {gap:.2f} "
        "rows)"
    ]
    if block < gap:
        out.append(
            f"miss: interarrival_cv is pulled strongly toward 1, because the "
            f"mean block length {block:.2f} rows (rows / {plan.target_R} "
            f"runs-test runs) is below the mean run gap {gap:.2f} rows (rows "
            f"/ {runs} repeat-runs), so block boundaries split most "
            "repeat-runs; this eases only when runs_z is negative enough for "
            "blocks to outgrow the mean run gap"
        )
    return out


def layout_sequence(
    support: np.ndarray,
    counts: np.ndarray,
    spec: DatasetSpec,
    rng: np.random.Generator,
    *,
    cv_override: float | None = None,
) -> SequenceResult:
    """Dispatch on which ordering targets the spec sets.

    Neither: a uniform shuffle. `runs_z` only: `layout_runs`. `interarrival_cv`
    only: `layout_interarrival`. Both: the combined path -- the runs block
    pattern decides which positions hold each dichotomy group, and each group
    is then laid out with the inter-arrival method inside its positions, once,
    with no retry. `cv_override` replaces the spec's CV target so Task 10's
    refine pass can re-lay with a trial value.

    A single-group dichotomy (a constant column, or any multiset whose split
    leaves one side empty) has no runs standard deviation, so a runs target
    cannot apply; it is dropped with a warning and the remaining targets
    decide the layout. The manifest measures every metric on every dataset,
    so this must not raise.

    A `runs_z` outside the band reachable from the measured split is clamped
    by `layout_runs` (design 4.3); this is the package's first runs-z clamp
    report (spikes/FINDINGS.md section 7), so the clamp is recorded here as
    a warning naming the requested z and the band, never silently, and the
    plan is returned so the caller can record the clamp under its solves.
    """
    support, counts = np.asarray(support), np.asarray(counts)
    x = np.repeat(support, counts)
    kind = dichotomy_kind_for(spec)
    z = spec.runs_z
    cv = spec.interarrival_cv if cv_override is None else cv_override
    warnings: list[str] = []
    plan = None

    if z is not None:
        plan = runs_plan(x, z, kind)
        if plan.n1 == 0 or plan.n2 == 0:
            warnings.append(
                "undefined: runs_z has no standard deviation for a single-group "
                f"dichotomy (kind={kind!r}, all {x.size} rows fall on one "
                "side), so the metric is undefined for this data; ignoring "
                "the target"
            )
            z, plan = None, None
        elif plan.clamped:
            warnings.append(
                f"clamp: runs_z={z:g} is outside the reachable band "
                f"[{plan.z_lo:.2f}, {plan.z_hi:.2f}] for the measured split "
                f"n1={plan.n1}, n2={plan.n2}; the target has been clamped to "
                f"{plan.target_R} runs"
            )

    if cv is not None:
        repeat_runs = int(np.count_nonzero(counts >= 2))
        if repeat_runs < MIN_REPEAT_RUNS:
            warnings.append(
                f"undefined: interarrival_cv needs at least {MIN_REPEAT_RUNS} "
                f"repeat-runs to have two gaps, and this multiset forms only "
                f"{repeat_runs} repeat-runs (values with count >= 2), so the "
                "metric is undefined for this data; laying out anyway"
            )

    if z is None and cv is None:
        return SequenceResult(rng.permutation(x), warnings, False)
    if cv is None:
        return SequenceResult(layout_runs(x, z, rng, kind), warnings, False, plan)
    if z is None:
        return SequenceResult(
            layout_interarrival(support, counts, cv, rng), warnings, False
        )

    pattern = dichotomise(layout_runs(x, z, rng, kind), kind)
    b = dichotomise(x, kind)
    out = np.empty_like(x)
    out[pattern] = layout_interarrival(*np.unique(x[b], return_counts=True), cv, rng)
    out[~pattern] = layout_interarrival(*np.unique(x[~b], return_counts=True), cv, rng)
    warnings.extend(_combined_warnings(plan, counts))
    return SequenceResult(out, warnings, True, plan)
