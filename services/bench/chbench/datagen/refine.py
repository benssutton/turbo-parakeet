"""The bounded CV refinement pass.

Stage 2's inter-arrival layout (`sequence.layout_interarrival`) draws its
start-to-start gaps from a gamma parameterised by the *target* CV, but the
achieved CV always undershoots: 5.9 percent relative at CV 0.5, rising to
33.6 percent at CV 2.0 (spikes/FINDINGS.md; Task 9). A single multiplicative
rescaling cannot correct this -- the miss grows by a factor of six across the
target range for the start-to-start gap mode in use -- so this instead
bisects the gamma's *input* CV against the metric of the regenerated
sequence, exactly as `solve.solve_monotone` already does for the stage 1
entropy and dispersion solves. Task 9 measured achieved CV strictly monotone
in input CV, which is what makes the bisection converge.

`runs_z` needs no refinement (the construction hits the integer run count
exactly, leaving only sub-tolerance rounding noise, and no corrective move
exists). When both `runs_z` and `interarrival_cv` are targeted, the combined
distortion in `layout_sequence` is inflicted on the CV *after* the runs block
pattern is laid out, so bisecting the input CV against the combined output
cannot recover the target; this pass is skipped in that case rather than
spending its whole cap failing to converge.

Contract for the skipped case: `values` is `None`, not a plausible-looking
but wrong array. This module only consumes `layout_interarrival`, never
`layout_sequence`, so it has no way to produce the combined runs+CV layout
that honours `runs_z`; the pipeline caller already holds that correct
layout from `layout_sequence` and must keep using it. Returning a same-
shaped array here would let a caller that forgets to check `skipped`
silently overwrite a correct layout with one that violates the runs target.
`None` instead fails loudly -- an `AttributeError` or a `TypeError` -- at
the first downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chbench.datagen.metrics import interarrival_cv
from chbench.datagen.sequence import MIN_REPEAT_RUNS, layout_interarrival
from chbench.datagen.solve import SolveResult, SolveStatus, solve_monotone
from chbench.datagen.spec import DatasetSpec

INPUT_CV_LO = 0.05
INPUT_CV_HI = 10.0


@dataclass(frozen=True)
class RefineResult:
    values: np.ndarray | None
    input_cv: float
    achieved: float
    solve: SolveResult
    skipped: bool
    reason: str


def refine_interarrival(
    support: np.ndarray,
    counts: np.ndarray,
    spec: DatasetSpec,
    rng: np.random.Generator,
    cap: int = 50,
    _on_evaluate=None,
) -> RefineResult:
    """Bisect the gamma's input CV so the regenerated sequence's achieved CV
    lands on `spec.interarrival_cv`.

    Each trial lays out with a fresh generator seeded identically (derived
    once from `rng`, before the search starts), so the objective is a
    deterministic function of the input CV and the whole pass is itself
    deterministic for a given caller state. The final `values` is the layout
    produced at the solved input CV using that same per-trial seeding, so
    `achieved` describes exactly the array returned.

    When `spec.runs_z` is also set, the pass is skipped: `skipped` is True,
    `reason` names `runs_z`, `values` is `None`, `achieved` is `nan`, and
    `input_cv` echoes the untouched target. The caller must not read
    `values` without checking `skipped` first -- it already holds the
    correct combined layout from `layout_sequence` and must keep using it.
    The pass is skipped the same way when the multiset forms fewer than
    `MIN_REPEAT_RUNS` repeat-runs, because the metric is then undefined at
    every input CV and there is nothing to bisect against.
    """
    support, counts = np.asarray(support), np.asarray(counts)
    target = spec.interarrival_cv

    if spec.runs_z is not None:
        reason = (
            "runs_z is also targeted: the combined distortion is inflicted "
            "after the gaps are laid out, so bisecting the input CV against "
            "the combined output cannot recover the target"
        )
        return RefineResult(
            values=None,
            input_cv=target,
            achieved=float("nan"),
            solve=SolveResult(
                target,
                float("nan"),
                SolveStatus.OK,
                None,
                "skipped: runs_z also targeted",
            ),
            skipped=True,
            reason=reason,
        )

    repeat_runs = int(np.count_nonzero(counts >= 2))
    if repeat_runs < MIN_REPEAT_RUNS:
        reason = (
            f"interarrival_cv is undefined for this data: it needs at least "
            f"{MIN_REPEAT_RUNS} repeat-runs to have two gaps and the multiset "
            f"forms only {repeat_runs} (values with count >= 2), so every "
            "trial layout would measure nan"
        )
        return RefineResult(
            values=None,
            input_cv=target,
            achieved=float("nan"),
            solve=SolveResult(
                target, float("nan"), SolveStatus.OK, None, "skipped: metric undefined"
            ),
            skipped=True,
            reason=reason,
        )

    trial_seed = int(rng.integers(0, 2**63 - 1))

    def layout_at(input_cv: float) -> np.ndarray:
        return layout_interarrival(
            support, counts, input_cv, np.random.default_rng(trial_seed)
        )

    def objective(input_cv: float) -> float:
        if _on_evaluate is not None:
            _on_evaluate(input_cv)
        return interarrival_cv(layout_at(input_cv), decimal_places=spec.decimal_places)

    solve = solve_monotone(
        objective,
        target,
        INPUT_CV_LO,
        INPUT_CV_HI,
        increasing=True,
        iters=cap,
        tol=spec.tolerances.interarrival_cv_relative * abs(target),
    )
    values = layout_at(solve.x)
    achieved = interarrival_cv(values, decimal_places=spec.decimal_places)

    return RefineResult(
        values=values,
        input_cv=solve.x,
        achieved=achieved,
        solve=solve,
        skipped=False,
        reason="",
    )
