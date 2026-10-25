"""One monotone bisection, shared by every search in the generator.

Plan 1 produced three hand-rolled searches with three different status
vocabularies, and the findings require a clamp and an invalid bracket to be
distinguished. This defines that vocabulary once.

Callers: the entropy solve on the power-law exponent, the
index-of-dispersion solve on the distribution's spread, and the
inter-arrival CV refinement on the gamma's input CV.

The warning vocabulary lives here too. Every warning any generator module
emits is `"<prefix> <metric> ..."` with the prefix one of `WARNING_PREFIXES`
and the metric one of the five targetable names, so a harness can classify a
warning by its first two words without parsing the prose:

  clamp:      the target lies outside the reachable band and the generator
              honoured the nearest bound (a solve clamp, a runs-z band clamp,
              a support capacity clamp).
  miss:       the target was reachable in principle but the achieved value is
              off it (a tolerance miss, a solve that did not converge or had
              no valid bracket, a support shortfall from collapse).
  skipped:    a step was not run, so a target was not pursued (the CV
              refinement under a combined runs_z target, the CV tolerance
              under the settled combined-target behaviour).
  undefined:  the metric has no value for this data (runs_z on a single-group
              dichotomy, interarrival_cv with fewer than three repeat-runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

WARNING_PREFIXES = ("clamp:", "miss:", "skipped:", "undefined:")


class SolveStatus(Enum):
    OK = "ok"
    CLAMPED_LOW = "clamped_low"
    CLAMPED_HIGH = "clamped_high"
    INVALID_BRACKET = "invalid_bracket"
    NOT_CONVERGED = "not_converged"

    @property
    def warning_prefix(self) -> str:
        """The vocabulary word for a warning about a solve in this state: a
        clamp is `clamp:`, anything else that is not ok is `miss:`."""
        if self in (SolveStatus.CLAMPED_LOW, SolveStatus.CLAMPED_HIGH):
            return "clamp:"
        return "miss:"


@dataclass(frozen=True)
class SolveResult:
    x: float
    achieved: float
    status: SolveStatus
    bound: float | None
    message: str

    @property
    def ok(self) -> bool:
        return self.status is SolveStatus.OK

    def warning(self, metric: str, target: float) -> str:
        """The vocabulary warning for a solve that did not land: prefix by
        status, then the metric name, then the target and this message."""
        return (
            f"{self.status.warning_prefix} {metric} target {target:g} could "
            f"not be reached: {self.message}"
        )


def solve_monotone(
    f: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    *,
    increasing: bool,
    iters: int = 60,
    guard: Callable[[float], bool] | None = None,
    tol: float | None = None,
) -> SolveResult:
    """Find x in [lo, hi] with f(x) ~= target, assuming f is monotone there.

    increasing: whether f rises with x. Both directions are supported because the
    entropy solve is decreasing in its exponent while the dispersion solve is
    increasing in its spread.

    guard: optional predicate marking the valid branch. If it fails at `lo` the
    bracket is unusable and INVALID_BRACKET is returned rather than a clamp. If it
    fails at `hi`, the bracket is first narrowed by bisecting on the guard, so a
    value computed beyond a discontinuity is never mistaken for a reachable bound.
    Any bound found this way is reported as approximate, because its precision is
    set by the narrowing iteration count rather than by the function.

    tol: when set, the final |f(x) - target| must be within it or the result is
    NOT_CONVERGED rather than OK. A bisection cannot tell from its bracket that
    f is not monotone -- it just lands somewhere -- so this is the one check
    that stops a non-monotone objective being reported as solved. Clamps are
    unaffected: they are decided from the bracket ends before any search.
    """
    if not (target == target) or target in (float("inf"), float("-inf")):
        raise ValueError(f"target must be finite, got {target!r}")
    if not lo < hi:
        raise ValueError(f"need lo < hi, got lo={lo!r} hi={hi!r}")

    approximate = False
    if guard is not None:
        if not guard(lo):
            return SolveResult(
                lo,
                f(lo),
                SolveStatus.INVALID_BRACKET,
                None,
                f"invalid bracket: guard fails at lo={lo:.6g}",
            )
        if not guard(hi):
            g_lo, g_hi = lo, hi
            for _ in range(40):
                mid = (g_lo + g_hi) / 2
                if guard(mid):
                    g_lo = mid
                else:
                    g_hi = mid
            hi = g_lo
            approximate = True

    f_lo, f_hi = f(lo), f(hi)
    low_val, high_val = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    low_x, high_x = (lo, hi) if increasing else (hi, lo)
    suffix = " (approximate, set by the bracket narrowing)" if approximate else ""

    if target <= low_val:
        return SolveResult(
            low_x,
            low_val,
            SolveStatus.CLAMPED_LOW,
            low_val,
            f"clamped: min reachable {low_val:.6g}{suffix}",
        )
    if target >= high_val:
        return SolveResult(
            high_x,
            high_val,
            SolveStatus.CLAMPED_HIGH,
            high_val,
            f"clamped: max reachable {high_val:.6g}{suffix}",
        )

    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    achieved = f(x)
    if tol is not None and not abs(achieved - target) <= tol:
        return SolveResult(
            x,
            achieved,
            SolveStatus.NOT_CONVERGED,
            None,
            f"not converged: achieved {achieved:.6g} against target {target:.6g} "
            f"is outside tolerance {tol:.6g} after {iters} iterations; the "
            "objective is not monotone on the bracket",
        )
    return SolveResult(x, achieved, SolveStatus.OK, None, "ok")
