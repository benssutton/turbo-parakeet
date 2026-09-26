"""Number-of-distinct-values estimators behind Describe's cardinality conclusions.

Pure functions of frequency counts, shared by every implementation. Intervals are
95% (z = 1.96) and closed-form. Estimators are picked by rule, never averaged:
Chao1 is a lower bound and Lincoln–Petersen/Schnabel is biased low when some
values are far more common than others, so an average has no meaningful interval.
"""

from __future__ import annotations

import math
from typing import Sequence

Z = 1.96


def chao1(d: int, f1: int, f2: int) -> tuple[float, float, float]:
    """Bias-corrected Chao1 with Chao's (1987) log-normal interval; variance as in
    the EstimateS user guide. Returns (estimate, low, high)."""
    s = d + f1 * (f1 - 1) / (2 * (f2 + 1))
    t = s - d
    if t <= 0:
        return float(s), float(d), float(d)
    if f2 > 0:
        var = (
            f1 * (f1 - 1) / (2 * (f2 + 1))
            + f1 * (2 * f1 - 1) ** 2 / (4 * (f2 + 1) ** 2)
            + f1**2 * f2 * (f1 - 1) ** 2 / (4 * (f2 + 1) ** 4)
        )
    else:
        var = f1 * (f1 - 1) / 2 + f1 * (2 * f1 - 1) ** 2 / 4 - f1**4 / (4 * s)
    k = math.exp(Z * math.sqrt(math.log(1 + max(var, 0.0) / t**2)))
    return float(s), d + t / k, d + t * k


def schnabel(history: Sequence[int], d: int, n: int) -> tuple[float, float, float] | None:
    """Schnabel (multi-sample Lincoln–Petersen) over the three split subsets.

    history[k-1] = distinct values whose subset mask is k (bit i = seen in subset i).
    Occasions in subset order: catches C_t = |S_t|, marked M_2 = |S1|, M_3 = |S1 ∪ S2|,
    recaptures R_2 = |S1 ∩ S2|, R_3 = |S3 ∩ (S1 ∪ S2)|. Interval: Byar's closed-form
    Poisson limits on R = R_2 + R_3. None unless n > 0, d/n < 0.5 and R ≥ 1.
    """
    if n == 0 or d / n >= 0.5:
        return None
    h = lambda *masks: sum(history[m - 1] for m in masks)
    s1, s2, s3 = h(1, 3, 5, 7), h(2, 3, 6, 7), h(4, 5, 6, 7)
    union12 = h(1, 2, 3, 5, 6, 7)
    r = h(3, 7) + h(5, 6, 7)
    if r < 1:
        return None
    a = s2 * s1 + s3 * union12
    r_lo = r * (1 - 1 / (9 * r) - Z / (3 * math.sqrt(r))) ** 3
    r_hi = (r + 1) * (1 - 1 / (9 * (r + 1)) + Z / (3 * math.sqrt(r + 1))) ** 3
    return a / (r + 1), a / r_hi, a / r_lo


def duj1(d: int, f1: int, n: int, q: float) -> float:
    """Haas–Stokes Duj1: sample of n non-null values (sampling fraction q) → population NDV."""
    return 0.0 if n == 0 else d / (1 - (1 - q) * f1 / n)


def estimate(d: int, n: int, f1: int, f2: int, history: Sequence[int], q: float | None) -> dict:
    """Every estimate plus the one picked by rule: q == 1 → exact; q < 1 → Duj1;
    Schnabel valid → Schnabel; else Chao1. q = frame rows / population rows (None: unknown)."""
    c, c_lo, c_hi = chao1(d, f1, f2)
    sch = schnabel(history, d, n)
    s, s_lo, s_hi = sch if sch else (None, None, None)
    if q == 1.0:
        method, est, lo, hi = "exact", float(d), float(d), float(d)
    elif q is not None:
        method, est, lo, hi = "duj1", duj1(d, f1, n, q), None, None
    elif sch:
        method, est, lo, hi = "schnabel", s, s_lo, s_hi
    else:
        method, est, lo, hi = "chao1", c, c_lo, c_hi
    return {
        "unique": d == n and n > 0,
        "chao1": c, "chao1_low": c_lo, "chao1_high": c_hi,
        "schnabel": s, "schnabel_low": s_lo, "schnabel_high": s_hi,
        "est_cardinality": est, "est_method": method, "est_low": lo, "est_high": hi,
        "estimates_agree": None if sch is None else (c_lo <= s_hi and s_lo <= c_hi),
    }
