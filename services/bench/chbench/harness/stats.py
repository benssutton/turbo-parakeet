"""Distribution-free summaries of repeated timings: the median with an order-statistic confidence
interval, and the median of paired per-round ratios with the same interval. Metrics only: whether
an interval excludes 1 is the caller's decision.

The interval's bounds are the order statistics at ranks k and n - k + 1, k the largest rank with
P(Binomial(n, 1/2) < k) <= (1 - level) / 2. It holds whatever the timing distribution, and its
exact coverage (`exact_coverage`) is at least `level`; below six values there is no 95% interval.
"""

from __future__ import annotations

import math

import numpy as np


def _rank(n: int, level: float) -> int:
    """Largest k with P(Binomial(n, 1/2) < k) <= (1 - level) / 2; 0 when no interval exists."""
    alpha = (1.0 - level) / 2.0
    k, cdf = 0, 0.0
    while k < n // 2:
        nxt = cdf + math.comb(n, k) / 2**n
        if nxt > alpha:
            break
        cdf, k = nxt, k + 1
    return k


def exact_coverage(n: int, level: float = 0.95) -> float | None:
    """The interval's exact coverage for n values, 1 - 2 P(B(n, 1/2) < k); None without one."""
    k = _rank(n, level)
    if k == 0:
        return None
    return 1.0 - 2.0 * sum(math.comb(n, i) for i in range(k)) / 2**n


def median_ci(values, level: float = 0.95) -> tuple:
    """(median, lo, hi); None and NaN are dropped; (None, None, None) when nothing is left."""
    v = np.array([x for x in values if x is not None], dtype=float)
    v = np.sort(v[~np.isnan(v)])
    if v.size == 0:
        return None, None, None
    med = float(np.median(v))
    k = _rank(v.size, level)
    if k == 0:
        return med, None, None
    return med, float(v[k - 1]), float(v[v.size - k])


def _positive(x) -> bool:
    return x is not None and x == x and x > 0


def paired_ratio(a: dict, b: dict, level: float = 0.95) -> tuple:
    """(ratio, lo, hi, n_rounds): `median_ci` of a[r] / b[r] over the rounds r both hold with a
    positive value. Pairing cancels what the two shared in that round (load, drift)."""
    rounds = sorted(
        r for r in a.keys() & b.keys() if _positive(a[r]) and _positive(b[r])
    )
    ratio, lo, hi = median_ci([a[r] / b[r] for r in rounds], level)
    return ratio, lo, hi, len(rounds)
