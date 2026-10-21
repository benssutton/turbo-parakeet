import math
from bisect import bisect_left
from collections import Counter
from fractions import Fraction

import polars as pl

from turbo_parakeet.techniques.delta.base import Delta

_BOUNDS = [1 << (8 * (k - 1) - 1) for k in range(2, 10)]  # sig > k-1 bytes iff |v| >= b


def _sig(v: int) -> int:
    return 1 + sum(v < -b or v >= b for b in _BOUNDS) if v else 0


def _sig_mean(x: list[int]) -> float:
    return sum(_sig(v) for v in x) / len(x) if x else math.nan


def _entropy(x: list[int]) -> float:
    if not x:
        return math.nan
    n = len(x)
    return -sum(c / n * math.log2(c / n) for c in Counter(x).values()) + 0.0


def _zero_fraction(x: list[int]) -> float:
    return x.count(0) / len(x) if x else math.nan


def _expected_bytes(values: list[int]) -> float:
    """Exact random-order mean sig(Δ) over ordered pairs of distinct positions, by bisect
    over the sorted values."""
    n = len(values)
    if n < 2:
        return math.nan
    v = sorted(values)
    pairs = n * (n - 1)
    equal = sum(c * (c - 1) for c in Counter(v).values())
    total = Fraction(pairs - equal, pairs)
    for b in _BOUNDS:
        inside = sum(bisect_left(v, x + b) - bisect_left(v, x - b) - 1 for x in v)
        total += Fraction(pairs - inside, pairs)
    return float(total)


def delta_statistics(s: pl.Series) -> dict:
    """The Delta metrics of one non-empty integer-backed column (Python ints throughout)."""
    x = s.drop_nulls().to_physical().cast(pl.Int128).to_list()
    d = [b - a for a, b in zip(x, x[1:])]
    dd = [b - a for a, b in zip(d, d[1:])]
    return {
        "n_rows": s.len(),
        "n_non_null": len(x),
        "n_deltas": len(d),
        "delta_min": min(d, default=None),
        "delta_max": max(d, default=None),
        "double_delta_min": min(dd, default=None),
        "double_delta_max": max(dd, default=None),
        "value_bytes_mean": _sig_mean(x),
        "delta_bytes_mean": _sig_mean(d),
        "double_delta_bytes_mean": _sig_mean(dd),
        "zero_delta_fraction": _zero_fraction(d),
        "zero_double_delta_fraction": _zero_fraction(dd),
        "delta_entropy": _entropy(d),
        "double_delta_entropy": _entropy(dd),
        "expected_delta_bytes_mean": _expected_bytes(x),
    }


class DeltaPolars(Delta):
    """Reference: Python-int differences of the physical values, sig by explicit byte
    bounds, entropy from a Counter, the baseline by bisect over the sorted values.
    Shares no code with the Rust kernel."""

    def _compute(self, frames, combos):
        rows = [delta_statistics(frames[f][c]) for ((f, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
