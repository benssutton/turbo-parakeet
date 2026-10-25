"""Spike 1: can we hit a target Shannon entropy (bits) for a given cardinality?

Method: every one of the k support values gets 1 row, the
remaining n-k rows are allocated by a power-law profile f_i ~ rank_i^-s using
largest-remainder rounding. Entropy is monotone decreasing in s, so bisect.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from spikes._common import cardinality_ratio, entropy_bits, largest_remainder


def powerlaw_counts(n: int, k: int, s: float) -> np.ndarray:
    w = np.arange(1, k + 1, dtype=np.float64) ** (-s)
    return largest_remainder(w, n - k) + 1


def entropy_of_counts(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def solve_exponent(
    n: int, k: int, target: float, lo: float = 0.0, hi: float = 30.0, iters: int = 60
) -> tuple[float, float, str]:
    h_lo = entropy_of_counts(powerlaw_counts(n, k, lo))  # max entropy (uniform)
    h_hi = entropy_of_counts(powerlaw_counts(n, k, hi))  # min entropy (one heavy value)
    if target >= h_lo:
        return lo, h_lo, f"clamped: max reachable {h_lo:.3f}"
    if target <= h_hi:
        return hi, h_hi, f"clamped: min reachable {h_hi:.3f}"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if entropy_of_counts(powerlaw_counts(n, k, mid)) > target:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return s, entropy_of_counts(powerlaw_counts(n, k, s)), "ok"


def expand(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = rng.choice(9_000_000, size=len(counts), replace=False) + 1_000_000
    return rng.permutation(np.repeat(values, counts))


def main() -> None:
    n = 100_000
    print(
        f"{'k':>6} {'target':>7} {'s':>8} {'achieved':>9} {'card':>7} {'ms':>6}  status"
    )
    for k in (1_000, 10_000):
        for target in (0.5, 2.0, 4.0, 8.0, 12.0):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                t0 = time.perf_counter()
                s, _, status = solve_exponent(n, k, target)
                x = expand(powerlaw_counts(n, k, s), rng)
                ms = (time.perf_counter() - t0) * 1000
                print(
                    f"{k:>6} {target:>7.2f} {s:>8.3f} {entropy_bits(x):>9.3f} "
                    f"{cardinality_ratio(x):>7.4f} {ms:>6.0f}  {status}"
                )


if __name__ == "__main__":
    main()
