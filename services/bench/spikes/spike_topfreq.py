"""Spike 6: does the runs layout work for the top-frequency dichotomy?

Every earlier layout row used the median dichotomy on numeric data. The design
routes string and categorical columns through the top-frequency dichotomy
instead, which for skewed data gives a far more lopsided split. This measures
what that does to the achievable runs z-score.

Run from the repository root or from inside spikes/: this file puts the
repository root on sys.path itself, resolved from its own __file__ rather
than the current directory, so `chbench` (installed in editable mode) and
`spikes.spike_runs` both import correctly either way.
"""

from __future__ import annotations

import pathlib
import sys

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from synthdata.column.metrics import (
    cardinality_ratio,
    dichotomise,
    entropy_bits,
    largest_remainder,
    run_count,
    runs_expected,
    runs_z,
    runs_z_bounds,
)
from spikes.spike_runs import layout_runs, target_run_count


def categorical_multiset(
    n: int, k: int, s: float, rng: np.random.Generator
) -> np.ndarray:
    """k distinct string labels with a power-law frequency profile of exponent s.

    s = 0 is uniform (a near-balanced top-frequency split); larger s concentrates
    mass on a few labels and pushes the split lopsided.
    """
    labels = np.array([f"cat_{i:05d}" for i in range(k)])
    w = np.arange(1, k + 1, dtype=np.float64) ** (-s)
    counts = largest_remainder(w, n - k) + 1
    return rng.permutation(np.repeat(labels, counts))


def main() -> None:
    n = 200_000
    print(
        f"{'s':>5} {'k':>5} {'n1':>8} {'n2':>8} {'split%':>7} "
        f"{'z lo':>9} {'z hi':>9} {'target':>7} {'R want':>8} {'R got':>8} "
        f"{'achieved':>9} {'multiset':>9}"
    )
    for s in (0.0, 0.8, 1.2, 2.0):
        for k in (50, 1000):
            rng = np.random.default_rng(0)
            x = categorical_multiset(n, k, s, rng)
            b = dichotomise(x, kind="topfreq")
            n1, n2 = int(b.sum()), int((~b).sum())
            lo, hi = runs_z_bounds(n1, n2)
            for target in (-0.25, -20.0, 20.0):
                rng2 = np.random.default_rng(1)
                y = layout_runs(x, target, rng2, kind="topfreq")
                R_want = target_run_count(n1, n2, target)
                ok = (
                    np.array_equal(np.sort(x), np.sort(y))
                    and entropy_bits(x) == entropy_bits(y)
                    and cardinality_ratio(x) == cardinality_ratio(y)
                )
                print(
                    f"{s:>5.1f} {k:>5} {n1:>8} {n2:>8} {100*n1/n:>7.1f} "
                    f"{lo:>9.1f} {hi:>9.1f} {target:>7.2f} {R_want:>8} "
                    f"{run_count(dichotomise(y, kind='topfreq')):>8} "
                    f"{runs_z(y, kind='topfreq'):>9.3f} {str(ok):>9}"
                )

    print("\n--- ceiling as the split gets lopsided (n = 200 000) ---")
    print(f"{'top set %':>10} {'n1':>8} {'n2':>8} {'sd':>10} {'z lo':>10} {'z hi':>10}")
    for frac in (0.5, 0.6, 0.75, 0.9, 0.99):
        n1 = int(n * frac)
        lo, hi = runs_z_bounds(n1, n - n1)
        _, sd = runs_expected(n1, n - n1)
        print(
            f"{100*frac:>10.0f} {n1:>8} {n-n1:>8} {sd:>10.2f} {lo:>10.1f} {hi:>10.1f}"
        )


if __name__ == "__main__":
    main()
