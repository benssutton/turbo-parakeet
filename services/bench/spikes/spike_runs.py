"""Spike 2: hit a target Wald-Wolfowitz runs z-score by block layout.

Method (spec 4.3 step 1): dichotomise the multiset, invert the runs-test
mean/sd for the group sizes to get a target run count R, then lay out R
alternating blocks whose lengths are random compositions of n1 and n2.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from spikes._common import dichotomise, entropy_bits, run_count, runs_expected, runs_z


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


def make_multiset(n: int, rng: np.random.Generator) -> np.ndarray:
    """Integers in [1e6, 1e7) with ~1000 distinct values, skewed frequencies."""
    support = rng.choice(np.arange(1_000_000, 10_000_000), size=1_000, replace=False)
    weights = np.arange(1, 1_001, dtype=np.float64) ** -1.2
    return rng.choice(support, size=n, p=weights / weights.sum())


def main() -> None:
    n = 100_000
    print(
        f"{'target z':>9} {'seed':>4} {'R target':>9} {'R got':>7} {'achieved z':>11} {'multiset ok':>11}"
    )
    for z in (-0.25, -5.0, 3.0, -50.0, 50.0, -400.0):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            x = make_multiset(n, rng)
            b = dichotomise(x)
            R_target = target_run_count(int(b.sum()), int((~b).sum()), z)
            y = layout_runs(x, z, rng)
            ok = np.array_equal(np.sort(x), np.sort(y)) and entropy_bits(
                x
            ) == entropy_bits(y)
            print(
                f"{z:>9.2f} {seed:>4} {R_target:>9} {run_count(dichotomise(y)):>7} "
                f"{runs_z(y):>11.3f} {str(ok):>11}"
            )


if __name__ == "__main__":
    main()
