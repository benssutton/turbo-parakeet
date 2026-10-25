"""Spike 3: hit a target coefficient of variation of the gaps between runs of
identical values, alone and combined with a runs z-score target.

Method: values with count >= 2 form one run each; singles
fill the space between runs. Free-gap lengths are drawn from a gamma with
shape 1/CV^2 and rescaled to sum to the number of singles.

mode="free":  gamma on the free gaps only (naive).
mode="start": gamma on start-to-start gaps, subtract run length, clip at 0,
              rescale (accounts for run lengths inflating the gap mean).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from spikes._common import (
    dichotomise,
    interarrival_cv,
    largest_remainder,
    repeat_runs,
    runs_z,
)
from spikes.spike_runs import layout_runs


def float_multiset(
    n: int, ratio: float, rng: np.random.Generator, dp: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    k = int(round(ratio * n))
    support = np.unique(np.round(rng.uniform(-1, 10, size=k), dp))
    counts = np.ones(len(support), dtype=np.int64)
    np.add.at(counts, rng.integers(0, len(support), size=n - len(support)), 1)
    return support, counts


def layout_interarrival(
    support: np.ndarray,
    counts: np.ndarray,
    cv: float,
    rng: np.random.Generator,
    mode: str = "start",
) -> np.ndarray:
    is_run = counts >= 2
    run_vals, run_len = support[is_run], counts[is_run]
    singles = rng.permutation(support[~is_run])
    m, S = len(run_vals), len(singles)
    order = rng.permutation(m)
    run_vals, run_len = run_vals[order], run_len[order]
    shape = 1.0 / cv**2
    if mode == "free":
        free = rng.gamma(shape, 1.0, size=m + 1)
    else:
        mean_gap = (S + run_len[:-1].sum()) / max(m - 1, 1)
        gaps = rng.gamma(shape, mean_gap / shape, size=m)
        free = np.concatenate(([0.0], np.maximum(gaps - run_len, 0.0)))
        free[0] = free.mean()  # leading segment: an average-sized free gap
    free_int = largest_remainder(free + 1e-12, S)
    parts, pos = [], 0
    for j in range(m):
        parts.append(singles[pos : pos + free_int[j]])
        pos += free_int[j]
        parts.append(np.repeat(run_vals[j], run_len[j]))
    parts.append(singles[pos:])
    return np.concatenate(parts)


def multiset_of(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(x, return_counts=True)


def layout_combined(
    x: np.ndarray, z: float, cv: float, rng: np.random.Generator
) -> np.ndarray:
    """Runs block pattern decides which positions hold above/below-median values;
    each group is then laid out with the inter-arrival method inside its positions."""
    pattern = dichotomise(layout_runs(x, z, rng))
    b = dichotomise(x)
    out = np.empty_like(x)
    out[pattern] = layout_interarrival(*multiset_of(x[b]), cv, rng)
    out[~pattern] = layout_interarrival(*multiset_of(x[~b]), cv, rng)
    return out


def main() -> None:
    n = 100_000
    print("--- inter-arrival alone ---")
    print(
        f"{'mode':>5} {'target cv':>9} {'seed':>4} {'runs':>6} {'achieved cv':>11} {'runs_z':>8}"
    )
    for mode in ("free", "start"):
        for cv in (0.5, 1.0, 2.0):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                support, counts = float_multiset(n, 0.75, rng)
                y = layout_interarrival(support, counts, cv, rng, mode)
                assert np.array_equal(np.sort(y), np.repeat(support, counts))
                print(
                    f"{mode:>5} {cv:>9.2f} {seed:>4} {len(repeat_runs(y)[0]):>6} "
                    f"{interarrival_cv(y):>11.3f} {runs_z(y):>8.2f}"
                )

    print("--- combined with runs z ---")
    print(
        f"{'target z':>9} {'target cv':>9} {'seed':>4} {'runs':>6} {'achieved cv':>11} {'achieved z':>10}"
    )
    for z in (-0.25, -20.0):
        for cv in (0.5, 1.0, 2.0):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                support, counts = float_multiset(n, 0.75, rng)
                x = rng.permutation(np.repeat(support, counts))
                y = layout_combined(x, z, cv, rng)
                assert np.array_equal(np.sort(x), np.sort(y))
                print(
                    f"{z:>9.2f} {cv:>9.2f} {seed:>4} {len(repeat_runs(y)[0]):>6} "
                    f"{interarrival_cv(y):>11.3f} {runs_z(y):>10.2f}"
                )


if __name__ == "__main__":
    main()
