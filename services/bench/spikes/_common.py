"""Metric functions shared by the feasibility spikes.

These are the definitions from the design spec (section 2, "Run definitions")
and are the seed of chbench/datagen/metrics.py.
"""
from __future__ import annotations

import numpy as np


def entropy_bits(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def cardinality_ratio(x: np.ndarray) -> float:
    return len(np.unique(x)) / len(x)


def index_of_dispersion(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(x.var() / x.mean())


def dichotomise(x: np.ndarray, kind: str = "median") -> np.ndarray:
    """Boolean labels for the runs test.

    kind="median":  value > median (numeric, date, time).
    kind="topfreq": value is in the most-frequent set covering >= half the rows.
    """
    x = np.asarray(x)
    if kind == "median":
        return x > np.median(x)
    if kind == "topfreq":
        vals, counts = np.unique(x, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        cum = np.cumsum(counts[order])
        n_top = int(np.searchsorted(cum, len(x) / 2, side="left")) + 1
        return np.isin(x, vals[order][:n_top])
    raise ValueError(f"unknown dichotomy kind {kind!r}")


def run_count(b: np.ndarray) -> int:
    b = np.asarray(b, dtype=bool)
    return 1 + int(np.count_nonzero(b[1:] != b[:-1]))


def runs_expected(n1: int, n2: int) -> tuple[float, float]:
    """Wald-Wolfowitz expected run count and its standard deviation."""
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1))
    return mu, float(np.sqrt(var))


def runs_z(x: np.ndarray, kind: str = "median") -> float:
    b = dichotomise(x, kind)
    n1 = int(b.sum())
    n2 = len(b) - n1
    mu, sd = runs_expected(n1, n2)
    return float((run_count(b) - mu) / sd)


def repeat_runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of every maximal run of identical values, length >= 2."""
    x = np.asarray(x)
    change = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(x)])))
    keep = lengths >= 2
    return starts[keep], lengths[keep]


def interarrival_cv(x: np.ndarray) -> float:
    starts, _ = repeat_runs(x)
    if len(starts) < 3:
        return float("nan")
    gaps = np.diff(starts)
    return float(gaps.std() / gaps.mean())


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` integer units proportionally to `weights`, summing exactly."""
    w = np.asarray(weights, dtype=np.float64)
    raw = w / w.sum() * total
    counts = np.floor(raw).astype(np.int64)
    short = int(total - counts.sum())
    if short > 0:
        idx = np.argsort(-(raw - counts), kind="stable")[:short]
        counts[idx] += 1
    return counts
