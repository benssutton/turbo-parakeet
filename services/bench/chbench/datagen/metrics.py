"""Metric definitions for the generator.

Every function takes a 1-D numpy array and returns a Python float. These are the
single arbiter of what a metric means: the solvers, the refinement pass, the
manifest and the tests all call these and never reimplement a formula.

Definitions are fixed by the design spec, section 2:
  - runs test: dichotomy is `x > median` for numeric, or membership of the
    most-frequent set covering half the rows for categorical.
  - inter-arrival: a run is a maximal stretch of identical values of length >= 2;
    gaps are measured between run starts; CV is std/mean.
  - float comparisons happen after rounding to `decimal_places`.

Degenerate inputs return nan rather than raising: a constant column and an empty
column are both legal DatasetSpecs, and the manifest measures every metric for
every dataset (spec section 2, best-effort).
"""

from __future__ import annotations

import numpy as np


def _round(x: np.ndarray, decimal_places: int | None) -> np.ndarray:
    """Apply the spec's float comparison rule. Integers pass through unchanged."""
    if decimal_places is None or not np.issubdtype(np.asarray(x).dtype, np.floating):
        return np.asarray(x)
    return np.round(x, decimal_places)


def entropy_bits(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    _, counts = np.unique(x, return_counts=True)
    return entropy_of_counts(counts)


def entropy_of_counts(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a count vector. Zero counts contribute nothing.

    entropy_bits delegates here so the stage 1 solver, which works in counts, and
    the manifest, which works in values, share one implementation.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def cardinality_ratio(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return len(np.unique(x)) / len(x)


def index_of_dispersion(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    mean = x.mean()
    if mean == 0.0:
        return float("nan")
    return float(x.var() / mean)


def dichotomise(x: np.ndarray, kind: str = "median") -> np.ndarray:
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
    if b.size == 0:
        return 0
    return 1 + int(np.count_nonzero(b[1:] != b[:-1]))


def runs_expected(n1: int, n2: int) -> tuple[float, float]:
    """Wald-Wolfowitz expected run count and standard deviation.

    A degenerate split (either group empty) has no variance, so sd is nan: with
    one group the run count is deterministically 1 and the z-score undefined.
    """
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return 1.0, float("nan")
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1))
    return mu, float(np.sqrt(var))


def runs_z(x: np.ndarray, kind: str = "median") -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    b = dichotomise(x, kind)
    n1 = int(b.sum())
    n2 = len(b) - n1
    mu, sd = runs_expected(n1, n2)
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float((run_count(b) - mu) / sd)


def runs_z_bounds(n1: int, n2: int) -> tuple[float, float]:
    """Reachable z range for a given split, computed from the measured n1, n2.

    Asymmetric: R = 2 is always reachable so the low bound is nearly independent
    of the split, while the ceiling (r_max - mu)/sd falls with min(n1, n2).
    Never assume a 50/50 split: the strict `x > median` dichotomy puts every tied
    value below, and skewed data produces splits as lopsided as 32/68.
    """
    mu, sd = runs_expected(n1, n2)
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan"), float("nan")
    r_max = 2 * min(n1, n2) + (0 if n1 == n2 else 1)
    return float((2 - mu) / sd), float((r_max - mu) / sd)


def repeat_runs(
    x: np.ndarray, decimal_places: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of every maximal run of identical values, length >= 2.

    Pass decimal_places for float columns: the spec compares floats after rounding,
    and sub-tolerance noise otherwise destroys every run.
    """
    x = _round(x, decimal_places)
    if x.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    change = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(x)])))
    keep = lengths >= 2
    return starts[keep], lengths[keep]


def interarrival_cv(x: np.ndarray, decimal_places: int | None = None) -> float:
    starts, _ = repeat_runs(x, decimal_places)
    if len(starts) < 3:
        return float("nan")
    gaps = np.diff(starts)
    return float(gaps.std() / gaps.mean())


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` integer units proportionally to `weights`, summing exactly.

    All-zero weights fall back to an even split rather than producing INT64_MIN:
    the default frequency profile is a probability density, which can underflow to
    all zeros for an extreme spread, and plausible-shaped nonsense is worse than
    a crash.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return np.array([], dtype=np.int64)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    total_w = w.sum()
    raw = np.full(w.size, total / w.size) if total_w <= 0 else w / total_w * total
    counts = np.floor(raw).astype(np.int64)
    short = int(total - counts.sum())
    if short > 0:
        idx = np.argsort(-(raw - counts), kind="stable")[:short]
        counts[idx] += 1
    return counts
