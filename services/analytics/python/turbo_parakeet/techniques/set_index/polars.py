import math
from collections import Counter

import numpy as np
import polars as pl

from turbo_parakeet.techniques._skip_index import (
    avoid,
    common_metrics,
    phi,
    ranked,
    windows,
)
from turbo_parakeet.techniques.set_index.base import SetIndex


def intervals(key: pl.Series, w: int, k: int) -> pl.DataFrame:
    """Per id, the merged ranges [lo, hi] of window starts whose window holds it: an
    occurrence at row r is in the windows starting at max(0, r − w + 1) … min(r, k − 1).
    """
    return (
        pl.DataFrame({"id": key, "r": pl.int_range(key.len(), eager=True)})
        .with_columns(
            lo=(pl.col("r") - (w - 1)).clip(lower_bound=0),
            hi=pl.col("r").clip(upper_bound=k - 1),
        )
        .sort("id", "r")
        .with_columns(
            new=(pl.col("lo") > pl.col("hi").shift(1).over("id") + 1).fill_null(True)
        )
        .group_by(pl.col("new").cum_sum().alias("interval"))
        .agg(pl.col("id").first(), pl.col("lo").min(), pl.col("hi").max())
    )


def per_window(iv: pl.DataFrame, weight: np.ndarray, k: int) -> np.ndarray:
    """Σ weight over the intervals covering each window start (difference array)."""
    diff = np.zeros(k + 1, dtype=np.int64)
    np.add.at(diff, iv["lo"].to_numpy(), weight)
    np.add.at(diff, iv["hi"].to_numpy() + 1, -weight)
    return np.cumsum(diff[:k])


def std_present(entries: Counter, n: int, w: int) -> float:
    """The std of a window's entry count, as `std_present` in set.rs."""
    q = {c: avoid(n, w, c) for c in entries}
    var = sum(m * q[c] * (1 - q[c]) for c, m in entries.items())
    for c, m in entries.items():
        for c2, m2 in entries.items():
            pairs = m * (m2 - (c == c2))
            if pairs:
                var += pairs * (avoid(n, w, c + c2) - q[c] * q[c2])
    return math.sqrt(max(var, 0.0))


def set_baseline(
    counts: np.ndarray, n_null: int, n: int, w: int, max_size: int
) -> tuple[float, float, float, float]:
    """(expected_skip_rows, expected_skip_values, expected_window_distinct,
    expected_overflow_fraction) under a random row order, over count classes.
    """
    values = Counter(counts.tolist())
    entries = Counter(counts.tolist() + ([n_null] if n_null else []))

    def mean_present(rows: int, without: int | None = None) -> float:
        return sum(
            (m - (c == without)) * (1 - avoid(rows, w, c))
            for c, m in entries.items()
            if m - (c == without)
        )

    mu = mean_present(n)
    sigma = std_present(entries, n, w) if max_size else 0.0

    def fits(mean: float) -> float:
        if not max_size:
            return 1.0
        x = max_size + 0.5 - mean
        return phi(x / sigma) if sigma > 0 else float(x >= 0)

    rows = vals = 0.0
    for c, m in values.items():
        absent = avoid(n, w, c)
        p = (
            absent
            if not max_size or absent == 0
            else absent * fits(mean_present(n - c, c))
        )
        rows += c * m * p
        vals += m * p
    nn, d = int(counts.sum()), int(counts.size)
    return (
        rows / nn if nn else math.nan,
        vals / d if d else math.nan,
        mu,
        1 - fits(mu),
    )


def set_statistics(s: pl.Series, window_rows: int, max_size: int) -> dict:
    """The SetIndex metrics of one non-empty column."""
    ids, counts = ranked(s)
    n, d, nn = s.len(), int(counts.size), int(counts.sum())
    w, k = windows(n, window_rows)
    iv = intervals(ids.fill_null(d), w, k)  # NULL is id d
    which = iv["id"].to_numpy().astype(np.int64)
    value = which < d
    present = per_window(iv, value.astype(np.int64), k)
    mass = per_window(iv, np.append(counts, 0)[which], k)
    nulls = per_window(iv, (~value).astype(np.int64), k)
    entries = present + (nulls > 0)
    over = entries > max_size if max_size else np.zeros(k, dtype=bool)
    rows = np.where(over, 0, nn - mass)
    values = np.where(over, 0, d - present)
    er, ev, mu, eo = set_baseline(counts, n - nn, n, w, max_size)
    return {
        **common_metrics(n, counts, w, rows, values, (er, ev)),
        "window_distinct_mean": float(entries.mean()),
        "window_distinct_max": int(entries.max()),
        "overflow_fraction": float(over.mean()),
        "expected_window_distinct": mu,
        "expected_overflow_fraction": eo,
    }


class SetIndexPolars(SetIndex):
    """Reference: each value's occurrences become merged intervals of window starts (Polars),
    summed per window with a difference array (NumPy) — no sliding window; the baseline in
    Python floats over count classes. Shares no code with the Rust kernel."""

    def _compute(self, frames, combos):
        rows = [
            set_statistics(frames[f][c], self.window_rows, self.max_size)
            for ((f, c),) in combos
        ]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
