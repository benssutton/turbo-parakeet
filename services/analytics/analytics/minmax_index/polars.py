import numpy as np
import polars as pl

from analytics._skip_index import avoid, common_metrics, ranked, windows
from analytics.minmax_index.base import MinMaxIndex


def minmax_baseline(counts: np.ndarray, n: int, w: int) -> tuple[float, float]:
    """(expected_skip_rows, expected_skip_values) under a random row order."""
    nn, d = int(counts.sum()), int(counts.size)
    if d == 0:
        return float("nan"), float("nan")
    all_null = avoid(n, w, nn)
    rows = values = 0.0
    below = 0  # non-null rows below v
    for c in counts.tolist():
        p = avoid(n, w, below + c) + avoid(n, w, nn - below) - all_null
        rows += c * p
        values += p
        below += c
    return rows / nn, values / d


def minmax_statistics(s: pl.Series, window_rows: int) -> dict:
    """The MinMaxIndex metrics of one non-empty column."""
    ids, counts = ranked(s)
    n, d, nn = s.len(), int(counts.size), int(counts.sum())
    w, k = windows(n, window_rows)
    lo = ids.rolling_min(window_size=w, min_samples=1).slice(w - 1)
    hi = ids.rolling_max(window_size=w, min_samples=1).slice(w - 1)
    empty = lo.is_null().to_numpy()  # an all-null window
    lo = lo.fill_null(0).to_numpy().astype(np.int64)
    hi = hi.fill_null(0).to_numpy().astype(np.int64)
    if d:
        cum = np.concatenate([[0], np.cumsum(counts)])
        rows = np.where(empty, nn, nn - (cum[hi + 1] - cum[lo]))
        values = np.where(empty, d, d - (hi - lo + 1))
    else:
        rows = values = np.zeros(k, dtype=np.int64)
    return common_metrics(n, counts, w, rows, values, minmax_baseline(counts, n, w))


class MinMaxIndexPolars(MinMaxIndex):
    """Reference: rank ids from a join against the sorted distinct values, each window's
    extremes from Polars' rolling_min / rolling_max (nulls ignored), rows covered from
    cumulative counts; the baseline in Python floats (math.lgamma). Shares no code with the
    Rust kernel."""

    def _compute(self, frames, combos):
        rows = [
            minmax_statistics(frames[f][c], self.window_rows) for ((f, c),) in combos
        ]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
