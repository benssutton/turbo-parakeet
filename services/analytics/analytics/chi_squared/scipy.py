import math

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

from analytics.chi_squared.base import ChiSquared


def _table(pair: pl.DataFrame) -> np.ndarray:
    """Dense contingency table (rows: values of column 0, cols: values of column 1)."""
    a, b = pair.columns
    counts = pair.group_by(a, b).len().with_columns(
        (pl.col(a).rank("dense").cast(pl.Int64) - 1),
        (pl.col(b).rank("dense").cast(pl.Int64) - 1),
    )
    table = np.zeros((counts[a].max() + 1, counts[b].max() + 1))
    table[counts[a].to_numpy(), counts[b].to_numpy()] = counts["len"].to_numpy()
    return table


def chi_squared_pair(df: pl.DataFrame, a: str, b: str) -> dict:
    pair = df.select(pl.col(a).cast(pl.String), pl.col(b).cast(pl.String)).drop_nulls()
    n = pair.height
    undefined = {"chi2_stat": math.nan, "p_value": math.nan, "cramers_v": math.nan, "low_expected_count": False, "n_valid": n}
    if n == 0:
        return undefined
    table = _table(pair)
    rows, cols = table.shape
    if rows < 2 or cols < 2:
        return undefined
    res = chi2_contingency(table, correction=False)
    stat = float(res.statistic)
    return {
        "chi2_stat": stat,
        "p_value": float(res.pvalue),
        "cramers_v": math.sqrt(stat / (n * (min(rows, cols) - 1))),
        "low_expected_count": bool(table.sum(axis=1).min() * table.sum(axis=0).min() / n < 5),
        "n_valid": n,
    }


class ChiSquaredScipy(ChiSquared):
    """Reference: scipy.stats.chi2_contingency (correction=False) on a Polars-built table, one pair at a time."""

    def _compute(self, frames, combos):
        rows = [chi_squared_pair(frames[f], a, b) for (f, a), (_, b) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})
