"""Shannon entropy of each column, null one category (per-column scope)."""

import polars as pl

from turbo_parakeet._dtypes import encodable
from turbo_parakeet.techniques.base import Technique


class MarginalEntropy(Technique):
    """H(col) in bits; null is its own category (the joint entropies' policy). Eligible:
    any column the Rust encoder accepts, in a frame with at least one row. Metrics only;
    `n_rows` counts nulls."""

    SCOPE = "per_column"
    ARITY = 1
    METRICS = {"h": pl.Float64, "n_rows": pl.UInt32}
    RTOL = 1e-9
    ATOL = 1e-12
    ACCURACY_METRICS = ("h",)

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)

    def entropy_rows(self, combos, entropy: dict, heights: dict) -> pl.DataFrame:
        """Metrics frame, one row per combo in `combos` order; `entropy` by column name,
        `heights` by frame name."""
        return self.metrics_frame(
            combos,
            {
                "h": [entropy[c] for ((_, c),) in combos],
                "n_rows": [heights[f] for ((f, _),) in combos],
            },
        )
