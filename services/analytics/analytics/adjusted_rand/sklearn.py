import math

import polars as pl
from sklearn.metrics import adjusted_rand_score

from analytics.adjusted_rand.base import AdjustedRand


def _labels(series: pl.Series) -> list[str]:
    """String labels; -0.0 and 0.0 are one label, as in the Rust encoder."""
    if series.dtype.is_float():
        series = series.to_frame().select(pl.when(pl.first() == 0).then(0.0).otherwise(pl.first())).to_series()
    return series.cast(pl.String).to_list()


class AdjustedRandSklearn(AdjustedRand):
    """Reference: sklearn.metrics.adjusted_rand_score, one pair at a time."""

    def _compute(self, frames, combos):
        ari, n_valid = [], []
        for (f, a), (_, b) in combos:
            pair = frames[f].select(a, b).drop_nulls()
            n_valid.append(pair.height)
            ari.append(adjusted_rand_score(_labels(pair[a]), _labels(pair[b])) if pair.height else math.nan)
        return self.metrics_frame(combos, {"ari": ari, "n_valid": n_valid})
