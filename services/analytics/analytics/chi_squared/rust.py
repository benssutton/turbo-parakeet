import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.chi_squared.base import ChiSquared


class ChiSquaredRust(ChiSquared):
    """Rust plugin `pairwise_chi_squared`: dense-id contingency tables, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            pairs = [(a, b) for (_, a), (_, b) in group]
            df = frames[frame].select(list(dict.fromkeys(c for p in pairs for c in p)))
            out = _plugin.pairwise_chi_squared(df, pairs).unnest("pairwise_chi_squared")
            parts.append(self.rows_from_plugin(frame, out))
        return pl.concat(parts)
