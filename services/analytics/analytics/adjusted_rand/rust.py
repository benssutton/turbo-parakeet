import polars as pl

from analytics import _plugin
from analytics.adjusted_rand.base import AdjustedRand
from analytics.base import group_by_frame


class AdjustedRandRust(AdjustedRand):
    """Rust extension `pairwise_adjusted_rand`: shared dense contingency builder, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            pairs = [(a, b) for (_, a), (_, b) in group]
            df = frames[frame].select(list(dict.fromkeys(c for p in pairs for c in p)))
            out = _plugin.pairwise_adjusted_rand(df, pairs)
            parts.append(self.rows_from_plugin(frame, out))
        return pl.concat(parts)
