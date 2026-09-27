import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyRust(ThreewayEntropy):
    """Rust extension `threeway_joint_entropy`: dense-id arithmetic keys, rayon-parallel across triplets."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(list(dict.fromkeys(c for k in group for _, c in k)))
            triplets = [tuple(c for _, c in k) for k in group]
            out = _plugin.threeway_joint_entropy(df, triplets)
            h = {frozenset((a, b, c)): e for a, b, c, e in out.iter_rows()}
            parts.append(
                self.metrics_frame(
                    group,
                    {"h_abc": [h[frozenset(t)] for t in triplets], "n_rows": [df.height] * len(group)},
                )
            )
        return pl.concat(parts)
