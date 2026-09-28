import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.pairwise_entropy.base import PairwiseEntropy


class PairwiseEntropyRust(PairwiseEntropy):
    """Rust extension functions `marginal_entropy` + `pairwise_joint_entropy`: dense-id
    encoding, flat-array counting, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(
                list(dict.fromkeys(c for k in group for _, c in k))
            )
            marginal = dict(_plugin.marginal_entropy(df).iter_rows())
            joint = _plugin.pairwise_joint_entropy(
                df, [(a, b) for (_, a), (_, b) in group]
            )
            h_ab = {frozenset((a, b)): h for a, b, h in joint.iter_rows()}
            parts.append(
                self.entropy_rows(
                    group,
                    [marginal[a] for (_, a), _ in group],
                    [marginal[b] for _, (_, b) in group],
                    [h_ab[frozenset((a, b))] for (_, a), (_, b) in group],
                    [df.height] * len(group),
                )
            )
        return pl.concat(parts)
