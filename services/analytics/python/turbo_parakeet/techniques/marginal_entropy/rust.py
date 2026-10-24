import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.marginal_entropy.base import MarginalEntropy


class MarginalEntropyRust(MarginalEntropy):
    """Rust extension function `marginal_entropy`: dense-id encoding, rayon-parallel
    across columns; one binding call per frame."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            entropy = dict(_plugin.marginal_entropy(df).iter_rows())
            parts.append(self.entropy_rows(group, entropy, {frame: df.height}))
        return pl.concat(parts)
