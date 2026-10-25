import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques._ams import AMS_ATOL_BITS, check_ams_params
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.marginal_entropy.base import MarginalEntropy


class MarginalEntropyAmsRust(MarginalEntropy):
    """Rust extension function `ams_marginal_entropy`: one Chakrabarti–Cormode–McGregor
    sketch (`num_samples` estimators, O(num_samples) memory) per column, rayon-parallel.
    """

    EXACT = False
    RTOL = 0.05
    ATOL = AMS_ATOL_BITS  # agrees within max(RTOL·|exact|, ATOL)

    def __init__(self, *, num_samples: int = 4096, seed: int = 0):
        super().__init__()
        check_ams_params(num_samples, seed)
        self.num_samples = num_samples
        self.seed = seed

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            entropy = dict(
                _plugin.ams_marginal_entropy(
                    df, self.num_samples, self.seed
                ).iter_rows()
            )
            parts.append(self.entropy_rows(group, entropy, {frame: df.height}))
        return pl.concat(parts)
