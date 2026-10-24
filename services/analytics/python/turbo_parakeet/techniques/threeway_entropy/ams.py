import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques._ams import check_ams_params
from turbo_parakeet.techniques.base import group_by_frame, metric_mismatches
from turbo_parakeet.techniques.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyAmsRust(ThreewayEntropy):
    """Rust extension `ams_threeway_joint_entropy`: one Chakrabarti–Cormode–McGregor sketch
    (`num_samples` estimators, O(num_samples) memory) per triplet, rayon-parallel; h_abc
    agrees with the reference within RTOL."""

    EXACT = False
    # mixed_dtypes(1_000) at num_samples=4096, seed=0: max relative error 3.9% (h_abc), 2.7% (h_ab).
    RTOL = 0.05
    ATOL = 1e-9

    def __init__(self, *, num_samples: int = 4096, seed: int = 0, **thresholds):
        super().__init__(**thresholds)
        check_ams_params(num_samples, seed)
        self.num_samples = num_samples
        self.seed = seed

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(
                list(dict.fromkeys(c for k in group for _, c in k))
            )
            triplets = [tuple(c for _, c in k) for k in group]
            out = _plugin.ams_threeway_joint_entropy(
                df, triplets, self.num_samples, self.seed
            )
            h = {frozenset((a, b, c)): e for a, b, c, e in out.iter_rows()}
            parts.append(
                self.metrics_frame(
                    group,
                    {
                        "h_abc": [h[frozenset(t)] for t in triplets],
                        "n_rows": [df.height] * len(group),
                    },
                )
            )
        return pl.concat(parts)

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """h_abc within RTOL, n_rows exact."""
        keys = self.key_columns()
        problems = metric_mismatches(
            result, reference, keys, ["h_abc"], self.RTOL, self.ATOL
        )
        if problems[:1] == ["key columns differ from the reference"]:
            return problems
        return problems + metric_mismatches(
            result, reference, keys, ["n_rows"], 0.0, 0.0
        )
