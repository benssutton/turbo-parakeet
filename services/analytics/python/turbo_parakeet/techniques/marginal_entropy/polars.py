from turbo_parakeet.techniques.marginal_entropy.base import MarginalEntropy
from turbo_parakeet.techniques.pairwise_entropy.polars import entropy_bits


class MarginalEntropyPolars(MarginalEntropy):
    """Reference: Polars value_counts + entropy per column (PairwiseEntropyPolars'
    marginal expression)."""

    def _compute(self, frames, combos):
        return self.metrics_frame(
            combos,
            {
                "h": [entropy_bits(frames[f].select(c), [c]) for ((f, c),) in combos],
                "n_rows": [frames[f].height for ((f, _),) in combos],
            },
        )
