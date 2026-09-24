from analytics.pairwise_entropy.polars import entropy_bits
from analytics.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyPolars(ThreewayEntropy):
    """Reference: native Polars value_counts + entropy over a 3-field struct, one triplet at a time."""

    def _compute(self, frames, combos):
        return self.metrics_frame(
            combos,
            {
                "h_abc": [entropy_bits(frames[k[0][0]], [c for _, c in k]) for k in combos],
                "n_rows": [frames[k[0][0]].height for k in combos],
            },
        )
