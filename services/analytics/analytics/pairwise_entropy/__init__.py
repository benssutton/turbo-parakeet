"""Pairwise joint entropy / mutual information (ordered scope). See PairwiseEntropy."""

from analytics.base import lazy_attributes
from analytics.pairwise_entropy.base import PairwiseEntropy
from analytics.pairwise_entropy.rust import PairwiseEntropyRust

REFERENCE = "PairwiseEntropyPolars"
IMPLEMENTATIONS = ("PairwiseEntropyRust", "PairwiseEntropyPolars")

__getattr__ = lazy_attributes(__name__, {"PairwiseEntropyPolars": ".polars"})
__all__ = [
    "PairwiseEntropy",
    "PairwiseEntropyRust",
    "PairwiseEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
