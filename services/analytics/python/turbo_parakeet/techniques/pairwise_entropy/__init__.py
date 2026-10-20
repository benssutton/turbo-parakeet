"""Pairwise joint entropy / mutual information (ordered scope). See PairwiseEntropy."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.pairwise_entropy.ams import PairwiseEntropyAmsRust
from turbo_parakeet.techniques.pairwise_entropy.base import PairwiseEntropy
from turbo_parakeet.techniques.pairwise_entropy.rust import PairwiseEntropyRust

REFERENCE = "PairwiseEntropyPolars"
IMPLEMENTATIONS = (
    "PairwiseEntropyRust",
    "PairwiseEntropyAmsRust",
    "PairwiseEntropyPolars",
)

__getattr__ = lazy_attributes(__name__, {"PairwiseEntropyPolars": ".polars"})
__all__ = [
    "PairwiseEntropy",
    "PairwiseEntropyRust",
    "PairwiseEntropyAmsRust",
    "PairwiseEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
