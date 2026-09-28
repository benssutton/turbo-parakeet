"""Set similarity — Jaccard / Overlap Coefficient (multi-set scope). See Similarity."""

from analytics.base import lazy_attributes
from analytics.similarity.base import Similarity
from analytics.similarity.exact import SimilarityExactLRU
from analytics.similarity.rust import MinHashRust, optimal_lsh_params

REFERENCE = "SimilarityExactLRU"
IMPLEMENTATIONS = ("MinHashRust", "MinHashDatasketch", "SimilarityExactLRU")

__getattr__ = lazy_attributes(__name__, {"MinHashDatasketch": ".datasketch"})
__all__ = [
    "Similarity",
    "MinHashRust",
    "MinHashDatasketch",
    "SimilarityExactLRU",
    "optimal_lsh_params",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
