"""Set similarity — Jaccard / Overlap Coefficient (multi-set scope). See Similarity."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.similarity.base import Similarity
from turbo_parakeet.techniques.similarity.exact import SimilarityExactLRU
from turbo_parakeet.techniques.similarity.rust import MinHashRust, optimal_lsh_params

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
