"""Adjusted Rand Index (ordered scope). See AdjustedRand for semantics."""

from turbo_parakeet.techniques.adjusted_rand.base import AdjustedRand
from turbo_parakeet.techniques.adjusted_rand.rust import AdjustedRandRust
from turbo_parakeet.techniques.base import lazy_attributes

REFERENCE = "AdjustedRandSklearn"
IMPLEMENTATIONS = ("AdjustedRandRust", "AdjustedRandSklearn")

__getattr__ = lazy_attributes(__name__, {"AdjustedRandSklearn": ".sklearn"})
__all__ = [
    "AdjustedRand",
    "AdjustedRandRust",
    "AdjustedRandSklearn",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
