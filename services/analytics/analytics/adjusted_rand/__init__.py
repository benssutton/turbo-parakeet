"""Adjusted Rand Index (ordered scope). See AdjustedRand for semantics."""

from analytics.adjusted_rand.base import AdjustedRand
from analytics.adjusted_rand.rust import AdjustedRandRust
from analytics.base import lazy_attributes

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
