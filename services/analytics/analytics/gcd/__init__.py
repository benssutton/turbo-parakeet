"""Whole-column GCD (per-column scope). See Gcd for semantics."""

from analytics.base import lazy_attributes
from analytics.gcd.base import Gcd
from analytics.gcd.rust import GcdRust

REFERENCE = "GcdMath"
IMPLEMENTATIONS = ("GcdRust", "GcdNumpy", "GcdMath")

__getattr__ = lazy_attributes(__name__, {"GcdNumpy": ".numpy", "GcdMath": ".math"})
__all__ = ["Gcd", "GcdRust", "GcdNumpy", "GcdMath", "REFERENCE", "IMPLEMENTATIONS"]
