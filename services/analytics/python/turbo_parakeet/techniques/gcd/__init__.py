"""Whole-column GCD (per-column scope). See Gcd for semantics."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.gcd.base import Gcd
from turbo_parakeet.techniques.gcd.rust import GcdRust

REFERENCE = "GcdMath"
IMPLEMENTATIONS = ("GcdRust", "GcdNumpy", "GcdMath")

__getattr__ = lazy_attributes(__name__, {"GcdNumpy": ".numpy", "GcdMath": ".math"})
__all__ = ["Gcd", "GcdRust", "GcdNumpy", "GcdMath", "REFERENCE", "IMPLEMENTATIONS"]
