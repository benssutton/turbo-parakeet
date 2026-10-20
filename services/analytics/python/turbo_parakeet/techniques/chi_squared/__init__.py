"""Chi-squared independence test (ordered scope). See ChiSquared for semantics."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.chi_squared.base import ChiSquared
from turbo_parakeet.techniques.chi_squared.rust import ChiSquaredRust

REFERENCE = "ChiSquaredScipy"
IMPLEMENTATIONS = ("ChiSquaredRust", "ChiSquaredScipy", "ChiSquaredPolarsDS")

__getattr__ = lazy_attributes(
    __name__, {"ChiSquaredScipy": ".scipy", "ChiSquaredPolarsDS": ".polars_ds"}
)
__all__ = [
    "ChiSquared",
    "ChiSquaredRust",
    "ChiSquaredScipy",
    "ChiSquaredPolarsDS",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
