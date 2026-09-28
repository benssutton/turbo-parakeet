"""Chi-squared independence test (ordered scope). See ChiSquared for semantics."""

from analytics.base import lazy_attributes
from analytics.chi_squared.base import ChiSquared
from analytics.chi_squared.rust import ChiSquaredRust

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
