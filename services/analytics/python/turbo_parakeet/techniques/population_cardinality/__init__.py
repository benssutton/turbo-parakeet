"""Distinct values in the population behind each column (per-column scope). See
PopulationCardinality."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.population_cardinality.base import PopulationCardinality
from turbo_parakeet.techniques.population_cardinality.rust import (
    PopulationCardinalityRust,
)

REFERENCE = "PopulationCardinalityPolars"
IMPLEMENTATIONS = ("PopulationCardinalityRust", "PopulationCardinalityPolars")

__getattr__ = lazy_attributes(__name__, {"PopulationCardinalityPolars": ".polars"})
__all__ = [
    "PopulationCardinality",
    "PopulationCardinalityRust",
    "PopulationCardinalityPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
