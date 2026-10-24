"""Distinct non-null values per column (per-column scope). See DistinctCount."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.distinct_count.base import DistinctCount
from turbo_parakeet.techniques.distinct_count.hll import DistinctCountHllRust

REFERENCE = "DistinctCountPolars"
IMPLEMENTATIONS = (
    "DistinctCountHllRust",
    "DistinctCountPolarsApprox",
    "DistinctCountPolars",
)

__getattr__ = lazy_attributes(
    __name__,
    {
        "DistinctCountPolars": ".polars",
        "DistinctCountPolarsApprox": ".polars",
    },
)
__all__ = [
    "DistinctCount",
    "DistinctCountHllRust",
    "DistinctCountPolarsApprox",
    "DistinctCountPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
