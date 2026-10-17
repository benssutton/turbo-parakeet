"""ClickHouse minmax skip-index suitability (per-column, order-dependent). See MinMaxIndex."""

from analytics.minmax_index.base import MinMaxIndex
from analytics.minmax_index.polars import MinMaxIndexPolars
from analytics.minmax_index.rust import MinMaxIndexRust

REFERENCE = "MinMaxIndexPolars"
IMPLEMENTATIONS = ("MinMaxIndexRust", "MinMaxIndexPolars")

__all__ = [
    "MinMaxIndex",
    "MinMaxIndexRust",
    "MinMaxIndexPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
