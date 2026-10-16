"""ClickHouse set(max_size) skip-index suitability (per-column, order-dependent). See
SetIndex."""

from analytics.set_index.base import SetIndex
from analytics.set_index.polars import SetIndexPolars
from analytics.set_index.rust import SetIndexRust

REFERENCE = "SetIndexPolars"
IMPLEMENTATIONS = ("SetIndexRust", "SetIndexPolars")

__all__ = ["SetIndex", "SetIndexRust", "SetIndexPolars", "REFERENCE", "IMPLEMENTATIONS"]
