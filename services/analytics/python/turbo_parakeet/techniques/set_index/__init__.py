"""ClickHouse set(max_size) skip-index suitability (per-column, order-dependent). See
SetIndex."""

from turbo_parakeet.techniques.set_index.base import SetIndex
from turbo_parakeet.techniques.set_index.polars import SetIndexPolars
from turbo_parakeet.techniques.set_index.rust import SetIndexRust

REFERENCE = "SetIndexPolars"
IMPLEMENTATIONS = ("SetIndexRust", "SetIndexPolars")

__all__ = ["SetIndex", "SetIndexRust", "SetIndexPolars", "REFERENCE", "IMPLEMENTATIONS"]
