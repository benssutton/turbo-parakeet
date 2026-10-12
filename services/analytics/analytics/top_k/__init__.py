"""Top-k most frequent values per column (per-column scope). See TopK."""

from analytics.base import lazy_attributes
from analytics.top_k.base import TopK

REFERENCE = "TopKPolars"
IMPLEMENTATIONS = ("TopKPolars",)

__getattr__ = lazy_attributes(__name__, {"TopKPolars": ".polars"})
__all__ = ["TopK", "TopKPolars", "REFERENCE", "IMPLEMENTATIONS"]
