"""Top-k most frequent values per column (per-column scope). See TopK."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.top_k.base import TopK
from turbo_parakeet.techniques.top_k.heavy_keeper import TopKHeavyKeeperRust
from turbo_parakeet.techniques.top_k.space_saving import TopKSpaceSavingRust

REFERENCE = "TopKPolars"
IMPLEMENTATIONS = ("TopKHeavyKeeperRust", "TopKSpaceSavingRust", "TopKPolars")

__getattr__ = lazy_attributes(__name__, {"TopKPolars": ".polars"})
__all__ = [
    "TopK",
    "TopKHeavyKeeperRust",
    "TopKSpaceSavingRust",
    "TopKPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
