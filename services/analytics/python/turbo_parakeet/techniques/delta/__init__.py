"""Delta and DoubleDelta statistics (per-column, order-dependent). See Delta for semantics."""

from turbo_parakeet.techniques.delta.base import Delta
from turbo_parakeet.techniques.delta.polars import DeltaPolars
from turbo_parakeet.techniques.delta.rust import DeltaRust

REFERENCE = "DeltaPolars"
IMPLEMENTATIONS = ("DeltaRust", "DeltaPolars")

__all__ = ["Delta", "DeltaRust", "DeltaPolars", "REFERENCE", "IMPLEMENTATIONS"]
