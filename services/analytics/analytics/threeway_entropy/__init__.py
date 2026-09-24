"""Three-way joint entropy (ordered scope, arity 3). See ThreewayEntropy."""

from analytics.base import lazy_attributes
from analytics.threeway_entropy.base import ThreewayEntropy
from analytics.threeway_entropy.rust import ThreewayEntropyRust

REFERENCE = "ThreewayEntropyPolars"
IMPLEMENTATIONS = ("ThreewayEntropyRust", "ThreewayEntropyPolars")

__getattr__ = lazy_attributes(__name__, {"ThreewayEntropyPolars": ".polars"})
__all__ = ["ThreewayEntropy", "ThreewayEntropyRust", "ThreewayEntropyPolars", "REFERENCE", "IMPLEMENTATIONS"]
