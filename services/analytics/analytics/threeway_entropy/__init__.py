"""Three-way joint entropy (ordered scope, arity 3). See ThreewayEntropy."""

from analytics.base import lazy_attributes
from analytics.threeway_entropy.ams import ThreewayEntropyAmsRust
from analytics.threeway_entropy.base import ThreewayEntropy
from analytics.threeway_entropy.rust import ThreewayEntropyRust

REFERENCE = "ThreewayEntropyPolars"
IMPLEMENTATIONS = (
    "ThreewayEntropyRust",
    "ThreewayEntropyAmsRust",
    "ThreewayEntropyPolars",
)

__getattr__ = lazy_attributes(__name__, {"ThreewayEntropyPolars": ".polars"})
__all__ = [
    "ThreewayEntropy",
    "ThreewayEntropyRust",
    "ThreewayEntropyAmsRust",
    "ThreewayEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
