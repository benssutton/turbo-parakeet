"""Three-way joint entropy (ordered scope, arity 3). See ThreewayEntropy."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.threeway_entropy.ams import ThreewayEntropyAmsRust
from turbo_parakeet.techniques.threeway_entropy.base import ThreewayEntropy
from turbo_parakeet.techniques.threeway_entropy.rust import ThreewayEntropyRust

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
