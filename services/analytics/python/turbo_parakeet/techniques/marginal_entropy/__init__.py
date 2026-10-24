"""Shannon entropy per column (per-column scope). See MarginalEntropy."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.marginal_entropy.ams import MarginalEntropyAmsRust
from turbo_parakeet.techniques.marginal_entropy.base import MarginalEntropy
from turbo_parakeet.techniques.marginal_entropy.rust import MarginalEntropyRust

REFERENCE = "MarginalEntropyPolars"
IMPLEMENTATIONS = (
    "MarginalEntropyRust",
    "MarginalEntropyAmsRust",
    "MarginalEntropyPolars",
)

__getattr__ = lazy_attributes(__name__, {"MarginalEntropyPolars": ".polars"})
__all__ = [
    "MarginalEntropy",
    "MarginalEntropyRust",
    "MarginalEntropyAmsRust",
    "MarginalEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
