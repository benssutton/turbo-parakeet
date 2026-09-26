"""Per-column profile for choosing narrower / more compressible Arrow types. See Describe."""

from analytics.base import lazy_attributes
from analytics.describe.base import Describe

REFERENCE = "DescribePolars"
IMPLEMENTATIONS = ("DescribePolars",)

__getattr__ = lazy_attributes(__name__, {"DescribePolars": ".polars"})
__all__ = ["Describe", "REFERENCE", "IMPLEMENTATIONS"]
