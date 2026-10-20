"""Per-column profile for choosing narrower / more compressible Arrow types. See Describe."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.describe.base import CONCLUSIONS, Describe
from turbo_parakeet.techniques.describe.rust import DescribeRust

REFERENCE = "DescribePolars"
IMPLEMENTATIONS = ("DescribeRust", "DescribeDataFusion", "DescribePolars")

__getattr__ = lazy_attributes(
    __name__, {"DescribePolars": ".polars", "DescribeDataFusion": ".datafusion"}
)
__all__ = [
    "CONCLUSIONS",
    "Describe",
    "DescribeRust",
    "DescribePolars",
    "DescribeDataFusion",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
