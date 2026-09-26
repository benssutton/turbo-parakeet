"""Per-column profile for choosing narrower / more compressible Arrow types. See Describe."""

from analytics.base import lazy_attributes
from analytics.describe.base import Describe
from analytics.describe.rust import DescribeRust

REFERENCE = "DescribePolars"
IMPLEMENTATIONS = ("DescribeRust", "DescribeDataFusion", "DescribePolars")

__getattr__ = lazy_attributes(__name__, {"DescribePolars": ".polars", "DescribeDataFusion": ".datafusion"})
__all__ = ["Describe", "DescribeRust", "DescribePolars", "DescribeDataFusion", "REFERENCE", "IMPLEMENTATIONS"]
