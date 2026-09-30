"""Narrowest value-preserving Arrow type per column, cast, verified and measured. See Recommend."""

from analytics.recommend.base import Recommend
from analytics.recommend.rust import RecommendRust
from analytics.recommend.streaming import StreamingRecommender

REFERENCE = "RecommendRust"
IMPLEMENTATIONS = ("RecommendRust",)

__all__ = ["Recommend", "RecommendRust", "StreamingRecommender", "REFERENCE", "IMPLEMENTATIONS"]
