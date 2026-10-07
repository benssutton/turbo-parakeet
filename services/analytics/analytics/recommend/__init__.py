"""Narrowest value-preserving Arrow type per column: OneShotRecommender (one frame,
exact) and StreamingRecommender (batches over time). Not techniques on the uniform
contract (spec docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md).
`to_arrow` exports a result with its top_k columns as Arrow maps."""

from analytics.recommend._maps import to_arrow
from analytics.recommend.oneshot import OneShotRecommender
from analytics.recommend.streaming import StreamingRecommender

__all__ = ["OneShotRecommender", "StreamingRecommender", "to_arrow"]
