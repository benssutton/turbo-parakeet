"""Narrowest value-preserving Arrow type per column: OneShotRecommender (one frame,
exact) and StreamingRecommender (batches over time). Not techniques on the uniform
contract. `to_arrow` exports a result with its top_k columns as Arrow maps."""

from turbo_parakeet.recommenders._maps import to_arrow
from turbo_parakeet.recommenders.chbench import to_chbench
from turbo_parakeet.recommenders.oneshot import OneShotRecommender
from turbo_parakeet.recommenders.streaming import StreamingRecommender

__all__ = ["OneShotRecommender", "StreamingRecommender", "to_arrow", "to_chbench"]
