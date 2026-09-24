from analytics.similarity.base import Similarity


class SimilarityExactLRU(Similarity):
    """Reference: brute-force exact Jaccard / Overlap for every pair. Distinct-value sets
    are memoised by the per-instance LRU cache — part of what the MinHash
    implementations are benchmarked against, not an incidental detail."""

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, self.verify(combos))
