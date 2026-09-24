"""
Set-similarity speed benchmark: MinHashRust vs MinHashDatasketch vs SimilarityExactLRU (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_similarity.py

The hypothesis here is "probabilistic pruning in Rust beats LRU-cached exact
checking of every pair"; `agrees` also enforces recall ≥ Similarity.MIN_RECALL.
"""

from harness import Dataset, large_dataset, run

from datagen import related_frames, similar_frames

if __name__ == "__main__":
    run(
        "analytics.similarity",
        [
            large_dataset(),
            Dataset("narrow related 3 frames x 1M rows", lambda: related_frames(1_000_000)),
            Dataset("medium 2 frames x 20 cols", lambda: similar_frames()),
            Dataset(
                "wide 2 frames x 100 cols",
                lambda: similar_frames(n_similar=50, n_independent=50, col_size=5_000, n_elements=100_000),
            ),
        ],
    )
