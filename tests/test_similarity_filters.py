"""
Test suite for column-similarity filters.

Fast correctness tests (no marker) use tiny synthetic data and assert that
MinHashLSHFilter and MinHashLSHFilter_datasketch return exactly the same pairs
as DeterministicSimilarityFilter (the brute-force ground truth).

Slow recall tests (pytest.mark.slow) use medium-scale synthetic data with
seeded similar pairs and assert recall >= RECALL_THRESHOLD against brute-force.

Run slow tests:   pytest -m slow
Skip slow tests:  pytest -m "not slow"
"""

import sys
import time
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analytics.deterministic_similarity_filter import DeterministicSimilarityFilter
from services.analytics.minhash_lsh_filter import MinHashLSHFilter
from services.analytics.minhash_lsh_filter_datasketch import MinHashLSHFilter_datasketch


JACCARD_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.9
RECALL_THRESHOLD = 0.85

PROBABILISTIC_FILTERS = [
    pytest.param(MinHashLSHFilter, id="MinHashLSHFilter"),
    pytest.param(MinHashLSHFilter_datasketch, id="MinHashLSHFilter_datasketch"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_pair_set(df: pl.DataFrame) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for row in df.iter_rows(named=True):
        a = f"{row['df_a']}|{row['col_a']}"
        b = f"{row['df_b']}|{row['col_b']}"
        pairs.add(tuple(sorted([a, b])))
    return pairs


def _ground_truth(
    lazyframes: Dict[str, pl.LazyFrame],
    j: float = JACCARD_THRESHOLD,
    oc: float = OVERLAP_THRESHOLD,
) -> Set[Tuple[str, str]]:
    f = DeterministicSimilarityFilter(jaccard_threshold=j, overlap_threshold=oc)
    f.add(lazyframes)
    return _extract_pair_set(f.get_similar_pairs())


def _generate_lazyframes(
    n_similar: int = 8,
    n_independent: int = 12,
    col_size: int = 200,
    n_elements: int = 2000,
    seed: int = 42,
) -> Dict[str, pl.LazyFrame]:
    """
    Two LazyFrames each with (n_similar + n_independent) columns.
    Each of the n_similar cross-frame pairs shares >= 93% of elements (OC > OVERLAP_THRESHOLD).
    Independent columns are drawn from disjoint regions of the element space to
    minimise accidental similarity.
    """
    rng = np.random.default_rng(seed)
    df0_cols: Dict[str, list] = {}
    df1_cols: Dict[str, list] = {}

    for i in range(n_similar):
        base = rng.choice(n_elements, size=col_size, replace=False).tolist()
        shared = int(col_size * 0.93)
        extra = rng.integers(0, n_elements, size=col_size - shared).tolist()
        df0_cols[f"sim_{i:02d}"] = base
        df1_cols[f"sim_{i:02d}"] = base[:shared] + extra

    region = n_elements // max(n_independent, 1)
    for i in range(n_independent):
        lo = (i * region) % n_elements
        pool = list(range(lo, min(lo + region, n_elements))) or list(range(n_elements))
        sz = int(min(len(pool), rng.integers(50, col_size)))
        df0_cols[f"ind0_{i:02d}"] = rng.choice(pool, size=sz, replace=False).tolist()
        df1_cols[f"ind1_{i:02d}"] = rng.choice(pool, size=sz, replace=False).tolist()

    def _to_lazy(cols: Dict[str, list]) -> pl.LazyFrame:
        max_len = max(len(v) for v in cols.values())
        return pl.DataFrame(
            {k: v + [None] * (max_len - len(v)) for k, v in cols.items()}
        ).lazy()

    return {"df0": _to_lazy(df0_cols), "df1": _to_lazy(df1_cols)}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_lazyframes() -> Dict[str, pl.LazyFrame]:
    df1 = pl.DataFrame({
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "C": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    }).lazy()
    df2 = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [4, 5, 6, 7, 8],
        "C": [10, 11, 12, 13, 14],
    }).lazy()
    return {"df1": df1, "df2": df2}


@pytest.fixture(scope="module")
def medium_lazyframes() -> Dict[str, pl.LazyFrame]:
    return _generate_lazyframes()


# ── Correctness tests (fast) ──────────────────────────────────────────────────

def test_deterministic_small(small_lazyframes):
    """DeterministicSimilarityFilter finds the 5 expected pairs in tiny data."""
    f = DeterministicSimilarityFilter(
        jaccard_threshold=JACCARD_THRESHOLD,
        overlap_threshold=OVERLAP_THRESHOLD,
    )
    f.add(small_lazyframes)
    pairs = _extract_pair_set(f.get_similar_pairs())
    cross = {p for p in pairs if p[0].split("|")[0] != p[1].split("|")[0]}
    assert len(pairs) == 5
    assert len(cross) == 4


@pytest.mark.parametrize("filter_cls", PROBABILISTIC_FILTERS)
def test_probabilistic_matches_deterministic(small_lazyframes, filter_cls):
    """Probabilistic filters return the same pairs as brute-force on tiny data."""
    expected = _ground_truth(small_lazyframes)
    f = filter_cls(jaccard_threshold=JACCARD_THRESHOLD, overlap_threshold=OVERLAP_THRESHOLD)
    f.add(small_lazyframes)
    actual = _extract_pair_set(f.get_similar_pairs())
    assert actual == expected, (
        f"{filter_cls.__name__} mismatch — "
        f"missing: {expected - actual}  extra: {actual - expected}"
    )


# ── Recall tests (slow) ───────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("filter_cls", PROBABILISTIC_FILTERS)
def test_recall_medium_scale(medium_lazyframes, filter_cls):
    """Probabilistic filters achieve >= RECALL_THRESHOLD recall on medium-scale data."""
    truth = _ground_truth(medium_lazyframes)
    if not truth:
        pytest.skip("No similar pairs in generated data")

    f = filter_cls(jaccard_threshold=JACCARD_THRESHOLD, overlap_threshold=OVERLAP_THRESHOLD)
    f.add(medium_lazyframes)
    t0 = time.perf_counter()
    result = _extract_pair_set(f.get_similar_pairs())
    elapsed = time.perf_counter() - t0

    fn = truth - result
    fp = result - truth
    recall = len(truth & result) / len(truth)

    print(
        f"\n{filter_cls.__name__}: {elapsed * 1000:.1f}ms | "
        f"truth={len(truth)} found={len(result)} recall={recall:.3f} "
        f"FN={len(fn)} FP={len(fp)}"
    )
    assert recall >= RECALL_THRESHOLD, (
        f"{filter_cls.__name__} recall {recall:.3f} < {RECALL_THRESHOLD}. "
        f"Missing pairs: {fn}"
    )
