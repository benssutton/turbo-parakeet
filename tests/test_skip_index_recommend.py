"""Skip-index recommendations (spec 2026-10-20)."""

import pytest

from analytics.recommend import OneShotRecommender, StreamingRecommender

BAD = [
    {"skip_granularities": ()},
    {"skip_granularities": (1, 1)},
    {"skip_granularities": (0,)},
    {"set_max_sizes": (0,)},
    {"bloom_fprs": (0.0,)},
    {"bloom_fprs": (1.0,)},
    {"min_skip": 1.5},
    {"max_overflow": -0.1},
]


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
@pytest.mark.parametrize(
    "kwargs", BAD, ids=lambda k: next(iter(k)) + str(next(iter(k.values())))
)
def test_keywords_are_validated(cls, kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        cls(**kwargs)
