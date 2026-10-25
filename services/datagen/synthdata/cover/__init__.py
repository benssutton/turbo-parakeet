"""The pairwise covering design and its two tiers."""

from synthdata.cover.classify import Coverage, classify, coverage, coverage_of
from synthdata.cover.tiers import (
    SMALL_ROWS,
    WINDOW_ROWS,
    cover,
    cover_manifest,
    cover_small,
    cover_spec,
    cover_windows,
    spread,
)

__all__ = [
    "Coverage",
    "SMALL_ROWS",
    "WINDOW_ROWS",
    "classify",
    "cover",
    "cover_manifest",
    "cover_small",
    "cover_spec",
    "cover_windows",
    "coverage",
    "coverage_of",
    "spread",
]
