"""The pairwise covering design and its two tiers."""

from synthdata.synth.classify import Coverage, classify, coverage, coverage_of
from synthdata.synth.tiers import (
    SMALL_ROWS,
    WINDOW_ROWS,
    synth_frame,
    synth_manifest,
    synth_small,
    synth_spec,
    synth_windows,
    spread,
)

__all__ = [
    "Coverage",
    "SMALL_ROWS",
    "WINDOW_ROWS",
    "classify",
    "synth_frame",
    "synth_manifest",
    "synth_small",
    "synth_spec",
    "synth_windows",
    "coverage",
    "coverage_of",
    "spread",
]
