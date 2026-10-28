"""Explicit planted-answer and shape scenarios (frames with a known truth)."""

from synthdata.scenarios.codecs import delta_frame, gcd_frame, skip_frame
from synthdata.scenarios.entropy_extremes import near_constant, near_unique
from synthdata.scenarios.integers import (
    integer_multiples,
    integer_random,
    runs_int,
    runs_str,
)
from synthdata.scenarios.measure import measure_frame
from synthdata.scenarios.sets import containment_pairs, related_frames, similar_frames
from synthdata.scenarios.shapes import describe_narrow, describe_nested, describe_wide
from synthdata.scenarios.streams import TOP_K_COLUMNS, population_samples, top_k_streams
from synthdata.scenarios.zoo import (
    describe_mixed,
    low_cardinality,
    make_column,
    mixed_dtypes,
    stringified,
)

__all__ = [
    "TOP_K_COLUMNS",
    "containment_pairs",
    "delta_frame",
    "describe_mixed",
    "describe_narrow",
    "describe_nested",
    "describe_wide",
    "gcd_frame",
    "integer_multiples",
    "integer_random",
    "low_cardinality",
    "make_column",
    "measure_frame",
    "mixed_dtypes",
    "near_constant",
    "near_unique",
    "population_samples",
    "related_frames",
    "runs_int",
    "runs_str",
    "similar_frames",
    "skip_frame",
    "stringified",
    "top_k_streams",
]
