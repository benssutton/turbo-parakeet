"""ClickHouse minmax skip-index suitability per column (spec
docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md)."""

from turbo_parakeet.techniques._skip_index import SkipIndex


class MinMaxIndex(SkipIndex):
    """Windows and metrics as SkipIndex. A window is skipped for `col = v` iff v < min or
    v > max over its non-null values; an all-null window is skipped for every v.

    Values compare as ClickHouse compares them: numbers and temporals by value, Boolean
    false < true, String and Binary bytewise, Categorical by its string, Enum by category
    order (ClickHouse Enum compares its numeric values); floats are canonical (−0.0 = 0.0,
    every NaN equal, after every number). The random-order baseline is exact.
    """
