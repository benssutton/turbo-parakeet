"""Canonical distinct values for the pure-Python multi-set implementations.

Mirrors how the Rust encoder identifies values: Polars `unique()` already treats
-0.0 == 0.0 and every NaN as one value; freezing keeps that true inside Python
sets (one shared NaN object, since nan != nan) and makes nested values hashable.
"""

import math

import polars as pl

NAN = float("nan")


def freeze(value):
    if isinstance(value, float):
        if math.isnan(value):
            return NAN
        return 0.0 if value == 0 else value
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    return value


def distinct_values(series: pl.Series) -> list:
    """Distinct non-null values of `series`, frozen for use as set members."""
    return [freeze(v) for v in series.drop_nulls().unique().to_list()]
