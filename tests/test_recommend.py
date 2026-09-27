"""
recommend accuracy tests — RecommendRust, the only implementation.

Oracles: hand-worked known answers; the pyarrow and Polars casts of each column to
the recommended types, measured by the pyarrow IPC oracle; predicted = measured;
the Python cardinality estimators. Accuracy only — nothing here is timed.
"""

import re
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from pytest import approx

from analytics.describe import _sizes
from datagen import describe_mixed, stringified
from harness import assert_contract, load, run

PKG = "analytics.recommend"
LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


def impl():
    return load(f"{PKG}:RecommendRust")


def rec(s: pl.Series, **params) -> dict:
    """recommend one Series; its result row as a dict."""
    return run(impl(), {"t": s.to_frame()}, **params).row(0, named=True)


def by_type(r: dict) -> dict:
    return {c["arrow_type"]: c for c in r["rec_candidates"]}


def ineligible_frame() -> pl.DataFrame:
    cols = [pl.Series("obj", [object(), object()], dtype=pl.Object), pl.Series("nul", [None, None], dtype=pl.Null)]
    return pl.DataFrame(cols)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract

def test_contract():
    frames = {"mixed": describe_mixed(300), "empty": describe_mixed(50).clear(), "bad": ineligible_frame()}
    cls = impl()
    result = run(cls, frames)
    assert_contract(cls, result, frames)
    computed = result.filter(pl.col("status") == "computed")
    assert computed["rec_arrow_type"].null_count() == 0  # describe_mixed nests no Int128
    assert computed["rec_polars_type"].null_count() == 0
    assert set(result.filter(pl.col("df_a") == "bad")["status"]) == {"ineligible"}


def test_constructor_validates_boolean_pairs():
    cls = impl()
    with pytest.raises(ValueError):
        cls(boolean_pairs=(("yes", "YES"),))
    with pytest.raises(ValueError):
        cls(boolean_pairs=(("y",),))


def test_population_rows_below_frame_rows_raise():
    with pytest.raises(ValueError):
        rec(pl.Series("x", [1, 2, 3]), population_rows=2)
