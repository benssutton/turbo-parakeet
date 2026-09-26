"""
describe accuracy tests — every implementation in analytics.describe.

Oracles: hand-worked known answers and DescribePolars, the technique's reference.
Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pytest import approx

from analytics.describe import estimators

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — estimators (pure functions)

def test_chao1_with_doubletons():
    assert estimators.chao1(10, 4, 2) == approx((12.0, 10.249903382590167, 26.00618590489368))


def test_chao1_without_doubletons():
    assert estimators.chao1(5, 3, 0) == approx((8.0, 5.369121767830802, 29.38219792045809))


def test_chao1_without_unseen_mass_is_exact():
    assert estimators.chao1(7, 0, 3) == (7.0, 7.0, 7.0)
    assert estimators.chao1(7, 1, 0) == (7.0, 7.0, 7.0)


def test_schnabel_when_every_value_is_in_every_subset():
    assert estimators.schnabel([0, 0, 0, 0, 0, 0, 10], 10, 30_000) == approx(
        (9.523809523809524, 6.474567576908709, 16.378255262343956)
    )


def test_schnabel_mixed_history():
    # |S1| = 13, |S2| = 13, |S3| = 7, |S1 ∪ S2| = 20, R = 6 + 5 = 11, A = 13·13 + 7·20 = 309
    assert estimators.schnabel([5, 5, 5, 2, 2, 2, 1], 22, 1_000) == approx(
        (25.75, 15.69849888622768, 56.349688010901595)
    )


def test_schnabel_invalid_cases():
    assert estimators.schnabel([5, 5, 5, 2, 2, 2, 1], 22, 44) is None  # d/n = 0.5
    # masks 1, 2, 4 only (single-subset membership) -> R = 0: no recaptures
    assert estimators.schnabel([3, 3, 0, 3, 0, 0, 0], 9, 100) is None
    assert estimators.schnabel([0] * 7, 0, 0) is None


def test_duj1():
    assert estimators.duj1(10, 4, 40, 0.5) == approx(10.526315789473685)
    assert estimators.duj1(0, 0, 0, 0.5) == 0.0


HISTORY_10 = [0, 0, 0, 0, 0, 0, 10]


def test_estimate_picks_exact_then_duj1_then_schnabel_then_chao1():
    exact = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=1.0)
    assert (exact["est_method"], exact["est_cardinality"], exact["est_low"], exact["est_high"]) == ("exact", 10.0, 10.0, 10.0)
    duj = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=0.5)
    assert duj["est_method"] == "duj1" and duj["est_cardinality"] == approx(10.526315789473685)
    assert duj["est_low"] is None and duj["est_high"] is None
    sch = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=None)
    assert sch["est_method"] == "schnabel" and sch["est_cardinality"] == approx(9.523809523809524)
    assert sch["estimates_agree"] is True  # [10.25, 26.0] overlaps [6.47, 16.38]
    chao = estimators.estimate(10, 15, 4, 2, HISTORY_10, q=None)  # d/n ≥ 0.5 → Schnabel invalid
    assert chao["est_method"] == "chao1" and chao["est_cardinality"] == 12.0
    assert chao["schnabel"] is None and chao["estimates_agree"] is None


def test_estimates_disagree_under_heavy_skew():
    e = estimators.estimate(10, 40, 8, 0, HISTORY_10, q=None)  # Chao1 [17.47, 114.95] vs Schnabel [6.47, 16.38]
    assert e["estimates_agree"] is False


def test_unique_flag():
    assert estimators.estimate(5, 5, 5, 0, [5, 0, 0, 0, 0, 0, 0], None)["unique"] is True
    assert estimators.estimate(0, 0, 0, 0, [0] * 7, None)["unique"] is False
