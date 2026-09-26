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

from analytics.describe import Describe, estimators
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — rendering, estimates, classification (via with_metrics)

DEFAULTS = {
    "n_rows": 4, "n_null": 0, "n_unique": 4, "entropy": 2.0, "f1": 4, "f2": 0,
    "argmin": 0, "argmax": 3, "top5_idx": [0, 1, 2, 3], "top5_count": [1, 1, 1, 1],
    "capture_history": [1, 1, 0, 1, 0, 0, 1],
    "size_bytes": 32, "size_zstd_bytes": 32, "size_polars_bytes": 32, "size_polars_zstd_bytes": 32,
}


def conclude(s: pl.Series, params: dict | None = None, **overrides) -> dict:
    """Describe's conclusions for one column whose metrics are DEFAULTS | overrides."""
    metrics = {m: None for m in Describe.METRICS} | DEFAULTS | overrides
    cls = with_metrics(Describe, **{k: [v] for k, v in metrics.items()})
    return run(cls, {"t": s.to_frame()}, **(params or {})).row(0, named=True)


def test_renders_min_max_and_top5_from_indices():
    r = conclude(pl.Series("x", [3.5, 1.25, None, 3.5]), argmin=1, argmax=0, top5_idx=[0, 1], top5_count=[2, 1])
    assert (r["min"], r["max"]) == ("1.25", "3.5")
    assert r["top5"] == [{"value": "3.5", "count": 2}, {"value": "1.25", "count": 1}]


def test_renders_inner_values_from_flattened_indices():
    s = pl.Series("x", [[1, 2], None, [], [3]], dtype=pl.List(pl.Int64))
    r = conclude(
        s, n_null=1, inner_n_values=3, inner_n_null=0, inner_n_unique=3, inner_f1=3, inner_f2=0,
        inner_argmin=0, inner_argmax=2, inner_top5_idx=[2], inner_top5_count=[1],
        inner_capture_history=[1, 1, 1, 0, 0, 0, 0],
    )
    assert (r["inner_min"], r["inner_max"]) == ("1", "3")
    assert r["inner_top5"] == [{"value": "3", "count": 1}]


def test_scalar_columns_have_null_inner_conclusions():
    r = conclude(pl.Series("x", [1, 2, 3, 4]))
    assert r["inner_min"] is None and r["inner_class"] is None and r["inner_est_method"] is None


def test_estimate_branches_follow_population_rows():
    s = pl.Series("x", [1, 2, 3, 4])
    assert conclude(s)["est_method"] == "chao1" and conclude(s)["est_cardinality"] == 10.0
    assert conclude(s, {"population_rows": 4})["est_method"] == "exact"
    duj = conclude(s, {"population_rows": 8})
    assert duj["est_method"] == "duj1" and duj["est_cardinality"] == approx(8.0)
    assert conclude(s, {"population_rows": {"t": 8}})["est_method"] == "duj1"
    with pytest.raises(ValueError, match="population_rows"):
        conclude(s, {"population_rows": 3})


def test_schnabel_branch_and_agreement_flag():
    s = pl.Series("x", list(range(10)) * 4)
    agree = conclude(s, n_rows=40, n_unique=10, f1=0, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10])
    assert agree["est_method"] == "schnabel" and agree["estimates_agree"] is True
    skew = conclude(s, n_rows=40, n_unique=10, f1=8, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10])
    assert skew["estimates_agree"] is False


D = datetime(2024, 1, 1)
CLASS_CASES = [
    pytest.param(pl.Series("x", [None] * 4, dtype=pl.Int64), dict(n_null=4, n_unique=0, argmin=None, argmax=None), {}, "null", id="null"),
    pytest.param(pl.Series("x", [7, 7, None, 7]), dict(n_null=1, n_unique=1), {}, "constant", id="constant"),
    pytest.param(pl.Series("x", [1, 5, 1, 5]), dict(n_unique=2, argmax=1), {}, "boolean", id="boolean"),
    pytest.param(pl.Series("x", [0, 4, 1, 2, 3]), dict(n_rows=5, n_unique=5, argmin=0, argmax=1), {}, "ordinal", id="ordinal_before_categorical"),
    pytest.param(pl.Series("x", [-3, 5, 1, 2]), dict(argmin=0, argmax=1), {}, "categorical", id="negative_not_ordinal"),
    pytest.param(pl.Series("x", [-3, 5, 1, 2]), dict(argmin=0, argmax=1), {"categorical_threshold": 5}, "discrete", id="over_threshold"),
    pytest.param(pl.Series("x", [0, 9, 1, 2]), dict(argmin=0, argmax=1), {}, "categorical", id="max_over_2N"),
    pytest.param(pl.Series("x", [0, 9, 1, 2]), dict(argmin=0, argmax=1), {"population_rows": 100}, "ordinal", id="2N_uses_population"),
    pytest.param(pl.Series("x", [0.0, 2.0, 1.0, 3.0]), dict(argmax=3, n_nan=0, n_inf=0, n_fractional=0), {}, "ordinal", id="whole_floats"),
    pytest.param(pl.Series("x", [0.0, 2.5, 1.0, 3.0]), dict(argmax=3, n_nan=0, n_inf=0, n_fractional=1), {}, "categorical", id="fractional_floats"),
    pytest.param(pl.Series("x", ["3", "1", "2", "0"]), dict(argmin=3, argmax=0, n_numeric_int=4, n_leading_zero=0, numeric_int_min=0, numeric_int_max=3), {}, "ordinal", id="integer_strings"),
    pytest.param(pl.Series("x", ["3", "01", "2", "0"]), dict(argmin=3, argmax=0, n_numeric_int=4, n_leading_zero=1, numeric_int_min=0, numeric_int_max=3), {}, "categorical", id="leading_zero_strings"),
    pytest.param(pl.Series("x", [D, D, D, D]).dt.date(), dict(), {}, "categorical", id="temporal_never_ordinal"),
]


@pytest.mark.parametrize("s, overrides, params, expected", CLASS_CASES)
def test_classification(s, overrides, params, expected):
    assert conclude(s, params, **overrides)["class"] == expected


def test_zero_row_column_conclusions():
    r = conclude(
        pl.Series("x", [], dtype=pl.Int64), n_rows=0, n_unique=0, entropy=float("nan"), f1=0, f2=0,
        argmin=None, argmax=None, top5_idx=[], top5_count=[], capture_history=[0] * 7,
    )
    assert r["class"] == "null" and r["top5"] == [] and r["min"] is None
    assert r["est_method"] == "chao1" and r["est_cardinality"] == 0.0 and r["unique"] is False


@pytest.mark.parametrize(
    "params",
    [{"categorical_threshold": -1}, {"zstd_level": 0}, {"zstd_level": 23}, {"seed": -1}, {"population_rows": -5}, {"population_rows": {"t": -1}}],
)
def test_constructor_validates(params):
    cls = with_metrics(Describe, **{m: [None] for m in Describe.METRICS})
    with pytest.raises(ValueError):
        cls(**params)


def test_agreement_tolerances():
    s = pl.Series("x", list(range(10)) * 4)
    base = dict(n_rows=40, n_unique=10, f1=0, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10], entropy=3.0, size_zstd_bytes=1_000)
    ref = run(with_metrics(Describe, **{k: [v] for k, v in ({m: None for m in Describe.METRICS} | DEFAULTS | base).items()}), {"t": s.to_frame()})

    def compare(**changes) -> list[str]:
        metrics = {m: None for m in Describe.METRICS} | DEFAULTS | base | changes
        cls = with_metrics(Describe, **{k: [v] for k, v in metrics.items()})
        return cls().agreement(run(cls, {"t": s.to_frame()}), ref)

    assert compare(entropy=3.0 + 1e-12, size_zstd_bytes=1_005) == []
    assert compare(capture_history=[0, 0, 0, 0, 0, 1, 9]) == []  # Schnabel moves < 10%
    assert any("size_zstd_bytes" in p for p in compare(size_zstd_bytes=1_100))
    assert any("n_unique" in p for p in compare(n_unique=11))
    # [5, 4, 0, 0, 0, 0, 1]: |S1| = 6, |S2| = 5, |S3| = 1, R = 2 → Schnabel 40/3 ≈ 13.3 vs 9.52 (> 10%)
    assert any("schnabel" in p for p in compare(capture_history=[5, 4, 0, 0, 0, 0, 1]))
