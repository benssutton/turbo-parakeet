"""
describe accuracy tests — every implementation in analytics.describe.

Oracles: hand-worked known answers and DescribePolars, the technique's reference.
Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from pytest import approx

from analytics.describe import Describe, _sizes, estimators
from datagen import describe_mixed, stringified
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
    with_metrics,
)

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — estimators (pure functions)


def test_chao1_with_doubletons():
    assert estimators.chao1(10, 4, 2) == approx(
        (12.0, 10.249903382590167, 26.00618590489368)
    )


def test_chao1_without_doubletons():
    assert estimators.chao1(5, 3, 0) == approx(
        (8.0, 5.369121767830802, 29.38219792045809)
    )


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


HISTORY_10 = [0, 0, 0, 0, 0, 0, 10]


def test_estimate_picks_schnabel_then_chao1():
    sch = estimators.estimate(10, 40, 4, 2, HISTORY_10)
    assert sch["est_method"] == "schnabel" and sch["est_cardinality"] == approx(
        9.523809523809524
    )
    assert sch["estimates_agree"] is True  # [10.25, 26.0] overlaps [6.47, 16.38]
    chao = estimators.estimate(10, 15, 4, 2, HISTORY_10)  # d/n ≥ 0.5 → Schnabel invalid
    assert chao["est_method"] == "chao1" and chao["est_cardinality"] == 12.0
    assert chao["schnabel"] is None and chao["estimates_agree"] is None


def test_estimates_disagree_under_heavy_skew():
    e = estimators.estimate(10, 40, 8, 0, HISTORY_10)  # Chao1 [17.47, 114.95] vs Schnabel [6.47, 16.38]
    assert e["estimates_agree"] is False


def test_unique_flag():
    assert (
        estimators.estimate(5, 5, 5, 0, [5, 0, 0, 0, 0, 0, 0])["unique"] is True
    )
    assert estimators.estimate(0, 0, 0, 0, [0] * 7)["unique"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — rendering, estimates, classification (via with_metrics)

DEFAULTS = {
    "n_rows": 4,
    "n_null": 0,
    "n_unique": 4,
    "entropy": 2.0,
    "f1": 4,
    "f2": 0,
    "argmin": 0,
    "argmax": 3,
    "top5_idx": [0, 1, 2, 3],
    "top5_count": [1, 1, 1, 1],
    "capture_history": [1, 1, 0, 1, 0, 0, 1],
    "size_bytes": 32,
    "size_zstd_bytes": 32,
    "size_polars_bytes": 32,
    "size_polars_zstd_bytes": 32,
}


def conclude(s: pl.Series, params: dict | None = None, **overrides) -> dict:
    """Describe's conclusions for one column whose metrics are DEFAULTS | overrides."""
    metrics = {m: None for m in Describe.METRICS} | DEFAULTS | overrides
    cls = with_metrics(Describe, **{k: [v] for k, v in metrics.items()})
    return run(cls, {"t": s.to_frame()}, **(params or {})).row(0, named=True)


def test_renders_min_max_and_top5_from_indices():
    r = conclude(
        pl.Series("x", [3.5, 1.25, None, 3.5]),
        argmin=1,
        argmax=0,
        top5_idx=[0, 1],
        top5_count=[2, 1],
    )
    assert (r["min"], r["max"]) == ("1.25", "3.5")
    assert r["top5"] == [{"value": "3.5", "count": 2}, {"value": "1.25", "count": 1}]


def test_renders_inner_values_from_flattened_indices():
    s = pl.Series("x", [[1, 2], None, [], [3]], dtype=pl.List(pl.Int64))
    r = conclude(
        s,
        n_null=1,
        inner_n_values=3,
        inner_n_null=0,
        inner_n_unique=3,
        inner_f1=3,
        inner_f2=0,
        inner_argmin=0,
        inner_argmax=2,
        inner_top5_idx=[2],
        inner_top5_count=[1],
        inner_capture_history=[1, 1, 1, 0, 0, 0, 0],
    )
    assert (r["inner_min"], r["inner_max"]) == ("1", "3")
    assert r["inner_top5"] == [{"value": "3", "count": 1}]


def test_scalar_columns_have_null_inner_conclusions():
    r = conclude(pl.Series("x", [1, 2, 3, 4]))
    assert (
        r["inner_min"] is None
        and r["inner_class"] is None
        and r["inner_est_method"] is None
    )


def test_schnabel_branch_and_agreement_flag():
    s = pl.Series("x", list(range(10)) * 4)
    agree = conclude(
        s, n_rows=40, n_unique=10, f1=0, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10]
    )
    assert agree["est_method"] == "schnabel" and agree["estimates_agree"] is True
    skew = conclude(
        s, n_rows=40, n_unique=10, f1=8, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10]
    )
    assert skew["estimates_agree"] is False


D = datetime(2024, 1, 1)
CLASS_CASES = [
    pytest.param(
        pl.Series("x", [None] * 4, dtype=pl.Int64),
        dict(n_null=4, n_unique=0, argmin=None, argmax=None),
        {},
        "null",
        id="null",
    ),
    pytest.param(
        pl.Series("x", [7, 7, None, 7]),
        dict(n_null=1, n_unique=1),
        {},
        "constant",
        id="constant",
    ),
    pytest.param(
        pl.Series("x", [1, 5, 1, 5]),
        dict(n_unique=2, argmax=1),
        {},
        "boolean",
        id="boolean",
    ),
    pytest.param(
        pl.Series("x", [0, 4, 1, 2, 3]),
        dict(n_rows=5, n_unique=5, argmin=0, argmax=1),
        {},
        "ordinal",
        id="ordinal_before_categorical",
    ),
    pytest.param(
        pl.Series("x", [-3, 5, 1, 2]),
        dict(argmin=0, argmax=1),
        {},
        "categorical",
        id="negative_not_ordinal",
    ),
    pytest.param(
        pl.Series("x", [-3, 5, 1, 2]),
        dict(argmin=0, argmax=1),
        {"categorical_threshold": 5},
        "discrete",
        id="over_threshold",
    ),
    pytest.param(
        pl.Series("x", [0, 9, 1, 2]),
        dict(argmin=0, argmax=1),
        {},
        "categorical",
        id="max_over_2N",
    ),
    pytest.param(
        pl.Series("x", [0.0, 2.0, 1.0, 3.0]),
        dict(argmax=3, n_nan=0, n_inf=0, n_fractional=0),
        {},
        "ordinal",
        id="whole_floats",
    ),
    pytest.param(
        pl.Series("x", [0.0, 2.5, 1.0, 3.0]),
        dict(argmax=3, n_nan=0, n_inf=0, n_fractional=1),
        {},
        "categorical",
        id="fractional_floats",
    ),
    pytest.param(
        pl.Series("x", ["3", "1", "2", "0"]),
        dict(
            argmin=3,
            argmax=0,
            n_numeric_int=4,
            n_leading_zero=0,
            numeric_int_min=0,
            numeric_int_max=3,
        ),
        {},
        "ordinal",
        id="integer_strings",
    ),
    pytest.param(
        pl.Series("x", ["3", "01", "2", "0"]),
        dict(
            argmin=3,
            argmax=0,
            n_numeric_int=4,
            n_leading_zero=1,
            numeric_int_min=0,
            numeric_int_max=3,
        ),
        {},
        "categorical",
        id="leading_zero_strings",
    ),
    pytest.param(
        pl.Series("x", [D, D, D, D]).dt.date(),
        dict(),
        {},
        "categorical",
        id="temporal_never_ordinal",
    ),
]


@pytest.mark.parametrize("s, overrides, params, expected", CLASS_CASES)
def test_classification(s, overrides, params, expected):
    assert conclude(s, params, **overrides)["class"] == expected


def test_zero_row_column_conclusions():
    r = conclude(
        pl.Series("x", [], dtype=pl.Int64),
        n_rows=0,
        n_unique=0,
        entropy=float("nan"),
        f1=0,
        f2=0,
        argmin=None,
        argmax=None,
        top5_idx=[],
        top5_count=[],
        capture_history=[0] * 7,
    )
    assert r["class"] == "null" and r["top5"] == [] and r["min"] is None
    assert (
        r["est_method"] == "chao1"
        and r["est_cardinality"] == 0.0
        and r["unique"] is False
    )


@pytest.mark.parametrize(
    "params",
    [
        {"categorical_threshold": -1},
        {"zstd_level": 0},
        {"zstd_level": 23},
        {"seed": -1},
    ],
)
def test_constructor_validates(params):
    cls = with_metrics(Describe, **{m: [None] for m in Describe.METRICS})
    with pytest.raises(ValueError):
        cls(**params)


def test_agreement_tolerances():
    s = pl.Series("x", list(range(10)) * 4)
    base = dict(
        n_rows=40,
        n_unique=10,
        f1=0,
        f2=0,
        capture_history=[0, 0, 0, 0, 0, 0, 10],
        entropy=3.0,
        size_zstd_bytes=1_000,
    )
    ref = run(
        with_metrics(
            Describe,
            **{
                k: [v]
                for k, v in (
                    {m: None for m in Describe.METRICS} | DEFAULTS | base
                ).items()
            },
        ),
        {"t": s.to_frame()},
    )

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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers — the pyarrow size oracle itself


def test_ipc_body_bytes_framing():
    seq = pa.array(np.arange(1_000, dtype=np.int32))
    assert _sizes.ipc_body_bytes(seq, None) == 4_000
    assert (
        _sizes.ipc_body_bytes(seq, 1) == 1_912
    )  # 8-byte prefix + ZSTD frame, padded to 8
    with_nulls = pa.array([None if i % 3 == 0 else i for i in range(1_000)], pa.int32())
    assert (
        _sizes.ipc_body_bytes(with_nulls, None) == 4_128
    )  # + 125-byte validity padded to 128
    assert _sizes.ipc_body_bytes(pa.array([], pa.int32()), None) == 0
    assert _sizes.ipc_body_bytes(pa.array([1], pa.int32()), 1) == 24
    assert _sizes.ipc_body_bytes(pa.array(["ab", None], pa.large_string()), None) == 40
    assert (
        _sizes.ipc_body_bytes(pa.array([], pa.large_string()), None) == 8
    )  # offsets [0]


def test_column_sizes_arrow_and_polars():
    s = pl.Series("x", np.arange(1_000, dtype=np.int32))
    assert _sizes.column_sizes(s, 1) == {
        "size_bytes": 4_000,
        "size_zstd_bytes": 1_912,
        "size_polars_bytes": 4_000,
        "size_polars_zstd_bytes": 1_912,
    }


def test_categorical_size_includes_its_dictionary():
    s = pl.Series("x", ["a", "b", "a"], dtype=pl.Categorical)
    assert (
        _sizes.column_sizes(s, 1)["size_bytes"] == 48
    )  # dictionary batch 32 + keys 16


def test_polars_size_is_the_native_ipc_body():
    assert (
        _sizes.column_sizes(pl.Series("x", ["ab", None]), 1)["size_polars_bytes"] == 40
    )  # validity 8 + 2 views × 16
    assert (
        _sizes.column_sizes(pl.Series("x", ["a" * 20, "b"]), 1)["size_polars_bytes"]
        == 56
    )  # views 32 + 20-byte buffer → 24


PKG = "analytics.describe"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def profile(impl: str, s: pl.Series, **params) -> dict:
    """describe one Series; its result row as a dict."""
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def ineligible_frame() -> pl.DataFrame:
    cols = [
        pl.Series("obj", [object(), object()], dtype=pl.Object),
        pl.Series("nul", [None, None], dtype=pl.Null),
    ]
    cols += [
        pl.Series("i128", [1, 2], dtype=pl.Int128),
        pl.Series("list_i128", [[1], [2]], dtype=pl.List(pl.Int128)),
    ]
    if hasattr(pl, "UInt128"):
        cols.append(pl.Series("u128", [1, 2], dtype=pl.UInt128))
    return pl.DataFrame(cols)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": describe_mixed(300),
        "empty": describe_mixed(50).clear(),
        "bad": ineligible_frame(),
    }
    cls = load(impl)
    result = run(cls, frames)
    assert_contract(cls, result, frames)
    assert set(result.filter(pl.col("df_a") == "bad")["status"]) == {"ineligible"}
    assert set(result.filter(pl.col("df_a") != "bad")["status"]) == {"computed"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {
        "mixed": describe_mixed(2_000),
        "strings": stringified(describe_mixed(500)),
        "empty": describe_mixed(50).clear(),
    }
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.slow
@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_on_large_dataset(impl):
    frames = {"large": pl.read_ipc(LARGE)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
def test_frequencies_entropy_and_top5(impl):
    r = profile(impl, pl.Series("x", ["a", "a", "b", None]))
    assert (r["n_rows"], r["n_null"], r["n_unique"], r["f1"], r["f2"]) == (
        4,
        1,
        2,
        1,
        1,
    )
    assert r["entropy"] == approx(1.5)
    assert (r["top5_idx"], r["top5_count"]) == ([0, 2], [2, 1])
    assert sum(r["capture_history"]) == 2


@pytest.mark.parametrize("impl", ALL)
def test_top5_ties_break_by_first_occurrence(impl):
    r = profile(impl, pl.Series("x", [3, 1, 2, 1, 2, 3, 4, 5, 6]))
    assert (r["top5_idx"], r["top5_count"]) == ([0, 1, 2, 6, 7], [2, 2, 2, 1, 1])


@pytest.mark.parametrize("impl", ALL)
def test_extremes_are_first_occurrences(impl):
    r = profile(impl, pl.Series("x", [5, 1, 3, 1, 5]))
    assert (r["argmin"], r["argmax"], r["min"], r["max"]) == (1, 0, "1", "5")


@pytest.mark.parametrize("impl", ALL)
def test_float_zero_and_nan_are_one_value_each(impl):
    r = profile(impl, pl.Series("x", [0.0, -0.0, float("nan"), float("nan"), 1.5]))
    assert (
        r["n_unique"],
        r["n_nan"],
        r["argmin"],
        r["argmax"],
        r["n_fractional"],
        r["max_frac_digits"],
    ) == (3, 2, 0, 4, 1, 1)


@pytest.mark.parametrize("impl", ALL)
def test_float_decimal_places_and_specials(impl):
    r = profile(impl, pl.Series("x", [0.1, 1e-7, 1.5e20, 3.0, float("inf"), None]))
    assert (r["max_frac_digits"], r["n_fractional"], r["n_inf"], r["n_nan"]) == (
        7,
        2,
        1,
        0,
    )


@pytest.mark.parametrize("impl", ALL)
def test_f32_round_trip(impl):
    assert profile(impl, pl.Series("x", [0.1]))["n_f32_inexact"] == 1
    assert profile(impl, pl.Series("x", [0.5, 3.0, None]))["n_f32_inexact"] == 0


@pytest.mark.parametrize("impl", ALL)
def test_float32_uses_its_own_shortest_repr(impl):
    r = profile(impl, pl.Series("x", [0.1, 0.25], dtype=pl.Float32))
    assert (r["max_frac_digits"], r["n_f32_inexact"]) == (2, None)


@pytest.mark.parametrize("impl", ALL)
def test_non_float_columns_have_null_float_stats(impl):
    r = profile(impl, pl.Series("x", [1, 2]))
    assert all(
        r[k] is None
        for k in ("n_nan", "n_inf", "n_fractional", "max_frac_digits", "n_f32_inexact")
    )


@pytest.mark.parametrize("impl", ALL)
def test_string_and_list_lengths(impl):
    s = profile(impl, pl.Series("x", ["ab", "", None, "héllo"]))
    assert (s["min_len"], s["max_len"]) == (0, 6)  # UTF-8 bytes
    lst = profile(impl, pl.Series("x", [[1, 2], [], None], dtype=pl.List(pl.Int64)))
    assert (lst["min_len"], lst["max_len"]) == (0, 2)
    assert profile(impl, pl.Series("x", [1, 2]))["min_len"] is None


@pytest.mark.parametrize("impl", ALL)
def test_enum_orders_by_category_and_categorical_by_string(impl):
    e = profile(impl, pl.Series("x", ["a", "z", "a"], dtype=pl.Enum(["z", "a"])))
    assert (e["argmin"], e["argmax"]) == (1, 0)
    c = profile(impl, pl.Series("x", ["z", "a", "z"], dtype=pl.Categorical))
    assert (c["argmin"], c["argmax"]) == (1, 0)


@pytest.mark.parametrize("impl", ALL)
def test_equal_struct_values_count_once(impl):
    r = profile(
        impl,
        pl.Series(
            "x", [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 1, "b": None}, None]
        ),
    )
    assert (r["n_unique"], r["n_null"]) == (2, 1)


@pytest.mark.parametrize("impl", ALL)
def test_list_whole_and_inner_values(impl):
    r = profile(
        impl, pl.Series("x", [[1, None], None, [], [3, 1]], dtype=pl.List(pl.Int64))
    )
    assert (r["n_rows"], r["n_null"], r["n_unique"], r["argmin"], r["argmax"]) == (
        4,
        1,
        3,
        2,
        3,
    )
    assert (r["inner_n_values"], r["inner_n_null"], r["inner_n_unique"]) == (
        4,
        1,
        2,
    )  # [1, None, 3, 1]
    assert (r["inner_argmin"], r["inner_argmax"], r["inner_f1"], r["inner_f2"]) == (
        0,
        2,
        1,
        1,
    )
    assert (r["inner_top5_idx"], r["inner_top5_count"]) == ([0, 2], [2, 1])
    assert (r["inner_min"], r["inner_max"]) == ("1", "3")


@pytest.mark.parametrize("impl", ALL)
def test_n_midnight_uses_local_time(impl):
    s = pl.Series(
        "x",
        [
            datetime(2024, 3, 31, 0, 0),
            datetime(2024, 3, 31, 12, 0),
            datetime(2024, 7, 1, 0, 0),
            datetime(2024, 10, 28, 0, 0, 0, 1),
        ],
    ).dt.replace_time_zone("Europe/London")
    # local midnight, noon, local midnight in BST (23:00 UTC — a UTC check would miss it), 1 µs past midnight
    assert profile(impl, s)["n_midnight"] == 2
    assert (
        profile(impl, pl.Series("x", [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)]))[
            "n_midnight"
        ]
        == 1
    )
    assert profile(impl, pl.Series("x", [1]))["n_midnight"] is None


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_and_all_null_columns(impl):
    z = profile(impl, pl.Series("x", [], dtype=pl.Int32))
    assert (
        z["n_rows"],
        z["n_unique"],
        z["argmin"],
        z["top5_idx"],
        z["capture_history"],
    ) == (0, 0, None, [], [0] * 7)
    assert math.isnan(z["entropy"]) and z["size_bytes"] == 0 and z["min_len"] is None
    a = profile(impl, pl.Series("x", [None, None], dtype=pl.String))
    assert (a["n_unique"], a["entropy"], a["argmin"]) == (0, 0.0, None)


@pytest.mark.parametrize("impl", ALL)
def test_capture_history(impl):
    everywhere = profile(impl, pl.Series("x", np.repeat(np.arange(10), 1_000)))
    assert everywhere["capture_history"] == [0, 0, 0, 0, 0, 0, 10]
    rng = np.random.default_rng(7)
    r = profile(impl, pl.Series("x", rng.integers(0, 5_000, 20_000)))
    assert sum(r["capture_history"]) == r["n_unique"]


@pytest.mark.parametrize("impl", ALL)
def test_sizes_through_the_technique(impl):
    r = profile(impl, pl.Series("x", np.arange(1_000, dtype=np.int32)))
    assert (r["size_bytes"], r["size_polars_bytes"]) == (4_000, 4_000)
    assert r["size_zstd_bytes"] == approx(1_912, rel=0.01)
    assert (
        profile(
            impl,
            pl.Series(
                "x", [None if i % 3 == 0 else i for i in range(1_000)], dtype=pl.Int32
            ),
        )["size_bytes"]
        == 4_128
    )
    assert profile(impl, pl.Series("x", ["ab", None]))["size_polars_bytes"] == 40


@pytest.mark.parametrize("impl", ALL)
def test_gcd_metric(impl):
    assert profile(impl, pl.Series("x", [10, None, 20, 30]))["gcd"] == 10
    assert profile(impl, pl.Series("x", [None, None], dtype=pl.Int32))["gcd"] == 0
    assert (
        profile(
            impl,
            pl.Series("x", [Decimal("1.20"), Decimal("3.40")], dtype=pl.Decimal(10, 2)),
        )["gcd"]
        == 20
    )
    days = pl.Series(
        "x", [datetime(2024, 1, 1), datetime(2024, 1, 2)], dtype=pl.Datetime("us")
    )
    assert profile(impl, days)["gcd"] == 86_400_000_000
    assert profile(impl, pl.Series("x", [1.5, 2.5]))["gcd"] is None
    assert profile(impl, pl.Series("x", ["a"], dtype=pl.Categorical))["gcd"] is None
    lists = profile(impl, pl.Series("x", [[4, 8], None, [12]]))
    assert (lists["gcd"], lists["inner_gcd"]) == (None, 4)


@pytest.mark.parametrize("impl", ALL)
def test_sum_len_metrics(impl):
    r = profile(impl, pl.Series("x", ["ab", "ab", "c", None, "héllo"]))
    assert (r["sum_len"], r["sum_len_unique"]) == (11, 9)  # 2+2+1+6, 2+1+6
    c = profile(impl, pl.Series("x", ["ab", "ab", "c"], dtype=pl.Categorical))
    assert (c["sum_len"], c["sum_len_unique"]) == (5, 3)
    b = profile(impl, pl.Series("x", [b"ab", b"ab", None]))
    assert (b["sum_len"], b["sum_len_unique"]) == (4, 2)
    assert profile(impl, pl.Series("x", [1, 2]))["sum_len"] is None
    lists = profile(impl, pl.Series("x", [["ab", "c"], ["ab"]]))
    assert (
        lists["sum_len"],
        lists["inner_sum_len"],
        lists["inner_sum_len_unique"],
    ) == (None, 5, 3)
    empty = profile(impl, pl.Series("x", [], dtype=pl.String))
    assert (empty["sum_len"], empty["sum_len_unique"]) == (0, 0)


@pytest.mark.parametrize("impl", ALL)
def test_numeric_fraction_and_significant_digits(impl):
    r = profile(impl, pl.Series("x", ["1.50", "0.00120", "7", "abc", None]))
    assert (
        r["numeric_min_frac_digits"],
        r["numeric_max_frac_digits"],
        r["numeric_max_sig_digits"],
    ) == (0, 4, 2)
    r = profile(impl, pl.Series("x", ["1200", "-0.0", "12.50"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (0, 4)
    r = profile(impl, pl.Series("x", ["0.25", "1.125"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (2, 4)
    r = profile(impl, pl.Series("x", ["abc"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (None, None)


@pytest.mark.parametrize("impl", ALL)
def test_iso_significant_fraction_digits(impl):
    r = profile(
        impl,
        pl.Series("x", ["10:00:00.120", "2024-01-05T10:00:00.000", "2024-01-05", None]),
    )
    assert (r["iso_max_frac_digits"], r["iso_max_sig_frac_digits"]) == (3, 2)
    r = profile(
        impl, pl.Series("x", ["2024-01-05 10:00", "2024-01-05T10:00:00.000000+02:00"])
    )
    assert (r["iso_max_frac_digits"], r["iso_max_sig_frac_digits"]) == (6, 0)
    assert (
        profile(impl, pl.Series("x", ["2024-01-05"]))["iso_max_sig_frac_digits"] is None
    )


NUMERIC_CASES = [
    pytest.param(
        ["5.", ".5", "1.2.3", "+5", "1e5", " 5", "5 ", "٣"],
        dict(
            n_numeric=0, n_numeric_int=0, n_leading_zero=0, numeric_max_int_digits=None
        ),
        id="rejected",
    ),
    pytest.param(
        ["007", "-012"],
        dict(
            n_numeric=2,
            n_numeric_int=2,
            n_leading_zero=2,
            numeric_int_min=-12,
            numeric_int_max=7,
        ),
        id="leading_zero",
    ),
    pytest.param(
        ["0", "-0"],
        dict(n_numeric=2, n_numeric_int=2, n_leading_zero=0, numeric_max_int_digits=0),
        id="zero_is_not_leading",
    ),
    pytest.param(
        ["007.50"],
        dict(
            n_numeric=1,
            n_numeric_int=0,
            n_leading_zero=0,
            numeric_max_int_digits=1,
            numeric_max_frac_digits=1,
        ),
        id="decimal_ignores_zeros",
    ),
    pytest.param(
        ["12", "-7", "300", "0.25"],
        dict(
            numeric_int_min=-7,
            numeric_int_max=300,
            numeric_max_int_digits=3,
            numeric_max_frac_digits=2,
        ),
        id="ranges",
    ),
    pytest.param(
        ["1" + "0" * 38],
        dict(
            n_numeric_int=1,
            numeric_int_min=None,
            numeric_int_max=None,
            numeric_max_int_digits=39,
        ),
        id="39_digits",
    ),
]


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("values, expected", NUMERIC_CASES)
def test_numeric_scanner(impl, values, expected):
    r = profile(impl, pl.Series("x", values))
    assert {k: r[k] for k in expected} == expected


ISO_CASES = [
    pytest.param(
        ["2024-02-29", "2023-02-29", "2024-13-01", "2024-1-05"],
        dict(n_iso_date=1, iso_max_frac_digits=None),
        id="calendar",
    ),
    pytest.param(
        [
            "23:59",
            "24:00",
            "23:59:60",
            "10:00:00.123456789",
            "10:00:00.1234567890",
            "10:00:00.",
        ],
        dict(n_iso_time=2, iso_max_frac_digits=9),
        id="times",
    ),
    pytest.param(
        ["2024-01-05T10:00", "2024-01-05 10:00:00", "2024-01-05t10:00"],
        dict(n_iso_datetime=2, n_iso_datetime_tz=0, iso_max_frac_digits=0),
        id="separator",
    ),
    pytest.param(
        [
            "2024-01-05T10:00:00Z",
            "2024-01-05 10:00:00+00:00",
            "2024-01-05T10:00-00:00",
            "2024-01-05T10:00+02:00",
            "2024-01-05T10:00+0200",
        ],
        dict(n_iso_datetime_tz=4, iso_n_offsets=2),
        id="offsets",
    ),
    pytest.param(
        [
            "2024-01-05T00:00:00.000",
            "2024-01-05T00:00",
            "2024-01-05T00:00:01",
            "2024-01-05T00:00Z",
        ],
        dict(n_iso_datetime=3, n_iso_datetime_tz=1, iso_n_midnight=3),
        id="midnight",
    ),
]


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("values, expected", ISO_CASES)
def test_iso_scanner(impl, values, expected):
    r = profile(impl, pl.Series("x", values))
    assert {k: r[k] for k in expected} == expected


@pytest.mark.parametrize("impl", ALL)
def test_scanners_on_categorical_and_non_strings(impl):
    c = profile(impl, pl.Series("x", ["12", "007", "x"], dtype=pl.Categorical))
    assert (c["n_numeric"], c["n_leading_zero"]) == (2, 1)
    assert profile(impl, pl.Series("x", [1, 2]))["n_numeric"] is None


@pytest.mark.parametrize("impl", ALL)
def test_adversarial_long_strings(impl):
    s = pl.Series(
        "x",
        [
            "0" * 10**6 + "." + "0" * 10**6 + ".",
            "0" * 10**6,
            "2024-01-05T" + "0" * 10**6,
            "9" * 39,
        ],
    )
    r = profile(impl, s)
    assert (r["n_numeric"], r["n_numeric_int"], r["n_leading_zero"]) == (2, 2, 1)
    assert r["numeric_int_min"] is None  # "9"*39 has more than 38 significant digits
    assert r["n_iso_date"] == r["n_iso_datetime"] == r["n_iso_datetime_tz"] == 0


def _non_null(frame: pl.DataFrame, col: str) -> int:
    return frame[col].len() - frame[col].null_count()


@pytest.mark.parametrize("impl", ALL)
def test_stringified_describe_mixed(impl):
    source = describe_mixed(500)
    text = stringified(source)
    out = {r["col_a"]: r for r in run(load(impl), {"s": text}).iter_rows(named=True)}
    for c in ("i8", "i16", "i32", "i64", "d38", "u8", "u16", "u32", "u64", "codes"):
        assert out[c]["n_numeric_int"] == _non_null(source, c), c
        assert out[c]["n_leading_zero"] == 0, c
    assert out["date"]["n_iso_date"] == _non_null(source, "date")
    assert out["dt_naive"]["n_iso_datetime"] == _non_null(
        source, "dt_naive"
    )  # "2024-01-05 10:00:00.000000"
    assert out["dt_tz"]["n_iso_datetime_tz"] == _non_null(
        source, "dt_tz"
    )  # "…+00:00" / "…+01:00"
    assert out["dt_tz"]["iso_n_offsets"] == 2  # GMT and BST across 2024
    assert out["time"]["n_iso_time"] == _non_null(source, "time")
    assert out["f64_price"]["n_numeric"] == _non_null(source, "f64_price")
    assert (
        out["dec"]["n_numeric"] == _non_null(source, "dec")
        and out["dec"]["numeric_max_frac_digits"] <= 2
    )
    # Polars writes "1e-7", "1.5e+20", "inf", "NaN": exponent and special forms are deliberately not numeric.
    assert out["f64"]["n_numeric"] == int(
        text["f64"].is_in(["0.0", "-0.0", "0.1"]).sum()
    )
    assert (
        out["bool"]["n_numeric"] == 0 and out["bool"]["n_iso_date"] == 0
    )  # "true" / "false"
    lst = out["list_i64"]
    assert lst["inner_n_numeric_int"] == lst["inner_n_values"] - lst["inner_n_null"]


@pytest.mark.parametrize("impl", ALL)
def test_stringified_large_dataset_columns(impl):
    large = pl.read_ipc(LARGE).head(5_000)
    cols = [
        c
        for c, dt in large.schema.items()
        if dt in (pl.Int32, pl.Int64, pl.Date, pl.Float64)
    ][:25]
    source = large.select(cols)
    out = {
        r["col_a"]: r
        for r in run(load(impl), {"s": stringified(source)}).iter_rows(named=True)
    }
    for c in cols:
        n = _non_null(source, c)
        if source[c].dtype in (pl.Int32, pl.Int64):
            assert out[c]["n_numeric_int"] == n, c
        elif source[c].dtype == pl.Date:
            assert out[c]["n_iso_date"] == n, c
        else:
            assert out[c]["n_numeric"] <= n, c


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_and_sliced_input(impl):
    parts = [
        pl.Series("x", ["b", None]),
        pl.Series("x", ["a", "c"]),
        pl.Series("x", ["a"]),
    ]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    r = profile(impl, s)
    assert (r["n_unique"], r["argmin"], r["argmax"], r["top5_idx"]) == (
        3,
        2,
        3,
        [2, 0, 3],
    )
    sliced = pl.Series("x", [[9], [1, 2], None, [3]], dtype=pl.List(pl.Int64)).slice(
        1, 3
    )
    q = profile(impl, sliced)
    assert (q["inner_n_values"], q["inner_argmin"], q["size_bytes"]) == (
        3,
        0,
        profile(impl, pl.Series("x", [[1, 2], None, [3]], dtype=pl.List(pl.Int64)))[
            "size_bytes"
        ],
    )


@pytest.mark.parametrize("impl", ALL)
def test_first_occurrence_across_parallel_chunks(impl):
    chunk = 1 << 16
    values = np.full(3 * chunk + 17, 5, dtype=np.int64)
    values[2 * chunk + 3] = 1  # first minimum, third chunk
    values[3 * chunk + 1] = 1  # later minimum, fourth chunk
    values[chunk + 7] = 9  # maximum, second chunk
    r = profile(impl, pl.Series("x", values))
    assert (r["argmin"], r["argmax"]) == (2 * chunk + 3, chunk + 7)
    assert (r["top5_idx"], r["top5_count"]) == (
        [0, 2 * chunk + 3, chunk + 7],
        [len(values) - 3, 2, 1],
    )


@pytest.mark.parametrize("impl", OTHERS)
def test_categorical_and_enum_sizes_agree(impl):
    frame = describe_mixed(1_000).select("cat", "enum", "str_free")
    exact = ["col_a", "size_bytes", "size_polars_bytes"]
    assert (
        run(load(impl), {"t": frame})
        .select(exact)
        .equals(run(reference(PKG), {"t": frame}).select(exact))
    )


@pytest.mark.parametrize("impl", ALL)
def test_nested_ordering_with_null_elements(impl):
    s = pl.Series("x", [[2], [None], [1, None], [], [1]], dtype=pl.List(pl.Int64))
    want = s.arg_sort(nulls_last=False)  # Polars order is the definition
    r = profile(impl, s)
    assert (r["argmin"], r["argmax"], r["n_unique"]) == (want[0], want[-1], 5)


# ─────────────────────────────────────────────────────────────────────────────
# Final-review regressions


@pytest.mark.parametrize("impl", ALL)
def test_array_lengths_with_null_rows(impl):
    # Polars marks arr.len() sorted even with a null in the middle, so a naive .max() returns None.
    r = profile(
        impl, pl.Series("x", [[1, 2], None, [3, 4]], dtype=pl.Array(pl.Int64, 2))
    )
    assert (r["min_len"], r["max_len"]) == (2, 2)
    nested = profile(
        impl,
        pl.Series("x", [[[1, 2], None, [3, 4]]], dtype=pl.List(pl.Array(pl.Int64, 2))),
    )
    assert (nested["inner_min_len"], nested["inner_max_len"]) == (2, 2)


_SLICED = """
import polars as pl
from analytics import describe
from harness import load, run
fresh = pl.DataFrame({{
    "arr": pl.Series([[3, 4], None, [5, None]], dtype=pl.Array(pl.Int64, 2)),
    "st": pl.Series([{{"a": 2}}, None, {{"a": 3}}]),
}})
full = pl.DataFrame({{
    "arr": pl.Series([[1, 2], [3, 4], None, [5, None]], dtype=pl.Array(pl.Int64, 2)),
    "st": pl.Series([{{"a": 1}}, {{"a": 2}}, None, {{"a": 3}}]),
}})
cls = load("{impl}")
got = run(cls, {{"t": full.slice(1, 3)}})
want = run(cls, {{"t": fresh}})
problems = cls().agreement(got, want)
print("OK" if not problems else problems)
"""


@pytest.mark.parametrize("impl", ALL)
def test_sliced_nested_with_nulls(impl):
    """A sliced Array/Struct with nulls must describe exactly like the same rows built
    fresh — and must never crash the interpreter (run in a subprocess to observe that).
    """
    import subprocess
    import sys

    tests_dir = Path(__file__).parent
    code = _SLICED.format(impl=impl)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tests_dir,
        timeout=300,
    )
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr[-2000:]}"
    assert out.stdout.strip().endswith("OK"), out.stdout[-2000:]


NESTED_CASES = [
    pytest.param(
        pl.Series("x", [[0.0], [-0.0], [1.5]]),
        dict(n_unique=2, argmin=0, argmax=2),
        id="list_negzero",
    ),
    pytest.param(
        pl.Series("x", [{"a": 0.0}, {"a": -0.0}, {"a": 1.0}]),
        dict(n_unique=2, argmin=0, argmax=2),
        id="struct_negzero",
    ),
    pytest.param(
        pl.Series("x", [["a"], ["z"], ["a"]], dtype=pl.List(pl.Enum(["z", "a"]))),
        dict(n_unique=2, argmin=1, argmax=0),
        id="list_enum",
    ),
    pytest.param(
        pl.Series("x", [["b"], ["a"], ["b"]], dtype=pl.List(pl.Categorical)),
        dict(n_unique=2, argmin=1, argmax=0),
        id="list_categorical",
    ),
    pytest.param(
        pl.Series(
            "x", [{"e": "a"}, {"e": "z"}], dtype=pl.Struct({"e": pl.Enum(["z", "a"])})
        ),
        dict(n_unique=2, argmin=1, argmax=0),
        id="struct_enum",
    ),
]


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("s, expected", NESTED_CASES)
def test_nested_floats_and_enums(impl, s, expected):
    r = profile(impl, s)
    assert {k: r[k] for k in expected} == expected
