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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers

LONDON = [datetime(2024, 1, 5), datetime(2024, 1, 6)]
OFFSETS = "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"

KNOWN = [
    pytest.param(pl.Series("x", [0, 1, 1, None]), {}, "bool", "Boolean", id="int_0_1"),
    pytest.param(pl.Series("x", [0, 5, 127]), {}, "uint8", "UInt8", id="uint_before_int_on_tie"),
    pytest.param(pl.Series("x", [-5, 100]), {}, "int8", "Int8", id="int8"),
    pytest.param(pl.Series("x", [-200, 5]), {}, "int16", "Int16", id="int16"),
    pytest.param(pl.Series("x", [10**20, -1], dtype=pl.Int128), {}, "decimal128(21, 0)", "Decimal(precision=21, scale=0)", id="int128_beyond_64_bits"),
    pytest.param(pl.Series("x", [Decimal("1.20"), Decimal("3.40")], dtype=pl.Decimal(10, 2)), {}, "decimal32(2, 1)", "Decimal(precision=2, scale=1)", id="decimal_scale_by_gcd"),
    pytest.param(pl.Series("x", [Decimal("1.00"), Decimal("300.00")], dtype=pl.Decimal(10, 2)), {}, "uint16", "UInt16", id="decimal_to_integer"),
    pytest.param(pl.Series("x", [123.45, 99.99]), {}, "decimal32(5, 2)", "Decimal(precision=5, scale=2)", id="float_to_decimal32"),
    pytest.param(pl.Series("x", [1234567.891, 2.5]), {}, "decimal64(10, 3)", "Decimal(precision=10, scale=3)", id="float_to_decimal64"),
    pytest.param(pl.Series("x", [1.0, 2.0, 300.0]), {}, "uint16", "UInt16", id="whole_floats"),
    pytest.param(pl.Series("x", [0.0009765625, 0.5]), {}, "float", "Float32", id="float32_beats_decimal64"),
    pytest.param(pl.Series("x", [0.5, 0.25]), {}, "decimal32(2, 2)", "Decimal(precision=2, scale=2)", id="decimal32_beats_float32_on_tie"),
    pytest.param(pl.Series("x", [1.5, float("nan")]), {}, "float", "Float32", id="nan_keeps_a_float"),
    pytest.param(pl.Series("x", [0.1, float("nan")]), {}, "double", "Float64", id="original_float64"),
    pytest.param(pl.Series("x", [1e10, 1e-9]), {}, "double", "Float64", id="float64_needing_decimal128_kept"),
    pytest.param(pl.Series("x", ["007", "12"]), {}, "string", "String", id="leading_zero_stays_string"),
    pytest.param(pl.Series("x", ["1.50", "2.2"]), {}, "decimal32(2, 1)", "Decimal(precision=2, scale=1)", id="string_decimal"),
    pytest.param(pl.Series("x", ["1234567890.1", "0.00000012345"]), {}, "double", "Float64", id="varying_places_float64"),
    pytest.param(pl.Series("x", ["1.5", "0.00000000000000000012"]), {}, "float", "Float32", id="varying_places_float32"),
    pytest.param(pl.Series("x", ["12345678901234567.8", "0.12"]), {}, "decimal128(19, 2)", "Decimal(precision=19, scale=2)", id="over_15_significant_digits"),
    pytest.param(pl.Series("x", ["12345678901234567.89", "0.12"]), {}, "decimal128(19, 2)", "Decimal(precision=19, scale=2)", id="fixed_places_decimal128"),
    pytest.param(pl.Series("x", ["true", "False", None]), {}, "bool", "Boolean", id="boolean_pair_default"),
    pytest.param(pl.Series("x", ["Y", "n", "y"]), {"boolean_pairs": (("true", "false"), ("y", "n"))}, "bool", "Boolean", id="boolean_pair_y_n"),
    pytest.param(pl.Series("x", ["2024-01-05", "2024-02-29"]), {}, "date32[day]", "Date", id="iso_date"),
    pytest.param(pl.Series("x", ["10:00", "23:59:30"]), {}, "time32[s]", "Time", id="iso_time"),
    pytest.param(pl.Series("x", ["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"]), {}, "timestamp[ms]", "Datetime(time_unit='ms', time_zone=None)", id="iso_naive"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00:00.000", "2024-01-05T11:30:00.000"]), {}, "timestamp[s]", "Datetime(time_unit='ms', time_zone=None)", id="iso_zero_fraction"),
    pytest.param(pl.Series("x", ["2024-01-05T00:00:00", "2024-01-06 00:00"]), {}, "date32[day]", "Date", id="iso_midnights"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-06T11:00:00+05:00"]), {}, "timestamp[s, tz=+05:00]", "Datetime(time_unit='ms', time_zone='+05:00')", id="iso_fixed_offset"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00Z", "2024-01-05T11:00+00:00"]), {}, "timestamp[s, tz=UTC]", "Datetime(time_unit='ms', time_zone='UTC')", id="iso_utc"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]), {}, OFFSETS, "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})", id="iso_varying_offsets"),
    pytest.param(pl.Series("x", LONDON, dtype=pl.Datetime("us")), {}, "date32[day]", "Date", id="naive_midnights"),
    pytest.param(pl.Series("x", LONDON, dtype=pl.Datetime("us")).dt.replace_time_zone("Europe/London"), {}, "timestamp[s, tz=Europe/London]", "Datetime(time_unit='ms', time_zone='Europe/London')", id="tz_aware_midnights"),
    pytest.param(pl.Series("x", [datetime(2024, 1, 5, 10, 0, 0, 120_000)], dtype=pl.Datetime("us")), {}, "timestamp[ms]", "Datetime(time_unit='ms', time_zone=None)", id="datetime_unit_by_gcd"),
    pytest.param(pl.Series("x", [timedelta(seconds=5), timedelta(minutes=1)], dtype=pl.Duration("us")), {}, "duration[s]", "Duration(time_unit='ms')", id="duration_unit_by_gcd"),
    pytest.param(pl.Series("x", [time(10, 0), time(11, 30, 15)]), {}, "time32[s]", "Time", id="time_unit_by_gcd"),
    pytest.param(pl.Series("x", [[1], [2], None]), {}, "uint8", "UInt8", id="single_item_lists"),
    pytest.param(pl.Series("x", [[2], [None], None]), {}, "list<item: uint8>", "List(UInt8)", id="null_lists_and_elements_stay_lists"),
    pytest.param(pl.Series("x", [[1], [None], None]), {}, "list<item: bool>", "List(Boolean)", id="list_of_0_1_becomes_list_of_bool"),
    pytest.param(pl.Series("x", [[1, 2], [3]]), {}, "list<item: uint8>", "List(UInt8)", id="large_list_to_list"),
    pytest.param(pl.Series("x", [None, None], dtype=pl.String), {}, "null", "Null", id="all_null"),
]


@pytest.mark.parametrize("s, params, arrow_type, polars_type", KNOWN)
def test_known_answers(s, params, arrow_type, polars_type):
    r = rec(s, **params)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == (arrow_type, polars_type)
    assert r["rec_nullable"] == (s.null_count() > 0)


def test_lossy_formatting():
    assert rec(pl.Series("x", ["1.50", "2.2"]))["rec_lossy_formatting"] is True
    assert rec(pl.Series("x", ["1.5", "2.2"]))["rec_lossy_formatting"] is False
    negative_zero = rec(pl.Series("x", [-0.0, 1.0, 1.0]))
    assert (negative_zero["rec_arrow_type"], negative_zero["rec_lossy_formatting"]) == ("bool", True)
    assert rec(pl.Series("x", [0, 1]))["rec_lossy_formatting"] is False
    assert rec(pl.Series("x", ["true", "False"]))["rec_lossy_formatting"] is True


def test_candidates_report_rule_evidence_and_outcomes():
    r = rec(pl.Series("x", [1e10, 1e-9]))
    c = by_type(r)
    assert c["double"]["outcome"] == "chosen"
    assert c["decimal128(20, 9)"]["outcome"] == "not_tried"
    assert "max_frac_digits=9" in c["decimal128(20, 9)"]["evidence"]
    tried = [x["projected_population_bytes"] for x in r["rec_candidates"] if x["outcome"] != "rejected"]
    assert tried == sorted(tried)


def test_failed_cast_falls_back_to_the_next_candidate():
    r = rec(pl.Series("x", ["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"]))
    assert r["rec_arrow_type"] == "string"
    failed = by_type(r)["timestamp[ns]"]
    assert failed["outcome"] == "failed" and failed["reason"]


def test_dictionary_polars_types():
    s = pl.Series("x", ["a", "b"] * 50)
    exact = rec(s, population_rows=100)
    assert exact["rec_arrow_type"] == "dictionary<values=string, indices=uint8, ordered=0>"
    assert exact["rec_polars_type"] == "Enum(categories=['a', 'b'])"
    assert rec(s)["rec_polars_type"] == 'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))'


@pytest.mark.parametrize(
    "d, arrow_key",
    [
        pytest.param(255, "uint8", id="255"),
        pytest.param(256, "uint8", id="256"),
        pytest.param(257, "uint16", id="257"),
        pytest.param(65_536, "uint16", id="65536", marks=pytest.mark.slow),
        pytest.param(65_537, "uint32", id="65537", marks=pytest.mark.slow),
    ],
)
def test_dictionary_key_widths(d, arrow_key):
    s = pl.Series("x", [f"v{i:05d}" for i in range(d)] * 2 if d > 1_000 else [f"v{i:05d}" for i in range(d)] * 40)
    r = rec(s, population_rows=s.len(), categorical_threshold=100_000)
    assert r["rec_arrow_type"] == f"dictionary<values=string, indices={arrow_key}, ordered=0>"
    # Polars reserves one key code: its own width (checked by casting) must match the measured layout.
    polars = s.cast(pl.Enum(s.unique(maintain_order=True).to_list()))
    assert r["rec_polars_size_bytes"] == _sizes.column_sizes(polars, 1)["size_polars_bytes"]


def test_dictionary_gate_rejects_above_threshold():
    r = rec(pl.Series("x", ["a", "b"] * 500), population_rows=1_000, categorical_threshold=1)
    assert r["rec_arrow_type"] == "string"
    dictionary = next(c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary"))
    assert dictionary["outcome"] == "rejected" and "categorical_threshold=1" in dictionary["reason"]


def test_population_projection_chooses_plain_over_dictionary():
    # 600 singletons, 200 doubletons, 200 values × 5: n = 2,000, d = 1,000 (d/n = 0.5, so
    # Chao1 — est_high ≈ 2,115 — sizes the dictionary); 30-byte values.
    ids = [f"id-{i:027d}" for i in range(1_000)]
    s = pl.Series("x", ids[:600] + ids[600:800] * 2 + ids[800:] * 5)
    r = rec(s)
    plain = by_type(r)["string"]
    dictionary = next(c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary"))
    assert dictionary["predicted_bytes"] < plain["predicted_bytes"]  # smaller on the frame …
    assert dictionary["projected_population_bytes"] > plain["projected_population_bytes"]  # … larger in the population
    assert r["rec_arrow_type"] == "string"
