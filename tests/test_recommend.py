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
    pytest.param(pl.Series("x", [[1], [None], [2]]), {}, "uint8", "UInt8", id="single_item_lists_with_null_items"),
    pytest.param(pl.Series("x", [["a"], [None], ["b"]] * 20), {}, "dictionary<values=string, indices=uint8, ordered=0>", 'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))', id="single_item_string_lists_with_null_items"),
    pytest.param(pl.Series("x", [[2], [None], None]), {}, "list<item: uint8>", "List(UInt8)", id="null_lists_and_elements_stay_lists"),
    pytest.param(pl.Series("x", [[1], [None], None]), {}, "list<item: bool>", "List(Boolean)", id="list_of_0_1_becomes_list_of_bool"),
    pytest.param(pl.Series("x", [[1, 2], [3]]), {}, "list<item: uint8>", "List(UInt8)", id="large_list_to_list"),
    pytest.param(pl.Series("x", [None, None], dtype=pl.String), {}, "null", "Null", id="all_null"),
    pytest.param(pl.Series("x", [b"ab", None, b"cde"]), {}, "binary", "Binary", id="binary"),
    pytest.param(pl.Series("x", [[1, 2], [3, 4], None], dtype=pl.Array(pl.Int64, 2)), {}, "fixed_size_list<item: uint8>[2]", "Array(UInt8, shape=(2,))", id="array_keeps_fixed_size"),
    pytest.param(pl.Series("x", ["123", "45", "-7"]), {}, "int8", "Int8", id="string_integer"),
]


@pytest.mark.parametrize("s, params, arrow_type, polars_type", KNOWN)
def test_known_answers(s, params, arrow_type, polars_type):
    r = rec(s, **params)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == (arrow_type, polars_type)
    # rec_nullable: the recommended array has nulls — a list → scalar recast turns a [null] item into a null row.
    assert r["rec_nullable"] == (_unlist(s, arrow_type).null_count() > 0)


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
    assert failed["outcome"] == "failed"
    assert "row 0" in failed["reason"] and "2300-01-01T00:00:00.123456789" in failed["reason"]


NUMERIC = re.compile(r"(u?int\d+|decimal\d+|float|double|halffloat)$|decimal\d+\(")


def test_leading_zero_string_has_no_numeric_candidate():
    r = rec(pl.Series("x", ["007", "12"]))
    assert not [c["arrow_type"] for c in r["rec_candidates"] if NUMERIC.match(c["arrow_type"])]


def test_over_15_significant_digits_has_no_float_candidate():
    r = rec(pl.Series("x", ["12345678901234567.8", "0.12"]))
    assert not {"float", "double", "halffloat"} & set(by_type(r))


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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Oracles (single implementation: no reference agreement)

_SIMPLE = {
    "null": pa.null(), "bool": pa.bool_(), "float": pa.float32(), "double": pa.float64(),
    "string": pa.string(), "large_string": pa.large_string(), "binary": pa.binary(), "large_binary": pa.large_binary(),
    "date32[day]": pa.date32(),
    **{n: getattr(pa, n)() for n in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")},
}


def pa_type(name: str) -> pa.DataType:
    """pyarrow type from the `str(type)` spellings recommend emits."""
    if name in _SIMPLE:
        return _SIMPLE[name]
    if m := re.fullmatch(r"decimal(32|64|128)\((\d+), (\d+)\)", name):
        return getattr(pa, f"decimal{m[1]}")(int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(time32|time64|duration)\[(\w+)\]", name):
        return getattr(pa, m[1])(m[2])
    if m := re.fullmatch(r"timestamp\[(\w+)(?:, tz=(.+))?\]", name):
        return pa.timestamp(m[1], m[2])
    if m := re.fullmatch(r"large_list<item: (.+)>", name):
        return pa.large_list(pa_type(m[1]))
    if m := re.fullmatch(r"list<item: (.+)>", name):
        return pa.list_(pa_type(m[1]))
    if m := re.fullmatch(r"fixed_size_list<item: (.+)>\[(\d+)\]", name):
        return pa.list_(pa_type(m[1]), int(m[2]))
    if m := re.fullmatch(r"dictionary<values=(.+), indices=(\w+), ordered=0>", name):
        return pa.dictionary(pa_type(m[2]), pa_type(m[1]))
    raise NotImplementedError(name)


def pl_dtype(name: str) -> pl.DataType:
    """Polars dtype from its Python `str(dtype)` spelling."""
    return eval(name, {"__builtins__": {}}, {"pl": pl, **{k: getattr(pl, k) for k in dir(pl) if k[:1].isupper()}})


def _string_source(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.List, pl.Array)):
        return _string_source(dtype.inner)
    return isinstance(dtype, (pl.String, pl.Categorical, pl.Enum))


def _outer_chosen(r: dict) -> dict:
    """The chosen candidate for the column itself (list columns also report the inner level's, rule "inner: …")."""
    return next(c for c in r["rec_candidates"] if c["outcome"] == "chosen" and not c["rule"].startswith("inner: "))


def _unlist(s: pl.Series, target: str) -> pl.Series:
    """A list → scalar recommendation (every row holds at most one item) is a first-item extraction, not a cast."""
    if isinstance(s.dtype, (pl.List, pl.Array)) and "list" not in target:
        return s.list.first() if isinstance(s.dtype, pl.List) else s.arr.first()
    return s


def list_frame(n: int = 600) -> pl.DataFrame:
    """List / Array columns: single-item ones become scalars (checked through `_unlist`), the rest stay lists."""
    return pl.DataFrame([
        pl.Series("one_int", [[i % 200] if i % 7 else None for i in range(n)], dtype=pl.List(pl.Int64)),
        pl.Series("one_str", [[f"s{i % 9}"] if i % 5 else None for i in range(n)], dtype=pl.List(pl.String)),
        pl.Series("one_float", [[i / 4] for i in range(n)], dtype=pl.List(pl.Float64)),
        pl.Series("arr1", [[i % 50] if i % 3 else None for i in range(n)], dtype=pl.Array(pl.Int64, 1)),
        pl.Series("arr2", [[i % 50, i % 3] for i in range(n)], dtype=pl.Array(pl.Int64, 2)),
        pl.Series("multi", [[i, i + 1][: 1 + i % 2] for i in range(n)], dtype=pl.List(pl.Int64)),
    ])


ORACLE_FRAMES = [
    pytest.param(lambda: {"mixed": describe_mixed(2_000)}, id="mixed"),
    pytest.param(lambda: {"lists": list_frame()}, id="lists"),
    pytest.param(lambda: {"strings": stringified(describe_mixed(500))}, id="stringified"),
    pytest.param(lambda: {"large": pl.read_ipc(LARGE)}, id="large", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("make", ORACLE_FRAMES)
def test_sizes_match_pyarrow_and_polars_casts(make):
    frames = make()
    result = run(impl(), frames)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        s = frames[r["df_a"]][r["col_a"]].rechunk()
        chosen = _outer_chosen(r)
        assert chosen["predicted_bytes"] == r["rec_arrow_size_bytes"], r["col_a"]
        original = chosen["rule"].endswith("original")
        if original:
            assert (r["rec_arrow_size_bytes"], r["rec_polars_size_bytes"]) == (r["size_bytes"], r["size_polars_bytes"])
            assert r["rec_polars_type"] == str(s.dtype), r["col_a"]  # the Polars side is an identity cast
            # Kept-original columns go through the same identity cast and measurement below. The only skip is a
            # type pa_type cannot spell (e.g. a generic struct<...>), where there is no pyarrow type to cast to.
            try:
                pa_type(r["rec_arrow_type"])
            except NotImplementedError:
                continue
        skippable = _string_source(s.dtype) and not original
        try:
            arrow = _sizes._to_arrow(_unlist(s, r["rec_arrow_type"]).rechunk(), pl.CompatLevel.oldest())
            arrow = arrow.cast(pa_type(r["rec_arrow_type"]), safe=False)
            polars = s if original else _unlist(s, r["rec_arrow_type"]).cast(pl_dtype(r["rec_polars_type"])).rechunk()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pl.exceptions.PolarsError, NotImplementedError):
            assert skippable, f"{r['col_a']}: pyarrow/Polars cannot cast a non-string column to {r['rec_arrow_type']}"
            continue
        if arrow.null_count != s.null_count() or polars.null_count() != s.null_count():
            assert skippable, f"{r['col_a']}: the library cast lost values"
            continue
        level = 1
        assert _sizes.ipc_body_bytes(arrow, None) == r["rec_arrow_size_bytes"], r["col_a"]
        assert _sizes.ipc_body_bytes(arrow, level) == approx(r["rec_arrow_size_zstd_bytes"], rel=0.01, abs=16), r["col_a"]
        native = _sizes.column_sizes(polars, level)
        assert native["size_polars_bytes"] == r["rec_polars_size_bytes"], r["col_a"]
        assert native["size_polars_zstd_bytes"] == approx(r["rec_polars_size_zstd_bytes"], rel=0.01, abs=16), r["col_a"]
        checked += 1
    assert checked >= min(10, result.height)


@pytest.mark.parametrize("population_rows", [None, 1_000_000])
def test_rust_cardinality_matches_python_estimators(population_rows):
    frames = {"mixed": describe_mixed(2_000), "strings": stringified(describe_mixed(500))}
    result = run(impl(), frames, population_rows=population_rows)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        for c in r["rec_candidates"]:
            m = re.search(r"c=(\S+) from (est_high|est_cardinality)", c["evidence"] or "")
            if m is None:
                continue
            prefix = "inner_" if c["rule"].startswith("inner: ") else ""
            # Rust floors the cardinality at the observed distinct count.
            expected = max(r[f"{prefix}{m[2]}"], r[f"{prefix}n_unique"])
            assert float(m[1]) == approx(expected, rel=1e-9), (r["col_a"], c["rule"])
            checked += 1
    assert checked >= 5
