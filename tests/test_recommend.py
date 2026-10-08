"""
recommend accuracy tests — OneShotRecommender.

Oracles: hand-worked known answers; the pyarrow and Polars casts of each column to
the recommended types, measured by the pyarrow IPC oracle; predicted = measured;
the Python cardinality estimators. Accuracy only — nothing here is timed.
"""

import re
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from analytics.describe import _sizes
from analytics.recommend import OneShotRecommender, StreamingRecommender
from datagen import describe_mixed, stringified

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


def rec(s: pl.Series, **params) -> dict:
    """recommend one Series; its result row as a dict."""
    return OneShotRecommender(**params).add(s.to_frame()).result().row(0, named=True)


def recommend_frames(frames: dict, **params) -> pl.DataFrame:
    """Each frame recommended on its own; `frame` names its source."""
    return pl.concat(
        OneShotRecommender(**params).add(f).result().with_columns(frame=pl.lit(name))
        for name, f in frames.items()
    )


def by_type(r: dict) -> dict:
    return {c["arrow_type"]: c for c in r["rec_candidates"]}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract

STREAMING_ONLY = ("first_row", "n_sampled_rows", "n_sampled_blocks")
F64_KEPT = [3.3e200 / 7, 9.9e200 / 7]  # beyond float32, ~200 integer digits: kept


def test_schema_is_streamings_without_the_streaming_columns():
    frame = describe_mixed(300)
    out = OneShotRecommender().add(frame).result()
    streamed = StreamingRecommender().add(frame).result()
    want = [(k, v) for k, v in streamed.schema.items() if k not in STREAMING_ONLY]
    assert list(out.schema.items()) == want
    assert out.to_arrow().num_rows == out.height
    assert out["column"].to_list() == frame.columns
    computed = out.filter(pl.col("status") == "computed")
    assert computed["rec_arrow_type"].null_count() == 0
    assert computed["rec_polars_type"].null_count() == 0


def test_result_before_add_is_empty_with_the_full_schema():
    empty = OneShotRecommender().result()
    assert empty.height == 0
    full = OneShotRecommender().add(pl.DataFrame({"a": [1]})).result()
    assert empty.schema == full.schema


def test_a_second_add_raises_and_keeps_the_first():
    rec = OneShotRecommender().add(pl.DataFrame({"a": [1, 2]}))
    second = pl.DataFrame({"a": [3]})
    wide = pl.DataFrame({"w": pl.Series([1], dtype=pl.Int128)})
    with pytest.raises(ValueError, match="already been added"):
        rec.add(second)
    with pytest.raises(ValueError, match="already been added"):
        rec.add(wide)
    assert rec.result()["n_rows"].to_list() == [2]


def test_result_is_repeatable():
    rec = OneShotRecommender().add(describe_mixed(100))
    assert rec.result().equals(rec.result())


@pytest.mark.parametrize(
    "convert, string_dtype",
    [
        (lambda f: f, "string_view"),
        (lambda f: f.lazy(), "string_view"),
        (lambda f: f.to_arrow(), "large_string"),
        (
            lambda f: pa.RecordBatchReader.from_batches(
                f.to_arrow().schema, f.to_arrow().to_batches(max_chunksize=2)
            ),
            "large_string",
        ),
    ],
    ids=["polars", "lazy", "pyarrow_table", "record_batch_reader"],
)
def test_inputs(convert, string_dtype):
    frame = pl.DataFrame({"a": [0, 5, 7, 9], "s": ["x", "y", "x", None]})
    want = OneShotRecommender().add(frame).result()
    out = OneShotRecommender().add(convert(frame)).result()
    assert out["n_rows"].to_list() == [4, 4]
    # `dtype` names the Arrow type received: py-polars exports String as string_view.
    assert out.drop("dtype").equals(want.drop("dtype"))
    assert out["dtype"][0] == "int64"
    assert out["dtype"][1] == string_dtype


def test_zero_row_frame():
    frame = describe_mixed(50)
    want = OneShotRecommender().add(frame).result()
    out = OneShotRecommender().add(frame.clear()).result()
    assert out.schema == want.schema
    assert out.height == frame.width
    assert out["n_rows"].to_list() == [0] * frame.width
    # No column is ineligible for being empty: every one is computed and recommended.
    assert out["status"].to_list() == ["computed"] * frame.width
    assert out["rec_arrow_type"].null_count() == 0


def test_lazy_input_with_an_ineligible_column():
    frame = pl.DataFrame(
        {
            "a": [0, 5, 7, 9],
            "w": pl.Series([1, 2, 3, 4], dtype=pl.Int128),
            "s": ["x", "y", "x", None],
        }
    )
    want = OneShotRecommender().add(frame).result()
    out = OneShotRecommender().add(frame.lazy()).result()
    assert out.equals(want)
    # Ineligible columns come first (spec §3), as in streaming.
    assert out["column"].to_list() == ["w", "a", "s"]
    assert out["status"].to_list() == ["ineligible", "computed", "computed"]


def test_ineligible_columns_are_listed():
    frame = pl.DataFrame(
        {
            "w": pl.Series([1, 2], dtype=pl.Int128),
            "a": [1, 2],
            "n": [None, None],
            "obj": pl.Series([object(), object()], dtype=pl.Object),
        }
    )
    rows = {
        r["column"]: r
        for r in OneShotRecommender().add(frame).result().iter_rows(named=True)
    }
    w = rows["w"]
    assert (w["status"], w["dtype"], w["n_rows"], w["n_null"]) == (
        "ineligible",
        "Int128",
        2,
        None,
    )
    assert w["rec_arrow_type"] is None
    assert (rows["obj"]["status"], rows["obj"]["dtype"]) == ("ineligible", "Object")
    n = rows["n"]
    assert (n["status"], n["dtype"], n["n_null"]) == ("ineligible", "null", 2)
    assert (rows["a"]["status"], rows["a"]["dtype"]) == ("computed", "int64")


def test_kept_original_names_its_polars_type():
    out = OneShotRecommender().add(pl.DataFrame({"f": F64_KEPT})).result()
    r = out.row(0, named=True)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == ("double", "Float64")


def test_constructor_validates_boolean_pairs():
    with pytest.raises(ValueError):
        OneShotRecommender(boolean_pairs=(("yes", "YES"),))


@pytest.mark.parametrize("pairs", [(("y",),), ((1, 2),), (("a", "b", "c"),)])
def test_boolean_pairs_must_be_pairs(pairs):
    with pytest.raises(ValueError, match="boolean_pairs must be pairs of two strings"):
        OneShotRecommender(boolean_pairs=pairs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers

LONDON = [datetime(2024, 1, 5), datetime(2024, 1, 6)]
DICT8 = "dictionary<values=string, indices=uint8, ordered=0>"
UNIQUE = [f"value-{i:03d}" for i in range(200)]
OFFSETS = (
    "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"
)

KNOWN = [
    pytest.param(pl.Series("x", [0, 1, 1, None]), {}, "bool", "Boolean", id="int_0_1"),
    pytest.param(
        pl.Series("x", [0, 5, 127]), {}, "uint8", "UInt8", id="uint_before_int_on_tie"
    ),
    pytest.param(pl.Series("x", [-5, 100]), {}, "int8", "Int8", id="int8"),
    pytest.param(pl.Series("x", [-200, 5]), {}, "int16", "Int16", id="int16"),
    pytest.param(
        pl.Series("x", [Decimal("1.20"), Decimal("3.40")], dtype=pl.Decimal(10, 2)),
        {},
        "decimal32(2, 1)",
        "Decimal(precision=2, scale=1)",
        id="decimal_scale_by_gcd",
    ),
    pytest.param(
        pl.Series("x", [Decimal("1.00"), Decimal("300.00")], dtype=pl.Decimal(10, 2)),
        {},
        "uint16",
        "UInt16",
        id="decimal_to_integer",
    ),
    pytest.param(
        pl.Series("x", [123.45, 99.99]),
        {},
        "decimal32(5, 2)",
        "Decimal(precision=5, scale=2)",
        id="float_to_decimal32",
    ),
    pytest.param(
        pl.Series("x", [1234567.891, 2.5]),
        {},
        "decimal64(10, 3)",
        "Decimal(precision=10, scale=3)",
        id="float_to_decimal64",
    ),
    pytest.param(
        pl.Series("x", [1.0, 2.0, 300.0]), {}, "uint16", "UInt16", id="whole_floats"
    ),
    pytest.param(
        pl.Series("x", [0.0009765625, 0.5]),
        {},
        "float",
        "Float32",
        id="float32_beats_decimal64",
    ),
    pytest.param(
        pl.Series("x", [0.5, 0.25]),
        {},
        "decimal32(2, 2)",
        "Decimal(precision=2, scale=2)",
        id="decimal32_beats_float32_on_tie",
    ),
    pytest.param(
        pl.Series("x", [1.5, float("nan")]),
        {},
        "float",
        "Float32",
        id="nan_keeps_a_float",
    ),
    pytest.param(
        pl.Series("x", [0.1, float("nan")]),
        {},
        "double",
        "Float64",
        id="original_float64",
    ),
    pytest.param(
        pl.Series("x", [1e10, 1e-9]),
        {},
        "double",
        "Float64",
        id="float64_needing_decimal128_kept",
    ),
    pytest.param(
        pl.Series("x", ["007", "12"]),
        {},
        "string",
        "String",
        id="leading_zero_stays_string",
    ),
    pytest.param(
        pl.Series("x", ["1.50", "2.2"]),
        {},
        "decimal32(2, 1)",
        "Decimal(precision=2, scale=1)",
        id="string_decimal",
    ),
    pytest.param(
        pl.Series("x", ["1234567890.1", "0.00000012345"]),
        {},
        "double",
        "Float64",
        id="varying_places_float64",
    ),
    pytest.param(
        pl.Series("x", ["1.5", "0.00000000000000000012"]),
        {},
        "float",
        "Float32",
        id="varying_places_float32",
    ),
    pytest.param(
        pl.Series("x", ["12345678901234567.8", "0.12"]),
        {},
        "decimal128(19, 2)",
        "Decimal(precision=19, scale=2)",
        id="over_15_significant_digits",
    ),
    pytest.param(
        pl.Series("x", ["12345678901234567.89", "0.12"]),
        {},
        "decimal128(19, 2)",
        "Decimal(precision=19, scale=2)",
        id="fixed_places_decimal128",
    ),
    pytest.param(
        pl.Series("x", ["true", "False", None]),
        {},
        "bool",
        "Boolean",
        id="boolean_pair_default",
    ),
    pytest.param(
        pl.Series("x", ["Y", "n", "y"]),
        {"boolean_pairs": (("true", "false"), ("y", "n"))},
        "bool",
        "Boolean",
        id="boolean_pair_y_n",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05", "2024-02-29"]),
        {},
        "date32[day]",
        "Date",
        id="iso_date",
    ),
    pytest.param(
        pl.Series("x", ["10:00", "23:59:30"]), {}, "time32[s]", "Time", id="iso_time"
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"]),
        {},
        "timestamp[ms]",
        "Datetime(time_unit='ms', time_zone=None)",
        id="iso_naive",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05T10:00:00.000", "2024-01-05T11:30:00.000"]),
        {},
        "timestamp[s]",
        "Datetime(time_unit='ms', time_zone=None)",
        id="iso_zero_fraction",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05T00:00:00", "2024-01-06 00:00"]),
        {},
        "date32[day]",
        "Date",
        id="iso_midnights",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-06T11:00:00+05:00"]),
        {},
        "timestamp[s, tz=+05:00]",
        "Datetime(time_unit='ms', time_zone='+05:00')",
        id="iso_fixed_offset",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05T10:00Z", "2024-01-05T11:00+00:00"]),
        {},
        "timestamp[s, tz=UTC]",
        "Datetime(time_unit='ms', time_zone='UTC')",
        id="iso_utc",
    ),
    pytest.param(
        pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]),
        {},
        OFFSETS,
        "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})",
        id="iso_varying_offsets",
    ),
    pytest.param(
        pl.Series("x", LONDON, dtype=pl.Datetime("us")),
        {},
        "date32[day]",
        "Date",
        id="naive_midnights",
    ),
    pytest.param(
        pl.Series("x", LONDON, dtype=pl.Datetime("us")).dt.replace_time_zone(
            "Europe/London"
        ),
        {},
        "timestamp[s, tz=Europe/London]",
        "Datetime(time_unit='ms', time_zone='Europe/London')",
        id="tz_aware_midnights",
    ),
    pytest.param(
        pl.Series(
            "x", [datetime(2024, 1, 5, 10, 0, 0, 120_000)], dtype=pl.Datetime("us")
        ),
        {},
        "timestamp[ms]",
        "Datetime(time_unit='ms', time_zone=None)",
        id="datetime_unit_by_gcd",
    ),
    pytest.param(
        pl.Series(
            "x", [timedelta(seconds=5), timedelta(minutes=1)], dtype=pl.Duration("us")
        ),
        {},
        "duration[s]",
        "Duration(time_unit='ms')",
        id="duration_unit_by_gcd",
    ),
    pytest.param(
        pl.Series("x", [time(10, 0), time(11, 30, 15)]),
        {},
        "time32[s]",
        "Time",
        id="time_unit_by_gcd",
    ),
    pytest.param(
        pl.Series("x", [[1], [2], None]), {}, "uint8", "UInt8", id="single_item_lists"
    ),
    pytest.param(
        pl.Series("x", [[1], [None], [2]]),
        {},
        "uint8",
        "UInt8",
        id="single_item_lists_with_null_items",
    ),
    pytest.param(
        pl.Series("x", [["a"], [None], ["b"]] * 20),
        {},
        "dictionary<values=string, indices=uint8, ordered=0>",
        'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))',
        id="single_item_string_lists_with_null_items",
    ),
    pytest.param(
        pl.Series("x", [[2], [None], None]),
        {},
        "list<item: uint8>",
        "List(UInt8)",
        id="null_lists_and_elements_stay_lists",
    ),
    pytest.param(
        pl.Series("x", [[1], [None], None]),
        {},
        "list<item: bool>",
        "List(Boolean)",
        id="list_of_0_1_becomes_list_of_bool",
    ),
    pytest.param(
        pl.Series("x", [[1, 2], [3]]),
        {},
        "list<item: uint8>",
        "List(UInt8)",
        id="large_list_to_list",
    ),
    pytest.param(
        pl.Series("x", [None, None], dtype=pl.String), {}, "null", "Null", id="all_null"
    ),
    pytest.param(
        pl.Series("x", [b"ab", None, b"cde"]), {}, "binary", "Binary", id="binary"
    ),
    pytest.param(
        pl.Series("x", [[1, 2], [3, 4], None], dtype=pl.Array(pl.Int64, 2)),
        {},
        "fixed_size_list<item: uint8>[2]",
        "Array(UInt8, shape=(2,))",
        id="array_keeps_fixed_size",
    ),
    pytest.param(
        pl.Series("x", ["123", "45", "-7"]), {}, "int8", "Int8", id="string_integer"
    ),
    # Categorical / Enum sources are already dictionaries: step 2 narrows the key or drops the dictionary (§5.2).
    # Categorical exports dictionary<uint32, large_string> (400 + 24 + 8 bytes); two values → UInt8 keys (104 + 16 + 8).
    pytest.param(
        pl.Series("x", ["a", "b"] * 50, dtype=pl.Categorical),
        {},
        DICT8,
        'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))',
        id="categorical_narrows_key",
    ),
    # Enum source, two values: Schnabel's est_high (~7.4) projects the narrower dictionary larger than the original, so the Enum is kept.
    pytest.param(
        pl.Series("x", ["a", "b"] * 50, dtype=pl.Enum(["a", "b"])),
        {},
        "dictionary<values=large_string, indices=uint8, ordered=0>",
        "Enum(categories=['a', 'b'])",
        id="enum_kept_when_cardinality_is_uncertain",
    ),
    # 200 singletons: Chao1 est_high ≈ 26,438 > categorical_threshold → dictionary rejected; Utf8 (2,608) beats the original (3,608).
    pytest.param(
        pl.Series("x", UNIQUE, dtype=pl.Enum(UNIQUE)),
        {},
        "string",
        "String",
        id="enum_of_singletons_drops_dictionary",
    ),
]


@pytest.mark.parametrize("s, params, arrow_type, polars_type", KNOWN)
def test_known_answers(s, params, arrow_type, polars_type):
    r = rec(s, **params)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == (arrow_type, polars_type)
    # rec_nullable: the recommended array has nulls — a list → scalar recast turns a [null] item into a null row.
    assert r["rec_nullable"] == (_unlist(s, arrow_type).null_count() > 0)


def null_lists_holding_values(lists: list) -> pl.Series:
    """`lists` with row 1 set null by pl.when/otherwise: the null row keeps its values behind the offsets."""
    s = pl.Series("x", lists)
    mask = pl.Series([i != 1 for i in range(len(lists))])
    out = pl.select(pl.when(mask).then(s).otherwise(None).alias("x")).to_series()
    offsets = out.to_arrow().offsets.to_pylist()
    assert offsets[2] > offsets[1], offsets  # the null row still spans values
    return out


@pytest.mark.parametrize(
    "lists, arrow_type, polars_type",
    [
        pytest.param(
            [[1, 2], [3, 4], [5]], "list<item: uint8>", "List(UInt8)", id="list"
        ),
        pytest.param([[1], [2], [3]], "uint8", "UInt8", id="scalar"),
        # inner ["a", "a"]: Utf8 (16 + 8) ties Dictionary (keys 8 + offsets 8 + values 8); Dictionary wins on rank
        pytest.param(
            [["a"], ["b"], ["a"]],
            "dictionary<values=string, indices=uint8, ordered=0>",
            'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))',
            id="strings",
        ),
    ],
)
def test_null_lists_holding_values_are_narrowed(lists, arrow_type, polars_type):
    s = null_lists_holding_values(lists)
    r = rec(s)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == (arrow_type, polars_type)
    assert r["rec_nullable"] is True
    assert _outer_chosen(r)["predicted_bytes"] == r["rec_arrow_size_bytes"]


def test_lossy_formatting():
    assert rec(pl.Series("x", ["1.50", "2.2"]))["rec_lossy_formatting"] is True
    assert rec(pl.Series("x", ["1.5", "2.2"]))["rec_lossy_formatting"] is False
    negative_zero = rec(pl.Series("x", [-0.0, 1.0, 1.0]))
    assert (negative_zero["rec_arrow_type"], negative_zero["rec_lossy_formatting"]) == (
        "bool",
        True,
    )
    assert rec(pl.Series("x", [0, 1]))["rec_lossy_formatting"] is False
    assert rec(pl.Series("x", ["true", "False"]))["rec_lossy_formatting"] is True


def test_candidates_report_rule_evidence_and_outcomes():
    r = rec(pl.Series("x", [1e10, 1e-9]))
    c = by_type(r)
    assert c["double"]["outcome"] == "chosen"
    assert c["decimal128(20, 9)"]["outcome"] == "not_tried"
    assert "max_frac_digits=9" in c["decimal128(20, 9)"]["evidence"]
    tried = [
        x["projected_population_bytes"]
        for x in r["rec_candidates"]
        if x["outcome"] != "rejected"
    ]
    assert tried == sorted(tried)


def test_failed_cast_falls_back_to_the_next_candidate():
    r = rec(pl.Series("x", ["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"]))
    assert r["rec_arrow_type"] == "string"
    failed = by_type(r)["timestamp[ns]"]
    assert failed["outcome"] == "failed"
    assert "row 0" in failed["reason"]
    assert "2300-01-01T00:00:00.123456789" in failed["reason"]


NUMERIC = re.compile(r"(u?int\d+|decimal\d+|float|double|halffloat)$|decimal\d+\(")


def test_leading_zero_string_has_no_numeric_candidate():
    r = rec(pl.Series("x", ["007", "12"]))
    assert not [
        c["arrow_type"] for c in r["rec_candidates"] if NUMERIC.match(c["arrow_type"])
    ]


def test_over_15_significant_digits_has_no_float_candidate():
    r = rec(pl.Series("x", ["12345678901234567.8", "0.12"]))
    assert not {"float", "double", "halffloat"} & set(by_type(r))


def test_dictionary_polars_types():
    s = pl.Series("x", ["a", "b"] * 50)
    assert (
        rec(s)["rec_polars_type"]
        == 'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))'
    )


@pytest.mark.parametrize(
    "d, arrow_key",
    [
        pytest.param(255, "uint8", id="255"),
        pytest.param(256, "uint8", id="256"),
        pytest.param(257, "uint16", id="257"),
        pytest.param(65_536, "uint16", id="65536"),
        pytest.param(65_537, "uint32", id="65537"),
    ],
)
def test_dictionary_key_widths(d, arrow_key):
    s = pl.Series(
        "x",
        [f"v{i:05d}" for i in range(d)] * 2,
    )
    r = rec(s, categorical_threshold=100_000)
    assert (
        r["rec_arrow_type"]
        == f"dictionary<values=string, indices={arrow_key}, ordered=0>"
    )
    # Polars reserves one key code: its own width (checked by casting) must match the measured layout.
    polars = s.cast(pl.Enum(s.unique(maintain_order=True).to_list()))
    assert (
        r["rec_polars_size_bytes"]
        == _sizes.column_sizes(polars, 1)["size_polars_bytes"]
    )


def test_dictionary_keys_are_frequency_ordered():
    # 300 values twice, then 20 000 draws from r230..r269: in first-seen order those hot
    # values straddle key 256 (a random high key byte); by frequency they all sit below it.
    hot = np.random.default_rng(0).integers(230, 270, 20_000)
    s = pl.Series("x", [f"r{i}" for i in range(300)] * 2 + [f"r{i}" for i in hot])
    r = rec(s)
    assert r["rec_arrow_type"].startswith("dictionary")
    chosen = next(c for c in r["rec_candidates"] if c["outcome"] == "chosen")
    assert chosen["evidence"].endswith(" key_order=frequency")
    arrow = s.to_arrow(compat_level=pl.CompatLevel.oldest()).cast(
        pa_type(r["rec_arrow_type"]), safe=False
    )
    want = r["rec_arrow_size_zstd_bytes"]
    assert _sizes.ipc_body_bytes(frequency_ordered(arrow), 1) == pytest.approx(
        want, rel=0.01, abs=16
    )
    assert _sizes.ipc_body_bytes(arrow, 1) != pytest.approx(want, rel=0.01, abs=16)


def test_dictionary_gate_rejects_above_threshold():
    r = rec(pl.Series("x", ["a", "b"] * 500), categorical_threshold=1)
    assert r["rec_arrow_type"] == "string"
    dictionary = next(
        c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary")
    )
    assert dictionary["outcome"] == "rejected"
    assert "categorical_threshold=1" in dictionary["reason"]


def test_projection_scales_the_dictionary_by_the_estimate():
    ids = [f"id-{i:027d}" for i in range(1_000)]
    s = pl.Series("x", ids[:600] + ids[600:800] * 2 + ids[800:] * 10)
    r = rec(s)
    dictionary = next(
        c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary")
    )
    assert r["est_method"] in ("schnabel", "chao1")
    assert r["est_cardinality"] > 1_000
    assert dictionary["projected_population_bytes"] > dictionary["predicted_bytes"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Oracles (single implementation: no reference agreement)

_SIMPLE = {
    "null": pa.null(),
    "bool": pa.bool_(),
    "float": pa.float32(),
    "double": pa.float64(),
    "string": pa.string(),
    "large_string": pa.large_string(),
    "binary": pa.binary(),
    "large_binary": pa.large_binary(),
    "date32[day]": pa.date32(),
    **{
        n: getattr(pa, n)()
        for n in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
    },
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
    return eval(
        name,
        {"__builtins__": {}},
        {"pl": pl, **{k: getattr(pl, k) for k in dir(pl) if k[:1].isupper()}},
    )


def _string_source(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.List, pl.Array)):
        return _string_source(dtype.inner)
    return isinstance(dtype, (pl.String, pl.Categorical, pl.Enum))


def _outer_chosen(r: dict) -> dict:
    """The chosen candidate for the column itself (list columns also report the inner level's, rule "inner: …")."""
    return next(
        c
        for c in r["rec_candidates"]
        if c["outcome"] == "chosen" and not c["rule"].startswith("inner: ")
    )


def _unlist(s: pl.Series, target: str) -> pl.Series:
    """A list → scalar recommendation (every row holds at most one item) is a first-item extraction, not a cast."""
    if isinstance(s.dtype, (pl.List, pl.Array)) and "list" not in target:
        return s.list.first() if isinstance(s.dtype, pl.List) else s.arr.first()
    return s


def frequency_ordered(arr: pa.Array) -> pa.Array:
    """`arr` with every dictionary's keys renumbered as the recommender measures them
    (spec 2026-10-07 §5): the most frequent value takes key 0, ties go to the value seen
    first; values never seen follow. Lists are rebuilt around their reordered values."""
    t = arr.type
    # The list branches assume compacted lists (no values behind null rows, not sliced), as
    # Polars-built rechunked test frames are.
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return type(arr).from_arrays(
            arr.offsets, frequency_ordered(arr.values), mask=arr.is_null()
        )
    if pa.types.is_fixed_size_list(t):
        return pa.FixedSizeListArray.from_arrays(
            frequency_ordered(arr.values), t.list_size, mask=arr.is_null()
        )
    if not pa.types.is_dictionary(t):
        return arr
    ranked = (
        pl.Series("code", arr.indices.cast(pa.int64()))
        .to_frame()
        .with_row_index("row")
        .drop_nulls("code")
        .group_by("code")
        .agg(n=pl.len(), first=pl.col("row").min())
        .sort(["n", "first"], descending=[True, False])["code"]
        .to_list()
    )
    seen = set(ranked)
    order = ranked + [c for c in range(len(arr.dictionary)) if c not in seen]
    new_code = [0] * len(order)
    for new, old in enumerate(order):
        new_code[old] = new
    indices = pc.take(pa.array(new_code, t.index_type), arr.indices)
    return pa.DictionaryArray.from_arrays(indices, arr.dictionary.take(pa.array(order)))


def list_frame(n: int = 600) -> pl.DataFrame:
    """List / Array columns: single-item ones become scalars (checked through `_unlist`), the rest stay lists."""
    return pl.DataFrame(
        [
            pl.Series(
                "one_int",
                [[i % 200] if i % 7 else None for i in range(n)],
                dtype=pl.List(pl.Int64),
            ),
            pl.Series(
                "one_str",
                [[f"s{i % 9}"] if i % 5 else None for i in range(n)],
                dtype=pl.List(pl.String),
            ),
            pl.Series(
                "one_float", [[i / 4] for i in range(n)], dtype=pl.List(pl.Float64)
            ),
            pl.Series(
                "arr1",
                [[i % 50] if i % 3 else None for i in range(n)],
                dtype=pl.Array(pl.Int64, 1),
            ),
            pl.Series(
                "arr2", [[i % 50, i % 3] for i in range(n)], dtype=pl.Array(pl.Int64, 2)
            ),
            pl.Series(
                "multi",
                [[i, i + 1][: 1 + i % 2] for i in range(n)],
                dtype=pl.List(pl.Int64),
            ),
        ]
    )


ORACLE_FRAMES = [
    pytest.param(lambda: {"mixed": describe_mixed(2_000)}, id="mixed"),
    pytest.param(lambda: {"lists": list_frame()}, id="lists"),
    pytest.param(
        lambda: {"strings": stringified(describe_mixed(500))}, id="stringified"
    ),
    pytest.param(
        lambda: {"large": pl.read_ipc(LARGE)}, id="large", marks=pytest.mark.slow
    ),
]


@pytest.mark.parametrize("make", ORACLE_FRAMES)
def test_sizes_match_pyarrow_and_polars_casts(make):
    frames = make()
    result = recommend_frames(frames)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        s = frames[r["frame"]][r["column"]].rechunk()
        chosen = _outer_chosen(r)
        assert chosen["predicted_bytes"] == r["rec_arrow_size_bytes"], r["column"]
        original = chosen["rule"].endswith("original")
        if original:
            assert (r["rec_arrow_size_bytes"], r["rec_polars_size_bytes"]) == (
                r["size_bytes"],
                r["size_polars_bytes"],
            )
            assert r["rec_polars_type"] == str(s.dtype), r[
                "column"
            ]  # the Polars side is an identity cast
            # Kept-original columns go through the same identity cast and measurement below. The only skip is a
            # type pa_type cannot spell (e.g. a generic struct<...>), where there is no pyarrow type to cast to.
            try:
                pa_type(r["rec_arrow_type"])
            except NotImplementedError:
                continue
        skippable = _string_source(s.dtype) and not original
        try:
            arrow = (
                _unlist(s, r["rec_arrow_type"])
                .rechunk()
                .to_arrow(compat_level=pl.CompatLevel.oldest())
            )
            arrow = arrow.cast(pa_type(r["rec_arrow_type"]), safe=False)
            polars = (
                s
                if original
                else _unlist(s, r["rec_arrow_type"])
                .cast(pl_dtype(r["rec_polars_type"]))
                .rechunk()
            )
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pl.exceptions.PolarsError,
            NotImplementedError,
        ):
            assert (
                skippable
            ), f"{r['column']}: pyarrow/Polars cannot cast a non-string column to {r['rec_arrow_type']}"
            continue
        if arrow.null_count != s.null_count() or polars.null_count() != s.null_count():
            assert skippable, f"{r['column']}: the library cast lost values"
            continue
        level = 1
        assert _sizes.ipc_body_bytes(arrow, None) == r["rec_arrow_size_bytes"], r[
            "column"
        ]
        assert _sizes.ipc_body_bytes(frequency_ordered(arrow), level) == pytest.approx(
            r["rec_arrow_size_zstd_bytes"], rel=0.01, abs=16
        ), r["column"]
        native = _sizes.column_sizes(polars, level)
        assert native["size_polars_bytes"] == r["rec_polars_size_bytes"], r["column"]
        if "dictionary" not in r["rec_arrow_type"]:
            # The recommender measures its Polars layout in frequency order too; a Polars
            # cast assigns codes in first-seen order, so only other targets compare.
            assert native["size_polars_zstd_bytes"] == pytest.approx(
                r["rec_polars_size_zstd_bytes"], rel=0.01, abs=16
            ), r["column"]
        checked += 1
    assert checked >= min(10, result.height)


def test_dictionary_evidence_reads_est_high():
    """The dictionary candidates' evidence cardinality is Rust's own conclusion
    (est_high / est_cardinality, floored at n_unique) from the same result."""
    frames = {
        "mixed": describe_mixed(2_000),
        "strings": stringified(describe_mixed(500)),
    }
    result = recommend_frames(frames)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        for c in r["rec_candidates"]:
            m = re.search(
                r"c=(\S+) from (est_high|est_cardinality)", c["evidence"] or ""
            )
            if m is None:
                continue
            prefix = "inner_" if c["rule"].startswith("inner: ") else ""
            # Rust floors the cardinality at the observed distinct count.
            expected = max(r[f"{prefix}{m[2]}"], r[f"{prefix}n_unique"])
            assert float(m[1]) == pytest.approx(expected, rel=1e-9), (
                r["column"],
                c["rule"],
            )
            checked += 1
    assert checked >= 5


def test_rust_conclusions_match_describe():
    from analytics.describe import CONCLUSIONS, DescribeRust

    frame = describe_mixed(2_000)
    a = OneShotRecommender().add(frame).result()
    b = DescribeRust().add({"t": frame}).result()
    # Same rule; float operation order may differ by an ulp.
    for c in CONCLUSIONS:
        if CONCLUSIONS[c] == pl.Float64:
            got = a[c].to_list()
            assert got == pytest.approx(b[c].to_list(), rel=1e-12, nan_ok=True), c
        else:
            assert a[c].to_list() == b[c].to_list(), c


# ─────────────────────────────────────────────────────────────────────────────
# top_k (spec docs/superpowers/specs/2026-10-07-top-k-frequencies-design.md)

WORDS = ["b", "a", "b", None, "c", "a", "b"]


def top_k(r: dict, column: str = "top_k") -> list | None:
    """A top_k cell as ordered (value, count) pairs."""
    cell = r[column]
    return None if cell is None else [(e["key"], e["value"]) for e in cell]


def test_top_k_column_type():
    out = OneShotRecommender().add(pl.DataFrame({"s": WORDS})).result()
    entry = pl.Struct({"key": pl.String, "value": pl.UInt64})
    assert out.schema["top_k"] == pl.List(entry)
    assert out.schema["inner_top_k"] == pl.List(entry)


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["c", "a", "b"])], ids=str
)
def test_top_k_known_answer(dtype):
    r = rec(pl.Series("s", WORDS, dtype=dtype))
    assert top_k(r) == [("b", 3), ("a", 2), ("c", 1)]


def test_top_k_ties_go_to_the_value_seen_first():
    assert top_k(rec(pl.Series("s", ["y", "x", "x", "y", "z"]))) == [
        ("y", 2),
        ("x", 2),
        ("z", 1),
    ]


@pytest.mark.parametrize(
    "k, want",
    [
        (1, [("b", 3)]),
        (2, [("b", 3), ("a", 2)]),
        (None, [("b", 3), ("a", 2), ("c", 1)]),
        (0, None),
    ],
)
def test_top_k_is_limited(k, want):
    assert top_k(rec(pl.Series("s", WORDS), top_k=k)) == want


def test_top_k_is_null_off_dictionary_candidates():
    frame = pl.DataFrame(
        {
            "s": WORDS,
            "i": list(range(7)),
            "n": pl.Series([None] * 7, dtype=pl.String),
        }
    )
    out = OneShotRecommender().add(frame).result()
    assert [c is None for c in out["top_k"].to_list()] == [False, True, True]
    gated = OneShotRecommender(categorical_threshold=2).add(frame).result()
    assert gated["top_k"].null_count() == 3


def test_top_k_default_limit_is_a_prefix_of_the_full_order():
    values = [f"v{i:03d}" for i in range(600) for _ in range((600 - i) % 37 + 1)]
    s = pl.Series("s", np.random.default_rng(0).permutation(np.array(values)))
    r = rec(s)
    assert r["rec_arrow_type"] == "dictionary<values=string, indices=uint16, ordered=0>"
    full = top_k(rec(s, top_k=None))
    assert len(full) == 600
    assert len({v for v, _ in full}) == 600
    assert len(top_k(r)) == 256
    assert top_k(r) == full[:256]


def test_inner_top_k_counts_list_values():
    s = pl.Series("l", [["b", "a"], None, ["b", None], [], ["c", "b"]])
    r = rec(s)
    assert r["top_k"] is None
    assert top_k(r, "inner_top_k") == [("b", 3), ("a", 1), ("c", 1)]


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
@pytest.mark.parametrize("k", [-1, 1.5, True, 2**64])
def test_top_k_must_be_a_non_negative_integer(cls, k):
    with pytest.raises(ValueError, match="top_k"):
        cls(top_k=k)


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
def test_top_k_accepts_numpy_integers(cls):
    cls(top_k=np.int64(2))


def test_to_arrow_restores_the_map_type():
    from analytics.recommend import to_arrow

    frame = pl.DataFrame(
        {"s": WORDS, "l": [["x"], None, ["y", "x"], [], None, None, None]}
    )
    t = to_arrow(OneShotRecommender().add(frame).result())
    for name in ("top_k", "inner_top_k"):
        assert t.schema.field(name).type == pa.map_(pa.string(), pa.uint64())
    assert t.schema.field("top_k").type.keys_sorted is False
    assert t.column("top_k").to_pylist() == [[("b", 3), ("a", 2), ("c", 1)], None]
    assert t.column("inner_top_k").to_pylist() == [None, [("x", 2), ("y", 1)]]


def polars_top_k(s: pl.Series, k: int | None) -> list:
    """The oracle: non-null values (a list's: its items) by count descending, ties to
    the first occurrence."""
    if isinstance(s.dtype, pl.List):
        s = s.drop_nulls().explode(empty_as_null=True)
    elif isinstance(s.dtype, pl.Array):
        s = s.drop_nulls().arr.explode(empty_as_null=True)
    ranked = (
        s.cast(pl.String)
        .to_frame("v")
        .with_row_index("row")
        .drop_nulls("v")
        .group_by("v")
        .agg(n=pl.len(), first=pl.col("row").min())
        .sort(["n", "first"], descending=[True, False])
    )
    if k is not None:
        ranked = ranked.head(k)
    return list(zip(ranked["v"].to_list(), ranked["n"].to_list()))


@pytest.mark.parametrize("make", ORACLE_FRAMES)
def test_top_k_matches_polars(make):
    frames = make()
    params = {"categorical_threshold": 50_000} if "large" in frames else {}
    result = recommend_frames(frames, **params)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        s = frames[r["frame"]][r["column"]]
        for column, prefix in (("top_k", ""), ("inner_top_k", "inner: ")):
            dictionary = [
                c
                for c in r["rec_candidates"]
                if c["rule"] == f"{prefix}string→dictionary"
            ]
            expected = bool(dictionary) and dictionary[0]["outcome"] != "rejected"
            assert (r[column] is not None) == expected, (r["column"], column)
            if expected:
                assert top_k(r, column) == polars_top_k(s, 256), (r["column"], column)
                checked += 1
    assert checked > 0
