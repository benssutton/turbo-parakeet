"""The recommended-type vocabulary (ruling R36, spec section 5.2, FINDINGS 9):
narrow integers, Float32, Bool, Decimal128, Date32, DateTime, DateTime64
precision and zone, FixedString, Enum8 / Enum16, Time64, and the GCD codec.

Every skip message asserted here was produced by the server in
`spikes/spike_ch_vocabulary.py` (ClickHouse 26.8.31.2)."""

import datetime as dt
import decimal
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from chbench.datagen import load_or_generate
from chbench.datagen.realdata import path_spec
from chbench.harness.ddl import table_name
from chbench.harness.loader import arrow_type_for, load_table
from chbench.harness.queries import (
    filter_sql,
    full_scan_sql,
    in_predicate,
    query_values,
    range_predicate,
)
from chbench.harness.suite import ClickHouseAxes, SuiteError, load_suite, parse_suite
from chbench.harness.validate import (
    KNOWN_CODECS,
    KNOWN_TYPES,
    TypeResolutionError,
    expand_grid,
    resolve_type,
)
from tests.harness.test_suite import MINIMAL
from tests.harness.test_suite_path import suite_file

V = "26.8.31.2"


def write(directory, array, file="x.arrow"):
    p = directory / file
    batch = pa.record_batch([array], names=["v"])
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    return p


def ts(values, unit, tz=None):
    return pa.array(values, pa.timestamp(unit, tz=tz))


def enum_strings(n):
    return pa.array([f"v{i:05d}" for i in range(n)] * 2)


# axis type -> (the dataset array, the DDL type it must resolve to)
CASES = {
    "UInt8": (pa.array([1, 200, 7, 7, None, 3, 7, 200], pa.uint8()), "UInt8", "1"),
    "Int8": (pa.array([-5, 100, 7, 7, None, 3, 7, 100], pa.int8()), "Int8", "-5"),
    "UInt16": (
        pa.array([300, 60000, 7, 7, None, 3, 7, 60000], pa.uint16()),
        "UInt16",
        "300",
    ),
    "Int16": (
        pa.array([-300, 3000, 7, 7, None, 3, 7, 3000], pa.int16()),
        "Int16",
        "-300",
    ),
    "UInt64": (
        pa.array([2**63 + 5, 1, 7, 7, None, 3, 7, 1], pa.uint64()),
        "UInt64",
        "9223372036854775813",
    ),
    "Int64": (
        pa.array([-(2**62), 1, 7, 7, None, 3, 7, 1], pa.int64()),
        "Int64",
        "-4611686018427387904",
    ),
    "Float32": (
        pa.array([1.5, 2.25, 7.0, 7.0, None, 3.0, 7.0, 2.25], pa.float32()),
        "Float32",
        "1.5",
    ),
    "Bool": (
        pa.array([True, False, True, True, None, False, True, False]),
        "Bool",
        "true",
    ),
    "Decimal128": (
        pa.array(
            [
                decimal.Decimal("12.3400"),
                decimal.Decimal("-1.5000"),
                None,
                decimal.Decimal("12.3400"),
                decimal.Decimal("7.0000"),
            ],
            pa.decimal128(20, 4),
        ),
        "Decimal128(4)",
        "12.34",
    ),
    "Date32": (
        pa.array(
            [dt.date(2020, 1, 2), dt.date(2021, 5, 6), None, dt.date(2020, 1, 2)],
            pa.date32(),
        ),
        "Date32",
        "2020-01-02",
    ),
    "DateTime": (
        ts([1700000123, None, 1700000123, 5], "s"),
        "DateTime",
        "2023-11-14 22:15:23",
    ),
    "DateTime@UTC": (
        ts([1700000123, None, 1700000123, 5], "s", "UTC"),
        "DateTime('UTC')",
        "2023-11-14 22:15:23",
    ),
    "DateTime@Paris": (
        ts([1700000123, None, 1700000123, 5], "s", "Europe/Paris"),
        "DateTime('Europe/Paris')",
        "2023-11-14 23:15:23",
    ),
    "DateTime64@s": (
        ts([1700000123, None, 1700000123, 5], "s"),
        "DateTime64(0)",
        "2023-11-14 22:15:23",
    ),
    "DateTime64@ms": (
        ts([1700000123456, None, 1700000123456, 5], "ms"),
        "DateTime64(3)",
        "2023-11-14 22:15:23.456",
    ),
    "DateTime64@us": (
        ts([1700000123456789, None, 1700000123456789, 5], "us"),
        "DateTime64(6)",
        "2023-11-14 22:15:23.456789",
    ),
    "DateTime64@ns": (
        ts([1700000123456789000, None, 1700000123456789000, 5], "ns"),
        "DateTime64(9)",
        "2023-11-14 22:15:23.456789000",
    ),
    "DateTime64@ns+UTC": (
        ts([1700000123456789000, None, 1700000123456789000, 5], "ns", "UTC"),
        "DateTime64(9, 'UTC')",
        "2023-11-14 22:15:23.456789000",
    ),
    "FixedString@binary": (
        pa.array([b"abcd", None, b"wxyz", b"abcd"], pa.binary(4)),
        "FixedString(4)",
        "abcd",
    ),
    "FixedString@string": (
        pa.array(["ab", None, "abc", "ab", "xyz"]),
        "FixedString(3)",
        "ab",
    ),
    "Enum8": (
        pa.array(["b", None, "a", "b", "c", "b"]),
        "Enum8('a' = 1, 'b' = 2, 'c' = 3)",
        "b",
    ),
    "Enum16": (
        enum_strings(130),
        "Enum16(" + ", ".join(f"'v{i:05d}' = {i + 1}" for i in range(130)) + ")",
        "v00000",
    ),
    "Time64@s": (
        pa.array([3723, None, 3723, 5], pa.time32("s")),
        "Time64(0)",
        "01:02:03",
    ),
    "Time64@ms": (
        pa.array([3723000, None, 3723000, 5], pa.time32("ms")),
        "Time64(3)",
        "01:02:03.000",
    ),
    "Time64@us": (
        pa.array([3723000000, None, 3723000000, 5], pa.time64("us")),
        "Time64(6)",
        "01:02:03.000000",
    ),
    "Time64@ns": (
        pa.array([3723000000000, None, 3723000000000, 5], pa.time64("ns")),
        "Time64(9)",
        "01:02:03.000000000",
    ),
}


def axis_of(case):
    return case.split("@")[0]


def path_suite(tmp_path, array, **axes):
    """A parsed suite whose dataset is `array` read from a file and whose
    clickhouse axes are MINIMAL's overridden by `axes`."""
    s = parse_suite(MINIMAL)
    ck = {k: list(v) for k, v in vars(s.clickhouse).items()}
    ck.update({k: list(v) for k, v in axes.items()})
    spec = path_spec(write(tmp_path, array), "v")
    return replace(s, dataset=spec, clickhouse=ClickHouseAxes(**ck))


def grid(tmp_path, array, *, value_min=0.0, value_max=1.0, null_count=None, **axes):
    s = path_suite(tmp_path, array, **axes)
    return expand_grid(
        s,
        null_count=s.dataset.null_count if null_count is None else null_count,
        value_min=value_min,
        value_max=value_max,
        clickhouse_version=V,
        array=array,
    )


def one(tmp_path, case, **axes):
    array = CASES[case][0]
    axes.setdefault("type", [axis_of(case)])
    [v] = grid(tmp_path, array, nullable=[True], **axes)
    return v


# --- the vocabulary -----------------------------------------------------------


def test_the_vocabulary_holds_exactly_the_measured_types_and_codecs():
    assert KNOWN_TYPES == {
        "Float64",
        "Float32",
        "Decimal32",
        "Decimal64",
        "Decimal128",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Bool",
        "String",
        "Date",
        "Date32",
        "DateTime",
        "DateTime64",
        "FixedString",
        "Enum8",
        "Enum16",
        "Time64",
    }
    assert KNOWN_CODECS == {
        "none",
        "Delta",
        "DoubleDelta",
        "Gorilla",
        "T64",
        "LowCardinality",
        "FPC",
        "GCD",
    }


# --- resolve_type: parameters come from the dataset ----------------------------


@pytest.mark.parametrize("case", list(CASES))
def test_resolve_type_takes_its_parameters_from_the_dataset(tmp_path, case):
    array, expected, _ = CASES[case]
    spec = path_spec(write(tmp_path, array), "v")
    assert resolve_type(axis_of(case), spec, array) == expected


def test_a_generated_dataset_keeps_datetime64_3(tmp_path):
    s = parse_suite(MINIMAL)
    assert resolve_type("DateTime64", s.dataset) == "DateTime64(3)"


def test_a_string_array_is_needed_for_fixedstring_and_enum_of_a_string_dataset(
    tmp_path,
):
    array = pa.array(["a", "b"])
    spec = path_spec(write(tmp_path, array), "v")
    with pytest.raises(ValueError, match="array"):
        resolve_type("FixedString", spec)
    with pytest.raises(ValueError, match="array"):
        resolve_type("Enum8", spec)


def test_fixedstring_width_is_the_longest_value_in_bytes(tmp_path):
    array = pa.array(["ab", "éé", None])  # two 2-byte characters
    spec = path_spec(write(tmp_path, array), "v")
    assert resolve_type("FixedString", spec, array) == "FixedString(4)"


def test_enum_names_are_sorted_and_escaped(tmp_path):
    array = pa.array(["it's", "a\\b", "it's", None])
    spec = path_spec(write(tmp_path, array), "v")
    assert resolve_type("Enum8", spec, array) == "Enum8('a\\\\b' = 1, 'it\\'s' = 2)"


def test_enum8_holds_127_values_and_enum16_more(tmp_path):
    ok = enum_strings(127)
    spec = path_spec(write(tmp_path, ok), "v")
    assert resolve_type("Enum8", spec, ok).startswith("Enum8('v00000' = 1")
    too_many = enum_strings(128)
    spec = path_spec(write(tmp_path, too_many, "y.arrow"), "v")
    with pytest.raises(TypeResolutionError, match=r"Enum8.*127.*128"):
        resolve_type("Enum8", spec, too_many)
    assert resolve_type("Enum16", spec, too_many).startswith("Enum16('v00000' = 1")


def test_an_enum_with_more_than_32767_values_or_none_cannot_be_resolved(tmp_path):
    big = enum_strings(32768)
    spec = path_spec(write(tmp_path, big), "v")
    with pytest.raises(TypeResolutionError, match=r"Enum16.*32767.*32768"):
        resolve_type("Enum16", spec, big)
    empty = pa.array([None, None], pa.string())
    spec = path_spec(write(tmp_path, empty, "e.arrow"), "v")
    with pytest.raises(TypeResolutionError, match="no non-null"):
        resolve_type("Enum8", spec, empty)


def test_an_unresolvable_type_is_a_skipped_cell_with_the_reason(tmp_path):
    too_many = enum_strings(128)
    [v] = grid(tmp_path, too_many, type=["Enum8"], nullable=[False])
    assert v.status == "skipped" and "127" in v.skip_reason and v.ch_type == "Enum8"
    [v] = grid(tmp_path, too_many, type=["Enum16"], nullable=[False])
    assert v.status == "ok" and v.ch_type.startswith("Enum16(")


# --- codecs: the measured matrix ----------------------------------------------

OK = None


@pytest.mark.parametrize(
    "case,codec,message",
    [
        # GCD: Int*, UInt*, Decimal*, Date*, DateTime* (and Bool) only
        ("UInt8", "GCD", OK),
        ("Int64", "GCD", OK),
        ("Bool", "GCD", OK),
        ("Decimal128", "GCD", OK),
        ("Date32", "GCD", OK),
        ("DateTime", "GCD", OK),
        ("DateTime64@us", "GCD", OK),
        (
            "Float32",
            "GCD",
            "Codec GCD cannot be applied to column Float32 because it can only be "
            "used with Int*, UInt*, Decimal*, Date* or DateTime* types.",
        ),
        (
            "FixedString@binary",
            "GCD",
            "Codec GCD cannot be applied to column FixedString(4) "
            "because it can only be used with Int*, UInt*, Decimal*, "
            "Date* or DateTime* types.",
        ),
        (
            "Enum8",
            "GCD",
            "Codec GCD cannot be applied to column Enum8('a' = 1, 'b' = 2, 'c' = 3) "
            "because it can only be used with Int*, UInt*, Decimal*, Date* or "
            "DateTime* types.",
        ),
        (
            "Time64@ms",
            "GCD",
            "Codec GCD cannot be applied to column Time64(3) because it can "
            "only be used with Int*, UInt*, Decimal*, Date* or DateTime* types.",
        ),
        # Delta / DoubleDelta
        ("Float32", "Delta", OK),
        ("Bool", "DoubleDelta", OK),
        ("Enum8", "Delta", OK),
        ("Time64@ms", "DoubleDelta", OK),
        ("FixedString@binary", "Delta", OK),
        (
            "Decimal128",
            "Delta",
            "Codec Delta is only applicable for data types of size 1, 2, 4, 8 "
            "bytes. Given type Decimal(38, 4).",
        ),
        (
            "Decimal128",
            "DoubleDelta",
            "Codec DoubleDelta is only applicable for data types of size "
            "1, 2, 4, 8 bytes. Given type Decimal(38, 4).",
        ),
        (
            "FixedString@string",
            "Delta",
            "Codec Delta is only applicable for data types of size "
            "1, 2, 4, 8 bytes. Given type FixedString(3).",
        ),
        (
            "FixedString@binary",
            "DoubleDelta",
            "Codec DoubleDelta is not applicable for "
            "FixedString(4) because the data type is not numeric.",
        ),
        # Gorilla: floats only
        ("Float32", "Gorilla", OK),
        (
            "Bool",
            "Gorilla",
            "The combination of compression codecs Gorilla(1), LZ4 is meaningless, "
            "because it does not make sense to apply a floating-point time series "
            "codec to non-floating-point columns (Note: you can enable setting "
            "'allow_suspicious_codecs' to skip this check).",
        ),
        (
            "Decimal128",
            "Gorilla",
            "Codec Gorilla is only applicable for data types of size 1, 2, 4, "
            "8 bytes. Given type Decimal(38, 4).",
        ),
        (
            "FixedString@string",
            "Gorilla",
            "Codec Gorilla is only applicable for data types of size "
            "1, 2, 4, 8 bytes. Given type FixedString(3).",
        ),
        # T64
        ("UInt8", "T64", OK),
        ("Bool", "T64", OK),
        ("Date32", "T64", OK),
        ("DateTime", "T64", OK),
        ("Enum8", "T64", OK),
        ("Time64@ms", "T64", OK),
        ("DateTime64@ns", "T64", OK),
        ("Float32", "T64", "T64 codec is not supported for specified type Float32."),
        (
            "Decimal128",
            "T64",
            "T64 codec is not supported for specified type Decimal(38, 4).",
        ),
        (
            "FixedString@binary",
            "T64",
            "T64 codec is not supported for specified type " "FixedString(4).",
        ),
        # FPC
        ("Float32", "FPC", OK),
        (
            "Decimal128",
            "FPC",
            "Codec 'FPC' is not applicable for Decimal(38, 4) because the data "
            "type is not Float*.",
        ),
        (
            "Date32",
            "FPC",
            "Codec 'FPC' is not applicable for Date32 because the data type is not "
            "Float*.",
        ),
        (
            "DateTime@UTC",
            "FPC",
            "Codec 'FPC' is not applicable for DateTime('UTC') because the "
            "data type is not Float*.",
        ),
        (
            "Bool",
            "FPC",
            "Codec 'FPC' is not applicable for Bool because the data type is not "
            "Float*.",
        ),
    ],
)
def test_codec_matrix(tmp_path, case, codec, message):
    v = one(tmp_path, case, codec=[codec])
    if message is OK:
        assert v.status == "ok", v.skip_reason
    else:
        assert v.status == "skipped" and v.skip_reason == message


@pytest.mark.parametrize(
    "case,width",
    [
        ("UInt8", 1),
        ("Int16", 2),
        ("UInt64", 8),
        ("Bool", 1),
        ("Date32", 4),
        ("DateTime", 4),
        ("DateTime@UTC", 4),
        ("DateTime64@s", 8),
        ("Time64@ms", 8),
        ("FixedString@binary", 4),
        ("Enum16", 2),
    ],
)
def test_gorilla_width_in_the_message_is_the_storage_width(tmp_path, case, width):
    v = one(tmp_path, case, codec=["Gorilla"], compression=["ZSTD(3)"])
    assert f"Gorilla({width}), ZSTD(3) is meaningless" in v.skip_reason


@pytest.mark.parametrize("case", ["Enum16", "Bool"])
def test_gorilla_on_a_two_axis_codec_cell_is_the_only_skip_for_enum_and_bool(
    tmp_path, case
):
    assert one(tmp_path, case, codec=["T64"]).status == "ok"


# --- LowCardinality: String and FixedString only -------------------------------


def test_low_cardinality_is_accepted_on_fixedstring(tmp_path):
    v = one(tmp_path, "FixedString@binary", codec=["LowCardinality"])
    assert v.status == "ok" and v.ch_type == "LowCardinality(Nullable(FixedString(4)))"


@pytest.mark.parametrize(
    "case,type_in_message",
    [
        ("UInt8", "UInt8"),
        ("Int64", "Int64"),
        ("Float32", "Float32"),
        ("Bool", "Bool"),
        ("Date32", "Date32"),
        ("DateTime", "DateTime"),
        ("DateTime@UTC", "DateTime('UTC')"),
    ],
)
def test_low_cardinality_on_numbers_bool_and_dates_is_the_code_455_skip(
    tmp_path, case, type_in_message
):
    v = one(tmp_path, case, codec=["LowCardinality"])
    assert v.skip_reason == (
        f"Creating columns of type LowCardinality(Nullable({type_in_message})) is prohibited by "
        "default due to expected negative impact on performance. It can be enabled with the "
        "`allow_suspicious_low_cardinality_types` setting."
    )


@pytest.mark.parametrize(
    "case,type_in_message",
    [
        ("Decimal128", "Decimal(38, 4)"),
        ("DateTime64@ns+UTC", "DateTime64(9, 'UTC')"),
        ("DateTime64@s", "DateTime64(0)"),
        ("Time64@ms", "Time64(3)"),
        ("Enum8", "Enum8('a' = 1, 'b' = 2, 'c' = 3)"),
    ],
)
def test_low_cardinality_on_decimal_datetime64_enum_and_time_is_the_code_43_skip(
    tmp_path, case, type_in_message
):
    v = one(tmp_path, case, codec=["LowCardinality"])
    assert v.skip_reason == (
        "DataTypeLowCardinality is supported only for numbers, strings, Date or DateTime, "
        f"but got Nullable({type_in_message})."
    )


# --- skip indexes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "case,type_in_message",
    [
        ("Decimal128", "Decimal(38, 4)"),
        ("Date32", "Date32"),
        ("Time64@ms", "Time64(3)"),
    ],
)
def test_bloom_filter_rejections_name_the_nullable_wrapper(
    tmp_path, case, type_in_message
):
    v = one(tmp_path, case, skip_index=["bloom_filter"])
    assert v.skip_reason == (
        f"Unexpected type Nullable({type_in_message}) of bloom filter "
        "index: When validating secondary index `ix`."
    )
    [plain] = grid(
        tmp_path,
        CASES[case][0],
        type=[axis_of(case)],
        nullable=[False],
        skip_index=["bloom_filter"],
        null_count=0,
    )
    assert plain.skip_reason == (
        f"Unexpected type {type_in_message} of bloom filter index: "
        "When validating secondary index `ix`."
    )


@pytest.mark.parametrize(
    "case",
    [
        "UInt8",
        "Float32",
        "Bool",
        "DateTime",
        "DateTime64@ns",
        "FixedString@binary",
        "Enum8",
        "Enum16",
    ],
)
@pytest.mark.parametrize("index", ["bloom_filter", "minmax", "set(100)"])
def test_other_types_accept_every_skip_index(tmp_path, case, index):
    assert one(tmp_path, case, skip_index=[index]).status == "ok"


@pytest.mark.parametrize("case", ["Decimal128", "Date32", "Time64@ms"])
def test_minmax_and_set_are_accepted_everywhere(tmp_path, case):
    for index in ("minmax", "set(100)"):
        assert one(tmp_path, case, skip_index=[index]).status == "ok"


# --- ranges of the narrow integers ----------------------------------------------


@pytest.mark.parametrize(
    "case,lo,hi,fragment",
    [
        ("UInt8", 0, 300, "maximum is 300"),
        ("Int8", -129, 5, "minimum is -129"),
        ("Int16", 0, 40000, "maximum is 40000"),
        ("UInt16", 0, 70000, "maximum is 70000"),
        ("Int64", -5, 2**63, "maximum is"),
    ],
)
def test_integers_that_do_not_fit_the_axis_type_are_skipped(
    tmp_path, case, lo, hi, fragment
):
    [v] = grid(
        tmp_path,
        CASES[case][0],
        type=[axis_of(case)],
        nullable=[True],
        value_min=lo,
        value_max=hi,
    )
    assert v.status == "skipped" and fragment in v.skip_reason
    assert axis_of(case) in v.skip_reason


def test_integers_inside_the_range_are_not_skipped(tmp_path):
    [v] = grid(
        tmp_path,
        CASES["UInt8"][0],
        type=["UInt8"],
        nullable=[True],
        value_min=0,
        value_max=255,
    )
    assert v.status == "ok"


def test_unsigned_minimum_rule_still_runs_first(tmp_path):
    [v] = grid(
        tmp_path,
        CASES["UInt8"][0],
        type=["UInt8"],
        nullable=[True],
        value_min=-1,
        value_max=5,
    )
    assert "Unsigned type UInt8" in v.skip_reason


def test_decimal128_range_check_uses_precision_38(tmp_path):
    array = CASES["Decimal128"][0]
    [ok] = grid(
        tmp_path,
        array,
        type=["Decimal128"],
        nullable=[True],
        value_min=-1.0,
        value_max=1e30,
    )
    assert ok.status == "ok" and ok.ch_type == "Nullable(Decimal128(4))"
    [bad] = grid(
        tmp_path,
        array,
        type=["Decimal128"],
        nullable=[True],
        value_min=-1.0,
        value_max=1e35,
    )
    assert bad.status == "skipped" and "Decimal128(4)" in bad.skip_reason


# --- the loader's Arrow types ---------------------------------------------------


@pytest.mark.parametrize(
    "ch_type,expected",
    [
        ("UInt8", pa.uint8()),
        ("Int16", pa.int16()),
        ("UInt64", pa.uint64()),
        ("Float32", pa.float32()),
        ("Bool", pa.bool_()),
        ("Date32", pa.date32()),
        ("DateTime", pa.timestamp("s")),
        ("DateTime('UTC')", pa.timestamp("s", tz="UTC")),
        ("DateTime64(0)", pa.timestamp("s")),
        ("DateTime64(9)", pa.timestamp("ns")),
        ("DateTime64(6, 'Europe/Paris')", pa.timestamp("us", tz="Europe/Paris")),
        ("Nullable(DateTime64(3, 'UTC'))", pa.timestamp("ms", tz="UTC")),
        ("Decimal128(4)", pa.decimal128(38, 4)),
        ("Time64(0)", pa.time32("s")),
        ("Time64(3)", pa.time32("ms")),
        ("Time64(6)", pa.time64("us")),
        ("Time64(9)", pa.time64("ns")),
        ("FixedString(4)", None),
        ("Enum8('a' = 1)", None),
        ("LowCardinality(String)", pa.string()),
    ],
)
def test_arrow_type_for_the_new_types(ch_type, expected):
    assert arrow_type_for(ch_type) == expected


# --- suites: the parse-time dataset / axis rules --------------------------------


def parse_path_suite(tmp_path, array, axis):
    write(tmp_path, array)
    return load_suite(suite_file(tmp_path, axis=axis))


@pytest.mark.parametrize("axis", sorted(KNOWN_TYPES))
def test_every_known_type_parses_on_a_dataset_that_can_feed_it(tmp_path, axis):
    case = {
        "Float64": "Float32",
        "Decimal32": "Decimal128",
        "Decimal64": "Decimal128",
        "Int32": "Int64",
        "UInt32": "UInt64",
        "String": "Enum8",
        "Date": "Date32",
        "Enum8": "Enum8",
        "Enum16": "Enum16",
        "DateTime64": "DateTime64@ms",
        "Time64": "Time64@ms",
        "FixedString": "FixedString@binary",
    }.get(axis, axis)
    s = parse_path_suite(tmp_path, CASES[case][0], axis)
    assert s.clickhouse.type == [axis]


@pytest.mark.parametrize(
    "case,axis,fragment",
    [
        ("UInt8", "Bool", "boolean"),
        ("UInt8", "Enum8", "string"),
        ("UInt8", "FixedString", "string or fixed_size_binary"),
        ("UInt8", "Time64", "time32 or time64"),
        ("Enum8", "Date32", "integer, date32 or timestamp"),
        ("Enum8", "DateTime", "integer, date32 or timestamp"),
        ("Enum8", "UInt8", "numeric"),
        ("Enum8", "Float32", "numeric"),
        ("Enum8", "Decimal128", "numeric"),
        ("Bool", "Int64", "numeric"),
        ("FixedString@binary", "Enum16", "string"),
    ],
)
def test_axis_types_that_the_dataset_cannot_feed_are_rejected_at_parse_time(
    tmp_path, case, axis, fragment
):
    with pytest.raises(SuiteError, match=rf"clickhouse\.type.*{axis}.*{fragment}"):
        parse_path_suite(tmp_path, CASES[case][0], axis)


def test_decimal128_needs_decimal_places(tmp_path):
    y = (
        MINIMAL.replace("arrow_type: float64", "arrow_type: int32")
        .replace("value_range: [-1, 10]", "value_range: [-100, 100]")
        .replace("  decimal_places: 8\n", "")
        .replace("type: [Float64]", "type: [Int32, Decimal128]")
    )
    with pytest.raises(SuiteError, match=r"Decimal128.*decimal_places"):
        parse_suite(y)


def test_the_unknown_type_error_names_the_vocabulary():
    with pytest.raises(SuiteError, match="UUID"):
        parse_suite(MINIMAL.replace("type: [Float64]", "type: [UUID]"))


def test_gcd_parses_as_a_codec():
    assert parse_suite(
        MINIMAL.replace("codec: [none]", "codec: [GCD]")
    ).clickhouse.codec == ["GCD"]


# --- the server: every admitted type loads and answers queries -------------------


@pytest.fixture
def table(ch):
    names = []

    def make(suffix):
        name = table_name("t_vocab", suffix)
        names.append(name)
        return name

    yield make
    for name in names:
        ch.command(f"DROP TABLE IF EXISTS {name}")


@pytest.mark.clickhouse
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("compression", ["LZ4", "ZSTD(3)"])
def test_each_admitted_type_loads_through_arrow_and_reads_back(
    ch, table, tmp_path, case, compression
):
    array, expected_type, first = CASES[case]
    axis = axis_of(case)
    [variant] = grid(
        tmp_path, array, type=[axis], nullable=[True], compression=[compression]
    )
    assert variant.status == "ok" and variant.ch_type == f"Nullable({expected_type})"
    dataset = load_or_generate(
        path_spec(write(tmp_path, array), "v"), tmp_path / "cache"
    )
    name = table(f"{abs(hash((case, compression))) % 10**8:08d}0000000000000000")
    r = load_table(ch, name, variant, dataset)
    assert r.status == "ok", r.error
    assert r.stats.rows == len(array) and r.stats.parts == 1
    [(count, nulls)] = ch.query_rows(f"SELECT count(v), countIf(v IS NULL) FROM {name}")
    assert (count, nulls) == (len(array) - array.null_count, array.null_count)
    [(value,)] = ch.query_rows(
        f"SELECT toString(v) FROM {name} ORDER BY row_id LIMIT 1"
    )
    assert value == first


@pytest.mark.clickhouse
@pytest.mark.parametrize(
    "case,codec",
    [
        ("UInt8", "GCD"),
        ("Int64", "GCD"),
        ("Bool", "GCD"),
        ("Decimal128", "GCD"),
        ("Date32", "GCD"),
        ("DateTime@UTC", "GCD"),
        ("DateTime64@ns+UTC", "GCD"),
        ("Float32", "Gorilla"),
        ("Float32", "FPC"),
        ("Enum8", "T64"),
        ("Time64@ms", "DoubleDelta"),
        ("FixedString@binary", "LowCardinality"),
    ],
)
def test_codec_cells_the_matrix_accepts_load(ch, table, tmp_path, case, codec):
    array, _, _ = CASES[case]
    [variant] = grid(
        tmp_path, array, type=[axis_of(case)], nullable=[True], codec=[codec]
    )
    assert variant.status == "ok"
    dataset = load_or_generate(
        path_spec(write(tmp_path, array), "v"), tmp_path / "cache"
    )
    name = table(f"{abs(hash((case, codec))) % 10**8:08d}1111111111111111")
    r = load_table(ch, name, variant, dataset)
    assert r.status == "ok", r.error


@pytest.mark.clickhouse
@pytest.mark.parametrize("case", list(CASES))
def test_filter_and_scan_queries_run_on_every_admitted_type(ch, table, tmp_path, case):
    array, _, _ = CASES[case]
    [variant] = grid(tmp_path, array, type=[axis_of(case)], nullable=[True])
    dataset = load_or_generate(
        path_spec(write(tmp_path, array), "v"), tmp_path / "cache"
    )
    name = table(f"{abs(hash(case)) % 10**8:08d}2222222222222222")
    r = load_table(ch, name, variant, dataset)
    assert r.status == "ok", r.error
    [(n, _)] = ch.query_rows(full_scan_sql(name, variant))
    assert n == len(array)
    values = query_values(dataset.array)
    sql, fraction = in_predicate(values, 0.5, 1, ch_type=variant.ch_type)
    [(matched, _)] = ch.query_rows(filter_sql(name, variant, sql))
    assert matched == round(fraction * len(array)), sql
    if axis_of(case) not in ("Bool", "FixedString", "Enum8", "Enum16"):
        sql, fraction = range_predicate(
            values, 0.5, np.random.default_rng(0), ch_type=variant.ch_type
        )
        [(matched, _)] = ch.query_rows(filter_sql(name, variant, sql))
        assert matched == round(fraction * len(array)), sql


# --- literals of the new types (no server) -----------------------------------------


@pytest.mark.parametrize(
    "value,ch_type,expected",
    [
        (3723456, "Time64(3)", "'01:02:03.456'"),
        (3723, "Nullable(Time64(0))", "'01:02:03'"),
        (3723000000001, "Time64(9)", "'01:02:03.000000001'"),
        (True, "Bool", "true"),
        (False, "Nullable(Bool)", "false"),
        (b"ab\0\0", "FixedString(4)", "'ab'"),
        (b"\xff\xfe", "FixedString(2)", "unhex('fffe')"),
        ("it's", "Enum8('a' = 1)", "'it\\'s'"),
        (
            decimal.Decimal("123456789012345678901234.5678"),
            "Decimal128(4)",
            "'123456789012345678901234.5678'",
        ),
        (np.datetime64("2020-01-02", "D"), "Date32", "'2020-01-02'"),
        (
            np.datetime64("2023-11-14T22:15:23.456789123", "ns"),
            "DateTime64(9)",
            "'2023-11-14 22:15:23.456789123'",
        ),
        (
            np.datetime64("2023-11-14T22:15:23", "s"),
            "DateTime('Europe/Paris')",
            "toDateTime('2023-11-14 22:15:23', 'UTC')",
        ),
        (
            np.datetime64("2023-11-14T22:15:23.5", "ms"),
            "DateTime64(3, 'UTC')",
            "toDateTime64('2023-11-14 22:15:23.500', 3, 'UTC')",
        ),
    ],
)
def test_literals_of_the_new_types(value, ch_type, expected):
    from chbench.harness.queries import _render_literal

    assert _render_literal(value, ch_type) == expected


def test_query_values_keeps_64_bit_integers_with_nulls_exact_and_times_as_ticks():
    big = pa.array([2**63 + 5, None, 1], pa.uint64())
    assert list(query_values(big)) == [2**63 + 5, None, 1]
    assert list(query_values(pa.array([3723000000001, 5], pa.time64("ns")))) == [
        3723000000001,
        5,
    ]
    assert query_values(pa.array([1, 2], pa.int32())).dtype == np.int32
