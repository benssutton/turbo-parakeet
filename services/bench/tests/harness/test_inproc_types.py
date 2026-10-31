"""chbench.harness.inproc_types: the polars / datafusion block vocabulary. Imports no engine."""

import pyarrow as pa
import pytest

from chbench.harness.inproc_types import (
    PolarsType,
    arrow_family,
    column_family,
    feed_error,
    ipc_batch_rows,
    ipc_codec,
    normalise_ipc_batch_rows,
    normalise_ipc_compression,
    parse_datafusion_type,
    parse_polars_type,
    type_label,
)


@pytest.mark.parametrize(
    "value, stored",
    [
        ("none", "none"),
        ("LZ4_FRAME", "LZ4_FRAME"),
        ("ZSTD", "ZSTD(1)"),
        ("ZSTD(1)", "ZSTD(1)"),
        ("ZSTD(22)", "ZSTD(22)"),
    ],
)
def test_compression_is_stored_in_one_spelling(value, stored):
    assert normalise_ipc_compression(value) == stored


@pytest.mark.parametrize("value", ["LZ4", "ZSTD(0)", "ZSTD(23)", "zstd", "ZSTD(3", 3])
def test_unknown_compression_is_refused(value):
    with pytest.raises(ValueError, match="none, LZ4_FRAME, ZSTD"):
        normalise_ipc_compression(value)


def test_ipc_batch_rows_is_whole_or_a_positive_count():
    assert normalise_ipc_batch_rows("whole") == "whole"
    assert normalise_ipc_batch_rows(65536) == "65536"
    assert ipc_batch_rows("whole") is None and ipc_batch_rows("65536") == 65536
    for bad in (0, -1, True, "big", "1000", 1.5):
        with pytest.raises(ValueError, match="whole' or a positive integer"):
            normalise_ipc_batch_rows(bad)


def test_ipc_codec():
    assert ipc_codec("none") is None
    assert ipc_codec("LZ4_FRAME").name == "lz4"
    z = ipc_codec("ZSTD(9)")
    assert z.name == "zstd" and z.compression_level == 9


@pytest.mark.parametrize(
    "name, label",
    [
        ("Int16", "Int16"),
        ("UInt64", "UInt64"),
        ("Float32", "Float32"),
        ("Boolean", "Boolean"),
        ("Date", "Date"),
        ("Time", "Time"),
        ("String", "String"),
        ("Binary", "Binary"),
        ("Categorical", "Categorical(UInt32)"),
        ("Categorical(UInt8)", "Categorical(UInt8)"),
        (
            'Categorical(Categories(name="v", namespace="", physical=pl.UInt16))',
            "Categorical(UInt16)",
        ),
        ('Categorical(Categories("v"))', "Categorical(UInt32)"),
        ("Decimal(precision=5, scale=2)", "Decimal(5, 2)"),
        ("Decimal(18, 0)", "Decimal(18, 0)"),
        ("Datetime(time_unit='us', time_zone=None)", "Datetime(us)"),
        ("Datetime(time_unit='ns', time_zone='UTC')", "Datetime(ns, UTC)"),
        ("Datetime(ms, Europe/London)", "Datetime(ms, Europe/London)"),
        ("Enum", "Enum"),
        ("Enum(categories=['a', \"it's\"])", "Enum"),
    ],
)
def test_polars_names_parse_to_their_short_label(name, label):
    assert parse_polars_type(name).label == label == type_label("polars", name)


def test_polars_type_parameters():
    assert parse_polars_type("Categorical(UInt8)") == PolarsType(
        "Categorical(UInt8)", "Categorical", width="UInt8"
    )
    d = parse_polars_type("Decimal(precision=5, scale=2)")
    assert (d.kind, d.precision, d.scale) == ("Decimal", 5, 2)
    t = parse_polars_type("Datetime(time_unit='ns', time_zone='UTC')")
    assert (t.kind, t.unit, t.tz) == ("Datetime", "ns", "UTC")
    assert parse_polars_type("Enum").categories is None
    assert parse_polars_type("Enum(categories=['b', 'a'])").categories == ("b", "a")


@pytest.mark.parametrize(
    "name",
    [
        "Categorical(UInt64)",
        "Float16",
        "Datetime(s)",
        "Decimal(40, 2)",
        "Decimal(5, 6)",
        "Enum(categories=[1, 2])",
        "Enum(categories=[])",
        "Enum(categories=['a', 'a'])",
        "Duration(time_unit='us')",
        "List(Int64)",
        "Utf8",
    ],
)
def test_names_outside_the_polars_vocabulary_are_refused(name):
    with pytest.raises(ValueError, match="polars type"):
        parse_polars_type(name)


@pytest.mark.parametrize(
    "name, t",
    [
        ("int16", pa.int16()),
        ("uint64", pa.uint64()),
        ("float", pa.float32()),
        ("double", pa.float64()),
        ("bool", pa.bool_()),
        ("string", pa.string()),
        ("large_string", pa.large_string()),
        ("string_view", pa.string_view()),
        ("binary", pa.binary()),
        ("date32[day]", pa.date32()),
        ("decimal32(5, 2)", pa.decimal32(5, 2)),
        ("decimal128(38, 10)", pa.decimal128(38, 10)),
        ("timestamp[us]", pa.timestamp("us")),
        ("timestamp[ns, tz=UTC]", pa.timestamp("ns", "UTC")),
        ("time32[ms]", pa.time32("ms")),
        ("time64[us]", pa.time64("us")),
        ("fixed_size_binary[4]", pa.binary(4)),
        (
            "dictionary<values=string, indices=int16, ordered=0>",
            pa.dictionary(pa.int16(), pa.string()),
        ),
        (
            "dictionary<values=large_string, indices=uint8, ordered=0>",
            pa.dictionary(pa.uint8(), pa.large_string()),
        ),
    ],
)
def test_datafusion_names_are_pyarrow_type_strings(name, t):
    assert parse_datafusion_type(name) == t
    assert type_label("datafusion", name) == name


def test_run_end_encoding_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="MIN / MAX"):
        parse_datafusion_type("run_end_encoded<run_ends: int32, values: string>")


@pytest.mark.parametrize(
    "name",
    [
        "float64",
        "utf8",
        "int",
        "decimal32(12, 2)",
        "time32[us]",
        "dictionary<values=int64, indices=int8, ordered=0>",
        "list<item: int64>",
        "duration[us]",
        "date64[ms]",
        "timestamp[us, tz=UTC",
    ],
)
def test_names_outside_the_datafusion_vocabulary_are_refused(name):
    with pytest.raises(ValueError, match="datafusion type"):
        parse_datafusion_type(name)


def test_families():
    assert arrow_family(pa.int8()) == arrow_family(pa.decimal64(10, 2)) == "numeric"
    assert arrow_family(pa.dictionary(pa.int8(), pa.string())) == "categorical"
    assert arrow_family(pa.dictionary(pa.int8(), pa.binary())) == "binary"
    assert arrow_family(pa.list_(pa.int64())) is None
    assert column_family("polars", "Enum") == "categorical"
    assert column_family("polars", "Datetime(us)") == "timestamp"
    assert column_family("datafusion", "date32[day]") == "date"


def test_feed_errors_name_both_sides():
    assert feed_error(pa.string(), "polars", "Categorical(UInt8)") is None
    assert (
        feed_error(
            pa.string(),
            "datafusion",
            "dictionary<values=string, indices=int16, ordered=0>",
        )
        is None
    )
    assert feed_error(pa.int64(), "datafusion", "string") is None
    msg = feed_error(pa.int64(), "polars", "Enum")
    assert (
        "polars.type value 'Enum'" in msg and "categorical" in msg and "numeric" in msg
    )
    assert "cannot feed" in feed_error(pa.list_(pa.int64()), "polars", "Int64")
