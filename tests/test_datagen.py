"""The planted structure that other tests and benchmarks rely on."""

import polars as pl
from polars.testing import assert_frame_equal

from datagen import (
    describe_mixed,
    describe_narrow,
    describe_nested,
    describe_wide,
    integer_multiples,
    integer_random,
    low_cardinality,
    mixed_dtypes,
    related_frames,
    similar_frames,
    stringified,
)


def test_mixed_dtypes_layout():
    df = mixed_dtypes(200)
    assert df.shape == (200, 18)
    assert df.columns[:3] == ["float64_skewed", "float64_high_unique", "float64_sparse"]
    assert df.schema["categorical_skewed"] == pl.Categorical


def test_mixed_dtypes_is_deterministic_per_seed():
    assert mixed_dtypes(100).equals(mixed_dtypes(100))
    assert not mixed_dtypes(100).equals(mixed_dtypes(100, seed=7))


def test_low_cardinality():
    df = low_cardinality(1_000, 10, cardinality=5)
    assert df.shape == (1_000, 10) and df.columns[0] == "c000"
    assert df["c000"].drop_nulls().n_unique() <= 5
    assert 0 < df["c000"].null_count() < 200


def test_related_frames_planted_relationships():
    f = related_frames(1_000)
    ids = set(f["customers"]["id"].to_list())
    assert f["customers"]["id"].n_unique() == 1_000
    assert set(f["orders"]["customer_id"].drop_nulls().to_list()) <= ids
    assert f["orders"]["customer_id"].null_count() > 0
    assert set(f["archive"]["id"].to_list()) == ids
    assert set(f["orders"]["region"].to_list()) < set(
        f["customers"]["region"].to_list()
    )
    shared = set(f["archive"]["name"].to_list()) & set(f["customers"]["name"].to_list())
    assert 0.9 < len(shared) / 1_000 < 0.95


def test_similar_frames():
    f = similar_frames()
    assert set(f) == {"df0", "df1"}
    a, b = set(f["df0"]["sim_00"].drop_nulls().to_list()), set(
        f["df1"]["sim_00"].drop_nulls().to_list()
    )
    assert len(a & b) / min(len(a), len(b)) > 0.9


def test_integer_generators():
    m = integer_multiples(1_000, 3, 12)
    assert m.columns == ["c0", "c1", "c2"] and (m["c0"] % 12 == 0).all()
    assert integer_random(1_000, 2).shape == (1_000, 2)


def test_describe_mixed_is_seeded_and_covers_every_family():
    a, b = describe_mixed(300), describe_mixed(300)
    assert_frame_equal(a, b)
    kinds = {type(dt) for dt in a.dtypes}
    for kind in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Decimal,
        pl.Boolean,
        pl.Date,
        pl.Datetime,
        pl.Duration,
        pl.Time,
        pl.String,
        pl.Categorical,
        pl.Enum,
        pl.Binary,
        pl.List,
        pl.Array,
        pl.Struct,
    ):
        assert kind in kinds, kind
    assert a["all_null"].null_count() == 300
    assert a["dt_tz"].dtype.time_zone == "Europe/London"


def test_stringified_casts_only_castable_columns():
    s = stringified(describe_mixed(100))
    assert all(dt == pl.String or dt == pl.List(pl.String) for dt in s.dtypes)
    assert {"i32", "f64", "dec", "date", "dt_tz", "time", "bool", "list_i64"} <= set(
        s.columns
    )
    assert not {"dur", "bin", "struct", "arr_i32", "str_int", "cat", "all_null"} & set(
        s.columns
    )


def test_describe_benchmark_shapes():
    narrow = describe_narrow(1_000)
    assert (
        narrow.columns == ["int", "float", "num_str", "iso_str"]
        and narrow.height == 1_000
    )
    wide = describe_wide(100, 8)
    assert wide.width == 8 and [str(dt) for dt in wide.dtypes[:4]] == [
        "Int64",
        "Float64",
        "String",
        "Date",
    ]
    nested = describe_nested(500)
    assert nested.schema == {
        "list_i64": pl.List(pl.Int64),
        "list_str": pl.List(pl.String),
        "struct": pl.Struct({"a": pl.Int64, "b": pl.String}),
    }
