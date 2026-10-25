import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import synthdata.scenarios as sc

# Height, width, schema and row-hash sum of each scenario, frozen when the scenarios
# moved here byte-identical from the former root fixture module: they stay stable.
FROZEN = json.loads((Path(__file__).with_name("frozen.json")).read_text())


def _summary(df: pl.DataFrame) -> dict:
    return {
        "height": df.height,
        "width": df.width,
        "schema": {k: str(v) for k, v in df.schema.items()},
        "hash": int(df.hash_rows().sum()),
    }


@pytest.mark.parametrize("name", list(FROZEN))
def test_scenarios_are_frozen(name):
    case = FROZEN[name]
    new = getattr(sc, name)(*case["args"])
    frames = new if isinstance(new, dict) else {"": new}
    assert {k: _summary(v) for k, v in frames.items()} == case["frames"]


def test_population_samples_and_measure_frame():
    df, truth = sc.population_samples(1_000, [(100, "uniform"), (50, ("zipf", 1.1))])
    assert truth == {"p0": 100, "p1": 50}
    man = sc.measure_frame(df)
    assert man.columns["p0"]["distinct"] <= 100 and man.rows == 1_000


# The planted structure that the root tests and benchmarks rely on.


def test_mixed_dtypes_layout():
    df = sc.mixed_dtypes(200)
    assert df.shape == (200, 18)
    assert df.columns[:3] == ["float64_skewed", "float64_high_unique", "float64_sparse"]
    assert df.schema["categorical_skewed"] == pl.Categorical


def test_mixed_dtypes_is_deterministic_per_seed():
    assert sc.mixed_dtypes(100).equals(sc.mixed_dtypes(100))
    assert not sc.mixed_dtypes(100).equals(sc.mixed_dtypes(100, seed=7))


def test_low_cardinality():
    df = sc.low_cardinality(1_000, 10, cardinality=5)
    assert df.shape == (1_000, 10)
    assert df.columns[0] == "c000"
    assert df["c000"].drop_nulls().n_unique() <= 5
    assert 0 < df["c000"].null_count() < 200


def test_related_frames_planted_relationships():
    f = sc.related_frames(1_000)
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
    f = sc.similar_frames()
    assert set(f) == {"df0", "df1"}
    a, b = set(f["df0"]["sim_00"].drop_nulls().to_list()), set(
        f["df1"]["sim_00"].drop_nulls().to_list()
    )
    assert len(a & b) / min(len(a), len(b)) > 0.9


def test_integer_generators():
    m = sc.integer_multiples(1_000, 3, 12)
    assert m.columns == ["c0", "c1", "c2"]
    assert (m["c0"] % 12 == 0).all()
    assert sc.integer_random(1_000, 2).shape == (1_000, 2)


def test_describe_mixed_is_seeded_and_covers_every_family():
    a, b = sc.describe_mixed(300), sc.describe_mixed(300)
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
    s = sc.stringified(sc.describe_mixed(100))
    assert all(dt == pl.String or dt == pl.List(pl.String) for dt in s.dtypes)
    assert {"i32", "f64", "dec", "date", "dt_tz", "time", "bool", "list_i64"} <= set(
        s.columns
    )
    assert not {"dur", "bin", "struct", "arr_i32", "str_int", "cat", "all_null"} & set(
        s.columns
    )


def test_describe_benchmark_shapes():
    narrow = sc.describe_narrow(1_000)
    assert narrow.columns == ["int", "float", "num_str", "iso_str"]
    assert narrow.height == 1_000
    wide = sc.describe_wide(100, 8)
    assert wide.width == 8
    assert [str(dt) for dt in wide.dtypes[:4]] == ["Int64", "Float64", "String", "Date"]
    nested = sc.describe_nested(500)
    assert nested.schema == {
        "list_i64": pl.List(pl.Int64),
        "list_str": pl.List(pl.String),
        "struct": pl.Struct({"a": pl.Int64, "b": pl.String}),
    }


def test_near_unique():
    df = sc.near_unique(10_000)
    assert df.columns == [
        "u0",
        "u1",
        "u2",
        "key_hi",
        "key_lo",
        "skewed",
        "mid",
        "low",
        "flag",
    ]
    assert df.select(pl.struct("key_hi", "key_lo").n_unique()).item() == 10_000
    assert 9_985 <= df["u0"].n_unique() < 10_000
    assert 0.93 < (df["skewed"] == 0).mean() < 0.97
    assert df["low"].n_unique() == 10
    assert df["flag"].sum() == 1
    assert sc.near_unique(1_000).equals(sc.near_unique(1_000))


def test_near_constant():
    df = sc.near_constant(100_000)
    assert df.shape == (100_000, 3)
    assert df.columns == ["f0", "f1", "f2"]
    assert df.sum().row(0) == (10, 10, 10)
    assert (df.sum_horizontal() <= 1).all()


def test_top_k_streams_shape_and_determinism():
    df = sc.top_k_streams(20_000)
    assert df.columns == ["zipf_15", "zipf_11", "zipf_08", "flat", "skewed", "zipf_str"]
    assert df.height == 20_000
    assert df.equals(sc.top_k_streams(20_000))
    one = sc.top_k_streams(20_000, columns=["zipf_08"])
    assert one["zipf_08"].equals(df["zipf_08"])
    share = {
        c: df[c].value_counts(sort=True)["count"][0] / df.height for c in df.columns
    }
    assert share["zipf_15"] > share["zipf_11"] > share["zipf_08"] > share["flat"]
    assert 0.93 < share["skewed"] < 0.97
    assert df["zipf_str"][0] == f"item_{df['zipf_11'][0]}"


def test_runs_generators_plant_mean_run_length():
    df = sc.runs_int(100_000, 2, mean_run=50, cardinality=1_000)
    assert df.columns == ["c0", "c1"] and df.height == 100_000
    assert_frame_equal(df, sc.runs_int(100_000, 2, mean_run=50, cardinality=1_000))
    for c in df.columns:
        n_runs = 1 + (df[c].diff() != 0).sum()
        assert 100_000 / n_runs == pytest.approx(50, rel=0.1)
    single = sc.runs_int(1_000, 1, mean_run=1, cardinality=1_000_000)
    assert (single["c0"].diff() != 0).sum() > 990  # every row its own draw
    s = sc.runs_str(1_000, 1, mean_run=10, cardinality=5)
    assert s.columns == ["s0"] and s.schema["s0"] == pl.String
    assert s["s0"][0].startswith("v")


def test_population_samples():
    spec = [(1_000, "uniform"), (50_000, ("zipf", 1.1))]
    frame, truth = sc.population_samples(200_000, spec, seed=3)
    assert frame.columns == ["p0", "p1"]
    assert frame.height == 200_000
    assert truth == {"p0": 1_000, "p1": 50_000}
    assert frame["p0"].n_unique() == 1_000  # uniform, 200 draws per label: all seen
    assert frame["p1"].n_unique() < 50_000  # Zipf: rare labels unseen
    assert frame.equals(sc.population_samples(200_000, spec, seed=3)[0])
