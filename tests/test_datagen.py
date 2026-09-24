"""The planted structure that other tests and benchmarks rely on."""

import polars as pl

from datagen import integer_multiples, integer_random, low_cardinality, mixed_dtypes, related_frames, similar_frames


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
    assert set(f["orders"]["region"].to_list()) < set(f["customers"]["region"].to_list())
    shared = set(f["archive"]["name"].to_list()) & set(f["customers"]["name"].to_list())
    assert 0.9 < len(shared) / 1_000 < 0.95


def test_similar_frames():
    f = similar_frames()
    assert set(f) == {"df0", "df1"}
    a, b = set(f["df0"]["sim_00"].drop_nulls().to_list()), set(f["df1"]["sim_00"].drop_nulls().to_list())
    assert len(a & b) / min(len(a), len(b)) > 0.9


def test_integer_generators():
    m = integer_multiples(1_000, 3, 12)
    assert m.columns == ["c0", "c1", "c2"] and (m["c0"] % 12 == 0).all()
    assert integer_random(1_000, 2).shape == (1_000, 2)
