"""The planted codec / skip-index frames (moved from the calibration scripts)."""

import polars as pl
import pytest

from synthdata.scenarios import delta_frame, gcd_frame, skip_frame


def test_gcd_frame_planted_divisors():
    df = gcd_frame(10_000)
    assert df.columns == ["gcd_1000", "gcd_even", "gcd_three", "ts_seconds"]
    assert (df["gcd_1000"] % 1000 == 0).all()
    assert (df["gcd_even"] % 2 == 0).all()
    assert (df["gcd_three"] % 3 == 0).all()
    assert df["ts_seconds"].dtype == pl.Datetime("us")
    assert (df["ts_seconds"].dt.epoch("us") % 1_000_000 == 0).all()


def test_delta_frame_planted_steps():
    df = delta_frame(10_000)
    assert df.columns == [
        "ts_regular",
        "ts_jitter",
        "walk",
        "ids_sorted",
        "walk_i16",
        "walk_i32",
        "random_ints",
    ]
    steps = df["ts_regular"].dt.epoch("ms").diff().drop_nulls()
    assert (steps == 1000).all()
    assert df["ids_sorted"].is_sorted()
    assert df["walk_i16"].dtype == pl.Int16 and df["walk_i32"].dtype == pl.Int32
    assert df["walk"].diff().drop_nulls().abs().max() <= 1000


def test_skip_frame_planted_layouts():
    df = skip_frame(10_000)
    assert df.columns == ["sorted_ids", "clustered", "random_50", "grouped_hashes"]
    assert df["sorted_ids"].is_sorted()
    assert df["clustered"].n_unique() == 1000
    assert (df["clustered"].rle_id().max() + 1) == 1000  # one run per value
    assert df["random_50"].n_unique() == 50
    assert (df["grouped_hashes"].rle_id().max() + 1) == 1000  # runs of 10


def test_skip_frame_rejects_rows_not_a_multiple_of_1000():
    with pytest.raises(ValueError, match="multiple of 1000"):
        skip_frame(1_500)


def test_frames_are_deterministic():
    for f in (gcd_frame, delta_frame, skip_frame):
        assert f(2_000).equals(f(2_000))
