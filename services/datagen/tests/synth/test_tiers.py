import polars as pl

from synthdata.synth import (
    WINDOW_ROWS,
    synth_frame,
    synth_manifest,
    synth_small,
    synth_windows,
    spread,
)
from synthdata.synth.classify import coverage
from synthdata.synth.pairwise import allowed_pairs


def test_small_tier_proves_its_coverage():
    df = synth_small()
    man, intended = synth_manifest(2_000)
    cov = coverage(man, intended)
    assert df.height == 2_000 and cov.mismatches == {} and cov.missing == set()
    assert cov.covered == allowed_pairs()
    assert {"list_int", "array_int", "struct"} <= set(df.columns)
    assert df.schema["struct"] == pl.Struct
    assert isinstance(df.schema["array_int"], pl.Array)
    assert any(c.startswith("rel_") for c in df.columns)


def test_window_tier_proves_its_coverage():
    df = synth_windows()
    man, intended = synth_manifest(WINDOW_ROWS)
    cov = coverage(man, intended)
    assert df.height == WINDOW_ROWS and cov.mismatches == {} and cov.missing == set()
    assert df.columns == synth_small().columns


def test_tiers_share_columns_and_seeds():
    assert synth_frame(500).columns == synth_small().columns
    assert WINDOW_ROWS == 9 * 8_192


def test_spread():
    df = synth_small()
    s = spread(df, 10)
    assert s.width == 10 and s.columns[0] == df.columns[0]
    assert spread(df, 10_000).width == df.width


def test_null_list_rows_hold_no_slots():
    lists = synth_small()["list_int"].to_arrow()
    assert 0 < lists.null_count < len(lists)
    lengths = pl.Series(lists.offsets).diff().drop_nulls()
    assert lengths.filter(pl.Series(lists.is_null())).sum() == 0
