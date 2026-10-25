import polars as pl

from synthdata.cover import (
    WINDOW_ROWS,
    cover,
    cover_manifest,
    cover_small,
    cover_windows,
    spread,
)
from synthdata.cover.classify import coverage
from synthdata.cover.pairwise import allowed_pairs


def test_small_tier_proves_its_coverage():
    df = cover_small()
    man, intended = cover_manifest(2_000)
    cov = coverage(man, intended)
    assert df.height == 2_000 and cov.mismatches == {} and cov.missing == set()
    assert cov.covered == allowed_pairs()
    assert {"list_int", "array_int", "struct"} <= set(df.columns)
    assert df.schema["struct"] == pl.Struct
    assert isinstance(df.schema["array_int"], pl.Array)
    assert any(c.startswith("rel_") for c in df.columns)


def test_window_tier_proves_its_coverage():
    df = cover_windows()
    man, intended = cover_manifest(WINDOW_ROWS)
    cov = coverage(man, intended)
    assert df.height == WINDOW_ROWS and cov.mismatches == {} and cov.missing == set()
    assert df.columns == cover_small().columns


def test_tiers_share_columns_and_seeds():
    assert cover(500).columns == cover_small().columns
    assert WINDOW_ROWS == 9 * 8_192


def test_spread():
    df = cover_small()
    s = spread(df, 10)
    assert s.width == 10 and s.columns[0] == df.columns[0]
    assert spread(df, 10_000).width == df.width
