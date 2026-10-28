import numpy as np
import polars as pl
import pytest

from evaluate.predict import (
    MARK_ROWS,
    SkipPredictor,
    parse_literal,
    parse_skip,
    value_of,
)


def test_parse_skip():
    assert parse_skip("minmax:2") == ("minmax", None, 2)
    assert parse_skip("set(125):4") == ("set", 125.0, 4)
    assert parse_skip("bloom_filter(0.025):1") == ("bloom_filter", 0.025, 1)


def test_parse_literal_and_value_of():
    assert parse_literal("42", pl.Int64) == 42
    assert parse_literal("'it\\'s'", pl.String) == "it's"
    assert parse_literal("true", pl.Boolean) == 1
    assert parse_literal("'1970-01-06'", pl.Date) == 5
    assert parse_literal("'2026-01-01'", pl.Int64) is None
    assert value_of("v IN (7)", pl.Int64) == 7
    assert value_of("v BETWEEN 1 AND 2", pl.Int64) is None
    # a multi-value list is not a point query
    assert value_of("v IN (1, 2)", pl.Int64) is None
    assert value_of("v IN ('a', 'b')", pl.String) is None


def test_minmax_on_sorted_values_skips_all_other_blocks():
    n = 4 * MARK_ROWS  # four marks
    p = SkipPredictor(pl.Series("v", np.arange(n)))
    assert p.total_marks == 4
    assert p.skip("minmax:1", 0) == pytest.approx(3 / 4)
    assert p.skip("minmax:2", 0) == pytest.approx(2 / 4)  # one block of two marks kept


def test_partial_last_block():
    n = 2 * MARK_ROWS + 10  # three marks, the last holds 10 rows
    p = SkipPredictor(pl.Series("v", np.arange(n)))
    assert p.total_marks == 3
    assert p.skip("minmax:2", 0) == pytest.approx(1 / 3)  # the 1-mark last block skips


def test_set_overflow_blocks_are_not_skipped():
    # block 0: one value; block 1: 200 distinct values (overflows set(125))
    v = np.concatenate(
        [np.zeros(MARK_ROWS, dtype=np.int64), np.arange(MARK_ROWS) % 200 + 1]
    )
    p = SkipPredictor(pl.Series("v", v))
    # block 1 overflows, block 0 holds 0
    assert p.skip("set(125):1", 0) == pytest.approx(0.0)
    # block 0 lacks 5, block 1 overflows
    assert p.skip("set(125):1", 5) == pytest.approx(0.5)


def test_bloom_is_discounted_by_fpr():
    v = np.concatenate(
        [np.zeros(MARK_ROWS, dtype=np.int64), np.ones(MARK_ROWS, dtype=np.int64)]
    )
    p = SkipPredictor(pl.Series("v", v))
    assert p.skip("bloom_filter(0.025):1", 0) == pytest.approx(0.5 * 0.975)


def test_none_skips_nothing_and_counts_rows():
    p = SkipPredictor(pl.Series("v", [1, 1, 2]))
    assert p.skip("none", 1) == 0.0
    assert p.rows_of(1) == 2


def test_quoted_number_for_an_integer_column_is_not_parsed():
    assert value_of("v IN ('-5')", pl.Int64) is None
    assert value_of("v IN (-5)", pl.Int64) == -5
    assert value_of("v IN ('-5')", pl.String) == "-5"


def _round_trip(arr, ch_type: str):
    from chbench.harness.loader import cast_column
    from chbench.harness.queries import in_predicate, query_values

    arr = cast_column(arr, ch_type)
    col = pl.from_arrow(arr)
    sql, frac = in_predicate(query_values(arr), 0.5, 1, ch_type=ch_type)
    v = value_of(sql, col.dtype)
    assert v is not None, sql
    assert SkipPredictor(col).rows_of(v) == round(frac * len(arr)), sql


def test_chbench_literals_round_trip_with_exact_counts():
    import decimal

    import pyarrow as pa

    days = [6, 20000, 6, 7]
    _round_trip(pa.array(days, pa.date32()), "Date")
    secs = [5, 1_700_000_000, 5, 86_399]
    _round_trip(pa.array(secs, pa.timestamp("s")), "DateTime")
    _round_trip(
        pa.array(secs, pa.timestamp("s", "Europe/London")), "DateTime('Europe/London')"
    )
    units = (
        ("ms", 3, [5, 1_700_000_000_123, 5, 7]),
        ("us", 6, [5000, 1_700_000_000_123_456, 5000, 9]),
    )
    units += (("ns", 9, [5, 1_700_000_000_123_456_789, 5, 7]),)
    for unit, p, vals in units:
        _round_trip(pa.array(vals, pa.timestamp(unit)), f"DateTime64({p})")
        for tz in ("UTC", "Europe/London", "America/New_York"):
            _round_trip(
                pa.array(vals, pa.timestamp(unit, tz)), f"DateTime64({p}, '{tz}')"
            )
    # Time64(9) keeps sub-microsecond ticks
    t = [5_999_999_999, 86_399_000_000_001, 5_999_999_999, 3]
    _round_trip(pa.array(t, pa.time64("ns")), "Time64(9)")
    dec = [decimal.Decimal(x) for x in ("1.5000", "-2.2500", "1.5000", "0.0001")]
    _round_trip(pa.array(dec, pa.decimal128(18, 4)), "Decimal64(4)")
