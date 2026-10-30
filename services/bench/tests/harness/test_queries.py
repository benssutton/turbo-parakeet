import uuid

import numpy as np
import pytest
import pyarrow as pa

from chbench.harness.queries import (
    _render_literal,
    expand_queries,
    filter_sql,
    full_scan_sql,
    in_predicate,
    query_id,
    range_predicate,
)
from chbench.harness.suite import QueryAxes, QueryKind
from chbench.harness.validate import TableVariant


def variant(ch_type="Float64"):
    return TableVariant(
        ch_type=ch_type,
        nullable=False,
        primary_key=False,
        projection=False,
        codec="none",
        skip_index="none",
        compression="LZ4",
        status="ok",
        skip_reason=None,
        warnings=(),
    )


def test_range_predicate_covers_the_target_fraction():
    vals = np.round(np.random.default_rng(0).uniform(-1, 10, 100_000), 8)
    sql, actual = range_predicate(vals, 0.10, np.random.default_rng(1))
    assert sql.startswith("v BETWEEN ")
    assert actual == pytest.approx(0.10, abs=0.001)


def test_range_predicate_offset_is_random_not_always_the_tail():
    vals = np.arange(10_000, dtype=float)
    a, _ = range_predicate(vals, 0.1, np.random.default_rng(1))
    b, _ = range_predicate(vals, 0.1, np.random.default_rng(2))
    assert a != b


def test_range_predicate_is_deterministic_for_a_seed():
    vals = np.arange(10_000, dtype=float)
    assert range_predicate(vals, 0.1, np.random.default_rng(5)) == range_predicate(
        vals, 0.1, np.random.default_rng(5)
    )


def test_in_predicate_picks_the_most_frequent_values_closest_to_target():
    vals = np.repeat(np.arange(10), [50, 30, 10, 5, 2, 1, 1, 1, 0, 0][:8] + [0, 0])
    sql, actual = in_predicate(vals, 0.20, 1)
    assert sql == "v IN (1)"  # 30/100 is closest to 0.20 among singles
    assert actual == pytest.approx(0.30)
    sql5, actual5 = in_predicate(vals, 0.20, 5)
    assert sql5.startswith("v IN (") and actual5 > 0


def test_in_predicate_quotes_strings():
    vals = np.array(["a'b"] * 5 + ["c"] * 5)
    sql, _ = in_predicate(vals, 0.5, 1)
    assert sql in ("v IN ('a\\'b')", "v IN ('c')")


def test_full_scan_sql_by_type():
    assert (
        full_scan_sql("bench.t", variant("Float64"))
        == "SELECT count(), sum(toFloat64(v)) FROM bench.t"
    )
    assert (
        full_scan_sql("bench.t", variant("String"))
        == "SELECT count(), sum(length(v)) FROM bench.t"
        " SETTINGS optimize_functions_to_subcolumns = 0"
    )


def test_string_full_scan_reads_the_characters_not_the_size_subcolumn():
    assert full_scan_sql("t", variant("String")).endswith(
        " SETTINGS optimize_functions_to_subcolumns = 0"
    )
    for ch_type in ("LowCardinality(String)", "Nullable(String)"):
        assert full_scan_sql("t", variant(ch_type)).endswith(
            " SETTINGS optimize_functions_to_subcolumns = 0"
        )
    assert "SETTINGS" not in full_scan_sql("t", variant("Float64"))
    assert "SETTINGS" not in full_scan_sql("t", variant("FixedString(4)"))


@pytest.mark.clickhouse
def test_string_full_scan_reads_the_characters_live(ch):
    t = "bench.test_string_scan"
    ch.command(f"DROP TABLE IF EXISTS {t}")
    ch.command(
        f"CREATE TABLE {t} (row_id UInt64, v String) ENGINE = MergeTree ORDER BY row_id "
        "SETTINGS min_bytes_for_wide_part = 0"
    )
    try:
        ch.command(
            f"INSERT INTO {t} SELECT number, repeat('x', 50) FROM numbers(100000)"
        )
        qid = uuid.uuid4().hex
        ch.query_rows(full_scan_sql(t, variant("String")), query_id=qid)
        ch.flush_logs()
        [(read_bytes,)] = ch.query_rows(
            "SELECT read_bytes FROM system.query_log "
            "WHERE query_id = {q:String} AND type = 'QueryFinish'",
            parameters={"q": qid},
        )
        assert read_bytes >= 100_000 * 50  # the characters, not only the 8-byte sizes
    finally:
        ch.command(f"DROP TABLE IF EXISTS {t}")


def test_filter_sql_reads_the_column_like_full_scan():
    assert (
        filter_sql("bench.t", variant("Float64"), "v BETWEEN 1 AND 2")
        == "SELECT count(), sum(toFloat64(v)) FROM bench.t WHERE v BETWEEN 1 AND 2"
    )
    assert (
        filter_sql("bench.t", variant("String"), "v IN ('a')")
        == "SELECT count(), sum(length(v)) FROM bench.t WHERE v IN ('a')"
    )


def test_expand_queries_gives_one_variant_per_kind():
    axes = QueryAxes(
        kinds=[
            QueryKind("full_scan", None, None, None),
            QueryKind("filter", 0.1, "range", None),
        ],
        samples=3,
    )
    vals = np.arange(1000, dtype=float)
    scan, filt = expand_queries(axes, vals, variant(), np.random.default_rng(0))
    assert scan.kind == "full_scan" and scan.predicate_sql is None
    assert filt.predicate_sql and filt.actual_fraction == pytest.approx(0.1, abs=0.01)


def test_query_id_is_stable_and_distinct_per_kind():
    axes = QueryAxes(
        kinds=[
            QueryKind("full_scan", None, None, None),
            QueryKind("filter", 0.1, "range", None),
        ],
        samples=1,
    )
    vals = np.arange(10.0)
    qs = expand_queries(axes, vals, variant(), np.random.default_rng(0))
    again = expand_queries(axes, vals, variant(), np.random.default_rng(0))
    ids = [query_id("t", q) for q in qs]
    assert ids == [query_id("t", q) for q in again]
    assert len(set(ids)) == 2 and all(len(i) == 16 for i in ids)


# --- Fix-report regression tests: literal rendering must follow the column
# type, not the value's runtime type, and nulls must not corrupt a predicate.


def test_in_predicate_quotes_decimal_literals_for_exact_matching():
    # A bare decimal literal is parsed by ClickHouse as Float64 and the IN
    # set is converted to the column type through float arithmetic, which
    # truncates: `materialize(toDecimal32('-0.54929124', 8)) IN (-0.54929124)`
    # returns 0 on a live server, but `IN ('-0.54929124')` (quoted) returns 1.
    # So a Decimal column's IN predicate must render its literals quoted.
    vals = np.array([-0.54929124] * 6 + [0.12345678] * 4)
    sql, actual = in_predicate(vals, 0.6, 1, ch_type="Decimal32(8)")
    assert sql == "v IN ('-0.54929124')"
    assert actual == pytest.approx(0.6)


def test_range_predicate_quotes_decimal_literals():
    vals = np.array([1.00000000, 2.00000000, 3.00000000, 4.00000000])
    sql, _ = range_predicate(
        vals, 1.0, np.random.default_rng(0), ch_type="Decimal64(8)"
    )
    assert sql == "v BETWEEN '1.00000000' AND '4.00000000'"


def test_string_column_quotes_a_non_string_value():
    # The loader casts an integer-valued dataset to String by stringifying;
    # the predicate literal must match that, quoted, not bare.
    vals = np.array([1, 2, 3, 4, 5])
    sql, _ = in_predicate(vals, 0.2, 1, ch_type="String")
    assert sql == "v IN ('1')"


def test_float_column_renders_bare():
    vals = np.arange(10, dtype=float)
    sql, _ = range_predicate(vals, 1.0, np.random.default_rng(0), ch_type="Float64")
    assert sql == "v BETWEEN 0.0 AND 9.0"
    assert "'" not in sql


def test_range_predicate_excludes_nan_and_measures_against_all_rows():
    # A nullable Arrow array becomes a float64 numpy array with NaN for the
    # nulls. NaN must never enter the sorted window or be compared, and the
    # achieved fraction must stay relative to every row, nulls included.
    vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    sql, actual = range_predicate(vals, 2 / 3, np.random.default_rng(0))
    assert "nan" not in sql.lower()
    assert sql == "v BETWEEN 1.0 AND 4.0"
    assert actual == pytest.approx(4 / 6)


def test_in_predicate_excludes_nan_values():
    # NaN is the most frequent "value" here (5/10 = the exact target), so an
    # unfixed implementation picks NaN itself as the IN literal.
    vals = np.array([1.0] * 3 + [2.0] * 2 + [np.nan] * 5)
    sql, actual = in_predicate(vals, 0.5, 1)
    assert "nan" not in sql.lower()
    assert sql == "v IN (1.0)"
    assert actual == pytest.approx(0.3)


# --- Round 2 fix-report regression tests.


def test_string_column_casts_numbers_like_the_loader_not_python_str():
    # The loader stores a float column cast to String through pyarrow's own
    # cast, not Python's str(): pyarrow renders 3.0 as '3' and 1e10 as
    # '1e+10', while Python's str() renders '3.0' and '10000000000.0'. A
    # predicate literal built with str() would then never match the stored
    # rows for a whole-number value.
    vals = np.array([3.0, 1e10])
    sql, _ = in_predicate(vals, 0.5, 2, ch_type="String")
    assert sql == "v IN ('3', '1e+10')"


def test_range_predicate_raises_on_an_all_null_array():
    vals = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        range_predicate(vals, 0.5, np.random.default_rng(0))


def test_in_predicate_raises_on_an_all_null_array():
    vals = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        in_predicate(vals, 0.5, 1)


# --- Final fix wave: a `Date` column crashed the renderer.


def _loader_stored(arr, ch_type):
    """What the loader's cast (`cast_column` -> `pc.cast`) would store for
    each non-null value of `arr` in a column of `ch_type`, as Python
    values."""
    import pyarrow.compute as pc
    from chbench.harness.loader import arrow_type_for

    return [
        x for x in pc.cast(arr, arrow_type_for(ch_type)).to_pylist() if x is not None
    ]


def _expected_literal(value):
    """How the module renders a loader-stored date, timestamp or decimal."""
    import datetime
    import decimal

    if isinstance(value, decimal.Decimal):
        return f"'{value}'"
    assert isinstance(value, (datetime.date, datetime.datetime))
    return f"'{value.isoformat().replace('T', ' ')}'"


@pytest.mark.parametrize(
    "arrow_type, ch_type, with_null",
    [
        (pa.date32(), "Date", False),
        (pa.date32(), "Nullable(Date)", True),
        (pa.timestamp("ms"), "DateTime64(3)", False),
        (pa.timestamp("ms"), "Nullable(DateTime64(3))", True),
        (pa.decimal64(18, 8), "Decimal64(8)", False),
        (pa.decimal64(18, 8), "Nullable(Decimal64(8))", True),
    ],
)
def test_literals_render_what_the_loader_stores(arrow_type, ch_type, with_null):
    # The predicate sees the dataset as `dataset.array.to_numpy(...)`, i.e.
    # `np.datetime64[D]` for date32, `np.datetime64[ms]` for timestamp[ms],
    # `decimal.Decimal` for decimal64. Whatever those become, the rendered
    # literal must name the value the loader's `pc.cast` put in the column,
    # or a BETWEEN/IN against a Date column never matches its own rows.
    # Before this fix, `pa.array([np.datetime64[D]], type=pa.date32())`
    # raised TypeError on pyarrow 24 and every Date suite crashed at the
    # first filter query.
    raw = [1, 2, 3, 4] + ([None] if with_null else [])
    arr = pa.array(raw, type=arrow_type)
    values = arr.to_numpy(zero_copy_only=False)
    stored = _loader_stored(arr, ch_type)

    sql, actual = range_predicate(
        values, 1.0, np.random.default_rng(0), ch_type=ch_type
    )
    assert sql == (
        f"v BETWEEN {_expected_literal(stored[0])} "
        f"AND {_expected_literal(stored[-1])}"
    )
    assert actual == pytest.approx(4 / len(raw))

    sql, actual = in_predicate(values, 0.25, 1, ch_type=ch_type)
    assert sql == f"v IN ({_expected_literal(stored[0])})"
    assert actual == pytest.approx(1 / len(raw))


@pytest.mark.clickhouse
@pytest.mark.parametrize(
    "arrow_type, ch_type, suffix",
    [
        (pa.date32(), "Date", "d1"),
        (pa.date32(), "Nullable(Date)", "d2"),
        (pa.timestamp("ms"), "DateTime64(3)", "d3"),
        (pa.timestamp("ms"), "Nullable(DateTime64(3))", "d4"),
    ],
)
def test_date_predicates_match_the_intended_rows_live(ch, arrow_type, ch_type, suffix):
    """The rendered predicate, run against a table loaded the way the
    harness loads it, selects exactly the rows the achieved fraction says."""
    from chbench.harness.ddl import create_table_sql, drop_table_sql, table_name
    from chbench.harness.loader import cast_column

    nullable = ch_type.startswith("Nullable")
    raw = list(range(100, 140)) + ([None] * 4 if nullable else [])
    arr = pa.array(raw, type=arrow_type)
    values = arr.to_numpy(zero_copy_only=False)
    v = variant(ch_type)
    name = table_name("t_qlit", f"{suffix}000000")
    ch.command(drop_table_sql(name))
    ch.command(create_table_sql(name, v))
    try:
        ch.insert_arrow(
            name,
            pa.table(
                {
                    "row_id": pa.array(
                        np.arange(len(raw), dtype=np.uint64), pa.uint64()
                    ),
                    "v": cast_column(arr, ch_type),
                }
            ),
        )
        sql, actual = range_predicate(
            values, 0.25, np.random.default_rng(3), ch_type=ch_type
        )
        [(count,)] = ch.query_rows(f"SELECT count() FROM {name} WHERE {sql}")
        assert count == round(actual * len(raw)) == round(0.25 * len(raw))
        sql, actual = in_predicate(values, 0.05, 2, ch_type=ch_type)
        [(count,)] = ch.query_rows(f"SELECT count() FROM {name} WHERE {sql}")
        assert count == round(actual * len(raw)) == 2
    finally:
        ch.command(drop_table_sql(name))


def test_date_literal_from_a_nullable_integer_dataset():
    """A nullable int dataset reaches the renderer as float64 (NaN holes);
    the literal must still name the date the loader stores for that row."""
    import pyarrow as pa
    import pyarrow.compute as pc

    from chbench.harness.loader import arrow_type_for
    from chbench.harness.queries import _render_literal_typed

    arr = pa.array([5, None, 7], type=pa.int32())
    value = arr.to_numpy(zero_copy_only=False)[0]  # np.float64(5.0)
    stored = pc.cast(arr, arrow_type_for("Nullable(Date)"))[0].as_py()
    assert _render_literal_typed(value, "Nullable(Date)") == f"'{stored.isoformat()}'"


@pytest.mark.parametrize("ch_type", ["Int32", "Nullable(UInt32)", "Nullable(Int64)"])
def test_integer_column_renders_a_float_value_as_an_integer(ch_type):
    # a nullable Int32 reaches the renderer as float64 (numpy has no nullable int)
    assert _render_literal(1061684000.0, ch_type=ch_type) == "1061684000"
    assert _render_literal(np.float64(-7.0), ch_type=ch_type) == "-7"


def test_integer_column_keeps_a_python_int_exact():
    assert _render_literal(2**63 + 5, ch_type="Nullable(UInt64)") == str(2**63 + 5)


def test_string_column_renders_bytes_as_text_or_unhex():
    assert _render_literal(b"ab'c", ch_type="String") == "'ab\\'c'"
    assert _render_literal(b"\xff\x00", ch_type="String") == "unhex('ff00')"
    # valid UTF-8 with a NUL, a backslash and a newline: only the backslash is escaped
    assert _render_literal(b"a\x00\\\n", ch_type="String") == "'a\x00\\\\\n'"
