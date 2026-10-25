import pytest

from chbench.harness.ddl import create_table_sql, drop_table_sql, optimize_sql, table_name
from chbench.harness.validate import TableVariant


def variant(**kw):
    d = dict(ch_type="Float64", nullable=False, primary_key=False, projection=False,
             codec="none", skip_index="none", compression="LZ4", status="ok",
             skip_reason=None, warnings=())
    d.update(kw)
    return TableVariant(**d)


WIDE = "SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"


def test_table_name():
    assert table_name("float_codecs", "0123456789abcdef") == "bench.float_codecs_01234567"


def test_minimal_create():
    sql = create_table_sql("bench.t", variant())
    assert sql == (
        "CREATE TABLE bench.t (row_id UInt64, v Float64 CODEC(LZ4)) "
        f"ENGINE = MergeTree ORDER BY tuple() {WIDE}"
    )


def test_codec_and_compression():
    assert "v Decimal32(8) CODEC(Delta, ZSTD)" in create_table_sql(
        "bench.t", variant(ch_type="Decimal32(8)", codec="Delta", compression="ZSTD"))


def test_nullable_wraps_the_type():
    assert "v Nullable(Int32) CODEC(T64, LZ4)" in create_table_sql(
        "bench.t", variant(ch_type="Nullable(Int32)", nullable=True, codec="T64"))


def test_low_cardinality_is_a_type_wrapper_not_a_codec():
    sql = create_table_sql("bench.t", variant(ch_type="LowCardinality(String)", codec="LowCardinality"))
    assert "v LowCardinality(String) CODEC(LZ4)" in sql
    assert "CODEC(LowCardinality" not in sql


def test_low_cardinality_nullable_type_renders_verbatim():
    sql = create_table_sql(
        "bench.t",
        variant(ch_type="LowCardinality(Nullable(String))", nullable=True, codec="LowCardinality"),
    )
    assert "v LowCardinality(Nullable(String)) CODEC(LZ4)" in sql


def test_primary_key_orders_by_v():
    assert "ORDER BY v " in create_table_sql("bench.t", variant(primary_key=True))


def test_projection_and_index_clauses():
    sql = create_table_sql("bench.t", variant(projection=True, skip_index="set(100)"))
    assert "INDEX ix v TYPE set(100) GRANULARITY 1" in sql
    assert "PROJECTION p (SELECT * ORDER BY v)" in sql


def test_every_create_forces_wide_parts():
    for kw in ({}, {"primary_key": True}, {"projection": True}, {"skip_index": "minmax"}):
        assert create_table_sql("bench.t", variant(**kw)).endswith(WIDE)


def test_drop_and_optimize():
    assert drop_table_sql("bench.t") == "DROP TABLE IF EXISTS bench.t"
    assert optimize_sql("bench.t") == "OPTIMIZE TABLE bench.t FINAL"


@pytest.mark.parametrize("value,clause", [
    ("minmax", "TYPE minmax GRANULARITY 1"),
    ("minmax:4", "TYPE minmax GRANULARITY 4"),
    ("set(125):2", "TYPE set(125) GRANULARITY 2"),
    ("bloom_filter(0.025):8", "TYPE bloom_filter(0.025) GRANULARITY 8"),
    ("bloom_filter(0.01)", "TYPE bloom_filter(0.01) GRANULARITY 1"),
])
def test_skip_index_granularity_suffix(value, clause):
    assert clause in create_table_sql("bench.t", variant(skip_index=value))
