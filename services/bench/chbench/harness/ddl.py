"""Render the DDL strings the harness sends to ClickHouse for a table
variant: `CREATE TABLE`, `DROP TABLE`, and `OPTIMIZE TABLE FINAL`.

This module is the only place SQL DDL strings are built. It applies no type
wrapping: `TableVariant.ch_type` already carries the complete column type in
the server's required order (`LowCardinality(Nullable(X))`), wrapped upstream
by `chbench.harness.validate.expand_grid`. `nullable` and `codec` remain on
the variant as axes for the store and report; `ddl.py` reads `codec` only to
decide the `CODEC(...)` clause — `LowCardinality` is a type wrapper, not a
codec, so it never appears inside `CODEC(...)`.

Every `CREATE TABLE` forces Wide parts (`SETTINGS min_bytes_for_wide_part = 0,
min_rows_for_wide_part = 0`): per-column storage statistics exist only for
Wide parts, and ClickHouse silently chooses Compact below a size threshold.
"""
from __future__ import annotations

from chbench.harness.validate import TableVariant, split_skip_index

_WIDE_SETTINGS = "SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"


def table_name(suite_name: str, table_id: str) -> str:
    """`bench.<suite>_<first 8 hex of table_id>`."""
    return f"bench.{suite_name}_{table_id[:8]}"


def _codec_clause(variant: TableVariant) -> str:
    if variant.codec in ("none", "LowCardinality"):
        return f"CODEC({variant.compression})"
    return f"CODEC({variant.codec}, {variant.compression})"


def create_table_sql(name: str, variant: TableVariant) -> str:
    """The `CREATE TABLE` statement for one table variant.

    Column order: `row_id`, then `v`. Clause order: columns, then INDEX
    (inside the column list), then PROJECTION (inside the column list), then
    ENGINE, ORDER BY, SETTINGS — the order ClickHouse's own DDL syntax
    requires.
    """
    columns = ["row_id UInt64", f"v {variant.ch_type} {_codec_clause(variant)}"]

    if variant.skip_index != "none":
        index, granularity = split_skip_index(variant.skip_index)
        columns.append(f"INDEX ix v TYPE {index} GRANULARITY {granularity}")

    if variant.projection:
        columns.append("PROJECTION p (SELECT * ORDER BY v)")

    order_by = "v" if variant.primary_key else "tuple()"

    return (
        f"CREATE TABLE {name} ({', '.join(columns)}) "
        f"ENGINE = MergeTree ORDER BY {order_by} {_WIDE_SETTINGS}"
    )


def drop_table_sql(name: str) -> str:
    return f"DROP TABLE IF EXISTS {name}"


def optimize_sql(name: str) -> str:
    return f"OPTIMIZE TABLE {name} FINAL"
