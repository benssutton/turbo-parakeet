"""Load one dataset into one table variant and read its storage statistics.

Lifted from `report_storage`, `insert_float` and `insert_decimal` in
`spikes/spike_ch_roundtrip.py`, with the prints turned into `LoadResult`
fields and the DDL taken from `chbench.harness.ddl`.

Storage statistics come **only** from `column_data_compressed_bytes`,
`column_data_uncompressed_bytes` and `column_marks_bytes` in
`system.parts_columns`, filtered to column `v`. The plain `data_*` and
`marks` columns there are part-level totals repeated identically on every
column's row (FINDINGS §5). Those per-column figures exist only for Wide
parts, which every `CREATE TABLE` from `ddl.py` forces.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from synthdata import Dataset
from chbench.harness.client import CHClient
from chbench.harness.ddl import create_table_sql, drop_table_sql, optimize_sql
from chbench.harness.validate import TableVariant


@dataclass(frozen=True)
class StorageStats:
    compressed_bytes: int
    uncompressed_bytes: int
    marks_bytes: int
    rows: int
    parts: int
    # skip-index sizes (`system.data_skipping_indices`); 0 when the table has no index
    index_compressed_bytes: int = 0
    index_uncompressed_bytes: int = 0
    index_marks_bytes: int = 0


# Rows per Arrow record batch sent by `load_table`: the server's default
# insert block (`min_insert_block_size_rows` = 1 048 449, rounded up). The
# server squashes small batches into blocks of that size but never splits a
# large one, and a `LowCardinality` column's `minmax` / `set` index costs
# O(block rows) per granule, so one 10M-row batch made those inserts ~3x
# slower than this (FINDINGS §11). The stored bytes are the same either way.
INSERT_BATCH_ROWS = 1_048_576


@dataclass(frozen=True)
class LoadResult:
    status: str  # "ok" | "error"
    error: str | None
    insert_ms: float | None  # None when the insert did not complete
    stats: StorageStats | None


_WRAPPER_RE = re.compile(r"^(?:LowCardinality|Nullable)\((.*)\)$")
_DECIMAL_RE = re.compile(r"^Decimal(32|64|128)\((\d+)\)$")
_DATETIME64_RE = re.compile(r"^DateTime64\((\d)(?:, '([^']+)')?\)$")
_DATETIME_ZONE_RE = re.compile(r"^DateTime\('([^']+)'\)$")
_TIME64_RE = re.compile(r"^Time64\((\d)\)$")

# Arrow types for the fixed-width scalar ClickHouse types.
_SCALAR_ARROW = {
    "Float32": pa.float32(),
    "Float64": pa.float64(),
    "Int8": pa.int8(),
    "Int16": pa.int16(),
    "Int32": pa.int32(),
    "Int64": pa.int64(),
    "UInt8": pa.uint8(),
    "UInt16": pa.uint16(),
    "UInt32": pa.uint32(),
    "UInt64": pa.uint64(),
    "String": pa.string(),
    "Bool": pa.bool_(),
    "Date": pa.date32(),
    "Date32": pa.date32(),
    "DateTime": pa.timestamp("s"),
}

# Precision each Decimal width is reported at: 9, 18, 38 digits.
_DECIMAL_ARROW = {
    "32": (pa.decimal32, 9),
    "64": (pa.decimal64, 18),
    "128": (pa.decimal128, 38),
}

_DATETIME64_UNIT = {"0": "s", "3": "ms", "6": "us", "9": "ns"}

# Time64(p): Arrow has time32 for seconds and milliseconds, time64 for the rest.
_TIME64_ARROW = {
    "0": pa.time32("s"),
    "3": pa.time32("ms"),
    "6": pa.time64("us"),
    "9": pa.time64("ns"),
}


def base_type(ch_type: str) -> str:
    """Strip `LowCardinality(...)` and `Nullable(...)` wrappers, in any nesting."""
    while m := _WRAPPER_RE.match(ch_type):
        ch_type = m.group(1)
    return ch_type


def arrow_type_for(ch_type: str) -> pa.DataType | None:
    """The Arrow type to cast the dataset column to before `insert_arrow`, or
    `None` when the ClickHouse type has no mapping here (the column is then
    inserted uncast and the server decides).

    Decimals: `Decimal32(S)` -> `pa.decimal32(9, S)`,
    `Decimal64(S)` -> `pa.decimal64(18, S)`, `Decimal128(S)` ->
    `pa.decimal128(38, S)`. `String` and `LowCardinality(String)` ->
    `pa.string()`. Integer targets -> the matching Arrow integer type.
    `DateTime('tz')` and `DateTime64(p[, 'tz'])` -> a timestamp of the unit
    and zone; `Time64(p)` -> time32 (p 0, 3) or time64 (p 6, 9). `FixedString`
    and `Enum*` have no target: the dataset (`fixed_size_binary(n)` or string)
    is inserted as it is and the server converts it (FINDINGS §9). Wrappers are
    stripped first; nulls survive the cast.
    """
    t = base_type(ch_type)
    if t in _SCALAR_ARROW:
        return _SCALAR_ARROW[t]
    m = _DECIMAL_RE.match(t)
    if m:
        ctor, precision = _DECIMAL_ARROW[m.group(1)]
        return ctor(precision, int(m.group(2)))
    m = _DATETIME64_RE.match(t)
    if m and m.group(1) in _DATETIME64_UNIT:
        return pa.timestamp(_DATETIME64_UNIT[m.group(1)], tz=m.group(2))
    m = _DATETIME_ZONE_RE.match(t)
    if m:
        return pa.timestamp("s", tz=m.group(1))
    m = _TIME64_RE.match(t)
    if m and m.group(1) in _TIME64_ARROW:
        return _TIME64_ARROW[m.group(1)]
    return None


def cast_column(array: pa.Array, ch_type: str) -> pa.Array:
    """Cast the dataset array to the Arrow type matching `ch_type`. A safe
    cast: an out-of-range value (a negative into `UInt32`, `10.0` into
    `Decimal32(8)`) raises `ArrowInvalid` rather than being truncated."""
    target = arrow_type_for(ch_type)
    if target is None or array.type == target:
        return array
    return pc.cast(array, target)


def storage_stats(client: CHClient, name: str) -> StorageStats:
    """Per-column storage of column `v` in `name`, summed over active parts.

    Uses `column_data_compressed_bytes`, `column_data_uncompressed_bytes` and
    `column_marks_bytes`: the true per-column figures. Never the part-level
    `data_compressed_bytes` / `data_uncompressed_bytes` / `marks`, which are
    the whole part's totals repeated on every column's row (FINDINGS §5).
    `rows` is the table's row count and `parts` the active part count. The
    `index_*` fields sum the table's skip indexes from
    `system.data_skipping_indices` (0 without one).
    """
    database, _, table = name.rpartition(".")
    database = database or client.connection.database
    rows = client.query_rows(
        "SELECT sum(column_data_compressed_bytes), sum(column_data_uncompressed_bytes), "
        "sum(column_marks_bytes), sum(rows), count() "
        "FROM system.parts_columns "
        "WHERE database = {db:String} AND table = {t:String} AND column = 'v' AND active",
        parameters={"db": database, "t": table},
    )
    compressed, uncompressed, marks, nrows, parts = rows[0]
    idx = client.query_rows(
        "SELECT sum(data_compressed_bytes), sum(data_uncompressed_bytes), sum(marks_bytes) "
        "FROM system.data_skipping_indices WHERE database = {db:String} AND table = {t:String}",
        parameters={"db": database, "t": table},
    )
    ic, iu, im = (int(x or 0) for x in idx[0])
    return StorageStats(
        compressed_bytes=int(compressed),
        uncompressed_bytes=int(uncompressed),
        marks_bytes=int(marks),
        rows=int(nrows),
        parts=int(parts),
        index_compressed_bytes=ic,
        index_uncompressed_bytes=iu,
        index_marks_bytes=im,
    )


def load_table(
    client: CHClient,
    name: str,
    variant: TableVariant,
    dataset: Dataset,
    *,
    insert_batch_rows: int | None = INSERT_BATCH_ROWS,
) -> LoadResult:
    """Drop, create, insert, optimise and measure one table variant.

    The insert is `insert_arrow` of a two-column table: `row_id UInt64` from
    0 and `v` cast to the variant's column type (`cast_column`), sent as
    record batches of at most `insert_batch_rows` rows (`None`: one batch;
    zero-copy slices either way). `insert_ms` times the insert alone, not the
    `OPTIMIZE TABLE ... FINAL` after it. Any exception at any step becomes
    `status="error"` with its message and `stats=None`, and the table is
    dropped (best effort); nothing is raised, so the run continues. Acceptance at DDL time is not proof that the
    insert works (deferred-findings note), so an insert failure on an
    accepted variant is a recorded error, not an impossibility.
    """
    insert_ms: float | None = None
    try:
        client.command(drop_table_sql(name))
        client.command(create_table_sql(name, variant))
        v = cast_column(dataset.array, variant.ch_type)
        row_id = pa.array(np.arange(len(v), dtype=np.uint64), pa.uint64())
        batch = pa.table({"row_id": row_id, "v": v})
        if insert_batch_rows is not None:
            batch = pa.Table.from_batches(
                batch.to_batches(max_chunksize=insert_batch_rows), schema=batch.schema
            )
        t0 = time.perf_counter()
        client.insert_arrow(name, batch)
        insert_ms = (time.perf_counter() - t0) * 1000
        client.command(optimize_sql(name))
        stats = storage_stats(client, name)
    except Exception as e:  # noqa: BLE001
        # Drop whatever was created so an errored variant never leaves a
        # half-loaded table in bench; a failure of the drop itself must not
        # mask the original error.
        try:
            client.command(drop_table_sql(name))
        except Exception:  # noqa: BLE001
            pass
        return LoadResult(
            status="error",
            error=f"{type(e).__name__}: {e}",
            insert_ms=insert_ms,
            stats=None,
        )
    return LoadResult(status="ok", error=None, insert_ms=insert_ms, stats=stats)
