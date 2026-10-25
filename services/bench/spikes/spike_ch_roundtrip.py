"""Spike 5: ClickHouse 26.8 round-trip via clickhouse-connect.

Checks: Arrow insert (float64 and decimal32/64, with a Float64-staging
fallback if the Arrow decimal path fails), OPTIMIZE FINAL, storage stats
from system.parts_columns, query with query_id + settings, query_log
capture, EXPLAIN indexes = 1, cache drop statements (each isolated so an
unsupported one does not abort the run), and a DDL applicability matrix
for (type x codec) and (type x skip index).

Corrections applied to the original brief (see task-6 brief + corrections):
1. Decimal32(8) probe restructured into Part A (real data, cast to
   decimal32(9,8), expected to succeed) and Part B (single value 10.0,
   isolated, expected to overflow) instead of assuming the whole dataset
   overflows.
2. Every `SYSTEM DROP ... CACHE` statement is wrapped in its own
   try/except so an unsupported cache-drop statement cannot abort the
   script before the matrices print.
3. The Arrow float->decimal cast and the Arrow insert are each allowed to
   fail independently; on failure the script falls back to inserting into
   a Float64 staging table and doing `INSERT INTO ... SELECT toDecimalN(v,
   8)`. Which path actually worked is recorded per decimal width.
"""
from __future__ import annotations

import sys
import time
import uuid

import clickhouse_connect
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

CH = dict(host="localhost", port=8123, username="CLICKHOUSE_USER", password="CLICKHOUSE_PASSWORD")


def report_storage(client, name: str) -> None:
    # NOTE: data_compressed_bytes / data_uncompressed_bytes on
    # system.parts_columns are PART-level totals (summed across every
    # column in the part) repeated identically on each column's row -- the
    # brief's literal SQL used those and so silently attributed the whole
    # part's size to each column. The true per-column figures are
    # column_data_compressed_bytes / column_data_uncompressed_bytes, with
    # column_marks_bytes for marks; those are used here instead.
    rows = client.query(
        "SELECT column, sum(column_data_compressed_bytes), sum(column_data_uncompressed_bytes), "
        "sum(column_marks_bytes), sum(rows), count() AS parts FROM system.parts_columns "
        "WHERE database = 'bench' AND table = {t:String} AND active GROUP BY column",
        parameters={"t": name}).result_rows
    for r in rows:
        print(f"  {name}.{r[0]}: compressed={r[1]} uncompressed={r[2]} marks_bytes={r[3]} rows={r[4]} parts={r[5]}")


def insert_float(client, name: str, col_ddl: str, row_id, v) -> None:
    client.command(f"DROP TABLE IF EXISTS bench.{name}")
    # Force Wide part format (rather than the Compact format MergeTree
    # would pick by default for a ~1.6MB table): ClickHouse only tracks
    # per-column column_data_compressed_bytes/column_data_uncompressed_bytes
    # for Wide parts -- a Compact part stores every column interleaved in
    # one file and reports 0 for all per-column byte counters. Without this,
    # the per-column storage query added for Critical 1 would silently
    # return all zeros instead of real numbers.
    client.command(f"CREATE TABLE bench.{name} (row_id UInt64, {col_ddl}) "
                    f"ENGINE = MergeTree ORDER BY tuple() "
                    f"SETTINGS min_bytes_for_wide_part=0, min_rows_for_wide_part=0")
    t0 = time.perf_counter()
    client.insert_arrow(f"bench.{name}", pa.table({"row_id": row_id, "v": pa.array(v, pa.float64())}))
    print(f"{name}: inserted (arrow float64) in {(time.perf_counter()-t0)*1000:.0f} ms")
    client.command(f"OPTIMIZE TABLE bench.{name} FINAL")
    report_storage(client, name)


def insert_decimal(client, name: str, col_ddl: str, width: int, precision: int, scale: int,
                    row_id, v) -> str:
    """Try Arrow decimal insert directly; fall back to Float64 staging + toDecimalN.

    Returns the path that worked: "arrow_decimal" or "float64_staging_fallback".
    Raises if both paths fail (caller decides how to report that).
    """
    client.command(f"DROP TABLE IF EXISTS bench.{name}")
    # See insert_float's comment: force Wide part format so the per-column
    # storage query returns real numbers instead of the 0s a Compact part
    # would give for column_data_compressed_bytes / column_data_uncompressed_bytes.
    client.command(f"CREATE TABLE bench.{name} (row_id UInt64, {col_ddl}) "
                    f"ENGINE = MergeTree ORDER BY tuple() "
                    f"SETTINGS min_bytes_for_wide_part=0, min_rows_for_wide_part=0")

    arrow_type = pa.decimal64(precision, scale) if width == 64 else pa.decimal32(precision, scale)
    # Only the cast and the insert belong inside the try: this record is what the
    # design's insert-path decision rests on, so an OPTIMIZE or storage-report
    # failure must not be misattributed to the Arrow path.
    path = None
    try:
        t0 = time.perf_counter()
        arr = pc.cast(pa.array(v), arrow_type)
        client.insert_arrow(f"bench.{name}", pa.table({"row_id": row_id, "v": arr}))
        print(f"{name}: inserted (arrow decimal{width}({precision},{scale})) in "
              f"{(time.perf_counter()-t0)*1000:.0f} ms")
        path = "arrow_decimal"
    except Exception as e:  # noqa: BLE001
        print(f"{name}: arrow decimal{width} cast/insert failed, falling back to Float64 "
              f"staging: {type(e).__name__}: {e}")

    if path is None:
        # Fallback: Float64 staging table -> INSERT ... SELECT toDecimalN(v, scale)
        client.command(f"DROP TABLE IF EXISTS bench.{name}")
        client.command(f"CREATE TABLE bench.{name} (row_id UInt64, {col_ddl}) "
                        f"ENGINE = MergeTree ORDER BY tuple() "
                        f"SETTINGS min_bytes_for_wide_part=0, min_rows_for_wide_part=0")
        client.command(f"DROP TABLE IF EXISTS bench.{name}_stage")
        client.command(f"CREATE TABLE bench.{name}_stage (row_id UInt64, v Float64) "
                        f"ENGINE = MergeTree ORDER BY tuple()")
        func = "toDecimal64" if width == 64 else "toDecimal32"
        t0 = time.perf_counter()
        try:
            client.insert_arrow(f"bench.{name}_stage",
                                pa.table({"row_id": row_id, "v": pa.array(v, pa.float64())}))
            client.command(f"INSERT INTO bench.{name} "
                           f"SELECT row_id, {func}(v, {scale}) FROM bench.{name}_stage")
            print(f"{name}: inserted (float64 staging + {func}) in "
                  f"{(time.perf_counter()-t0)*1000:.0f} ms")
        finally:
            # Drop the staging table on the failure path too, so it cannot leak.
            client.command(f"DROP TABLE IF EXISTS bench.{name}_stage")
        path = "float64_staging_fallback"

    client.command(f"OPTIMIZE TABLE bench.{name} FINAL")
    report_storage(client, name)
    return path


def main() -> None:
    # EXPLAIN output can contain box-drawing characters (U+2502 etc.); the
    # default console encoding on Windows (cp1252) can't encode them, so
    # force stdout to UTF-8 rather than let a print() crash the run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    client = clickhouse_connect.get_client(**CH)
    print("version:", client.command("SELECT version()"))
    print("clickhouse_connect:", clickhouse_connect.__version__)
    client.command("CREATE DATABASE IF NOT EXISTS bench")

    rng = np.random.default_rng(0)
    v = np.round(rng.uniform(-1, 10, 100_000), 8)
    row_id = pa.array(np.arange(100_000, dtype=np.uint64))

    print("\n--- arrow insert: spike_f64 (Float64) ---")
    insert_float(client, "spike_f64", "v Float64 CODEC(Gorilla, ZSTD)", row_id, v)

    print("\n--- arrow insert: spike_d64 (Decimal64(8)) ---")
    path_d64 = insert_decimal(client, "spike_d64", "v Decimal64(8) CODEC(Delta, LZ4)",
                               64, 18, 8, row_id, v)

    print("\n--- arrow insert: spike_d32 Part A (Decimal32(8), real data -- expected to fit) ---")
    print("Decimal32(8) representable range: -9.99999999 to +9.99999999 "
          "(9 significant digits, scale 8). rng.uniform(-1, 10) draws values "
          "strictly < 10, but rounding to 8 dp sends any draw in "
          "[9.999999995, 10) to exactly 10.00000000, which overflows: the "
          "range check must be applied to the rounded values, not the draw "
          "bounds.")
    path_d32 = insert_decimal(client, "spike_d32", "v Decimal32(8) CODEC(Delta, LZ4)",
                               32, 9, 8, row_id, v)

    print("\n--- arrow insert: spike_d32 Part B (Decimal32(8), single value 10.0 -- expected overflow) ---")
    client.command("DROP TABLE IF EXISTS bench.spike_d32_overflow")
    client.command("CREATE TABLE bench.spike_d32_overflow (v Decimal32(8)) "
                    "ENGINE = MergeTree ORDER BY tuple()")
    try:
        arr10 = pc.cast(pa.array([10.0], pa.float64()), pa.decimal32(9, 8))
        client.insert_arrow("bench.spike_d32_overflow", pa.table({"v": arr10}))
        print("spike_d32_overflow: arrow cast+insert of 10.0 unexpectedly SUCCEEDED")
    except Exception as e:  # noqa: BLE001
        print(f"spike_d32_overflow: arrow cast/insert of 10.0 failed: {type(e).__name__}: {e}")
    try:
        client.command("INSERT INTO bench.spike_d32_overflow VALUES (10.0)")
        print("spike_d32_overflow: plain INSERT VALUES (10.0) unexpectedly SUCCEEDED")
    except Exception as e:  # noqa: BLE001
        print(f"spike_d32_overflow: plain INSERT VALUES (10.0) failed: {type(e).__name__}: {e}")
    client.command("DROP TABLE IF EXISTS bench.spike_d32_overflow")

    print(f"\ndecimal insert path used: decimal64={path_d64}  decimal32={path_d32}")

    print("\n--- timed query + query_log ---")
    sql = "SELECT count(), sum(v) FROM bench.spike_f64 WHERE v BETWEEN 2 AND 3"
    for drop in (True, False):
        if drop:
            print("  cache-drop attempts:")
            for stmt in ("SYSTEM DROP MARK CACHE", "SYSTEM DROP UNCOMPRESSED CACHE",
                         "SYSTEM DROP QUERY CONDITION CACHE"):
                try:
                    client.command(stmt)
                    print(f"    {stmt}: ok")
                except Exception as e:  # noqa: BLE001
                    print(f"    {stmt}: unavailable: {type(e).__name__}: {e}")
        qid = str(uuid.uuid4())
        t0 = time.perf_counter()
        # clickhouse_connect 1.8.0's Client.query() has no separate `query_id`
        # keyword arg (passing query_id=... raises TypeError: unexpected
        # keyword argument). Passing it via `transport_settings={"query_id":
        # ...}` also does NOT work: transport_settings are sent as HTTP
        # headers, and ClickHouse's HTTP interface only recognizes query_id
        # as a URL parameter, so the server silently ignores the header and
        # autogenerates its own query_id (confirmed: system.query_log never
        # gets a row for the id we sent that way). The call shape that
        # actually works on this version is to put query_id INSIDE the
        # `settings` dict -- "query_id" is one of HttpClient's
        # valid_transport_settings, so `_validate_settings` passes it through
        # unfiltered into the URL params ClickHouse actually reads.
        client.query(sql, settings={"query_id": qid, "use_query_condition_cache": 0,
                                     "use_query_cache": 0})
        client_ms = (time.perf_counter() - t0) * 1000
        client.command("SYSTEM FLUSH LOGS")
        log = client.query(
            "SELECT query_duration_ms, read_rows, read_bytes, memory_usage, "
            "ProfileEvents['SelectedMarks'], ProfileEvents['SelectedRanges'], "
            "ProfileEvents['SelectedParts'], ProfileEvents['QueryConditionCacheHits'], "
            "ProfileEvents['QueryConditionCacheMisses'], "
            # Indexing a Map with [] returns the value type's zero for an
            # absent key, so a 0 above is ambiguous between "key exists
            # with value 0" and "key does not exist at all". mapContains
            # settles it directly instead of relying on that ambiguous 0.
            "mapContains(ProfileEvents, 'QueryConditionCacheHits'), "
            "mapContains(ProfileEvents, 'QueryConditionCacheMisses') "
            "FROM system.query_log WHERE query_id = {q:String} AND type = 'QueryFinish'",
            parameters={"q": qid}).result_rows
        print(f"cold={drop} client_ms={client_ms:.1f} query_log={log}")

    print("\n--- EXPLAIN indexes = 1 ---")
    for line in client.query("EXPLAIN indexes = 1 " + sql).result_rows:
        print("  " + line[0])

    print("\n--- DDL applicability matrix: type x codec ---")
    types = ["Float64", "Decimal32(8)", "Decimal64(8)", "Int32", "UInt32", "String",
             "LowCardinality(String)", "Date", "DateTime64(3)", "Nullable(Int32)"]
    codecs = ["Delta", "DoubleDelta", "Gorilla", "T64", "FPC"]
    print(f"{'type':<24}" + "".join(f"{c:>12}" for c in codecs))
    for t in types:
        row = f"{t:<24}"
        for c in codecs:
            client.command("DROP TABLE IF EXISTS bench.spike_probe")
            try:
                client.command(f"CREATE TABLE bench.spike_probe (v {t} CODEC({c}, LZ4)) "
                               f"ENGINE = MergeTree ORDER BY tuple()")
                row += f"{'ok':>12}"
            except Exception as e:  # noqa: BLE001
                row += f"{'REJECT':>12}"
                # Full text, untruncated: these become quotable validate.py
                # messages, so the retained script must reproduce them
                # completely from a single run rather than relying on an
                # out-of-band probe.
                print(f"    {t} + {c}: {e}")
        print(row)

    print("\n--- DDL applicability matrix: type x skip index ---")
    indexes = ["minmax", "bloom_filter", "set(100)"]
    print(f"{'type':<24}" + "".join(f"{i:>14}" for i in indexes))
    for t in types:
        row = f"{t:<24}"
        for i in indexes:
            client.command("DROP TABLE IF EXISTS bench.spike_probe")
            try:
                client.command(f"CREATE TABLE bench.spike_probe (v {t}, INDEX ix v TYPE {i} GRANULARITY 1) "
                               f"ENGINE = MergeTree ORDER BY tuple()")
                row += f"{'ok':>14}"
            except Exception as e:  # noqa: BLE001
                row += f"{'REJECT':>14}"
                print(f"    {t} + {i}: {e}")
        print(row)
    client.command("DROP TABLE IF EXISTS bench.spike_probe")

    print("\n--- cleanup ---")
    for name in ("spike_f64", "spike_d64", "spike_d32", "spike_d32_overflow", "spike_probe",
                 "spike_d64_stage", "spike_d32_stage"):
        client.command(f"DROP TABLE IF EXISTS bench.{name}")
        print(f"  dropped bench.{name} (if existed)")


if __name__ == "__main__":
    main()
