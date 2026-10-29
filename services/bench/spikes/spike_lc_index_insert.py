"""Spike 11: why does LowCardinality(String) with a minmax or set(100) index insert ~15x slower
(45.7 / 46.5 s against 3.2 s with bloom_filter, int_encodings at 10M rows; ToDo 3)?

Two parts, both on server 26.8.31.2 at the time of writing:

- `arrow`: int_encodings' dataset (its own `dataset:` block; 1000 distinct values at 1M and at
  10M rows) loaded as String and LowCardinality(String) under no index, minmax, set(100) and
  bloom_filter, through the harness's own `load_table`, once with the whole table in one Arrow
  record batch (`insert_batch_rows=None`, the loader before this spike) and once in batches of
  `INSERT_BATCH_ROWS`. For each load: the client-timed `insert_ms` (the insert alone), the
  `system.query_log` rows of the insert (matched by `has(tables, 'bench.<table>')`, which works
  on this server; the `Insert` row plus, for an insert under 10 MiB, which the server runs as
  an async insert, the `AsyncInsertFlush` row that writes the part — their ProfileEvents
  summed), the `OPTIMIZE TABLE ... FINAL` row after it (matched by its query text), and the
  stored bytes (`storage_stats`).
- `sql`: the reproduction without Arrow or the harness: `INSERT INTO ... SELECT` from
  `numbers()` into the same DDL, varying rows, the insert block size (`max_block_size`,
  `min_insert_block_size_rows`, `max_insert_block_size`) and `index_granularity`. If the
  index cost scales with rows at a fixed block size it is per row; if it scales with
  block rows x granules per block, each granule's index update costs O(block rows).

Prints a TSV and writes spike_lc_index_insert_results.tsv beside this file (the last column is
the Insert's full ProfileEvents as JSON). Needs the chbench-ch container; ~10 min.

    python -m spikes.spike_lc_index_insert      (from services/bench/)
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from synthdata.cache import load_or_generate
from chbench.cli import DEFAULT_CACHE_DIR
from chbench.harness.client import CHClient, Connection
from chbench.harness.ddl import drop_table_sql, table_name
from chbench.harness.loader import INSERT_BATCH_ROWS, load_table
from chbench.harness.orchestrate import _achieved_bounds
from chbench.harness.suite import parse_suite
from chbench.harness.validate import expand_grid

# int_encodings' dataset block, with `rows` and `cardinality_ratio` free (1000 distinct values
# at both sizes; at 10M rows the ratio is int_encodings' own 0.0001, so its cached dataset is
# reused).
SUITE = """
name: spike_lc
dataset:
  rows: {rows}
  arrow_type: int32
  value_range: [1000000, 9999999]
  null_count: 0
  cardinality_ratio: {ratio}
  entropy_bits: 4.0
  runs_z: -0.25
  seed: 42
clickhouse:
  type: [String]
  nullable: [false]
  primary_key: [false]
  projection: [false]
  codec: [none, LowCardinality]
  skip_index: [none, minmax, "set(100)", bloom_filter]
  compression: [LZ4]
queries:
  kinds: [{{kind: full_scan}}]
  samples: 1
results:
  percentiles: [50]
"""

SKIP_US = "MergeTreeDataWriterSkipIndicesCalculationMicroseconds"
EVENTS = ("UserTimeMicroseconds", "SystemTimeMicroseconds", "OSCPUWaitMicroseconds")
HEADER = (
    "path\trows\tbatch_rows\tindex_granularity\tvariant\tinsert_ms\tserver_ms\tskip_index_us"
    "\tparts\tasync_flushes\t"
    + "\t".join(EVENTS)
    + "\toptimize_ms\tcompressed_bytes\tprofile_json"
)
# An insert under `async_insert_max_data_size` (10 MiB) runs as an async insert (`async_insert`
# is 1 by default on 26.8): its `Insert` row then carries only the receive, and the part is
# written by an `AsyncInsertFlush` row whose ProfileEvents hold the writer's events. Both
# kinds are summed, over the rows since the load started.
_NOW_US = "SELECT toUnixTimestamp64Micro(now64(6))"
_INSERTS_SINCE = """
SELECT query_kind, query_duration_ms, ProfileEvents FROM system.query_log
WHERE type = 'QueryFinish' AND query_kind IN ('Insert', 'AsyncInsertFlush')
  AND has(tables, {t:String}) AND toUnixTimestamp64Micro(event_time_microseconds) >= {s:Int64}
"""
_LAST_OPTIMIZE = """
SELECT query_duration_ms FROM system.query_log
WHERE type = 'QueryFinish' AND query = {q:String}
ORDER BY event_time_microseconds DESC LIMIT 1
"""
_BY_ID = """
SELECT query_duration_ms, ProfileEvents FROM system.query_log
WHERE type = 'QueryFinish' AND query_id = {q:String}
"""
SQL_TABLE = "bench.spike_lc_sql"


def _line(
    path,
    rows,
    batch,
    ig,
    label,
    insert_ms,
    server_ms,
    profile,
    optimize_ms,
    nbytes,
    flushes=0,
):
    return [
        path,
        str(rows),
        str(batch),
        str(ig),
        label,
        f"{insert_ms:.0f}",
        str(server_ms),
        str(profile.get(SKIP_US, 0)),
        str(profile.get("InsertedWideParts", 0)),
        str(flushes),
        *(str(profile.get(k, 0)) for k in EVENTS),
        str(optimize_ms),
        str(nbytes),
        json.dumps(profile, sort_keys=True),
    ]


def arrow_part(client: CHClient) -> list[list[str]]:
    out = []
    for rows in (1_000_000, 10_000_000):
        suite = parse_suite(SUITE.format(rows=rows, ratio=1000 / rows))
        dataset = load_or_generate(suite.dataset, Path(DEFAULT_CACHE_DIR))
        lo, hi = _achieved_bounds(dataset.array)
        variants = expand_grid(
            suite,
            null_count=0,
            value_min=lo,
            value_max=hi,
            clickhouse_version=client.version(),
            array=dataset.array,
        )
        for batch in (None, INSERT_BATCH_ROWS):
            for i, v in enumerate(x for x in variants if x.status == "ok"):
                name = table_name("spike_lc", f"{i:08x}")
                since = client.query_rows(_NOW_US)[0][0]
                load = load_table(client, name, v, dataset, insert_batch_rows=batch)
                assert load.status == "ok", load.error
                client.flush_logs()
                log = client.query_rows(
                    _INSERTS_SINCE, parameters={"t": name, "s": since}
                )
                server_ms = next((ms for kind, ms, _ in log if kind == "Insert"), None)
                flushes = sum(kind == "AsyncInsertFlush" for kind, _, _ in log)
                profile: dict = {}
                for _, _, events in log:
                    for k, x in dict(events).items():
                        profile[k] = profile.get(k, 0) + x
                opt = client.query_rows(
                    _LAST_OPTIMIZE, parameters={"q": f"OPTIMIZE TABLE {name} FINAL"}
                )
                label = f"{v.ch_type}/{v.skip_index}"  # ch_type carries the LowCardinality wrapper
                line = _line(
                    "arrow",
                    rows,
                    batch or rows,
                    8192,
                    label,
                    load.insert_ms,
                    server_ms,
                    profile,
                    opt[0][0] if opt else None,
                    load.stats.compressed_bytes,
                    flushes,
                )
                out.append(line)
                print("\t".join(line[:-1]), flush=True)
                client.command(drop_table_sql(name))
    return out


def sql_insert(client, ch_type, skip_index, rows, block=None, ig=8192) -> list[str]:
    """One `INSERT ... SELECT` from `numbers()`: 1000 distinct strings, scattered."""
    client.command(drop_table_sql(SQL_TABLE))
    ix = "" if skip_index == "none" else f", INDEX ix v TYPE {skip_index} GRANULARITY 1"
    client.command(
        f"CREATE TABLE {SQL_TABLE} (row_id UInt64, v {ch_type} CODEC(LZ4){ix}) "
        f"ENGINE = MergeTree ORDER BY tuple() SETTINGS index_granularity = {ig}, "
        "min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"
    )
    settings = (
        {
            "max_block_size": block,
            "min_insert_block_size_rows": block,
            "max_insert_block_size": block,
        }
        if block
        else None
    )
    qid = str(uuid.uuid4())
    t0 = time.perf_counter()
    client.query_rows(
        f"INSERT INTO {SQL_TABLE} SELECT number, toString(1000000 + (number * 7919) % 1000) "
        f"FROM numbers({rows})",
        query_id=qid,
        settings=settings,
    )
    ms = (time.perf_counter() - t0) * 1000
    client.flush_logs()
    log = client.query_rows(_BY_ID, parameters={"q": qid})
    server_ms, profile = (log[0][0], dict(log[0][1])) if log else (None, {})
    label = f"{ch_type}/{skip_index}"
    return _line(
        "sql", rows, block or "default", ig, label, ms, server_ms, profile, "", ""
    )


def sql_part(client: CHClient) -> list[list[str]]:
    lc, s = "LowCardinality(String)", "String"
    cells = [
        (t, ix, 10_000_000)
        for t in (s, lc)
        for ix in ("none", "minmax", "set(100)", "bloom_filter")
    ]
    cells += [(lc, "minmax", 1_000_000)]
    cells += [(lc, "minmax", 10_000_000, b) for b in (8192, 65_536, 262_144)]
    cells += [(lc, "minmax", 10_000_000, None, ig) for ig in (1024, 65_536)]
    out = []
    for cell in cells:
        line = sql_insert(client, *cell)
        out.append(line)
        print("\t".join(line[:-1]), flush=True)
    client.command(drop_table_sql(SQL_TABLE))
    return out


def main() -> None:
    client = CHClient(Connection.from_env())
    print(HEADER.rsplit("\t", 1)[0], flush=True)
    lines = arrow_part(client) + sql_part(client)
    path = Path(__file__).with_name("spike_lc_index_insert_results.tsv")
    text = "\n".join([HEADER] + ["\t".join(x) for x in lines]) + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
