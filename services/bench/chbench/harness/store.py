"""The SQLite results store (spec section 6.1).

The five tables there, verbatim, with one correction from the task brief:
the storage-marks column is `marks_bytes`, not `marks` (matching
`StorageStats.marks_bytes` and `column_marks_bytes` in `system.parts_columns`,
never the part-level `marks`, which `loader.py`'s docstring explains is a
different, unrelated figure).

Two additions go beyond the spec's literal list of five tables:

- `tables.warnings` (ruling R19: approved, to be recorded as a spec 6.1
  addendum later): the `TableVariant`'s applicability warnings (e.g.
  "projection is redundant with primary_key"). Every other JSON-ish input
  (the manifest's achieved/targets/within_tolerance/solves, the sample's
  profile_events) is folded into the columns the spec already lists
  (`datasets.achieved_json`, `samples.profile_events_json`) rather than
  given a column of its own; only this one had nowhere to go.
- `queries.status` and `queries.error`: the `RunResult`'s own outcome and
  message. Spec 6.4 requires an insert or query failure to be recorded with
  its message and the run to continue; recording it only on `tables` leaves
  a query whose *run* failed indistinguishable from one the run never
  reached, which is exactly the distinction an unattended overnight run
  needs the next morning.
- `suite_tables(suite_name, table_id)`: a join table, not a `suite_name`
  column on `tables` (ruling R18), because `table_id` is derived from the
  dataset and variant alone, so two suites can legitimately share one table
  id and a column would attribute it to whichever suite ran last.
  `upsert_table`'s optional `suite_name` keyword populates it, so
  `tables_for_suite` can find a table `clean` must drop even when the suite's
  run was interrupted before any query produced a sample (the gap the
  runs/samples join used to leave open).

Every write to `datasets`, `tables` and `queries` is an `INSERT ... ON
CONFLICT DO UPDATE`, idempotent on the table's primary key (`created_at` is
never in the UPDATE's SET list, so a re-run keeps the row's original creation
time). `samples` has no natural key -- it is append-only by design (task
brief point 4: samples accumulate across runs so `samples_held` can report
how many are already banked) -- so `insert_sample` is a plain INSERT,
committed immediately after it runs: a Ctrl-C between samples loses at most
the sample in flight, never an earlier one.

`Manifest`, `TableVariant`, `LoadResult`/`StorageStats`, `QueryVariant`,
`Sample` and `Suite` are consumed by attribute access only (never
isinstance-checked), so the same code accepts either the real dataclasses
or a duck-typed stand-in exposing the same names -- which is what this
module's own tests use.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from chbench.harness.ddl import table_name

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    achieved_json TEXT NOT NULL,
    warnings TEXT NOT NULL,
    arrow_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tables (
    table_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    ch_type TEXT,
    nullable INTEGER,
    primary_key INTEGER,
    projection INTEGER,
    codec TEXT,
    skip_index TEXT,
    compression TEXT,
    status TEXT,
    skip_reason TEXT,
    compressed_bytes INTEGER,
    uncompressed_bytes INTEGER,
    marks_bytes INTEGER,
    index_compressed_bytes INTEGER,
    index_uncompressed_bytes INTEGER,
    index_marks_bytes INTEGER,
    rows INTEGER,
    insert_ms REAL,
    clickhouse_version TEXT,
    created_at TEXT NOT NULL,
    warnings TEXT
);

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    table_id TEXT NOT NULL,
    kind TEXT,
    predicate_kind TEXT,
    target_fraction REAL,
    actual_fraction REAL,
    predicate_sql TEXT,
    cache_mode TEXT,
    qcc INTEGER,
    explain_plan TEXT,
    status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    query_id TEXT NOT NULL,
    sample_no INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    client_ms REAL,
    server_ms REAL,
    read_rows INTEGER,
    read_bytes INTEGER,
    memory_bytes INTEGER,
    selected_marks INTEGER,
    selected_granules INTEGER,
    profile_events_json TEXT,
    ran_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    suite_name TEXT NOT NULL,
    suite_yaml TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS suite_tables (
    suite_name TEXT NOT NULL,
    table_id TEXT NOT NULL,
    PRIMARY KEY (suite_name, table_id)
);
"""

# Columns returned by query_rows, in this order, mapped to the aliased SELECT
# below. Only the two join keys (table_id, query_id) are deduplicated: every
# other column name is unique to the table it comes from (checked against
# the full column lists of tables/queries/samples above), so nothing else
# collides.
_QUERY_ROWS_SQL = """
SELECT
    tb.table_id AS table_id,
    tb.dataset_id AS dataset_id,
    tb.ch_type AS ch_type,
    tb.nullable AS nullable,
    tb.primary_key AS primary_key,
    tb.projection AS projection,
    tb.codec AS codec,
    tb.skip_index AS skip_index,
    tb.compression AS compression,
    tb.status AS status,
    tb.skip_reason AS skip_reason,
    tb.compressed_bytes AS compressed_bytes,
    tb.uncompressed_bytes AS uncompressed_bytes,
    tb.marks_bytes AS marks_bytes,
    tb.index_compressed_bytes AS index_compressed_bytes,
    tb.index_uncompressed_bytes AS index_uncompressed_bytes,
    tb.index_marks_bytes AS index_marks_bytes,
    tb.rows AS rows,
    tb.insert_ms AS insert_ms,
    tb.clickhouse_version AS clickhouse_version,
    tb.created_at AS created_at,
    tb.warnings AS table_warnings,
    q.query_id AS query_id,
    q.kind AS kind,
    q.predicate_kind AS predicate_kind,
    q.target_fraction AS target_fraction,
    q.actual_fraction AS actual_fraction,
    q.predicate_sql AS predicate_sql,
    q.cache_mode AS cache_mode,
    q.qcc AS qcc,
    q.explain_plan AS explain_plan,
    q.status AS query_status,
    q.error AS query_error,
    s.sample_no AS sample_no,
    s.run_id AS run_id,
    s.client_ms AS client_ms,
    s.server_ms AS server_ms,
    s.read_rows AS read_rows,
    s.read_bytes AS read_bytes,
    s.memory_bytes AS memory_bytes,
    s.selected_marks AS selected_marks,
    s.selected_granules AS selected_granules,
    s.profile_events_json AS profile_events_json,
    s.ran_at AS ran_at
FROM samples s
JOIN queries q ON q.query_id = s.query_id
JOIN tables tb ON tb.table_id = q.table_id
JOIN suite_tables st ON st.table_id = tb.table_id
WHERE st.suite_name = ?
"""


# Columns added to `_SCHEMA` after the first `results.db` files existed.
# `init_schema` adds each one to a store that predates it; see
# `Store._add_missing_columns`.
_ADDED_COLUMNS = (
    ("queries", "status", "TEXT"),
    ("queries", "error", "TEXT"),
    ("tables", "index_compressed_bytes", "INTEGER"),
    ("tables", "index_uncompressed_bytes", "INTEGER"),
    ("tables", "index_marks_bytes", "INTEGER"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o):
    """Fallback for `json.dumps`: a numpy scalar (achieved metrics can carry
    e.g. `np.float64` or `np.int64`) converts via `.item()`, which returns
    the matching native Python type -- `int` stays `int`, not `1.0` -- so
    integer-ness survives the round trip; anything without `.item()` falls
    back to `float()`, then to `str()` rather than raising. `.item()` is
    itself guarded: on a multi-element array it raises, and the fallback
    chain must still hold rather than aborting the whole dump."""
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _as_plain(value):
    """A `SolveRecord` (or any dataclass) becomes a plain dict so `json.dumps`
    does not need to know about it; everything else passes through."""
    return asdict(value) if is_dataclass(value) else value


def _ran_at_text(ran_at) -> str:
    """`Sample.ran_at` is a `datetime.datetime` in the real dataclass; store
    it as ISO text. Accepts a plain string too (duck-typed callers), so this
    never assumes the attribute is a `datetime`."""
    to_iso = getattr(ran_at, "isoformat", None)
    return to_iso() if callable(to_iso) else str(ran_at)


class Store:
    """A connection to `results.db` plus the schema and query helpers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.conn.close()
        return False

    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self._add_missing_columns()
        self.conn.commit()

    def _add_missing_columns(self) -> None:
        """Add columns that a `results.db` created by an earlier version has
        not got. `CREATE TABLE IF NOT EXISTS` is a no-op on an existing
        table, so a new column would otherwise only reach a database created
        from scratch, and the first `upsert_query` against an older one would
        fail on `no such column`. Samples accumulate across runs by design,
        so an existing store is the normal case, not the exception, and
        losing one to a schema change would throw away measurements.
        """
        for table, column, decl in _ADDED_COLUMNS:
            existing = {
                row[1]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # -- datasets ---------------------------------------------------------

    def upsert_dataset(self, manifest, arrow_path: str) -> None:
        """`manifest.achieved`, `.targets`, `.within_tolerance`, `.solves`,
        `.clip_collapsed` and `.round_collapsed` are bundled into one JSON
        object in `achieved_json`, since spec 6.1 gives `datasets` no column
        for each separately; `.warnings` keeps its own listed column."""
        achieved_json = json.dumps(
            {
                "achieved": manifest.achieved,
                "targets": manifest.targets,
                "within_tolerance": manifest.within_tolerance,
                "solves": {k: _as_plain(v) for k, v in manifest.solves.items()},
                "clip_collapsed": manifest.clip_collapsed,
                "round_collapsed": manifest.round_collapsed,
            },
            default=_json_default,
        )
        self.conn.execute(
            """
            INSERT INTO datasets (dataset_id, spec_json, achieved_json, warnings,
                                   arrow_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                spec_json=excluded.spec_json,
                achieved_json=excluded.achieved_json,
                warnings=excluded.warnings,
                arrow_path=excluded.arrow_path
            """,
            (
                manifest.dataset_id,
                manifest.spec.to_json(),
                achieved_json,
                json.dumps(list(manifest.warnings), default=_json_default),
                arrow_path,
                _now(),
            ),
        )
        self.conn.commit()

    # -- tables -------------------------------------------------------------

    def upsert_table(
        self,
        table_id: str,
        dataset_id: str,
        variant,
        load_result,
        clickhouse_version: str,
        *,
        suite_name: str | None = None,
    ) -> None:
        """`status`/`skip_reason` reconcile two sources: a `variant` marked
        `skipped` never reached the loader, so its own status/reason wins;
        otherwise (`load_result` present) the load's own outcome ("ok" or
        "error", with the load's error text) is the table's real status --
        `variant.status` alone would say "ok" even for a table whose insert
        failed. Storage stats and `insert_ms` are None whenever there was no
        successful load to measure (skipped, or load_result is None).

        `suite_name` is optional (keyword-only, default `None`) so existing
        callers that only need the table row are unaffected; when given, the
        `(suite_name, table_id)` pair is also upserted into `suite_tables`
        (ruling R18), which is how `tables_for_suite` finds this table even
        before any query against it has produced a sample."""
        if variant.status == "skipped":
            status, reason, stats, insert_ms = (
                "skipped",
                variant.skip_reason,
                None,
                None,
            )
        elif load_result is not None:
            status, reason = load_result.status, load_result.error
            stats, insert_ms = load_result.stats, load_result.insert_ms
        else:
            status, reason, stats, insert_ms = (
                variant.status,
                variant.skip_reason,
                None,
                None,
            )

        self.conn.execute(
            """
            INSERT INTO tables (table_id, dataset_id, ch_type, nullable, primary_key,
                                 projection, codec, skip_index, compression, status,
                                 skip_reason, compressed_bytes, uncompressed_bytes,
                                 marks_bytes, index_compressed_bytes,
                                 index_uncompressed_bytes, index_marks_bytes,
                                 rows, insert_ms, clickhouse_version,
                                 created_at, warnings)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(table_id) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                ch_type=excluded.ch_type,
                nullable=excluded.nullable,
                primary_key=excluded.primary_key,
                projection=excluded.projection,
                codec=excluded.codec,
                skip_index=excluded.skip_index,
                compression=excluded.compression,
                status=excluded.status,
                skip_reason=excluded.skip_reason,
                compressed_bytes=excluded.compressed_bytes,
                uncompressed_bytes=excluded.uncompressed_bytes,
                marks_bytes=excluded.marks_bytes,
                index_compressed_bytes=excluded.index_compressed_bytes,
                index_uncompressed_bytes=excluded.index_uncompressed_bytes,
                index_marks_bytes=excluded.index_marks_bytes,
                rows=excluded.rows,
                insert_ms=excluded.insert_ms,
                clickhouse_version=excluded.clickhouse_version,
                warnings=excluded.warnings
            """,
            (
                table_id,
                dataset_id,
                variant.ch_type,
                variant.nullable,
                variant.primary_key,
                variant.projection,
                variant.codec,
                variant.skip_index,
                variant.compression,
                status,
                reason,
                stats.compressed_bytes if stats else None,
                stats.uncompressed_bytes if stats else None,
                stats.marks_bytes if stats else None,
                stats.index_compressed_bytes if stats else None,
                stats.index_uncompressed_bytes if stats else None,
                stats.index_marks_bytes if stats else None,
                stats.rows if stats else None,
                insert_ms,
                clickhouse_version,
                _now(),
                json.dumps(list(variant.warnings), default=_json_default),
            ),
        )
        if suite_name is not None:
            self.conn.execute(
                """
                INSERT INTO suite_tables (suite_name, table_id) VALUES (?, ?)
                ON CONFLICT(suite_name, table_id) DO NOTHING
                """,
                (suite_name, table_id),
            )
        self.conn.commit()

    # -- queries --------------------------------------------------------

    def upsert_query(
        self,
        query_id: str,
        table_id: str,
        qv,
        explain: str,
        *,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """`status` and `error` carry the `RunResult`'s own outcome ("ok",
        "error" or "interrupted", with the runner's message).

        Spec section 6.4 asks for a failure to be recorded on the *query*,
        not only on the table: without these two columns a query whose run
        failed is indistinguishable, after an unattended run, from one the
        run never reached. Both are keyword-only and default to `None`, so a
        caller that only has the query's shape -- the pre-run write in the
        orchestrator, and this module's own tests -- is unaffected.
        """
        self.conn.execute(
            """
            INSERT INTO queries (query_id, table_id, kind, predicate_kind, target_fraction,
                                  actual_fraction, predicate_sql, cache_mode, qcc,
                                  explain_plan, status, error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(query_id) DO UPDATE SET
                table_id=excluded.table_id,
                kind=excluded.kind,
                predicate_kind=excluded.predicate_kind,
                target_fraction=excluded.target_fraction,
                actual_fraction=excluded.actual_fraction,
                predicate_sql=excluded.predicate_sql,
                cache_mode=excluded.cache_mode,
                qcc=excluded.qcc,
                explain_plan=excluded.explain_plan,
                status=excluded.status,
                error=excluded.error
            """,
            (
                query_id,
                table_id,
                qv.kind,
                qv.predicate_kind,
                qv.target_fraction,
                qv.actual_fraction,
                qv.predicate_sql,
                qv.cache_mode,
                qv.qcc,
                explain,
                status,
                error,
            ),
        )
        self.conn.commit()

    def explain_for(self, query_id: str) -> str:
        """The EXPLAIN plan already stored for `query_id`, or `""`.

        The orchestrator writes a query's row before its samples, so a
        Ctrl-C between the first sample and the EXPLAIN cannot strand
        committed samples behind an inner join; it needs to carry the
        existing plan across that write rather than blank it.
        """
        row = self.conn.execute(
            "SELECT explain_plan FROM queries WHERE query_id = ?", (query_id,)
        ).fetchone()
        return (row[0] or "") if row else ""

    def latest_suite_yaml(self, suite_name: str) -> str | None:
        """The suite YAML stored with this suite's most recent run, or None.

        `chbench report` is given a suite *name*, not a path, so the file on
        disk may since have been edited, moved or deleted; this is the exact
        text that produced the samples being summarised.
        """
        row = self.conn.execute(
            "SELECT suite_yaml FROM runs WHERE suite_name = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (suite_name,),
        ).fetchone()
        return row[0] if row and row[0] else None

    # -- runs -------------------------------------------------------------

    def begin_run(self, suite) -> str:
        run_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO runs (run_id, suite_name, suite_yaml, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, NULL)",
            (run_id, suite.name, suite.source_yaml, _now()),
        )
        self.conn.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?", (_now(), run_id)
        )
        self.conn.commit()

    def unfinished_runs(self, suite_name: str) -> list[tuple[str, str]]:
        """`(run_id, started_at)` of every run of `suite_name` whose
        `finished_at` is still NULL, oldest first.

        `finish_run` is called from a `finally`, so such a row means a
        process died without reaching it -- or is still running against
        this store (ruling R33). The orchestrator refuses to start a second
        run over it unless told the first is gone.
        """
        rows = self.conn.execute(
            "SELECT run_id, started_at FROM runs "
            "WHERE suite_name = ? AND finished_at IS NULL ORDER BY started_at",
            (suite_name,),
        ).fetchall()
        return [(run_id, started_at) for run_id, started_at in rows]

    # -- samples --------------------------------------------------------

    def insert_sample(self, query_id: str, run_id: str, sample) -> None:
        """The runner's `on_sample` hook: one row per sample, committed
        immediately so a Ctrl-C between samples never loses one already
        delivered."""
        self.conn.execute(
            """
            INSERT INTO samples (query_id, sample_no, run_id, client_ms, server_ms,
                                  read_rows, read_bytes, memory_bytes, selected_marks,
                                  selected_granules, profile_events_json, ran_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                query_id,
                sample.sample_no,
                run_id,
                sample.client_ms,
                sample.server_ms,
                sample.read_rows,
                sample.read_bytes,
                sample.memory_bytes,
                sample.selected_marks,
                sample.selected_granules,
                json.dumps(dict(sample.profile_events), default=_json_default),
                _ran_at_text(sample.ran_at),
            ),
        )
        self.conn.commit()

    def samples_held(self, query_id: str) -> int:
        row = self.conn.execute(
            "SELECT count(*) FROM samples WHERE query_id = ?", (query_id,)
        ).fetchone()
        return int(row[0])

    # -- report / clean ---------------------------------------------------

    def query_rows(self, suite_name: str) -> list[dict]:
        """Every column of the three-way join of `tables`, `queries` and
        `samples` for one suite's tables (`tables.warnings` -> `table_warnings`,
        since a bare `warnings` would shadow nothing here but the rename
        keeps it unambiguous next to a future `queries`/`samples` warnings
        column). `table_id` and `query_id` are each selected once, from
        their owning table; `run_id` is the sample's own.

        Filtered through `suite_tables`, not `runs` (ruling R32): a sample
        is owned by the table, and `samples_held` counts by `query_id`
        regardless of which run took it. Two suites sharing a cell share
        its `table_id` and `query_id`; the second one to run skips every
        query as already held, and a filter through `runs.suite_name` would
        then show it nothing, because the samples sit under the first
        suite's run."""
        cur = self.conn.execute(_QUERY_ROWS_SQL, (suite_name,))
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def tables_for_suite(self, suite_name: str) -> list[str]:
        """The rendered ClickHouse table names (`ddl.table_name(suite_name,
        table_id)` -- the exact string `create_table_sql`/`drop_table_sql`
        used, so `clean` can `DROP TABLE` it directly) of every table
        `upsert_table` recorded against this suite (ruling R18: from the
        `suite_tables` join table, not derived through `queries`/`samples`/
        `runs` -- that join missed a table whose run was interrupted before
        its first query produced a sample, even though the table was already
        created in ClickHouse and needed dropping)."""
        rows = self.conn.execute(
            "SELECT table_id FROM suite_tables WHERE suite_name = ?",
            (suite_name,),
        ).fetchall()
        return [table_name(suite_name, row[0]) for row in rows]
