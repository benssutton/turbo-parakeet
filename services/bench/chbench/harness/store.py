"""The SQLite results store.

Five tables (`datasets`, `tables`, `queries`, `samples`, `runs`), where
the storage-marks column is `marks_bytes`, not `marks` (matching
`StorageStats.marks_bytes` and `column_marks_bytes` in `system.parts_columns`,
never the part-level `marks`, which `loader.py`'s docstring explains is a
different, unrelated figure).

Three additions go beyond those five tables as first designed:

- `tables.warnings`: the `TableVariant`'s applicability warnings (e.g.
  "projection is redundant with primary_key"). Every other JSON-ish input
  (the manifest's achieved/targets/within_tolerance/solves, the sample's
  profile_events) is folded into the columns the five tables already have
  (`datasets.achieved_json`, `samples.profile_events_json`) rather than
  given a column of its own; only this one had nowhere to go.
- `queries.status` and `queries.error`: the query's own outcome over the run's
  rounds and its message. An insert or query failure is recorded with its message and the
  run continues; recording it only on `tables` leaves
  a query whose *run* failed indistinguishable from one the run never
  reached, which is exactly the distinction an unattended overnight run
  needs the next morning.
- `suite_tables(suite_name, table_id)`: a join table, not a `suite_name`
  column on `tables`, because `table_id` is derived from the
  dataset and variant alone, so two suites can legitimately share one table
  id and a column would attribute it to whichever suite ran last.
  `upsert_table`'s optional `suite_name` keyword populates it, so
  `tables_for_suite` can find a table `clean` must drop even when the suite's
  run was interrupted before any query produced a sample (the gap the
  runs/samples join used to leave open).

`SCHEMA_VERSION` (stamped in `PRAGMA user_version`) guards the layout: `init_schema` raises
`StoreVersionError` on a populated older store, whose samples cannot be paired by round, while
`query_rows` reads it anyway, substituting NULL for the columns it lacks.

Every write to `datasets`, `tables` and `queries` is an `INSERT ... ON
CONFLICT DO UPDATE`, idempotent on the table's primary key (`created_at` is
never in the UPDATE's SET list, so a re-run keeps the row's original creation
time). `samples` has no natural key -- it is append-only by design
(samples accumulate across runs so `samples_held` can report how many are
already banked) -- so `insert_sample` is a plain INSERT,
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
    explain_plan TEXT,
    status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    query_id TEXT NOT NULL,
    sample_no INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    round_no INTEGER,
    client_ms REAL,
    server_ms REAL,
    cpu_us REAL,
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
    finished_at TEXT,
    quiet_before TEXT,
    quiet_after TEXT
);

CREATE TABLE IF NOT EXISTS suite_tables (
    suite_name TEXT NOT NULL,
    table_id TEXT NOT NULL,
    PRIMARY KEY (suite_name, table_id)
);
"""

SCHEMA_VERSION = (
    2  # 2: interleaved rounds (samples.round_no, samples.cpu_us, runs.quiet_*), ToDo 3
)


class StoreVersionError(RuntimeError):
    """A store written before the current schema: readable, but new samples need a new store."""


# Columns `query_rows` returns, per source table: a name, or (column, alias). A column an older
# store lacks reads as NULL, so `report` and the evaluation loop can still read it.
_ROW_COLUMNS = (
    (
        "tb",
        "tables",
        (
            "table_id",
            "dataset_id",
            "ch_type",
            "nullable",
            "primary_key",
            "projection",
            "codec",
            "skip_index",
            "compression",
            "status",
            "skip_reason",
            "compressed_bytes",
            "uncompressed_bytes",
            "marks_bytes",
            "index_compressed_bytes",
            "index_uncompressed_bytes",
            "index_marks_bytes",
            "rows",
            "insert_ms",
            "clickhouse_version",
            "created_at",
            ("warnings", "table_warnings"),
        ),
    ),
    (
        "q",
        "queries",
        (
            "query_id",
            "kind",
            "predicate_kind",
            "target_fraction",
            "actual_fraction",
            "predicate_sql",
            "explain_plan",
            ("status", "query_status"),
            ("error", "query_error"),
        ),
    ),
    (
        "s",
        "samples",
        (
            "sample_no",
            "run_id",
            "round_no",
            "client_ms",
            "server_ms",
            "cpu_us",
            "read_rows",
            "read_bytes",
            "memory_bytes",
            "selected_marks",
            "selected_granules",
            "profile_events_json",
            "ran_at",
        ),
    ),
)

_QUERY_ROWS_FROM = """
FROM samples s
JOIN queries q ON q.query_id = s.query_id
JOIN tables tb ON tb.table_id = q.table_id
JOIN suite_tables st ON st.table_id = tb.table_id
WHERE st.suite_name = ?
"""


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
        """Create the schema (idempotent) and stamp `user_version`. A store written by an
        earlier schema raises `StoreVersionError`: its samples were taken without rounds and
        cannot be paired with new ones, so new samples go to a new store. A store stamped by
        a newer chbench raises too, rather than being re-stamped down."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        populated = self.conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'samples'"
        ).fetchone()[0]
        if populated and version < SCHEMA_VERSION:
            raise StoreVersionError(
                f"{self.path} is a schema-{version} store, written before interleaved rounds "
                f"(ToDo 3); `chbench report` still reads it, but new samples need a new store: "
                f"pass --store"
            )
        if version > SCHEMA_VERSION:
            raise StoreVersionError(
                f"{self.path} is a schema-{version} store, written by a newer chbench (this "
                f"one writes schema {SCHEMA_VERSION}): upgrade chbench, or pass --store"
            )
        self.conn.executescript(_SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _query_rows_sql(self) -> str:
        parts = []
        for alias, table, columns in _ROW_COLUMNS:
            present = {
                r[1]
                for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for c in columns:
                column, name = c if isinstance(c, tuple) else (c, c)
                parts.append(
                    f"{alias}.{column} AS {name}"
                    if column in present
                    else f"NULL AS {name}"
                )
        return "SELECT " + ", ".join(parts) + _QUERY_ROWS_FROM

    # -- datasets ---------------------------------------------------------

    def upsert_dataset(self, manifest, arrow_path: str) -> None:
        """`manifest.achieved`, `.targets`, `.within_tolerance`, `.solves`,
        `.clip_collapsed` and `.round_collapsed` are bundled into one JSON
        object in `achieved_json`, since `datasets` has no column
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
        `(suite_name, table_id)` pair is also upserted into `suite_tables`,
        which is how `tables_for_suite` finds this table even
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
        """`status` and `error` carry the query's own outcome over the run's
        rounds ("ok", "error" or "interrupted", with the runner's message).

        A failure is recorded on the *query*,
        not only on the table: without these two columns a query whose run
        failed is indistinguishable, after an unattended run, from one the
        run never reached. Both are keyword-only and default to `None`, so a
        caller that only has the query's shape -- the pre-run write in the
        orchestrator, and this module's own tests -- is unaffected.
        """
        self.conn.execute(
            """
            INSERT INTO queries (query_id, table_id, kind, predicate_kind, target_fraction,
                                  actual_fraction, predicate_sql, explain_plan, status, error)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(query_id) DO UPDATE SET
                table_id=excluded.table_id,
                kind=excluded.kind,
                predicate_kind=excluded.predicate_kind,
                target_fraction=excluded.target_fraction,
                actual_fraction=excluded.actual_fraction,
                predicate_sql=excluded.predicate_sql,
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
            "ORDER BY started_at DESC, rowid DESC LIMIT 1",  # rowid: same-tick runs
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

    def record_quiet(self, run_id: str, when: str, snapshot: dict) -> None:
        """The quietness snapshot (`quiet.snapshot`) taken `when` ("before" / "after") the rounds."""
        column = {"before": "quiet_before", "after": "quiet_after"}[when]
        self.conn.execute(
            f"UPDATE runs SET {column} = ? WHERE run_id = ?",
            (json.dumps(snapshot, default=_json_default), run_id),
        )
        self.conn.commit()

    def next_round(self, suite_name: str) -> int:
        """One past the highest `round_no` among the suite's samples; 0 when there are none.
        New rounds continue the numbering, so samples added by `--samples-add`, which re-runs
        every query, stay paired by round; a query added later runs only with the other
        newcomers, so it pairs with them alone.
        """
        row = self.conn.execute(
            "SELECT max(s.round_no) FROM samples s "
            "JOIN queries q ON q.query_id = s.query_id "
            "JOIN suite_tables st ON st.table_id = q.table_id "
            "WHERE st.suite_name = ?",
            (suite_name,),
        ).fetchone()
        return 0 if row[0] is None else int(row[0]) + 1

    def unfinished_runs(self, suite_name: str) -> list[tuple[str, str]]:
        """`(run_id, started_at)` of every run of `suite_name` whose
        `finished_at` is still NULL, oldest first.

        `finish_run` is called from a `finally`, so such a row means a
        process died without reaching it -- or is still running against
        this store. The orchestrator refuses to start a second
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
            INSERT INTO samples (query_id, sample_no, run_id, round_no, client_ms, server_ms,
                                  cpu_us, read_rows, read_bytes, memory_bytes, selected_marks,
                                  selected_granules, profile_events_json, ran_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                query_id,
                sample.sample_no,
                run_id,
                getattr(sample, "round_no", None),
                sample.client_ms,
                sample.server_ms,
                getattr(sample, "cpu_us", None),
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

        Filtered through `suite_tables`, not `runs`: a sample
        is owned by the table, and `samples_held` counts by `query_id`
        regardless of which run took it. Two suites sharing a cell share
        its `table_id` and `query_id`; the second one to run skips every
        query as already held, and a filter through `runs.suite_name` would
        then show it nothing, because the samples sit under the first
        suite's run."""
        cur = self.conn.execute(self._query_rows_sql(), (suite_name,))
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def table_rows(self, suite_name: str) -> list[dict]:
        """Every `tables` column of each table recorded against `suite_name`, including cells
        `expand_grid` skipped (status `skipped`, with `skip_reason`) and tables whose queries
        hold no samples — which `query_rows`, joined through `samples`, leaves out."""
        cur = self.conn.execute(
            "SELECT tb.* FROM tables tb "
            "JOIN suite_tables st ON st.table_id = tb.table_id "
            "WHERE st.suite_name = ? ORDER BY tb.table_id",
            (suite_name,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def tables_for_suite(self, suite_name: str) -> list[str]:
        """The rendered ClickHouse table names (`ddl.table_name(suite_name,
        table_id)` -- the exact string `create_table_sql`/`drop_table_sql`
        used, so `clean` can `DROP TABLE` it directly) of every table
        `upsert_table` recorded against this suite (from the
        `suite_tables` join table, not derived through `queries`/`samples`/
        `runs` -- that join missed a table whose run was interrupted before
        its first query produced a sample, even though the table was already
        created in ClickHouse and needed dropping)."""
        rows = self.conn.execute(
            "SELECT table_id FROM suite_tables WHERE suite_name = ?",
            (suite_name,),
        ).fetchall()
        return [table_name(suite_name, row[0]) for row in rows]
