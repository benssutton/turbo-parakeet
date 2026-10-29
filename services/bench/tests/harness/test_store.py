import json
import sqlite3
from types import SimpleNamespace as NS

import numpy as np

from synthdata.generate import SolveRecord
from chbench.harness.store import Store


def manifest(dataset_id="d1"):
    return NS(
        dataset_id=dataset_id,
        spec=NS(to_json=lambda: '{"rows": 1}'),
        achieved={"cardinality_ratio": 0.5},
        warnings=["miss: entropy_bits: x"],
        targets={},
        within_tolerance={},
        solves={},
        clip_collapsed=0,
        round_collapsed=0,
    )


def variant():
    return NS(
        ch_type="Float64",
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


def load_result():
    return NS(
        status="ok",
        error=None,
        insert_ms=12.5,
        stats=NS(
            compressed_bytes=100,
            uncompressed_bytes=400,
            marks_bytes=8,
            rows=50,
            parts=1,
            index_compressed_bytes=30,
            index_uncompressed_bytes=60,
            index_marks_bytes=4,
        ),
    )


def qv():
    return NS(
        kind="filter",
        predicate_kind="range",
        target_fraction=0.1,
        values_n=None,
        predicate_sql="v BETWEEN 1 AND 2",
        actual_fraction=0.11,
    )


def sample(n):
    return NS(
        sample_no=n,
        client_ms=1.5,
        server_ms=1.0,
        read_rows=10,
        read_bytes=80,
        memory_bytes=1000,
        selected_marks=2,
        selected_granules=2,
        profile_events={"SelectedParts": 1},
        ran_at="2026-09-15T00:00:00",
    )


def test_schema_has_the_five_tables(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        names = {
            r[0]
            for r in s.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert names >= {"datasets", "tables", "queries", "samples", "runs"}


def test_tables_uses_marks_bytes_not_marks(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        cols = [r[1] for r in s.conn.execute("PRAGMA table_info(tables)")]
    assert "marks_bytes" in cols and "marks" not in cols


def test_upserts_are_idempotent(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        for _ in range(2):
            s.upsert_dataset(manifest(), "datasets/d1.arrow")
            s.upsert_table("t1", "d1", variant(), load_result(), "26.8.2.7")
            s.upsert_query("q1", "t1", qv(), "EXPLAIN text")
        assert s.conn.execute("SELECT count() FROM datasets").fetchone()[0] == 1
        assert s.conn.execute("SELECT count() FROM tables").fetchone()[0] == 1
        assert s.conn.execute("SELECT count() FROM queries").fetchone()[0] == 1


def test_samples_accumulate_and_are_counted(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v")
        s.upsert_query("q1", "t1", qv(), "")
        run = s.begin_run(NS(name="suite", source_yaml="name: suite"))
        for n in range(3):
            s.insert_sample("q1", run, sample(n))
        s.finish_run(run)
        assert s.samples_held("q1") == 3
        assert s.samples_held("q_other") == 0


def test_query_rows_joins_for_the_report(tmp_path):
    # `suite_name` is what puts the table in `suite_tables`, which is the
    # join `query_rows` filters through; the orchestrator
    # always passes it.
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v", suite_name="suite")
        s.upsert_query("q1", "t1", qv(), "")
        run = s.begin_run(NS(name="suite", source_yaml="name: suite"))
        s.insert_sample("q1", run, sample(0))
        rows = s.query_rows("suite")
    assert len(rows) == 1
    r = rows[0]
    assert r["codec"] == "none" and r["kind"] == "filter" and r["client_ms"] == 1.5
    assert r["compressed_bytes"] == 100 and r["actual_fraction"] == 0.11
    assert (
        r["index_compressed_bytes"],
        r["index_uncompressed_bytes"],
        r["index_marks_bytes"],
    ) == (30, 60, 4)


def test_older_store_gains_the_index_size_columns(tmp_path):
    path = tmp_path / "old.db"
    with Store(path) as s:
        s.init_schema()
        for c in (
            "index_compressed_bytes",
            "index_uncompressed_bytes",
            "index_marks_bytes",
        ):
            s.conn.execute(f"ALTER TABLE tables DROP COLUMN {c}")
        s.conn.commit()
    with Store(path) as s:
        s.init_schema()
        cols = {r[1] for r in s.conn.execute("PRAGMA table_info(tables)")}
    assert {
        "index_compressed_bytes",
        "index_uncompressed_bytes",
        "index_marks_bytes",
    } <= cols


def test_skipped_table_is_stored_with_its_reason(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        v = variant()
        v.status = "skipped"
        v.skip_reason = "T64 codec is not supported for specified type Float64."
        s.upsert_table("t2", "d1", v, None, "v")
        row = s.conn.execute(
            "SELECT status, skip_reason, compressed_bytes FROM tables WHERE table_id='t2'"
        ).fetchone()
    assert row == (
        "skipped",
        "T64 codec is not supported for specified type Float64.",
        None,
    )


def test_each_sample_is_committed_immediately(tmp_path):
    """Ctrl-C safety: a sample must be durable before the next one starts."""
    p = tmp_path / "r.db"
    with Store(p) as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v")
        s.upsert_query("q1", "t1", qv(), "")
        run = s.begin_run(NS(name="suite", source_yaml=""))
        s.insert_sample("q1", run, sample(0))
        other = sqlite3.connect(p)
        assert other.execute("SELECT count() FROM samples").fetchone()[0] == 1
        other.close()


def test_tables_for_suite_uses_ddl_table_name(tmp_path):
    from chbench.harness.ddl import table_name

    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v", suite_name="suite")
        s.upsert_query("q1", "t1", qv(), "")
        run = s.begin_run(NS(name="suite", source_yaml=""))
        s.insert_sample("q1", run, sample(0))
        names = s.tables_for_suite("suite")
    assert names == [table_name("suite", "t1")]


def test_dataset_json_survives_numpy_scalars_and_solve_records(tmp_path):
    """`_json_default`/`_as_plain` are otherwise dead code: the plain
    `manifest()` helper only ever carries Python floats and an empty
    `solves` dict. Exercise both with a real `np.float64`, a real
    `np.int64` (which must come back as an int, not `1.0`) and a real
    `SolveRecord` (not a stub)."""
    m = manifest()
    m.achieved = {"cardinality_ratio": np.float64(0.5), "entropy_bits": np.int64(4)}
    m.solves = {"entropy_bits": SolveRecord(status="ok", message="converged")}
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(m, "p")
        row = s.conn.execute(
            "SELECT achieved_json FROM datasets WHERE dataset_id = 'd1'"
        ).fetchone()
    payload = json.loads(row[0])
    assert payload["achieved"]["cardinality_ratio"] == 0.5
    entropy = payload["achieved"]["entropy_bits"]
    assert entropy == 4 and isinstance(entropy, int)
    assert payload["solves"]["entropy_bits"] == {"status": "ok", "message": "converged"}


def test_tables_for_suite_finds_table_with_no_samples_yet(tmp_path):
    """A table whose run was interrupted before any query produced a
    sample still needs to be dropped by `clean` -- it was already created in
    ClickHouse. Deriving `tables_for_suite` through queries/samples/runs
    would miss it; the `suite_tables` join table (populated by
    `upsert_table`'s `suite_name` keyword) must not."""
    from chbench.harness.ddl import table_name

    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v", suite_name="suite")
        s.begin_run(NS(name="suite", source_yaml=""))
        # No upsert_query, no insert_sample.
        names = s.tables_for_suite("suite")
    assert names == [table_name("suite", "t1")]


def test_json_fallback_survives_a_multi_element_array():
    """`.item()` raises on a multi-element array; the fallback chain must
    still hold rather than aborting the whole dump."""
    import json

    import numpy as np

    from chbench.harness.store import _json_default

    assert json.dumps({"a": np.arange(3)}, default=_json_default)


def test_errored_query_stores_its_status_and_message(tmp_path):
    """A query failure is recorded with its message and the run
    continues. Without this the row is indistinguishable from a query the
    run never reached."""
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v")
        # The orchestrator's pre-run write, then the post-run rewrite.
        s.upsert_query("q1", "t1", qv(), "")
        s.upsert_query(
            "q1",
            "t1",
            qv(),
            "",
            status="error",
            error="DatabaseError: Code: 241. Memory limit exceeded",
        )
        row = s.conn.execute(
            "SELECT status, error FROM queries WHERE query_id='q1'"
        ).fetchone()
    assert row[0] == "error"
    assert row[1].startswith("DatabaseError: Code: 241")


def test_explain_for_returns_the_stored_plan_and_empty_when_unknown(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v")
        s.upsert_query("q1", "t1", qv(), "ReadFromMergeTree")
        assert s.explain_for("q1") == "ReadFromMergeTree"
        assert s.explain_for("never-stored") == ""
        # A row written with no plan reads back as "", never None.
        s.upsert_query("q2", "t1", qv(), "")
        assert s.explain_for("q2") == ""


def test_latest_suite_yaml_is_the_most_recent_runs_text(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        assert s.latest_suite_yaml("suite") is None
        s.begin_run(NS(name="suite", source_yaml="name: suite\npercentiles: old"))
        s.begin_run(NS(name="suite", source_yaml="name: suite\npercentiles: new"))
        s.begin_run(NS(name="other", source_yaml="name: other"))
        assert s.latest_suite_yaml("suite").endswith("new")
        assert s.latest_suite_yaml("nosuch") is None


def test_init_schema_adds_columns_missing_from_an_older_store(tmp_path):
    """A results.db created before `queries.status`/`error` existed must gain
    them, not fail on the next upsert: samples accumulate across runs, so an
    existing store is the normal case."""
    path = tmp_path / "old.db"
    with Store(path) as s:
        s.init_schema()
        s.conn.execute("ALTER TABLE queries DROP COLUMN status")
        s.conn.execute("ALTER TABLE queries DROP COLUMN error")
        s.conn.commit()
    with Store(path) as s:
        s.init_schema()
        columns = {
            r[1] for r in s.conn.execute("PRAGMA table_info(queries)").fetchall()
        }
        assert {"status", "error"} <= columns
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v")
        s.upsert_query("q1", "t1", qv(), "", status="error", error="boom")
        assert (
            s.conn.execute("SELECT error FROM queries WHERE query_id='q1'").fetchone()[
                0
            ]
            == "boom"
        )


def test_query_rows_follows_suite_tables_not_the_run_that_sampled(tmp_path):
    """A sample belongs to the table (and so, through
    `suite_tables`, to every suite that owns that table), not to the suite
    whose run happened to take it. Two suites sharing a cell share its
    `table_id` and `query_id`; suite B would otherwise skip every query as
    already held (`samples_held` counts globally) and then report nothing,
    because the samples sit under suite A's run."""
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_dataset(manifest(), "p")
        s.upsert_table("t1", "d1", variant(), load_result(), "v", suite_name="A")
        s.upsert_table("t1", "d1", variant(), load_result(), "v", suite_name="B")
        s.upsert_query("q1", "t1", qv(), "")
        run_a = s.begin_run(NS(name="A", source_yaml=""))
        s.insert_sample("q1", run_a, sample(0))
        s.insert_sample("q1", run_a, sample(1))
        rows_a, rows_b, rows_c = s.query_rows("A"), s.query_rows("B"), s.query_rows("C")
    assert len(rows_a) == 2 and len(rows_b) == 2 and rows_c == []
    assert {r["run_id"] for r in rows_b} == {run_a}


def test_unfinished_runs_lists_only_runs_without_finished_at(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        done = s.begin_run(NS(name="A", source_yaml=""))
        s.finish_run(done)
        stale = s.begin_run(NS(name="A", source_yaml=""))
        s.begin_run(NS(name="B", source_yaml=""))
        started = s.conn.execute(
            "SELECT started_at FROM runs WHERE run_id = ?", (stale,)
        ).fetchone()[0]
        assert s.unfinished_runs("A") == [(stale, started)]
        assert s.unfinished_runs("C") == []


def test_table_rows_lists_skipped_tables_without_samples(tmp_path):
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        s.upsert_table("t_ok", "d1", variant(), load_result(), "26.8", suite_name="su")
        skipped = variant()
        skipped.status, skipped.skip_reason, skipped.codec = (
            "skipped",
            "GCD on Float64",
            "GCD",
        )
        s.upsert_table("t_skip", "d1", skipped, None, "26.8", suite_name="su")
        s.upsert_table(
            "t_other", "d1", variant(), load_result(), "26.8", suite_name="other"
        )
        rows = {r["table_id"]: r for r in s.table_rows("su")}
    assert set(rows) == {"t_ok", "t_skip"}
    assert rows["t_ok"]["compressed_bytes"] == 100
    assert rows["t_skip"]["status"] == "skipped"
    assert rows["t_skip"]["skip_reason"] == "GCD on Float64"
    assert rows["t_skip"]["compressed_bytes"] is None
