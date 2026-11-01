"""run_suite with in-process engines and no server, no real engine: every worker runs
tests/harness/fake_worker.py (faults through CHBENCH_FAKE); ClickHouse, where a suite has it, is
the FakeClient. Every worker the run starts is checked closed at the end of each test.
"""

import io
import json
import sqlite3

import pytest
from rich.console import Console

from chbench.harness import orchestrate, quiet, versions
from chbench.harness.engines import inproc
from chbench.harness.orchestrate import run_suite
from chbench.harness.store import Store
from chbench.harness.suite import parse_suite
from tests.harness.fakes import FakeClient, rounds
from tests.harness.test_rounds import SNAP, _ok_load
from tests.harness.test_suite import MINIMAL, drop_block

POLARS_BLOCK = (
    "polars:\n"
    "  type: [Float64, Float32]\n"
    "  storage: [memory, ipc]\n"
    "  compression: [none, ZSTD(3)]\n"
    "  threads: [2]\n"
)
SUITE = (
    drop_block(MINIMAL, "clickhouse")
    .replace("name: t", "name: t_inproc")
    .replace("queries:", POLARS_BLOCK + "queries:")
    .replace(
        "    - {kind: full_scan}",
        "    - {kind: full_scan}\n    - {kind: filter, target_fraction: 0.1, predicate: in, values: 1}",
    )
)  # 8 cells, 2 skipped (memory + ZSTD(3)), 6 tables x 2 queries, samples 3
MIXED = MINIMAL.replace("name: t", "name: t_mixed").replace(
    "queries:",
    "polars:\n  type: [Float64]\n  storage: [memory]\n  compression: [none]\n"
    "  threads: [2]\nqueries:",
)  # 1 ClickHouse + 1 Polars table, 1 query each, samples 3
ONE_QUERY = SUITE.replace(
    "\n    - {kind: filter, target_fraction: 0.1, predicate: in, values: 1}", ""
)  # the same suite with its first query only


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setitem(
        inproc.FACTORIES, "polars", "tests.harness.fake_worker:FakeWorker"
    )
    monkeypatch.setitem(
        inproc.FACTORIES, "datafusion", "tests.harness.fake_worker:FakeWorker"
    )
    monkeypatch.setattr(quiet, "snapshot", lambda client: dict(SNAP))
    monkeypatch.setattr(
        inproc, "TARGET_BATCH_NS", 0
    )  # k = 1: one fake run per warm-up job
    monkeypatch.delenv("CHBENCH_FAKE", raising=False)
    holder = {"client": FakeClient()}
    monkeypatch.setattr(orchestrate, "CHClient", lambda conn: holder["client"])
    monkeypatch.setattr(orchestrate, "load_table", _ok_load)
    started = []
    real_init = inproc.InprocEngine.__init__

    def recording(self, *a, **k):
        real_init(self, *a, **k)
        started.append(self)

    monkeypatch.setattr(inproc.InprocEngine, "__init__", recording)
    yield holder
    # No orphan worker, whatever the test did (a failing or interrupted run included).
    alive = [e.pid for e in started if e._proc.is_alive()]
    for e in started:
        e.close()
    assert alive == [], f"worker process(es) left running: {alive}"


def _run(tmp_path, text=SUITE, **kw):
    kw.setdefault("console", Console(file=io.StringIO(), width=400))
    return run_suite(
        parse_suite(text), store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds", **kw
    )


def _queries(tmp_path):
    with Store(tmp_path / "r.db") as s:
        return s.conn.execute("SELECT status, error FROM queries").fetchall()


def test_an_in_process_suite_runs_without_a_server(fake, tmp_path):
    r = _run(tmp_path)
    assert (r.tables_ok, r.tables_skipped, r.tables_error) == (6, 2, 0)
    assert r.queries_run == 12
    assert r.samples_written == 36
    assert fake["client"].calls == []  # no ClickHouse at all
    with Store(tmp_path / "r.db") as s:
        rows = s.query_rows("t_inproc")
        tables = s.table_rows("t_inproc")
        (text,) = s.conn.execute("SELECT engine_versions FROM runs").fetchone()
    assert {(r["engine"], r["threads"]) for r in rows} == {("polars", 2)}
    assert all(r["server_ms"] > 0 and r["cpu_us"] is not None for r in rows)
    ipc = [t for t in tables if t["storage"] == "ipc" and t["status"] == "ok"]
    # a real PrepareResult reaches the store's ipc_bytes column
    assert len(ipc) == 4
    assert all(t["ipc_bytes"] > 0 for t in ipc)
    assert "clickhouse" not in json.loads(text)
    assert json.loads(text)["polars"]
    assert {st for st, _ in _queries(tmp_path)} == {"ok"}


def test_ipc_files_are_reused_on_resume_and_a_full_rerun_prepares_nothing(
    fake, tmp_path, monkeypatch
):
    _run(tmp_path)
    files = sorted((tmp_path / "ds" / "ipc").glob("*.arrow"))
    stamps = [f.stat().st_mtime_ns for f in files]
    assert len(files) == 4
    r2 = _run(tmp_path, samples_add=1)
    assert r2.samples_written == 12
    assert [f.stat().st_mtime_ns for f in files] == stamps
    calls = []
    real = inproc.InprocEngine.prepare
    monkeypatch.setattr(
        inproc.InprocEngine,
        "prepare",
        lambda self, *a: calls.append(a) or real(self, *a),
    )
    r3 = _run(tmp_path)
    assert r3.samples_written == 0
    assert r3.queries_skipped_existing == 12
    assert calls == []


def test_a_wrong_first_answer_is_an_error_with_both_answers_and_takes_no_samples(
    fake, tmp_path, monkeypatch
):
    # execute #1 is the first job's discarded cold run, #2 its first answer (the one compared)
    monkeypatch.setenv("CHBENCH_FAKE", "wrong=2")
    r = _run(tmp_path)
    assert r.samples_written == 33
    errors = [e for st, e in _queries(tmp_path) if st == "error"]
    assert len(errors) == 1
    assert "pyarrow reference" in errors[0]


def test_a_samples_add_run_checks_the_answers_again(fake, tmp_path, monkeypatch):
    _run(tmp_path)
    monkeypatch.setenv("CHBENCH_FAKE", "wrong=2")
    r = _run(tmp_path, samples_add=1)
    assert r.samples_written == 11  # the mismatching query takes none
    assert sum(st == "error" for st, _ in _queries(tmp_path)) == 1


def test_a_query_added_later_is_warmed_up_and_checked(fake, tmp_path, monkeypatch):
    _run(tmp_path, ONE_QUERY)
    assert not [1 for st, _ in _queries(tmp_path) if st != "ok"]
    # The second run's pending queries are the 6 new ones, in job order: 4th execute is the 2nd
    # job's first answer (each job runs a cold run, then the compared one).
    monkeypatch.setenv("CHBENCH_FAKE", "wrong=4")
    r = _run(tmp_path)
    assert r.queries_skipped_existing == 6
    assert r.queries_run == 6
    assert r.samples_written == 5 * 3
    statuses = _queries(tmp_path)
    assert sum(st == "error" for st, _ in statuses) == 1
    assert len(statuses) == 12


def test_a_planted_skip_is_recorded_with_its_reason(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("CHBENCH_FAKE", "skip=Float32")
    r = _run(tmp_path)
    assert (r.tables_ok, r.tables_skipped) == (3, 5)
    with Store(tmp_path / "r.db") as s:
        reasons = [
            t["skip_reason"]
            for t in s.table_rows("t_inproc")
            if t["status"] == "skipped"
        ]
    assert sum("planted skip" in x for x in reasons) == 3


def test_a_worker_that_dies_errors_its_pending_queries_and_the_run_finishes(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setenv("CHBENCH_FAKE", "die=3")
    r = _run(tmp_path)
    assert r.samples_written == 0
    statuses = _queries(tmp_path)
    assert {st for st, _ in statuses} == {"error"}
    assert any("died" in e for _, e in statuses)
    with Store(tmp_path / "r.db") as s:
        assert s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]


def test_a_mixed_suite_runs_both_engines_in_every_round(fake, tmp_path):
    r = _run(tmp_path, MIXED)
    assert r.tables_ok == 2
    assert r.samples_written == 6
    assert len(rounds(fake["client"].calls)) == 3  # one query_log read per timed round
    with Store(tmp_path / "r.db") as s:
        got = s.conn.execute(
            "SELECT tb.engine, s.round_no FROM samples s JOIN queries q ON q.query_id = s.query_id "
            "JOIN tables tb ON tb.table_id = q.table_id ORDER BY s.round_no, tb.engine"
        ).fetchall()
    assert got == [(e, n) for n in range(3) for e in ("clickhouse", "polars")]


def test_a_failure_escaping_the_rounds_marks_the_open_queries_error(
    fake, tmp_path, monkeypatch
):
    # The store breaks on the third sample (a second run sharing results.db, say): every query
    # still short of its samples, ClickHouse and Polars alike, ends `error` with the exception.
    real, n = Store.insert_sample, {"n": 0}

    def insert_sample(self, *a):
        n["n"] += 1
        if n["n"] == 3:
            raise sqlite3.OperationalError("database is locked")
        return real(self, *a)

    monkeypatch.setattr(Store, "insert_sample", insert_sample)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _run(tmp_path, MIXED)
    statuses = _queries(tmp_path)
    assert len(statuses) == 2
    assert set(statuses) == {("error", "OperationalError: database is locked")}
    with Store(tmp_path / "r.db") as s:
        assert s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]


def test_the_banner_names_each_engine_and_its_threads(fake, tmp_path):
    console = Console(file=io.StringIO(), width=400)
    _run(tmp_path, dry_run=True, console=console)
    out = console.file.getvalue()
    assert f"polars {versions.installed()['polars']}, threads [2]" in out


def test_a_dry_run_starts_no_worker(fake, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a dry run must not start a worker")

    monkeypatch.setattr(inproc.InprocEngine, "__init__", refuse)
    r = _run(tmp_path, dry_run=True)
    assert (r.tables_ok, r.tables_skipped) == (6, 2)
    assert not (tmp_path / "r.db").exists()


def test_a_polars_version_off_its_pin_warns(fake, tmp_path, monkeypatch):
    pin = tmp_path / "versions.yaml"
    pin.write_text('polars: "0.0.1"\n', encoding="utf-8")
    monkeypatch.setattr(versions, "PINNED_PATH", pin)
    console = Console(file=io.StringIO(), width=400)
    _run(tmp_path, dry_run=True, console=console)
    assert "warning versions: polars" in console.file.getvalue()
    assert "differs from the pinned 0.0.1" in console.file.getvalue()


def test_drift_is_checked_for_the_engines_the_suite_uses_and_no_clickhouse(
    fake, tmp_path, monkeypatch
):
    seen = []
    real = versions.drift
    monkeypatch.setattr(
        versions,
        "drift",
        lambda pinned, actual: seen.append(dict(actual)) or real(pinned, actual),
    )
    _run(tmp_path, dry_run=True)
    installed = versions.installed()
    assert seen == [{"polars": installed["polars"], "pyarrow": installed["pyarrow"]}]


def test_a_block_without_threads_takes_the_docker_vm_count(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(quiet, "docker_cpus", lambda: 3)
    console = Console(file=io.StringIO(), width=400)
    _run(tmp_path, SUITE.replace("  threads: [2]\n", ""), dry_run=True, console=console)
    assert "threads=3" in console.file.getvalue()


def test_the_default_thread_count_is_resolved_once_per_run(fake, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrate, "default_threads", lambda: calls.append(1) or 3)
    _run(tmp_path, SUITE.replace("  threads: [2]\n", ""), dry_run=True)
    assert calls == [1]


def test_each_query_keeps_its_k_across_rounds_and_a_capped_k_is_recorded(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(inproc, "TARGET_BATCH_NS", 10**15)
    monkeypatch.setattr(inproc, "MAX_REPEATS", 4)
    console = Console(file=io.StringIO(), width=400)
    _run(tmp_path, console=console)
    with Store(tmp_path / "r.db") as s:
        got = s.conn.execute(
            "SELECT query_id, count(DISTINCT round_no), group_concat(DISTINCT repeats), "
            "group_concat(DISTINCT repeats_capped) FROM samples GROUP BY query_id"
        ).fetchall()
    assert len(got) == 12
    assert all(n == 3 and k == "4" and c == "1" for _, n, k, c in got)
    assert "warning warm-up" in console.file.getvalue()
    assert "(the cap)" in console.file.getvalue()


def test_ipc_batch_rows_cells_run_and_are_stored(fake, tmp_path):
    text = SUITE.replace(
        "  threads: [2]\n", "  threads: [2]\n  ipc_batch_rows: [whole, 100]\n"
    )
    r = _run(tmp_path, text)
    # memory (no batch axis): 2 types x none ok, 2 x ZSTD(3) skipped; ipc: 2 types x 2 comp x 2 batches ok
    assert (r.tables_ok, r.tables_skipped) == (10, 2)
    with Store(tmp_path / "r.db") as s:
        rows = {
            t["ipc_batch_rows"] for t in s.table_rows("t_inproc") if t["status"] == "ok"
        }
    assert rows == {None, "whole", "100"}  # memory tables store NULL


def test_ctrl_c_in_a_timed_round_marks_the_queries_it_never_ran_interrupted(
    fake, tmp_path, monkeypatch
):
    real = inproc.InprocRound.run
    timed = {"n": 0}

    def run(self, job):
        if self.timed:
            timed["n"] += 1
            if timed["n"] == 5:
                raise KeyboardInterrupt
        return real(self, job)

    monkeypatch.setattr(inproc.InprocRound, "run", run)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    with Store(tmp_path / "r.db") as s:
        n = s.conn.execute("SELECT count(*) FROM samples").fetchone()[0]
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    # four samples landed; every query is short of its three, so none is left unmarked
    assert n == 4
    assert finished
    assert {st for st, _ in _queries(tmp_path)} == {"interrupted"}


def test_ctrl_c_in_the_warm_up_marks_every_in_process_query_interrupted(
    fake, tmp_path, monkeypatch
):
    real = inproc.InprocRound.run
    warm = {"n": 0}

    def run(self, job):
        if not self.timed:
            warm["n"] += 1
            if warm["n"] == 3:
                raise KeyboardInterrupt
        return real(self, job)

    monkeypatch.setattr(inproc.InprocRound, "run", run)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    statuses = _queries(tmp_path)
    assert len(statuses) == 12
    assert {st for st, _ in statuses} == {"interrupted"}


def test_a_failure_while_preparing_still_closes_the_workers(
    fake, tmp_path, monkeypatch
):
    def boom(self, *a):
        raise RuntimeError("prepare blew up")

    monkeypatch.setattr(inproc.InprocEngine, "prepare", boom)
    with pytest.raises(RuntimeError, match="prepare blew up"):
        _run(tmp_path)
    with Store(tmp_path / "r.db") as s:
        assert s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]


TWO_WORKERS = ONE_QUERY.replace("threads: [2]", "threads: [1, 2]")


def _failing_close(monkeypatch):
    """Every engine's first close() runs for real, then raises (the fixture's own later close
    must not); returns the list of engines it saw."""
    real, seen = inproc.InprocEngine.close, []

    def close(self):
        real(self)
        if self not in seen:
            seen.append(self)
            raise OSError("broken pipe")

    monkeypatch.setattr(inproc.InprocEngine, "close", close)
    return seen


def test_one_close_failing_does_not_skip_the_others_or_mask_the_error(
    fake, tmp_path, monkeypatch
):
    def boom(self, *a):
        raise RuntimeError("prepare blew up")

    monkeypatch.setattr(inproc.InprocEngine, "prepare", boom)
    seen = _failing_close(monkeypatch)
    with pytest.raises(RuntimeError, match="prepare blew up") as info:
        _run(tmp_path, TWO_WORKERS)
    assert (
        len(seen) == 2
    )  # the second worker was closed although the first close raised
    assert any("broken pipe" in n for n in info.value.__notes__)


def test_a_close_failing_after_a_good_run_is_a_printed_warning(
    fake, tmp_path, monkeypatch
):
    seen = _failing_close(monkeypatch)
    out = io.StringIO()
    summary = _run(tmp_path, TWO_WORKERS, console=Console(file=out, width=400))
    assert len(seen) == 2
    assert summary.samples_written > 0
    assert out.getvalue().count("broken pipe") == 2
