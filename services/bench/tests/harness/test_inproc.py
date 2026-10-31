"""engines.inproc: the worker process behind InprocEngine (spawned for real, running the
FakeWorker): start, version and threads, prepare, rounds, the equal-results check, worker errors
and death, close."""

import multiprocessing
import os
import subprocess
import types

import psutil

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from chbench.harness import quiet
from chbench.harness.engines import inproc
from chbench.harness.queries import QueryVariant, reference_result
from chbench.harness.runner import Job
from chbench.harness.validate import TableVariant

ARRAY = pa.array([1.0, 2.0, 2.0, None])
SCAN = QueryVariant("full_scan", None, None, None, None, None)
VARIANT = TableVariant(
    ch_type="Float64",
    nullable=None,
    primary_key=None,
    projection=None,
    codec=None,
    skip_index=None,
    compression="none",
    status="ok",
    skip_reason=None,
    warnings=(),
    engine="fake",
    storage="memory",
    threads=2,
    type_spec="Float64",
    ipc_batch_rows=None,
)
PAYLOAD = {"predicate_kind": None, "values": None, "sql": None}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "d.arrow"
    table = pa.table({"value": ARRAY})
    with ipc.new_file(str(path), table.schema) as w:
        w.write_table(table)
    return path


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setitem(
        inproc.FACTORIES, "fake", "tests.harness.fake_worker:FakeWorker"
    )
    made = []

    def make(threads=2, fake="", **kw):
        monkeypatch.setenv("CHBENCH_FAKE", fake)
        e = inproc.InprocEngine("fake", threads, **kw)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close()


def job(key="q1", variant=VARIANT):
    return Job(
        key,
        "t1",
        variant,
        SCAN,
        payload=PAYLOAD,
        reference=reference_result(ARRAY, None, None),
    )


def test_a_worker_reports_its_engine_and_the_threads_its_environment_gave_it(engine):
    before = os.environ.get("POLARS_MAX_THREADS")
    e = engine(threads=3)
    assert e.info == {"engine": "fake", "version": "0.0.0", "threads": 3}
    assert e.version() == "0.0.0" and e.pid != os.getpid()
    assert os.environ.get("POLARS_MAX_THREADS") == before


def test_prepare_then_a_timed_round_delivers_a_sample(engine, dataset, tmp_path):
    e = engine()
    assert e.load(dataset, "value") == 4
    r = e.prepare(VARIANT, "t1", tmp_path / "ipc" / "t1.arrow", None)
    assert (r.status, r.rows, r.ipc_bytes) == ("ok", 4, None) and r.memory_bytes > 0
    seen = []
    rd = e.start_round(5, sample_no=2, on_sample=lambda k, s: seen.append((k, s)))
    rd.run(job())
    result = rd.finish()
    [(key, s)] = seen
    assert key == "q1" and result.samples == {"q1": s} and not result.errors
    assert (s.round_no, s.sample_no) == (5, 2)
    assert s.server_ms > 0 and s.cpu_us >= 0 and s.memory_bytes >= 0 and s.client_ms > 0


def test_ipc_prepare_writes_the_file(engine, dataset, tmp_path):
    e = engine()
    e.load(dataset, "value")
    v = TableVariant(
        **{
            **vars(VARIANT),
            "storage": "ipc",
            "compression": "ZSTD(3)",
            "ipc_batch_rows": "122880",
        }
    )
    path = tmp_path / "ipc" / "t1.arrow"
    r = e.prepare(v, "t1", path, None)
    assert r.status == "ok" and r.ipc_bytes == path.stat().st_size


def test_the_warm_up_checks_the_first_answer(engine, dataset, tmp_path):
    e = engine(fake="wrong=2")  # run 1 is the discarded cold run
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(None, timed=False)
    rd.run(job("q1"))
    rd.run(job("q2"))
    errors = rd.finish().errors
    assert list(errors) == ["q1"]
    assert "answered count 5" in errors["q1"] and "reference is count 4" in errors["q1"]


def test_a_skip_comes_back_as_skipped(engine, dataset, tmp_path):
    e = engine(fake="skip=Float64")
    e.load(dataset, "value")
    r = e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    assert r.status == "skipped" and "planted skip" in r.error


def test_a_worker_error_is_that_jobs_error_and_the_round_carries_on(
    engine, dataset, tmp_path
):
    e = engine(fake="fail=1")
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(0)
    rd.run(job("q1"))
    rd.run(job("q2"))
    result = rd.finish()
    assert result.errors["q1"].startswith("WorkerError: RuntimeError: planted failure")
    assert list(result.samples) == ["q2"]
    # no warm-up ran: both jobs were sampled uncalibrated, the failed one too
    assert [w.split(":")[0] for w in result.warnings] == ["q1", "q2"]
    assert all("not calibrated" in w for w in result.warnings)


def test_a_dead_worker_fails_every_later_call(engine, dataset, tmp_path):
    e = engine(fake="die=1")
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(0)
    rd.run(job("q1"))
    rd.run(job("q2"))
    result = rd.finish()
    errors = result.errors
    assert set(errors) == {"q1", "q2"} and all("died" in t for t in errors.values())
    assert [w.split(":")[0] for w in result.warnings] == ["q1", "q2"]
    assert all("not calibrated" in w for w in result.warnings)
    assert e.prepare(VARIANT, "t2", tmp_path / "y.arrow", None).status == "error"


def test_a_failure_in_the_warm_ups_cold_run_is_that_jobs_error_and_leaves_it_uncalibrated(
    engine, dataset, tmp_path
):
    e = engine(fake="fail=1")  # run 1 is calibration's discarded cold run
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(None, timed=False)
    rd.run(job("q1"))
    rd.run(job("q2"))
    result = rd.finish()
    assert list(result.errors) == ["q1"]
    assert result.errors["q1"].startswith("WorkerError: RuntimeError: planted failure")
    assert "q1" not in e.repeats and "q2" in e.repeats


def test_a_death_in_the_warm_ups_cold_run_fails_the_job_and_kills_the_engine(
    engine, dataset, tmp_path
):
    e = engine(fake="die=1")
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(None, timed=False)
    rd.run(job("q1"))
    rd.run(job("q2"))
    errors = rd.finish().errors
    assert set(errors) == {"q1", "q2"}
    assert all(t.startswith("WorkerDied: ") and "died" in t for t in errors.values())
    assert e.repeats == {}
    e._proc.join(5)  # the pipe can break a moment before the process is reaped
    assert e._proc.exitcode == 3
    with pytest.raises(inproc.WorkerDied, match="died"):
        e.load(dataset, "value")


def test_close_ends_the_process(engine):
    e = engine()
    e.close()
    assert not e._proc.is_alive()


def test_a_stuck_worker_times_out_is_terminated_and_fails_every_later_call(
    engine, dataset, tmp_path
):
    e = engine(fake="hang=run", timeouts={"run": 0.5})
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(0)
    rd.run(job("q1"))
    rd.run(job("q2"))
    errors = rd.finish().errors
    assert set(errors) == {"q1", "q2"}
    assert all("timed out after 0.5s in run" in t for t in errors.values())
    assert not e._proc.is_alive()
    with pytest.raises(inproc.WorkerDied, match="timed out"):
        e.load(dataset, "value")


def test_a_worker_stuck_at_start_times_out_and_leaves_no_process(monkeypatch):
    monkeypatch.setitem(
        inproc.FACTORIES, "fake", "tests.harness.fake_worker:FakeWorker"
    )
    monkeypatch.setenv("CHBENCH_FAKE", "hang=version")
    with pytest.raises(inproc.WorkerDied, match="timed out after 1s in version"):
        inproc.InprocEngine("fake", 1, timeouts={"version": 1})
    assert multiprocessing.active_children() == []


def test_the_default_timeouts_cover_every_op():
    assert set(inproc.TIMEOUTS_S) == {"version", "load", "prepare", "run", "calibrate"}
    assert all(t >= 60 for t in inproc.TIMEOUTS_S.values())


@pytest.fixture
def pipes(monkeypatch):
    """Every Pipe end InprocEngine opens."""
    opened = []
    real = inproc._CONTEXT.Pipe

    def pipe(*a, **k):
        ends = real(*a, **k)
        opened.extend(ends)
        return ends

    monkeypatch.setattr(inproc._CONTEXT, "Pipe", pipe)
    monkeypatch.setitem(
        inproc.FACTORIES, "fake", "tests.harness.fake_worker:FakeWorker"
    )
    monkeypatch.setenv("CHBENCH_FAKE", "")
    return opened


def test_a_start_that_raises_closes_both_pipe_ends(monkeypatch, pipes):
    class Unstartable(inproc._CONTEXT.Process):
        def start(self):
            raise OSError("planted start failure")

    monkeypatch.setattr(inproc._CONTEXT, "Process", Unstartable)
    with pytest.raises(OSError, match="planted start failure"):
        inproc.InprocEngine("fake", 1)
    assert len(pipes) == 2 and all(c.closed for c in pipes)
    assert multiprocessing.active_children() == []


def test_a_child_gone_before_psutil_sees_it_is_worker_died(monkeypatch, pipes):
    def gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(inproc.psutil, "Process", gone)
    # the engine is dead before it sent `close`, so close() waits out a join on a live child
    monkeypatch.setattr(inproc, "_CLOSE_WAIT_S", 0.2)
    with pytest.raises(inproc.WorkerDied, match="died at start"):
        inproc.InprocEngine("fake", 1)
    assert len(pipes) == 2 and all(c.closed for c in pipes)
    assert multiprocessing.active_children() == []


def test_a_worker_that_exits_at_once_is_worker_died(monkeypatch, pipes):
    monkeypatch.setenv("CHBENCH_FAKE", "exit=start")
    with pytest.raises(inproc.WorkerDied, match="died"):
        inproc.InprocEngine("fake", 1)
    assert len(pipes) == 2 and all(c.closed for c in pipes)
    assert multiprocessing.active_children() == []


class ScriptedRss:
    """A psutil.Process stand-in: `memory_info()` follows `script` (an int is the RSS, None
    raises AccessDenied), its last entry repeating."""

    def __init__(self, *script):
        self.script = list(script)

    def memory_info(self):
        rss = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if rss is None:
            raise psutil.AccessDenied()
        return types.SimpleNamespace(rss=rss)


@pytest.mark.parametrize(
    "script", [(None, 500), (100, None)], ids=["baseline", "every-peak"]
)
def test_memory_is_unknown_when_rss_reads_fail(engine, dataset, tmp_path, script):
    e = engine()
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    e._ps = ScriptedRss(*script)
    rd = e.start_round(0)
    rd.run(job())
    assert rd.finish().samples["q1"].memory_bytes is None


def test_memory_is_the_peak_above_the_baseline(engine, dataset, tmp_path):
    e = engine()
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    e._ps = ScriptedRss(100, 700, 300)
    rd = e.start_round(0)
    rd.run(job())
    assert rd.finish().samples["q1"].memory_bytes == 600


def test_the_docker_cpu_lookup_runs_once_per_process(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "6 1000", "")

    monkeypatch.setattr(quiet.subprocess, "run", run)
    quiet.docker_cpus.cache_clear()
    try:
        assert quiet.docker_cpus() == 6 and quiet.docker_cpus() == 6
        assert inproc.default_threads() == 6 and len(calls) == 1
    finally:
        quiet.docker_cpus.cache_clear()


def test_default_threads_follow_the_docker_vm(monkeypatch):
    monkeypatch.setattr(quiet, "docker_cpus", lambda: 6)
    assert inproc.default_threads() == 6
    monkeypatch.setattr(quiet, "docker_cpus", lambda: None)
    assert inproc.default_threads() == (os.cpu_count() or 1)


def test_ipc_path(tmp_path):
    assert inproc.ipc_path(tmp_path, "abc") == tmp_path / "ipc" / "abc.arrow"


def test_interrupt_after_finish_delivers_nothing_twice_and_talks_to_nobody(
    engine, dataset, tmp_path
):
    e = engine()
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    seen = []
    rd = e.start_round(0, on_sample=lambda k, s: seen.append(k))
    rd.run(job())
    result = rd.finish()
    calls = []
    real = e.call
    e.call = lambda *a, **k: calls.append(a) or real(*a, **k)
    rd.interrupt()
    rd.interrupt()
    assert seen == ["q1"] and rd.finish() is result and not calls


def test_a_worker_that_cannot_start_leaves_no_process(monkeypatch):
    monkeypatch.setitem(inproc.FACTORIES, "fake", "no.such.module:Nothing")
    with pytest.raises(inproc.WorkerDied):
        inproc.InprocEngine("fake", 1)
    assert multiprocessing.active_children() == []


class StubEngine:
    """An engine whose worker answers one fixed reply: the parent-side arithmetic alone."""

    name = "stub"

    def __init__(self, reply):
        self.reply, self.repeats, self.calls = reply, {}, []

    def call(self, op, *args, sample_memory=False):
        self.calls.append((op, *args))
        return dict(self.reply), (100 if sample_memory else None)


def test_a_timed_sample_is_the_batch_divided_by_k():
    e = StubEngine(
        {"count": 4, "total": 5.0, "wall_ns": 8_000_000, "cpu_ns": 4_000_000}
    )
    e.repeats["q1"] = (4, False)
    rd = inproc.InprocRound(e, 0)
    rd.run(job())
    s = rd.finish().samples["q1"]
    assert (s.server_ms, s.cpu_us, s.repeats, s.repeats_capped) == (
        2.0,
        1000.0,
        4,
        False,
    )
    assert s.memory_bytes == 100 and e.calls == [("run", "t1", PAYLOAD, 4)]


def test_the_warm_up_calibrates_and_every_timed_round_reuses_k(
    engine, dataset, tmp_path, monkeypatch
):
    monkeypatch.setattr(inproc, "TARGET_BATCH_NS", 10**15)
    monkeypatch.setattr(inproc, "MAX_REPEATS", 8)
    e = engine()
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    warm = e.start_round(None, timed=False)
    warm.run(job())
    assert e.repeats["q1"] == (8, True)
    assert any("8 repeats (the cap)" in w for w in warm.finish().warnings)
    for r in (0, 1):
        rd = e.start_round(r)
        rd.run(job())
        s = rd.finish().samples["q1"]
        assert (s.round_no, s.repeats, s.repeats_capped) == (r, 8, True)


def test_a_job_never_calibrated_runs_at_k_1_and_warns():
    e = StubEngine(
        {"count": 4, "total": 5.0, "wall_ns": 8_000_000, "cpu_ns": 4_000_000}
    )
    rd = inproc.InprocRound(e, 0)
    rd.run(job())
    r = rd.finish()
    assert r.samples["q1"].repeats == 1 and e.calls == [("run", "t1", PAYLOAD, 1)]
    assert (
        len(r.warnings) == 1
        and "q1" in r.warnings[0]
        and "not calibrated" in r.warnings[0]
    )
