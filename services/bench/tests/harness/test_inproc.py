"""engines.inproc: the worker process behind InprocEngine (spawned for real, running the
FakeWorker): start, version and threads, prepare, rounds, the equal-results check, worker errors
and death, close."""

import multiprocessing
import os

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
    ipc_batch_rows="whole",
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

    def make(threads=2, fake=""):
        monkeypatch.setenv("CHBENCH_FAKE", fake)
        e = inproc.InprocEngine("fake", threads)
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
    v = TableVariant(**{**vars(VARIANT), "storage": "ipc", "compression": "ZSTD(3)"})
    path = tmp_path / "ipc" / "t1.arrow"
    r = e.prepare(v, "t1", path, None)
    assert r.status == "ok" and r.ipc_bytes == path.stat().st_size


def test_the_warm_up_checks_the_first_answer(engine, dataset, tmp_path):
    e = engine(fake="wrong=1")
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


def test_a_dead_worker_fails_every_later_call(engine, dataset, tmp_path):
    e = engine(fake="die=1")
    e.load(dataset, "value")
    e.prepare(VARIANT, "t1", tmp_path / "x.arrow", None)
    rd = e.start_round(0)
    rd.run(job("q1"))
    rd.run(job("q2"))
    errors = rd.finish().errors
    assert set(errors) == {"q1", "q2"} and all("died" in t for t in errors.values())
    assert e.prepare(VARIANT, "t2", tmp_path / "y.arrow", None).status == "error"


def test_close_ends_the_process(engine):
    e = engine()
    e.close()
    assert not e._proc.is_alive()


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
