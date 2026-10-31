"""engines.worker: the round-trip check, the clock and BaseWorker's prepare / run, in process
(the FakeWorker; real processes are test_inproc.py's)."""

import gc
import multiprocessing
import threading

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from chbench.harness.engines import worker as worker_module
from chbench.harness.engines.worker import calibrate_repeats, mismatch, timed
from tests.harness.fake_worker import FakeWorker

ARRAY = pa.array([1.0, 2.0, 2.0, None])
PAYLOAD = {"predicate_kind": None, "values": None, "sql": None}


def spec(
    tmp_path,
    *,
    storage="memory",
    compression="none",
    expect=None,
    tid="t1",
    type_spec="Float64",
    batch_rows="whole",
):
    return {
        "table_id": tid,
        "type_spec": type_spec,
        "storage": storage,
        "compression": compression,
        "ipc_batch_rows": batch_rows,
        "ipc_path": str(tmp_path / "ipc" / f"{tid}.arrow"),
        "expect_ipc_bytes": expect,
    }


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.delenv("CHBENCH_FAKE", raising=False)
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    path = tmp_path / "d.arrow"
    table = pa.table({"value": ARRAY})
    with ipc.new_file(str(path), table.schema) as w:
        w.write_table(table)
    w = FakeWorker(2)
    assert w.load(str(path), "value") == 4
    return w


def test_load_reads_the_column(worker):
    assert worker.array.equals(ARRAY)
    assert worker.version() == {"engine": "fake", "version": "0.0.0", "threads": 2}


def test_prepare_memory_holds_the_table(worker, tmp_path):
    r = worker.prepare(spec(tmp_path))
    assert r["status"] == "ok" and r["reason"] is None and r["rows"] == 4
    assert r["ipc_bytes"] is None and r["memory_bytes"] > 0 and r["prepare_ms"] >= 0
    assert "t1" in worker.tables


def test_prepare_ipc_writes_once_and_reuses_the_file(worker, tmp_path):
    first = worker.prepare(spec(tmp_path, storage="ipc", compression="ZSTD(3)"))
    path = tmp_path / "ipc" / "t1.arrow"
    assert first["status"] == "ok" and first["ipc_bytes"] == path.stat().st_size
    stamp = path.stat().st_mtime_ns
    again = worker.prepare(
        spec(tmp_path, storage="ipc", compression="ZSTD(3)", expect=first["ipc_bytes"])
    )
    assert again["ipc_bytes"] == first["ipc_bytes"] and path.stat().st_mtime_ns == stamp
    assert not list((tmp_path / "ipc").glob("*.tmp"))


def test_the_ipc_file_has_record_batches_of_the_cells_size(worker, tmp_path):
    worker.prepare(spec(tmp_path, storage="ipc", tid="w"))
    worker.prepare(spec(tmp_path, storage="ipc", tid="b3", batch_rows="3"))
    with pa.OSFile(str(tmp_path / "ipc" / "w.arrow"), "rb") as f:
        assert ipc.open_file(f).num_record_batches == 1
    with pa.OSFile(str(tmp_path / "ipc" / "b3.arrow"), "rb") as f:
        assert ipc.open_file(f).num_record_batches == 2  # 4 rows in batches of 3


def test_a_planted_cast_failure_is_a_skip(tmp_path, monkeypatch, worker):
    monkeypatch.setenv("CHBENCH_FAKE", "skip=Float64")
    w = FakeWorker(1)
    w.array = ARRAY
    r = w.prepare(spec(tmp_path))
    assert r["status"] == "skipped" and "cast to Float64 failed" in r["reason"]
    assert "planted skip" in r["reason"]


def test_run_answers_with_its_clocks(worker, tmp_path):
    worker.prepare(spec(tmp_path))
    out = worker.run("t1", {"predicate_kind": None, "values": None, "sql": None})
    assert (out["count"], out["total"]) == (4, 5.0)
    assert out["wall_ns"] > 0 and out["cpu_ns"] >= 0


def test_mismatch_names_the_first_differing_row():
    assert mismatch(pa.array([1, 2, 3]), pa.array([1, 2, 3], pa.int32())) is None
    assert mismatch(pa.array([1, 2, 3]), pa.array([1, 5, 3])) == "row 1: 2 became 5"
    assert mismatch(pa.array([1, None]), pa.array([1, 2])) == "the null rows differ"
    assert (
        mismatch(pa.array([float("nan"), 1.0]), pa.array([float("nan"), 1.0])) is None
    )
    assert "cannot be read back as int8" in mismatch(
        pa.array([1], pa.int8()), pa.array([300])
    )
    assert mismatch(pa.array([1, 2]), pa.array([1])) == "1 row(s), the dataset has 2"


def test_timed_pauses_the_collector():
    seen = []
    out, wall, cpu = timed(lambda: seen.append(gc.isenabled()) or 7)
    assert out == 7 and seen == [False] and gc.isenabled() and wall > 0 and cpu >= 0


def test_mismatch_accepts_a_chunked_array():
    chunked = pa.chunked_array([[1, 2], [3]])
    assert mismatch(pa.array([1, 2, 3]), chunked) is None
    assert mismatch(pa.array([1, 2, 4]), chunked) == "row 2: 4 became 3"
    empty = pa.chunked_array([pa.array([], pa.int64())])
    assert mismatch(pa.array([1, 2, 3]), empty) == "0 row(s), the dataset has 3"


def test_a_cast_that_changes_values_is_a_skip(worker, tmp_path, monkeypatch):
    monkeypatch.setattr(
        FakeWorker, "values", lambda self, held: pa.array([1.0, 9.0, 2.0, None])
    )
    r = worker.prepare(spec(tmp_path))
    assert (
        r["status"] == "skipped" and "does not hold the dataset's values" in r["reason"]
    )
    assert "row 1: 2.0 became 9.0" in r["reason"] and "t1" not in worker.tables


def test_an_ipc_file_of_the_wrong_size_is_rewritten(worker, tmp_path):
    path = tmp_path / "ipc" / "t1.arrow"
    path.parent.mkdir()
    path.write_bytes(b"not an arrow file")
    r = worker.prepare(spec(tmp_path, storage="ipc", expect=999_999))
    assert r["status"] == "ok" and r["ipc_bytes"] == path.stat().st_size != 17
    assert mismatch(ARRAY, worker.read_ipc(path)) is None
    assert not list(path.parent.glob("*.tmp"))


def test_timed_enables_the_collector_again_after_the_function_raises():
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        timed(boom)
    assert gc.isenabled()


@pytest.fixture
def loop(monkeypatch):
    """worker.main over a real Pipe, on a thread (SIGINT handling is the main thread's)."""
    monkeypatch.setattr(worker_module.signal, "signal", lambda *a: None)
    monkeypatch.delenv("CHBENCH_FAKE", raising=False)
    parent, child = multiprocessing.Pipe()
    t = threading.Thread(
        target=worker_module.main,
        args=(child, "tests.harness.fake_worker:FakeWorker", 1),
        daemon=True,
    )
    t.start()
    yield parent, t
    parent.close()
    t.join(10)
    assert not t.is_alive()


def ask(parent, *msg):
    parent.send(msg)
    assert parent.poll(10)
    return parent.recv()


def test_the_loop_answers_ok_and_reports_a_raising_op_and_carries_on(loop):
    parent, t = loop
    assert ask(parent, "version") == (
        "ok",
        {"engine": "fake", "version": "0.0.0", "threads": 1},
    )
    status, text = ask(parent, "load", "/no/such/file.arrow", "value")
    assert status == "error" and ": " in text
    assert ask(parent, "version")[0] == "ok" and t.is_alive()


def test_an_unknown_op_is_an_error_and_the_loop_carries_on(loop):
    parent, t = loop
    status, text = ask(parent, "no_such_op")
    assert status == "error" and text.startswith("AttributeError")
    assert ask(parent, "version")[0] == "ok"


def test_close_ends_the_loop(loop):
    parent, t = loop
    parent.send(("close",))
    t.join(10)
    assert not t.is_alive()


def test_eof_ends_the_loop_cleanly(loop):
    parent, t = loop
    parent.close()
    t.join(10)
    assert not t.is_alive()


def test_calibration_picks_the_smallest_power_of_two_over_the_target():
    walls = {1: 30, 2: 60, 4: 120, 8: 240, 16: 480}
    asked = []

    def wall(k):
        asked.append(k)
        return walls[k]

    assert calibrate_repeats(wall, 250, 1024) == (16, False) and asked == [
        1,
        2,
        4,
        8,
        16,
    ]
    assert calibrate_repeats(lambda k: 10 * k, 250, 8) == (8, True)
    assert calibrate_repeats(lambda k: 999, 250, 1024) == (1, False)


def test_run_repeats_the_query_and_answers_the_first(worker, tmp_path):
    worker.prepare(spec(tmp_path))
    out = worker.run("t1", PAYLOAD, 5)
    assert worker.runs == 5 and (out["count"], out["total"], out["repeats"]) == (
        4,
        5.0,
        5,
    )


def test_calibrate_answers_the_first_run_and_its_batch_size(worker, tmp_path):
    worker.prepare(spec(tmp_path))
    out = worker.calibrate("t1", PAYLOAD, 0, 1024)
    assert out == {"count": 4, "total": 5.0, "repeats": 1, "capped": False}
    assert worker.runs == 1
    capped = worker.calibrate("t1", PAYLOAD, 10**15, 4)
    assert (capped["repeats"], capped["capped"]) == (4, True)
    assert worker.runs == 1 + 1 + 2 + 4  # batches of 1, 2 and 4
