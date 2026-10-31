"""engines.worker: the round-trip check, the clock and BaseWorker's prepare / run, in process
(the FakeWorker; real processes are test_inproc.py's)."""

import gc

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from chbench.harness.engines.worker import mismatch, timed
from tests.harness.fake_worker import FakeWorker

ARRAY = pa.array([1.0, 2.0, 2.0, None])


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
