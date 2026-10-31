"""engines.datafusion_engine, in process: SQL answers against the pyarrow reference on every
DataFusion type in the vocabulary, memory and ipc, partitions, and the skip rules."""

import itertools

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from chbench.harness.engines.datafusion_engine import DataFusionWorker
from chbench.harness.inproc_types import arrow_family, parse_datafusion_type
from chbench.harness.queries import (
    choice_values_arrow,
    choose_in,
    choose_range,
    datafusion_sql,
    datafusion_table,
    query_values,
    reference_result,
    render_datafusion,
    results_agree,
)
from tests.harness.test_queries_engines import DF_CASES

_TABLES = itertools.count()
_IDS = [f"{a.type}->{t}" for a, ts in DF_CASES for t in ts]


def spec(
    tid, type_spec, tmp_path, storage="memory", compression="none", batch_rows="whole"
):
    return {
        "table_id": tid,
        "type_spec": type_spec,
        "storage": storage,
        "compression": compression,
        "ipc_batch_rows": batch_rows,
        "ipc_path": str(tmp_path / f"{tid}.arrow"),
        "expect_ipc_bytes": None,
    }


def worker_over(array, threads=2):
    w = DataFusionWorker(threads)
    w.array = array
    return w


@pytest.mark.parametrize("storage", ["memory", "ipc"])
@pytest.mark.parametrize(
    "array, name", [(a, t) for a, ts in DF_CASES for t in ts], ids=_IDS
)
def test_answers_equal_the_pyarrow_reference(tmp_path, array, name, storage):
    w = worker_over(array)
    tid = f"t{next(_TABLES)}"
    compression = "LZ4_FRAME" if storage == "ipc" else "none"
    r = w.prepare(spec(tid, name, tmp_path, storage, compression))
    assert r["status"] == "ok", r["reason"]
    column_type = parse_datafusion_type(name)
    family = arrow_family(column_type)
    values = query_values(array)
    choices = [None, choose_in(values, 0.4, 1)]
    if family not in ("boolean", "binary"):
        choices.append(choose_range(values, 0.5, np.random.default_rng(1)))
    for c in choices:
        arrow_values = None if c is None else choice_values_arrow(c, array.type)
        kind = None if c is None else c.predicate_kind
        predicate = (
            None if c is None else render_datafusion(kind, arrow_values, column_type)
        )
        sql = datafusion_sql(datafusion_table(tid), column_type, predicate)
        out = w.run(tid, {"predicate_kind": kind, "values": arrow_values, "sql": sql})
        ref = reference_result(array, kind, arrow_values, family)
        assert results_agree(ref, out["count"], out["total"]), (sql, out, ref)


def test_a_memory_table_has_one_partition_per_thread(tmp_path):
    w = worker_over(pa.array(np.arange(300_000, dtype=np.int64)), threads=3)
    w.prepare(spec("t1", "int64", tmp_path))
    plan = w.ctx.sql(
        f"EXPLAIN SELECT sum(v) FROM {datafusion_table('t1')}"
    ).to_arrow_table()
    assert "partitions=3" in "\n".join(plan.column("plan").to_pylist())


def test_dictionary_indices_too_narrow_are_skipped(tmp_path):
    w = worker_over(pa.array([f"v{i}" for i in range(300)]))
    r = w.prepare(
        spec("t1", "dictionary<values=string, indices=int8, ordered=0>", tmp_path)
    )
    assert r["status"] == "skipped" and "cast to dictionary" in r["reason"]


@pytest.mark.parametrize("width", ["int8", "int16", "uint16", "int32"])
def test_the_ipc_file_keeps_the_dictionary_width(tmp_path, width):
    w = worker_over(pa.array(["a", "b", None] * 10))
    name = f"dictionary<values=string, indices={width}, ordered=0>"
    r = w.prepare(spec("t1", name, tmp_path, "ipc", "ZSTD(1)"))
    assert r["status"] == "ok", r["reason"]
    with pa.OSFile(str(tmp_path / "t1.arrow"), "rb") as f:
        t = ipc.open_file(f).schema.field("v").type
    assert t == pa.dictionary(getattr(pa, width)(), pa.string())


def test_ipc_batch_rows_sets_the_files_record_batches(tmp_path):
    w = worker_over(pa.array(["a", "b", None] * 10))
    assert (
        w.prepare(spec("t4", "string", tmp_path, "ipc", "LZ4_FRAME", "10"))["status"]
        == "ok"
    )
    with pa.OSFile(str(tmp_path / "t4.arrow"), "rb") as f:
        assert ipc.open_file(f).num_record_batches == 3
    sql = datafusion_sql(datafusion_table("t4"), pa.string(), None)
    assert (
        w.run("t4", {"predicate_kind": None, "values": None, "sql": sql})["count"] == 30
    )


def test_version_names_datafusion():
    import datafusion

    info = DataFusionWorker(3).version()
    assert info == {
        "engine": "datafusion",
        "version": datafusion.__version__,
        "threads": 3,
    }


def test_the_ipc_file_can_be_deleted_while_the_worker_lives(tmp_path):
    w = worker_over(pa.array(["a", "b", None] * 10))
    assert w.prepare(spec("t1", "string", tmp_path, "ipc", "ZSTD(1)"))["status"] == "ok"
    sql = datafusion_sql(datafusion_table("t1"), pa.string(), None)
    q = {"predicate_kind": None, "values": None, "sql": sql}
    w.run("t1", q)
    (tmp_path / "t1.arrow").unlink()  # PermissionError on Windows if still open
