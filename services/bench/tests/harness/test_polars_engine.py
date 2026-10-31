"""engines.polars_engine, in process: dtypes, predicates and totals against the pyarrow reference
on every Polars type in the vocabulary, memory and ipc, and the skip rules."""

import decimal
import itertools

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from chbench.harness.engines.polars_engine import PolarsWorker, polars_dtype
from chbench.harness.inproc_types import column_family, parse_polars_type
from chbench.harness.queries import (
    choice_values_arrow,
    choose_in,
    choose_range,
    query_values,
    reference_result,
    results_agree,
)

# a deprecated Polars call (is_in with a plain list) must fail the tests, not warn
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")
D = decimal.Decimal
_TABLES = itertools.count()
PL_CASES = [
    (pa.array([1, 2, 2, None, 3] * 20, pa.int64()), ["Int16", "UInt64", "Float64"]),
    (
        pa.array([0.5, 1.25, 1.25, None, -3.0] * 20),
        ["Float64", "Float32", "Decimal(5, 2)"],
    ),
    (
        pa.array(
            [D("0.50"), D("1.25"), D("1.25"), None, D("-3.00")] * 20, pa.decimal32(5, 2)
        ),
        ["Decimal(precision=5, scale=2)", "Decimal(18, 2)", "Float64"],
    ),
    (pa.array([True, False, True, None] * 20), ["Boolean"]),
    (
        pa.array(["a", "b'q", "b'q", None, "ccc"] * 20),
        [
            "String",
            "Categorical(UInt8)",
            "Categorical(UInt16)",
            "Categorical",
            'Categorical(Categories(name="v", namespace="", physical=pl.UInt8))',
            "Enum",
            "Enum(categories=['ccc', \"b'q\", 'a'])",
            "Binary",
        ],
    ),
    (pa.array([b"\x00\xff", b"ab", b"ab", None] * 20), ["Binary"]),
    (pa.array([b"\x00\xff", b"ab", b"ab", None] * 20, pa.binary(2)), ["Binary"]),
    (pa.array([0, 1, 1, None, 400] * 20, pa.date32()), ["Date", "Datetime(us)"]),
    (
        pa.array(
            [0, 1_500_000, 1_500_000, None, 9_000_000] * 20,
            pa.timestamp("us", tz="Europe/London"),
        ),
        ["Datetime(time_unit='us', time_zone='Europe/London')", "Datetime(ns, UTC)"],
    ),
    (
        pa.array([0, 1_500, 1_500, None, 90_000] * 20, pa.timestamp("ms")),
        ["Datetime(ms)", "Datetime(us)"],
    ),
    (pa.array([0, 1_500, 1_500, None, 90_000] * 20, pa.time32("ms")), ["Time"]),
    (
        pa.array([0, 1_500_000_001, 1_500_000_001, None, 3] * 20, pa.time64("ns")),
        ["Time"],
    ),
]
_IDS = [f"{a.type}->{t}" for a, ts in PL_CASES for t in ts]


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


def worker_over(array):
    w = PolarsWorker(2)
    w.array = array
    return w


def test_polars_dtypes():
    a = pa.array(["b", "a", None, "b"])
    assert polars_dtype(
        parse_polars_type("Categorical(UInt8)"), "ns1", a
    ) == pl.Categorical(pl.Categories("v", namespace="ns1", physical=pl.UInt8))
    assert polars_dtype(parse_polars_type("Enum"), "x", a) == pl.Enum(["b", "a"])
    assert polars_dtype(
        parse_polars_type("Enum(categories=['a', 'b'])"), "x", a
    ) == pl.Enum(["a", "b"])
    assert polars_dtype(parse_polars_type("Decimal(5, 2)"), "x", a) == pl.Decimal(5, 2)
    assert polars_dtype(parse_polars_type("Datetime(ns, UTC)"), "x", a) == pl.Datetime(
        "ns", "UTC"
    )
    assert polars_dtype(parse_polars_type("UInt16"), "x", a) == pl.UInt16


@pytest.mark.parametrize("storage", ["memory", "ipc"])
@pytest.mark.parametrize(
    "array, name", [(a, t) for a, ts in PL_CASES for t in ts], ids=_IDS
)
def test_answers_equal_the_pyarrow_reference(tmp_path, array, name, storage):
    w = worker_over(array)
    tid = f"t{next(_TABLES)}"
    compression = "ZSTD(3)" if storage == "ipc" else "none"
    r = w.prepare(spec(tid, name, tmp_path, storage, compression))
    assert r["status"] == "ok", r["reason"]
    values = query_values(array)
    family = column_family("polars", name)
    choices = [None, choose_in(values, 0.4, 1)]
    if family not in ("boolean", "binary"):
        choices.append(choose_range(values, 0.5, np.random.default_rng(1)))
    for c in choices:
        arrow_values = None if c is None else choice_values_arrow(c, array.type)
        kind = None if c is None else c.predicate_kind
        out = w.run(tid, {"predicate_kind": kind, "values": arrow_values, "sql": None})
        ref = reference_result(array, kind, arrow_values, family)
        assert results_agree(ref, out["count"], out["total"]), (kind, out, ref)


def test_a_categorical_too_narrow_for_the_column_is_skipped(tmp_path):
    w = worker_over(pa.array([f"v{i}" for i in range(300)]))
    r = w.prepare(spec("t1", "Categorical(UInt8)", tmp_path))
    assert (
        r["status"] == "skipped" and "cast to Categorical(UInt8) failed" in r["reason"]
    )


def test_a_lossy_cast_is_skipped(tmp_path):
    w = worker_over(pa.array([0.5, 1.0]))
    r = w.prepare(spec("t1", "Int64", tmp_path))
    assert r["status"] == "skipped" and "row 0: 0.5 became 0.0" in r["reason"]


def test_two_tables_do_not_share_categories(tmp_path):
    w = worker_over(pa.array([f"a{i}" for i in range(200)]))
    assert w.prepare(spec("t1", "Categorical(UInt8)", tmp_path))["status"] == "ok"
    w.array = pa.array([f"b{i}" for i in range(200)])
    assert w.prepare(spec("t2", "Categorical(UInt8)", tmp_path))["status"] == "ok"


def test_the_ipc_file_is_in_polars_layout(tmp_path):
    w = worker_over(pa.array(["a", "b", None] * 10))
    w.prepare(spec("t1", "Categorical(UInt16)", tmp_path, "ipc", "LZ4_FRAME"))
    w.prepare(spec("t2", "String", tmp_path, "ipc", "none"))
    with pa.OSFile(str(tmp_path / "t1.arrow"), "rb") as f:
        t1 = ipc.open_file(f).schema.field("v").type
    with pa.OSFile(str(tmp_path / "t2.arrow"), "rb") as f:
        t2 = ipc.open_file(f).schema.field("v").type
    assert t1 == pa.dictionary(pa.uint16(), pa.string_view())
    assert t2 == pa.string_view()


def test_ipc_batch_rows_sets_the_files_record_batches(tmp_path):
    array = pa.array(["a", "b", None] * 10)
    w = worker_over(array)
    assert (
        w.prepare(spec("t3", "Categorical(UInt8)", tmp_path, "ipc", "ZSTD(3)", "7"))[
            "status"
        ]
        == "ok"
    )
    with pa.OSFile(str(tmp_path / "t3.arrow"), "rb") as f:
        assert ipc.open_file(f).num_record_batches == 5  # 30 rows in batches of 7
    out = w.run("t3", {"predicate_kind": None, "values": None, "sql": None})
    assert out["count"] == 30


def test_version_names_polars_and_its_thread_pool():
    info = PolarsWorker(2).version()
    assert info["engine"] == "polars" and info["version"] == pl.__version__
    assert info["threads"] == pl.thread_pool_size()


def test_the_ipc_file_can_be_deleted_while_the_worker_lives(tmp_path):
    w = worker_over(pa.array(["a", "b", None] * 10))
    assert (
        w.prepare(spec("t1", "Categorical(UInt8)", tmp_path, "ipc", "ZSTD(3)"))[
            "status"
        ]
        == "ok"
    )
    w.run("t1", {"predicate_kind": None, "values": None, "sql": None})
    (
        tmp_path / "t1.arrow"
    ).unlink()  # PermissionError on Windows if a map were still live
