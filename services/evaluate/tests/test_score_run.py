import json

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from evaluate.predict import MARK_ROWS
from evaluate.rundir import RunDir
from evaluate.score import FILES, query_cost, score_run
from tests.fakes import Q, T, make_store


def test_query_cost_p50s_and_cpu():
    entry = {"dataset": "d", "column": "c", "suite": "s"}
    rows = [
        {
            "table_id": "t",
            "query_id": "q",
            "ch_type": "UInt32",
            "codec": "none",
            "compression": "LZ4",
            "skip_index": "none",
            "kind": "filter",
            "target_fraction": 0.01,
            "server_ms": ms,
            "client_ms": ms + 50,
            "memory_bytes": mem,
            "read_rows": 10,
            "read_bytes": 80,
            "profile_events_json": json.dumps(
                {"UserTimeMicroseconds": u, "SystemTimeMicroseconds": 1}
            ),
        }
        for ms, mem, u in ((1.0, 100, 9), (3.0, 300, 19), (2.0, 200, 29))
    ]
    (r,) = query_cost(entry, rows)
    assert (
        r["samples"] == 3 and r["server_ms_p50"] == 2.0 and r["memory_bytes_p50"] == 200
    )
    assert r["cpu_us_p50"] == 20 and r["client_ms_p50"] == 52.0


def _run_dir(tmp_path):
    run = RunDir.new(tmp_path)
    run.write_info(
        {"versions": {"git_sha": "abc"}, "datasets": {"d": {"status": "ok"}}}
    )
    ds = run.dataset("d")
    suites = run.suites("d")
    suites.mkdir()
    col = np.repeat(np.array([0, 1, 1, 1], dtype=np.int64), MARK_ROWS)
    pa_table = pa.table({"c": col})
    with pa.ipc.new_file(suites / "d__c.arrow", pa_table.schema) as w:
        w.write_table(pa_table)
    entries = [
        {
            "dataset": "d",
            "column": "c",
            "suite": "d__c",
            "role": "size",
            "type_role": "recommended",
            "base": "UInt32",
            "yaml": "d__c.yaml",
            "arrow": "d__c.arrow",
            "predicted": "d__c.predicted.json",
            "rec_codec": "none",
            "rec_compression": "LZ4",
        },
        {
            "dataset": "d",
            "column": "c",
            "suite": "d__c__skip",
            "role": "skip",
            "type_role": "recommended",
            "base": "UInt32",
            "yaml": "d__c__skip.yaml",
            "arrow": "d__c.arrow",
            "predicted": "d__c__skip.predicted.json",
            "rec_codec": "none",
            "rec_compression": "LZ4",
        },
    ]
    (suites / "suites.json").write_text(json.dumps(entries))
    (suites / "skipped.json").write_text("[]")
    (suites / "d__c.predicted.json").write_text(
        json.dumps(
            {
                "UInt32|none|LZ4": {
                    "bytes": 1100,
                    "origin": "candidate",
                    "outcome": "chosen",
                }
            }
        )
    )
    (suites / "d__c__skip.predicted.json").write_text(
        json.dumps(
            {
                "minmax:1": {"outcome": "chosen", "index_bytes": 40, "skip_rows": 0.75},
                "none": {"outcome": "none", "index_bytes": None, "skip_rows": 0.0},
            }
        )
    )
    stores = run.stores("d")
    stores.mkdir()
    make_store(
        stores / "d__c.db", "d__c", [T("t1", queries=[Q("q1", "full_scan", marks=4)])]
    )
    make_store(
        stores / "d__c__skip.db",
        "d__c__skip",
        [
            T(
                "tm",
                skip_index="minmax:1",
                index_bytes=42,
                queries=[
                    Q("m0", "full_scan", marks=4),
                    Q("m1", predicate_sql="v IN (0)", marks=1),
                ],
            ),
            T(
                "tn",
                skip_index="none",
                queries=[
                    Q("n0", "full_scan", marks=4),
                    Q("n1", predicate_sql="v IN (0)", marks=4),
                ],
            ),
        ],
    )
    (ds / "cost.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "rows": 4 * MARK_ROWS,
                "cols": 1,
                "wall_s": 1.0,
                "cpu_s": 2.0,
                "peak_mb": 10.0,
                "frame_mb": 0.25,
                "exit_code": 0,
                "error": None,
            }
        )
    )
    return run


def test_score_run_writes_every_file(tmp_path):
    run = _run_dir(tmp_path)
    notes = score_run(run)
    assert notes == {"unparsed": 0, "marks_mismatch": 0, "missing_stores": 0}
    for name in FILES:
        assert (run.path / f"{name}.arrow").is_file(), name
    cells = pl.read_ipc(run.path / "size_cells.arrow")
    assert cells["rel_error"].to_list() == [pytest.approx(0.1)]
    skip = pl.read_ipc(run.path / "skip_regret.arrow")
    assert skip["chosen_skip_freq"].to_list() == [pytest.approx(0.75)]
    cost = pl.read_ipc(run.path / "recommend_cost.arrow")
    assert cost["cpu_s"].to_list() == [2.0] and cost["peak_mb"].to_list() == [10.0]
    qc = pl.read_ipc(run.path / "query_cost.arrow")
    assert qc.height > 0 and qc["cpu_us_p50"].null_count() == 0
    assert qc["memory_bytes_p50"].null_count() == 0
    meta = pa.ipc.open_file(run.path / "size_cells.arrow").schema.metadata
    assert meta[b"git_sha"] == b"abc" and meta[b"run_id"] == run.run_id.encode()


def test_missing_store_is_counted(tmp_path):
    run = _run_dir(tmp_path)
    (run.stores("d") / "d__c__skip.db").unlink()
    notes = score_run(run)
    assert notes["missing_stores"] == 1
    assert pl.read_ipc(run.path / "skip_queries.arrow").height == 0


def test_missing_cost_json_is_a_row(tmp_path):
    run = _run_dir(tmp_path)
    run.cost("d").unlink()
    score_run(run)
    cost = pl.read_ipc(run.path / "recommend_cost.arrow")
    assert cost["status"].to_list() == ["missing"] and cost["cpu_s"].to_list() == [None]
