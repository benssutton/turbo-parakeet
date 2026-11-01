"""Polars and DataFusion for real, end to end through run_suite at small row counts: both engines x
both storage modes x compression, the equal-results check on every query, resume (IPC files reused),
`report --baseline` within an engine and `clean`. No server: runs in CI. The two suites run once per
module (about 30 s all told); the tests read what those runs left behind."""

import csv
import io
import multiprocessing
import textwrap

import pytest
from rich.console import Console

from chbench.cli import main
from chbench.harness import quiet
from chbench.harness.engines import inproc
from chbench.harness.orchestrate import run_suite
from chbench.harness.report import summarise
from chbench.harness.store import Store
from chbench.harness.suite import parse_suite
from tests.harness.test_rounds import SNAP

STRINGS = textwrap.dedent("""
    name: t_inproc_strings
    dataset:
      rows: 3000
      arrow_type: string
      string_length: 6
      null_count: 30
      cardinality_ratio: 0.01
      seed: 3
    polars:
      type: [String, 'Categorical(UInt8)', Enum]
      storage: [memory, ipc]
      compression: [none, ZSTD(3)]
      threads: [2]
    datafusion:
      type: [string, 'dictionary<values=string, indices=uint8, ordered=0>']
      storage: [memory, ipc]
      compression: [none, ZSTD(3)]
      threads: [2]
    queries:
      kinds:
        - {kind: full_scan}
        - {kind: filter, target_fraction: 0.01, predicate: in, values: 1}
        - {kind: filter, target_fraction: 0.2, predicate: range}
      samples: 2
    results:
      percentiles: [50]
""")
NUMBERS = textwrap.dedent("""
    name: t_inproc_numbers
    dataset:
      rows: 3000
      arrow_type: int64
      value_range: [0, 30000]
      null_count: 0
      cardinality_ratio: 0.1
      seed: 4
    polars:
      type: [Int64, UInt16, Float64]
      storage: [ipc]
      compression: [LZ4_FRAME]
      threads: [1]
    datafusion:
      type: [int64, uint16, double]
      storage: [memory]
      compression: [none]
      threads: [1, 2]
    queries:
      kinds:
        - {kind: full_scan}
        - {kind: filter, target_fraction: 0.05, predicate: in, values: 2}
        - {kind: filter, target_fraction: 0.1, predicate: range}
      samples: 2
    results:
      percentiles: [50]
""")


@pytest.fixture(scope="module", autouse=True)
def small_batches():
    # Batches of ~2 ms (at most 16 runs) instead of 250 ms, so CI spends seconds, not minutes.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(quiet, "snapshot", lambda client: dict(SNAP))
        mp.setattr(inproc, "TARGET_BATCH_NS", 2_000_000)
        mp.setattr(inproc, "MAX_REPEATS", 16)
        yield


def _run(text, root, **kw):
    before = set(multiprocessing.active_children())
    r = run_suite(
        parse_suite(text),
        store_path=root / "r.db",
        cache_dir=root / "ds",
        console=Console(file=io.StringIO(), width=400),
        **kw,
    )
    # Every worker this run started is closed: the multiprocessing children it added, not
    # psutil's children(), which on Linux include the spawn context's long-lived
    # resource_tracker (no worker); children alive before the run are not this run's.
    assert set(multiprocessing.active_children()) <= before
    return r


@pytest.fixture(scope="module")
def strings(tmp_path_factory):
    """STRINGS run once, then once more with --samples-add (what the resume test needs); the first
    run's rows are read in between."""
    root = tmp_path_factory.mktemp("strings")
    first = _run(STRINGS, root)
    with Store(root / "r.db") as s:
        seen = {
            "statuses": s.conn.execute(
                "SELECT DISTINCT status, error FROM queries"
            ).fetchall(),
            "samples": s.conn.execute(
                "SELECT repeats, repeats_capped FROM samples"
            ).fetchall(),
            "tables": s.table_rows("t_inproc_strings"),
            "rows": s.query_rows("t_inproc_strings"),
        }
    files = sorted((root / "ds" / "ipc").glob("*.arrow"))
    stamps = [f.stat().st_mtime_ns for f in files]
    second = _run(STRINGS, root, samples_add=1)
    return root, first, seen, files, stamps, second


@pytest.fixture(scope="module")
def numbers(tmp_path_factory):
    root = tmp_path_factory.mktemp("numbers")
    r = _run(NUMBERS, root)
    with Store(root / "r.db") as s:
        seen = {
            "statuses": s.conn.execute(
                "SELECT DISTINCT status, error FROM queries"
            ).fetchall(),
            "samples": s.conn.execute(
                "SELECT repeats, repeats_capped FROM samples"
            ).fetchall(),
            "tables": s.table_rows("t_inproc_numbers"),
        }
    return root, r, seen


@pytest.mark.parametrize(
    "fixture, ok, skipped", [("strings", 15, 5), ("numbers", 9, 0)]
)
def test_every_query_answers_the_reference(fixture, ok, skipped, request):
    run = request.getfixturevalue(fixture)
    r, seen = run[1], run[2]
    assert (r.tables_ok, r.tables_skipped, r.tables_error) == (ok, skipped, 0)
    assert seen["statuses"] == [("ok", None)]
    assert r.samples_written == ok * 3 * 2
    ks = seen["samples"]
    assert ks
    assert all(1 <= k <= 16 and capped in (0, 1) for k, capped in ks)
    for t in seen["tables"]:
        if t["status"] != "ok":
            continue
        if t["storage"] == "ipc":
            assert t["ipc_bytes"] > 0
            assert t["ipc_batch_rows"] == "122880"
        else:
            assert t["ipc_bytes"] is None
            assert t["ipc_batch_rows"] is None
            assert t["memory_bytes"] > 0


def test_ipc_compression_shrinks_strings_and_files_are_reused(strings):
    _, _, seen, files, stamps, second = strings
    tables = [t for t in seen["tables"] if t["storage"] == "ipc"]
    by = {(t["engine"], t["ch_type"], t["compression"]): t["ipc_bytes"] for t in tables}
    assert by[("polars", "String", "ZSTD(3)")] < by[("polars", "String", "none")]
    assert (
        by[("datafusion", "string", "ZSTD(3)")] < by[("datafusion", "string", "none")]
    )
    assert len(files) == 10
    assert all(f.is_file() for f in files)
    assert second.samples_written == 15 * 3
    assert [f.stat().st_mtime_ns for f in files] == stamps


def test_a_baseline_pairs_within_each_engine(strings):
    root = strings[0]
    with Store(root / "r.db") as s:
        rows = s.query_rows("t_inproc_strings")
    summary = summarise(rows, [50], baseline="ch_type=String")
    cat = [x for x in summary if x["ch_type"] == "Categorical(UInt8)"]
    # Paired on wall time: process CPU time ticks in 15.6 ms steps on Windows, so cpu_ratio and
    # n_rounds are mostly empty for 2 ms batches.
    assert len(cat) == 3 * 3
    assert all(x["server_ratio"] is not None for x in cat)
    assert all(
        x["server_ratio"] is None for x in summary if x["engine"] == "datafusion"
    )
    out = root / "polars.csv"
    code = main(
        [
            "report",
            "--suite",
            "t_inproc_strings",
            "--store",
            str(root / "r.db"),
            "--baseline",
            "ch_type=String",
            "--where",
            "engine=polars",
            "--csv",
            str(out),
        ]
    )
    with open(out, newline="") as f:
        written = list(csv.DictReader(f))
    assert code == 0
    assert {r["engine"] for r in written} == {"polars"}
    assert any(
        r["ch_type"] == "Categorical(UInt8)" and r["server_ratio"] for r in written
    )


def test_clean_removes_the_ipc_files(numbers):
    root = numbers[0]
    files = sorted((root / "ds" / "ipc").glob("*.arrow"))
    assert len(files) == 3  # the three Polars ipc tables
    code = main(
        [
            "clean",
            "--suite",
            "t_inproc_numbers",
            "--store",
            str(root / "r.db"),
            "--cache-dir",
            str(root / "ds"),
        ]
    )
    assert code == 0
    assert not any(f.exists() for f in files)
