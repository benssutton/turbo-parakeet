"""`run_suite` over a real-data dataset (`dataset: {path, column}`).

Tables are named `bench.t_orchpath_<hex>`; the module fixture drops them.
"""

import numpy as np
import pyarrow as pa
import pytest

from chbench.harness.orchestrate import run_suite
from chbench.harness.store import Store
from chbench.harness.suite import load_suite

SUITE = """
name: t_orchpath
dataset: {path: x.arrow, column: v}
clickhouse:
  type: [Int32, UInt32]
  nullable: [true]
  primary_key: [false]
  projection: [false]
  codec: [none]
  skip_index: [none]
  compression: [LZ4]
queries:
  kinds:
    - {kind: full_scan}
    - {kind: filter, target_fraction: 0.1, predicate: range}
  samples: 2
results:
  percentiles: [50]
"""


@pytest.fixture(scope="module", autouse=True)
def drop_tables(request):
    yield
    try:
        client = request.getfixturevalue("ch")
    except Exception:  # noqa: BLE001
        return
    for (name,) in client.query_rows(
        "SELECT name FROM system.tables WHERE database = 'bench' "
        "AND name LIKE 't\\_orchpath\\_%'"
    ):
        client.command(f"DROP TABLE IF EXISTS bench.{name}")


@pytest.mark.clickhouse
def test_a_path_suite_runs_end_to_end_and_the_store_records_the_dataset(ch, tmp_path):
    values = np.random.default_rng(0).integers(0, 500, 1000).astype(np.int32)
    array = pa.array(values, mask=np.arange(1000) % 97 == 0)
    batch = pa.record_batch([array], names=["v"])
    with pa.OSFile(str(tmp_path / "x.arrow"), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    (tmp_path / "s.yaml").write_text(SUITE, encoding="utf-8")
    suite = load_suite(tmp_path / "s.yaml")

    cache = tmp_path / "ds"
    r = run_suite(suite, store_path=tmp_path / "r.db", cache_dir=cache)
    assert r.tables_ok == 2 and r.tables_error == 0
    assert r.queries_run == 4 and r.samples_written == 8
    assert not cache.exists()  # a path dataset is not copied

    with Store(tmp_path / "r.db") as s:
        ds = s.conn.execute(
            "SELECT dataset_id, spec_json, arrow_path FROM datasets"
        ).fetchall()
        n = s.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert len(ds) == 1 and ds[0][0] == suite.dataset.dataset_id
    assert '"path"' in ds[0][1] and ds[0][2] == suite.dataset.path
    assert n == 8


BINARY_SUITE = """
name: t_orchpath
dataset: {path: x.arrow, column: v}
clickhouse:
  type: [String]
  nullable: [true]
  primary_key: [false]
  projection: [false]
  codec: [none, LowCardinality]
  skip_index: [none]
  compression: [LZ4]
queries:
  kinds:
    - {kind: full_scan}
    - {kind: filter, target_fraction: 0.01, predicate: in, values: 2}
    - {kind: filter, target_fraction: 0.1, predicate: in, values: 20}
  samples: 2
results:
  percentiles: [50]
"""


@pytest.mark.clickhouse
def test_a_binary_path_suite_loads_into_string_and_both_literal_forms_match(
    ch, tmp_path
):
    # half the values are ASCII (quoted literals), half non-UTF-8 behind a 0xff
    # lead byte (`unhex` literals); every 97th row is null
    codes = np.random.default_rng(0).integers(0, 100, 1000)
    values = [
        None if i % 97 == 0 else bytes([0xFF, c]) if i % 2 else b"a%d" % c
        for i, c in enumerate(codes)
    ]
    batch = pa.record_batch([pa.array(values, pa.binary())], names=["v"])
    with pa.OSFile(str(tmp_path / "x.arrow"), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    (tmp_path / "s.yaml").write_text(BINARY_SUITE, encoding="utf-8")
    suite = load_suite(tmp_path / "s.yaml")
    assert suite.dataset.arrow_type == "binary"

    r = run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    assert r.tables_ok == 2 and r.tables_error == 0 and r.queries_run == 6

    with Store(tmp_path / "r.db") as s:
        tables = dict(s.conn.execute("SELECT table_id, codec FROM tables").fetchall())
        rows = s.conn.execute(
            "SELECT table_id, kind, status, actual_fraction, predicate_sql "
            "FROM queries"
        ).fetchall()
    assert sorted(tables.values()) == ["LowCardinality", "none"]
    for table_id in tables:
        mine = [q for q in rows if q[0] == table_id]
        assert sorted(q[1] for q in mine) == ["filter", "filter", "full_scan"]
        assert all(q[2] == "ok" for q in mine)
        filters = [q for q in mine if q[1] == "filter"]
        predicates = " ".join(q[4] for q in filters)
        assert "unhex(" in predicates and "'a" in predicates
        name = f"bench.t_orchpath_{table_id[:8]}"
        assert ch.query_rows(f"SELECT count() FROM {name}")[0][0] == 1000
        for _, _, _, fraction, predicate in filters:
            # the server, not the client, counts the rows the literals select
            (matched,) = ch.query_rows(f"SELECT count() FROM {name} WHERE {predicate}")[
                0
            ]
            assert fraction > 0 and matched == round(fraction * 1000)
