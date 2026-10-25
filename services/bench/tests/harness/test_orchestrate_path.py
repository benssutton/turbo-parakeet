"""`run_suite` over a real-data dataset (`dataset: {path, column}`, ruling R35).

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
  query_condition_cache: [false]
  cache_mode: [warm]
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
