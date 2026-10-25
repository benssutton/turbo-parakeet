"""End-to-end tests for `run_suite` against the live container.

Every table these create is named `bench.t_orch_<8 hex>` (`ddl.table_name`
of the suite name below), so the autouse fixture can drop the lot by prefix
after the module finishes, whatever the tests did to the store.
"""

import pytest

from chbench.harness.orchestrate import _samples_to_run, run_suite
from chbench.harness.runner import RunResult
from chbench.harness.store import Store
from chbench.harness.suite import parse_suite
from tests.harness.test_suite import MINIMAL

SMALL = (
    MINIMAL.replace("rows: 1000", "rows: 20000")
    .replace("name: t", "name: t_orch")
    .replace("codec: [none]", "codec: [none, T64]")
)  # T64 on Float64 is skipped


@pytest.fixture(scope="module", autouse=True)
def drop_orch_tables(request):
    """Drop every `bench.t_orch_*` table this module created, at teardown.

    Depends on `ch` lazily (through `request.getfixturevalue`) so the module
    still collects and skips cleanly when the container is unreachable.
    """
    yield
    try:
        client = request.getfixturevalue("ch")
    except Exception:  # noqa: BLE001 - nothing to clean up if there is no server
        return
    names = client.query_rows(
        "SELECT name FROM system.tables WHERE database = 'bench' "
        "AND name LIKE 't\\_orch\\_%'"
    )
    for (name,) in names:
        client.command(f"DROP TABLE IF EXISTS bench.{name}")


@pytest.mark.clickhouse
def test_run_suite_end_to_end(ch, tmp_path):
    suite = parse_suite(SMALL)
    r = run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    assert r.tables_ok == 1 and r.tables_skipped == 1 and r.tables_error == 0
    assert r.queries_run == 1 and r.samples_written == 3
    with Store(tmp_path / "r.db") as s:
        rows = s.query_rows("t_orch")
        skipped = s.conn.execute(
            "SELECT skip_reason FROM tables WHERE status='skipped'"
        ).fetchone()
    assert len(rows) == 3
    assert skipped[0].startswith("T64 codec is not supported")


@pytest.mark.clickhouse
def test_rerun_skips_queries_that_already_hold_enough_samples(ch, tmp_path):
    suite = parse_suite(SMALL)
    run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    r2 = run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    assert r2.queries_skipped_existing == 1 and r2.samples_written == 0
    r3 = run_suite(
        suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds", samples_add=2
    )
    assert r3.samples_written == 2
    r4 = run_suite(
        suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds", force=True
    )
    assert r4.samples_written == 3


@pytest.mark.clickhouse
def test_rerun_does_not_reload_a_table_whose_queries_are_all_held(
    ch, tmp_path, monkeypatch
):
    """The query variants depend only on the variant and the
    seeded rng, so they can be expanded before the load; when every one of
    them already holds its target, the load (drop, create, insert, optimise)
    is skipped and the cell still counts as ok. `--force` or `--samples-add`
    make the queries want samples again, and then the table is loaded."""
    from chbench.harness import orchestrate

    suite = parse_suite(SMALL)
    run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    with Store(tmp_path / "r.db") as s:
        before = s.conn.execute(
            "SELECT table_id, insert_ms, created_at FROM tables WHERE status='ok'"
        ).fetchall()
    assert len(before) == 1

    real_load = orchestrate.load_table
    calls: list[str] = []

    def counting_load(client, name, variant, dataset):
        calls.append(name)
        return real_load(client, name, variant, dataset)

    monkeypatch.setattr(orchestrate, "load_table", counting_load)

    r2 = run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    assert calls == []
    assert r2.tables_ok == 1 and r2.tables_skipped == 1 and r2.tables_error == 0
    assert r2.queries_skipped_existing == 1 and r2.samples_written == 0
    with Store(tmp_path / "r.db") as s:
        after = s.conn.execute(
            "SELECT table_id, insert_ms, created_at FROM tables WHERE status='ok'"
        ).fetchall()
    assert after == before

    r3 = run_suite(
        suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds", samples_add=1
    )
    assert len(calls) == 1 and r3.samples_written == 1


@pytest.mark.clickhouse
def test_dry_run_touches_nothing(ch, tmp_path):
    r = run_suite(
        parse_suite(SMALL),
        store_path=tmp_path / "r.db",
        cache_dir=tmp_path / "ds",
        dry_run=True,
    )
    assert r.tables_ok == 1 and r.tables_skipped == 1 and r.samples_written == 0
    assert not (tmp_path / "r.db").exists()


def test_samples_to_run_gives_a_never_run_query_the_full_sample_count():
    """Minor 5: the target is max(samples, held + samples_add), so a newly
    added codec is not left with fewer samples than the incumbents it exists
    to be compared against."""
    assert _samples_to_run(held=0, samples=3, force=False, samples_add=0) == 3
    assert _samples_to_run(held=3, samples=3, force=False, samples_add=0) == 0
    # A query already holding 3 takes 2 more.
    assert _samples_to_run(held=3, samples=3, force=False, samples_add=2) == 2
    # A query that never ran still takes the suite's full `samples`, not 2.
    assert _samples_to_run(held=0, samples=3, force=False, samples_add=2) == 3
    # force ignores the bank entirely.
    assert _samples_to_run(held=5, samples=3, force=True, samples_add=0) == 3
    # Never negative: run_query(samples=-2) would report a mean of nan.
    assert _samples_to_run(held=0, samples=3, force=True, samples_add=-2) == 0
    assert _samples_to_run(held=5, samples=3, force=False, samples_add=-9) == 0


@pytest.mark.clickhouse
def test_query_error_is_recorded_on_the_query_row(ch, tmp_path, monkeypatch):
    """The message lands on the query, and the run continues."""
    from chbench.harness import orchestrate

    def failing_run_query(client, name, variant, qv, samples, *, on_sample=None):
        return RunResult(
            status="error",
            error="DatabaseError: Code: 241. Memory limit exceeded",
            samples=[],
            explain="",
        )

    monkeypatch.setattr(orchestrate, "run_query", failing_run_query)
    r = run_suite(
        parse_suite(SMALL), store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds"
    )
    # The run carried on to the end rather than aborting.
    assert r.tables_ok == 1 and r.queries_run == 1 and r.samples_written == 0
    with Store(tmp_path / "r.db") as s:
        status, error = s.conn.execute("SELECT status, error FROM queries").fetchone()
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert status == "error"
    assert error.startswith("DatabaseError: Code: 241")
    assert finished is not None


@pytest.mark.clickhouse
def test_finish_run_is_called_when_something_other_than_ctrl_c_escapes(
    ch, tmp_path, monkeypatch
):
    """Important 2: `finished_at` must not stay NULL because a
    `database is locked` or an `expand_queries` ValueError escaped the loop."""
    from chbench.harness import orchestrate

    def exploding(*args, **kwargs):
        raise ValueError("range_predicate: values are all null")

    monkeypatch.setattr(orchestrate, "expand_queries", exploding)
    with pytest.raises(ValueError):
        run_suite(
            parse_suite(SMALL), store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds"
        )
    with Store(tmp_path / "r.db") as s:
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert finished is not None


def test_achieved_bounds_of_a_decimal64_array():
    """`pc.min_max` has no kernel for decimal32/decimal64 on
    pyarrow 24 and raised `ArrowNotImplementedError` before the grid was
    even printed, so a decimal dataset could not be dry-run."""
    import decimal
    import pyarrow as pa

    from chbench.harness.orchestrate import _achieved_bounds

    arr = pa.array(
        [
            decimal.Decimal("-1.50000000"),
            decimal.Decimal("9.99999999"),
            None,
            decimal.Decimal("0.00000001"),
        ],
        type=pa.decimal64(18, 8),
    )
    lo, hi = _achieved_bounds(arr)
    assert (lo, hi) == (decimal.Decimal("-1.50000000"), decimal.Decimal("9.99999999"))
    arr32 = pa.array(
        [decimal.Decimal("3.25"), decimal.Decimal("-0.75")], type=pa.decimal32(9, 2)
    )
    assert _achieved_bounds(arr32) == (
        decimal.Decimal("-0.75"),
        decimal.Decimal("3.25"),
    )
    # The common case is unchanged.
    assert _achieved_bounds(pa.array([3.0, -1.0, 2.0])) == (-1.0, 3.0)


@pytest.mark.clickhouse
def test_refuses_to_start_while_an_unfinished_run_of_the_suite_exists(ch, tmp_path):
    """`finish_run` sits in a `finally`, so a `runs` row with
    `finished_at` NULL means a process died mid-run (or is still running
    against this store); a second run must say so and stop, unless told
    that the first is gone."""
    from chbench.harness.orchestrate import UnfinishedRunError

    suite = parse_suite(SMALL)
    with Store(tmp_path / "r.db") as s:
        s.init_schema()
        stale = s.begin_run(suite)
    with pytest.raises(UnfinishedRunError, match=rf"{stale}.*--ignore-unfinished"):
        run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    with Store(tmp_path / "r.db") as s:
        assert s.conn.execute("SELECT count() FROM runs").fetchone()[0] == 1
    r = run_suite(
        suite,
        store_path=tmp_path / "r.db",
        cache_dir=tmp_path / "ds",
        ignore_unfinished=True,
    )
    assert r.run_id is not None and r.samples_written == 3


@pytest.mark.clickhouse
def test_ctrl_c_mid_query_records_the_interrupted_query_and_finishes_the_run(
    ch, tmp_path, monkeypatch
):
    """Driven through the real `CHClient` and the real `Store`:
    a `KeyboardInterrupt` raised inside the second timed sample query (the
    `_InterruptOnTimedCall` proxy from the runner tests) must propagate,
    leave the first sample committed, mark the query row `interrupted`
    rather than leave the pre-run upsert's NULL, and finish the run row."""
    from chbench.harness import orchestrate
    from chbench.harness.client import CHClient
    from tests.harness.test_runner import _InterruptOnTimedCall

    monkeypatch.setattr(
        orchestrate,
        "CHClient",
        lambda conn: _InterruptOnTimedCall(CHClient(conn), at=2),
    )
    suite = parse_suite(SMALL)
    with pytest.raises(KeyboardInterrupt):
        run_suite(suite, store_path=tmp_path / "r.db", cache_dir=tmp_path / "ds")
    with Store(tmp_path / "r.db") as s:
        samples = s.conn.execute("SELECT sample_no FROM samples").fetchall()
        queries = s.conn.execute(
            "SELECT status, error, explain_plan FROM queries"
        ).fetchall()
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert samples == [(0,)]
    assert len(queries) == 1
    status, error, explain = queries[0]
    assert status == "interrupted" and error is None
    assert finished is not None
