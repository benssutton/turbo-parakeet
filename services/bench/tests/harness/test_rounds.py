"""`run_suite`'s round scheduler, without a server: `CHClient`, `load_table` and the quietness
snapshot are replaced; the dataset, grid, queries, runner and store are real."""

import pytest

from chbench.harness import orchestrate, quiet
from chbench.harness.loader import LoadResult, StorageStats
from chbench.harness.orchestrate import run_suite
from chbench.harness.store import Store
from chbench.harness.suite import parse_suite
from tests.harness.fakes import FakeClient, rounds
from tests.harness.test_suite import MINIMAL

SUITE = (
    MINIMAL.replace("name: t", "name: t_rounds")
    .replace("codec: [none]", "codec: [none, Delta]")
    .replace("compression: [LZ4]", "compression: [LZ4, ZSTD]")
    .replace(
        "    - {kind: full_scan}",
        "    - {kind: full_scan}\n    - {kind: filter, target_fraction: 0.1, predicate: range}",
    )
)  # 4 tables x 2 queries = 8 pairs, samples 3 (MINIMAL is dedented: 2-space YAML)

SNAP = {"host_cpu_pct": 1.0, "ch_active_merges": 0}


def _ok_load(client, name, variant, dataset):
    stats = StorageStats(
        compressed_bytes=10, uncompressed_bytes=40, marks_bytes=1, rows=1000, parts=1
    )
    return LoadResult(status="ok", error=None, insert_ms=1.0, stats=stats)


@pytest.fixture
def fake(monkeypatch):
    holder = {"client": FakeClient()}
    monkeypatch.setattr(orchestrate, "CHClient", lambda conn: holder["client"])
    monkeypatch.setattr(orchestrate, "load_table", _ok_load)
    monkeypatch.setattr(quiet, "snapshot", lambda client: dict(SNAP))
    return holder


def _run(tmp_path, **kw):
    return run_suite(
        parse_suite(SUITE),
        store_path=tmp_path / "r.db",
        cache_dir=tmp_path / "ds",
        **kw
    )


def test_the_suite_is_four_tables_of_two_queries():
    s = parse_suite(SUITE)
    assert s.clickhouse.codec == ["none", "Delta"]
    assert s.clickhouse.compression == ["LZ4", "ZSTD"]
    assert [k.kind for k in s.queries.kinds] == ["full_scan", "filter"]


def test_every_pair_runs_once_per_round_after_an_untimed_warm_up(fake, tmp_path):
    r = _run(tmp_path)
    assert r.tables_ok == 4 and r.tables_skipped == 0
    assert r.queries_run == 8 and r.samples_written == 24
    calls = fake["client"].calls
    assert [c[2] for c in calls[:8]] == [None] * 8  # the warm-up round
    rs = rounds(calls)
    assert len(rs) == 3 and all(len({c[1] for c in rd}) == 8 for rd in rs)


def test_round_order_is_shuffled_per_round_and_reproducible(fake, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _run(tmp_path / "a")
    first = [[c[1] for c in rd] for rd in rounds(fake["client"].calls)]
    fake["client"] = FakeClient()
    _run(tmp_path / "b")
    second = [[c[1] for c in rd] for rd in rounds(fake["client"].calls)]
    assert first == second and len({tuple(o) for o in first}) > 1


def test_samples_carry_their_round_and_samples_add_continues_the_numbering(
    fake, tmp_path
):
    _run(tmp_path)
    fake["client"] = FakeClient()
    r2 = _run(tmp_path, samples_add=2)
    assert r2.samples_written == 16
    with Store(tmp_path / "r.db") as s:
        got = s.conn.execute(
            "SELECT round_no, count(*) FROM samples GROUP BY round_no ORDER BY round_no"
        ).fetchall()
        assert all(r["cpu_us"] == 150.0 for r in s.query_rows("t_rounds"))
    assert got == [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8)]


def test_quietness_is_recorded_before_and_after(fake, tmp_path):
    r = _run(tmp_path)
    with Store(tmp_path / "r.db") as s:
        before, after = s.conn.execute(
            "SELECT quiet_before, quiet_after FROM runs WHERE run_id = ?", (r.run_id,)
        ).fetchone()
    assert before and after


def test_a_failing_query_is_recorded_and_the_others_complete(fake, tmp_path):
    fake["client"] = FakeClient(fail_sql="BETWEEN")  # every filter query fails
    r = _run(tmp_path)
    assert r.samples_written == 12
    with Store(tmp_path / "r.db") as s:
        status = dict(
            s.conn.execute(
                "SELECT kind, group_concat(DISTINCT status) FROM queries GROUP BY kind"
            ).fetchall()
        )
        error = s.conn.execute(
            "SELECT DISTINCT error FROM queries WHERE kind = 'filter'"
        ).fetchall()
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert status == {"filter": "error", "full_scan": "ok"} and finished
    assert error == [("RuntimeError: boom",)]


def test_ctrl_c_mid_round_keeps_the_samples_and_marks_the_rest_interrupted(
    fake, tmp_path
):
    fake["client"] = FakeClient(interrupt_at=12)  # round 1, fourth call
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    with Store(tmp_path / "r.db") as s:
        n = s.conn.execute("SELECT count(*) FROM samples").fetchone()[0]
        status = s.conn.execute("SELECT status FROM queries").fetchall()
        finished = s.conn.execute("SELECT finished_at FROM runs").fetchone()[0]
    assert n == 11 and {st for (st,) in status} == {"interrupted"} and finished


def _unsolved(monkeypatch):
    from synthdata.generate import SolveRecord

    real = orchestrate.load_or_generate

    def load(spec, cache_dir):
        ds = real(spec, cache_dir)
        ds.manifest.solves["entropy_bits"] = SolveRecord(
            "invalid_bracket", "invalid bracket: guard fails at lo=0.01"
        )
        ds.manifest.targets["entropy_bits"] = 3.0
        return ds

    monkeypatch.setattr(orchestrate, "load_or_generate", load)


def test_an_invalid_bracket_is_refused_before_anything_is_loaded(
    fake, tmp_path, monkeypatch
):
    _unsolved(monkeypatch)
    with pytest.raises(
        orchestrate.UnsolvedDatasetError, match="entropy_bits.*--allow-unsolved"
    ):
        _run(tmp_path)
    assert not (tmp_path / "r.db").exists()


def test_allow_unsolved_runs_it_with_a_warning(fake, tmp_path, monkeypatch):
    _unsolved(monkeypatch)
    assert _run(tmp_path, allow_unsolved=True).samples_written == 24
