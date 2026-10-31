"""Tests for the `chbench` console script, driven through `subprocess`.

The CLI is tested as a process, not by calling `main` in-process, because
two of its responsibilities only exist there: the exit code, and the UTF-8
reconfiguration of a real (non-tty) stdout that would otherwise raise
`UnicodeEncodeError` on the box-drawing characters in an EXPLAIN plan
(FINDINGS §5).

`run --dry-run` is exercised against a small temporary suite rather than
`suites/float_codecs.yaml`: the grid assertion needed is the
T64-on-Float64 skip reason, and `float_codecs.yaml`'s codec axis is
`[Gorilla, Delta, DoubleDelta]` with no T64 at all, while its dataset is
10 000 000 rows -- a five-second generation and an 80 MB Arrow file for a
test whose subject is the CLI's plumbing, not the generator's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from synthdata import DatasetSpec
from synthdata.cache import load_or_generate
from chbench import cli
from chbench.cli import main
from chbench.harness.store import Store
from tests.harness.test_store import _v2_store
from tests.harness.test_orchestrate import SMALL

ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "chbench.cli", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_help_lists_the_four_commands():
    p = run_cli("--help")
    assert p.returncode == 0, p.stderr
    for command in ("run", "report", "datasets", "clean"):
        assert command in p.stdout


@pytest.mark.clickhouse
def test_run_dry_run_prints_the_grid_and_creates_no_store(ch, tmp_path):
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    p = run_cli(
        "run",
        str(suite_path),
        "--dry-run",
        "--store",
        str(tmp_path / "r.db"),
        "--cache-dir",
        str(tmp_path / "ds"),
    )
    assert p.returncode == 0, p.stderr
    assert "T64 codec is not supported" in p.stdout
    assert not (tmp_path / "r.db").exists()


@pytest.mark.clickhouse
def test_clean_drops_the_suites_tables(ch, tmp_path):
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    store = tmp_path / "r.db"
    try:
        p = run_cli(
            "run",
            str(suite_path),
            "--store",
            str(store),
            "--cache-dir",
            str(tmp_path / "ds"),
        )
        assert p.returncode == 0, p.stderr
        assert (
            ch.query_rows(
                "SELECT count() FROM system.tables WHERE database = 'bench' "
                "AND name LIKE 't\\_orch\\_%'"
            )[0][0]
            > 0
        )

        c = run_cli("clean", "--suite", "t_orch", "--store", str(store))
        assert c.returncode == 0, c.stderr
        assert (
            ch.query_rows(
                "SELECT count() FROM system.tables WHERE database = 'bench' "
                "AND name LIKE 't\\_orch\\_%'"
            )[0][0]
            == 0
        )
    finally:
        for (name,) in ch.query_rows(
            "SELECT name FROM system.tables WHERE database = 'bench' "
            "AND name LIKE 't\\_orch\\_%'"
        ):
            ch.command(f"DROP TABLE IF EXISTS bench.{name}")


def test_report_on_an_unknown_suite_exits_non_zero(tmp_path):
    store = tmp_path / "r.db"
    p = run_cli("report", "--suite", "nosuch", "--store", str(store))
    assert p.returncode != 0
    assert "nosuch" in p.stdout + p.stderr
    # The guard must run before the store is opened: sqlite3.connect would
    # otherwise leave an empty results.db behind for a suite that has none.
    assert not store.exists()


def test_clean_without_a_store_exits_non_zero(tmp_path):
    p = run_cli("clean", "--suite", "nosuch", "--store", str(tmp_path / "r.db"))
    assert p.returncode != 0
    assert "nosuch" in p.stdout + p.stderr


def test_datasets_lists_cached_manifests(tmp_path):
    cache = tmp_path / "ds"
    spec = DatasetSpec(
        rows=200,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=3,
    )
    dataset = load_or_generate(spec, cache)

    p = run_cli("datasets", "--cache-dir", str(cache))
    assert p.returncode == 0, p.stderr
    assert dataset.manifest.dataset_id[:8] in p.stdout
    assert "float64" in p.stdout

    m = run_cli("datasets", "--cache-dir", str(cache), "--metrics")
    assert m.returncode == 0, m.stderr
    assert "entropy" in m.stdout


def test_datasets_on_an_empty_cache_says_so(tmp_path):
    p = run_cli("datasets", "--cache-dir", str(tmp_path / "empty"))
    assert p.returncode == 0, p.stderr
    assert "no cached datasets" in p.stdout.lower()


def test_negative_samples_add_is_rejected_at_parse_time(tmp_path):
    """Without the guard this reaches a negative want, which takes
    nothing and reports a mean of nan."""
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    p = run_cli(
        "run",
        str(suite_path),
        "--samples-add",
        "-2",
        "--store",
        str(tmp_path / "r.db"),
        "--cache-dir",
        str(tmp_path / "ds"),
    )
    assert p.returncode != 0
    assert "zero or greater" in p.stdout + p.stderr
    assert not (tmp_path / "r.db").exists()


def test_unreachable_server_fails_before_generating_the_dataset(tmp_path):
    """A wrong CH_PORT must cost a message, not a full generation followed by
    a urllib3 traceback. The empty cache dir afterwards is the evidence that
    the connection was tested first."""
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    cache = tmp_path / "ds"
    env = dict(os.environ, CH_PORT="9")
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "chbench.cli",
            "run",
            str(suite_path),
            "--store",
            str(tmp_path / "r.db"),
            "--cache-dir",
            str(cache),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert p.returncode == 2
    assert "cannot reach ClickHouse at localhost:9" in p.stdout + p.stderr
    assert not cache.exists()


@pytest.mark.clickhouse
def test_explain_is_stored_always_and_printed_only_on_request(ch, tmp_path):
    """The plan is evidence, so it is always recorded; it is also
    ten lines per query, so it is printed only under --explain."""
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    store = tmp_path / "r.db"
    try:
        quiet = run_cli(
            "run",
            str(suite_path),
            "--store",
            str(store),
            "--cache-dir",
            str(tmp_path / "ds"),
        )
        assert quiet.returncode == 0, quiet.stderr
        assert "ReadFromMergeTree" not in quiet.stdout

        with Store(store) as s:
            stored = s.conn.execute("SELECT explain_plan FROM queries").fetchone()[0]
        assert "ReadFromMergeTree" in stored

        loud = run_cli(
            "run",
            str(suite_path),
            "--store",
            str(store),
            "--cache-dir",
            str(tmp_path / "ds"),
            "--force",
            "--explain",
        )
        assert loud.returncode == 0, loud.stderr
        assert "ReadFromMergeTree" in loud.stdout
    finally:
        for (name,) in ch.query_rows(
            "SELECT name FROM system.tables WHERE database = 'bench' "
            "AND name LIKE 't\\_orch\\_%'"
        ):
            ch.command(f"DROP TABLE IF EXISTS bench.{name}")


@pytest.mark.clickhouse
def test_run_refuses_an_unfinished_run_unless_told_to_ignore_it(ch, tmp_path):
    """Exit 2 naming the stale run, and `--ignore-unfinished`
    is the way past it."""
    from types import SimpleNamespace

    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    store = tmp_path / "r.db"
    with Store(store) as s:
        s.init_schema()
        stale = s.begin_run(SimpleNamespace(name="t_orch", source_yaml=SMALL))
    common = ("--store", str(store), "--cache-dir", str(tmp_path / "ds"))
    p = run_cli("run", str(suite_path), *common)
    assert p.returncode == 2
    assert stale in p.stderr and "--ignore-unfinished" in p.stderr
    p = run_cli("run", str(suite_path), "--ignore-unfinished", *common)
    assert p.returncode == 0, p.stderr


def test_run_offers_allow_unsolved():
    assert "--allow-unsolved" in run_cli("run", "--help").stdout


def test_run_reports_an_old_schema_store_cleanly(tmp_path):
    suite_path = tmp_path / "small.yaml"
    suite_path.write_text(SMALL, encoding="utf-8")
    old = tmp_path / "old.db"
    with Store(old) as s:
        s.init_schema()
        s.conn.execute("PRAGMA user_version = 0")
        s.conn.commit()
    p = run_cli(
        "run", str(suite_path), "--store", str(old), "--cache-dir", str(tmp_path / "ds")
    )
    assert p.returncode == 2
    assert "--store" in p.stderr and "Traceback" not in p.stderr


def _inproc_variant():
    return NS(
        ch_type="String",
        nullable=None,
        primary_key=None,
        projection=None,
        codec=None,
        skip_index=None,
        compression="none",
        status="ok",
        skip_reason=None,
        warnings=(),
        engine="polars",
        storage="ipc",
        threads=2,
        type_spec="String",
        ipc_batch_rows=None,
    )


_PREPARED = NS(
    status="ok",
    error=None,
    insert_ms=1.0,
    stats=None,
    ipc_bytes=3,
    memory_bytes=9,
    rows=1,
)


def _inproc_store(db, *ids):
    with Store(db) as s:
        s.init_schema()
        for tid in ids:
            s.upsert_table(
                tid, "d1", _inproc_variant(), _PREPARED, "2.0.0", suite_name="p"
            )


def _clean(db, cache):
    return main(
        ["clean", "--suite", "p", "--store", str(db), "--cache-dir", str(cache)]
    )


def test_clean_deletes_the_ipc_files_without_a_server(tmp_path, capsys):
    db = tmp_path / "r.db"
    _inproc_store(db, "abc123")
    ipc_file = tmp_path / "ds" / "ipc" / "abc123.arrow"
    ipc_file.parent.mkdir(parents=True)
    ipc_file.write_bytes(b"abc")
    other = ipc_file.with_name("other.arrow")
    other.write_bytes(b"x")
    code = _clean(db, tmp_path / "ds")
    assert code == 0 and not ipc_file.exists() and other.exists()
    assert "deleted 1 IPC file(s) for suite p" in capsys.readouterr().out


def test_clean_deletes_a_left_over_tmp_of_a_recorded_table_only(tmp_path, capsys):
    db = tmp_path / "r.db"
    _inproc_store(db, "abc123")
    ipc = tmp_path / "ds" / "ipc"
    ipc.mkdir(parents=True)
    tmp = ipc / "abc123.tmp"  # a prepare that died mid-write
    tmp.write_bytes(b"half")
    other = ipc / "other.tmp"
    other.write_bytes(b"x")
    assert _clean(db, tmp_path / "ds") == 0
    assert not tmp.exists() and other.exists()
    assert "deleted 1 IPC file(s) for suite p" in capsys.readouterr().out


def test_clean_with_a_missing_file_is_fine(tmp_path, capsys):
    db = tmp_path / "r.db"
    _inproc_store(db, "abc123")
    assert _clean(db, tmp_path / "ds") == 0
    assert "deleted 0 IPC file(s) for suite p" in capsys.readouterr().out


def test_clean_reports_an_undeletable_file_and_goes_on(tmp_path, capsys, monkeypatch):
    db = tmp_path / "r.db"
    _inproc_store(db, "aaa111", "bbb222")
    ipc = tmp_path / "ds" / "ipc"
    ipc.mkdir(parents=True)
    for tid in ("aaa111", "bbb222"):
        (ipc / f"{tid}.arrow").write_bytes(b"x")
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self.name == "aaa111.arrow":
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked)
    code = _clean(db, tmp_path / "ds")
    out = capsys.readouterr()
    assert code == 1
    assert (ipc / "aaa111.arrow").exists() and not (ipc / "bbb222.arrow").exists()
    assert "aaa111.arrow" in out.out + out.err and "deleted 1 IPC file(s)" in out.out


class _FakeClient:
    def __init__(self):
        self.commands = []
        self.connection = NS(database="bench")

    def command(self, sql, *a, **k):
        self.commands.append(sql)

    def query_rows(self, *a, **k):
        return []


def test_clean_on_a_v2_store_drops_its_tables_and_touches_no_files(
    tmp_path, monkeypatch
):
    db = tmp_path / "v2.db"
    _v2_store(db)
    fake = _FakeClient()
    monkeypatch.setattr(cli, "CHClient", lambda conn: fake)
    monkeypatch.setattr(cli.Connection, "from_env", staticmethod(lambda: None))
    cache = tmp_path / "ds"
    code = main(
        ["clean", "--suite", "A", "--store", str(db), "--cache-dir", str(cache)]
    )
    assert code == 0 and not cache.exists()
    with Store(db) as s:
        (name,) = s.tables_for_suite("A")
    assert fake.commands == [f"DROP TABLE IF EXISTS {name}"]
