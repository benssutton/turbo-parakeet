"""Tests for the `chbench` console script, driven through `subprocess`.

The CLI is tested as a process, not by calling `main` in-process, because
two of its responsibilities only exist there: the exit code, and the UTF-8
reconfiguration of a real (non-tty) stdout that would otherwise raise
`UnicodeEncodeError` on the box-drawing characters in an EXPLAIN plan
(FINDINGS §5).

`run --dry-run` is exercised against a small temporary suite rather than
`suites/float_codecs.yaml`: the grid assertion the task brief names is the
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

import pytest

from chbench.datagen import DatasetSpec
from chbench.datagen.cache import load_or_generate
from chbench.harness.store import Store
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
    """Without the guard this reaches run_query(samples=-2), which runs
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
