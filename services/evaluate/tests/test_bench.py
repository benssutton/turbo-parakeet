from pathlib import Path

import pytest

from evaluate import bench


def test_check_server_wraps_connection_errors(monkeypatch):
    class Boom:
        def __init__(self, *a):
            raise ConnectionRefusedError("refused")

    monkeypatch.setattr(bench, "CHClient", Boom)
    with pytest.raises(bench.ServerUnavailable, match="docker start chbench-ch"):
        bench.check_server()


def test_run_all_inline_collects_errors(tmp_path):
    calls = []

    def runner(yaml_path: Path, store: Path, cache: Path) -> str | None:
        calls.append((yaml_path.name, store.name))
        return "boom" if yaml_path.stem == "b" else None

    jobs = [
        (tmp_path / "a.yaml", tmp_path / "s"),
        (tmp_path / "b.yaml", tmp_path / "s"),
    ]
    errors = bench.run_all(jobs, tmp_path / "cache", workers=1, runner=runner)
    assert errors == {"a": None, "b": "boom"}
    assert calls == [("a.yaml", "a.db"), ("b.yaml", "b.db")]


def test_run_one_reports_exceptions(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: [", encoding="utf-8")
    err = bench.run_one(tmp_path / "bad.yaml", tmp_path / "bad.db", tmp_path / "cache")
    assert err and err.startswith(("SuiteError", "ScannerError", "ParserError"))


def _stub_runner(yaml_path: Path, store: Path, cache: Path) -> str | None:
    return f"ran {store.name}"


def test_run_all_pool(tmp_path):
    jobs = [
        (tmp_path / "a.yaml", tmp_path / "s"),
        (tmp_path / "b.yaml", tmp_path / "s"),
    ]
    errors = bench.run_all(jobs, tmp_path / "cache", workers=2, runner=_stub_runner)
    assert errors == {"a": "ran a.db", "b": "ran b.db"}


def test_drop_failure_does_not_mask_run_error(tmp_path, monkeypatch):
    class Suite:
        name = "x"

    def bad_run(*a, **k):
        raise ValueError("run broke")

    def bad_drop(*a):
        raise ConnectionError("down")

    store = tmp_path / "x.db"
    store.write_bytes(b"")
    monkeypatch.setattr(bench, "load_suite", lambda p: Suite())
    monkeypatch.setattr(bench, "run_suite", bad_run)
    monkeypatch.setattr(bench, "drop_tables", bad_drop)
    err = bench.run_one(tmp_path / "x.yaml", store, tmp_path / "cache")
    assert err == "ValueError: run broke; drop failed: ConnectionError: down"
