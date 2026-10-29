import json

import pytest

from evaluate import bench
from evaluate.cli import main


def test_dry_run_needs_no_server(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        bench, "check_server", lambda: pytest.fail("server checked in a dry run")
    )
    assert (
        main(["run", "synth_frame:2000", "--dry-run", "--runs-dir", str(tmp_path)]) == 0
    )
    out = capsys.readouterr().out
    assert "synth_frame:" in out and "table loads" in out
    (run_path,) = tmp_path.iterdir()
    info = json.loads((run_path / "run.json").read_text())
    assert info["datasets"]["synth_frame"]["rows"] == 2000
    assert (run_path / "synth_frame" / "suites" / "suites.json").is_file()


def test_unknown_dataset_exits_2_before_anything(tmp_path, capsys):
    assert main(["run", "nope", "--runs-dir", str(tmp_path)]) == 2
    assert "unknown dataset 'nope'" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_no_server_exits_3_and_keeps_the_steps(tmp_path, capsys, monkeypatch):
    def down():
        raise bench.ServerUnavailable(
            "ClickHouse not reachable; start it with `docker start chbench-ch`"
        )

    monkeypatch.setattr(bench, "check_server", down)
    assert main(["run", "synth_frame:2000", "--runs-dir", str(tmp_path)]) == 3
    assert "docker start chbench-ch" in capsys.readouterr().err
    (run_path,) = tmp_path.iterdir()
    assert (run_path / "synth_frame" / "result.arrow").is_file()


def test_resume_skips_steps_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench,
        "check_server",
        lambda: (_ for _ in ()).throw(bench.ServerUnavailable("x")),
    )
    main(["run", "synth_frame:2000", "--runs-dir", str(tmp_path)])
    (run_path,) = tmp_path.iterdir()
    files = [
        run_path / "synth_frame" / f
        for f in ("frame.arrow", "result.arrow", "cost.json")
    ]
    files.append(run_path / "synth_frame" / "suites" / "suites.json")
    before = [f.stat().st_mtime_ns for f in files]
    main(["run", "synth_frame:2000", "--resume", str(run_path)])
    assert [f.stat().st_mtime_ns for f in files] == before


def test_report_on_a_scored_run(tmp_path, capsys):
    from evaluate.score import score_run
    from evaluate.summary import summarise
    from tests.test_score_run import _run_dir

    run = _run_dir(tmp_path)
    summarise(run, score_run(run))
    assert main(["report", str(run.path)]) == 0
    assert "columns 1" in capsys.readouterr().out


def test_a_failing_dataset_does_not_stop_the_others(tmp_path, capsys, monkeypatch):
    import evaluate.run as run_mod

    real = run_mod.build

    def build(name, *a, **k):
        if name == "synth_frame":
            raise RuntimeError("boom")
        return real(name, *a, **k)

    monkeypatch.setattr(run_mod, "build", build)
    args = ["run", "synth_frame:2000", "gcd:2000", "--dry-run"]
    assert main([*args, "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "synth_frame: failed (RuntimeError: boom)" in out and "gcd:" in out
    (run_path,) = tmp_path.iterdir()
    datasets = json.loads((run_path / "run.json").read_text())["datasets"]
    assert datasets["synth_frame"] == {
        **datasets["synth_frame"],
        "status": "failed",
        "error": "RuntimeError: boom",
    }
    assert datasets["gcd"]["status"] == "prepared"
    # resume after the fix: the stale error is dropped
    monkeypatch.setattr(run_mod, "build", real)
    assert (
        main(["run", "synth_frame:2000", "--dry-run", "--resume", str(run_path)]) == 0
    )
    entry = json.loads((run_path / "run.json").read_text())["datasets"]["synth_frame"]
    assert entry["status"] == "prepared" and "error" not in entry
    assert not list((run_path / "synth_frame").glob("*.tmp"))


@pytest.mark.parametrize(
    "cmd",
    [
        ["report", "{missing}"],
        ["compare", "{missing}", "{missing}"],
        ["run", "synth_frame:2000", "--resume", "{missing}"],
    ],
)
def test_missing_run_dir_exits_2(tmp_path, capsys, cmd):
    missing = str(tmp_path / "nope")
    assert main([a.format(missing=missing) for a in cmd]) == 2
    assert "is not a run directory" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_workers_default_to_one():
    from evaluate.cli import _parser

    args = _parser().parse_args(["run", "gcd"])
    assert args.workers == 1
