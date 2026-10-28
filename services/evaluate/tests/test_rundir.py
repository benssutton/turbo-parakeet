import json

from evaluate.rundir import RunDir, versions


def test_new_run_creates_a_timestamped_directory(tmp_path):
    run = RunDir.new(tmp_path)
    assert run.path.parent == tmp_path and run.path.is_dir()
    assert run.run_id == run.path.name
    assert len(run.run_id) == len("20261028-101530")


def test_dataset_layout(tmp_path):
    run = RunDir.new(tmp_path)
    d = run.dataset("gcd")
    assert d == run.path / "gcd" and d.is_dir()
    assert run.frame("gcd") == d / "frame.arrow"
    assert run.result("gcd") == d / "result.arrow"
    assert run.cost("gcd") == d / "cost.json"
    assert run.suites("gcd") == d / "suites"
    assert run.stores("gcd") == d / "stores"


def test_run_json_round_trip_and_update(tmp_path):
    run = RunDir.new(tmp_path)
    run.write_info({"datasets": {"gcd": {"status": "pending"}}})
    run.update_dataset("gcd", status="ok", warnings=["w"])
    info = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
    assert info["datasets"]["gcd"] == {"status": "ok", "warnings": ["w"]}
    assert RunDir.open(run.path).info() == info


def test_versions_names_the_packages():
    v = versions()
    assert {
        "git_sha",
        "turbo_parakeet",
        "synthdata",
        "chbench",
        "polars",
        "pyarrow",
    } <= set(v)
