"""scripts/check_layering.py: the Rust crate's dependency rule (bindings -> recommenders ->
techniques -> common), on synthetic trees and on the real one.
Also: no Parquet anywhere (CLAUDE.md § Project Objectives, Formats)."""

import importlib.util
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_layering", HERE.parent / "scripts" / "check_layering.py"
)
layering = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layering)


def tree(tmp_path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_downward_and_same_layer_imports_are_allowed(tmp_path):
    src = tree(
        tmp_path,
        {
            "lib.rs": "mod common;\nuse crate::bindings::api;\n",  # exempt
            "common/error.rs": "pub struct Error;\n",
            "common/text.rs": "use crate::common::error::Error;\n",
            "techniques/gcd.rs": "use crate::common::text::x;\nuse crate::techniques::hll;\n",
            "recommenders/engine/mod.rs": "use crate::techniques::describe;\nuse crate::common::text;\n",
            "bindings/api/mod.rs": "use crate::recommenders::oneshot;\nuse crate::common::error;\n",
        },
    )
    assert layering.violations(src) == []


def test_upward_imports_are_reported_with_file_and_line(tmp_path):
    src = tree(
        tmp_path,
        {
            "common/text.rs": "fn ok() {}\nuse crate::techniques::describe::x;\n",
            "techniques/describe/mod.rs": "let a = crate::recommenders::engine::canon(s);\n",
            "recommenders/oneshot.rs": "use crate::bindings::api::Error;\n",
        },
    )
    found = "\n".join(layering.violations(src))
    assert found.count("must not import") == 3
    assert "common/text.rs:2: common must not import techniques" in found
    assert (
        "techniques/describe/mod.rs:1: techniques must not import recommenders" in found
    )
    assert "recommenders/oneshot.rs:1: recommenders must not import bindings" in found


def test_flat_top_level_modules_and_stray_folders_are_reported(tmp_path):
    src = tree(
        tmp_path,
        {
            "techniques/gcd.rs": "use crate::shared::encode_series;\n",
            "stray/x.rs": "fn f() {}\n",
        },
    )
    found = layering.violations(src)
    assert any("crate::shared is not a layer" in v for v in found)
    assert any("stray/x.rs: not inside one of" in v for v in found)


def test_comments_are_ignored(tmp_path):
    src = tree(
        tmp_path,
        {
            "common/a.rs": "// see crate::techniques::describe\n/// uses crate::bindings\nfn f() {}\n"
        },
    )
    assert layering.violations(src) == []


def test_the_real_tree_has_no_violations():
    assert layering.violations() == []


ROOT = HERE.parent
PARQUET = re.compile(
    r"read_parquet|write_parquet|scan_parquet|sink_parquet|pyarrow\.parquet|\.parquet[\"']"
)


def parquet_uses(paths: list[Path]) -> list[str]:
    """`file:line` for every line that reads or writes Parquet."""
    return [
        f"{p.name}:{i}"
        for p in paths
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if PARQUET.search(line)
    ]


def test_parquet_uses_are_found(tmp_path):
    files = tree(
        tmp_path,
        {
            "a.py": 'df.write_parquet(out)\nprint("no Parquet here")\n',
            "b.py": 'import pyarrow.parquet as pq\nx = "r.parquet"\n',
        },
    )
    assert parquet_uses(sorted(files.glob("*.py"))) == ["a.py:1", "b.py:1", "b.py:2"]


def test_no_python_file_uses_parquet():
    tracked = subprocess.run(
        ["git", "ls-files", "--", "tests/*.py", "services/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = [ROOT / t for t in tracked if ROOT / t != Path(__file__).resolve()]
    assert paths and parquet_uses(paths) == []


def test_polars_crate_builds_without_parquet():
    cargo = (ROOT / "services" / "analytics" / "Cargo.toml").read_text(encoding="utf-8")
    polars = next(line for line in cargo.splitlines() if line.startswith("polars ="))
    assert "default-features = false" in polars
    assert "parquet" not in cargo
