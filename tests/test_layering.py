"""scripts/check_layering.py: the Rust crate's dependency rule (bindings -> recommenders ->
techniques -> common), on synthetic trees and on the real one."""

import importlib.util
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
