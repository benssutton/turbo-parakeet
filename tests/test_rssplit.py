"""scripts/rssplit.py: the refactoring tool that splits a Rust module by item."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "rssplit", Path(__file__).resolve().parent.parent / "scripts" / "rssplit.py"
)
rssplit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rssplit)

SOURCE = """// ───────────────────────────
// demo — a module
// ───────────────────────────

use std::fmt;
use crate::x::{a, b};

// ── part one ──────────────────
/// Documented.
pub(crate) fn alpha() -> u8 { 1 }

fn beta() -> u8 {
    alpha() + 1
}

// ── part two ──────────────────
#[derive(Debug)]
struct Gamma {
    v: u8,
}

impl Gamma {
    fn new() -> Self { Gamma { v: 3 } }
}

const LIMIT: usize = 4;

fn tail() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn helper() -> u8 { 9 }

    #[test]
    fn alpha_is_one() {
        assert_eq!(alpha(), 1);
    }

    #[test]
    fn gamma_is_three() {
        assert_eq!(Gamma::new().v, 3);
    }
}
"""

LOSSY = """use std::fmt;

// ── one ──────────────────
thread_local! {
    static X: u8 = 1;
}

pub(crate) fn f() {}

// ── two ──────────────────
fn g() {}

// ── dropme ──────────────────
"""


def run(tmp_path, spec_extra, source=SOURCE):
    src = tmp_path / "mod.rs"
    src.write_text(source, encoding="utf-8")
    spec = {"source": str(src), "outdir": str(tmp_path / "out"), **spec_extra}
    return rssplit.split(spec)


def test_items_are_chunked_with_their_docs():
    lines = SOURCE.splitlines(keepends=True)
    cs = rssplit.chunks(lines)
    keys = [k for _, _, k in cs]
    assert ("fn", "alpha") in keys and ("struct", "Gamma") in keys
    assert ("impl", "Gamma") in keys and ("mod", "tests") in keys
    assert any(k[0] == "banner" and "part one" in k[1] for k in keys)
    start = next(s for s, _, k in cs if k == ("fn", "alpha"))
    assert lines[start].startswith("/// Documented.")


def test_split_assigns_every_item_once_and_bumps_visibility(tmp_path):
    files = run(
        tmp_path,
        {
            "mod_header": "mod one;\nmod two;\n",
            "glob_super": True,
            "code": [
                ["one.rs", "banner", "part one"],
                ["two.rs", "banner", "part two"],
                ["../up.rs", "fn", "tail"],
            ],
            "tests": [
                [["one.rs", "two.rs"], "fn", "helper"],
                ["one.rs", "fn", "alpha_is_one"],
                ["two.rs", "fn", "gamma_is_three"],
            ],
        },
    )
    assert set(files) == {"one.rs", "two.rs", "mod.rs", "../up.rs"}
    one, two = files["one.rs"], files["two.rs"]
    assert one.startswith("use super::*;\n")
    assert not files["../up.rs"].startswith("use super::*;")  # outside the folder
    assert "use std::fmt;" in one and "use crate::x::{a, b};" in two
    assert "pub(crate) fn alpha" in one and "pub(crate) fn beta" in one
    assert "pub(crate) struct Gamma" in two and "pub(crate) const LIMIT" in two
    assert "    fn new()" in two  # nested items are not bumped
    assert "alpha_is_one" in one and "gamma_is_three" not in one
    assert "gamma_is_three" in two and "alpha_is_one" not in two
    assert "fn helper" in one and "fn helper" in two  # shared helper duplicated
    assert one.count("use std::collections::HashMap;") == 1
    assert files["mod.rs"] == "mod one;\nmod two;\n"  # no code stays in mod.rs
    everything = "".join(files.values())
    for needle in (
        "fn alpha()",
        "fn beta",
        "struct Gamma",
        "impl Gamma",
        "const LIMIT",
    ):
        assert everything.count(needle) == 1, needle


def test_mod_rs_keeps_its_code_and_the_header_uses(tmp_path):
    files = run(
        tmp_path,
        {
            "mod_header": "mod one;\n",
            "code": [
                ["one.rs", "banner", "part one"],
                ["mod.rs", "banner", "part two"],
            ],
            "tests": [["one.rs", "fn", "helper"]],
        },
    )
    mod = files["mod.rs"]
    assert mod.startswith("mod one;\n") and "use std::fmt;" in mod
    assert "struct Gamma" in mod and "pub(crate) struct" not in mod  # not bumped


def test_a_tests_file_gets_the_module_body_dedented(tmp_path):
    files = run(
        tmp_path,
        {
            "mod_header": "#[cfg(test)]\nmod tests;\n",
            "code": [["mod.rs", "banner", "part one"]],
            "tests": [["tests.rs", "fn", "helper"]],
        },
    )
    text = files["tests.rs"]
    assert text.startswith("use super::*;")
    assert "\nfn helper()" in text and "mod tests" not in text
    assert "#[test]\nfn alpha_is_one" in text


def test_unknown_marker_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, {"code": [["a.rs", "fn", "nope"]], "tests": []})


def test_no_line_is_lost_and_banners_can_be_dropped(tmp_path):
    files = run(
        tmp_path,
        {
            "code": [
                ["a.rs", "banner", "one"],
                ["b.rs", "banner", "two"],
                ["-", "banner", "dropme"],
            ],
        },
        source=LOSSY,
    )
    assert "static X: u8 = 1;" in files["a.rs"]  # kept with the banner above it
    assert "dropme" not in "".join(files.values())
    assert "fn g()" in files["b.rs"] and "fn f()" in files["a.rs"]


def test_a_tests_module_without_a_tests_spec_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="tests"):
        run(tmp_path, {"code": [["a.rs", "banner", "part one"]]})


def test_rs_paths_rewrites_old_module_paths_in_one_pass():
    spec = importlib.util.spec_from_file_location(
        "rs_paths", Path(__file__).resolve().parent.parent / "scripts" / "rs_paths.py"
    )
    rs_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs_paths)
    text = (
        "use crate::recommend::{a, b};\n"
        "use crate::conclusions::conclude;\n"
        "let x = crate::describe::f(crate::shared::g());\n"
        "use crate::common::text::canon;\n"  # already new: untouched
        "use crate::recommenders::engine::h;\n"  # prefix of an old name: untouched
        "use crate::streaming::holds_nested_null;\n"
        "use crate::partial::LevelStats;\n"
    )
    assert rs_paths.rewrite(text) == (
        "use crate::recommenders::engine::{a, b};\n"
        "use crate::techniques::describe::conclusions::conclude;\n"
        "let x = crate::techniques::describe::f(crate::common::encode::g());\n"
        "use crate::common::text::canon;\n"
        "use crate::recommenders::engine::h;\n"
        "use crate::recommenders::streaming::holds_nested_null;\n"
        "use crate::recommenders::streaming::partial::LevelStats;\n"
    )


DOCS = """//! The original module docs.
//! Second line.

use std::fmt;

// ── one ──────────────────
pub(crate) fn f() {}

// ── two ──────────────────
fn g() {}
"""


def test_inner_docs_stay_on_mod_rs_and_new_files_get_their_own(tmp_path):
    files = run(
        tmp_path,
        {
            "mod_header": "mod a;\nmod b;\n",
            "glob_super": True,
            "docs": {"a.rs": "Part one."},
            "code": [["a.rs", "banner", "one"], ["b.rs", "banner", "two"]],
        },
        source=DOCS,
    )
    assert files["mod.rs"].startswith(
        "//! The original module docs.\n//! Second line.\n\nmod a;"
    )
    assert files["a.rs"].startswith("//! Part one.\n\nuse super::*;\nuse std::fmt;")
    assert "//! The original" not in files["b.rs"]
    assert files["b.rs"].startswith("use super::*;\nuse std::fmt;")  # no docs requested
