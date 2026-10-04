"""One-off refactoring tool: splits a Rust module file into several files by top-level item.

    python scripts/rssplit.py SPEC.json

SPEC.json:
    {
      "source":  "services/analytics/src/recommenders/engine/mod.rs",  # read first, then overwritten
      "outdir":  "services/analytics/src/recommenders/engine",
      "mod_header": "//! doc\\n\\nmod a;\\npub(crate) use a::*;\\n",    # text placed in mod.rs after the source's own `//!` docs
      "code":  [["a.rs", "fn", "first_item"], ["mod.rs", "banner", "text in a banner"], ...],
      "tests": [["a.rs", "fn", "first_test_or_helper"], ...],          # optional
      "glob_super": true,    # new files in the same folder start with `use super::*;`
      "docs": {"a.rs": "one-line purpose"}                             # `//!` line for a new file
    }

Each list is an ordered partition: a segment runs from its marker's item to the next marker's
item. A marker is (kind, name) of a top-level item (fn, struct, enum, impl <Type>, const, type,
mod, ...) or ("banner", substring of a `// ───` banner block). A destination may be a list of
file names, to duplicate a segment (shared test helpers); the name "-" drops it. Chunks before the first marker are the
header: its `use` lines are copied to every new file. Items that were private become `pub(crate)`
in every destination except mod.rs, so siblings can see them. Tests (the body of the file's
`mod tests`) go to a `#[cfg(test)] mod tests { … }` in each destination; a destination named
`tests.rs` receives them as a whole file instead (declare `mod tests;` in mod_header).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ITEM = re.compile(
    r"^ *(?:pub(?:\([a-z]+\))? +)?(?:unsafe +)?(?:async +)?"
    r"(?P<kind>fn|struct|enum|impl|const|static|type|trait|use|mod|thread_local!|macro_rules!)"
    r"\b[ <!]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)?"
)
IMPL_FOR = re.compile(
    r"^ *impl(?:<[^>]*>)? +(?:[A-Za-z0-9_:]+(?:<[^>]*>)? +for +)?(?P<ty>[A-Za-z_][A-Za-z0-9_]*)"
)
PRIVATE_ITEM = re.compile(
    r"^(?P<ind> *)(?P<rest>(?:unsafe +|async +)?(?:fn|struct|enum|const|static|type|trait)\b.*)$"
)


def ident(line: str):
    m = ITEM.match(line)
    if not m:
        return None
    kind, name = m["kind"], m["name"] or ""
    if kind == "impl":
        i = IMPL_FOR.match(line)
        name = i["ty"] if i else name
    return kind, name


def chunks(lines: list[str], indent: int = 0):
    """[(start, end, (kind, name))]: top-level items with the comments / attributes above them;
    each banner block is its own chunk."""
    pad = " " * indent
    out, cur, pending, i, n = [], None, None, 0, len(lines)

    def close(end):
        nonlocal cur
        if cur is not None:
            out.append((cur[0], end, cur[1]))
            cur = None

    while i < n:
        s = lines[i].rstrip("\r\n")
        if not s.strip():
            i += 1
            continue
        col = len(s) - len(s.lstrip(" "))
        if col == indent and s.lstrip().startswith("// ──"):
            close(i)
            j = i
            while (
                j < n
                and lines[j].startswith(pad + "//")
                and not lines[j].startswith((pad + "///", pad + "//!"))
            ):
                j += 1
            text = " ".join(lines[k].strip(" /─\r\n") for k in range(i, j)).strip()
            out.append((i, j, ("banner", text)))
            i, pending = j, None
            continue
        if col == indent and s.lstrip().startswith(("//", "#[", "#!")):
            if pending is None:
                close(i)
                pending = i
            i += 1
            continue
        if col == indent and ident(s):
            start = pending if pending is not None else i
            close(start)
            cur, pending = (start, ident(s)), None
        i += 1
    close(n)
    if out:
        stray = "".join(lines[: out[0][0]]).strip()
        if stray:
            raise SystemExit(
                f"text before the first item is not covered: {stray[:80]!r}"
            )
    # contiguous: lines between items (e.g. a thread_local! block) stay with the item above
    return [
        (a, out[k + 1][0] if k + 1 < len(out) else n, key)
        for k, (a, _, key) in enumerate(out)
    ]


def _find(cs, marker, after):
    kind, name = marker
    for idx in range(after, len(cs)):
        k = cs[idx][2]
        if k[0] == kind and (name in k[1] if kind == "banner" else k[1] == name):
            return idx
    raise SystemExit(f"marker not found (after chunk {after}): {marker}")


def segments(lines, starts, indent=0):
    """({dest: [segment text]}, header `use` text); each chunk after the header lands in
    exactly one segment (or several destinations, for a list of names)."""
    cs = chunks(lines, indent)
    idxs, after = [], 0
    for _, kind, name in starts:
        idxs.append(_find(cs, (kind, name), after))
        after = idxs[-1] + 1
    header = "".join(
        l
        for a, b, key in cs[: idxs[0]]
        if key[0] == "use"
        for l in lines[a:b]
        if not l.startswith(("//!", "#!["))
    )
    header = header.strip("\n") + "\n" if header.strip() else ""
    out: dict[str, list[str]] = {}
    for (dest, _, _), a, b in zip(starts, idxs, idxs[1:] + [len(cs)]):
        text = "".join("".join(lines[s:e]) for s, e, _ in cs[a:b])
        for d in [dest] if isinstance(dest, str) else dest:
            if d != "-":
                out.setdefault(d, []).append(text)
    return out, header


def bump(text: str) -> str:
    """Private top-level items (column 0) become pub(crate)."""
    out = []
    for line in text.splitlines(keepends=True):
        m = PRIVATE_ITEM.match(line.rstrip("\r\n"))
        out.append(f"pub(crate) {line}" if m and not m["ind"] else line)
    return "".join(out)


def join(parts: list[str]) -> str:
    return "\n".join(p.strip("\n") + "\n" for p in parts)


def split(spec: dict) -> dict[str, str]:
    lines = Path(spec["source"]).read_text(encoding="utf-8").splitlines(keepends=True)
    cs = chunks(lines)
    tests = next((c for c in cs if c[2] == ("mod", "tests")), None)
    code_lines = lines[: tests[0]] if tests else lines
    code, use_lines = segments(code_lines, spec["code"])
    files: dict[str, str] = {}
    for dest, parts in code.items():
        body = join(parts)
        if dest == "mod.rs":
            files[dest] = use_lines + "\n" + body
            continue
        sibling = (
            "use super::*;\n"
            if spec.get("glob_super") and not dest.startswith("..")
            else ""
        )
        files[dest] = sibling + use_lines + "\n" + bump(body)
    if tests and "tests" not in spec:
        raise SystemExit(
            'the source has a `mod tests` but the spec has no "tests" list'
        )
    if tests:
        body_lines = lines[tests[0] : tests[1]]
        first = next(
            i
            for i, l in enumerate(body_lines)
            if re.match(r"(pub(\([a-z]+\))? )?mod tests\b", l)
        )
        last = max(i for i, l in enumerate(body_lines) if l.rstrip() == "}")
        by_dest, t_uses = segments(
            body_lines[first + 1 : last], spec["tests"], indent=4
        )
        for dest, parts in by_dest.items():
            body = join(parts)
            if dest.endswith("tests.rs"):
                text = (t_uses + "\n" + body).splitlines(keepends=True)
                files[dest] = "".join(
                    l[4:] if l.startswith("    ") else l for l in text
                )
            else:
                files[dest] = (
                    files.get(dest, "")
                    + "\n#[cfg(test)]\nmod tests {\n"
                    + t_uses
                    + "\n"
                    + body
                    + "}\n"
                )
    for dest, doc in spec.get("docs", {}).items():
        files[dest] = f"//! {doc}\n\n" + files[dest]
    if "mod_header" in spec:
        inner = []
        for l in lines:  # the source's leading `//!` block stays with mod.rs
            if not l.startswith(("//!", "#![")):
                break
            inner.append(l)
        mod = files.get("mod.rs")
        files["mod.rs"] = (
            "".join(inner)
            + ("\n" if inner else "")
            + spec["mod_header"]
            + ("\n" + mod if mod else "")
        )
    return files


def main(argv: list[str]) -> None:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(spec["outdir"])
    for name, text in split(spec).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {out / name} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main(sys.argv)
