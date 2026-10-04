"""Layering check for the Rust crate: bindings -> recommenders -> techniques -> common.

    python scripts/check_layering.py

A file may import (`crate::<layer>::…`) its own layer and the layers below it, never above:
`common` imports only `common`; `techniques` only `common` / `techniques`; `recommenders` those and
`recommenders`; `bindings` anything. Every `crate::<name>` must name a layer (no flat modules
at the top of `src/`). Comments are ignored. Spec: docs/superpowers/specs/
2026-10-04-rust-module-structure-design.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "services" / "analytics" / "src"
LAYERS = ("common", "techniques", "recommenders", "bindings")  # low -> high
CRATE_PATH = re.compile(r"\bcrate::([A-Za-z_][A-Za-z0-9_]*)")


def violations(src: Path = SRC) -> list[str]:
    """One message per offending `crate::…` path; files directly in `src/` (lib.rs) are exempt."""
    out = []
    for path in sorted(src.rglob("*.rs")):
        rel = path.relative_to(src)
        if len(rel.parts) == 1:
            continue
        layer = rel.parts[0]
        if layer not in LAYERS:
            out.append(f"{rel.as_posix()}: not inside one of {', '.join(LAYERS)}")
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in CRATE_PATH.finditer(re.sub(r"//.*", "", line)):
                target = m[1]
                where = f"{rel.as_posix()}:{n}"
                if target not in LAYERS:
                    out.append(
                        f"{where}: crate::{target} is not a layer ({', '.join(LAYERS)})"
                    )
                elif LAYERS.index(target) > LAYERS.index(layer):
                    out.append(
                        f"{where}: {layer} must not import {target} (crate::{target})"
                    )
    return out


def main() -> None:
    found = violations()
    for v in found:
        print(v)
    if found:
        print(f"\n{len(found)} layering violation(s)")
        sys.exit(1)
    print("layering ok")


if __name__ == "__main__":
    main()
