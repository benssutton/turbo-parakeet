"""One-off refactoring tool: rewrites `crate::<old module>` paths to the new folder layout.

    python scripts/rs_paths.py            # rewrites every .rs under services/analytics/src

Single pass, so a rewritten path is never rewritten again. Paths whose first segment is not an
old module name (common, techniques, recommenders, bindings, …) are left alone.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "services" / "analytics" / "src"

MAP = {
    "arrow_io": "common::arrow_io",
    "shared": "common::encode",
    "sizes": "common::ipc_sizes",
    "gcd": "techniques::gcd",
    "hll": "techniques::hll",
    "cardinality_estimators": "techniques::cardinality_estimators",
    "describe": "techniques::describe",
    "conclusions": "techniques::describe::conclusions",
    "bloomfilter": "techniques::bloomfilter",
    "minhash": "techniques::minhash",
    "chi_squared": "techniques::chi_squared",
    "ari": "techniques::ari",
    "contingency": "techniques::contingency",
    "entropy": "techniques::joint_entropy",
    "recommend": "recommenders::engine",
    "oneshot": "recommenders::oneshot",
    "streaming": "recommenders::streaming",
    "partial": "recommenders::streaming::partial",
    "reservoir": "recommenders::streaming::reservoir",
    "distinct_sample": "recommenders::streaming::distinct_sample",
    "api": "bindings::api",
}
PATH = re.compile(r"\bcrate::([A-Za-z_][A-Za-z0-9_]*)\b")


def rewrite(text: str) -> str:
    return PATH.sub(lambda m: "crate::" + MAP.get(m[1], m[1]), text)


def main() -> None:
    changed = 0
    for path in sorted(SRC.rglob("*.rs")):
        old = path.read_text(encoding="utf-8")
        new = rewrite(old)
        if new != old:
            path.write_text(new, encoding="utf-8", newline="\n")
            changed += 1
    print(f"rewrote paths in {changed} files")


if __name__ == "__main__":
    main()
