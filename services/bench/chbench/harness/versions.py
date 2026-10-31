"""The engine versions the calibrations last ran on (`docs/engines/versions.yaml`, the single source of
truth) and the drift between them and the versions a run actually uses. A drift is a warning, never a
refusal: the store records the actual versions either way (ToDo 6)."""

from __future__ import annotations

from pathlib import Path

import yaml

# chbench/harness/versions.py -> the repository root is four levels up.
PINNED_PATH = Path(__file__).resolve().parents[4] / "docs" / "engines" / "versions.yaml"
_HINT = "(docs/engines/versions.yaml; after an upgrade see docs/engines/README.md)"


def load_pinned(path: str | Path | None = None) -> dict:
    """The pinned versions; {} when the file is absent (an installed chbench outside the repository)."""
    p = Path(path) if path is not None else PINNED_PATH
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _pinned_version(pinned: dict, engine: str) -> str | None:
    entry = pinned.get(engine)
    if isinstance(entry, dict):
        entry = entry.get("version")
    return None if entry is None else str(entry)


def drift(pinned: dict, actual: dict[str, str | None]) -> list[str]:
    """One warning per engine in `actual` whose version differs from its pin, in `actual`'s order.
    Engines without a pin, or with no actual version, are not compared."""
    out = []
    for engine, version in actual.items():
        want = _pinned_version(pinned, engine)
        if want is None or version is None or str(version) == want:
            continue
        out.append(f"{engine} {version} differs from the pinned {want} {_HINT}")
    return out
