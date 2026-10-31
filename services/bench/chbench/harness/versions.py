"""The engine versions the calibrations last ran on (`docs/engines/versions.yaml`, the single source of
truth) and the drift between them and the versions a run actually uses. A drift is a warning, never a
refusal: the store records the actual versions either way (ToDo 6)."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import yaml

# chbench/harness/versions.py -> the repository root is four levels up.
PINNED_PATH = Path(__file__).resolve().parents[4] / "docs" / "engines" / "versions.yaml"
_HINT = "(docs/engines/versions.yaml; after an upgrade see docs/engines/README.md)"


def load_pinned(path: str | Path | None = None) -> dict:
    """The pinned versions; {} when the file is absent (an installed chbench outside the
    repository) or not a mapping."""
    p = Path(path) if path is not None else PINNED_PATH
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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


def installed() -> dict[str, str | None]:
    """The installed Polars, DataFusion and pyarrow versions (None for one not installed) and
    Python's, read from the distributions' metadata: neither engine is imported, so a run can
    record and compare them before its worker processes start (each with its thread count).
    """
    out: dict[str, str | None] = {}
    for package in ("polars", "datafusion", "pyarrow"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    out["python"] = platform.python_version()
    return out
