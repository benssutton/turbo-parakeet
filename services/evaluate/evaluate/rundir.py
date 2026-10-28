"""One directory per run: runs/<YYYYmmdd-HHMMSS>/ holding run.json and, per dataset,
frame.arrow, result.arrow, cost.json, suites/ and stores/; the score files at the top.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository
DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "runs"


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    try:
        out["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_sha"] = None
    for pkg in (
        "turbo_parakeet",
        "synthdata",
        "chbench",
        "evaluate",
        "polars",
        "pyarrow",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.run_id = self.path.name

    @classmethod
    def new(cls, root: Path = DEFAULT_RUNS) -> "RunDir":
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(root) / stamp
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> "RunDir":
        if not (Path(path) / "run.json").is_file():
            raise FileNotFoundError(f"{path} is not a run directory (no run.json)")
        return cls(path)

    def dataset(self, name: str) -> Path:
        d = self.path / name
        d.mkdir(exist_ok=True)
        return d

    def frame(self, name: str) -> Path:
        return self.dataset(name) / "frame.arrow"

    def result(self, name: str) -> Path:
        return self.dataset(name) / "result.arrow"

    def cost(self, name: str) -> Path:
        return self.dataset(name) / "cost.json"

    def suites(self, name: str) -> Path:
        return self.dataset(name) / "suites"

    def stores(self, name: str) -> Path:
        return self.dataset(name) / "stores"

    def info(self) -> dict:
        return json.loads((self.path / "run.json").read_text(encoding="utf-8"))

    def write_info(self, info: dict) -> None:
        (self.path / "run.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )

    def update_dataset(self, name: str, **fields) -> None:
        info = self.info()
        info.setdefault("datasets", {})[name] = fields
        self.write_info(info)
