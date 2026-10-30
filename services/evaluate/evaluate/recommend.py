"""The one-shot or streaming recommender in a child process per dataset (peak memory is per process), with its
cost: wall_s, cpu_s (user + system, all threads), peak_mb above the loaded frame, frame_mb.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl


def _memory_mb() -> tuple[float | None, float | None]:
    """(current resident MiB, peak MiB) of this process, where the OS reports them."""
    current = None
    try:
        import psutil

        info = psutil.Process().memory_info()
        current = info.rss / 2**20
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is not None:
            return current, peak / 2**20
    except ImportError:
        pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return current, rss / 2**20 if sys.platform == "darwin" else rss / 1024
    except ImportError:
        return current, None


BLOCK_ROWS = 65_536  # StreamingRecommender's block_rows default


def _child(frame_path: Path, out_dir: Path, kind: str) -> None:
    from turbo_parakeet.recommenders import OneShotRecommender, StreamingRecommender

    frame = pl.read_ipc(frame_path.read_bytes())
    current, _ = _memory_mb()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    if kind == "streaming":
        rec = StreamingRecommender()
        for block in frame.iter_slices(BLOCK_ROWS):
            rec.add(block)
        result = rec.result()
    else:
        result = OneShotRecommender().add(frame).result()
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    _, peak = _memory_mb()
    result.write_ipc(out_dir / "result.arrow")
    cost = {
        "status": "ok",
        "recommender": kind,
        "rows": frame.height,
        "cols": frame.width,
        "wall_s": wall,
        "cpu_s": cpu,
        "peak_mb": (
            None if peak is None or current is None else max(peak - current, 0.0)
        ),
        "frame_mb": frame.estimated_size() / 2**20,
        "exit_code": 0,
        "error": None,
    }
    (out_dir / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")


def recommend(frame_path: Path, out_dir: Path, kind: str = "oneshot") -> dict:
    """Run the recommender on `frame_path` in a child; return (and write) its cost.json.
    A crash is a `failed` cost row with the exit code and the stderr tail."""
    out_dir = Path(out_dir)
    cost_path = out_dir / "cost.json"
    cost_path.unlink(missing_ok=True)
    (out_dir / "result.arrow").unlink(missing_ok=True)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluate.recommend",
            str(frame_path),
            str(out_dir),
            kind,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if run.returncode == 0 and cost_path.is_file():
        return json.loads(cost_path.read_text(encoding="utf-8"))
    cost = {
        "status": "failed",
        "recommender": kind,
        "rows": None,
        "cols": None,
        "wall_s": None,
        "cpu_s": None,
        "peak_mb": None,
        "frame_mb": None,
        "exit_code": run.returncode,
        "error": (run.stderr or run.stdout)[-2000:] or "no output",
    }
    cost_path.write_text(json.dumps(cost, indent=2), encoding="utf-8")
    return cost


if __name__ == "__main__":
    _child(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
