"""
Streaming entropy past the exact phase: the bottom-k sample estimate vs the AMS sketch
(the default rule: `sample` unless its error exceeds 5% on a shape where `ams` stays within
5%).

Per shape and estimator, in a child process (peak memory is per process): one column of
10^6 rows streamed in 100 000-row batches through StreamingRecommender; reported: estimate,
interval, relative error and coverage against the exact entropy (one-shot), add() seconds,
peak working set above the loaded data.

Run: python tests/performance/benchmark_streaming_entropy.py
Results -> tests/performance/results/streaming_entropy.arrow
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # harness

from synthdata.synth import synth_frame  # noqa: E402
from harness import memory_mb  # noqa: E402

from turbo_parakeet.recommenders import (  # noqa: E402
    OneShotRecommender,
    StreamingRecommender,
)

RESULTS_DIR = HERE / "results"
ROWS = int(os.environ.get("BENCH_ROWS", 1_000_000))
BATCH = max(ROWS // 10, 1)


def zipf(alpha: float, d: int = 1_000_000) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    w = 1.0 / np.arange(1, d + 1) ** alpha
    return pl.DataFrame({"x": rng.choice(d, size=ROWS, p=w / w.sum())})


SHAPES = {
    "uniform 1e4": lambda: pl.DataFrame(
        {"x": np.random.default_rng(0).integers(0, 10_000, ROWS)}
    ),
    "uniform 1e5": lambda: pl.DataFrame(
        {"x": np.random.default_rng(0).integers(0, 100_000, ROWS)}
    ),
    "uniform 1e6": lambda: pl.DataFrame(
        {"x": np.random.default_rng(0).integers(0, 1_000_000, ROWS)}
    ),
    "zipf 0.8": lambda: zipf(0.8),
    "zipf 1.1": lambda: zipf(1.1),
    "zipf 1.5": lambda: zipf(1.5),
    "near-constant": lambda: pl.DataFrame(
        {
            "x": np.where(
                np.random.default_rng(2).random(ROWS) < 0.999, 0, np.arange(ROWS)
            )
        }
    ),
    "synth_frame 50K": lambda: synth_frame(50_000),
}
if os.environ.get("BENCH_SHAPES"):
    SHAPES = {
        k: v for k, v in SHAPES.items() if k in os.environ["BENCH_SHAPES"].split(",")
    }


def child(shape: str, estimator: str) -> None:
    frame = SHAPES[shape]()
    gc.collect()
    current_before, _ = memory_mb()
    rec = StreamingRecommender(entropy_estimator=estimator, reservoir_rows=0)
    t0 = time.perf_counter()
    for off in range(0, frame.height, BATCH):
        rec.add(frame.slice(off, BATCH))
    add_s = time.perf_counter() - t0
    out = rec.result().select(
        "column", "entropy", "entropy_low", "entropy_high", "entropy_method"
    )
    _, peak = memory_mb()
    print(
        json.dumps(
            {"add_s": add_s, "peak_mb": peak - current_before, "rows": out.to_dicts()}
        )
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for shape, make in SHAPES.items():
        frame = make()
        exact = {
            r["column"]: r["entropy"]
            for r in OneShotRecommender().add(frame).result().iter_rows(named=True)
        }
        for estimator in ("sample", "ams"):
            print(f"  {shape}: {estimator} ...", file=sys.stderr, flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", shape, estimator],
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
            )
            info = json.loads(proc.stdout.strip().splitlines()[-1])
            for r in info["rows"]:
                h = exact[r["column"]]
                if r["entropy_method"] == "exact":
                    continue  # never overflowed: nothing estimated
                rows.append(
                    {
                        "shape": shape,
                        "column": r["column"],
                        "estimator": estimator,
                        "exact": h,
                        "estimate": r["entropy"],
                        "low": r["entropy_low"],
                        "high": r["entropy_high"],
                        "rel_error": abs(r["entropy"] - h) / h if h else 0.0,
                        "covered": r["entropy_low"] <= h <= r["entropy_high"],
                        "add_s": info["add_s"],
                        "peak_mb": info["peak_mb"],
                    }
                )
    df = pl.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_ipc(RESULTS_DIR / "streaming_entropy.arrow", compression="zstd")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(
            df.group_by("shape", "estimator")
            .agg(
                pl.col("rel_error").max().alias("max_rel_error"),
                pl.col("covered").mean().alias("coverage"),
                pl.col("add_s").first(),
                pl.col("peak_mb").first(),
            )
            .sort("shape", "estimator")
        )
    seen = set(df["shape"].unique()) if df.height else set()
    print("never overflowed:", [s for s in SHAPES if s not in seen])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
    else:
        main()
