"""
Streaming recommender speed: add throughput (rows/s) and finish time, beside one-shot
RecommendRust on the same data. Standalone — the shared harness assumes IMPLEMENTATIONS.

Run: /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_streaming_recommend.py

Results → tests/performance/results/streaming_recommend.parquet (git-ignored).
`peak_mb` is the process's peak memory so far (dataset included), where the OS reports it.
"""

import statistics
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/: datagen

from datagen import describe_narrow, describe_wide  # noqa: E402

from analytics.recommend import RecommendRust, StreamingRecommender  # noqa: E402

DATASETS = {
    "large_dataset": lambda: pl.read_ipc(HERE.parent / "data" / "large_dataset.arrow"),
    "narrow 1M x 4": lambda: describe_narrow(1_000_000),
    "wide 50K x 100": lambda: describe_wide(50_000, 100),
}
BATCH_ROWS = (10_000, 100_000)
RUNS = 3


def peak_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
    except ImportError:
        try:
            import psutil
        except ImportError:
            return None
        return getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20


def stream(frame: pl.DataFrame, batch_rows: int) -> tuple[float, float]:
    rec = StreamingRecommender()
    t0 = time.perf_counter()
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    t1 = time.perf_counter()
    rec.finish()
    return t1 - t0, time.perf_counter() - t1


def main() -> None:
    rows = []
    for name, make in DATASETS.items():
        frame = make()
        one_shot = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            RecommendRust().add({"t": frame}).result()
            one_shot.append(time.perf_counter() - t0)
        for batch_rows in BATCH_ROWS:
            runs = [stream(frame, batch_rows) for _ in range(RUNS)]
            add_s = statistics.median(r[0] for r in runs)
            rows.append(
                {
                    "dataset": name,
                    "rows": frame.height,
                    "cols": frame.width,
                    "batch_rows": batch_rows,
                    "add_s": add_s,
                    "add_rows_per_s": frame.height / add_s,
                    "finish_s": statistics.median(r[1] for r in runs),
                    "one_shot_s": statistics.median(one_shot),
                    "peak_mb": peak_mb(),
                }
            )
            print(rows[-1])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "streaming_recommend.parquet")


if __name__ == "__main__":
    main()
