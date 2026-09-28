"""
Shared speed-benchmark harness. Every tests/performance/benchmark_<technique>.py is:

    from harness import Dataset, large_dataset, run
    from datagen import ...
    if __name__ == "__main__":
        run("analytics.<technique>", [large_dataset(), Dataset("narrow …", …), Dataset("wide …", …)])

What is timed: Impl(**params).add(frames).result() — the whole analytical question,
fresh instance per run (so caches never carry over), 1 warm-up + `runs` timed runs,
median and min reported. An implementation whose warm-up exceeds `budget_s` is
reported "skipped: over budget" and not timed.

Hypothesis split: each *Rust implementation is also timed at 1 thread in a child
process (RAYON_NUM_THREADS=1, POLARS_MAX_THREADS=1 — both pools are fixed at process
start), giving
    algorithmic = fastest non-Rust ÷ Rust@1 thread
    parallel    = Rust@1 thread   ÷ Rust@N threads
    total       = fastest non-Rust ÷ Rust@N threads

Sanity check (never a gate): each result is compared with the reference via
impl.agreement(); the pytest suite is the correctness gate.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

sys.path.insert(
    0, str(Path(__file__).parents[1])
)  # tests/ -> `import datagen` in benchmark scripts

import analytics  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
LARGE_DATASET = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
SINGLE_THREAD_ENV = {"RAYON_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
_CHILD_ENV = "ANALYTICS_BENCH_CHILD"


@dataclass(frozen=True)
class Dataset:
    name: str
    make: Callable[[], dict[str, pl.DataFrame]]
    exclude: tuple[
        str, ...
    ] = ()  # implementations not run on this dataset (e.g. would exhaust memory)


def large_dataset(columns: int | None = None, exclude: tuple[str, ...] = ()) -> Dataset:
    """tests/data/large_dataset.arrow (50K rows × 101 cols), optionally only its first `columns`."""
    label = "large_dataset.arrow" + (f" (first {columns} cols)" if columns else "")

    def make() -> dict[str, pl.DataFrame]:
        df = pl.read_ipc(LARGE_DATASET)
        return {"large": df.select(df.columns[:columns]) if columns else df}

    return Dataset(label, make, exclude)


def time_call(
    fn: Callable[[], object], runs: int, budget_s: float
) -> tuple[list[float] | None, object]:
    t0 = time.perf_counter()
    result = fn()
    if time.perf_counter() - t0 > budget_s:
        return None, None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def speedups(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        ok = [r for r in rows if r["dataset"] == dataset and r["status"] == "ok"]
        others = [
            r["median_ms"] for r in ok if not r["implementation"].endswith("Rust")
        ]
        best = min(others) if others else None
        for impl in dict.fromkeys(
            r["implementation"] for r in ok if r["implementation"].endswith("Rust")
        ):
            n = next(
                (
                    r["median_ms"]
                    for r in ok
                    if r["implementation"] == impl and r["threads"] == "N"
                ),
                None,
            )
            one = next(
                (
                    r["median_ms"]
                    for r in ok
                    if r["implementation"] == impl and r["threads"] == "1"
                ),
                None,
            )
            out.append(
                {
                    "dataset": dataset,
                    "implementation": impl,
                    "algorithmic": best / one if best and one else None,
                    "parallel": one / n if one and n else None,
                    "total": best / n if best and n else None,
                }
            )
    return out


def _load(module, name: str):
    try:
        return getattr(module, name)
    except ImportError as exc:
        print(f"  {name}: unavailable ({exc})", file=sys.stderr)
        return None


def _measure(
    cls, frames, params: dict, runs: int, budget_s: float
) -> tuple[dict, object]:
    times, result = time_call(
        lambda: cls(**params).add(frames).result(), runs, budget_s
    )
    if times is None:
        return {
            "status": f"skipped: over budget ({budget_s:.0f}s)",
            "median_ms": None,
            "min_ms": None,
        }, None
    return {
        "status": "ok",
        "median_ms": statistics.median(times) * 1e3,
        "min_ms": min(times) * 1e3,
    }, result


def run(
    package: str,
    datasets: list[Dataset],
    params: dict | None = None,
    impl_params: dict[str, dict] | None = None,
    runs: int = 5,
    budget_s: float = 60.0,
) -> pl.DataFrame | None:
    params, impl_params = params or {}, impl_params or {}
    module = importlib.import_module(package)
    names = [module.REFERENCE] + [
        n for n in module.IMPLEMENTATIONS if n != module.REFERENCE
    ]
    classes = {n: c for n in names if (c := _load(module, n)) is not None}
    rust = [n for n in classes if n.endswith("Rust")]
    child = os.environ.get(_CHILD_ENV) == "1"

    rows: list[dict] = []
    for ds in datasets:
        frames = ds.make()
        ref_result = None
        for name in rust if child else classes:
            base = {
                "dataset": ds.name,
                "implementation": name,
                "threads": "N" if name in rust else "default",
            }
            if name in ds.exclude:
                rows.append(
                    {
                        **base,
                        "status": "excluded",
                        "median_ms": None,
                        "min_ms": None,
                        "agrees": None,
                    }
                )
                continue
            print(f"  {ds.name}: {name} …", file=sys.stderr, flush=True)
            p = {**params, **impl_params.get(name, {})}
            measured, result = _measure(classes[name], frames, p, runs, budget_s)
            agrees = None
            if not child:
                if name == module.REFERENCE:
                    ref_result = result
                elif result is not None and ref_result is not None:
                    agrees = not classes[name](**p).agreement(result, ref_result)
            rows.append({**base, **measured, "agrees": agrees})

    if child:
        print(json.dumps(rows))
        return None
    if rust:
        rows += _single_thread_rows()
    return _report(package, rows)


def _single_thread_rows() -> list[dict]:
    print(
        "  re-running Rust implementations at 1 thread …", file=sys.stderr, flush=True
    )
    proc = subprocess.run(
        [sys.executable, *sys.argv],
        env={**os.environ, **SINGLE_THREAD_ENV, _CHILD_ENV: "1"},
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    rows = json.loads(proc.stdout.strip().splitlines()[-1])
    return [{**r, "threads": "1", "agrees": None} for r in rows]


def _fmt(x, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def _report(package: str, rows: list[dict]) -> pl.DataFrame:
    technique = package.rsplit(".", 1)[-1]
    print(
        f"\n{technique} — median / min of timed runs, fresh instance per run, {os.cpu_count()} CPUs"
    )
    print(
        f"{'dataset':<36} {'implementation':<28} {'threads':>7} {'median ms':>11} {'min ms':>11} {'agrees':>7}  status"
    )
    for r in rows:
        agrees = {True: "yes", False: "NO", None: "—"}[r["agrees"]]
        print(
            f"{r['dataset']:<36} {r['implementation']:<28} {r['threads']:>7} "
            f"{_fmt(r['median_ms'], '11.2f')} {_fmt(r['min_ms'], '11.2f')} {agrees:>7}  {r['status']}"
        )
    print(
        f"\n{'dataset':<36} {'Rust implementation':<28} {'algorithmic':>12} {'parallel':>9} {'total':>9}"
    )
    for s in speedups(rows):
        print(
            f"{s['dataset']:<36} {s['implementation']:<28} "
            f"{_fmt(s['algorithmic'], '11.1f')}x {_fmt(s['parallel'], '8.1f')}x {_fmt(s['total'], '8.1f')}x"
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        technique=pl.lit(technique),
        cpu_count=pl.lit(os.cpu_count()),
        processor=pl.lit(platform.processor()),
        polars=pl.lit(pl.__version__),
        numpy=pl.lit(np.__version__),
        analytics=pl.lit(analytics.__version__),
        timestamp=pl.lit(stamp),
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{technique}_{stamp}.parquet"
    frame.write_parquet(path)
    print(f"\nsaved {path}")
    return frame
