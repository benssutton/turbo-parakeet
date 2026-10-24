"""
Shared benchmark harness: speed, peak memory and accuracy of every implementation of one
technique (spec docs/superpowers/specs/2026-10-24-benchmark-uniformity-design.md). Every
tests/performance/benchmark_<technique>.py is:

    from harness import Dataset, large_dataset, run
    from datagen import ...
    if __name__ == "__main__":
        run("turbo_parakeet.techniques.<technique>", [large_dataset(), Dataset("narrow …", …), Dataset("wide …", …)])

One child process per measurement, i.e. per (dataset, implementation, variant, threads),
reading the dataset from Arrow IPC files the parent wrote once. The child (this file,
`--child spec.json`) collects garbage, notes its working set (`load_mb`), runs the warm-up
Impl(**params).add(frames).result() and reports `peak_mb`, the peak working set above
`load_mb` (the OS peak never resets, so only a fresh process gives a clean figure;
`peak_is_lower_bound` when the warm-up never exceeded the load-time peak), then times `runs`
runs (fresh instance each) → median / min. A warm-up over `budget_s` is "skipped: over
budget". Params cross as JSON (tuples arrive as lists).

Rust implementations run at N threads and at 1 (RAYON_NUM_THREADS=1, POLARS_MAX_THREADS=1 —
both pools are fixed at process start):
    algorithmic = fastest non-Rust ÷ Rust@1 thread
    parallel    = Rust@1 thread   ÷ Rust@N threads
    total       = fastest non-Rust ÷ Rust@N threads

Accuracy (never a gate; the pytest suite is the correctness gate): each result is compared
with the reference's (`agrees` = impl.agreement() finds nothing, plus impl.accuracy()'s
figures) and an estimator's (EXACT = False) also with its exact Rust sibling's — the one
Rust implementation with EXACT = True, if there is exactly one — as `exact_rust_*` and
`speedup_vs_exact_rust`. Variants: `Variant.estimator_params` go to estimators only, so the
exact implementations run once per distinct `Variant.params` (their rows carry the first
such variant's label).
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

sys.path.insert(
    0, str(Path(__file__).parents[1])
)  # tests/ -> `import datagen` in benchmark scripts

import turbo_parakeet  # noqa: E402
from datagen import large_dataset as generated_large_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
SINGLE_THREAD_ENV = {"RAYON_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
CHILD_TIMEOUT_S = 1800
COLUMNS = [
    "dataset",
    "implementation",
    "variant",
    "threads",
    "status",
    "median_ms",
    "min_ms",
    "load_mb",
    "peak_mb",
    "peak_is_lower_bound",
    "agrees",
    "exact_rust_agrees",
    "algorithmic",
    "parallel",
    "total",
    "speedup_vs_exact_rust",
]
FIXED_DTYPES = {
    **dict.fromkeys(
        (
            "median_ms",
            "min_ms",
            "load_mb",
            "peak_mb",
            "algorithmic",
            "parallel",
            "total",
        ),
        pl.Float64,
    ),
    **dict.fromkeys(("peak_is_lower_bound", "agrees", "exact_rust_agrees"), pl.Boolean),
    "speedup_vs_exact_rust": pl.Float64,
}


@dataclass(frozen=True)
class Dataset:
    name: str
    make: Callable[[], dict[str, pl.DataFrame]]
    exclude: tuple[
        str, ...
    ] = ()  # implementations not run on this dataset (e.g. would exhaust memory)


@dataclass(frozen=True)
class Variant:
    label: str = "default"
    params: dict = field(default_factory=dict)  # every implementation
    estimator_params: dict = field(
        default_factory=dict
    )  # estimators (EXACT = False) only


def large_dataset(columns: int | None = None, exclude: tuple[str, ...] = ()) -> Dataset:
    """datagen.large_dataset() (50K rows × 101 cols, generated), optionally only its first `columns`."""
    label = "large_dataset" + (f" (first {columns} cols)" if columns else "")

    def make() -> dict[str, pl.DataFrame]:
        df = generated_large_dataset()
        return {"large": df.select(df.columns[:columns]) if columns else df}

    return Dataset(label, make, exclude)


def memory_mb() -> tuple[float, float]:
    """This process's (current, peak) working set (resident set) so far, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        assert ok, "GetProcessMemoryInfo failed"
        return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # bytes
        return maxrss / 2**20, maxrss / 2**20
    with open("/proc/self/statm") as f:  # pages
        current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return current / 2**20, maxrss / 2**10  # ru_maxrss is KiB on Linux


def time_call(
    fn: Callable[[], object],
    runs: int,
    budget_s: float,
    on_warmup: Callable[[], None] | None = None,
) -> tuple[list[float] | None, object]:
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    if on_warmup is not None:
        on_warmup()
    if elapsed > budget_s:
        return None, None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def speedups(rows: list[dict]) -> list[dict]:
    """Per (dataset, variant, Rust implementation); the non-Rust competitors are the rows of
    the same dataset and params group (rows without a group form one)."""
    out = []
    ok = [r for r in rows if r["status"] == "ok"]
    for dataset, group in dict.fromkeys((r["dataset"], r.get("group")) for r in ok):
        same = [r for r in ok if r["dataset"] == dataset and r.get("group") == group]
        others = [
            r["median_ms"] for r in same if not r["implementation"].endswith("Rust")
        ]
        best = min(others) if others else None
        rust = dict.fromkeys(
            (r["implementation"], r.get("variant"))
            for r in same
            if r["implementation"].endswith("Rust")
        )
        for impl, variant in rust:

            def median(threads: str) -> float | None:
                return next(
                    (
                        r["median_ms"]
                        for r in same
                        if r["implementation"] == impl
                        and r.get("variant") == variant
                        and r["threads"] == threads
                    ),
                    None,
                )

            n, one = median("N"), median("1")
            out.append(
                {
                    "dataset": dataset,
                    "variant": variant,
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


def _threads(name: str) -> tuple[str, ...]:
    return ("N", "1") if name.endswith("Rust") else ("default",)


def _plan(
    classes: dict, variants: list[Variant], params: dict, impl_params: dict
) -> list[tuple[str, str, str, str, dict]]:
    """(implementation, variant, group, threads, kwargs) per measurement. The group is the
    first variant with the same `params`; non-estimators run once per group."""
    first: dict[str, str] = {}
    group = {
        v.label: first.setdefault(json.dumps(v.params, sort_keys=True), v.label)
        for v in variants
    }
    plan = []
    for name, cls in classes.items():
        for v in variants:
            estimator = not cls.EXACT
            if not estimator and group[v.label] != v.label:
                continue
            kwargs = {
                **params,
                **v.params,
                **(v.estimator_params if estimator else {}),
                **impl_params.get(name, {}),
            }
            for threads in _threads(name):
                plan.append((name, v.label, group[v.label], threads, kwargs))
    return plan


def _measure(
    package: str,
    name: str,
    kwargs: dict,
    frames: dict[str, str],
    threads: str,
    runs: int,
    budget_s: float,
    out: str,
) -> dict:
    spec = Path(out).with_suffix(".json")
    try:
        spec.write_text(
            json.dumps(
                {
                    "package": package,
                    "implementation": name,
                    "params": kwargs,
                    "frames": frames,
                    "runs": runs,
                    "budget_s": budget_s,
                    "out": out,
                }
            )
        )
    except (TypeError, ValueError) as exc:  # e.g. params that are not JSON
        print(
            f"    error: parameters not serialisable: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return {"status": "error"}
    env = {**os.environ, **(SINGLE_THREAD_ENV if threads == "1" else {})}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(spec)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").splitlines()[-20:])
        print(f"    error:\n{tail}", file=sys.stderr, flush=True)
        return {"status": "error"}
    except subprocess.TimeoutExpired:
        print("    timeout", file=sys.stderr, flush=True)
        return {"status": "timeout"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        print(
            f"    error: child printed no result:\n{(proc.stderr or '')[-500:]}",
            file=sys.stderr,
            flush=True,
        )
        return {"status": "error"}


def _child(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    frames = {n: pl.read_ipc(p, memory_map=False) for n, p in spec["frames"].items()}
    cls = getattr(importlib.import_module(spec["package"]), spec["implementation"])
    gc.collect()
    load_mb, peak_before = memory_mb()
    peak: list[float] = []
    times, result = time_call(
        lambda: cls(**spec["params"]).add(frames).result(),
        spec["runs"],
        spec["budget_s"],
        on_warmup=lambda: peak.append(memory_mb()[1]),
    )
    info = {
        "load_mb": load_mb,
        "peak_mb": peak[0] - load_mb,
        "peak_is_lower_bound": peak[0] == peak_before,
    }
    if times is None:
        info.update(
            status=f"skipped: over budget ({spec['budget_s']:.0f}s)",
            median_ms=None,
            min_ms=None,
        )
    else:
        result.write_ipc(spec["out"])
        info.update(
            status="ok",
            median_ms=statistics.median(times) * 1e3,
            min_ms=min(times) * 1e3,
        )
    print(json.dumps(info))


def _compare(
    impl, frames: dict, result: pl.DataFrame, other: pl.DataFrame, prefix: str
) -> dict:
    out = {}
    try:  # a failed comparison must not lose the run's other rows
        out["agrees"] = not impl.agreement(result, other)
    except Exception as exc:
        print(f"    agreement failed: {exc!r}", file=sys.stderr, flush=True)
    try:
        out.update(impl.accuracy(frames, result, other))
    except Exception as exc:
        print(f"    accuracy failed: {exc!r}", file=sys.stderr, flush=True)
    return {prefix + k: v for k, v in out.items()}


def _score(
    rows: list[dict],
    results: dict,
    kwargs_of: dict,
    frames: dict,
    classes: dict,
    reference: str,
    sibling: str | None,
) -> None:
    """Fill one dataset's agreement / accuracy columns in place."""
    row_of = {(r["implementation"], r["variant"], r["threads"]): r for r in rows}
    for row in rows:
        key = (row["implementation"], row["variant"], row["threads"])
        result = results.get(key)
        if result is None or row["implementation"] == reference:
            continue
        cls = classes[row["implementation"]]
        try:
            impl = cls(**kwargs_of[key])
        except Exception as exc:
            print(f"    scoring failed: {exc!r}", file=sys.stderr, flush=True)
            continue
        ref = results.get((reference, row["group"], _threads(reference)[0]))
        if ref is not None:
            row.update(_compare(impl, frames, result, ref, ""))
        if sibling is None or cls.EXACT:
            continue
        sib = results.get((sibling, row["group"], row["threads"]))
        if sib is not None:
            row.update(_compare(impl, frames, result, sib, "exact_rust_"))
            sib_ms = row_of[sibling, row["group"], row["threads"]]["median_ms"]
            row["speedup_vs_exact_rust"] = sib_ms / row["median_ms"]


def run(
    package: str,
    datasets: list[Dataset],
    params: dict | None = None,
    impl_params: dict[str, dict] | None = None,
    variants: list[Variant] | None = None,
    runs: int = 5,
    budget_s: float = 60.0,
    results_name: str | None = None,
) -> pl.DataFrame:
    params, impl_params = params or {}, impl_params or {}
    variants = variants or [Variant()]
    if len({v.label for v in variants}) != len(variants):
        raise ValueError("variant labels must be unique")
    module = importlib.import_module(package)
    names = [module.REFERENCE] + [
        n for n in module.IMPLEMENTATIONS if n != module.REFERENCE
    ]
    classes = {n: c for n in names if (c := _load(module, n)) is not None}
    exact_rust = [n for n, c in classes.items() if n.endswith("Rust") and c.EXACT]
    sibling = exact_rust[0] if len(exact_rust) == 1 else None
    if len({d.name for d in datasets}) != len(datasets):
        raise ValueError("dataset names must be unique")
    plan = _plan(classes, variants, params, impl_params)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ds in enumerate(datasets):
            frames = {
                n: f.collect() if isinstance(f, pl.LazyFrame) else f
                for n, f in ds.make().items()
            }
            folder = Path(tmp) / f"dataset{i}"
            folder.mkdir()
            paths = {}
            for j, (n, f) in enumerate(frames.items()):
                paths[n] = str(folder / f"frame{j}.arrow")
                f.write_ipc(paths[n])
            ds_rows, results, kwargs_of = [], {}, {}
            for j, (name, label, group, threads, kwargs) in enumerate(plan):
                row = {
                    "dataset": ds.name,
                    "implementation": name,
                    "variant": label,
                    "threads": threads,
                    "group": group,
                }
                ds_rows.append(row)
                if name in ds.exclude:
                    row["status"] = "excluded"
                    continue
                print(
                    f"  {ds.name} | {label}: {name} @{threads} …",
                    file=sys.stderr,
                    flush=True,
                )
                out = str(folder / f"result{j}.arrow")
                row.update(
                    _measure(package, name, kwargs, paths, threads, runs, budget_s, out)
                )
                if row["status"] == "ok":
                    results[name, label, threads] = pl.read_ipc(out, memory_map=False)
                    kwargs_of[name, label, threads] = kwargs
            _score(
                ds_rows, results, kwargs_of, frames, classes, module.REFERENCE, sibling
            )
            rows += ds_rows
            del frames, results

    by_key = {
        (s["dataset"], s["variant"], s["implementation"]): s for s in speedups(rows)
    }
    for r in rows:
        s = by_key.get((r["dataset"], r["variant"], r["implementation"]))
        if s is not None and r["threads"] == "N":
            r.update(
                algorithmic=s["algorithmic"], parallel=s["parallel"], total=s["total"]
            )
        del r["group"]
    estimators = {n for n, c in classes.items() if not c.EXACT}
    return _report(
        results_name or package.rsplit(".", 1)[-1], package, rows, estimators
    )


def _report(
    name: str,
    package: str,
    rows: list[dict],
    estimators: set[str] | frozenset = frozenset(),
) -> pl.DataFrame:
    for stream in (sys.stdout, sys.stderr):  # tables and "—" on a cp1252 console / pipe
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    rows = [{**dict.fromkeys(COLUMNS), **r} for r in rows]
    frame = pl.from_dicts(rows, infer_schema_length=None)
    extras = [c for c in frame.columns if c not in COLUMNS]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame = frame.select(*COLUMNS, *extras).with_columns(
        pl.col(c).cast(t) for c, t in FIXED_DTYPES.items()
    )
    frame = frame.with_columns(
        technique=pl.lit(package.rsplit(".", 1)[-1]),
        cpu_count=pl.lit(os.cpu_count()),
        processor=pl.lit(platform.processor()),
        polars=pl.lit(pl.__version__),
        numpy=pl.lit(np.__version__),
        analytics=pl.lit(turbo_parakeet.__version__),
        timestamp=pl.lit(stamp),
    )
    print(
        f"\n{name} — median / min of timed runs, one process per measurement, {os.cpu_count()} CPUs"
    )
    keys = ["dataset", "implementation", "variant", "threads"]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=40):
        print(
            frame.select(
                *keys,
                "median_ms",
                "min_ms",
                "peak_mb",
                "peak_is_lower_bound",
                "agrees",
                "status",
            )
        )
        print(
            frame.filter(pl.col("parallel").is_not_null()).select(
                *keys[:3], "algorithmic", "parallel", "total"
            )
        )
        print(
            frame.filter(
                pl.col("implementation").is_in(sorted(estimators))
                | (pl.col("agrees") == False),  # noqa: E712
                pl.col("threads") != "1",
            ).select(*keys[:3], "agrees", *extras, "speedup_vs_exact_rust")
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_{stamp}.arrow"
    frame.write_ipc(path, compression="zstd")
    print(f"\nsaved {path}")
    return frame


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    _child(sys.argv[2])
