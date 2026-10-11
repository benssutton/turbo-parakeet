"""
AMS entropy memory and error report: every implementation of analytics.pairwise_entropy
and analytics.threeway_entropy on the speed benchmarks' datasets (near-unique 1M x 9
and near-constant 1M x 3 included).

Each (dataset, implementation) runs once in a fresh child process. After loading the data
and a gc.collect(), the child notes its current working set, runs result() and reports
`peak_mb` = the process's peak working set afterwards minus that current working set (what
result() needed on top of the loaded data). `peak_is_lower_bound` is true when the peak did
not move during result(), i.e. result() never exceeded the load-time peak: peak_mb then
only bounds the true figure from below. Windows: WorkingSetSize / PeakWorkingSetSize;
Linux: /proc/self/statm and ru_maxrss; macOS: ru_maxrss for both. The parent compares each
result with the reference's: max_rel_err / p95_rel_err over every computed entropy,
n_missing (null estimates where the reference computed), max_abs_mi_err (pairwise only).
Each child has a 1800 s timeout (status "timeout"); implementations a dataset excludes get
status "excluded".

The AMS path holds 8 B per row per column after converting keys in place; the exact path
about 9. peak_mb differences therefore show mostly the counting state; the sketch's
advantage of not keeping rows at all is a streaming one and is not measured here.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_ams_entropy.py
Results -> tests/performance/results/ams_entropy.parquet
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
from harness import RESULTS_DIR, Dataset, large_dataset

from datagen import low_cardinality, near_constant, near_unique

DATASETS = {
    "analytics.pairwise_entropy": [
        large_dataset(),
        Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
        Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
        Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
    ],
    "analytics.threeway_entropy": [
        large_dataset(columns=40),
        Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
        Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
        Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
        Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
    ],
}
ENTROPIES = {
    "analytics.pairwise_entropy": ["h_a", "h_b", "h_ab"],
    "analytics.threeway_entropy": ["h_abc"],
}


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


def child(package: str, index: int, name: str, out: str) -> None:
    """Load the dataset, run the implementation once, write its result to `out` and print
    the peak working set it needed above the loaded data."""
    frames = DATASETS[package][index].make()
    cls = getattr(importlib.import_module(package), name)
    gc.collect()
    current_before, peak_before = memory_mb()
    result = cls().add(frames).result()
    _, peak_after = memory_mb()
    result.write_parquet(out)
    print(
        json.dumps(
            {
                "peak_mb": peak_after - current_before,
                "peak_is_lower_bound": peak_after == peak_before,
            }
        )
    )


def errors(package: str, got: pl.DataFrame, want: pl.DataFrame) -> dict:
    """Relative error of every computed entropy; nulls in `got` where `want` computed;
    absolute MI error (pairwise only)."""
    keys = [c for c in want.columns if c.startswith(("df_", "col_"))]
    assert got.select(keys).equals(want.select(keys)), "result rows are not aligned"
    computed = want["status"] == "computed"
    got, want = got.filter(computed), want.filter(computed)
    rels, missing = [], 0
    for c in ENTROPIES[package]:
        missing += got[c].null_count()
        rel = (got[c] - want[c]).abs() / want[c]
        # Only 0/0 (both exactly 0) is no error; NaN elsewhere stays visible.
        rels.append(
            pl.select(
                pl.when((got[c] == 0) & (want[c] == 0)).then(0.0).otherwise(rel)
            ).to_series()
        )
    rel = pl.concat(rels)
    mi = (got["mi"] - want["mi"]).abs().max() if "mi" in want.columns else None
    return {
        "max_rel_err": rel.max(),
        "p95_rel_err": rel.quantile(0.95),
        "n_missing": missing,
        "max_abs_mi_err": mi,
    }


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for package, datasets in DATASETS.items():
            module = importlib.import_module(package)
            names = [module.REFERENCE] + [
                n for n in module.IMPLEMENTATIONS if n != module.REFERENCE
            ]
            for index, ds in enumerate(datasets):
                results = {}
                for name in names:
                    row = {
                        "package": package.split(".")[-1],
                        "dataset": ds.name,
                        "implementation": name,
                        "status": "ok",
                        "peak_mb": None,
                        "peak_is_lower_bound": None,
                        "max_rel_err": None,
                        "p95_rel_err": None,
                        "n_missing": None,
                        "max_abs_mi_err": None,
                    }
                    rows.append(row)
                    if name in ds.exclude:
                        row["status"] = "excluded"
                        continue
                    print(
                        f"  {package} | {ds.name}: {name} ...",
                        file=sys.stderr,
                        flush=True,
                    )
                    out = str(Path(tmp) / f"{name}.parquet")
                    start = time.perf_counter()
                    try:
                        proc = subprocess.run(
                            [sys.executable, __file__, "--child", package]
                            + [str(index), name, out],
                            stdout=subprocess.PIPE,
                            text=True,
                            check=True,
                            timeout=1800,
                        )
                    except subprocess.TimeoutExpired:
                        row["status"] = "timeout"
                        print("    timeout", file=sys.stderr, flush=True)
                        continue
                    print(
                        f"    {time.perf_counter() - start:.1f} s",
                        file=sys.stderr,
                        flush=True,
                    )
                    info = json.loads(proc.stdout.strip().splitlines()[-1])
                    row["peak_mb"] = info["peak_mb"]
                    row["peak_is_lower_bound"] = info["peak_is_lower_bound"]
                    results[name] = pl.read_parquet(out)
                    if name != module.REFERENCE and module.REFERENCE in results:
                        row.update(
                            errors(package, results[name], results[module.REFERENCE])
                        )
    df = pl.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_parquet(RESULTS_DIR / "ams_entropy.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        main()
