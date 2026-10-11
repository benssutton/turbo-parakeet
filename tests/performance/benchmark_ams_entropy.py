"""
AMS entropy memory and error report: every implementation of analytics.pairwise_entropy
and analytics.threeway_entropy on the speed benchmarks' datasets (near-unique 1M x 9
included).

Each (dataset, implementation) runs once in a fresh child process, which reports the peak
working set its result() added above the peak after loading the data (`peak_mb`; Windows
PeakWorkingSetSize, elsewhere ru_maxrss). The parent compares each result with the
reference's: max_rel_err / p95_rel_err over every computed entropy, max_abs_mi_err
(pairwise only).

Both Rust implementations hold the encoded columns (~9 B per row per column), so peak_mb
differences show the counting state only; the sketch's advantage of not keeping rows at
all is a streaming one and is not measured here.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_ams_entropy.py
Results -> tests/performance/results/ams_entropy.parquet
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
from harness import RESULTS_DIR, Dataset, large_dataset

from datagen import low_cardinality, near_unique

DATASETS = {
    "analytics.pairwise_entropy": [
        large_dataset(),
        Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
        Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
    ],
    "analytics.threeway_entropy": [
        large_dataset(columns=40),
        Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
        Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
        Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
    ],
}
ENTROPIES = {
    "analytics.pairwise_entropy": ["h_a", "h_b", "h_ab"],
    "analytics.threeway_entropy": ["h_abc"],
}


def peak_mb() -> float:
    """This process's peak working set (resident set) so far, in MiB."""
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
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (
        2**20 if sys.platform == "darwin" else 2**10
    )  # bytes on macOS, KiB on Linux


def child(package: str, index: int, name: str, out: str) -> None:
    """Load the dataset, run the implementation once, write its result to `out` and print
    the peak working set it added."""
    frames = DATASETS[package][index].make()
    before = peak_mb()
    cls = getattr(importlib.import_module(package), name)
    result = cls().add(frames).result()
    after = peak_mb()
    result.write_parquet(out)
    print(json.dumps({"peak_mb": after - before}))


def errors(package: str, got: pl.DataFrame, want: pl.DataFrame) -> dict:
    """Relative error of every computed entropy; absolute MI error (pairwise only)."""
    computed = want["status"] == "computed"
    got, want = got.filter(computed), want.filter(computed)
    rel = pl.concat(
        [((got[c] - want[c]).abs() / want[c]).fill_nan(0.0) for c in ENTROPIES[package]]
    )  # 0/0 (both exactly 0) -> 0; x/0 -> inf
    mi = (got["mi"] - want["mi"]).abs().max() if "mi" in want.columns else None
    return {
        "max_rel_err": rel.max(),
        "p95_rel_err": rel.quantile(0.95),
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
                    print(
                        f"  {package} | {ds.name}: {name} …",
                        file=sys.stderr,
                        flush=True,
                    )
                    out = str(Path(tmp) / f"{name}.parquet")
                    proc = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--child",
                            package,
                            str(index),
                            name,
                            out,
                        ],
                        stdout=subprocess.PIPE,
                        text=True,
                        check=True,
                    )
                    peak = json.loads(proc.stdout.strip().splitlines()[-1])["peak_mb"]
                    results[name] = pl.read_parquet(out)
                    err = {
                        "max_rel_err": None,
                        "p95_rel_err": None,
                        "max_abs_mi_err": None,
                    }
                    if name != module.REFERENCE:
                        err = errors(package, results[name], results[module.REFERENCE])
                    rows.append(
                        {
                            "package": package.split(".")[-1],
                            "dataset": ds.name,
                            "implementation": name,
                            "peak_mb": peak,
                            **err,
                        }
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
