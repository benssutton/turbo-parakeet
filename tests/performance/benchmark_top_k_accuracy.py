"""
Top-k accuracy and memory report: TopKHeavyKeeperRust and TopKSpaceSavingRust against the
exact TopKPolars on datagen.top_k_streams (10M rows: Zipf α = 1.5 / 1.1 / 0.8, flat, skewed,
string), k ∈ {5, 256} × memory_bytes ∈ {¼, 1, 4} × the default max(16 KiB, k × 1 KiB).

The parent writes the data once to a temporary Arrow IPC file and computes every column's
exact counts. Each (column, k, budget, implementation) then runs in a fresh child process
that reads its one column, notes its working set, runs result() once, and reports peak_mb
(peak working set minus the post-load working set, as benchmark_ams_entropy.py), time_s and
the reported items. The reference runs once per (column, k); its budget is null.

Scores, per run (`top` = the reported items, true = exact counts):
  precision            tie-aware: reported values whose true count ≥ the true k-th count,
                       ÷ min(k, distinct values)
  are, aae             mean relative / absolute count error over reported values (the paper's)
  max_bound_violation  SpaceSaving only: max over items of how far the true count falls
                       outside [count − error, count] (0 = bounds hold)
  sketch_bytes         the sketch's accounted bytes (null for the reference)

--rows N (e.g. --rows 100_000_000): the scale run instead — zipf_11 only, k = 5, 1× budget,
written to results/top_k_scale.arrow.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_top_k_accuracy.py
Results -> tests/performance/results/top_k_accuracy.arrow
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
from benchmark_ams_entropy import memory_mb
from harness import RESULTS_DIR

from datagen import TOP_K_COLUMNS, top_k_streams

SKETCHES = ("TopKHeavyKeeperRust", "TopKSpaceSavingRust")
REFERENCE = "TopKPolars"


def default_budget(k: int) -> int:
    return max(16 * 1024, k * 1024)


def child(data: str, column: str, name: str, k: int, budget: int, out: str) -> None:
    import turbo_parakeet.techniques.top_k

    frame = pl.read_ipc(data, columns=[column], memory_map=False)
    cls = getattr(turbo_parakeet.techniques.top_k, name)
    gc.collect()
    current_before, peak_before = memory_mb()
    start = time.perf_counter()
    result = cls(k=k, memory_bytes=budget).add({"t": frame}).result()
    elapsed = time.perf_counter() - start
    _, peak_after = memory_mb()
    result.select("top_k", "sketch_bytes").write_ipc(out)
    print(
        json.dumps(
            {
                "peak_mb": peak_after - current_before,
                "peak_is_lower_bound": peak_after == peak_before,
                "time_s": elapsed,
            }
        )
    )


def truth_of(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """value (text) → exact count, every distinct non-null value, count desc."""
    return (
        frame.select(pl.col(column).cast(pl.String).alias("value"))
        .drop_nulls()
        .group_by("value")
        .agg(true=pl.len())
        .sort("true", descending=True)
    )


def score(top: list[dict], truth: pl.DataFrame, k: int) -> dict:
    floor = truth["true"][k - 1] if truth.height >= k else 0
    reported = pl.DataFrame(
        top, schema={"value": pl.String, "count": pl.UInt64, "error": pl.UInt64}
    ).join(truth, on="value", how="left")
    count = reported["count"].cast(pl.Int64)
    true = reported["true"].cast(pl.Int64)
    err = (count - true).abs()
    violation = None
    if reported["error"].null_count() == 0 and reported.height:
        lower = count - reported["error"].cast(pl.Int64)
        violation = max(0, (lower - true).max(), (true - count).max())
    return {
        "precision": (true >= floor).sum() / min(k, truth.height),
        "are": (err / true).mean(),
        "aae": err.mean(),
        "max_bound_violation": violation,
    }


def run_one(data, column, name, k, budget, out) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--child",
            data,
            column,
            name,
            str(k),
            str(budget),
            out,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        timeout=1800,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(n_rows: int, scale: bool) -> None:
    columns = ["zipf_11"] if scale else list(TOP_K_COLUMNS)
    ks = (5,) if scale else (5, 256)
    multipliers = (1.0,) if scale else (0.25, 1.0, 4.0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        data = str(Path(tmp) / "streams.arrow")
        frame = top_k_streams(n_rows, columns=columns)
        frame.write_ipc(data)
        for column in columns:
            truth = truth_of(frame, column)
            for k in ks:
                runs = [(REFERENCE, None)] + [
                    (name, int(default_budget(k) * m))
                    for m in multipliers
                    for name in SKETCHES
                ]
                for name, budget in runs:
                    row = {
                        "n_rows": n_rows,
                        "column": column,
                        "k": k,
                        "budget_bytes": budget,
                        "implementation": name,
                        "status": "ok",
                    }
                    rows.append(row)
                    print(
                        f"  {column} k={k} budget={budget}: {name} ...",
                        file=sys.stderr,
                        flush=True,
                    )
                    out = str(Path(tmp) / "out.arrow")
                    try:
                        info = run_one(
                            data, column, name, k, budget or default_budget(k), out
                        )
                    except subprocess.CalledProcessError as exc:
                        row["status"] = "error"
                        print(
                            "\n".join((exc.stderr or "").splitlines()[-20:]),
                            file=sys.stderr,
                        )
                        continue
                    except subprocess.TimeoutExpired:
                        row["status"] = "timeout"
                        continue
                    result = pl.read_ipc(out, memory_map=False)
                    row.update(info)
                    row["sketch_bytes"] = result["sketch_bytes"][0]
                    row.update(score(result["top_k"][0].to_list(), truth, k))
    df = pl.DataFrame(rows, infer_schema_length=None)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_ipc(
        RESULTS_DIR / ("top_k_scale.arrow" if scale else "top_k_accuracy.arrow"),
        compression="zstd",
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        a = sys.argv[2:]
        child(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--rows", type=int, default=None)
        args = parser.parse_args()
        main(args.rows or 10_000_000, scale=args.rows is not None)
