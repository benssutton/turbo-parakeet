"""Calibrate the ClickHouse size emulator against bytes measured by chbench on a live server.

Standalone (never collected by pytest).  Needs the `chbench-ch` container (services/bench/CLAUDE.md,
Environment).  For each frame: recommend, `to_chbench`, run every suite with chbench (storage bytes only),
join the measured compressed bytes of the data column with `<name>.predicted.json`, write
`results/clickhouse_calibration.arrow` and print a summary.

    python tests/performance/calibrate_clickhouse_sizes.py [--workers N] [--keep]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import yaml
from synthdata.synth import synth_frame

from turbo_parakeet.recommenders import OneShotRecommender, to_chbench

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "services" / "bench"  # chbench runs with this as its working directory

N = 1_000_000
RESULT = Path(__file__).parent / "results" / "clickhouse_calibration.arrow"
MINIMAL_QUERIES = {
    "kinds": [{"kind": "full_scan"}],
    "query_condition_cache": [False],
    "cache_mode": ["warm"],
    "samples": 1,
}
FIXED_WIDTH = (
    "UInt8 UInt16 UInt32 UInt64 Int8 Int16 Int32 Int64 Float32 Float64 Decimal32 Decimal64 "
    "Decimal128 Date Date32 DateTime DateTime64 Bool Enum8 Enum16 Time64"
).split()


def gcd_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(0)
    multiples = rng.integers(0, 10**9 + 1, N, dtype=np.uint64) * np.uint64(1000)
    even = rng.integers(2**31, 2**32, N, dtype=np.uint64) * np.uint64(2)
    three = rng.integers(0, 2**40 // 3, N, dtype=np.int64) * 3
    secs = rng.integers(0, 2 * 10**9, N, dtype=np.int64)
    ts = pl.Series("ts_seconds", secs * 1_000_000).cast(pl.Datetime("us"))
    return {
        "gcd_1000": pl.DataFrame({"gcd_1000": multiples}),
        "gcd_even": pl.DataFrame({"gcd_even": even}),
        "gcd_three": pl.DataFrame({"gcd_three": three}),
        "ts_seconds": pl.DataFrame([ts]),
    }


def delta_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(1)
    step = np.full(N, 1000, dtype=np.int64)
    jit = step + rng.integers(-5, 6, N)
    gaps = rng.random(N) < 0.01
    jit[gaps] = rng.integers(1000, 60_000, int(gaps.sum()))
    t0 = 1_700_000_000_000
    walk = np.cumsum(rng.integers(-1000, 1001, N), dtype=np.int64)
    ids = np.cumsum(rng.integers(1, 11, N), dtype=np.uint64)
    w16 = np.clip(np.cumsum(rng.integers(-3, 4, N)), -32768, 32767).astype(np.int16)
    w32 = np.cumsum(rng.integers(-100, 101, N)).astype(np.int32)
    ts = lambda name, d: pl.DataFrame(  # noqa: E731
        [pl.Series(name, t0 + np.cumsum(d), dtype=pl.Int64).cast(pl.Datetime("ms"))]
    )
    return {
        "ts_regular": ts("ts_regular", step),
        "ts_jitter": ts("ts_jitter", jit),
        "walk": pl.DataFrame({"walk": walk}),
        "ids_sorted": pl.DataFrame({"ids_sorted": ids}),
        "walk_i16": pl.DataFrame({"walk_i16": w16}),
        "walk_i32": pl.DataFrame({"walk_i32": w32}),
        "random_ints": pl.DataFrame(
            {"random_ints": rng.integers(-(2**62), 2**62, N, dtype=np.int64)}
        ),
    }


def widen(yaml_path: Path) -> None:
    """Measure every transform on a delta-frame suite, not only the recommended one: the codec
    axis becomes none / GCD / Delta / DoubleDelta and the compression axis LZ4 / ZSTD(1) / ZSTD(3),
    predictions from the emulator on the suite's own dataset."""
    from turbo_parakeet import _plugin

    suite = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    ch = suite["clickhouse"]
    if "LowCardinality" in ch["codec"]:
        return
    one = pa.ipc.open_file(yaml_path.parent / suite["dataset"]["path"]).read_all()
    col, base = suite["dataset"]["column"], ch["type"][0]
    comps = ["LZ4", "ZSTD(1)", "ZSTD(3)"]
    codecs = ["none", "GCD", "Delta", "DoubleDelta"]
    predicted = {}
    for k in codecs:
        for c in comps:
            chain = c if k == "none" else f"{k}, {c}"
            predicted[f"{base}|{k}|{c}"] = _plugin.clickhouse_profile(one, col, chain)[
                "bytes"
            ]
    ch["codec"], ch["compression"], ch["skip_index"] = codecs, comps, ["none"]
    yaml_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    (yaml_path.parent / f"{yaml_path.stem}.predicted.json").write_text(
        json.dumps(predicted), encoding="utf-8"
    )


def run_suite(yaml_path: Path, work: Path) -> tuple[str, str]:
    name = yaml_path.stem
    suite = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    suite["queries"] = MINIMAL_QUERIES
    yaml_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    store, cache, out = work / f"{name}.db", work / "cache", work / f"{name}.csv"
    py = sys.executable
    base = [py, "-m", "chbench.cli"]
    run = subprocess.run(
        [
            *base,
            "run",
            str(yaml_path),
            "--store",
            str(store),
            "--cache-dir",
            str(cache),
        ],
        cwd=BENCH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if run.returncode:
        return name, f"run failed: {run.stderr[-500:] or run.stdout[-500:]}"
    rep = subprocess.run(
        [*base, "report", "--suite", name, "--csv", str(out), "--store", str(store)],
        cwd=BENCH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return name, (
        ""
        if rep.returncode == 0
        else f"report failed: {rep.stderr[-500:] or rep.stdout[-500:]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    work = Path(tempfile.mkdtemp(prefix="calibrate_ch_"))
    frames = {"synth_frame": synth_frame(50_000), **gcd_frames(), **delta_frames()}

    delta_names = set(delta_frames())
    suites: list[tuple[str, Path]] = []  # (frame, yaml)
    for fname, frame in frames.items():
        result = OneShotRecommender().add(frame).result()
        d = work / fname
        d.mkdir()
        skipped = to_chbench(result, frame, d)
        print(
            f"{fname}: {len(list(d.glob('*.yaml')))} suites, {len(skipped)} columns skipped"
        )
        if fname in delta_names:
            for p in d.glob("*.yaml"):
                widen(p)
        suites += [(fname, p) for p in sorted(d.glob("*.yaml"))]
    # Suite names are column names; make them unique across frames for the stores.
    errors = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_suite, p, p.parent) for _, p in suites]
        for i, f in enumerate(futs):
            name, err = f.result()
            if err:
                errors[name] = err
            if (i + 1) % 10 == 0:
                print(f"  ran {i + 1}/{len(futs)}", flush=True)
    for n, e in errors.items():
        print(f"ERROR {n}: {e}")

    rows, n_null_pred, n_skipped = [], 0, 0
    for fname, yp in suites:
        name = yp.stem
        csv_path = yp.parent / f"{name}.csv"
        if not csv_path.exists():
            continue
        predicted = json.loads((yp.parent / f"{name}.predicted.json").read_text())
        suite = yaml.safe_load(yp.read_text(encoding="utf-8"))
        column = suite["dataset"]["column"]
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                base = r["ch_type"]
                while base.startswith(("Nullable(", "LowCardinality(")):
                    base = base[base.index("(") + 1 : -1]
                base = base.split("(", 1)[0]
                key = f"{base}|{r['codec']}|{r['compression']}"
                measured = r["compressed_bytes"]
                if measured in ("", None) or key not in predicted:
                    n_skipped += 1
                    continue
                p = predicted[key]
                if p is None:
                    n_null_pred += 1
                    continue
                m = int(measured)
                rows.append(
                    (
                        fname,
                        column,
                        base,
                        r["codec"],
                        r["compression"],
                        p,
                        m,
                        (p - m) / m if m else float("nan"),
                    )
                )
    if not rows:
        sys.exit("no calibration cells")
    cols = list(zip(*rows))
    names = [
        "frame",
        "column",
        "type",
        "codec",
        "compression",
        "predicted",
        "measured",
        "rel_error",
    ]
    types = [pa.string()] * 5 + [pa.int64(), pa.int64(), pa.float64()]
    table = pa.table({n: pa.array(c, t) for n, c, t in zip(names, cols, types)})
    RESULT.parent.mkdir(exist_ok=True)
    with pa.ipc.new_file(RESULT, table.schema) as w:
        w.write_table(table)
    print(
        f"\nwrote {RESULT} ({table.num_rows} cells); skipped by chbench: {n_skipped}; "
        f"null predictions excluded: {n_null_pred}; suites failed: {len(errors)}"
    )

    df = pl.from_arrow(table).with_columns(abs_err=pl.col("rel_error").abs())
    g = (
        df.group_by("type", "codec", "compression")
        .agg(n=pl.len(), median=pl.col("abs_err").median(), max=pl.col("abs_err").max())
        .sort("type", "codec", "compression")
    )
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=40):
        print(g)
        print("\nworst cells")
        print(df.sort("abs_err", descending=True).head(15).drop("abs_err"))
    a = df["abs_err"].drop_nans()
    print(
        f"\noverall: n={len(a)} median={a.median():.4f} p90={a.quantile(0.9):.4f} max={a.max():.4f}"
    )
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print("kept", work)


if __name__ == "__main__":
    main()
