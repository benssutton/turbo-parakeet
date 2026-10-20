"""Calibrate the skip-index predictions (skip share, index bytes) against chbench on a live server.

Standalone (never collected by pytest).  Needs the `chbench-ch` container and the sibling repo
`clickhouse_benchmark` (spec 2026-10-20 §8.3, §10).  For each frame: recommend, `to_chbench`, run
every suite whose recommender chose a skip index (point queries only, one sample), read the results
store directly (predicate SQL, `selected_marks`, `index_compressed_bytes`), compute for each queried
value the exact predicted skip share from the suite's dataset in stored order, and write
`results/skip_index_calibration.arrow` plus a summary.

    python tests/performance/calibrate_skip_indexes.py [--chbench PATH] [--workers N] [--keep]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests")]

import datagen  # noqa: E402
from turbo_parakeet.recommenders import OneShotRecommender, to_chbench  # noqa: E402

N = 1_000_000
LARGE_ROWS = 200_000
MARK_ROWS = 8192
RESULT = Path(__file__).parent / "results" / "skip_index_calibration.arrow"
EXTRA_FRACTIONS = (0.0005, 0.002, 0.005, 0.02, 0.1, 0.2)  # more distinct queried values


def known_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(1)
    clustered = np.repeat(rng.permutation(1000) * 1_000_003 % 2**31, 1000)
    grouped = np.repeat(np.random.default_rng(3).integers(0, 2**62, N // 10), 10)
    return {
        "sorted_ids": pl.DataFrame({"sorted_ids": np.arange(N)}),
        "clustered": pl.DataFrame({"clustered": clustered}),
        "random_50": pl.DataFrame(
            {"random_50": np.random.default_rng(2).integers(0, 50, N)}
        ),
        "grouped_hashes": pl.DataFrame({"grouped_hashes": grouped}),
    }


def prepare_suite(path: Path) -> dict:
    """Trim the suite: one codec / compression, the point queries plus extra fractions, 1 sample."""
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    ch = suite["clickhouse"]
    ch["codec"] = ["none"] if "none" in ch["codec"] else ch["codec"][:1]
    ch["compression"] = ch["compression"][:1]
    kinds = [k for k in suite["queries"]["kinds"] if k.get("predicate") != "range"]
    kinds += [
        {"kind": "filter", "target_fraction": f, "predicate": "in", "values": 1}
        for f in EXTRA_FRACTIONS
    ]
    suite["queries"]["kinds"] = kinds
    suite["queries"]["samples"] = 1
    path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    return suite


def run_suite(chbench_root: Path, yaml_path: Path, store: Path, cache: Path) -> str:
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "chbench.cli",
            "run",
            str(yaml_path),
            "--store",
            str(store),
            "--cache-dir",
            str(cache),
        ],
        cwd=chbench_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (
        ""
        if run.returncode == 0
        else f"run failed: {(run.stderr or run.stdout)[-500:]}"
    )


def parse_skip(skip: str) -> tuple[str, float | int | None, int]:
    m = re.fullmatch(r"(minmax|set|bloom_filter)(?:\(([\d.]+)\))?:(\d+)", skip)
    kind, arg, g = m.group(1), m.group(2), int(m.group(3))
    return kind, (None if arg is None else float(arg)), g


def parse_literal(lit: str, dtype: pl.DataType):
    lit = lit.strip()
    if lit.lower() in ("true", "false"):
        return int(lit.lower() == "true")
    if dtype.is_integer():
        return int(lit)
    if dtype.is_float():
        return float(lit)
    if dtype == pl.Boolean:
        return int(lit.lower() in ("true", "1"))
    if dtype in (pl.String, pl.Categorical) or isinstance(
        dtype, (pl.Enum, pl.Categorical)
    ):
        if lit.startswith("'") and lit.endswith("'"):
            return re.sub(r"\\(.)", r"\1", lit[1:-1])
    return None  # dates, decimals, ...: not parsed


class BlockStats:
    """Per index block (G marks) of a column in stored order."""

    def __init__(self, col: pl.Series, g: int, kind: str):
        if col.dtype == pl.Boolean:
            col = col.cast(pl.UInt8)
        self.col, self.kind = col, kind
        self.rows_per_block = g * MARK_ROWS
        n = len(col)
        self.n_blocks = math.ceil(n / self.rows_per_block)
        self.marks_per_block = np.full(self.n_blocks, g, dtype=np.int64)
        total_marks = math.ceil(n / MARK_ROWS)
        self.marks_per_block[-1] = total_marks - g * (self.n_blocks - 1)
        self.total_marks = total_marks
        df = pl.DataFrame({"v": col}).with_columns(
            b=(pl.int_range(pl.len()) // self.rows_per_block).cast(pl.Int64)
        )
        self.df = df
        if kind == "minmax":
            agg = (
                df.group_by("b")
                .agg(lo=pl.col("v").min(), hi=pl.col("v").max())
                .sort("b")
            )
            self.lo, self.hi = agg["lo"], agg["hi"]
        if kind == "set":
            agg = df.group_by("b").agg(d=pl.col("v").n_unique()).sort("b")
            self.distinct = agg["d"].to_numpy()  # includes the NULL entry

    def skipped_marks(self, v, max_size: int | None) -> float:
        """Marks in blocks the index must be able to skip (exact set semantics)."""
        if self.kind == "minmax":
            out = ((self.lo > v) | (self.hi < v)).fill_null(True).to_numpy()
        else:
            has = np.zeros(self.n_blocks, dtype=bool)
            blocks = self.df.filter(pl.col("v") == v)["b"].unique().to_numpy()
            has[blocks] = True
            out = ~has
            if self.kind == "set" and max_size:
                out &= self.distinct <= max_size
        return float(self.marks_per_block[out].sum())


def collect(
    frame_name: str, yp: Path, store: Path, predicted: dict, column: str
) -> tuple[list, dict]:
    suite = yaml.safe_load(yp.read_text(encoding="utf-8"))
    ds = yp.parent / suite["dataset"]["path"]
    with pa.ipc.open_file(ds) as r:
        col = pl.from_arrow(r.read_all())[suite["dataset"]["column"]]
    db = sqlite3.connect(store)
    db.row_factory = sqlite3.Row
    rows_sql = """
        SELECT t.table_id, t.ch_type, t.codec, t.compression, t.skip_index, t.rows,
               t.index_compressed_bytes AS idx, q.kind, q.predicate_sql, q.actual_fraction,
               s.selected_marks
        FROM tables t JOIN queries q USING (table_id) JOIN samples s USING (query_id)
        JOIN suite_tables st ON st.table_id = t.table_id
        WHERE st.suite_name = ? AND t.status = 'ok' AND q.status = 'ok'"""
    recs = [dict(r) for r in db.execute(rows_sql, (suite["name"],))]
    stats: dict[tuple, BlockStats] = {}
    out, notes = [], {"unparsed": 0, "marks_mismatch": 0}
    totals = {
        r["table_id"]: r["selected_marks"] for r in recs if r["kind"] == "full_scan"
    }
    for r in recs:
        if r["kind"] != "filter":
            continue
        base = r["ch_type"]
        while base.startswith(("Nullable(", "LowCardinality(")):
            base = base[base.index("(") + 1 : -1]
        base = base.split("(", 1)[0]
        skip = r["skip_index"]
        total = totals.get(r["table_id"])
        if total is None:
            continue
        if total != math.ceil(r["rows"] / MARK_ROWS):
            notes["marks_mismatch"] += 1
        measured = 1 - r["selected_marks"] / total
        m = re.fullmatch(r"v IN \((.*)\)", r["predicate_sql"] or "")
        v = parse_literal(m.group(1), col.dtype) if m else None
        if v is None:
            notes["unparsed"] += 1
            continue
        if skip == "none":
            pred = pred_b = 0.0
            kind, g, fpr, idx_pred = "none", 0, None, None
        else:
            kind, arg, g = parse_skip(skip)
            max_size = int(arg) if kind == "set" else None
            fpr = arg if kind == "bloom_filter" else None
            if col.dtype == pl.Boolean:
                col = col.cast(pl.UInt8)
            bs = stats.setdefault((kind, g), BlockStats(col, g, kind))
            pred = bs.skipped_marks(v, max_size) / bs.total_marks
            pred_b = pred * (1 - fpr) if fpr is not None else pred
            key = f"{base}|{r['codec']}|{r['compression']}|{skip}"
            cell = predicted.get(key)
            idx_pred = cell["index_bytes"] if cell else None
        out.append(
            (
                frame_name,
                column,
                skip,
                kind,
                g,
                base,
                r["codec"],
                r["compression"],
                str(v),
                float(r["actual_fraction"] or math.nan),
                pred,
                pred_b,
                measured,
                idx_pred,
                int(r["idx"] or 0),
            )
        )
    return out, notes


NAMES = [
    "frame",
    "column",
    "skip_index",
    "kind",
    "granularity",
    "type",
    "codec",
    "compression",
    "value",
    "actual_fraction",
    "predicted_skip",
    "predicted_skip_bloom_adjusted",
    "measured_skip",
    "predicted_index_bytes",
    "measured_index_bytes",
]
TYPES = (
    [pa.string()] * 3
    + [pa.string(), pa.int64()]
    + [pa.string()] * 4
    + [pa.float64()] * 4
    + [pa.int64()] * 2
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chbench",
        default=os.environ.get(
            "CHBENCH_DIR", str(ROOT.parent / "clickhouse_benchmark")
        ),
    )
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse", help="a kept work dir: skip recommending and running")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    chbench_root = Path(args.chbench)
    work = Path(args.reuse or tempfile.mkdtemp(prefix="calibrate_skip_"))
    suites: list[tuple[str, Path, str]] = []
    frames = (
        {}
        if args.reuse
        else {**known_frames(), "large": datagen.large_dataset_mixed(LARGE_ROWS)}
    )
    if args.reuse:
        for d in sorted(p for p in work.iterdir() if p.is_dir() and p.name != "cache"):
            for p in sorted(d.glob("*.yaml")):
                s = yaml.safe_load(p.read_text(encoding="utf-8"))
                if not p.stem.endswith("_original") and any(
                    x != "none" for x in s["clickhouse"].get("skip_index", ["none"])
                ):
                    suites.append((d.name, p, s["dataset"]["column"]))
    for fname, frame in frames.items():
        result = OneShotRecommender().add(frame).result()
        d = work / fname
        d.mkdir()
        to_chbench(result, frame, d)
        n = 0
        for p in sorted(d.glob("*.yaml")):
            if p.stem.endswith("_original"):
                continue
            s = yaml.safe_load(p.read_text(encoding="utf-8"))
            if any(x != "none" for x in s["clickhouse"].get("skip_index", ["none"])):
                prepare_suite(p)
                suites.append((fname, p, s["dataset"]["column"]))
                n += 1
        print(f"{fname}: {n} suites with a chosen skip index", flush=True)

    cache = work / "cache"
    errors = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = (
            []
            if args.reuse
            else [
                pool.submit(run_suite, chbench_root, p, work / f"{p.stem}.db", cache)
                for _, p, _ in suites
            ]
        )
        for i, f in enumerate(futs):
            if err := f.result():
                errors[suites[i][1].stem] = err
            if (i + 1) % 5 == 0:
                print(f"  ran {i + 1}/{len(futs)}", flush=True)
    for n, e in errors.items():
        print(f"ERROR {n}: {e}")

    rows, notes = [], {"unparsed": 0, "marks_mismatch": 0}
    for fname, yp, column in suites:
        db = work / f"{yp.stem}.db"
        if not db.exists() or yp.stem in errors:
            continue
        predicted = json.loads(
            (yp.parent / f"{yp.stem}.predicted.json").read_text(encoding="utf-8")
        )
        r, nt = collect(fname, yp, db, predicted, column)
        rows += r
        for k in notes:
            notes[k] += nt[k]
    if not rows:
        sys.exit("no calibration cells")
    table = pa.table({n: pa.array(c, t) for n, c, t in zip(NAMES, zip(*rows), TYPES)})
    RESULT.parent.mkdir(exist_ok=True)
    with pa.ipc.new_file(RESULT, table.schema) as w:
        w.write_table(table)
    print(
        f"\nwrote {RESULT} ({table.num_rows} cells); suites failed: {len(errors)}; "
        f"queries with unparsed values: {notes['unparsed']}; "
        f"total_marks != ceil(rows/8192): {notes['marks_mismatch']}"
    )
    summarise(pl.from_arrow(table))
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print("kept", work)


def summarise(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=40):
        ctl = df.filter(pl.col("kind") == "none")
        print(
            "\ncontrol (no index): n",
            len(ctl),
            "min measured skip",
            ctl["measured_skip"].min(),
            "max",
            ctl["measured_skip"].max(),
        )
        d = df.filter(pl.col("kind") != "none").with_columns(
            err=(pl.col("predicted_skip") - pl.col("measured_skip")).abs(),
            err_b=(
                pl.col("predicted_skip_bloom_adjusted") - pl.col("measured_skip")
            ).abs(),
            signed=pl.col("measured_skip") - pl.col("predicted_skip"),
        )
        print(
            "\nskip share by kind x G (|predicted - measured|; bloom adj = x(1 - fpr))"
        )
        print(
            d.group_by("kind", "granularity")
            .agg(
                n=pl.len(),
                mean_abs=pl.col("err").mean(),
                max_abs=pl.col("err").max(),
                mean_signed_meas_minus_pred=pl.col("signed").mean(),
                mean_abs_bloom_adj=pl.col("err_b").mean(),
                max_abs_bloom_adj=pl.col("err_b").max(),
            )
            .sort("kind", "granularity")
        )
        print("\nworst skip cells")
        print(
            d.sort("err", descending=True)
            .head(15)
            .select(
                "frame",
                "column",
                "skip_index",
                "value",
                "predicted_skip",
                "predicted_skip_bloom_adjusted",
                "measured_skip",
            )
        )
        t = (
            df.filter(pl.col("kind") != "none")
            .unique(["frame", "column", "skip_index", "type", "codec", "compression"])
            .with_columns(
                rel=(
                    (pl.col("predicted_index_bytes") - pl.col("measured_index_bytes"))
                    / pl.col("measured_index_bytes")
                ),
                diff=pl.col("predicted_index_bytes") - pl.col("measured_index_bytes"),
            )
            .with_columns(abs_rel=pl.col("rel").abs(), abs_diff=pl.col("diff").abs())
        )
        print("\nindex bytes by kind x G")
        print(
            t.group_by("kind", "granularity")
            .agg(
                n=pl.len(),
                median_rel=pl.col("abs_rel").median(),
                max_rel=pl.col("abs_rel").max(),
                max_abs_bytes=pl.col("abs_diff").max(),
                mean_signed_rel=pl.col("rel").mean(),
            )
            .sort("kind", "granularity")
        )
        print("\nworst index-size cells")
        print(
            t.sort("abs_rel", descending=True)
            .head(15)
            .select(
                "frame",
                "column",
                "skip_index",
                "predicted_index_bytes",
                "measured_index_bytes",
                "rel",
            )
        )


if __name__ == "__main__":
    main()
