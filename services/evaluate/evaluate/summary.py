"""summary.arrow (sections size / size_regret / skip / skip_regret / cost / coverage), the printed
report, and compare. Metrics only: no thresholds, no verdicts. Means / medians skip nulls; the
`n_null_*` counts say how many rows had one."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from evaluate.rundir import RunDir

GROUPS = {
    "size": ["type", "codec", "compression"],
    "size_regret": ["recommended_chain"],
    "skip": ["index", "granularity"],
    "skip_regret": ["chosen_index"],
}


def _read(run: RunDir, name: str) -> pl.DataFrame:
    return pl.read_ipc(run.path / f"{name}.arrow")


def _coverage(run: RunDir, notes: dict) -> dict:
    columns, skipped = set(), 0
    datasets = run.info().get("datasets", {})
    for name in datasets:
        d = run.suites(name)
        if (d / "suites.json").is_file():
            columns |= {
                (name, e["column"])
                for e in json.loads((d / "suites.json").read_text(encoding="utf-8"))
            }
        if (d / "skipped.json").is_file():
            skipped += len(json.loads((d / "skipped.json").read_text(encoding="utf-8")))
    cells = _read(run, "size_cells")
    measured = cells.filter(pl.col("measured_bytes").is_not_null())
    return {
        "columns": len(columns),
        "columns_skipped": skipped,
        "cells_measured": measured.height,
        "cells_total": cells.height,
        "cells_skipped_by_clickhouse": cells["skip_reason"].is_not_null().sum(),
        "null_predictions": measured["predicted_bytes"].is_null().sum(),
        "notes": sum(notes.values()),
        "failed_datasets": sum(v.get("status") == "failed" for v in datasets.values()),
    }


def coverage_line(c: dict) -> str:
    return (
        f"columns {c['columns']} (skipped {c['columns_skipped']}) · cells {c['cells_measured']} measured / "
        f"{c['cells_total']} ({c['cells_skipped_by_clickhouse']} skipped by ClickHouse) · "
        f"null predictions {c['null_predictions']} · notes {c['notes']} · failed datasets {c['failed_datasets']}"
    )


def _errors(col: str) -> list[pl.Expr]:
    """n = rows with an error; n_null_error = rows without (unmeasured or unpredicted)."""
    a = pl.col(col).abs()
    return [
        pl.col(col).count().alias("n"),
        pl.col(col).null_count().alias("n_null_error"),
        a.median().alias("median_abs_error"),
        a.quantile(0.9).alias("p90_abs_error"),
        a.max().alias("max_abs_error"),
    ]


def summarise(run: RunDir, notes: dict) -> pl.DataFrame:
    """Aggregate the score files of `run` into summary.arrow (and return it)."""
    size = _read(run, "size_cells").group_by(GROUPS["size"]).agg(_errors("rel_error"))
    size_regret = (
        _read(run, "size_regret")
        .filter(pl.col("type_role") == "recommended")
        .group_by(GROUPS["size_regret"])
        .agg(
            n=pl.len(),
            n_null_regret=pl.col("regret_rel").null_count(),
            regret_rel_sum=pl.col("regret_rel").sum(),
            regret_rel_worst=pl.col("regret_rel").max(),
            type_gain_measured_mean=pl.col("type_gain_measured").mean(),
            n_null_type_gain=pl.col("type_gain_measured").null_count(),
        )
    )
    skip = (
        _read(run, "skip_queries").group_by(GROUPS["skip"]).agg(_errors("skip_error"))
    )
    skip_regret = (
        _read(run, "skip_regret")
        .group_by(GROUPS["skip_regret"])
        .agg(
            n=pl.len(),
            n_null_regret=pl.col("regret_freq").null_count(),
            regret_freq_mean=pl.col("regret_freq").mean(),
            regret_freq_worst=pl.col("regret_freq").max(),
            regret_uniform_mean=pl.col("regret_uniform").mean(),
            regret_uniform_worst=pl.col("regret_uniform").max(),
        )
    )
    cost = _read(run, "recommend_cost").select(
        n=pl.len(),
        n_missing=(pl.col("status") == "missing").sum(),
        wall_s=pl.col("wall_s").sum(),
        cpu_s=pl.col("cpu_s").sum(),
        peak_mb=pl.col("peak_mb").max(),
        rows=pl.col("rows").sum(),
    )
    sections = {
        "size": size,
        "size_regret": size_regret,
        "skip": skip,
        "skip_regret": skip_regret,
        "cost": cost,
        "coverage": pl.DataFrame([_coverage(run, notes)]),
    }
    parts = [
        (df.sort(GROUPS[name]) if name in GROUPS else df).with_columns(
            section=pl.lit(name)
        )
        for name, df in sections.items()
    ]
    out = pl.concat(parts, how="diagonal_relaxed").select(
        "section", pl.exclude("section")
    )
    out.write_ipc(run.path / "summary.arrow")
    return out


def report(run: RunDir) -> None:
    s = pl.read_ipc(run.path / "summary.arrow")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=40):
        for section in ("size", "size_regret", "skip", "skip_regret", "cost"):
            part = s.filter(pl.col("section") == section)
            if part.is_empty():
                print(f"\n{section}\n(no rows)")
                continue
            part = part.select(
                [c for c in part.columns if part[c].null_count() < part.height]
            )
            print(f"\n{section}\n{part}")
    cov = s.filter(pl.col("section") == "coverage").row(0, named=True)
    print("\n" + coverage_line(cov))


def compare(a: Path, b: Path) -> pl.DataFrame:
    """b - a for every numeric summary metric, joined on section + group keys."""
    sa = pl.read_ipc(Path(a) / "summary.arrow")
    sb = pl.read_ipc(Path(b) / "summary.arrow")
    keys = ["section", *dict.fromkeys(k for g in GROUPS.values() for k in g)]
    keys = [k for k in keys if k in sa.columns and k in sb.columns]
    metrics = [
        c
        for c in sa.columns
        if c not in keys and sa[c].dtype.is_numeric() and c in sb.columns
    ]
    j = sa.join(sb, on=keys, how="full", coalesce=True, suffix="_b", nulls_equal=True)
    return j.select(
        *keys,
        *[
            (pl.col(f"{m}_b").cast(pl.Float64) - pl.col(m).cast(pl.Float64)).alias(
                f"{m}_delta"
            )
            for m in metrics
        ],
    )
