"""The columns where a faster chain beats the recommended one on full-scan CPU (FINDINGS §6).

Per recommended-type size suite of a scored run: the cell with the lowest full-scan CPU median, kept when it is not
the recommended chain and its paired 95% cpu_ratio interval (not adjusted) lies below 1. Prints each with the extra
bytes and the CPU saved against the recommended chain, then the medians and the faster chains' counts. No server.

    python services/evaluate/scripts/faster_chains.py services/evaluate/runs/<run>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def _chain(codec: str, compression: str) -> pl.Expr:
    return (
        pl.when(pl.col(codec) == "none")
        .then(pl.col(compression))
        .otherwise(pl.concat_str(codec, pl.lit(", "), compression))
    )


def faster_chains(run_dir: Path) -> pl.DataFrame:
    f = pl.read_ipc(Path(run_dir) / "size_frontier.arrow").filter(
        pl.col("measured_bytes").is_not_null(),
        # `recommended` is set on recommended-type suites only
        pl.col("recommended").any().over("suite"),
    )
    rec = f.filter("recommended").select(
        "suite",
        rec_chain=_chain("codec", "compression"),
        rec_bytes="measured_bytes",
        rec_cpu_ms=pl.col("cpu_us_p50") / 1e3,
    )
    fastest = (
        f.sort("cpu_us_p50")
        .group_by("suite")
        .first()
        .filter(~pl.col("recommended"), pl.col("cpu_ratio_hi") < 1)
    )
    return (
        fastest.join(rec, on="suite")
        .select(
            "dataset",
            "column",
            "rec_chain",
            fast_chain=_chain("codec", "compression"),
            rec_bytes="rec_bytes",
            fast_bytes="measured_bytes",
            more_bytes_pct=(pl.col("measured_bytes") / pl.col("rec_bytes") - 1) * 100,
            cpu_saved_pct=(1 - pl.col("cpu_ratio")) * 100,
            rec_cpu_ms="rec_cpu_ms",
        )
        .sort("dataset", "column")
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.reconfigure(encoding="utf-8")
    t = faster_chains(Path(argv[0]))
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        fmt_str_lengths=40,
        tbl_width_chars=250,
        float_precision=1,
        tbl_hide_dataframe_shape=True,
    ):
        print(t)
        print(
            f"{t.height} columns; median +{t['more_bytes_pct'].median():.0f}% bytes, "
            f"-{t['cpu_saved_pct'].median():.0f}% CPU"
        )
        print(t.group_by("fast_chain").len().sort("len", "fast_chain", descending=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
