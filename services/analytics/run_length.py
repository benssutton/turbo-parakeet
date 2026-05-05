"""
Run-length encoding analysis for Apache Arrow REE compression assessment.

For each column, walks values in physical order, identifies spans of consecutive
identical values (treating consecutive nulls as a single run), and quantifies
the runs by size.
"""
from typing import Sequence

import polars as pl


def column_run_stats(
    df: pl.DataFrame | pl.LazyFrame,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Compute per-column run-length statistics for assessing REE compression benefit.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames are collected.
    columns : sequence of str, optional
        Specific columns to analyze. Defaults to all columns.

    Returns
    -------
    pl.DataFrame
        One row per column with fields:
        - col_name: String           - column name
        - n_rows: UInt64             - total rows
        - n_runs: UInt64             - run count (column's REE length)
        - compression_ratio: Float64 - n_rows / n_runs
        - mean_run_length: Float64
        - max_run_length: UInt64
        - run_length_histogram: List[Struct{size: UInt64, count: UInt64}]
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    cols = list(columns) if columns is not None else df.columns
    n_rows = df.height

    rows = []
    for col in cols:
        # rle() returns Struct{len: UInt32, value: T}; consecutive nulls are
        # collapsed into one run, matching REE null-bitmap semantics.
        lengths = df.select(
            pl.col(col).rle().struct.field("len").alias("len")
        )["len"]

        n_runs = lengths.len()

        if n_runs == 0:
            rows.append({
                "col_name": col,
                "n_rows": n_rows,
                "n_runs": 0,
                "compression_ratio": float("nan"),
                "mean_run_length": 0.0,
                "max_run_length": 0,
                "run_length_histogram": [],
            })
            continue

        hist_df = (
            lengths.to_frame()
            .group_by("len")
            .agg(pl.len().alias("count"))
            .sort("len")
        )
        histogram = [
            {"size": int(r["len"]), "count": int(r["count"])}
            for r in hist_df.iter_rows(named=True)
        ]

        rows.append({
            "col_name": col,
            "n_rows": n_rows,
            "n_runs": n_runs,
            "compression_ratio": n_rows / n_runs,
            "mean_run_length": float(lengths.mean()),
            "max_run_length": int(lengths.max()),
            "run_length_histogram": histogram,
        })

    return pl.DataFrame(
        rows,
        schema={
            "col_name": pl.String,
            "n_rows": pl.UInt64,
            "n_runs": pl.UInt64,
            "compression_ratio": pl.Float64,
            "mean_run_length": pl.Float64,
            "max_run_length": pl.UInt64,
            "run_length_histogram": pl.List(
                pl.Struct({"size": pl.UInt64, "count": pl.UInt64})
            ),
        },
    )
