"""Narrow, wide and nested frames for describe-style workloads."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from synthdata.scenarios.zoo import _WORDS


def describe_narrow(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """int, float, numeric-string and ISO-datetime-string columns (benchmarks)."""
    rng = np.random.default_rng(seed)
    ints = rng.integers(0, 1_000_000, n_rows)
    return pl.DataFrame(
        {"int": ints, "float": np.round(rng.normal(100.0, 15.0, n_rows), 2)}
    ).with_columns(
        pl.col("int").cast(pl.String).alias("num_str"),
        (pl.datetime(2024, 1, 1) + pl.duration(seconds=pl.col("int") * 30))
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        .alias("iso_str"),
    )


def describe_wide(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """n_cols columns c000… cycling Int64 (low cardinality) / Float64 / String / Date."""
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(n_cols):
        kind = i % 4
        if kind == 0:
            cols.append(pl.Series(f"c{i:03d}", rng.integers(0, 50, n_rows)))
        elif kind == 1:
            cols.append(
                pl.Series(f"c{i:03d}", np.round(rng.normal(0.0, 1.0, n_rows), 3))
            )
        elif kind == 2:
            cols.append(pl.Series(f"c{i:03d}", _WORDS[rng.integers(0, 5, n_rows)]))
        else:
            cols.append(
                pl.Series(f"c{i:03d}", rng.integers(19_000, 20_000, n_rows))
                .cast(pl.Int32)
                .cast(pl.Date)
            )
    return pl.DataFrame(cols)


def describe_nested(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """List(Int64), List(String) (0–4 elements) and Struct{a: Int64, b: String} columns."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 5, n_rows)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = rng.integers(0, 100, int(offsets[-1]))
    list_i64 = pl.from_arrow(
        pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(values))
    )
    return pl.DataFrame(
        [
            list_i64.alias("list_i64"),
            list_i64.cast(pl.List(pl.String)).alias("list_str"),
            pl.DataFrame(
                {
                    "a": rng.integers(0, 3, n_rows),
                    "b": _WORDS[rng.integers(0, 2, n_rows)],
                }
            ).to_struct("struct"),
        ]
    )
