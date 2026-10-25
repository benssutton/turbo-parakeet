"""Per-dtype columns, mixed-dtype and low-cardinality frames, describe fixtures."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from synthdata.scenarios.entropy_extremes import (
    _high_unique_tokens,
    _skewed_tokens,
    _sparse_tokens,
)


def _with_nulls(name: str, values: np.ndarray, null_mask: np.ndarray) -> pl.Series:
    return pl.from_arrow(pa.array(values, mask=null_mask)).alias(name)


def make_column(
    name: str,
    dtype_spec: str,
    shape: str,
    n_rows: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Build one pl.Series with the given dtype and distribution shape.

    dtype_spec : "float64" | "uint32" | "boolean" | "categorical" | "list" | "arr"
    shape      : "skewed" | "high_unique" | "sparse"
    """
    if shape == "sparse":
        tokens, null_mask = _sparse_tokens(n_rows, rng)
    else:
        tokens = (
            _skewed_tokens(n_rows, rng)
            if shape == "skewed"
            else _high_unique_tokens(n_rows, rng)
        )
        null_rate = 0.10 if shape == "skewed" else 0.05
        null_mask = rng.random(n_rows) < null_rate

    values: list = []
    for i in range(n_rows):
        if null_mask[i]:
            values.append(None)
            continue
        t = int(tokens[i])
        if dtype_spec == "float64":
            values.append(float(t))
        elif dtype_spec == "uint32":
            values.append(t)
        elif dtype_spec == "boolean":
            values.append(bool(t % 2))
        elif dtype_spec == "categorical":
            values.append(f"cat_{t}")
        elif dtype_spec == "list":
            values.append([t, t + 1])
        else:  # arr
            values.append([t, t + 1, t + 2])

    if dtype_spec == "float64":
        return pl.Series(name, values, dtype=pl.Float64)
    if dtype_spec == "uint32":
        return pl.Series(name, values, dtype=pl.UInt32)
    if dtype_spec == "boolean":
        return pl.Series(name, values, dtype=pl.Boolean)
    if dtype_spec == "categorical":
        return pl.Series(name, values, dtype=pl.String).cast(pl.Categorical)
    if dtype_spec == "list":
        return pl.Series(name, values, dtype=pl.List(pl.Int32))
    return pl.Series(name, values, dtype=pl.Array(pl.Int32, 3))


def mixed_dtypes(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame:
    """18 columns (6 dtypes × 3 shapes) named {dtype_spec}_{shape}, e.g. categorical_sparse."""
    rng = np.random.default_rng(seed)
    cols = []
    for dtype_spec in ("float64", "uint32", "boolean", "categorical", "list", "arr"):
        for shape in ("skewed", "high_unique", "sparse"):
            cols.append(
                make_column(f"{dtype_spec}_{shape}", dtype_spec, shape, n_rows, rng)
            )
    return pl.DataFrame(cols)


def low_cardinality(
    n_rows: int,
    n_cols: int,
    cardinality: int = 20,
    null_rate: float = 0.05,
    seed: int = 42,
) -> pl.DataFrame:
    """Int64 columns c000… with `cardinality` values each. Every 5th column is a noisy
    copy of the previous one, so some pairs are genuinely associated."""
    rng = np.random.default_rng(seed)
    cols, prev = [], None
    for i in range(n_cols):
        if prev is not None and i % 5 == 4:
            values = (prev + rng.integers(0, 2, n_rows)) % cardinality
        else:
            values = rng.integers(0, cardinality, n_rows)
        prev = values
        cols.append(_with_nulls(f"c{i:03d}", values, rng.random(n_rows) < null_rate))
    return pl.DataFrame(cols)


_WORDS = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])


def describe_mixed(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame:
    """One column per dtype family `describe` profiles, with ~10% nulls, float
    specials (±0, NaN, ±inf, 0.1, 1e-7, 1.5e20), strings for the numeric and ISO
    scanners (incl. leading zeros and mixed offsets), nested and all-null columns."""
    rng = np.random.default_rng(seed)
    n = n_rows

    def nulls(values, rate: float = 0.1) -> list:
        mask = rng.random(n) < rate
        return [None if m else v for v, m in zip(values, mask)]

    ints = rng.integers(-1_000, 1_000, n)
    floats = np.round(rng.normal(100.0, 15.0, n), 2)
    specials = rng.choice(
        np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 0.1, 1e-7, 1.5e20]), n
    )
    days = rng.integers(0, 366, n)
    micros = (
        rng.integers(0, 86_400, n) * 1_000_000 * (rng.random(n) < 0.5)
    )  # ~half at midnight
    datetimes = [
        datetime(2024, 1, 1) + timedelta(days=int(d), microseconds=int(u))
        for d, u in zip(days, micros)
    ]
    offsets = np.array(["Z", "+02:00", "-05:30"])
    lists = [
        (
            None
            if i % 11 == 0
            else (
                []
                if i % 13 == 0
                else [int(x) for x in rng.integers(0, 20, rng.integers(1, 4))]
            )
        )
        for i in range(n)
    ]

    def words():
        return _WORDS[rng.integers(0, 5, n)]

    return pl.DataFrame(
        [
            pl.Series("i8", nulls(rng.integers(-128, 128, n).tolist()), dtype=pl.Int8),
            pl.Series("i16", nulls(ints.tolist()), dtype=pl.Int16),
            pl.Series("i32", nulls((ints * 1_000).tolist()), dtype=pl.Int32),
            pl.Series(
                "i64", nulls((ints.astype(np.int64) * 10**12).tolist()), dtype=pl.Int64
            ),
            pl.Series(
                "d38", nulls([int(v) * 10**25 for v in ints]), dtype=pl.Decimal(38, 0)
            ),
            pl.Series("u8", nulls(rng.integers(0, 256, n).tolist()), dtype=pl.UInt8),
            pl.Series("u16", rng.integers(0, 65_536, n).tolist(), dtype=pl.UInt16),
            pl.Series(
                "u32", nulls(rng.integers(0, 2**32, n).tolist()), dtype=pl.UInt32
            ),
            pl.Series(
                "u64",
                nulls(([2**64 - 1] + rng.integers(0, 2**63, n).tolist())[:n]),
                dtype=pl.UInt64,
            ),
            pl.Series("codes", rng.integers(0, 5, n).tolist(), dtype=pl.Int64),
            pl.Series("f32", nulls(floats.tolist()), dtype=pl.Float32),
            pl.Series("f64", nulls(specials.tolist()), dtype=pl.Float64),
            pl.Series("f64_price", nulls(floats.tolist()), dtype=pl.Float64),
            pl.Series("f64_whole", np.arange(n, dtype=np.float64)),
            pl.Series(
                "dec",
                nulls([Decimal(f"{v:.2f}") for v in floats]),
                dtype=pl.Decimal(10, 2),
            ),
            pl.Series("bool", nulls((ints > 0).tolist()), dtype=pl.Boolean),
            pl.Series(
                "date",
                nulls([date(2024, 1, 1) + timedelta(days=int(d)) for d in days]),
                dtype=pl.Date,
            ),
            pl.Series("dt_naive", nulls(datetimes), dtype=pl.Datetime("us")),
            pl.Series(
                "dt_tz", nulls(datetimes), dtype=pl.Datetime("us")
            ).dt.replace_time_zone(
                "Europe/London", ambiguous="earliest", non_existent="null"
            ),
            pl.Series(
                "dur",
                nulls([timedelta(seconds=int(v)) for v in ints]),
                dtype=pl.Duration("us"),
            ),
            pl.Series(
                "time",
                nulls(
                    [
                        time(int(v) // 3600, int(v) // 60 % 60, int(v) % 60)
                        for v in rng.integers(0, 86_400, n)
                    ]
                ),
                dtype=pl.Time,
            ),
            pl.Series("str_free", nulls([f"{w} {v}" for w, v in zip(words(), ints)])),
            pl.Series("str_int", nulls([str(v) for v in ints])),
            pl.Series("str_lead", nulls([f"{abs(v):05d}" for v in ints])),
            pl.Series("str_dec", nulls([f"{v:.2f}" for v in floats])),
            pl.Series(
                "str_date",
                nulls([str(date(2024, 1, 1) + timedelta(days=int(d))) for d in days]),
            ),
            pl.Series(
                "str_dt", nulls([d.strftime("%Y-%m-%dT%H:%M:%S.%f") for d in datetimes])
            ),
            pl.Series(
                "str_dt_tz",
                nulls(
                    [
                        d.strftime("%Y-%m-%dT%H:%M:%S") + o
                        for d, o in zip(datetimes, offsets[rng.integers(0, 3, n)])
                    ]
                ),
            ),
            pl.Series("cat", nulls(words().tolist()), dtype=pl.Categorical),
            pl.Series("enum", nulls(words().tolist()), dtype=pl.Enum(_WORDS.tolist())),
            pl.Series("bin", nulls([w.encode() for w in words()]), dtype=pl.Binary),
            pl.Series("list_i64", lists, dtype=pl.List(pl.Int64)),
            pl.Series(
                "list_str",
                [None if v is None else [str(x) for x in v] for v in lists],
                dtype=pl.List(pl.String),
            ),
            pl.Series(
                "arr_i32",
                nulls([[int(x) for x in rng.integers(0, 10, 3)] for _ in range(n)]),
                dtype=pl.Array(pl.Int32, 3),
            ),
            pl.Series(
                "struct",
                nulls(
                    [
                        {"a": int(a), "b": str(b)}
                        for a, b in zip(
                            rng.integers(0, 3, n), _WORDS[rng.integers(0, 2, n)]
                        )
                    ]
                ),
            ),
            pl.Series("all_null", [None] * n, dtype=pl.String),
        ]
    )


def _castable(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or isinstance(
        dtype, (pl.Date, pl.Datetime, pl.Time, pl.Boolean)
    )


def stringified(frame: pl.DataFrame) -> pl.DataFrame:
    """Every non-string column Polars can cast to String, cast to String
    (List(<castable>) → List(String)). Other columns are dropped: string-like,
    Duration (Polars cannot cast it to String), Binary, Array, Struct, all-null."""
    out = []
    for name, dtype in frame.schema.items():
        if _castable(dtype):
            out.append(frame[name].cast(pl.String))
        elif isinstance(dtype, pl.List) and _castable(dtype.inner):
            out.append(frame[name].cast(pl.List(pl.String)))
    return pl.DataFrame(out)
