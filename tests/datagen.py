"""
Seeded toy-data generators shared by accuracy tests (small sizes) and speed
benchmarks (large sizes). Every generator is deterministic for a given seed.

Ordered techniques:   mixed_dtypes, low_cardinality
Multi-set techniques: related_frames, similar_frames
Per-column (GCD):     integer_multiples, integer_random
Per-column (describe): describe_mixed, stringified (tests); describe_narrow, describe_wide, describe_nested (benchmarks)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa


def _with_nulls(name: str, values: np.ndarray, null_mask: np.ndarray) -> pl.Series:
    return pl.from_arrow(pa.array(values, mask=null_mask)).alias(name)


# ── ordered techniques ────────────────────────────────────────────────────────


def _skewed_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """30 tokens drawn with power-law weights (heavy skew, ~30 distinct values)."""
    k = np.arange(1, 31)
    w = 1.0 / k**0.8
    w /= w.sum()
    return rng.choice(30, size=n, p=w)


def _high_unique_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """All n tokens distinct (random permutation)."""
    return rng.permutation(n)


def _sparse_tokens(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """200 unique tokens at random positions; remaining positions forced null.
    Returns (tokens, null_mask) — tokens[i] is only valid when null_mask[i]=False.
    """
    tokens = np.empty(n, dtype=np.int64)
    positions = rng.choice(n, size=min(200, n), replace=False)
    tokens[positions] = np.arange(len(positions))
    null_mask = np.ones(n, dtype=bool)
    null_mask[positions] = False
    return tokens, null_mask


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


# ── multi-set techniques ──────────────────────────────────────────────────────


def related_frames(n_rows: int = 1_000, seed: int = 42) -> dict[str, pl.DataFrame]:
    """Three frames with planted relationships (enumeration order customers, orders, archive):

    customers.id     ~ orders.customer_id    pk_fk   (FK, 10% null)
    customers.id     ~ archive.id            pk_pk   (same id set, shuffled)
    customers.name   ~ orders.customer_name  pk_fk
    customers.name   ~ archive.name          ~93% shared - similar, not contained
    customers.region ~ orders.region         b_in_a  (orders uses 2 of 4 regions)
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_rows) + 1_000_000
    names = [f"cust_{i}" for i in ids]
    regions = np.array(["north", "south", "east", "west"])
    customers = pl.DataFrame(
        {
            "id": ids,
            "name": names,
            "region": rng.choice(regions, n_rows),
            "score": rng.integers(0, 100, n_rows),
        }
    )
    pick = rng.integers(0, n_rows, n_rows)
    orders = pl.DataFrame(
        [
            pl.Series("order_id", np.arange(n_rows) + 5_000_000),
            _with_nulls("customer_id", ids[pick], rng.random(n_rows) < 0.10),
            pl.Series("customer_name", [names[i] for i in pick]),
            pl.Series("region", rng.choice(regions[:2], n_rows)),
            pl.Series("amount", rng.integers(1, 500, n_rows)),
        ]
    )
    archive_names = [names[i] for i in rng.permutation(n_rows)]
    for i in np.flatnonzero(rng.random(n_rows) < 0.07):
        archive_names[i] = f"new_{i}"
    archive = pl.DataFrame({"id": rng.permutation(ids), "name": archive_names})
    return {"customers": customers, "orders": orders, "archive": archive}


def containment_pairs(
    n_pairs: int = 20, small_size: int = 5, large_size: int = 200, seed: int = 42
) -> dict[str, pl.DataFrame]:
    """`n_pairs` column pairs (small.small_i, large.large_i) where the small column's
    values are a subset of the large column's — overlap == 1.0 (fully contained) but
    Jaccard = small_size / large_size (~0.025 at the defaults), far below the MinHash
    LSH candidate threshold (~0.43). Each pair draws from its own disjoint numeric
    range, so off-diagonal pairs (small_i vs large_j, i != j; small_i vs small_j) are
    ~disjoint and contribute no accidental truth positives.

    Exercises the documented containment-recall gap on MinHashRust/MinHashDatasketch
    (CLAUDE.md Next Steps / I2): the LSH candidate threshold is Jaccard-calibrated, so
    a small-in-large containment pair with low Jaccard is pruned before verification
    ever runs, even though its Overlap Coefficient is 1.0.
    """
    rng = np.random.default_rng(seed)
    span = large_size * 10
    small_cols, large_cols = {}, {}
    for i in range(n_pairs):
        base = i * span
        pool = rng.choice(np.arange(base, base + span), size=large_size, replace=False)
        small = rng.choice(pool, size=small_size, replace=False)
        small_cols[f"small_{i:02d}"] = small.tolist()
        large_cols[f"large_{i:02d}"] = pool.tolist()
    return {"small": pl.DataFrame(small_cols), "large": pl.DataFrame(large_cols)}


def similar_frames(
    n_similar: int = 8,
    n_independent: int = 12,
    col_size: int = 200,
    n_elements: int = 2_000,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Two frames each with (n_similar + n_independent) columns. Each of the n_similar
    cross-frame pairs sim_ii shares >= 93% of elements (OC above 0.9). Independent
    columns come from disjoint regions of the element space to minimise accidental
    similarity. Shorter columns are null-padded."""
    rng = np.random.default_rng(seed)
    df0: dict[str, list] = {}
    df1: dict[str, list] = {}
    for i in range(n_similar):
        base = rng.choice(n_elements, size=col_size, replace=False).tolist()
        shared = int(col_size * 0.93)
        extra = rng.integers(0, n_elements, size=col_size - shared).tolist()
        df0[f"sim_{i:02d}"] = base
        df1[f"sim_{i:02d}"] = base[:shared] + extra
    region = n_elements // max(n_independent, 1)
    for i in range(n_independent):
        lo = (i * region) % n_elements
        pool = list(range(lo, min(lo + region, n_elements))) or list(range(n_elements))
        size = int(min(len(pool), rng.integers(50, col_size)))
        df0[f"ind0_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()
        df1[f"ind1_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()

    def to_frame(cols: dict[str, list]) -> pl.DataFrame:
        longest = max(len(v) for v in cols.values())
        return pl.DataFrame(
            {k: v + [None] * (longest - len(v)) for k, v in cols.items()}
        )

    return {"df0": to_frame(df0), "df1": to_frame(df1)}


# ── per-column (GCD) ──────────────────────────────────────────────────────────


def integer_multiples(n_rows: int, n_cols: int, g: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… whose values are k·g, k uniform in [-2**40, 2**40)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * g
            for i in range(n_cols)
        }
    )


def integer_random(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… of random values; whole-column GCD is 1 almost surely (early exit)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64)
            for i in range(n_cols)
        }
    )


# ── per-column (describe) ─────────────────────────────────────────────────────

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
