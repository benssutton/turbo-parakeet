"""
Whole-column GCD correctness tests: Rust plugin vs off-the-shelf references.

References:
    math.gcd(*values)  — primary oracle; arbitrary precision covers Int128,
                         Decimal, u64::MAX and i64::MIN magnitudes.
    numpy.gcd.reduce   — independent second oracle for ≤64-bit physical values
                         (skipped where a MIN magnitude can't be represented).

Semantics under test (ClickHouse GCD-codec method): GCD of the magnitudes of
the raw physical integer values; nulls skipped; all-null / all-zero / zero-row
→ 0; non-integer-backed dtypes (incl. Categorical/Enum) → null; a magnitude of
2**127 (only i128::MIN) → null.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import decimal
import math
import random
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

_analytics = pytest.importorskip("analytics")
column_gcd = _analytics.column_gcd

CHUNK = 1 << 16  # gcd.rs parallel chunk size
I128_LIMIT = 2**127
_DEC_CTX = decimal.Context(prec=80)

UNSIGNED = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plugin_gcds(df: pl.DataFrame | pl.LazyFrame) -> dict[str, int | None]:
    out = column_gcd(df).unnest("column_gcd")
    return dict(zip(out["column"].to_list(), out["gcd"].to_list()))


def plugin_gcd(s: pl.Series) -> int | None:
    return plugin_gcds(s.to_frame())[s.name]


def physical_dtype(dtype: pl.DataType) -> pl.DataType:
    if dtype == pl.Date:
        return pl.Int32()
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.Int64()
    return dtype


def from_physical(name: str, ints: list[int | None], dtype: pl.DataType) -> pl.Series:
    """Series of `dtype` whose physical values are exactly `ints` (None = null)."""
    if isinstance(dtype, pl.Decimal):
        vals = [None if v is None else Decimal(v).scaleb(-dtype.scale, context=_DEC_CTX) for v in ints]
        return pl.Series(name, vals, dtype=dtype)
    return pl.Series(name, ints, dtype=physical_dtype(dtype)).cast(dtype)


def ref_gcd(s: pl.Series) -> int | None:
    """math.gcd over physical values, with the plugin's 2**127 → null rule."""
    g = math.gcd(*s.to_physical().drop_nulls().to_list())
    return None if g >= I128_LIMIT else g


def numpy_gcd(s: pl.Series) -> int:
    return int(np.gcd.reduce(s.to_physical().drop_nulls().to_numpy()))


def numpy_applicable(s: pl.Series) -> bool:
    """numpy has no 128-bit ints, and |MIN| overflows its own dtype."""
    phys = s.to_physical()
    if phys.dtype in (pl.Int128,) or isinstance(s.dtype, pl.Decimal):
        return False
    if phys.dtype in UNSIGNED:
        return True
    bits = {pl.Int8: 8, pl.Int16: 16, pl.Int32: 32, pl.Int64: 64}[phys.dtype.base_type()]
    return -(2 ** (bits - 1)) not in phys.drop_nulls().to_list()


def is_signed(dtype: pl.DataType) -> bool:
    return not (isinstance(dtype, UNSIGNED) or dtype == pl.Time)


# (dtype, g, k_lo, k_hi): physical values are k·g, k ∈ [k_lo, k_hi], always incl. k=1.
CASES = [
    pytest.param(pl.Int8(), 4, -32, 31, id="Int8"),
    pytest.param(pl.Int16(), 12, -2_000, 2_000, id="Int16"),
    pytest.param(pl.Int32(), 1_000, -2_000_000, 2_000_000, id="Int32"),
    pytest.param(pl.Int64(), 3_600, -(2**40), 2**40, id="Int64"),
    pytest.param(pl.Int128(), 10**20, -(10**15), 10**15, id="Int128"),
    pytest.param(pl.UInt8(), 5, 0, 51, id="UInt8"),
    pytest.param(pl.UInt16(), 12, 0, 5_000, id="UInt16"),
    pytest.param(pl.UInt32(), 1_000, 0, 4_000_000, id="UInt32"),
    pytest.param(pl.UInt64(), 2**40, 0, 2**23, id="UInt64"),
    pytest.param(pl.Decimal(38, 4), 25, -(10**30), 10**30, id="Decimal38_4"),
    pytest.param(pl.Date(), 7, -5_000, 5_000, id="Date"),
    pytest.param(pl.Datetime("ms"), 60_000, -(10**6), 10**6, id="Datetime_ms"),
    pytest.param(pl.Datetime("us"), 3_600_000_000, -(10**5), 10**5, id="Datetime_us"),
    pytest.param(pl.Datetime("ns", "UTC"), 86_400 * 10**9, -(10**4), 10**4, id="Datetime_ns_UTC"),
    pytest.param(pl.Duration("us"), 250, -(10**9), 10**9, id="Duration_us"),
    pytest.param(pl.Time(), 15 * 60 * 10**9, 0, 95, id="Time"),
]
ALL_DTYPES = [pytest.param(p.values[0], id=p.id) for p in CASES]


# ─────────────────────────────────────────────────────────────────────────────
# Values
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype, g, k_lo, k_hi", CASES)
def test_multiples_of_known_gcd(dtype, g, k_lo, k_hi):
    rng = random.Random(1234)
    ints = [g] + [
        None if rng.random() < 0.1 else rng.randint(k_lo, k_hi) * g for _ in range(1_000)
    ]
    s = from_physical("x", ints, dtype)
    got = plugin_gcd(s)
    assert got == ref_gcd(s) == g
    if numpy_applicable(s):
        assert got == numpy_gcd(s)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_coprime_values_give_one(dtype):
    s = from_physical("x", [6, 10, 15], dtype)
    assert plugin_gcd(s) == ref_gcd(s) == 1


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_single_value_is_its_magnitude(dtype):
    v = -42 if is_signed(dtype) else 42
    s = from_physical("x", [v], dtype)
    assert plugin_gcd(s) == ref_gcd(s) == 42


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_zeros_and_nulls(dtype):
    assert plugin_gcd(from_physical("x", [0, 0, 12, None, 18, 0], dtype)) == 6
    assert plugin_gcd(from_physical("x", [0, 0, 0], dtype)) == 0
    assert plugin_gcd(from_physical("x", [None, None], dtype)) == 0
    assert plugin_gcd(from_physical("x", [None, 12, None, 18, None], dtype)) == 6


@pytest.mark.parametrize(
    "dtype, ints, expected",
    [
        pytest.param(pl.Int8(), [-128], 128, id="i8_min"),
        pytest.param(pl.Int8(), [-128, 64], 64, id="i8_min_and_64"),
        pytest.param(pl.UInt64(), [2**64 - 1], 2**64 - 1, id="u64_max"),
        pytest.param(pl.Int64(), [-(2**63)], 2**63, id="i64_min"),
        pytest.param(pl.Int64(), [-(2**63), 2**62], 2**62, id="i64_min_and_2^62"),
        pytest.param(pl.Int128(), [2**127 - 1], 2**127 - 1, id="i128_max"),
        pytest.param(pl.Int128(), [-(2**127), 2**126], 2**126, id="i128_min_and_2^126"),
        pytest.param(pl.Int128(), [-(2**127)], None, id="i128_min_unrepresentable"),
        pytest.param(pl.Int128(), [-(2**127), None, 0], None, id="i128_min_with_null_zero"),
    ],
)
def test_dtype_extremes(dtype, ints, expected):
    s = from_physical("x", ints, dtype)
    assert plugin_gcd(s) == ref_gcd(s) == expected


def test_zero_row_frame():
    schema = {f"c{i}": p.values[0] for i, p in enumerate(CASES)} | {"s": pl.String()}
    got = plugin_gcds(pl.DataFrame(schema=schema))
    assert got == {**{f"c{i}": 0 for i in range(len(CASES))}, "s": None}


def test_zero_column_frame():
    out = column_gcd(pl.DataFrame())
    assert out.height == 0
    assert out.schema == pl.Schema(
        {"column_gcd": pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arrow layout
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_chunk_series():
    parts = [pl.Series("x", [12, 24]), pl.Series("x", [None, 36]), pl.Series("x", [18])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    assert plugin_gcd(s) == ref_gcd(s) == 6


def test_long_series_crosses_parallel_chunks():
    n = 3 * CHUNK + 17
    values = np.full(n, 12, dtype=np.int64)
    values[2 * CHUNK + 5] = 18
    mask = np.arange(n) % 7 == 3  # scattered nulls
    assert not mask[2 * CHUNK + 5]  # the spoiler stays valid
    s = pl.from_arrow(pa.array(values, mask=mask))
    assert plugin_gcd(s) == ref_gcd(s) == numpy_gcd(s) == 6


def test_null_payloads_ignored():
    # 1s live under null slots. pyarrow keeps the payload bytes; a correct
    # null mask must ignore them — otherwise the GCD collapses to 1, and a
    # masked 1 would also wrongly trigger the gcd == 1 early exit.
    values = np.tile(np.array([12, 1, 18, 1], dtype=np.int64), 50_000)
    mask = values == 1
    arr = pa.array(values, mask=mask)
    assert np.frombuffer(arr.buffers()[1], dtype=np.int64)[1] == 1  # payload really is there
    s = pl.from_arrow(arr)
    assert plugin_gcd(s) == ref_gcd(s) == 6


def test_sliced_series():
    # Leading 7s are sliced away; offsets are deliberately not multiples of 8
    # so the validity bitmap has a sub-byte offset.
    n = 2 * CHUNK + 100
    values = np.full(n, 12, dtype=np.int64)
    values[:5] = 7
    values[CHUNK + 1] = 18
    mask = np.zeros(n, dtype=bool)
    mask[CHUNK + 2 :: 11] = True
    values[mask] = 7  # payloads under nulls
    s = pl.from_arrow(pa.array(values, mask=mask)).slice(5, 2 * CHUNK + 50)
    assert plugin_gcd(s) == ref_gcd(s) == 6
    s3 = pl.from_arrow(pa.array(values, mask=mask)).slice(3)  # keeps two leading 7s
    assert plugin_gcd(s3) == ref_gcd(s3) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Physical-unit results
# ─────────────────────────────────────────────────────────────────────────────

def test_hourly_datetime_us():
    s = pl.datetime_range(datetime(2024, 1, 1, 7), datetime(2024, 1, 3), "1h", time_unit="us", eager=True)
    assert plugin_gcd(s) == ref_gcd(s) == 3_600_000_000


def test_decimal_quarter_steps():
    s = pl.Series("p", [Decimal("1.25"), Decimal("0.50"), Decimal("0.75"), Decimal("-2.00")], dtype=pl.Decimal(10, 2))
    assert plugin_gcd(s) == ref_gcd(s) == 25


def test_weekly_dates():
    s = pl.date_range(date(2024, 1, 1), date(2024, 6, 30), "1w", eager=True)
    assert plugin_gcd(s) == ref_gcd(s) == numpy_gcd(s)  # raw epoch days, not the 7-day step


# ─────────────────────────────────────────────────────────────────────────────
# Output contract
# ─────────────────────────────────────────────────────────────────────────────

def test_non_integer_dtypes_are_null():
    df = pl.DataFrame(
        {
            "f64": pl.Series([2.0, 4.0], dtype=pl.Float64),
            "f32": pl.Series([2.0, 4.0], dtype=pl.Float32),
            "str": ["a", "b"],
            "bool": [True, False],
            "cat": pl.Series(["x", "y"], dtype=pl.Categorical),
            "enum": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
            "list": [[2, 4], [6]],
            "arr": pl.Series([[2, 4], [6, 8]], dtype=pl.Array(pl.Int64, 2)),
            "struct": [{"a": 2}, {"a": 4}],
            "bin": [b"\x02", b"\x04"],
            "null": pl.Series([None, None], dtype=pl.Null),
        }
    )
    assert plugin_gcds(df) == {c: None for c in df.columns}


def test_dtype_strings():
    df = pl.DataFrame(
        {
            "i64": pl.Series([1], dtype=pl.Int64),
            "u8": pl.Series([1], dtype=pl.UInt8),
            "i128": pl.Series([1], dtype=pl.Int128),
            "dec": pl.Series([Decimal("1.00")], dtype=pl.Decimal(10, 2)),
            "date": [date(2024, 1, 1)],
            "dt_us": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("us")),
            "dt_ns_utc": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ns", "UTC")),
            "dur_ms": pl.Series([1], dtype=pl.Duration("ms")),
            "f64": [1.0],
            "str": ["a"],
            "bool": [True],
        }
    )
    out = column_gcd(df).unnest("column_gcd")
    assert dict(zip(out["column"], out["dtype"])) == {
        "i64": "i64",
        "u8": "u8",
        "i128": "i128",
        "dec": "decimal[10,2]",
        "date": "date",
        "dt_us": "datetime[μs]",
        "dt_ns_utc": "datetime[ns, UTC]",
        "dur_ms": "duration[ms]",
        "f64": "f64",
        "str": "str",
        "bool": "bool",
    }


def test_output_schema_and_order():
    df = pl.DataFrame({"z": [4, 8], "a": ["x", "y"], "m": [9, 6]})
    out = column_gcd(df)
    assert out.schema == pl.Schema(
        {"column_gcd": pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})}
    )
    assert out.unnest("column_gcd")["column"].to_list() == ["z", "a", "m"]


def test_lazyframe_input():
    lf = pl.LazyFrame({"a": [12, 18], "b": [5, 10]})
    assert plugin_gcds(lf) == {"a": 6, "b": 5}


# ─────────────────────────────────────────────────────────────────────────────
# Seeded fuzz
# ─────────────────────────────────────────────────────────────────────────────

_FUZZ_INT_RANGES = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.Int64: (-(2**63), 2**63 - 1),
    pl.Int128: (-(2**127), 2**127 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
    pl.UInt64: (0, 2**64 - 1),
}


@pytest.mark.parametrize("seed", range(40))
def test_seeded_fuzz(seed):
    rng = random.Random(seed)
    dtype, (lo, hi) = rng.choice(list(_FUZZ_INT_RANGES.items()))
    g = rng.choice([1, 2, 3, 6, 7, 12, 1_000, 2**20, rng.randint(1, hi)])
    k_lo, k_hi = -(-lo // g), hi // g  # ceil(lo/g), floor(hi/g): k·g stays in range
    n = rng.choice([0, 1, 17, 1_000, CHUNK + rng.randint(1, 5_000)])
    null_rate = rng.choice([0.0, 0.05, 0.5, 1.0])
    ints = [None if rng.random() < null_rate else rng.randint(k_lo, k_hi) * g for _ in range(n)]

    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat(
        [pl.Series("x", ints[:cut], dtype=dtype), pl.Series("x", ints[cut:], dtype=dtype)],
        rechunk=False,
    )
    got = plugin_gcd(s)
    assert got == ref_gcd(s), f"seed={seed} dtype={dtype} g={g} n={n}"
    if numpy_applicable(s):
        assert got == numpy_gcd(s), f"numpy mismatch seed={seed}"


@pytest.mark.skipif(not hasattr(pl, "UInt128"), reason="polars build without UInt128")
def test_uint128_columns_are_null_not_crash():
    # Rust polars 0.51 (pyo3-polars 0.24) has no UInt128: handing one to the
    # plugin aborts the interpreter. Such columns must come back as null rows,
    # in input order, alongside correctly computed neighbours.
    df = pl.DataFrame(
        {
            "a": pl.Series([12, 18], dtype=pl.Int64),
            "u128": pl.Series([12, 18], dtype=pl.UInt128),
            "b": pl.Series([10, 15], dtype=pl.UInt8),
            "lu128": pl.Series([[1], [2]], dtype=pl.List(pl.UInt128)),
        }
    )
    out = column_gcd(df)
    assert out.schema == pl.Schema(
        {"column_gcd": pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})}
    )
    rows = out.unnest("column_gcd")
    assert rows["column"].to_list() == ["a", "u128", "b", "lu128"]
    assert rows["gcd"].to_list() == [6, None, 5, None]
    assert rows["dtype"].to_list() == ["i64", str(pl.UInt128), "u8", str(pl.List(pl.UInt128))]
