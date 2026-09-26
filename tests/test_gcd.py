"""
Whole-column GCD accuracy tests — every implementation in analytics.gcd.

Oracles: known answers (values built as k·g) and GcdMath (math.gcd, arbitrary
precision), the technique's reference. Semantics (ClickHouse GCD-codec method):
GCD of the magnitudes of the raw physical integer values; nulls skipped;
all-null / all-zero / zero-row → 0; non-integer-backed dtypes → status
"ineligible" with gcd null; a GCD of more than 38 digits (Decimal(38, 0)) → null.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import decimal
import random
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from analytics.gcd import Gcd
from datagen import integer_multiples, mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.gcd"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

CHUNK = 1 << 16  # gcd.rs parallel chunk size
_DEC_CTX = decimal.Context(prec=80)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def gcds(impl: str, df: pl.DataFrame | pl.LazyFrame) -> dict[str, int | None]:
    out = run(load(impl), {"t": df})
    return dict(zip(out["col_a"].to_list(), out["gcd"].to_list()))


def gcd_of(impl: str, s: pl.Series) -> int | None:
    return gcds(impl, s.to_frame())[s.name]


def math_gcd(s: pl.Series) -> int | None:
    return gcd_of(f"{PKG}:GcdMath", s)


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


UNSIGNED = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


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
# 1. Contract

@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"mixed": mixed_dtypes(200), "ints": integer_multiples(500, 3, 12)}
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement

@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"ints": integer_multiples(5_000, 4, 3_600), "mixed": mixed_dtypes(500)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)

@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype, g, k_lo, k_hi", CASES)
def test_multiples_of_known_gcd(impl, dtype, g, k_lo, k_hi):
    rng = random.Random(1234)
    ints = [g] + [None if rng.random() < 0.1 else rng.randint(k_lo, k_hi) * g for _ in range(1_000)]
    assert gcd_of(impl, from_physical("x", ints, dtype)) == g


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_coprime_values_give_one(impl, dtype):
    assert gcd_of(impl, from_physical("x", [6, 10, 15], dtype)) == 1


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_single_value_is_its_magnitude(impl, dtype):
    v = -42 if is_signed(dtype) else 42
    assert gcd_of(impl, from_physical("x", [v], dtype)) == 42


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_zeros_and_nulls(impl, dtype):
    assert gcd_of(impl, from_physical("x", [0, 0, 12, None, 18, 0], dtype)) == 6
    assert gcd_of(impl, from_physical("x", [0, 0, 0], dtype)) == 0
    assert gcd_of(impl, from_physical("x", [None, None], dtype)) == 0
    assert gcd_of(impl, from_physical("x", [None, 12, None, 18, None], dtype)) == 6


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize(
    "dtype, ints, expected",
    [
        pytest.param(pl.Int8(), [-128], 128, id="i8_min"),
        pytest.param(pl.Int8(), [-128, 64], 64, id="i8_min_and_64"),
        pytest.param(pl.UInt64(), [2**64 - 1], 2**64 - 1, id="u64_max"),
        pytest.param(pl.Int64(), [-(2**63)], 2**63, id="i64_min"),
        pytest.param(pl.Int64(), [-(2**63), 2**62], 2**62, id="i64_min_and_2^62"),
        pytest.param(pl.Int128(), [10**38 - 1], 10**38 - 1, id="decimal38_max"),
        pytest.param(pl.Int128(), [10**38], None, id="beyond_38_digits"),
        pytest.param(pl.Int128(), [2**127 - 1], None, id="i128_max"),
        pytest.param(pl.Int128(), [-(2**127), 2**126], 2**126, id="i128_min_and_2^126"),
        pytest.param(pl.Int128(), [-(2**127)], None, id="i128_min_unrepresentable"),
        pytest.param(pl.Int128(), [-(2**127), None, 0], None, id="i128_min_with_null_zero"),
    ],
)
def test_dtype_extremes(impl, dtype, ints, expected):
    assert gcd_of(impl, from_physical("x", ints, dtype)) == expected


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame(impl):
    schema = {f"c{i}": p.values[0] for i, p in enumerate(CASES)} | {"s": pl.String()}
    assert gcds(impl, pl.DataFrame(schema=schema)) == {**{f"c{i}": 0 for i in range(len(CASES))}, "s": None}


@pytest.mark.parametrize("impl", ALL)
def test_zero_column_frame(impl):
    cls = load(impl)
    out = run(cls, {"t": pl.DataFrame()})
    assert out.height == 0
    assert_contract(cls, out, {"t": pl.DataFrame()})


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_series(impl):
    parts = [pl.Series("x", [12, 24]), pl.Series("x", [None, 36]), pl.Series("x", [18])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    assert gcd_of(impl, s) == 6


@pytest.mark.parametrize("impl", ALL)
def test_long_series_crosses_parallel_chunks(impl):
    n = 3 * CHUNK + 17
    values = np.full(n, 12, dtype=np.int64)
    values[2 * CHUNK + 5] = 18
    mask = np.arange(n) % 7 == 3  # scattered nulls
    assert not mask[2 * CHUNK + 5]  # the spoiler stays valid
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask))) == 6


@pytest.mark.parametrize("impl", ALL)
def test_null_payloads_ignored(impl):
    # 1s live under null slots; a correct null mask ignores them (otherwise the
    # GCD collapses to 1, and a masked 1 would also trigger the early exit).
    values = np.tile(np.array([12, 1, 18, 1], dtype=np.int64), 50_000)
    arr = pa.array(values, mask=values == 1)
    assert np.frombuffer(arr.buffers()[1], dtype=np.int64)[1] == 1  # payload really is there
    assert gcd_of(impl, pl.from_arrow(arr)) == 6


@pytest.mark.parametrize("impl", ALL)
def test_sliced_series(impl):
    # Leading 7s are sliced away; offsets are not multiples of 8, so the
    # validity bitmap has a sub-byte offset.
    n = 2 * CHUNK + 100
    values = np.full(n, 12, dtype=np.int64)
    values[:5] = 7
    values[CHUNK + 1] = 18
    mask = np.zeros(n, dtype=bool)
    mask[CHUNK + 2 :: 11] = True
    values[mask] = 7  # payloads under nulls
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask)).slice(5, 2 * CHUNK + 50)) == 6
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask)).slice(3)) == 1  # keeps two leading 7s


@pytest.mark.parametrize("impl", ALL)
def test_physical_unit_results(impl):
    hourly = pl.datetime_range(datetime(2024, 1, 1, 7), datetime(2024, 1, 3), "1h", time_unit="us", eager=True)
    assert gcd_of(impl, hourly) == 3_600_000_000
    quarters = pl.Series("p", [Decimal("1.25"), Decimal("0.50"), Decimal("0.75"), Decimal("-2.00")], dtype=pl.Decimal(10, 2))
    assert gcd_of(impl, quarters) == 25
    weekly = pl.date_range(date(2024, 1, 1), date(2024, 6, 30), "1w", eager=True)
    assert gcd_of(impl, weekly) == math_gcd(weekly)  # raw epoch days, not the 7-day step


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


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(40))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype, (lo, hi) = rng.choice(list(_FUZZ_INT_RANGES.items()))
    g = rng.choice([1, 2, 3, 6, 7, 12, 1_000, 2**20, rng.randint(1, hi)])
    k_lo, k_hi = -(-lo // g), hi // g  # ceil(lo/g), floor(hi/g): k·g stays in range
    n = rng.choice([0, 1, 17, 1_000, CHUNK + rng.randint(1, 5_000)])
    null_rate = rng.choice([0.0, 0.05, 0.5, 1.0])
    ints = [None if rng.random() < null_rate else rng.randint(k_lo, k_hi) * g for _ in range(n)]
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([pl.Series("x", ints[:cut], dtype=dtype), pl.Series("x", ints[cut:], dtype=dtype)], rechunk=False)
    assert gcd_of(impl, s) == math_gcd(s), f"seed={seed} dtype={dtype} g={g} n={n}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions (technique base, once)

def test_conclusions():
    Fixed = with_metrics(Gcd, gcd=[0, 1, 2, None])
    df = pl.DataFrame({"zero": [0], "one": [1], "two": [2], "big": pl.Series([0], dtype=pl.Int128)})
    out = Fixed().add({"t": df}).result()
    assert out["gcd_compressible"].to_list() == [False, False, True, None]


# ─────────────────────────────────────────────────────────────────────────────
# Output contract details

@pytest.mark.parametrize("impl", ALL)
def test_ineligible_dtypes_are_reported_with_their_dtype(impl):
    cols = {
        "f64": pl.Series([2.0, 4.0], dtype=pl.Float64),
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
    if hasattr(pl, "UInt128"):  # the plugin's Rust polars cannot receive UInt128
        cols["u128"] = pl.Series([12, 18], dtype=pl.UInt128)
    df = pl.DataFrame(cols)
    out = run(load(impl), {"t": df})
    assert out["status"].to_list() == ["ineligible"] * df.width
    assert out["gcd"].null_count() == df.width
    assert out["gcd_compressible"].null_count() == df.width
    assert out["dtype"].to_list() == [str(dt) for dt in df.dtypes]


@pytest.mark.parametrize("impl", ALL)
def test_order_lazyframes_and_multiple_frames(impl):
    df = pl.DataFrame({"z": [4, 8], "a": ["x", "y"], "m": [9, 6]})
    out = run(load(impl), {"first": df.lazy(), "second": df})
    assert out.select("df_a", "col_a").rows() == [
        ("first", "z"), ("first", "a"), ("first", "m"), ("second", "z"), ("second", "a"), ("second", "m")
    ]
    assert out["gcd"].to_list() == [4, None, 3, 4, None, 3]
    assert out["dtype"].to_list() == ["Int64", "String", "Int64"] * 2
