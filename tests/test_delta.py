"""
Delta accuracy tests — every implementation in turbo_parakeet.techniques.delta.

Oracles: hand-worked known answers (reference included), DeltaPolars (Python-int
reference), and an O(n²) brute force for the random-order baseline. Semantics: differences
of consecutive non-null raw physical integers, exact (no wrapping); sig = bytes of the
signed form (0 for 0, up to 9); floats, text and zero-row columns ineligible.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import datetime as dt
import math
import random
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from datagen import large_dataset, large_dataset_mixed, mixed_dtypes
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.delta"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

I64_MIN, I64_MAX = -(2**63), 2**63 - 1


def stats(impl: str, s: pl.Series) -> dict:
    return run(load(impl), {"t": s.to_frame()}).row(0, named=True)


def eligible_frame(n: int, seed: int = 1) -> pl.DataFrame:
    """Every eligible dtype, with nulls."""
    rng = np.random.default_rng(seed)
    nulls = rng.random(n) < 0.1

    def mk(lo, hi, dtype):
        v = rng.integers(lo, hi, n)
        s = pl.Series([None if z else int(x) for x, z in zip(v, nulls)], dtype=pl.Int64)
        return s.cast(dtype)

    return pl.DataFrame(
        {
            "i8": mk(-128, 128, pl.Int8),
            "i16": mk(-30_000, 30_000, pl.Int16),
            "i32": mk(-(2**31), 2**31, pl.Int32),
            "i64": mk(-(2**62), 2**62, pl.Int64),
            "u8": mk(0, 256, pl.UInt8),
            "u16": mk(0, 65_536, pl.UInt16),
            "u32": mk(0, 2**32, pl.UInt32),
            "u64": mk(0, 2**62, pl.UInt64),
            "dec": mk(-100_000, 100_000, pl.Decimal(10, 2)),
            "date": mk(0, 20_000, pl.Date),
            "dt_us": mk(0, 10**12, pl.Datetime("us")),
            "dt_ns": mk(0, 10**15, pl.Datetime("ns", "UTC")),
            "dur": mk(-(10**6), 10**6, pl.Duration("ms")),
            "time": mk(0, 86_400 * 10**9, pl.Time),
            "const": pl.Series([7] * n, dtype=pl.Int32),
        }
    )


# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "large": large_dataset_mixed(),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
        "ineligible": pl.DataFrame(
            {
                "list": [[1], [2]],
                "f": [1.5, 2.5],
                "s": ["a", "b"],
                "b": [True, False],
                "c": pl.Series(["a", "b"], dtype=pl.Categorical),
            }
        ),
    }
    cls = load(impl)
    result = run(cls, frames)
    assert_contract(cls, result, frames)
    ineligible = result.filter(pl.col("df_a") == "ineligible")
    assert ineligible["status"].to_list() == ["ineligible"] * 5
    assert result.filter(pl.col("df_a") == "empty")["status"].to_list() == [
        "ineligible"
    ]


# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_large_dataset(impl, make):
    frames = {"large": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_every_eligible_dtype(impl):
    frames = {"t": eligible_frame(3_000)}
    cls = load(impl)
    result = run(cls, frames)
    assert set(result["status"]) == {"computed"}
    assert_agrees(cls(), result, run(reference(PKG), frames))


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_across_chunk_boundaries(impl):
    rng = np.random.default_rng(3)
    n = 300_000
    walk = np.cumsum(rng.integers(-1_000, 1_001, n)) + 10**12
    nulls = rng.random(n) < 0.05
    ts = np.cumsum(rng.integers(995, 1_006, n)).astype("datetime64[ms]")
    frames = {
        "t": pl.DataFrame(
            {
                "walk": walk,
                "walk_nulls": pl.Series(walk).zip_with(
                    pl.Series(~nulls), pl.Series([None] * n, dtype=pl.Int64)
                ),
                "ts": ts,
            }
        )
    }
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers (reference included)


@pytest.mark.parametrize("impl", ALL)
def test_regular_series(impl):
    r = stats(impl, pl.Series("x", [10, 20, 30, 40]))
    assert (r["n_rows"], r["n_non_null"], r["n_deltas"]) == (4, 4, 3)
    assert (r["delta_min"], r["delta_max"]) == (10, 10)
    assert (r["double_delta_min"], r["double_delta_max"]) == (0, 0)
    assert r["double_delta_bytes_mean"] == 0.0
    assert r["delta_bytes_mean"] == 1.0
    assert r["zero_delta_fraction"] == 0.0
    assert r["zero_double_delta_fraction"] == 1.0
    assert r["double_delta_entropy"] == 0.0
    assert r["delta_entropy"] == 0.0


@pytest.mark.parametrize("impl", ALL)
def test_constant_column(impl):
    r = stats(impl, pl.Series("x", [5, 5, 5, 5]))
    assert r["delta_bytes_mean"] == 0.0
    assert r["delta_entropy"] == 0.0
    assert r["zero_delta_fraction"] == 1.0
    assert r["value_bytes_mean"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_int64_extremes_do_not_wrap(impl):
    r = stats(impl, pl.Series("x", [I64_MIN, I64_MAX], dtype=pl.Int64))
    assert r["delta_min"] == r["delta_max"] == Decimal(2**64 - 1)
    assert r["delta_bytes_mean"] == 9.0
    assert r["value_bytes_mean"] == 8.0
    assert r["n_deltas"] == 1 and r["double_delta_min"] is None


@pytest.mark.parametrize("impl", ALL)
def test_nulls_are_skipped(impl):
    r = stats(impl, pl.Series("x", [1, None, 3, None, 6]))
    assert (r["n_rows"], r["n_non_null"], r["n_deltas"]) == (5, 3, 2)
    assert (r["delta_min"], r["delta_max"]) == (2, 3)
    assert (r["double_delta_min"], r["double_delta_max"]) == (1, 1)


@pytest.mark.parametrize("impl", ALL)
def test_one_value(impl):
    r = stats(impl, pl.Series("x", [42]))
    assert (r["n_non_null"], r["n_deltas"]) == (1, 0)
    for m in ("delta_min", "delta_max", "double_delta_min", "double_delta_max"):
        assert r[m] is None
    for m in (
        "delta_bytes_mean",
        "double_delta_bytes_mean",
        "zero_delta_fraction",
        "delta_entropy",
        "expected_delta_bytes_mean",
    ):
        assert math.isnan(r[m]), m
    assert r["value_bytes_mean"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_all_null_column(impl):
    r = stats(impl, pl.Series("x", [None, None], dtype=pl.Int64))
    assert (r["n_rows"], r["n_non_null"], r["n_deltas"]) == (2, 0, 0)
    assert math.isnan(r["value_bytes_mean"])


@pytest.mark.parametrize("impl", ALL)
def test_date(impl):
    s = pl.Series("x", [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 4)])
    r = stats(impl, s)
    assert (r["delta_min"], r["delta_max"]) == (1, 2)


@pytest.mark.parametrize("impl", ALL)
def test_datetime_ms(impl):
    s = pl.Series("x", [0, 1_000, 2_000, 3_000]).cast(pl.Datetime("ms"))
    r = stats(impl, s)
    assert (r["delta_min"], r["delta_max"]) == (1_000, 1_000)
    assert r["delta_bytes_mean"] == 2.0  # 1000 needs two bytes


@pytest.mark.parametrize("impl", ALL)
def test_decimal_uses_unscaled_values(impl):
    s = pl.Series("x", [Decimal("1.00"), Decimal("1.50")], dtype=pl.Decimal(10, 2))
    r = stats(impl, s)
    assert (r["delta_min"], r["delta_max"]) == (50, 50)


@pytest.mark.parametrize("impl", ALL)
def test_sig_boundaries(impl):
    # deltas 127 and -128 fit one byte; 128 and -129 need two
    one = stats(impl, pl.Series("x", [0, 127, -1], dtype=pl.Int64))
    two = stats(impl, pl.Series("x", [0, 128, -1], dtype=pl.Int64))
    assert one["delta_bytes_mean"] == 1.0
    assert two["delta_bytes_mean"] == 2.0


# 4. Baseline against an O(n²) brute force (no conclusions: the technique has none)


def py_sig(v: int) -> int:
    if v == 0:
        return 0
    return next(
        k for k in range(1, 10) if -(1 << (8 * k - 1)) <= v < (1 << (8 * k - 1))
    )


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(20))
def test_expected_baseline_matches_brute_force(impl, seed):
    rng = random.Random(seed)
    n = rng.randint(2, 60)
    mags = [0, 100, 300, 2**15, 2**23, 2**31, 2**39, 2**47, 2**55, 2**62]
    values = [rng.choice([-1, 1]) * rng.randint(0, rng.choice(mags)) for _ in range(n)]
    if seed % 3 == 0:
        values = [rng.choice(values) for _ in range(n)]  # repeated values
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    want = sum(py_sig(values[j] - values[i]) for i, j in pairs) / len(pairs)
    got = stats(impl, pl.Series("x", values, dtype=pl.Int64))[
        "expected_delta_bytes_mean"
    ]
    assert got == pytest.approx(want, rel=1e-12)
