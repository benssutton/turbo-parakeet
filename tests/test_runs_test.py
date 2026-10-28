"""
Runs-test accuracy tests — every implementation in turbo_parakeet.techniques.runs_test.

Oracles: hand-worked known answers, a plain-Python run counter, RunsTestPolars (the
reference: exact Python-int moments), a permutation Monte Carlo of the closed-form
moments and, for two-valued columns, statsmodels' binary runs test. Semantics: R = runs of
equal consecutive values (null one value, floats canonical); p one-sided, lower tail;
Var = 0 → z, p NaN; zero-row and nested / 128-bit columns ineligible.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
import random

import numpy as np
import polars as pl
import pytest

from turbo_parakeet.techniques.runs_test.polars import exact_moments
from synthdata.synth import synth_small, synth_windows, spread
from synthdata.scenarios import runs_int, runs_str
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.runs_test"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

CHUNK = 1 << 16  # runs_test.rs parallel chunk size


# ─────────────────────────────────────────────────────────────────────────────
# Helpers


def stats(impl: str, s: pl.Series) -> dict:
    return run(load(impl), {"t": s.to_frame()}).row(0, named=True)


def py_runs(values: list) -> int:
    """Runs of equal consecutive values; None equals None."""
    return 1 + sum(a != b for a, b in zip(values, values[1:]))


def phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


SEQ = [0, 0, 1, 1, 1, 2]  # A A B B B C: R = 3, E[R] = 14/3, Var = 136/180
SEQ_Z = (3 - 14 / 3) / math.sqrt(136 / 180)


def as_dtype(values: list[int], dtype: pl.DataType) -> pl.Series:
    s = pl.Series("x", values)
    if dtype in (pl.String, pl.Binary, pl.Categorical) or isinstance(dtype, pl.Enum):
        return s.cast(pl.String).cast(dtype)
    return s.cast(dtype)


ELIGIBLE_DTYPES = [
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
    pl.Decimal(10, 2),
    pl.Date(),
    pl.Datetime("us"),
    pl.Datetime("ns", "UTC"),
    pl.Duration("ms"),
    pl.Time(),
    pl.String(),
    pl.Binary(),
    pl.Categorical(),
    pl.Enum(["0", "1", "2"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "cover": spread(synth_small(), 18).head(300),
        "runs": runs_int(1_000, 2, 10, 50),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_cover_windows(impl):
    frames = {"windows": synth_windows()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_seeded(impl):
    frames = {
        "cover": spread(synth_small(), 18),
        "ints": runs_int(200_000, 3, 25, 100),
        "strs": runs_str(200_000, 2, 5, 1_000),
    }
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(30))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype = ELIGIBLE_DTYPES[seed % len(ELIGIBLE_DTYPES)]
    n = rng.choice([1, 2, 17, 1_000, CHUNK + rng.randint(1, 5_000)])
    k = rng.choice([1, 2, 3, 10, 50])
    if isinstance(dtype, pl.Enum):
        dtype = pl.Enum([str(i) for i in range(k)])
    null_rate = rng.choice([0.0, 0.1, 0.5])
    values, cur = [], 0
    for _ in range(n):  # sticky random walk over k values → runs of varied length
        if rng.random() < 0.3:
            cur = rng.randrange(k)
        values.append(None if rng.random() < null_rate else cur)
    s = as_dtype(values, dtype)
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([s.slice(0, cut), s.slice(cut)], rechunk=False)
    frames = {"t": s.to_frame()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def test_kernel_rejects_unsupported_dtype():
    from turbo_parakeet import _plugin

    with pytest.raises(ValueError, match="unsupported dtype"):
        _plugin.runs_test(pl.DataFrame({"x": [[1], [2]]}))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ELIGIBLE_DTYPES, ids=str)
def test_known_answer_every_dtype(impl, dtype):
    r = stats(impl, as_dtype(SEQ, dtype))
    assert (r["n_rows"], r["n_runs"]) == (6, 3)
    assert r["expected_runs"] == pytest.approx(14 / 3, rel=1e-12)
    assert r["z"] == pytest.approx(SEQ_Z, rel=1e-12)
    assert r["p_value"] == pytest.approx(phi(SEQ_Z), rel=1e-12)
    assert r["mean_run_length"] == 2.0


def test_known_answers_cover_every_eligible_class():
    """Every class in ELIGIBLE (mirrored by the Rust kernel's `supported`) runs through both
    implementations in test_known_answer_every_dtype (Boolean: test_boolean)."""
    from turbo_parakeet.techniques.runs_test.base import ELIGIBLE

    assert {type(dt) for dt in ELIGIBLE_DTYPES} | {pl.Boolean} == set(ELIGIBLE)


@pytest.mark.parametrize("impl", ALL)
def test_boolean(impl):
    r = stats(impl, pl.Series("x", [True, True, False, False, False, True]))
    assert (r["n_runs"], r["expected_runs"]) == (3, 4.0)


@pytest.mark.parametrize("impl", ALL)
def test_alternating_is_upper_tail(impl):
    r = stats(impl, pl.Series("x", [0, 1, 0, 1, 0, 1]))
    assert (r["n_runs"], r["expected_runs"]) == (6, 4.0)
    assert r["z"] == pytest.approx(2 / math.sqrt(1.2), rel=1e-12)
    assert r["p_value"] > 0.5  # one-sided, lower tail: more runs than chance → p near 1


@pytest.mark.parametrize("impl", ALL)
def test_nulls_are_one_value(impl):
    assert stats(impl, pl.Series("x", [1, None, None, 1]))["n_runs"] == 3
    r = stats(impl, pl.Series("x", [None, None, None], dtype=pl.Int64))
    assert (r["n_runs"], r["expected_runs"], r["mean_run_length"]) == (1, 1.0, 3.0)
    assert math.isnan(r["z"]) and math.isnan(r["p_value"])


@pytest.mark.parametrize("impl", ALL)
def test_floats_compare_canonically(impl):
    other_nan = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
    s = pl.Series("x", [0.0, -0.0, float("nan"), other_nan])
    for series in (s, s.cast(pl.Float32)):
        r = stats(impl, series)
        # counts {0: 2, NaN: 2} → E = 1 + (16 − 8) / 4 = 3: counting is canonical too
        assert (r["n_runs"], r["expected_runs"]) == (2, 3.0)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize(
    "values, n_runs, expected",
    [
        pytest.param([7], 1, 1.0, id="one_row"),
        pytest.param([5, 5, 5, 5], 1, 1.0, id="constant"),
        pytest.param([1, 2, 3, 4], 4, 4.0, id="all_distinct"),
    ],
)
def test_zero_variance_is_nan(impl, values, n_runs, expected):
    r = stats(impl, pl.Series("x", values))
    assert (r["n_runs"], r["expected_runs"]) == (n_runs, expected)
    assert r["mean_run_length"] == len(values) / n_runs
    assert math.isnan(r["z"]) and math.isnan(r["p_value"])


@pytest.mark.parametrize("impl", ALL)
def test_string_categorical_enum_identical(impl):
    values = ["b", "a", "a", "c", "c", "c", "a", None, None, "b"]
    s = pl.Series("x", values)

    def metrics(r):  # dtype differs by design
        return {m: r[m] for m in load(impl).METRICS}

    want = metrics(stats(impl, s))
    assert metrics(stats(impl, s.cast(pl.Categorical))) == want
    assert metrics(stats(impl, s.cast(pl.Enum(["c", "b", "a"])))) == want


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_run_spans_boundary(impl):
    parts = [pl.Series("x", [1, 1]), pl.Series("x", [1, 2]), pl.Series("x", [2])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    assert stats(impl, s)["n_runs"] == 2


@pytest.mark.parametrize("impl", ALL)
def test_long_column_crosses_parallel_chunks(impl):
    n = 3 * CHUNK + 17
    values = (np.arange(n) // 1000).astype(np.int64).tolist()
    for i in range(CHUNK - 3, CHUNK + 3):  # a null run straddling the first boundary
        values[i] = None
    assert stats(impl, pl.Series("x", values, dtype=pl.Int64))["n_runs"] == py_runs(
        values
    )


@pytest.mark.parametrize("impl", ALL)
def test_ineligible_columns_are_listed(impl):
    df = pl.DataFrame(
        {
            "list": [[1], [2]],
            "arr": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
            "struct": [{"a": 1}, {"a": 2}],
            "null": pl.Series([None, None], dtype=pl.Null),
            "i128": pl.Series([1, 2], dtype=pl.Int128),
            "ok": [1, 1],
        }
    )
    out = run(load(impl), {"t": df, "empty": pl.DataFrame(schema={"e": pl.Int64})})
    assert out["status"].to_list() == ["ineligible"] * 5 + ["computed", "ineligible"]
    assert out["dtype"].to_list() == [str(dt) for dt in df.dtypes] + ["Int64"]
    assert out["n_runs"].to_list() == [None] * 5 + [1, None]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical correctness (no conclusions to test: the technique reports metrics only)


def test_moments_match_permutation_monte_carlo():
    counts, n = [16, 12, 8, 4], 40
    base = np.repeat(np.arange(len(counts)), counts)
    shuffles = np.random.default_rng(7).permuted(np.tile(base, (20_000, 1)), axis=1)
    runs = 1 + (shuffles[:, 1:] != shuffles[:, :-1]).sum(axis=1)
    expected, variance = exact_moments(n, counts)
    assert abs(runs.mean() - expected) < 4 * math.sqrt(variance / runs.size)
    assert runs.var(ddof=1) == pytest.approx(variance, rel=0.05)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(5))
def test_two_valued_z_matches_statsmodels(impl, seed):
    sm_runs = pytest.importorskip(
        "statsmodels.sandbox.stats.runs", reason="statsmodels not installed"
    )
    rng = np.random.default_rng(seed)
    x = (rng.random(500) < rng.uniform(0.2, 0.8)).astype(np.int64)
    x[: 50 * seed] = np.sort(x[: 50 * seed])  # a clustered prefix varies z
    z_sm, _ = sm_runs.Runs(x).runs_test(correction=False)
    r = stats(impl, pl.Series("x", x))
    assert r["z"] == pytest.approx(z_sm, rel=1e-9)
    assert r["p_value"] == pytest.approx(phi(z_sm), rel=1e-9)  # ours: one-sided


@pytest.mark.parametrize("impl", ALL)
def test_near_constant_variance_is_exact(impl):
    # Var's numerator terms are ~1e24 here and cancel to ~4e14: f64 would lose ~7 digits.
    n = 1_000_000
    x = np.zeros(n, dtype=np.int64)
    x[::100_000] = 1
    n_runs = 1 + int(np.count_nonzero(np.diff(x)))
    expected, variance = exact_moments(n, [n - 10, 10])
    r = stats(impl, pl.Series("x", x))
    assert r["n_runs"] == n_runs
    assert r["z"] == pytest.approx((n_runs - expected) / math.sqrt(variance), rel=1e-9)
