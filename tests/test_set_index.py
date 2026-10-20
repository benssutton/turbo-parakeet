"""
Set skip-index accuracy tests — every implementation in turbo_parakeet.techniques.set_index.

Oracles: hand-worked known answers, a brute-force window-by-window oracle
(skip_index_helpers.oracle), SetIndexPolars (the reference) and a Monte Carlo of the
random-order baseline (exact when max_size = 0, a normal approximation otherwise).
Semantics: spec 2026-10-16-skip-index-techniques §2.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
import random

import numpy as np
import polars as pl
import pytest

from datagen import large_dataset, large_dataset_mixed, mixed_dtypes, runs_int, runs_str
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)
from skip_index_helpers import (
    ELIGIBLE_DTYPES,
    as_dtype,
    assert_metrics,
    oracle,
    sampled,
    zipf_column,
)

PKG = "turbo_parakeet.techniques.set_index"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

TOY = [0, 0, 0, 1, 0, 0, 0, 2]  # A A A B A A A C (spec §7)


def stats(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def set_oracle(values: list, window: int, max_size: int = 0) -> dict:
    return oracle(values, window, max_size)[1]


def agree(impl: str, frames: dict, **params) -> None:
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "runs": runs_int(1_000, 2, 10, 50),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames, index_granularity=16, max_size=5), frames)


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
    assert out["window_distinct_max"].to_list() == [None] * 5 + [1, None]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "params", [{}, {"index_granularity": 1_024, "granularity": 2, "max_size": 0}]
)
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_with_reference_large_dataset(impl, params, make):
    agree(impl, {"large": make()}, **params)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("max_size", [0, 10])
def test_agrees_with_reference_seeded(impl, max_size):
    # 200 000 rows: more window starts than one parallel range (2^16)
    frames = {
        "mixed": mixed_dtypes(2_000),
        "ints": runs_int(200_000, 3, 25, 100),
        "strs": runs_str(200_000, 2, 5, 1_000),
    }
    agree(impl, frames, index_granularity=512, max_size=max_size)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(30))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype = ELIGIBLE_DTYPES[seed % len(ELIGIBLE_DTYPES)]
    n = rng.choice([1, 2, 17, 1_000, 70_000])
    k = rng.choice([1, 2, 3, 10])
    if isinstance(dtype, pl.Enum):
        dtype = pl.Enum([str(i) for i in range(k)])
    null_rate = rng.choice([0.0, 0.1, 0.5])
    values, cur = [], 0
    for _ in range(n):  # sticky random walk → clustered values
        if rng.random() < 0.3:
            cur = rng.randrange(k)
        values.append(None if rng.random() < null_rate else cur)
    s = as_dtype(values, dtype)
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([s.slice(0, cut), s.slice(cut)], rechunk=False)
    agree(
        impl,
        {"t": s.to_frame()},
        index_granularity=rng.choice([1, 3, 64, 5_000]),
        max_size=rng.choice([0, 1, 2, 5]),
    )


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_oracle(impl, seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    k = rng.choice([1, 2, 3, 5, 10])
    null_rate = rng.choice([0.0, 0.1, 0.5, 1.0])
    values = [None if rng.random() < null_rate else rng.randrange(k) for _ in range(n)]
    window, max_size = rng.randint(1, 12), rng.choice([0, 1, 2, 3])
    s = pl.Series("x", values, dtype=pl.Int64)
    got = stats(impl, s, index_granularity=window, max_size=max_size)
    assert_metrics(got, set_oracle(values, window, max_size))


def test_kernel_rejects_bad_input():
    from turbo_parakeet import _plugin

    with pytest.raises(ValueError, match="unsupported dtype"):
        _plugin.set_index(pl.DataFrame({"x": [[1], [2]]}), 4, 0)
    with pytest.raises(ValueError, match="window_rows"):
        _plugin.set_index(pl.DataFrame({"x": [1, 2]}), 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ELIGIBLE_DTYPES, ids=str)
def test_known_answer_every_dtype(impl, dtype):
    r = stats(impl, as_dtype(TOY, dtype), index_granularity=4, max_size=0)
    assert (r["n_windows"], r["window_distinct_mean"], r["window_distinct_max"]) == (
        5,
        2,
        2,
    )
    assert r["skip_rows_mean"] == pytest.approx(0.125, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(1 / 3, rel=1e-12)
    assert r["skip_rows_std"] == pytest.approx(0.0, abs=1e-15)
    assert r["skip_values_std"] == pytest.approx(0.0, abs=1e-15)
    assert r["overflow_fraction"] == 0.0


@pytest.mark.parametrize("impl", ALL)
def test_window_is_index_granularity_times_granularity(impl):
    r = stats(impl, pl.Series("x", list(range(20))), index_granularity=2, granularity=3)
    assert (r["window_rows"], r["n_windows"]) == (6, 15)


def test_known_answers_cover_every_eligible_class():
    from turbo_parakeet._dtypes import ORDERED_SCALARS

    assert {type(dt) for dt in ELIGIBLE_DTYPES} | {pl.Boolean} == set(ORDERED_SCALARS)


@pytest.mark.parametrize("impl", ALL)
def test_boolean(impl):
    values = [False, False, True, True]
    r = stats(impl, pl.Series("x", values), index_granularity=2, max_size=0)
    assert_metrics(r, set_oracle(values, 2))


@pytest.mark.parametrize("impl", ALL)
def test_overflow_skips_nothing(impl):
    r = stats(impl, pl.Series("x", TOY), index_granularity=4, max_size=1)
    assert (r["skip_rows_mean"], r["skip_values_mean"], r["overflow_fraction"]) == (
        0.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("impl", ALL)
def test_null_counts_toward_max_size(impl):
    values = [1, None, 2, None]  # every 2-row window: one value + NULL = 2 entries
    s = pl.Series("x", values)
    r = stats(impl, s, index_granularity=2, max_size=1)
    assert (r["overflow_fraction"], r["window_distinct_max"]) == (1.0, 2)
    r = stats(impl, s, index_granularity=2, max_size=2)
    assert r["overflow_fraction"] == 0.0
    assert_metrics(r, set_oracle(values, 2, 2))


def test_default_max_size_is_100():
    assert reference(PKG)().max_size == 100


@pytest.mark.parametrize("impl", ALL)
def test_sorted_column(impl):
    r = stats(impl, pl.Series("x", list(range(1, 9))), index_granularity=4)
    assert (r["skip_rows_mean"], r["skip_values_mean"]) == (0.5, 0.5)
    assert r["window_distinct_mean"] == 4.0


@pytest.mark.parametrize("impl", ALL)
def test_all_null_column(impl):
    r = stats(impl, pl.Series("x", [None] * 3, dtype=pl.Int64), index_granularity=2)
    assert (r["n_distinct"], r["window_distinct_mean"], r["window_distinct_max"]) == (
        0,
        1.0,
        1,
    )
    assert r["expected_window_distinct"] == pytest.approx(1.0, rel=1e-12)
    assert math.isnan(r["skip_rows_mean"]) and math.isnan(r["expected_skip_values"])


@pytest.mark.parametrize("impl", ALL)
def test_column_shorter_than_window_is_one_window(impl):
    r = stats(impl, pl.Series("x", [3, 1, 2]), index_granularity=8, max_size=0)
    assert (r["window_rows"], r["n_windows"], r["skip_rows_mean"]) == (3, 1, 0.0)
    assert r["expected_skip_rows"] == 0.0
    assert r["expected_window_distinct"] == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("impl", ALL)
def test_string_categorical_enum_identical(impl):
    # a set needs identity, not order: every string-like dtype gives the same result
    s = pl.Series("x", ["b", "a", "a", "c", "c", "c", "a", None, None, "b"])

    def metrics(r):  # dtype differs by design
        return {m: r[m] for m in load(impl).METRICS}

    want = metrics(stats(impl, s, index_granularity=3, max_size=2))
    for dtype in (pl.Categorical, pl.Enum(["c", "b", "a"])):
        got = metrics(stats(impl, s.cast(dtype), index_granularity=3, max_size=2))
        assert got == want, dtype


@pytest.mark.parametrize("impl", ALL)
def test_floats_canonical(impl):
    other_nan = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
    s = pl.Series("x", [0.0, -0.0, float("nan"), other_nan, 1.0, -1.0])
    for series in (s, s.cast(pl.Float32)):
        r = stats(impl, series, index_granularity=2, max_size=0)
        assert r["n_distinct"] == 4
        assert_metrics(r, set_oracle([1, 1, 3, 3, 2, 0], 2))


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_equals_one_chunk(impl):
    s = runs_int(5_000, 1, 7, 40)["c0"]
    parts = pl.concat([s.slice(0, 1_234), s.slice(1_234)], rechunk=False)
    assert parts.n_chunks() == 2
    params = {"index_granularity": 64, "max_size": 3}
    assert stats(impl, parts, **params) == stats(impl, s, **params)


@pytest.mark.parametrize("impl", ALL)
def test_baseline_hand_worked(impl):
    # 1 2 3 4, w = 2, unlimited: each value is absent from half the pairs; 2 entries each
    r = stats(impl, pl.Series("x", [1, 2, 3, 4]), index_granularity=2, max_size=0)
    assert r["expected_skip_rows"] == pytest.approx(0.5, rel=1e-9)
    assert r["expected_skip_values"] == pytest.approx(0.5, rel=1e-9)
    assert r["expected_window_distinct"] == pytest.approx(2.0, rel=1e-9)
    assert r["expected_overflow_fraction"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"max_size": -1}, {"max_size": 1.5}, {"index_granularity": 0}],
)
def test_keywords_are_validated(kwargs):
    with pytest.raises(ValueError, match="must be an integer"):
        reference(PKG)(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical correctness (no conclusions to test: the technique reports metrics only)

CASES = [
    (1, 30, 400, 20, 11, 1.0, 0.0),
    (2, 50, 1_000, 30, 13, 1.2, 0.05),
    (3, 20, 300, 15, 8, 1.5, 0.1),
]  # seed, values, rows, window, max_size (near the mean entry count), Zipf α, null rate


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("case", CASES)
def test_unlimited_baseline_matches_monte_carlo(impl, case):
    seed, k, n, w, _, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    s = pl.Series("x", values, dtype=pl.Int64)
    r = stats(impl, s, index_granularity=w, max_size=0)
    mc = sampled(values, w, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "set_rows"),
        ("expected_skip_values", "set_values"),
        ("expected_window_distinct", "entries"),
    ):
        mean, se = mc[key]
        assert abs(r[metric] - mean) <= 5 * se + 1e-9, (metric, r[metric], mean, se)
    assert r["expected_overflow_fraction"] == 0.0


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("case", CASES)
def test_finite_baseline_within_approximation(impl, case):
    seed, k, n, w, max_size, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    s = pl.Series("x", values, dtype=pl.Int64)
    r = stats(impl, s, index_granularity=w, max_size=max_size)
    mc = sampled(values, w, max_size, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "set_rows"),
        ("expected_skip_values", "set_values"),
        ("expected_overflow_fraction", "overflow"),
    ):
        assert abs(r[metric] - mc[key][0]) <= 0.02, (metric, r[metric], mc[key])
    mean, se = mc["entries"]  # exact whatever max_size
    assert abs(r["expected_window_distinct"] - mean) <= 5 * se + 1e-9
