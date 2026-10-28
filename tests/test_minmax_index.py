"""
MinMax skip-index accuracy tests — every implementation in turbo_parakeet.techniques.minmax_index.

Oracles: hand-worked known answers, a brute-force window-by-window oracle
(skip_index_helpers.oracle), MinMaxIndexPolars (the reference) and a Monte Carlo of the
random-order baseline.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
import random

import numpy as np
import polars as pl
import pytest

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
from skip_index_helpers import (
    ELIGIBLE_DTYPES,
    as_dtype,
    assert_metrics,
    oracle,
    sampled,
    zipf_column,
)

PKG = "turbo_parakeet.techniques.minmax_index"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

TOY = [0, 0, 0, 1, 0, 0, 0, 2]  # A A A B A A A C


def stats(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def minmax_oracle(values: list, window: int) -> dict:
    return oracle(values, window)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "synth_frame": spread(synth_small(), 18).head(300),
        "runs": runs_int(1_000, 2, 10, 50),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames, index_granularity=16), frames)


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
    assert out["n_windows"].to_list() == [None] * 5 + [1, None]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("params", [{}, {"index_granularity": 1_024, "granularity": 2}])
def test_agrees_with_reference_cover_windows(impl, params):
    frames = {"windows": synth_windows()}
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_seeded(impl):
    # 200 000 rows: more window starts than one parallel range (2^16)
    frames = {
        "synth_frame": spread(synth_small(), 18),
        "ints": runs_int(200_000, 3, 25, 100),
        "strs": runs_str(200_000, 2, 5, 1_000),
    }
    cls = load(impl)
    params = {"index_granularity": 512}
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


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
    frames = {"t": s.to_frame()}
    params = {"index_granularity": rng.choice([1, 3, 64, 5_000])}
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_oracle(impl, seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    k = rng.choice([1, 2, 3, 5, 10])
    null_rate = rng.choice([0.0, 0.1, 0.5, 1.0])
    values = [None if rng.random() < null_rate else rng.randrange(k) for _ in range(n)]
    window = rng.randint(1, 12)
    got = stats(impl, pl.Series("x", values, dtype=pl.Int64), index_granularity=window)
    assert_metrics(got, minmax_oracle(values, window))


def test_kernel_rejects_bad_input():
    from turbo_parakeet import _plugin

    with pytest.raises(ValueError, match="unsupported dtype"):
        _plugin.minmax_index(pl.DataFrame({"x": [[1], [2]]}), 4)
    with pytest.raises(ValueError, match="window_rows"):
        _plugin.minmax_index(pl.DataFrame({"x": [1, 2]}), 0)


def test_window_must_fit_in_64_bits():
    with pytest.raises(ValueError, match="64 bits"):
        reference(PKG)(index_granularity=2**40, granularity=2**40)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ELIGIBLE_DTYPES, ids=str)
def test_known_answer_every_dtype(impl, dtype):
    r = stats(impl, as_dtype(TOY, dtype), index_granularity=4)
    counts = tuple(
        r[m] for m in ("n_rows", "n_non_null", "n_distinct", "window_rows", "n_windows")
    )
    assert counts == (8, 8, 3, 4, 5)
    assert r["skip_rows_mean"] == pytest.approx(0.1, rel=1e-12)
    assert r["skip_rows_std"] == pytest.approx(0.05, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(4 / 15, rel=1e-12)
    assert r["skip_values_std"] == pytest.approx(2 / 15, rel=1e-12)


def test_known_answers_cover_every_eligible_class():
    """Every class in ORDERED_SCALARS (mirrored by the Rust kernels' `supported`) runs
    through both implementations in test_known_answer_every_dtype (Boolean: test_boolean).
    """
    from turbo_parakeet._dtypes import ORDERED_SCALARS

    assert {type(dt) for dt in ELIGIBLE_DTYPES} | {pl.Boolean} == set(ORDERED_SCALARS)


@pytest.mark.parametrize("impl", ALL)
def test_boolean(impl):
    # windows {F}, {F, T}, {T}: each one-valued window skips the other half
    r = stats(impl, pl.Series("x", [False, False, True, True]), index_granularity=2)
    assert r["skip_rows_mean"] == pytest.approx(1 / 3, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(1 / 3, rel=1e-12)


@pytest.mark.parametrize("impl", ALL)
def test_sorted_column(impl):
    r = stats(impl, pl.Series("x", list(range(1, 9))), index_granularity=4)
    assert (r["skip_rows_mean"], r["skip_values_mean"]) == (0.5, 0.5)
    assert r["skip_rows_std"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("impl", ALL)
def test_all_null_window_skips_everything(impl):
    values = [1, None, None, 2]  # windows {1}, {}, {2}
    r = stats(impl, pl.Series("x", values), index_granularity=2)
    assert r["skip_rows_mean"] == pytest.approx(2 / 3, rel=1e-12)
    assert_metrics(r, minmax_oracle(values, 2))


@pytest.mark.parametrize("impl", ALL)
def test_all_null_column_is_nan(impl):
    r = stats(impl, pl.Series("x", [None] * 3, dtype=pl.Int64), index_granularity=2)
    assert (r["n_non_null"], r["n_distinct"], r["n_windows"]) == (0, 0, 2)
    for m in ("skip_rows_mean", "skip_values_std", "expected_skip_rows"):
        assert math.isnan(r[m]), m


@pytest.mark.parametrize("impl", ALL)
def test_column_shorter_than_window_is_one_window(impl):
    r = stats(impl, pl.Series("x", [3, 1, 2]), index_granularity=8)
    assert (r["window_rows"], r["n_windows"]) == (3, 1)
    assert (r["skip_rows_mean"], r["expected_skip_rows"]) == (0.0, 0.0)


@pytest.mark.parametrize("impl", ALL)
def test_window_is_index_granularity_times_granularity(impl):
    r = stats(impl, pl.Series("x", list(range(20))), index_granularity=2, granularity=3)
    assert (r["window_rows"], r["n_windows"]) == (6, 15)


@pytest.mark.parametrize("impl", ALL)
def test_categorical_by_string_enum_by_category_order(impl):
    values = ["b", "b", "c", "a", "a", "c", "b", None]
    s = pl.Series("x", values)
    cat = stats(impl, s.cast(pl.Categorical), index_granularity=2)
    assert_metrics(cat, minmax_oracle(values, 2))  # strings compare lexically
    enum = stats(impl, s.cast(pl.Enum(["c", "b", "a"])), index_granularity=2)
    rank = {"c": 0, "b": 1, "a": 2, None: None}
    assert_metrics(enum, minmax_oracle([rank[v] for v in values], 2))


@pytest.mark.parametrize("impl", ALL)
def test_floats_canonical_nan_last(impl):
    other_nan = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
    s = pl.Series("x", [0.0, -0.0, float("nan"), other_nan, 1.0, -1.0])
    for series in (s, s.cast(pl.Float32)):
        r = stats(impl, series, index_granularity=2)
        assert r["n_distinct"] == 4
        # ranks: −1 → 0, ±0 → 1, 1 → 2, NaN → 3
        assert_metrics(r, minmax_oracle([1, 1, 3, 3, 2, 0], 2))


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_equals_one_chunk(impl):
    s = runs_int(5_000, 1, 7, 40)["c0"]
    parts = pl.concat([s.slice(0, 1_234), s.slice(1_234)], rechunk=False)
    assert parts.n_chunks() == 2
    assert stats(impl, parts, index_granularity=64) == stats(
        impl, s, index_granularity=64
    )


@pytest.mark.parametrize("impl", ALL)
def test_baseline_hand_worked(impl):
    # 1 2 3 4, w = 2: 6 equally likely pairs, 8 of 24 (pair, query) combinations skipped
    r = stats(impl, pl.Series("x", [1, 2, 3, 4]), index_granularity=2)
    assert r["expected_skip_rows"] == pytest.approx(1 / 3, rel=1e-9)
    assert r["expected_skip_values"] == pytest.approx(1 / 3, rel=1e-9)
    assert r["skip_rows_mean"] == 0.5  # sorted beats a random order


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_granularity": 0},
        {"granularity": 0},
        {"index_granularity": 1.5},
        {"granularity": True},
    ],
)
def test_keywords_are_validated(kwargs):
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        reference(PKG)(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical correctness (no conclusions to test: the technique reports metrics only)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize(
    "case",
    [
        (1, 30, 400, 20, 1.0, 0.0),
        (2, 50, 1_000, 30, 1.2, 0.05),
        (3, 20, 300, 15, 1.5, 0.1),
    ],
)
def test_baseline_matches_monte_carlo(impl, case):
    seed, k, n, w, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    r = stats(impl, pl.Series("x", values, dtype=pl.Int64), index_granularity=w)
    mc = sampled(values, w, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "mm_rows"),
        ("expected_skip_values", "mm_values"),
    ):
        mean, se = mc[key]
        assert abs(r[metric] - mean) <= 5 * se + 1e-9, (metric, r[metric], mean, se)
