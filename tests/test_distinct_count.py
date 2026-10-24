"""
Distinct-count accuracy tests — every implementation in
turbo_parakeet.techniques.distinct_count. Oracles: hand-worked counts and
DistinctCountPolars (the exact reference). Estimators are compared within their own
tolerance (HLL: 4 standard errors; Polars approx_n_unique: 5%).

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import polars as pl
import pytest

from datagen import large_dataset, low_cardinality, mixed_dtypes
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.distinct_count"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def count(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def tolerance(impl: str) -> float:
    """Relative tolerance of an implementation's estimate: 0 for exact ones."""
    cls = load(impl)
    return 0.0 if cls.EXACT else cls().RTOL


# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [
        lambda: large_dataset(),
        lambda: mixed_dtypes(2_000),
        lambda: low_cardinality(200_000, 4, cardinality=100_000),
    ],
    ids=["large", "mixed", "highcard"],
)
def test_agrees_with_reference(impl, make):
    frames = {"t": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers


@pytest.mark.parametrize("impl", ALL)
def test_small_column_counts_non_null_values(impl):
    s = pl.Series("x", [(i % 50) if i % 7 else None for i in range(1_000)])
    r = count(impl, s)
    assert r["n_non_null"] == 857
    # HLL p=14 reads 49.07 here (a register collision, deterministic): within the
    # implementation's own tolerance, exact for the reference.
    assert r["n_distinct"] == pytest.approx(50, rel=tolerance(impl))
    if r["std_error"] is not None:  # Polars' sketch exposes no interval
        assert r["low"] <= r["n_distinct"] <= r["high"] <= 857


@pytest.mark.parametrize("impl", ALL)
def test_floats_are_canonical(impl):
    r = count(impl, pl.Series("f", [0.0, -0.0, float("nan"), float("nan"), None]))
    assert r["n_distinct"] == pytest.approx(2, rel=tolerance(impl))


@pytest.mark.parametrize("impl", ALL)
def test_categorical_counts_like_string(impl):
    s = pl.Series("s", ["a", "b", "a", "c"] * 50)
    for other in (pl.Categorical, pl.Enum(["a", "b", "c"])):
        assert count(impl, s.cast(other))["n_distinct"] == count(impl, s)["n_distinct"]


@pytest.mark.parametrize("impl", ALL)
def test_other_dtypes_are_counted(impl):
    import datetime as dt
    from decimal import Decimal

    frame = pl.DataFrame(
        {
            "dec": pl.Series([Decimal("1.5"), Decimal("2.5"), Decimal("1.5")] * 10),
            "date": pl.Series([dt.date(2020, 1, d) for d in (1, 2, 3)] * 10),
            "dtm": pl.Series([dt.datetime(2020, 1, 1, h) for h in (1, 2, 3)] * 10),
            "dur": pl.Series([dt.timedelta(seconds=s) for s in (1, 2, 3)] * 10),
            "time": pl.Series([dt.time(1, m) for m in (1, 2, 3)] * 10),
            "bool": pl.Series([True, False, True] * 10),
        }
    )
    got = run(load(impl), {"t": frame}).select("col_a", "status", "n_distinct")
    want = {"dec": 2, "date": 3, "dtm": 3, "dur": 3, "time": 3, "bool": 2}
    for row in got.iter_rows(named=True):
        assert row["status"] == "computed"
        assert row["n_distinct"] == pytest.approx(
            want[row["col_a"]], rel=tolerance(impl)
        )


def test_reference_is_exact_and_hll_reports_its_error():
    from turbo_parakeet.techniques.distinct_count import (
        DistinctCountHllRust,
        DistinctCountPolars,
    )

    s = pl.Series("x", list(range(10_000)))
    exact = run(DistinctCountPolars, {"t": s.to_frame()}).row(0, named=True)
    assert exact["n_distinct"] == exact["low"] == exact["high"] == 10_000
    assert exact["std_error"] == 0.0 and exact["sketch_bytes"] is None
    hll = run(DistinctCountHllRust, {"t": s.to_frame()}, precision=10).row(
        0, named=True
    )
    assert hll["std_error"] == pytest.approx(1.04 / 32)
    assert hll["sketch_bytes"] == 1024
    assert DistinctCountHllRust.EXACT is False
    with pytest.raises(ValueError):
        DistinctCountHllRust(precision=3)


def test_polars_approx_accepts_and_ignores_precision():
    from turbo_parakeet.techniques.distinct_count import DistinctCountPolarsApprox

    s = pl.Series("x", list(range(5_000))).to_frame()
    a = run(DistinctCountPolarsApprox, {"t": s}).row(0, named=True)
    b = run(DistinctCountPolarsApprox, {"t": s}, precision=10).row(0, named=True)
    assert a["n_distinct"] == b["n_distinct"]
    assert a["std_error"] is None and a["low"] is None
