"""
Top-k accuracy tests — every implementation in analytics.top_k.

Oracles: known answers and TopKPolars (exact group-by counts), the reference. Semantics
(spec docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md): each eligible
column's k most frequent non-null values as arrow-rs text, count desc then first row;
`error` is SpaceSaving's over-count bound (0 exact, null HeavyKeeper); `dominated` = top-1
share ≥ dominance_threshold.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from analytics.top_k import TopK
from datagen import mixed_dtypes, top_k_streams
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
    with_metrics,
)

PKG = "analytics.top_k"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
# Columns whose k-th count is not tied (agreement precision is a lower bound under ties).
UNTIED = ["zipf_15", "zipf_11", "zipf_08", "zipf_str"]


def tops(impl: str, frame: pl.DataFrame, **params) -> dict[str, list | None]:
    out = run(load(impl), {"t": frame}, **params)
    return {
        c: None if t is None else [(i["value"], i["count"]) for i in t]
        for c, t in zip(out["col_a"], out["top_k"].to_list())
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "zipf": top_k_streams(20_000),
        "empty": pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"zipf": top_k_streams(200_000, columns=UNTIED)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers


@pytest.mark.parametrize("impl", ALL)
def test_fewer_values_than_k_are_exact(impl):
    s = pl.Series("s", ["a", "a", "a", None, "b", "b", "c"])
    assert tops(impl, s.to_frame())["s"] == [("a", 3), ("b", 2), ("c", 1)]


@pytest.mark.parametrize("impl", ALL)
def test_zipf_top_5(impl):
    frame = top_k_streams(100_000, columns=["zipf_15"])
    want = (
        frame["zipf_15"]
        .value_counts(sort=True)
        .head(5)["zipf_15"]
        .cast(pl.String)
        .to_list()
    )
    assert [v for v, _ in tops(impl, frame)["zipf_15"]] == want


@pytest.mark.parametrize("impl", ALL)
def test_ties_rank_by_first_row(impl):
    s = pl.Series("x", [3, 1, 2, 2, 1, 3])
    assert tops(impl, s.to_frame())["x"] == [("3", 2), ("1", 2), ("2", 2)]


@pytest.mark.parametrize("impl", ALL)
def test_all_null_gives_an_empty_list(impl):
    out = run(
        load(impl), {"t": pl.DataFrame({"x": pl.Series([None] * 4, dtype=pl.Int64)})}
    )
    assert out["top_k"].to_list() == [[]]
    assert out["n_non_null"].to_list() == [0]
    assert out["dominated"].to_list() == [None]


@pytest.mark.parametrize("impl", ALL)
def test_values_render_as_arrow_text(impl):
    frame = pl.DataFrame(
        {
            "cat": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "enum": pl.Series(["p", "p", "q"], dtype=pl.Enum(["q", "p"])),
            "date": [date(2024, 1, 2)] * 2 + [date(2024, 1, 3)],
            "bool": [True, True, False],
            "float": [2.5, 2.5, 1.0],
            "dec": pl.Series(
                [Decimal("1.50"), Decimal("1.50"), Decimal("2.00")],
                dtype=pl.Decimal(10, 2),
            ),
            "bin": [b"ab", b"ab", b"c"],
        }
    )
    got = {c: t[0][0] for c, t in tops(impl, frame).items()}
    assert got == {
        "cat": "x",
        "enum": "p",
        "date": "2024-01-02",
        "bool": "true",
        "float": "2.5",
        "dec": "1.50",
        "bin": "ab",
    }


@pytest.mark.parametrize("impl", ALL)
def test_non_utf8_binary_renders_as_null(impl):
    s = pl.Series("bin", [b"\xff", b"\xff", b"ok"])
    assert tops(impl, s.to_frame())["bin"] == [(None, 2), ("ok", 1)]


@pytest.mark.parametrize("impl", ALL)
def test_nested_and_wide_columns_are_ineligible(impl):
    frame = pl.DataFrame(
        {
            "l": pl.Series([[1], [2]], dtype=pl.List(pl.Int64)),
            "w": pl.Series([1, 2], dtype=pl.Int128),
            "i": [1, 1],
        }
    )
    out = run(load(impl), {"t": frame})
    assert out["status"].to_list() == ["ineligible", "ineligible", "computed"]


@pytest.mark.parametrize("impl", ALL)
def test_parameters_are_checked(impl):
    cls = load(impl)
    for bad in (
        {"k": 0},
        {"k": 1.5},
        {"memory_bytes": -1},
        {"dominance_threshold": 1.5},
    ):
        with pytest.raises(ValueError):
            cls(**bad)


def test_default_budget_follows_k():
    cls = reference(PKG)
    assert cls().memory_bytes == 16 * 1024
    assert cls(k=256).memory_bytes == 256 * 1024
    assert cls(k=5, memory_bytes=1000).memory_bytes == 1000


@pytest.mark.parametrize("impl", OTHERS)
def test_a_budget_below_k_entries_raises(impl):
    with pytest.raises(ValueError, match="memory_bytes"):
        run(load(impl), {"t": pl.DataFrame({"x": [1, 2]})}, memory_bytes=64)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("k", [5, 256])
def test_sketch_bytes_fit_the_budget(impl, k):
    out = run(load(impl), {"t": pl.DataFrame({"x": [1, 2]})}, k=k)
    assert 0 < out["sketch_bytes"][0] <= max(16 * 1024, k * 1024)


def test_space_saving_is_exact_while_values_fit():
    s = pl.Series("x", [i for i in range(100) for _ in range(i + 1)])
    out = run(load(f"{PKG}:TopKSpaceSavingRust"), {"t": s.to_frame()})
    assert out["top_k"].to_list()[0] == [
        {"value": str(v), "count": v + 1, "error": 0} for v in (99, 98, 97, 96, 95)
    ]


def test_heavy_keeper_is_deterministic_per_seed():
    frame = top_k_streams(50_000, columns=["zipf_08"])
    cls = load(f"{PKG}:TopKHeavyKeeperRust")
    assert run(cls, {"t": frame}, seed=3).equals(run(cls, {"t": frame}, seed=3))
    with pytest.raises(ValueError):
        cls(seed=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions


def test_dominated():
    item = lambda n: {"value": "a", "count": n, "error": 0}  # noqa: E731
    cls = with_metrics(
        TopK,
        n_non_null=[10, 10, 0],
        top_k=[[item(6)], [item(4)], []],
        sketch_bytes=[None, None, None],
    )
    frame = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = run(cls, {"t": frame})
    assert out["dominated"].to_list() == [True, False, None]
