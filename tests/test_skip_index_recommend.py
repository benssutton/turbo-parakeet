"""Skip-index recommendations (spec 2026-10-20): keyword validation and one-shot known answers."""

import numpy as np
import polars as pl
import pytest

from analytics.recommend import OneShotRecommender, StreamingRecommender

BAD = [
    {"skip_granularities": ()},
    {"skip_granularities": (1, 1)},
    {"skip_granularities": (0,)},
    {"set_max_sizes": (0,)},
    {"bloom_fprs": (0.0,)},
    {"bloom_fprs": (1.0,)},
    {"min_skip": 1.5},
    {"max_overflow": -0.1},
]


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
@pytest.mark.parametrize(
    "kwargs", BAD, ids=lambda k: next(iter(k)) + str(next(iter(k.values())))
)
def test_keywords_are_validated(cls, kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        cls(**kwargs)


N = 200_000


def recommend(values, dtype=None, **kw):
    s = pl.Series("x", values)
    if dtype is not None:
        s = s.cast(dtype)
    row = OneShotRecommender(**kw).add(pl.DataFrame([s])).result().row(0, named=True)
    return row["clickhouse_skip_index"], row["clickhouse_skip_indexes"]


def show(rows):
    return "\n".join(str(r) for r in rows)


def sorted_ids():
    return np.arange(N)


def clustered():
    rng = np.random.default_rng(1)
    return np.repeat(rng.permutation(200) * 1_000_003 % 2**31, 1_000)


def random_50():
    return np.random.default_rng(2).integers(0, 50, N)


def grouped_hashes(bits=62):
    return np.repeat(np.random.default_rng(3).integers(0, 2**bits, N // 10), 10)


def test_sorted_unique_column_gets_minmax():
    index, rows = recommend(sorted_ids())
    assert index is not None and index.startswith("minmax"), show(rows)


def test_clustered_high_cardinality_gets_set():
    index, rows = recommend(clustered())
    assert index is not None and index.startswith("set(125)"), show(rows)


def test_random_order_gets_no_index():
    index, rows = recommend(random_50())
    assert index is None, show(rows)
    assert rows and all(r["outcome"] in ("rejected", "pruned") for r in rows), show(
        rows
    )


def test_grouped_random_hashes_get_bloom():
    index, rows = recommend(grouped_hashes())
    assert index is not None and index.startswith("bloom_filter(0.025)"), show(rows)


def test_decimal_column_has_no_bloom_candidate():
    # scale 2: a scale-0 Decimal is narrowed to UInt64 by the type rules, which does take a bloom filter
    values = pl.Series("x", (grouped_hashes(56) % 10**17) / 100.0).cast(
        pl.Decimal(18, 2)
    )
    index, rows = recommend(values)
    bloom = [r for r in rows if r["kind"] == "bloom_filter"]
    assert all(r["outcome"] == "pruned" for r in bloom), show(rows)
    assert bloom and all("R36" in r["evidence"] for r in bloom), show(rows)
    assert index is None or not index.startswith("bloom_filter")


def test_min_skip_moves_the_boundary():
    index, rows = recommend(sorted_ids(), min_skip=0.99)
    assert index is None, show(rows)


def test_skip_granularities_limit_the_rows():
    _, rows = recommend(sorted_ids(), skip_granularities=(1,))
    measured = [r for r in rows if r["skip_rows"] is not None]
    assert measured and all(r["granularity"] == 1 for r in measured), show(rows)


def test_set_max_sizes_above_the_window_distinct_leave_the_gate():
    # ~820 distinct per window: set(1000) does not overflow, so the gate (not the overflow rule) decides
    _, rows = recommend(
        grouped_hashes(), skip_granularities=(1,), set_max_sizes=(1000,)
    )
    sets = [r for r in rows if r["kind"] == "set"]
    assert sets and any(r["outcome"] != "pruned" for r in sets), show(rows)


FRAMES = {
    "sorted": sorted_ids,
    "clustered": clustered,
    "random": random_50,
    "hashes": grouped_hashes,
}


def streamed_row(frame, batch=50_000, **kw):
    rec = StreamingRecommender(**kw)
    for off in range(0, frame.height, batch):
        rec.add(frame.slice(off, batch))
    return rec.result().row(0, named=True)


def chosen(r):
    return next(
        (c for c in r["clickhouse_skip_indexes"] if c["outcome"] == "chosen"), None
    )


@pytest.mark.parametrize("name", FRAMES)
def test_streaming_skip_index_matches_one_shot(name):
    frame = pl.DataFrame({"x": FRAMES[name]()})
    one = OneShotRecommender().add(frame).result().row(0, named=True)
    row = streamed_row(frame, reservoir_rows=frame.height + 65_536)
    assert row["clickhouse_skip_index"] == one["clickhouse_skip_index"], show(
        row["clickhouse_skip_indexes"]
    )
    a, b = chosen(row), chosen(one)
    assert (a is None) == (b is None)
    if a is not None:
        assert a["skip_rows"] == pytest.approx(b["skip_rows"], abs=0.02)


def test_streaming_skip_index_without_sample():
    frame = pl.DataFrame({"x": sorted_ids()})
    row = streamed_row(frame, reservoir_rows=0)
    assert row["clickhouse_skip_index"] is None
    assert not row["clickhouse_skip_indexes"]


def test_streaming_granularity_capped_by_block_rows():
    frame = pl.DataFrame({"x": sorted_ids()})
    row = streamed_row(frame, reservoir_rows=frame.height, block_rows=16_384)
    rows = row["clickhouse_skip_indexes"]
    for r in rows:
        if r["granularity"] >= 4:
            assert r["outcome"] == "pruned" and "exceeds block_rows" in r["evidence"]
    assert {r["granularity"] for r in rows if r["skip_rows"] is not None} <= {1, 2}
    assert any(r["skip_rows"] is not None for r in rows), show(rows)
    assert {4, 8} <= {r["granularity"] for r in rows if r["outcome"] == "pruned"}


# ---- to_chbench: the skip-index axis and point queries (spec §8.2) ------------------------


def chbench_suites(tmp_path, frames, **kw):
    import json

    import yaml

    from analytics.recommend import to_chbench

    frame = pl.DataFrame(frames)
    result = OneShotRecommender(**kw).add(frame).result()
    to_chbench(result, frame, tmp_path)
    suites = {p.stem: yaml.safe_load(p.read_text()) for p in tmp_path.glob("*.yaml")}
    predicted = {
        p.name.removesuffix(".predicted.json"): json.loads(p.read_text())
        for p in tmp_path.glob("*.predicted.json")
    }
    return result, suites, predicted


def test_to_chbench_skip_index_axis_and_predictions(tmp_path):
    result, suites, predicted = chbench_suites(
        tmp_path,
        {
            "sorted": sorted_ids(),
            "clustered": clustered(),
            "bloom": grouped_hashes(),
            "random": random_50(),
        },
    )
    rows = {r["column"]: r for r in result.iter_rows(named=True)}
    assert suites["sorted"]["clickhouse"]["skip_index"] == ["minmax:1", "none"]
    assert suites["clustered"]["clickhouse"]["skip_index"] == ["set(125):1", "none"]
    assert suites["bloom"]["clickhouse"]["skip_index"] == [
        "bloom_filter(0.025):1",
        "none",
    ]
    assert suites["random"]["clickhouse"]["skip_index"] == ["none"]
    for name, skip in (("sorted", "minmax:1"), ("clustered", "set(125):1")):
        ch = suites[name]["clickhouse"]
        chosen = [
            c for c in rows[name]["clickhouse_skip_indexes"] if c["outcome"] == "chosen"
        ][0]
        for k in predicted[name]:
            parts = k.split("|")
            if len(parts) == 4:
                assert parts[3] == skip
                assert predicted[name][k]["index_bytes"] == chosen["bytes"] > 0
                assert (
                    predicted[name]["|".join(parts[:3])]
                    == predicted[name][k]["column_bytes"]
                )
        assert any(len(k.split("|")) == 4 for k in predicted[name])
        cells = {
            f"{t}|{k}|{c}"
            for t in ch["type"]
            for k in ch["codec"]
            for c in ch["compression"]
        }
        assert {k for k in predicted[name] if k.count("|") == 2} == cells
    assert all(k.count("|") == 2 for k in predicted["random"])
    for name, s in suites.items():
        assert s["queries"]["kinds"][0] == {"kind": "full_scan"}
        ins = [k for k in s["queries"]["kinds"] if k.get("predicate") == "in"]
        assert [k["target_fraction"] for k in ins] == [0.001, 0.01, 0.05]
        assert all(k["values"] == 1 for k in ins)


def test_to_chbench_original_suite_keeps_no_index(tmp_path):
    _, suites, predicted = chbench_suites(tmp_path, {"sorted": sorted_ids()})
    assert suites["sorted"]["clickhouse"]["skip_index"][0].startswith("minmax")
    assert suites["sorted_original"]["clickhouse"]["skip_index"] == ["none"]
    assert all(k.count("|") == 2 for k in predicted["sorted_original"])


def test_to_chbench_granularity_suffix(tmp_path):
    _, suites, predicted = chbench_suites(
        tmp_path, {"sorted": sorted_ids()}, skip_granularities=(4,)
    )
    assert suites["sorted"]["clickhouse"]["skip_index"] == ["minmax:4", "none"]
    assert any(k.endswith("|minmax:4") for k in predicted["sorted"])


def test_to_chbench_skip_suites_load_in_chbench(tmp_path):
    import os
    import sys
    from pathlib import Path

    root = Path(
        os.environ.get("CHBENCH_DIR")
        or Path(__file__).resolve().parents[1].parent / "clickhouse_benchmark"
    )
    if not root.exists():
        pytest.skip("clickhouse_benchmark not found")
    sys.path.insert(0, str(root))
    pytest.importorskip("chbench")
    from chbench.harness.suite import load_suite

    _, suites, _ = chbench_suites(
        tmp_path,
        {"sorted": sorted_ids(), "clustered": clustered(), "bloom": grouped_hashes()},
        skip_granularities=(1, 4),
    )
    for name in suites:
        s = load_suite(tmp_path / f"{name}.yaml")
        assert s.clickhouse.skip_index[-1] == "none"


# ---- review fixes ---------------------------------------------------------------------------


def enum_minmax_skip(categories, cls=OneShotRecommender):
    n = 16_384
    s = pl.Series("x", ["a"] * n + ["b"] * n + ["c"] * n).cast(pl.Enum(categories))
    r = cls().add(pl.DataFrame([s])).result().row(0, named=True)
    assert r["clickhouse_type"].startswith("Enum8("), r["clickhouse_type"]
    (row,) = [
        c
        for c in r["clickhouse_skip_indexes"]
        if c["kind"] == "minmax" and c["granularity"] == 1
    ]
    return row["skip_rows"]


def enum_expected(mixed_ab_skip):
    # Three runs of 16 384 rows, windows of 8 192 rows at every start: 24 579 pure windows skip
    # the two other values (2/3 of the rows); 8 191 windows straddle each of the two run
    # boundaries. a|b skips c iff c lies outside [a, b] by category order; b|c always skips a.
    windows = 3 * 16_384 - 8_192 + 1
    pure = windows - 2 * 8_191
    return (pure * 2 / 3 + 8_191 * mixed_ab_skip + 8_191 / 3) / windows


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
def test_enum_minmax_follows_category_order(cls):
    in_order = enum_minmax_skip(["a", "b", "c"], cls)
    assert in_order == pytest.approx(enum_expected(1 / 3), abs=1e-3)
    # Categories (a, c, b): [a, b] spans c's code, so the a|b windows skip nothing.
    swapped = enum_minmax_skip(["a", "c", "b"], cls)
    assert swapped == pytest.approx(enum_expected(0.0), abs=1e-3)
    assert in_order - swapped > 0.05


def test_set_max_sizes_above_uint32_is_refused():
    with pytest.raises(ValueError, match="set_max_sizes"):
        OneShotRecommender(set_max_sizes=(2**32,))


def test_unmeasured_bloom_fpr_is_pruned_with_the_rule():
    _, rows = recommend(grouped_hashes(), bloom_fprs=(0.1,), skip_granularities=(1,))
    bloom = [r for r in rows if r["kind"] == "bloom_filter"]
    assert bloom and all(r["outcome"] == "pruned" for r in bloom), show(rows)
    assert all(
        r["evidence"]
        == "bloom_filter fpr 0.1 not measured (chbench R37: 0.001, 0.01, 0.025, 0.05)"
        for r in bloom
    ), show(rows)
