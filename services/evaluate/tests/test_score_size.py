import pytest

from evaluate.score import (
    measured_choice,
    size_cells,
    size_frontier,
    size_regret,
    type_gain,
)

ENTRY = {
    "dataset": "d",
    "column": "c",
    "suite": "d__c",
    "role": "size",
    "type_role": "recommended",
    "base": "UInt32",
    "rec_codec": "Delta",
    "rec_compression": "ZSTD(1)",
}
PRED = {
    "UInt32|Delta|ZSTD(1)": {"bytes": 110, "origin": "candidate", "outcome": "chosen"},
    "UInt32|none|LZ4": {"bytes": 400, "origin": "grid", "outcome": "grid"},
    "UInt32|GCD|LZ4": {"bytes": 390, "origin": "grid", "outcome": "grid"},
    "UInt32|DoubleDelta|LZ4": {"bytes": None, "origin": "grid", "outcome": "grid"},
}
TABLES = [
    {
        "table_id": "t1",
        "ch_type": "UInt32",
        "codec": "Delta",
        "compression": "ZSTD(1)",
        "status": "ok",
        "skip_reason": None,
        "compressed_bytes": 100,
        "uncompressed_bytes": 400,
        "insert_ms": 3.0,
    },
    {
        "table_id": "t2",
        "ch_type": "UInt32",
        "codec": "none",
        "compression": "LZ4",
        "status": "ok",
        "skip_reason": None,
        "compressed_bytes": 90,
        "uncompressed_bytes": 400,
        "insert_ms": 3.0,
    },
    {
        "table_id": "t3",
        "ch_type": "UInt32",
        "codec": "GCD",
        "compression": "LZ4",
        "status": "skipped",
        "skip_reason": "GCD not applicable",
        "compressed_bytes": None,
        "uncompressed_bytes": None,
        "insert_ms": None,
    },
    {
        "table_id": "t4",
        "ch_type": "Nullable(UInt32)",
        "codec": "DoubleDelta",
        "compression": "LZ4",
        "status": "ok",
        "skip_reason": None,
        "compressed_bytes": 300,
        "uncompressed_bytes": 400,
        "insert_ms": 3.0,
    },
]


def test_size_cells():
    cells = {(c["codec"], c["compression"]): c for c in size_cells(ENTRY, TABLES, PRED)}
    rec = cells[("Delta", "ZSTD(1)")]
    assert (rec["predicted_bytes"], rec["measured_bytes"]) == (110, 100)
    assert rec["rel_error"] == pytest.approx(0.1)
    assert (rec["origin"], rec["outcome"], rec["original_type"]) == (
        "candidate",
        "chosen",
        False,
    )
    skipped = cells[("GCD", "LZ4")]
    assert skipped["measured_bytes"] is None and skipped["rel_error"] is None
    assert skipped["skip_reason"] == "GCD not applicable"
    nullpred = cells[("DoubleDelta", "LZ4")]  # Nullable(...) base resolves to UInt32
    assert nullpred["predicted_bytes"] is None and nullpred["rel_error"] is None


def test_size_regret_and_rank():
    r = size_regret(ENTRY, size_cells(ENTRY, TABLES, PRED))
    assert (r["recommended_chain"], r["best_chain"]) == ("Delta, ZSTD(1)", "LZ4")
    assert (r["recommended_measured"], r["best_measured"]) == (100, 90)
    assert r["regret_bytes"] == 10 and r["regret_rel"] == pytest.approx(10 / 90)
    assert r["rank"] == 2 and r["n_cells"] == 3


def test_size_regret_null_when_the_recommended_cell_is_missing():
    orig = {**ENTRY, "type_role": "original", "rec_codec": "LowCardinality"}
    r = size_regret(orig, size_cells(orig, TABLES, PRED))
    assert (
        r["recommended_measured"] is None
        and r["regret_bytes"] is None
        and r["rank"] is None
    )
    assert r["best_measured"] == 90


def test_type_gain():
    rec = [
        {"measured_bytes": 100, "predicted_bytes": 110},
        {"measured_bytes": 150, "predicted_bytes": 140},
    ]
    orig = [{"measured_bytes": 200, "predicted_bytes": 220}]
    assert type_gain(rec, orig) == (
        pytest.approx(110 / 220 - 1),
        pytest.approx(100 / 200 - 1),
    )
    assert type_gain(rec, None) == (0.0, 0.0)


AGREE_ENTRY = {
    **ENTRY,
    "rec_codec": "none",
    "rec_compression": "ZSTD(1)",
}


def agree_cell(codec, comp, measured, origin="candidate", outcome="rejected"):
    return {
        "codec": codec,
        "compression": comp,
        "measured_bytes": measured,
        "origin": origin,
        "outcome": outcome,
    }


def test_measured_choice_reapplies_the_rule_to_the_compared_chains():
    cells = [
        agree_cell("none", "LZ4", 1000),
        agree_cell("none", "ZSTD(1)", 950, outcome="chosen"),  # saves only 5%
        agree_cell("GCD", "LZ4", 10, origin="grid", outcome="grid"),
        agree_cell("Delta", "LZ4", 5, outcome="pruned"),
    ]
    assert measured_choice(cells) == "LZ4"
    r = size_regret(AGREE_ENTRY, cells)
    assert (r["measured_choice"], r["agrees"]) == ("LZ4", False)


def test_measured_choice_is_none_when_a_compared_cell_was_not_measured():
    cells = [
        agree_cell("none", "LZ4", None),
        agree_cell("none", "ZSTD(1)", 900, outcome="chosen"),
    ]
    assert measured_choice(cells) is None
    assert size_regret(AGREE_ENTRY, cells)["agrees"] is None


def test_agrees_when_the_rule_picks_the_recommended_chain():
    cells = [
        agree_cell("none", "LZ4", 1000),
        agree_cell("none", "ZSTD(1)", 500, outcome="chosen"),
    ]
    r = size_regret(AGREE_ENTRY, cells)
    assert (r["measured_choice"], r["agrees"]) == ("ZSTD(1)", True)


def test_agrees_for_a_low_cardinality_recommendation():
    entry = {**ENTRY, "rec_codec": "LowCardinality", "rec_compression": "ZSTD(1)"}
    cells = [
        agree_cell("LowCardinality", "LZ4", 1000),
        agree_cell("LowCardinality", "ZSTD(1)", 500, outcome="chosen"),
    ]
    r = size_regret(entry, cells)
    assert (r["recommended_chain"], r["measured_choice"], r["agrees"]) == (
        "ZSTD(1)",
        "ZSTD(1)",
        True,
    )


def test_agreement_is_none_when_the_rule_refuses_the_rows():
    cells = [agree_cell("none", "ZSTD(1)", 500, outcome="chosen")]  # no plain LZ4
    assert measured_choice(cells) is None
    assert size_regret(AGREE_ENTRY, cells)["agrees"] is None


def cost(codec, comp, p50, ratio=None, kind="full_scan"):
    return {
        "codec": codec,
        "compression": comp,
        "skip_index": "none",
        "query_kind": kind,
        "cpu_us_p50": p50,
        "cpu_us_lo": p50 - 1,
        "cpu_us_hi": p50 + 1,
        "cpu_ratio": ratio,
        "cpu_ratio_lo": ratio,
        "cpu_ratio_hi": ratio,
    }


def test_size_frontier_marks_the_pareto_cells_and_the_recommendation():
    cells = [
        agree_cell("none", "ZSTD(1)", 800, outcome="chosen"),  # smallest, dearest
        agree_cell("none", "LZ4", 1000),  # larger, cheaper: on the frontier
        agree_cell("GCD", "LZ4", 1100, origin="grid", outcome="grid"),  # dominated
    ]
    costs = [
        cost("none", "ZSTD(1)", 30.0),
        cost("none", "LZ4", 20.0, 0.66),
        cost("GCD", "LZ4", 25.0, 0.83),
    ]
    rows = {
        (r["codec"], r["compression"]): r
        for r in size_frontier(AGREE_ENTRY, cells, costs)
    }
    assert rows[("none", "ZSTD(1)")]["on_frontier"]
    assert rows[("none", "ZSTD(1)")]["recommended"]
    assert rows[("none", "LZ4")]["on_frontier"]
    assert rows[("none", "LZ4")]["cpu_ratio"] == 0.66
    assert not rows[("none", "LZ4")]["recommended"]
    assert not rows[("GCD", "LZ4")]["on_frontier"]


def test_size_frontier_leaves_out_cells_without_bytes_or_a_scan_cost():
    cells = [
        agree_cell("none", "ZSTD(1)", 800, outcome="chosen"),
        agree_cell("none", "LZ4", None),
        agree_cell("GCD", "LZ4", 900),
    ]
    costs = [cost("none", "ZSTD(1)", 30.0), cost("none", "LZ4", 20.0)]
    rows = size_frontier(AGREE_ENTRY, cells, costs)
    assert [(r["codec"], r["compression"]) for r in rows] == [("none", "ZSTD(1)")]


def test_size_frontier_keeps_ties_on_the_frontier():
    cells = [agree_cell("none", "LZ4", 1000), agree_cell("GCD", "LZ4", 1000)]
    costs = [cost("none", "LZ4", 20.0), cost("GCD", "LZ4", 20.0)]
    assert all(r["on_frontier"] for r in size_frontier(AGREE_ENTRY, cells, costs))


def test_size_frontier_ignores_filter_rows():
    cells = [agree_cell("none", "ZSTD(1)", 800), agree_cell("none", "LZ4", 1000)]
    costs = [
        cost("none", "ZSTD(1)", 30.0),
        cost("none", "LZ4", 20.0),
        cost("none", "ZSTD(1)", 1.0, kind="filter"),
    ]
    rows = {r["compression"]: r for r in size_frontier(AGREE_ENTRY, cells, costs)}
    assert rows["ZSTD(1)"]["cpu_us_p50"] == 30.0 and rows["ZSTD(1)"]["on_frontier"]
