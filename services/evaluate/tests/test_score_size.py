import pytest

from evaluate.score import size_cells, size_regret, type_gain

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
