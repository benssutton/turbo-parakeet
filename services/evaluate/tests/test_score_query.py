import pytest

from evaluate.score import chosen_table, query_cost, query_regret

SIZE = {
    "dataset": "d",
    "column": "c",
    "suite": "s",
    "role": "size",
    "type_role": "recommended",
    "rec_codec": "none",
    "rec_compression": "LZ4",
}
CPU = [100.0, 120.0, 90.0, 110.0, 105.0, 95.0, 100.0, 115.0]


def rows(table_id, codec, factor, kind="full_scan", sql=None):
    return [
        {
            "table_id": table_id,
            "query_id": f"{table_id}-{kind}",
            "ch_type": "UInt32",
            "codec": codec,
            "compression": "LZ4",
            "skip_index": "none",
            "kind": kind,
            "target_fraction": None,
            "predicate_sql": sql,
            "round_no": i,
            "cpu_us": c * factor,
            "server_ms": 10.0 * factor,
            "client_ms": 60.0,
            "memory_bytes": 100,
            "read_rows": 10,
            "read_bytes": 80,
            "profile_events_json": None,
        }
        for i, c in enumerate(CPU)
    ]


def test_chosen_table_matches_the_recommended_chain_or_the_chosen_index():
    r = rows("t1", "none", 1.0) + rows("t2", "Delta", 0.5)
    assert chosen_table(SIZE, r, {}) == "t1"
    assert chosen_table({**SIZE, "type_role": "original"}, r, {}) is None
    skip = {**SIZE, "role": "skip"}
    r2 = [dict(x, skip_index="minmax:1") for x in rows("t3", "none", 1.0)]
    assert chosen_table(skip, r2, {"minmax:1": {"outcome": "chosen"}}) == "t3"


def test_query_cost_gains_intervals_and_ratios_against_the_chosen_table():
    cost = {
        c["codec"]: c
        for c in query_cost(
            SIZE, rows("t1", "none", 1.0) + rows("t2", "Delta", 0.5), "t1"
        )
    }
    d, n = cost["Delta"], cost["none"]
    assert (
        d["cpu_ratio"] == d["cpu_ratio_lo"] == d["cpu_ratio_hi"] == pytest.approx(0.5)
    )
    assert d["server_ratio"] == pytest.approx(0.5) and d["n_rounds"] == 8
    assert n["cpu_ratio"] is None and n["n_rounds"] is None
    assert n["cpu_us_lo"] == 90.0 and n["cpu_us_hi"] == 120.0  # n = 8: ranks 1 and 8


def test_query_regret_names_the_fastest_resolved_alternative():
    r = rows("t1", "none", 1.0) + rows("t2", "Delta", 0.5) + rows("t3", "T64", 1.2)
    (q,) = query_regret(SIZE, r, "t1")
    assert (q["chosen_variant"], q["best_variant"]) == ("LZ4", "Delta, LZ4")
    assert q["resolved"] is True and q["regret_cpu"] == pytest.approx(0.5)


def test_a_slower_alternative_gives_no_regret():
    (q,) = query_regret(SIZE, rows("t1", "none", 1.0) + rows("t3", "T64", 1.2), "t1")
    assert q["resolved"] is False and q["regret_cpu"] == 0.0
    assert q["cpu_ratio"] == pytest.approx(1.2)


def test_no_chosen_table_no_regret_rows():
    assert query_regret(SIZE, rows("t1", "none", 1.0), None) == []
