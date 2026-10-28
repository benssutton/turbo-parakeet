from chbench.harness.store import Store

from tests.fakes import Q, T, make_store


def test_make_store_round_trips(tmp_path):
    db = make_store(
        tmp_path / "s.db",
        "su",
        [
            T(
                "t1",
                codec="Delta",
                compression="ZSTD(1)",
                compressed=500,
                queries=[Q("q1", "full_scan", marks=4)],
            ),
            T("t2", codec="GCD", status="skipped", skip_reason="no GCD"),
        ],
    )
    with Store(db) as s:
        tables = {r["table_id"]: r for r in s.table_rows("su")}
        rows = s.query_rows("su")
    assert (
        tables["t1"]["compressed_bytes"] == 500 and tables["t2"]["status"] == "skipped"
    )
    assert len(rows) == 1 and rows[0]["selected_marks"] == 4
