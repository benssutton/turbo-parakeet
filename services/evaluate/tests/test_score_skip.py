import datetime

import numpy as np
import polars as pl
import pytest

from evaluate.predict import MARK_ROWS, SkipPredictor
from evaluate.score import skip_queries, skip_regret

ENTRY = {
    "dataset": "d",
    "column": "c",
    "suite": "d__c__skip",
    "role": "skip",
    "type_role": "recommended",
    "base": "UInt32",
    "rec_codec": "none",
    "rec_compression": "LZ4",
}
PRED = {
    "minmax:1": {"outcome": "chosen", "index_bytes": 40, "skip_rows": 0.75},
    "set(125):1": {"outcome": "rejected", "index_bytes": 60, "skip_rows": 0.5},
    "none": {"outcome": "none", "index_bytes": None, "skip_rows": 0.0},
}
ROWS = 4 * MARK_ROWS


def _row(table, skip, kind, marks, sql=None, idx=0, q=None):
    return {
        "table_id": table,
        "skip_index": skip,
        "kind": kind,
        "selected_marks": marks,
        "predicate_sql": sql,
        "rows": ROWS,
        "query_id": q or f"{table}-{kind}-{sql}",
        "index_compressed_bytes": idx,
        "status": "ok",
        "query_status": "ok",
    }


def _rows():
    rows = []
    for t, skip, idx, marks_for in (
        ("tm", "minmax:1", 42, {0: 1, 1: 1}),
        ("ts", "set(125):1", 61, {0: 2, 1: 4}),
        ("tn", "none", 0, {0: 4, 1: 4}),
    ):
        rows.append(_row(t, skip, "full_scan", 4, idx=idx))
        for v, m in marks_for.items():
            rows.append(_row(t, skip, "filter", m, sql=f"v IN ({v})", idx=idx))
            rows.append(
                _row(t, skip, "filter", m, sql=f"v IN ({v})", idx=idx)
            )  # second sample
    return rows


def _predictor():
    # value 0 fills mark 0 only, value 1 fills marks 1-3
    return SkipPredictor(pl.Series("v", np.repeat(np.array([0, 1, 1, 1]), MARK_ROWS)))


def test_skip_queries_one_row_per_query():
    out, notes = skip_queries(ENTRY, _rows(), _predictor(), PRED)
    assert len(out) == 6 and notes == {"unparsed": 0, "marks_mismatch": 0}
    r = next(r for r in out if r["index"] == "minmax:1" and r["value_sql"] == "0")
    assert r["total_marks"] == 4 and r["measured_skip"] == pytest.approx(0.75)
    assert r["predicted_skip"] == pytest.approx(0.75) and r[
        "skip_error"
    ] == pytest.approx(0.0)
    assert (r["predicted_index_bytes"], r["measured_index_bytes"]) == (40, 42)
    assert (r["granularity"], r["outcome"], r["value_rows"]) == (1, "chosen", MARK_ROWS)


def test_unparsed_values_are_noted():
    rows = [
        _row("tm", "minmax:1", "full_scan", 4),
        _row("tm", "minmax:1", "filter", 1, sql="v BETWEEN 1 AND 2"),
    ]
    out, notes = skip_queries(ENTRY, rows, _predictor(), PRED)
    assert (
        notes["unparsed"] == 1
        and out[0]["predicted_skip"] is None
        and out[0]["note"] == "unparsed"
    )


def test_marks_mismatch_is_noted():
    rows = [
        _row("tm", "minmax:1", "full_scan", 3),
        _row("tm", "minmax:1", "filter", 1, sql="v IN (0)"),
    ]
    out, notes = skip_queries(ENTRY, rows, _predictor(), PRED)
    assert notes["marks_mismatch"] == 1 and out[0]["note"] == "marks_mismatch"


def test_skip_regret_both_weightings():
    out, _ = skip_queries(ENTRY, _rows(), _predictor(), PRED)
    r = skip_regret(ENTRY, out, PRED)
    # minmax measured: v0 0.75 (rows 8192), v1 0.75 (rows 24576); set: v0 0.5, v1 0.0; none 0
    assert r["chosen_index"] == "minmax:1" and r["best_index_freq"] == "minmax:1"
    assert r["chosen_skip_freq"] == pytest.approx(0.75) and r[
        "regret_freq"
    ] == pytest.approx(0.0)
    assert r["chosen_skip_uniform"] == pytest.approx(0.75) and r[
        "regret_uniform"
    ] == pytest.approx(0.0)
    assert (r["chosen_index_bytes"], r["best_index_bytes"]) == (42, 42)
    assert r["recommender_skip_rows"] == 0.75 and r[
        "measured_skip_rows"
    ] == pytest.approx(0.75)


def test_skip_regret_when_an_alternative_wins():
    rows = [
        (
            r | {"selected_marks": 4}
            if r["skip_index"] == "minmax:1" and r["kind"] == "filter"
            else r
        )
        for r in _rows()
    ]
    out, _ = skip_queries(ENTRY, rows, _predictor(), PRED)
    r = skip_regret(ENTRY, out, PRED)
    assert (
        r["best_index_freq"] == "set(125):1" and r["best_index_uniform"] == "set(125):1"
    )
    # set freq-weighted: (0.5 * 8192 + 0.0 * 24576) / 32768 = 0.125; uniform 0.25
    assert r["regret_freq"] == pytest.approx(0.125) and r[
        "regret_uniform"
    ] == pytest.approx(0.25)


def test_temporal_literals_parse_against_the_logical_dtype():
    col = pl.Series(
        "v",
        [datetime.date(2020, 1, 1)] * MARK_ROWS
        + [datetime.date(2020, 1, 2)] * MARK_ROWS,
    )
    rows = [
        _row("tm", "minmax:1", "full_scan", 2) | {"rows": 2 * MARK_ROWS},
        _row("tm", "minmax:1", "filter", 1, sql="v IN ('2020-01-02')")
        | {"rows": 2 * MARK_ROWS},
    ]
    out, notes = skip_queries(ENTRY, rows, SkipPredictor(col), PRED)
    assert notes["unparsed"] == 0 and out[0]["predicted_skip"] == pytest.approx(0.5)
    assert out[0]["value_rows"] == MARK_ROWS


def test_ties_go_to_the_chosen_index_whatever_the_row_order():
    # every index skips the same: the chosen one wins the tie, also when its rows come last
    rows = [r | {"selected_marks": 4} for r in _rows()]
    for ordered in (rows, rows[::-1]):
        out, _ = skip_queries(ENTRY, ordered, _predictor(), PRED)
        r = skip_regret(ENTRY, out, PRED)
        assert r["best_index_freq"] == "minmax:1" and r["regret_freq"] == 0


def test_non_point_predicate_keeps_its_raw_value_sql():
    rows = [
        _row("tm", "minmax:1", "full_scan", 4),
        _row("tm", "minmax:1", "filter", 1, sql="v BETWEEN 1 AND 2"),
    ]
    out, _ = skip_queries(ENTRY, rows, _predictor(), PRED)
    assert out[0]["value_sql"] == "v BETWEEN 1 AND 2"
