import json

import polars as pl
import pytest

from evaluate.score import score_run
from evaluate.summary import compare, coverage_line, report, summarise
from tests.test_score_run import _run_dir


def test_summary_sections(tmp_path):
    run = _run_dir(tmp_path)
    notes = score_run(run)
    s = summarise(run, notes)
    assert set(s["section"]) == {
        "size",
        "size_regret",
        "skip",
        "skip_regret",
        "query_regret",
        "cost",
        "coverage",
    }
    reg = s.filter(pl.col("section") == "size_regret").row(0, named=True)
    assert (reg["recommended_chain"], reg["n"], reg["regret_rel_worst"]) == (
        "LZ4",
        1,
        0.0,
    )
    sk = s.filter(pl.col("section") == "skip_regret").row(0, named=True)
    assert sk["chosen_index"] == "minmax:1"
    assert sk["regret_freq_mean"] == pytest.approx(0.0)
    size = s.filter(pl.col("section") == "size").row(0, named=True)
    assert (size["type"], size["codec"], size["compression"], size["n"]) == (
        "UInt32",
        "none",
        "LZ4",
        1,
    )
    assert size["median_abs_error"] == pytest.approx(0.1)
    cov = s.filter(pl.col("section") == "coverage").row(0, named=True)
    assert (cov["columns"], cov["cells_measured"], cov["failed_datasets"]) == (1, 1, 0)
    assert (run.path / "summary.arrow").is_file()


def test_coverage_line_text():
    cov = {
        "columns": 77,
        "columns_skipped": 3,
        "cells_measured": 1412,
        "cells_total": 1436,
        "cells_skipped_by_clickhouse": 24,
        "null_predictions": 0,
        "notes": 2,
        "failed_datasets": 0,
    }
    assert coverage_line(cov) == (
        "columns 77 (skipped 3) · cells 1412 measured / 1436 (24 skipped by ClickHouse) · "
        "null predictions 0 · notes 2 · failed datasets 0"
    )


def test_compare_deltas(tmp_path):
    a, b = _run_dir(tmp_path / "a"), _run_dir(tmp_path / "b")
    summarise(a, score_run(a))
    # b: the recommender predicted 1000 instead of 1100 -> error 0
    p = b.path / "d" / "suites" / "d__c.predicted.json"
    entry = {"bytes": 1000, "origin": "candidate", "outcome": "chosen"}
    p.write_text(json.dumps({"UInt32|none|LZ4": entry}))
    summarise(b, score_run(b))
    d = compare(a.path, b.path)
    row = d.filter(pl.col("section") == "size").row(0, named=True)
    assert row["median_abs_error_delta"] == pytest.approx(-0.1)


def test_report_prints_the_coverage_line(tmp_path, capsys):
    run = _run_dir(tmp_path)
    summarise(run, score_run(run))
    report(run)
    assert "columns 1 (skipped 0)" in capsys.readouterr().out


def test_compare_count_decrease_is_negative(tmp_path):
    for name, n in (("a", 2), ("b", 1)):
        (tmp_path / name).mkdir()
        df = pl.DataFrame({"section": ["size"], "type": ["UInt32"], "n": [n]})
        df.with_columns(pl.col("n").cast(pl.UInt32)).write_ipc(
            tmp_path / name / "summary.arrow"
        )
    d = compare(tmp_path / "a", tmp_path / "b")
    assert d["n_delta"].to_list() == [-1.0]


def test_query_regret_section_counts(tmp_path):
    run = _run_dir(tmp_path)
    s = summarise(run, score_run(run))
    q = s.filter(pl.col("section") == "query_regret")
    assert q.height > 0 and q["n"].sum() >= 1
