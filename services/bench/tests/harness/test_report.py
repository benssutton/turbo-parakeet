import csv

import pytest

from chbench.harness.report import (
    _cell,
    filter_where,
    render_table,
    summarise,
    write_csv,
)


def rows():
    base = dict(
        table_id="t",
        ch_type="Float64",
        nullable=0,
        primary_key=0,
        projection=0,
        codec="Delta",
        skip_index="none",
        compression="LZ4",
        kind="filter",
        predicate_kind="range",
        target_fraction=0.1,
        actual_fraction=0.101,
        compressed_bytes=500,
        uncompressed_bytes=1000,
        index_compressed_bytes=30,
        index_uncompressed_bytes=60,
        query_id="q",
    )
    return [
        dict(base, sample_no=i, client_ms=float(v), server_ms=float(v) - 0.5)
        for i, v in enumerate((10, 20, 30, 40, 50))
    ]


def test_summary_stats():
    [s] = summarise(rows(), [50, 90])
    assert s["n"] == 5 and s["client_mean"] == 30 and s["client_max"] == 50
    assert s["client_p50"] == 30 and s["client_p90"] == 46
    assert s["server_mean"] == 29.5
    assert s["compression_ratio"] == 2.0 and s["actual_fraction"] == 0.101
    assert s["index_compressed_bytes"] == 30 and s["index_uncompressed_bytes"] == 60


def test_csv_has_the_index_size_columns(tmp_path):
    path = tmp_path / "o.csv"
    write_csv(summarise(rows(), [50]), path)
    with open(path, newline="") as f:
        [row] = list(csv.DictReader(f))
    assert (
        row["index_compressed_bytes"] == "30"
        and row["index_uncompressed_bytes"] == "60"
    )


def test_groups_by_query_and_sorts_by_axes():
    r = rows()
    other = [
        dict(x, query_id="q2", codec="none", client_ms=1.0, server_ms=0.5) for x in r
    ]
    summary = summarise(other + r, [50])
    assert [s["codec"] for s in summary] == ["Delta", "none"]


def test_render_table_has_a_column_per_requested_percentile():
    t = render_table(summarise(rows(), [50, 99]), [50, 99])
    headers = [c.header for c in t.columns]
    assert (
        "p50 client" in headers
        and "p99 client" in headers
        and "p90 client" not in headers
    )


def test_write_csv_round_trips(tmp_path):
    p = tmp_path / "out.csv"
    write_csv(summarise(rows(), [50]), p)
    with open(p, newline="") as f:
        [row] = list(csv.DictReader(f))
    assert row["codec"] == "Delta" and float(row["client_mean"]) == 30


def test_filter_where():
    r = rows()
    assert filter_where(r, "codec=Delta") == r
    assert filter_where(r, "codec=T64") == []
    assert filter_where(r, "codec=Delta,kind=filter") == r
    assert filter_where(r, None) == r


# -- coverage beyond the brief's fixture ---------------------------------


def test_skipped_table_has_none_compression_ratio():
    """A skipped table has NULL storage stats (store.py's
    `test_skipped_table_is_stored_with_its_reason`): `compressed_bytes` and
    `uncompressed_bytes` come back as `None` from SQLite, and the ratio must
    stay `None` rather than raising `ZeroDivisionError` or `TypeError`."""
    base = dict(
        table_id="t",
        ch_type="Float64",
        nullable=0,
        primary_key=0,
        projection=0,
        codec="Delta",
        skip_index="none",
        compression="LZ4",
        kind="full_scan",
        predicate_kind=None,
        target_fraction=None,
        actual_fraction=None,
        compressed_bytes=None,
        uncompressed_bytes=None,
        query_id="q",
    )
    r = [dict(base, sample_no=0, client_ms=1.0, server_ms=1.0)]
    [s] = summarise(r, [50])
    assert s["compression_ratio"] is None


def test_filter_where_unknown_axis_raises():
    with pytest.raises(ValueError, match="nope"):
        filter_where(rows(), "nope=1")


def test_filter_where_numeric_value_matches_string_form():
    """CLI values arrive as strings; row values are ints/floats. `nullable=0`,
    `nullable=1` and `target_fraction=0.1` must all match by comparing the
    row value's string form to the filter value."""
    r = rows()
    assert filter_where(r, "nullable=0") == r
    assert filter_where(r, "target_fraction=0.1") == r
    assert filter_where(r, "nullable=0") == r
    assert filter_where(r, "nullable=1") == []


def test_summarise_single_sample_group():
    base = dict(
        table_id="t",
        ch_type="Float64",
        nullable=0,
        primary_key=0,
        projection=0,
        codec="Delta",
        skip_index="none",
        compression="LZ4",
        kind="full_scan",
        predicate_kind=None,
        target_fraction=None,
        actual_fraction=None,
        compressed_bytes=500,
        uncompressed_bytes=1000,
        query_id="q",
    )
    r = [dict(base, sample_no=0, client_ms=7.0, server_ms=6.5)]
    [s] = summarise(r, [50])
    assert s["n"] == 1 and s["client_mean"] == 7.0 and s["client_p50"] == 7.0


def test_sort_tolerates_none_vs_float_on_the_same_axis():
    """Reviewer's reproduction: two `full_scan` query blocks that agree on
    every axis up to `target_fraction` (`suite.py`'s `_parse_query_kind`
    requires `target_fraction` for `filter` but never forbids it for
    `full_scan`), one with `target_fraction=None` and one with a float.
    `_sort_key`'s bare `tuple(s[a] for a in _AXES)` would compare `None` to
    `0.5` at that position and raise `TypeError`; `summarise` must instead
    return both in a deterministic order."""
    base = dict(
        table_id="t",
        ch_type="Float64",
        nullable=0,
        primary_key=0,
        projection=0,
        codec="Delta",
        skip_index="none",
        compression="LZ4",
        kind="full_scan",
        predicate_kind=None,
        actual_fraction=None,
        compressed_bytes=500,
        uncompressed_bytes=1000,
        sample_no=0,
        client_ms=1.0,
        server_ms=1.0,
    )
    rows = [
        dict(base, query_id="q_none", target_fraction=None),
        dict(base, query_id="q_half", target_fraction=0.5),
    ]
    summary = summarise(rows, [50])
    assert {s["query_id"] for s in summary} == {"q_none", "q_half"}
    assert len(summary) == 2


def rounds_rows(query_id, codec, cpu, server):
    base = dict(rows()[0], query_id=query_id, table_id=query_id, codec=codec)
    return [
        dict(base, sample_no=i, round_no=i, cpu_us=c, server_ms=s, client_ms=s + 50)
        for i, (c, s) in enumerate(zip(cpu, server))
    ]


def test_cpu_and_server_intervals():
    r = rounds_rows("q", "Delta", [float(v) for v in range(10, 0, -1)], [1.0] * 10)
    [s] = summarise(r, [50])
    assert (s["cpu_p50"], s["cpu_lo"], s["cpu_hi"]) == (5.5, 2.0, 9.0)
    assert (s["server_lo"], s["server_hi"]) == (1.0, 1.0)


def test_baseline_ratios_pair_by_round():
    cpu = [100.0, 120.0, 90.0, 110.0, 105.0, 95.0, 100.0, 115.0]
    base = rounds_rows("qb", "none", cpu, [10.0] * 8)
    half = rounds_rows("qd", "Delta", [c / 2 for c in cpu], [20.0] * 8)
    by = {s["codec"]: s for s in summarise(base + half, [50], baseline="codec=none")}
    d = by["Delta"]
    assert d["cpu_ratio"] == d["cpu_ratio_lo"] == d["cpu_ratio_hi"] == 0.5
    assert d["server_ratio"] == 2.0 and d["n_rounds"] == 8
    assert d["clocks_disagree"] is True  # CPU resolved faster, wall resolved slower
    assert by["none"]["cpu_ratio"] == 1.0 and by["none"]["clocks_disagree"] is False


def test_a_group_without_its_baseline_row_has_no_ratio():
    full = [
        dict(r, kind="full_scan", target_fraction=None, predicate_kind=None)
        for r in rounds_rows("qf", "Delta", [1.0] * 6, [1.0] * 6)
    ]
    rows_ = (
        rounds_rows("qb", "none", [2.0] * 6, [1.0] * 6)
        + rounds_rows("qd", "Delta", [1.0] * 6, [1.0] * 6)
        + full
    )
    by = {s["query_id"]: s for s in summarise(rows_, [50], baseline="codec=none")}
    assert by["qd"]["cpu_ratio"] == 0.5
    s = by["qf"]  # the full_scan group has no codec=none row
    assert s["cpu_ratio"] is None and s["n_rounds"] == 0
    assert s["clocks_disagree"] is None


def _shaped(query_id, codec, cpu, predicate_kind, values_n):
    return [
        dict(r, predicate_kind=predicate_kind, values_n=values_n)
        for r in rounds_rows(query_id, codec, cpu, [1.0] * 6)
    ]


def test_baseline_groups_by_the_predicate_shape():
    cpu = [100.0, 120.0, 90.0, 110.0, 105.0, 95.0]
    rows_ = (
        _shaped("rb", "none", cpu, "range", None)
        + _shaped("rd", "Delta", [c / 2 for c in cpu], "range", None)
        + _shaped("ib", "none", [c * 3 for c in cpu], "in", 1)
        + _shaped("id", "Delta", [c / 4 for c in cpu], "in", 1)
        + _shaped("jb", "none", [c * 5 for c in cpu], "in", 3)
        + _shaped("jd", "Delta", [c / 5 for c in cpu], "in", 3)
    )
    by = {s["query_id"]: s for s in summarise(rows_, [50], baseline="codec=none")}
    assert by["rd"]["cpu_ratio"] == 0.5
    assert by["id"]["cpu_ratio"] == pytest.approx(1 / 12)
    assert by["jd"]["cpu_ratio"] == pytest.approx(1 / 25)


@pytest.mark.parametrize(
    "baseline", ["codec=None", "primary_key=true", "target_fraction=0.10", "codec"]
)
def test_a_baseline_matching_no_row_is_refused(baseline):
    r = rounds_rows("q", "none", [1.0] * 6, [1.0] * 6)
    with pytest.raises(ValueError, match="1/0"):
        summarise(r, [50], baseline=baseline)


def test_clocks_disagree_renders_true_false_and_none_apart():
    assert [
        _cell({"clocks_disagree": v}, "clocks_disagree") for v in (True, False, None)
    ] == ["y", "-", ""]


def test_an_unknown_baseline_axis_is_named():
    with pytest.raises(ValueError, match="kodec"):
        summarise(rows(), [50], baseline="kodec=none")


def test_csv_carries_the_ratio_columns_with_a_baseline(tmp_path):
    p = tmp_path / "o.csv"
    r = rounds_rows("q", "none", [1.0] * 6, [1.0] * 6)
    write_csv(summarise(r, [50], baseline="codec=none"), p)
    with open(p, newline="") as f:
        [row] = list(csv.DictReader(f))
    assert row["cpu_ratio"] == "1.0" and row["cpu_p50"] == "1.0"
