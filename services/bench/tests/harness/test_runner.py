import numpy as np
import pytest

from synthdata import DatasetSpec, generate
from chbench.harness.ddl import table_name
from chbench.harness.loader import load_table
from chbench.harness.queries import expand_queries
from chbench.harness.runner import Job, explain_query, explain_sql, run_round
from chbench.harness.suite import QueryAxes, QueryKind
from chbench.harness.validate import TableVariant


@pytest.fixture(scope="module")
def loaded(ch):
    ds = generate(
        DatasetSpec(
            rows=200_000,
            arrow_type="float64",
            value_range=(-1.0, 10.0),
            decimal_places=8,
            null_count=0,
            seed=1,
            cardinality_ratio=0.5,
        )
    )
    v = TableVariant(
        ch_type="Float64",
        nullable=False,
        primary_key=True,
        projection=False,
        codec="none",
        skip_index="none",
        compression="LZ4",
        status="ok",
        skip_reason=None,
        warnings=(),
    )
    name = table_name("t_runner", "ffffffff00000000")
    r = load_table(ch, name, v, ds)
    assert r.status == "ok", r.error
    yield name, v, ds
    ch.command(f"DROP TABLE IF EXISTS {name}")


def queries_for(v, ds):
    """Seeded deterministically: the range predicate's interval depends on
    the generator, and comparable results across variants need the same
    seed per variant (the orchestrator's job; here it is `default_rng(0)`)."""
    axes = QueryAxes(
        kinds=[
            QueryKind("full_scan", None, None, None),
            QueryKind("filter", 0.1, "range", None),
        ],
        samples=3,
    )
    return expand_queries(
        axes, ds.array.to_numpy(zero_copy_only=False), v, np.random.default_rng(0)
    )


@pytest.mark.clickhouse
def test_filter_read_rows_covers_the_matched_rows_not_just_boundary_granules(
    ch, loaded
):
    """A bare `count()` under a primary-key range is answered from the index
    (the exact-count projection) and reads only the boundary granules; the
    filter must read the column so `read_rows` is at least the matched rows."""
    name, v, ds = loaded
    [_, filt] = queries_for(v, ds)
    matched = filt.actual_fraction * len(ds.array)
    r = run_round(ch, [Job("filt", name, v, filt)], 0)
    assert not r.errors, r.errors
    assert r.samples["filt"].read_rows >= matched


class _InterruptOnTimedCall:
    """Proxy over `CHClient` that raises `KeyboardInterrupt` in place of the
    `at`-th timed sample query (the ones carrying a `query_id`), simulating
    Ctrl-C while that sample is running."""

    def __init__(self, inner, at):
        self._inner, self._at, self._n = inner, at, 0

    def query_rows(self, sql, **kw):
        if kw.get("query_id") is not None:
            self._n += 1
            if self._n == self._at:
                raise KeyboardInterrupt
        return self._inner.query_rows(sql, **kw)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def test_explain_sql():
    assert (
        explain_sql("bench.t", "SELECT count() FROM bench.t")
        == "EXPLAIN indexes = 1 SELECT count() FROM bench.t"
    )


@pytest.mark.clickhouse
def test_a_live_round_records_server_timings_and_cpu(ch, loaded):
    name, v, ds = loaded
    scan, filt = queries_for(v, ds)
    js = [Job("scan", name, v, scan), Job("filt", name, v, filt)]
    run_round(ch, js, timed=False)
    r = run_round(ch, js, 0)
    assert not r.errors, r.errors
    s, f = r.samples["scan"], r.samples["filt"]
    assert s.cpu_us and s.cpu_us > 0 and s.round_no == 0
    assert s.client_ms > 0 and s.server_ms >= 0 and "SelectedParts" in s.profile_events
    assert f.read_rows < s.read_rows and f.selected_marks < s.selected_marks
    plan, error = explain_query(ch, js[0])
    assert error is None and "ReadFromMergeTree" in plan
