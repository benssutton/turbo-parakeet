import math

import numpy as np
import pytest

from synthdata import DatasetSpec, generate
from chbench.harness.ddl import table_name
from chbench.harness.loader import load_table
from chbench.harness.queries import QueryVariant, expand_queries
from chbench.harness.runner import explain_sql, run_query
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


def queries_for(v, ds, cache_mode, qcc=False):
    """Seeded deterministically: the range predicate's interval depends on
    the generator, and comparable results across variants need the same
    seed per variant (the orchestrator's job; here it is `default_rng(0)`)."""
    axes = QueryAxes(
        kinds=[
            QueryKind("full_scan", None, None, None),
            QueryKind("filter", 0.1, "range", None),
        ],
        query_condition_cache=[qcc],
        cache_mode=[cache_mode],
        samples=3,
    )
    return expand_queries(
        axes, ds.array.to_numpy(zero_copy_only=False), v, np.random.default_rng(0)
    )


@pytest.mark.clickhouse
def test_warm_run_collects_every_sample_with_server_timings(ch, loaded):
    name, v, ds = loaded
    for qv in queries_for(v, ds, "warm"):
        r = run_query(ch, name, v, qv, samples=3)
        assert r.status == "ok", r.error
        assert len(r.samples) == 3
        for s in r.samples:
            assert s.client_ms > 0 and s.server_ms >= 0
            assert s.read_rows > 0 and s.selected_marks > 0
            assert "SelectedParts" in s.profile_events
        assert r.explain and "ReadFromMergeTree" in r.explain


@pytest.mark.clickhouse
def test_filter_reads_fewer_rows_than_full_scan_on_a_sorted_table(ch, loaded):
    name, v, ds = loaded
    scan, filt = queries_for(v, ds, "warm")
    rs = run_query(ch, name, v, scan, samples=1).samples[0]
    rf = run_query(ch, name, v, filt, samples=1).samples[0]
    assert rf.read_rows < rs.read_rows
    assert rf.selected_marks < rs.selected_marks


@pytest.mark.clickhouse
def test_cold_mode_drops_caches_and_still_collects(ch, loaded):
    name, v, ds = loaded
    [scan, _] = queries_for(v, ds, "cold")
    r = run_query(ch, name, v, scan, samples=2)
    assert r.status == "ok" and len(r.samples) == 2


@pytest.mark.clickhouse
def test_qcc_setting_is_passed_and_hits_are_reported(ch, loaded):
    """With the query condition cache enabled, the second sample of the same
    filter should record cache hits in ProfileEvents."""
    name, v, ds = loaded
    [_, filt] = queries_for(v, ds, "warm", qcc=True)
    r = run_query(ch, name, v, filt, samples=3)
    assert any(
        s.profile_events.get("QueryConditionCacheHits", 0) > 0 for s in r.samples[1:]
    )


@pytest.mark.clickhouse
def test_filter_read_rows_covers_the_matched_rows_not_just_boundary_granules(
    ch, loaded
):
    """A bare `count()` under a primary-key range is answered from the index
    (the exact-count projection) and reads only the boundary granules; the
    filter must read the column so `read_rows` is at least the matched rows."""
    name, v, ds = loaded
    [_, filt] = queries_for(v, ds, "warm")
    matched = filt.actual_fraction * len(ds.array)
    s = run_query(ch, name, v, filt, samples=1).samples[0]
    assert s.read_rows >= matched


@pytest.mark.clickhouse
def test_on_sample_hook_fires_per_sample_with_complete_samples(ch, loaded):
    name, v, ds = loaded
    [scan, _] = queries_for(v, ds, "warm")
    seen = (
        []
    )  # snapshots taken at hook time, so a later in-place fill cannot mask a gap
    run_query(
        ch,
        name,
        v,
        scan,
        samples=3,
        on_sample=lambda s: seen.append((s.sample_no, s.server_ms, s.read_rows)),
    )
    assert [no for no, _, _ in seen] == [0, 1, 2]
    assert all(math.isfinite(ms) and rows > 0 for _, ms, rows in seen)


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


@pytest.mark.clickhouse
def test_keyboard_interrupt_mid_loop_delivers_finished_samples_then_propagates(
    ch, loaded
):
    name, v, ds = loaded
    [scan, _] = queries_for(v, ds, "warm")
    seen = []
    with pytest.raises(KeyboardInterrupt) as excinfo:
        run_query(
            _InterruptOnTimedCall(ch, at=3),
            name,
            v,
            scan,
            samples=3,
            on_sample=lambda s: seen.append((s.sample_no, s.server_ms, s.read_rows)),
        )
    assert [no for no, _, _ in seen] == [0, 1]
    assert all(math.isfinite(ms) and rows > 0 for _, ms, rows in seen)
    r = excinfo.value.run_result
    assert r.status == "interrupted" and len(r.samples) == 2
    assert all(math.isfinite(s.server_ms) for s in r.samples)


@pytest.mark.clickhouse
def test_keyboard_interrupt_from_hook_delivers_first_complete_then_propagates(
    ch, loaded
):
    name, v, ds = loaded
    [scan, _] = queries_for(v, ds, "warm")
    seen = []

    def hook(s):
        seen.append((s.sample_no, s.server_ms, s.read_rows))
        if s.sample_no == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as excinfo:
        run_query(ch, name, v, scan, samples=3, on_sample=hook)
    assert [no for no, _, _ in seen] == [0, 1]
    assert all(math.isfinite(ms) and rows > 0 for _, ms, rows in seen)
    r = excinfo.value.run_result
    assert r.status == "interrupted"
    # every sample had already run and been filled; delivery stopped at the interrupt
    assert len(r.samples) == 3 and all(math.isfinite(s.server_ms) for s in r.samples)


@pytest.mark.clickhouse
def test_query_error_is_recorded_not_raised(ch, loaded):
    name, v, ds = loaded
    bad = QueryVariant(
        kind="filter",
        predicate_kind="range",
        target_fraction=0.1,
        values_n=None,
        qcc=False,
        cache_mode="warm",
        predicate_sql="no_such_column > 1",
        actual_fraction=0.1,
    )
    r = run_query(ch, name, v, bad, samples=2)
    assert r.status == "error" and "no_such_column" in r.error


def test_explain_sql():
    assert (
        explain_sql("bench.t", "SELECT count() FROM bench.t")
        == "EXPLAIN indexes = 1 SELECT count() FROM bench.t"
    )


# ---------------------------------------------------------------------------
# Fast-suite tests: no live ClickHouse needed. `FakeClient` stands in for
# `CHClient`, driven entirely off call order (like `_InterruptOnTimedCall`
# above), so these exercise the retry/warning/interrupt bookkeeping in
# `run_query` without a server.

_FAKE_VARIANT = TableVariant(
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

_FULL_SCAN = QueryVariant(
    kind="full_scan",
    predicate_kind=None,
    target_fraction=None,
    values_n=None,
    qcc=False,
    cache_mode="warm",
    predicate_sql=None,
    actual_fraction=None,
)

_FULL_SCAN_COLD = QueryVariant(
    kind="full_scan",
    predicate_kind=None,
    target_fraction=None,
    values_n=None,
    qcc=False,
    cache_mode="cold",
    predicate_sql=None,
    actual_fraction=None,
)


class FakeClient:
    """Minimal stand-in for `CHClient`. No server: every `query_rows` call is
    dispatched by inspecting the SQL text, and `sample_ids` records the
    `query_id` of every timed sample call, in order, so a test can target
    "the id for sample N" without depending on what `uuid4` produced.

    - `missing_indices`: sample indices whose `system.query_log` row is
      never returned, simulating a row that never lands.
    - `fail_at_index`: a sample index whose timed query raises
      `RuntimeError` instead of succeeding.
    - `dropped_caches`: the set `drop_caches()` reports as dropped (defaults
      to all three).
    - `log_read_raises_at`: the 1-based count of the `system.query_log` read
      at which to raise `KeyboardInterrupt` instead of returning rows.
    """

    def __init__(
        self,
        *,
        missing_indices=(),
        fail_at_index=None,
        dropped_caches=None,
        log_read_raises_at=None,
    ):
        self.sample_ids: list[str] = []
        self._missing_indices = set(missing_indices)
        self._fail_at_index = fail_at_index
        self._dropped_caches = (
            set(dropped_caches)
            if dropped_caches is not None
            else {"MARK", "UNCOMPRESSED", "QUERY CONDITION"}
        )
        self._log_read_raises_at = log_read_raises_at
        self._log_reads = 0

    def query_rows(self, sql, *, query_id=None, settings=None, parameters=None):
        if "system.query_log" in sql:
            self._log_reads += 1
            if self._log_read_raises_at == self._log_reads:
                raise KeyboardInterrupt
            rows = []
            for qid in parameters["ids"]:
                idx = self.sample_ids.index(qid)
                if idx in self._missing_indices:
                    continue
                rows.append((qid, 123.0, 1000, 2000, 3000, {"SelectedMarks": 5}))
            return rows
        if sql.startswith("EXPLAIN"):
            return [("plan",)]
        # Warm-up or timed sample query.
        if query_id is not None:
            idx = len(self.sample_ids)
            if self._fail_at_index == idx:
                raise RuntimeError("boom")
            self.sample_ids.append(query_id)
        return [(1,)]

    def flush_logs(self) -> None:
        pass

    def drop_caches(self):
        return set(self._dropped_caches)


def test_missing_log_row_leaves_nan_fields_with_a_naming_warning():
    client = FakeClient(missing_indices={1})
    seen = []
    r = run_query(
        client,
        "bench.t",
        _FAKE_VARIANT,
        _FULL_SCAN,
        samples=2,
        on_sample=lambda s: seen.append(s),
    )
    assert r.status == "ok"
    missing = [s for s in r.samples if s.sample_no == 1][0]
    assert math.isnan(missing.server_ms)
    assert math.isnan(missing.read_rows)
    assert any("sample 1" in w and client.sample_ids[1] in w for w in r.warnings)
    ok = [s for s in r.samples if s.sample_no == 0][0]
    assert math.isfinite(ok.server_ms)
    assert [s.sample_no for s in seen] == [0, 1]


def test_exception_on_a_later_sample_still_delivers_the_earlier_ones():
    client = FakeClient(fail_at_index=2)
    seen = []
    r = run_query(
        client,
        "bench.t",
        _FAKE_VARIANT,
        _FULL_SCAN,
        samples=3,
        on_sample=lambda s: seen.append(s),
    )
    assert r.status == "error" and "boom" in r.error
    assert len(r.samples) == 2
    assert [s.sample_no for s in r.samples] == [0, 1]
    assert all(
        math.isfinite(s.server_ms) and math.isfinite(s.read_rows) for s in r.samples
    )
    assert [s.sample_no for s in seen] == [0, 1]


def test_cold_mode_drop_failure_is_recorded_as_a_warning():
    client = FakeClient(
        dropped_caches={"MARK", "UNCOMPRESSED"}
    )  # QUERY CONDITION missing
    r = run_query(client, "bench.t", _FAKE_VARIANT, _FULL_SCAN_COLD, samples=1)
    assert r.status == "ok"
    assert any("QUERY CONDITION" in w for w in r.warnings)


def test_keyboard_interrupt_inside_log_read_still_warns_every_pending_sample():
    """Regression for the `filled` flag window: a `KeyboardInterrupt` raised
    from inside the `system.query_log` read must still leave every sample
    that reaches `on_sample` either filled or named by a warning, so the
    outer handler's `if not filled: fill()` (which will not re-run here,
    since a fill attempt already happened) is not the only safety net."""
    client = FakeClient(log_read_raises_at=1)
    seen = []
    with pytest.raises(KeyboardInterrupt) as excinfo:
        run_query(
            client,
            "bench.t",
            _FAKE_VARIANT,
            _FULL_SCAN,
            samples=2,
            on_sample=lambda s: seen.append(s),
        )
    r = excinfo.value.run_result
    assert r.status == "interrupted"
    assert len(seen) == 2  # both collected samples are delivered once
    for s in seen:
        named = any(f"sample {s.sample_no}" in w for w in r.warnings)
        assert math.isfinite(s.server_ms) or named
