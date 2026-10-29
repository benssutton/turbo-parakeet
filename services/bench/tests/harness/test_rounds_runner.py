import math

import pytest

from chbench.harness.queries import QueryVariant
from chbench.harness.runner import Job, explain_query, run_round
from chbench.harness.validate import TableVariant
from tests.harness.fakes import FakeClient

VARIANT = TableVariant(
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
SCAN = QueryVariant(
    kind="full_scan",
    predicate_kind=None,
    target_fraction=None,
    values_n=None,
    predicate_sql=None,
    actual_fraction=None,
)


def jobs(*names):
    return [Job(key=n, name=f"bench.{n}", variant=VARIANT, qv=SCAN) for n in names]


def test_a_round_runs_each_job_once_and_delivers_filled_samples_in_order():
    client, seen = FakeClient(), []
    r = run_round(
        client,
        jobs("a", "b", "c"),
        7,
        sample_no=2,
        on_sample=lambda k, s: seen.append((k, s)),
    )
    assert [k for k, _ in seen] == ["a", "b", "c"] and not r.errors and not r.warnings
    for _, s in seen:
        assert (s.round_no, s.sample_no, s.cpu_us, s.server_ms) == (7, 2, 150.0, 12.0)
        assert s.selected_marks == 2 and s.client_ms > 0
    assert [c for c in client.calls if c[0] == "log"] == [
        ("log",)
    ]  # one read per round


def test_every_query_runs_with_both_caches_off():
    client = FakeClient()
    run_round(client, jobs("a"), 0)
    assert client.settings == [{"use_query_condition_cache": 0, "use_query_cache": 0}]


def test_a_failing_job_is_recorded_and_the_round_carries_on():
    client = FakeClient(fail_sql="bench.b")
    r = run_round(client, jobs("a", "b", "c"), 0)
    assert list(r.samples) == ["a", "c"] and r.errors["b"] == "RuntimeError: boom"


def test_a_missing_log_row_keeps_nan_and_names_the_id():
    client = FakeClient(missing={1})
    r = run_round(client, jobs("a", "b"), 3)
    assert math.isnan(r.samples["b"].server_ms) and r.samples["b"].cpu_us is None
    assert any("round 3" in w and client.ids[1] in w for w in r.warnings)


def test_an_untimed_round_takes_no_samples_and_reads_no_log():
    client = FakeClient()
    r = run_round(client, jobs("a", "b"), timed=False)
    assert r.samples == {} and client.ids == []
    assert all(c[0] == "query" and c[2] is None for c in client.calls)


def test_cpu_us_is_none_when_the_profile_has_neither_counter():
    r = run_round(FakeClient(profile={"SelectedMarks": 1}), jobs("a"), 0)
    assert r.samples["a"].cpu_us is None


def test_ctrl_c_mid_round_delivers_what_ran_then_propagates():
    client, seen = FakeClient(interrupt_at=2), []
    with pytest.raises(KeyboardInterrupt) as e:
        run_round(client, jobs("a", "b", "c"), 0, on_sample=lambda k, s: seen.append(k))
    assert seen == ["a"] and list(e.value.round_result.samples) == ["a"]
    assert math.isfinite(e.value.round_result.samples["a"].server_ms)


def test_ctrl_c_inside_the_log_read_still_warns_every_pending_sample():
    client, seen = FakeClient(log_interrupt_at=1), []
    with pytest.raises(KeyboardInterrupt) as e:
        run_round(client, jobs("a", "b"), 0, on_sample=lambda k, s: seen.append(s))
    assert len(seen) == 2
    for s in seen:
        named = any(f"sample {s.sample_no}" in w for w in e.value.round_result.warnings)
        assert math.isfinite(s.server_ms) or named


def test_explain_query_returns_the_plan_or_the_error():
    (job,) = jobs("a")
    assert explain_query(FakeClient(), job) == ("ReadFromMergeTree", None)

    class Failing(FakeClient):
        def query_rows(self, sql, **kw):
            raise RuntimeError("nope")

    assert explain_query(Failing(), job) == ("", "RuntimeError: nope")
