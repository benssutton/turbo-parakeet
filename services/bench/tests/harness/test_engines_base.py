"""engines.base.run_round: one round over several engines, the jobs in the order given."""

import dataclasses

import pytest

from chbench.harness.engines.base import _merge, job_engine, run_round
from chbench.harness.engines.clickhouse import ClickHouseEngine
from chbench.harness.runner import Job, RoundResult
from chbench.harness.runner import run_round as ch_round
from tests.harness.fakes import FakeClient
from tests.harness.test_rounds_runner import SCAN, VARIANT

POLARS = dataclasses.replace(VARIANT, engine="polars", threads=2, storage="memory")


class FakeRound:
    def __init__(self, log, name, fail=()):
        self.log, self.name, self.fail = log, name, fail
        self.result = RoundResult()

    def run(self, job):
        self.log.append(("run", self.name, job.key))
        if job.key == "stop":
            raise KeyboardInterrupt
        if job.key == "boom":
            raise RuntimeError("worker died")
        self.result.samples[job.key] = self.name

    def finish(self):
        self.log.append(("finish", self.name))
        if "finish_ki" in self.fail:
            raise KeyboardInterrupt
        if "finish" in self.fail:
            raise RuntimeError(f"{self.name} finish failed")
        return self.result

    def interrupt(self):
        self.log.append(("interrupt", self.name))
        if "interrupt" in self.fail:
            raise RuntimeError(f"{self.name} interrupt failed")


class FakeEngine:
    def __init__(self, log, name, fail=()):
        self.log, self.name, self.fail = log, name, fail

    def start_round(self, round_no, *, sample_no, timed, on_sample):
        self.log.append(("start", self.name, round_no, timed))
        return FakeRound(self.log, self.name, self.fail)


def job(key, variant):
    return Job(key=key, name=key, variant=variant, qv=SCAN)


def test_job_engine():
    assert job_engine(job("a", VARIANT)) == "clickhouse"
    assert job_engine(job("b", POLARS)) == "polars:2"


def test_jobs_run_in_order_on_their_engines_then_every_round_finishes():
    log = []
    engines = {"clickhouse": FakeEngine(log, "ch"), "polars:2": FakeEngine(log, "pl")}
    r = run_round(engines, [job("a", VARIANT), job("b", POLARS), job("c", VARIANT)], 4)
    runs = [e for e in log if e[0] == "run"]
    assert runs == [("run", "ch", "a"), ("run", "pl", "b"), ("run", "ch", "c")]
    last_run = max(i for i, e in enumerate(log) if e[0] == "run")
    assert log.index(("finish", "ch")) > last_run
    assert log.index(("finish", "pl")) > last_run
    assert ("start", "ch", 4, True) in log and r.samples == {
        "a": "ch",
        "b": "pl",
        "c": "ch",
    }


def test_an_engine_without_jobs_is_not_started():
    log = []
    engines = {"clickhouse": FakeEngine(log, "ch"), "polars:2": FakeEngine(log, "pl")}
    run_round(engines, [job("b", POLARS)], timed=False)
    assert [e[1] for e in log if e[0] == "start"] == ["pl"]


def test_ctrl_c_interrupts_every_started_round_and_attaches_what_ran():
    log = []
    engines = {"clickhouse": FakeEngine(log, "ch"), "polars:2": FakeEngine(log, "pl")}
    with pytest.raises(KeyboardInterrupt) as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS), job("stop", POLARS)])
    assert ("interrupt", "ch") in log and ("interrupt", "pl") in log
    assert info.value.round_result.samples == {"a": "ch", "b": "pl"}


PL = "polars:2"


def test_any_failure_in_run_interrupts_every_started_round_and_attaches_what_ran():
    log = []
    engines = {"clickhouse": FakeEngine(log, "ch"), PL: FakeEngine(log, "pl")}
    with pytest.raises(RuntimeError, match="worker died") as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS), job("boom", POLARS)])
    assert ("interrupt", "ch") in log and ("interrupt", "pl") in log
    assert not any(e[0] == "finish" for e in log)
    assert info.value.round_result.samples == {"a": "ch", "b": "pl"}


def test_a_raising_finish_still_finishes_the_other_rounds():
    log = []
    engines = {
        "clickhouse": FakeEngine(log, "ch", fail=("finish",)),
        PL: FakeEngine(log, "pl"),
    }
    with pytest.raises(RuntimeError, match="ch finish failed") as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS)])
    assert ("finish", "pl") in log
    assert info.value.round_result.samples == {"a": "ch", "b": "pl"}


def test_every_raising_finish_is_named_on_the_first():
    log = []
    engines = {
        "clickhouse": FakeEngine(log, "ch", fail=("finish",)),
        PL: FakeEngine(log, "pl", fail=("finish",)),
    }
    with pytest.raises(RuntimeError, match="ch finish failed") as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS)])
    assert any("pl finish failed" in n for n in info.value.__notes__)


def test_ctrl_c_while_finishing_interrupts_every_round():
    log = []
    engines = {
        "clickhouse": FakeEngine(log, "ch", fail=("finish_ki",)),
        PL: FakeEngine(log, "pl"),
    }
    with pytest.raises(KeyboardInterrupt) as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS)])
    assert ("interrupt", "ch") in log and ("interrupt", "pl") in log
    assert info.value.round_result.samples == {"a": "ch", "b": "pl"}


def test_a_raising_interrupt_skips_no_other_round_and_keeps_the_original():
    log = []
    engines = {
        "clickhouse": FakeEngine(log, "ch", fail=("interrupt",)),
        PL: FakeEngine(log, "pl"),
    }
    with pytest.raises(KeyboardInterrupt) as info:
        run_round(engines, [job("a", VARIANT), job("b", POLARS), job("stop", POLARS)])
    assert ("interrupt", "pl") in log
    assert info.value.round_result.samples == {"a": "ch", "b": "pl"}
    assert any("ch interrupt failed" in n for n in info.value.__notes__)


def test_an_unknown_engine_is_refused_before_any_job_runs():
    log = []
    engines = {"clickhouse": FakeEngine(log, "ch")}
    with pytest.raises(ValueError, match=r"no engine started for \['polars:2'\]"):
        run_round(engines, [job("a", VARIANT), job("b", POLARS)])
    assert log == []


def test_merge_takes_every_engines_samples_errors_and_warnings_in_start_order():
    a, b = FakeRound([], "ch"), FakeRound([], "df")
    a.result = RoundResult(samples={"a": 1}, errors={"x": "e1"}, warnings=["w1"])
    b.result = RoundResult(samples={"b": 2}, errors={"y": "e2"}, warnings=["w2", "w3"])
    m = _merge([a, b])
    assert m.samples == {"a": 1, "b": 2} and m.errors == {"x": "e1", "y": "e2"}
    assert m.warnings == ["w1", "w2", "w3"]


def test_merge_refuses_a_key_two_engines_share():
    a, b = FakeRound([], "ch"), FakeRound([], "df")
    a.result.samples["k"] = 1
    b.result.samples["k"] = 2
    with pytest.raises(AssertionError):
        _merge([a, b])


def test_clickhouse_behind_the_protocol_is_the_runners_round():
    a, b = FakeClient(), FakeClient()
    jobs = [job("x", VARIANT), job("y", VARIANT)]
    got = run_round({"clickhouse": ClickHouseEngine(a)}, jobs, 2, sample_no=1)
    want = ch_round(b, jobs, 2, sample_no=1)
    assert list(got.samples) == list(want.samples) == ["x", "y"]
    assert [c[0] for c in a.calls] == [c[0] for c in b.calls]
    assert ClickHouseEngine(a).version() == "26.8.31.2"
