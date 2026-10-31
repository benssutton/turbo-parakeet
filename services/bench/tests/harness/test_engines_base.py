"""engines.base.run_round: one round over several engines, the jobs in the order given."""

import dataclasses

import pytest

from chbench.harness.engines.base import job_engine, run_round
from chbench.harness.engines.clickhouse import ClickHouseEngine
from chbench.harness.runner import Job, RoundResult
from chbench.harness.runner import run_round as ch_round
from tests.harness.fakes import FakeClient
from tests.harness.test_rounds_runner import SCAN, VARIANT

POLARS = dataclasses.replace(VARIANT, engine="polars", threads=2, storage="memory")


class FakeRound:
    def __init__(self, log, name):
        self.log, self.name = log, name
        self.result = RoundResult()

    def run(self, job):
        self.log.append(("run", self.name, job.key))
        if job.key == "stop":
            raise KeyboardInterrupt
        self.result.samples[job.key] = self.name

    def finish(self):
        self.log.append(("finish", self.name))
        return self.result

    def interrupt(self):
        self.log.append(("interrupt", self.name))


class FakeEngine:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def start_round(self, round_no, *, sample_no, timed, on_sample):
        self.log.append(("start", self.name, round_no, timed))
        return FakeRound(self.log, self.name)


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
    assert log.index(("finish", "ch")) > log.index(("run", "ch", "c"))
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


def test_clickhouse_behind_the_protocol_is_the_runners_round():
    a, b = FakeClient(), FakeClient()
    jobs = [job("x", VARIANT), job("y", VARIANT)]
    got = run_round({"clickhouse": ClickHouseEngine(a)}, jobs, 2, sample_no=1)
    want = ch_round(b, jobs, 2, sample_no=1)
    assert list(got.samples) == list(want.samples) == ["x", "y"]
    assert [c[0] for c in a.calls] == [c[0] for c in b.calls]
    assert ClickHouseEngine(a).version() == "26.8.31.2"
