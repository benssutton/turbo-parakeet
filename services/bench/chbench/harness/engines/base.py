"""The engine boundary the orchestrator schedules rounds through (ToDo 5).

An engine is ClickHouse (`clickhouse.ClickHouseEngine`: a server, its samples read from
`system.query_log` once per round) or an in-process engine in a worker process
(`inproc.InprocEngine`: Polars or DataFusion, one process per engine and thread count). Each gives
the orchestrator `version()`, `start_round(...)` (a `Round`: `run(job)` per job, then `finish()`;
`interrupt()` after a Ctrl-C) and `close()`. Preparing a table is not part of the protocol:
ClickHouse loads a named table from the dataset (`loader.load_table`), an in-process engine casts
the column its worker already holds (`InprocEngine.prepare`); the orchestrator calls each through
its own helper.

`run_round` is one round over every engine: the jobs run in the order given (the orchestrator's
shuffle by `(seed, round_no)`), each on its own engine, so the engines' queries interleave; then
every engine's round is finished (ClickHouse reads its log) and the results are merged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from chbench.harness.runner import Job, RoundResult, Sample
from chbench.harness.validate import engine_key


@dataclass(frozen=True)
class PrepareResult:
    """An in-process table made ready, or not: `status` ok | skipped | error, `error` the skip
    reason or the error text, `insert_ms` the prepare time (the cast and its round-trip check,
    plus for `ipc` the write and the file's check), `ipc_bytes` the file's size (`ipc` only),
    `memory_bytes` the resident size, `rows`. `stats` is None: the store's ClickHouse storage
    columns stay NULL."""

    status: str
    error: str | None
    insert_ms: float | None
    ipc_bytes: int | None
    memory_bytes: int | None
    rows: int | None
    stats: None = None


class Round(Protocol):
    result: RoundResult

    def run(self, job: Job) -> None: ...

    def finish(self) -> RoundResult: ...

    def interrupt(self) -> None: ...


class Engine(Protocol):
    def version(self) -> str: ...

    def start_round(
        self,
        round_no: int | None,
        *,
        sample_no: int,
        timed: bool,
        on_sample: Callable[[str, Sample], None] | None,
    ) -> Round: ...

    def close(self) -> None: ...


def job_engine(job: Job) -> str:
    """The key of the engine a job runs on (`validate.engine_key`)."""
    return engine_key(job.variant)


def _merge(rounds) -> RoundResult:
    """The rounds' results as one, in engine-start order. Sample and error keys are job keys,
    each job on one engine, so no key is in two rounds (asserted: an update would drop one);
    warnings are concatenated in the same order."""
    out = RoundResult()
    for rd in rounds:
        r = rd.result
        assert not (out.samples.keys() & r.samples.keys()), "a sample key in two rounds"
        assert not (out.errors.keys() & r.errors.keys()), "an error key in two rounds"
        out.samples.update(r.samples)
        out.errors.update(r.errors)
        out.warnings.extend(r.warnings)
    return out


def _interrupt_all(rounds: dict[str, Round], exc: BaseException) -> None:
    """`interrupt()` on every started round; one that raises (a second Ctrl-C, a dead worker)
    is noted on `exc` and the others still run, so `exc` is what propagates."""
    for key, rd in rounds.items():
        try:
            rd.interrupt()
        except BaseException as err:  # noqa: BLE001
            exc.add_note(f"{key}: interrupt() also failed: {err!r}")


def run_round(
    engines: dict,
    jobs: list[Job],
    round_no: int | None = None,
    *,
    sample_no: int = 0,
    timed: bool = True,
    on_sample: Callable[[str, Sample], None] | None = None,
) -> RoundResult:
    """Every job once, in the order given, each on `engines[job_engine(job)]`; an engine is started
    on its first job and finished after the last job of the round. A job whose engine is not in
    `engines` is refused (ValueError) before any job runs.

    Any exception, Ctrl-C included, while the jobs run (or a Ctrl-C while the rounds finish):
    every started round delivers what ran (`interrupt`; one that raises is noted on the original
    exception and the rest still run), the merged partial result is attached as `round_result`,
    and the original exception propagates. A `finish()` that raises anything else does not stop
    the others finishing: the first such exception propagates once all are done, with the later
    ones as notes and the merged result as `round_result`."""
    missing = {job_engine(j) for j in jobs} - engines.keys()
    if missing:
        raise ValueError(f"no engine started for {sorted(missing)}")
    rounds: dict[str, Round] = {}
    first: Exception | None = None
    try:
        for job in jobs:
            key = job_engine(job)
            if key not in rounds:
                rounds[key] = engines[key].start_round(
                    round_no, sample_no=sample_no, timed=timed, on_sample=on_sample
                )
            rounds[key].run(job)
        for key, rd in rounds.items():
            try:
                rd.finish()
            except Exception as err:  # noqa: BLE001
                if first is None:
                    first = err
                else:
                    first.add_note(f"{key}: finish() also failed: {err!r}")
    except BaseException as e:
        if first is not None:
            e.add_note(f"after a finish() failed: {first!r}")
        _interrupt_all(rounds, e)
        e.round_result = _merge(rounds.values())  # type: ignore[attr-defined]
        raise
    if first is not None:
        first.round_result = _merge(rounds.values())  # type: ignore[attr-defined]
        raise first
    return _merge(rounds.values())
