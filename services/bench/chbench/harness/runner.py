"""Run one query variant against one table and collect its samples.

The run, step by step:

1. One untimed warm-up run.
2. `samples` timed runs. Each has a fresh `uuid4` query id (delivered only
   through `CHClient.query_rows`'s `query_id` argument, which is the sole
   path that reaches the server on clickhouse-connect 1.8.0 — see that
   module's docstring) and the settings `use_query_condition_cache = 0` and
   `use_query_cache = 0`. The client wall clock is measured around the query
   call alone.
3. `SYSTEM FLUSH LOGS`, then **one** read of `system.query_log` for every
   sample's id at once (`_QUERY_LOG_SQL`), matched back to the samples by
   id. If a row is missing the read is retried up to three times with a
   short sleep; a sample still missing after that keeps nan server fields
   and a warning naming its id.
4. `EXPLAIN indexes = 1 <sql>` once, joined into one string.

`on_sample` receives only complete samples (server fields filled), so it
runs after step 3. Ctrl-C (`KeyboardInterrupt`) during the sample loop is
caught: the samples that finished are flushed, read and delivered, the
result's status becomes `"interrupted"`, the `RunResult` is attached to the
exception as `run_result`, and the exception is re-raised.

Nothing here prints: the plan text contains box-drawing characters that
raise `UnicodeEncodeError` on a cp1252 console, and printing is the
orchestrator's job under a UTF-8 reconfiguration (FINDINGS §5).
"""

from __future__ import annotations

import datetime
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from chbench.harness.client import CHClient
from chbench.harness.queries import QueryVariant, filter_sql, full_scan_sql
from chbench.harness.validate import TableVariant

_LOG_READ_ATTEMPTS = 3
_LOG_READ_SLEEP_S = 0.5
_NAN = float("nan")

_QUERY_LOG_SQL = """
SELECT query_id, query_duration_ms, read_rows, read_bytes, memory_usage, ProfileEvents
FROM system.query_log
WHERE type = 'QueryFinish'
  AND event_date >= today() - 1
  AND query_id IN {ids:Array(String)}
"""


@dataclass
class Sample:
    """One timed run of a query.

    `client_ms` is the wall clock around the client call alone. The server
    fields (`server_ms` = `query_duration_ms`, `read_rows`, `read_bytes`,
    `memory_bytes` = `memory_usage`, `selected_marks`, `selected_granules`,
    `profile_events`) come from `system.query_log`, read once after every
    sample has run; they are nan (and `profile_events` empty) only if the
    log row never appeared, which `RunResult.warnings` then names.

    `selected_granules` is recorded equal to `ProfileEvents['SelectedMarks']`:
    the server does not expose a per-query granule count separately from the
    mark count, and no conversion is invented here.
    """

    sample_no: int
    client_ms: float
    server_ms: float
    read_rows: int | float
    read_bytes: int | float
    memory_bytes: int | float
    selected_marks: int | float
    selected_granules: int | float
    profile_events: dict[str, int]
    ran_at: datetime.datetime
    round_no: int | None = None
    cpu_us: float | None = None


@dataclass
class RunResult:
    status: str  # "ok" | "error" | "interrupted"
    error: str | None
    samples: list[Sample]
    explain: str
    warnings: list[str] = field(default_factory=list)


def explain_sql(name: str, sql: str) -> str:
    """`EXPLAIN indexes = 1 <sql>`; `name` is accepted for symmetry with the
    other SQL builders and is not needed to render the statement."""
    return f"EXPLAIN indexes = 1 {sql}"


def query_sql(name: str, variant: TableVariant, qv: QueryVariant) -> str:
    """The SQL a `QueryVariant` runs: the full-scan aggregate
    (`count(), sum(toFloat64(v))` or `count(), sum(length(v))`) for
    `full_scan`, otherwise that same aggregate with the variant's predicate
    appended as a `WHERE` clause (`filter_sql`) -- not a bare `count()`,
    because a primary-key range can be answered from the index alone
    without reading the column (see `filter_sql`'s docstring)."""
    if qv.kind == "full_scan":
        return full_scan_sql(name, variant)
    if qv.predicate_sql is None:
        raise ValueError(f"query kind {qv.kind!r} has no predicate_sql")
    return filter_sql(name, variant, qv.predicate_sql)


# Every sample runs warm with both result caches off (ToDo 3 retired the cold / qcc axes).
_SETTINGS = {"use_query_condition_cache": 0, "use_query_cache": 0}


def _error_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _read_query_log(client: CHClient, ids: list[str]) -> dict[str, tuple]:
    rows = client.query_rows(_QUERY_LOG_SQL, parameters={"ids": ids})
    return {str(r[0]): r for r in rows}


def _cpu_us(profile: dict[str, int]) -> float | None:
    """User + System CPU microseconds; ClickHouse omits zero counters, so None when neither is present."""
    parts = [
        profile[k]
        for k in ("UserTimeMicroseconds", "SystemTimeMicroseconds")
        if k in profile
    ]
    return float(sum(parts)) if parts else None


def _fill_from_log(sample: Sample, row: tuple) -> None:
    _, duration_ms, read_rows, read_bytes, memory, events = row
    profile = {str(k): int(v) for k, v in (events or {}).items()}
    sample.server_ms = float(duration_ms)
    sample.read_rows = int(read_rows)
    sample.read_bytes = int(read_bytes)
    sample.memory_bytes = int(memory)
    sample.selected_marks = profile.get("SelectedMarks", 0)
    sample.selected_granules = sample.selected_marks
    sample.profile_events = profile
    sample.cpu_us = _cpu_us(profile)


def run_query(
    client: CHClient,
    name: str,
    variant: TableVariant,
    qv: QueryVariant,
    samples: int,
    *,
    on_sample: Callable[[Sample], None] | None = None,
) -> RunResult:
    """Run `qv` against table `name` `samples` times and collect the results.

    The caller is responsible for building `qv` with a deterministically
    seeded generator: `expand_queries` takes an `rng`, and the range
    predicate's interval depends on it, so results are only comparable
    across the table variants of a suite if the orchestrator seeds that
    generator identically for every variant. This function never calls
    `expand_queries`.

    `on_sample`, when given, is called once per `Sample`, in order, only
    once the sample is complete: after the single `system.query_log` read
    that follows the sample loop has filled its server fields. The caller
    can persist each one as it is delivered.

    Ctrl-C: a `KeyboardInterrupt` raised while the samples are running is
    caught after the current sample; the ids collected so far are flushed
    and read (the same single read), each of those samples is delivered to
    `on_sample`, `status` is set to `"interrupted"`, the partial
    `RunResult` is attached to the exception as `run_result`, and the
    exception is re-raised so the caller still stops. A `KeyboardInterrupt`
    raised from inside `on_sample` (or during the EXPLAIN) is handled the
    same way except that no further samples are delivered; the samples that
    ran are still in `run_result.samples`, filled.

    Any other exception during the warm-up, a sample, the flush, the log
    read or the EXPLAIN sets `status = "error"` with the message and
    returns whatever was collected; nothing is raised. Exceptions raised by
    `on_sample` itself are the caller's and propagate.
    """
    result = RunResult(status="ok", error=None, samples=[], explain="")
    ids: list[str] = []
    filled = False
    delivering = False
    delivered = 0

    def fail(e: BaseException) -> None:
        if result.status == "ok":
            result.status = "error"
            result.error = _error_text(e)

    def fill() -> None:
        """One `system.query_log` read for every collected id, retried while
        rows are missing; fills the samples in place.

        The attempt loop runs inside a `try/finally`: whatever is still
        `pending` when the loop exits -- normally, or via an exception
        (including `KeyboardInterrupt` raised by `flush_logs()`, the log
        read, or the sleep between attempts) -- gets its warning appended in
        the `finally`. That guarantees a sample that reaches `on_sample`
        with nan server fields always has a warning naming it: the caller
        cannot observe a `filled = True` fill() that produced no warnings
        for the ids it never filled.
        """
        nonlocal filled
        pending = dict(zip(ids, result.samples))
        try:
            for attempt in range(_LOG_READ_ATTEMPTS):
                if attempt:
                    time.sleep(_LOG_READ_SLEEP_S)
                client.flush_logs()
                for qid, row in _read_query_log(client, list(pending)).items():
                    sample = pending.pop(qid, None)
                    if sample is not None:
                        _fill_from_log(sample, row)
                if not pending:
                    break
        finally:
            filled = True
            for qid, sample in pending.items():
                result.warnings.append(
                    f"sample {sample.sample_no}: no QueryFinish row in system.query_log "
                    f"for query_id {qid} after {_LOG_READ_ATTEMPTS} reads"
                )

    def deliver() -> None:
        nonlocal delivering, delivered
        delivering = True
        while delivered < len(result.samples):
            sample = result.samples[delivered]
            delivered += 1
            if on_sample is not None:
                on_sample(sample)

    try:
        try:
            sql = query_sql(name, variant, qv)
            settings = _SETTINGS
            client.query_rows(sql, settings=settings)
            for i in range(samples):
                qid = str(uuid.uuid4())
                ran_at = datetime.datetime.now(datetime.timezone.utc)
                t0 = time.perf_counter()
                client.query_rows(sql, query_id=qid, settings=settings)
                client_ms = (time.perf_counter() - t0) * 1000.0
                ids.append(qid)
                result.samples.append(
                    Sample(
                        sample_no=i,
                        client_ms=client_ms,
                        server_ms=_NAN,
                        read_rows=_NAN,
                        read_bytes=_NAN,
                        memory_bytes=_NAN,
                        selected_marks=_NAN,
                        selected_granules=_NAN,
                        profile_events={},
                        ran_at=ran_at,
                    )
                )
        except Exception as e:  # noqa: BLE001
            fail(e)

        if ids:
            try:
                fill()
            except Exception as e:  # noqa: BLE001
                fail(e)
            deliver()

        if result.status == "ok":
            try:
                rows = client.query_rows(explain_sql(name, sql))
                result.explain = "\n".join(str(r[0]) for r in rows)
            except Exception as e:  # noqa: BLE001
                fail(e)
    except KeyboardInterrupt as e:
        # Attached first: `result` is mutated in place below, so even a
        # second Ctrl-C during this handler's own fill()/deliver() still
        # leaves the caller with whatever was filled/delivered by then.
        e.run_result = result  # type: ignore[attr-defined]
        result.status = "interrupted"
        if ids and not filled:
            try:
                fill()
            except Exception as e2:  # noqa: BLE001
                result.error = _error_text(e2)
        if ids and not delivering:
            deliver()
        raise

    return result


@dataclass(frozen=True)
class Job:
    """One (table, query) pair of a round; `key` is its store `query_id`."""

    key: str
    name: str
    variant: TableVariant
    qv: QueryVariant


@dataclass
class RoundResult:
    samples: dict[str, Sample] = field(
        default_factory=dict
    )  # job key -> sample, in job order
    errors: dict[str, str] = field(default_factory=dict)  # job key -> error text
    warnings: list[str] = field(default_factory=list)


def _fill(client: CHClient, pending: dict[str, Sample], warnings: list[str]) -> None:
    """One `system.query_log` read for every pending id, retried while rows are missing; a
    sample still missing when the attempts end -- normally or through an exception, Ctrl-C
    included -- gets a warning naming its id (the `finally`)."""
    try:
        for attempt in range(_LOG_READ_ATTEMPTS):
            if attempt:
                time.sleep(_LOG_READ_SLEEP_S)
            client.flush_logs()
            for qid, row in _read_query_log(client, list(pending)).items():
                sample = pending.pop(qid, None)
                if sample is not None:
                    _fill_from_log(sample, row)
            if not pending:
                break
    finally:
        for qid, sample in pending.items():
            warnings.append(
                f"round {sample.round_no} sample {sample.sample_no}: no QueryFinish row in "
                f"system.query_log for query_id {qid} after {_LOG_READ_ATTEMPTS} reads"
            )


def run_round(
    client: CHClient,
    jobs: list[Job],
    round_no: int | None = None,
    *,
    sample_no: int = 0,
    timed: bool = True,
    on_sample: Callable[[str, Sample], None] | None = None,
) -> RoundResult:
    """Run every job once, in the order given, then read the round's log rows in one go.

    A job whose query raises is recorded in `errors` and the round carries on. `on_sample(key,
    sample)` receives each complete sample, in job order. `timed=False` (the warm-up) runs the
    queries without ids and takes no samples. Ctrl-C: the samples already taken are read and
    delivered, the partial result is attached as `round_result`, and the interrupt propagates.
    """
    result = RoundResult()
    ids: dict[str, str] = {}  # query id -> job key
    state = {"filled": False, "delivering": False, "delivered": 0}

    def finish() -> None:
        if not state["filled"]:
            state["filled"] = True
            pending = {qid: result.samples[key] for qid, key in ids.items()}
            if pending:
                try:
                    _fill(client, pending, result.warnings)
                except (
                    Exception
                ) as e:  # noqa: BLE001 - the samples keep nan, named by _fill
                    result.warnings.append(
                        f"system.query_log read failed: {_error_text(e)}"
                    )
        state["delivering"] = True
        keys = list(result.samples)
        while state["delivered"] < len(keys):
            key = keys[state["delivered"]]
            state["delivered"] += 1
            if on_sample is not None:
                on_sample(key, result.samples[key])

    try:
        for job in jobs:
            sql = query_sql(job.name, job.variant, job.qv)
            if not timed:
                try:
                    client.query_rows(sql, settings=_SETTINGS)
                except Exception as e:  # noqa: BLE001
                    result.errors[job.key] = _error_text(e)
                continue
            qid = str(uuid.uuid4())
            ran_at = datetime.datetime.now(datetime.timezone.utc)
            t0 = time.perf_counter()
            try:
                client.query_rows(sql, query_id=qid, settings=_SETTINGS)
            except Exception as e:  # noqa: BLE001
                result.errors[job.key] = _error_text(e)
                continue
            client_ms = (time.perf_counter() - t0) * 1000.0
            ids[qid] = job.key
            result.samples[job.key] = Sample(
                sample_no=sample_no,
                client_ms=client_ms,
                server_ms=_NAN,
                read_rows=_NAN,
                read_bytes=_NAN,
                memory_bytes=_NAN,
                selected_marks=_NAN,
                selected_granules=_NAN,
                profile_events={},
                ran_at=ran_at,
                round_no=round_no,
            )
        finish()
    except KeyboardInterrupt as e:
        e.round_result = result  # type: ignore[attr-defined]
        if not state["delivering"]:
            try:
                finish()
            except KeyboardInterrupt:
                pass
        raise
    return result


def explain_query(client: CHClient, job: Job) -> tuple[str, str | None]:
    """The job's `EXPLAIN indexes = 1` plan and None, or ("", the error text)."""
    try:
        rows = client.query_rows(
            explain_sql(job.name, query_sql(job.name, job.variant, job.qv))
        )
    except Exception as e:  # noqa: BLE001
        return "", _error_text(e)
    return "\n".join(str(r[0]) for r in rows), None
