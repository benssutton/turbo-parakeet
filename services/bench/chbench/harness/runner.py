"""Run one round of (table, query) jobs and collect one sample per job.

A run is rounds, scheduled by the orchestrator; one round, step by step:

1. The warm-up round (`timed=False`) runs every job once without a query id
   and takes no samples; a job that fails there is reported in `errors`.
2. A timed round runs every job once, in the order given. Each call has a
   fresh `uuid4` query id (delivered only through `CHClient.query_rows`'s
   `query_id` argument, which is the sole path that reaches the server on
   clickhouse-connect 1.8.0 — see that module's docstring) and the settings
   `use_query_condition_cache = 0` and `use_query_cache = 0`. The client
   wall clock is measured around the query call alone. A job whose query
   raises is recorded in `errors` and the round carries on.
3. `SYSTEM FLUSH LOGS`, then **one** read of `system.query_log` for every
   id of the round at once (`_QUERY_LOG_SQL`), matched back to the samples
   by id. If a row is missing the read is retried up to three times with a
   short sleep; a sample still missing after that keeps nan server fields
   and a warning naming its round, sample and id.
4. `on_sample(key, sample)` receives each complete sample (server fields
   and `cpu_us` filled), in job order, after that read.

Ctrl-C (`KeyboardInterrupt`) during a round delivers what ran: the samples
already taken are read and delivered, the partial `RoundResult` is attached
to the exception as `round_result`, and the exception is re-raised.
`explain_query` gives each job's `EXPLAIN indexes = 1` plan, once per job
after the rounds.

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
    log row never appeared, which `RoundResult.warnings` then names.

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
                except Exception as e:  # noqa: BLE001
                    # The samples keep nan, named by _fill.
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
