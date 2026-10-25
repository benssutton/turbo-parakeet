"""Run one query variant against one table and collect its samples.

Spec section 5.4, step by step:

1. One untimed warm-up run, skipped in `cold` mode.
2. `samples` timed runs. Each has a fresh `uuid4` query id (delivered only
   through `CHClient.query_rows`'s `query_id` argument, which is the sole
   path that reaches the server on clickhouse-connect 1.8.0 — see that
   module's docstring) and the settings `use_query_condition_cache` from the
   variant's `qcc` and `use_query_cache = 0`. In `cold` mode
   `CHClient.drop_caches()` runs before every sample. The client wall clock
   is measured around the query call alone.
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
orchestrator's job under a UTF-8 reconfiguration (FINDINGS section 5).
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

_EXPECTED_DROPS = frozenset({"MARK", "UNCOMPRESSED", "QUERY CONDITION"})
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


@dataclass
class RunResult:
    status: str                 # "ok" | "error" | "interrupted"
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


def _query_settings(qv: QueryVariant) -> dict[str, int]:
    return {"use_query_condition_cache": int(bool(qv.qcc)), "use_query_cache": 0}


def _error_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _read_query_log(client: CHClient, ids: list[str]) -> dict[str, tuple]:
    rows = client.query_rows(_QUERY_LOG_SQL, parameters={"ids": ids})
    return {str(r[0]): r for r in rows}


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


def run_query(client: CHClient, name: str, variant: TableVariant, qv: QueryVariant,
              samples: int, *,
              on_sample: Callable[[Sample], None] | None = None) -> RunResult:
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

    In `cold` mode `drop_caches()` runs before every sample; if any of the
    three caches (`MARK`, `UNCOMPRESSED`, `QUERY CONDITION`) was not dropped
    a warning naming it is appended to `RunResult.warnings`, because a cold
    measurement without the mark cache dropped is not cold.

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
                    f"for query_id {qid} after {_LOG_READ_ATTEMPTS} reads")

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
            settings = _query_settings(qv)
            cold = qv.cache_mode == "cold"
            if not cold:
                client.query_rows(sql, settings=settings)
            for i in range(samples):
                if cold:
                    dropped = client.drop_caches()
                    missing = sorted(_EXPECTED_DROPS - dropped)
                    if missing:
                        result.warnings.append(
                            f"sample {i}: cold mode but these caches were not dropped: "
                            + ", ".join(missing))
                qid = str(uuid.uuid4())
                ran_at = datetime.datetime.now(datetime.timezone.utc)
                t0 = time.perf_counter()
                client.query_rows(sql, query_id=qid, settings=settings)
                client_ms = (time.perf_counter() - t0) * 1000.0
                ids.append(qid)
                result.samples.append(Sample(
                    sample_no=i, client_ms=client_ms, server_ms=_NAN,
                    read_rows=_NAN, read_bytes=_NAN, memory_bytes=_NAN,
                    selected_marks=_NAN, selected_granules=_NAN,
                    profile_events={}, ran_at=ran_at))
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
