"""A stand-in for `CHClient` with no server, for the runner and orchestrator tests.

Every call is dispatched on its SQL text. `calls` records, in order, ("query", sql, query_id) for
each sample or warm-up query and ("log",) for each `system.query_log` read, so a test can split
the call stream into rounds. `ids` holds every timed query id in order.

- fail_sql: a substring; a query whose SQL contains it raises RuntimeError("boom").
- interrupt_at: the 1-based timed call that raises KeyboardInterrupt instead of running.
- untimed_interrupt_at: the 1-based untimed query call (a warm-up query, no query id) that
  raises KeyboardInterrupt instead of running.
- log_interrupt_at: the 1-based log read that raises KeyboardInterrupt.
- missing: indices into `ids` whose log row never appears.
- profile: the ProfileEvents each log row carries.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

PROFILE = {
    "SelectedMarks": 2,
    "UserTimeMicroseconds": 100,
    "SystemTimeMicroseconds": 50,
}


class FakeClient:
    def __init__(
        self,
        *,
        fail_sql=None,
        interrupt_at=None,
        untimed_interrupt_at=None,
        log_interrupt_at=None,
        missing=(),
        profile=None,
    ):
        self.connection = NS(host="fake", port=0, database="bench")
        self.calls: list[tuple] = []
        self.ids: list[str] = []
        self.settings: list = []
        self._fail_sql = fail_sql
        self._interrupt_at = interrupt_at
        self._untimed_interrupt_at = untimed_interrupt_at
        self._log_interrupt_at = log_interrupt_at
        self._missing = set(missing)
        self._profile = dict(PROFILE if profile is None else profile)
        self._timed = 0
        self._untimed = 0
        self._log_reads = 0

    def version(self) -> str:
        return "26.8.31.2"

    def command(self, sql):
        return None

    def flush_logs(self) -> None:
        pass

    def query_rows(self, sql, *, query_id=None, settings=None, parameters=None):
        if "system.query_log" in sql:
            self._log_reads += 1
            self.calls.append(("log",))
            if self._log_interrupt_at == self._log_reads:
                raise KeyboardInterrupt
            return [
                (qid, 12.0, 1000, 8000, 4096, dict(self._profile))
                for qid in parameters["ids"]
                if self.ids.index(qid) not in self._missing
            ]
        if sql.startswith("EXPLAIN"):
            return [("ReadFromMergeTree",)]
        if "system." in sql:  # quietness probes
            return [(0,)]
        self.calls.append(("query", sql, query_id))
        self.settings.append(settings)
        if query_id is not None:
            self._timed += 1
            if self._interrupt_at == self._timed:
                raise KeyboardInterrupt
        else:
            self._untimed += 1
            if self._untimed_interrupt_at == self._untimed:
                raise KeyboardInterrupt
        if self._fail_sql and self._fail_sql in sql:
            raise RuntimeError("boom")
        if query_id is not None:
            self.ids.append(query_id)
        return [(1,)]


def rounds(calls: list[tuple]) -> list[list[tuple]]:
    """The timed query calls of each round: the stream split at every log read."""
    out, current = [], []
    for c in calls:
        if c[0] == "log":
            out.append(current)
            current = []
        elif c[2] is not None:
            current.append(c)
    return out
