"""Thin wrapper over `clickhouse_connect.Client` for the harness.

This is the only module that talks to the client library directly, so the
two clickhouse-connect 1.8.0 gotchas measured in FINDINGS §5 are
encoded here once and no later module can trip on them:

1. `query_id` is delivered **inside the `settings` mapping** (`query_rows`).
   A bare `query_id=` keyword to `Client.query` raises `TypeError`; passing
   it as a transport setting sends an HTTP header the server ignores, so the
   server autogenerates its own id and every `system.query_log` lookup by
   the id we sent returns zero rows, silently.
2. Server-side parameter binding (`{name:Type}`) goes through the library's
   own `parameters=` argument, never through settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import clickhouse_connect
import pyarrow as pa


@dataclass(frozen=True)
class Connection:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "Connection":
        """Read `CH_HOST`, `CH_PORT`, `CH_USER`, `CH_PASSWORD`, `CH_DATABASE`,
        defaulting to the Docker container in `CLAUDE.md` (the user and
        password are the literal strings `CLICKHOUSE_USER` and
        `CLICKHOUSE_PASSWORD`)."""
        return cls(
            host=os.environ.get("CH_HOST", "localhost"),
            port=int(os.environ.get("CH_PORT", "8123")),
            user=os.environ.get("CH_USER", "CLICKHOUSE_USER"),
            password=os.environ.get("CH_PASSWORD", "CLICKHOUSE_PASSWORD"),
            database=os.environ.get("CH_DATABASE", "bench"),
        )


class CHClient:
    """The harness's view of a ClickHouse connection."""

    def __init__(self, conn: Connection | None = None):
        conn = conn or Connection.from_env()
        self.connection = conn
        self._client = clickhouse_connect.get_client(
            host=conn.host,
            port=conn.port,
            username=conn.user,
            password=conn.password,
            database=conn.database,
        )

    def command(self, sql: str) -> Any:
        return self._client.command(sql)

    def query_rows(
        self,
        sql: str,
        *,
        query_id: str | None = None,
        settings: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[tuple]:
        """Run `sql` and return its rows as tuples.

        `query_id` is merged into `settings` and passed no other way. From
        FINDINGS §5: "clickhouse_connect 1.8.0's Client.query() has no
        separate `query_id` keyword arg (passing query_id=... raises
        TypeError). Passing it via `transport_settings={"query_id": ...}`
        also does NOT work: transport_settings are sent as HTTP headers, and
        ClickHouse's HTTP interface only recognizes query_id as a URL
        parameter, so the server silently ignores the header and
        autogenerates its own query_id. The call shape that actually works on
        this version is to put query_id INSIDE the `settings` dict."

        `parameters` binds server-side `{name:Type}` placeholders through the
        library's own `parameters=` argument.
        """
        merged = dict(settings) if settings else {}
        if query_id is not None:
            merged["query_id"] = query_id
        result = self._client.query(
            sql,
            parameters=dict(parameters) if parameters else None,
            settings=merged or None,
        )
        return [tuple(r) for r in result.result_rows]

    def insert_arrow(self, table: str, arrow_table: pa.Table) -> None:
        self._client.insert_arrow(table, arrow_table)

    def version(self) -> str:
        return str(self._client.command("SELECT version()"))

    def flush_logs(self) -> None:
        """`SYSTEM FLUSH LOGS`, so `system.query_log` holds the rows for
        queries that have already finished."""
        self._client.command("SYSTEM FLUSH LOGS")
