"""A real chbench SQLite store filled through Store's own writers with chosen numbers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

from chbench.harness.store import Store


@dataclass
class Q:
    query_id: str
    kind: str = "filter"  # "full_scan" | "filter"
    predicate_sql: str | None = None
    target_fraction: float | None = None
    marks: int = 0
    status: str = "ok"
    # a query with status != "ok" writes no samples; else one dict per sample: server_ms, memory_bytes, user_us, system_us, round_no
    samples: list[dict] = field(default_factory=lambda: [{}])


@dataclass
class T:
    table_id: str
    ch_type: str = "UInt32"
    codec: str = "none"
    compression: str = "LZ4"
    skip_index: str = "none"
    status: str = "ok"
    skip_reason: str | None = None
    compressed: int | None = 1000
    index_bytes: int = 0
    rows: int = 4 * 8192
    queries: list[Q] = field(default_factory=list)


def make_store(path: Path, suite_name: str, tables: list[T]) -> Path:
    with Store(path) as s:
        s.init_schema()
        run_id = s.begin_run(NS(name=suite_name, source_yaml="name: x"))
        for t in tables:
            variant = NS(
                ch_type=t.ch_type,
                nullable=False,
                primary_key=False,
                projection=False,
                codec=t.codec,
                skip_index=t.skip_index,
                compression=t.compression,
                status="skipped" if t.status == "skipped" else "ok",
                skip_reason=t.skip_reason,
                warnings=(),
            )
            load = None
            if t.status != "skipped":
                stats = None
                if t.status == "ok":
                    stats = NS(
                        compressed_bytes=t.compressed,
                        uncompressed_bytes=4 * t.rows,
                        marks_bytes=16,
                        rows=t.rows,
                        parts=1,
                        index_compressed_bytes=t.index_bytes,
                        index_uncompressed_bytes=t.index_bytes,
                        index_marks_bytes=0,
                    )
                load = NS(
                    status=t.status, error=t.skip_reason, insert_ms=5.0, stats=stats
                )
            s.upsert_table(
                t.table_id, "d1", variant, load, "26.8", suite_name=suite_name
            )
            for q in t.queries:
                qv = NS(
                    kind=q.kind,
                    predicate_kind="in" if q.kind == "filter" else None,
                    target_fraction=q.target_fraction,
                    values_n=1 if q.kind == "filter" else None,
                    actual_fraction=None,
                    predicate_sql=q.predicate_sql,
                )
                s.upsert_query(q.query_id, t.table_id, qv, "", status=q.status)
                for i, smp in enumerate(q.samples if q.status == "ok" else []):
                    server_ms = smp.get("server_ms", 1.0)
                    s.insert_sample(
                        q.query_id,
                        run_id,
                        NS(
                            sample_no=i,
                            round_no=smp.get("round_no", i),
                            cpu_us=smp.get("user_us", 100) + smp.get("system_us", 10),
                            client_ms=server_ms + 50,
                            server_ms=server_ms,
                            read_rows=100,
                            read_bytes=800,
                            memory_bytes=smp.get("memory_bytes", 1000),
                            selected_marks=q.marks,
                            selected_granules=q.marks,
                            profile_events={
                                "UserTimeMicroseconds": smp.get("user_us", 100),
                                "SystemTimeMicroseconds": smp.get("system_us", 10),
                            },
                            ran_at="2026-10-28T00:00:00",
                        ),
                    )
        s.finish_run(run_id)
    return path
