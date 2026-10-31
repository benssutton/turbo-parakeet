"""The engines a suite runs on (ToDo 5): ClickHouse (a server) and the in-process engines, Polars
and DataFusion, each in worker processes. `base` is the protocol the orchestrator schedules rounds
through; `clickhouse`, `inproc` (the parent side of a worker), `worker` (its child loop) and
`polars_engine` / `datafusion_engine` (what a worker does) implement it."""
