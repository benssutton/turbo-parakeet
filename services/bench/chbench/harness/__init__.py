"""The harness: a suite YAML in, samples in SQLite and a report out.

`run_suite` is the whole of it — it owns the order the modules below it run
in and is what `chbench run` calls. The individual modules
(`suite`, `validate`, `ddl`, `loader`, `queries`, `runner`, `store`,
`report`, `stats`, `quiet`, `client`) stay importable on their own, since each is separately
useful and separately tested; only the names a caller of the harness as a
whole needs are re-exported here.

The re-exports are lazy (PEP 562): an in-process engine's worker process imports
`chbench.harness.engines.worker`, and an eager `orchestrate` import here would load ClickHouse's
client, synthdata and Polars into every worker (about 1.2 s per spawn) for nothing.
"""

import importlib

_EXPORTS = {
    "RunSummary": "chbench.harness.orchestrate",
    "run_suite": "chbench.harness.orchestrate",
    "Suite": "chbench.harness.suite",
    "SuiteError": "chbench.harness.suite",
    "load_suite": "chbench.harness.suite",
    "parse_suite": "chbench.harness.suite",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
