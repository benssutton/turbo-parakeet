"""The harness: a suite YAML in, samples in SQLite and a report out.

`run_suite` is the whole of it — it owns the order the modules below it run
in and is what `chbench run` calls. The individual modules
(`suite`, `validate`, `ddl`, `loader`, `queries`, `runner`, `store`,
`report`, `client`) stay importable on their own, since each is separately
useful and separately tested; only the names a caller of the harness as a
whole needs are re-exported here.
"""

from chbench.harness.orchestrate import RunSummary, run_suite
from chbench.harness.suite import Suite, SuiteError, load_suite, parse_suite

__all__ = [
    "RunSummary",
    "Suite",
    "SuiteError",
    "load_suite",
    "parse_suite",
    "run_suite",
]
