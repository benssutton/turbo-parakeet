"""Canaries: engine limitations the recommendations depend on, as strict xfails. When an engine release
lifts one, its test XPASSes, which fails the run: the reason says what to do. Full operator surveys stay in
the spikes (tests/performance/spike_datafusion_ree.py). Procedure after an upgrade: docs/engines/README.md.
"""

import datafusion
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datafusion import SessionContext


def _min_max(table: pa.Table, column: str) -> list:
    ctx = SessionContext()
    ctx.register_record_batches("t", [table.to_batches()])
    return (
        ctx.sql(f"select min({column}), max({column}) from t")
        .to_arrow_table()
        .to_pylist()
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        f"DataFusion {datafusion.__version__} rejects MIN / MAX over run-end-encoded columns. "
        "If this XPASSes, they work now: ToDo 6, re-run tests/performance/spike_datafusion_ree.py, "
        "then add REE sizes and a recommendation and retire this canary."
    ),
)
@pytest.mark.parametrize(
    "values",
    [
        pa.array([1, 1, 2, None, 3, 3], pa.int64()),
        pa.array(["a", "a", "b", None, "c", "c"]),
    ],
    ids=["int64", "utf8"],
)
def test_datafusion_min_max_over_ree(values):
    plain = pa.table({"v": values})
    ree = pa.table({"v": pc.run_end_encode(values)})
    assert _min_max(ree, "v") == _min_max(plain, "v")
