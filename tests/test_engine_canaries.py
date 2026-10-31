"""Canaries: engine limitations the recommendations depend on. A canary reports XFAIL while its limitation
holds and FAILS, saying what to do, when a release lifts it or it changes. Full operator surveys stay in
the spikes (tests/performance/spike_datafusion_ree.py). Procedure after an upgrade: docs/engines/README.md.
"""

import datafusion
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datafusion import SessionContext

_EXPECTED = "Min/Max accumulator not implemented"


def _min_max(table: pa.Table, column: str) -> list:
    ctx = SessionContext()
    ctx.register_record_batches("t", [table.to_batches()])
    return (
        ctx.sql(f"select min({column}), max({column}) from t")
        .to_arrow_table()
        .to_pylist()
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
    # Plain baseline first: a failure here is infrastructure and fails loudly.
    expected = _min_max(pa.table({"v": values}), "v")
    ree = pa.table({"v": pc.run_end_encode(values)})
    try:
        got = _min_max(ree, "v")
    except Exception as e:  # DataFusion raises a bare Exception
        # pytest.fail / pytest.xfail raise BaseException subclasses: this except does not catch them.
        if _EXPECTED not in str(e):
            pytest.fail(f"MIN / MAX over REE failed differently: {e}", pytrace=False)
        pytest.xfail(
            f"DataFusion {datafusion.__version__} rejects MIN / MAX over run-end-encoded columns"
        )
    pytest.fail(
        f"DataFusion {datafusion.__version__} now runs MIN / MAX over run-end-encoded columns "
        f"(result matches plain: {got == expected}): ToDo 6, re-run "
        "tests/performance/spike_datafusion_ree.py, then add REE sizes and a recommendation "
        "and retire this canary.",
        pytrace=False,
    )
