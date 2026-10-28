"""End to end against the live ClickHouse container; skips without it."""

import numpy as np
import polars as pl
import pytest

from evaluate import bench
from evaluate.cli import main

pytestmark = pytest.mark.clickhouse


@pytest.fixture(scope="module")
def server():
    try:
        return bench.check_server()
    except bench.ServerUnavailable as e:
        pytest.skip(str(e))


def test_tiny_run(server, tmp_path):
    n = 50_000
    frame = tmp_path / "tiny.arrow"
    pl.DataFrame(
        {
            "ids": np.arange(n, dtype=np.int64),
            "walk": np.cumsum(np.random.default_rng(0).integers(-3, 4, n)).astype(
                np.int64
            ),
            "cat": np.random.default_rng(1).choice(["x", "y", "z"], n),
        }
    ).write_ipc(frame)
    runs = tmp_path / "runs"
    args = ["run", str(frame), "--workers", "2", "--samples", "1"]
    assert main(args + ["--runs-dir", str(runs)]) == 0
    (run,) = runs.iterdir()
    for name in (
        "size_cells",
        "size_regret",
        "skip_queries",
        "skip_regret",
        "query_cost",
        "recommend_cost",
    ):
        assert (run / f"{name}.arrow").is_file(), name
    summary = pl.read_ipc(run / "summary.arrow")
    cov = summary.filter(pl.col("section") == "coverage").row(0, named=True)
    assert cov["failed_datasets"] == 0 and cov["cells_measured"] > 0
    cells = pl.read_ipc(run / "size_cells.arrow").filter(
        pl.col("rel_error").is_not_null() & (pl.col("type") != "String")
    )
    err = (pl.col("predicted_bytes") - pl.col("measured_bytes")).abs()
    bound = pl.max_horizontal(pl.col("measured_bytes") * 0.02, pl.lit(256))
    assert cells.filter(err > bound).height == 0, cells.filter(err > bound)
    skips = pl.read_ipc(run / "skip_queries.arrow").filter(
        pl.col("index").str.starts_with("minmax")
        | pl.col("index").str.starts_with("set")
    )
    assert skips.height > 0 and (skips["skip_error"].abs() < 1e-12).all()
    cost = pl.read_ipc(run / "query_cost.arrow")
    assert cost["memory_bytes_p50"].is_not_null().all()
    assert (
        cost["cpu_us_p50"].is_not_null().any()
    )  # ProfileEvents may omit CPU on a trivial query
    from chbench.harness.client import CHClient, Connection

    left = CHClient(Connection.from_env()).query_rows(
        r"SELECT name FROM system.tables WHERE database = 'bench' AND name LIKE 'tiny\_\_%'"
    )
    assert not left, left  # every suite's tables were dropped after it ran
