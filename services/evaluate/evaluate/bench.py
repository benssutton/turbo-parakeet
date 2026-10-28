"""The only module that talks to ClickHouse: the server check, and chbench's run_suite in a
process pool with one SQLite store per suite. Resume is chbench's: a rerun skips samples already
held (a table whose queries are all held is not reloaded, so dropping it is safe). Each store
belongs to one suite of this run, so an unfinished run row means a dead worker of this run
(ignore_unfinished). A suite's tables are dropped once it has run, so a large run does not fill
the server's disk."""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from chbench.harness.client import CHClient, Connection
from chbench.harness.orchestrate import run_suite
from chbench.harness.store import Store
from chbench.harness.suite import load_suite
from rich.console import Console


class ServerUnavailable(RuntimeError):
    """ClickHouse is not reachable."""


def check_server() -> str:
    """The server version; ServerUnavailable when it cannot be reached."""
    try:
        return CHClient(Connection.from_env()).version()
    except Exception as e:  # noqa: BLE001 - any client failure means no server
        raise ServerUnavailable(
            f"ClickHouse not reachable ({type(e).__name__}: {e}); "
            "start it with `docker start chbench-ch`"
        ) from e


def drop_tables(store: Path, suite_name: str) -> None:
    """DROP every table the store recorded for `suite_name` (as `chbench clean` does)."""
    with Store(store) as s:
        names = s.tables_for_suite(suite_name)
    client = CHClient(Connection.from_env())
    for name in names:
        client.command(f"DROP TABLE IF EXISTS {name}")


def run_one(yaml_path: Path, store: Path, cache: Path) -> str | None:
    """Run one suite into its own store, then drop its tables; the error text, or None. A failed
    drop is reported after the run's own error (the store still holds the results)."""
    error = None
    suite = None
    try:
        suite = load_suite(yaml_path)
        run_suite(
            suite,
            store_path=store,
            cache_dir=cache,
            ignore_unfinished=True,
            console=Console(file=io.StringIO(), width=200),
        )
    except Exception as e:  # noqa: BLE001 - recorded, the run carries on
        error = f"{type(e).__name__}: {e}"
    if suite is not None and store.is_file():
        try:
            drop_tables(store, suite.name)
        except (
            Exception
        ) as e:  # noqa: BLE001 - cleanup failure must not mask the run's error
            drop = f"drop failed: {type(e).__name__}: {e}"
            error = f"{error}; {drop}" if error else drop
    return error


Runner = Callable[[Path, Path, Path], "str | None"]


def _result(future) -> str | None:
    try:
        return future.result()
    except (
        Exception
    ) as e:  # noqa: BLE001 - e.g. BrokenProcessPool: keep the other suites
        return f"{type(e).__name__}: {e}"


def run_all(
    jobs: list[tuple[Path, Path]],
    cache: Path,
    *,
    workers: int,
    runner: Runner = run_one,
) -> dict[str, str | None]:
    """`jobs` are (suite yaml, stores dir); each suite writes `<stores>/<suite stem>.db`. Returns
    suite stem -> error text or None."""
    for _, stores in jobs:
        stores.mkdir(parents=True, exist_ok=True)
    args = [(y, stores / f"{y.stem}.db", cache) for y, stores in jobs]
    if workers <= 1:
        return {y.stem: runner(y, s, c) for y, s, c in args}
    with ProcessPoolExecutor(workers) as pool:
        futures = {y.stem: pool.submit(runner, y, s, c) for y, s, c in args}
        return {stem: _result(f) for stem, f in futures.items()}
