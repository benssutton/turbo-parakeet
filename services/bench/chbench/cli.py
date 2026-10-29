"""The `chbench` console script: `run`, `report`, `datasets`, `clean`.

The surface:

    chbench run  suites/x.yaml [--force] [--samples-add N] [--dry-run]
                 [--explain] [--ignore-unfinished] [--allow-unsolved]
    chbench report --suite x [--csv out.csv] [--where "codec=T64"] [--baseline EXPR]
    chbench datasets [--metrics]
    chbench clean --suite x

Two options beyond these are added to every command that needs
them: `--store` (default `results.db`) and `--cache-dir` (default
`datasets/`), which are the two paths `CLAUDE.md` fixes at the project root.
They exist because `datasets` has to be told which cache to list and because
nothing else would let a test, or a second experiment, run against its own
store instead of the project's.

The connection is read from the environment by `Connection.from_env`
(`CH_HOST`, `CH_PORT`, `CH_USER`, `CH_PASSWORD`, `CH_DATABASE`); the CLI
adds no flags for it, so there is one way to point the harness at a server.

Exit codes: 0 on success, 2 on a usage or "nothing to work with" error (no
such store, no results for that suite, no such suite file, no reachable
server), 130 on Ctrl-C -- the shell convention, so a `run` interrupted
halfway is distinguishable from one that failed. Every error message goes to stderr, and the guards run
*before* the store is opened: `sqlite3.connect` creates its file, so
reporting on a suite with no results must not be what creates `results.db`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from clickhouse_connect.driver.exceptions import OperationalError
from rich.console import Console
from rich.table import Table

from synthdata.generate import Manifest
from chbench.harness.client import CHClient, Connection
from chbench.harness.orchestrate import (
    UnfinishedRunError,
    UnsolvedDatasetError,
    default_console,
    force_utf8,
    run_suite,
)
from chbench.harness.report import filter_where, render_table, summarise, write_csv
from chbench.harness.store import Store, StoreVersionError
from chbench.harness.suite import SuiteError, load_suite, parse_suite

DEFAULT_STORE = "results.db"
DEFAULT_CACHE_DIR = "datasets"

# Used only when a suite's own stored YAML cannot supply them; the two
# example suites both ask for exactly these.
_DEFAULT_PERCENTILES = (50, 90, 99)

# The metrics a manifest can carry, in the order `datasets --metrics` shows
# them. Mirrors `synthdata.generate._TARGETABLE`; a metric absent from a
# manifest's `achieved` renders as a dash rather than an error, so the two
# lists drifting apart degrades the display instead of breaking it.
_METRICS = (
    "cardinality_ratio",
    "entropy_bits",
    "index_of_dispersion",
    "runs_z",
    "interarrival_cv",
)

# What `ddl.table_name` appends to a suite name: an underscore and the first
# eight hex characters of the table id. `clean` matches on this rather than
# on the bare `<suite>_` prefix, so suite `t_a` does not claim suite `t_a_b`'s
# tables -- they were correctly never dropped, but they were wrongly reported
# as orphans.
_TABLE_SUFFIX_RE = re.compile(r"^[0-9a-f]{8}$")


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = default_console()
    try:
        return args.handler(args, console)
    except KeyboardInterrupt:
        # `run_suite` has already finished the run and committed every
        # delivered sample; this only sets the exit code.
        console.print("interrupted", style="yellow")
        return 130
    except OperationalError as e:
        # A wrong CH_HOST/CH_PORT, or a stopped container. Worth naming
        # rather than showing a urllib3 traceback, since the connection is
        # configured through the environment and the message has to say
        # which host and port were actually tried.
        conn = Connection.from_env()
        return _fail(
            f"cannot reach ClickHouse at {conn.host}:{conn.port} "
            f"as user {conn.user!r}: {e}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chbench",
        description="Benchmark ClickHouse storage and query optimisations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite: load the grid and take samples")
    run.add_argument("suite", help="path to the suite YAML file")
    run.add_argument(
        "--force",
        action="store_true",
        help="re-run every query, ignoring the samples already held",
    )
    run.add_argument(
        "--samples-add",
        type=_non_negative,
        default=0,
        metavar="N",
        help="raise each query's target to at least N more than it holds",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the expanded grid and stop; nothing is created",
    )
    run.add_argument(
        "--allow-unsolved",
        action="store_true",
        help="run a dataset whose solve could not bracket a target (invalid_bracket)",
    )
    run.add_argument(
        "--explain",
        action="store_true",
        help="also print each query's EXPLAIN plan (always stored)",
    )
    run.add_argument(
        "--ignore-unfinished",
        action="store_true",
        help="start even though the store holds an unfinished run of "
        "this suite (only if that process is no longer running)",
    )
    _add_store(run)
    _add_cache_dir(run)
    run.set_defaults(handler=_cmd_run)

    report = sub.add_parser("report", help="summarise a suite's stored samples")
    report.add_argument("--suite", required=True, help="suite name")
    report.add_argument("--csv", metavar="PATH", help="also write the summary as CSV")
    report.add_argument(
        "--where",
        metavar="EXPR",
        help='filter, e.g. "codec=T64,kind=filter"; the boolean '
        "axes nullable, primary_key and projection are "
        "written 1/0, not the y/- the table shows",
    )
    report.add_argument(
        "--baseline",
        metavar="EXPR",
        help='reference row per group, e.g. "codec=none,compression=LZ4": adds CPU and wall '
        "ratios paired by round, with intervals",
    )
    _add_store(report)
    report.set_defaults(handler=_cmd_report)

    datasets = sub.add_parser("datasets", help="list the cached datasets")
    datasets.add_argument(
        "--metrics",
        action="store_true",
        help="also show each dataset's achieved metrics",
    )
    _add_cache_dir(datasets)
    datasets.set_defaults(handler=_cmd_datasets)

    clean = sub.add_parser("clean", help="drop a suite's ClickHouse tables")
    clean.add_argument("--suite", required=True, help="suite name")
    _add_store(clean)
    clean.set_defaults(handler=_cmd_clean)

    return parser


def _add_store(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        metavar="PATH",
        help=f"SQLite results database (default {DEFAULT_STORE})",
    )


def _add_cache_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        metavar="PATH",
        help=f"Arrow dataset cache (default {DEFAULT_CACHE_DIR}/)",
    )


def _non_negative(text: str) -> int:
    """An argparse type for `--samples-add`. A negative count would
    give a query a negative want, which takes nothing and then reports a mean
    of nan; rejecting it at parse time is the only place it can be caught
    before the run has started."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or greater, got {value}")
    return value


def _fail(message: str) -> int:
    print(f"chbench: {message}", file=sys.stderr)
    return 2


def _existing_store(path: str) -> Path | None:
    """The store path if it is already a file, else None. Never opens it:
    `sqlite3.connect` would create an empty database as a side effect of a
    command that has nothing to read."""
    p = Path(path)
    return p if p.is_file() else None


# -- run ---------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    suite_path = Path(args.suite)
    if not suite_path.is_file():
        return _fail(f"no such suite file: {suite_path}")
    try:
        suite = load_suite(suite_path)
    except SuiteError as e:
        return _fail(f"{suite_path}: {e}")

    try:
        summary = run_suite(
            suite,
            store_path=args.store,
            cache_dir=args.cache_dir,
            force=args.force,
            samples_add=args.samples_add,
            dry_run=args.dry_run,
            explain=args.explain,
            ignore_unfinished=args.ignore_unfinished,
            allow_unsolved=args.allow_unsolved,
            console=console,
        )
    except (UnfinishedRunError, UnsolvedDatasetError, StoreVersionError) as e:
        return _fail(str(e))

    console.print(
        f"\ntables: {summary.tables_ok} ok, {summary.tables_skipped} skipped, "
        f"{summary.tables_error} error | queries: {summary.queries_run} run, "
        f"{summary.queries_skipped_existing} already held | "
        f"samples written: {summary.samples_written}",
        style="bold",
        markup=False,
        highlight=False,
    )
    return 0


# -- report ------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    store_path = _existing_store(args.store)
    if store_path is None:
        return _fail(
            f"no results database at {args.store}; "
            f"nothing to report for suite {args.suite!r}"
        )

    with Store(store_path) as store:
        rows = store.query_rows(args.suite)
        percentiles = _percentiles_for(store, args.suite)

    if not rows:
        return _fail(f"no results for suite {args.suite!r} in {args.store}")

    try:
        rows = filter_where(rows, args.where)
    except ValueError as e:
        return _fail(str(e))
    if not rows:
        return _fail(f"no rows for suite {args.suite!r} match {args.where!r}")

    try:
        summary = summarise(rows, percentiles, baseline=args.baseline)
    except ValueError as e:
        return _fail(str(e))
    console.print(
        f"report: {args.suite} ({len(summary)} queries, {len(rows)} samples)",
        style="bold",
        markup=False,
        highlight=False,
    )
    console.print(render_table(summary, percentiles))

    if args.csv:
        write_csv(summary, args.csv)
        console.print(f"wrote {args.csv}", markup=False, highlight=False)
    return 0


def _percentiles_for(store: Store, suite_name: str) -> list[int]:
    """The percentiles the suite asked for, taken from the suite YAML stored
    with its most recent run.

    `Store.latest_suite_yaml` holds the exact text that produced these
    samples, which is the text whose `results.percentiles` they should be
    summarised under -- the file on disk may since have been edited, moved or
    deleted. Falls back to `_DEFAULT_PERCENTILES` when there is no stored
    YAML or it no longer parses.
    """
    yaml_text = store.latest_suite_yaml(suite_name)
    if yaml_text is None:
        return list(_DEFAULT_PERCENTILES)
    try:
        return parse_suite(yaml_text).results.percentiles
    except SuiteError:
        return list(_DEFAULT_PERCENTILES)


# -- datasets ----------------------------------------------------------------


def _cmd_datasets(args: argparse.Namespace, console: Console) -> int:
    cache_dir = Path(args.cache_dir)
    paths = sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    if not paths:
        console.print(
            f"no cached datasets in {cache_dir}", markup=False, highlight=False
        )
        return 0

    # Read and parse each manifest once: the warnings printed under the table
    # come from the same objects the rows were built from.
    manifests = [
        (path, Manifest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        for path in paths
    ]

    table = Table(title=f"cached datasets in {cache_dir}")
    for column in (
        "dataset id",
        "rows",
        "arrow type",
        "nulls",
        "seed",
        "MB",
        "warnings",
    ):
        table.add_column(column)
    if args.metrics:
        for metric in _METRICS:
            table.add_column(metric.replace("_", " "))

    for path, manifest in manifests:
        spec = manifest.spec
        arrow = path.with_suffix(".arrow")
        megabytes = f"{arrow.stat().st_size / 1e6:.1f}" if arrow.is_file() else "-"
        row = [
            manifest.dataset_id,
            str(spec.rows),
            spec.arrow_type,
            str(spec.null_count),
            str(spec.seed),
            megabytes,
            str(len(manifest.warnings)),
        ]
        if args.metrics:
            row += [_metric_cell(manifest, metric) for metric in _METRICS]
        table.add_row(*row)

    console.print(table)
    if args.metrics:
        for _, manifest in manifests:
            for warning in manifest.warnings:
                console.print(
                    f"warning {manifest.dataset_id}: {warning}",
                    style="yellow",
                    markup=False,
                    highlight=False,
                    soft_wrap=True,
                )
    return 0


def _metric_cell(manifest: Manifest, metric: str) -> str:
    """`achieved` alone for a metric that was only measured; `achieved
    (target)` with a tick or cross for one that was targeted, so a miss is
    visible without reading the warnings. `within_tolerance` is `None` for
    `interarrival_cv` when `runs_z` was also targeted -- the settled
    combined-target behaviour records the achieved CV without applying its
    tolerance -- and renders as `?`."""
    if metric not in manifest.achieved:
        return "-"
    achieved = f"{manifest.achieved[metric]:.4g}"
    if metric not in manifest.targets:
        return achieved
    within = manifest.within_tolerance.get(metric)
    mark = "?" if within is None else ("ok" if within else "MISS")
    return f"{achieved} ({manifest.targets[metric]:g}) {mark}"


# -- clean -------------------------------------------------------------------


def _cmd_clean(args: argparse.Namespace, console: Console) -> int:
    store_path = _existing_store(args.store)
    if store_path is None:
        return _fail(
            f"no results database at {args.store}; "
            f"nothing to clean for suite {args.suite!r}"
        )

    with Store(store_path) as store:
        known = store.tables_for_suite(args.suite)

    if not known:
        return _fail(f"no tables recorded for suite {args.suite!r} in {args.store}")

    client = CHClient(Connection.from_env())
    present = _present_tables(client, args.suite)

    # Every recorded name is dropped, because the store is the authority on
    # what this suite owns. Only the ones the server actually has are
    # counted and announced: a grid cell `expand_grid` skipped is recorded
    # in `tables` for its skip reason, and so is in `suite_tables`, but no
    # table was ever created for it, and reporting it as dropped would
    # overstate what happened.
    dropped = 0
    for name in known:
        client.command(f"DROP TABLE IF EXISTS {name}")
        if name in present:
            dropped += 1
            console.print(f"dropped {name}", markup=False, highlight=False)

    for name in sorted(present - set(known)):
        console.print(
            f"warning: {name} matches suite {args.suite} but is not in the store; "
            f"left in place",
            style="yellow",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )

    never_created = len(known) - dropped
    tail = f" ({never_created} recorded but never created)" if never_created else ""
    console.print(
        f"dropped {dropped} table(s) for suite {args.suite}{tail}",
        style="bold",
        markup=False,
        highlight=False,
    )
    return 0


def _present_tables(client: CHClient, suite_name: str) -> set[str]:
    """Every `<database>.<suite_name>_*` table the server currently has.

    Used for two things `clean` cannot get from the store alone: which
    recorded names really exist (a skipped grid cell is recorded but was
    never created), and which existing tables the store has never heard of.

    An orphan -- a table matching the suite's prefix that the store does not
    know about -- is named by `clean` but never dropped. The store records a
    table the moment its load finishes, so an orphan means a
    process died between `CREATE TABLE` and that write. Dropping a table this
    store never claimed would be guessing at ownership from a name prefix,
    and the project's rule is to record what it measured, never to coerce --
    so the orphan is named and left for the user to drop deliberately.
    """
    database = client.connection.database
    prefix = f"{suite_name}_"
    rows = client.query_rows(
        "SELECT name FROM system.tables WHERE database = {db:String} "
        "AND startsWith(name, {prefix:String})",
        parameters={"db": database, "prefix": prefix},
    )
    # `startsWith` alone is not enough: suite `t_a` would match suite
    # `t_a_b`'s tables and report every one of them as an orphan, since they
    # are in the store under a different suite name. Only a remainder that is
    # exactly the eight hex characters `ddl.table_name` appends belongs to
    # this suite.
    return {
        f"{database}.{name}"
        for (name,) in rows
        if _TABLE_SUFFIX_RE.match(name[len(prefix) :])
    }


if __name__ == "__main__":
    raise SystemExit(main())
