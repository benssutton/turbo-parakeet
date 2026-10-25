"""Wire every harness module into one command: `run_suite`.

This is the data flow of spec section 3, and the only module that knows the
order of it: dataset -> connection -> grid -> tables -> queries -> samples ->
report. Nothing above it exists; everything below it is already tested in
isolation, so this module contains no numerics and no SQL of its own, only
the sequence and the bookkeeping. Every read it needs from the store goes
through a `Store` method (`explain_for`, `samples_held`), never a SELECT
written here.

`RunContext` carries what every step of the run needs -- the console, the
store, the client, the suite, the server version, the run id, the counters
and the four flags -- so the two helpers below take a context and the one
or two things that actually vary between their calls, instead of a dozen
positional parameters each.

Five decisions here are load-bearing and are not free to change:

1. **`expand_grid` is given explicit achieved values, not the manifest**
   (ruling R1). `null_count` comes from the spec, where it is exact -- the
   generator injects precisely that many -- and `value_min`/`value_max` from
   `pyarrow.compute.min_max` of the *final* array, which is already rounded
   and cast, so its bounds are the values ClickHouse will really store. The
   unsigned-minimum and decimal-overflow rules have to run against those,
   not against the requested `value_range`, or a grid cell that overflows
   would be created and fail at insert instead of being skipped with a
   reason.

2. **Every table variant gets a freshly seeded predicate generator.** The
   range predicate's interval is drawn from the `rng` handed to
   `expand_queries` (see `run_query`'s docstring), so a single generator
   reused across variants would give the Delta variant a different interval
   from the Gorilla variant and the two timings would not be comparable.
   Seeding a new `np.random.default_rng(suite.dataset.seed)` inside the
   per-variant loop makes every variant of the suite run the identical
   query. This is the difference between a benchmark and noise.

3. **A failure is recorded, not raised.** `load_table` never raises, so its
   `status` is checked; a table that failed to load is stored with its error
   and its queries are skipped. A `RunResult` with `status == "error"` is
   stored the same way and the loop carries on to the next query. Only a
   `KeyboardInterrupt` stops the run, and even then the samples the runner
   already delivered through `on_sample` are committed, the interrupted
   query's row is marked `interrupted` (ruling R34) and the run is
   finished before it is re-raised (spec section 6.4).

4. **UTF-8 is forced on the console before anything is printed.** The
   `EXPLAIN indexes = 1` output contains box-drawing characters, which raise
   `UnicodeEncodeError` on the default Windows console encoding. Measured in
   the spikes (FINDINGS section 5), not a precaution. `force_utf8()` stays
   ahead of the print path whether or not the plan is printed, because a
   skip reason or an error message can carry non-ASCII too.

5. **The EXPLAIN plan is always stored and never printed by default**
   (ruling R21). It is evidence, so every run records it on the query row;
   it is also ten lines per query, which on a full `int_encodings` grid
   (36 valid cells x 12 query variants) is thousands of lines of console.
   `explain=True` (`chbench run --explain`) prints it.

`finish_run` is in a `finally`, not on the success path: a
`sqlite3.OperationalError` from a second run sharing `results.db`, or an
`expand_queries` `ValueError` on an all-null dataset, would otherwise leave
`finished_at` NULL forever. It is called from exactly one place, so it can
never run twice.

Warnings reach the user from all three sources that produce them: the
dataset `Manifest`'s (a metric missed its target), each `TableVariant`'s (the
cell is degraded, e.g. a projection redundant with the primary key) and each
`RunResult`'s (a cache drop failed, or a `system.query_log` row never
appeared). None of them is allowed to vanish into the store unread.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rich.console import Console

from chbench.datagen.cache import load_or_generate
from chbench.harness.client import CHClient, Connection
from chbench.harness.ddl import table_name
from chbench.harness.loader import load_table
from chbench.harness.queries import QueryVariant, expand_queries, query_id, query_values
from chbench.harness.report import render_table, summarise
from chbench.harness.runner import run_query
from chbench.harness.store import Store
from chbench.harness.suite import Suite
from chbench.harness.validate import TableVariant, expand_grid, table_id

# Rich falls back to 80 columns when stdout is not a terminal (a pipe, a log
# file, a test subprocess) and truncates every cell of the report's two dozen
# columns to a few characters. 320 is what the widest report (seven table
# axes, four query axes, three percentiles of both clocks) needs to render
# untruncated. A terminal keeps its real width; `--csv` is the way out when
# that terminal is narrower than the table.
_PIPED_WIDTH = 320


class UnfinishedRunError(RuntimeError):
    """A run of this suite has a `runs` row with no `finished_at` (ruling
    R33). The message names the run and says how to get past it."""


@dataclass
class RunSummary:
    """What one `run_suite` call did.

    `run_id` is `None` for a dry run, which never opens the store. The three
    table counters partition the grid: `tables_skipped` counts cells
    `expand_grid` marked invalid, `tables_error` cells whose load failed,
    `tables_ok` cells that loaded or, on a rerun, were reused without a
    reload because every query already held its samples (ruling R30; in a
    dry run, cells that *would* be loaded). `queries_run` counts query
    variants `run_query` was called for,
    `queries_skipped_existing` those already holding enough samples, and
    `samples_written` rows `insert_sample` committed.
    """
    run_id: str | None = None
    tables_ok: int = 0
    tables_skipped: int = 0
    tables_error: int = 0
    queries_run: int = 0
    queries_skipped_existing: int = 0
    samples_written: int = 0


@dataclass(frozen=True)
class RunContext:
    """Everything one run's steps share, gathered once (ruling R24).

    Frozen because none of it changes during a run -- `summary` is a mutable
    dataclass held by reference, and is the only thing the steps write to.
    The alternative, threading these twelve values positionally through each
    helper, made adding a single flag a change to every signature and every
    call site between here and the flag's one use.
    """
    console: Console
    store: Store
    client: CHClient
    suite: Suite
    version: str
    run_id: str
    summary: RunSummary
    force: bool
    samples_add: int
    explain: bool
    ignore_unfinished: bool


def force_utf8() -> None:
    """Reconfigure stdout and stderr to UTF-8 with replacement.

    `EXPLAIN indexes = 1` returns box-drawing characters, and printing them
    to a cp1252 console raises `UnicodeEncodeError` mid-run (FINDINGS
    section 5). `errors="replace"` is the second line of defence for a
    stream that cannot re-encode a character at all. A stream with no
    `reconfigure` (a `StringIO` in a test) or one whose buffer is detached
    is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def default_console() -> Console:
    """A `rich.Console` wide enough for the report when stdout is piped."""
    console = Console()
    return console if console.is_terminal else Console(width=_PIPED_WIDTH)


def _say(console: Console, text: str, style: str | None = None) -> None:
    """Print one line of text that may have come from the server or the data.

    `markup=False` unconditionally: ClickHouse's rejection messages and its
    `EXPLAIN` plans contain square brackets (`Keys: [v]`, `Condition: [v, 1]`),
    which rich would parse as a style tag and then raise `MissingStyle` on
    mid-run. Styling is applied to the whole line through `style` instead of
    inline. `soft_wrap=True` keeps a long skip reason on one line, so it can
    be grepped out of a log intact, and `highlight=False` stops rich
    recolouring the numbers inside a message.
    """
    console.print(text, style=style, markup=False, highlight=False, soft_wrap=True)


def _variant_label(variant: TableVariant) -> str:
    return (f"{variant.ch_type} codec={variant.codec} pk={variant.primary_key} "
            f"proj={variant.projection} idx={variant.skip_index} "
            f"comp={variant.compression}")


def _query_label(qv: QueryVariant) -> str:
    label = f"{qv.kind} cache={qv.cache_mode} qcc={qv.qcc}"
    if qv.target_fraction is not None:
        label += f" target={qv.target_fraction:g} actual={qv.actual_fraction:.4f}"
    return label


def _warn(console: Console, source: str, warnings) -> None:
    """Print every warning from one source, naming where it came from."""
    for warning in warnings:
        _say(console, f"warning {source}: {warning}", style="yellow")


def _print_grid(console: Console, variants: list[TableVariant]) -> None:
    """One line per grid cell, with its skip reason and its warnings.

    Deliberately not a `rich.Table`: a skip reason is ClickHouse's own
    rejection message, often well over a hundred characters, and a table
    cell would truncate exactly the part the user needs to read.
    """
    ok = sum(1 for v in variants if v.status == "ok")
    _say(console, f"\ngrid: {len(variants)} cells, {ok} valid, "
                  f"{len(variants) - ok} skipped", style="bold")
    for variant in variants:
        if variant.status == "ok":
            _say(console, f"  ok      {_variant_label(variant)}", style="green")
        else:
            _say(console, f"  skipped {_variant_label(variant)}", style="yellow")
            _say(console, f"          -> {variant.skip_reason}")
        _warn(console, _variant_label(variant), variant.warnings)


def _achieved_bounds(array: pa.Array) -> tuple:
    """`(min, max)` of `array` as Python values, nulls ignored.

    A `decimal32`/`decimal64` array (a decimal dataset, spec section 4.1) is
    cast to `decimal128` at the same precision and scale first: on pyarrow
    24 `pc.min_max` has kernels for `decimal128` and `decimal256` only, and
    raised `ArrowNotImplementedError` on the narrower widths before the
    grid could be printed -- even under `--dry-run`. The cast is exact
    (wider storage for the same digits), so the bounds are still the values
    ClickHouse will hold.
    """
    if pa.types.is_decimal32(array.type) or pa.types.is_decimal64(array.type):
        array = array.cast(pa.decimal128(array.type.precision, array.type.scale))
    bounds = pc.min_max(array)
    return bounds["min"].as_py(), bounds["max"].as_py()


def _samples_to_run(held: int, samples: int, force: bool, samples_add: int) -> int:
    """How many samples this run should take, or 0 to skip the query.

    `force` ignores what is banked and takes a full `samples` again.
    Otherwise the target is `max(samples, held + samples_add)` and the query
    takes the shortfall; a query already at or above the target is skipped
    (spec section 6.1: "re-running a suite skips queries already holding
    >= N samples").

    The target is a `max`, not `held + samples_add`, so that a query which
    has never run still gets the suite's full `samples`. Under the plain sum,
    `--samples-add 2` against a grid where the incumbents hold 5 would give a
    newly added codec only 2, and two rows meant to be compared would carry
    different `n` -- the report would then be comparing a 5-sample median
    with a 2-sample one.

    Never returns a negative count: `run_query(samples=-2)` would run no
    samples and report a mean of nan. The CLI rejects a negative
    `--samples-add` at parse time; this is the guard for a direct caller.
    """
    if force:
        return max(0, samples_add or samples)
    target = max(samples, held + samples_add)
    return max(0, target - held)


def _print_report(console: Console, store: Store, suite: Suite) -> None:
    """The suite's report, over every sample banked for it rather than only
    this run's: samples accumulate across runs by design (that is what
    `samples_held` counts), so the percentiles a user wants to see after a
    run are the ones over the whole bank. A re-run that added nothing still
    prints what is held.
    """
    rows = store.query_rows(suite.name)
    if not rows:
        _say(console, "\nno samples recorded for this suite", style="yellow")
        return
    summary = summarise(rows, suite.results.percentiles)
    _say(console, f"\nreport: {suite.name} "
                  f"({len(summary)} queries, {len(rows)} samples)", style="bold")
    console.print(render_table(summary, suite.results.percentiles))


def run_suite(suite: Suite, *, store_path: str | Path, cache_dir: str | Path,
              force: bool = False, samples_add: int = 0, dry_run: bool = False,
              explain: bool = False, ignore_unfinished: bool = False,
              console: Console | None = None) -> RunSummary:
    """Run one suite end to end and return what it did.

    The order is fixed by spec section 3 and by three constraints on it.

    The **connection comes first**, before the dataset: it is the cheapest
    thing that can fail and the likeliest to be misconfigured, and generating
    first would spend seconds (use case 1) or minutes (an `interarrival_cv`
    suite) before reporting a wrong `CH_PORT`.

    The dataset is loaded or generated before the grid, because the grid's
    validity rules need the achieved value range (module docstring, point 1);
    a dry run therefore still generates it.

    The grid is expanded and printed *before* the store is opened, so
    `--dry-run` leaves no `results.db` behind.

    Once the store is open and before `begin_run`, a `runs` row for this
    suite with `finished_at` NULL stops the run with `UnfinishedRunError`
    (ruling R33): `finish_run` is in a `finally`, so the row means either
    a process that died mid-run or one still running against this store,
    and two live runs would interleave samples and contend for the file.
    `ignore_unfinished` (`--ignore-unfinished`) is the user's statement
    that the first process is gone.
    """
    force_utf8()
    console = console or default_console()
    summary = RunSummary()

    client = CHClient(Connection.from_env())
    version = client.version()
    _say(console, f"clickhouse {version} at "
                  f"{client.connection.host}:{client.connection.port}", style="bold")

    dataset = load_or_generate(suite.dataset, Path(cache_dir))
    manifest = dataset.manifest
    # `cache.save` writes `<cache_dir>/<dataset_id>.arrow`; the store records
    # that path so a result can be traced back to the exact bytes loaded.
    # A path dataset (ruling R35) is read in place: the store records its file.
    arrow_path = (Path(suite.dataset.path) if suite.dataset.path is not None
                  else Path(cache_dir) / f"{manifest.dataset_id}.arrow")
    _say(console, f"dataset {manifest.dataset_id}: {suite.dataset.rows} rows of "
                  f"{suite.dataset.arrow_type}, {arrow_path}", style="bold")
    _warn(console, "dataset", manifest.warnings)

    # R1: the achieved bounds of the final, rounded, cast array -- the values
    # the server will actually hold.
    value_min, value_max = _achieved_bounds(dataset.array)
    variants = expand_grid(
        suite, null_count=suite.dataset.null_count,
        value_min=value_min, value_max=value_max, clickhouse_version=version,
        array=dataset.array)
    _print_grid(console, variants)

    if dry_run:
        summary.tables_ok = sum(1 for v in variants if v.status == "ok")
        summary.tables_skipped = len(variants) - summary.tables_ok
        _say(console, "\ndry run: nothing loaded, nothing stored", style="bold")
        return summary

    values = query_values(dataset.array)

    with Store(store_path) as store:
        store.init_schema()
        unfinished = store.unfinished_runs(suite.name)
        if unfinished and not ignore_unfinished:
            run_id, started_at = unfinished[-1]
            raise UnfinishedRunError(
                f"suite {suite.name!r} has an unfinished run {run_id} started "
                f"at {started_at} in {store.path}. If that process is no longer "
                f"running, rerun with --ignore-unfinished.")
        store.upsert_dataset(manifest, str(arrow_path))
        run_id = store.begin_run(suite)
        summary.run_id = run_id
        context = RunContext(
            console=console, store=store, client=client, suite=suite,
            version=version, run_id=run_id, summary=summary, force=force,
            samples_add=samples_add, explain=explain,
            ignore_unfinished=ignore_unfinished)
        try:
            for variant in variants:
                _run_variant(context, dataset, values, variant)
        except KeyboardInterrupt:
            # `run_query` already delivered every completed sample through
            # `on_sample`, and `insert_sample` commits each one as it lands,
            # so there is nothing left to flush here: say what survived and
            # let the interrupt through (spec section 6.4). `finish_run` is
            # the `finally`'s job, below.
            _say(console, f"\ninterrupted: {summary.samples_written} sample(s) "
                          f"written and committed", style="yellow")
            raise
        finally:
            # The only call: a run row must never keep `finished_at` NULL
            # because something other than Ctrl-C escaped the loop -- a
            # `database is locked` from a second run sharing this file, or
            # `expand_queries` raising on an all-null dataset. Reached once
            # per run, on every path, so it cannot run twice.
            store.finish_run(run_id)
        # Deliberately outside the `finally`: rendering a report while an
        # exception propagates would mask it if the store is the thing that
        # broke. `chbench report` shows the bank in that case.
        _print_report(console, store, suite)

    return summary


def _run_variant(ctx: RunContext, dataset, values: np.ndarray,
                 variant: TableVariant) -> None:
    """Load one table variant and run every query against it.

    `suite_name` goes to `upsert_table` (ruling R18) so `clean` can drop this
    table even if the run is interrupted before its first sample lands: the
    table exists in ClickHouse from the moment the load succeeds, and the
    store has to know about it from that same moment.
    """
    suite, store, console = ctx.suite, ctx.store, ctx.console
    dataset_id = dataset.manifest.dataset_id
    tid = table_id(variant, dataset_id, ctx.version)
    name = table_name(suite.name, tid)

    if variant.status == "skipped":
        store.upsert_table(tid, dataset_id, variant, None, ctx.version,
                           suite_name=suite.name)
        ctx.summary.tables_skipped += 1
        return

    # A generator seeded identically for every variant, constructed here and
    # never reused across variants: module docstring, point 2. Two variants
    # that differ only in codec must run the byte-identical query, or the
    # comparison between them measures the predicate, not the codec.
    rng = np.random.default_rng(suite.dataset.seed)
    query_variants = expand_queries(suite.queries, values, variant, rng)

    # Ruling R30: the query variants depend on the variant and the seeded
    # rng, not on the table, so they are expanded before the load, and a
    # table none of whose queries wants a sample is not dropped and reloaded
    # for nothing. The first runs measured the reload at a few seconds per
    # table, a minute over the int suite, for a rerun that then skipped
    # every query. The `tables` row is left as the run that loaded it wrote
    # it, so its `insert_ms` and `created_at` stay that run's evidence.
    if not ctx.force and all(
            _samples_to_run(store.samples_held(query_id(tid, qv)),
                            suite.queries.samples, ctx.force, ctx.samples_add) == 0
            for qv in query_variants):
        ctx.summary.tables_ok += 1
        _say(console, f"\n{name} {_variant_label(variant)}", style="bold")
        _say(console, f"  reused: every query already holds its samples; not reloaded",
             style="dim")
        for qv in query_variants:
            _run_query_variant(ctx, name, variant, qv, tid)
        return

    load = load_table(ctx.client, name, variant, dataset)
    store.upsert_table(tid, dataset_id, variant, load, ctx.version,
                       suite_name=suite.name)

    if load.status == "error":
        ctx.summary.tables_error += 1
        _say(console, f"\nerror {name} {_variant_label(variant)}", style="red")
        _say(console, f"          -> {load.error}")
        return

    ctx.summary.tables_ok += 1
    stats = load.stats
    ratio = (stats.uncompressed_bytes / stats.compressed_bytes
             if stats.compressed_bytes else float("nan"))
    _say(console, f"\n{name} {_variant_label(variant)}", style="bold")
    _say(console, f"  loaded {stats.rows} rows in {load.insert_ms:.0f} ms, "
                  f"{stats.compressed_bytes} compressed bytes, ratio {ratio:.2f}")

    for qv in query_variants:
        _run_query_variant(ctx, name, variant, qv, tid)


def _run_query_variant(ctx: RunContext, name: str, variant: TableVariant,
                       qv: QueryVariant, tid: str) -> None:
    store, console, summary = ctx.store, ctx.console, ctx.summary
    qid = query_id(tid, qv)
    held = store.samples_held(qid)
    want = _samples_to_run(held, ctx.suite.queries.samples, ctx.force, ctx.samples_add)

    if want == 0:
        summary.queries_skipped_existing += 1
        _say(console, f"  skip {_query_label(qv)} ({held} sample(s) already held)",
             style="dim")
        return

    # The query row is written before the samples, carrying whatever EXPLAIN
    # it already had, and rewritten with the fresh one afterwards. The store
    # joins samples to queries, so a sample committed under a `query_id` with
    # no `queries` row would be invisible to the report -- which is exactly
    # what a Ctrl-C between the first sample and the EXPLAIN would produce if
    # the row were written only at the end.
    prior_explain = store.explain_for(qid)
    store.upsert_query(qid, tid, qv, prior_explain)

    _say(console, f"  run  {_query_label(qv)} ({want} sample(s), {held} held)")

    def on_sample(sample) -> None:
        # No KeyboardInterrupt handling here: `run_query` owns that, and a
        # second handler would swallow the interrupt it re-raises.
        store.insert_sample(qid, ctx.run_id, sample)
        summary.samples_written += 1

    summary.queries_run += 1
    try:
        result = run_query(ctx.client, name, variant, qv, want, on_sample=on_sample)
    except KeyboardInterrupt as e:
        # `run_query` attaches the partial `RunResult` (status
        # "interrupted", the samples that completed, the EXPLAIN if it got
        # that far) to the exception (ruling R34). Without this write the
        # pre-run upsert above leaves the query's status NULL, and after
        # the fact the interrupted query looks like one the run never
        # reached. The samples are already committed by `on_sample`.
        partial = e.run_result
        store.upsert_query(qid, tid, qv, partial.explain or prior_explain,
                           status=partial.status, error=partial.error)
        raise
    # The run's own outcome goes on the query row (spec section 6.4): without
    # it, a query whose run failed is indistinguishable, after an unattended
    # run, from one the run never reached.
    store.upsert_query(qid, tid, qv, result.explain or prior_explain,
                       status=result.status, error=result.error)
    _warn(console, _query_label(qv), result.warnings)

    if result.status == "error":
        _say(console, f"  error {_query_label(qv)}: {result.error}", style="red")
        return

    client_ms = [s.client_ms for s in result.samples]
    _say(console, f"       {len(client_ms)} sample(s), "
                  f"client mean {np.mean(client_ms):.1f} ms")
    if ctx.explain and result.explain:
        # R21: stored always, printed only on request. UTF-8 was forced at
        # the top of `run_suite`, which is what makes this box-drawing text
        # printable at all.
        _say(console, result.explain, style="dim")
