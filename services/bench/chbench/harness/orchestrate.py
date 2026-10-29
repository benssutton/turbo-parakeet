"""Wire every harness module into one command: `run_suite`.

This is the benchmark's data flow, and the only module that knows the
order of it: connection -> dataset -> grid -> every table -> rounds of
samples -> report. Nothing above it exists; everything below it is already tested in
isolation, so this module contains no numerics and no SQL of its own, only
the sequence and the bookkeeping. Every read it needs from the store goes
through a `Store` method (`explain_for`, `samples_held`, `next_round`), never a SELECT
written here.

`RunContext` carries what every step of the run needs -- the console, the
store, the client, the suite, the server version, the run id, the counters
and the four flags -- so the helpers below take a context and the one
or two things that actually vary between their calls, instead of a dozen
positional parameters each.

Six decisions here are load-bearing and are not free to change:

1. **`expand_grid` is given explicit achieved values, not the manifest**
   `null_count` comes from the DatasetSpec, where it is exact -- the
   generator injects precisely that many -- and `value_min`/`value_max` from
   `pyarrow.compute.min_max` of the *final* array, which is already rounded
   and cast, so its bounds are the values ClickHouse will really store. The
   unsigned-minimum and decimal-overflow rules have to run against those,
   not against the requested `value_range`, or a grid cell that overflows
   would be created and fail at insert instead of being skipped with a
   reason.

2. **Every table variant gets a freshly seeded predicate generator.** The
   range predicate's interval is drawn from the `rng` handed to
   `expand_queries`, so a single generator
   reused across variants would give the Delta variant a different interval
   from the Gorilla variant and the two timings would not be comparable.
   Seeding a new `np.random.default_rng(suite.dataset.seed)` inside the
   per-variant loop makes every variant of the suite run the identical
   query. This is the difference between a benchmark and noise.

3. **Load first, then interleaved rounds.** Every table of the grid is loaded
   (or reused) before any timed query; then one untimed warm-up round and
   `samples` timed rounds, each running every query once in a freshly
   shuffled order. Identical tables sampled back to back disagreed by up to
   2x (ToDo 3); interleaving spreads drift and load over every variant, and
   `round_no` pairs them for the report's ratios. A quietness snapshot
   (`quiet.snapshot`) is stored on the run row before the warm-up and after
   the last round.

4. **A failure is recorded, not raised.** `load_table` never raises, so its
   `status` is checked; a table that failed to load is stored with its error
   and its queries are skipped. A query that fails in a round (the warm-up
   included) or in its EXPLAIN is stored as `error` with the message, takes
   no further samples, and the rounds carry on. Only a `KeyboardInterrupt`
   stops the run, and even then the samples `run_round` delivered are
   committed, every query short of its samples is marked `interrupted` --
   whichever phase was running: a load, the warm-up, a round or the final
   EXPLAINs -- and the run is finished before it is re-raised.

5. **UTF-8 is forced on the console before anything is printed.** The
   `EXPLAIN indexes = 1` output contains box-drawing characters, which raise
   `UnicodeEncodeError` on the default Windows console encoding. Measured in
   the spikes (FINDINGS §5), not a precaution. `force_utf8()` stays
   ahead of the print path whether or not the plan is printed, because a
   skip reason or an error message can carry non-ASCII too.

6. **The EXPLAIN plan is always stored and never printed by default**
   It is evidence, so every run records it on the query row;
   it is also ten lines per query, which on a full `int_encodings` grid
   (36 valid cells x 12 query variants) is thousands of lines of console.
   `explain=True` (`chbench run --explain`) prints it.

`finish_run` is in a `finally`, not on the success path: a
`sqlite3.OperationalError` from a second run sharing `results.db`, or an
`expand_queries` `ValueError` on an all-null dataset, would otherwise leave
`finished_at` NULL forever. It is called from exactly one place, so it can
never run twice.

Warnings reach the user from every source that produces them: the
dataset `Manifest`'s (a metric missed its target), each `TableVariant`'s (the
cell is degraded, e.g. a projection redundant with the primary key), each
round's (a `system.query_log` row never appeared) and the quietness
snapshots' (a busy host, active merges). None of them is allowed to vanish
into the store unread.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rich.console import Console

from synthdata.cache import load_or_generate
from chbench.harness import quiet
from chbench.harness.client import CHClient, Connection
from chbench.harness.ddl import table_name
from chbench.harness.loader import load_table
from chbench.harness.queries import QueryVariant, expand_queries, query_id, query_values
from chbench.harness.report import render_table, summarise
from chbench.harness.runner import Job, explain_query, run_round
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
    """A run of this suite has a `runs` row with no `finished_at`. The
    message names the run and says how to get past it."""


class UnsolvedDatasetError(RuntimeError):
    """The dataset's solve could not bracket a target (`invalid_bracket`); the message names
    each such metric with its target and says how to run it anyway."""


@dataclass
class RunSummary:
    """What one `run_suite` call did.

    `run_id` is `None` for a dry run, which never opens the store. The three
    table counters partition the grid: `tables_skipped` counts cells
    `expand_grid` marked invalid, `tables_error` cells whose load failed,
    `tables_ok` cells that loaded or, on a rerun, were reused without a
    reload because every query already held its samples (in a
    dry run, cells that *would* be loaded). `queries_run` counts query
    variants that took part in the rounds,
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
    """Everything one run's steps share, gathered once.

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


@dataclass
class _Pending:
    """One query of the grid that takes samples this run, and what the rounds made of it."""

    job: Job
    tid: str
    want: int
    prior_explain: str
    status: str = "ok"  # "ok" | "error" | "interrupted"
    error: str | None = None
    taken: int = 0
    done: bool = False  # its outcome row is written (`_finish_queries`)


def force_utf8() -> None:
    """Reconfigure stdout and stderr to UTF-8 with replacement.

    `EXPLAIN indexes = 1` returns box-drawing characters, and printing them
    to a cp1252 console raises `UnicodeEncodeError` mid-run
    (FINDINGS §5). `errors="replace"` is the second line of defence for a
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
    return (
        f"{variant.ch_type} codec={variant.codec} pk={variant.primary_key} "
        f"proj={variant.projection} idx={variant.skip_index} "
        f"comp={variant.compression}"
    )


def _query_label(qv: QueryVariant) -> str:
    label = qv.kind
    if qv.target_fraction is not None:
        label += f" target={qv.target_fraction:g} actual={qv.actual_fraction:.4f}"
    return label


def _warn(console: Console, source: str, warnings) -> None:
    """Print every warning from one source, naming where it came from."""
    for warning in warnings:
        _say(console, f"warning {source}: {warning}", style="yellow")


def _check_solves(manifest, allow_unsolved: bool, console: Console) -> None:
    bad = [
        (m, rec)
        for m, rec in manifest.solves.items()
        if rec.status == "invalid_bracket"
    ]
    if not bad:
        return
    text = "; ".join(
        f"{m} (target {manifest.targets.get(m)!r}): {rec.message}" for m, rec in bad
    )
    if not allow_unsolved:
        raise UnsolvedDatasetError(
            f"the dataset's solve could not bracket {text}. Rerun with --allow-unsolved to "
            f"run it anyway."
        )
    _say(console, f"warning dataset: running unsolved targets: {text}", style="yellow")


def _print_grid(console: Console, variants: list[TableVariant]) -> None:
    """One line per grid cell, with its skip reason and its warnings.

    Deliberately not a `rich.Table`: a skip reason is ClickHouse's own
    rejection message, often well over a hundred characters, and a table
    cell would truncate exactly the part the user needs to read.
    """
    ok = sum(1 for v in variants if v.status == "ok")
    _say(
        console,
        f"\ngrid: {len(variants)} cells, {ok} valid, " f"{len(variants) - ok} skipped",
        style="bold",
    )
    for variant in variants:
        if variant.status == "ok":
            _say(console, f"  ok      {_variant_label(variant)}", style="green")
        else:
            _say(console, f"  skipped {_variant_label(variant)}", style="yellow")
            _say(console, f"          -> {variant.skip_reason}")
        _warn(console, _variant_label(variant), variant.warnings)


def _achieved_bounds(array: pa.Array) -> tuple:
    """`(min, max)` of `array` as Python values, nulls ignored.

    A `decimal32`/`decimal64` array (a decimal dataset) is
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
    (re-running a suite skips queries already holding >= N samples).

    The target is a `max`, not `held + samples_add`, so that a query which
    has never run still gets the suite's full `samples`. Under the plain sum,
    `--samples-add 2` against a grid where the incumbents hold 5 would give a
    newly added codec only 2, and two rows meant to be compared would carry
    different `n` -- the report would then be comparing a 5-sample median
    with a 2-sample one.

    Never returns a negative count: a negative want would take no samples
    and report a mean of nan. The CLI rejects a negative
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
    _say(
        console,
        f"\nreport: {suite.name} " f"({len(summary)} queries, {len(rows)} samples)",
        style="bold",
    )
    console.print(render_table(summary, suite.results.percentiles))


def run_suite(
    suite: Suite,
    *,
    store_path: str | Path,
    cache_dir: str | Path,
    force: bool = False,
    samples_add: int = 0,
    dry_run: bool = False,
    explain: bool = False,
    ignore_unfinished: bool = False,
    allow_unsolved: bool = False,
    console: Console | None = None,
) -> RunSummary:
    """Run one suite end to end and return what it did.

    The order is fixed by three constraints.

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
    suite with `finished_at` NULL stops the run with `UnfinishedRunError`:
    `finish_run` is in a `finally`, so the row means either
    a process that died mid-run or one still running against this store,
    and two live runs would interleave samples and contend for the file.
    `ignore_unfinished` (`--ignore-unfinished`) is the user's statement
    that the first process is gone.

    A dataset whose solve ended `invalid_bracket` does not hit its target, so the run would
    measure the wrong data: `UnsolvedDatasetError` stops it (before the grid, so before the
    store) unless `allow_unsolved`. Clamped and `not_converged` solves stay warnings.
    """
    force_utf8()
    console = console or default_console()
    summary = RunSummary()

    client = CHClient(Connection.from_env())
    version = client.version()
    _say(
        console,
        f"clickhouse {version} at "
        f"{client.connection.host}:{client.connection.port}",
        style="bold",
    )

    dataset = load_or_generate(suite.dataset, Path(cache_dir))
    manifest = dataset.manifest
    # `cache.save` writes `<cache_dir>/<dataset_id>.arrow`; the store records
    # that path so a result can be traced back to the exact bytes loaded.
    # A path dataset is read in place: the store records its file.
    arrow_path = (
        Path(suite.dataset.path)
        if suite.dataset.path is not None
        else Path(cache_dir) / f"{manifest.dataset_id}.arrow"
    )
    _say(
        console,
        f"dataset {manifest.dataset_id}: {suite.dataset.rows} rows of "
        f"{suite.dataset.arrow_type}, {arrow_path}",
        style="bold",
    )
    _warn(console, "dataset", manifest.warnings)
    _check_solves(manifest, allow_unsolved, console)

    # The achieved bounds of the final, rounded, cast array -- the values
    # the server will actually hold.
    value_min, value_max = _achieved_bounds(dataset.array)
    variants = expand_grid(
        suite,
        null_count=suite.dataset.null_count,
        value_min=value_min,
        value_max=value_max,
        clickhouse_version=version,
        array=dataset.array,
    )
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
                f"running, rerun with --ignore-unfinished."
            )
        store.upsert_dataset(manifest, str(arrow_path))
        run_id = store.begin_run(suite)
        summary.run_id = run_id
        context = RunContext(
            console=console,
            store=store,
            client=client,
            suite=suite,
            version=version,
            run_id=run_id,
            summary=summary,
            force=force,
            samples_add=samples_add,
            explain=explain,
            ignore_unfinished=ignore_unfinished,
        )
        # Every query row written so far, so an interrupt in any phase can close it.
        pending: list[_Pending] = []
        try:
            # Every table first, then the rounds: module docstring, point 3.
            for variant in variants:
                pending += _prepare_variant(context, dataset, values, variant)
            _run_rounds(context, pending)
        except KeyboardInterrupt:
            # `run_round` already delivered every completed sample through
            # `on_sample`, and `insert_sample` commits each one as it lands,
            # so there is nothing left to flush. Without this write the
            # pre-run upsert leaves each query's status NULL, and after the
            # fact an interrupted query looks like one the run never reached;
            # a row `_finish_queries` already wrote keeps its status.
            # `finish_run` is the `finally`'s job, below.
            unfinished = [p for p in pending if not p.done]
            for p in unfinished:
                if p.status == "ok" and p.taken < p.want:
                    p.status = "interrupted"
            _finish_queries(context, unfinished, explain=False)
            _say(
                console,
                f"\ninterrupted: {summary.samples_written} sample(s) "
                f"written and committed",
                style="yellow",
            )
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


def _prepare_variant(
    ctx: RunContext, dataset, values: np.ndarray, variant: TableVariant
) -> list[_Pending]:
    """Load one table variant (or reuse it) and return its queries that want samples.

    `suite_name` goes to `upsert_table` so `clean` can drop this
    table even if the run is interrupted before its first sample lands: the
    table exists in ClickHouse from the moment the load succeeds, and the
    store has to know about it from that same moment.
    """
    suite, store, console = ctx.suite, ctx.store, ctx.console
    dataset_id = dataset.manifest.dataset_id
    tid = table_id(variant, dataset_id, ctx.version)
    name = table_name(suite.name, tid)

    if variant.status == "skipped":
        store.upsert_table(
            tid, dataset_id, variant, None, ctx.version, suite_name=suite.name
        )
        ctx.summary.tables_skipped += 1
        return []

    # A generator seeded identically for every variant, constructed here and
    # never reused across variants: module docstring, point 2. Two variants
    # that differ only in codec must run the byte-identical query, or the
    # comparison between them measures the predicate, not the codec.
    rng = np.random.default_rng(suite.dataset.seed)
    query_variants = expand_queries(suite.queries, values, variant, rng)

    # The query variants depend on the variant and the seeded
    # rng, not on the table, so they are expanded before the load, and a
    # table none of whose queries wants a sample is not dropped and reloaded
    # for nothing. The first runs measured the reload at a few seconds per
    # table, a minute over the int suite, for a rerun that then skipped
    # every query. The `tables` row is left as the run that loaded it wrote
    # it, so its `insert_ms` and `created_at` stay that run's evidence.
    if not ctx.force and all(
        _samples_to_run(
            store.samples_held(query_id(tid, qv)),
            suite.queries.samples,
            ctx.force,
            ctx.samples_add,
        )
        == 0
        for qv in query_variants
    ):
        ctx.summary.tables_ok += 1
        _say(console, f"\n{name} {_variant_label(variant)}", style="bold")
        _say(
            console,
            "  reused: every query already holds its samples; not reloaded",
            style="dim",
        )
        return _pending_queries(ctx, name, variant, query_variants, tid)

    load = load_table(ctx.client, name, variant, dataset)
    store.upsert_table(
        tid, dataset_id, variant, load, ctx.version, suite_name=suite.name
    )

    if load.status == "error":
        ctx.summary.tables_error += 1
        _say(console, f"\nerror {name} {_variant_label(variant)}", style="red")
        _say(console, f"          -> {load.error}")
        return []

    ctx.summary.tables_ok += 1
    stats = load.stats
    ratio = (
        stats.uncompressed_bytes / stats.compressed_bytes
        if stats.compressed_bytes
        else float("nan")
    )
    _say(console, f"\n{name} {_variant_label(variant)}", style="bold")
    _say(
        console,
        f"  loaded {stats.rows} rows in {load.insert_ms:.0f} ms, "
        f"{stats.compressed_bytes} compressed bytes, ratio {ratio:.2f}",
    )
    return _pending_queries(ctx, name, variant, query_variants, tid)


def _pending_queries(
    ctx: RunContext, name: str, variant: TableVariant, query_variants, tid: str
) -> list[_Pending]:
    """The queries of one loaded table that want samples, each with its pre-run query row.

    The row is written before any sample, carrying the EXPLAIN it already had, and rewritten
    with the fresh one after the rounds. The store joins samples to queries, so a sample
    committed under a `query_id` with no `queries` row would be invisible to the report --
    what a Ctrl-C in the first round would otherwise produce."""
    store, console, summary = ctx.store, ctx.console, ctx.summary
    out = []
    for qv in query_variants:
        qid = query_id(tid, qv)
        held = store.samples_held(qid)
        want = _samples_to_run(
            held, ctx.suite.queries.samples, ctx.force, ctx.samples_add
        )
        if want == 0:
            summary.queries_skipped_existing += 1
            _say(
                console,
                f"  skip {_query_label(qv)} ({held} sample(s) already held)",
                style="dim",
            )
            continue
        prior = store.explain_for(qid)
        store.upsert_query(qid, tid, qv, prior)
        summary.queries_run += 1
        _say(console, f"  want {_query_label(qv)} ({want} sample(s), {held} held)")
        out.append(_Pending(Job(qid, name, variant, qv), tid, want, prior))
    return out


def _mark_errors(by_key: dict[str, _Pending], errors: dict[str, str]) -> None:
    """A query that failed in a round keeps its first error and takes no more samples."""
    for key, error in errors.items():
        p = by_key[key]
        if p.status == "ok":
            p.status, p.error = "error", error


def _quiet(ctx: RunContext, when: str) -> None:
    snap = quiet.snapshot(ctx.client)
    ctx.store.record_quiet(ctx.run_id, when, snap)
    _warn(ctx.console, f"quietness {when}", quiet.warnings(snap))


def _run_rounds(ctx: RunContext, pending: list[_Pending]) -> None:
    """The warm-up round, then `max(want)` timed rounds over every pending query.

    Round r runs each query still wanting a sample once, in an order shuffled by
    `default_rng([seed, r])`: reproducible, different per round, so drift and load bursts
    spread over every variant instead of landing on one. Rounds continue the store's
    numbering (`next_round`), so samples added by a later `--samples-add` stay paired. A
    query added to a suite that already holds its samples runs alone, in rounds the others
    never shared: a warning says so.
    A sample's `sample_no` is its index among this run's samples of the query (0, 1, ...),
    as it always was; `round_no` is what pairs samples across runs."""
    if not pending:
        return
    store, console = ctx.store, ctx.console
    by_key = {p.job.key: p for p in pending}
    _quiet(ctx, "before")
    warm_up = run_round(ctx.client, [p.job for p in pending], timed=False)
    _mark_errors(by_key, warm_up.errors)
    first = store.next_round(ctx.suite.name)
    total = max(p.want for p in pending)
    seed = ctx.suite.dataset.seed or 0
    _say(
        console,
        f"\n{len(pending)} queries, {total} round(s) from round {first}",
        style="bold",
    )
    held = ctx.summary.queries_skipped_existing
    if held and not ctx.samples_add:
        _say(
            console,
            f"warning: {len(pending)} of {len(pending) + held} queries run without the "
            f"others (the rest already hold their samples): their rounds pair only among "
            f"themselves; --samples-add K re-runs every query so all stay paired",
            style="yellow",
        )

    def on_sample(key: str, sample) -> None:
        # No KeyboardInterrupt handling here: `run_round` owns that, and a
        # second handler would swallow the interrupt it re-raises.
        store.insert_sample(key, ctx.run_id, sample)
        by_key[key].taken += 1
        ctx.summary.samples_written += 1

    # A Ctrl-C anywhere here is `run_suite`'s to handle: it marks the queries.
    for i in range(total):
        due = [p for p in pending if p.status == "ok" and p.want > i]
        if not due:
            break
        r = first + i
        order = np.random.default_rng([seed, r]).permutation(len(due))
        result = run_round(
            ctx.client,
            [due[k].job for k in order],
            r,
            sample_no=i,
            on_sample=on_sample,
        )
        _mark_errors(by_key, result.errors)
        _warn(console, f"round {r}", result.warnings)
    _finish_queries(ctx, pending, explain=True)
    _quiet(ctx, "after")


def _finish_queries(ctx: RunContext, pending: list[_Pending], *, explain: bool) -> None:
    """Each query's EXPLAIN (stored always, printed with --explain) and its outcome row.

    The outcome goes on the query row: without it, a query whose run failed
    is indistinguishable, after an unattended run, from one the run never
    reached."""
    for p in pending:
        plan = p.prior_explain
        if explain and p.status == "ok":
            text, error = explain_query(ctx.client, p.job)
            if error is not None:
                p.status, p.error = "error", error
            plan = text or plan
        ctx.store.upsert_query(
            p.job.key, p.tid, p.job.qv, plan, status=p.status, error=p.error
        )
        p.done = True
        label = f"{p.job.name} {_query_label(p.job.qv)}"
        if p.status == "error":
            _say(ctx.console, f"  error {label}: {p.error}", style="red")
        elif explain and ctx.explain and plan:
            # UTF-8 was forced at the top of `run_suite`, which is what makes
            # this box-drawing text printable at all.
            _say(ctx.console, f"  plan {label}", style="bold")
            _say(ctx.console, plan, style="dim")
