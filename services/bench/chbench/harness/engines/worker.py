"""The worker process of an in-process engine (one per engine and thread count; the parent side
is `inproc.InprocEngine`).

Protocol: the parent sends tuples `(op, *args)` over a `multiprocessing` Pipe and reads one reply
per request, `("ok", value)` or `("error", "<Type>: <message>")`; `("close",)` (or the pipe closing)
ends the loop. The ops are `BaseWorker`'s public methods: `version()`, `load(path, column)`,
`prepare(spec)`, `run(table_id, query, repeats)` and
`calibrate(table_id, query, target_ns, cap)`. The worker ignores SIGINT: Ctrl-C is the parent's,
which closes the workers and marks the queries.

`BaseWorker` is the engine-independent half: it reads the dataset column once (`load`), prepares
a table (`prepare`: cast, row-by-row round trip back to the dataset's values, for `ipc` the file
written once with the cell's compression, reused when its size matches the store's, and its
values read back and checked) and times a batch of repeats of a query (`run`: the collector run first and paused,
`perf_counter_ns` wall and `process_time_ns` CPU — User + System over every thread of the worker —
around the whole query, for `ipc` from opening the file to the collected answer; `calibrate` picks the batch size in the warm-up). An engine
subclass (`polars_engine.PolarsWorker`, `datafusion_engine.DataFusionWorker`) provides `engine`,
`engine_version()`, `threads_in_use()`, `cast(type_spec, table_id)` (the held object), `values(held)`
(its column as Arrow), `nbytes(held)`, `write_ipc(held, path, codec, batch_rows)` (record batches of `batch_rows` rows; None: one batch), `read_ipc(path)`,
`hold(table_id, held)`, `attach(table_id, path, held)`, `plan(table_id, query)` (work outside the
clock) and `execute(table_id, planned)` -> (count, total).
"""

from __future__ import annotations

import gc
import importlib
import signal
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from chbench.harness.inproc_types import ipc_batch_rows, ipc_codec


def main(conn, factory: str, threads: int) -> None:
    """The child loop (the `multiprocessing` target). `factory` is `module:Class` of a
    `BaseWorker` subclass, built with `threads`."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    module, _, name = factory.partition(":")
    impl = getattr(importlib.import_module(module), name)(threads)
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return
            if msg[0] == "close":
                return
            try:
                reply = ("ok", getattr(impl, msg[0])(*msg[1:]))
            # any failure of an op is reported to the parent; the loop carries on
            except Exception as e:
                reply = ("error", f"{type(e).__name__}: {e}")
            conn.send(reply)
    finally:
        conn.close()


def timed(fn):
    """`fn()` with the collector run first and paused throughout: (result, wall ns, CPU ns). The
    CPU clock is `process_time_ns` (User + System of every thread of this process); on Windows it
    advances in 15.625 ms ticks."""
    gc.collect()
    gc.disable()
    try:
        w0, c0 = time.perf_counter_ns(), time.process_time_ns()
        out = fn()
        c1, w1 = time.process_time_ns(), time.perf_counter_ns()
    finally:
        gc.enable()
    return out, w1 - w0, c1 - c0


def calibrate_repeats(batch_wall_ns, target_ns: int, cap: int) -> tuple[int, bool]:
    """The batch size of a query: the smallest power of two k whose batch of k runs lasts at least
    `target_ns` of wall time (`batch_wall_ns(k)`), with False; `cap` and True when a batch of `cap`
    runs is still shorter. `process_time_ns` advances in 15.625 ms steps on Windows, so one run of
    a fast query cannot be timed on its own."""
    k = 1
    while batch_wall_ns(k) < target_ns:
        if k >= cap:
            return cap, True
        k = min(k * 2, cap)
    return k, False


def mismatch(expected: pa.Array, got: pa.Array | pa.ChunkedArray) -> str | None:
    """None when `got` holds `expected`'s values row by row (after a cast back to its type; NaN
    equal to NaN); else what differs first. `got` may be chunked."""
    if isinstance(got, pa.ChunkedArray):
        got = got.combine_chunks()
    if got.type != expected.type:
        try:
            got = got.cast(expected.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
            return f"the column cannot be read back as {expected.type}: {e}"
    if len(got) != len(expected):
        return f"{len(got)} row(s), the dataset has {len(expected)}"
    if not expected.is_null().equals(got.is_null()):
        return "the null rows differ"
    eq = pc.equal(expected, got)
    if pa.types.is_floating(expected.type):
        eq = pc.or_kleene(eq, pc.and_(pc.is_nan(expected), pc.is_nan(got)))
    i = pc.index(eq.fill_null(True), False).as_py()
    if i < 0:
        return None
    return f"row {i}: {expected[i].as_py()!r} became {got[i].as_py()!r}"


def _skipped(reason: str) -> dict:
    return {
        "status": "skipped",
        "reason": reason,
        "prepare_ms": None,
        "ipc_bytes": None,
        "memory_bytes": None,
        "rows": None,
    }


class BaseWorker:
    """See the module docstring."""

    engine = ""

    def __init__(self, threads: int):
        self.threads = threads
        self.array: pa.Array | None = None
        self.tables: dict = {}

    def version(self) -> dict:
        return {
            "engine": self.engine,
            "version": self.engine_version(),
            "threads": self.threads_in_use(),
        }

    def load(self, path: str, column: str) -> int:
        """The dataset column, read once (as the parent reads it: `realdata.read_column`)."""
        from synthdata.column.realdata import read_column

        self.array = read_column(path, column)
        return len(self.array)

    def prepare(self, spec: dict) -> dict:
        t0 = time.perf_counter()
        tid, type_spec = spec["table_id"], spec["type_spec"]
        try:
            held = self.cast(type_spec, tid)
        # a cast the engine refuses, however it fails, is a skipped cell
        except Exception as e:
            return _skipped(f"cast to {type_spec} failed: {type(e).__name__}: {e}")
        why = mismatch(self.array, self.values(held))
        if why is not None:
            return _skipped(f"{type_spec} does not hold the dataset's values: {why}")
        memory = self.nbytes(held)
        ipc_bytes = None
        if spec["storage"] == "ipc":
            path = Path(spec["ipc_path"])
            if not (path.is_file() and path.stat().st_size == spec["expect_ipc_bytes"]):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                try:
                    self.write_ipc(
                        held,
                        tmp,
                        ipc_codec(spec["compression"]),
                        ipc_batch_rows(spec["ipc_batch_rows"]),
                    )
                    tmp.replace(path)
                except BaseException:
                    # a half-written file must not linger; `chbench clean` also removes one a
                    # killed worker left behind
                    tmp.unlink(missing_ok=True)
                    raise
            ipc_bytes = path.stat().st_size
            why = mismatch(self.array, self.read_ipc(path))
            if why is not None:
                return _skipped(
                    f"the IPC file does not hold the dataset's values: {why}"
                )
            self.attach(tid, path, held)
        else:
            self.hold(tid, held)
        return {
            "status": "ok",
            "reason": None,
            "prepare_ms": (time.perf_counter() - t0) * 1000.0,
            "ipc_bytes": ipc_bytes,
            "memory_bytes": memory,
            "rows": len(self.array),
        }

    def run(self, table_id: str, query: dict, repeats: int = 1) -> dict:
        """`repeats` runs of the query back to back, timed as one batch (`timed`): the first run's
        answer and the batch's wall and CPU totals."""
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        planned = self.plan(table_id, query)

        def batch():
            first = self.execute(table_id, planned)
            for _ in range(repeats - 1):
                self.execute(table_id, planned)
            return first

        (count, total), wall, cpu = timed(batch)
        return {
            "count": count,
            "total": total,
            "wall_ns": wall,
            "cpu_ns": cpu,
            "repeats": repeats,
        }

    def calibrate(self, table_id: str, query: dict, target_ns: int, cap: int) -> dict:
        """The warm-up of one query: its first answer (what the equal-results check compares) and
        its batch size (`calibrate_repeats` over batches of 1, 2, 4, ... runs). One run, untimed
        and discarded, comes first: the first run of a query is cold (caches, lazy initialisation),
        and a query slow only then must not be calibrated as a slow one."""
        self.execute(table_id, self.plan(table_id, query))
        first = self.run(table_id, query)
        walls = {1: first["wall_ns"]}

        def batch_wall_ns(k: int) -> int:
            if k not in walls:
                walls[k] = self.run(table_id, query, k)["wall_ns"]
            return walls[k]

        k, capped = calibrate_repeats(batch_wall_ns, target_ns, cap)
        return {
            "count": first["count"],
            "total": first["total"],
            "repeats": k,
            "capped": capped,
        }
