"""The worker process of an in-process engine (one per engine and thread count; the parent side
is `inproc.InprocEngine`).

Protocol: the parent sends tuples `(op, *args)` over a `multiprocessing` Pipe and reads one reply
per request, `("ok", value)` or `("error", "<Type>: <message>")`; `("close",)` (or the pipe closing)
ends the loop. The ops are `BaseWorker`'s public methods: `version()`, `load(path, column)`,
`prepare(spec)` and `run(table_id, query)`. The worker ignores SIGINT: Ctrl-C is the parent's,
which closes the workers and marks the queries.

`BaseWorker` is the engine-independent half: it reads the dataset column once (`load`), prepares
a table (`prepare`: cast, row-by-row round trip back to the dataset's values, for `ipc` the file
written once with the cell's compression, reused when its size matches the store's, and its
values read back and checked) and times a query (`run`: the collector run first and paused,
`perf_counter_ns` wall and `process_time_ns` CPU — User + System over every thread of the worker —
around the whole query, for `ipc` from opening the file to the collected answer). An engine
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
            except (
                Exception
            ) as e:  # noqa: BLE001 - reported to the parent; the loop carries on
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


def mismatch(expected: pa.Array, got: pa.Array) -> str | None:
    """None when `got` holds `expected`'s values row by row (after a cast back to its type; NaN
    equal to NaN); else what differs first."""
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
        except (
            Exception
        ) as e:  # noqa: BLE001 - a cast the engine refuses is a skipped cell
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
                self.write_ipc(
                    held,
                    tmp,
                    ipc_codec(spec["compression"]),
                    ipc_batch_rows(spec["ipc_batch_rows"]),
                )
                tmp.replace(path)
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

    def run(self, table_id: str, query: dict) -> dict:
        planned = self.plan(table_id, query)
        (count, total), wall, cpu = timed(lambda: self.execute(table_id, planned))
        return {"count": count, "total": total, "wall_ns": wall, "cpu_ns": cpu}
