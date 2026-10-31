"""The parent side of an in-process engine: `InprocEngine` starts one worker process
(`worker.main`, `multiprocessing` spawn context, over a Pipe) for one engine and thread count, and
`InprocRound` runs a round's jobs on it.

- Threads are fixed when the worker starts. `POLARS_MAX_THREADS` is set in this process's
  environment around `Process.start()`, so the child has it before it imports anything (spawn
  re-imports the parent's `__main__`, which may import Polars before the worker's own code runs);
  DataFusion's `SessionConfig().with_target_partitions(threads)` is built by the worker. The worker
  reports the count it runs (`info["threads"]`); `run_suite` warns on a mismatch.
- Memory: while a timed query runs, the parent samples the worker's RSS (psutil) about every
  millisecond — `time.sleep(0.001)` between `Connection.poll()` checks, because `poll(timeout)` on
  Windows waits a whole ~15 ms timer tick — and the sample's `memory_bytes` is the peak minus the
  RSS just before the query (a per-query peak cannot be reset in process on Windows: `peak_wset`
  is lifetime).
  A sample whose baseline read (or every peak read) fails has memory None: unknown, not guessed.
- Failure: the worker's exception comes back as `WorkerError` (that job's error, the round carries
  on); a worker that exits or breaks the pipe raises `WorkerDied`, and every later call to it
  raises it again, so its pending queries all end `error`. A call that outlives its op's timeout
  (`TIMEOUTS_S`, overridable per engine) terminates the worker and is `WorkerDied` the same way.
  A failure while starting (before or during the `version` call) closes the pipe and the process.
- The first answer of each job (the warm-up's) is compared with its pyarrow reference
  (`queries.results_agree`); a mismatch is the job's error, both answers in the message.
"""

from __future__ import annotations

import contextlib
import datetime
import multiprocessing
import os
import time
from pathlib import Path
from typing import Callable

import psutil

from chbench.harness import quiet
from chbench.harness.engines import worker
from chbench.harness.engines.base import PrepareResult
from chbench.harness.queries import results_agree
from chbench.harness.runner import Job, RoundResult, Sample

FACTORIES = {
    "polars": "chbench.harness.engines.polars_engine:PolarsWorker",
    "datafusion": "chbench.harness.engines.datafusion_engine:DataFusionWorker",
}
# A timed sample of an in-process query is a batch of k runs: k is the smallest power of two whose
# batch lasts TARGET_BATCH_NS of wall time (process_time_ns moves in 15.625 ms steps on Windows),
# at most MAX_REPEATS, calibrated in the warm-up and fixed for the run's timed rounds.
TARGET_BATCH_NS = 250_000_000
MAX_REPEATS = 1024
_CONTEXT = multiprocessing.get_context("spawn")
_POLL_S = 0.001
# Per-op call timeouts (s), generous: they catch a stuck worker (a deadlocked engine, a runaway
# query), never a slow one. `version` is the spawn plus importing Polars / DataFusion, seconds even
# on a cold disk under a virus scanner. `load` reads one column of the dataset and `prepare` casts,
# verifies row by row and writes (ZSTD at a high level) then re-reads an IPC file: at 10M rows tens
# of seconds, minutes for the slow codecs, so half an hour. `run` is one sample, a calibrated batch of
# k <= 1024 repeats of about 250 ms or more, and `calibrate` is the warm-up's batches of 1, 2, 4, ...
# up to k (about 2 x k runs); only a single query far slower than any the bench plans reaches 5 minutes.
TIMEOUTS_S = {
    "version": 60.0,
    "load": 1800.0,
    "prepare": 1800.0,
    "run": 300.0,
    "calibrate": 300.0,
}
_CLOSE_WAIT_S = 5.0
_NAN = float("nan")


class WorkerError(RuntimeError):
    """The worker raised: the message is its `<Type>: <message>`."""


class WorkerDied(RuntimeError):
    """The worker process exited, or its pipe broke."""


def default_threads() -> int:
    """An in-process engine's thread count when a block names none: the Docker VM's CPU count
    (the ClickHouse server's budget, so the engines compare at equal CPU), else this host's.
    """
    return quiet.docker_cpus() or os.cpu_count() or 1


def ipc_path(cache_dir, table_id: str) -> Path:
    """Where an in-process table's IPC file lives: `<cache_dir>/ipc/<table_id>.arrow`."""
    return Path(cache_dir) / "ipc" / f"{table_id}.arrow"


@contextlib.contextmanager
def _environment(**values: str):
    """`values` set in `os.environ` for the block (it mutates the process environment: not
    thread-safe, so engines are started one at a time)."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class InprocEngine:
    """One worker process of `engine` (`FACTORIES`) at `threads`, started at construction; `info`
    is its `version()` reply. `timeouts` overrides `TIMEOUTS_S` per op."""

    def __init__(
        self, engine: str, threads: int, *, timeouts: dict[str, float] | None = None
    ):
        self.name, self.threads = engine, threads
        self.timeouts = {**TIMEOUTS_S, **(timeouts or {})}
        self._dead: str | None = None
        # job key -> (k, capped), from the warm-up; k is per run: the next run's warm-up calibrates again
        self.repeats: dict[str, tuple[int, bool]] = {}
        self._proc = None
        self._conn, child = _CONTEXT.Pipe()
        try:
            proc = _CONTEXT.Process(
                target=worker.main,
                args=(child, FACTORIES[engine], threads),
                name=f"chbench-{engine}-{threads}",
                daemon=True,
            )
            with _environment(POLARS_MAX_THREADS=str(threads)):
                proc.start()
            self._proc = proc
            child.close()
            try:
                self._ps = psutil.Process(proc.pid)
            except psutil.Error as e:
                self._dead = (
                    f"{engine} worker (pid {proc.pid}) died at start: "
                    f"{type(e).__name__}"
                )
                raise WorkerDied(self._dead) from e
            self.info: dict = self.call("version")[0]
        except BaseException:
            child.close()
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._proc.pid

    def _rss(self) -> int | None:
        """The worker's RSS; None when psutil cannot read it."""
        try:
            return self._ps.memory_info().rss
        except psutil.Error:
            return None

    def _peak(self, peak: int | None) -> int | None:
        rss = self._rss()
        return rss if peak is None or (rss is not None and rss > peak) else peak

    def call(self, op: str, *args, sample_memory: bool = False):
        """One request and its reply: (value, the RSS peak above the RSS just before the request
        when `sample_memory` and both were read, else None). `WorkerError` when the worker raised,
        `WorkerDied` when it is gone or outlived the op's timeout (then it is terminated).
        """
        if self._dead is not None:
            raise WorkerDied(self._dead)
        limit = self.timeouts.get(op, TIMEOUTS_S["run"])
        base = self._rss() if sample_memory else None
        peak = None
        try:
            self._conn.send((op, *args))
            deadline = time.perf_counter() + limit
            while not self._conn.poll():
                if not self._proc.is_alive():
                    raise EOFError(f"exit code {self._proc.exitcode}")
                if time.perf_counter() > deadline:
                    self._dead = (
                        f"{self.name} worker (pid {self._proc.pid}) timed out after "
                        f"{limit:g}s in {op}"
                    )
                    self._proc.terminate()
                    self._proc.join(_CLOSE_WAIT_S)
                    raise WorkerDied(self._dead)
                if sample_memory:
                    peak = self._peak(peak)
                time.sleep(_POLL_S)
            status, value = self._conn.recv()
        except (EOFError, OSError) as e:
            self._dead = f"{self.name} worker (pid {self._proc.pid}) died: {e or type(e).__name__}"
            raise WorkerDied(self._dead) from e
        memory = None
        if sample_memory:
            peak = self._peak(peak)
            if base is not None and peak is not None:
                memory = max(0, peak - base)
        if status == "error":
            raise WorkerError(value)
        return value, memory

    def version(self) -> str:
        return str(self.info["version"])

    def load(self, path, column: str) -> int:
        """The worker reads the dataset column once (`BaseWorker.load`)."""
        return self.call("load", str(path), column)[0]

    def prepare(
        self, variant, table_id: str, ipc_file: Path, expect_ipc_bytes: int | None
    ) -> PrepareResult:
        """Prepare one table on the worker; an `ipc` file of `expect_ipc_bytes` is reused."""
        spec = {
            "table_id": table_id,
            "type_spec": variant.type_spec,
            "storage": variant.storage,
            "compression": variant.compression,
            "ipc_batch_rows": variant.ipc_batch_rows,
            "ipc_path": str(ipc_file),
            "expect_ipc_bytes": expect_ipc_bytes,
        }
        try:
            r, _ = self.call("prepare", spec)
        except (WorkerError, WorkerDied) as e:
            return PrepareResult(
                "error", f"{type(e).__name__}: {e}", None, None, None, None
            )
        return PrepareResult(
            r["status"],
            r["reason"],
            r["prepare_ms"],
            r["ipc_bytes"],
            r["memory_bytes"],
            r["rows"],
        )

    def start_round(
        self,
        round_no: int | None,
        *,
        sample_no: int = 0,
        timed: bool = True,
        on_sample: Callable[[str, Sample], None] | None = None,
    ) -> "InprocRound":
        return InprocRound(
            self, round_no, sample_no=sample_no, timed=timed, on_sample=on_sample
        )

    def close(self) -> None:
        """Ask the loop to end, wait, then terminate; safe to call twice, and on an engine whose
        process never started."""
        proc = self._proc
        if proc is not None:
            if self._dead is None and proc.is_alive():
                with contextlib.suppress(OSError):
                    self._conn.send(("close",))
            proc.join(_CLOSE_WAIT_S)
            if proc.is_alive():
                proc.terminate()
                proc.join(_CLOSE_WAIT_S)
        self._conn.close()


class InprocRound:
    """One round of an in-process engine's jobs. A timed job's sample is delivered (`on_sample`)
    as soon as its answer returns; the untimed warm-up calibrates each job's batch size
    (`BaseWorker.calibrate`; kept in `engine.repeats` for the run's timed rounds, a capped k a
    round warning) and compares its first answer with the job's reference, recording a mismatch as
    the job's error. A sample's `client_ms` / k includes one pipe round trip per batch. A Ctrl-C during a batch interrupts `call` as during one query: the batch's
    sample is not taken, and `close()` waits for the worker to finish the batch."""

    def __init__(
        self,
        engine: InprocEngine,
        round_no: int | None = None,
        *,
        sample_no: int = 0,
        timed: bool = True,
        on_sample: Callable[[str, Sample], None] | None = None,
    ):
        self.engine = engine
        self.round_no = round_no
        self.sample_no = sample_no
        self.timed = timed
        self.on_sample = on_sample
        self.result = RoundResult()

    def run(self, job: Job) -> None:
        ran_at = datetime.datetime.now(datetime.timezone.utc)
        t0 = time.perf_counter()
        try:
            if self.timed:
                if job.key not in self.engine.repeats:
                    self.result.warnings.append(
                        f"{job.key}: not calibrated (no warm-up ran); sampled at 1 repeat"
                    )
                k, capped = self.engine.repeats.get(job.key, (1, False))
                reply, memory = self.engine.call(
                    "run", job.name, job.payload, k, sample_memory=True
                )
            else:
                reply, _ = self.engine.call(
                    "calibrate", job.name, job.payload, TARGET_BATCH_NS, MAX_REPEATS
                )
        except (WorkerError, WorkerDied) as e:
            self.result.errors[job.key] = f"{type(e).__name__}: {e}"
            return
        client_ms = (time.perf_counter() - t0) * 1000.0
        if not self.timed:
            self.engine.repeats[job.key] = (reply["repeats"], reply["capped"])
            if reply["capped"]:
                self.result.warnings.append(
                    f"{job.key}: {reply['repeats']} repeats (the cap) last under "
                    f"{TARGET_BATCH_NS / 1e6:.0f} ms; its samples are recorded as capped"
                )
            ref = job.reference
            if ref is not None and not results_agree(
                ref, reply["count"], reply["total"]
            ):
                self.result.errors[job.key] = (
                    f"{self.engine.name} answered count {reply['count']}, total "
                    f"{reply['total']!r}; the pyarrow reference is count {ref.count}, total "
                    f"{ref.total!r}"
                )
            return
        sample = Sample(
            sample_no=self.sample_no,
            client_ms=client_ms / k,
            server_ms=reply["wall_ns"] / k / 1e6,
            read_rows=_NAN,
            read_bytes=_NAN,
            memory_bytes=memory,
            selected_marks=_NAN,
            selected_granules=_NAN,
            profile_events={},
            ran_at=ran_at,
            round_no=self.round_no,
            cpu_us=reply["cpu_ns"] / k / 1e3,
            repeats=k,
            repeats_capped=capped,
        )
        self.result.samples[job.key] = sample
        if self.on_sample is not None:
            self.on_sample(job.key, sample)

    def finish(self) -> RoundResult:
        return self.result

    def interrupt(self) -> None:
        """Nothing to deliver: each sample was delivered as it landed."""
