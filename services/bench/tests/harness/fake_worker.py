"""A worker engine with no engine behind it, for the worker-protocol tests: it holds the dataset as
it is and answers every query with the pyarrow reference. `CHBENCH_FAKE` (read when the worker
starts; comma-joined) plants faults: `skip=<type>` refuses that type at prepare, `wrong=<n>` makes
the n-th run answer one row too many, `fail=<n>` raises on the n-th run, `die=<n>` exits the
process (exit code 3) on the n-th run, `hang=<op>` makes that op (`version`, `load`, `prepare`,
`run`) sleep for an hour, `exit=start` exits the process as the worker is built. For `wrong`,
`fail` and `die`, n counts every `execute` across all jobs: each repeat of a batch, and in a
warm-up calibration's discarded cold run first (so `fail=1` there fails the cold run). Its thread
count is `POLARS_MAX_THREADS` when set (proving the parent's environment reached the child), else
the one it was given."""

from __future__ import annotations

import os
import time

import pyarrow as pa
import pyarrow.ipc as ipc

from chbench.harness.engines.worker import BaseWorker
from chbench.harness.queries import reference_result


class FakeWorker(BaseWorker):
    engine = "fake"

    def __init__(self, threads: int):
        super().__init__(threads)
        text = os.environ.get("CHBENCH_FAKE", "")
        self.faults = dict(f.split("=", 1) for f in text.split(",") if f)
        self.runs = 0
        if self.faults.get("exit") == "start":
            os._exit(4)
        if "hang" in self.faults:
            setattr(self, self.faults["hang"], lambda *a: time.sleep(3600))

    def engine_version(self) -> str:
        return "0.0.0"

    def threads_in_use(self) -> int:
        return int(os.environ.get("POLARS_MAX_THREADS", self.threads))

    def cast(self, type_spec, table_id):
        if self.faults.get("skip") == type_spec:
            raise ValueError(f"planted skip of {type_spec}")
        return pa.table({"v": self.array})

    def values(self, held):
        return held["v"].combine_chunks()

    def nbytes(self, held) -> int:
        return int(held.nbytes)

    def write_ipc(self, held, path, codec, batch_rows) -> None:
        options = ipc.IpcWriteOptions(compression=codec)
        with ipc.new_file(str(path), held.schema, options=options) as w:
            w.write_table(held.combine_chunks(), max_chunksize=batch_rows)

    def read_ipc(self, path):
        with pa.OSFile(str(path), "rb") as f:
            return ipc.open_file(f).read_all()["v"].combine_chunks()

    def hold(self, table_id, held) -> None:
        self.tables[table_id] = held

    def attach(self, table_id, path, held) -> None:
        self.tables[table_id] = held

    def plan(self, table_id, query):
        return query

    def execute(self, table_id, query):
        self.runs += 1
        n = str(self.runs)
        if self.faults.get("die") == n:
            os._exit(3)
        if self.faults.get("fail") == n:
            raise RuntimeError("planted failure")
        ref = reference_result(self.array, query["predicate_kind"], query["values"])
        extra = 1 if self.faults.get("wrong") == n else 0
        return ref.count + extra, ref.total
