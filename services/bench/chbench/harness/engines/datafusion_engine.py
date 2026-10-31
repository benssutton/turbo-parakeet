"""DataFusion in a worker process (`worker.BaseWorker`).

- One `SessionContext(SessionConfig().with_target_partitions(threads))` per worker.
- Cast: `pyarrow.compute.cast` of the dataset column to the cell's pyarrow type (dictionaries
  included: a dictionary whose indices cannot hold the distinct values fails, a skipped cell).
- `memory_bytes` is the Arrow table's `nbytes`: every buffer, a dictionary's values included. It
  is informational, not comparable across engines.
- `memory`: the column registered with `register_record_batches` as `threads` partitions, dealt
  round-robin in batches of min(65 536, ceil(rows / threads)) rows, so every thread gets a
  partition however small the table (a memory cell has no `ipc_batch_rows`).
  `ipc`: written once by pyarrow's IPC writer in the Arrow type (the layout the recommenders
  measure for DataFusion), in record batches of `ipc_batch_rows`, read back for the check, and
  registered (`register_arrow`) and deregistered around each query, so a run measures decode +
  query and no handle on the file outlives a query (it can be deleted while the worker lives).
- Query: the SQL the parent rendered (`queries.datafusion_sql`), collected to Arrow; the answer is
  its first row.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from datafusion import SessionConfig, SessionContext

from chbench.harness.engines.worker import BaseWorker
from chbench.harness.inproc_types import parse_datafusion_type
from chbench.harness.queries import datafusion_table

_BATCH_ROWS = 65_536


class DataFusionWorker(BaseWorker):
    engine = "datafusion"

    def __init__(self, threads: int):
        super().__init__(threads)
        self.ctx = SessionContext(SessionConfig().with_target_partitions(threads))

    def engine_version(self) -> str:
        return datafusion.__version__

    def threads_in_use(self) -> int:
        return self.threads

    def cast(self, type_spec, table_id):
        return pa.table({"v": pc.cast(self.array, parse_datafusion_type(type_spec))})

    def values(self, held) -> pa.Array:
        return held["v"].combine_chunks()

    def nbytes(self, held) -> int:
        return int(held.nbytes)

    def write_ipc(self, held, path, codec, batch_rows) -> None:
        options = ipc.IpcWriteOptions(compression=codec)
        with ipc.new_file(str(path), held.schema, options=options) as w:
            w.write_table(held.combine_chunks(), max_chunksize=batch_rows)

    def read_ipc(self, path) -> pa.Array:
        with pa.OSFile(str(path), "rb") as f:
            return ipc.open_file(f).read_all()["v"].combine_chunks()

    def hold(self, table_id, held) -> None:
        rows = max(1, min(_BATCH_ROWS, -(-held.num_rows // self.threads)))
        batches = held.to_batches(max_chunksize=rows)
        parts = [batches[i :: self.threads] for i in range(self.threads)]
        self.ctx.register_record_batches(
            datafusion_table(table_id), [p for p in parts if p] or [batches]
        )
        self.tables[table_id] = ("memory", None)

    def attach(self, table_id, path, held) -> None:
        self.tables[table_id] = ("ipc", str(path))

    def plan(self, table_id, query):
        return query["sql"]

    def execute(self, table_id, planned):
        kind, path = self.tables[table_id]
        name = datafusion_table(table_id)
        if kind == "ipc":
            self.ctx.register_arrow(name, path)
        try:
            answer = self.ctx.sql(planned).to_arrow_table()
        finally:
            if kind == "ipc":
                self.ctx.deregister_table(name)
        return answer.column(0)[0].as_py(), answer.column(1)[0].as_py()
