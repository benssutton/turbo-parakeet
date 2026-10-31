"""Polars in a worker process (`worker.BaseWorker`). Imported only by the worker, after the parent
has set `POLARS_MAX_THREADS` (`inproc.InprocEngine`).

- Cast: the dataset column as a Polars Series cast (strict) to the cell's dtype (`polars_dtype`):
  a Categorical of any width to categories named "v" in the table's own namespace (two tables never
  share a mapping, and none touches Polars' process-global one, ToDo 11); `Enum` to the column's
  values in first-seen order unless the type lists them. A table id (`validate.table_id`) hashes
  the variant and the dataset id, so a namespace never sees two datasets (a reused id with new
  values would add to its categories, and a narrow width would overflow).
- `memory_bytes` is Polars' own `DataFrame.estimated_size()`: the physical buffers only, leaving
  out a string column's 16-byte views and a categorical's dictionary (1M rows of
  Categorical(UInt8) are 1,000,000 bytes, the codes). It is informational, not comparable across
  engines.
- `memory`: the single-chunk DataFrame, queried through `.lazy()`. `ipc`: the frame in Polars' own
  layout (`to_arrow(compat_level=CompatLevel.newest())`: string_view, dictionary<…string_view>) written
  by pyarrow's IPC writer — Polars' `write_ipc` takes no ZSTD level, and the recommenders' Polars
  sizes mirror pyarrow's writer on this layout — then read back from its bytes for the check (a
  memory-mapped read would lock the file on Windows), and scanned per query with `pl.scan_ipc`.
  `scan_ipc` memory-maps the file whatever its compression (Polars 2.0). An uncompressed file in
  record batches is queried from the mapped pages without a private copy (the "uncompressed IPC"
  cell); one whole-file batch is copied into private memory first, and a compressed file is
  decoded into private memory batch by batch.
- Query: `.select(pl.len(), <polars_total>)`, filtered by `polars_predicate`, `.collect()`;
  Categorical / Enum columns are compared and totalled through their string values.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from chbench.harness.engines.worker import BaseWorker
from chbench.harness.inproc_types import PolarsType, parse_polars_type

_DIGITS = {"ms": 3, "us": 6, "ns": 9}


def polars_dtype(t: PolarsType, namespace: str, array: pa.Array):
    """The Polars dtype of a parsed type; `namespace` names a Categorical's categories, `array`
    (the dataset column) gives a bare Enum its categories in first-seen order."""
    if t.kind == "Decimal":
        return pl.Decimal(t.precision, t.scale)
    if t.kind == "Datetime":
        return pl.Datetime(t.unit, t.tz)
    if t.kind == "Categorical":
        return pl.Categorical(
            pl.Categories("v", namespace=namespace, physical=getattr(pl, t.width))
        )
    if t.kind == "Enum":
        if t.categories is not None:
            return pl.Enum(list(t.categories))
        seen = pl.Series(array).drop_nulls().unique(maintain_order=True)
        return pl.Enum(seen.cast(pl.String).to_list())
    return getattr(pl, t.kind)


def _textual(dtype) -> bool:
    return dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))


def polars_total(dtype) -> pl.Expr:
    """The aggregate beside `pl.len()`, mirroring DataFusion's (`queries.datafusion_total`):
    bytes for strings, categoricals and binaries, days for dates, seconds for datetimes and times,
    the value as Float64 otherwise."""
    v = pl.col("v")
    if _textual(dtype):
        return v.cast(pl.String).str.len_bytes().cast(pl.Int64).sum()
    if dtype == pl.Binary:
        return v.bin.size().cast(pl.Int64).sum()
    if dtype == pl.Date:
        return v.to_physical().cast(pl.Float64).sum()
    if isinstance(dtype, pl.Datetime):
        return (
            v.to_physical().cast(pl.Float64) / float(10 ** _DIGITS[dtype.time_unit])
        ).sum()
    if dtype == pl.Time:
        return (v.to_physical().cast(pl.Float64) / 1e9).sum()
    return v.cast(pl.Float64).sum()


def polars_predicate(predicate_kind: str, values: pa.Array, dtype) -> pl.Expr:
    """`v` in `values` (`in`) or between `values[0]` and `values[1]` (`range`), the values (the
    dataset's type) cast to the column's dtype — to String for Categorical / Enum, whose range is
    compared on the strings."""
    target = pl.String if _textual(dtype) else dtype
    s = pl.Series("values", values).cast(target, strict=True)
    col = pl.col("v")
    if predicate_kind == "range":
        if isinstance(dtype, (pl.Categorical, pl.Enum)):
            col = col.cast(pl.String)
        return col.is_between(
            pl.lit(s.slice(0, 1)).first(), pl.lit(s.slice(1, 1)).first()
        )
    return col.is_in(s.implode())


class PolarsWorker(BaseWorker):
    engine = "polars"

    def engine_version(self) -> str:
        return pl.__version__

    def threads_in_use(self) -> int:
        return pl.thread_pool_size()

    def cast(self, type_spec, table_id):
        dtype = polars_dtype(parse_polars_type(type_spec), table_id, self.array)
        return pl.DataFrame({"v": pl.Series("v", self.array).cast(dtype, strict=True)})

    def values(self, held) -> pa.Array:
        return held.to_arrow()["v"].combine_chunks()

    def nbytes(self, held) -> int:
        return int(held.estimated_size())

    def write_ipc(self, held, path, codec, batch_rows) -> None:
        table = held.to_arrow(compat_level=pl.CompatLevel.newest()).combine_chunks()
        options = ipc.IpcWriteOptions(compression=codec)
        with ipc.new_file(str(path), table.schema, options=options) as w:
            w.write_table(table, max_chunksize=batch_rows)

    def read_ipc(self, path) -> pa.Array:
        return pl.read_ipc(Path(path).read_bytes())["v"].to_arrow()

    def hold(self, table_id, held) -> None:
        self.tables[table_id] = ("memory", held, held.schema["v"])

    def attach(self, table_id, path, held) -> None:
        self.tables[table_id] = ("ipc", str(path), held.schema["v"])

    def plan(self, table_id, query):
        _, _, dtype = self.tables[table_id]
        values = query["values"]
        where = (
            None
            if values is None
            else polars_predicate(query["predicate_kind"], values, dtype)
        )
        return where, polars_total(dtype)

    def execute(self, table_id, planned):
        kind, source, _ = self.tables[table_id]
        where, total = planned
        lf = pl.scan_ipc(source) if kind == "ipc" else source.lazy()
        if where is not None:
            lf = lf.filter(where)
        return lf.select(pl.len(), total).collect().row(0)
