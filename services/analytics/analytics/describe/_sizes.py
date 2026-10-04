"""Arrow IPC body sizes shared by the Python implementations; pyarrow is the oracle.

A column's size is the body length of the IPC messages that carry it (dictionary
batches + record batch). pyarrow pads every buffer to 8 bytes, writes a validity
buffer only when the array has nulls, and gives an empty buffer no space. With
ZSTD, each non-empty buffer is an 8-byte uncompressed-length prefix plus one ZSTD
frame — pyarrow never falls back to raw bytes. src/common/ipc_sizes.rs mirrors this.

Arrow sizes use the classic layout (CompatLevel.oldest(): LargeUtf8, LargeList);
Polars sizes are the IPC body of its native layout, plain and ZSTD
(CompatLevel.newest(): Utf8View/BinaryView) — what `write_ipc(compression="zstd")`
writes.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

SIZE_KEYS = (
    "size_bytes",
    "size_zstd_bytes",
    "size_polars_bytes",
    "size_polars_zstd_bytes",
)


def ipc_body_bytes(arr: pa.Array, zstd_level: int | None) -> int:
    batch = pa.record_batch([arr], names=["x"])
    codec = (
        None if zstd_level is None else pa.Codec("zstd", compression_level=zstd_level)
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(
        sink, batch.schema, options=pa.ipc.IpcWriteOptions(compression=codec)
    ) as writer:
        writer.write_batch(batch)
    return sum(
        m.body.size
        for m in pa.ipc.MessageReader.open_stream(sink.getvalue())
        if m.type != "schema"
    )


def column_sizes(s: pl.Series, zstd_level: int) -> dict[str, int]:
    s = s.rechunk()
    classic = s.to_arrow(compat_level=pl.CompatLevel.oldest())
    native = s.to_arrow(compat_level=pl.CompatLevel.newest())
    return {
        "size_bytes": ipc_body_bytes(classic, None),
        "size_zstd_bytes": ipc_body_bytes(classic, zstd_level),
        "size_polars_bytes": ipc_body_bytes(native, None),
        "size_polars_zstd_bytes": ipc_body_bytes(native, zstd_level),
    }
