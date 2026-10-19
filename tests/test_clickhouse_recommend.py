"""ClickHouse recommendations, Arrow IPC compression choice and the chbench handoff (spec 2026-10-19)."""

import polars as pl
import pyarrow as pa
import pytest
from datagen import describe_mixed, large_dataset_mixed

from analytics import _plugin
from analytics.describe._sizes import ipc_body_bytes


def pyarrow_body(arr: pa.Array, compression: str | int | None) -> int:
    """Body length of the record-batch IPC messages pyarrow writes for one column
    (`compression`: None, a ZSTD level, or "lz4" = LZ4_FRAME)."""
    return ipc_body_bytes(arr, compression)


@pytest.mark.parametrize(
    "make_frame",
    [lambda: describe_mixed(2_000), large_dataset_mixed],
    ids=["describe", "large"],
)
def test_lz4_frame_body_sizes_equal_pyarrow(make_frame):
    frame = make_frame()
    for name in frame.columns:
        arr = frame[name].rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if pa.types.is_dictionary(arr.type) or pa.types.is_null(arr.type):
            continue
        t = pa.table({"x": arr})
        assert _plugin.ipc_body_sizes(t, "lz4")["x"] == pyarrow_body(arr, "lz4"), name
        assert _plugin.ipc_body_sizes(t, "none")["x"] == pyarrow_body(arr, None), name
        assert _plugin.ipc_body_sizes(t, "zstd:1")["x"] == pyarrow_body(arr, 1), name
