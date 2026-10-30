"""Spike 13: sorted Decimal32 + ZSTD is 2.09x the size of Decimal64 (17.1 vs 8.2 MB at 10M rows,
decimal32_codecs, ORDER BY v; ToDo 3). Why?

Loads decimal32_codecs' own dataset (its cached 10M-row array, float64 at 7 dp) as Decimal32(7),
Decimal64(7), Int32 and Int64 (the unscaled integers, read from the decimals' own storage, so
exact), each sorted by the table (`ORDER BY v`, as the suite's primary_key cells) and, as a
control, in generation order (`ORDER BY tuple()`), under LZ4, ZSTD(1), ZSTD(3), ZSTD(9), ZSTD(19),
Delta + ZSTD(1) and T64 + ZSTD(1). Prints the column's compressed and uncompressed bytes.

If the integer pair repeats the Decimal ratio, the cause is the byte width, not Decimal. If a
transform (Delta, T64) removes it, the cause is how ZSTD finds matches in the raw stream.
ZSTD(19) is the one level here whose minimum match length is 3 bytes (FINDINGS §13).

Prints a TSV and writes spike_decimal32_zstd_results.tsv beside this file. Needs the chbench-ch
container; ~15 min at 10M rows.

    python -m spikes.spike_decimal32_zstd      (from services/bench/)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from synthdata.cache import load_or_generate

from chbench.cli import DEFAULT_CACHE_DIR
from chbench.harness.client import CHClient, Connection
from chbench.harness.ddl import drop_table_sql, optimize_sql
from chbench.harness.loader import INSERT_BATCH_ROWS
from chbench.harness.suite import load_suite

HERE = Path(__file__).parent
CHAINS = [
    "LZ4",
    "ZSTD(1)",
    "ZSTD(3)",
    "ZSTD(9)",
    "ZSTD(19)",
    "Delta, ZSTD(1)",
    "T64, ZSTD(1)",
]
ORDERS = {"sorted": "v", "generation": "tuple()"}
TABLE = "bench.spike_dec32"


def columns(arr: pa.Array, scale: int) -> dict[str, tuple[str, pa.Array]]:
    """The four column types over the same values; the integers are the decimals' storage."""
    d32 = pc.cast(
        arr, pa.decimal32(9, scale)
    )  # safe cast: raises if a value does not fit
    d64 = pc.cast(arr, pa.decimal64(18, scale))
    i32 = pa.Array.from_buffers(pa.int32(), len(d32), d32.buffers())
    i64 = pa.Array.from_buffers(pa.int64(), len(d64), d64.buffers())
    assert np.array_equal(i32.to_numpy(), i64.to_numpy()), "unscaled values must agree"
    return {
        "Decimal32": (f"Decimal32({scale})", d32),
        "Decimal64": (f"Decimal64({scale})", d64),
        "Int32": ("Int32", i32),
        "Int64": ("Int64", i64),
    }


def measure(
    c: CHClient, ch_type: str, values: pa.Array, chain: str, order_by: str
) -> tuple:
    c.command(drop_table_sql(TABLE))
    c.command(
        f"CREATE TABLE {TABLE} (row_id UInt64, v {ch_type} CODEC({chain})) ENGINE = MergeTree "
        f"ORDER BY {order_by} SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"
    )
    row_id = pa.array(np.arange(len(values), dtype=np.uint64), pa.uint64())
    t = pa.table({"row_id": row_id, "v": values})
    t = pa.Table.from_batches(
        t.to_batches(max_chunksize=INSERT_BATCH_ROWS), schema=t.schema
    )
    c.insert_arrow(TABLE, t)
    c.command(optimize_sql(TABLE))
    [(comp, uncomp, parts)] = c.query_rows(
        "SELECT sum(column_data_compressed_bytes), sum(column_data_uncompressed_bytes), count() "
        "FROM system.parts_columns WHERE database = 'bench' AND table = 'spike_dec32' "
        "AND column = 'v' AND active"
    )
    c.command(drop_table_sql(TABLE))
    return int(comp), int(uncomp), int(parts)


def main() -> None:
    suite = load_suite(HERE.parent / "suites" / "decimal32_codecs.yaml")
    arr = load_or_generate(suite.dataset, Path(DEFAULT_CACHE_DIR)).array
    cols = columns(arr, suite.dataset.decimal_places)
    c = CHClient(Connection.from_env())
    lines = [f"# server {c.version()}, {len(arr)} rows"]
    lines.append("order\ttype\tchain\tcompressed_bytes\tuncompressed_bytes\tparts")
    print("\n".join(lines), flush=True)
    for order, order_by in ORDERS.items():
        for label, (ch_type, values) in cols.items():
            for chain in CHAINS:
                comp, uncomp, parts = measure(c, ch_type, values, chain, order_by)
                lines.append(f"{order}\t{label}\t{chain}\t{comp}\t{uncomp}\t{parts}")
                print(lines[-1], flush=True)
    (HERE / "spike_decimal32_zstd_results.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
