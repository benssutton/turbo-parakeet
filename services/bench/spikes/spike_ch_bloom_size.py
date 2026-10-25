"""Spike 10: how ClickHouse sizes `bloom_filter`, `minmax` and `set` skip
indexes (task 1 of the skip-index cross-repository plan).

For each shape of an Int64 column of 1 000 000 rows:

- `distinct`: every value distinct (8 192 x G distinct values per index block);
- `clustered`: 100 distinct values per 8 192-row granule, each value confined
  to one granule (100 x G distinct values per index block);
- `ten`: 10 distinct values overall, in pseudo-random order (10 per block);

and for each GRANULARITY G in {1, 4}, the table (chbench's DDL form: Wide
parts forced, `ORDER BY tuple()`, `INDEX ix v TYPE <t> GRANULARITY G`) is
loaded, optimised to one part, and the index's compressed / uncompressed
bytes and marks bytes are read from `system.data_skipping_indices`. Indexes:
`bloom_filter(fpr)` for fpr in {0.001, 0.01, 0.025, 0.05}, `minmax`, and
`set(125)`. The distinct values per index block are measured from the
stored data, not assumed.

Runnable standalone: `python spikes/spike_ch_bloom_size.py`; prints the
tables and writes every cell to `spikes/spike_ch_bloom_size_results.tsv`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import clickhouse_connect

CH = dict(host="localhost", port=8123, username="CLICKHOUSE_USER", password="CLICKHOUSE_PASSWORD")
ROWS = 1_000_000
GRANULE = 8192
TABLE = "bench.spike_bloom_size"
SETTINGS = ("SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0, "
            "index_granularity = 8192")

SHAPES = {
    "distinct": "toInt64(number)",
    "clustered": f"toInt64(intDiv(number, {GRANULE}) * 100 + number % 100)",
    "ten": "toInt64(cityHash64(number) % 10)",
}
FPRS = [0.001, 0.01, 0.025, 0.05]
GRANULARITIES = [1, 4]


def indexes():
    for g in GRANULARITIES:
        for fpr in FPRS:
            yield f"bloom_filter({fpr})", g
        yield "minmax", g
        yield "set(125)", g


def block_distinct(client, g: int) -> tuple[float, int]:
    """Mean and max distinct values per index block of the stored column."""
    rows = client.query(
        "SELECT avg(u), max(u) FROM (SELECT uniqExact(v) AS u FROM "
        f"(SELECT v, intDiv(row_id, {GRANULE * g}) AS b FROM {TABLE}) GROUP BY b)"
    ).result_rows
    return float(rows[0][0]), int(rows[0][1])


def measure(client, shape: str, expr: str, index: str, g: int) -> dict:
    client.command(f"DROP TABLE IF EXISTS {TABLE}")
    client.command(
        f"CREATE TABLE {TABLE} (row_id UInt64, v Int64 CODEC(LZ4), "
        f"INDEX ix v TYPE {index} GRANULARITY {g}) "
        f"ENGINE = MergeTree ORDER BY tuple() {SETTINGS}")
    client.command(
        f"INSERT INTO {TABLE} SELECT number, {expr} FROM numbers({ROWS}) "
        "SETTINGS max_threads = 1, max_insert_threads = 1")
    client.command(f"OPTIMIZE TABLE {TABLE} FINAL")
    parts = client.query(
        f"SELECT count() FROM system.parts WHERE database = 'bench' AND table = 'spike_bloom_size' "
        "AND active").result_rows[0][0]
    comp, unc, marks = client.query(
        "SELECT sum(data_compressed_bytes), sum(data_uncompressed_bytes), sum(marks_bytes) "
        "FROM system.data_skipping_indices WHERE database = 'bench' AND table = 'spike_bloom_size'"
    ).result_rows[0]
    d_mean, d_max = block_distinct(client, g)
    blocks = math.ceil(ROWS / (GRANULE * g))
    return dict(shape=shape, index=index, granularity=g, parts=int(parts), blocks=blocks,
                distinct_per_block_mean=d_mean, distinct_per_block_max=d_max,
                compressed=int(comp), uncompressed=int(unc), marks_bytes=int(marks))


def main() -> None:
    client = clickhouse_connect.get_client(**CH)
    cells = []
    for shape, expr in SHAPES.items():
        for index, g in indexes():
            cells.append(measure(client, shape, expr, index, g))
            c = cells[-1]
            print(f"{shape:10s} G={g} {index:20s} blocks={c['blocks']:4d} "
                  f"D={c['distinct_per_block_mean']:8.1f} comp={c['compressed']:9d} "
                  f"unc={c['uncompressed']:9d} marks={c['marks_bytes']}", flush=True)
    client.command(f"DROP TABLE IF EXISTS {TABLE}")

    out = Path(__file__).with_name("spike_ch_bloom_size_results.tsv")
    cols = ["shape", "index", "granularity", "parts", "blocks", "distinct_per_block_mean",
            "distinct_per_block_max", "compressed", "uncompressed", "marks_bytes",
            "uncompressed_bits_per_row", "uncompressed_bits_per_block_value",
            "compressed_bits_per_block_value", "compressed_over_uncompressed"]
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for c in cells:
            d = c["distinct_per_block_mean"] * c["blocks"]
            c["uncompressed_bits_per_row"] = c["uncompressed"] * 8 / ROWS
            c["uncompressed_bits_per_block_value"] = c["uncompressed"] * 8 / d
            c["compressed_bits_per_block_value"] = c["compressed"] * 8 / d
            c["compressed_over_uncompressed"] = (c["compressed"] / c["uncompressed"]
                                                 if c["uncompressed"] else float("nan"))
            f.write("\t".join(f"{c[k]:.4f}" if isinstance(c[k], float) else str(c[k])
                              for k in cols) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
