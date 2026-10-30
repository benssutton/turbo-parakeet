"""Spike 13: sorted Decimal32 + ZSTD is 2.09x the size of Decimal64 (17.1 vs 8.2 MB at 10M rows,
decimal32_codecs, ORDER BY v; ToDo 3). Why?

Loads decimal32_codecs' own dataset (its cached 10M-row array, float64 at 7 dp) as Decimal32(7),
Decimal64(7), Int32 and Int64 (the unscaled integers, read from the decimals' own storage, so
exact), each sorted by the table (`ORDER BY v`, as the suite's primary_key cells) and, as a
control, in generation order (`ORDER BY tuple()`), under LZ4, ZSTD(1), ZSTD(3), ZSTD(9), ZSTD(19),
Delta + ZSTD(1) and T64 + ZSTD(1); sorted only, also ZSTD(13), (14), (17) and (18). Prints the
column's compressed and uncompressed bytes.

If the integer pair repeats the Decimal ratio, the cause is the byte width, not Decimal. If a
transform (Delta, T64) removes it, the cause is how ZSTD finds matches in the raw stream.
ClickHouse compresses these columns in 64 KiB blocks, for which python-zstandard's libzstd 1.5.7
level table gives ZSTD(13) btopt with a 4-byte minimum match, (14) btopt with 3, (17) / (18)
btultra with 3 and (19) btultra2 with 3 (FINDINGS §13).

Prints a TSV and writes spike_decimal32_zstd_results.tsv beside this file. Needs the chbench-ch
container; ~15 min at 10M rows.

    python -m spikes.spike_decimal32_zstd      (from services/bench/)
    python -m spikes.spike_decimal32_zstd --chains "ZSTD(13)" "ZSTD(14)" --orders sorted --append
    python -m spikes.spike_decimal32_zstd --emulator     (no server)

`--append` adds the cells to the TSV instead of rewriting it. `--emulator` runs no server: it
prints the sorted arrays' adjacent-pair statistics and `turbo_parakeet`'s ClickHouse size emulator
(`_plugin.clickhouse_profile`) for Decimal32 / Decimal64 over every chain, then python-zstandard's
libzstd over the raw sorted bytes with a level's own parameters and only the minimum match
changed (`libzstd_controls`), and writes spike_decimal32_zstd_emulator.tsv.
"""

from __future__ import annotations

import argparse
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
RESULTS = HERE / "spike_decimal32_zstd_results.tsv"
EMULATOR = HERE / "spike_decimal32_zstd_emulator.tsv"
CHAINS = [
    "LZ4",
    "ZSTD(1)",
    "ZSTD(3)",
    "ZSTD(9)",
    "ZSTD(19)",
    "Delta, ZSTD(1)",
    "T64, ZSTD(1)",
]
SORTED_ONLY = ["ZSTD(13)", "ZSTD(14)", "ZSTD(17)", "ZSTD(18)"]
ORDERS = {"sorted": "v", "generation": "tuple()"}
TABLE = "bench.spike_dec32"
HEADER = "order\ttype\tchain\tcompressed_bytes\tuncompressed_bytes\tparts"


def columns(arr: pa.Array, scale: int) -> dict[str, tuple[str, pa.Array]]:
    """The four column types over the same values; the integers are the decimals' storage."""
    # safe cast: raises if a value does not fit
    d32 = pc.cast(arr, pa.decimal32(9, scale))
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


def adjacent_pairs(unscaled: np.ndarray) -> tuple[float, float]:
    """Shares of sorted adjacent pairs that are equal, and that agree in every byte above the
    low one (the 8-byte two's complement value)."""
    x = np.sort(unscaled).astype(np.uint64)
    return float(np.mean(x[1:] == x[:-1])), float(
        np.mean((x[1:] >> 8) == (x[:-1] >> 8))
    )


def libzstd_controls(cols: dict[str, tuple[str, pa.Array]]) -> list[str]:
    """python-zstandard's bundled libzstd over the raw sorted bytes in 64 KiB blocks (the block
    size the emulator's ClickHouse block split gives these columns: two 32 KiB granules of
    Decimal32, one 64 KiB granule of Decimal64), each level's own parameters for a 64 KiB source
    with only `min_match` changed, to separate the minimum match from the strategy. Strategy
    numbers are libzstd's: 1 fast, 2 dfast, 3 greedy, 4 lazy, 5 lazy2, 6 btlazy2, 7 btopt,
    8 btultra, 9 btultra2."""
    import zstandard as zs

    blk = 1 << 16
    v = ".".join(map(str, zs.ZSTD_VERSION))
    lines = [
        f"# libzstd {v} (python-zstandard {zs.__version__}), 64 KiB blocks",
        "type\tlevel\tstrategy\tmin_match\tbytes",
    ]
    cells = [(1, None), (1, 3), (9, None), (9, 3), (13, None), (13, 3), (14, None)]
    cells += [(14, 4), (19, None), (19, 4)]
    for label in ("Decimal32", "Decimal64"):
        a = cols[label][1]
        raw = memoryview(a.buffers()[1])[: len(a) * a.type.byte_width]
        for level, mm in cells:
            kw = {} if mm is None else {"min_match": mm}
            p = zs.ZstdCompressionParameters.from_level(level, source_size=blk, **kw)
            c = zs.ZstdCompressor(compression_params=p)
            b = sum(len(c.compress(raw[i : i + blk])) for i in range(0, len(raw), blk))
            lines.append(f"{label}\t{level}\t{p.strategy}\t{p.min_match}\t{b}")
    return lines


def emulator(arr: pa.Array, scale: int) -> None:
    from turbo_parakeet import _plugin

    cols = columns(pc.take(arr, pc.sort_indices(arr)), scale)
    equal, high = adjacent_pairs(cols["Int64"][1].to_numpy())
    lines = [
        f"# sorted adjacent pairs: equal {equal:.4f}, equal above the low byte {high:.4f}",
        "type\tchain\temulated_bytes",
    ]
    for label in ("Decimal32", "Decimal64"):
        t = pa.table({"v": cols[label][1]})
        for chain in CHAINS[:4] + SORTED_ONLY + CHAINS[4:]:
            try:
                b = _plugin.clickhouse_profile(t, "v", chain)["bytes"]
            except Exception as e:  # noqa: BLE001
                b = f"unsupported ({type(e).__name__})"
            lines.append(f"{label}\t{chain}\t{b}")
    lines += libzstd_controls(cols)
    text = "\n".join(lines)
    print(text)
    EMULATOR.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chains", nargs="+", help="default: CHAINS, and SORTED_ONLY sorted"
    )
    ap.add_argument("--orders", nargs="+", choices=list(ORDERS), default=list(ORDERS))
    ap.add_argument("--append", action="store_true", help="append rows to the TSV")
    ap.add_argument(
        "--emulator", action="store_true", help="the size emulator; no server"
    )
    args = ap.parse_args()

    suite = load_suite(HERE.parent / "suites" / "decimal32_codecs.yaml")
    arr = load_or_generate(suite.dataset, Path(DEFAULT_CACHE_DIR)).array
    scale = suite.dataset.decimal_places
    if args.emulator:
        emulator(arr, scale)
        return
    cols = columns(arr, scale)
    c = CHClient(Connection.from_env())
    lines = [] if args.append else [f"# server {c.version()}, {len(arr)} rows", HEADER]
    print("\n".join(lines or [HEADER]), flush=True)
    for order in args.orders:
        chains = args.chains or CHAINS + (SORTED_ONLY if order == "sorted" else [])
        for label, (ch_type, values) in cols.items():
            for chain in chains:
                comp, uncomp, parts = measure(c, ch_type, values, chain, ORDERS[order])
                lines.append(f"{order}\t{label}\t{chain}\t{comp}\t{uncomp}\t{parts}")
                print(lines[-1], flush=True)
    with RESULTS.open("a" if args.append else "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
