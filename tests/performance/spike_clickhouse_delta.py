"""
Delta / DoubleDelta / GCD on-disk layout: emulator vs the live ClickHouse server (spec
docs/superpowers/specs/2026-10-23-delta-codecs-design.md §3, §7).

For each column and codec chain: create a one-column Wide MergeTree table (as chbench does:
min_bytes_for_wide_part = 0), insert the column as Arrow IPC over HTTP, OPTIMIZE FINAL, read
`data_compressed_bytes` of the active part and compare with the emulator
(`_plugin.clickhouse_profile`).  Needs the `chbench-ch` container (http://localhost:8123, user
CLICKHOUSE_USER, password CLICKHOUSE_PASSWORD, database bench).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/spike_clickhouse_delta.py [--rows N]
"""

from __future__ import annotations

import argparse
import base64
import urllib.parse
import urllib.request

import numpy as np
import pyarrow as pa

from turbo_parakeet import _plugin

URL = "http://localhost:8123/"
AUTH = "Basic " + base64.b64encode(b"CLICKHOUSE_USER:CLICKHOUSE_PASSWORD").decode()
TABLE = "bench.spike_delta"
CODECS = [
    "NONE",
    "Delta",
    "DoubleDelta",
    "GCD",
    "LZ4",
    "Delta, LZ4",
    "DoubleDelta, LZ4",
    "GCD, LZ4",
    "Delta, ZSTD(1)",
    "DoubleDelta, ZSTD(1)",
    "GCD, ZSTD(1)",
]


def query(sql: str, body: bytes | None = None) -> bytes:
    params = (
        {"allow_suspicious_codecs": "1", "query": sql}
        if body is not None
        else {"allow_suspicious_codecs": "1"}
    )
    url = URL + "?" + urllib.parse.urlencode(params)
    data = body if body is not None else sql.encode()
    req = urllib.request.Request(url, data=data, headers={"Authorization": AUTH})
    try:
        with urllib.request.urlopen(req) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from e


def columns(n: int) -> dict[str, pa.Array]:
    rng = np.random.default_rng(0)
    jitter = np.cumsum(60_000 + rng.integers(-5, 6, n))
    gaps = rng.random(n) < 0.01
    jitter = jitter + np.cumsum(np.where(gaps, rng.integers(1_000, 60_000, n), 0))
    return {
        "regular_u32": pa.array(1_600_000_000 + 60 * np.arange(n), pa.uint32()),
        "jitter_i64": pa.array(1_600_000_000_000 + jitter, pa.int64()),
        "walk_i64": pa.array(np.cumsum(rng.integers(-1000, 1001, n)), pa.int64()),
        "ids_u64": pa.array(
            np.cumsum(rng.integers(1, 11, n)).astype(np.uint64), pa.uint64()
        ),
        "walk_i32": pa.array(np.cumsum(rng.integers(-100, 101, n)), pa.int32()),
        "walk_i16": pa.array(
            np.clip(np.cumsum(rng.integers(-3, 4, n)), -30000, 30000), pa.int16()
        ),
        "random_u16": pa.array(rng.integers(0, 65536, n), pa.uint16()),
        "random_i8": pa.array(rng.integers(-128, 128, n), pa.int8()),
        "gcd_i64": pa.array(rng.integers(-(2**40) // 3, 2**40 // 3, n) * 3, pa.int64()),
        "gcd_ts_ms": pa.array(
            1_600_000_000_000 + 1000 * rng.integers(0, 10**7, n), pa.int64()
        ),
        "gcd_u16": pa.array(2 * rng.integers(0, 30000, n), pa.uint16()),
        "gcd_i8": pa.array(5 * rng.integers(-25, 25, n), pa.int8()),
        # most negative double-deltas: alternate extremes
        "extreme_i8": pa.array(
            np.tile([-128, 127, -128, 0], n // 4 + 1)[:n], pa.int8()
        ),
        "extreme_i16": pa.array(
            np.tile([-32768, 32767, 0, -32768], n // 4 + 1)[:n], pa.int16()
        ),
        "extreme_i32": pa.array(
            np.tile([-(2**31), 2**31 - 1, 0, 5], n // 4 + 1)[:n], pa.int32()
        ),
        "extreme_i64": pa.array(
            np.tile([-(2**63), 2**63 - 1, 0, 7], n // 4 + 1)[:n], pa.int64()
        ),
    }


def measure(name: str, arr: pa.Array, ch_type: str, codec: str) -> int:
    query(f"DROP TABLE IF EXISTS {TABLE}")
    query(
        f"CREATE TABLE {TABLE} (v {ch_type} CODEC({codec})) ENGINE = MergeTree ORDER BY tuple() "
        "SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"
    )
    try:
        sink = pa.BufferOutputStream()
        t = pa.table({"v": arr})
        with pa.ipc.new_stream(sink, t.schema) as w:
            w.write_table(t)
        query(f"INSERT INTO {TABLE} FORMAT ArrowStream", sink.getvalue().to_pybytes())
        query(f"OPTIMIZE TABLE {TABLE} FINAL")
        out = query(
            "SELECT sum(column_data_compressed_bytes), count() FROM system.parts_columns "
            "WHERE database = 'bench' AND table = 'spike_delta' AND column = 'v' AND active "
            "FORMAT TSV"
        ).decode()
        total, parts = (int(x) for x in out.split())
        assert parts == 1, f"{name}/{codec}: {parts} active parts"
        return total
    finally:
        query(f"DROP TABLE IF EXISTS {TABLE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    print(
        f"{'column':14} {'type':8} {'codec':22} {'emulated':>10} {'measured':>10} {'diff %':>8}"
    )
    for name, arr in columns(args.rows).items():
        if args.only and args.only not in name:
            continue
        t = pa.table({"v": arr})
        for codec in CODECS:
            p = _plugin.clickhouse_profile(t, "v", codec)
            ch_type, emu = p["clickhouse_type"], p["bytes"]
            try:
                got = measure(name, arr, ch_type, codec)
            except RuntimeError as e:
                print(f"{name:14} {ch_type:8} {codec:22} ERROR {str(e)[:80]!r}")
                continue
            diff = (emu - got) / got * 100
            print(f"{name:14} {ch_type:8} {codec:22} {emu:>10} {got:>10} {diff:>+8.2f}")


if __name__ == "__main__":
    main()
