"""Spike 9: the ClickHouse type / codec / skip-index vocabulary behind the
turbo-parakeet recommendations (task 13 of the cross-repository plan).

For every type the recommender may emit, the script runs chbench's DDL form
(Wide parts forced, `CODEC(<codec>, <compression>)`, `LowCardinality` as a
type wrapper) and a small INSERT ... SELECT of representative values, and
records accepted / rejected with the server's own message and the stage
(`create` or `insert`) that rejected it. The matrices are factored like
spec 5.2's:

- codec x compression, skip index `none`;
- skip index x compression `LZ4`, codec `none`;
- both over nullable false / true.

Then it measures the GCD codec on disk: an Int64 column of multiples of 1000
and the same values / 1000, `GCD, LZ4` against `LZ4`.

Runnable standalone: `python spikes/spike_ch_vocabulary.py`; prints the
matrices and writes every cell to `spikes/spike_ch_vocabulary_results.tsv`.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import clickhouse_connect

CH = dict(
    host="localhost",
    port=8123,
    username="CLICKHOUSE_USER",
    password="CLICKHOUSE_PASSWORD",
)

ENUM16 = "Enum16(" + ", ".join(f"'v{i:03d}' = {i}" for i in range(1, 201)) + ")"

# (type DDL, SQL expression of `number` giving a representative value)
TYPES = [
    ("UInt8", "toUInt8(number % 200)"),
    ("UInt16", "toUInt16(number * 7 % 60000)"),
    ("UInt64", "number * 1000"),
    ("Int8", "toInt8(number % 100 - 50)"),
    ("Int16", "toInt16(number * 7 % 60000 - 30000)"),
    ("Int64", "toInt64(number) * 1000 - 500000"),
    ("Float32", "toFloat32(number / 8)"),
    ("Bool", "number % 2 = 1"),
    ("Decimal128(4)", "toDecimal128(number / 10, 4)"),
    ("Date32", "toDate32('2020-01-01') + number"),
    ("DateTime", "toDateTime('2020-01-01 00:00:00') + number * 60"),
    ("DateTime('UTC')", "toDateTime('2020-01-01 00:00:00', 'UTC') + number * 60"),
    ("DateTime64(0)", "toDateTime64('2020-01-01 00:00:00', 0) + number"),
    ("DateTime64(6)", "toDateTime64('2020-01-01 00:00:00', 6) + number"),
    ("DateTime64(9, 'UTC')", "toDateTime64('2020-01-01 00:00:00', 9, 'UTC') + number"),
    ("FixedString(4)", "toFixedString(toString(number % 1000), 4)"),
    (
        "Enum8('a' = 1, 'b' = 2)",
        "CAST(if(number % 2 = 0, 'a', 'b') AS Enum8('a' = 1, 'b' = 2))",
    ),
    (
        ENUM16,
        f"CAST(concat('v', leftPad(toString(number % 200 + 1), 3, '0')) AS {ENUM16})",
    ),
    ("Time64(3)", "CAST(number * 1000 AS Time64(3))"),
    ("FixedString(3)", "toFixedString(toString(number % 1000), 3)"),
    ("FixedString(16)", "toFixedString(toString(number), 16)"),
    # the vocabulary measured earlier (spec 5.2), for comparison and for the
    # GCD column the earlier spike predates
    ("Float64", "number / 8"),
    ("Decimal32(8)", "toDecimal32(number / 1000, 8)"),
    ("Decimal64(8)", "toDecimal64(number / 1000, 8)"),
    ("Int32", "toInt32(number)"),
    ("UInt32", "toUInt32(number)"),
    ("String", "toString(number % 50)"),
    ("Date", "toDate('2020-01-01') + number"),
    ("DateTime64(3)", "toDateTime64('2020-01-01 00:00:00', 3) + number"),
]
SETTINGS_TIME64 = {"enable_time_time64_type": 1}

CODECS = [
    "none",
    "GCD",
    "Delta",
    "DoubleDelta",
    "Gorilla",
    "T64",
    "FPC",
    "LowCardinality",
]
COMPRESSIONS = ["LZ4", "ZSTD(3)"]
SKIPS = ["none", "minmax", "set(100)", "bloom_filter"]

WIDE = "SETTINGS min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"
ROWS = 1000


def short(label: str) -> str:
    return label if len(label) < 28 else label[:24] + "..."


def clean(exc: Exception) -> str:
    """The server's message without the HTTP wrapper: `code N NAME: message`."""
    text = str(exc)
    text = re.sub(r"\s*\(for url [^)]*\)\s*$", "", text.strip())
    m = re.search(r"Code: (\d+)\. DB::Exception: (.*)", text, re.S)
    body = (m.group(2) if m else text).strip().splitlines()[0]
    body = re.sub(r"\s*\(version [^)]*\)\s*$", "", body)
    nm = re.search(r"\(([A-Z_0-9]+)\)\s*$", body)
    name = nm.group(1) if nm else ""
    if nm:
        body = body[: nm.start()].rstrip()
    return f"code {m.group(1) if m else '?'} {name}: {body}"


def run_cell(
    client,
    tid: int,
    ch_type: str,
    expr: str,
    nullable: bool,
    codec: str,
    compression: str,
    skip: str,
    settings: dict,
) -> tuple[str, str, str]:
    """-> (status, stage, message). Status `accepted` / `rejected`."""
    name = f"bench.vocab_{tid}"
    col = ch_type
    if nullable:
        col = f"Nullable({col})"
    if codec == "LowCardinality":
        col = f"LowCardinality({col})"
    clause = (
        f"CODEC({compression})"
        if codec in ("none", "LowCardinality")
        else f"CODEC({codec}, {compression})"
    )
    cols = ["row_id UInt64", f"v {col} {clause}"]
    if skip != "none":
        cols.append(f"INDEX ix v TYPE {skip} GRANULARITY 1")
    ddl = f"CREATE TABLE {name} ({', '.join(cols)}) ENGINE = MergeTree ORDER BY tuple() {WIDE}"
    client.command(f"DROP TABLE IF EXISTS {name}")
    try:
        client.command(ddl, settings=settings)
    except Exception as e:  # noqa: BLE001
        return "rejected", "create", clean(e)
    try:
        value = f"if(number % 7 = 0, NULL, {expr})" if nullable else expr
        client.command(
            f"INSERT INTO {name} SELECT number, {value} FROM numbers({ROWS})",
            settings=settings,
        )
        n = client.command(f"SELECT count() FROM {name}")
        assert int(n) == ROWS
    except Exception as e:  # noqa: BLE001
        client.command(f"DROP TABLE IF EXISTS {name}")
        return "rejected", "insert", clean(e)
    client.command(f"DROP TABLE IF EXISTS {name}")
    return "accepted", "", ""


def matrices(client) -> list[tuple]:
    rows = []
    for ch_type, expr in TYPES:
        variants = [("default", {})]
        if ch_type.startswith("Time64"):
            variants = [("default", {}), ("enable_time_time64_type=1", SETTINGS_TIME64)]
        for setting_label, settings in variants:
            label = (
                ch_type
                if setting_label == "default"
                else f"{ch_type} [{setting_label}]"
            )
            for nullable in (False, True):
                cells = [(c, z, "none") for c in CODECS for z in COMPRESSIONS]
                cells += [("none", "LZ4", s) for s in SKIPS if s != "none"]
                for codec, compression, skip in cells:
                    status, stage, msg = run_cell(
                        client,
                        0,
                        ch_type,
                        expr,
                        nullable,
                        codec,
                        compression,
                        skip,
                        settings,
                    )
                    rows.append(
                        (label, nullable, codec, compression, skip, status, stage, msg)
                    )
    return rows


def print_matrix(rows: list[tuple]) -> None:
    """Per type, one line per (codec, skip index) over nullable x compression;
    identical outcomes collapse, differing ones list the cells they hold for
    (n0 / n1 = nullable false / true, then the compression)."""
    by_type = collections.OrderedDict()
    for r in rows:
        by_type.setdefault(r[0], []).append(r)
    for label, group in by_type.items():
        print(f"\n=== {label}")
        seen: dict[tuple, dict] = collections.OrderedDict()
        for _, nullable, codec, compression, skip, status, stage, msg in group:
            outcome = (status, f"@{stage}: {msg}" if status == "rejected" else "")
            seen.setdefault((codec, skip), {}).setdefault(outcome, []).append(
                f"n{int(nullable)}/{compression}"
            )
        for (codec, skip), outcomes in seen.items():
            tag = f"codec={codec:<14} skip={skip:<13}"
            for (status, msg), cells in outcomes.items():
                where = "" if len(outcomes) == 1 else f" [{' '.join(cells)}]"
                print(f"  {tag}{where} {status} {msg}")


def storage(client, name: str) -> tuple[int, int]:
    r = client.query(
        "SELECT sum(column_data_compressed_bytes), sum(column_data_uncompressed_bytes) "
        "FROM system.parts_columns WHERE database = 'bench' AND table = {t:String} "
        "AND column = 'v' AND active",
        parameters={"t": name.split(".")[1]},
    ).result_rows[0]
    return int(r[0]), int(r[1])


def gcd_measure(client) -> list[tuple]:
    out = []
    n = 1_000_000
    inputs = {
        "x*1000 (multiples of 1000, < 1e12)": "toInt64(intHash64(number) % 1000000000) * 1000",
        "x (the same / 1000)": "toInt64(intHash64(number) % 1000000000)",
        "x*1000 sorted": "toInt64(number) * 1000",
        "x sorted": "toInt64(number)",
    }
    for label, expr in inputs.items():
        for codec in ("LZ4", "GCD, LZ4", "ZSTD(3)", "GCD, ZSTD(3)"):
            name = "bench.vocab_gcd"
            client.command(f"DROP TABLE IF EXISTS {name}")
            client.command(
                f"CREATE TABLE {name} (row_id UInt64, v Int64 CODEC({codec})) "
                f"ENGINE = MergeTree ORDER BY tuple() {WIDE}"
            )
            client.command(
                f"INSERT INTO {name} SELECT number, {expr} FROM numbers({n})"
            )
            client.command(f"OPTIMIZE TABLE {name} FINAL")
            comp, uncomp = storage(client, name)
            out.append((label, codec, comp, uncomp))
            client.command(f"DROP TABLE IF EXISTS {name}")
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    client = clickhouse_connect.get_client(**CH)
    client.command("CREATE DATABASE IF NOT EXISTS bench")
    print("version:", client.command("SELECT version()"))
    rows = matrices(client)
    print_matrix(rows)

    out = Path(__file__).with_name("spike_ch_vocabulary_results.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "type\tnullable\tcodec\tcompression\tskip_index\tstatus\tstage\tmessage\n"
        )
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    print(f"\nwrote {len(rows)} cells to {out}")

    print("\n=== GCD codec on disk (Int64, 1 000 000 rows, OPTIMIZE FINAL)")
    print(f"{'input':<40} {'codec':<14} {'compressed':>12} {'uncompressed':>13}")
    for label, codec, comp, uncomp in gcd_measure(client):
        print(f"{label:<40} {codec:<14} {comp:>12} {uncomp:>13}")


if __name__ == "__main__":
    main()
