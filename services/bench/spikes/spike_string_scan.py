"""Spike 12: what does the String full scan read? (ToDo 3)

`SELECT count(), sum(length(v))` over a String column is rewritten to the size subcolumn by
`optimize_functions_to_subcolumns`, so it reads offsets only. Measures read_bytes and
ReadCompressedBytes for three forms of the scan on a 1M-row, 20-character String table:
the current form, the same with the rewrite disabled, and `sum(lengthUTF8(v))`.

    python -m spikes.spike_string_scan      (from services/bench/, chbench-ch running)
"""

from __future__ import annotations

import uuid

from chbench.harness.client import CHClient, Connection

TABLE = "bench.spike_string_scan"
FORMS = {
    "length": "SELECT count(), sum(length(v)) FROM {t}",
    "length_no_subcolumns": (
        "SELECT count(), sum(length(v)) FROM {t} SETTINGS optimize_functions_to_subcolumns = 0"
    ),
    "lengthUTF8": "SELECT count(), sum(lengthUTF8(v)) FROM {t}",
}


def main() -> None:
    c = CHClient(Connection.from_env())
    c.command(f"DROP TABLE IF EXISTS {TABLE}")
    c.command(
        f"CREATE TABLE {TABLE} (row_id UInt64, v String) ENGINE = MergeTree ORDER BY row_id "
        "SETTINGS min_bytes_for_wide_part = 0"
    )
    c.command(
        f"INSERT INTO {TABLE} SELECT number, randomPrintableASCII(20) FROM numbers(1000000)"
    )
    c.command(f"OPTIMIZE TABLE {TABLE} FINAL")
    ids = {}
    for name, sql in FORMS.items():
        for _ in range(2):  # the second run is the warm one we read
            ids[name] = uuid.uuid4().hex
            c.query_rows(
                sql.format(t=TABLE), query_id=ids[name], settings={"use_query_cache": 0}
            )
    c.flush_logs()
    print("form\tread_bytes\tReadCompressedBytes\tcpu_us")
    for name, qid in ids.items():
        [(rb, rcb, cpu)] = c.query_rows(
            "SELECT read_bytes, ProfileEvents['ReadCompressedBytes'], "
            "ProfileEvents['UserTimeMicroseconds'] + ProfileEvents['SystemTimeMicroseconds'] "
            "FROM system.query_log WHERE query_id = {q:String} AND type = 'QueryFinish'",
            parameters={"q": qid},
        )
        print(f"{name}\t{rb}\t{rcb}\t{cpu}")
    c.command(f"DROP TABLE {TABLE}")


if __name__ == "__main__":
    main()
