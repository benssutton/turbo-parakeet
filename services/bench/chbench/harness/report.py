"""Turn `Store.query_rows` output into the report spec section 6.2 describes:
one summary row per query with sample count, mean, max and requested
percentiles of `client_ms`/`server_ms`, table compressed bytes and
compression ratio, and achieved selectivity -- rendered with `rich`, sorted
by variant axes so codec or type comparisons sit adjacent, or exported to
CSV. `chbench report` re-queries SQLite without running anything, so this
module never touches ClickHouse or the generator; it consumes the flat dict
rows `Store.query_rows` returns and nothing else.

`Store.query_rows` returns every column of the `tables`/`queries`/`samples`
join (see its `_QUERY_ROWS_SQL`), which is wider than what a summary needs --
`status`, `skip_reason`, `table_warnings`, `marks_bytes`, `explain_plan`,
`profile_events_json` and the rest are of interest to a future dump/debug
view, not this one. `summarise` reads only the keys it needs and otherwise
ignores a row's shape, so it tolerates that wider row without change.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rich.table import Table

# The variant axes, in sort order. Table axes first (so two rows that differ
# only in one table axis -- e.g. codec -- sit adjacent), then the query axes,
# narrowest-scope last (`target_fraction` distinguishes rows that already
# share every coarser axis). This is also the column order `render_table`
# and `write_csv` use.
_TABLE_AXES = (
    "ch_type",
    "nullable",
    "primary_key",
    "projection",
    "codec",
    "skip_index",
    "compression",
)
_QUERY_AXES = ("kind", "cache_mode", "qcc", "target_fraction")
_AXES = _TABLE_AXES + _QUERY_AXES

# Row keys rendered as a tick/blank rather than a bare 0/1 integer, per task
# brief point 2: SQLite gives these back as ints, and a column of 0s and 1s
# next to numeric columns reads as data, not a flag.
_BOOL_AXES = ("nullable", "primary_key", "projection", "qcc")


def _sort_key(s: dict):
    """`(v is None, v)` per axis, not a bare `v`, so a `None` sorts
    consistently (at the low end) instead of ever being compared against a
    non-`None` value of another summary's tuple at the same position -- that
    comparison raises `TypeError` in Python 3.

    `target_fraction` is `None` for `full_scan` queries and a `float` for
    `filter` queries, and nothing stops a suite from having two `full_scan`
    query blocks that agree on every other axis (ruling R20: this is fixed
    here, generically, rather than by having `suite.py` forbid the
    combination -- the report reads back from SQLite, where any column can
    be `NULL` for reasons unrelated to suite validation, e.g. a skipped
    table or a partial run)."""
    return tuple((v is None, v) for v in (s[a] for a in _AXES))


def summarise(rows: list[dict], percentiles: list[int]) -> list[dict]:
    """Group `rows` (as `Store.query_rows` returns them) by `query_id` and
    reduce each group to one summary dict, sorted by the variant axes.

    Every row sharing a `query_id` shares its table and query axes by
    construction (one query is run against one table); the variant axes are
    taken from the group's first row, and only `client_ms`/`server_ms`
    across the group's samples are aggregated. `compressed_bytes`,
    `uncompressed_bytes`, `compression_ratio` and `actual_fraction` are also
    per-table/per-query, so they come from the first row too.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["query_id"], []).append(row)

    summaries = [_summarise_group(group, percentiles) for group in groups.values()]
    summaries.sort(key=_sort_key)
    return summaries


def _summarise_group(group: list[dict], percentiles: list[int]) -> dict:
    first = group[0]
    client = np.array([r["client_ms"] for r in group], dtype=float)
    server = np.array([r["server_ms"] for r in group], dtype=float)

    summary = {"query_id": first["query_id"], "table_id": first["table_id"]}
    for axis in _AXES:
        summary[axis] = first[axis]

    summary["n"] = len(group)
    summary["client_mean"] = float(np.mean(client))
    summary["client_max"] = float(np.max(client))
    summary["server_mean"] = float(np.mean(server))
    summary["server_max"] = float(np.max(server))
    for p in percentiles:
        summary[f"client_p{p}"] = float(np.percentile(client, p))
        summary[f"server_p{p}"] = float(np.percentile(server, p))

    summary["compressed_bytes"] = first["compressed_bytes"]
    summary["uncompressed_bytes"] = first["uncompressed_bytes"]
    summary["compression_ratio"] = _compression_ratio(
        first["compressed_bytes"], first["uncompressed_bytes"]
    )
    summary["index_compressed_bytes"] = first.get("index_compressed_bytes")
    summary["index_uncompressed_bytes"] = first.get("index_uncompressed_bytes")
    summary["actual_fraction"] = first["actual_fraction"]

    return summary


def _compression_ratio(compressed_bytes, uncompressed_bytes) -> float | None:
    """`uncompressed_bytes / compressed_bytes`. A skipped table has NULL
    storage stats (store.py's `test_skipped_table_is_stored_with_its_reason`),
    which SQLite gives back as `None`: `None` in, `None` out, never a
    `ZeroDivisionError` and never a silent zero standing in for "no data".
    `compressed_bytes == 0` is unreachable with real storage stats but is
    guarded the same way rather than raising `ZeroDivisionError`."""
    if compressed_bytes is None or uncompressed_bytes is None or compressed_bytes == 0:
        return None
    return uncompressed_bytes / compressed_bytes


def _column_order(percentiles: list[int]) -> list[str]:
    # `query_id` leads: two summaries can tie on every other rendered column
    # (e.g. differ only in `predicate_kind` or IN-list size, neither of
    # which is rendered) with nothing else to trace a row back to its
    # source query.
    return (
        ["query_id"]
        + list(_AXES)
        + ["n", "client_mean", "client_max"]
        + [f"client_p{p}" for p in percentiles]
        + ["server_mean", "server_max"]
        + [f"server_p{p}" for p in percentiles]
        + [
            "compressed_bytes",
            "compression_ratio",
            "index_compressed_bytes",
            "index_uncompressed_bytes",
            "actual_fraction",
        ]
    )


_HEADERS = {
    "query_id": "query id",
    "ch_type": "type",
    "nullable": "nullable",
    "primary_key": "primary key",
    "projection": "projection",
    "codec": "codec",
    "skip_index": "skip index",
    "compression": "compression",
    "kind": "kind",
    "cache_mode": "cache",
    "qcc": "qcc",
    "target_fraction": "target fraction",
    "n": "n",
    "client_mean": "mean client",
    "client_max": "max client",
    "server_mean": "mean server",
    "server_max": "max server",
    "compressed_bytes": "compressed bytes",
    "compression_ratio": "ratio",
    "index_compressed_bytes": "index compressed",
    "index_uncompressed_bytes": "index uncompressed",
    "actual_fraction": "selectivity",
}


def _header(key: str) -> str:
    if key.startswith("client_p"):
        return f"p{key[len('client_p'):]} client"
    if key.startswith("server_p"):
        return f"p{key[len('server_p'):]} server"
    return _HEADERS.get(key, key)


def _cell(summary: dict, key: str) -> str:
    value = summary[key]
    if key in _BOOL_AXES:
        return "y" if value else "-"
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(summary: list[dict], percentiles: list[int]) -> Table:
    """Build (but do not print -- that is the CLI's job) a `rich.table.Table`
    with one column per field in `_column_order` and one row per summary
    dict, already sorted by `summarise`."""
    table = Table()
    columns = _column_order(percentiles)
    for key in columns:
        table.add_column(_header(key))
    for row in summary:
        table.add_row(*[_cell(row, key) for key in columns])
    return table


def write_csv(summary: list[dict], path: str | Path) -> None:
    """One row per summary dict, header line included, column order matching
    `render_table`'s (`_column_order`)."""
    percentiles = sorted(
        {
            int(key[len("client_p") :])
            for row in summary
            for key in row
            if key.startswith("client_p")
        }
    )
    columns = _column_order(percentiles)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in columns})


def filter_where(rows: list[dict], where: str | None) -> list[dict]:
    """Filter `rows` (as `Store.query_rows` returns them, or `summarise`'s
    output -- both are flat dicts) to those matching every `axis=value` pair
    in `where`, comma-joined (e.g. `"codec=Delta,cache_mode=warm"`).
    `where=None` returns `rows` unchanged.

    Comparison rule: `where` always arrives as a string (it comes from the
    CLI), while row values are the row's own type (`int`, `float`, `str`,
    `None`). A row matches when `str(row[axis]) == value` -- comparing the
    row value's string form to the filter string -- so `nullable=1`,
    `qcc=0` and `target_fraction=0.1` all work without the caller needing to
    know or convert each axis's stored type.

    An axis name that is not a key of the rows is almost always a typo, and
    a silently-empty filter would waste a user's whole run reading it, so
    this raises `ValueError` naming the unknown axis and the valid ones
    (the first row's keys) rather than matching nothing.
    """
    if where is None:
        return rows
    if not rows:
        return rows

    valid = set(rows[0])
    pairs = []
    for clause in where.split(","):
        axis, _, value = clause.partition("=")
        if axis not in valid:
            raise ValueError(
                f"unknown filter axis {axis!r}; valid axes are {sorted(valid)!r}"
            )
        pairs.append((axis, value))

    return [
        row for row in rows if all(str(row[axis]) == value for axis, value in pairs)
    ]
