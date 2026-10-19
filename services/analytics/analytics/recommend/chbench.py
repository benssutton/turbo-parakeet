"""`to_chbench`: a recommender result as runnable suites for the sibling benchmark project
`clickhouse_benchmark` (`chbench`) — spec 2026-10-19 §7. A helper, not a recommender."""

import json
import re
from pathlib import Path

import polars as pl
import yaml

from analytics import _plugin

_NESTED = ("Array(", "Tuple(", "Map(", "Nested(")
_RANGE_TYPES = re.compile(
    r"^(U?Int\d+|Float\d+|Decimal\d*|Date|Date32|DateTime|DateTime64)$"
)


def _base(ch_type: str) -> str:
    """`Nullable(...)` / `LowCardinality(...)` stripped, parameters dropped."""
    t = ch_type
    while t.startswith(("Nullable(", "LowCardinality(")) and t.endswith(")"):
        t = t[t.index("(") + 1 : -1]
    return t.split("(", 1)[0]


def _scale(ch_type: str) -> int | None:
    m = re.search(r"Decimal\d*\((?:\d+,\s*)?(\d+)\)", ch_type)
    return int(m.group(1)) if m else None


def _sanitise(name: str, taken: set[str]) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", name.lower()) or "c"
    if s[0].isdigit():
        s = "c_" + s
    out, i = s, 2
    while out in taken:
        out, i = f"{s}_{i}", i + 1
    taken.add(out)
    return out


def _plain(arr):
    """Lossless normalisation to the types `chbench` parses."""
    import pyarrow as pa

    t = arr.type
    if pa.types.is_dictionary(t):
        return _plain(arr.dictionary_decode())
    if t in (pa.string_view(), pa.large_string()):
        return arr.cast(pa.string())
    if t in (pa.binary_view(), pa.large_binary()):
        return arr.cast(pa.binary())
    return arr


def _table(frame):
    import pyarrow as pa

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    return pa.table(frame)


def _compression(codecs: str) -> str:
    return codecs.split(", ")[-1]


def to_chbench(
    result: pl.DataFrame,
    frame,
    out_dir,
    *,
    columns: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Write `<name>.arrow`, `<name>.yaml` and `<name>.predicted.json` into `out_dir` for each
    computed column of `result` (or of `columns`) whose recommended and original ClickHouse
    types are expressible as `chbench` base types; return the skipped columns as
    (column, reason). `frame` is the frame the result was computed from. Spec §7."""
    import pyarrow as pa

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = _table(frame)
    rows = {r["column"]: r for r in result.iter_rows(named=True)}
    wanted = columns if columns is not None else list(rows)
    skipped: list[tuple[str, str]] = []
    taken: set[str] = set()
    for col in wanted:
        r = rows.get(col)
        if r is None or col not in table.column_names:
            skipped.append((col, "not in the result" if r is None else "not in frame"))
            continue
        if r["status"] != "computed":
            skipped.append((col, f"status {r['status']}"))
            continue
        rec_type = r["clickhouse_type"]
        if rec_type is None:
            skipped.append((col, "no ClickHouse type"))
            continue
        if any(n in rec_type for n in _NESTED):
            skipped.append((col, "nested type"))
            continue
        arr = _plain(table[col].combine_chunks())
        one = pa.table({col: arr})
        orig = _plugin.clickhouse_profile(one, col, "LZ4")
        if orig["clickhouse_type"] is None or any(
            n in orig["clickhouse_type"] for n in _NESTED
        ):
            skipped.append((col, "original type has no chbench base type"))
            continue
        name = _sanitise(col, taken)
        _write(out, name, col, one, r, orig)
    return skipped


def _write(out: Path, name: str, col: str, one, r: dict, orig: dict) -> None:
    import pyarrow as pa

    rec_type = r["clickhouse_type"]
    rec_base, orig_base = _base(rec_type), _base(orig["clickhouse_type"])
    chosen = r["clickhouse_codecs"]
    comp = _compression(chosen)
    rec_codec = (
        "LowCardinality"
        if rec_type.startswith("LowCardinality(")
        else "GCD" if chosen.startswith("GCD") else "none"
    )
    types = list(dict.fromkeys([rec_base, orig_base]))
    codecs = list(dict.fromkeys([rec_codec, "none"]))
    comps = list(dict.fromkeys([comp, "LZ4"]))

    candidates = {
        c["codecs"]: c["predicted_bytes"] for c in r["clickhouse_candidates"] or []
    }
    profile: dict[str, int | None] = {}

    def original(codec: str, c: str) -> int | None:
        chain = f"GCD, {c}" if codec == "GCD" else c
        if chain not in profile:
            profile[chain] = _plugin.clickhouse_profile(one, col, chain)["bytes"]
        return profile[chain]

    predicted: dict[str, int | None] = {}
    for t in types:
        for k in codecs:
            for c in comps:
                if t == rec_base and not (
                    rec_codec == "LowCardinality" and k == "none"
                ):
                    chain = f"GCD, {c}" if k == "GCD" else c
                    value = candidates.get(chain)
                elif k == "LowCardinality":
                    value = None
                else:
                    value = original(k, c)
                predicted[f"{t}|{k}|{c}"] = value

    dataset = {"path": f"{name}.arrow", "column": col}
    scale = _scale(rec_type)
    if scale is None:
        scale = _scale(orig["clickhouse_type"])
    if any(t.startswith("Decimal") for t in types) and scale is not None:
        dataset["decimal_places"] = scale
    kinds = [
        {"kind": "full_scan"},
        {"kind": "filter", "target_fraction": 0.05, "predicate": "in", "values": 1},
    ]
    if all(_RANGE_TYPES.match(t) for t in types):
        kinds.append({"kind": "filter", "target_fraction": 0.10, "predicate": "range"})
    suite = {
        "name": name,
        "dataset": dataset,
        "clickhouse": {
            "type": types,
            "nullable": [bool(r["nullable"])],
            "primary_key": [False],
            "projection": [False],
            "codec": codecs,
            "skip_index": ["none"],
            "compression": comps,
        },
        "queries": {
            "kinds": kinds,
            "query_condition_cache": [False],
            "cache_mode": ["warm"],
            "samples": 10,
        },
        "results": {"percentiles": [50, 90, 99]},
    }
    (out / f"{name}.yaml").write_text(
        yaml.safe_dump(suite, sort_keys=False), encoding="utf-8"
    )
    (out / f"{name}.predicted.json").write_text(
        json.dumps(predicted, indent=2), encoding="utf-8"
    )
    with pa.ipc.new_file(
        out / f"{name}.arrow",
        one.schema,
        options=pa.ipc.IpcWriteOptions(compression=None),
    ) as w:
        w.write_table(one.combine_chunks(), max_chunksize=None)
