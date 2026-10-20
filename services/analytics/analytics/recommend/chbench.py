"""`to_chbench`: a recommender result as runnable suites for the sibling benchmark project
`clickhouse_benchmark` (`chbench`) — spec 2026-10-19 §7, skip indexes 2026-10-20 §8.2. A helper,
not a recommender.

Skip index: when the result chose one (`clickhouse_skip_index`, e.g. `"set(125) GRANULARITY 4"`),
the recommended suite's `skip_index` axis is `[<chosen in chbench's grammar>, "none"]`, e.g.
`["set(125):4", "none"]` (chbench ruling R38: `<index>[:<G>]`); the original suite keeps `["none"]`.
Queries: full_scan, point `in` filters (`values: 1`) at target fractions 0.001, 0.01 and 0.05,
and a range filter on range-capable types.

`<name>.predicted.json` maps `"<type>|<codec>|<compression>"` to the predicted column bytes, as
before (also the `skip = none` cells, so the calibration script is unchanged), and for the cells of
the chosen index `"<type>|<codec>|<compression>|<skip>"` to
`{"column_bytes": …, "index_bytes": …}` (the chosen candidate's predicted index size).
"""

import json
import re
from pathlib import Path

import polars as pl

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


def _sanitise(name: str, taken: set[str]) -> str:
    """A file-name stem unique among `taken`; the stem and `<stem>_original` are reserved."""
    s = re.sub(r"[^a-z0-9_]", "_", name.lower()) or "c"
    if s[0].isdigit():
        s = "c_" + s
    out, i = s, 2
    while out in taken or f"{out}_original" in taken:
        out, i = f"{s}_{i}", i + 1
    taken.update((out, f"{out}_original"))
    return out


def _normal(t):
    """The type `_plain` leaves a column of type `t` in."""
    import pyarrow as pa

    if t in (pa.string_view(), pa.large_string()):
        return pa.string()
    if t in (pa.binary_view(), pa.large_binary()):
        return pa.binary()
    return t


def _plain(arr):
    """Lossless normalisation to the types `chbench` parses."""
    import pyarrow as pa

    if pa.types.is_dictionary(arr.type):
        return _plain(arr.dictionary_decode())
    t = _normal(arr.type)
    return arr if t == arr.type else arr.cast(t)


def _arrow_type(name: str):
    """The pyarrow type for a recommendation's `datafusion_type` text. A dictionary stands for
    its plain values (its LowCardinality is a codec-axis value). ValueError when unparseable.
    """
    import pyarrow as pa

    if name.startswith("dictionary<"):
        values = re.match(r"dictionary<values=(\w+),", name)
        return _normal(pa.type_for_alias(values.group(1) if values else "string"))
    if m := re.fullmatch(r"decimal(32|64|128)\((\d+), (\d+)\)", name):
        return getattr(pa, f"decimal{m.group(1)}")(int(m.group(2)), int(m.group(3)))
    if m := re.fullmatch(r"timestamp\[(\w+)(?:, tz=(.+))?\]", name):
        return pa.timestamp(m.group(1), m.group(2))
    if m := re.fullmatch(r"(time32|time64|duration)\[(\w+)\]", name):
        return getattr(pa, m.group(1))(m.group(2))
    if name in ("date32[day]", "date64[ms]"):
        return getattr(pa, name[:6])()
    return _normal(pa.type_for_alias(name))


def _table(frame):
    import pyarrow as pa

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    return pa.table(frame)


def _skip_axis(text: str | None) -> str | None:
    """`"set(125) GRANULARITY 4"` -> chbench's `"set(125):4"`; None when no index was chosen."""
    if not text:
        return None
    index, _, g = text.partition(" GRANULARITY ")
    return f"{index}:{g}" if g else index


def _index_bytes(r: dict) -> int | None:
    chosen = [
        c for c in r.get("clickhouse_skip_indexes") or [] if c["outcome"] == "chosen"
    ]
    return chosen[0]["bytes"] if chosen else None


def _compression(codecs: str) -> str:
    return codecs.split(", ")[-1]


def to_chbench(
    result: pl.DataFrame,
    frame,
    out_dir,
    *,
    columns: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Write suites into `out_dir` for each computed column of `result` (or of `columns`) whose
    recommended and original ClickHouse types are expressible as `chbench` base types; return
    the skipped columns as (column, reason). `frame` is the frame the result was computed from.

    chbench takes a type's parameters (Decimal scale, DateTime64 precision, ...) from the dataset
    file, so a column whose recommended Arrow type differs from the original gets two suites:
    `<name>` (the column cast to the recommended Arrow type) and `<name>_original` (the column
    as it is), each with its own `.arrow` and `.predicted.json`; otherwise one `<name>` suite.
    Spec §7."""
    import pyarrow as pa
    import pyarrow.compute as pc

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
        if r["clickhouse_codecs"] is None:
            skipped.append((col, "no codec chosen (no sample)"))
            continue
        arr = _plain(table[col].combine_chunks())
        orig_one = pa.table({col: arr})
        orig = _plugin.clickhouse_profile(orig_one, col, "LZ4")
        if orig["clickhouse_type"] is None or any(
            n in orig["clickhouse_type"] for n in _NESTED
        ):
            skipped.append((col, "original type has no chbench base type"))
            continue
        name = _sanitise(col, taken)
        rec_one, same = orig_one, True
        try:
            target = _arrow_type(r["datafusion_type"])
            if target != arr.type:
                same = False
                cast = pc.cast(arr, target, safe=True)
                if cast.null_count != arr.null_count:
                    raise ValueError("the cast changed the null count")
                rec_one = pa.table({col: cast})
        except (pa.ArrowException, ValueError, NotImplementedError) as e:
            skipped.append(
                (
                    col,
                    f"recommended type not reproducible by a lossless pyarrow cast: {e}",
                )
            )
            rec_one = None
        if rec_one is not None:
            _write(out, name, col, rec_one, r, orig, original=False)
        if not same:
            _write(out, f"{name}_original", col, orig_one, r, orig, original=True)
    return skipped


def _write(
    out: Path, name: str, col: str, one, r: dict, orig: dict, *, original: bool
) -> None:
    """One suite over the dataset `one`: the recommended type (predictions from the result's
    candidates) or, with `original`, the original type (predictions from the emulator).
    """
    import pyarrow as pa
    import yaml

    rec_type = r["clickhouse_type"]
    base = _base(orig["clickhouse_type"] if original else rec_type)
    chosen = r["clickhouse_codecs"]
    comp = _compression(chosen)
    rec_codec = (
        "LowCardinality"
        if rec_type.startswith("LowCardinality(")
        else "GCD" if chosen.startswith("GCD") else "none"
    )
    comps = list(dict.fromkeys([comp, "LZ4"]))
    profile: dict[str, int | None] = {}

    def measured(chain: str) -> int | None:
        if chain not in profile:
            profile[chain] = _plugin.clickhouse_profile(one, col, chain)["bytes"]
        return profile[chain]

    if original:
        gcd = rec_codec == "GCD" and measured(f"GCD, {comp}") is not None
        codecs = ["none", "GCD"] if gcd else ["none"]
    else:
        codecs = list(dict.fromkeys([rec_codec, "none"]))
    candidates = {
        c["codecs"]: c["predicted_bytes"] for c in r["clickhouse_candidates"] or []
    }

    def predict(k: str, c: str) -> int | None:
        chain = f"GCD, {c}" if k == "GCD" else c
        if original or (rec_codec == "LowCardinality" and k == "none"):
            return None if k == "LowCardinality" else measured(chain)
        return candidates.get(chain)

    predicted = {f"{base}|{k}|{c}": predict(k, c) for k in codecs for c in comps}
    skip = None if original else _skip_axis(r.get("clickhouse_skip_index"))
    if skip:
        index_bytes = _index_bytes(r)
        for k in codecs:
            for c in comps:
                predicted[f"{base}|{k}|{c}|{skip}"] = {
                    "column_bytes": predict(k, c),
                    "index_bytes": index_bytes,
                }

    dataset = {"path": f"{name}.arrow", "column": col}
    t = one.schema.field(col).type
    if base.startswith("Decimal") and pa.types.is_decimal(t):
        dataset["decimal_places"] = t.scale
    kinds = [{"kind": "full_scan"}] + [
        {"kind": "filter", "target_fraction": f, "predicate": "in", "values": 1}
        for f in (0.001, 0.01, 0.05)
    ]
    if _RANGE_TYPES.match(base):
        kinds.append({"kind": "filter", "target_fraction": 0.10, "predicate": "range"})
    suite = {
        "name": name,
        "dataset": dataset,
        "clickhouse": {
            "type": [base],
            "nullable": [bool(r["nullable"])],
            "primary_key": [False],
            "projection": [False],
            "codec": codecs,
            "skip_index": [skip, "none"] if skip else ["none"],
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
