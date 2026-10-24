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

from turbo_parakeet import _plugin

_TRANSFORMS = ("GCD", "Delta", "DoubleDelta")
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


def _row_skip(r: dict | None, in_frame: bool) -> str | None:
    """Why a result row cannot become a suite; None when it can."""
    if r is None:
        return "not in the result"
    if not in_frame:
        return "not in frame"
    if r["status"] != "computed":
        return f"status {r['status']}"
    rec_type = r["clickhouse_type"]
    if rec_type is None:
        return "no ClickHouse type"
    if any(n in rec_type for n in _NESTED):
        return "nested type"
    if r["clickhouse_codecs"] is None:
        return "no codec chosen (no sample)"
    return None


def _no_base_type(orig: dict) -> bool:
    t = orig["clickhouse_type"]
    return t is None or any(n in t for n in _NESTED)


def _recommended_table(col: str, arr, datafusion_type: str):
    """(the dataset cast to the recommended Arrow type or None when it cannot be, whether the
    type equals the original's, the failure text or None)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    same = True
    try:
        target = _arrow_type(datafusion_type)
        if target == arr.type:
            return pa.table({col: arr}), True, None
        same = False
        cast = pc.cast(arr, target, safe=True)
        if cast.null_count != arr.null_count:
            raise ValueError("the cast changed the null count")
        return pa.table({col: cast}), False, None
    except (pa.ArrowException, ValueError, NotImplementedError) as e:
        return None, same, str(e)


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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = _table(frame)
    rows = {r["column"]: r for r in result.iter_rows(named=True)}
    wanted = columns if columns is not None else list(rows)
    skipped: list[tuple[str, str]] = []
    taken: set[str] = set()
    for col in wanted:
        r = rows.get(col)
        reason = _row_skip(r, col in table.column_names)
        if reason:
            skipped.append((col, reason))
            continue
        arr = _plain(table[col].combine_chunks())
        orig_one = pa.table({col: arr})
        orig = _plugin.clickhouse_profile(orig_one, col, "LZ4")
        if _no_base_type(orig):
            skipped.append((col, "original type has no chbench base type"))
            continue
        name = _sanitise(col, taken)
        rec_one, same, error = _recommended_table(col, arr, r["datafusion_type"])
        if rec_one is None:
            skipped.append(
                (
                    col,
                    f"recommended type not reproducible by a lossless pyarrow cast: {error}",
                )
            )
        else:
            _write(out, name, col, rec_one, r, orig, original=False)
        if not same:
            _write(out, f"{name}_original", col, orig_one, r, orig, original=True)
    return skipped


def _recommended_codec(rec_type: str, chosen: str) -> str:
    """The codec-axis value of the recommendation: LowCardinality, the transform or none."""
    head = chosen.split(", ")[0]
    if rec_type.startswith("LowCardinality("):
        return "LowCardinality"
    if head in _TRANSFORMS:
        return head
    return "none"


class _Cells:
    """The predicted column bytes of the (codec, compression) cells of one suite."""

    def __init__(self, one, col: str, r: dict, rec_codec: str, original: bool):
        self.one, self.col, self.rec_codec, self.original = (
            one,
            col,
            rec_codec,
            original,
        )
        self.candidates = {
            c["codecs"]: c["predicted_bytes"] for c in r["clickhouse_candidates"] or []
        }
        self.profile: dict[str, int | None] = {}

    def measured(self, chain: str) -> int | None:
        if chain not in self.profile:
            self.profile[chain] = _plugin.clickhouse_profile(self.one, self.col, chain)[
                "bytes"
            ]
        return self.profile[chain]

    def codecs(self, comp: str) -> list[str]:
        if not self.original:
            return list(dict.fromkeys([self.rec_codec, "none"]))
        rec = self.rec_codec
        has = rec in _TRANSFORMS and self.measured(f"{rec}, {comp}") is not None
        return ["none", rec] if has else ["none"]

    def predict(self, k: str, c: str) -> int | None:
        chain = f"{k}, {c}" if k in _TRANSFORMS else c
        if self.original or (self.rec_codec == "LowCardinality" and k == "none"):
            return None if k == "LowCardinality" else self.measured(chain)
        return self.candidates.get(chain)


def _predicted(
    base: str, cells: _Cells, codecs: list[str], comps: list[str], skip, r: dict
) -> dict:
    """`predicted.json`: column bytes per cell, plus the chosen index's cells with its bytes."""
    predicted = {f"{base}|{k}|{c}": cells.predict(k, c) for k in codecs for c in comps}
    if skip:
        index_bytes = _index_bytes(r)
        for k in codecs:
            for c in comps:
                predicted[f"{base}|{k}|{c}|{skip}"] = {
                    "column_bytes": cells.predict(k, c),
                    "index_bytes": index_bytes,
                }
    return predicted


def _queries(base: str) -> dict:
    kinds = [{"kind": "full_scan"}] + [
        {"kind": "filter", "target_fraction": f, "predicate": "in", "values": 1}
        for f in (0.001, 0.01, 0.05)
    ]
    if _RANGE_TYPES.match(base):
        kinds.append({"kind": "filter", "target_fraction": 0.10, "predicate": "range"})
    return {
        "kinds": kinds,
        "query_condition_cache": [False],
        "cache_mode": ["warm"],
        "samples": 10,
    }


def _write_arrow(path: Path, one) -> None:
    import pyarrow as pa

    with pa.ipc.new_file(
        path, one.schema, options=pa.ipc.IpcWriteOptions(compression=None)
    ) as w:
        w.write_table(one.combine_chunks(), max_chunksize=None)


def _write(
    out: Path, name: str, col: str, one, r: dict, orig: dict, *, original: bool
) -> None:
    """One suite over the dataset `one`: the recommended type (predictions from the result's
    candidates) or, with `original`, the original type (predictions from the emulator).
    """
    import pyarrow as pa
    import yaml

    chosen = r["clickhouse_codecs"]
    base = _base(orig["clickhouse_type"] if original else r["clickhouse_type"])
    comp = _compression(chosen)
    comps = list(dict.fromkeys([comp, "LZ4"]))
    rec_codec = _recommended_codec(r["clickhouse_type"], chosen)
    cells = _Cells(one, col, r, rec_codec, original)
    codecs = cells.codecs(comp)
    skip = None if original else _skip_axis(r.get("clickhouse_skip_index"))
    predicted = _predicted(base, cells, codecs, comps, skip, r)

    dataset = {"path": f"{name}.arrow", "column": col}
    t = one.schema.field(col).type
    if base.startswith("Decimal") and pa.types.is_decimal(t):
        dataset["decimal_places"] = t.scale
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
        "queries": _queries(base),
        "results": {"percentiles": [50, 90, 99]},
    }
    (out / f"{name}.yaml").write_text(
        yaml.safe_dump(suite, sort_keys=False), encoding="utf-8"
    )
    (out / f"{name}.predicted.json").write_text(
        json.dumps(predicted, indent=2), encoding="utf-8"
    )
    _write_arrow(out / f"{name}.arrow", one)
