"""Predicted joined with measured: the six score files of a run (spec, Score files). Pure functions
over chbench store rows (Store.table_rows / query_rows) and the suites' predictions; `score_run`
reads a run directory and writes the files. Metrics only: no thresholds."""

from __future__ import annotations

import json
import math
import re
import statistics

import polars as pl
import pyarrow as pa
from chbench.harness.store import Store

from evaluate.predict import MARK_ROWS, SkipPredictor, parse_skip, value_of
from evaluate.rundir import RunDir


def base_type(ch_type: str) -> str:
    """`Nullable(...)` / `LowCardinality(...)` stripped, parameters dropped."""
    t = ch_type
    while t.startswith(("Nullable(", "LowCardinality(")) and t.endswith(")"):
        t = t[t.index("(") + 1 : -1]
    return t.split("(", 1)[0]


def chain(codec: str, compression: str) -> str:
    return (
        compression
        if codec in ("none", "LowCardinality")
        else f"{codec}, {compression}"
    )


def _keys(entry: dict) -> dict:
    return {k: entry[k] for k in ("dataset", "column", "suite")}


def size_cells(entry: dict, tables: list[dict], predicted: dict) -> list[dict]:
    out = []
    for t in tables:
        base = base_type(t["ch_type"])
        p = predicted.get(f"{base}|{t['codec']}|{t['compression']}", {})
        pred = p.get("bytes")
        meas = t["compressed_bytes"] if t["status"] == "ok" else None
        out.append(
            {
                **_keys(entry),
                "type": base,
                "original_type": entry["type_role"] == "original",
                "codec": t["codec"],
                "compression": t["compression"],
                "origin": p.get("origin", "grid"),
                "outcome": p.get("outcome", "grid"),
                "predicted_bytes": pred,
                "measured_bytes": meas,
                "rel_error": (
                    (pred - meas) / meas if pred is not None and meas else None
                ),
                "uncompressed_bytes": t["uncompressed_bytes"],
                "insert_ms": t["insert_ms"],
                "skip_reason": t["skip_reason"],
            }
        )
    return out


def size_regret(entry: dict, cells: list[dict]) -> dict:
    measured = [c for c in cells if c["measured_bytes"] is not None]
    best = min(measured, key=lambda c: c["measured_bytes"], default=None)
    rec = next(
        (
            c
            for c in measured
            if (c["codec"], c["compression"])
            == (entry["rec_codec"], entry["rec_compression"])
        ),
        None,
    )
    regret = (
        None
        if rec is None or best is None
        else rec["measured_bytes"] - best["measured_bytes"]
    )
    return {
        **_keys(entry),
        "type_role": entry["type_role"],
        "recommended_chain": chain(entry["rec_codec"], entry["rec_compression"]),
        "recommended_measured": rec["measured_bytes"] if rec else None,
        "best_chain": chain(best["codec"], best["compression"]) if best else None,
        "best_measured": best["measured_bytes"] if best else None,
        "regret_bytes": regret,
        "regret_rel": (
            regret / best["measured_bytes"]
            if regret is not None and best["measured_bytes"]
            else None
        ),
        "rank": (
            None
            if rec is None
            else 1 + sum(c["measured_bytes"] < rec["measured_bytes"] for c in measured)
        ),
        "n_cells": len(measured),
        "type_gain_predicted": None,
        "type_gain_measured": None,
    }


def _best(cells: list[dict], key: str) -> int | None:
    vals = [c[key] for c in cells if c[key] is not None]
    return min(vals) if vals else None


def type_gain(
    rec: list[dict], orig: list[dict] | None
) -> tuple[float | None, float | None]:
    """(predicted, measured): best on the recommended type / best on the original - 1; (0, 0)
    when the recommended type is the original (no original suite)."""
    if orig is None:
        return 0.0, 0.0
    out = []
    for key in ("predicted_bytes", "measured_bytes"):
        r, o = _best(rec, key), _best(orig, key)
        out.append(r / o - 1 if r is not None and o else None)
    return out[0], out[1]


def skip_queries(
    entry: dict, rows: list[dict], predictor: SkipPredictor, predicted: dict
) -> tuple[list[dict], dict]:
    """One row per filter query of a skip suite; the notes counts (unparsed, marks_mismatch)."""
    notes = {"unparsed": 0, "marks_mismatch": 0}
    totals = {
        r["table_id"]: r["selected_marks"] for r in rows if r["kind"] == "full_scan"
    }
    out, seen = [], set()
    # table_id order (Store.table_rows' ORDER BY): the output order, hence skip_regret's ties, is fixed
    for r in sorted(rows, key=lambda r: r["table_id"]):
        if r["kind"] != "filter" or r["query_status"] != "ok" or r["query_id"] in seen:
            continue
        seen.add(r["query_id"])
        total = totals.get(r["table_id"])
        if not total:
            continue
        note = None
        if total != math.ceil(r["rows"] / MARK_ROWS):
            notes["marks_mismatch"] += 1
            note = "marks_mismatch"
        skip = r["skip_index"]
        m = re.fullmatch(r"v IN \((.*)\)", r["predicate_sql"] or "", re.DOTALL)
        v = value_of(r["predicate_sql"], predictor.dtype)
        if v is None:
            notes["unparsed"] += 1
            note = "unparsed"
        p = predicted.get(skip, {})
        pred = None if v is None else predictor.skip(skip, v)
        meas = 1 - r["selected_marks"] / total
        out.append(
            {
                **_keys(entry),
                "index": skip,
                "granularity": 0 if skip == "none" else parse_skip(skip)[2],
                "outcome": p.get("outcome", "rejected"),
                "value_sql": m.group(1) if m else r["predicate_sql"],
                "value_rows": None if v is None else predictor.rows_of(v),
                "total_marks": total,
                "predicted_skip": pred,
                "measured_skip": meas,
                "skip_error": None if pred is None else pred - meas,
                "predicted_index_bytes": p.get("index_bytes"),
                "measured_index_bytes": r["index_compressed_bytes"],
                "note": note,
            }
        )
    return out, notes


def skip_regret(entry: dict, queries: list[dict], predicted: dict) -> dict:
    by: dict[str, list[dict]] = {}
    for q in queries:
        by.setdefault(q["index"], []).append(q)

    def mean(qs: list[dict], weighted: bool) -> float | None:
        qs = [q for q in qs if not weighted or q["value_rows"]]
        if not qs:
            return None
        w = [q["value_rows"] if weighted else 1 for q in qs]
        return sum(q["measured_skip"] * x for q, x in zip(qs, w)) / sum(w)

    chosen = next((k for k, p in predicted.items() if p["outcome"] == "chosen"), None)
    order = sorted(
        by, key=lambda k: k != chosen
    )  # chosen first, then queries' (table_id) order
    out = {**_keys(entry), "chosen_index": chosen}
    for label, weighted in (("freq", True), ("uniform", False)):
        means = {k: mean(by[k], weighted) for k in order}
        ranked = [k for k in order if means[k] is not None]
        best = max(ranked, key=lambda k: means[k], default=None)
        c = means.get(chosen)
        out[f"best_index_{label}"] = best
        out[f"chosen_skip_{label}"] = c
        out[f"best_skip_{label}"] = means[best] if best else None
        out[f"regret_{label}"] = None if c is None or best is None else means[best] - c
    idx = {k: by[k][0]["measured_index_bytes"] for k in order}
    out["chosen_index_bytes"] = idx.get(chosen)
    out["best_index_bytes"] = idx.get(out["best_index_freq"])
    out["recommender_skip_rows"] = predicted.get(chosen, {}).get("skip_rows")
    out["measured_skip_rows"] = out["chosen_skip_freq"]
    return out


def _p50(values: list) -> float | None:
    vals = [v for v in values if v is not None and v == v]  # drop None / nan
    return statistics.median(vals) if vals else None


def _cpu_us(events_json: str | None) -> float | None:
    if not events_json:
        return None
    e = json.loads(events_json)
    if "UserTimeMicroseconds" not in e:
        return None
    return e["UserTimeMicroseconds"] + e.get("SystemTimeMicroseconds", 0)


def query_cost(entry: dict, rows: list[dict]) -> list[dict]:
    """One row per (table variant, query) with the samples' p50s; CPU and memory included."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["query_id"], []).append(r)
    out = []
    for samples in by.values():
        r = samples[0]
        out.append(
            {
                **_keys(entry),
                "type": base_type(r["ch_type"]),
                "codec": r["codec"],
                "compression": r["compression"],
                "skip_index": r["skip_index"],
                "query_kind": r["kind"],
                "target_fraction": r["target_fraction"],
                "samples": len(samples),
                **{
                    f"{k}_p50": _p50([s[k] for s in samples])
                    for k in (
                        "server_ms",
                        "client_ms",
                        "memory_bytes",
                        "read_rows",
                        "read_bytes",
                    )
                },
                "cpu_us_p50": _p50(
                    [_cpu_us(s.get("profile_events_json")) for s in samples]
                ),
            }
        )
    return out


_KEYS = [
    ("run_id", pa.string()),
    ("dataset", pa.string()),
    ("column", pa.string()),
    ("suite", pa.string()),
]
_S, _I, _F, _B = pa.string(), pa.int64(), pa.float64(), pa.bool_()
FILES: dict[str, list[tuple[str, pa.DataType]]] = {
    "size_cells": [
        ("type", _S),
        ("original_type", _B),
        ("codec", _S),
        ("compression", _S),
        ("origin", _S),
        ("outcome", _S),
        ("predicted_bytes", _I),
        ("measured_bytes", _I),
        ("rel_error", _F),
        ("uncompressed_bytes", _I),
        ("insert_ms", _F),
        ("skip_reason", _S),
    ],
    "size_regret": [
        ("type_role", _S),
        ("recommended_chain", _S),
        ("recommended_measured", _I),
        ("best_chain", _S),
        ("best_measured", _I),
        ("regret_bytes", _I),
        ("regret_rel", _F),
        ("rank", _I),
        ("n_cells", _I),
        ("type_gain_predicted", _F),
        ("type_gain_measured", _F),
    ],
    "skip_queries": [
        ("index", _S),
        ("granularity", _I),
        ("outcome", _S),
        ("value_sql", _S),
        ("value_rows", _I),
        ("total_marks", _I),
        ("predicted_skip", _F),
        ("measured_skip", _F),
        ("skip_error", _F),
        ("predicted_index_bytes", _I),
        ("measured_index_bytes", _I),
        ("note", _S),
    ],
    "skip_regret": [
        ("chosen_index", _S),
        ("best_index_freq", _S),
        ("best_index_uniform", _S),
        ("chosen_skip_freq", _F),
        ("best_skip_freq", _F),
        ("regret_freq", _F),
        ("chosen_skip_uniform", _F),
        ("best_skip_uniform", _F),
        ("regret_uniform", _F),
        ("chosen_index_bytes", _I),
        ("best_index_bytes", _I),
        ("recommender_skip_rows", _F),
        ("measured_skip_rows", _F),
    ],
    "query_cost": [
        ("type", _S),
        ("codec", _S),
        ("compression", _S),
        ("skip_index", _S),
        ("query_kind", _S),
        ("target_fraction", _F),
        ("samples", _I),
        ("server_ms_p50", _F),
        ("client_ms_p50", _F),
        ("memory_bytes_p50", _F),
        ("read_rows_p50", _F),
        ("read_bytes_p50", _F),
        ("cpu_us_p50", _F),
    ],
    "recommend_cost": [
        ("rows", _I),
        ("cols", _I),
        ("wall_s", _F),
        ("cpu_s", _F),
        ("peak_mb", _F),
        ("frame_mb", _F),
        ("status", _S),
        ("error", _S),
    ],
}
UNIFORM_NOTE = "skip 'uniform' weighting is uniform over the queried values, not every distinct value"


def write_scores(run: RunDir, rows: dict[str, list[dict]]) -> None:
    info = run.info()
    meta = {"run_id": run.run_id, "skip_uniform": UNIFORM_NOTE}
    meta.update({k: str(v) for k, v in info.get("versions", {}).items()})
    for name, fields in FILES.items():
        schema = pa.schema(_KEYS + fields, metadata=meta)
        data = [{**r, "run_id": run.run_id} for r in rows.get(name, [])]
        table = pa.Table.from_pylist(data, schema=schema)
        with pa.ipc.new_file(run.path / f"{name}.arrow", schema) as w:
            w.write_table(table)


def score_run(run: RunDir) -> dict:
    """Score every dataset of `run`; write the six files; return the notes counts."""
    rows: dict[str, list[dict]] = {n: [] for n in FILES}
    notes = {"unparsed": 0, "marks_mismatch": 0, "missing_stores": 0}
    for name in run.info().get("datasets", {}):
        cost_path = run.cost(name)
        c = (
            json.loads(cost_path.read_text(encoding="utf-8"))
            if cost_path.is_file()
            else {"status": "missing"}  # a row with its status, never silently dropped
        )
        rows["recommend_cost"].append(
            {
                "dataset": name,
                "column": None,
                "suite": None,
                **{k: c.get(k) for k, _ in FILES["recommend_cost"]},
            }
        )
        index = run.suites(name) / "suites.json"
        if not index.is_file():
            continue
        suites_dir = index.parent
        by_column: dict[str, dict[str, list[dict]]] = {}
        regrets: dict[str, dict] = {}
        for e in json.loads(index.read_text(encoding="utf-8")):
            db = run.stores(name) / f"{e['suite']}.db"
            if not db.is_file():
                notes["missing_stores"] += 1
                continue
            predicted = json.loads(
                (suites_dir / e["predicted"]).read_text(encoding="utf-8")
            )
            with Store(db) as s:
                tables, qrows = s.table_rows(e["suite"]), s.query_rows(e["suite"])
            rows["query_cost"] += query_cost(e, qrows)
            if e["role"] == "size":
                cells = size_cells(e, tables, predicted)
                rows["size_cells"] += cells
                by_column.setdefault(e["column"], {})[e["type_role"]] = cells
                r = size_regret(e, cells)
                rows["size_regret"].append(r)
                if e["type_role"] == "recommended":
                    regrets[e["column"]] = r
            else:
                col = pl.read_ipc(suites_dir / e["arrow"])[e["column"]]
                q, n = skip_queries(e, qrows, SkipPredictor(col), predicted)
                rows["skip_queries"] += q
                for k in ("unparsed", "marks_mismatch"):
                    notes[k] += n[k]
                rows["skip_regret"].append(skip_regret(e, q, predicted))
        for column, r in regrets.items():
            types = by_column[column]
            r["type_gain_predicted"], r["type_gain_measured"] = type_gain(
                types["recommended"], types.get("original")
            )
    write_scores(run, rows)
    return notes
