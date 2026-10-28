"""Predicted joined with measured: the six score files of a run (spec, Score files). Pure functions
over chbench store rows (Store.table_rows / query_rows) and the suites' predictions; `score_run`
reads a run directory and writes the files. Metrics only: no thresholds."""

from __future__ import annotations

import math

from evaluate.predict import MARK_ROWS, SkipPredictor, parse_skip, value_of


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
    for r in rows:
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
                "value_sql": (r["predicate_sql"] or "")[len("v IN (") : -1],
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
    order = list(by)  # the suite's axis order: chosen first
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
