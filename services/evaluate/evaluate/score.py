"""Predicted joined with measured: the six score files of a run (spec, Score files). Pure functions
over chbench store rows (Store.table_rows / query_rows) and the suites' predictions; `score_run`
reads a run directory and writes the files. Metrics only: no thresholds."""

from __future__ import annotations


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
