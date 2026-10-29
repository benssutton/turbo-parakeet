"""to_chbench, then each suite extended for the loop: a size suite per type (the codec x
compression grid plus the recommender's candidates) and a skip suite (the chosen and rejected skip
candidates plus none, on the recommended chain). Writes predicted.json per suite, suites.json (the
index the later steps read) and skipped.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import yaml
from chbench.harness.suite import load_suite
from turbo_parakeet import _plugin
from turbo_parakeet.recommenders import to_chbench

from evaluate.registry import sanitise

TRANSFORMS = ("GCD", "Delta", "DoubleDelta")
GRID_CODECS = ("none", "GCD", "Delta", "DoubleDelta")
GRID_COMPRESSIONS = ("LZ4", "ZSTD(1)", "ZSTD(3)")
SIZE_FRACTIONS = (0.001, 0.01, 0.05)
SKIP_FRACTIONS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def skip_value(c: dict) -> str:
    """A `clickhouse_skip_indexes` candidate in chbench's `<index>[:<G>]` grammar."""
    kind = c["kind"]
    if kind == "set":
        kind = f"set({c['max_size']})"
    elif kind == "bloom_filter":
        kind = f"bloom_filter({c['fpr']:g})"
    return f"{kind}:{c['granularity']}"


def rec_codec(rec_type: str, chain: str) -> str:
    """The codec-axis value of the recommendation: LowCardinality, the transform or none."""
    if rec_type.startswith("LowCardinality("):
        return "LowCardinality"
    head = chain.split(", ")[0]
    return head if head in TRANSFORMS else "none"


def _cell(chain: str, plain: str) -> tuple[str, str]:
    """(codec-axis value, compression) of a codec chain; `plain` names the no-transform codec."""
    parts = chain.split(", ")
    return (parts[0], parts[1]) if parts[0] in TRANSFORMS else (plain, parts[0])


def _queries(fractions, samples: int) -> dict:
    return {
        "kinds": [{"kind": "full_scan"}]
        + [
            {"kind": "filter", "target_fraction": f, "predicate": "in", "values": 1}
            for f in fractions
        ],
        "samples": samples,
    }


def _emulated(table: pa.Table, col: str, codec: str, comp: str) -> int | None:
    if codec == "LowCardinality":
        return None
    chain = comp if codec == "none" else f"{codec}, {comp}"
    try:
        return _plugin.clickhouse_profile(table, col, chain)["bytes"]
    except ValueError:
        return None


def _size_suite(base_suite, table, col, row, codec_value, original, samples):
    """(suite dict, predicted dict) of one size suite."""
    cands = row["clickhouse_candidates"] or []
    comp_rec = row["clickhouse_codecs"].split(", ")[-1]
    by_cell: dict[tuple[str, str], dict] = {}
    if not original:
        plain = "LowCardinality" if codec_value == "LowCardinality" else "none"
        for c in cands:
            # a pruned transform with no compression: no cell
            if c["codecs"] in TRANSFORMS:
                continue
            by_cell[_cell(c["codecs"], plain)] = c
    if original:
        # holds every transform; no LowCardinality cell (no prediction)
        codecs = list(GRID_CODECS)
        comps = list(dict.fromkeys([*GRID_COMPRESSIONS, comp_rec]))
    else:
        codecs = list(
            dict.fromkeys([codec_value, *(k for k, _ in by_cell), *GRID_CODECS])
        )
        comps = list(dict.fromkeys([*(c for _, c in by_cell), *GRID_COMPRESSIONS]))
    suite = dict(base_suite)
    ch = dict(suite["clickhouse"])
    ch.update(codec=codecs, compression=comps, skip_index=["none"])
    suite["clickhouse"] = ch
    suite["queries"] = _queries(SIZE_FRACTIONS, samples)
    base = ch["type"][0]
    predicted = {}
    for k in codecs:
        for c in comps:
            cand = by_cell.get((k, c))
            if cand is not None:
                # a pruned candidate has no bytes: the emulator fills the cell
                nbytes = cand["predicted_bytes"]
                if nbytes is None:
                    nbytes = _emulated(table, col, k, c)
                predicted[f"{base}|{k}|{c}"] = {
                    "bytes": nbytes,
                    "origin": "candidate",
                    "outcome": cand["outcome"],
                }
            else:
                predicted[f"{base}|{k}|{c}"] = {
                    "bytes": _emulated(table, col, k, c),
                    "origin": "grid",
                    "outcome": "grid",
                }
    return suite, predicted


def _skip_suite(base_suite, row, codec_value, samples):
    cands = row["clickhouse_skip_indexes"] or []
    chosen = [c for c in cands if c["outcome"] == "chosen"]
    if not chosen:
        return None, None
    rejected = [c for c in cands if c["outcome"] == "rejected"]
    values = list(
        dict.fromkeys(
            [skip_value(chosen[0]), *(skip_value(c) for c in rejected), "none"]
        )
    )
    predicted = {
        skip_value(c): {
            "outcome": c["outcome"],
            "index_bytes": c["bytes"],
            "skip_rows": c["skip_rows"],
        }
        for c in [
            *rejected,
            chosen[0],
        ]  # chosen last: it wins a value shared with a rejected one
    }
    predicted["none"] = {"outcome": "none", "index_bytes": None, "skip_rows": 0.0}
    suite = dict(base_suite)
    ch = dict(suite["clickhouse"])
    ch.update(
        codec=[codec_value],
        compression=[row["clickhouse_codecs"].split(", ")[-1]],
        skip_index=values,
    )
    suite["clickhouse"] = ch
    suite["queries"] = _queries(SKIP_FRACTIONS, samples)
    return suite, predicted


def build(
    dataset: str, result: pl.DataFrame, frame, out: Path, *, samples: int
) -> list[dict]:
    """Write the loop's suites for `result` (computed from `frame`) into `out`; return the index
    (also written to `out/suites.json`)."""
    out = Path(out)
    handoff = out / "handoff"
    shutil.rmtree(handoff, ignore_errors=True)
    skipped = to_chbench(result, frame, handoff)
    (out / "skipped.json").write_text(
        json.dumps([{"column": c, "reason": r} for c, r in skipped], indent=2),
        encoding="utf-8",
    )
    rows = {r["column"]: r for r in result.iter_rows(named=True)}
    prefix = sanitise(dataset)
    entries: list[dict] = []
    dropped: list[dict] = []
    for yp in sorted(handoff.glob("*.yaml")):
        base_suite = yaml.safe_load(yp.read_text(encoding="utf-8"))
        col = base_suite["dataset"]["column"]
        row = rows[col]
        original = yp.stem.endswith("_original")
        stem = yp.stem.removesuffix("_original")
        arrow = f"{prefix}__{yp.stem}.arrow"
        shutil.copyfile(handoff / f"{yp.stem}.arrow", out / arrow)
        with pa.ipc.open_file(out / arrow) as r:
            table = r.read_all()
        codec_value = rec_codec(row["clickhouse_type"], row["clickhouse_codecs"])
        base_suite["dataset"] = {**base_suite["dataset"], "path": arrow}
        made = [
            (
                "size",
                f"{prefix}__{yp.stem}",
                *_size_suite(
                    base_suite, table, col, row, codec_value, original, samples
                ),
            )
        ]
        if not original:
            made.append(
                (
                    "skip",
                    f"{prefix}__{stem}__skip",
                    *_skip_suite(base_suite, row, codec_value, samples),
                )
            )
        for role, name, suite, predicted in made:
            if suite is None:
                continue
            suite["name"] = name
            (out / f"{name}.yaml").write_text(
                yaml.safe_dump(suite, sort_keys=False), encoding="utf-8"
            )
            (out / f"{name}.predicted.json").write_text(
                json.dumps(predicted, indent=2), encoding="utf-8"
            )
            try:
                load_suite(out / f"{name}.yaml")
            except Exception as e:  # chbench cannot express the column
                for f in (f"{name}.yaml", f"{name}.predicted.json"):
                    (out / f).unlink()
                dropped.append({"column": col, "reason": f"{name}: {e}"})
                continue
            entries.append(
                {
                    "dataset": dataset,
                    "column": col,
                    "suite": name,
                    "role": role,
                    "type_role": "original" if original else "recommended",
                    "base": suite["clickhouse"]["type"][0],
                    "yaml": f"{name}.yaml",
                    "arrow": arrow,
                    "predicted": f"{name}.predicted.json",
                    "rec_codec": codec_value,
                    "rec_compression": row["clickhouse_codecs"].split(", ")[-1],
                }
            )
    shutil.rmtree(handoff, ignore_errors=True)
    if dropped:
        for a in {a for a in (out.glob("*.arrow"))} - {
            out / e["arrow"] for e in entries
        }:
            a.unlink()
        (out / "skipped.json").write_text(
            json.dumps(
                [{"column": c, "reason": r} for c, r in skipped] + dropped, indent=2
            ),
            encoding="utf-8",
        )
    (out / "suites.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries
