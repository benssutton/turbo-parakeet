import json

import numpy as np
import polars as pl
import yaml
from turbo_parakeet.recommenders import OneShotRecommender

from evaluate.suites import (
    GRID_CODECS,
    GRID_COMPRESSIONS,
    SIZE_FRACTIONS,
    SKIP_FRACTIONS,
    TRANSFORMS,
    build,
    skip_value,
)

N = 100_000


def _frame():
    return pl.DataFrame(
        {
            "ids": np.arange(
                N, dtype=np.int64
            ),  # UInt32 recommended, Delta, minmax index
            "s": np.random.default_rng(0).choice(["a", "b", "c"], N),  # LowCardinality
        }
    )


def _build(tmp_path, samples=3):
    frame = _frame()
    result = OneShotRecommender().add(frame).result()
    return frame, result, build("my.ds", result, frame, tmp_path, samples=samples)


def test_skip_value_grammar():
    assert (
        skip_value({"kind": "minmax", "granularity": 2, "max_size": None, "fpr": None})
        == "minmax:2"
    )
    assert (
        skip_value({"kind": "set", "granularity": 4, "max_size": 125, "fpr": None})
        == "set(125):4"
    )
    assert (
        skip_value(
            {"kind": "bloom_filter", "granularity": 1, "max_size": None, "fpr": 0.025}
        )
        == "bloom_filter(0.025):1"
    )


def test_suite_index(tmp_path):
    _, _, entries = _build(tmp_path)
    names = {e["suite"]: e for e in entries}
    assert {"my_ds__ids", "my_ds__ids_original", "my_ds__ids__skip"} <= set(names)
    assert (
        names["my_ds__ids"]["role"] == "size"
        and names["my_ds__ids"]["type_role"] == "recommended"
    )
    assert names["my_ds__ids_original"]["type_role"] == "original"
    assert names["my_ds__ids__skip"]["role"] == "skip"
    assert json.loads((tmp_path / "suites.json").read_text()) == entries
    for e in entries:
        assert (tmp_path / e["yaml"]).is_file() and (tmp_path / e["arrow"]).is_file()


def test_size_suite_axes(tmp_path):
    _, result, entries = _build(tmp_path)
    e = next(e for e in entries if e["suite"] == "my_ds__ids")
    suite = yaml.safe_load((tmp_path / e["yaml"]).read_text())
    ch = suite["clickhouse"]
    assert suite["name"] == "my_ds__ids"
    assert set(GRID_CODECS) <= set(ch["codec"])
    chains = [c["codecs"] for c in result.row(0, named=True)["clickhouse_candidates"]]
    comps = {c.split(", ")[-1] for c in chains if c not in TRANSFORMS}
    assert set(GRID_COMPRESSIONS) | comps == set(ch["compression"])
    assert ch["skip_index"] == ["none"]
    fractions = [k.get("target_fraction") for k in suite["queries"]["kinds"]]
    assert fractions == [None, *SIZE_FRACTIONS]
    assert suite["queries"]["samples"] == 3


def test_size_predictions_cover_every_cell(tmp_path):
    _, result, entries = _build(tmp_path)
    e = next(e for e in entries if e["suite"] == "my_ds__ids")
    suite = yaml.safe_load((tmp_path / e["yaml"]).read_text())
    pred = json.loads((tmp_path / e["predicted"]).read_text())
    ch = suite["clickhouse"]
    keys = {f"{e['base']}|{k}|{c}" for k in ch["codec"] for c in ch["compression"]}
    assert set(pred) == keys
    row = result.row(0, named=True)
    chosen = row["clickhouse_codecs"]  # e.g. "Delta, ZSTD(3)"
    head, comp = chosen.split(", ")
    cell = pred[f"{e['base']}|{head}|{comp}"]
    assert cell["origin"] == "candidate" and cell["outcome"] == "chosen"
    assert cell["bytes"] == row["clickhouse_size_bytes"]
    assert pred[f"{e['base']}|none|LZ4"]["bytes"] is not None


def test_lowcardinality_original_has_no_lowcardinality_cell(tmp_path):
    _, _, entries = _build(tmp_path)
    rec = next(e for e in entries if e["suite"] == "my_ds__s")
    assert rec["rec_codec"] == "LowCardinality"
    orig = next((e for e in entries if e["suite"] == "my_ds__s_original"), None)
    if orig is not None:
        ch = yaml.safe_load((tmp_path / orig["yaml"]).read_text())["clickhouse"]
        assert "LowCardinality" not in ch["codec"]


def test_skip_suite(tmp_path):
    _, result, entries = _build(tmp_path)
    e = next(e for e in entries if e["suite"] == "my_ds__ids__skip")
    suite = yaml.safe_load((tmp_path / e["yaml"]).read_text())
    ch = suite["clickhouse"]
    row = result.row(0, named=True)
    assert ch["codec"] == [e["rec_codec"]] and ch["compression"] == [
        e["rec_compression"]
    ]
    cands = row["clickhouse_skip_indexes"]
    chosen = skip_value(next(c for c in cands if c["outcome"] == "chosen"))
    rejected = [skip_value(c) for c in cands if c["outcome"] == "rejected"]
    assert ch["skip_index"] == [chosen, *dict.fromkeys(rejected), "none"]
    pred = json.loads((tmp_path / e["predicted"]).read_text())
    assert pred[chosen]["outcome"] == "chosen" and pred["none"]["outcome"] == "none"
    fractions = [k.get("target_fraction") for k in suite["queries"]["kinds"]]
    assert fractions == [None, *SKIP_FRACTIONS]


def test_skipped_columns_are_recorded(tmp_path):
    frame = pl.DataFrame({"l": [[1, 2], [3]] * 10})
    result = OneShotRecommender().add(frame).result()
    entries = build("n", result, frame, tmp_path, samples=1)
    assert entries == []
    skipped = json.loads((tmp_path / "skipped.json").read_text())
    assert skipped and skipped[0]["column"] == "l"
