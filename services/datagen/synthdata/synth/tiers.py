"""The covering frame at two lengths, plus a planted-relations block and nested extras."""

from __future__ import annotations

import functools

import numpy as np
import polars as pl
import pyarrow as pa

from synthdata.cache import load_or_generate_frame
from synthdata.synth.classify import coverage
from synthdata.synth.factors import column_spec, mid_distinct
from synthdata.synth.pairwise import covering
from synthdata.frame import (
    ColumnSpec,
    CompositeKey,
    ForeignKey,
    FrameSpec,
    NoisyCopy,
    Relabel,
    Subset,
    TargetMissed,
)

SMALL_ROWS = 2_000
WINDOW_ROWS = (
    9 * 8_192
)  # crosses the 65 536-row streaming block; a full window at G <= 8

_ABBR = {
    "int": "int",
    "uint": "uint",
    "float": "flt",
    "decimal": "dec",
    "string": "str",
    "categorical": "cat",
    "boolean": "bool",
    "temporal": "time",
    "binary": "bin",
    "constant": "const",
    "binary_": "two",
    "low": "low",
    "mid": "mid",
    "near_unique": "uniq",
    "uniform": "uni",
    "heavy": "heavy",
    "none": "none",
    "sparse": "sparse",
    "all": "all",
    "random": "rand",
    "runs": "runs",
    "clustered": "clus",
    "sorted": "sort",
    "plain": "plain",
    "multiples": "mult",
    "walk": "walk",
    None: "na",
}


def column_name(index: int, levels: dict) -> str:
    card = "binary_" if levels["cardinality"] == "binary" else levels["cardinality"]
    parts = [
        levels["family"],
        card,
        levels["skew"],
        levels["nulls"],
        levels["order"],
        levels["structure"],
    ]
    return f"c{index:02d}_" + "_".join(_ABBR[p] for p in parts)


def relation_block(rows: int) -> dict:
    mid = mid_distinct(rows)
    return {
        "rel_p_str_low": ColumnSpec("string", distinct=10),
        "rel_p_int_mid": ColumnSpec("int64", distinct=mid),
        "rel_p_str_unique": ColumnSpec("string", distinct_share=1.0),
        "rel_p_int_unique": ColumnSpec("int64", distinct_share=1.0),
        "rel_noisy05_str_low": NoisyCopy("rel_p_str_low", 0.05),
        "rel_noisy50_int_mid": NoisyCopy("rel_p_int_mid", 0.5),
        "rel_relabel_str_low": Relabel("rel_p_str_low", "int64"),
        "rel_relabel_int_mid": Relabel("rel_p_int_mid", "string"),
        "rel_fk_str_unique": ForeignKey("rel_p_str_unique", null_share=0.1),
        "rel_fk_int_unique": ForeignKey("rel_p_int_unique", zipf_exponent=1.1),
        "rel_subset90_int_mid": Subset("rel_p_int_mid", 0.9),
        "rel_subset50_str_unique": Subset("rel_p_str_unique", 0.5),
        "rel_key_lo_int_mid": CompositeKey("rel_p_int_mid"),
        "rel_indep_int_mid": ColumnSpec("int64", distinct=mid),
    }


def synth_spec(rows: int, seed: int = 0) -> tuple[FrameSpec, dict]:
    """(FrameSpec, intended levels per covering column)."""
    columns, intended = {}, {}
    for i, levels in enumerate(covering()):
        name = column_name(i, levels)
        columns[name] = column_spec(levels, i, rows)
        intended[name] = levels
    columns.update(relation_block(rows))
    return FrameSpec(rows=rows, columns=columns, seed=seed), intended


def synth_manifest(rows: int, seed: int = 0):
    """(FrameManifest, intended levels) of the covering frame at `rows`."""
    spec, intended = synth_spec(rows, seed)
    _, manifest = load_or_generate_frame(spec, strict=True)
    return manifest, intended


@functools.cache
def synth_frame(rows: int, seed: int = 0) -> pl.DataFrame:
    """The covering frame at any length (cached in process and on disk). Strict: every
    covering column's measured levels equal its intended ones."""
    spec, intended = synth_spec(rows, seed)
    df, manifest = load_or_generate_frame(spec, strict=True)
    cov = coverage(manifest, intended)
    if cov.mismatches:
        name, diff = next(iter(cov.mismatches.items()))
        raise TargetMissed(
            name,
            "levels",
            {f: w for f, (w, _) in diff.items()},
            {f: g for f, (_, g) in diff.items()},
        )
    return _with_nested(df, seed)


def synth_small(seed: int = 0) -> pl.DataFrame:
    return synth_frame(SMALL_ROWS, seed)


def synth_windows(seed: int = 0) -> pl.DataFrame:
    return synth_frame(WINDOW_ROWS, seed)


def _with_nested(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """List(Int32) with 0-3 items and 10% nulls (a null row holds no slots, as Polars and
    pyarrow build them), Array(Int32, 3), Struct{a: Int64, b: String}."""
    rng = np.random.default_rng([seed, 99])
    n = df.height
    lengths = rng.integers(0, 4, n)
    null = rng.random(n) < 0.1
    lengths[null] = 0
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    values = pa.array(rng.integers(0, 100, int(offsets[-1])), pa.int32())
    lists = pa.ListArray.from_arrays(pa.array(offsets), values, mask=pa.array(null))
    items = pa.array(rng.integers(0, 10, 3 * n), pa.int32())
    arrays = pa.FixedSizeListArray.from_arrays(items, 3)
    return df.with_columns(
        pl.Series("list_int", lists),
        pl.Series("array_int", arrays),
        pl.struct(a=pl.col("rel_p_int_mid"), b=pl.col("rel_p_str_low")).alias("struct"),
    )


def spread(frame: pl.DataFrame, k: int) -> pl.DataFrame:
    """k columns evenly spread over the frame (all of them if k >= width), first
    included."""
    if k >= frame.width:
        return frame
    idx = np.linspace(0, frame.width - 1, k).round().astype(int)
    return frame.select([frame.columns[i] for i in idx])
