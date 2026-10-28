"""The factors a column's statistics depend on, their levels, the constraints between
them, and each level's concrete targets."""

from __future__ import annotations

import math

from synthdata.frame.spec import (
    Clustered,
    ColumnSpec,
    Multiples,
    Plain,
    Random,
    Runs,
    Sorted,
    Walk,
)

FACTORS = {
    "family": (
        "int",
        "uint",
        "float",
        "decimal",
        "string",
        "categorical",
        "boolean",
        "temporal",
        "binary",
    ),
    "cardinality": ("constant", "binary", "low", "mid", "near_unique"),
    "skew": ("uniform", "heavy"),
    "nulls": ("none", "sparse", "heavy", "all"),
    "order": ("random", "runs", "clustered", "sorted"),
    "structure": ("plain", "multiples", "walk"),
}
STRUCTURED = frozenset({"int", "uint", "decimal", "temporal"})
TEMPORAL_TYPES = (
    "date32",
    "timestamp",
    "duration",
    "time64",
)  # rotated by column index
TYPE_OF = {
    "int": "int64",
    "uint": "uint32",
    "float": "float64",
    "decimal": "decimal",
    "string": "string",
    "categorical": "categorical",
    "boolean": "bool",
    "binary": "binary",
}
GCD = {
    "int64": 1_000,
    "uint32": 1_000,
    "decimal": 100,
    "date32": 7,
    "timestamp": 1_000_000,
    "duration": 1_000_000,
    "time64": 1_000_000,
}
LOW, NEAR_UNIQUE_SHARE = 10, 0.95
HEAVY_ENTROPY, HEAVY_BINARY_ENTROPY = 0.4, 0.47  # H / log2(d); binary: ≈ 90 / 10
NULL_SHARE = {"none": 0.0, "sparse": 0.05, "heavy": 0.5, "all": 1.0}
MEAN_RUN, DISORDER, MAX_STEP = 8.0, 0.1, 8


def mid_distinct(rows: int) -> int:
    return max(21, round(math.sqrt(rows)))


def normalise(levels: dict) -> dict | None:
    """The canonical form of a combination: None if impossible; factors that do not apply
    to it become None."""
    lv = dict(levels)
    if lv["nulls"] == "all":
        return {
            f: (lv["family"] if f == "family" else "all" if f == "nulls" else None)
            for f in FACTORS
        }
    if lv["family"] not in STRUCTURED:
        if lv["structure"] not in (None, "plain"):
            return None
        lv["structure"] = None
    if lv["family"] == "boolean" and lv["cardinality"] not in ("constant", "binary"):
        return None
    if lv["structure"] == "walk":
        lv["cardinality"] = lv["skew"] = lv["order"] = None
        return lv
    if lv["cardinality"] in ("constant", "binary") and lv["structure"] is not None:
        lv["structure"] = None
    if lv["cardinality"] == "constant":
        lv["skew"] = lv["order"] = None
    if lv["cardinality"] == "near_unique":
        if lv["order"] == "runs":
            return None
        lv["skew"] = None
    return lv


def column_spec(levels: dict, index: int, rows: int) -> ColumnSpec:
    """The ColumnSpec of a normalised combination; temporal columns rotate their type."""
    family = levels["family"]
    type_ = TEMPORAL_TYPES[index % 4] if family == "temporal" else TYPE_OF[family]
    if levels["nulls"] == "all":
        return ColumnSpec(type_, distinct=1, null_share=1.0)
    null_share = NULL_SHARE[levels["nulls"]]
    if levels["structure"] == "walk":
        return ColumnSpec(type_, null_share=null_share, structure=Walk(MAX_STEP))
    card = levels["cardinality"]
    distinct = {"constant": 1, "binary": 2, "low": LOW, "mid": mid_distinct(rows)}.get(
        card
    )
    share = NEAR_UNIQUE_SHARE if card == "near_unique" else None
    entropy = None
    if levels["skew"] == "heavy":
        entropy = HEAVY_BINARY_ENTROPY if card == "binary" else HEAVY_ENTROPY
    order = {
        None: Random(),
        "random": Random(),
        "runs": Runs(MEAN_RUN),
        "clustered": Clustered(DISORDER),
        "sorted": Sorted(),
    }[levels["order"]]
    structure = Multiples(GCD[type_]) if levels["structure"] == "multiples" else Plain()
    return ColumnSpec(
        type_,
        distinct=distinct,
        distinct_share=share,
        entropy=entropy,
        null_share=null_share,
        order=order,
        structure=structure,
    )
