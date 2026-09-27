"""Narrowest value-preserving Arrow type per column, cast, verified and measured
(Spec B: docs/superpowers/specs/2026-09-26-recommend-technique-design.md).

Recommend is Describe plus recommendation metrics: its implementations fill
Describe's METRICS and REC_METRICS; Describe's conclusions are unchanged.
"""

from __future__ import annotations

import polars as pl

from analytics.describe.base import Describe

OUTCOME = pl.Enum(["chosen", "failed", "rejected", "not_tried"])
CANDIDATE = pl.Struct(
    {
        "arrow_type": pl.String,
        "rule": pl.String,
        "evidence": pl.String,
        "predicted_bytes": pl.UInt64,
        "projected_population_bytes": pl.Float64,
        "outcome": OUTCOME,
        "reason": pl.String,
    }
)
REC_METRICS = {
    "rec_nullable": pl.Boolean,
    "rec_arrow_type": pl.String,
    "rec_arrow_size_bytes": pl.UInt64,
    "rec_arrow_size_zstd_bytes": pl.UInt64,
    "rec_polars_type": pl.String,
    "rec_polars_size_bytes": pl.UInt64,
    "rec_polars_size_zstd_bytes": pl.UInt64,
    "rec_lossy_formatting": pl.Boolean,
    "rec_candidates": pl.List(CANDIDATE),
}


class Recommend(Describe):
    """Describe, then per column: candidate Arrow types from the type hierarchy
    (Spec B §4) and dictionary encoding for strings (§5.2), each with a predicted
    IPC size; tried smallest projected population size first (ties: hierarchy
    rank), cast, verified row by row and measured. `rec_arrow_type` is pyarrow's
    spelling, `rec_polars_type` Python's `str(dtype)` of the equivalent Polars type;
    sizes are Arrow IPC bodies (Polars: its native layout), plain and ZSTD.
    `rec_candidates` lists every candidate with the rule, the metric values it
    tested, its predicted / projected size and its outcome.

    `boolean_pairs`: (true text, false text) pairs a two-valued string column may
    map to Boolean, compared case-insensitively.
    """

    METRICS = {**Describe.METRICS, **REC_METRICS}

    def __init__(self, *, boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),), **describe_params) -> None:
        super().__init__(**describe_params)
        pairs = tuple(tuple(p) for p in boolean_pairs)
        if any(len(p) != 2 or not all(isinstance(v, str) and v for v in p) or p[0].lower() == p[1].lower() for p in pairs):
            raise ValueError(f"boolean_pairs must be pairs of distinct non-empty strings, got {boolean_pairs!r}")
        self.boolean_pairs = pairs
