"""Top-k most frequent values per column (spec
docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md).

Implementations rank items as rows of the RANKED schema (one per column and rank: the row
of an occurrence, the count and the error bound); the base renders each value from its row
(arrow-rs text, `_plugin.render`, as Describe's min / max), concludes `dominated` and
compares sketches with the exact reference.
"""

from __future__ import annotations

import polars as pl

from analytics._dtypes import INTEGERS_64, STRING_LIKE
from analytics.base import Technique, check_unit, computed, metric_mismatches

ITEM = pl.Struct({"value": pl.String, "count": pl.UInt64, "error": pl.UInt64})
TOP_K = pl.List(ITEM)
ELIGIBLE = (
    *INTEGERS_64,
    pl.Boolean,
    pl.Float32,
    pl.Float64,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    *STRING_LIKE,
    pl.Decimal,
    pl.Binary,
)
# The Rust kernels' output (and TopKPolars'): one row per (column, rank); a column without
# values has one row with null rank / row / count / error.
RANKED = {
    "col_name": pl.String,
    "rank": pl.UInt32,
    "row": pl.UInt64,
    "count": pl.UInt64,
    "error": pl.UInt64,
    "n_non_null": pl.UInt64,
    "sketch_bytes": pl.UInt64,
}
MIN_BUDGET = 16 * 1024  # the per-column HyperLogLog's size (p = 14)


def default_budget(k: int) -> int:
    return max(MIN_BUDGET, k * 1024)


def _utf8(b: bytes | None) -> str | None:
    try:
        return None if b is None else b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _binary_as_text(s: pl.Series) -> pl.Series:
    """Binary as String, non-UTF-8 values null (Polars' non-strict cast raises on them)."""
    try:
        return s.cast(pl.String, strict=False)
    except pl.exceptions.ComputeError:
        return pl.Series(s.name, [_utf8(b) for b in s.to_list()], dtype=pl.String)


class TopK(Technique):
    """Each eligible column's k most frequent non-null values, ranked by count desc then the
    row of an occurrence (the first row, for the exact reference), with `n_non_null` and the
    sketch's accounted `sketch_bytes` (null for the reference). `top_k` items are
    {value (arrow-rs text; null for non-UTF-8 binary), count, error}: error is the
    implementation's over-count bound (true count ∈ [count − error, count]), 0 when exact and
    null without a bound. `dominated` = top-1 count / n_non_null ≥ dominance_threshold.
    Eligible: the scalar dtypes the Rust encoder keys; List / Array / Struct / Null / Object
    and 128-bit integers are not. `memory_bytes` (default max(16 KiB, k × 1 KiB)) sizes the
    sketches; the reference ignores it.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {"n_non_null": pl.UInt64, "top_k": TOP_K, "sketch_bytes": pl.UInt64}
    CONCLUSIONS = {"dominated": pl.Boolean}
    MIN_PRECISION = 0.9
    MAX_ARE: float | None = (
        None  # mean relative count error over found values, if checked
    )

    def __init__(
        self,
        *,
        k: int = 5,
        memory_bytes: int | None = None,
        dominance_threshold: float = 0.5,
    ):
        super().__init__()
        if type(k) is not int or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if memory_bytes is not None and (
            type(memory_bytes) is not int or memory_bytes < 0
        ):
            raise ValueError(
                f"memory_bytes must be a non-negative integer, got {memory_bytes!r}"
            )
        check_unit("dominance_threshold", dominance_threshold)
        self.k = k
        self.memory_bytes = default_budget(k) if memory_bytes is None else memory_bytes
        self.dominance_threshold = dominance_threshold

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, ELIGIBLE)

    def describe(self, frames, combos):
        return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    def top_k_rows(
        self, frames, combos, ranked: dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """metrics_frame for `combos` from each frame's RANKED rows, values rendered from
        their rows."""
        from analytics import _plugin

        parts = {
            frame: r.cast(RANKED)
            .sort("col_name", "rank")
            .partition_by("col_name", as_dict=True)
            for frame, r in ranked.items()
        }
        n_non_null, top, sketch_bytes = [], [], []
        for ((frame, col),) in combos:
            r = parts[frame][(col,)]
            items = r.filter(pl.col("rank").is_not_null())
            values = []
            if items.height:
                s = frames[frame][col].gather(items["row"])
                if (
                    s.dtype == pl.Binary
                ):  # as text; non-UTF-8 → null, as arrow-rs renders it
                    s = _binary_as_text(s)
                values = _plugin.render(s)
            top.append(
                [
                    {"value": v, "count": c, "error": e}
                    for v, c, e in zip(values, items["count"], items["error"])
                ]
            )
            n_non_null.append(r["n_non_null"][0])
            sketch_bytes.append(r["sketch_bytes"][0])
        return self.metrics_frame(
            combos,
            {"n_non_null": n_non_null, "top_k": top, "sketch_bytes": sketch_bytes},
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        top1 = pl.col("top_k").list.first().struct.field("count").cast(pl.Float64)
        share = top1 / pl.col("n_non_null").cast(pl.Float64)
        return out.with_columns(dominated=computed(share >= self.dominance_threshold))

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """`n_non_null` exact; aggregate precision (reported values the reference lists ÷
        values it lists — a lower bound of tie-aware precision) ≥ MIN_PRECISION; mean
        relative count error ≤ MAX_ARE when set; per-item checks (`_item_problems`).
        `dominated` is not compared: an estimate can cross the threshold."""
        keys = self.key_columns()
        problems = metric_mismatches(result, reference, keys, ["n_non_null"], 0.0, 0.0)
        if problems:
            return problems
        found = wanted = 0
        rel: list[float] = []
        rows = zip(
            result.select(keys).rows(),
            result["top_k"].to_list(),
            reference["top_k"].to_list(),
        )
        for (df, col), got, want in rows:
            if want is None:
                continue
            true = {w["value"]: w["count"] for w in want}
            floor = want[-1]["count"] if len(want) == self.k else 0
            wanted += len(want)
            for item in got:
                t = true.get(item["value"])
                if t is not None:
                    found += 1
                    rel.append(abs(item["count"] - t) / t)
                problems += self._item_problems(f"{df}.{col}", item, t, floor)
        precision = found / wanted if wanted else 1.0
        if precision < self.MIN_PRECISION:
            problems.append(f"precision {precision:.3f} < {self.MIN_PRECISION}")
        are = sum(rel) / len(rel) if rel else 0.0
        if self.MAX_ARE is not None and are > self.MAX_ARE:
            problems.append(f"mean relative count error {are:.4f} > {self.MAX_ARE}")
        return problems

    def _item_problems(
        self, label: str, item: dict, true: int | None, floor: int
    ) -> list[str]:
        """Per-item checks; `true` is the exact count when the reference lists the value,
        else None (its true count is then at most `floor`, the reference's k-th count).
        """
        return []
