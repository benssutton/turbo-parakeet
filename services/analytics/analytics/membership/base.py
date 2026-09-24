"""Set membership / containment between the distinct-value sets of two columns."""

import math

import polars as pl

from analytics._dtypes import encodable, value_family
from analytics.base import Technique, at_least, check_unit, computed, metric_mismatches, same_value

RELATIONSHIPS = ["pk_pk", "fk_pk", "pk_fk", "mutual", "a_in_b", "b_in_a", "none"]
_COUNTS = ["n_distinct_a", "n_distinct_b", "n_non_null_a", "n_non_null_b"]


class Membership(Technique):
    """Directional containment of distinct non-null values, both ways.

    ratio_a_in_b = |distinct(A) ∩ distinct(B)| / |distinct(A)| (NaN if A has no
    values; 0.0 if only B is empty). A column is unique when every non-null value
    is distinct. With t = containment_threshold:
        pk_pk   both ratios ≥ t, both unique
        fk_pk   A ⊂ B (ratio_a_in_b ≥ t), B unique, A not      (pk_fk: the mirror)
        mutual  both ratios ≥ t otherwise
        a_in_b / b_in_a   one direction ≥ t;  none   otherwise
    Pairs are compared only within one value family (analytics._dtypes.value_family).
    """

    SCOPE = "multi_set"
    ARITY = 2
    METRICS = {
        "ratio_a_in_b": pl.Float64,
        "ratio_b_in_a": pl.Float64,
        "n_distinct_a": pl.UInt32,
        "n_distinct_b": pl.UInt32,
        "n_non_null_a": pl.UInt32,
        "n_non_null_b": pl.UInt32,
    }
    CONCLUSIONS = {"unique_a": pl.Boolean, "unique_b": pl.Boolean, "relationship": pl.Enum(RELATIONSHIPS)}

    def __init__(self, *, containment_threshold: float = 0.95):
        super().__init__()
        check_unit("containment_threshold", containment_threshold)
        self.containment_threshold = containment_threshold

    def eligible(self, series: pl.Series) -> bool:
        return encodable(series.dtype)

    def compatible(self, dtypes) -> bool:
        return len({value_family(d) for d in dtypes}) == 1

    def membership_rows(self, frames, combos, n_distinct: dict, contained: dict) -> pl.DataFrame:
        """Metrics from exact distinct counts and directional ratios
        contained[(x, y)] = fraction of x's distinct values found in y."""

        def ratio(x, y):
            return contained.get((x, y), 0.0) if n_distinct[x] else math.nan

        non_null = {c: frames[c[0]][c[1]].len() - frames[c[0]][c[1]].null_count() for c in n_distinct}
        return self.metrics_frame(
            combos,
            {
                "ratio_a_in_b": [ratio(a, b) for a, b in combos],
                "ratio_b_in_a": [ratio(b, a) for a, b in combos],
                "n_distinct_a": [n_distinct[a] for a, _ in combos],
                "n_distinct_b": [n_distinct[b] for _, b in combos],
                "n_non_null_a": [non_null[a] for a, _ in combos],
                "n_non_null_b": [non_null[b] for _, b in combos],
            },
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        t = self.containment_threshold
        a_in_b, b_in_a = at_least("ratio_a_in_b", t), at_least("ratio_b_in_a", t)
        ua = (pl.col("n_distinct_a") == pl.col("n_non_null_a")) & (pl.col("n_non_null_a") > 0)
        ub = (pl.col("n_distinct_b") == pl.col("n_non_null_b")) & (pl.col("n_non_null_b") > 0)
        rel = (
            pl.when(a_in_b & b_in_a & ua & ub).then(pl.lit("pk_pk"))
            .when(a_in_b & ub & ~ua).then(pl.lit("fk_pk"))
            .when(b_in_a & ua & ~ub).then(pl.lit("pk_fk"))
            .when(a_in_b & b_in_a).then(pl.lit("mutual"))
            .when(a_in_b).then(pl.lit("a_in_b"))
            .when(b_in_a).then(pl.lit("b_in_a"))
            .otherwise(pl.lit("none"))
        )
        return out.with_columns(
            unique_a=computed(ua),
            unique_b=computed(ub),
            relationship=computed(rel).cast(pl.Enum(RELATIONSHIPS)),
        )


class BloomMembership(Membership):
    """Shared by the Bloom-filter implementations: `fp_rate` and the probabilistic
    agreement bound (no false negatives; aggregate FP rate ≤ FP_TOLERANCE × fp_rate)."""

    EXACT = False
    FP_TOLERANCE = 3.0

    def __init__(self, *, fp_rate: float = 0.01, **thresholds):
        super().__init__(**thresholds)
        if not 0.0 < fp_rate < 1.0:
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")
        self.fp_rate = fp_rate

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        problems = metric_mismatches(result, reference, self.key_columns(), _COUNTS, 0.0, 0.0)
        if problems[:1] == ["key columns differ from the reference"]:
            return problems
        false_pos = negatives = 0.0
        for side, n_col in (("ratio_a_in_b", "n_distinct_a"), ("ratio_b_in_a", "n_distinct_b")):
            for got, want, n in zip(result[side].to_list(), reference[side].to_list(), reference[n_col].to_list()):
                if want is None or math.isnan(want):
                    if not same_value(got, want, 0.0, 0.0):
                        problems.append(f"{side}: {got!r} where the reference has {want!r}")
                    continue
                if got is None or got < want - 1e-12:
                    problems.append(f"{side}: false negative ({got!r} < exact {want!r})")
                    continue
                false_pos += (got - want) * n
                negatives += (1.0 - want) * n
        # Below ~100 negatives, a single false positive can push the observed rate well
        # past FP_TOLERANCE x fp_rate by chance (e.g. 1/14 = 0.071 vs a 3% bound) — not
        # evidence the filter is miscalibrated. The no-false-negative check above still
        # always applies.
        if negatives >= 100 and false_pos / negatives > self.FP_TOLERANCE * self.fp_rate:
            problems.append(
                f"false-positive rate {false_pos / negatives:.4f} > {self.FP_TOLERANCE} × fp_rate {self.fp_rate}"
            )
        return problems
