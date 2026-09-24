"""Set similarity (Jaccard index, Overlap Coefficient) between distinct-value sets."""

import functools
import math

import polars as pl

from analytics._dtypes import encodable, value_family
from analytics._sets import distinct_values
from analytics.base import Technique, at_least, check_unit, computed, metric_mismatches


class Similarity(Technique):
    """Jaccard = |A∩B| / |A∪B| (NaN when both sets are empty); Overlap = |A∩B| / min(|A|, |B|)
    (NaN when either set is empty). High Jaccard ⇒ heavy overlap of distinct values;
    high Overlap ⇒ one set is (nearly) contained in the other (catches PK-FK-like pairs).

    Sets are distinct non-null values; pairs from different value families are
    ineligible. Verification is always exact, memoised by a per-instance LRU cache of
    distinct-value sets (`cache_size`) that add() clears and no other instance shares.
    Probabilistic implementations mark pairs they never evaluate "pruned".
    """

    SCOPE = "multi_set"
    ARITY = 2
    METRICS = {"jaccard": pl.Float64, "overlap": pl.Float64}
    CONCLUSIONS = {"passes_jaccard": pl.Boolean, "passes_overlap": pl.Boolean}
    MIN_RECALL = 0.85

    def __init__(self, *, jaccard_threshold: float = 0.6, overlap_threshold: float = 0.95, cache_size: int = 4096):
        super().__init__()
        check_unit("jaccard_threshold", jaccard_threshold)
        check_unit("overlap_threshold", overlap_threshold)
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self.jaccard_threshold = jaccard_threshold
        self.overlap_threshold = overlap_threshold
        self._distinct = functools.lru_cache(maxsize=cache_size)(self._distinct_values)

    def eligible(self, series: pl.Series) -> bool:
        return encodable(series.dtype)

    def compatible(self, dtypes) -> bool:
        return len({value_family(d) for d in dtypes}) == 1

    def _on_add(self) -> None:
        self._distinct.cache_clear()

    def _distinct_values(self, frame: str, column: str) -> frozenset:
        return frozenset(distinct_values(self._collected[frame][column]))

    def verify(self, combos) -> dict[str, list[float]]:
        """Exact Jaccard / Overlap for `combos` from (cached) distinct-value sets."""
        jaccard, overlap = [], []
        for (fa, ca), (fb, cb) in combos:
            a, b = self._distinct(fa, ca), self._distinct(fb, cb)
            shared = len(a & b)
            union = len(a) + len(b) - shared
            smaller = min(len(a), len(b))
            jaccard.append(shared / union if union else math.nan)
            overlap.append(shared / smaller if smaller else math.nan)
        return {"jaccard": jaccard, "overlap": overlap}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            passes_jaccard=computed(at_least("jaccard", self.jaccard_threshold)),
            passes_overlap=computed(at_least("overlap", self.overlap_threshold)),
        )

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        if self.EXACT:
            return super().agreement(result, reference)
        keys = self.key_columns()
        if result.select(keys).rows() != reference.select(keys).rows():
            return ["key columns differ from the reference"]
        got, want = result["status"].to_list(), reference["status"].to_list()
        problems = [
            f"row {i}: status {g} vs reference {w}"
            for i, (g, w) in enumerate(zip(got, want))
            if (g == "ineligible") != (w == "ineligible")
        ]
        evaluated = [i for i, g in enumerate(got) if g == "computed"]
        problems += metric_mismatches(result[evaluated], reference[evaluated], keys, list(self.METRICS), 0.0, 0.0)

        def passing(df: pl.DataFrame) -> set[int]:
            flags = (df["passes_jaccard"].fill_null(False) | df["passes_overlap"].fill_null(False)).to_list()
            return {i for i, p in enumerate(flags) if p}

        truth = passing(reference)
        if truth:
            recall = len(truth & passing(result)) / len(truth)
            if recall < self.MIN_RECALL:
                problems.append(f"recall {recall:.3f} < {self.MIN_RECALL} (missed rows {sorted(truth - passing(result))[:10]})")
        return problems
