"""Adjusted Rand Index between the row partitions induced by two columns."""

import polars as pl

from analytics._dtypes import encodable, is_nested
from analytics.base import Technique, at_least, check_unit, computed


class AdjustedRand(Technique):
    """Chance-corrected agreement between two partitions (rows sharing a value form
    a cluster). 1.0 = identical partitions, ≈0 = chance, floor -0.5.

    Null policy: rows where either column is null are dropped (n_valid counts the
    rest). No overlapping rows → NaN; degenerate denominator (e.g. both columns
    constant) → 1.0, matching sklearn. Eligible: any non-nested column the Rust
    encoder accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {"ari": pl.Float64, "n_valid": pl.UInt32}
    CONCLUSIONS = {"same_partition": pl.Boolean}
    RTOL = 1e-9
    ATOL = 1e-12

    def __init__(self, *, ari_threshold: float = 0.9):
        super().__init__()
        check_unit("ari_threshold", ari_threshold)
        self.ari_threshold = ari_threshold

    def eligible(self, series: pl.Series) -> bool:
        return (
            series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            same_partition=computed(at_least("ari", self.ari_threshold))
        )
