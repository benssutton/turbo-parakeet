"""Three-way joint entropy H(A,B,C)."""

import polars as pl

from analytics._dtypes import encodable
from analytics.base import Technique, computed
from analytics.pairwise_entropy.base import near_unique


class ThreewayEntropy(Technique):
    """H(A,B,C) in bits for every column triplet within a frame. Joint entropy near
    log2(n_rows) ⇒ the triplet (almost) identifies rows.

    Null policy: null is its own category. Eligible: any column the Rust encoder
    accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 3
    METRICS = {"h_abc": pl.Float64, "n_rows": pl.UInt32}
    CONCLUSIONS = {"near_unique": pl.Boolean}
    RTOL = 1e-5
    ATOL = 1e-12

    def __init__(self, *, near_unique_margin: float = 0.1):
        super().__init__()
        if near_unique_margin < 0:
            raise ValueError(f"near_unique_margin must be >= 0, got {near_unique_margin}")
        self.near_unique_margin = near_unique_margin

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(near_unique=computed(near_unique("h_abc", self.near_unique_margin)))
