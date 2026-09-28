"""Pairwise joint entropy and (normalised) mutual information."""

import math

import polars as pl

from analytics._dtypes import encodable
from analytics.base import Technique, at_least, check_unit, computed


def near_unique(h_column: str, margin: float) -> pl.Expr:
    """Joint entropy within `margin` bits of log2(n_rows): combinations are (almost) all distinct."""
    return (pl.col("n_rows") > 1) & (
        pl.col(h_column) >= pl.col("n_rows").cast(pl.Float64).log(2) - margin
    )


class PairwiseEntropy(Technique):
    """H(A), H(B), H(A,B) in bits; MI = H(A) + H(B) − H(A,B); NMI = MI / min(H(A), H(B))
    (NaN when either marginal entropy is 0). NMI ≈ 1 ⇒ one column predicts the other.

    Null policy: null is its own category. Eligible: any column the Rust encoder
    accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {
        "h_a": pl.Float64,
        "h_b": pl.Float64,
        "h_ab": pl.Float64,
        "mi": pl.Float64,
        "nmi": pl.Float64,
        "n_rows": pl.UInt32,
    }
    CONCLUSIONS = {"redundant": pl.Boolean, "near_unique": pl.Boolean}
    RTOL = 1e-5
    ATOL = 1e-12

    def __init__(self, *, nmi_threshold: float = 0.9, near_unique_margin: float = 0.1):
        super().__init__()
        check_unit("nmi_threshold", nmi_threshold)
        if near_unique_margin < 0:
            raise ValueError(
                f"near_unique_margin must be >= 0, got {near_unique_margin}"
            )
        self.nmi_threshold = nmi_threshold
        self.near_unique_margin = near_unique_margin

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)

    def entropy_rows(self, combos, h_a, h_b, h_ab, n_rows) -> pl.DataFrame:
        """Metrics frame from marginal and joint entropies (MI/NMI derived here, once)."""
        mi = [a + b - ab for a, b, ab in zip(h_a, h_b, h_ab)]
        nmi = [
            m / min(a, b) if min(a, b) > 0 else math.nan
            for m, a, b in zip(mi, h_a, h_b)
        ]
        return self.metrics_frame(
            combos,
            {
                "h_a": h_a,
                "h_b": h_b,
                "h_ab": h_ab,
                "mi": mi,
                "nmi": nmi,
                "n_rows": n_rows,
            },
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            redundant=computed(at_least("nmi", self.nmi_threshold)),
            near_unique=computed(near_unique("h_ab", self.near_unique_margin)),
        )
