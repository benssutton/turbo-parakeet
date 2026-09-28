"""Chi-squared independence test between pairs of categorical columns."""

import polars as pl

from analytics._dtypes import INTEGERS_64, STRING_LIKE
from analytics.base import Technique, at_least, check_unit, computed

CATEGORICAL = (pl.Boolean, *STRING_LIKE, *INTEGERS_64)


class ChiSquared(Technique):
    """χ² test of independence per column pair within a frame.

    Null policy: rows where either column is null are dropped (n_valid counts the
    rest). Constant columns and empty overlaps → NaN statistics. At realistic N,
    p-values are vanishingly small for almost any pair, so Cramér's V (effect size)
    drives the `associated` verdict. low_expected_count: rarest row marginal ×
    rarest column marginal / N < 5.

    Eligible: Boolean, String, Categorical, Enum and integer columns with at least
    one row and at most `max_unique` distinct values (None disables the cap).
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {
        "chi2_stat": pl.Float64,
        "p_value": pl.Float64,
        "cramers_v": pl.Float64,
        "low_expected_count": pl.Boolean,
        "n_valid": pl.UInt32,
    }
    CONCLUSIONS = {"associated": pl.Boolean}
    RTOL = 1e-4
    ATOL = 1e-12

    def __init__(
        self, *, cramers_v_threshold: float = 0.3, max_unique: int | None = 1000
    ):
        super().__init__()
        check_unit("cramers_v_threshold", cramers_v_threshold)
        if max_unique is not None and max_unique < 2:
            raise ValueError(f"max_unique must be >= 2 or None, got {max_unique}")
        self.cramers_v_threshold = cramers_v_threshold
        self.max_unique = max_unique

    def eligible(self, series: pl.Series) -> bool:
        return (
            series.len() > 0
            and isinstance(series.dtype, CATEGORICAL)
            and (self.max_unique is None or series.n_unique() <= self.max_unique)
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            associated=computed(at_least("cramers_v", self.cramers_v_threshold))
        )
