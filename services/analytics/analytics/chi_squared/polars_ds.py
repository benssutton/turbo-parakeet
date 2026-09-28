import math

import polars as pl
import polars_ds as pds

from analytics.chi_squared.base import ChiSquared


def chi_squared_pair(df: pl.DataFrame, a: str, b: str) -> dict:
    pair = df.select(a, b).drop_nulls()
    n = pair.height
    undefined = {
        "chi2_stat": math.nan,
        "p_value": math.nan,
        "cramers_v": math.nan,
        "low_expected_count": False,
        "n_valid": n,
    }
    if n == 0:
        return undefined
    unique_a, unique_b = pair[a].n_unique(), pair[b].n_unique()
    if unique_a < 2 or unique_b < 2:
        return undefined
    result = pair.select(pds.chi2(a, b).alias("r")).unnest("r")
    if "statistic" not in result.columns or "pvalue" not in result.columns:
        raise RuntimeError(
            f"polars-ds chi2 returned unexpected struct fields: {result.columns}"
        )
    stat = float(result["statistic"][0])
    low = (
        pair[a].value_counts()["count"].min()
        * pair[b].value_counts()["count"].min()
        / n
        < 5
    )
    return {
        "chi2_stat": stat,
        "p_value": float(result["pvalue"][0]),
        "cramers_v": math.sqrt(stat / (n * (min(unique_a, unique_b) - 1))),
        "low_expected_count": bool(low),
        "n_valid": n,
    }


class ChiSquaredPolarsDS(ChiSquared):
    """polars-ds `chi2` expression, one pair at a time (the original pure-Python baseline)."""

    def _compute(self, frames, combos):
        rows = [chi_squared_pair(frames[f], a, b) for (f, a), (_, b) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
