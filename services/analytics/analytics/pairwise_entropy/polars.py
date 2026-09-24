import polars as pl

from analytics._dtypes import is_nested
from analytics.pairwise_entropy.base import PairwiseEntropy


def entropy_bits(df: pl.DataFrame, columns: list[str]) -> float:
    """H(columns) in bits via Polars value_counts; null is its own category.

    Falls back to casting List/Array columns to String if Polars cannot group a
    struct containing nested fields (behaviour is version-dependent).
    """

    def h(frame: pl.DataFrame) -> float:
        key = frame.select(pl.struct(columns).alias("k")).to_series()
        return key.value_counts().get_column("count").entropy(base=2, normalize=True)

    try:
        return h(df)
    except Exception:
        nested = [c for c in columns if is_nested(df.schema[c])]
        if not nested:
            raise
        return h(df.with_columns(pl.col(c).cast(pl.String) for c in nested))


class PairwiseEntropyPolars(PairwiseEntropy):
    """Reference: native Polars value_counts + entropy, one pair at a time (marginals memoised per call)."""

    def _compute(self, frames, combos):
        marginal: dict = {}

        def h1(col):
            if col not in marginal:
                marginal[col] = entropy_bits(frames[col[0]], [col[1]])
            return marginal[col]

        return self.entropy_rows(
            combos,
            [h1(a) for a, _ in combos],
            [h1(b) for _, b in combos],
            [entropy_bits(frames[f], [a, b]) for (f, a), (_, b) in combos],
            [frames[f].height for (f, _), _ in combos],
        )
