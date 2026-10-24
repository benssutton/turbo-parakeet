import polars as pl

from turbo_parakeet.techniques.distinct_count.base import DistinctCount, canonical


def _values(frames, combos):
    return [canonical(frames[f][c]) for ((f, c),) in combos]


class DistinctCountPolars(DistinctCount):
    """Reference: exact Polars n_unique of the canonical non-null values."""

    def _compute(self, frames, combos):
        vs = _values(frames, combos)
        n = len(vs)
        return self.interval_rows(
            combos,
            [v.len() for v in vs],
            [float(v.n_unique()) for v in vs],
            [0.0] * n,
            [None] * n,
        )


class DistinctCountPolarsApprox(DistinctCount):
    """Polars approx_n_unique (Polars' HyperLogLog) of the canonical non-null values; no
    precision setting, std_error not exposed (null)."""

    EXACT = False
    RTOL = 0.05
    # one or two register collisions at small d (Polars' sketch is a HyperLogLog too)
    ATOL = 2.0

    def __init__(self, *, precision: int | None = None):
        """`precision` is accepted and ignored so benchmark variants can pass it to every
        estimator; Polars' sketch has no precision setting."""
        super().__init__()

    def _compute(self, frames, combos):
        vs = _values(frames, combos)
        n = len(vs)
        return self.interval_rows(
            combos,
            [v.len() for v in vs],
            [
                float(v.to_frame().select(pl.first().approx_n_unique()).item())
                for v in vs
            ],
            [None] * n,
            [None] * n,
        )
