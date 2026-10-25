"""An ARITY-2 fake technique for tests/test_benchmark_harness.py: the sum of two numeric
columns. FakePairPolars is the reference, FakePairRust its Rust-named sibling."""

import polars as pl

from turbo_parakeet.techniques.base import Technique


class FakePair(Technique):
    SCOPE = "ordered"
    ARITY = 2
    METRICS = {"total": pl.Float64}

    def eligible(self, series: pl.Series) -> bool:
        return series.dtype.is_numeric()

    def _compute(self, frames, combos):
        totals = [
            float(frames[a][x].sum() + frames[b][y].sum()) for (a, x), (b, y) in combos
        ]
        return self.metrics_frame(combos, {"total": totals})


class FakePairPolars(FakePair):
    pass


class FakePairRust(FakePair):
    pass


REFERENCE = "FakePairPolars"
IMPLEMENTATIONS = ("FakePairRust", "FakePairPolars")
