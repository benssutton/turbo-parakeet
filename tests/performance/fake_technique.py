"""A tiny technique for tests/test_benchmark_harness.py: the sum of each numeric column.
FakeSumPolars is the reference, FakeSumRust its exact Rust-named sibling (with a ballast
allocation, to check the memory figure), FakeSumSketchRust an estimator off by `noise`, and
FakeSumBroken raises."""

import polars as pl

from turbo_parakeet.techniques.base import Technique


class FakeSum(Technique):
    SCOPE = "per_column"
    ARITY = 1
    METRICS = {"total": pl.Float64}

    def __init__(self, *, noise: float = 0.0, ballast_mb: int = 0):
        super().__init__()
        self.noise = noise
        self.ballast_mb = ballast_mb

    def eligible(self, series: pl.Series) -> bool:
        return series.dtype.is_numeric()

    def _compute(self, frames, combos):
        ballast = b"x" * (
            self.ballast_mb * 2**20
        )  # written, so it is in the working set
        totals = [float(frames[n][c].sum()) * (1 + self.noise) for ((n, c),) in combos]
        del ballast
        return self.metrics_frame(combos, {"total": totals})


class FakeSumPolars(FakeSum):
    pass


class FakeSumRust(FakeSum):
    pass


class FakeSumSketchRust(FakeSum):
    EXACT = False


class FakeSumBroken(FakeSum):
    def _compute(self, frames, combos):
        raise RuntimeError("broken on purpose")


REFERENCE = "FakeSumPolars"
IMPLEMENTATIONS = ("FakeSumRust", "FakeSumSketchRust", "FakeSumBroken", "FakeSumPolars")
