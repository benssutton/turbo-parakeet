import math

import polars as pl

from analytics.runs_test.base import RunsTest


def exact_moments(n: int, counts: list[int]) -> tuple[float, float]:
    """(E[R], Var[R]) under random ordering of n rows with category `counts`, in exact
    integer arithmetic up to the final divisions (Var NaN when n < 2)."""
    s2 = sum(c * c for c in counts)
    s3 = sum(c**3 for c in counts)
    expected = 1 + (n * n - s2) / n
    numerator = s2 * (s2 + n * (n + 1)) - 2 * n * s3 - n**3
    denominator = n * n * (n - 1)
    return expected, numerator / denominator if denominator else math.nan


FLOATS = (pl.Float32, pl.Float64)


def _canonical(s: pl.Series) -> pl.Series:
    """−0.0 → 0.0 and every NaN payload → one NaN. Defensive: current Polars already
    merges them in group_by, but the semantics the kernel promises (−0.0 = 0.0, every
    NaN equal) are made explicit here rather than left to Polars' float semantics."""
    if s.dtype in FLOATS:
        return (s + 0.0).fill_nan(float("nan"))
    return s


def runs_statistics(s: pl.Series) -> dict[str, int | float]:
    """The RunsTest metrics of one non-empty column."""
    s = _canonical(s)
    n = s.len()
    after, before = s.slice(1), s.slice(0, n - 1)
    changed = after.ne_missing(before)
    if s.dtype in FLOATS:
        # NaN continues a NaN run, whatever `!=` says (defensive, like _canonical)
        changed = changed & ~(after.is_nan() & before.is_nan()).fill_null(False)
    n_runs = 1 + int(changed.sum())
    counts = pl.DataFrame({"v": s}).group_by("v").len()["len"].to_list()
    expected, variance = exact_moments(n, counts)
    # R − E exactly, as (n(R − 1) − (n² − S₂)) / n: a fixed R (Var = 0) never gives ±inf
    d = n * (n_runs - 1) - (n * n - sum(c * c for c in counts))
    z = (d / n) / math.sqrt(variance) if variance > 0 else math.nan
    p = math.nan if math.isnan(z) else 0.5 * math.erfc(-z / math.sqrt(2))
    return {
        "n_rows": n,
        "n_runs": n_runs,
        "expected_runs": expected,
        "z": z,
        "p_value": p,
        "mean_run_length": n / n_runs,
    }


class RunsTestPolars(RunsTest):
    """Reference: R from neighbour comparisons (`ne_missing`, null-aware), category
    counts from a group-by (nulls one group), moments in exact Python ints; only the final
    divisions and Φ (math.erfc) are floating point. Shares no code with the Rust kernel.
    """

    def _compute(self, frames, combos):
        rows = [runs_statistics(frames[f][c]) for ((f, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
