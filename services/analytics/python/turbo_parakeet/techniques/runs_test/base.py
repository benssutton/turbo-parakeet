"""One-sample runs test over k categories: does row order cluster equal values?"""

import polars as pl

from turbo_parakeet._dtypes import ORDERED_SCALARS
from turbo_parakeet.techniques.base import Technique

# Mirrors the Rust kernel's `supported` (src/techniques/single_column/ordered/runs_test.rs).
ELIGIBLE = ORDERED_SCALARS


class RunsTest(Technique):
    """Runs of equal consecutive values per column, and whether row order makes them
    fewer than chance would — the signal for Arrow run-end encoding (REE).

    `n_runs` (R) is the number of maximal runs of equal consecutive values: exactly the
    length of an REE array's run_ends. Nulls are one value (consecutive nulls are one
    run, as in REE); floats compare canonically (−0.0 = 0.0, every NaN equal).

    Under random ordering of the same values — n rows, k categories of counts nᵢ; the
    k-category generalisation of the one-sample Wald–Wolfowitz runs test (Barton & David
    1957):

        E[R]   = 1 + (n² − Σnᵢ²) / n
        Var[R] = [Σnᵢ²(Σnᵢ² + n(n+1)) − 2nΣnᵢ³ − n³] / (n²(n−1))

    `z` = (R − E[R]) / √Var[R]. `p_value` is ONE-SIDED, LOWER TAIL: Φ(z) = P(R ≤ observed),
    "fewer runs than chance" — the only direction that matters for REE. Columns that
    alternate more than chance (A B A B …) get z > 0 and p near 1. p uses the asymptotic
    normal approximation, without continuity correction. Var = 0 (one row, one distinct
    value, or every value distinct: R is then fixed) → z and p are NaN.

    p is not the effect size: at large n it vanishes even when runs are barely longer than
    one row. `mean_run_length` = n / R is the effect size; p says only whether the
    clustering is real. The technique reports metrics only — whether REE pays off for a
    dtype is the recommenders' decision.

    Order-dependent: implementations read rows in the frame's own order. Eligible:
    integers (≤ 64-bit), Boolean, Float, Decimal, temporal, String, Binary, Categorical and
    Enum columns with at least one row; List / Array / Struct / Null / Object, 128-bit
    integers and zero-row columns are ineligible. `dtype` (Python's str(dtype)) is
    reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_rows": pl.UInt64,
        "n_runs": pl.UInt64,
        "expected_runs": pl.Float64,
        "z": pl.Float64,
        "p_value": pl.Float64,
        "mean_run_length": pl.Float64,
    }
    RTOL = 1e-9
    # p underflows to 0 near z ≈ −38.5; two erfc implementations may reach the subnormal
    # range one value apart.
    ATOL = 1e-300

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, ELIGIBLE) and series.len() > 0

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}
