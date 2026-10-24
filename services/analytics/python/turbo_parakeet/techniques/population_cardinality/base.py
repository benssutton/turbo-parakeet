"""Distinct values in the population behind each column, unseen ones included
(per-column scope; spec docs/superpowers/specs/2026-10-24-cardinality-entropy-techniques-design.md §3).
"""

import math

import polars as pl

from turbo_parakeet._dtypes import encodable, is_nested
from turbo_parakeet.techniques.base import Technique

ESTIMATORS = ("chao1", "schnabel")


class PopulationCardinality(Technique):
    """Observed `n_distinct` (d of `n_non_null`), singletons `f1`, doubletons `f2`, and two
    estimates of the population's distinct count with 95% intervals: bias-corrected
    Chao1 (log-normal interval) and Schnabel over a seeded 3-way row split (Byar's
    interval; NaN when d/n >= 0.5 or no recapture). Both are always reported: choosing one
    is the recommenders' decision. Values keyed as the Rust encoder keys them (floats
    canonical, Categorical / Enum by string). Eligible: as DistinctCount."""

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_non_null": pl.UInt64,
        "n_distinct": pl.UInt64,
        "f1": pl.UInt64,
        "f2": pl.UInt64,
        **{f"{e}{s}": pl.Float64 for e in ESTIMATORS for s in ("", "_low", "_high")},
    }
    RTOL = 1e-9
    ATOL = 1e-9

    def __init__(self, *, seed: int = 0):
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed

    def eligible(self, series):
        return (
            series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)
        )

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def accuracy(self, frames, result, reference, truth=None):
        """The default figures (agreement with the reference) and, with `truth` ((frame,
        column) -> the population's distinct count D), over the computed columns:
        observed_rel_err = max |d - D| / D; <est>_rel_err = max |estimate - D| / D (NaN
        estimates skipped, counted in <est>_n_undefined); <est>_coverage = share of
        columns whose [low, high] contains D."""
        out = super().accuracy(frames, result, reference)
        if truth is None:
            return out
        rows = [
            r
            for r in result.iter_rows(named=True)
            if r["status"] == "computed" and (r["df_a"], r["col_a"]) in truth
        ]
        if not rows:
            return out
        d_true = [truth[r["df_a"], r["col_a"]] for r in rows]
        out["observed_rel_err"] = max(
            abs(r["n_distinct"] - t) / t for r, t in zip(rows, d_true)
        )
        for e in ESTIMATORS:
            pairs = [(r, t) for r, t in zip(rows, d_true) if not math.isnan(r[e])]
            out[f"{e}_n_undefined"] = len(rows) - len(pairs)
            out[f"{e}_rel_err"] = max(
                (abs(r[e] - t) / t for r, t in pairs), default=None
            )
            out[f"{e}_coverage"] = (
                sum(r[f"{e}_low"] <= t <= r[f"{e}_high"] for r, t in pairs) / len(pairs)
                if pairs
                else None
            )
        return out
