"""Delta / DoubleDelta statistics: how well ClickHouse's Delta and DoubleDelta codecs suit a
column in its row order (spec docs/superpowers/specs/2026-10-23-delta-codecs-design.md §2).
"""

import polars as pl

from turbo_parakeet.techniques.base import Technique
from turbo_parakeet.techniques.gcd.base import INTEGER_BACKED


class Delta(Technique):
    """Differences of consecutive non-null raw physical integers (as for GCD: Decimal →
    unscaled, Date → days, Datetime / Duration → time unit, Time → ns), in the frame's
    own row order; nulls are skipped (the delta spans them).

    Δᵢ = xᵢ₊₁ − xᵢ and ΔΔᵢ = Δᵢ₊₁ − Δᵢ, computed without wrapping (exact, 128-bit);
    `delta_min/max` and `double_delta_min/max` are Decimal(38, 0), null where there is no
    delta (`n_deltas` = 0) or no double delta. `sig(v)` is the number of bytes (0..9) of
    the signed two's-complement form of v with trailing sign bytes dropped, 0 for v = 0
    (so the full 64-bit range of a delta needs 9). `value_bytes_mean`, `delta_bytes_mean`
    and `double_delta_bytes_mean` are the means of sig over the values, deltas and double
    deltas (NaN when there are none); `zero_*_fraction` the share of zero deltas / double
    deltas; `*_entropy` the Shannon entropy in bits of the delta / double-delta values.
    `expected_delta_bytes_mean` is the exact mean of sig(xⱼ − xᵢ) over all ordered pairs
    of distinct positions: the baseline for a random row order (observed − expected =
    gain from the order).

    Floats are ineligible (Gorilla is out of scope), as are Boolean, text, Categorical,
    nested, 128-bit and zero-row columns. Metrics only: whether a codec pays off belongs
    to the recommenders. `dtype` (Python's str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_rows": pl.UInt64,
        "n_non_null": pl.UInt64,
        "n_deltas": pl.UInt64,
        "delta_min": pl.Decimal(38, 0),
        "delta_max": pl.Decimal(38, 0),
        "double_delta_min": pl.Decimal(38, 0),
        "double_delta_max": pl.Decimal(38, 0),
        "value_bytes_mean": pl.Float64,
        "delta_bytes_mean": pl.Float64,
        "double_delta_bytes_mean": pl.Float64,
        "zero_delta_fraction": pl.Float64,
        "zero_double_delta_fraction": pl.Float64,
        "delta_entropy": pl.Float64,
        "double_delta_entropy": pl.Float64,
        "expected_delta_bytes_mean": pl.Float64,
    }
    RTOL = 1e-9
    ATOL = 0.0

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, INTEGER_BACKED) and series.len() > 0

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}
