import numpy as np
import polars as pl

from turbo_parakeet.techniques.describe import estimators
from turbo_parakeet.techniques.distinct_count.base import canonical
from turbo_parakeet.techniques.population_cardinality.base import PopulationCardinality

_U = np.uint64


def subsets(seed: int, n: int) -> np.ndarray:
    """SplitMix64 finaliser of seed + row, mod 3, for rows 0..n (describe/frequencies.rs
    `subset`)."""
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=_U) + _U(seed) + _U(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
        return ((z ^ (z >> _U(31))) % _U(3)).astype(np.uint8)


def statistics(s: pl.Series, seed: int) -> dict:
    keep = s.is_not_null().to_numpy()
    bits = (np.uint8(1) << subsets(seed, s.len()))[keep]
    groups = (
        pl.DataFrame({"v": canonical(s), "bit": bits})
        .group_by("v")
        .agg(count=pl.len(), mask=pl.col("bit").bitwise_or())
    )
    counts = groups["count"].to_numpy()
    history = np.bincount(groups["mask"].to_numpy(), minlength=8)[1:].tolist()
    n, d = int(keep.sum()), groups.height
    f1, f2 = int((counts == 1).sum()), int((counts == 2).sum())
    chao = estimators.chao1(d, f1, f2)
    sch = estimators.schnabel(history, d, n) or (float("nan"),) * 3
    return {
        "n_non_null": n,
        "n_distinct": d,
        "f1": f1,
        "f2": f2,
        "chao1": chao[0],
        "chao1_low": chao[1],
        "chao1_high": chao[2],
        "schnabel": sch[0],
        "schnabel_low": sch[1],
        "schnabel_high": sch[2],
    }


class PopulationCardinalityPolars(PopulationCardinality):
    """Reference: counts and capture masks by a Polars group_by, the split by SplitMix64
    in NumPy uint64 (bit-identical to the Rust `subset`), the formulas of
    describe/estimators.py. Shares no counting code with the Rust kernel."""

    def _compute(self, frames, combos):
        rows = [statistics(frames[f][c], self.seed) for ((f, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
