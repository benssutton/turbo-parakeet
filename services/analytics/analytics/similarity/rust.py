import polars as pl

from analytics import _plugin
from analytics.base import columns_of
from analytics.similarity.base import Similarity


def optimal_lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """(bands, rows_per_band) whose LSH S-curve best separates pairs around `threshold`."""
    min_error = float("inf")
    best_b, best_r = 1, num_perm
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            # P(candidate | similarity = s) = 1 - (1 - s^r)^b, evaluated at the threshold
            fp = 1 - (1 - threshold**r) ** b
            fn = (1 - threshold**r) ** b
            error = abs(fp - 0.9) + abs(fn - 0.1)
            if error < min_error:
                min_error = error
                best_b, best_r = b, r
    return best_b, best_r


class MinHashRust(Similarity):
    """Rust extension functions `minhash` (signatures for all columns, rayon-parallel) and
    `lsh_candidates` (banded LSH); candidates then verified exactly, the rest pruned."""

    EXACT = False

    def __init__(self, *, num_perm: int = 128, **thresholds):
        super().__init__(**thresholds)
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        self.num_perm = num_perm
        # The LSH S-curve is calibrated against Jaccard, not Overlap Coefficient, so a
        # pair that passes only on Overlap (a small set inside a large one, Jaccard
        # low) needs a lower candidate threshold to survive the candidate stage.
        # 0.45 × the overlap threshold (carried over unchanged from the deleted
        # minhash_lsh_filter.py) is a fixed discount, not a guarantee: at the default
        # thresholds it resolves to min(0.54, 0.4275) = 0.4275 (bands=28, rows=3), i.e.
        # LSH only surfaces pairs whose Jaccard is roughly >= 0.43. A containment pair
        # (overlap == 1.0) with Jaccard below that is pruned before verification ever
        # runs, even for ordinary low-cardinality columns with no extreme size skew
        # (e.g. 3 vs 9 distinct values, Jaccard 0.33 < 0.43) — not only pairs with huge
        # cardinality differences. Measured on large_dataset.arrow: recall 0.62
        # (389/625), all 236 misses are containment pairs with Jaccard from 0.00002 to
        # 0.36 (median 0.10) that never became LSH candidates.
        self.lsh_threshold = min(self.jaccard_threshold * 0.9, self.overlap_threshold * 0.45)
        self.bands, self.rows_per_band = optimal_lsh_params(self.lsh_threshold, num_perm)

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        empty = {c for c in columns if frames[c[0]][c[1]].null_count() == frames[c[0]][c[1]].len()}
        names = list(frames)
        # Frame *index* prefixes the qualified name, so frame names containing "|"
        # are safe; split on the first "|" only, so column names may contain it.
        signatures = [
            _plugin.minhash(frames[name].select(cols), str(i), self.num_perm)
            for i, name in enumerate(names)
            if (cols := [c for f, c in columns if f == name and (f, c) not in empty])
        ]
        found: set = set()
        if signatures:
            sigs = pl.concat(signatures)
            if sigs.height > 1:
                pairs = _plugin.lsh_candidates(sigs, num_bands=self.bands, rows_per_band=self.rows_per_band)

                def parse(qualified: str):
                    i, column = qualified.split("|", 1)
                    return names[int(i)], column

                found = {frozenset((parse(a), parse(b))) for a, b in pairs.iter_rows()}
        chosen = [k for k in combos if frozenset(k) in found or k[0] in empty or k[1] in empty]
        chosen_set = set(chosen)
        pruned = [k for k in combos if k not in chosen_set]
        return pl.concat([self.metrics_frame(chosen, self.verify(chosen)), self.null_frame(pruned, "pruned")])
