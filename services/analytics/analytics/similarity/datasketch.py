import polars as pl
from datasketch import MinHash, MinHashLSH

from analytics.base import columns_of
from analytics.similarity.base import Similarity


class MinHashDatasketch(Similarity):
    """datasketch MinHash + MinHashLSH (pure Python, one column at a time); candidates
    then verified exactly, the rest pruned."""

    EXACT = False

    def __init__(self, *, num_perm: int = 128, **thresholds):
        super().__init__(**thresholds)
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        self.num_perm = num_perm
        # Kept from the original datasketch filter: 0.45 × the lower threshold. This is
        # a different formula from MinHashRust's (min(jaccard*0.9, overlap*0.45)), not
        # the same rule restated — at the defaults it resolves to min(0.6, 0.95)*0.45 =
        # 0.27, versus MinHashRust's 0.4275, so the two implementations run at
        # different candidate thresholds and benchmark comparisons between them are at
        # different operating points, not an apples-to-apples LSH configuration.
        self.lsh_threshold = min(self.jaccard_threshold, self.overlap_threshold) * 0.45

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm, weights=(0.9, 0.1))
        sketches = {}
        for i, (f, c) in enumerate(columns):
            values = [str(v).encode("utf8") for v in self._distinct(f, c)]
            if values:
                sketch = MinHash(num_perm=self.num_perm)
                sketch.update_batch(values)
                sketches[i] = sketch
                lsh.insert(i, sketch)
        found = {frozenset((columns[i], columns[j])) for i, s in sketches.items() for j in lsh.query(s) if j != i}
        empty = {c for i, c in enumerate(columns) if i not in sketches}
        chosen = [k for k in combos if frozenset(k) in found or k[0] in empty or k[1] in empty]
        chosen_set = set(chosen)
        pruned = [k for k in combos if k not in chosen_set]
        return pl.concat([self.metrics_frame(chosen, self.verify(chosen)), self.null_frame(pruned, "pruned")])
