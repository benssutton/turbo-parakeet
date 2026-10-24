import math

from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.distinct_count.base import DistinctCount


class DistinctCountHllRust(DistinctCount):
    """Rust extension `hll_distinct`: one 2^precision-register HyperLogLog per 64K-row
    chunk, merged by register max, Ertl's estimator; rayon across columns and chunks."""

    EXACT = False

    def __init__(self, *, precision: int = 14):
        super().__init__()
        if type(precision) is not int or not 4 <= precision <= 18:
            raise ValueError(
                f"precision must be an integer in 4..=18, got {precision!r}"
            )
        self.precision = precision
        # agreement: within 4 standard errors of the exact count
        self.RTOL = 4 * 1.04 / math.sqrt(2**precision)

    def _compute(self, frames, combos):
        rows = {}  # (frame, column) -> plugin row; one binding call per frame
        for frame, group in group_by_frame(combos).items():
            out = _plugin.hll_distinct(
                frames[frame].select([c for ((_, c),) in group]), self.precision
            )
            rows.update(((frame, r["col_name"]), r) for r in out.iter_rows(named=True))
        cols = [rows[k] for (k,) in combos]  # canonical order
        return self.interval_rows(
            combos,
            [r["n_non_null"] for r in cols],
            [r["estimate"] for r in cols],
            [r["std_error"] for r in cols],
            [r["sketch_bytes"] for r in cols],
        )
