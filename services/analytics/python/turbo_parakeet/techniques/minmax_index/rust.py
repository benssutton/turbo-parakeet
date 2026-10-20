import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.minmax_index.base import MinMaxIndex


class MinMaxIndexRust(MinMaxIndex):
    """Rust extension `minmax_index`: dense rank ids, then two monotonic deques give each
    window's extremes in O(1) amortised per row; skips summed exactly as integers;
    rayon-parallel across columns and ranges of window starts."""

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.minmax_index(
                        frames[frame].select([c for ((_, c),) in group]),
                        self.window_rows,
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
