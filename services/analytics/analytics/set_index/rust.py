import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.set_index.base import SetIndex


class SetIndexRust(SetIndex):
    """Rust extension `set_index`: dense ids, then a sliding window keeps per-id row counts,
    its distinct values and their population rows, O(1) per row entering and leaving; skips
    summed exactly as integers; rayon-parallel across columns and ranges of window starts.
    """

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.set_index(
                        frames[frame].select([c for ((_, c),) in group]),
                        self.window_rows,
                        self.max_size,
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
