import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.runs_test.base import RunsTest


class RunsTestRust(RunsTest):
    """Rust extension `runs_test`: one pass per column comparing neighbours natively and
    hashing once per run (not per row); rayon-parallel across columns and 2^16-row
    chunks, whose partials merge in row order. The variance numerator is exact (i128).
    """

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.runs_test(
                        frames[frame].select([c for ((_, c),) in group])
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
