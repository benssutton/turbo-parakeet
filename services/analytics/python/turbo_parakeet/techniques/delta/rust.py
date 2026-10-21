import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.delta.base import Delta


class DeltaRust(Delta):
    """Rust extension `delta`: one pass per column over the non-null raw values, rayon-parallel
    across columns and chunks; the expected-bytes baseline from the sorted values."""

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.delta(
                        frames[frame].select([c for ((_, c),) in group])
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
