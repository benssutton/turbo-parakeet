from analytics import _plugin
from analytics.base import group_by_frame
from analytics.gcd.base import Gcd


class GcdRust(Gcd):
    """Rust plugin `column_gcd`: rayon-parallel across columns and 64K-value chunks;
    binary GCD after one hardware remainder per value; early exit once the GCD is 1."""

    def _compute(self, frames, combos):
        gcd = {}
        for frame, group in group_by_frame(combos).items():
            columns = [c for ((_, c),) in group]
            out = _plugin.column_gcd(frames[frame].select(columns))
            gcd.update(((frame, c), g) for c, g in zip(out["column"].to_list(), out["gcd"].to_list()))
        return self.metrics_frame(combos, {"gcd": [gcd[k[0]] for k in combos]})
