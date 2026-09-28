from analytics import _plugin
from analytics.base import group_by_frame
from analytics.describe.base import Describe


class DescribeRust(Describe):
    """Rust extension: `describe_columns` (one pass per column; rayon across columns and
    64K-row chunks; hash-map frequencies, byte scanners, row-encoded extremes) and
    `column_sizes` (Arrow buffer walk + zstd, mirroring pyarrow's IPC writer)."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            stats = _plugin.describe_columns(df, self.seed).join(
                _plugin.column_sizes(df, self.zstd_level), on="column"
            )
            for r in stats.iter_rows(named=True):
                rows[frame, r["column"]] = r
        return self.metrics_frame(
            combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS}
        )
