import polars as pl

from analytics.base import group_by_frame
from analytics.top_k.base import RANKED, TopK


class TopKPolars(TopK):
    """Reference: exact counts per column (Polars group_by), ranked by count desc then first
    row. Holds every distinct value: memory grows with the column's cardinality."""

    def _compute(self, frames, combos):
        ranked = {
            frame: pl.concat([self._ranked(frames[frame][c]) for ((_, c),) in group])
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)

    def _ranked(self, s: pl.Series) -> pl.DataFrame:
        n = s.len() - s.null_count()
        rows = (
            pl.DataFrame({"v": s})
            .with_row_index("row")
            .drop_nulls("v")
            .group_by("v")
            .agg(count=pl.len(), row=pl.col("row").min())
            .sort(["count", "row"], descending=[True, False])
            .head(self.k)
            .select("row", "count")
            .rows()
        )
        if not rows:
            rows = [(None, None)]
        ranks = [None] if rows[0][0] is None else list(range(1, len(rows) + 1))
        return pl.DataFrame(
            {
                "col_name": [s.name] * len(rows),
                "rank": ranks,
                "row": [r for r, _ in rows],
                "count": [c for _, c in rows],
                "error": [None if r is None else 0 for r, _ in rows],
                "n_non_null": [n] * len(rows),
                "sketch_bytes": [None] * len(rows),
            },
            schema=RANKED,
        )
