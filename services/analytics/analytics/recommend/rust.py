from analytics import _plugin
from analytics.base import group_by_frame
from analytics.recommend.base import Recommend


class RecommendRust(Recommend):
    """Rust extension `describe_and_recommend`: Describe's metrics and sizes, then the
    candidate types cast, verified and measured with arrow-rs (src/recommend.rs)."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            pop = self._population(frame)
            if pop is not None and pop < df.height:
                raise ValueError(f"population_rows {pop} < {df.height} rows in frame {frame!r}")
            out = _plugin.describe_and_recommend(
                df,
                seed=self.seed,
                zstd_level=self.zstd_level,
                population_rows=pop,
                categorical_threshold=self.categorical_threshold,
                boolean_pairs=self.boolean_pairs,
            )
            for r in out.iter_rows(named=True):
                if r["rec_arrow_type"] is not None and r["rec_polars_type"] is None:  # the original type was kept
                    r["rec_polars_type"] = str(df.schema[r["column"]])
                rows[frame, r["column"]] = r
        return self.metrics_frame(combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS})
