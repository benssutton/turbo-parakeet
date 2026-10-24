from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.population_cardinality.base import PopulationCardinality


class PopulationCardinalityRust(PopulationCardinality):
    """Rust extension `population_cardinality`: Describe's frequency pass (exact counts,
    capture masks over a seeded 3-way row split) and its Chao1 / Schnabel formulas; one
    binding call per frame."""

    def _compute(self, frames, combos):
        rows = {}
        for frame, group in group_by_frame(combos).items():
            out = _plugin.population_cardinality(
                frames[frame].select([c for ((_, c),) in group]), self.seed
            )
            rows.update(((frame, r["col_name"]), r) for r in out.iter_rows(named=True))
        cols = [rows[k] for (k,) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in cols] for m in self.METRICS}
        )
