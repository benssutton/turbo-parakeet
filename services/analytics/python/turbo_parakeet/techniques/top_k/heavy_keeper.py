from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.top_k.base import TopK


class TopKHeavyKeeperRust(TopK):
    """Rust extension `heavy_keeper_top_k`: HeavyKeeper (Gong et al., ATC 2018) with a heap of
    k and 2 × width (fingerprint, count) buckets filling the rest of `memory_bytes`; columns in
    parallel (rayon), each fed in row order (the sketch does not merge). Seeded per column
    from `seed` and the column name; no per-item bound (error null)."""

    EXACT = False
    MAX_ARE = 0.05

    def __init__(self, *, seed: int = 0, **params):
        super().__init__(**params)
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError(f"seed must be an integer in [0, 2**64 - 1], got {seed!r}")
        self.seed = seed

    def _compute(self, frames, combos):
        ranked = {
            frame: _plugin.heavy_keeper_top_k(
                frames[frame].select([c for ((_, c),) in group]),
                self.k,
                self.memory_bytes,
                self.seed,
            )
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)
