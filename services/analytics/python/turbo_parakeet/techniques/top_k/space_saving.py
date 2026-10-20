from turbo_parakeet import _plugin
from turbo_parakeet.techniques.base import group_by_frame
from turbo_parakeet.techniques.top_k.base import TopK


class TopKSpaceSavingRust(TopK):
    """Rust extension `space_saving_top_k`: SpaceSaving with as many counters as fit
    `memory_bytes` (~40–56 B each); 2^20-row chunks summarised in parallel and merged in
    order (rayon across columns and chunks). Deterministic; true count ∈ [count − error,
    count] for every item, and every value above n / counters is found."""

    EXACT = False

    def _compute(self, frames, combos):
        ranked = {
            frame: _plugin.space_saving_top_k(
                frames[frame].select([c for ((_, c),) in group]),
                self.k,
                self.memory_bytes,
            )
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)

    def _item_problems(self, label, item, true, floor):
        lower, upper = item["count"] - item["error"], item["count"]
        if true is not None and not lower <= true <= upper:
            return [
                f"{label}: {item['value']!r} true count {true} outside [{lower}, {upper}]"
            ]
        if true is None and lower > floor:
            return [
                f"{label}: {item['value']!r} lower bound {lower} above the k-th count {floor}"
            ]
        return []
