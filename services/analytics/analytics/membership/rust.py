import math

import polars as pl

from analytics import _plugin
from analytics.base import columns_of
from analytics.membership.base import BloomMembership


def bloom_geometry(n: int, fp_rate: float) -> tuple[int, int]:
    """Optimal (m bits, k hashes) for n items: m = -n·ln(p)/ln(2)², rounded up to
    whole bytes (the plugin expects m in bits and ceil(m/8) bytes); k = (m/n)·ln 2."""
    m = -(n * math.log(fp_rate)) / (math.log(2) ** 2)
    m = (int(math.ceil(m)) + 7) // 8 * 8
    return m, max(1, int(math.ceil((m / n) * math.log(2))))


class BloomRust(BloomMembership):
    """Rust plugin: one Bloom filter per column over its distinct values (`bloom_filter`),
    then `membership_ratio` checks all partner columns against it in one rayon-parallel
    call (partners' distinct values null-padded to one frame; ratio_non_null is the
    distinct-value ratio)."""

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        distinct = {c: frames[c[0]][c[1]].drop_nulls().unique() for c in columns}
        partners: dict = {c: [] for c in columns}
        for a, b in combos:
            partners[a].append(b)
            partners[b].append(a)
        contained = {}
        for y in columns:
            queries = [x for x in partners[y] if distinct[x].len()]
            if not distinct[y].len() or not queries:
                continue
            m, k = bloom_geometry(distinct[y].len(), self.fp_rate)
            bits = _plugin.bloom_filter_bits(distinct[y], k=k, m=m)
            longest = max(distinct[x].len() for x in queries)
            padded = pl.DataFrame(
                [distinct[x].extend_constant(None, longest - distinct[x].len()).alias(f"q{i}") for i, x in enumerate(queries)]
            )
            ratios = _plugin.membership_ratio(padded, bit_array_bytes=bits, k=k, m=m).unnest("membership_ratio")
            contained.update(((x, y), r) for x, r in zip(queries, ratios["ratio_non_null"].to_list()))
        return self.membership_rows(frames, combos, {c: distinct[c].len() for c in columns}, contained)
