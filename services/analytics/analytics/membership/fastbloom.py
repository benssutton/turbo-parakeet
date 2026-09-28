from fastbloom_rs import BloomFilter as FastBloomFilter

from analytics._sets import distinct_values
from analytics.base import columns_of
from analytics.membership.base import BloomMembership


class BloomFastbloom(BloomMembership):
    """fastbloom-rs (Rust-backed Bloom filter, one column at a time from Python);
    values keyed by their string form."""

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        keys = {
            c: [str(v) for v in distinct_values(frames[c[0]][c[1]])] for c in columns
        }
        partners: dict = {c: [] for c in columns}
        for a, b in combos:
            partners[a].append(b)
            partners[b].append(a)
        contained = {}
        for y in columns:
            if not keys[y]:
                continue
            bloom = FastBloomFilter(len(keys[y]), self.fp_rate)
            bloom.add_str_batch(keys[y])
            for x in partners[y]:
                if keys[x]:
                    contained[(x, y)] = sum(bloom.contains_str_batch(keys[x])) / len(
                        keys[x]
                    )
        return self.membership_rows(
            frames, combos, {c: len(k) for c, k in keys.items()}, contained
        )
