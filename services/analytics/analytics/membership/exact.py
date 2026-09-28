from analytics._sets import distinct_values
from analytics.base import columns_of
from analytics.membership.base import Membership


class MembershipExact(Membership):
    """Reference: exact containment from Python sets of distinct non-null values."""

    def _compute(self, frames, combos):
        sets = {
            c: frozenset(distinct_values(frames[c[0]][c[1]]))
            for c in columns_of(combos)
        }
        contained = {}
        for a, b in combos:
            shared = len(sets[a] & sets[b])
            if sets[a]:
                contained[(a, b)] = shared / len(sets[a])
            if sets[b]:
                contained[(b, a)] = shared / len(sets[b])
        return self.membership_rows(
            frames, combos, {c: len(s) for c, s in sets.items()}, contained
        )
