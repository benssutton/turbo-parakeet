from synthdata.cover.factors import normalise
from synthdata.cover.pairwise import allowed_pairs, candidates, covering, pairs


def test_covering_covers_every_allowed_pair_and_only_allowed_columns():
    cols = covering()
    covered = set().union(*(pairs(c) for c in cols))
    assert covered == allowed_pairs()
    assert all(normalise(c) == c for c in cols)
    # Lower bound 58: a family needs a column per cardinality level, one all-null, and
    # (integer-backed families) one walk; boolean needs its four orders.
    assert 55 <= len(cols) <= 65, len(cols)


def test_covering_is_deterministic():
    assert covering() == covering()


def test_pairs_skip_factors_that_do_not_apply():
    c = dict(
        family="int",
        cardinality=None,
        skew=None,
        nulls="all",
        order=None,
        structure=None,
    )
    assert pairs(c) == {(("family", "int"), ("nulls", "all"))}
    assert c in candidates()
