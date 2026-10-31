"""The engine-neutral half of a filter (`ValueChoice`) and the per-engine renderers. The ClickHouse
output is pinned by test_queries.py, unchanged."""

import numpy as np
import pytest

from chbench.harness.queries import (
    ValueChoice,
    choose_in,
    choose_queries,
    choose_range,
    expand_queries,
    in_predicate,
    range_predicate,
    render_clickhouse,
    render_queries,
)
from chbench.harness.suite import QueryAxes, QueryKind
from tests.harness.test_queries import variant

AXES = QueryAxes(
    kinds=[
        QueryKind("full_scan", None, None, None),
        QueryKind("filter", 0.1, "range", None),
        QueryKind("filter", 0.05, "in", 2),
    ],
    samples=1,
)


def _vals():
    return np.round(np.random.default_rng(0).uniform(-1, 10, 10_000), 2)


def test_one_choice_renders_the_old_clickhouse_predicates():
    vals = _vals()
    c = choose_range(vals, 0.1, np.random.default_rng(3))
    assert c.predicate_kind == "range" and len(c.values) == 2
    assert range_predicate(vals, 0.1, np.random.default_rng(3), ch_type="Float64") == (
        render_clickhouse(c, "Float64"),
        c.actual_fraction,
    )
    i = choose_in(vals, 0.01, 3)
    assert i.predicate_kind == "in" and len(i.values) == 3
    assert in_predicate(vals, 0.01, 3) == (render_clickhouse(i), i.actual_fraction)


def test_choose_queries_is_what_expand_queries_renders():
    vals = _vals()
    choices = choose_queries(AXES, vals, np.random.default_rng(7))
    assert choices[0] is None
    assert [c.predicate_kind for c in choices[1:]] == ["range", "in"]
    assert render_queries(AXES, choices, variant()) == expand_queries(
        AXES, vals, variant(), np.random.default_rng(7)
    )


def test_a_value_choice_is_frozen():
    c = ValueChoice("in", np.array([1.0]), 0.5)
    with pytest.raises(AttributeError):
        c.actual_fraction = 0.1
