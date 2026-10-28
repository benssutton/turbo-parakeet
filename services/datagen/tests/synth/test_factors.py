import pytest

from synthdata.synth.factors import FACTORS, column_spec, normalise
from synthdata.frame import ColumnSpec, FrameSpec

FULL = dict(
    family="int",
    cardinality="low",
    skew="heavy",
    nulls="sparse",
    order="runs",
    structure="plain",
)


def test_normalise_rules():
    assert normalise(FULL) == FULL
    assert normalise({**FULL, "family": "boolean"}) is None
    b = normalise(
        {**FULL, "family": "boolean", "cardinality": "binary", "structure": "plain"}
    )
    assert b["structure"] is None
    assert normalise({**FULL, "family": "string", "structure": "walk"}) is None
    n = normalise({**FULL, "nulls": "all"})
    assert n == dict(
        family="int",
        cardinality=None,
        skew=None,
        nulls="all",
        order=None,
        structure=None,
    )
    w = normalise({**FULL, "structure": "walk"})
    assert w["cardinality"] is None and w["skew"] is None and w["order"] is None
    assert normalise({**FULL, "cardinality": "near_unique", "order": "runs"}) is None
    assert normalise({**FULL, "cardinality": "constant"})["skew"] is None


@pytest.mark.parametrize("family", FACTORS["family"])
def test_every_family_maps_to_a_valid_column(family):
    levels = normalise({**FULL, "family": family, "cardinality": "binary"})
    FrameSpec(rows=2_000, columns={"c": column_spec(levels, 0, 2_000)}).validate()
    assert isinstance(column_spec(levels, 0, 2_000), ColumnSpec)
