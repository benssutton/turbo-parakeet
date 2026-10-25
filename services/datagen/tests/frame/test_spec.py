import pytest

from synthdata import SpecError
from synthdata.frame.spec import (
    ColumnSpec,
    CompositeKey,
    ForeignKey,
    FrameSpec,
    Multiples,
    NoisyCopy,
    Runs,
    Walk,
    frame_id,
)


def test_valid_frame():
    FrameSpec(
        rows=100,
        columns={"a": ColumnSpec("int64", distinct=10), "b": NoisyCopy("a", 0.1)},
    ).validate()


@pytest.mark.parametrize(
    "columns, match",
    [
        ({"a": ColumnSpec("nope", distinct=1)}, "type"),
        ({"a": ColumnSpec("int64")}, "exactly one of distinct"),
        (
            {"a": ColumnSpec("int64", distinct=2, distinct_share=0.5)},
            "exactly one of distinct",
        ),
        ({"a": ColumnSpec("int64", distinct=0)}, "distinct"),
        ({"a": ColumnSpec("int64", distinct=5, null_share=1.5)}, "null_share"),
        ({"a": ColumnSpec("string", distinct=5, structure=Multiples(10))}, "structure"),
        (
            {"a": ColumnSpec("int64", distinct=5, structure=Walk(), order=Runs())},
            "walk",
        ),
        ({"a": ColumnSpec("int64", distinct=5, entropy=1.5)}, "entropy"),
        ({"b": NoisyCopy("a", 0.1)}, "parent"),
        (
            {
                "a": ColumnSpec("int64", distinct=5),
                "b": ForeignKey("a", null_share=2.0),
            },
            "null_share",
        ),
        (
            {
                "a": ColumnSpec("int64", distinct=5, null_share=0.1),
                "b": CompositeKey("a"),
            },
            "nulls",
        ),
    ],
)
def test_invalid(columns, match):
    with pytest.raises(SpecError, match=match):
        FrameSpec(rows=100, columns=columns).validate()


def test_frame_id_depends_on_content_only():
    a = FrameSpec(rows=10, columns={"a": ColumnSpec("int64", distinct=3)}, seed=1)
    b = FrameSpec(rows=10, columns={"a": ColumnSpec("int64", distinct=3)}, seed=1)
    c = FrameSpec(rows=10, columns={"a": ColumnSpec("int64", distinct=4)}, seed=1)
    assert frame_id(a) == frame_id(b) != frame_id(c)
