import polars as pl
import pytest

from synthdata.frame import (
    ColumnSpec,
    ForeignKey,
    FrameManifest,
    FrameSpec,
    Relabel,
    Sorted,
    TargetMissed,
    generate_frame,
)


def spec(**columns):
    return FrameSpec(rows=1_000, columns=columns, seed=5)


def test_frame_and_manifest():
    df, man = generate_frame(
        spec(
            a=ColumnSpec("categorical", distinct=10, null_share=0.1),
            b=Relabel("a", "int64"),
            c=ColumnSpec("decimal", distinct_share=0.95, order=Sorted()),
        )
    )
    assert df.columns == ["a", "b", "c"] and df.height == 1_000
    assert df.schema["a"] == pl.Categorical and df.schema["c"] == pl.Decimal(12, 2)
    assert man.columns["a"]["null_count"] == 100 and man.columns["a"]["distinct"] == 10
    assert man.columns["c"]["descents"] == 0
    assert abs(man.relations["b"]["nmi"] - 1) < 1e-9
    assert FrameManifest.from_json(man.to_json()) == man


def test_a_column_does_not_depend_on_its_siblings():
    one, _ = generate_frame(spec(x=ColumnSpec("int64", distinct=30)))
    two, _ = generate_frame(
        spec(y=ColumnSpec("string", distinct=4), x=ColumnSpec("int64", distinct=30))
    )
    assert one["x"].equals(two["x"])


def test_frame_level_order():
    df, man = generate_frame(
        FrameSpec(
            rows=500,
            columns={
                "k": ColumnSpec("int64", distinct=20),
                "v": ColumnSpec("string", distinct=50),
            },
            order_by=("k",),
            seed=1,
        )
    )
    assert df["k"].is_sorted()


def test_strict_raises_on_a_missed_target():
    with pytest.raises(TargetMissed, match="entropy"):
        generate_frame(
            spec(a=ColumnSpec("int64", distinct=10, entropy=0.01)), strict=True
        )
    generate_frame(
        spec(a=ColumnSpec("int64", distinct=10, entropy=0.01))
    )  # best effort: no raise


def test_strict_checks_relations():
    _, man = generate_frame(
        spec(
            p=ColumnSpec("string", distinct_share=1.0),
            f=ForeignKey("p", null_share=0.2),
        ),
        strict=True,
    )
    assert man.relations["f"]["containment"] == 1.0
