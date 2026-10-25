from synthdata.cache import load_or_generate_frame
from synthdata.frame import ColumnSpec, FrameSpec


def test_cached_round_trip(tmp_path):
    spec = FrameSpec(
        rows=300,
        columns={
            "a": ColumnSpec("categorical", distinct=7),
            "b": ColumnSpec("int64", distinct=30),
        },
        seed=2,
    )
    df1, m1 = load_or_generate_frame(spec, directory=tmp_path)
    files = sorted(p.suffix for p in tmp_path.iterdir())
    assert files == [".arrow", ".json"]
    df2, m2 = load_or_generate_frame(spec, directory=tmp_path)
    assert df1.equals(df2) and m1 == m2 and df2.schema == df1.schema
    other = FrameSpec(rows=301, columns=spec.columns, seed=2)
    load_or_generate_frame(other, directory=tmp_path)
    assert len(list(tmp_path.iterdir())) == 4
