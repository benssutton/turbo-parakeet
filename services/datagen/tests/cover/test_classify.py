import polars as pl

from synthdata.cover.classify import classify, coverage, coverage_of, family_of
from synthdata.cover.factors import column_spec, normalise
from synthdata.frame import FrameSpec, generate_frame

FULL = dict(
    family="int",
    cardinality="low",
    skew="heavy",
    nulls="sparse",
    order="runs",
    structure="plain",
)


def test_generated_columns_classify_as_intended():
    variants = [
        FULL,
        {**FULL, "order": "sorted"},
        {**FULL, "order": "clustered", "skew": "uniform"},
        {**FULL, "cardinality": "near_unique", "order": "random"},
        {**FULL, "structure": "walk"},
        {**FULL, "structure": "multiples", "nulls": "none"},
        {**FULL, "family": "boolean", "cardinality": "binary"},
        {**FULL, "nulls": "all"},
        # A skewed binary column clustered: its rho is bounded near 0.5.
        {**FULL, "family": "string", "cardinality": "binary", "order": "clustered"},
    ]
    intended = {f"c{i}": normalise(v) for i, v in enumerate(variants)}
    columns = {
        n: column_spec(lv, i, 2_000) for i, (n, lv) in enumerate(intended.items())
    }
    _, man = generate_frame(FrameSpec(rows=2_000, columns=columns), strict=True)
    cov = coverage(man, intended)
    assert cov.mismatches == {}, cov.mismatches


def test_family_of_polars_dtypes():
    assert family_of(pl.Int32) == "int" and family_of(pl.UInt8) == "uint"
    assert (
        family_of(pl.Date) == "temporal" and family_of(pl.Categorical) == "categorical"
    )
    assert family_of(pl.List(pl.Int64)) is None


def test_coverage_of_any_frame():
    df = pl.DataFrame({"a": [1, 2, 3, 4] * 50, "s": ["x", "y"] * 100})
    cov = coverage_of(df)
    assert cov.measured["a"]["family"] == "int"
    assert cov.measured["s"]["cardinality"] == "binary"
    assert cov.covered and cov.missing


def test_classify_constant():
    stats = dict(
        rows=10,
        null_count=0,
        distinct=1,
        entropy_norm=float("nan"),
        rho=float("nan"),
        run_ratio=float("nan"),
        descents=0,
        gcd=5,
        max_step=0,
    )
    assert classify(stats, "int")["cardinality"] == "constant"
