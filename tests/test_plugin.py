"""The Arrow boundary: 128-bit integers, the Python binding, and Arrow inputs."""

import importlib

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from analytics import analytics as rs
from analytics._dtypes import holds_wide_integer
from datagen import mixed_dtypes
from harness import load, run

PACKAGES = (
    "analytics.gcd", "analytics.describe", "analytics.recommend", "analytics.membership",
    "analytics.similarity", "analytics.chi_squared", "analytics.pairwise_entropy",
    "analytics.threeway_entropy", "analytics.adjusted_rand",
)
EVERY_IMPLEMENTATION = [
    pytest.param(f"{p}:{n}", id=n) for p in PACKAGES for n in importlib.import_module(p).IMPLEMENTATIONS
]


def wide_integer_frame() -> tuple[pl.DataFrame, list[str]]:
    cols = {
        "ok_a": [1, 2, 3],
        "ok_b": [3, 2, 1],
        "i128": pl.Series([1, 2, 3], dtype=pl.Int128),
        "list_i128": pl.Series([[1], [2], [3]], dtype=pl.List(pl.Int128)),
    }
    if hasattr(pl, "UInt128"):
        cols["u128"] = pl.Series([1, 2, 3], dtype=pl.UInt128)
    return pl.DataFrame(cols), [c for c in cols if not c.startswith("ok_")]


def test_holds_wide_integer_detects_nested_dtype_classes():
    assert all(holds_wide_integer(d) for d in (pl.Int128, pl.List(pl.Int128), pl.Struct({"a": pl.Int128})))


@pytest.mark.parametrize("spec", EVERY_IMPLEMENTATION)
def test_128_bit_integer_columns_are_ineligible(spec):
    frame, wide = wide_integer_frame()
    out = run(load(spec), {"t": frame})
    names = [c for c in out.columns if c.startswith("col_")]
    touches_wide = out.filter(pl.any_horizontal(pl.col(c).is_in(wide) for c in names))
    assert touches_wide.height > 0
    assert touches_wide["status"].unique().to_list() == ["ineligible"]


# ── the binding (analytics.analytics) ─────────────────────────────────────────


def test_binding_takes_and_returns_arrow():
    out = pl.DataFrame(rs.column_gcd(pl.DataFrame({"a": [12, 18], "b": [7, 14]})))
    assert out.columns == ["column", "dtype", "gcd"]
    assert out["gcd"].to_list() == [6, 7]
    assert pa.table(rs.column_gcd(pa.table({"a": [12, 18]}))).num_rows == 1


def test_multi_chunk_table_is_accepted():
    table = pa.concat_tables([pa.table({"a": [12, 18]}), pa.table({"a": [24]})])
    assert pa.table(rs.column_gcd(table))["gcd"].to_pylist() == [6]


def test_multi_batch_record_batch_reader_is_accepted():
    table = pa.concat_tables([pa.table({"a": [12, 18]}), pa.table({"a": [24]})])
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    assert pa.table(rs.column_gcd(reader))["gcd"].to_pylist() == [6]


def test_result_table_can_be_exported_twice():
    out = rs.column_gcd(pl.DataFrame({"a": [12, 18]}))
    as_arrow = pa.table(out)
    as_polars = pl.DataFrame(out)
    assert as_arrow["gcd"].to_pylist() == as_polars["gcd"].to_list()


def test_bloom_bits_are_bytes():
    bits = rs.bloom_filter(pl.DataFrame({"a": [1, 2, 3]}), 3, 64)
    assert isinstance(bits, bytes) and len(bits) == 8
    ratios = pl.DataFrame(rs.membership_ratio(pl.DataFrame({"a": [1, 2, 3]}), bits, 3, 64))
    assert ratios["ratio_non_null"].to_list() == [1.0]


def test_unknown_pair_column_is_value_error():
    with pytest.raises(ValueError, match='unknown column "nope"'):
        rs.pairwise_joint_entropy(pl.DataFrame({"a": [1, 2]}), [("a", "nope")])


def test_wrong_bloom_length_is_value_error():
    with pytest.raises(ValueError, match="requires 8 bytes"):
        rs.membership_ratio(pl.DataFrame({"a": [1]}), b"\x00", 3, 64)


def test_wrong_column_type_is_value_error():
    # api.rs classifies kernel type errors (SchemaMismatch etc.) as InvalidInput; the
    # Compute → RuntimeError mapping is covered by api.rs's unit test of `compute`.
    with pytest.raises(ValueError):
        rs.lsh_candidates(pl.DataFrame({"qualified_name": ["a"], "minhash": [1]}), 1, 1)


def test_zero_bloom_parameters_are_value_errors():
    with pytest.raises(ValueError):
        rs.bloom_filter(pl.DataFrame({"a": [1]}), 0, 64)


def test_non_arrow_input_is_type_error():
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        rs.column_gcd(42)


WIDE = [
    pl.Series("x", [1], dtype=pl.Int128),
    pl.Series("x", [[1]], dtype=pl.List(pl.Int128)),
    pl.Series("x", [{"v": 1}], dtype=pl.Struct({"v": pl.Int128})),
]
if hasattr(pl, "UInt128"):
    WIDE.append(pl.Series("x", [1], dtype=pl.UInt128))


@pytest.mark.parametrize("s", WIDE, ids=lambda s: str(s.dtype))
def test_128_bit_integers_are_rejected_at_the_binding(s):
    with pytest.raises(ValueError, match='column "x" holds .*128-bit'):
        rs.column_gcd(s.to_frame())


# ── Arrow inputs to the technique classes ─────────────────────────────────────

RUST_AND_REFERENCE = [
    pytest.param(f"{p}:{n}", id=n)
    for p in PACKAGES
    for n in importlib.import_module(p).IMPLEMENTATIONS
    if n.endswith("Rust") or n == importlib.import_module(p).REFERENCE
]


@pytest.mark.parametrize("kind", ["table", "record_batch", "reader"])
@pytest.mark.parametrize("spec", RUST_AND_REFERENCE)
def test_arrow_inputs_match_polars(spec, kind):
    df = mixed_dtypes(200, seed=1)
    table = pa.table(df)  # through __arrow_c_stream__: keeps Polars' Categorical metadata
    arrow = {"table": table, "record_batch": table.combine_chunks().to_batches()[0], "reader": table.to_reader()}[kind]
    cls = load(spec)
    assert_frame_equal(run(cls, {"t": arrow}), run(cls, {"t": df}))
