"""
Correctness tests for the Rust joint-entropy plugin.

Uses the shared 18-column test DataFrame from conftest.py (6 dtypes × 3 shapes,
N_ROWS rows). See conftest.make_dataset for the column layout.

pairwise_joint_entropy  → C(18,2) = 153 pairs
threeway_joint_entropy  → C(18,3) = 816 triplets

Every result row is verified against the Polars reference:
    pl.struct(cols).value_counts().get_column("count").entropy(base=2, normalize=True)

Boolean columns only have 2 distinct values (True/False) + None — cardinality
assertions are skipped for that dtype; entropy correctness assertions still hold.
"""

import math

import polars as pl
import pytest

_analytics = pytest.importorskip("analytics")
pairwise_joint_entropy = _analytics.pairwise_joint_entropy
threeway_joint_entropy = _analytics.threeway_joint_entropy


# ── Reference implementation ──────────────────────────────────────────────────

def _is_nested(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.List, pl.Array))


def reference_entropy(df: pl.DataFrame, col_names: list[str]) -> float:
    """H(col_names) in bits via Polars value_counts.

    Falls back to casting List/Array cols to String if Polars cannot group structs
    containing nested-type fields (behaviour is version-dependent).
    """
    def _compute(frame: pl.DataFrame) -> float:
        key = frame.select(pl.struct(col_names).alias("k")).to_series()
        return key.value_counts().get_column("count").entropy(base=2, normalize=True)

    try:
        return _compute(df)
    except Exception:
        nested_cols = {c for c in col_names if _is_nested(df.schema[c])}
        if not nested_cols:
            raise
        cast_df = df.with_columns([pl.col(c).cast(pl.String) for c in nested_cols])
        return _compute(cast_df)


# ── Assertion helpers ─────────────────────────────────────────────────────────

def assert_all_pairs(result: pl.DataFrame, source: pl.DataFrame) -> None:
    rows = result.unnest(result.columns[0])
    failures = []
    for row in rows.iter_rows(named=True):
        col_a, col_b, h_plugin = row["col_a"], row["col_b"], row["entropy"]
        h_ref = reference_entropy(source, [col_a, col_b])
        if not math.isclose(h_plugin, h_ref, rel_tol=1e-5):
            failures.append(
                f"  ({col_a}, {col_b}): plugin={h_plugin:.8f} ref={h_ref:.8f}"
                f"  diff={abs(h_plugin - h_ref):.3e}"
            )
    if failures:
        raise AssertionError("Entropy mismatch for pairs:\n" + "\n".join(failures))


def assert_all_triplets(result: pl.DataFrame, source: pl.DataFrame) -> None:
    rows = result.unnest(result.columns[0])
    failures = []
    for row in rows.iter_rows(named=True):
        col_a, col_b, col_c = row["col_a"], row["col_b"], row["col_c"]
        h_plugin = row["entropy"]
        h_ref = reference_entropy(source, [col_a, col_b, col_c])
        if not math.isclose(h_plugin, h_ref, rel_tol=1e-5):
            failures.append(
                f"  ({col_a}, {col_b}, {col_c}): plugin={h_plugin:.8f} ref={h_ref:.8f}"
                f"  diff={abs(h_plugin - h_ref):.3e}"
            )
    if failures:
        raise AssertionError("Entropy mismatch for triplets:\n" + "\n".join(failures))


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_pairwise_entropy(dataset):
    """C(18,2) = 153 pairs across all dtype × shape combinations."""
    result = pairwise_joint_entropy(dataset)
    assert_all_pairs(result, dataset)


@pytest.mark.slow
def test_threeway_entropy(dataset):
    """C(18,3) = 816 triplets across all dtype × shape combinations."""
    result = threeway_joint_entropy(dataset)
    assert_all_triplets(result, dataset)
