"""
Correctness tests for the Rust joint-entropy plugin.

Uses the shared 18-column test DataFrame from conftest.py (6 dtypes × 3 shapes,
N_ROWS rows). See conftest.make_dataset for the column layout.

pairwise_joint_entropy  → C(18,2) = 153 pairs
threeway_joint_entropy  → C(18,3) = 816 triplets
marginal_entropy        → 18 columns

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
marginal_entropy = _analytics.marginal_entropy


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


def reference_marginal_entropy(df: pl.DataFrame, col_name: str) -> float:
    """H(col_name) in bits.

    A single-field struct's entropy is exactly that field's marginal entropy,
    so this reuses reference_entropy's struct-based grouping and nested-type
    fallback rather than reimplementing null/List-column handling separately
    (Series.cast can't stringify a List directly in this Polars version, but
    reference_entropy's struct-first path already handles it).
    """
    return reference_entropy(df, [col_name])


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


def assert_all_marginals(result: pl.DataFrame, source: pl.DataFrame) -> None:
    rows = result.unnest(result.columns[0])
    failures = []
    for row in rows.iter_rows(named=True):
        col_name, h_plugin = row["col_name"], row["entropy"]
        h_ref = reference_marginal_entropy(source, col_name)
        if not math.isclose(h_plugin, h_ref, rel_tol=1e-5):
            failures.append(
                f"  {col_name}: plugin={h_plugin:.8f} ref={h_ref:.8f}"
                f"  diff={abs(h_plugin - h_ref):.3e}"
            )
    if failures:
        raise AssertionError("Entropy mismatch for columns:\n" + "\n".join(failures))


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_pairwise_entropy(dataset):
    """C(18,2) = 153 pairs across all dtype × shape combinations."""
    result = pairwise_joint_entropy(dataset)
    assert_all_pairs(result, dataset)


@pytest.mark.slow
def test_pairwise_entropy_subadditivity(dataset):
    """H(A,B) <= H(A) + H(B) must hold for every pair (subadditivity of joint
    entropy). A violation is mathematically impossible for a correct
    implementation and indicates a bug in the plugin's counting, not a
    tolerance issue — so the bound uses a tight epsilon for floating-point
    slack only, not a relative tolerance.
    """
    result = pairwise_joint_entropy(dataset)
    rows = result.unnest(result.columns[0])
    failures = []
    for row in rows.iter_rows(named=True):
        col_a, col_b, h_joint = row["col_a"], row["col_b"], row["entropy"]
        h_a = reference_marginal_entropy(dataset, col_a)
        h_b = reference_marginal_entropy(dataset, col_b)
        bound = h_a + h_b
        if h_joint > bound + 1e-6:
            failures.append(
                f"  ({col_a}, {col_b}): H(A,B)={h_joint:.8f} > H(A)+H(B)={bound:.8f}"
                f"  (H(A)={h_a:.8f}, H(B)={h_b:.8f}, excess={h_joint - bound:.3e})"
            )
    if failures:
        raise AssertionError("Subadditivity violated for pairs:\n" + "\n".join(failures))


@pytest.mark.slow
def test_pairwise_entropy_subadditivity_rust_marginal(dataset):
    """H(A,B) <= H(A) + H(B) must hold for every pair, using the Rust plugin's
    own marginal_entropy output for H(A)/H(B) instead of the Polars reference.
    This keeps the comparison entirely inside the Rust plugin: joint entropy
    is computed via the dense-id/flat-array counting path (joint_entropy_pair)
    while marginal entropy is computed via an independent HashMap-based path
    (see marginal_entropy_impl's docstring in entropy.rs) — a violation here
    points at a mismatch between those two Rust paths specifically, not at
    Polars vs Rust disagreement (that's test_pairwise_entropy_subadditivity).
    """
    joint_result = pairwise_joint_entropy(dataset)
    marginal_result = marginal_entropy(dataset)

    marginal_rows = marginal_result.unnest(marginal_result.columns[0])
    h_marginal = {
        row["col_name"]: row["entropy"] for row in marginal_rows.iter_rows(named=True)
    }

    joint_rows = joint_result.unnest(joint_result.columns[0])
    failures = []
    for row in joint_rows.iter_rows(named=True):
        col_a, col_b, h_joint = row["col_a"], row["col_b"], row["entropy"]
        h_a = h_marginal[col_a]
        h_b = h_marginal[col_b]
        bound = h_a + h_b
        if h_joint > bound + 1e-6:
            failures.append(
                f"  ({col_a}, {col_b}): H(A,B)={h_joint:.8f} > H(A)+H(B)={bound:.8f}"
                f"  (H(A)={h_a:.8f}, H(B)={h_b:.8f}, excess={h_joint - bound:.3e})"
            )
    if failures:
        raise AssertionError(
            "Subadditivity violated vs Rust marginal entropy for pairs:\n"
            + "\n".join(failures)
        )


@pytest.mark.slow
def test_marginal_entropy(dataset):
    """H(col) for every one of the 18 columns in the shared dataset."""
    result = marginal_entropy(dataset)
    assert_all_marginals(result, dataset)


@pytest.mark.slow
def test_threeway_entropy(dataset):
    """C(18,3) = 816 triplets across all dtype × shape combinations."""
    result = threeway_joint_entropy(dataset)
    assert_all_triplets(result, dataset)
