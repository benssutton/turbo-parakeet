"""
Pairwise Adjusted Rand Index correctness tests: Rust plugin vs scikit-learn.

Eligible columns from the shared 18-column dataset (see conftest.make_dataset):
    boolean_*   (3 columns) — Boolean
    uint32_*    (3 columns) — UInt32
    categorical_* (3 columns) — Categorical
    float64_*   (3 columns) — Float64 (discrete labels; ARI is defined for any labelling)

Excluded: list_* and arr_* (nested types; sklearn cannot label them).

12 eligible columns → C(12,2) = 66 pairs tested.

The reference applies the plugin's null policy (drop rows where either column
is null) before calling sklearn.metrics.adjusted_rand_score. Labels are cast
to String so every dtype becomes uniform hashable labels — an injective
relabelling, which ARI is invariant to.

Skipped automatically if scikit-learn is not installed.
"""

import math

import polars as pl
import pytest

sk_metrics = pytest.importorskip("sklearn.metrics")

_analytics = pytest.importorskip("analytics")
pairwise_adjusted_rand = _analytics.pairwise_adjusted_rand


_ARI_ELIGIBLE_PREFIXES = ("boolean_", "uint32_", "categorical_", "float64_")


def ari_eligible_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(_ARI_ELIGIBLE_PREFIXES)]


def sklearn_ari(df: pl.DataFrame, col_a: str, col_b: str) -> tuple[float, int]:
    """Reference ARI with the plugin's drop-null policy. Returns (ari, n_valid)."""
    sub = df.select([col_a, col_b]).drop_nulls()
    n_valid = sub.height
    if n_valid == 0:
        return float("nan"), 0
    labels_a = sub[col_a].cast(pl.String).to_list()
    labels_b = sub[col_b].cast(pl.String).to_list()
    return sk_metrics.adjusted_rand_score(labels_a, labels_b), n_valid


def test_matches_sklearn_on_shared_dataset(dataset: pl.DataFrame) -> None:
    cols = ari_eligible_columns(dataset)
    assert len(cols) == 12
    sub_df = dataset.select(cols)

    result = pairwise_adjusted_rand(sub_df)
    rows = result.unnest(result.columns[0])
    assert rows.height == 66  # C(12,2)

    failures = []
    for row in rows.iter_rows(named=True):
        col_a, col_b = row["col_a"], row["col_b"]
        expected_ari, expected_n = sklearn_ari(sub_df, col_a, col_b)

        if row["n_valid"] != expected_n:
            failures.append(
                f"  ({col_a}, {col_b}): n_valid {row['n_valid']} != {expected_n}"
            )
            continue
        if math.isnan(expected_ari) != math.isnan(row["ari"]):
            failures.append(
                f"  ({col_a}, {col_b}): NaN mismatch rust={row['ari']} sklearn={expected_ari}"
            )
            continue
        if math.isnan(expected_ari):
            continue
        # Both sides count exact integers; only the final division is float.
        # abs_tol covers ARI values at/near zero where rel_tol is meaningless.
        if not math.isclose(row["ari"], expected_ari, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(
                f"  ({col_a}, {col_b}): ari rust={row['ari']} sklearn={expected_ari}"
            )

    assert not failures, "ARI mismatches vs sklearn:\n" + "\n".join(failures)


def test_identical_column_is_one(dataset: pl.DataFrame) -> None:
    df = dataset.select(
        pl.col("uint32_skewed").alias("x"),
        pl.col("uint32_skewed").alias("y"),
    )
    result = pairwise_adjusted_rand(df)
    row = result.unnest(result.columns[0]).row(0, named=True)
    assert math.isclose(row["ari"], 1.0, rel_tol=1e-12)


def test_specific_pairs_subset(dataset: pl.DataFrame) -> None:
    cols = ari_eligible_columns(dataset)
    wanted = [(cols[0], cols[1]), (cols[0], cols[2])]
    result = pairwise_adjusted_rand(dataset.select(cols), pairs=wanted)
    rows = result.unnest(result.columns[0])
    assert rows.height == 2
    assert set(zip(rows["col_a"], rows["col_b"])) == set(wanted)


def test_constant_columns_convention() -> None:
    # Both constant → sklearn returns 1.0; plugin must agree.
    df = pl.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2]})
    row = pairwise_adjusted_rand(df).unnest("pairwise_adjusted_rand").row(0, named=True)
    assert math.isclose(row["ari"], 1.0, rel_tol=1e-12)
    assert sk_metrics.adjusted_rand_score(df["a"].to_list(), df["b"].to_list()) == 1.0


def test_no_overlap_is_nan() -> None:
    df = pl.DataFrame({"a": [1, 2, None, None], "b": [None, None, 1, 2]})
    row = pairwise_adjusted_rand(df).unnest("pairwise_adjusted_rand").row(0, named=True)
    assert math.isnan(row["ari"])
    assert row["n_valid"] == 0
