"""
Pairwise chi-squared correctness tests: Rust plugin vs polars-ds baseline.

Eligible columns from the shared 18-column dataset (see conftest.make_dataset):
    boolean_*     (3 columns) — Boolean
    uint32_*      (3 columns) — UInt32 (integer type)
    cat_*         (3 columns) — Categorical

Excluded: float64_* (continuous), list_* and arr_* (nested types).

9 eligible columns → C(9,2) = 36 pairs tested.

Skipped automatically if polars-ds is not installed.
"""

import math
from itertools import combinations

import polars as pl
import pytest

pytest.importorskip("polars_ds")  # skip entire module if polars-ds not installed

_analytics = pytest.importorskip("analytics")
pairwise_chi_squared_rust = _analytics.pairwise_chi_squared

from chi_squared_polarsds import pairwise_chi_squared as pairwise_chi_squared_pds


# ── Column filter ─────────────────────────────────────────────────────────────

_CHI2_ELIGIBLE = (
    pl.Boolean,
    pl.String,
    pl.Categorical,
    pl.Enum,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)


def chi2_eligible_columns(df: pl.DataFrame) -> list[str]:
    """Column names whose dtype is suitable for chi-squared independence testing."""
    return [c for c, dtype in df.schema.items() if isinstance(dtype, _CHI2_ELIGIBLE)]


# ── Assertion helper ──────────────────────────────────────────────────────────

def assert_chi2_matches(rust_result: pl.DataFrame, pds_result: pl.DataFrame) -> None:
    """Assert that Rust and polars-ds chi-squared results agree within rtol=1e-4."""
    rust_rows = rust_result.unnest(rust_result.columns[0])
    pds_rows = pds_result.unnest(pds_result.columns[0])

    pds_map = {
        (row["col_a"], row["col_b"]): row
        for row in pds_rows.iter_rows(named=True)
    }

    failures = []
    for row in rust_rows.iter_rows(named=True):
        col_a, col_b = row["col_a"], row["col_b"]
        pds_row = pds_map.get((col_a, col_b))
        if pds_row is None:
            failures.append(f"  ({col_a}, {col_b}): pair missing from polars-ds result")
            continue

        r_chi2, p_chi2 = row["chi2_stat"], pds_row["chi2_stat"]
        r_v, p_v = row["cramers_v"], pds_row["cramers_v"]

        # Both NaN → degenerate pair (e.g. constant column after null-drop); skip
        if math.isnan(r_chi2) and math.isnan(p_chi2):
            continue

        if math.isnan(r_chi2) != math.isnan(p_chi2):
            failures.append(
                f"  ({col_a}, {col_b}): chi2_stat NaN mismatch"
                f" rust={r_chi2} pds={p_chi2}"
            )
            continue

        if not math.isclose(r_chi2, p_chi2, rel_tol=1e-4):
            failures.append(
                f"  ({col_a}, {col_b}): chi2_stat rust={r_chi2:.8f} pds={p_chi2:.8f}"
                f"  diff={abs(r_chi2 - p_chi2):.3e}"
            )

        if not (math.isnan(r_v) and math.isnan(p_v)):
            if not math.isclose(r_v, p_v, rel_tol=1e-4):
                failures.append(
                    f"  ({col_a}, {col_b}): cramers_v rust={r_v:.8f} pds={p_v:.8f}"
                    f"  diff={abs(r_v - p_v):.3e}"
                )

    if failures:
        raise AssertionError("Chi-squared mismatch (Rust vs polars-ds):\n" + "\n".join(failures))


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_pairwise_chi_squared(dataset):
    """C(9,2) = 36 pairs; Rust plugin chi2_stat and cramers_v match polars-ds."""
    eligible = chi2_eligible_columns(dataset)
    pairs = list(combinations(eligible, 2))

    rust_result = pairwise_chi_squared_rust(dataset, pairs=pairs)
    pds_result = pairwise_chi_squared_pds(dataset, pairs=pairs, max_unique=None)

    assert_chi2_matches(rust_result, pds_result)
