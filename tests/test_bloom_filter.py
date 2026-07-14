"""
Tests for BloomFilter (services/analytics/bloom_filter.py).

Two implementation families are tested:
  - Custom Rust plugin (BloomFilter) — all non-boolean dtypes (15 columns)
  - fastbloom-rs (FastBloomFilter) — scalar non-boolean dtypes only (9 columns)

Cross-implementation comparison is at the FP-rate level only. The two
implementations use incompatible hash functions, so per-item membership
decisions on false positives are expected to differ and are not compared.

Categorical/enum columns encode by string value (encode_series casts to String
and hashes with foldhash), so they behave like string columns here and are
covered by every test group below.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analytics.bloom_filter import BloomFilter

try:
    from fastbloom_rs import BloomFilter as FastBloomFilter
    FASTBLOOM_AVAILABLE = True
except ImportError:
    FASTBLOOM_AVAILABLE = False


FP_RATE = 0.01
N_NEGATIVES = 1_000
# At FP=1%, N=1000 negatives: E[FP]=10, σ≈3.1.
# 3× tolerance gives headroom for statistical variance.
FP_RATE_TOLERANCE = 3.0

NON_BOOLEAN_COLS = [
    f"{dtype}_{shape}"
    for dtype in ("float64", "uint32", "categorical", "list", "arr")
    for shape in ("skewed", "high_unique", "sparse")
]
SCALAR_NON_BOOLEAN_COLS = [
    f"{dtype}_{shape}"
    for dtype in ("float64", "uint32", "categorical")
    for shape in ("skewed", "high_unique", "sparse")
]


def _dtype_spec(col: str) -> str:
    return col.split("_")[0]


def _unique_nonnull(series: pl.Series) -> list:
    """Unique non-null values as a Python list; safe for List and Array dtypes."""
    dropped = series.drop_nulls()
    if _dtype_spec(series.name) in ("list", "arr"):
        seen: set = set()
        result = []
        for v in dropped.to_list():
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                result.append(v)
        return result
    return dropped.unique().to_list()


def _as_series(col: str, values: list, source_dtype: pl.DataType) -> pl.Series:
    """Build a typed Series suitable for passing to BloomFilter."""
    if _dtype_spec(col) == "categorical":
        return pl.Series(col, values, dtype=pl.String).cast(pl.Categorical)
    return pl.Series(col, values, dtype=source_dtype)


def _make_negatives(col: str) -> list:
    """
    Return N_NEGATIVES values guaranteed outside the test dataset's value space.

    Dataset token ranges by dtype:
      float64/uint32 : 0 .. n_rows-1  (skewed/sparse draw from [0, 30))
      categorical    : "cat_0" .. "cat_29"
      list           : [t, t+1] for t in small range
      arr            : [t, t+1, t+2] for t in small range
    """
    spec = _dtype_spec(col)
    n = N_NEGATIVES
    if spec == "float64":
        return [float(1_000_000 + i) for i in range(n)]
    if spec == "uint32":
        return [10_000 + i for i in range(n)]
    if spec == "categorical":
        return [f"__neg_{i}" for i in range(n)]
    if spec == "list":
        return [[99_999, i] for i in range(n)]
    if spec == "arr":
        return [[99_999, i, i + 1] for i in range(n)]
    raise ValueError(f"No negative generator for dtype spec '{spec}'")


def _fastbloom_key(value, col: str):
    """Canonical key for fastbloom-rs (accepts str | int | bytes)."""
    return int(value) if _dtype_spec(col) == "uint32" else str(value)


# ── No false negatives ────────────────────────────────────────────────────────

@pytest.mark.parametrize("col", NON_BOOLEAN_COLS)
def test_no_false_negatives_custom(dataset: pl.DataFrame, col: str) -> None:
    """Items added to the custom bloom filter must always be found."""
    dtype = dataset[col].dtype
    values = _unique_nonnull(dataset[col])
    if not values:
        pytest.skip(f"No non-null values in {col}")

    series = _as_series(col, values, dtype)
    bf = BloomFilter(len(values), FP_RATE)
    bf.add(pl.DataFrame({col: series}))

    fn_count = (~bf.membership(pl.DataFrame({col: series})).to_series()).sum()
    assert fn_count == 0, f"{col}: {fn_count}/{len(values)} false negatives"


@pytest.mark.skipif(not FASTBLOOM_AVAILABLE, reason="fastbloom-rs not installed")
@pytest.mark.parametrize("col", SCALAR_NON_BOOLEAN_COLS)
def test_no_false_negatives_fastbloom(dataset: pl.DataFrame, col: str) -> None:
    """Items added to fastbloom-rs must always be found."""
    dtype = dataset[col].dtype
    values = _unique_nonnull(dataset[col])
    if not values:
        pytest.skip(f"No non-null values in {col}")

    keys = [_fastbloom_key(v, col) for v in values]
    bf = FastBloomFilter(len(values), FP_RATE)
    for k in keys:
        bf.add(k)

    fn_count = sum(1 for k in keys if not bf.contains(k))
    assert fn_count == 0, f"{col}: {fn_count}/{len(values)} false negatives"


def test_cross_frame_categorical_membership() -> None:
    """
    A categorical value must be found by its string identity, not its physical
    code. Build the filter on one Series, then query with a separately built
    Series where the SAME strings get DIFFERENT physical codes (achieved by
    prepending distinct padding categories). Under code-based encoding the
    shared values map to codes the filter never saw → false negatives. Under
    string-based encoding they hash identically → all found.
    """
    shared = [f"tok_{i}" for i in range(50)]
    filter_series = pl.Series("c", shared, dtype=pl.String).cast(pl.Categorical)
    bf = BloomFilter(len(shared), FP_RATE)
    bf.add(pl.DataFrame({"c": filter_series}))

    # Prepend 50 distinct padding categories so tok_i is assigned code 50+i
    # in the query Series instead of code i.
    pad = [f"pad_{i}" for i in range(50)]
    query_series = pl.Series("c", pad + shared, dtype=pl.String).cast(pl.Categorical)
    result = bf.membership(pl.DataFrame({"c": query_series})).to_series()

    # Only the shared tail must be fully found (the pad prefix should not be).
    shared_found = int(result.slice(len(pad)).sum())
    assert shared_found == len(shared), (
        f"cross-frame categorical: {shared_found}/{len(shared)} shared values "
        "found; a categorical must be identified by string value, not code"
    )


# ── False positive rate ───────────────────────────────────────────────────────

@pytest.mark.parametrize("col", NON_BOOLEAN_COLS)
def test_false_positive_rate_custom(dataset: pl.DataFrame, col: str) -> None:
    """Custom filter FP rate must stay within FP_RATE_TOLERANCE × the configured rate."""
    dtype = dataset[col].dtype
    values = _unique_nonnull(dataset[col])
    if not values:
        pytest.skip(f"No non-null values in {col}")

    series = _as_series(col, values, dtype)
    bf = BloomFilter(len(values), FP_RATE)
    bf.add(pl.DataFrame({col: series}))

    neg_values = _make_negatives(col)
    neg_series = _as_series(col, neg_values, dtype)
    fp_count = bf.membership(pl.DataFrame({col: neg_series})).to_series().sum()
    actual_rate = fp_count / N_NEGATIVES

    assert actual_rate <= FP_RATE * FP_RATE_TOLERANCE, (
        f"{col}: FP rate {actual_rate:.4f} > {FP_RATE * FP_RATE_TOLERANCE:.4f} "
        f"({fp_count}/{N_NEGATIVES} FPs, target {FP_RATE:.4f})"
    )


@pytest.mark.skipif(not FASTBLOOM_AVAILABLE, reason="fastbloom-rs not installed")
@pytest.mark.parametrize("col", SCALAR_NON_BOOLEAN_COLS)
def test_false_positive_rate_fastbloom(dataset: pl.DataFrame, col: str) -> None:
    """fastbloom-rs FP rate must stay within FP_RATE_TOLERANCE × the configured rate."""
    dtype = dataset[col].dtype
    values = _unique_nonnull(dataset[col])
    if not values:
        pytest.skip(f"No non-null values in {col}")

    bf = FastBloomFilter(len(values), FP_RATE)
    for v in values:
        bf.add(_fastbloom_key(v, col))

    neg_values = _make_negatives(col)
    fp_count = sum(1 for v in neg_values if bf.contains(_fastbloom_key(v, col)))
    actual_rate = fp_count / N_NEGATIVES

    assert actual_rate <= FP_RATE * FP_RATE_TOLERANCE, (
        f"{col}: FP rate {actual_rate:.4f} > {FP_RATE * FP_RATE_TOLERANCE:.4f} "
        f"({fp_count}/{N_NEGATIVES} FPs, target {FP_RATE:.4f})"
    )


# ── Cross-implementation FP rate agreement ────────────────────────────────────

@pytest.mark.skipif(not FASTBLOOM_AVAILABLE, reason="fastbloom-rs not installed")
@pytest.mark.parametrize("col", SCALAR_NON_BOOLEAN_COLS)
def test_fp_rate_agreement_vs_fastbloom(dataset: pl.DataFrame, col: str) -> None:
    """
    Custom and fastbloom-rs must both achieve a FP rate within tolerance.

    Categorical is included: both implementations now hash the string value
    (the custom filter casts categorical to String in encode_series), so the
    rate comparison is meaningful.

    Per-item membership results on negatives are deliberately NOT compared — the
    two implementations use different hash functions, so per-item disagreement on
    false positives is expected and correct.
    """
    dtype = dataset[col].dtype
    values = _unique_nonnull(dataset[col])
    if not values:
        pytest.skip(f"No non-null values in {col}")

    neg_values = _make_negatives(col)
    series = _as_series(col, values, dtype)
    neg_series = _as_series(col, neg_values, dtype)

    # Custom filter
    custom_bf = BloomFilter(len(values), FP_RATE)
    custom_bf.add(pl.DataFrame({col: series}))
    custom_fp = custom_bf.membership(pl.DataFrame({col: neg_series})).to_series().sum() / N_NEGATIVES

    # fastbloom-rs
    fast_bf = FastBloomFilter(len(values), FP_RATE)
    for v in values:
        fast_bf.add(_fastbloom_key(v, col))
    fast_fp = sum(1 for v in neg_values if fast_bf.contains(_fastbloom_key(v, col))) / N_NEGATIVES

    assert custom_fp <= FP_RATE * FP_RATE_TOLERANCE, (
        f"{col}: custom FP rate {custom_fp:.4f} exceeds {FP_RATE * FP_RATE_TOLERANCE:.4f}"
    )
    assert fast_fp <= FP_RATE * FP_RATE_TOLERANCE, (
        f"{col}: fastbloom FP rate {fast_fp:.4f} exceeds {FP_RATE * FP_RATE_TOLERANCE:.4f}"
    )
