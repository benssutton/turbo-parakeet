"""
Correctness test for the Rust joint-entropy plugin (services/analytics/src/entropy.rs).

Fabricates a controlled two-column frame with a known joint value-count distribution
(36 distinct pairs, ~20.8K rows, with nulls contributing in BOTH columns) and asserts
that the plugin's H(A,B) equals Polars' own entropy computed from value_counts.

The Polars reference "concatenates" the two columns into a single composite key with
pl.struct (NOT concat_str — that collapses every null-bearing row into one bucket and
would not match the plugin's per-position null semantics), runs value_counts, and takes
.entropy(base=2). This exactly mirrors the plugin's NULL_SENTINEL treatment, where
(None, x), (x, None) and (None, None) are distinct joint keys.

The test is parametrized over three dtype combinations that route through different
series_to_u64 branches: two strings, two integers, and one of each.

Run:  pytest tests/test_entropy.py -v
"""

import math
import sys
from pathlib import Path

import polars as pl
import pytest

# Expose the maturin-compiled `analytics` package (services/analytics/analytics/) as a
# top-level import. importorskip cleanly skips the module if the plugin isn't built.
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analytics"))
pairwise_joint_entropy = pytest.importorskip("analytics").pairwise_joint_entropy


# ── Fabrication ────────────────────────────────────────────────────────────────

# 36 joint-pair frequencies, summing to 20813 rows.
COUNTS = [
    7000, 4000, 3300, 1700, 1300, 1000, 950, 600, 300, 200,
    75, 60, 45, 30, 27, 25, 23, 22, 20, 20,
    18, 16, 14, 11, 10, 8, 8, 8, 7, 6,
    3, 2, 2, 1, 1, 1,
]
TOTAL_ROWS = sum(COUNTS)  # 20813
EXPECTED_ENTROPY = 2.878815  # joint H (bits) of the COUNTS distribution

A_NULL = {5, 15, 20}   # pair indices where col_a is null
B_NULL = {10, 15, 25}  # pair indices where col_b is null   (index 15 => (null, null))


def _pair_tokens(i: int, kind: str):
    """Return the (a, b) token pair for joint-pair index `i` and dtype `kind`.

    Non-null tokens are unique per index, so all 36 pairs stay distinct even after
    inserting nulls: null-a pairs differ by b, null-b pairs differ by a, and (None, None)
    occurs exactly once.

    Regression pin for the old NULL_SENTINEL scheme: for the int kind, pair 0 is
    (-1, 1005) — the same b token as pair 5's (None, 1005). -1 sign-extends to
    u64::MAX, so a sentinel-based encoding would alias it with null and MERGE the
    two pairs (35 keys instead of 36), shifting the entropy. With out-of-band null
    tracking they stay distinct and the entropy matches EXPECTED_ENTROPY.
    """
    if kind == "str":
        a, b = f"a{i}", f"b{i}"
    elif kind == "int":
        a, b = (-1, 1005) if i == 0 else (i, 1000 + i)
    elif kind == "mixed":
        a, b = f"a{i}", 1000 + i
    else:  # pragma: no cover - guard against typos in parametrize
        raise ValueError(f"unknown kind: {kind}")

    if i in A_NULL:
        a = None
    if i in B_NULL:
        b = None
    return a, b


def build_frame(kind: str) -> pl.DataFrame:
    """Materialize the frame by repeating each pair's tokens COUNTS[i] times."""
    col_a: list = []
    col_b: list = []
    for i, count in enumerate(COUNTS):
        a, b = _pair_tokens(i, kind)
        col_a.extend([a] * count)
        col_b.extend([b] * count)

    # Explicit schema so null-bearing integer columns stay Int64 (not coerced to Float64).
    a_dtype = pl.String if kind in ("str", "mixed") else pl.Int64
    b_dtype = pl.String if kind == "str" else pl.Int64
    return pl.DataFrame(
        {"col_a": col_a, "col_b": col_b},
        schema={"col_a": a_dtype, "col_b": b_dtype},
    )


# ── Reference + plugin under test ────────────────────────────────────────────────

def polars_joint_entropy(df: pl.DataFrame) -> float:
    """Concatenate the two columns into one key, value_counts, then entropy in bits."""
    key = df.select(pl.struct(["col_a", "col_b"]).alias("k")).to_series()
    vc = key.value_counts()  # includes null keys, distinct per position
    counts = vc.get_column("count")
    return counts.entropy(base=2, normalize=True)  # normalize -> -sum(p * log2 p)


def plugin_joint_entropy(df: pl.DataFrame) -> float:
    """Run the Rust plugin (single pair -> single row) and extract the entropy."""
    res = pairwise_joint_entropy(df)
    return res.unnest(res.columns[0]).get_column("entropy")[0]


# ── Test ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["str", "int", "mixed"])
def test_joint_entropy_matches_polars(kind):
    df = build_frame(kind)

    # Frame is shaped as intended and nulls contribute in both columns.
    assert df.height == TOTAL_ROWS
    assert df["col_a"].null_count() > 0
    assert df["col_b"].null_count() > 0

    h_plugin = plugin_joint_entropy(df)
    h_polars = polars_joint_entropy(df)

    assert math.isclose(h_plugin, h_polars, rel_tol=1e-5), (
        f"{kind}: plugin={h_plugin} polars={h_polars} diff={abs(h_plugin - h_polars):.3e}"
    )
    # Sanity: both agree with the analytically-known entropy of the COUNTS distribution.
    assert math.isclose(h_plugin, EXPECTED_ENTROPY, rel_tol=1e-4)
