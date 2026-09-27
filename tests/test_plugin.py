"""The Arrow boundary: 128-bit integers, the Python binding, and Arrow inputs."""

import importlib

import polars as pl
import pytest

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


@pytest.mark.parametrize("spec", EVERY_IMPLEMENTATION)
def test_128_bit_integer_columns_are_ineligible(spec):
    frame, wide = wide_integer_frame()
    out = run(load(spec), {"t": frame})
    names = [c for c in out.columns if c.startswith("col_")]
    touches_wide = out.filter(pl.any_horizontal(pl.col(c).is_in(wide) for c in names))
    assert touches_wide.height > 0
    assert touches_wide["status"].unique().to_list() == ["ineligible"]
