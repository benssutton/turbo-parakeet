import time

import pytest

from chbench.datagen import generate
from chbench.datagen.metrics import runs_expected
from chbench.datagen.multiset import entropy_ceiling, entropy_floor
from chbench.datagen.spec import DatasetSpec, Distribution

pytestmark = pytest.mark.slow

ROWS = 10_000_000


def test_use_case_1_generates_at_full_scale():
    s = DatasetSpec(
        rows=ROWS,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=42,
        cardinality_ratio=0.75,
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=0.25, shift=-1.0),
    )
    t0 = time.perf_counter()
    d = generate(s)
    elapsed = time.perf_counter() - t0
    assert len(d.array) == ROWS
    assert d.manifest.achieved["cardinality_ratio"] == pytest.approx(0.75, abs=0.01)
    print(f"\nuse case 1 at {ROWS:,} rows: {elapsed:.1f}s")


def test_use_case_2_generates_at_full_scale():
    """Spec section 5.1: 1000 distinct values, entropy 4 bits, runs z -0.25."""
    s = DatasetSpec(
        rows=ROWS,
        arrow_type="int32",
        value_range=(1_000_000, 9_999_999),
        null_count=0,
        seed=42,
        cardinality_ratio=0.0001,
        entropy_bits=4.0,
        runs_z=-0.25,
    )
    t0 = time.perf_counter()
    d = generate(s)
    elapsed = time.perf_counter() - t0
    assert d.manifest.achieved["entropy_bits"] == pytest.approx(4.0, abs=0.01)
    assert d.manifest.achieved["runs_z"] == pytest.approx(-0.25, abs=0.01)
    assert d.manifest.achieved["cardinality_ratio"] == pytest.approx(0.0001, abs=1e-6)
    print(f"\nuse case 2 at {ROWS:,} rows: {elapsed:.1f}s")


def test_entropy_bounds_at_full_scale():
    assert entropy_floor(ROWS, 1_000) < 4.0 < entropy_ceiling(1_000)


def test_runs_z_band_scales_with_the_square_root_of_rows():
    _, sd_small = runs_expected(50_000, 50_000)
    _, sd_large = runs_expected(ROWS // 2, ROWS // 2)
    assert sd_large / sd_small == pytest.approx(10.0, rel=0.01)
    assert 0.5 / sd_large == pytest.approx(0.000316, abs=1e-5)


def test_interarrival_at_full_scale_is_the_slow_path():
    """Profiling marker: about 2.1 million Python-level iterations at this scale."""
    s = DatasetSpec(
        rows=ROWS,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=42,
        cardinality_ratio=0.75,
        interarrival_cv=1.0,
    )
    t0 = time.perf_counter()
    d = generate(s)
    elapsed = time.perf_counter() - t0
    print(f"\ninter-arrival at {ROWS:,} rows: {elapsed:.1f}s")
    assert len(d.array) == ROWS
