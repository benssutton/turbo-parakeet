import numpy as np
import pytest

from chbench.datagen.metrics import (
    cardinality_ratio,
    dichotomise,
    entropy_bits,
    entropy_of_counts,
    index_of_dispersion,
    interarrival_cv,
    largest_remainder,
    repeat_runs,
    run_count,
    runs_expected,
    runs_z,
    runs_z_bounds,
)

# --- ported from spikes/test_common.py: these must keep passing unchanged ---


def test_entropy_two_equiprobable_values_is_one_bit():
    assert entropy_bits(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_entropy_eight_distinct_values_is_three_bits():
    assert entropy_bits(np.arange(8)) == pytest.approx(3.0)


def test_entropy_single_value_is_zero():
    assert entropy_bits(np.zeros(10)) == pytest.approx(0.0)


def test_cardinality_ratio():
    assert cardinality_ratio(np.array([1, 1, 2, 3])) == pytest.approx(0.75)


def test_index_of_dispersion_uses_population_variance():
    x = np.array([2.0, 4.0])  # mean 3, population var 1
    assert index_of_dispersion(x) == pytest.approx(1.0 / 3.0)


def test_dichotomise_median_is_strictly_above():
    b = dichotomise(np.array([1, 2, 3, 4]))  # median 2.5
    assert b.tolist() == [False, False, True, True]


def test_dichotomise_topfreq_covers_half_the_rows():
    x = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    b = dichotomise(x, kind="topfreq")
    assert b.tolist() == [True] * 5 + [False] * 5


def test_run_count():
    assert run_count(np.array([True, True, False, True])) == 3
    assert run_count(np.array([True] * 5)) == 1


def test_runs_expected_known_values():
    mu, sd = runs_expected(10, 10)
    assert mu == pytest.approx(11.0)
    # Wald-Wolfowitz sd for n1 = n2 = 10: sqrt(2*100*(200-20) / (400*19)) = 2.17643
    assert sd == pytest.approx(2.17643, abs=1e-5)


def test_runs_z_sign():
    assert runs_z(np.tile([0, 1], 50)) > 5
    assert runs_z(np.repeat([0, 1], 50)) < -5


def test_repeat_runs():
    starts, lengths = repeat_runs(np.array([1, 1, 2, 3, 3, 3, 4]))
    assert starts.tolist() == [0, 3]
    assert lengths.tolist() == [2, 3]


def test_interarrival_cv_equal_gaps_is_zero():
    x = np.arange(40)
    for i, v in zip((0, 10, 20, 30), (100, 101, 102, 103)):
        x[i : i + 2] = v
    assert interarrival_cv(x) == pytest.approx(0.0)


def test_interarrival_cv_nan_when_too_few_runs():
    assert np.isnan(interarrival_cv(np.arange(10)))


def test_largest_remainder_sums_exactly():
    c = largest_remainder(np.array([0.5, 0.3, 0.2]), 7)
    assert c.sum() == 7
    assert c.tolist() == [4, 2, 1]


# --- new: the three defects the Plan 1 final review found ---


def test_runs_z_constant_column_returns_nan_not_raises():
    """A constant column is a legal DatasetSpec, and the manifest computes every
    metric for every dataset, so this must not abort generation."""
    assert np.isnan(runs_z(np.ones(100)))


def test_runs_expected_degenerate_group_returns_nan_sd():
    mu, sd = runs_expected(0, 100)
    assert mu == pytest.approx(1.0)
    assert np.isnan(sd)


def test_largest_remainder_zero_weights_spreads_evenly():
    """All-zero weights are reachable: the default frequency profile is a density
    that can underflow. Returning INT64_MIN silently is worse than a crash."""
    c = largest_remainder(np.zeros(4), 10)
    assert c.sum() == 10
    assert c.tolist() == [3, 3, 2, 2]


def test_largest_remainder_negative_weights_raises():
    with pytest.raises(ValueError, match="negative"):
        largest_remainder(np.array([1.0, -1.0]), 4)


def test_interarrival_cv_rounds_to_decimal_places():
    """Runs compare floats after rounding to decimal_places.
    Sub-tolerance noise must not destroy the runs."""
    x = np.repeat(np.arange(30.0), 2)
    jittered = x + np.random.default_rng(0).normal(0, 1e-12, x.size)
    assert interarrival_cv(x) == pytest.approx(0.0)
    assert np.isnan(interarrival_cv(jittered))  # unrounded: runs destroyed
    assert interarrival_cv(jittered, decimal_places=8) == pytest.approx(0.0)


def test_repeat_runs_rounds_to_decimal_places():
    x = np.array([1.0, 1.0 + 1e-12, 2.0])
    assert repeat_runs(x)[0].size == 0
    assert repeat_runs(x, decimal_places=8)[0].tolist() == [0]


def test_empty_input_is_consistent_across_metrics():
    e = np.array([], dtype=np.float64)
    for fn in (entropy_bits, cardinality_ratio, index_of_dispersion, interarrival_cv):
        assert np.isnan(fn(e)), f"{fn.__name__} should return nan on empty input"


def test_entropy_of_counts_matches_entropy_bits():
    counts = np.array([5, 3, 2], dtype=np.int64)
    values = np.repeat(np.arange(3), counts)
    assert entropy_of_counts(counts) == entropy_bits(values)


def test_entropy_of_counts_ignores_zero_counts():
    assert entropy_of_counts(np.array([5, 0, 5])) == pytest.approx(1.0)


def test_runs_z_bounds_from_measured_split():
    """The band is asymmetric: R = 2 is always reachable so the negative bound is
    stable, while the positive ceiling falls with min(n1, n2)."""
    lo_bal, hi_bal = runs_z_bounds(48849, 51151)
    lo_skew, hi_skew = runs_z_bounds(31852, 68148)
    assert lo_bal == pytest.approx(-316.22, abs=0.01)
    assert hi_bal == pytest.approx(302.00, abs=0.01)
    assert lo_skew == pytest.approx(-316.22, abs=0.01)
    assert hi_skew == pytest.approx(147.80, abs=0.01)


def test_runs_z_of_a_layout_lies_within_its_own_bounds():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    b = dichotomise(x)
    lo, hi = runs_z_bounds(int(b.sum()), int((~b).sum()))
    assert lo <= runs_z(x) <= hi
