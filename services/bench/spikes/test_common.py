import numpy as np
import pytest

from spikes._common import (
    cardinality_ratio,
    dichotomise,
    entropy_bits,
    index_of_dispersion,
    interarrival_cv,
    largest_remainder,
    repeat_runs,
    run_count,
    runs_expected,
    runs_z,
)


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
    # "a" alone covers 5 of 10 rows, which is >= half
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
    alternating = np.tile([0, 1], 50)
    blocked = np.repeat([0, 1], 50)
    assert runs_z(alternating) > 5
    assert runs_z(blocked) < -5


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
    w = np.array([0.5, 0.3, 0.2])
    c = largest_remainder(w, 7)
    assert c.sum() == 7
    assert c.tolist() == [4, 2, 1]  # raw 3.5, 2.1, 1.4 -> floors 3,2,1 -> +1 to .5
