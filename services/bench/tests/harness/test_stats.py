import numpy as np
import pytest

from chbench.harness.stats import exact_coverage, median_ci, paired_ratio


def test_fewer_than_six_values_have_no_95_interval():
    assert median_ci([3.0, 1.0, 2.0, 5.0, 4.0]) == (3.0, None, None)


def test_six_values_take_the_extremes():
    assert median_ci([6.0, 1.0, 5.0, 2.0, 4.0, 3.0]) == (3.5, 1.0, 6.0)


def test_ten_values_take_the_second_and_ninth():
    # P(B(10, 1/2) < 2) = 11/1024 <= 0.025 < P(B < 3) = 56/1024
    x = [float(v) for v in range(10, 0, -1)]
    assert median_ci(x) == (5.5, 2.0, 9.0)
    assert exact_coverage(10) == pytest.approx(1 - 2 * 11 / 1024)


def test_none_and_nan_are_dropped():
    assert median_ci([None, float("nan"), 2.0]) == (2.0, None, None)
    assert median_ci([]) == (None, None, None)


@pytest.mark.parametrize("n", [10, 30])
def test_interval_covers_the_true_median_as_often_as_its_exact_coverage(n):
    rng = np.random.default_rng(7)
    trials, hits = 2000, 0
    for _ in range(trials):
        _, lo, hi = median_ci(rng.lognormal(0.0, 0.5, n))  # true median exp(0) = 1
        hits += lo <= 1.0 <= hi
    assert hits / trials == pytest.approx(exact_coverage(n), abs=0.015)
    assert exact_coverage(n) >= 0.95


def test_paired_ratio_pairs_by_round_and_skips_missing_or_non_positive():
    a = {0: 50.0, 1: 60.0, 2: 0.0, 3: 40.0, 4: 55.0, 5: 45.0, 6: 50.0, 7: None, 9: 1.0}
    b = {0: 100.0, 1: 120.0, 2: 100.0, 3: 80.0, 4: 110.0, 5: 90.0, 6: 100.0, 7: 9.0}
    ratio, lo, hi, n = paired_ratio(a, b)
    assert n == 6 and ratio == pytest.approx(0.5) and lo == hi == pytest.approx(0.5)


def test_paired_ratio_without_shared_rounds():
    assert paired_ratio({0: 1.0}, {1: 1.0}) == (None, None, None, 0)
