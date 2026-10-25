import pytest

from synthdata.column.solve import SolveStatus, solve_monotone


def test_solves_an_increasing_function():
    r = solve_monotone(lambda x: x**2, target=4.0, lo=0.0, hi=10.0, increasing=True)
    assert r.status is SolveStatus.OK
    assert r.x == pytest.approx(2.0, abs=1e-6)
    assert r.achieved == pytest.approx(4.0, abs=1e-6)


def test_solves_a_decreasing_function():
    r = solve_monotone(lambda x: -x, target=-3.0, lo=0.0, hi=10.0, increasing=False)
    assert r.status is SolveStatus.OK
    assert r.x == pytest.approx(3.0, abs=1e-6)


def test_target_above_range_clamps_high_and_reports_the_bound():
    r = solve_monotone(lambda x: x, target=99.0, lo=0.0, hi=10.0, increasing=True)
    assert r.status is SolveStatus.CLAMPED_HIGH
    assert r.bound == pytest.approx(10.0)
    assert r.x == pytest.approx(10.0)
    assert "reachable" in r.message


def test_target_below_range_clamps_low_and_reports_the_bound():
    r = solve_monotone(lambda x: x, target=-5.0, lo=0.0, hi=10.0, increasing=True)
    assert r.status is SolveStatus.CLAMPED_LOW
    assert r.bound == pytest.approx(0.0)


def test_decreasing_function_clamps_on_the_right_side():
    """For a decreasing f the low end of the bracket gives the HIGH value, so the
    clamp sides must not be swapped."""
    r = solve_monotone(lambda x: -x, target=5.0, lo=0.0, hi=10.0, increasing=False)
    assert r.status is SolveStatus.CLAMPED_HIGH
    assert r.bound == pytest.approx(0.0)


def test_guard_failing_at_the_low_end_is_an_invalid_bracket():
    r = solve_monotone(
        lambda x: x, target=1.0, lo=0.0, hi=10.0, increasing=True, guard=lambda x: False
    )
    assert r.status is SolveStatus.INVALID_BRACKET
    assert "bracket" in r.message


def test_guard_narrows_the_bracket_to_the_valid_branch():
    """Mirrors index_of_dispersion: f is only meaningful where the guard holds, and
    beyond the crossing it returns a large negative number that would otherwise be
    read as a reachable maximum."""

    def f(x):
        return x if x < 5.0 else -1000.0

    r = solve_monotone(
        f, target=4.0, lo=0.0, hi=10.0, increasing=True, guard=lambda x: x < 5.0
    )
    assert r.status is SolveStatus.OK
    assert r.x == pytest.approx(4.0, abs=1e-3)
    assert r.achieved == pytest.approx(4.0, abs=1e-3)


def test_guard_narrowing_reports_an_approximate_bound_when_clamped():
    """A max computed next to the crossing reflects the iteration count, not a
    physical ceiling, so it is reported as approximate."""

    def f(x):
        return x if x < 5.0 else -1000.0

    r = solve_monotone(
        f, target=99.0, lo=0.0, hi=10.0, increasing=True, guard=lambda x: x < 5.0
    )
    assert r.status is SolveStatus.CLAMPED_HIGH
    assert r.bound == pytest.approx(5.0, abs=0.01)
    assert "approximate" in r.message


def test_is_deterministic():
    args = dict(target=4.0, lo=0.0, hi=10.0, increasing=True)
    a = solve_monotone(lambda x: x**2, **args)
    b = solve_monotone(lambda x: x**2, **args)
    assert (a.x, a.achieved, a.status) == (b.x, b.achieved, b.status)


def test_evaluation_count_is_bounded():
    calls = []
    solve_monotone(
        lambda x: calls.append(x) or x,
        target=5.0,
        lo=0.0,
        hi=10.0,
        increasing=True,
        iters=20,
    )
    assert len(calls) <= 25  # 2 endpoints + iters + a final confirmation


def test_non_finite_target_raises():
    with pytest.raises(ValueError):
        solve_monotone(
            lambda x: x, target=float("nan"), lo=0.0, hi=1.0, increasing=True
        )


def test_a_non_monotone_objective_is_reported_as_not_converged_not_ok():
    """The bisection assumes monotonicity and cannot detect its violation from
    the bracket alone; with a tolerance it can at least check where it landed.
    f is increasing on the bracket ends but dips in the middle, so the search
    stops on an arbitrary point that is not near the target."""

    def f(x):
        return x if x < 4.0 or x > 6.0 else 100.0 - x

    r = solve_monotone(f, target=5.0, lo=0.0, hi=10.0, increasing=True, tol=0.1)
    assert r.status is SolveStatus.NOT_CONVERGED
    assert abs(r.achieved - 5.0) > 0.1
    assert "not converged" in r.message and "5" in r.message
    assert not r.ok


def test_tolerance_check_passes_a_converged_solve():
    r = solve_monotone(
        lambda x: x**2, target=4.0, lo=0.0, hi=10.0, increasing=True, tol=1e-6
    )
    assert r.status is SolveStatus.OK


def test_no_tolerance_means_no_convergence_check():
    def f(x):
        return x if x < 4.0 or x > 6.0 else 100.0 - x

    r = solve_monotone(f, target=5.0, lo=0.0, hi=10.0, increasing=True)
    assert r.status is SolveStatus.OK
