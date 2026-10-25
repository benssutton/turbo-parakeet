import numpy as np
import pytest

from chbench.datagen.multiset import build_multiset, draw_support
from chbench.datagen.spec import DatasetSpec, Distribution
from chbench.datagen.multiset import (
    density_counts,
    entropy_ceiling,
    entropy_floor,
    powerlaw_counts,
    solve_entropy,
)
from chbench.datagen.metrics import entropy_of_counts
from chbench.datagen.solve import SolveStatus
from chbench.datagen.multiset import solve_dispersion
from chbench.datagen.metrics import index_of_dispersion


def spec(**kw) -> DatasetSpec:
    d = dict(
        rows=100_000,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
    )
    d.update(kw)
    return DatasetSpec(**d)


def test_uniform_draw_reaches_k_with_no_collapse():
    s = spec()
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.values.size == s.support_size
    assert d.clip_collapsed == 0
    assert d.clipped_frac == pytest.approx(0.0)
    assert d.warnings == []


def test_support_values_are_sorted_unique_and_in_range():
    s = spec()
    d = draw_support(s, np.random.default_rng(s.seed))
    assert np.all(np.diff(d.values) > 0)
    assert d.values.min() >= -1.0 and d.values.max() <= 10.0


def test_support_values_are_rounded_to_decimal_places():
    s = spec(decimal_places=3)
    d = draw_support(s, np.random.default_rng(s.seed))
    assert np.allclose(d.values, np.round(d.values, 3))


def test_clip_collapse_is_counted_separately_and_not_redrawn():
    """A lognormal wide enough to push mass past the upper bound. Every such draw
    lands on exactly one support point, and redrawing cannot recover them."""
    s = spec(
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=1.0, shift=-1.0)
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.clipped_frac > 0.01
    assert d.clip_collapsed > 0
    assert d.values.size < d.requested_k
    assert (d.values == 10.0).sum() <= 1  # all merged onto the one bound
    assert any(w.startswith("miss: cardinality_ratio at risk") for w in d.warnings)


def test_rounding_collapse_is_redrawn_away():
    """Coarse rounding on a narrow range forces collisions with no clipping at all.
    These are recoverable, so the redraw loop should close most of the gap."""
    s = spec(
        rows=10_000, cardinality_ratio=0.5, value_range=(0.0, 1.0), decimal_places=4
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.clipped_frac == pytest.approx(0.0)
    assert d.clip_collapsed == 0
    assert d.values.size >= 0.99 * d.requested_k


def test_shortfall_records_which_mechanism_caused_it():
    s = spec(
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=1.0, shift=-1.0)
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.clip_collapsed + d.round_collapsed == d.requested_k - d.values.size


def test_categories_are_used_verbatim():
    s = DatasetSpec(
        rows=1000, arrow_type="string", categories=["a", "b", "c"], null_count=0, seed=0
    )
    d = draw_support(s, np.random.default_rng(0))
    assert sorted(d.values.tolist()) == ["a", "b", "c"]
    assert d.clip_collapsed == d.round_collapsed == 0


def test_integer_range_draw_is_within_bounds_and_integral():
    s = DatasetSpec(
        rows=10_000,
        arrow_type="int32",
        value_range=(1_000_000, 9_999_999),
        null_count=0,
        seed=0,
        cardinality_ratio=0.1,
    )
    d = draw_support(s, np.random.default_rng(0))
    assert d.values.dtype.kind in "iu"
    assert d.values.min() >= 1_000_000 and d.values.max() <= 9_999_999
    assert d.values.size == s.support_size


def test_list_of_integer_draws_an_integer_support():
    """The integer check must look at the list's element type, not the list
    type itself; otherwise list<int32> falls into the float path and rounds
    with decimal_places=None."""
    s = DatasetSpec(
        rows=10_000,
        arrow_type="list<int32>",
        value_range=(0, 99_999),
        null_count=0,
        seed=0,
        cardinality_ratio=0.1,
        array_len_range=(2, 4),
    )
    d = draw_support(s, np.random.default_rng(0))
    assert d.values.dtype.kind in "iu"
    assert d.values.min() >= 0 and d.values.max() <= 99_999
    assert d.values.size == s.support_size


@pytest.mark.parametrize(
    "arrow_type,lo,hi",
    [
        ("date32", 0, 20_000),
        ("timestamp[ms]", 0, 10**12),
        ("time32[s]", 0, 86_399),
        ("bool", 0, 1),
    ],
)
def test_temporal_and_bool_draw_from_the_integer_lattice(arrow_type, lo, hi):
    """Design 4.2: dates and times are integers in the unit's resolution, then
    cast; bool is the lattice {0, 1}. None of them has decimal_places."""
    s = DatasetSpec(
        rows=10_000,
        arrow_type=arrow_type,
        value_range=(lo, hi),
        null_count=0,
        seed=0,
        cardinality_ratio=0.0002,
    )
    d = draw_support(s, np.random.default_rng(0))
    assert d.values.dtype.kind in "iu"
    assert d.values.min() >= lo and d.values.max() <= hi
    assert d.values.size == s.support_size


def test_a_cardinality_target_beyond_the_integer_lattice_is_clamped_with_a_warning():
    s = DatasetSpec(
        rows=1_000,
        arrow_type="bool",
        value_range=(0, 1),
        null_count=0,
        seed=0,
        cardinality_ratio=0.5,
    )
    d = draw_support(s, np.random.default_rng(0))
    assert d.values.tolist() == [0, 1]
    assert any("lattice" in w for w in d.warnings)


def test_string_length_draws_distinct_strings_of_that_length():
    s = DatasetSpec(
        rows=10_000,
        arrow_type="string",
        string_length=6,
        null_count=0,
        seed=0,
        cardinality_ratio=0.1,
    )
    d = draw_support(s, np.random.default_rng(0))
    assert d.values.size == s.support_size
    assert len(set(d.values.tolist())) == s.support_size
    assert all(len(v) == 6 and v.isalnum() for v in d.values.tolist())


def test_uncapped_multiset_draws_rows_values_directly():
    """Design 4.2 step 1: without cardinality_ratio, draw rows values and skip
    the profile; the multiset is their unique values with counts."""
    s = spec(cardinality_ratio=None)
    m = build_multiset(s, np.random.default_rng(0))
    assert int(m.counts.sum()) == s.rows
    assert np.all(np.diff(m.support) > 0)
    assert m.support.min() >= -1.0 and m.support.max() <= 10.0


def test_is_deterministic_for_a_given_seed():
    s = spec()
    a = draw_support(s, np.random.default_rng(s.seed)).values
    b = draw_support(s, np.random.default_rng(s.seed)).values
    assert np.array_equal(a, b)


def test_sigma_override_is_used_when_supplied():
    """The dispersion solver calls draw_support repeatedly with trial sigmas, so the
    override must bypass the spec's own value."""
    s = spec(
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=0.25, shift=-1.0)
    )
    narrow = draw_support(s, np.random.default_rng(s.seed), sigma=0.25)
    wide = draw_support(s, np.random.default_rng(s.seed), sigma=1.0)
    assert wide.clipped_frac > narrow.clipped_frac


def test_redraw_drops_out_of_range_draws_instead_of_clipping():
    """Redraws must never merge onto a bound: an out-of-range redraw is discarded,
    not clipped, so clip_collapsed is unaffected by the loop and no bound value
    appears that the first draw did not already produce.

    sigma=0.14, k=80, seed=0 is a case (found by search) where the first draw of
    80 normal variates on [0, 1] has zero draws outside the range -- so
    clip_collapsed is 0 and the redraw gate opens purely for rounding collapse --
    but the much smaller redraw batches occasionally do land outside [0, 1]. A
    clip-based redraw loop turns one of those into a spurious 0.0 in the support;
    a drop-based loop must not.
    """
    s = DatasetSpec(
        rows=1000,
        arrow_type="float64",
        value_range=(0.0, 1.0),
        decimal_places=2,
        null_count=0,
        seed=0,
        cardinality_ratio=0.08,
        distribution=Distribution(kind="normal", mean=0.5, sigma=0.14),
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.clip_collapsed == 0
    assert 0.0 not in d.values.tolist()
    assert 1.0 not in d.values.tolist()


def test_normal_without_sigma_raises_informative_error():
    """distribution.sigma defaults to None (meaning "let the dispersion solver
    choose"), and validate() does not require it, so draw_support must fail with
    a clear message of its own rather than a bare TypeError from `sigma ** 2`."""
    s = spec(distribution=Distribution(kind="normal", mean=5.0, sigma=None))
    with pytest.raises(ValueError, match="sigma"):
        draw_support(s, np.random.default_rng(s.seed))


def test_lognormal_without_sigma_raises_informative_error():
    s = spec(
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=None, shift=-1.0)
    )
    with pytest.raises(ValueError, match="sigma"):
        draw_support(s, np.random.default_rng(s.seed))


def test_lognormal_without_mean_raises_informative_error():
    """validate() now rejects a lognormal with no mean, but _raw_draw should state
    its own precondition rather than rely on the caller having validated."""
    s = spec(
        distribution=Distribution(kind="lognormal", mean=None, sigma=1.0, shift=-1.0)
    )
    with pytest.raises(ValueError, match="mean"):
        draw_support(s, np.random.default_rng(s.seed))


def test_two_sided_clip_does_not_overstate_clip_collapse():
    """A normal centred in the range with sigma wide enough to clip both bounds
    merges those draws onto two retained points, not one. The one-sided formula
    (outside count minus one) overstates clip_collapsed by one here and drives
    round_collapsed negative; sigma=3.0, k=1000, seed=0 is a case (found by
    search) where the old formula gives round_collapsed == -1."""
    s = spec(
        cardinality_ratio=0.01,
        distribution=Distribution(kind="normal", mean=4.5, sigma=3.0),
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.round_collapsed >= 0
    assert d.clip_collapsed + d.round_collapsed == d.requested_k - d.values.size
    assert (d.values == -1.0).sum() <= 1
    assert (d.values == 10.0).sum() <= 1


def test_entropy_floor_matches_the_measured_values():
    """FINDINGS section 1 measured these floors directly."""
    assert entropy_floor(100_000, 1_000) == pytest.approx(0.18027, abs=1e-4)
    assert entropy_floor(100_000, 10_000) == pytest.approx(1.79759, abs=1e-4)


def test_entropy_floor_rises_with_k():
    assert entropy_floor(100_000, 100) < entropy_floor(100_000, 10_000)


def test_entropy_ceiling_is_log2_k():
    assert entropy_ceiling(1_000) == pytest.approx(9.9657842, abs=1e-6)


def test_powerlaw_counts_sum_exactly_and_give_every_value_one():
    c = powerlaw_counts(100_000, 1_000, 1.2)
    assert c.sum() == 100_000
    assert c.min() >= 1


def test_entropy_is_monotone_decreasing_in_the_exponent():
    h = [
        entropy_of_counts(powerlaw_counts(100_000, 1_000, s))
        for s in (0.0, 0.5, 1.0, 2.0, 5.0)
    ]
    assert h == sorted(h, reverse=True)


@pytest.mark.parametrize("k,target", [(1_000, 4.0), (10_000, 4.0), (1_000, 8.0)])
def test_solve_entropy_hits_reachable_targets(k, target):
    r = solve_entropy(100_000, k, target)
    assert r.status is SolveStatus.OK
    assert r.achieved == pytest.approx(target, abs=0.01)


def test_solve_entropy_clamps_below_the_floor():
    r = solve_entropy(100_000, 10_000, 0.5)  # floor is 1.798
    assert r.status is SolveStatus.CLAMPED_LOW
    assert r.bound == pytest.approx(entropy_floor(100_000, 10_000), abs=1e-3)


def test_solve_entropy_clamps_above_the_ceiling():
    r = solve_entropy(100_000, 1_000, 12.0)  # ceiling is 9.966
    assert r.status is SolveStatus.CLAMPED_HIGH
    assert r.bound == pytest.approx(entropy_ceiling(1_000), abs=1e-3)


def test_use_case_2_entropy_target_is_reachable_at_10m_rows():
    """1000 distinct values at 10M rows, entropy 4 bits (spec section 5.1)."""
    r = solve_entropy(10_000_000, 1_000, 4.0)
    assert r.status is SolveStatus.OK
    assert r.achieved == pytest.approx(4.0, abs=0.01)


def test_build_multiset_counts_sum_to_rows():
    s = spec()
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert m.counts.sum() == s.rows
    assert m.support.size == m.counts.size


def test_build_multiset_hits_an_entropy_target():
    s = spec(rows=50_000, cardinality_ratio=0.02, entropy_bits=4.0)
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert entropy_of_counts(m.counts) == pytest.approx(4.0, abs=0.01)


def test_density_profile_is_used_when_no_entropy_target_is_set():
    """Without an entropy target the profile follows the distribution's density, so
    counts are not uniform for a skewed distribution."""
    s = spec(
        rows=20_000,
        cardinality_ratio=0.1,
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=0.5, shift=-1.0),
    )
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert m.counts.std() > 0
    assert m.counts.sum() == s.rows


@pytest.mark.parametrize("arrow_type", ["int32", "date32", "timestamp[ms]"])
def test_uncapped_lattice_draws_honour_the_distribution(arrow_type):
    """Design 4.2 step 1: without cardinality_ratio the rows are drawn from the
    distribution itself. A narrow normal centred in a wide range must give a
    column concentrated near the mean, not uniform over the lattice."""
    s = DatasetSpec(
        rows=20_000,
        arrow_type=arrow_type,
        value_range=(0, 100_000),
        null_count=0,
        seed=0,
        distribution=Distribution(kind="normal", mean=50_000.0, sigma=100.0),
    )
    m = build_multiset(s, np.random.default_rng(0))
    x = np.repeat(m.support, m.counts)
    assert x.dtype.kind in "iu" and int(m.counts.sum()) == s.rows
    assert x.min() >= 0 and x.max() <= 100_000
    uniform_sd = 100_000 / np.sqrt(12)
    assert x.std() < 0.01 * uniform_sd
    assert abs(x.mean() - 50_000) < 10


def test_uncapped_lattice_uniform_draws_cover_the_range():
    s = DatasetSpec(
        rows=20_000, arrow_type="int32", value_range=(0, 999), null_count=0, seed=0
    )
    m = build_multiset(s, np.random.default_rng(0))
    x = np.repeat(m.support, m.counts)
    assert x.min() == 0 and x.max() == 999
    assert abs(x.std() - 1000 / np.sqrt(12)) < 20


def test_numeric_categories_take_a_density_profile_without_value_range():
    s = DatasetSpec(
        rows=1_000,
        arrow_type="float64",
        categories=[1.0, 2.0, 3.0],
        decimal_places=1,
        null_count=0,
        seed=0,
        distribution=Distribution(kind="normal", mean=2.0, sigma=0.5),
    )
    m = build_multiset(s, np.random.default_rng(0))
    assert int(m.counts.sum()) == 1_000
    assert m.counts[1] > m.counts[0] and m.counts[1] > m.counts[2]


def test_density_counts_raises_on_nan_densities_instead_of_masking_them():
    """A sigma of zero or below gives nan densities everywhere. Validation
    rejects such a sigma, so the old fallback that masked nan to an even split
    is unreachable; keeping it would only hide a regression in validate() or
    in a caller's sigma override."""
    s = DatasetSpec(
        rows=1_000,
        arrow_type="float64",
        categories=[1.0, 2.0, 3.0],
        decimal_places=1,
        null_count=0,
        seed=0,
        distribution=Distribution(kind="normal", mean=2.0, sigma=0.5),
    )
    support = np.array([1.0, 2.0, 3.0])
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="nan"), np.errstate(
            divide="ignore", invalid="ignore"
        ):
            density_counts(support, s, 1_000, sigma=bad)
    with pytest.raises(ValueError, match="nan"):
        density_counts(np.array([1.0, np.nan, 3.0]), s, 1_000, sigma=None)
    counts = density_counts(support, s, 1_000, sigma=None)
    assert int(counts.sum()) == 1_000


def test_density_counts_still_floors_underflowed_zero_densities():
    """The far tail of a narrow normal underflows to a density of exactly 0,
    which is a real value, not a nan; it keeps the smallest-positive floor so
    every support value still appears at least once."""
    s = DatasetSpec(
        rows=1_000,
        arrow_type="float64",
        categories=[0.0, 0.5, 100.0],
        decimal_places=1,
        null_count=0,
        seed=0,
        distribution=Distribution(kind="normal", mean=0.5, sigma=0.1),
    )
    counts = density_counts(np.array([0.0, 0.5, 100.0]), s, 1_000, sigma=None)
    assert int(counts.sum()) == 1_000 and counts.min() >= 1
    assert counts[1] > counts[0] and counts[1] > counts[2]


def test_density_allocation_pulls_the_mean_below_the_support_mean():
    """FINDINGS section 4: replication weight follows the density, which for a
    right-skewed distribution concentrates below the mean. This is the dominant
    mean-drift mechanism, not clipping."""
    s = spec(
        rows=100_000,
        cardinality_ratio=0.75,
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=0.25, shift=-1.0),
    )
    m = build_multiset(s, np.random.default_rng(s.seed))
    expanded_mean = np.repeat(m.support, m.counts).mean()
    assert expanded_mean < m.support.mean()


def uc1(**kw) -> DatasetSpec:
    """Use case 1's shape, at a row count fast enough for a unit test."""
    d = dict(
        rows=20_000,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        distribution=Distribution(kind="lognormal", mean=4.0, shift=-1.0),
    )
    d.update(kw)
    return DatasetSpec(**d)


@pytest.mark.parametrize("target", [0.5, 1.0, 2.0])
def test_solve_dispersion_hits_reachable_targets(target):
    s = uc1(index_of_dispersion=target)
    r = solve_dispersion(s, np.random.default_rng(s.seed), target)
    assert r.status is SolveStatus.OK
    assert r.achieved == pytest.approx(target, abs=0.01)


def test_solve_dispersion_does_not_report_a_false_clamp():
    """The Plan 1 defect: a naive bracket straddles the mean-sign crossing, so the
    high end is negative and every positive target trivially clamps."""
    s = uc1(index_of_dispersion=2.0)
    r = solve_dispersion(s, np.random.default_rng(s.seed), 2.0)
    assert r.status is not SolveStatus.CLAMPED_HIGH


def test_solve_dispersion_narrows_to_the_positive_mean_branch():
    s = uc1(index_of_dispersion=2.0)
    r = solve_dispersion(s, np.random.default_rng(s.seed), 2.0)
    m = build_multiset(s, np.random.default_rng(s.seed), sigma=r.x)
    assert np.repeat(m.support, m.counts).mean() > 0


def test_solve_dispersion_reports_an_invalid_bracket_when_no_branch_exists():
    """A shift that drives the generated mean negative across the whole bracket.
    This must be INVALID_BRACKET, not a clamp: there is no valid branch to search,
    which is a different condition from a target beyond a reachable bound."""
    s = DatasetSpec(
        rows=5_000,
        arrow_type="float64",
        value_range=(-20.0, 20.0),
        decimal_places=4,
        null_count=0,
        seed=0,
        cardinality_ratio=0.5,
        distribution=Distribution(kind="lognormal", mean=4.0, shift=-10.0),
        index_of_dispersion=1.0,
    )
    r = solve_dispersion(s, np.random.default_rng(s.seed), 1.0)
    assert r.status is SolveStatus.INVALID_BRACKET


def test_solve_dispersion_uses_the_measured_not_analytic_value():
    """Analytic ID overstates the measured value once clipping removes the right
    tail: measured was 80, 72 and 36 percent of analytic at sigma 0.25, 0.5, 1.0."""
    s = uc1()
    r = solve_dispersion(s, np.random.default_rng(s.seed), 1.0)
    sigma = r.x
    m = build_multiset(s, np.random.default_rng(s.seed), sigma=sigma)
    measured = index_of_dispersion(np.repeat(m.support, m.counts))
    analytic_var = (np.exp(sigma**2) - 1) * np.exp(2 * np.log(4.0))
    analytic = analytic_var / (4.0 - 1.0)
    assert measured == pytest.approx(1.0, abs=0.01)
    assert measured < analytic


def test_build_multiset_records_the_dispersion_solve():
    s = uc1(index_of_dispersion=1.0)
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert "index_of_dispersion" in m.solves
    assert m.solves["index_of_dispersion"].status is SolveStatus.OK


def test_powerlaw_ranks_are_not_tied_to_support_order():
    """powerlaw_counts puts the heaviest rank at index 0 and the support is sorted,
    so without a shuffle the most frequent value is always the smallest. The
    assignment of counts to values must be permuted with rng."""
    heaviest_is_min = []
    for seed in range(6):
        s = spec(rows=50_000, cardinality_ratio=0.02, entropy_bits=4.0, seed=seed)
        m = build_multiset(s, np.random.default_rng(s.seed))
        heaviest_is_min.append(m.support[m.counts.argmax()] == m.support.min())
        assert m.counts.sum() == s.rows
        assert m.counts.min() >= 1
    assert not all(heaviest_is_min)


def test_dispersion_achieved_is_remeasured_on_the_returned_multiset():
    """The solve picks sigma against builds from a fresh generator seeded from the
    spec, but the final draw uses the caller's rng. If that generator is not at
    its initial state the two disagree, and the recorded achieved must describe
    the multiset actually returned, with a warning that says so. k=100 at 5000
    rows with 10 draws consumed moves the measured value by about 0.037, well
    past the 0.01 tolerance."""
    s = uc1(rows=5_000, cardinality_ratio=0.02, index_of_dispersion=1.0)

    consumed = np.random.default_rng(s.seed)
    consumed.random(10)
    m = build_multiset(s, consumed)
    measured = index_of_dispersion(np.repeat(m.support, m.counts))
    assert m.solves["index_of_dispersion"].achieved == measured
    assert measured != pytest.approx(1.0, abs=s.tolerances.index_of_dispersion)
    assert any("initial state" in w for w in m.warnings)

    fresh = build_multiset(s, np.random.default_rng(s.seed))
    fresh_measured = index_of_dispersion(np.repeat(fresh.support, fresh.counts))
    assert fresh.solves["index_of_dispersion"].achieved == fresh_measured
    assert fresh_measured == pytest.approx(1.0, abs=0.01)
    assert not any("initial state" in w for w in fresh.warnings)


def test_capped_lattice_support_draw_honours_the_distribution():
    """Design 4.2 step 1 draws the support *from the distribution*. When k is far
    below the lattice span, a uniform `rng.choice` support leaves the density
    profile nowhere to put its mass: int32 over [1e6, 1e7] with normal(5e6, 1e5)
    spanned the whole range (sd 192 572) while float64 on the same spec gave
    sd 72 745. The integer support must be shaped like the float one."""
    common = dict(
        rows=1_000_000,
        value_range=(1_000_000, 10_000_000),
        null_count=0,
        seed=0,
        cardinality_ratio=0.001,
        distribution=Distribution(kind="normal", mean=5e6, sigma=1e5),
    )
    ints = DatasetSpec(arrow_type="int32", **common)
    floats = DatasetSpec(arrow_type="float64", decimal_places=2, **common)
    di = draw_support(ints, np.random.default_rng(0))
    df = draw_support(floats, np.random.default_rng(0))
    assert di.values.dtype.kind in "iu"
    assert di.values.size == ints.support_size
    assert 0.5 < di.values.std() / df.values.std() < 2.0


def test_dispersion_solve_under_an_entropy_target_reports_not_converged():
    """validate() rejects the combination; this guards the solver half. The
    entropy target's rank shuffle makes the objective non-monotone in sigma
    (the sorted support reorders while the count permutation stays fixed), so
    the bisection lands off target and must say so rather than report ok."""
    s = DatasetSpec(
        rows=20_000,
        arrow_type="float64",
        value_range=(0.0, 100.0),
        decimal_places=2,
        null_count=0,
        seed=0,
        cardinality_ratio=0.5,
        entropy_bits=8.0,
        index_of_dispersion=0.5,
        distribution=Distribution(kind="normal", mean=50.0),
    )
    r = solve_dispersion(s, np.random.default_rng(0), 0.5)
    assert r.status is SolveStatus.NOT_CONVERGED
    assert abs(r.achieved - 0.5) > s.tolerances.index_of_dispersion


def test_float_support_beyond_the_grid_capacity_is_clamped_with_a_warning():
    """Range [0, 1] at 2 decimal places is a 101-point grid; 10 000 distinct
    values cannot exist on it. The float path must clamp and warn like the
    lattice path does, instead of burning the redraw cap in silence."""
    s = spec(
        rows=20_000, cardinality_ratio=0.5, value_range=(0.0, 1.0), decimal_places=2
    )
    d = draw_support(s, np.random.default_rng(s.seed))
    assert d.values.size <= 101
    assert any("cardinality" in w and "101" in w for w in d.warnings)


def test_build_multiset_warns_about_collapse_that_survives_the_redraw_cap():
    """A normal narrow enough to concentrate on about twenty grid points cannot
    supply 50 distinct values however often it is redrawn: 29 of the 50
    requested collapse by rounding and the shortfall was silent because
    build_multiset forwarded only the draw's own warnings."""
    s = DatasetSpec(
        rows=1_000,
        arrow_type="float64",
        value_range=(0.0, 1.0),
        decimal_places=2,
        null_count=0,
        seed=0,
        cardinality_ratio=0.05,
        distribution=Distribution(kind="normal", mean=0.5, sigma=0.03),
    )
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert m.round_collapsed > 0 and m.clip_collapsed == 0
    rounding = [w for w in m.warnings if "rounding collapse" in w]
    assert len(rounding) == 1
    assert str(m.round_collapsed) in rounding[0]
    assert not any("clip collapse" in w for w in m.warnings)


def test_build_multiset_warns_about_clip_collapse_with_its_count():
    s = spec(
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=1.0, shift=-1.0)
    )
    m = build_multiset(s, np.random.default_rng(s.seed))
    assert m.clip_collapsed > 0
    clip = [w for w in m.warnings if "clip collapse" in w]
    assert len(clip) == 1 and str(m.clip_collapsed) in clip[0]
