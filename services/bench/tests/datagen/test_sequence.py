import numpy as np
import pytest
from chbench.datagen.metrics import (
    cardinality_ratio,
    dichotomise,
    entropy_bits,
    run_count,
    runs_z,
    runs_z_bounds,
)
from chbench.datagen.sequence import (
    composition,
    dichotomy_kind_for,
    layout_runs,
    runs_plan,
    target_run_count,
)
from chbench.datagen.solve import SolveStatus
from chbench.datagen.spec import DatasetSpec
from chbench.datagen.metrics import interarrival_cv
from chbench.datagen.sequence import layout_interarrival, layout_sequence

# Reference output for test_layout_interarrival_matches_the_retained_spike_reference,
# captured from the loop implementation at commit fdfd875 before this task's step 3
# replaced it with the vectorised body.
REFERENCE_FIRST_12 = [
    7.76682403,
    3.87193531,
    3.87193531,
    3.87193531,
    -0.41640574,
    5.15332078,
    2.38113361,
    0.37719861,
    8.71855197,
    2.02386528,
    1.56544907,
    1.56544907,
]
REFERENCE_CHECKSUM = 1935508784


def skewed(n=20_000, k=1_000, s=1.2, seed=0):
    rng = np.random.default_rng(seed)
    support = rng.choice(np.arange(1_000_000, 10_000_000), size=k, replace=False)
    w = np.arange(1, k + 1, dtype=np.float64) ** (-s)
    return rng.choice(support, size=n, p=w / w.sum())


def test_composition_splits_into_positive_parts_summing_exactly():
    rng = np.random.default_rng(0)
    c = composition(1000, 7, rng)
    assert c.sum() == 1000 and c.size == 7 and c.min() >= 1


def test_composition_handles_a_single_part():
    assert composition(50, 1, np.random.default_rng(0)).tolist() == [50]


@pytest.mark.parametrize("z", [-0.25, -5.0, 3.0, -50.0])
def test_layout_hits_the_target_run_count_exactly(z):
    x = skewed()
    b = dichotomise(x)
    want = target_run_count(int(b.sum()), int((~b).sum()), z)
    y = layout_runs(x, z, np.random.default_rng(1))
    assert run_count(dichotomise(y)) == want


@pytest.mark.parametrize("z", [-0.25, -5.0, 3.0, -50.0])
def test_layout_preserves_the_multiset_exactly(z):
    """Load-bearing for the whole two-stage design: stage 2 only permutes."""
    x = skewed()
    y = layout_runs(x, z, np.random.default_rng(1))
    assert np.array_equal(np.sort(x), np.sort(y))
    assert entropy_bits(x) == entropy_bits(y)
    assert cardinality_ratio(x) == cardinality_ratio(y)


def test_achieved_z_is_within_half_a_run_step_of_target():
    x = skewed()
    b = dichotomise(x)
    n1, n2 = int(b.sum()), int((~b).sum())
    from chbench.datagen.metrics import runs_expected

    _, sd = runs_expected(n1, n2)
    y = layout_runs(x, -0.25, np.random.default_rng(1))
    assert runs_z(y) == pytest.approx(-0.25, abs=0.5 / sd + 1e-9)


def test_runs_plan_reports_the_asymmetric_band_from_measured_split():
    x = skewed()
    p = runs_plan(x, -0.25, "median")
    assert p.n1 + p.n2 == x.size
    assert (p.z_lo, p.z_hi) == runs_z_bounds(p.n1, p.n2)
    assert p.z_lo < 0 < p.z_hi
    assert not p.clamped


def test_runs_plan_flags_a_target_outside_the_band():
    x = skewed()
    p = runs_plan(x, -10_000.0, "median")
    assert p.clamped
    assert p.target_R == 2


def test_layout_clamps_rather_than_failing_outside_the_band():
    x = skewed()
    y = layout_runs(x, -10_000.0, np.random.default_rng(1))
    assert run_count(dichotomise(y)) == 2
    assert np.array_equal(np.sort(x), np.sort(y))


def test_topfreq_dichotomy_layout_also_preserves_the_multiset():
    labels = np.repeat(np.array([f"c{i}" for i in range(50)]), np.arange(50, 0, -1) * 8)
    y = layout_runs(labels, -0.25, np.random.default_rng(1), kind="topfreq")
    assert np.array_equal(np.sort(labels), np.sort(y))


def test_dichotomy_kind_is_chosen_by_type():
    num = DatasetSpec(
        rows=10, arrow_type="float64", value_range=(0.0, 1.0), null_count=0, seed=0
    )
    cat = DatasetSpec(
        rows=10, arrow_type="string", categories=["a", "b"], null_count=0, seed=0
    )
    assert dichotomy_kind_for(num) == "median"
    assert dichotomy_kind_for(cat) == "topfreq"


def test_no_ordering_target_gives_a_uniform_shuffle():
    x = skewed()
    y = layout_runs(x, None, np.random.default_rng(1))
    assert np.array_equal(np.sort(x), np.sort(y))
    assert abs(runs_z(y)) < 5


def float_mset(n=40_000, ratio=0.75, seed=0):
    rng = np.random.default_rng(seed)
    support = np.unique(np.round(rng.uniform(-1, 10, int(ratio * n)), 8))
    counts = np.ones(support.size, dtype=np.int64)
    np.add.at(counts, rng.integers(0, support.size, n - support.size), 1)
    return support, counts


@pytest.mark.parametrize("cv", [0.5, 1.0, 2.0])
def test_interarrival_layout_preserves_the_multiset(cv):
    support, counts = float_mset()
    y = layout_interarrival(support, counts, cv, np.random.default_rng(1))
    assert np.array_equal(np.sort(y), np.repeat(support, counts))


@pytest.mark.parametrize("cv", [0.5, 1.0, 2.0])
def test_interarrival_layout_moves_achieved_cv_toward_target(cv):
    """The constructive draw undershoots; refinement closes the gap. Here
    we only require the ordering to be right."""
    support, counts = float_mset()
    y = layout_interarrival(support, counts, cv, np.random.default_rng(1))
    got = interarrival_cv(y, decimal_places=8)
    assert 0 < got <= cv * 1.1


def test_achieved_cv_is_monotone_in_the_input_cv():
    """This is what makes the refine bisection converge."""
    support, counts = float_mset()
    got = [
        interarrival_cv(
            layout_interarrival(support, counts, c, np.random.default_rng(1)),
            decimal_places=8,
        )
        for c in (0.4, 0.8, 1.2, 1.6, 2.0)
    ]
    assert got == sorted(got)


def test_low_cv_targets_are_unreachable_by_construction():
    """Mean gap is fixed by the multiset and gaps are integers >= 2, so CV near zero
    cannot be produced. The generator must not pretend otherwise."""
    support, counts = float_mset()
    y = layout_interarrival(support, counts, 0.01, np.random.default_rng(1))
    assert interarrival_cv(y, decimal_places=8) > 0.05


def test_combined_targets_lay_out_once_and_warn():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-0.25,
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert r.combined is True
    assert any("trade off" in w for w in r.warnings)
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_combined_case_still_honours_the_runs_target():
    """runs_z is achieved to the same precision whether or not CV is also set,
    because the block pattern is fixed before any CV randomness is drawn."""
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-20.0,
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert runs_z(r.values) == pytest.approx(-20.0, abs=0.05)


def test_combined_case_warns_most_strongly_for_near_random_z():
    support, counts = float_mset()

    def warns(z):
        s = DatasetSpec(
            rows=int(counts.sum()),
            arrow_type="float64",
            value_range=(-1.0, 10.0),
            decimal_places=8,
            null_count=0,
            seed=0,
            cardinality_ratio=0.75,
            runs_z=z,
            interarrival_cv=0.5,
        )
        return layout_sequence(support, counts, s, np.random.default_rng(1)).warnings

    assert any("strongly" in w for w in warns(-0.25))


def test_no_ordering_targets_gives_a_shuffle():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert r.combined is False
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_runs_only_dispatches_to_the_runs_layout():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-20.0,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert r.combined is False
    assert runs_z(r.values) == pytest.approx(-20.0, abs=0.05)
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_cv_only_dispatches_to_the_interarrival_layout():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    direct = layout_interarrival(support, counts, 0.5, np.random.default_rng(1))
    assert r.combined is False
    assert np.array_equal(r.values, direct)
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_cv_override_replaces_the_spec_target():
    """The refine pass re-lays with a trial CV without rebuilding the DatasetSpec."""
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1), cv_override=2.0)
    direct = layout_interarrival(support, counts, 2.0, np.random.default_rng(1))
    assert np.array_equal(r.values, direct)


def test_interarrival_layout_handles_no_repeat_runs():
    """All-singles multiset: nothing to gap, so the result is just a permutation."""
    support = np.arange(100, dtype=np.float64)
    counts = np.ones(100, dtype=np.int64)
    y = layout_interarrival(support, counts, 0.5, np.random.default_rng(1))
    assert np.array_equal(np.sort(y), support)


def test_constant_column_with_runs_target_does_not_crash():
    """A single-group dichotomy has no runs sd, so `target_run_count` would raise on
    int(nan). The manifest measures every metric on every dataset, constant columns
    included, so the dispatcher must warn and fall through instead."""
    support, counts = np.array([3.0]), np.array([1_000], dtype=np.int64)
    s = DatasetSpec(
        rows=1_000,
        arrow_type="float64",
        value_range=(3.0, 3.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        runs_z=-0.25,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert r.combined is False
    assert any("single-group" in w for w in r.warnings)
    assert np.array_equal(r.values, np.repeat(support, counts))


def test_out_of_band_runs_target_is_clamped_with_a_warning_naming_the_band():
    """A runs target outside the reachable band is clamped and recorded as
    a reachable-bound violation, never silently. The band comes from the measured
    split, so the warning must carry those numbers."""
    support, counts = float_mset()
    x = np.repeat(support, counts)
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-10_000.0,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    p = runs_plan(x, -10_000.0, "median")
    assert p.clamped
    clamp = [w for w in r.warnings if "clamped" in w]
    assert len(clamp) == 1
    assert "-10000" in clamp[0]
    assert f"{p.z_lo:.2f}" in clamp[0] and f"{p.z_hi:.2f}" in clamp[0]
    assert run_count(dichotomise(r.values)) == p.target_R == 2
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_in_band_runs_target_does_not_warn_about_clamping():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-0.25,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert not any("clamped" in w for w in r.warnings)


def test_strong_combined_warning_states_the_condition_not_a_z_label():
    """The threshold is mean block length below mean run gap. At 40 000 rows that
    holds for z = -100, which is neither near-random nor positive, so the message
    must state the numbers and not characterise z."""
    support, counts = float_mset()
    n = int(counts.sum())
    runs = int((counts >= 2).sum())
    s = DatasetSpec(
        rows=n,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-100.0,
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    strong = [w for w in r.warnings if "strongly" in w]
    assert len(strong) == 1
    assert "near-random" not in strong[0] and "positive" not in strong[0]
    assert "mean block length" in strong[0] and "mean run gap" in strong[0]
    assert f"{n / runs:.2f}" in strong[0]


def test_runs_plan_tolerates_a_single_group_split():
    """A constant column has one group: run count is deterministically 1, the
    band is undefined and nothing is clamped. `layout_sequence` reads the plan's
    n1, n2 to detect this, so the plan must not raise."""
    p = runs_plan(np.full(100, 3.0), -0.25, "median")
    assert (p.n1, p.n2) == (0, 100)
    assert p.target_R == 1
    assert not p.clamped
    assert np.isnan(p.z_lo) and np.isnan(p.z_hi)


def test_cv_target_with_too_few_repeat_runs_warns_that_the_metric_is_undefined():
    """Too few duplicates to form runs is a warning. interarrival_cv needs three
    run starts to have two gaps, so a multiset with fewer than three repeated
    values cannot define it."""
    support = np.arange(100, dtype=np.float64)
    counts = np.ones(100, dtype=np.int64)
    counts[:2] = 3  # two repeat-runs: one gap, no CV
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(0.0, 100.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=100 / int(counts.sum()),
        interarrival_cv=0.5,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    undefined = [w for w in r.warnings if w.startswith("undefined: interarrival_cv")]
    assert len(undefined) == 1
    assert "2 repeat-runs" in undefined[0]
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_layout_sequence_returns_its_runs_plan_so_the_clamp_can_be_recorded():
    support, counts = float_mset()
    s = DatasetSpec(
        rows=int(counts.sum()),
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
        cardinality_ratio=0.75,
        runs_z=-10_000.0,
    )
    r = layout_sequence(support, counts, s, np.random.default_rng(1))
    assert r.plan is not None and r.plan.clamped
    assert r.plan.status is SolveStatus.CLAMPED_LOW
    no_target = layout_sequence(
        support,
        counts,
        DatasetSpec(
            rows=int(counts.sum()),
            arrow_type="float64",
            value_range=(-1.0, 10.0),
            decimal_places=8,
            null_count=0,
            seed=0,
            cardinality_ratio=0.75,
        ),
        np.random.default_rng(1),
    )
    assert no_target.plan is None


def test_layout_interarrival_matches_the_retained_spike_reference():
    """FINDINGS §8: the vectorised layout must reproduce the loop
    version byte for byte. This pins the reference output so the swap cannot
    silently change generated data."""
    support, counts = float_mset(n=40_000, ratio=0.75, seed=3)
    out = layout_interarrival(support, counts, 1.0, np.random.default_rng(11))
    # First 12 values and a checksum of the whole, captured from the loop
    # implementation at commit fdfd875 before this task changed it.
    assert out.size == counts.sum()
    assert np.array_equal(np.sort(out), np.repeat(support, counts))
    ref = REFERENCE_FIRST_12
    assert np.allclose(out[:12], ref)
    assert int(np.round(out, 8).view(np.int64).sum() % (2**31)) == REFERENCE_CHECKSUM
