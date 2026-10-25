import numpy as np
import pytest

from chbench.datagen.metrics import interarrival_cv
from chbench.datagen.refine import refine_interarrival
from chbench.datagen.spec import DatasetSpec
from tests.datagen.test_sequence import float_mset


def spec_for(counts, **kw) -> DatasetSpec:
    d = dict(rows=int(counts.sum()), arrow_type="float64", value_range=(-1.0, 10.0),
             decimal_places=8, null_count=0, seed=0, cardinality_ratio=0.75)
    d.update(kw)
    return DatasetSpec(**d)


@pytest.mark.parametrize("target", [0.5, 1.0, 1.5])
def test_refinement_beats_the_unrefined_draw(target):
    support, counts = float_mset()
    from chbench.datagen.sequence import layout_interarrival
    raw = interarrival_cv(layout_interarrival(support, counts, target,
                                              np.random.default_rng(1)),
                          decimal_places=8)
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=target),
                            np.random.default_rng(1))
    assert abs(r.achieved - target) < abs(raw - target)


@pytest.mark.parametrize("target", [0.5, 1.0, 1.5])
def test_refinement_lands_inside_the_ten_percent_tolerance(target):
    support, counts = float_mset()
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=target),
                            np.random.default_rng(1))
    assert r.achieved == pytest.approx(target, rel=0.10)


def test_refinement_preserves_the_multiset():
    support, counts = float_mset()
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=1.0),
                            np.random.default_rng(1))
    assert np.array_equal(np.sort(r.values), np.repeat(support, counts))


def test_refinement_is_skipped_when_runs_z_is_also_targeted():
    """Settled by the project owner, 2026-09-14: the combined distortion is inflicted
    after the gaps are laid out, so the bisection cannot recover the target and would
    spend the whole cap failing to."""
    support, counts = float_mset()
    r = refine_interarrival(support, counts,
                            spec_for(counts, interarrival_cv=1.0, runs_z=-0.25),
                            np.random.default_rng(1))
    assert r.skipped is True
    assert "runs_z" in r.reason
    assert r.values is None


def test_refinement_respects_its_iteration_cap():
    support, counts = float_mset()
    calls = []
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=1.0),
                            np.random.default_rng(1), cap=5,
                            _on_evaluate=calls.append)
    assert len(calls) <= 10


def test_unreachable_low_target_is_clamped_not_chased():
    support, counts = float_mset()
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=0.01),
                            np.random.default_rng(1))
    assert r.solve.status.name.startswith("CLAMPED")


def test_is_deterministic():
    support, counts = float_mset()
    a = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=1.0),
                            np.random.default_rng(1))
    b = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=1.0),
                            np.random.default_rng(1))
    assert np.array_equal(a.values, b.values) and a.input_cv == b.input_cv


def test_refinement_is_skipped_when_the_metric_is_undefined():
    """Fewer than three repeat-runs leave interarrival_cv nan at every input
    CV, so the bisection has nothing to measure and must not run."""
    support = np.arange(100, dtype=np.float64)
    counts = np.ones(100, dtype=np.int64)
    counts[:2] = 3
    r = refine_interarrival(support, counts, spec_for(counts, interarrival_cv=0.5),
                            np.random.default_rng(1))
    assert r.skipped is True
    assert "undefined" in r.reason and "repeat-runs" in r.reason
    assert r.values is None
