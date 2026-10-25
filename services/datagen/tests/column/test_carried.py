import re

import numpy as np
import pytest

from synthdata import DatasetSpec, Distribution, SpecError, Tolerances, generate
from synthdata.column import multiset


def spec(**kw):
    base = dict(
        rows=1_000, arrow_type="int64", null_count=0, seed=1, value_range=(0, 10_000)
    )
    base.update(kw)
    return DatasetSpec(**base)


def test_bool_categories_accept_python_bools():
    s = DatasetSpec(
        rows=100, arrow_type="bool", null_count=0, seed=1, categories=[True, False]
    )
    assert set(generate(s).array.to_pylist()) == {True, False}


def test_bool_categories_stay_rejected_for_a_numeric_type():
    with pytest.raises(SpecError, match="numeric"):
        spec(value_range=None, categories=[1, True]).validate()


def test_uint64_above_int64_max_rejected_off_the_uniform_path():
    """Values above 2**63 - 1 come only from the uniform lattice draw of a
    capped support; the uncapped and the non-uniform draws go through float
    and int64, which cannot hold them."""
    full = dict(arrow_type="uint64", value_range=(0, 2**64 - 1))
    with pytest.raises(SpecError, match="2\\*\\*63"):
        spec(**full).validate()
    with pytest.raises(SpecError, match="2\\*\\*63"):
        spec(
            **full,
            cardinality_ratio=0.5,
            distribution=Distribution(kind="normal", sigma=1.0),
        ).validate()
    spec(**full, cardinality_ratio=0.5, entropy_bits=3.0).validate()
    spec(arrow_type="uint64", value_range=(0, 2**63 - 1)).validate()


def test_dataset_id_ignores_tolerances_and_names_numpy():
    a = spec(cardinality_ratio=0.1)
    b = spec(cardinality_ratio=0.1, tolerances=Tolerances(entropy_bits=0.5))
    assert a.dataset_id == b.dataset_id
    assert np.__version__ in a.id_payload()
    assert "tolerances" not in a.id_payload()


def test_manifest_records_nulls_min_max_and_decimal_places():
    m = generate(spec(null_count=100, cardinality_ratio=0.1)).manifest
    assert m.null_count == 100
    assert m.value_min is not None and m.value_max is not None
    assert 0 <= m.value_min <= m.value_max <= 10_000
    assert m.decimal_places_observed is None
    f = generate(
        spec(arrow_type="float64", decimal_places=2, value_range=(0, 10))
    ).manifest
    assert f.decimal_places_observed is not None and f.decimal_places_observed <= 2
    s = generate(spec(arrow_type="string", value_range=None, string_length=4))
    assert s.manifest.value_min is None and s.manifest.value_max is None


def test_manifest_round_trips_without_the_new_fields():
    from synthdata import Manifest

    d = generate(spec(cardinality_ratio=0.1)).manifest.to_dict()
    for k in ("null_count", "value_min", "value_max", "decimal_places_observed"):
        d.pop(k)
    back = Manifest.from_dict(d)
    assert back.null_count == 0 and back.value_min is None


def test_warning_tokens_are_kind_metric_colon():
    m = generate(spec(cardinality_ratio=0.5, entropy_bits=0.0001)).manifest
    assert m.warnings
    for w in m.warnings:
        kind, metric, _ = w.split(": ", 2)
        assert kind in ("clamp", "miss", "skipped", "undefined")
        assert re.match(r"^(\w+): (\w+): ", w)


def test_dispersion_solve_builds_one_multiset_per_sigma(monkeypatch):
    calls = []
    real = multiset._generated

    def counting(spec_, seed, sigma):
        calls.append(sigma)
        return real(spec_, seed, sigma)

    monkeypatch.setattr(multiset, "_generated", counting)
    s = DatasetSpec(
        rows=2_000,
        arrow_type="float64",
        value_range=(0.0, 100.0),
        decimal_places=2,
        null_count=0,
        seed=3,
        cardinality_ratio=0.5,
        distribution=Distribution(kind="normal", mean=50.0),
        index_of_dispersion=0.05,
    )
    result = multiset.solve_dispersion(s, np.random.default_rng(3), 0.05)
    assert result.ok
    assert len(calls) == len(set(calls))
