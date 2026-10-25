import json

import pyarrow as pa
import pytest

from chbench.datagen.spec import (
    DatasetSpec,
    Distribution,
    SpecError,
    Tolerances,
    parse_arrow_type,
)


def base(**kw) -> DatasetSpec:
    d = dict(rows=1000, arrow_type="float64", value_range=(0.0, 10.0),
             decimal_places=4, null_count=0, seed=42)
    d.update(kw)
    return DatasetSpec(**d)


@pytest.mark.parametrize("s,expected", [
    ("int8", pa.int8()), ("uint64", pa.uint64()), ("float32", pa.float32()),
    ("float64", pa.float64()), ("bool", pa.bool_()), ("string", pa.string()),
    ("date32", pa.date32()), ("decimal32(9,8)", pa.decimal32(9, 8)),
    ("decimal64(18,8)", pa.decimal64(18, 8)), ("decimal128(38,10)", pa.decimal128(38, 10)),
    ("timestamp[ms]", pa.timestamp("ms")), ("time64[us]", pa.time64("us")),
    ("list<int32>", pa.list_(pa.int32())),
])
def test_parse_arrow_type(s, expected):
    assert parse_arrow_type(s) == expected


@pytest.mark.parametrize("s", ["int7", "decimal32(10,2)", "decimal64(19,2)",
                               "decimal128(39,2)", "list<list<int32>>", "wat"])
def test_parse_arrow_type_rejects(s):
    with pytest.raises(SpecError):
        parse_arrow_type(s)


def test_decimal_precision_limits_are_enforced_per_width():
    assert parse_arrow_type("decimal32(9,8)") == pa.decimal32(9, 8)
    with pytest.raises(SpecError, match="precision"):
        parse_arrow_type("decimal32(10,8)")


def test_default_tolerances_match_the_spec_table():
    t = Tolerances()
    assert t.entropy_bits == 0.01
    assert t.cardinality_ratio == 0.001
    assert t.index_of_dispersion == 0.01
    assert t.runs_z == 0.01
    assert t.interarrival_cv_relative == 0.10


def test_dataset_id_is_stable_and_depends_on_every_field():
    a, b = base(), base()
    assert a.dataset_id == b.dataset_id
    assert len(a.dataset_id) == 16
    assert base(seed=43).dataset_id != a.dataset_id
    assert base(rows=1001).dataset_id != a.dataset_id
    assert base(cardinality_ratio=0.5).dataset_id != a.dataset_id


def test_round_trips_through_json():
    s = base(distribution=Distribution(kind="lognormal", mean=4.0, shift=-1.0),
             cardinality_ratio=0.75)
    assert DatasetSpec.from_dict(json.loads(s.to_json())) == s


def test_entropy_target_requires_a_finite_support_set():
    with pytest.raises(SpecError, match="entropy_bits requires"):
        base(entropy_bits=4.0).validate()
    base(entropy_bits=4.0, cardinality_ratio=0.1).validate()
    DatasetSpec(rows=100, arrow_type="string", categories=["a", "b"],
                null_count=0, seed=1, entropy_bits=1.0).validate()


def test_null_count_cannot_exceed_rows():
    with pytest.raises(SpecError, match="null_count"):
        base(null_count=1001).validate()


def test_cardinality_ratio_must_yield_at_least_one_value():
    with pytest.raises(SpecError, match="cardinality_ratio"):
        base(cardinality_ratio=0.0).validate()
    with pytest.raises(SpecError, match="cardinality_ratio"):
        base(cardinality_ratio=1.5).validate()


def test_exactly_one_value_source_is_required():
    with pytest.raises(SpecError, match="exactly one"):
        DatasetSpec(rows=10, arrow_type="float64", null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="exactly one"):
        DatasetSpec(rows=10, arrow_type="float64", value_range=(0.0, 1.0),
                    categories=["a"], null_count=0, seed=1).validate()


def test_value_range_must_be_ordered():
    with pytest.raises(SpecError, match="value_range"):
        base(value_range=(10.0, 0.0)).validate()


def test_unsigned_type_rejects_a_negative_range():
    with pytest.raises(SpecError, match="unsigned"):
        DatasetSpec(rows=10, arrow_type="uint32", value_range=(-5, 5),
                    null_count=0, seed=1).validate()


def test_array_parameters_require_a_list_type():
    with pytest.raises(SpecError, match="array_len_range"):
        base(array_len_range=(1, 5)).validate()
    DatasetSpec(rows=10, arrow_type="list<int32>", value_range=(0, 10),
                null_count=0, seed=1, array_len_range=(1, 5)).validate()


def test_string_type_requires_a_length_or_categories():
    with pytest.raises(SpecError, match="string_length"):
        DatasetSpec(rows=10, arrow_type="string", null_count=0, seed=1).validate()


def test_interarrival_cv_must_be_positive():
    with pytest.raises(SpecError, match="interarrival_cv"):
        base(interarrival_cv=0.0).validate()


def test_valid_spec_validates_silently():
    base(cardinality_ratio=0.75, entropy_bits=4.0, runs_z=-0.25).validate()


def test_decimal_places_required_for_float_and_decimal_types():
    with pytest.raises(SpecError, match="decimal_places"):
        DatasetSpec(rows=10, arrow_type="float64", value_range=(0.0, 1.0),
                    null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="decimal_places"):
        DatasetSpec(rows=10, arrow_type="decimal64(18,8)", value_range=(0.0, 1.0),
                    null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="decimal_places"):
        DatasetSpec(rows=10, arrow_type="list<float32>", value_range=(0.0, 1.0),
                    null_count=0, seed=1, array_len_range=(1, 3)).validate()


def test_decimal_places_must_be_non_negative():
    with pytest.raises(SpecError, match="decimal_places"):
        base(decimal_places=-1).validate()


def test_decimal_places_must_match_decimal_scale():
    with pytest.raises(SpecError, match="decimal_places"):
        DatasetSpec(rows=10, arrow_type="decimal64(18,8)", value_range=(0.0, 1.0),
                    decimal_places=5, null_count=0, seed=1).validate()
    DatasetSpec(rows=10, arrow_type="decimal64(18,8)", value_range=(0.0, 1.0),
                decimal_places=8, null_count=0, seed=1).validate()


def test_distribution_kind_must_be_recognised():
    with pytest.raises(SpecError, match="kind"):
        base(distribution=Distribution(kind="triangular")).validate()


@pytest.mark.parametrize("kind", ["normal", "lognormal"])
def test_spread_distributions_require_sigma_or_a_dispersion_target(kind):
    """Neither sigma nor index_of_dispersion fixes the spread, so the draw
    cannot proceed; that must be a SpecError here, not a ValueError later."""
    with pytest.raises(SpecError, match="sigma.*index_of_dispersion"):
        base(distribution=Distribution(kind=kind, mean=5.0)).validate()
    base(distribution=Distribution(kind=kind, mean=5.0, sigma=1.0)).validate()
    base(distribution=Distribution(kind=kind, mean=5.0),
         index_of_dispersion=0.5).validate()


def test_list_type_requires_array_len_range():
    with pytest.raises(SpecError, match="array_len_range.*list"):
        base(arrow_type="list<int32>", value_range=(0, 100),
             decimal_places=None).validate()
    base(arrow_type="list<int32>", value_range=(0, 100), decimal_places=None,
         array_len_range=(1, 3)).validate()


@pytest.mark.parametrize("field", ["cardinality_ratio", "entropy_bits",
                                   "index_of_dispersion", "runs_z",
                                   "interarrival_cv"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_float_targets_must_be_finite(field, value):
    kw = {field: value}
    if field == "entropy_bits":
        kw["cardinality_ratio"] = 0.5
    with pytest.raises(SpecError, match=f"{field}.*finite"):
        base(**kw).validate()


@pytest.mark.parametrize("bad", [(float("nan"), 1.0), (0.0, float("inf"))])
def test_value_range_bounds_must_be_finite(bad):
    with pytest.raises(SpecError, match="value_range.*finite"):
        base(value_range=bad).validate()


def test_string_length_must_be_positive():
    with pytest.raises(SpecError, match="string_length must be positive"):
        base(arrow_type="string", string_length=0, value_range=None,
             decimal_places=None).validate()


def test_bool_value_range_must_lie_within_the_lattice():
    with pytest.raises(SpecError, match="bool.*\[0, 1\]"):
        base(arrow_type="bool", value_range=(0, 5), decimal_places=None).validate()
    base(arrow_type="bool", value_range=(0, 1), decimal_places=None).validate()


@pytest.mark.parametrize("support", [
    dict(string_length=4, arrow_type="string"),
    dict(categories=["a", "b"], arrow_type="string"),
])
def test_a_density_profile_needs_a_numeric_support(support):
    """normal and lognormal profile the support by density; strings have none."""
    kw = dict(value_range=None, decimal_places=None, **support,
              distribution=Distribution(kind="normal", mean=1.0, sigma=1.0))
    with pytest.raises(SpecError, match="normal.*density.*numeric"):
        base(**kw).validate()


@pytest.mark.parametrize("support", [
    dict(string_length=4, arrow_type="string"),
    dict(categories=["a", "b"], arrow_type="string"),
])
def test_a_dispersion_target_needs_a_numeric_support(support):
    """index_of_dispersion is var/mean of the values; strings have neither."""
    with pytest.raises(SpecError, match="index_of_dispersion.*numeric"):
        base(value_range=None, decimal_places=None, index_of_dispersion=1.0,
             **support).validate()


def test_a_normal_with_no_mean_needs_value_range_to_centre_it():
    with pytest.raises(SpecError, match="normal.*mean.*value_range"):
        base(value_range=None, categories=[1.0, 2.0, 3.0],
             distribution=Distribution(kind="normal", sigma=1.0)).validate()
    base(value_range=None, categories=[1.0, 2.0, 3.0],
         distribution=Distribution(kind="normal", mean=2.0, sigma=1.0)).validate()


def test_lognormal_requires_a_positive_mean():
    with pytest.raises(SpecError, match="mean"):
        base(distribution=Distribution(kind="lognormal")).validate()
    with pytest.raises(SpecError, match="mean"):
        base(distribution=Distribution(kind="lognormal", mean=-1.0)).validate()
    base(distribution=Distribution(kind="lognormal", mean=1.0, sigma=0.5)).validate()


def test_categories_must_be_non_empty():
    with pytest.raises(SpecError, match="categories"):
        DatasetSpec(rows=10, arrow_type="string", categories=[],
                    null_count=0, seed=1).validate()


def test_array_null_count_must_be_within_the_inner_count():
    """The inner count `wrap` actually produces is rows * round(mean(range)),
    not rows * upper: 10 rows of (1, 5) hold 30 values, and 31 inner nulls
    made `_null_mask` raise a ValueError from `rng.choice`."""
    with pytest.raises(SpecError, match="array_null_count"):
        DatasetSpec(rows=10, arrow_type="list<int32>", value_range=(0, 10),
                    null_count=0, seed=1, array_len_range=(1, 5),
                    array_null_count=-1).validate()
    with pytest.raises(SpecError, match="array_null_count.*30"):
        DatasetSpec(rows=10, arrow_type="list<int32>", value_range=(0, 10),
                    null_count=0, seed=1, array_len_range=(1, 5),
                    array_null_count=31).validate()
    DatasetSpec(rows=10, arrow_type="list<int32>", value_range=(0, 10),
                null_count=0, seed=1, array_len_range=(1, 5),
                array_null_count=30).validate()


def test_categories_cannot_outnumber_rows():
    """Three categories over two rows cannot all appear at least once, which the
    multiset build otherwise turns into a zero count for one support value."""
    with pytest.raises(SpecError, match=r"categories.*rows"):
        DatasetSpec(rows=2, arrow_type="string", categories=["a", "b", "c"],
                    null_count=0, seed=1).validate()
    DatasetSpec(rows=3, arrow_type="string", categories=["a", "b", "c"],
                null_count=0, seed=1).validate()


def test_float32_rejects_more_decimal_places_than_it_can_represent():
    """Values are drawn and measured as float64 at decimal_places and then cast;
    float32 has 24 significant bits, so a grid finer than that collapses
    distinct values after measurement (use case 1's shape at 2 million rows:
    manifest cardinality 0.749989, array 0.7133). Reject when
    max(|lo|, |hi|) * 10**decimal_places exceeds 2**24."""
    with pytest.raises(SpecError, match="float32.*decimal_places"):
        base(arrow_type="float32", value_range=(-1.0, 10.0),
             decimal_places=8).validate()
    base(arrow_type="float32", value_range=(-1.0, 10.0), decimal_places=6).validate()
    base(arrow_type="float32", value_range=(0.0, 1.0), decimal_places=3).validate()
    with pytest.raises(SpecError, match="float32.*decimal_places"):
        base(arrow_type="list<float32>", value_range=(0.0, 1000.0),
             decimal_places=5, array_len_range=(1, 3)).validate()


def test_sigma_must_be_strictly_positive_when_set():
    """sigma=0 made scipy's density nan everywhere and the draw a constant."""
    with pytest.raises(SpecError, match="sigma.*positive"):
        base(distribution=Distribution(kind="normal", mean=5.0, sigma=0.0)).validate()
    with pytest.raises(SpecError, match="sigma.*positive"):
        base(distribution=Distribution(kind="lognormal", mean=5.0, sigma=-1.0)).validate()


@pytest.mark.parametrize("arrow_type,value_range", [
    ("int8", (0, 200)),
    ("uint16", (0, 70_000)),
    ("date32", (0, 2**40)),
    ("time32[s]", (0, 100_000)),
    ("time32[ms]", (0, 10**9)),
    ("time64[us]", (0, 10**12)),
    ("time64[ns]", (0, 10**15)),
    ("timestamp[ns]", (0, 2**70)),
    ("decimal32(9,2)", (0.0, 10**8)),
    ("list<date32>", (0, 2**40)),
])
def test_value_range_must_fit_the_element_type(arrow_type, value_range):
    """Every scalar family, not just integers and decimals: date32 beyond int32
    raised ArrowInvalid from the cast, timestamp beyond int64 a ValueError from
    the draw, and the time types were silently accepted past a day."""
    kw = dict(arrow_type=arrow_type, value_range=value_range, decimal_places=None)
    if arrow_type.startswith("decimal"):
        kw["decimal_places"] = 2
    if arrow_type.startswith("list"):
        kw["array_len_range"] = (1, 3)
    with pytest.raises(SpecError, match="does not fit"):
        base(**kw).validate()


@pytest.mark.parametrize("arrow_type,value_range", [
    ("time32[s]", (0, 86_399)),
    ("time64[ns]", (0, 86_399_999_999_999)),
    ("timestamp[ns]", (0, 2**62)),
    ("int64", (-2**63, 2**63 - 1)),
    ("uint64", (0, 2**64 - 1)),
])
def test_value_range_at_the_type_limit_is_accepted(arrow_type, value_range):
    base(arrow_type=arrow_type, value_range=value_range, decimal_places=None).validate()


def test_array_len_range_must_be_ordered_positive_and_average_at_least_one():
    def lst(rng):
        return DatasetSpec(rows=10, arrow_type="list<int32>", value_range=(0, 10),
                           null_count=0, seed=1, array_len_range=rng)
    with pytest.raises(SpecError, match="array_len_range"):
        lst((5, 1)).validate()
    with pytest.raises(SpecError, match="array_len_range"):
        lst((-1, 3)).validate()
    with pytest.raises(SpecError, match="array_len_range"):
        lst((0, 0)).validate()
    lst((1, 1)).validate()
    lst((1, 5)).validate()


def test_categories_must_match_the_arrow_type():
    """String categories under a numeric type reached the median dichotomy and
    raised a UFuncTypeError; numeric categories under string were cast."""
    with pytest.raises(SpecError, match="categories.*numeric"):
        DatasetSpec(rows=10, arrow_type="int32", categories=["a", "b"],
                    null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="categories.*string"):
        DatasetSpec(rows=10, arrow_type="string", categories=[1, 2],
                    null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="categories.*numeric"):
        DatasetSpec(rows=10, arrow_type="float64", categories=[1.0, True],
                    decimal_places=1, null_count=0, seed=1).validate()
    DatasetSpec(rows=10, arrow_type="int32", categories=[1, 2, 3],
                null_count=0, seed=1).validate()
    DatasetSpec(rows=10, arrow_type="string", categories=["a", "b"],
                null_count=0, seed=1).validate()


def test_dataset_id_accepts_numpy_scalar_categories():
    import numpy as np
    s = DatasetSpec(rows=10, arrow_type="int32", categories=[np.int64(1), np.int64(2)],
                    null_count=0, seed=1)
    s.validate()
    assert s.dataset_id == DatasetSpec(rows=10, arrow_type="int32",
                                       categories=[1, 2], null_count=0,
                                       seed=1).dataset_id


def test_numeric_categories_must_fit_the_element_type():
    """Categories bypass value_range, so they need the same fit check, or
    `wrap` would be the first to notice."""
    with pytest.raises(SpecError, match="categories.*does not fit"):
        DatasetSpec(rows=10, arrow_type="int8", categories=[1, 300],
                    null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match="categories.*does not fit"):
        DatasetSpec(rows=10, arrow_type="decimal32(9,8)", categories=[1.0, 10.0],
                    decimal_places=8, null_count=0, seed=1).validate()


def test_numeric_categories_must_sit_on_the_decimal_places_grid():
    """Categories [1.001, 1.002, 2.0] at 2 places measured a cardinality of 3
    before `wrap` rounded them to two distinct values, so the manifest lied.
    Every numeric category must equal itself rounded to decimal_places; the
    lattice types round to 0 places."""
    with pytest.raises(SpecError, match=r"categories.*1\.001.*decimal_places"):
        DatasetSpec(rows=10, arrow_type="float64", categories=[1.001, 1.002, 2.0],
                    decimal_places=2, null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match=r"categories.*0\.123456789.*decimal_places"):
        DatasetSpec(rows=10, arrow_type="decimal32(9,8)", categories=[0.123456789],
                    decimal_places=8, null_count=0, seed=1).validate()
    with pytest.raises(SpecError, match=r"categories.*1\.5"):
        DatasetSpec(rows=10, arrow_type="int32", categories=[1, 1.5],
                    null_count=0, seed=1).validate()
    DatasetSpec(rows=10, arrow_type="float64", categories=[1.0, 1.01, 2.0],
                decimal_places=2, null_count=0, seed=1).validate()
    DatasetSpec(rows=10, arrow_type="int32", categories=[1, 2.0],
                null_count=0, seed=1).validate()


def test_entropy_and_dispersion_targets_together_are_rejected():
    """The profile solve and the spread solve interact: under the entropy
    target's rank shuffle the sorted support reorders with sigma while the
    count permutation stays fixed, so the dispersion objective is not monotone
    in sigma and the bisection lands on an arbitrary point."""
    with pytest.raises(SpecError, match="entropy_bits.*index_of_dispersion.*not supported"):
        base(cardinality_ratio=0.1, entropy_bits=4.0, index_of_dispersion=0.5,
             distribution=Distribution(kind="lognormal", mean=4.0)).validate()


def test_dispersion_target_needs_a_distribution_with_a_spread_parameter():
    """uniform has no sigma, so the solve had nothing to bisect on and
    reported a clamp against a fixed value."""
    with pytest.raises(SpecError, match="index_of_dispersion.*uniform"):
        base(cardinality_ratio=0.1, index_of_dispersion=0.5).validate()


def test_explicit_sigma_and_a_dispersion_target_are_rejected():
    """One fixes the spread, the other asks to solve for it; the solve was
    silently skipped with no record."""
    with pytest.raises(SpecError, match="sigma.*index_of_dispersion"):
        base(cardinality_ratio=0.1, index_of_dispersion=0.5,
             distribution=Distribution(kind="normal", mean=5.0, sigma=1.0)).validate()
    base(cardinality_ratio=0.1, index_of_dispersion=0.5,
         distribution=Distribution(kind="normal", mean=5.0)).validate()
