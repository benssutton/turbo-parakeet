import numpy as np
import pyarrow as pa
import pytest

from chbench.datagen.spec import DatasetSpec
from chbench.datagen.wrap import wrap


def s(**kw) -> DatasetSpec:
    d = dict(
        rows=1000,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=0,
    )
    d.update(kw)
    return DatasetSpec(**d)


def vals(n=1000):
    return np.round(np.random.default_rng(0).uniform(-1, 10, n), 8)


@pytest.mark.parametrize(
    "t,pt",
    [
        ("float64", pa.float64()),
        ("float32", pa.float32()),
        ("int32", pa.int32()),
        ("uint32", pa.uint32()),
        ("decimal32(9,8)", pa.decimal32(9, 8)),
        ("decimal64(18,8)", pa.decimal64(18, 8)),
        ("string", pa.string()),
    ],
)
def test_casts_to_the_requested_arrow_type(t, pt):
    v = vals() if "decimal" in t or "float" in t else np.arange(1000)
    a = wrap(v, s(arrow_type=t, value_range=(0.0, 1000.0)), np.random.default_rng(0))
    assert a.type == pt
    assert len(a) == 1000


def test_exact_null_count_is_injected():
    a = wrap(vals(), s(null_count=137), np.random.default_rng(0))
    assert a.null_count == 137
    assert len(a) == 1000


def test_zero_nulls_gives_a_column_with_no_nulls():
    assert wrap(vals(), s(), np.random.default_rng(0)).null_count == 0


def test_null_positions_are_deterministic_for_a_seed():
    a = wrap(vals(), s(null_count=50), np.random.default_rng(7))
    b = wrap(vals(), s(null_count=50), np.random.default_rng(7))
    assert a.equals(b)


def test_round_trip_preserves_values_exactly_for_float64():
    v = vals()
    a = wrap(v, s(), np.random.default_rng(0))
    assert np.array_equal(a.to_numpy(zero_copy_only=False), v)


def test_decimal32_accepts_the_use_case_1_range():
    """Decimal32(8) holds -9.99999999 to +9.99999999, which covers [-1, 10] for every
    rounded value at or below the bound."""
    a = wrap(vals(), s(arrow_type="decimal32(9,8)"), np.random.default_rng(0))
    assert a.type == pa.decimal32(9, 8)


def test_decimal32_overflow_is_reported_against_rounded_values():
    """A draw in [9.999999995, 10) rounds to exactly 10.00000000 and overflows. The
    check must run on the rounded data, not on the spec's bound. Validation
    rejects any value_range whose rounded bounds overflow, so reaching this
    check is a generator bug, not a spec error: it is an AssertionError."""
    v = np.array([9.999999995, 1.0])
    with pytest.raises(AssertionError, match="validation"):
        wrap(v, s(arrow_type="decimal32(9,8)", rows=2), np.random.default_rng(0))


def test_integer_overflow_is_an_assertion_not_a_spec_error():
    with pytest.raises(AssertionError, match="validation"):
        wrap(
            np.array([300, 1]),
            s(arrow_type="int8", value_range=(0, 300), rows=2, decimal_places=None),
            np.random.default_rng(0),
        )


def test_list_type_chunks_into_arrays():
    v = np.arange(5000)
    spec = s(
        rows=1000,
        arrow_type="list<int32>",
        value_range=(0.0, 5000.0),
        array_len_range=(3, 7),
        decimal_places=None,
    )
    a = wrap(v, spec, np.random.default_rng(0))
    assert pa.types.is_list(a.type)
    assert a.type == spec.arrow_dtype
    assert len(a) == 1000
    assert sum(len(x) for x in a.to_pylist()) == 5000


def test_list_outer_nulls_are_injected():
    v = np.arange(5000)
    spec = s(
        rows=1000,
        arrow_type="list<int32>",
        value_range=(0.0, 5000.0),
        array_len_range=(3, 7),
        null_count=50,
        array_null_count=100,
        decimal_places=None,
    )
    a = wrap(v, spec, np.random.default_rng(0))
    assert a.null_count == 50
    assert len(a) == 1000
    assert a.type == spec.arrow_dtype
    # Inner nulls live in the child array and survive outer nulling even when
    # an inner-null position falls inside a row that also became outer-null
    # (that row's own nulls are then hidden by to_pylist but still present in
    # the underlying values buffer, which is what "preserved" means here).
    assert a.values.null_count == 100


def test_list_inner_nulls_are_injected():
    v = np.arange(5000)
    spec = s(
        rows=1000,
        arrow_type="list<int32>",
        value_range=(0.0, 5000.0),
        array_len_range=(3, 7),
        array_null_count=100,
        decimal_places=None,
    )
    a = wrap(v, spec, np.random.default_rng(0))
    inner = [e for row in a.to_pylist() for e in row]
    assert sum(e is None for e in inner) == 100


@pytest.mark.parametrize(
    "t,pt,v",
    [
        ("timestamp[ms]", pa.timestamp("ms"), np.arange(1000) * 86_400_000),
        ("date32", pa.date32(), np.arange(1000)),
        ("time32[ms]", pa.time32("ms"), np.arange(1000)),
    ],
)
def test_date_and_timestamp_cast_from_integers(t, pt, v):
    a = wrap(
        v,
        s(arrow_type=t, value_range=(0.0, 1e12), decimal_places=None),
        np.random.default_rng(0),
    )
    assert a.type == pt
