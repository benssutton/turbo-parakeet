"""The generator's last stage: nulls, list wrapping, and the Arrow cast.

This is the only module that touches pyarrow types. It takes the flat numpy
array of values produced by the two-stage generator and produces the
pyarrow.Array the loader writes to ClickHouse.

Order of operations, fixed:
  1. Round float/decimal values to `decimal_places`. The generator already
     rounds at draw time, but this must not assume that.
  2. Range-check the *rounded* values against the target type's representable
     range. `DatasetSpec.validate()` already rejects any value_range or
     category whose rounded bounds overflow the type, and every generated
     value lies inside those bounds, so this check is defence in depth: it
     raises AssertionError, not SpecError, because reaching it means the
     generator broke its own invariant. It matters most for decimals:
     Decimal32(9,8) holds -9.99999999..9.99999999, and a draw in
     [9.999999995, 10) rounds to exactly 10 and overflows even though the
     pre-round value looked safe (spikes/FINDINGS.md).
  3. Chunk into list<T> arrays if the type is a list.
  4. Inject nulls -- null_count at the outer level, array_null_count at the
     inner level for lists -- at uniformly random positions drawn from rng.
  5. Cast with pyarrow.

List types: the number of inner values is decided by
the caller (rows * round(mean(array_len_range))); wrap treats len(values) as
authoritative and draws `rows` lengths uniformly from array_len_range, then
adjusts them so they sum to exactly len(values), staying within the range
where possible.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from chbench.datagen.spec import DatasetSpec, representable_range


def wrap(values: np.ndarray, spec: DatasetSpec, rng: np.random.Generator) -> pa.Array:
    """Round, range-check, chunk, null-inject and cast `values` per `spec`.

    `values` is the flat sequence produced by the generator: `spec.rows`
    scalars, or for list<T>, `spec.rows * round(mean(array_len_range))` inner
    values. Every random draw goes through `rng`.
    """
    dtype = spec.arrow_dtype
    is_list = pa.types.is_list(dtype)
    elem_dtype = dtype.value_type if is_list else dtype

    values = _round_to_places(values, spec.decimal_places, elem_dtype)
    _range_check(values, spec.arrow_type, elem_dtype)

    if is_list:
        return _wrap_list(values, spec, elem_dtype, rng)
    return _wrap_scalar(values, spec, dtype, rng)


def _round_to_places(
    values: np.ndarray, decimal_places: int | None, elem_dtype: pa.DataType
) -> np.ndarray:
    values = np.asarray(values)
    if decimal_places is None:
        return values
    if pa.types.is_floating(elem_dtype) or pa.types.is_decimal(elem_dtype):
        return np.round(values, decimal_places)
    return values


def _range_check(values: np.ndarray, arrow_type: str, elem_dtype: pa.DataType) -> None:
    """Assert the (already rounded) values sit inside elem_dtype's representable
    range. Validation guarantees this for every value the generator can
    produce, so a failure here is a generator defect."""
    values = np.asarray(values)
    bounds = representable_range(elem_dtype)
    if values.size == 0 or bounds is None or values.dtype.kind not in "biuf":
        return
    lo, hi = bounds
    bad = (values < lo) | (values > hi)
    if np.any(bad):
        offender = values[bad][0]
        raise AssertionError(
            f"value {offender!r} does not fit in {arrow_type!r} (representable "
            f"range is [{lo}, {hi}]); validation should have caught this, so "
            "the generator produced a value outside the DatasetSpec's bounds"
        )


def _wrap_scalar(
    values: np.ndarray, spec: DatasetSpec, dtype: pa.DataType, rng: np.random.Generator
) -> pa.Array:
    mask = _null_mask(len(values), spec.null_count, rng)
    return _cast(values, dtype, mask)


def _wrap_list(
    values: np.ndarray,
    spec: DatasetSpec,
    elem_dtype: pa.DataType,
    rng: np.random.Generator,
) -> pa.Array:
    low, high = spec.array_len_range
    lengths = _chunk_lengths(len(values), spec.rows, low, high, rng)

    inner_mask = _null_mask(len(values), spec.array_null_count, rng)
    inner = _cast(values, elem_dtype, inner_mask)

    offsets = np.empty(spec.rows + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    outer_mask = _null_mask(spec.rows, spec.null_count, rng)
    # Nulling via the offsets array's own validity bitmap does not behave as a
    # simple "this row is null" flag (the arrow C++ implementation folds a
    # null offset's span into its neighbour), but ListArray.from_arrays takes
    # a separate `mask` argument for exactly this -- it nulls the outer rows
    # while leaving the offsets, and any inner nulls already injected, alone.
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int64()), inner, mask=pa.array(outer_mask)
    )


def _chunk_lengths(
    total: int, rows: int, low: int, high: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw `rows` lengths uniformly from [low, high], then nudge them so they
    sum to exactly `total`, staying within [low, high] where capacity allows.
    """
    lengths = rng.integers(low, high + 1, size=rows)
    diff = int(total) - int(lengths.sum())
    if diff == 0 or rows == 0:
        return lengths

    order = rng.permutation(rows)
    step = 1 if diff > 0 else -1
    remaining = abs(diff)

    capacity = (high - lengths)[order] if step > 0 else (lengths - low)[order]
    cum = np.cumsum(capacity)
    total_capacity = int(cum[-1])

    if remaining <= total_capacity:
        idx_full = int(np.searchsorted(cum, remaining, side="left"))
        adjust = np.zeros(rows, dtype=lengths.dtype)
        adjust[order[:idx_full]] = capacity[:idx_full]
        already = int(cum[idx_full - 1]) if idx_full > 0 else 0
        remainder = remaining - already
        if idx_full < rows:
            adjust[order[idx_full]] = remainder
        lengths = lengths + step * adjust
    else:
        # Even maxing (or minning) every element out can't reach `total`
        # while staying in range -- spill the rest outside [low, high].
        overflow = remaining - total_capacity
        lengths[order] = high if step > 0 else low
        spill_idx = rng.integers(0, rows, size=overflow)
        np.add.at(lengths, spill_idx, step)

    return lengths


def _null_mask(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if count:
        positions = rng.choice(n, size=count, replace=False)
        mask[positions] = True
    return mask


def _cast(values: np.ndarray, dtype: pa.DataType, mask: np.ndarray) -> pa.Array:
    if pa.types.is_decimal(dtype):
        # Phase one confirmed the float64 -> decimal path works on this
        # pyarrow version; go through pyarrow's own cast rather than
        # constructing decimal scalars by hand.
        src = pa.array(values, type=pa.float64(), mask=mask)
    elif (
        pa.types.is_timestamp(dtype)
        or pa.types.is_date(dtype)
        or pa.types.is_time(dtype)
    ):
        # Values are integers in the unit's resolution. The source integer
        # width must match the target's bit width: on this pyarrow version an
        # int64 -> date32/time32 cast raises ArrowNotImplementedError, while
        # int64 -> the 64-bit temporals (date64, timestamp, time64) works.
        int_type = pa.int32() if dtype.bit_width == 32 else pa.int64()
        src = pa.array(values, type=int_type, mask=mask)
    else:
        src = pa.array(values, mask=mask)
    return src.cast(dtype) if src.type != dtype else src
