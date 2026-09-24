# Column GCD — Design

**Date:** 2026-09-23
**Status:** Approved in brainstorming, pending spec review

## Purpose

Compute the whole-column greatest common divisor of every integer-backed column
in a DataFrame, using the same method as ClickHouse's `GCD` compression codec.

## Semantics (matches ClickHouse `CompressionCodecGCD`)

- **GCD of raw physical integer values** — not of differences from an offset
  (ClickHouse leaves offset removal to `Delta`/`DoubleDelta`).
- **Signed values use magnitudes** (`unsigned_abs`, so `MIN` never overflows).
- **Nulls are skipped.** Zeros are the GCD identity (`gcd(g, 0) = g`).
- **All-null, all-zero, or zero-row column → `0`** (matches `math.gcd()` and
  `numpy.gcd.reduce` on empty input).
- **Results are in physical units.** Hourly `Datetime(us)` → `3_600_000_000`;
  `Decimal(10,2)` in 0.25 steps → `25`.
- **Early exit at the dtype's minimum increment.** Once the running GCD
  reaches the smallest non-zero value the dtype can express (one physical
  unit), no further value can lower it, so the scan stops. For every
  supported dtype that unit is physical `1`: 1 for integers, `10^-scale` for
  `Decimal`, 1 day for `Date`, 1 time-unit for `Datetime`/`Duration`, 1 ns
  for `Time`. While the running GCD is above 1, the scan continues.

### Supported dtypes

| Polars dtype | Physical | Magnitude path |
|---|---|---|
| `Int8`, `Int16`, `Int32`, `Int64` | itself | `unsigned_abs` → u64 |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | itself | u64 |
| `Int128`, `Decimal(p, s)` | i128 | `unsigned_abs` → u128 |
| `Date` | i32 (days) | `unsigned_abs` → u64 |
| `Datetime(unit, tz)`, `Duration(unit)`, `Time` | i64 | `unsigned_abs` → u64 |

Every other dtype (Float, String, Boolean, Categorical, List, Array, Struct,
…) produces a row with `gcd = null`.

**`i128::MIN` edge case:** an `Int128` column whose GCD magnitude is 2¹²⁷ (only
possible when every non-zero value is `i128::MIN`) cannot be represented in
`Int128` → `gcd = null`. Decimals cannot reach this (precision ≤ 38 digits).

## Rust implementation — `services/analytics/src/gcd.rs`

- New dependency: the `gcd` crate. `binary_u64` / `binary_u128` are
  Stein's (binary GCD) algorithm.
- Register `mod gcd;` in `lib.rs`.
- Plugin entry point, following the `ari.rs` pattern:

  ```rust
  #[polars_expr(output_type_func = gcd_output_type)]
  fn column_gcd(inputs: &[Series]) -> PolarsResult<Series>
  ```

  Output: `Struct { column: String, dtype: String, gcd: Int128 }`, one row per
  input column, in input column order. `dtype` is the Polars dtype's `Display`
  string (e.g. `"i64"`, `"datetime[μs]"`, `"decimal[10,2]"`, `"date"`).

- **Dispatch:** `series.to_physical_repr()`, then match the physical dtype to
  the u64 or u128 kernel. `encode_series` is *not* reused. It hashes decimals
  and reinterprets signed bits, which destroys the magnitudes a GCD needs.

- **Kernel** (generic over the u64 / u128 paths):
  - For each Arrow chunk (`downcast_iter()`), take the values slice and
    validity bitmap (respecting the array offset, for sliced series).
  - Split the slice into fixed chunks of ~64K values;
    `par_chunks(CHUNK).map(fold binary_gcd).reduce(|| 0, binary_gcd)`.
  - Mask null slots to `0` branch-free (GCD identity), so the inner loop has
    no null check. When a chunk has no validity bitmap, skip masking entirely.
  - Combine chunk results with `binary_gcd`.
  - **Fold step** (added during implementation): `gcd(g, v) = binary_gcd(g, v % g)`.
    Pure Stein on a small running `g` and a large `v` needs ~log₂(v) rounds per
    value and benchmarked slower than numpy's Euclid; one hardware remainder
    first leaves Stein two small operands (10M × 4 Int64: 494 ms → 28 ms).
  - **Early exit:** each parallel chunk folds in blocks of 1,024 values. One
    `AtomicBool` is shared across all of a column's chunks and Arrow arrays.
    It is checked before each block and set as soon as any block's running
    GCD reaches 1; once it is set, every chunk returns 1 at its next check.
    A `1` stored under a null slot is masked out and never triggers the exit.
- **Parallelism:** outer `par_iter` over columns, inner `par_chunks` within a
  column. rayon's work stealing covers both wide frames (many columns) and
  narrow, long frames (few columns).
- **Rust unit tests** (`#[cfg(test)]`): kernel on known multiples, coprimes,
  zeros, signed extremes, masked nulls, early exit (stops scanning; a masked
  `1` does not trigger it).

## Python API — `services/analytics/analytics/__init__.py`

```python
def column_gcd(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame
```

Mirrors `pairwise_adjusted_rand`: collects a LazyFrame, passes
`df.get_columns()` to `register_plugin_function(..., function_name="column_gcd",
is_elementwise=False, changes_length=True)`, returns a single struct column
`column_gcd`. The docstring documents physical units and the null policy.

## Testing — strict accuracy / performance split

### Convention (new, project-wide)

- `tests/test_*.py` — **accuracy only.** Asserts correctness against
  off-the-shelf reference implementations. Never times anything.
- `tests/performance/benchmark_*.py` — **performance only.** Standalone
  scripts run as `python tests/performance/benchmark_x.py`. They cross-check
  results for sanity but are never the correctness gate.
- Enforced by `pytest.ini`: `norecursedirs = .* __pycache__ performance`.
- Shared data stays in `tests/data/`.

### Migration

`git mv` the five existing scripts, renaming them to match the names CLAUDE.md
already uses:

| From | To |
|---|---|
| `tests/speed_benchmark_adjusted_rand.py` | `tests/performance/benchmark_adjusted_rand.py` |
| `tests/speed_benchmark_bloom_filter.py` | `tests/performance/benchmark_bloom_filter.py` |
| `tests/speed_benchmark_chi_squared.py` | `tests/performance/benchmark_chi_squared.py` |
| `tests/speed_benchmark_entropy.py` | `tests/performance/benchmark_entropy.py` |
| `tests/speed_benchmark_jaccard.py` | `tests/performance/benchmark_jaccard.py` |

Fix their `Path(__file__)`-relative paths (`parents[2]` for the repo root,
`parents[1] / "data"` for datasets). Each migrated script must still run.

### Accuracy — `tests/test_gcd.py`

References:
- **`math.gcd(*values)`** — primary oracle; arbitrary precision covers Int128,
  Decimal, `u64::MAX`, `i64::MIN`.
- **`numpy.gcd.reduce`** — second, independent oracle for native types of
  64 bits or fewer (excluding `i64::MIN` magnitudes, which numpy's int64 can't
  represent).

Scenarios, parametrised across `Int8–Int128`, `UInt8–UInt64`, `Decimal(p,s)`,
`Date`, `Datetime(ms/us/ns)`, `Duration`, `Time`:
- **Values:** multiples of a known g (`k·g`, random k) → g; coprime
  values → 1; single value → |v|; negatives; dtype extremes (`i8 -128` → 128,
  `u64::MAX`, `i64::MIN` → 2⁶³); zeros mixed with non-zeros; all zeros → 0;
  the `i128::MIN` edge → null.
- **Nulls:** scattered nulls; all-null → 0; zero-row frame → 0.
- **Arrow layout:** multi-chunk series (`rechunk=False`); series longer than
  the ~64K parallel chunk; sliced series (non-zero offset, validity-bitmap
  alignment).
- **Meaning checks:** hourly `Datetime(us)` → `3_600_000_000`; `Decimal(10,2)`
  in 0.25 steps → `25`; weekly `Date` → gcd of the epoch-day values.
- **Output contract:** non-integer dtypes (Float, String, Boolean, List)
  → `gcd = null`; `dtype` strings; row order equals input column order;
  output schema is `{column: String, dtype: String, gcd: Int128}`.
- **Seeded fuzz:** random frames compared against `math.gcd`. No new
  dependencies.

### Performance — `tests/performance/benchmark_gcd.py`

3 runs averaged; reports time and speedup; cross-checks equality with the
baselines.
- **Stress shape 1:** 10M rows × 4 `Int64` columns of `k·g`, no nulls.
- **Stress shape 2:** 1M rows × 100 `Int64` columns.
- **Realistic:** `tests/data/large_dataset.arrow` (50K rows × 101 columns).
- **Early exit:** 10M rows × 4 random `Int64` columns (GCD 1).

Shapes 1–2 have GCD `g` > 1, so they measure a full scan.
- **Baselines:** `numpy.gcd.reduce` per column (C ufunc); `math.gcd(*col)`
  on the realistic dataset only.

## Documentation

Update `CLAUDE.md`:
- a **Testing convention** section (as above);
- the project-structure tree (`gcd.rs`, `tests/test_gcd.py`,
  `tests/performance/`);
- "Exposed functions": `column_gcd(df)`;
- a seventh analytical technique: whole-column GCD (ClickHouse GCD-codec method).

## Out of scope

- Offset-invariant (step) GCD.
- Float columns.
