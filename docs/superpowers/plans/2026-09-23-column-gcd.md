# Column GCD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Rust plugin function `column_gcd(df)` that returns the whole-column GCD of every integer-backed column (ClickHouse GCD-codec method). Also introduce a strict accuracy/performance split in `tests/`.

**Architecture:** A new `src/gcd.rs` reads each column's physical integer buffer directly (not through `encode_series`). It folds magnitudes with Stein's binary GCD from the `gcd` crate, runs columns in parallel and 64K-value chunks within each column in parallel with rayon, and masks null slots to 0 (the GCD identity). A shared per-column flag stops the scan once the running GCD reaches 1, the minimum increment for every supported dtype. The result is a struct Series `{column: String, dtype: String, gcd: Int128}` with one row per input column, exposed to Python through `analytics.column_gcd`.

**Tech Stack:** Rust (polars 0.51, polars-arrow 0.51, pyo3-polars 0.24, rayon, `gcd` 2.3), Python 3.12 (polars 1.41, numpy, pyarrow, pytest).

**Spec:** `docs/superpowers/specs/2026-09-23-column-gcd-design.md`

## Global Constraints

- The GCD is taken over the **raw physical integer values' magnitudes**, never over differences from an offset.
- **Early exit at the dtype's minimum increment, which is physical `1` for every supported dtype.** Once the running GCD reaches 1, the column's scan stops. While it is above 1, the scan continues. `BLOCK = 1 << 10` values are folded between checks of the shared flag.
- Nulls are skipped. All-null, all-zero and zero-row columns give `0`.
- Supported logical dtypes: `Int8/16/32/64`, `UInt8/16/32/64`, `Int128`, `Decimal`, `Date`, `Datetime`, `Duration`, `Time`. **Every other dtype, including Categorical/Enum (whose physical type is integer codes), gives `gcd = null`.**
- Output struct field names and types, exactly: `column: String`, `dtype: String`, `gcd: Int128`. The struct is named `column_gcd`. `dtype` is Rust `DataType`'s `Display` string (`"i64"`, `"datetime[μs]"`, `"decimal[10,2]"`, `"date"`).
- A GCD magnitude of 2¹²⁷ (every non-zero value is `i128::MIN`) gives `gcd = null`.
- Parallel chunk size: `CHUNK = 1 << 16` (65,536 values).
- Test layout: `tests/test_*.py` checks accuracy only and never times anything. `tests/performance/benchmark_*.py` holds standalone performance scripts that pytest never collects.
- Python env: `C:\Users\Ben\miniconda3\envs\p312\python.exe`. Every Bash command that builds or tests needs this environment block first:

  ```bash
  export CONDA_PREFIX='C:\Users\Ben\miniconda3\envs\p312'
  export PYO3_PYTHON='C:\Users\Ben\miniconda3\envs\p312\python.exe'
  export PATH="/c/Users/Ben/miniconda3/envs/p312:/c/Users/Ben/miniconda3/envs/p312/Scripts:$PATH"
  ```

  (`cargo test` needs `PYO3_PYTHON`, and it also needs the env dir on `PATH`, or the test exe fails with `STATUS_DLL_NOT_FOUND`.)
- Rust tests are always filtered to `gcd::`. `ari::tests::test_null_rows_dropped` already fails in debug builds (a `comb2(0)` underflow at `ari.rs:61`). That is outside this plan's scope, so don't fix it here.

## Review Focus

1. **Categorical / Enum columns.** Their physical representation is u32 codes, so a physical-type-only dispatch would report a bogus GCD. They must give `null`. Pinned in Task 2 (`non_integer_dtypes_are_none`) and Task 3 (`test_non_integer_dtypes_are_null`).
2. **Null slots carrying non-zero payloads** (Arrow data from pyarrow keeps whatever bytes sit under a null). The payloads must be ignored, and a `1` under a null must not trigger the early exit. Pinned in Task 2 (`masked_null_payloads_ignored`) and Task 3 (`test_null_payloads_ignored`).
3. **Sliced series with offsets that aren't multiples of 8.** The validity bitmap must stay aligned with the values. Pinned in Task 3 (`test_sliced_series`).
4. **Zero-row and zero-column frames.** These must not crash: zero-row gives 0 for integer columns, and zero-column gives an empty result with the right schema. Pinned in Task 3 (`test_zero_row_frame`, `test_zero_column_frame`).
5. **Timezone-aware Datetime.** It must behave like naive Datetime, using the physical i64. Pinned in Task 3 (the `Datetime_ns_UTC` case in `CASES`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pytest.ini` | Modify | Exclude `tests/performance/` from collection |
| `tests/performance/benchmark_{adjusted_rand,bloom_filter,chi_squared,entropy,jaccard}.py` | Move (`git mv`) + path fixes | Existing benchmarks |
| `services/analytics/Cargo.toml` | Modify | Add `gcd = "2.3"` |
| `services/analytics/src/lib.rs` | Modify | `mod gcd;` |
| `services/analytics/src/gcd.rs` | Create | Kernel, dtype dispatch, plugin entry, Rust unit tests |
| `services/analytics/analytics/__init__.py` | Modify | `column_gcd` Python wrapper |
| `tests/test_gcd.py` | Create | Accuracy suite against `math.gcd` / `numpy.gcd.reduce` |
| `tests/performance/benchmark_gcd.py` | Create | Stress benchmark |
| `CLAUDE.md` | Modify | Technique #7, exposed function, tree, testing convention |

---

### Task 1: Accuracy/performance test layout

**Files:**
- Modify: `pytest.ini`
- Move: `tests/speed_benchmark_*.py` → `tests/performance/benchmark_*.py`

**Interfaces:**
- Produces: the `tests/performance/` directory, which pytest never collects, and the `Path(__file__).parents[2]` (repo root) / `parents[1] / "data"` path convention that Task 4 follows.

- [ ] **Step 1: Record the current collection baseline**

Run (from repo root, with the env block):
```bash
python -m pytest --collect-only -q 2>&1 | tail -3
```
Note the "N tests collected" count. It must stay the same after this task.

- [ ] **Step 2: Move the benchmarks**

```bash
mkdir -p tests/performance
git mv tests/speed_benchmark_adjusted_rand.py tests/performance/benchmark_adjusted_rand.py
git mv tests/speed_benchmark_bloom_filter.py  tests/performance/benchmark_bloom_filter.py
git mv tests/speed_benchmark_chi_squared.py   tests/performance/benchmark_chi_squared.py
git mv tests/speed_benchmark_entropy.py       tests/performance/benchmark_entropy.py
git mv tests/speed_benchmark_jaccard.py       tests/performance/benchmark_jaccard.py
```

- [ ] **Step 3: Fix the paths (each file moved one level deeper)**

Apply these exact line replacements:

`tests/performance/benchmark_adjusted_rand.py`:
```python
_ANALYTICS_ROOT = Path(__file__).parents[2] / "services" / "analytics"
...
DATA_PATH = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
```
`tests/performance/benchmark_chi_squared.py` and `tests/performance/benchmark_jaccard.py`: the same two replacements, for the lines that currently read `Path(__file__).parent.parent / "services" / "analytics"` and `Path(__file__).parent / "data" / "large_dataset.arrow"`.

`tests/performance/benchmark_bloom_filter.py`:
```python
sys.path.insert(0, str(Path(__file__).parents[2]))
```
`tests/performance/benchmark_entropy.py`:
```python
DATA_PATH = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
```

Then confirm nothing old is left:
```bash
grep -n "parent.parent\|Path(__file__).parent /" tests/performance/*.py
```
Expected: no output.

- [ ] **Step 4: Enforce the split in `pytest.ini`**

Add this line under `[pytest]`:
```ini
norecursedirs = .* __pycache__ performance
```

- [ ] **Step 5: Verify collection is unchanged and excludes performance**

```bash
python -m pytest --collect-only -q 2>&1 | tail -3
python -m pytest --collect-only -q 2>&1 | grep -c "performance" || true
```
Expected: the same count as Step 1, and `0` for the grep.

- [ ] **Step 6: Verify every migrated script still runs**

Run each one from the repo root. `benchmark_entropy.py` includes the ~65 s three-way pass, so allow up to 10 min per script:
```bash
for f in tests/performance/benchmark_*.py; do echo "== $f"; python "$f" > /dev/null 2>&1 && echo OK || echo FAIL; done
```
Expected: `OK` for all five. `benchmark_bloom_filter.py` prints a warning but still exits 0 if `fastbloom-rs` is missing. If any script shows FAIL, rerun it without redirection to see the traceback and fix its path.

- [ ] **Step 7: Commit**

```bash
git add pytest.ini tests/performance
git commit -m "test: move benchmarks to tests/performance, exclude from pytest collection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Rust GCD kernel and plugin entry point

**Files:**
- Modify: `services/analytics/Cargo.toml` (under `[dependencies]`)
- Modify: `services/analytics/src/lib.rs` (module list)
- Create: `services/analytics/src/gcd.rs`

**Interfaces:**
- Consumes: `gcd::binary_u64(u64, u64) -> u64` and `gcd::binary_u128(u128, u128) -> u128` (Stein's algorithm; `gcd(0, x) = x`).
- Produces:
  - `pub(crate) fn series_gcd(s: &Series) -> PolarsResult<Option<i128>>`
  - `pub(crate) fn column_gcd_impl(inputs: &[Series]) -> PolarsResult<Series>`, returning `StructChunked` `"column_gcd"` with fields `column: String`, `dtype: String`, `gcd: Int128`
  - The plugin symbol `column_gcd` (no kwargs), called from Python in Task 3.

- [ ] **Step 1: Add the dependency and module**

`services/analytics/Cargo.toml`, after the `#Statistics` block:
```toml
#GCD (Stein's binary algorithm)
gcd = "2.3"
```
`services/analytics/src/lib.rs`, after `mod ari;`:
```rust
mod gcd;
```

- [ ] **Step 2: Write the failing Rust tests**

Create `services/analytics/src/gcd.rs` with only the tests for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use polars_arrow::bitmap::Bitmap;

    fn gcd_of(s: Series) -> Option<i128> {
        series_gcd(&s).unwrap()
    }

    #[test]
    fn multiples_of_known_gcd() {
        assert_eq!(gcd_of(Series::new("a".into(), &[12i64, -18, 30])), Some(6));
    }

    #[test]
    fn coprime_values_give_one() {
        assert_eq!(gcd_of(Series::new("a".into(), &[6i32, 10, 15])), Some(1));
    }

    #[test]
    fn zeros_are_identity_and_empty_is_zero() {
        assert_eq!(gcd_of(Series::new("a".into(), &[0i64, 0, 21, 0])), Some(21));
        assert_eq!(gcd_of(Series::new("a".into(), &[0i64, 0])), Some(0));
        assert_eq!(gcd_of(Series::new_empty("a".into(), &DataType::Int64)), Some(0));
    }

    #[test]
    fn nulls_skipped_and_all_null_is_zero() {
        assert_eq!(gcd_of(Series::new("a".into(), &[Some(12i64), None, Some(18)])), Some(6));
        assert_eq!(gcd_of(Series::new("a".into(), &[None::<i64>, None])), Some(0));
    }

    #[test]
    fn signed_and_unsigned_extremes() {
        assert_eq!(gcd_of(Series::new("a".into(), &[i8::MIN])), Some(128));
        assert_eq!(gcd_of(Series::new("a".into(), &[i64::MIN])), Some(1i128 << 63));
        assert_eq!(gcd_of(Series::new("a".into(), &[i64::MIN, 1i64 << 62])), Some(1i128 << 62));
        assert_eq!(gcd_of(Series::new("a".into(), &[u64::MAX])), Some(u64::MAX as i128));
        assert_eq!(gcd_of(Series::new("a".into(), &[i128::MIN, 1i128 << 126])), Some(1i128 << 126));
        // Magnitude 2^127 is not representable as Int128 → null.
        assert_eq!(gcd_of(Series::new("a".into(), &[i128::MIN])), None);
    }

    fn slice_gcd(values: &[u64], validity: Option<&Bitmap>) -> u64 {
        gcd_slice(values, validity, |v: u64| v, binary_u64, &AtomicBool::new(false))
    }

    #[test]
    fn masked_null_payloads_ignored() {
        // 1s sit under null slots. A correct mask ignores them: the result stays
        // 6, and they must not trigger the gcd == 1 early exit either.
        let values = vec![12u64, 1, 18, 1];
        let validity = Bitmap::from_iter([true, false, true, false]);
        assert_eq!(slice_gcd(&values, Some(&validity)), 6);
    }

    #[test]
    fn crosses_parallel_chunk_and_block_boundaries() {
        let n = 3 * CHUNK + 17;
        let mut values = vec![12u64; n];
        values[2 * CHUNK + BLOCK + 5] = 18;
        assert_eq!(slice_gcd(&values, None), 6);

        // A masked spoiler just past the first chunk boundary, and a valid 18
        // in a later block of the same chunk (bit iterator must stay aligned).
        let mut masked = vec![12u64; n];
        masked[CHUNK + 3] = 1;
        masked[CHUNK + 3 * BLOCK + 1] = 18;
        let validity = Bitmap::from_iter((0..n).map(|i| i != CHUNK + 3));
        assert_eq!(slice_gcd(&masked, Some(&validity)), 6);
    }

    #[test]
    fn early_exit_stops_scanning() {
        use std::sync::atomic::AtomicUsize;
        // A 1 in the first block: the column's GCD is final after BLOCK values.
        let n = 64 * CHUNK;
        let mut values = vec![12u64; n];
        values[0] = 1;
        let visited = AtomicUsize::new(0);
        let mag = |v: u64| {
            visited.fetch_add(1, Ordering::Relaxed);
            v
        };
        let g = gcd_slice(&values[..], None, mag, binary_u64, &AtomicBool::new(false));
        assert_eq!(g, 1);
        // Chunks already running when the flag is set stop at their next block,
        // so only a small fraction of the n values is ever read.
        let seen = visited.load(Ordering::Relaxed);
        assert!(seen < n / 4, "early exit did not stop the scan: visited {seen} of {n}");
    }

    #[test]
    fn no_early_exit_while_gcd_above_one() {
        // GCD bottoms out at 2 (never 1), so every value must be read.
        use std::sync::atomic::AtomicUsize;
        let n = 4 * CHUNK;
        let mut values = vec![4u64; n];
        values[n - 1] = 2;
        let visited = AtomicUsize::new(0);
        let mag = |v: u64| {
            visited.fetch_add(1, Ordering::Relaxed);
            v
        };
        assert_eq!(gcd_slice(&values[..], None, mag, binary_u64, &AtomicBool::new(false)), 2);
        assert_eq!(visited.load(Ordering::Relaxed), n);
    }

    #[test]
    fn early_exit_across_arrow_chunks() {
        // Chunk 1 reaches 1; chunk 2 (a huge multiple of 12) must not change it.
        let mut s = Series::new("a".into(), &[12i64, 7]);
        s.append(&Series::new("a".into(), vec![12i64; 2 * CHUNK])).unwrap();
        assert_eq!(s.n_chunks(), 2);
        assert_eq!(gcd_of(s), Some(1));
    }

    #[test]
    fn temporal_uses_physical_values() {
        let d = Series::new("d".into(), &[7i32, 14, 21]).cast(&DataType::Date).unwrap();
        assert_eq!(gcd_of(d), Some(7));
        let dt = Series::new("t".into(), &[3_600_000_000i64, 7_200_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(gcd_of(dt), Some(3_600_000_000));
    }

    #[test]
    fn non_integer_dtypes_are_none() {
        assert_eq!(gcd_of(Series::new("f".into(), &[2.0f64, 4.0])), None);
        assert_eq!(gcd_of(Series::new("s".into(), &["a", "b"])), None);
        assert_eq!(gcd_of(Series::new("b".into(), &[true, false])), None);
        // Categorical is physically u32 codes — must still be None.
        let cat = Series::new("c".into(), &["x", "y", "x"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        assert_eq!(gcd_of(cat), None);
    }

    #[test]
    fn impl_rows_follow_input_order() {
        let a = Series::new("a".into(), &[4i64, 8]);
        let b = Series::new("b".into(), &["x", "y"]);
        let c = Series::new("c".into(), &[9u8, 6]);
        let out = column_gcd_impl(&[a, b, c]).unwrap();
        let df = out.into_frame().unnest(["column_gcd"]).unwrap();
        let cols: Vec<_> = df.column("column").unwrap().str().unwrap().into_no_null_iter().collect();
        let dtypes: Vec<_> = df.column("dtype").unwrap().str().unwrap().into_no_null_iter().collect();
        let gcds: Vec<_> = df.column("gcd").unwrap().i128().unwrap().into_iter().collect();
        assert_eq!(cols, ["a", "b", "c"]);
        assert_eq!(dtypes, ["i64", "str", "u8"]);
        assert_eq!(gcds, [Some(4), None, Some(3)]);
    }
}
```

If `DataType::from_categories(Categories::global())` doesn't compile in polars 0.51 (the categorical API changed in 0.50), look up the constructor with `grep -rn "pub fn from_categories\|fn global" ~/.cargo/registry/src/*/polars-core-0.51.0/src/datatypes/` and use it. The test only needs *some* categorical Series.

- [ ] **Step 3: Run the tests and confirm they fail**

```bash
cd services/analytics && cargo test --lib gcd:: 2>&1 | tail -5
```
Expected: compile errors `cannot find function series_gcd` / `gcd_slice` / `column_gcd_impl`, and `CHUNK` not found.

- [ ] **Step 4: Implement**

Add this above the test module in `services/analytics/src/gcd.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// Whole-column GCD (ClickHouse GCD-codec method)
// ─────────────────────────────────────────────────────────────────────────────
//
// For each integer-backed column, the GCD of the magnitudes of its raw physical
// values — the quantity ClickHouse's `GCD` codec divides by. Decimal uses its
// unscaled i128; Date/Datetime/Duration/Time use their i32/i64 physical values.
//
// Null policy: nulls are skipped (masked to 0, the GCD identity). All-null,
// all-zero and zero-row columns → 0. Non-integer-backed dtypes → null.
//
// `encode_series` is deliberately not used: it hashes decimals and
// reinterprets signed bits, destroying the magnitudes a GCD needs.
//
// Parallelism: columns in parallel, and each column's values in CHUNK-sized
// slices in parallel (GCD is associative and commutative).
//
// Early exit: the smallest non-zero GCD any supported dtype can have is one
// physical unit — physical 1 (1 for integers, 10^-scale for Decimal, 1 day
// for Date, 1 time-unit for Datetime/Duration, 1 ns for Time). Once any block
// of a column reaches 1 the column's GCD is final, so a shared flag stops
// every other chunk at its next block boundary.

use gcd::{binary_u128, binary_u64};
use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

const CHUNK: usize = 1 << 16;
/// Values folded between checks of the early-exit flag.
const BLOCK: usize = 1 << 10;

// ─────────────────────────────────────────────────────────────────────────────
// Output type
// ─────────────────────────────────────────────────────────────────────────────

fn gcd_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("column".into(), DataType::String),
        Field::new("dtype".into(), DataType::String),
        Field::new("gcd".into(), DataType::Int128),
    ];
    Ok(Field::new("column_gcd".into(), DataType::Struct(fields)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// GCD of `mag(v)` over the valid slots of one Arrow array's values.
///
/// `values` and `validity` are both indexed from the array's logical start
/// (Arrow slices carry their own offsets), so chunk `i` covers bits
/// `i*CHUNK .. i*CHUNK + len`. Null slots contribute 0 via a select, not a
/// branch; without nulls the mask is skipped entirely.
///
/// Early exit: each chunk folds BLOCK values at a time, checking `done` before
/// each block and setting it when its running GCD reaches 1. A chunk that sees
/// `done` returns 1 — correct, because another chunk's GCD is 1, so the whole
/// column's is. Masked (null) slots contribute 0, so a 1 stored under a null
/// can never trigger the exit.
fn gcd_slice<T, U, M, G>(
    values: &[T],
    validity: Option<&Bitmap>,
    mag: M,
    gcd: G,
    done: &AtomicBool,
) -> U
where
    T: Copy + Sync,
    U: Copy + Send + Sync + Default + PartialEq + From<u8>,
    M: Fn(T) -> U + Sync,
    G: Fn(U, U) -> U + Sync + Send,
{
    let zero = U::default();
    let one = U::from(1u8);
    let validity = validity.filter(|bm| bm.unset_bits() > 0);
    values
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            let chunk_bits = validity.map(|bm| bm.clone().sliced(i * CHUNK, chunk.len()));
            let mut bits = chunk_bits.as_ref().map(|bm| bm.iter());
            let mut g = zero;
            for block in chunk.chunks(BLOCK) {
                if done.load(Ordering::Relaxed) {
                    return one;
                }
                g = match bits.as_mut() {
                    None => block.iter().fold(g, |g, &v| gcd(g, mag(v))),
                    // `zip` pulls from `block` first, so `it` advances exactly
                    // block.len() bits and stays aligned for the next block.
                    Some(it) => block
                        .iter()
                        .zip(it.by_ref())
                        .fold(g, |g, (&v, ok)| gcd(g, if ok { mag(v) } else { zero })),
                };
                if g == one {
                    done.store(true, Ordering::Relaxed);
                    return one;
                }
            }
            g
        })
        .reduce(|| zero, &gcd)
}

/// GCD across every Arrow chunk of a ChunkedArray. One early-exit flag spans
/// all of the column's Arrow chunks.
fn ca_gcd<T, U, M, G>(ca: &ChunkedArray<T>, mag: M, gcd: G) -> U
where
    T: PolarsNumericType,
    U: Copy + Send + Sync + Default + PartialEq + From<u8>,
    M: Fn(T::Native) -> U + Sync + Copy,
    G: Fn(U, U) -> U + Sync + Send + Copy,
{
    let done = AtomicBool::new(false);
    ca.downcast_iter()
        .map(|arr| gcd_slice(arr.values().as_slice(), arr.validity(), mag, gcd, &done))
        .fold(U::default(), gcd)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Checked on the LOGICAL dtype: Categorical/Enum are physically integer
/// codes and must not qualify.
fn is_integer_backed(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int128
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal(_, _)
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
    )
}

/// Whole-column GCD of `s`. `None` for non-integer-backed dtypes, and for the
/// unrepresentable magnitude 2¹²⁷ (every non-zero value is `i128::MIN`).
pub(crate) fn series_gcd(s: &Series) -> PolarsResult<Option<i128>> {
    if !is_integer_backed(s.dtype()) {
        return Ok(None);
    }
    let phys = s.to_physical_repr();
    let g: u128 = match phys.dtype() {
        DataType::Int8 => ca_gcd(phys.i8()?, |v: i8| v.unsigned_abs() as u64, binary_u64) as u128,
        DataType::Int16 => ca_gcd(phys.i16()?, |v: i16| v.unsigned_abs() as u64, binary_u64) as u128,
        DataType::Int32 => ca_gcd(phys.i32()?, |v: i32| v.unsigned_abs() as u64, binary_u64) as u128,
        DataType::Int64 => ca_gcd(phys.i64()?, |v: i64| v.unsigned_abs(), binary_u64) as u128,
        DataType::UInt8 => ca_gcd(phys.u8()?, |v: u8| v as u64, binary_u64) as u128,
        DataType::UInt16 => ca_gcd(phys.u16()?, |v: u16| v as u64, binary_u64) as u128,
        DataType::UInt32 => ca_gcd(phys.u32()?, |v: u32| v as u64, binary_u64) as u128,
        DataType::UInt64 => ca_gcd(phys.u64()?, |v: u64| v, binary_u64) as u128,
        DataType::Int128 => ca_gcd(phys.i128()?, |v: i128| v.unsigned_abs(), binary_u128),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("column_gcd: unexpected physical dtype {dt}").into(),
            ))
        }
    };
    Ok(i128::try_from(g).ok())
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn column_gcd_impl(inputs: &[Series]) -> PolarsResult<Series> {
    let gcds: Vec<Option<i128>> = inputs
        .par_iter()
        .map(series_gcd)
        .collect::<PolarsResult<_>>()?;

    let dtypes: Vec<String> = inputs.iter().map(|s| s.dtype().to_string()).collect();

    let column_s = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let dtype_s = StringChunked::from_iter(dtypes.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("dtype".into());
    let gcd_s = Int128Chunked::from_iter_options("gcd".into(), gcds.into_iter()).into_series();

    let struct_ca = StructChunked::from_series(
        "column_gcd".into(),
        inputs.len(),
        [column_s, dtype_s, gcd_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin entry point
// ─────────────────────────────────────────────────────────────────────────────

#[polars_expr(output_type_func=gcd_output_type)]
fn column_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    column_gcd_impl(inputs)
}
```

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
cd services/analytics && cargo test --lib gcd:: 2>&1 | tail -5
```
Expected: `test result: ok. 13 passed; 0 failed`.

- [ ] **Step 6: Confirm the release build compiles cleanly**

```bash
cd services/analytics && cargo build --release 2>&1 | grep -E "^(warning|error)" | grep -i gcd || echo "no gcd warnings"
```
Expected: `no gcd warnings`.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/Cargo.toml services/analytics/Cargo.lock services/analytics/src/lib.rs services/analytics/src/gcd.rs
git commit -m "feat: column_gcd Rust plugin — whole-column binary GCD, rayon-parallel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(If `Cargo.lock` isn't tracked, `git add` will complain; drop it from the command.)

---

### Task 3: Python wrapper and accuracy suite

**Files:**
- Modify: `services/analytics/analytics/__init__.py` (append a "Column GCD" section after `pairwise_adjusted_rand`, before the `"""Min Hash LSH"""` banner)
- Create: `tests/test_gcd.py`

**Interfaces:**
- Consumes: the plugin symbol `column_gcd` (Task 2), with no kwargs.
- Produces: `analytics.column_gcd(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame`, a single struct column `column_gcd` with fields `column: String`, `dtype: String`, `gcd: Int128`, one row per input column in input order. Task 4 uses it.

- [ ] **Step 1: Write the failing accuracy suite**

Create `tests/test_gcd.py`:

```python
"""
Whole-column GCD correctness tests: Rust plugin vs off-the-shelf references.

References:
    math.gcd(*values)  — primary oracle; arbitrary precision covers Int128,
                         Decimal, u64::MAX and i64::MIN magnitudes.
    numpy.gcd.reduce   — independent second oracle for ≤64-bit physical values
                         (skipped where a MIN magnitude can't be represented).

Semantics under test (ClickHouse GCD-codec method): GCD of the magnitudes of
the raw physical integer values; nulls skipped; all-null / all-zero / zero-row
→ 0; non-integer-backed dtypes (incl. Categorical/Enum) → null; a magnitude of
2**127 (only i128::MIN) → null.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import decimal
import math
import random
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

_analytics = pytest.importorskip("analytics")
column_gcd = _analytics.column_gcd

CHUNK = 1 << 16  # gcd.rs parallel chunk size
I128_LIMIT = 2**127
_DEC_CTX = decimal.Context(prec=80)

UNSIGNED = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plugin_gcds(df: pl.DataFrame | pl.LazyFrame) -> dict[str, int | None]:
    out = column_gcd(df).unnest("column_gcd")
    return dict(zip(out["column"].to_list(), out["gcd"].to_list()))


def plugin_gcd(s: pl.Series) -> int | None:
    return plugin_gcds(s.to_frame())[s.name]


def physical_dtype(dtype: pl.DataType) -> pl.DataType:
    if dtype == pl.Date:
        return pl.Int32()
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.Int64()
    return dtype


def from_physical(name: str, ints: list[int | None], dtype: pl.DataType) -> pl.Series:
    """Series of `dtype` whose physical values are exactly `ints` (None = null)."""
    if isinstance(dtype, pl.Decimal):
        vals = [None if v is None else Decimal(v).scaleb(-dtype.scale, context=_DEC_CTX) for v in ints]
        return pl.Series(name, vals, dtype=dtype)
    return pl.Series(name, ints, dtype=physical_dtype(dtype)).cast(dtype)


def ref_gcd(s: pl.Series) -> int | None:
    """math.gcd over physical values, with the plugin's 2**127 → null rule."""
    g = math.gcd(*s.to_physical().drop_nulls().to_list())
    return None if g >= I128_LIMIT else g


def numpy_gcd(s: pl.Series) -> int:
    return int(np.gcd.reduce(s.to_physical().drop_nulls().to_numpy()))


def numpy_applicable(s: pl.Series) -> bool:
    """numpy has no 128-bit ints, and |MIN| overflows its own dtype."""
    phys = s.to_physical()
    if phys.dtype in (pl.Int128,) or isinstance(s.dtype, pl.Decimal):
        return False
    if phys.dtype in UNSIGNED:
        return True
    bits = {pl.Int8: 8, pl.Int16: 16, pl.Int32: 32, pl.Int64: 64}[phys.dtype.base_type()]
    return -(2 ** (bits - 1)) not in phys.drop_nulls().to_list()


def is_signed(dtype: pl.DataType) -> bool:
    return not (isinstance(dtype, UNSIGNED) or dtype == pl.Time)


# (dtype, g, k_lo, k_hi): physical values are k·g, k ∈ [k_lo, k_hi], always incl. k=1.
CASES = [
    pytest.param(pl.Int8(), 4, -32, 31, id="Int8"),
    pytest.param(pl.Int16(), 12, -2_000, 2_000, id="Int16"),
    pytest.param(pl.Int32(), 1_000, -2_000_000, 2_000_000, id="Int32"),
    pytest.param(pl.Int64(), 3_600, -(2**40), 2**40, id="Int64"),
    pytest.param(pl.Int128(), 10**20, -(10**15), 10**15, id="Int128"),
    pytest.param(pl.UInt8(), 5, 0, 51, id="UInt8"),
    pytest.param(pl.UInt16(), 12, 0, 5_000, id="UInt16"),
    pytest.param(pl.UInt32(), 1_000, 0, 4_000_000, id="UInt32"),
    pytest.param(pl.UInt64(), 2**40, 0, 2**23, id="UInt64"),
    pytest.param(pl.Decimal(38, 4), 25, -(10**30), 10**30, id="Decimal38_4"),
    pytest.param(pl.Date(), 7, -5_000, 5_000, id="Date"),
    pytest.param(pl.Datetime("ms"), 60_000, -(10**6), 10**6, id="Datetime_ms"),
    pytest.param(pl.Datetime("us"), 3_600_000_000, -(10**5), 10**5, id="Datetime_us"),
    pytest.param(pl.Datetime("ns", "UTC"), 86_400 * 10**9, -(10**4), 10**4, id="Datetime_ns_UTC"),
    pytest.param(pl.Duration("us"), 250, -(10**9), 10**9, id="Duration_us"),
    pytest.param(pl.Time(), 15 * 60 * 10**9, 0, 95, id="Time"),
]
ALL_DTYPES = [pytest.param(p.values[0], id=p.id) for p in CASES]


# ─────────────────────────────────────────────────────────────────────────────
# Values
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype, g, k_lo, k_hi", CASES)
def test_multiples_of_known_gcd(dtype, g, k_lo, k_hi):
    rng = random.Random(1234)
    ints = [g] + [
        None if rng.random() < 0.1 else rng.randint(k_lo, k_hi) * g for _ in range(1_000)
    ]
    s = from_physical("x", ints, dtype)
    got = plugin_gcd(s)
    assert got == ref_gcd(s) == g
    if numpy_applicable(s):
        assert got == numpy_gcd(s)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_coprime_values_give_one(dtype):
    s = from_physical("x", [6, 10, 15], dtype)
    assert plugin_gcd(s) == ref_gcd(s) == 1


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_single_value_is_its_magnitude(dtype):
    v = -42 if is_signed(dtype) else 42
    s = from_physical("x", [v], dtype)
    assert plugin_gcd(s) == ref_gcd(s) == 42


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_zeros_and_nulls(dtype):
    assert plugin_gcd(from_physical("x", [0, 0, 12, None, 18, 0], dtype)) == 6
    assert plugin_gcd(from_physical("x", [0, 0, 0], dtype)) == 0
    assert plugin_gcd(from_physical("x", [None, None], dtype)) == 0
    assert plugin_gcd(from_physical("x", [None, 12, None, 18, None], dtype)) == 6


@pytest.mark.parametrize(
    "dtype, ints, expected",
    [
        pytest.param(pl.Int8(), [-128], 128, id="i8_min"),
        pytest.param(pl.Int8(), [-128, 64], 64, id="i8_min_and_64"),
        pytest.param(pl.UInt64(), [2**64 - 1], 2**64 - 1, id="u64_max"),
        pytest.param(pl.Int64(), [-(2**63)], 2**63, id="i64_min"),
        pytest.param(pl.Int64(), [-(2**63), 2**62], 2**62, id="i64_min_and_2^62"),
        pytest.param(pl.Int128(), [2**127 - 1], 2**127 - 1, id="i128_max"),
        pytest.param(pl.Int128(), [-(2**127), 2**126], 2**126, id="i128_min_and_2^126"),
        pytest.param(pl.Int128(), [-(2**127)], None, id="i128_min_unrepresentable"),
        pytest.param(pl.Int128(), [-(2**127), None, 0], None, id="i128_min_with_null_zero"),
    ],
)
def test_dtype_extremes(dtype, ints, expected):
    s = from_physical("x", ints, dtype)
    assert plugin_gcd(s) == ref_gcd(s) == expected


def test_zero_row_frame():
    schema = {f"c{i}": p.values[0] for i, p in enumerate(CASES)} | {"s": pl.String()}
    got = plugin_gcds(pl.DataFrame(schema=schema))
    assert got == {**{f"c{i}": 0 for i in range(len(CASES))}, "s": None}


def test_zero_column_frame():
    out = column_gcd(pl.DataFrame())
    assert out.height == 0
    assert out.schema == pl.Schema(
        {"column_gcd": pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arrow layout
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_chunk_series():
    parts = [pl.Series("x", [12, 24]), pl.Series("x", [None, 36]), pl.Series("x", [18])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    assert plugin_gcd(s) == ref_gcd(s) == 6


def test_long_series_crosses_parallel_chunks():
    n = 3 * CHUNK + 17
    values = np.full(n, 12, dtype=np.int64)
    values[2 * CHUNK + 5] = 18
    mask = np.arange(n) % 7 == 3  # scattered nulls
    assert not mask[2 * CHUNK + 5]  # the spoiler stays valid
    s = pl.from_arrow(pa.array(values, mask=mask))
    assert plugin_gcd(s) == ref_gcd(s) == numpy_gcd(s) == 6


def test_null_payloads_ignored():
    # 1s live under null slots. pyarrow keeps the payload bytes; a correct
    # null mask must ignore them — otherwise the GCD collapses to 1, and a
    # masked 1 would also wrongly trigger the gcd == 1 early exit.
    values = np.tile(np.array([12, 1, 18, 1], dtype=np.int64), 50_000)
    mask = values == 1
    arr = pa.array(values, mask=mask)
    assert np.frombuffer(arr.buffers()[1], dtype=np.int64)[1] == 1  # payload really is there
    s = pl.from_arrow(arr)
    assert plugin_gcd(s) == ref_gcd(s) == 6


def test_sliced_series():
    # Leading 7s are sliced away; offsets are deliberately not multiples of 8
    # so the validity bitmap has a sub-byte offset.
    n = 2 * CHUNK + 100
    values = np.full(n, 12, dtype=np.int64)
    values[:5] = 7
    values[CHUNK + 1] = 18
    mask = np.zeros(n, dtype=bool)
    mask[CHUNK + 2 :: 11] = True
    values[mask] = 7  # payloads under nulls
    s = pl.from_arrow(pa.array(values, mask=mask)).slice(5, 2 * CHUNK + 50)
    assert plugin_gcd(s) == ref_gcd(s) == 6
    s3 = pl.from_arrow(pa.array(values, mask=mask)).slice(3)  # keeps two leading 7s
    assert plugin_gcd(s3) == ref_gcd(s3) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Physical-unit results
# ─────────────────────────────────────────────────────────────────────────────

def test_hourly_datetime_us():
    s = pl.datetime_range(datetime(2024, 1, 1, 7), datetime(2024, 1, 3), "1h", time_unit="us", eager=True)
    assert plugin_gcd(s) == ref_gcd(s) == 3_600_000_000


def test_decimal_quarter_steps():
    s = pl.Series("p", [Decimal("1.25"), Decimal("0.50"), Decimal("0.75"), Decimal("-2.00")], dtype=pl.Decimal(10, 2))
    assert plugin_gcd(s) == ref_gcd(s) == 25


def test_weekly_dates():
    s = pl.date_range(date(2024, 1, 1), date(2024, 6, 30), "1w", eager=True)
    assert plugin_gcd(s) == ref_gcd(s) == numpy_gcd(s)  # raw epoch days, not the 7-day step


# ─────────────────────────────────────────────────────────────────────────────
# Output contract
# ─────────────────────────────────────────────────────────────────────────────

def test_non_integer_dtypes_are_null():
    df = pl.DataFrame(
        {
            "f64": pl.Series([2.0, 4.0], dtype=pl.Float64),
            "f32": pl.Series([2.0, 4.0], dtype=pl.Float32),
            "str": ["a", "b"],
            "bool": [True, False],
            "cat": pl.Series(["x", "y"], dtype=pl.Categorical),
            "enum": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
            "list": [[2, 4], [6]],
            "arr": pl.Series([[2, 4], [6, 8]], dtype=pl.Array(pl.Int64, 2)),
            "struct": [{"a": 2}, {"a": 4}],
            "bin": [b"\x02", b"\x04"],
            "null": pl.Series([None, None], dtype=pl.Null),
        }
    )
    assert plugin_gcds(df) == {c: None for c in df.columns}


def test_dtype_strings():
    df = pl.DataFrame(
        {
            "i64": pl.Series([1], dtype=pl.Int64),
            "u8": pl.Series([1], dtype=pl.UInt8),
            "i128": pl.Series([1], dtype=pl.Int128),
            "dec": pl.Series([Decimal("1.00")], dtype=pl.Decimal(10, 2)),
            "date": [date(2024, 1, 1)],
            "dt_us": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("us")),
            "dt_ns_utc": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ns", "UTC")),
            "dur_ms": pl.Series([1], dtype=pl.Duration("ms")),
            "f64": [1.0],
            "str": ["a"],
            "bool": [True],
        }
    )
    out = column_gcd(df).unnest("column_gcd")
    assert dict(zip(out["column"], out["dtype"])) == {
        "i64": "i64",
        "u8": "u8",
        "i128": "i128",
        "dec": "decimal[10,2]",
        "date": "date",
        "dt_us": "datetime[μs]",
        "dt_ns_utc": "datetime[ns, UTC]",
        "dur_ms": "duration[ms]",
        "f64": "f64",
        "str": "str",
        "bool": "bool",
    }


def test_output_schema_and_order():
    df = pl.DataFrame({"z": [4, 8], "a": ["x", "y"], "m": [9, 6]})
    out = column_gcd(df)
    assert out.schema == pl.Schema(
        {"column_gcd": pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})}
    )
    assert out.unnest("column_gcd")["column"].to_list() == ["z", "a", "m"]


def test_lazyframe_input():
    lf = pl.LazyFrame({"a": [12, 18], "b": [5, 10]})
    assert plugin_gcds(lf) == {"a": 6, "b": 5}


# ─────────────────────────────────────────────────────────────────────────────
# Seeded fuzz
# ─────────────────────────────────────────────────────────────────────────────

_FUZZ_INT_RANGES = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.Int64: (-(2**63), 2**63 - 1),
    pl.Int128: (-(2**127), 2**127 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
    pl.UInt64: (0, 2**64 - 1),
}


@pytest.mark.parametrize("seed", range(40))
def test_seeded_fuzz(seed):
    rng = random.Random(seed)
    dtype, (lo, hi) = rng.choice(list(_FUZZ_INT_RANGES.items()))
    g = rng.choice([1, 2, 3, 6, 7, 12, 1_000, 2**20, rng.randint(1, hi)])
    k_lo, k_hi = -(-lo // g), hi // g  # ceil(lo/g), floor(hi/g): k·g stays in range
    n = rng.choice([0, 1, 17, 1_000, CHUNK + rng.randint(1, 5_000)])
    null_rate = rng.choice([0.0, 0.05, 0.5, 1.0])
    ints = [None if rng.random() < null_rate else rng.randint(k_lo, k_hi) * g for _ in range(n)]

    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat(
        [pl.Series("x", ints[:cut], dtype=dtype), pl.Series("x", ints[cut:], dtype=dtype)],
        rechunk=False,
    )
    got = plugin_gcd(s)
    assert got == ref_gcd(s), f"seed={seed} dtype={dtype} g={g} n={n}"
    if numpy_applicable(s):
        assert got == numpy_gcd(s), f"numpy mismatch seed={seed}"
```

- [ ] **Step 2: Run the suite and confirm it fails**

```bash
python -m pytest tests/test_gcd.py -q 2>&1 | tail -3
```
Expected: the whole module is skipped or errors with `AttributeError: module 'analytics' has no attribute 'column_gcd'`.

- [ ] **Step 3: Implement the Python wrapper**

In `services/analytics/analytics/__init__.py`, insert this between the end of `pairwise_adjusted_rand` and the `"""\nMin Hash LSH\n"""` banner:

```python
"""
Column GCD
"""

_COLUMN_GCD_SCHEMA = pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})


def column_gcd(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Whole-column greatest common divisor, using the Rust plugin.

    Follows ClickHouse's GCD codec: the GCD of the magnitudes of each column's
    raw physical integer values. Results are in physical units — Decimal uses
    the unscaled integer (Decimal(10,2) in 0.25 steps -> 25), Date uses days,
    Datetime/Duration use their time unit, Time uses nanoseconds.

    Null policy: nulls are skipped. All-null, all-zero and zero-row columns
    -> 0. Non-integer-backed dtypes (float, string, boolean, categorical,
    nested, ...) -> null. A magnitude of 2**127 (only i128::MIN values) is not
    representable as Int128 -> null.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.

    Returns
    -------
    pl.DataFrame
        Single column "column_gcd" containing one struct per input column,
        in input order:
        - column: String - Column name
        - dtype: String - Polars dtype (Rust Display form, e.g. "datetime[μs]")
        - gcd: Int128 - Whole-column GCD, or null if not applicable
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if df.width == 0:
        return pl.DataFrame(schema={"column_gcd": _COLUMN_GCD_SCHEMA})
    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="column_gcd",
            args=df.get_columns(),
            is_elementwise=False,
            changes_length=True,
        ).alias("column_gcd")
    )
```

- [ ] **Step 4: Build the extension**

```bash
cd services/analytics && maturin develop --release 2>&1 | tail -3
```
Expected: `Installed analytics-0.1.0`.

- [ ] **Step 5: Run the suite and confirm it passes**

```bash
python -m pytest tests/test_gcd.py -q 2>&1 | tail -3
```
Expected: all tests pass with 0 failures.

If `test_zero_row_frame` fails because Polars doesn't call the plugin for zero-row input (the result has 0 rows instead of one row per column), add a zero-row guard to the wrapper right after the width check. It gives the plugin a one-row all-null frame, which the policy maps to the same answer (all-null → 0):
```python
    if df.height == 0:
        df = df.clear(n=1)
```

- [ ] **Step 6: Run the full accuracy suite to catch regressions**

```bash
python -m pytest -q 2>&1 | tail -3
```
Expected: no new failures compared with the Task 1 baseline (plus the new `test_gcd.py` passes).

- [ ] **Step 7: Commit**

```bash
git add services/analytics/analytics/__init__.py tests/test_gcd.py
git commit -m "feat: column_gcd Python API + accuracy suite vs math.gcd / numpy.gcd

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Performance benchmark

**Files:**
- Create: `tests/performance/benchmark_gcd.py`

**Interfaces:**
- Consumes: `analytics.column_gcd(df) -> pl.DataFrame` (Task 3) and the path convention from Task 1.

- [ ] **Step 1: Write the benchmark**

Create `tests/performance/benchmark_gcd.py`:

```python
"""
Benchmark: whole-column GCD — Rust plugin vs numpy.gcd.reduce vs math.gcd.

Performance only; correctness is gated by tests/test_gcd.py. Results are
cross-checked here purely as a sanity guard.

Shapes:
 1. 10M rows × 4 Int64 columns of k·g, no nulls  — narrow & long (intra-column parallelism)
 2. 1M rows × 100 Int64 columns of k·g          — wide (inter-column parallelism)
 3. tests/data/large_dataset.arrow (50K × 101)  — realistic mix; adds a math.gcd baseline
 4. 10M rows × 4 random Int64 columns (GCD 1)    — early exit at the minimum increment

Shapes 1–2 have GCD 3600 > 1, so the plugin must scan every value (full scan).
Shape 4 shows the early exit; numpy.gcd.reduce has none and always scans in full.

Baselines receive pre-extracted numpy arrays / Python lists (extraction is
not timed); the plugin is timed end-to-end from a DataFrame. RUNS runs averaged.

Run: python tests/performance/benchmark_gcd.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parents[2] / "services" / "analytics"))

from analytics import column_gcd

DATA_PATH = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
RUNS = 3
SEED = 42
G = 3_600


def timed(fn):
    times = []
    result = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, sum(times) / RUNS


def plugin(df: pl.DataFrame) -> dict[str, int | None]:
    out = column_gcd(df).unnest("column_gcd")
    return dict(zip(out["column"].to_list(), out["gcd"].to_list()))


def integer_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in df.schema.items() if dt.is_integer() or dt.is_temporal()]


def make_multiples(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    return pl.DataFrame(
        {f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * G for i in range(n_cols)}
    )


def make_random(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    return pl.DataFrame(
        {f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64) for i in range(n_cols)}
    )


def run_shape(name: str, df: pl.DataFrame, with_math: bool) -> None:
    cols = integer_columns(df)
    arrays = {c: df[c].to_physical().drop_nulls().to_numpy() for c in cols}

    got, t_plugin = timed(lambda: plugin(df))
    exp_np, t_np = timed(lambda: {c: int(np.gcd.reduce(a)) for c, a in arrays.items()})

    print(f"\n{name}: {df.height:,} rows × {df.width} cols ({len(cols)} integer-backed)")
    print(f"  plugin (column_gcd)  {t_plugin * 1e3:10.2f} ms")
    print(f"  numpy.gcd.reduce     {t_np * 1e3:10.2f} ms   speedup {t_np / t_plugin:6.1f}x")
    mismatches = [c for c in cols if got[c] != exp_np[c]]

    if with_math:
        lists = {c: a.tolist() for c, a in arrays.items()}
        exp_math, t_math = timed(lambda: {c: math.gcd(*v) for c, v in lists.items()})
        print(f"  math.gcd(*col)       {t_math * 1e3:10.2f} ms   speedup {t_math / t_plugin:6.1f}x")
        mismatches += [c for c in cols if got[c] != exp_math[c]]

    if mismatches:
        raise SystemExit(f"  RESULT MISMATCH in {sorted(set(mismatches))}")
    print("  results match")


def main() -> None:
    print(f"Whole-column GCD benchmark ({RUNS} runs averaged)")
    run_shape("Shape 1 — narrow/long", make_multiples(10_000_000, 4), with_math=False)
    run_shape("Shape 2 — wide", make_multiples(1_000_000, 100), with_math=False)
    run_shape("Shape 3 — large_dataset.arrow", pl.read_ipc(DATA_PATH), with_math=True)
    run_shape("Shape 4 — early exit (GCD 1)", make_random(10_000_000, 4), with_math=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python tests/performance/benchmark_gcd.py
```
Expected: four sections, each ending in `results match`, with plugin timings and speedups printed. Exit code 0.

- [ ] **Step 3: Confirm pytest doesn't collect it**

```bash
python -m pytest --collect-only -q 2>&1 | grep -c benchmark_gcd || true
```
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add tests/performance/benchmark_gcd.py
git commit -m "test: column_gcd stress benchmark vs numpy.gcd.reduce / math.gcd

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the benchmark numbers printed in Task 4, Step 2 (for the "Current Focus" line).

- [ ] **Step 1: Analytical Functions — add technique #7**

Change `Six techniques for surfacing column relationships within and between dataframes:` to `Seven techniques for surfacing column properties and relationships within and between dataframes:`. Then append after technique 6:

```markdown
**7. Whole-column GCD — ClickHouse GCD-codec method**
For each integer-backed column (Int/UInt 8–64, Int128, Decimal, Date, Datetime, Duration, Time), the GCD of the magnitudes of its raw physical values — the quantity ClickHouse's `GCD` codec divides by. Results are in physical units (Decimal → unscaled integer, Date → days, Datetime/Duration → time unit, Time → ns). Nulls skipped; all-null / all-zero / zero-row → 0; other dtypes (incl. Categorical/Enum) → null; magnitude 2¹²⁷ (only `i128::MIN`) → null. Early exit once the running GCD reaches physical 1 (the dtype's minimum increment); rayon-parallel across columns and 64K-value chunks.
```

- [ ] **Step 2: Project Structure tree**

In the `src/` listing, add after `ari.rs`:
```
│       │   ├── gcd.rs              # Whole-column GCD (binary/Stein, rayon-parallel)
```
Replace the whole `tests/` subtree with:
```
└── tests/
    ├── conftest.py                 # pytest markers (slow), shared dataset fixture
    ├── data/
    │   └── large_dataset.arrow     # 50K rows, 101 columns
    ├── test_similarity_filters.py  # Correctness + recall tests for similarity filters
    ├── test_adjusted_rand.py       # ARI correctness vs scikit-learn
    ├── test_bloom_filter.py        # Bloom filter correctness
    ├── test_chi_squared.py         # Chi-squared correctness vs scipy
    ├── test_entropy.py             # Joint entropy correctness
    ├── test_gcd.py                 # Column GCD correctness vs math.gcd / numpy.gcd
    └── performance/                # Benchmarks only — never collected by pytest
        ├── benchmark_adjusted_rand.py  # ARI benchmark vs scikit-learn
        ├── benchmark_bloom_filter.py   # Bloom filter benchmark vs fastbloom-rs
        ├── benchmark_chi_squared.py    # Chi-squared: Rust vs polars-ds vs scipy
        ├── benchmark_entropy.py        # Joint entropy: plugin vs native Polars
        ├── benchmark_jaccard.py        # MinHash/Jaccard similarity benchmark
        └── benchmark_gcd.py            # Column GCD vs numpy.gcd.reduce / math.gcd
```
(The `benchmark_rle.py` line in the old tree refers to a file that doesn't exist; drop it.)

- [ ] **Step 3: Exposed functions and testing convention**

Under "Exposed functions", add:
```markdown
- `column_gcd(df)` — whole-column GCD per column (struct: `column`, `dtype`, `gcd: Int128`)
```
After the `# Rust Plugin (analytics)` section, add:
```markdown
# Testing Convention
- `tests/test_*.py` — **accuracy only**: assert correctness against off-the-shelf reference implementations (scipy, scikit-learn, `math.gcd`, numpy, …). Never time anything.
- `tests/performance/benchmark_*.py` — **performance only**: standalone scripts (`python tests/performance/benchmark_x.py`). May sanity-check results, but are never the correctness gate. Excluded from pytest collection via `norecursedirs` in `pytest.ini`.
- Rust unit tests: `cargo test --lib <module>::` from `services/analytics/` with `PYO3_PYTHON` set to the env's `python.exe` and the env dir on `PATH` (else `STATUS_DLL_NOT_FOUND`).
```

- [ ] **Step 4: Current Focus**

Append one sentence to the "Current Focus" paragraph, using the Shape 1 and Shape 3 numbers from Task 4, Step 2. Fill in the measured values; don't leave the letters:
```markdown
Whole-column GCD (`column_gcd`) is complete: validated against `math.gcd`/`numpy.gcd.reduce`; <X> ms for 10M rows × 4 Int64 cols and <Y> ms on large_dataset.arrow (<Z>x faster than numpy.gcd.reduce).
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document column_gcd technique and accuracy/performance test convention

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
