# Skip-Index Techniques (minmax, set) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `analytics.minmax_index` and `analytics.set_index` — per-column, order-dependent techniques that measure how well a column, in its row order, suits ClickHouse's `minmax` and `set(max_size)` data-skipping indexes — with Rust kernels, exact Polars references, accuracy tests and speed benchmarks.

**Architecture:** A shared Rust core (`src/techniques/single_column/ordered/skip_index/mod.rs`) turns a column into a `Population` (dense rank ids in value order + counts) and sums exact integer skip numerators over windows (`WindowSums`, mergeable by addition — the streaming hook). `minmax.rs` slides monotonic deques; `set.rs` slides per-id counts; both add closed-form random-order baselines (`avoid`). Exposed as `api::{minmax_index, set_index}` → pyo3 → `_plugin` → `*Rust` classes. `*Polars` references (★) recompute everything with different algorithms (rolling min/max; occurrence intervals + difference arrays). Spec: `docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md`.

**Tech Stack:** Rust (polars 0.51, polars-arrow, rayon, foldhash, libm), pyo3/maturin, Python 3.12, Polars 1.44, NumPy, pytest.

---

## Conventions for every task

- Python: `C:/Users/Alexander/miniconda3/envs/p312/python.exe` (written `$PY` below; in Git Bash: `PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe`).
- Rust tests run from `services/analytics/`:
  `PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib skip_index::`
  (without the env dir on PATH: `STATUS_DLL_NOT_FOUND`).
- Rebuild the extension after any Rust change, from `services/analytics/`: `$PY -m maturin develop --release`. Python changes need no rebuild.
- pytest runs from the repo root: `$PY -m pytest tests/test_minmax_index.py -n 0 -q`.
- **Commits:** the pre-commit hook (rustfmt + black) needs `python` on PATH. Prefix every `git commit` with
  `PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"`. Never use `--no-verify`.
- `docs/superpowers/` is git-ignored but tracked: add files there with `git add -f`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Semantics in one paragraph (read before coding)

n rows; the population is the N′ non-null rows with d distinct values, value v occurring n_v times. The window is
w = min(n, `index_granularity × granularity`) rows; it slides one row at a time: K = n − w + 1 windows (K = 0 only
for n = 0). Each window gets two integer **skip numerators**: a_rows over N′ and a_values over d.
**minmax:** lo/hi = the window's non-null extremes (by rank); a_rows = N′ − (rows with value in [lo, hi]),
a_values = d − (hi − lo + 1); an all-null window gets a_rows = N′, a_values = d. **set:** entries = distinct non-null
values in the window + 1 if it holds a null; if `max_size` > 0 and entries > `max_size` → a_rows = a_values = 0
(overflow); else a_rows = N′ − Σ n_v over its values, a_values = d − (its distinct values). mean = Σa/(K·den),
std = population std. N′ = 0 or d = 0 → NaN. Baselines: §2.5 of the spec (`avoid(n, w, m)` = C(n − m, w)/C(n, w)).

## File map

| File | Action | Responsibility |
|---|---|---|
| `services/analytics/src/techniques/single_column/ordered/skip_index/mod.rs` | create | `Population`, float order keys, `WindowSums`/`Moments`, `windows`, `over_ranges`, `avoid`, `phi`, `Common`, output helpers |
| `services/analytics/src/techniques/single_column/ordered/skip_index/minmax.rs` | create | deque scan, minmax baseline, `minmax_index_impl` |
| `services/analytics/src/techniques/single_column/ordered/skip_index/set.rs` | create | per-id window scan, set baseline, `set_index_impl` |
| `services/analytics/src/techniques/single_column/ordered/mod.rs` | modify | `pub(crate) mod skip_index;` |
| `services/analytics/src/techniques/single_column/ordered/runs_test.rs` | modify | `supported` → `pub(crate)` (shared eligibility) |
| `services/analytics/src/techniques/single_column/mod.rs` | modify | doc comment |
| `services/analytics/src/techniques/mod.rs` | modify | re-export `skip_index` |
| `services/analytics/src/bindings/api/techniques.rs` | modify | `pub fn minmax_index`, `pub fn set_index` |
| `services/analytics/src/bindings/python.rs` | modify | pyfunctions + registration |
| `services/analytics/analytics/_plugin.py` | modify | `minmax_index`, `set_index` wrappers |
| `services/analytics/analytics/_dtypes.py` | modify | `ORDERED_SCALARS` |
| `services/analytics/analytics/runs_test/base.py` | modify | `ELIGIBLE = ORDERED_SCALARS` |
| `services/analytics/analytics/_skip_index.py` | create | `SkipIndex` base + the references' shared arithmetic |
| `services/analytics/analytics/minmax_index/{__init__,base,polars,rust}.py` | create | technique package |
| `services/analytics/analytics/set_index/{__init__,base,polars,rust}.py` | create | technique package |
| `tests/skip_index_helpers.py` | create | dtypes, brute-force oracle, Monte Carlo sampler, `assert_metrics` |
| `tests/test_minmax_index.py`, `tests/test_set_index.py` | create | accuracy tests |
| `tests/performance/benchmark_minmax_index.py`, `benchmark_set_index.py` | create | speed benchmarks |
| `CLAUDE.md`, `ToDo.md`, spec | modify | documentation |

---

### Task 1: Shared Rust core — population, window sums, baselines' building blocks

**Files:**
- Create: `services/analytics/src/techniques/single_column/ordered/skip_index/mod.rs`
- Create (stubs, filled in Tasks 2–3): `.../skip_index/minmax.rs`, `.../skip_index/set.rs`
- Modify: `services/analytics/src/techniques/single_column/ordered/mod.rs`
- Modify: `services/analytics/src/techniques/single_column/ordered/runs_test.rs` (the `fn supported` line and its doc comment)
- Modify: `services/analytics/src/techniques/single_column/mod.rs:1-2`
- Modify: `services/analytics/src/techniques/mod.rs:12`

- [ ] **Step 1: Wire the module tree**

`services/analytics/src/techniques/single_column/ordered/mod.rs` becomes:

```rust
//! Single-column techniques where row order matters: the runs test and the skip-index techniques
//! (ClickHouse minmax / set).

pub(crate) mod runs_test;
pub(crate) mod skip_index;
```

In `services/analytics/src/techniques/single_column/mod.rs`, replace the first two lines with:

```rust
//! Techniques that look at one column at a time: `unordered` ignores row order, `ordered`
//! (the runs test, the skip indexes) depends on it.
```

In `services/analytics/src/techniques/mod.rs`, replace `pub(crate) use single_column::ordered::runs_test;` with:

```rust
pub(crate) use single_column::ordered::{runs_test, skip_index};
```

In `runs_test.rs`, make the eligibility list shared — replace

```rust
/// Logical dtypes the kernel accepts — mirrored by analytics/runs_test/base.py::ELIGIBLE.
fn supported(dtype: &DataType) -> bool {
```

with

```rust
/// Logical dtypes the order-dependent kernels accept (runs test, skip indexes) — mirrored by
/// analytics/_dtypes.py::ORDERED_SCALARS.
pub(crate) fn supported(dtype: &DataType) -> bool {
```

Create the two stubs so the tree compiles:

`.../skip_index/minmax.rs`:

```rust
//! The minmax index (filled in by the next task).
```

`.../skip_index/set.rs`:

```rust
//! The set index (filled in by a later task).
```

- [ ] **Step 2: Write `mod.rs` with its failing tests**

`services/analytics/src/techniques/single_column/ordered/skip_index/mod.rs`:

```rust
//! Skip-index techniques (spec docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md):
//! how well a column, in its row order, suits ClickHouse's `minmax` and `set(max_size)`
//! data-skipping indexes.
//!
//! A window of w rows (one index block: `index_granularity × GRANULARITY`) slides one row at a
//! time, so every granule alignment counts once. For a point query `col = v`, a window's skip is
//! the share of query values it lets a query skip — v drawn by row frequency (`rows`, over the N′
//! non-null rows) or uniformly over the d distinct values (`values`). Each skip is an integer
//! numerator over N′ or d, summed exactly in `WindowSums`, so sums over any split of the windows
//! merge exactly. Closed-form baselines give the same expectations under a uniformly random row
//! order (`avoid`).
//!
//! Results = merge(scan(population, segment) for each segment) + baseline(population, w).
//! One-shot passes the whole column as one segment; the streaming recommender will pass its
//! reservoir's blocks.

pub(crate) mod minmax;
pub(crate) mod set;

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;

use foldhash::fast::RandomState;
use polars::prelude::*;
use polars_arrow::array::Array;
use rayon::prelude::*;

use crate::common::encode::{canon_f32, canon_f64};
use crate::techniques::runs_test::supported;

/// The id of a null row.
pub(crate) const NULL: u32 = u32::MAX;
/// Fewest window starts per parallel range.
const RANGE: usize = 1 << 16;
const ONE_CHUNK: &str = "rechunked to one chunk";

// ─────────────────────────────────────────────────────────────────────────────
// Population
// ─────────────────────────────────────────────────────────────────────────────

/// A column as dense rank ids in value order, with its value counts: what the window scans and
/// the baselines need.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Population {
    /// Each row's rank among the distinct non-null values (0 = smallest); `NULL` for a null.
    pub ids: Vec<u32>,
    /// Rows per id.
    pub counts: Vec<u64>,
    /// `cum[i]` = rows with id < i (d + 1 entries).
    pub cum: Vec<u64>,
    pub n_null: u64,
}

impl Population {
    /// Ranks rows `0..n` by key; `get(i)` is row i's key (None = null).
    fn from_keys<K, G>(n: usize, get: G) -> PolarsResult<Self>
    where
        K: Copy + Ord + Hash,
        G: Fn(usize) -> Option<K>,
    {
        let mut rank: HashMap<K, u32, RandomState> = HashMap::default();
        for k in (0..n).filter_map(&get) {
            rank.entry(k).or_insert(0);
        }
        if rank.len() >= NULL as usize {
            polars_bail!(ComputeError: "skip_index: {} distinct values exceed u32 ids", rank.len());
        }
        let mut keys: Vec<K> = rank.keys().copied().collect();
        keys.sort_unstable();
        for (i, k) in keys.iter().enumerate() {
            rank.insert(*k, i as u32);
        }
        let mut counts = vec![0u64; keys.len()];
        let ids: Vec<u32> = (0..n)
            .map(|i| match get(i) {
                Some(k) => {
                    let id = rank[&k];
                    counts[id as usize] += 1;
                    id
                }
                None => NULL,
            })
            .collect();
        Ok(Self::from_ids(ids, counts))
    }

    /// A population from row ids (`NULL` for nulls) and rows per id.
    pub(crate) fn from_ids(ids: Vec<u32>, counts: Vec<u64>) -> Self {
        let mut cum = Vec::with_capacity(counts.len() + 1);
        let mut total = 0u64;
        cum.push(0);
        for &c in &counts {
            total += c;
            cum.push(total);
        }
        let n_null = ids.len() as u64 - total;
        Self {
            ids,
            counts,
            cum,
            n_null,
        }
    }

    pub(crate) fn n(&self) -> u64 {
        self.ids.len() as u64
    }

    pub(crate) fn d(&self) -> u64 {
        self.counts.len() as u64
    }

    pub(crate) fn non_null(&self) -> u64 {
        self.cum[self.counts.len()]
    }
}

/// Canonical f64 bits (−0.0 = 0.0, one NaN) mapped so that integer order is numeric order, the
/// NaN after every number.
fn ordered_f64(x: f64) -> u64 {
    let b = canon_f64(x);
    if b >> 63 == 1 {
        !b
    } else {
        b | 1 << 63
    }
}

/// `ordered_f64` for f32.
fn ordered_f32(x: f32) -> u32 {
    let b = canon_f32(x) as u32;
    if b >> 31 == 1 {
        !b
    } else {
        b | 1 << 31
    }
}

/// Integer keys (temporal values, Decimal's unscaled i128, Enum codes).
fn numeric<T>(ca: &ChunkedArray<T>) -> PolarsResult<Population>
where
    T: PolarsNumericType,
    T::Native: Ord + Hash,
{
    let arr = ca.downcast_iter().next().expect(ONE_CHUNK);
    Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
}

/// A supported column's population, ordered as ClickHouse compares values: numbers and
/// temporals by value (Decimal by its unscaled i128 at the column's fixed scale), Boolean
/// false < true, String / Binary bytewise, Categorical by its string, Enum by category order
/// (its codes), floats canonical with the NaN last.
pub(crate) fn population(s: &Series) -> PolarsResult<Population> {
    if !supported(s.dtype()) {
        polars_bail!(
            InvalidOperation: "skip_index: unsupported dtype {} (column {})", s.dtype(), s.name()
        );
    }
    let s = match s.dtype() {
        DataType::Categorical(_, _) => s.cast(&DataType::String)?,
        _ => s.clone(),
    };
    // One chunk, so row i is index i of a single Arrow array (zero-copy if already one).
    let phys = s.to_physical_repr().rechunk();
    match phys.dtype() {
        DataType::Int8 => numeric(phys.i8()?),
        DataType::Int16 => numeric(phys.i16()?),
        DataType::Int32 => numeric(phys.i32()?),
        DataType::Int64 => numeric(phys.i64()?),
        DataType::UInt8 => numeric(phys.u8()?),
        DataType::UInt16 => numeric(phys.u16()?),
        DataType::UInt32 => numeric(phys.u32()?),
        DataType::UInt64 => numeric(phys.u64()?),
        DataType::Int128 => numeric(phys.i128()?),
        DataType::Float32 => {
            let arr = phys.f32()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| {
                arr.is_valid(i).then(|| ordered_f32(arr.value(i)))
            })
        }
        DataType::Float64 => {
            let arr = phys.f64()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| {
                arr.is_valid(i).then(|| ordered_f64(arr.value(i)))
            })
        }
        DataType::Boolean => {
            let arr = phys.bool()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::String => {
            let arr = phys.str()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        DataType::Binary => {
            let arr = phys.binary()?.downcast_iter().next().expect(ONE_CHUNK);
            Population::from_keys(arr.len(), |i| arr.is_valid(i).then(|| arr.value(i)))
        }
        dt => polars_bail!(ComputeError: "skip_index: unexpected physical dtype {dt}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Windows and their sums
// ─────────────────────────────────────────────────────────────────────────────

/// (effective window, number of windows): w = min(n, window_rows); n − w + 1 windows (none for
/// an empty column).
pub(crate) fn windows(n: u64, window_rows: u64) -> (u64, u64) {
    let w = n.min(window_rows);
    (w, if n == 0 { 0 } else { n - w + 1 })
}

/// Σa and Σa² of one skip numerator over windows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Moments {
    pub sum: u128,
    pub sum_sq: u128,
}

impl Moments {
    fn push(&mut self, a: u64) {
        let a = u128::from(a);
        self.sum += a;
        self.sum_sq += a * a;
    }

    fn merge(self, o: Self) -> Self {
        Self {
            sum: self.sum + o.sum,
            sum_sq: self.sum_sq + o.sum_sq,
        }
    }

    /// (mean, population std) of a / `den` over `k` windows; NaN without a window or a
    /// denominator.
    pub(crate) fn mean_std(self, k: u64, den: u64) -> (f64, f64) {
        if k == 0 || den == 0 {
            return (f64::NAN, f64::NAN);
        }
        let scale = k as f64 * den as f64;
        (
            self.sum as f64 / scale,
            spread(k, self.sum, self.sum_sq).sqrt() / scale,
        )
    }
}

/// K·Σa² − (Σa)² (≥ 0): exact in i128, f64 once that overflows.
fn spread(k: u64, sum: u128, sum_sq: u128) -> f64 {
    let exact = || -> Option<i128> {
        let (k, s, s2) = (
            i128::from(k),
            i128::try_from(sum).ok()?,
            i128::try_from(sum_sq).ok()?,
        );
        k.checked_mul(s2)?.checked_sub(s.checked_mul(s)?)
    };
    exact().map_or_else(
        || (k as f64 * sum_sq as f64 - (sum as f64).powi(2)).max(0.0),
        |v| v as f64,
    )
}

/// Sums over a set of windows. `merge` is exact and associative, so any split of the windows
/// gives the same result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WindowSums {
    pub windows: u64,
    /// Numerators over N′.
    pub rows: Moments,
    /// Numerators over d.
    pub values: Moments,
    /// Set only: Σ entries, the most entries in a window, windows whose set overflowed.
    pub entries: u128,
    pub max_entries: u64,
    pub overflowed: u64,
}

impl WindowSums {
    pub(crate) fn push(&mut self, rows: u64, values: u64) {
        self.windows += 1;
        self.rows.push(rows);
        self.values.push(values);
    }

    pub(crate) fn merge(self, o: Self) -> Self {
        Self {
            windows: self.windows + o.windows,
            rows: self.rows.merge(o.rows),
            values: self.values.merge(o.values),
            entries: self.entries + o.entries,
            max_entries: self.max_entries.max(o.max_entries),
            overflowed: self.overflowed + o.overflowed,
        }
    }
}

/// Window starts `0..k`, in parallel ranges of at least max(2^16, 4w, `min_len`) starts,
/// merged. Each range re-reads the w − 1 rows before its first full window, so the ranges are
/// long next to w; `min_len` lets the set scan amortise its per-id array.
pub(crate) fn over_ranges<F>(k: u64, w: u64, min_len: u64, scan: F) -> WindowSums
where
    F: Fn(Range<usize>) -> WindowSums + Sync,
{
    let k = k as usize;
    let len = (RANGE as u64).max(4 * w).max(min_len) as usize;
    (0..k.div_ceil(len))
        .into_par_iter()
        .map(|r| scan(r * len..((r + 1) * len).min(k)))
        .reduce(WindowSums::default, WindowSums::merge)
}

// ─────────────────────────────────────────────────────────────────────────────
// Baselines' building blocks
// ─────────────────────────────────────────────────────────────────────────────

/// P(a w-row window of a uniformly random order of n rows holds none of m given rows)
/// = C(n − m, w) / C(n, w), in log space.
pub(crate) fn avoid(n: u64, w: u64, m: u64) -> f64 {
    if m == 0 {
        return 1.0;
    }
    if n - m < w {
        return 0.0;
    }
    let lg = |x: u64| libm::lgamma(x as f64);
    (lg(n - m + 1) - lg(n - m - w + 1) - lg(n + 1) + lg(n - w + 1)).exp()
}

/// Φ, the standard normal CDF (libm's erfc: statrs' CDF is only ~1e-10 accurate).
pub(crate) fn phi(t: f64) -> f64 {
    0.5 * libm::erfc(-t / std::f64::consts::SQRT_2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Result rows
// ─────────────────────────────────────────────────────────────────────────────

/// The metrics both techniques report for one column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Common {
    pub n_rows: u64,
    pub n_non_null: u64,
    pub n_distinct: u64,
    pub window_rows: u64,
    pub n_windows: u64,
    pub skip_rows_mean: f64,
    pub skip_rows_std: f64,
    pub skip_values_mean: f64,
    pub skip_values_std: f64,
    pub expected_skip_rows: f64,
    pub expected_skip_values: f64,
}

impl Common {
    /// `expected` = (expected_skip_rows, expected_skip_values).
    pub(crate) fn new(pop: &Population, w: u64, sums: &WindowSums, expected: (f64, f64)) -> Self {
        let (skip_rows_mean, skip_rows_std) = sums.rows.mean_std(sums.windows, pop.non_null());
        let (skip_values_mean, skip_values_std) = sums.values.mean_std(sums.windows, pop.d());
        Self {
            n_rows: pop.n(),
            n_non_null: pop.non_null(),
            n_distinct: pop.d(),
            window_rows: w,
            n_windows: sums.windows,
            skip_rows_mean,
            skip_rows_std,
            skip_values_mean,
            skip_values_std,
            expected_skip_rows: expected.0,
            expected_skip_values: expected.1,
        }
    }

    /// One series per metric, named and ordered as the Python base's COMMON_METRICS.
    pub(crate) fn fields(rows: &[Self]) -> Vec<Series> {
        vec![
            u64s("n_rows", rows.iter().map(|r| r.n_rows)),
            u64s("n_non_null", rows.iter().map(|r| r.n_non_null)),
            u64s("n_distinct", rows.iter().map(|r| r.n_distinct)),
            u64s("window_rows", rows.iter().map(|r| r.window_rows)),
            u64s("n_windows", rows.iter().map(|r| r.n_windows)),
            f64s("skip_rows_mean", rows.iter().map(|r| r.skip_rows_mean)),
            f64s("skip_rows_std", rows.iter().map(|r| r.skip_rows_std)),
            f64s("skip_values_mean", rows.iter().map(|r| r.skip_values_mean)),
            f64s("skip_values_std", rows.iter().map(|r| r.skip_values_std)),
            f64s("expected_skip_rows", rows.iter().map(|r| r.expected_skip_rows)),
            f64s("expected_skip_values", rows.iter().map(|r| r.expected_skip_values)),
        ]
    }
}

pub(crate) fn u64s(name: &str, values: impl Iterator<Item = u64>) -> Series {
    UInt64Chunked::from_vec(name.into(), values.collect()).into_series()
}

pub(crate) fn f64s(name: &str, values: impl Iterator<Item = f64>) -> Series {
    Float64Chunked::from_vec(name.into(), values.collect()).into_series()
}

/// `column` then `fields`: one struct row per input column.
pub(crate) fn output(name: &str, inputs: &[Series], fields: Vec<Series>) -> PolarsResult<Series> {
    let column = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("column".into());
    let all: Vec<Series> = std::iter::once(column).chain(fields).collect();
    Ok(StructChunked::from_series(name.into(), inputs.len(), all.iter())?.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_keys_order_numerically_nan_last() {
        let xs = [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1e-300,
            1.0,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
        ];
        let keys: Vec<u64> = xs.iter().map(|&x| ordered_f64(x)).collect();
        assert!(keys.windows(2).all(|p| p[0] <= p[1]), "{keys:?}");
        assert_eq!(keys[2], keys[3]); // −0.0 = 0.0
        assert_eq!(keys[7], keys[8]); // one NaN
        assert!(keys[6] < keys[7]); // NaN after +inf
        let k32: Vec<u32> = [f32::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY, f32::NAN]
            .iter()
            .map(|&x| ordered_f32(x))
            .collect();
        assert!(k32.windows(2).all(|p| p[0] <= p[1]), "{k32:?}");
        assert_eq!(k32[2], k32[3]);
        assert!(k32[5] < k32[6]);
    }

    #[test]
    fn population_ranks_in_value_order() {
        let s = Series::new(
            "x".into(),
            &[Some("b"), None, Some("a"), Some("b"), Some("c"), None],
        );
        let p = population(&s).unwrap();
        assert_eq!(p.ids, [1, NULL, 0, 1, 2, NULL]);
        assert_eq!(p.counts, [1, 2, 1]);
        assert_eq!(p.cum, [0, 1, 3, 4]);
        assert_eq!((p.n(), p.non_null(), p.d(), p.n_null), (6, 4, 3, 2));
    }

    #[test]
    fn population_of_every_physical_family() {
        let ints = Series::new("x".into(), &[2i64, 0, 1, 0]);
        for dt in [
            DataType::Int8,
            DataType::UInt16,
            DataType::Float32,
            DataType::Float64,
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
            DataType::Duration(TimeUnit::Milliseconds),
            DataType::Time,
            DataType::Decimal(Some(10), Some(2)),
            DataType::String,
            DataType::Binary,
        ] {
            let s = if dt == DataType::Binary {
                ints.cast(&DataType::String).unwrap().cast(&dt).unwrap()
            } else {
                ints.cast(&dt).unwrap()
            };
            assert_eq!(population(&s).unwrap().ids, [2, 0, 1, 0], "{dt}");
        }
        let b = Series::new("x".into(), &[true, false, true]);
        assert_eq!(population(&b).unwrap().ids, [1, 0, 1]);
    }

    #[test]
    fn unsupported_dtype_is_invalid_operation() {
        let s = Series::new("x".into(), &[1i128, 2]);
        assert!(matches!(
            population(&s),
            Err(PolarsError::InvalidOperation(_))
        ));
    }

    #[test]
    fn windows_clamp_to_the_column() {
        assert_eq!(windows(10, 4), (4, 7));
        assert_eq!(windows(3, 8), (3, 1));
        assert_eq!(windows(0, 8), (0, 0));
    }

    #[test]
    fn avoid_edges_and_small_cases() {
        assert_eq!(avoid(5, 2, 0), 1.0);
        assert_eq!(avoid(4, 2, 3), 0.0);
        assert_eq!(avoid(3, 3, 1), 0.0); // the window is every row
        assert!((avoid(4, 2, 1) - 0.5).abs() < 1e-12);
        assert!((avoid(10, 3, 2) - 56.0 / 120.0).abs() < 1e-12);
    }

    #[test]
    fn mean_std_from_integer_sums() {
        let mut m = Moments::default();
        for a in [1u64, 3, 3, 5] {
            m.push(a);
        }
        let (mean, std) = m.mean_std(4, 8);
        let xs = [1.0f64 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0];
        let mu = xs.iter().sum::<f64>() / 4.0;
        let sd = (xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean - mu).abs() < 1e-15 && (std - sd).abs() < 1e-15);
        assert!(Moments::default().mean_std(0, 8).0.is_nan());
        assert!(m.mean_std(4, 0).1.is_nan());
    }

    #[test]
    fn spread_overflow_falls_back_to_f64() {
        // K·Σa² = 10⁴⁰ overflows i128: 10¹⁰ windows of a = 10¹⁰ each, true spread 0
        let (k, a) = (10_000_000_000u64, 10_000_000_000u128);
        let s = spread(k, u128::from(k) * a, u128::from(k) * a * a);
        assert!((0.0..=1e25).contains(&s), "{s}");
    }

    #[test]
    fn window_sums_merge_adds() {
        let mut a = WindowSums::default();
        a.push(1, 2);
        a.entries = 3;
        a.max_entries = 3;
        let mut b = WindowSums::default();
        b.push(4, 0);
        b.entries = 5;
        b.max_entries = 5;
        b.overflowed = 1;
        let m = a.merge(b);
        assert_eq!(
            (
                m.windows,
                m.rows.sum,
                m.rows.sum_sq,
                m.values.sum,
                m.entries,
                m.max_entries,
                m.overflowed
            ),
            (2, 5, 17, 2, 8, 5, 1)
        );
        assert_eq!(WindowSums::default().merge(a), a);
    }

    #[test]
    fn over_ranges_covers_every_start_once() {
        let k = 3 * RANGE as u64 + 17;
        let sums = over_ranges(k, 4, 0, |r| {
            let mut s = WindowSums::default();
            for i in r {
                s.push(i as u64, 1);
            }
            s
        });
        assert_eq!(sums.windows, k);
        assert_eq!(sums.rows.sum, (u128::from(k) - 1) * u128::from(k) / 2);
        assert_eq!(
            over_ranges(0, 4, 0, |_| unreachable!()),
            WindowSums::default()
        );
    }
}
```

- [ ] **Step 3: Run the tests**

Run (from `services/analytics/`): `PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib skip_index::`
Expected: the 10 tests above pass (the kernel files are stubs; dead-code warnings for `phi`, `output`, `Common`, `from_ids` are expected until Tasks 2–3). Also run `cargo test --lib runs_test::` — still passes.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/techniques
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Skip index: shared Rust core (population, window sums, avoid)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Rust minmax kernel

**Files:**
- Modify (replace the stub): `services/analytics/src/techniques/single_column/ordered/skip_index/minmax.rs`

- [ ] **Step 1: Write the kernel and its tests**

`minmax.rs`:

```rust
//! The minmax index: a window is skipped for `col = v` iff v < lo or v > hi, its non-null
//! extremes; an all-null window is skipped for every v. Two monotonic deques of row positions
//! give each window's extremes in O(1) amortised per row.

use std::collections::VecDeque;
use std::ops::Range;

use polars::prelude::*;
use rayon::prelude::*;

use super::{avoid, output, over_ranges, population, windows, Common, Population, WindowSums, NULL};

/// The windows starting at `starts` (each `w` rows), in one pass over rows
/// `starts.start .. starts.end + w − 1`.
pub(crate) fn scan(pop: &Population, w: usize, starts: Range<usize>) -> WindowSums {
    let (ids, nn, d) = (&pop.ids, pop.non_null(), pop.d());
    let mut sums = WindowSums::default();
    let (mut lows, mut highs) = (VecDeque::<usize>::new(), VecDeque::<usize>::new());
    for r in starts.start..starts.end + w - 1 {
        let id = ids[r];
        if id != NULL {
            while lows.back().is_some_and(|&b| ids[b] >= id) {
                lows.pop_back();
            }
            lows.push_back(r);
            while highs.back().is_some_and(|&b| ids[b] <= id) {
                highs.pop_back();
            }
            highs.push_back(r);
        }
        if r + 1 < starts.start + w {
            continue;
        }
        let s = r + 1 - w; // the window is rows s..=r
        while lows.front().is_some_and(|&f| f < s) {
            lows.pop_front();
        }
        while highs.front().is_some_and(|&f| f < s) {
            highs.pop_front();
        }
        match (lows.front(), highs.front()) {
            (Some(&lo), Some(&hi)) => {
                let (lo, hi) = (ids[lo] as usize, ids[hi] as usize);
                sums.push(nn - (pop.cum[hi + 1] - pop.cum[lo]), d - (hi - lo + 1) as u64);
            }
            _ => sums.push(nn, d),
        }
    }
    sums
}

/// (expected_skip_rows, expected_skip_values) under a uniformly random row order: a window
/// skips v iff it holds no row ≤ v or no row ≥ v; both events include the all-null window,
/// which is counted once.
pub(crate) fn baseline(pop: &Population, w: u64) -> (f64, f64) {
    let (n, nn) = (pop.n(), pop.non_null());
    let all_null = avoid(n, w, nn);
    let (mut rows, mut values) = (0.0, 0.0);
    for (v, &c) in pop.counts.iter().enumerate() {
        let p = avoid(n, w, pop.cum[v + 1]) + avoid(n, w, nn - pop.cum[v]) - all_null;
        rows += c as f64 * p;
        values += p;
    }
    (rows / nn as f64, values / pop.d() as f64)
}

/// One column's metrics.
pub(crate) fn column(s: &Series, window_rows: u64) -> PolarsResult<Common> {
    let pop = population(s)?;
    let (w, k) = windows(pop.n(), window_rows);
    let sums = over_ranges(k, w, 0, |r| scan(&pop, w as usize, r));
    Ok(Common::new(&pop, w, &sums, baseline(&pop, w)))
}

/// `column` + MinMaxIndex's metrics per column; columns in parallel (and each column's window
/// ranges).
pub(crate) fn minmax_index_impl(inputs: &[Series], window_rows: u64) -> PolarsResult<Series> {
    let rows: Vec<Common> = inputs
        .par_iter()
        .map(|s| column(s, window_rows))
        .collect::<PolarsResult<_>>()?;
    output("minmax_index", inputs, Common::fields(&rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(values: &[Option<i64>]) -> Population {
        population(&Series::new("x".into(), values)).unwrap()
    }

    fn whole(p: &Population, w: usize) -> WindowSums {
        scan(p, w, 0..p.ids.len() - w + 1)
    }

    /// A A A B A A A C (spec §7).
    const TOY: [Option<i64>; 8] = [
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(0),
        Some(0),
        Some(0),
        Some(2),
    ];

    #[test]
    fn toy_known_answer() {
        let p = pop(&TOY);
        let s = whole(&p, 4);
        // windows 0–3 cover [A, B] (1 row of 8 outside, 1 value of 3); window 4 covers [A, C]
        assert_eq!(
            (s.windows, s.rows.sum, s.rows.sum_sq, s.values.sum, s.values.sum_sq),
            (5, 4, 4, 4, 4)
        );
        let c = Common::new(&p, 4, &s, baseline(&p, 4));
        assert!((c.skip_rows_mean - 0.1).abs() < 1e-15);
        assert!((c.skip_rows_std - 0.05).abs() < 1e-15);
        assert!((c.skip_values_mean - 4.0 / 15.0).abs() < 1e-15);
        assert!((c.skip_values_std - 2.0 / 15.0).abs() < 1e-15);
    }

    #[test]
    fn all_null_window_skips_everything() {
        let p = pop(&[Some(1), None, None, Some(2)]);
        // windows {1}, {}, {2}: rows numerators 1, 2, 1; values 1, 2, 1
        let s = whole(&p, 2);
        assert_eq!((s.rows.sum, s.values.sum), (4, 4));
    }

    #[test]
    fn merged_ranges_equal_one_scan_at_every_split() {
        let values: Vec<Option<i64>> = (0..40)
            .map(|i| (i % 7 != 3).then_some((i * 13 % 9) as i64))
            .collect();
        let p = pop(&values);
        for w in [1, 3, 8] {
            let k = values.len() - w + 1;
            let all = scan(&p, w, 0..k);
            for i in 1..k {
                assert_eq!(
                    scan(&p, w, 0..i).merge(scan(&p, w, i..k)),
                    all,
                    "w {w}, split {i}"
                );
            }
        }
    }

    #[test]
    fn baseline_hand_worked() {
        // 1 2 3 4, w = 2: 6 equally likely pairs, 8 of 24 (pair, query) combinations skipped
        let (rows, values) = baseline(&pop(&[Some(1), Some(2), Some(3), Some(4)]), 2);
        assert!((rows - 1.0 / 3.0).abs() < 1e-12);
        assert!((values - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn short_and_all_null_columns() {
        let c = column(&Series::new("x".into(), &[3i64, 1, 2]), 8).unwrap();
        assert_eq!(
            (c.window_rows, c.n_windows, c.skip_rows_mean, c.expected_skip_rows),
            (3, 1, 0.0, 0.0)
        );
        let c = column(&Series::new("x".into(), &[None::<i64>, None, None]), 2).unwrap();
        assert_eq!((c.n_distinct, c.n_windows), (0, 2));
        assert!(c.skip_rows_mean.is_nan() && c.skip_values_std.is_nan());
        assert!(c.expected_skip_values.is_nan());
    }

    #[test]
    fn impl_returns_one_struct_row_per_column() {
        let out = minmax_index_impl(
            &[
                Series::new("a".into(), &[1i64, 1, 2]),
                Series::new("b".into(), &["x", "y", "y"]),
            ],
            2,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        let fields = out.struct_().unwrap().fields_as_series();
        let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
        assert_eq!(
            names,
            [
                "column",
                "n_rows",
                "n_non_null",
                "n_distinct",
                "window_rows",
                "n_windows",
                "skip_rows_mean",
                "skip_rows_std",
                "skip_values_mean",
                "skip_values_std",
                "expected_skip_rows",
                "expected_skip_values"
            ]
        );
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib skip_index::`
Expected: all `skip_index::tests` and `skip_index::minmax::tests` pass.

- [ ] **Step 3: Commit**

```bash
git add services/analytics/src/techniques/single_column/ordered/skip_index/minmax.rs
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Skip index: minmax kernel (monotonic deques) and baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Rust set kernel

**Files:**
- Modify (replace the stub): `services/analytics/src/techniques/single_column/ordered/skip_index/set.rs`

- [ ] **Step 1: Write the kernel and its tests**

`set.rs`:

```rust
//! The set(max_size) index: a window is skipped for `col = v` iff v is not among its values and
//! its stored set has not overflowed — a set of more than `max_size` entries (its distinct
//! values, plus NULL when the window holds one; 0 = unlimited) is stored empty and skips nothing.
//! The scan keeps per-id row counts for the window, O(1) per row entering and leaving.

use std::ops::Range;

use polars::prelude::*;
use rayon::prelude::*;

use super::{
    avoid, f64s, output, over_ranges, phi, population, u64s, windows, Common, Population,
    WindowSums, NULL,
};

/// One window's contents: rows per id, its distinct values and their population rows, its
/// nulls.
struct Window<'a> {
    pop: &'a Population,
    inside: Vec<u32>,
    present: u64,
    mass: u64,
    nulls: u64,
}

impl<'a> Window<'a> {
    fn new(pop: &'a Population) -> Self {
        Self {
            pop,
            inside: vec![0; pop.counts.len()],
            present: 0,
            mass: 0,
            nulls: 0,
        }
    }

    fn enter(&mut self, id: u32) {
        if id == NULL {
            self.nulls += 1;
            return;
        }
        let slot = &mut self.inside[id as usize];
        if *slot == 0 {
            self.present += 1;
            self.mass += self.pop.counts[id as usize];
        }
        *slot += 1;
    }

    fn leave(&mut self, id: u32) {
        if id == NULL {
            self.nulls -= 1;
            return;
        }
        let slot = &mut self.inside[id as usize];
        *slot -= 1;
        if *slot == 0 {
            self.present -= 1;
            self.mass -= self.pop.counts[id as usize];
        }
    }

    /// The stored set's entries: the distinct values, plus NULL.
    fn entries(&self) -> u64 {
        self.present + u64::from(self.nulls > 0)
    }
}

/// The windows starting at `starts` (each `w` rows), in one pass over rows
/// `starts.start .. starts.end + w − 1`.
pub(crate) fn scan(pop: &Population, w: usize, max_size: u64, starts: Range<usize>) -> WindowSums {
    let (nn, d) = (pop.non_null(), pop.d());
    let mut win = Window::new(pop);
    let mut sums = WindowSums::default();
    for r in starts.start..starts.end + w - 1 {
        win.enter(pop.ids[r]);
        if r + 1 < starts.start + w {
            continue;
        }
        let s = r + 1 - w; // the window is rows s..=r
        if s > starts.start {
            win.leave(pop.ids[s - 1]);
        }
        let entries = win.entries();
        sums.entries += u128::from(entries);
        sums.max_entries = sums.max_entries.max(entries);
        if max_size > 0 && entries > max_size {
            sums.overflowed += 1;
            sums.push(0, 0);
        } else {
            sums.push(nn - win.mass, d - win.present);
        }
    }
    sums
}

/// The set baseline under a uniformly random row order (spec §2.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Baseline {
    pub skip_rows: f64,
    pub skip_values: f64,
    pub window_distinct: f64,
    pub overflow: f64,
}

/// (count, multiplicity) classes of `counts`, ascending: entries with equal counts have equal
/// absence probabilities, so the baseline's sums run over classes.
fn classes(counts: impl IntoIterator<Item = u64>) -> Vec<(u64, u64)> {
    let mut sorted: Vec<u64> = counts.into_iter().collect();
    sorted.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for c in sorted {
        match out.last_mut() {
            Some((last, m)) if *last == c => *m += 1,
            _ => out.push((c, 1)),
        }
    }
    out
}

/// Expected entries present in a w-row window of a random order of n rows, one entry of count
/// `without` (if any) set aside.
fn mean_present(n: u64, w: u64, entries: &[(u64, u64)], without: Option<u64>) -> f64 {
    entries
        .iter()
        .map(|&(c, m)| {
            let m = m - u64::from(Some(c) == without);
            if m == 0 {
                0.0
            } else {
                m as f64 * (1.0 - avoid(n, w, c))
            }
        })
        .sum()
}

/// Exact std of the entries present: Σ q(1 − q) + Σ_{u≠u′} [avoid(n_u + n_u′) − q_u q_u′].
fn std_present(n: u64, w: u64, entries: &[(u64, u64)]) -> f64 {
    let q: Vec<f64> = entries.iter().map(|&(c, _)| avoid(n, w, c)).collect();
    let mut var = 0.0;
    for (i, &(c, m)) in entries.iter().enumerate() {
        var += m as f64 * q[i] * (1.0 - q[i]);
        for (j, &(c2, m2)) in entries.iter().enumerate() {
            let pairs = m * (m2 - u64::from(i == j));
            if pairs > 0 {
                var += pairs as f64 * (avoid(n, w, c + c2) - q[i] * q[j]);
            }
        }
    }
    var.max(0.0).sqrt()
}

/// Exact with `max_size` = 0; otherwise the entry count is approximated as normal (exact mean,
/// variance and conditional means; continuity-corrected).
pub(crate) fn baseline(pop: &Population, w: u64, max_size: u64) -> Baseline {
    let n = pop.n();
    let values = classes(pop.counts.iter().copied());
    let entries = classes(
        pop.counts
            .iter()
            .copied()
            .chain((pop.n_null > 0).then_some(pop.n_null)),
    );
    let window_distinct = mean_present(n, w, &entries, None);
    let sigma = if max_size == 0 {
        0.0
    } else {
        std_present(n, w, &entries)
    };
    // P(entry count ≤ max_size) for a mean entry count; 1 when unlimited.
    let fits = |mean: f64| {
        if max_size == 0 {
            return 1.0;
        }
        let x = max_size as f64 + 0.5 - mean;
        if sigma > 0.0 {
            phi(x / sigma)
        } else if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    };
    let (mut rows, mut skip_values) = (0.0, 0.0);
    for &(c, m) in &values {
        let absent = avoid(n, w, c);
        let p = if max_size == 0 || absent == 0.0 {
            absent
        } else {
            absent * fits(mean_present(n - c, w, &entries, Some(c)))
        };
        rows += (c * m) as f64 * p;
        skip_values += m as f64 * p;
    }
    Baseline {
        skip_rows: rows / pop.non_null() as f64,
        skip_values: skip_values / pop.d() as f64,
        window_distinct,
        overflow: 1.0 - fits(window_distinct),
    }
}

/// One column's result row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SetStats {
    pub common: Common,
    pub window_distinct_mean: f64,
    pub window_distinct_max: u64,
    pub overflow_fraction: f64,
    pub expected_window_distinct: f64,
    pub expected_overflow_fraction: f64,
}

pub(crate) fn column(s: &Series, window_rows: u64, max_size: u64) -> PolarsResult<SetStats> {
    let pop = population(s)?;
    let (w, k) = windows(pop.n(), window_rows);
    let sums = over_ranges(k, w, pop.d(), |r| scan(&pop, w as usize, max_size, r));
    let b = baseline(&pop, w, max_size);
    let per_window = |x: f64| x / sums.windows as f64; // NaN without a window
    Ok(SetStats {
        common: Common::new(&pop, w, &sums, (b.skip_rows, b.skip_values)),
        window_distinct_mean: per_window(sums.entries as f64),
        window_distinct_max: sums.max_entries,
        overflow_fraction: per_window(sums.overflowed as f64),
        expected_window_distinct: b.window_distinct,
        expected_overflow_fraction: b.overflow,
    })
}

/// `column` + SetIndex's metrics per column; columns in parallel (and each column's window
/// ranges).
pub(crate) fn set_index_impl(
    inputs: &[Series],
    window_rows: u64,
    max_size: u64,
) -> PolarsResult<Series> {
    let rows: Vec<SetStats> = inputs
        .par_iter()
        .map(|s| column(s, window_rows, max_size))
        .collect::<PolarsResult<_>>()?;
    let commons: Vec<Common> = rows.iter().map(|r| r.common).collect();
    let mut fields = Common::fields(&commons);
    fields.extend([
        f64s("window_distinct_mean", rows.iter().map(|r| r.window_distinct_mean)),
        u64s("window_distinct_max", rows.iter().map(|r| r.window_distinct_max)),
        f64s("overflow_fraction", rows.iter().map(|r| r.overflow_fraction)),
        f64s("expected_window_distinct", rows.iter().map(|r| r.expected_window_distinct)),
        f64s("expected_overflow_fraction", rows.iter().map(|r| r.expected_overflow_fraction)),
    ]);
    output("set_index", inputs, fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(values: &[Option<i64>]) -> Population {
        population(&Series::new("x".into(), values)).unwrap()
    }

    fn whole(p: &Population, w: usize, max_size: u64) -> WindowSums {
        scan(p, w, max_size, 0..p.ids.len() - w + 1)
    }

    /// A A A B A A A C (spec §7).
    const TOY: [Option<i64>; 8] = [
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(0),
        Some(0),
        Some(0),
        Some(2),
    ];

    #[test]
    fn toy_known_answer() {
        let p = pop(&TOY);
        // windows 0–3 hold {A, B} (1 row of 8 outside, 1 value of 3); window 4 holds {A, C}
        let s = whole(&p, 4, 0);
        assert_eq!(
            (s.windows, s.rows.sum, s.values.sum, s.entries, s.max_entries, s.overflowed),
            (5, 5, 5, 10, 2, 0)
        );
        let o = whole(&p, 4, 1);
        assert_eq!((o.rows.sum, o.values.sum, o.overflowed), (0, 0, 5));
    }

    #[test]
    fn null_counts_toward_max_size() {
        let p = pop(&[Some(1), None, Some(2), None]);
        assert_eq!(whole(&p, 2, 1).overflowed, 3);
        assert_eq!(whole(&p, 2, 2).overflowed, 0);
    }

    #[test]
    fn merged_ranges_equal_one_scan_at_every_split() {
        let values: Vec<Option<i64>> = (0..40)
            .map(|i| (i % 7 != 3).then_some((i * 13 % 9) as i64))
            .collect();
        let p = pop(&values);
        for (w, max_size) in [(1, 0), (3, 0), (8, 0), (8, 4)] {
            let k = values.len() - w + 1;
            let all = scan(&p, w, max_size, 0..k);
            for i in 1..k {
                assert_eq!(
                    scan(&p, w, max_size, 0..i).merge(scan(&p, w, max_size, i..k)),
                    all,
                    "w {w}, max_size {max_size}, split {i}"
                );
            }
        }
    }

    /// Entry counts of every w-subset of `values` (each equally likely under a random order),
    /// keeping only subsets that `keep`.
    fn enumerate(values: &[Option<i64>], w: usize, keep: impl Fn(&[Option<i64>]) -> bool) -> Vec<f64> {
        let n = values.len();
        (0u32..1 << n)
            .filter(|mask| mask.count_ones() as usize == w)
            .filter_map(|mask| {
                let mut held: Vec<Option<i64>> =
                    (0..n).filter(|i| mask >> i & 1 == 1).map(|i| values[i]).collect();
                if !keep(&held) {
                    return None;
                }
                held.sort();
                held.dedup();
                Some(held.len() as f64)
            })
            .collect()
    }

    const SMALL: [Option<i64>; 9] = [
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(1),
        Some(2),
        None,
        None,
        Some(3),
    ];

    #[test]
    fn baseline_moments_match_enumeration() {
        let (n, w) = (SMALL.len() as u64, 3u64);
        let p = pop(&SMALL);
        let sizes = enumerate(&SMALL, 3, |_| true);
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let var = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        let entries = classes(p.counts.iter().copied().chain([p.n_null]));
        assert!((mean_present(n, w, &entries, None) - mean).abs() < 1e-12);
        assert!((std_present(n, w, &entries) - var.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn conditional_mean_matches_enumeration() {
        // value 1 (2 rows) absent; NULL also has 2 rows, so `without` removes one count-2 entry
        let (n, w) = (SMALL.len() as u64, 3u64);
        let p = pop(&SMALL);
        let sizes = enumerate(&SMALL, 3, |held| !held.contains(&Some(1)));
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let entries = classes(p.counts.iter().copied().chain([p.n_null]));
        assert!((mean_present(n - 2, w, &entries, Some(2)) - mean).abs() < 1e-12);
    }

    #[test]
    fn baseline_unlimited_hand_worked() {
        // 1 2 3 4, w = 2: each value is absent from half the pairs
        let b = baseline(&pop(&[Some(1), Some(2), Some(3), Some(4)]), 2, 0);
        assert!((b.skip_rows - 0.5).abs() < 1e-12 && (b.skip_values - 0.5).abs() < 1e-12);
        assert!((b.window_distinct - 2.0).abs() < 1e-12);
        assert_eq!(b.overflow, 0.0);
    }

    #[test]
    fn baseline_with_max_size_is_a_step_when_the_count_is_fixed() {
        // every window holds exactly 2 entries: σ = 0
        let p = pop(&[Some(1), Some(2), Some(3), Some(4)]);
        let b = baseline(&p, 2, 1);
        assert_eq!((b.skip_rows, b.overflow), (0.0, 1.0));
        let b = baseline(&p, 2, 2);
        assert!((b.skip_rows - 0.5).abs() < 1e-12);
        assert_eq!(b.overflow, 0.0);
    }

    #[test]
    fn impl_returns_one_struct_row_per_column() {
        let out = set_index_impl(&[Series::new("a".into(), &[1i64, 1, 2])], 2, 0).unwrap();
        let fields = out.struct_().unwrap().fields_as_series();
        let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
        assert_eq!(names.len(), 17);
        assert_eq!(
            names[12..],
            [
                "window_distinct_mean",
                "window_distinct_max",
                "overflow_fraction",
                "expected_window_distinct",
                "expected_overflow_fraction"
            ]
        );
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib skip_index::`
Expected: every `skip_index` test passes; no dead-code warnings remain in `skip_index/`.

- [ ] **Step 3: Commit**

```bash
git add services/analytics/src/techniques/single_column/ordered/skip_index/set.rs
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Skip index: set kernel (per-id window counts) and baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Bindings — api, pyo3, `_plugin`

**Files:**
- Modify: `services/analytics/src/bindings/api/techniques.rs` (after `runs_test`, ~line 23)
- Modify: `services/analytics/src/bindings/python.rs` (after `fn runs_test`, ~line 287; registration ~line 500)
- Modify: `services/analytics/analytics/_plugin.py` (after `def runs_test`, ~line 137)

- [ ] **Step 1: api entry points**

In `bindings/api/techniques.rs`, after `pub fn runs_test`:

```rust
/// `window_rows ≥ 1`, else InvalidInput.
fn window(window_rows: u64) -> Result<u64> {
    if window_rows == 0 {
        return Err(Error::InvalidInput("window_rows must be >= 1".into()));
    }
    Ok(window_rows)
}

/// `column` + MinMaxIndex's metrics per column: windows of `window_rows` rows sliding over the
/// batch's row order.
pub fn minmax_index(batch: &RecordBatch, window_rows: u64) -> Result<RecordBatch> {
    let w = window(window_rows)?;
    table(crate::techniques::skip_index::minmax::minmax_index_impl(
        &columns(batch)?,
        w,
    ))
}

/// `column` + SetIndex's metrics per column; `max_size` 0 = unlimited.
pub fn set_index(batch: &RecordBatch, window_rows: u64, max_size: u64) -> Result<RecordBatch> {
    let w = window(window_rows)?;
    table(crate::techniques::skip_index::set::set_index_impl(
        &columns(batch)?,
        w,
        max_size,
    ))
}
```

- [ ] **Step 2: pyo3 functions**

In `bindings/python.rs`, after `fn runs_test`:

```rust
#[pyfunction]
#[pyo3(signature = (data, window_rows))]
fn minmax_index(py: Python<'_>, data: &Bound<'_, PyAny>, window_rows: u64) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::minmax_index(&batch, window_rows)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, window_rows, max_size))]
fn set_index(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    window_rows: u64,
    max_size: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::set_index(&batch, window_rows, max_size)).map(ArrowTable)
}
```

and register them after `runs_test`:

```rust
    m.add_function(wrap_pyfunction!(minmax_index, m)?)?;
    m.add_function(wrap_pyfunction!(set_index, m)?)?;
```

- [ ] **Step 3: `_plugin` wrappers**

In `analytics/_plugin.py`, after `def runs_test`:

```python
def minmax_index(df: Frame, window_rows: int) -> pl.DataFrame:
    """column + MinMaxIndex's metrics per column — windows of `window_rows` rows sliding
    over the frame's row order."""
    return pl.DataFrame(_rs.minmax_index(_collected(df), window_rows))


def set_index(df: Frame, window_rows: int, max_size: int) -> pl.DataFrame:
    """column + SetIndex's metrics per column — windows of `window_rows` rows sliding over
    the frame's row order; `max_size` 0 = unlimited."""
    return pl.DataFrame(_rs.set_index(_collected(df), window_rows, max_size))
```

- [ ] **Step 4: Build and smoke-test**

Run (from `services/analytics/`): `$PY -m maturin develop --release`
Then (from the repo root):

```bash
$PY -c "
import polars as pl
from analytics import _plugin
df = pl.DataFrame({'x': [0, 0, 0, 1, 0, 0, 0, 2]})
print(_plugin.minmax_index(df, 4))
print(_plugin.set_index(df, 4, 0))
"
```

Expected: one row each; minmax `skip_rows_mean` 0.1, set `skip_rows_mean` 0.125, `window_distinct_mean` 2.0.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/bindings services/analytics/analytics/_plugin.py
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Skip index: api, pyo3 and _plugin entry points

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Shared Python base, test helpers, and the minmax reference

**Files:**
- Modify: `services/analytics/analytics/_dtypes.py` (after `WIDE_INTEGERS` / `holds_wide_integer`)
- Modify: `services/analytics/analytics/runs_test/base.py:1-24`
- Create: `services/analytics/analytics/_skip_index.py`
- Create: `services/analytics/analytics/minmax_index/{__init__,base,polars,rust}.py`
- Create: `tests/skip_index_helpers.py`
- Create: `tests/test_minmax_index.py`

- [ ] **Step 1: `ORDERED_SCALARS`**

In `analytics/_dtypes.py`, after `holds_wide_integer`:

```python
# Scalar dtypes with a value order and identity: the order-dependent per-column techniques'
# eligibility (runs_test, minmax_index, set_index). Mirrors the Rust kernels' `supported`
# (src/techniques/single_column/ordered/runs_test.rs).
ORDERED_SCALARS = (
    *INTEGERS_64,
    pl.Boolean,
    pl.Float32,
    pl.Float64,
    pl.Decimal,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    pl.String,
    pl.Binary,
    pl.Categorical,
    pl.Enum,
)
```

In `analytics/runs_test/base.py`, replace the import of `INTEGERS_64` and the whole `ELIGIBLE = (...)` tuple with:

```python
from analytics._dtypes import ORDERED_SCALARS
from analytics.base import Technique

# Mirrors the Rust kernel's `supported` (src/techniques/single_column/ordered/runs_test.rs).
ELIGIBLE = ORDERED_SCALARS
```

Run `$PY -m pytest tests/test_runs_test.py -n 0 -q` — still passes.

- [ ] **Step 2: `analytics/_skip_index.py`**

```python
"""Shared base of the skip-index techniques — analytics.minmax_index and analytics.set_index
(spec docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md) — and the arithmetic
their Polars references share (they share none with the Rust kernels)."""

import math

import numpy as np
import polars as pl

from analytics._dtypes import ORDERED_SCALARS
from analytics.base import Technique, metric_mismatches

COMMON_METRICS = {
    "n_rows": pl.UInt64,
    "n_non_null": pl.UInt64,
    "n_distinct": pl.UInt64,
    "window_rows": pl.UInt64,
    "n_windows": pl.UInt64,
    "skip_rows_mean": pl.Float64,
    "skip_rows_std": pl.Float64,
    "skip_values_mean": pl.Float64,
    "skip_values_std": pl.Float64,
    "expected_skip_rows": pl.Float64,
    "expected_skip_values": pl.Float64,
}


def positive(name: str, value) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


class SkipIndex(Technique):
    """How well a column, in its row order, suits a ClickHouse data-skipping index.

    A window of w = index_granularity × granularity rows — one index block — slides one row
    at a time over the column (a column shorter than w is one window), so every granule
    alignment counts once. For a point query `col = v`, a window's skip is the share of
    queries it lets ClickHouse skip, v drawn from the non-null values either by row
    frequency (`skip_rows_*`: frequent values are queried more) or uniformly over the
    distinct values (`skip_values_*`: rare-value search). `*_mean` / `*_std` are the mean
    and population std over windows: the expected skip under a random alignment, and its
    spread.

    `expected_skip_rows` / `expected_skip_values` are the same expectations had the rows
    arrived in a uniformly random order (closed form, from the value counts): observed minus
    expected is the gain due to ordering; expected itself is the gain due to rarity, which
    any order keeps.

    Metrics only — no conclusions: choosing an index, max_size or GRANULARITY is the
    recommenders' decision. Eligible: integers (≤ 64-bit), Boolean, Float, Decimal,
    temporal, String, Binary, Categorical and Enum columns with at least one row; a column
    without non-null values has NaN skips (no query value exists). `dtype` (Python's
    str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = COMMON_METRICS
    RTOL = 1e-9
    ATOL = 1e-12
    # The baselines sum lgamma differences (~10⁸ at 10⁷ rows) from two libms.
    BASELINE_RTOL = 1e-6

    def __init__(self, *, index_granularity: int = 8192, granularity: int = 1):
        super().__init__()
        positive("index_granularity", index_granularity)
        positive("granularity", granularity)
        self.index_granularity = index_granularity
        self.granularity = granularity

    @property
    def window_rows(self) -> int:
        return self.index_granularity * self.granularity

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, ORDERED_SCALARS) and series.len() > 0

    def describe(self, frames, combos):
        dtypes = {
            (n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()
        }
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def agreement(self, result, reference):
        keys = self.key_columns()
        baselines = [m for m in self.METRICS if m.startswith("expected_")]
        observed = [m for m in self.METRICS if m not in baselines]
        problems = metric_mismatches(
            result, reference, keys, observed, self.RTOL, self.ATOL
        ) + metric_mismatches(
            result, reference, keys, baselines, self.BASELINE_RTOL, self.ATOL
        )
        return list(dict.fromkeys(problems))  # both calls report status mismatches


# ── the references' arithmetic ─────────────────────────────────────────────────


def ranked(s: pl.Series) -> tuple[pl.Series, np.ndarray]:
    """(ids, counts): each row's rank among the distinct non-null values (UInt32; null for a
    null row) and the rows per rank. Order as the kernels: Categorical by its string, Enum by
    category order, floats canonical (−0.0 = 0.0, one NaN, sorted last)."""
    if s.dtype == pl.Categorical:
        s = s.cast(pl.String)
    elif isinstance(s.dtype, pl.Enum):
        s = s.to_physical()
    elif s.dtype in (pl.Float32, pl.Float64):
        s = s.set((s == 0).fill_null(False), 0.0).fill_nan(float("nan"))
    distinct = s.drop_nulls().unique().sort()
    ids = pl.DataFrame({"v": s}).join(
        distinct.to_frame("v").with_row_index("id"),
        on="v",
        how="left",
        maintain_order="left",
    )["id"]
    counts = np.bincount(ids.drop_nulls().to_numpy(), minlength=distinct.len())
    return ids, counts.astype(np.int64)


def windows(n: int, window_rows: int) -> tuple[int, int]:
    """(w, number of windows) for a non-empty column."""
    w = min(n, window_rows)
    return w, n - w + 1


def mean_std(a: np.ndarray, den: int) -> tuple[float, float]:
    """Mean (exact, Python ints) and population std (two-pass) of a / den; NaN without a
    denominator."""
    if den == 0 or a.size == 0:
        return math.nan, math.nan
    return int(a.sum()) / (a.size * den), float(np.std(a / den))


def avoid(n: int, w: int, m: int) -> float:
    """P(a w-row window of a random order of n rows holds none of m given rows)."""
    if m == 0:
        return 1.0
    if n - m < w:
        return 0.0
    lg = math.lgamma
    return math.exp(lg(n - m + 1) - lg(n - m - w + 1) - lg(n + 1) + lg(n - w + 1))


def phi(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2))


def common_metrics(
    n: int, counts: np.ndarray, w: int, rows: np.ndarray, values: np.ndarray, expected
) -> dict:
    """COMMON_METRICS from the per-window numerators and (expected rows, expected values)."""
    nn, d = int(counts.sum()), int(counts.size)
    (rm, rs), (vm, vs) = mean_std(rows, nn), mean_std(values, d)
    return {
        "n_rows": n,
        "n_non_null": nn,
        "n_distinct": d,
        "window_rows": w,
        "n_windows": n - w + 1,
        "skip_rows_mean": rm,
        "skip_rows_std": rs,
        "skip_values_mean": vm,
        "skip_values_std": vs,
        "expected_skip_rows": expected[0],
        "expected_skip_values": expected[1],
    }
```

- [ ] **Step 3: the minmax package (base + reference)**

`analytics/minmax_index/base.py`:

```python
"""ClickHouse minmax skip-index suitability per column (spec
docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md)."""

from analytics._skip_index import SkipIndex


class MinMaxIndex(SkipIndex):
    """Windows and metrics as SkipIndex. A window is skipped for `col = v` iff v < min or
    v > max over its non-null values; an all-null window is skipped for every v.

    Values compare as ClickHouse compares them: numbers and temporals by value, Boolean
    false < true, String and Binary bytewise, Categorical by its string, Enum by category
    order (ClickHouse Enum compares its numeric values); floats are canonical (−0.0 = 0.0,
    every NaN equal, after every number). The random-order baseline is exact.
    """
```

`analytics/minmax_index/polars.py`:

```python
import numpy as np
import polars as pl

from analytics._skip_index import avoid, common_metrics, ranked, windows
from analytics.minmax_index.base import MinMaxIndex


def minmax_baseline(counts: np.ndarray, n: int, w: int) -> tuple[float, float]:
    """(expected_skip_rows, expected_skip_values) under a random row order."""
    nn, d = int(counts.sum()), int(counts.size)
    if d == 0:
        return float("nan"), float("nan")
    all_null = avoid(n, w, nn)
    rows = values = 0.0
    below = 0  # non-null rows below v
    for c in counts.tolist():
        p = avoid(n, w, below + c) + avoid(n, w, nn - below) - all_null
        rows += c * p
        values += p
        below += c
    return rows / nn, values / d


def minmax_statistics(s: pl.Series, window_rows: int) -> dict:
    """The MinMaxIndex metrics of one non-empty column."""
    ids, counts = ranked(s)
    n, d, nn = s.len(), int(counts.size), int(counts.sum())
    w, k = windows(n, window_rows)
    lo = ids.rolling_min(window_size=w, min_samples=1).slice(w - 1)
    hi = ids.rolling_max(window_size=w, min_samples=1).slice(w - 1)
    empty = lo.is_null().to_numpy()  # an all-null window
    lo = lo.fill_null(0).to_numpy().astype(np.int64)
    hi = hi.fill_null(0).to_numpy().astype(np.int64)
    if d:
        cum = np.concatenate([[0], np.cumsum(counts)])
        rows = np.where(empty, nn, nn - (cum[hi + 1] - cum[lo]))
        values = np.where(empty, d, d - (hi - lo + 1))
    else:
        rows = values = np.zeros(k, dtype=np.int64)
    return common_metrics(n, counts, w, rows, values, minmax_baseline(counts, n, w))


class MinMaxIndexPolars(MinMaxIndex):
    """Reference: rank ids from a join against the sorted distinct values, each window's
    extremes from Polars' rolling_min / rolling_max (nulls ignored), rows covered from
    cumulative counts; the baseline in Python floats (math.lgamma). Shares no code with the
    Rust kernel."""

    def _compute(self, frames, combos):
        rows = [minmax_statistics(frames[f][c], self.window_rows) for ((f, c),) in combos]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
```

`analytics/minmax_index/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.minmax_index.base import MinMaxIndex


class MinMaxIndexRust(MinMaxIndex):
    """Rust extension `minmax_index`: dense rank ids, then two monotonic deques give each
    window's extremes in O(1) amortised per row; skips summed exactly as integers;
    rayon-parallel across columns and ranges of window starts."""

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.minmax_index(
                        frames[frame].select([c for ((_, c),) in group]),
                        self.window_rows,
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
```

`analytics/minmax_index/__init__.py`:

```python
"""ClickHouse minmax skip-index suitability (per-column, order-dependent). See MinMaxIndex."""

from analytics.minmax_index.base import MinMaxIndex
from analytics.minmax_index.polars import MinMaxIndexPolars
from analytics.minmax_index.rust import MinMaxIndexRust

REFERENCE = "MinMaxIndexPolars"
IMPLEMENTATIONS = ("MinMaxIndexRust", "MinMaxIndexPolars")

__all__ = [
    "MinMaxIndex",
    "MinMaxIndexRust",
    "MinMaxIndexPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
```

- [ ] **Step 4: `tests/skip_index_helpers.py`**

```python
"""Shared by tests/test_minmax_index.py and tests/test_set_index.py: dtypes to cover, a
brute-force oracle evaluating every window directly from the spec's definitions
(docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md §2.3), and a Monte Carlo of
the random-order baselines (under a random order every window is a uniform w-subset of the
rows)."""

import math
from collections import Counter

import numpy as np
import polars as pl
import pytest

ELIGIBLE_DTYPES = [
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
    pl.Decimal(10, 2),
    pl.Date(),
    pl.Datetime("us"),
    pl.Datetime("ns", "UTC"),
    pl.Duration("ms"),
    pl.Time(),
    pl.String(),
    pl.Binary(),
    pl.Categorical(),
    pl.Enum(["0", "1", "2"]),
]


def as_dtype(values: list[int], dtype: pl.DataType) -> pl.Series:
    """Single-digit ints keep their order under every cast (strings: "0" < "1" < "2")."""
    s = pl.Series("x", values)
    if dtype in (pl.String, pl.Binary, pl.Categorical) or isinstance(dtype, pl.Enum):
        return s.cast(pl.String).cast(dtype)
    return s.cast(dtype)


def zipf_column(seed: int, k: int, n: int, alpha: float, null_rate: float) -> list:
    rng = np.random.default_rng(seed)
    p = np.arange(1, k + 1) ** -alpha
    p /= p.sum()
    return [
        None if rng.random() < null_rate else int(v) for v in rng.choice(k, size=n, p=p)
    ]


def _mean_std(a: list[int], den: int) -> tuple[float, float]:
    if not den:
        return math.nan, math.nan
    x = np.array(a, dtype=float) / den
    return float(x.mean()), float(x.std())


def oracle(values: list, window_rows: int, max_size: int = 0) -> tuple[dict, dict]:
    """(minmax, set) observed metrics of `values` (comparable Python values, None = null),
    every window evaluated directly."""
    n = len(values)
    w = min(n, window_rows)
    k = n - w + 1
    counts = Counter(v for v in values if v is not None)
    dist = sorted(counts)
    nn, d = sum(counts.values()), len(dist)
    mm_rows, mm_values, set_rows, set_values, entries = [], [], [], [], []
    for s in range(k):
        win = values[s : s + w]
        held = {v for v in win if v is not None}
        if held:
            lo, hi = min(held), max(held)
            outside = [v for v in dist if v < lo or v > hi]
            mm_rows.append(sum(counts[v] for v in outside))
            mm_values.append(len(outside))
        else:
            mm_rows.append(nn)
            mm_values.append(d)
        e = len(held) + (None in win)
        entries.append(e)
        if max_size and e > max_size:
            set_rows.append(0)
            set_values.append(0)
        else:
            set_rows.append(nn - sum(counts[v] for v in held))
            set_values.append(d - len(held))
    common = dict(n_rows=n, n_non_null=nn, n_distinct=d, window_rows=w, n_windows=k)
    mm, st = dict(common), dict(common)
    for out, rows, vals in ((mm, mm_rows, mm_values), (st, set_rows, set_values)):
        out["skip_rows_mean"], out["skip_rows_std"] = _mean_std(rows, nn)
        out["skip_values_mean"], out["skip_values_std"] = _mean_std(vals, d)
    st.update(
        window_distinct_mean=float(np.mean(entries)),
        window_distinct_max=max(entries),
        overflow_fraction=float(np.mean([max_size > 0 and e > max_size for e in entries])),
    )
    return mm, st


def sampled(
    values: list, w: int, max_size: int = 0, samples: int = 20_000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """(mean, standard error) of each skip over uniformly random w-row subsets: keys
    mm_rows, mm_values, set_rows, set_values, entries, overflow."""
    rng = np.random.default_rng(seed)
    n = len(values)
    counts = Counter(v for v in values if v is not None)
    dist = sorted(counts)
    nn, d = sum(counts.values()), len(dist)
    rank = {v: i for i, v in enumerate(dist)}
    codes = np.array([-1 if v is None else rank[v] for v in values])
    cnt = np.array([counts[v] for v in dist])
    cum = np.concatenate([[0], np.cumsum(cnt)])
    keys = ("mm_rows", "mm_values", "set_rows", "set_values", "entries", "overflow")
    acc = {key: np.empty(samples) for key in keys}
    for i in range(samples):
        win = codes[rng.choice(n, w, replace=False)]
        held = np.unique(win[win >= 0])
        if held.size:
            lo, hi = held[0], held[-1]
            acc["mm_rows"][i] = (nn - (cum[hi + 1] - cum[lo])) / nn
            acc["mm_values"][i] = (d - (hi - lo + 1)) / d
        else:
            acc["mm_rows"][i] = acc["mm_values"][i] = 1.0
        e = held.size + bool((win < 0).any())
        over = max_size > 0 and e > max_size
        acc["entries"][i] = e
        acc["overflow"][i] = over
        acc["set_rows"][i] = 0 if over else (nn - cnt[held].sum()) / nn
        acc["set_values"][i] = 0 if over else (d - held.size) / d
    return {key: (a.mean(), a.std() / math.sqrt(samples)) for key, a in acc.items()}


def assert_metrics(got: dict, want: dict, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    for m, v in want.items():
        g = got[m]
        if isinstance(v, float) and math.isnan(v):
            assert isinstance(g, float) and math.isnan(g), f"{m}: {g!r} != nan"
        else:
            assert g == pytest.approx(v, rel=rtol, abs=atol), f"{m}: {g!r} != {v!r}"
```

- [ ] **Step 5: Write the minmax tests (reference-only blocks first)**

`tests/test_minmax_index.py`:

```python
"""
MinMax skip-index accuracy tests — every implementation in analytics.minmax_index.

Oracles: hand-worked known answers, a brute-force window-by-window oracle
(skip_index_helpers.oracle), MinMaxIndexPolars (the reference) and a Monte Carlo of the
random-order baseline. Semantics: spec 2026-10-16-skip-index-techniques §2.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datagen import mixed_dtypes, runs_int, runs_str
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)
from skip_index_helpers import (
    ELIGIBLE_DTYPES,
    as_dtype,
    assert_metrics,
    oracle,
    sampled,
    zipf_column,
)

PKG = "analytics.minmax_index"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"

TOY = [0, 0, 0, 1, 0, 0, 0, 2]  # A A A B A A A C (spec §7)


def stats(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def minmax_oracle(values: list, window: int) -> dict:
    return oracle(values, window)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "runs": runs_int(1_000, 2, 10, 50),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames, index_granularity=16), frames)


@pytest.mark.parametrize("impl", ALL)
def test_ineligible_columns_are_listed(impl):
    df = pl.DataFrame(
        {
            "list": [[1], [2]],
            "arr": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
            "struct": [{"a": 1}, {"a": 2}],
            "null": pl.Series([None, None], dtype=pl.Null),
            "i128": pl.Series([1, 2], dtype=pl.Int128),
            "ok": [1, 1],
        }
    )
    out = run(load(impl), {"t": df, "empty": pl.DataFrame(schema={"e": pl.Int64})})
    assert out["status"].to_list() == ["ineligible"] * 5 + ["computed", "ineligible"]
    assert out["dtype"].to_list() == [str(dt) for dt in df.dtypes] + ["Int64"]
    assert out["n_windows"].to_list() == [None] * 5 + [1, None]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ELIGIBLE_DTYPES, ids=str)
def test_known_answer_every_dtype(impl, dtype):
    r = stats(impl, as_dtype(TOY, dtype), index_granularity=4)
    counts = tuple(
        r[m] for m in ("n_rows", "n_non_null", "n_distinct", "window_rows", "n_windows")
    )
    assert counts == (8, 8, 3, 4, 5)
    assert r["skip_rows_mean"] == pytest.approx(0.1, rel=1e-12)
    assert r["skip_rows_std"] == pytest.approx(0.05, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(4 / 15, rel=1e-12)
    assert r["skip_values_std"] == pytest.approx(2 / 15, rel=1e-12)


def test_known_answers_cover_every_eligible_class():
    """Every class in ORDERED_SCALARS (mirrored by the Rust kernels' `supported`) runs
    through both implementations in test_known_answer_every_dtype (Boolean: test_boolean)."""
    from analytics._dtypes import ORDERED_SCALARS

    assert {type(dt) for dt in ELIGIBLE_DTYPES} | {pl.Boolean} == set(ORDERED_SCALARS)


@pytest.mark.parametrize("impl", ALL)
def test_boolean(impl):
    # windows {F}, {F, T}, {T}: each one-valued window skips the other half
    r = stats(impl, pl.Series("x", [False, False, True, True]), index_granularity=2)
    assert r["skip_rows_mean"] == pytest.approx(1 / 3, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(1 / 3, rel=1e-12)


@pytest.mark.parametrize("impl", ALL)
def test_sorted_column(impl):
    r = stats(impl, pl.Series("x", list(range(1, 9))), index_granularity=4)
    assert (r["skip_rows_mean"], r["skip_values_mean"]) == (0.5, 0.5)
    assert r["skip_rows_std"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("impl", ALL)
def test_all_null_window_skips_everything(impl):
    values = [1, None, None, 2]  # windows {1}, {}, {2}
    r = stats(impl, pl.Series("x", values), index_granularity=2)
    assert r["skip_rows_mean"] == pytest.approx(2 / 3, rel=1e-12)
    assert_metrics(r, minmax_oracle(values, 2))


@pytest.mark.parametrize("impl", ALL)
def test_all_null_column_is_nan(impl):
    r = stats(impl, pl.Series("x", [None] * 3, dtype=pl.Int64), index_granularity=2)
    assert (r["n_non_null"], r["n_distinct"], r["n_windows"]) == (0, 0, 2)
    for m in ("skip_rows_mean", "skip_values_std", "expected_skip_rows"):
        assert math.isnan(r[m]), m


@pytest.mark.parametrize("impl", ALL)
def test_column_shorter_than_window_is_one_window(impl):
    r = stats(impl, pl.Series("x", [3, 1, 2]), index_granularity=8)
    assert (r["window_rows"], r["n_windows"]) == (3, 1)
    assert (r["skip_rows_mean"], r["expected_skip_rows"]) == (0.0, 0.0)


@pytest.mark.parametrize("impl", ALL)
def test_window_is_index_granularity_times_granularity(impl):
    r = stats(impl, pl.Series("x", list(range(20))), index_granularity=2, granularity=3)
    assert (r["window_rows"], r["n_windows"]) == (6, 15)


@pytest.mark.parametrize("impl", ALL)
def test_categorical_by_string_enum_by_category_order(impl):
    values = ["b", "b", "c", "a", "a", "c", "b", None]
    s = pl.Series("x", values)
    cat = stats(impl, s.cast(pl.Categorical), index_granularity=2)
    assert_metrics(cat, minmax_oracle(values, 2))  # strings compare lexically
    enum = stats(impl, s.cast(pl.Enum(["c", "b", "a"])), index_granularity=2)
    rank = {"c": 0, "b": 1, "a": 2, None: None}
    assert_metrics(enum, minmax_oracle([rank[v] for v in values], 2))


@pytest.mark.parametrize("impl", ALL)
def test_floats_canonical_nan_last(impl):
    other_nan = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
    s = pl.Series("x", [0.0, -0.0, float("nan"), other_nan, 1.0, -1.0])
    for series in (s, s.cast(pl.Float32)):
        r = stats(impl, series, index_granularity=2)
        assert r["n_distinct"] == 4
        # ranks: −1 → 0, ±0 → 1, 1 → 2, NaN → 3
        assert_metrics(r, minmax_oracle([1, 1, 3, 3, 2, 0], 2))


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_equals_one_chunk(impl):
    s = runs_int(5_000, 1, 7, 40)["c0"]
    parts = pl.concat([s.slice(0, 1_234), s.slice(1_234)], rechunk=False)
    assert parts.n_chunks() == 2
    assert stats(impl, parts, index_granularity=64) == stats(
        impl, s, index_granularity=64
    )


@pytest.mark.parametrize("impl", ALL)
def test_baseline_hand_worked(impl):
    # 1 2 3 4, w = 2: 6 equally likely pairs, 8 of 24 (pair, query) combinations skipped
    r = stats(impl, pl.Series("x", [1, 2, 3, 4]), index_granularity=2)
    assert r["expected_skip_rows"] == pytest.approx(1 / 3, rel=1e-9)
    assert r["expected_skip_values"] == pytest.approx(1 / 3, rel=1e-9)
    assert r["skip_rows_mean"] == 0.5  # sorted beats a random order


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_granularity": 0},
        {"granularity": 0},
        {"index_granularity": 1.5},
        {"granularity": True},
    ],
)
def test_keywords_are_validated(kwargs):
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        reference(PKG)(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical correctness (no conclusions to test: the technique reports metrics only)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize(
    "case",
    [(1, 30, 400, 20, 1.0, 0.0), (2, 50, 1_000, 30, 1.2, 0.05), (3, 20, 300, 15, 1.5, 0.1)],
)
def test_baseline_matches_monte_carlo(impl, case):
    seed, k, n, w, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    r = stats(impl, pl.Series("x", values, dtype=pl.Int64), index_granularity=w)
    mc = sampled(values, w, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "mm_rows"),
        ("expected_skip_values", "mm_values"),
    ):
        mean, se = mc[key]
        assert abs(r[metric] - mean) <= 5 * se + 1e-9, (metric, r[metric], mean, se)
```

- [ ] **Step 6: Run the tests**

Run: `$PY -m pytest tests/test_minmax_index.py -n 0 -q`
Expected: all pass for both `MinMaxIndexPolars` and `MinMaxIndexRust` (the extension was built in Task 4). If a Rust case fails here, debug it against the reference before Task 6.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/analytics/_dtypes.py services/analytics/analytics/runs_test/base.py services/analytics/analytics/_skip_index.py services/analytics/analytics/minmax_index tests/skip_index_helpers.py tests/test_minmax_index.py
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "MinMax index: technique package, Polars reference, known-answer tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: MinMax reference agreement and oracle fuzz

**Files:**
- Modify: `tests/test_minmax_index.py` (insert a "2. Reference agreement" block between blocks 1 and 3)

- [ ] **Step 1: Add the agreement tests**

Insert after `test_ineligible_columns_are_listed`:

```python
# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("params", [{}, {"index_granularity": 1_024, "granularity": 2}])
def test_agrees_with_reference_large_dataset(impl, params):
    frames = {"large": pl.read_ipc(LARGE)}
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_seeded(impl):
    # 200 000 rows: more window starts than one parallel range (2^16)
    frames = {
        "mixed": mixed_dtypes(2_000),
        "ints": runs_int(200_000, 3, 25, 100),
        "strs": runs_str(200_000, 2, 5, 1_000),
    }
    cls = load(impl)
    params = {"index_granularity": 512}
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(30))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype = ELIGIBLE_DTYPES[seed % len(ELIGIBLE_DTYPES)]
    n = rng.choice([1, 2, 17, 1_000, 70_000])
    k = rng.choice([1, 2, 3, 10])
    if isinstance(dtype, pl.Enum):
        dtype = pl.Enum([str(i) for i in range(k)])
    null_rate = rng.choice([0.0, 0.1, 0.5])
    values, cur = [], 0
    for _ in range(n):  # sticky random walk → clustered values
        if rng.random() < 0.3:
            cur = rng.randrange(k)
        values.append(None if rng.random() < null_rate else cur)
    s = as_dtype(values, dtype)
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([s.slice(0, cut), s.slice(cut)], rechunk=False)
    frames = {"t": s.to_frame()}
    params = {"index_granularity": rng.choice([1, 3, 64, 5_000])}
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_oracle(impl, seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    k = rng.choice([1, 2, 3, 5, 10])
    null_rate = rng.choice([0.0, 0.1, 0.5, 1.0])
    values = [None if rng.random() < null_rate else rng.randrange(k) for _ in range(n)]
    window = rng.randint(1, 12)
    got = stats(impl, pl.Series("x", values, dtype=pl.Int64), index_granularity=window)
    assert_metrics(got, minmax_oracle(values, window))


def test_kernel_rejects_bad_input():
    from analytics import _plugin

    with pytest.raises(ValueError, match="unsupported dtype"):
        _plugin.minmax_index(pl.DataFrame({"x": [[1], [2]]}), 4)
    with pytest.raises(ValueError, match="window_rows"):
        _plugin.minmax_index(pl.DataFrame({"x": [1, 2]}), 0)
```

- [ ] **Step 2: Run the tests**

Run: `$PY -m pytest tests/test_minmax_index.py -n 0 -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_minmax_index.py
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "MinMax index: reference agreement and brute-force oracle fuzz

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Set package, reference and tests

**Files:**
- Create: `services/analytics/analytics/set_index/{__init__,base,polars,rust}.py`
- Create: `tests/test_set_index.py`

- [ ] **Step 1: The set package**

`analytics/set_index/base.py`:

```python
"""ClickHouse set(max_size) skip-index suitability per column (spec
docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md)."""

import polars as pl

from analytics._skip_index import COMMON_METRICS, SkipIndex


class SetIndex(SkipIndex):
    """Windows and metrics as SkipIndex. A window stores the set of its entries — its
    distinct non-null values, plus NULL when it holds a null (ClickHouse stores NULL for
    Nullable columns). A set of more than `max_size` entries is stored empty and skips
    nothing (0 = unlimited); otherwise the window is skipped for `col = v` iff v is not in
    it. NULL counts toward max_size only: it never helps a `col = v` query.

    Also reported: `window_distinct_mean` / `window_distinct_max` (entries per window) and
    `overflow_fraction` (windows whose set overflowed), with their random-order
    expectations `expected_window_distinct` (exact) and `expected_overflow_fraction`. The
    skip baseline is exact when max_size = 0; otherwise the entry count is approximated as
    normal with its exact mean, variance and conditional means (within ~0.01 of Monte Carlo).
    """

    METRICS = {
        **COMMON_METRICS,
        "window_distinct_mean": pl.Float64,
        "window_distinct_max": pl.UInt64,
        "overflow_fraction": pl.Float64,
        "expected_window_distinct": pl.Float64,
        "expected_overflow_fraction": pl.Float64,
    }

    def __init__(
        self, *, index_granularity: int = 8192, granularity: int = 1, max_size: int = 100
    ):
        super().__init__(index_granularity=index_granularity, granularity=granularity)
        if type(max_size) is not int or max_size < 0:
            raise ValueError(f"max_size must be an integer >= 0, got {max_size!r}")
        self.max_size = max_size
```

`analytics/set_index/polars.py`:

```python
import math
from collections import Counter

import numpy as np
import polars as pl

from analytics._skip_index import avoid, common_metrics, phi, ranked, windows
from analytics.set_index.base import SetIndex


def intervals(key: pl.Series, w: int, k: int) -> pl.DataFrame:
    """Per id, the merged ranges [lo, hi] of window starts whose window holds it: an
    occurrence at row r is in the windows starting at max(0, r − w + 1) … min(r, k − 1)."""
    return (
        pl.DataFrame({"id": key, "r": pl.int_range(key.len(), eager=True)})
        .with_columns(
            lo=(pl.col("r") - (w - 1)).clip(lower_bound=0),
            hi=pl.col("r").clip(upper_bound=k - 1),
        )
        .sort("id", "r")
        .with_columns(
            new=(pl.col("lo") > pl.col("hi").shift(1).over("id") + 1).fill_null(True)
        )
        .group_by(pl.col("new").cum_sum().alias("interval"))
        .agg(pl.col("id").first(), pl.col("lo").min(), pl.col("hi").max())
    )


def per_window(iv: pl.DataFrame, weight: np.ndarray, k: int) -> np.ndarray:
    """Σ weight over the intervals covering each window start (difference array)."""
    diff = np.zeros(k + 1, dtype=np.int64)
    np.add.at(diff, iv["lo"].to_numpy(), weight)
    np.add.at(diff, iv["hi"].to_numpy() + 1, -weight)
    return np.cumsum(diff[:k])


def set_baseline(
    counts: np.ndarray, n_null: int, n: int, w: int, max_size: int
) -> tuple[float, float, float, float]:
    """(expected_skip_rows, expected_skip_values, expected_window_distinct,
    expected_overflow_fraction) under a random row order (spec §2.5), over count classes."""
    values = Counter(counts.tolist())
    entries = Counter(counts.tolist() + ([n_null] if n_null else []))

    def mean_present(rows: int, without: int | None = None) -> float:
        return sum(
            (m - (c == without)) * (1 - avoid(rows, w, c))
            for c, m in entries.items()
            if m - (c == without)
        )

    mu = mean_present(n)
    sigma = 0.0
    if max_size:
        q = {c: avoid(n, w, c) for c in entries}
        var = sum(m * q[c] * (1 - q[c]) for c, m in entries.items())
        for c, m in entries.items():
            for c2, m2 in entries.items():
                pairs = m * (m2 - (c == c2))
                if pairs:
                    var += pairs * (avoid(n, w, c + c2) - q[c] * q[c2])
        sigma = math.sqrt(max(var, 0.0))

    def fits(mean: float) -> float:
        if not max_size:
            return 1.0
        x = max_size + 0.5 - mean
        return phi(x / sigma) if sigma > 0 else float(x >= 0)

    rows = vals = 0.0
    for c, m in values.items():
        absent = avoid(n, w, c)
        p = absent if not max_size or absent == 0 else absent * fits(mean_present(n - c, c))
        rows += c * m * p
        vals += m * p
    nn, d = int(counts.sum()), int(counts.size)
    return (
        rows / nn if nn else math.nan,
        vals / d if d else math.nan,
        mu,
        1 - fits(mu),
    )


def set_statistics(s: pl.Series, window_rows: int, max_size: int) -> dict:
    """The SetIndex metrics of one non-empty column."""
    ids, counts = ranked(s)
    n, d, nn = s.len(), int(counts.size), int(counts.sum())
    w, k = windows(n, window_rows)
    iv = intervals(ids.fill_null(d), w, k)  # NULL is id d
    which = iv["id"].to_numpy().astype(np.int64)
    value = which < d
    present = per_window(iv, value.astype(np.int64), k)
    mass = per_window(iv, np.append(counts, 0)[which], k)
    nulls = per_window(iv, (~value).astype(np.int64), k)
    entries = present + (nulls > 0)
    over = entries > max_size if max_size else np.zeros(k, dtype=bool)
    rows = np.where(over, 0, nn - mass)
    values = np.where(over, 0, d - present)
    er, ev, mu, eo = set_baseline(counts, n - nn, n, w, max_size)
    return {
        **common_metrics(n, counts, w, rows, values, (er, ev)),
        "window_distinct_mean": float(entries.mean()),
        "window_distinct_max": int(entries.max()),
        "overflow_fraction": float(over.mean()),
        "expected_window_distinct": mu,
        "expected_overflow_fraction": eo,
    }


class SetIndexPolars(SetIndex):
    """Reference: each value's occurrences become merged intervals of window starts (Polars),
    summed per window with a difference array (NumPy) — no sliding window; the baseline in
    Python floats over count classes. Shares no code with the Rust kernel."""

    def _compute(self, frames, combos):
        rows = [
            set_statistics(frames[f][c], self.window_rows, self.max_size)
            for ((f, c),) in combos
        ]
        return self.metrics_frame(
            combos, {m: [r[m] for r in rows] for m in self.METRICS}
        )
```

`analytics/set_index/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.set_index.base import SetIndex


class SetIndexRust(SetIndex):
    """Rust extension `set_index`: dense ids, then a sliding window keeps per-id row counts,
    its distinct values and their population rows, O(1) per row entering and leaving; skips
    summed exactly as integers; rayon-parallel across columns and ranges of window starts."""

    def _compute(self, frames, combos):
        return pl.concat(
            [
                self.rows_from_plugin(
                    frame,
                    _plugin.set_index(
                        frames[frame].select([c for ((_, c),) in group]),
                        self.window_rows,
                        self.max_size,
                    ).rename({"column": "col_a"}),
                )
                for frame, group in group_by_frame(combos).items()
            ]
        )
```

`analytics/set_index/__init__.py`:

```python
"""ClickHouse set(max_size) skip-index suitability (per-column, order-dependent). See
SetIndex."""

from analytics.set_index.base import SetIndex
from analytics.set_index.polars import SetIndexPolars
from analytics.set_index.rust import SetIndexRust

REFERENCE = "SetIndexPolars"
IMPLEMENTATIONS = ("SetIndexRust", "SetIndexPolars")

__all__ = ["SetIndex", "SetIndexRust", "SetIndexPolars", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 2: `tests/test_set_index.py`**

```python
"""
Set skip-index accuracy tests — every implementation in analytics.set_index.

Oracles: hand-worked known answers, a brute-force window-by-window oracle
(skip_index_helpers.oracle), SetIndexPolars (the reference) and a Monte Carlo of the
random-order baseline (exact when max_size = 0, a normal approximation otherwise).
Semantics: spec 2026-10-16-skip-index-techniques §2.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datagen import mixed_dtypes, runs_int, runs_str
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)
from skip_index_helpers import (
    ELIGIBLE_DTYPES,
    as_dtype,
    assert_metrics,
    oracle,
    sampled,
    zipf_column,
)

PKG = "analytics.set_index"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"

TOY = [0, 0, 0, 1, 0, 0, 0, 2]  # A A A B A A A C (spec §7)


def stats(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def set_oracle(values: list, window: int, max_size: int = 0) -> dict:
    return oracle(values, window, max_size)[1]


def agree(impl: str, frames: dict, **params) -> None:
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "runs": runs_int(1_000, 2, 10, 50),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames, index_granularity=16, max_size=5), frames)


@pytest.mark.parametrize("impl", ALL)
def test_ineligible_columns_are_listed(impl):
    df = pl.DataFrame(
        {
            "list": [[1], [2]],
            "arr": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
            "struct": [{"a": 1}, {"a": 2}],
            "null": pl.Series([None, None], dtype=pl.Null),
            "i128": pl.Series([1, 2], dtype=pl.Int128),
            "ok": [1, 1],
        }
    )
    out = run(load(impl), {"t": df, "empty": pl.DataFrame(schema={"e": pl.Int64})})
    assert out["status"].to_list() == ["ineligible"] * 5 + ["computed", "ineligible"]
    assert out["window_distinct_max"].to_list() == [None] * 5 + [1, None]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "params", [{}, {"index_granularity": 1_024, "granularity": 2, "max_size": 0}]
)
def test_agrees_with_reference_large_dataset(impl, params):
    agree(impl, {"large": pl.read_ipc(LARGE)}, **params)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("max_size", [0, 10])
def test_agrees_with_reference_seeded(impl, max_size):
    # 200 000 rows: more window starts than one parallel range (2^16)
    frames = {
        "mixed": mixed_dtypes(2_000),
        "ints": runs_int(200_000, 3, 25, 100),
        "strs": runs_str(200_000, 2, 5, 1_000),
    }
    agree(impl, frames, index_granularity=512, max_size=max_size)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(30))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype = ELIGIBLE_DTYPES[seed % len(ELIGIBLE_DTYPES)]
    n = rng.choice([1, 2, 17, 1_000, 70_000])
    k = rng.choice([1, 2, 3, 10])
    if isinstance(dtype, pl.Enum):
        dtype = pl.Enum([str(i) for i in range(k)])
    null_rate = rng.choice([0.0, 0.1, 0.5])
    values, cur = [], 0
    for _ in range(n):  # sticky random walk → clustered values
        if rng.random() < 0.3:
            cur = rng.randrange(k)
        values.append(None if rng.random() < null_rate else cur)
    s = as_dtype(values, dtype)
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([s.slice(0, cut), s.slice(cut)], rechunk=False)
    agree(
        impl,
        {"t": s.to_frame()},
        index_granularity=rng.choice([1, 3, 64, 5_000]),
        max_size=rng.choice([0, 1, 2, 5]),
    )


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_oracle(impl, seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    k = rng.choice([1, 2, 3, 5, 10])
    null_rate = rng.choice([0.0, 0.1, 0.5, 1.0])
    values = [None if rng.random() < null_rate else rng.randrange(k) for _ in range(n)]
    window, max_size = rng.randint(1, 12), rng.choice([0, 1, 2, 3])
    s = pl.Series("x", values, dtype=pl.Int64)
    got = stats(impl, s, index_granularity=window, max_size=max_size)
    assert_metrics(got, set_oracle(values, window, max_size))


def test_kernel_rejects_bad_input():
    from analytics import _plugin

    with pytest.raises(ValueError, match="unsupported dtype"):
        _plugin.set_index(pl.DataFrame({"x": [[1], [2]]}), 4, 0)
    with pytest.raises(ValueError, match="window_rows"):
        _plugin.set_index(pl.DataFrame({"x": [1, 2]}), 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ELIGIBLE_DTYPES, ids=str)
def test_known_answer_every_dtype(impl, dtype):
    r = stats(impl, as_dtype(TOY, dtype), index_granularity=4, max_size=0)
    assert (r["n_windows"], r["window_distinct_mean"], r["window_distinct_max"]) == (5, 2, 2)
    assert r["skip_rows_mean"] == pytest.approx(0.125, rel=1e-12)
    assert r["skip_values_mean"] == pytest.approx(1 / 3, rel=1e-12)
    assert r["skip_rows_std"] == pytest.approx(0.0, abs=1e-15)
    assert r["overflow_fraction"] == 0.0


def test_known_answers_cover_every_eligible_class():
    from analytics._dtypes import ORDERED_SCALARS

    assert {type(dt) for dt in ELIGIBLE_DTYPES} | {pl.Boolean} == set(ORDERED_SCALARS)


@pytest.mark.parametrize("impl", ALL)
def test_boolean(impl):
    values = [False, False, True, True]
    r = stats(impl, pl.Series("x", values), index_granularity=2, max_size=0)
    assert_metrics(r, set_oracle(values, 2))


@pytest.mark.parametrize("impl", ALL)
def test_overflow_skips_nothing(impl):
    r = stats(impl, pl.Series("x", TOY), index_granularity=4, max_size=1)
    assert (r["skip_rows_mean"], r["skip_values_mean"], r["overflow_fraction"]) == (
        0.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("impl", ALL)
def test_null_counts_toward_max_size(impl):
    values = [1, None, 2, None]  # every 2-row window: one value + NULL = 2 entries
    s = pl.Series("x", values)
    r = stats(impl, s, index_granularity=2, max_size=1)
    assert (r["overflow_fraction"], r["window_distinct_max"]) == (1.0, 2)
    r = stats(impl, s, index_granularity=2, max_size=2)
    assert r["overflow_fraction"] == 0.0
    assert_metrics(r, set_oracle(values, 2, 2))


def test_default_max_size_is_100():
    assert reference(PKG)().max_size == 100


@pytest.mark.parametrize("impl", ALL)
def test_sorted_column(impl):
    r = stats(impl, pl.Series("x", list(range(1, 9))), index_granularity=4)
    assert (r["skip_rows_mean"], r["skip_values_mean"]) == (0.5, 0.5)
    assert r["window_distinct_mean"] == 4.0


@pytest.mark.parametrize("impl", ALL)
def test_all_null_column(impl):
    r = stats(impl, pl.Series("x", [None] * 3, dtype=pl.Int64), index_granularity=2)
    assert (r["n_distinct"], r["window_distinct_mean"], r["window_distinct_max"]) == (
        0,
        1.0,
        1,
    )
    assert r["expected_window_distinct"] == pytest.approx(1.0, rel=1e-12)
    assert math.isnan(r["skip_rows_mean"]) and math.isnan(r["expected_skip_values"])


@pytest.mark.parametrize("impl", ALL)
def test_column_shorter_than_window_is_one_window(impl):
    r = stats(impl, pl.Series("x", [3, 1, 2]), index_granularity=8, max_size=0)
    assert (r["window_rows"], r["n_windows"], r["skip_rows_mean"]) == (3, 1, 0.0)
    assert r["expected_skip_rows"] == 0.0
    assert r["expected_window_distinct"] == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("impl", ALL)
def test_string_categorical_enum_identical(impl):
    # a set needs identity, not order: every string-like dtype gives the same result
    s = pl.Series("x", ["b", "a", "a", "c", "c", "c", "a", None, None, "b"])

    def metrics(r):  # dtype differs by design
        return {m: r[m] for m in load(impl).METRICS}

    want = metrics(stats(impl, s, index_granularity=3, max_size=2))
    for dtype in (pl.Categorical, pl.Enum(["c", "b", "a"])):
        got = metrics(stats(impl, s.cast(dtype), index_granularity=3, max_size=2))
        assert got == want, dtype


@pytest.mark.parametrize("impl", ALL)
def test_floats_canonical(impl):
    other_nan = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
    s = pl.Series("x", [0.0, -0.0, float("nan"), other_nan, 1.0, -1.0])
    for series in (s, s.cast(pl.Float32)):
        r = stats(impl, series, index_granularity=2, max_size=0)
        assert r["n_distinct"] == 4
        assert_metrics(r, set_oracle([1, 1, 3, 3, 2, 0], 2))


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_equals_one_chunk(impl):
    s = runs_int(5_000, 1, 7, 40)["c0"]
    parts = pl.concat([s.slice(0, 1_234), s.slice(1_234)], rechunk=False)
    assert parts.n_chunks() == 2
    params = {"index_granularity": 64, "max_size": 3}
    assert stats(impl, parts, **params) == stats(impl, s, **params)


@pytest.mark.parametrize("impl", ALL)
def test_baseline_hand_worked(impl):
    # 1 2 3 4, w = 2, unlimited: each value is absent from half the pairs; 2 entries each
    r = stats(impl, pl.Series("x", [1, 2, 3, 4]), index_granularity=2, max_size=0)
    assert r["expected_skip_rows"] == pytest.approx(0.5, rel=1e-9)
    assert r["expected_skip_values"] == pytest.approx(0.5, rel=1e-9)
    assert r["expected_window_distinct"] == pytest.approx(2.0, rel=1e-9)
    assert r["expected_overflow_fraction"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"max_size": -1}, {"max_size": 1.5}, {"index_granularity": 0}],
)
def test_keywords_are_validated(kwargs):
    with pytest.raises(ValueError, match="must be an integer"):
        reference(PKG)(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical correctness (no conclusions to test: the technique reports metrics only)

CASES = [
    (1, 30, 400, 20, 11, 1.0, 0.0),
    (2, 50, 1_000, 30, 13, 1.2, 0.05),
    (3, 20, 300, 15, 8, 1.5, 0.1),
]  # seed, values, rows, window, max_size (near the mean entry count), Zipf α, null rate


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("case", CASES)
def test_unlimited_baseline_matches_monte_carlo(impl, case):
    seed, k, n, w, _, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    s = pl.Series("x", values, dtype=pl.Int64)
    r = stats(impl, s, index_granularity=w, max_size=0)
    mc = sampled(values, w, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "set_rows"),
        ("expected_skip_values", "set_values"),
        ("expected_window_distinct", "entries"),
    ):
        mean, se = mc[key]
        assert abs(r[metric] - mean) <= 5 * se + 1e-9, (metric, r[metric], mean, se)
    assert r["expected_overflow_fraction"] == 0.0


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("case", CASES)
def test_finite_baseline_within_approximation(impl, case):
    seed, k, n, w, max_size, alpha, null_rate = case
    values = zipf_column(seed, k, n, alpha, null_rate)
    s = pl.Series("x", values, dtype=pl.Int64)
    r = stats(impl, s, index_granularity=w, max_size=max_size)
    mc = sampled(values, w, max_size, seed=seed)
    for metric, key in (
        ("expected_skip_rows", "set_rows"),
        ("expected_skip_values", "set_values"),
        ("expected_overflow_fraction", "overflow"),
    ):
        assert abs(r[metric] - mc[key][0]) <= 0.02, (metric, r[metric], mc[key])
    mean, se = mc["entries"]  # exact whatever max_size
    assert abs(r["expected_window_distinct"] - mean) <= 5 * se + 1e-9
```

- [ ] **Step 3: Run the tests**

Run: `$PY -m pytest tests/test_set_index.py -n 0 -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/analytics/set_index tests/test_set_index.py
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Set index: technique package, Polars reference and accuracy tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Speed benchmarks

**Files:**
- Create: `tests/performance/benchmark_minmax_index.py`
- Create: `tests/performance/benchmark_set_index.py`

- [ ] **Step 1: `benchmark_minmax_index.py`**

```python
"""
MinMax skip-index speed benchmark: MinMaxIndexRust vs MinMaxIndexPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_minmax_index.py

Default window (8 192 rows). Shapes: large_dataset.arrow (realistic mix); clustered 10M x 4
(runs of ~1 000 rows, cardinality 1 000, int + string: index-friendly); random 10M x 4
(cardinality 100 000: wide window ranges); wide 1M x 100 (mean runs 1 / 10 / 100 / 1 000;
parallelism across columns). Results: spec 2026-10-16-skip-index-techniques §10.
"""

import polars as pl
from harness import Dataset, large_dataset, run

from datagen import runs_int, runs_str


def clustered() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.concat(
            [
                runs_int(10_000_000, 2, 1_000, 1_000),
                runs_str(10_000_000, 2, 1_000, 1_000),
            ],
            how="horizontal",
        )
    }


def wide() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.concat(
            [
                runs_int(1_000_000, 25, m, 10_000, prefix=f"m{m}_", seed=m)
                for m in (1, 10, 100, 1_000)
            ],
            how="horizontal",
        )
    }


DATASETS = [
    large_dataset(),
    Dataset("clustered 10M x 4", clustered),
    Dataset("random 10M x 4", lambda: {"t": runs_int(10_000_000, 4, 1, 100_000)}),
    Dataset("wide 1M x 100", wide),
]

if __name__ == "__main__":
    run("analytics.minmax_index", DATASETS)
```

- [ ] **Step 2: `benchmark_set_index.py`**

```python
"""
Set skip-index speed benchmark: SetIndexRust vs SetIndexPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_set_index.py

Default window (8 192 rows) and max_size (100). The datasets of benchmark_minmax_index.py:
clustered columns stay under max_size; random ones overflow every window. Results: spec
2026-10-16-skip-index-techniques §10.
"""

from benchmark_minmax_index import DATASETS
from harness import run

if __name__ == "__main__":
    run("analytics.set_index", DATASETS)
```

- [ ] **Step 3: Run both**

Run: `$PY tests/performance/benchmark_minmax_index.py` then `$PY tests/performance/benchmark_set_index.py`
Expected: each prints a table per dataset (median/min ms, algorithmic / parallel / total speedups, `agrees` True) and writes `tests/performance/results/*.parquet`. Note the numbers for Task 9. If `agrees` is False on any dataset, stop and debug (superpowers:systematic-debugging) before continuing.

- [ ] **Step 4: Commit**

```bash
git add tests/performance/benchmark_minmax_index.py tests/performance/benchmark_set_index.py
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Benchmark: minmax and set skip-index speed (Rust vs Polars)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (§1b, project structure, Rust source layout, `_plugin` list, Current Focus)
- Modify: `ToDo.md:11`
- Modify: `docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md` (append §10 Outcome)

- [ ] **Step 1: CLAUDE.md §1b** — after the Runs test entry, add:

```markdown
**MinMax index — `analytics.minmax_index`** (`MinMaxIndexRust`, ★`MinMaxIndexPolars`) and **Set index — `analytics.set_index`** (`SetIndexRust`, ★`SetIndexPolars`). How well a column, in its row order, suits ClickHouse's `minmax` / `set(max_size)` skip indexes (spec: docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md). A window of w = `index_granularity × granularity` rows slides one row at a time (every granule alignment; a column shorter than w is one window); per window, the share of point queries `col = v` it lets skip — v by row frequency (`skip_rows_mean/std`) or uniform over distinct values (`skip_values_mean/std`), mean and population std over windows — and `expected_skip_rows/values` under a uniformly random row order (closed form from the counts: hypergeometric `avoid(m)` = C(n − m, w)/C(n, w)): observed − expected = gain from ordering. Both: `n_rows, n_non_null, n_distinct, window_rows, n_windows`. minmax: skipped iff v outside [min, max] of the window's non-null values (all-null window: always); order natural, strings bytewise, Categorical by string, Enum by category order, floats canonical with NaN last; baseline exact. set: entries = distinct values + NULL; > `max_size` (0 = unlimited) → stored empty, skips nothing; also `window_distinct_mean/max`, `overflow_fraction`, `expected_window_distinct` (exact), `expected_overflow_fraction`; baseline exact when unlimited, else a normal approximation with exact mean / variance / conditional means over count classes (O(C²) ≤ O(2n); ≤ 0.01 from Monte Carlo). No conclusions. Keywords `index_granularity=8192`, `granularity=1`, set `max_size=100`. Ineligible: as the runs test (`_dtypes.ORDERED_SCALARS`). Rust: techniques/single_column/ordered/skip_index/{mod.rs (`Population` = dense rank ids + counts; `WindowSums` = exact u128 Σa, Σa², merged by addition — the streaming hook: results = merge(scan(population, segment)) + baseline; `avoid`), minmax.rs (monotonic deques), set.rs (per-id window counts)}; ranges of max(2¹⁶, 4w[, d]) window starts in parallel. References: rolling min/max (minmax); occurrence intervals + difference array (set). Agreement: counts exact, observed RTOL 1e-9 (ATOL 1e-12), baselines 1e-6; plus a brute-force oracle fuzz (tests/skip_index_helpers.py). Recommender use: a later spec (reservoir blocks as segments).
```

Also in CLAUDE.md:
- Project structure: in the `techniques/` comment line, `single_column/ordered/{runs_test}` → `single_column/ordered/{runs_test, skip_index/ (minmax, set)}`; the `_dtypes.py, _sets.py` line → `_dtypes.py, _sets.py, _skip_index.py  # dtype groupings / value families; canonical distinct values; the skip-index base`; the `<technique>/` list → add `minmax_index, set_index` after `runs_test`; under `tests/`, add `├── skip_index_helpers.py                # skip-index oracle, Monte Carlo sampler`.
- Private `_plugin` list: after `runs_test`, add `minmax_index`, `set_index`.
- Current Focus: "All ten techniques (GCD, Describe, Top-k; Runs test; …" → "All twelve techniques (GCD, Describe, Top-k; Runs test, MinMax index, Set index; …"; "Next: the runs test's use by the StreamingRecommender" → "Next: the runs test's and skip indexes' use by the recommenders (the StreamingRecommender on the reservoir's row blocks)".

- [ ] **Step 2: ToDo.md** — replace line 11:

```markdown
- ~~Window functions of a specified width … minmax and skipindexes~~ — delivered as techniques `analytics.minmax_index` / `analytics.set_index` (spec 2026-10-16); recommender use pending (item 5).
```

- [ ] **Step 3: Spec outcome** — append to the spec:

```markdown
## 10. Outcome

Benchmark (tests/performance/benchmark_{minmax,set}_index.py, <CPUs> CPUs, median ms; `agrees` on every dataset):

| Technique | Dataset | Polars | Rust (N threads) | Rust @ 1 thread | algorithmic | parallel | total |
|---|---|---|---|---|---|---|---|
```

with one row per (technique, dataset) copied from Task 8's output (replace `<CPUs>` with the machine's count), followed by 2–4 bullets on what the numbers show (where Rust wins or loses, and why).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md ToDo.md
git add -f docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Document the skip-index techniques (CLAUDE.md §1b, ToDo, spec outcome)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Full check

- [ ] **Step 1: Run what CI runs**

Run: `$PY scripts/check.py check`
Expected: rustfmt, black, clippy (both feature sets), ruff, layering, build, `cargo test --lib`, pytest — all pass. Fix any clippy / ruff / black findings in the files this plan touched (run `$PY scripts/check.py fmt` for formatting), then re-run.

- [ ] **Step 2: Commit any fixes**

```bash
git add -u
PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "Skip index: lint and format fixes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Skip the commit if nothing changed.)
