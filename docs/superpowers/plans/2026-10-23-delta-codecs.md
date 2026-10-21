# Delta and DoubleDelta Codecs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `delta` technique (order-dependent metrics on consecutive differences) and `Delta` / `DoubleDelta` codec families in both recommenders' ClickHouse codec choice, emulated byte-exact and calibrated against ClickHouse.

**Architecture:** Increment 1 (Tasks 1–2): a Rust kernel `techniques/single_column/ordered/delta.rs` with a pure `DeltaStats` core over `&[i128]`, wrapped as the technique `turbo_parakeet.techniques.delta` (Rust + Polars reference). Increment 2 (Tasks 3–7): the emulator's `Chain` gains a `Transform` (None / Gcd / Delta / DoubleDelta); a spike pins the on-disk layouts against the live server; the decision prunes the delta families with `DeltaStats` computed on the emulator's integer streams (one code path for one-shot and streaming) and chooses among families with `transform_min_saving` (renamed from `gcd_min_saving` in every binding); `to_chbench` and the calibration learn the new codecs.

**Tech Stack:** Rust (polars 0.51 Series, arrow-rs, rayon, foldhash), pyo3, Python 3.12 / Polars, Java 25 FFM, ClickHouse 26.8 (Docker `chbench-ch`), chbench.

**Spec:** `docs/superpowers/specs/2026-10-23-delta-codecs-design.md` (read it first; section numbers below refer to it).

Conventions (every task): repo `c:/Users/Alexander/turbo-parakeet`, branch `delta-codecs` (Task 0). Every shell: `E=/c/Users/Alexander/miniconda3/envs/p312; export PATH="$E:$E/Scripts:$E/Library/bin:/c/Users/Alexander/.cargo/bin:$PATH" PYO3_PYTHON="$E/python.exe"; PY=$E/python.exe`. Rust unit tests: `cd services/analytics && cargo test --lib <module>::`. After Rust changes the extension needs `$PY scripts/check.py build` (warm ≈ 2.5 min) before pytest sees them. Before committing: `$PY scripts/check.py fmt && $PY scripts/check.py lint` (revert unrelated files fmt touched). Commit as `git -c user.email=benssutton@hotmail.com -c user.name=benssutton commit …`, never `--no-verify`; messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `docs/superpowers/` needs `git add -f`. ToDo.md items 15–16 stay byte-for-byte intact. Never push. Use the Write / Edit tools for files; use `git grep` (not recursive grep) under `services/analytics` (its `target/` is huge). Clippy runs with `-D warnings`; for items used only by later tasks use a module-level `#![cfg_attr(not(test), allow(dead_code))]`, removed once used.

---

### Task 0: Branch

- [ ] `git switch -c delta-codecs`

### Task 1: Rust kernel `delta.rs` and its binding

**Files:**
- Create: `services/analytics/src/techniques/single_column/ordered/delta.rs`
- Modify: `services/analytics/src/techniques/single_column/ordered/mod.rs` (add `pub(crate) mod delta;`, doc line), `services/analytics/src/techniques/mod.rs:12` (re-export `delta`), `services/analytics/src/bindings/api/techniques.rs` (entry point), `services/analytics/src/bindings/python.rs` (pyfunction + `m.add_function`), `services/analytics/python/turbo_parakeet/_plugin.py` (wrapper)

- [ ] **Step 1: the pure core.** In `delta.rs`:

```rust
//! Consecutive differences of an integer-backed column, in row order (spec 2026-10-23 §2):
//! ranges, mean significant bytes, zero shares and entropies of the deltas and double-deltas,
//! and the mean significant bytes of deltas under a random row order.

use foldhash::fast::RandomState;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::common::entropy_math::entropy_from_counts_iter;

/// Bytes of the smallest signed integer holding `v`; 0 for 0 (spec §2).
pub(crate) fn sig(v: i128) -> u32 {
    if v == 0 {
        return 0;
    }
    let m = if v >= 0 { v } else { !v };
    let bits = 128 - m.leading_zeros(); // magnitude bits; + 1 sign bit
    (bits + 8) / 8
}

const CHUNK: usize = 1 << 16;

/// Statistics of one column's non-null values in row order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeltaStats {
    pub n_values: u64,
    pub n_deltas: u64,
    pub n_double_deltas: u64,
    pub delta_min: Option<i128>,
    pub delta_max: Option<i128>,
    pub double_delta_min: Option<i128>,
    pub double_delta_max: Option<i128>,
    pub value_bytes: u64,        // Σ sig(value)
    pub delta_bytes: u64,        // Σ sig(delta)
    pub double_delta_bytes: u64, // Σ sig(double delta)
    pub zero_deltas: u64,
    pub zero_double_deltas: u64,
}

impl DeltaStats {
    pub(crate) fn value_bytes_mean(&self) -> f64 { mean(self.value_bytes, self.n_values) }
    pub(crate) fn delta_bytes_mean(&self) -> f64 { mean(self.delta_bytes, self.n_deltas) }
    pub(crate) fn double_delta_bytes_mean(&self) -> f64 {
        mean(self.double_delta_bytes, self.n_double_deltas)
    }
}

fn mean(sum: u64, n: u64) -> f64 {
    if n == 0 { f64::NAN } else { sum as f64 / n as f64 }
}

/// Byte-level statistics (no entropies, no baseline): the recommender's pruning input.
/// Deltas are taken within `values` only (a caller with several blocks merges with `merge`).
pub(crate) fn scan(values: &[i128]) -> DeltaStats { /* see Step 2 */ }

/// Sum of two scans of disjoint blocks (no delta across the blocks).
pub(crate) fn merge(a: DeltaStats, b: DeltaStats) -> DeltaStats { /* field-wise sums, min/max */ }
```

- [ ] **Step 2: implement `scan`** — `n = values.len()`; deltas `d[i] = values[i+1] − values[i]` (i < n−1), double-deltas `dd[i] = d[i+1] − d[i]` (i < n−2), both computed on the fly in i128 (never overflows: |dd| < 2^66). Parallel over index chunks of `CHUNK` (`(0..n).into_par_iter().step_by(CHUNK)` or `par_chunks` over index ranges), each chunk reading `values[i..i+3]` across its end (no merge state needed), reduced by `merge`-style addition. `value_bytes = Σ sig(v)`.

- [ ] **Step 3: entropies and baseline** (technique only):

```rust
/// Shannon entropy (bits) of a multiset of i128 keys; NaN when empty.
fn entropy(keys: impl ParallelIterator<Item = i128>) -> f64 {
    let counts: HashMap<i128, u64, RandomState> = keys
        .fold(HashMap::default, |mut m, k| { *m.entry(k).or_insert(0) += 1; m })
        .reduce(HashMap::default, |mut a, b| {
            for (k, c) in b { *a.entry(k).or_insert(0) += c; }
            a
        });
    let r: u64 = counts.values().sum();
    if r == 0 { return f64::NAN; }
    entropy_from_counts_iter(counts.into_values(), (r as f64).log2(), r as f64)
}

/// Mean sig(Δ) of Δ = x_j − x_i over ordered pairs of distinct positions (spec §2):
/// Σ_{k=1..9} P(sig ≥ k); sig ≥ 1 iff Δ ≠ 0, sig ≥ k (k ≥ 2) iff Δ ∉ [−B, B), B = 2^(8(k−1)−1).
pub(crate) fn expected_delta_bytes(values: &[i128]) -> f64 {
    let n = values.len();
    if n < 2 { return f64::NAN; }
    let mut v = values.to_vec();
    v.par_sort_unstable();
    let pairs = (n as f64) * (n as f64 - 1.0);
    // k = 1: pairs with equal values
    let mut equal: u128 = 0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && v[j] == v[i] { j += 1; }
        let c = (j - i) as u128;
        equal += c * (c - 1);
        i = j;
    }
    let mut expected = 1.0 - equal as f64 / pairs;
    for k in 2..=9u32 {
        let b: i128 = 1i128 << (8 * (k - 1) - 1);
        // inside = #ordered (i, j), i ≠ j, with v[i] − b ≤ v[j] < v[i] + b
        let (mut lo, mut hi, mut inside) = (0usize, 0usize, 0u128);
        for i in 0..n {
            while lo < n && v[lo] < v[i] - b { lo += 1; }
            while hi < n && v[hi] < v[i] + b { hi += 1; }
            inside += (hi - lo - 1) as u128;
        }
        expected += 1.0 - inside as f64 / pairs;
    }
    expected
}
```
(`v[i] ± b` cannot overflow i128: values are ≤ 2^64 in magnitude, b ≤ 2^63.)

- [ ] **Step 4: Series entry** — `fn physical_values(s: &Series) -> PolarsResult<Option<Vec<i128>>>`: `None` for ineligible dtypes; else the non-null physical values as i128, following `gcd.rs`'s per-physical-dtype `match` (`s.to_physical_repr()`, then `i8()…u64()`, and `i128()` for Decimal) — eligible: Int/UInt 8–64, Decimal, Date, Datetime, Duration, Time (same set as `gcd.rs::is_integer_backed`). Then

```rust
/// `column, n_rows, n_non_null, n_deltas, delta_min, delta_max, double_delta_min,
/// double_delta_max, value_bytes_mean, delta_bytes_mean, double_delta_bytes_mean,
/// zero_delta_fraction, zero_double_delta_fraction, delta_entropy, double_delta_entropy,
/// expected_delta_bytes_mean` per column (columns in parallel). Ineligible columns: an
/// InvalidOperation error naming the column (the Python base never sends them).
pub(crate) fn delta_impl(inputs: &[Series]) -> PolarsResult<Series>
```
Ranges as `Decimal(38, 0)` Series (`Int128Chunked::from_iter_options(..).into_decimal_unchecked(Some(38), 0)` — copy how `gcd.rs` builds its Decimal column); fractions = zeros / count (NaN when count 0); entropies via Step 3 over `(0..n−1)` deltas and `(0..n−2)` double-deltas (`into_par_iter().map(...)`). Columns in parallel (`inputs.par_iter()`), as `runs_test_impl` does.

- [ ] **Step 5: unit tests** (in `delta.rs`, `#[cfg(test)]`):
  - `sig`: 0→0, 1→1, −1→1, 127→1, 128→2, −128→1, −129→2, `i64::MAX`→8, `u64::MAX as i128`→9, `i64::MIN`→8.
  - `scan(&[10, 20, 30, 40])`: n_deltas 3, delta 10..10, double-deltas 2 of 0, zero_double_deltas 2, double_delta_bytes 0, value_bytes 4.
  - `scan(&[])`, `scan(&[5])`: counts 0, ranges None, means NaN.
  - chunk boundary: a 200 000-value random walk (seeded `rand` not available → build with a simple LCG) — `scan` equals a sequential reference loop written in the test.
  - `merge` of `scan(a)` and `scan(b)` equals field-wise sums (no cross delta).
  - `expected_delta_bytes` equals an O(n²) brute force over all ordered pairs for 50 seeded random vectors of length 2..40 with values in ±300 and ±70 000 (crossing the 1- and 2-byte boundaries), and for `[i64::MIN, i64::MAX, 0]` as i128.
  - `entropy` of deltas of `[0, 1, 2, 4]` = entropy of {1, 1, 2} = 0.9183 bits.
  Run: `cargo test --lib techniques::single_column::ordered::delta::` → pass.

- [ ] **Step 6: bindings.** `api/techniques.rs`:

```rust
/// `column` + the delta technique's metrics per column (rows in batch order).
pub fn delta(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::techniques::delta::delta_impl(&columns(batch)?))
}
```
`python.rs` (next to `runs_test`, and register it):

```rust
#[pyfunction]
fn delta(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    let batch = /* exactly as runs_test reads its batch */;
    run(py, || api::delta(&batch)).map(ArrowTable)
}
```
`_plugin.py`:

```python
def delta(df: Frame) -> pl.DataFrame:
    """column + Delta's metrics per column — rows in the frame's order."""
    return pl.DataFrame(_rs.delta(_collected(df)))
```

- [ ] **Step 7:** `$PY scripts/check.py build`, then `$PY -c "import polars as pl; from turbo_parakeet import _plugin; print(_plugin.delta(pl.DataFrame({'t': [10, 20, 30, 41]})))"` shows one row with delta_min 10, delta_max 11. fmt, lint; commit `"delta technique: Rust kernel and binding (spec 2026-10-23 §2)"`.

### Task 2: Python technique package, tests, benchmark

**Files:**
- Create: `services/analytics/python/turbo_parakeet/techniques/delta/{__init__.py, base.py, rust.py, polars.py}`, `tests/test_delta.py`, `tests/performance/benchmark_delta.py`
- Modify: `services/analytics/python/turbo_parakeet/techniques/__init__.py` (docstring list)

- [ ] **Step 1: `base.py`** — `class Delta(Technique)`: `SCOPE = "per_column"`, `ARITY = 1`, `DESCRIPTORS = {"dtype": pl.String}`, `RTOL = 1e-9`, `ATOL = 0.0`, no CONCLUSIONS, and

```python
METRICS = {
    "n_rows": pl.UInt64,
    "n_non_null": pl.UInt64,
    "n_deltas": pl.UInt64,
    "delta_min": pl.Decimal(38, 0),
    "delta_max": pl.Decimal(38, 0),
    "double_delta_min": pl.Decimal(38, 0),
    "double_delta_max": pl.Decimal(38, 0),
    "value_bytes_mean": pl.Float64,
    "delta_bytes_mean": pl.Float64,
    "double_delta_bytes_mean": pl.Float64,
    "zero_delta_fraction": pl.Float64,
    "zero_double_delta_fraction": pl.Float64,
    "delta_entropy": pl.Float64,
    "double_delta_entropy": pl.Float64,
    "expected_delta_bytes_mean": pl.Float64,
}
```
`eligible(series)`: dtype in GCD's `INTEGER_BACKED` (import it from `turbo_parakeet.techniques.gcd.base`) and `series.len() > 0`; `describe` as `RunsTest.describe`. Docstring: the spec §2 semantics (nulls skipped, sig definition, baseline meaning, floats ineligible because Gorilla is out of scope, metrics only). `n_rows` is computed by the Rust class / reference (the kernel does not return it: add it in the Python wrapper from the frame height if the kernel omits it — or have the kernel return it; pick the kernel, simpler).

- [ ] **Step 2: `rust.py`** — copy `runs_test/rust.py`, calling `_plugin.delta`.

- [ ] **Step 3: `polars.py`** — the reference, sharing no code with the kernel:

```python
def _sig_mean(x: pl.Series) -> float:
    """Mean significant bytes of an Int128 series (spec §2); NaN when empty."""
    if x.len() == 0:
        return math.nan
    total = (x != 0).sum()
    for k in range(2, 10):
        b = 1 << (8 * (k - 1) - 1)
        total += ((x < -b) | (x >= b)).sum()
    return total / x.len()


def _entropy(x: pl.Series) -> float:
    if x.len() == 0:
        return math.nan
    p = x.value_counts()["count"].cast(pl.Float64) / x.len()
    return float(-(p * p.log(2)).sum())


def _expected_bytes(values: list[int]) -> float:
    """Exact random-order mean sig(Δ), by bisect over the sorted values."""
    n = len(values)
    if n < 2:
        return math.nan
    v = sorted(values)
    pairs = n * (n - 1)
    equal = sum(c * (c - 1) for c in Counter(v).values())
    total = Fraction(pairs - equal, pairs)
    for k in range(2, 10):
        b = 1 << (8 * (k - 1) - 1)
        inside = sum(bisect_left(v, x + b) - bisect_left(v, x - b) - 1 for x in v)
        total += Fraction(pairs - inside, pairs)
    return float(total)
```
Per column: `x = s.drop_nulls().to_physical().cast(pl.Int128)`, `d = x.diff().slice(1)`, `dd = d.diff().slice(1)`; ranges as Python ints → `pl.Decimal(38, 0)`; fractions `(d == 0).sum() / d.len()` (NaN when empty); `value_bytes_mean = _sig_mean(x)`. Build the result frame like `RunsTestPolars` does (one dict per column → `self.rows(...)` / the base's helper — copy RunsTestPolars' `_compute`).

- [ ] **Step 4: `__init__.py`** — `REFERENCE = "DeltaPolars"`, `IMPLEMENTATIONS = ("DeltaRust", "DeltaPolars")`, `__all__`. Add `delta` to the "Per-column, ordered" line of `techniques/__init__.py`'s docstring.

- [ ] **Step 5: `tests/test_delta.py`** — the four blocks (copy `tests/test_runs_test.py`'s structure, `PKG = "turbo_parakeet.techniques.delta"`):
  1. contract over `large_dataset_mixed()` (every implementation): `assert_contract`; floats / strings / Boolean / Categorical / lists ineligible; zero-row column ineligible.
  2. agreement Rust vs reference: `assert_agrees` on `large_dataset()` and on a frame of every eligible dtype (Int8…UInt64, Decimal(10, 2), Date, Datetime("us"), Datetime("ns", "UTC"), Duration("ms"), Time) with nulls; plus columns longer than 2¹⁶ (chunk boundaries): a 300 000-row random walk.
  3. known answers (reference included, parametrised over ALL): regular series `[10, 20, 30, 40]` → delta 10..10, double-delta bytes mean 0.0, zero_double_delta_fraction 1.0, double_delta_entropy 0.0; constant column → delta_bytes_mean 0.0, delta_entropy 0.0; Int64 `[i64::MIN, i64::MAX]` → delta_max 2^64 − 1 as Decimal, delta_bytes_mean 9.0; nulls between values `[1, None, 3, None, 6]` → n_non_null 3, deltas 2 and 3; one value → n_deltas 0, delta_* null, means NaN; Date `[2020-01-01, 2020-01-02, 2020-01-04]` → deltas 1 and 2 (days); Datetime("ms") steps of 1 000 → delta 1000..1000; Decimal(10, 2) `[1.00, 1.50]` → delta 50.
  4. baseline: `expected_delta_bytes_mean` of DeltaRust equals an O(n²) Python brute force (mean of sig over all ordered pairs of distinct positions) on 20 seeded random Int64 columns of 2..60 values spanning the byte boundaries.
  (No conclusions block: the technique has none — add a one-line comment saying so.)
  Run: `$PY -m pytest tests/test_delta.py -q` → pass.

- [ ] **Step 6: `tests/performance/benchmark_delta.py`** — copy `benchmark_runs_test.py`'s form: `run("turbo_parakeet.techniques.delta", [large_dataset(), Dataset("timestamps 10M x 4", ...), Dataset("wide 1M x 100", ...)])` with seeded numpy timestamps (regular step + jitter) and random walks. Run it once (it is slow; report the speedups in the task report, not in git).

- [ ] **Step 7:** fmt, lint, `$PY -m pytest tests/test_delta.py tests/test_plugin.py -q`; commit `"delta technique: Python package, accuracy tests, benchmark (spec 2026-10-23 §2)"`.

### Task 3: Emulator transforms

**Files:** `services/analytics/src/common/clickhouse_sizes.rs`, and every user of `Chain { gcd, .. }`: `services/analytics/src/recommenders/engine/compression.rs`, `services/analytics/src/recommenders/engine/column.rs`, `services/analytics/src/bindings/api/techniques.rs` (`parse_chain`, `clickhouse_profile`), plus their tests (`git grep -n "gcd:" services/analytics/src` finds them).

- [ ] **Step 1: types.**

```rust
/// The transform codec before the compression (spec 2026-10-23 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Transform { None, Gcd, Delta, DoubleDelta }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Compression {
    /// No generic compression: the spike and `clickhouse_profile` only, never chosen.
    None,
    Lz4,
    Zstd(i32),
}

pub(crate) struct Chain { pub transform: Transform, pub compression: Compression }
```
`Chain::ddl()`: transform name (`"GCD"`, `"Delta"`, `"DoubleDelta"`) and compression (`"LZ4"`, `"ZSTD(l)"`) joined by `", "`, omitting `None` parts; `Transform::None` + `Compression::None` → `"NONE"`. Rename `Stream::gcd_width` to `int_width` (doc: "(value width, signed) for integer streams: the GCD, Delta and DoubleDelta codecs apply") and `ChLayout::Fixed::gcd_applies` to `int_codecs` — both via `git grep`, every use. `compressed_len(_, Compression::None)` = the input length.

- [ ] **Step 2: transforms.** Next to `gcd_transform`:

```rust
/// ClickHouse `Delta(width)` on one block (CompressionCodecDelta): header [width,
/// bytes_to_skip], the unaligned tail bytes verbatim, then each value minus the previous
/// (the first minus 0), wrapping at the width. Layout verified by the spike (Task 4).
fn delta_transform(block: &[u8], width: usize) -> Vec<u8>

/// ClickHouse `DoubleDelta` on one block (CompressionCodecDoubleDelta): header [width,
/// bytes_to_skip], tail bytes, UInt32 item count, the first value, the first delta (width
/// bytes), then per double-delta a bit-packed code: 0 → bit `0`; otherwise a prefix
/// `10` / `110` / `1110` / `11110` / `11111` selecting 7 / 9 / 12 / 32 / 64 data bits (sign
/// + magnitude), flushed to whole bytes. Exact bucket bounds, sign handling and the widths'
/// variants are taken from ClickHouse's source and confirmed by the spike (Task 4).
fn double_delta_transform(block: &[u8], width: usize) -> Vec<u8>
```
Write them from ClickHouse's source: `src/Compression/CompressionCodecDelta.cpp` and `CompressionCodecDoubleDelta.cpp` (ClickHouse 26.8; fetch `https://raw.githubusercontent.com/ClickHouse/ClickHouse/master/src/Compression/CompressionCodecDoubleDelta.cpp` and the Delta file with WebFetch, or read them in the server container: they are not shipped there, so WebFetch). Implement a `BitWriter` (MSB-first or LSB-first exactly as ClickHouse's `BitHelpers.h` — fetch it too). In `emulate`, dispatch on `chain.transform` for streams with `int_width: Some((w, signed))` (GCD unchanged; Delta / DoubleDelta need `w ∈ {1, 2, 4, 8}`, else the block passes through unchanged); streams without `int_width` pass through.

- [ ] **Step 3: chains in the emulated size.** A chain of two codecs is ClickHouse's `CompressionCodecMultiple`: model its extra header (method count + method bytes, and whether each inner codec writes its own 9-byte header) from `CompressionCodecMultiple.cpp` / `ICompressionCodec.cpp`; the existing `"GCD, LZ4"` cells (calibrated within 1.86%) must not get worse — if the model changes GCD sizes, the spike decides.

- [ ] **Step 4: `parse_chain`** accepts `[GCD|Delta|DoubleDelta][, ][LZ4|ZSTD(n)]` and a lone `GCD` / `Delta` / `DoubleDelta` / `NONE` (compression None); `clickhouse_profile` emulates a transform only where the layout has `int_codecs` (else bytes null, as for GCD today).

- [ ] **Step 5: unit tests** (clickhouse_sizes.rs): `ddl()` strings for every combination; `delta_transform` on `[1u32, 3, 6]` little-endian → header + `1, 2, 3` (+ tail handling with a 2-byte remainder); `delta_transform` wraps (`u8` 250 → 5 = 11); `double_delta_transform` on a regular `u64` series of 1 000 values has size header + 4 + 8 + 8 + ⌈998 / 8⌉ (all zero bits); round-trip property: write a test-only `double_delta_decode` from the same source and check decode(encode(x)) = x for 200 seeded random blocks at widths 1, 2, 4, 8 including extremes (i64::MIN / MAX steps). Existing emulator tests updated to the new `Chain` and still pass. Run `cargo test --lib common::clickhouse_sizes::` and `cargo test --lib bindings::api::techniques::`.

- [ ] **Step 6:** `compression.rs` / `column.rs` compile with the new `Chain` (`gcd: true` → `transform: Transform::Gcd`), no behaviour change yet; `$PY scripts/check.py check` passes. Commit `"emulator: Delta and DoubleDelta transforms (spec 2026-10-23 §3)"`.

### Task 4: Spike — emulated vs measured bytes on the live server

**Files:** Create `tests/performance/spike_clickhouse_delta.py`; modify the spec §7.

- [ ] **Step 1:** check the server: `docker ps --format '{{.Names}} {{.Status}}'` lists `chbench-ch` Up (else `docker start chbench-ch`). Credentials: user `CLICKHOUSE_USER`, password `CLICKHOUSE_PASSWORD`, `http://localhost:8123`, database `bench`.

- [ ] **Step 2: the script** (standalone, never collected; docstring with the run line, like `spike_datafusion_ree.py`). For each column below and each codec in `NONE, Delta, DoubleDelta, GCD, LZ4, Delta, LZ4, DoubleDelta, LZ4, Delta, ZSTD(1), DoubleDelta, ZSTD(1)`: `CREATE TABLE bench.spike_delta (v <type> CODEC(<codec>)) ENGINE = MergeTree ORDER BY tuple() SETTINGS min_bytes_for_wide_part = 0` (with `allow_suspicious_codecs = 1` as a query setting for codec-only chains), `INSERT … FORMAT Arrow` with the column's Arrow IPC stream bytes as the HTTP body (`urllib.request`, basic auth), `OPTIMIZE TABLE … FINAL`, read `data_compressed_bytes` from `system.columns`, drop the table; compare with `_plugin.clickhouse_profile(frame, "v", codec)["bytes"]` (the emulator, Task 3). Columns (1 000 000 rows, seeded): regular `UInt32` seconds (step 60), `DateTime64(3)`-like `Int64` ms with jitter ±5 and gaps, random walk `Int64` (steps ±1 000), sorted `UInt64` ids with random gaps 1–10, random `UInt16`, random `Int8`. Print a table: column, type, codec, emulated, measured, diff %.

- [ ] **Step 3: iterate** on Task 3's transforms / chain header until codec-only rows (`Delta`, `DoubleDelta`, `GCD` with `NONE` compression) are byte-exact and chained rows are within max(2 % × measured, 256 bytes). Commit emulator fixes with the script: `"Spike: Delta / DoubleDelta on-disk layout against ClickHouse 26.8"`.

- [ ] **Step 4:** fill the spec §7 with the final table and the layout facts learned (headers, bit order, buckets, the Multiple header); commit `"Spec: Delta / DoubleDelta spike findings (§7)"` (`git add -f`).

If byte-exactness cannot be reached for DoubleDelta after reading the source, STOP and report the residual table (the decision then needs a tolerance from the user).

### Task 5: Decision — delta families, pruning, `transform_min_saving`

**Files:** `services/analytics/src/recommenders/engine/compression.rs`, `services/analytics/src/recommenders/engine/column.rs`, the keyword rename across `services/analytics/src/bindings/{api/recommenders.rs, capi.rs, python.rs}`, `services/analytics/python/turbo_parakeet/{_plugin.py, recommenders/oneshot.py, recommenders/streaming.py}`, `services/analytics/bindings/java/src/main/java/io/github/benssutton/analytics/{OneShotParams, StreamingParams, OneShotRecommender, StreamingRecommender}.java` (+ Java tests using the parameter), `tests/test_clickhouse_recommend.py`.

- [ ] **Step 1: rename** `gcd_min_saving` → `transform_min_saving` (Java `gcdMinSaving` → `transformMinSaving`) everywhere (`git grep -n -i "gcd_min_saving\|gcdMinSaving"` → nothing left outside docs/superpowers and CLAUDE.md); docstrings: "a transform codec (GCD, Delta, DoubleDelta) is chosen only if it saves at least this fraction over no transform (default 0.01)". `$PY scripts/check.py check` and `test-java` pass before going on.

- [ ] **Step 2: stream statistics** (`column.rs`): a function

```rust
/// Delta statistics of the integer streams of `st` (one entry per array: one-shot one array,
/// streaming the sampled blocks), deltas within each stream; None when no stream is an
/// integer stream of width 1, 2, 4 or 8.
fn stream_delta_stats(st: &[Vec<Stream>]) -> Option<DeltaStats>
```
decoding each `Stream` with `int_width: Some((w, signed))` into `Vec<i128>` (little-endian, sign-extended when `signed`, nulls are the zeros already in the stream) and folding `delta::scan` with `delta::merge`.

- [ ] **Step 3: `decide`** — replace `gcd: Option<(u128, u128, bool)>` + `gcd_note` by a `Transforms` input:

```rust
pub(crate) struct Transforms<'a> {
    /// Some((g, max_abs, signed)) when GCD applies (unchanged byte-width rule).
    pub gcd: Option<(u128, u128, bool)>,
    pub gcd_note: &'a str,
    /// Delta statistics when Delta / DoubleDelta apply (integer stream, width 1/2/4/8).
    pub delta: Option<DeltaStats>,
}

/// Spec §4: Delta survives iff mean delta bytes < mean value bytes; DoubleDelta iff mean
/// double-delta bytes < min(mean value bytes, mean delta bytes). (survives, evidence).
pub(crate) fn delta_survives(s: &DeltaStats) -> (bool, String)
pub(crate) fn double_delta_survives(s: &DeltaStats) -> (bool, String)
```
Evidence strings: `"delta {:.2} B vs value {:.2} B"`, `"double delta {:.2} B vs min(value {:.2} B, delta {:.2} B)"`; NaN means (fewer than 2 / 3 values) → pruned with evidence "too few values". `Family { transform: Transform, .. }`; families = plain + each surviving transform; each measured as today (LZ4, then ZSTD levels unless the shortcut fires on that family's histogram). Winner: among transform families whose pick saves `transform_min_saving` over the plain pick, the smallest pick bytes (ties: GCD, Delta, DoubleDelta order); none → plain. Candidate rows: as today, with "rejected" evidence `"{transform} saves < {pct}% over no transform ({b} vs {plain})"` for a qualifying-family pick that is not the winner use `"{winner} smaller ({w} vs {b})"`.

- [ ] **Step 4: Rust unit tests** (compression.rs): delta_survives / double_delta_survives on hand-made stats; `decide` with a fake `measure` closure (as the existing tests do): Delta family chosen when it saves 5%; rejected when it saves 0.5%; DoubleDelta beats Delta when smaller; pruned families emit `pruned` rows and are never measured (assert on the closure's recorded calls); GCD behaviour unchanged (existing tests pass with the new signature).

- [ ] **Step 5: wire `column.rs`** — compute `stream_delta_stats(&st)` when the layout's `int_codecs` holds and the width ∈ {1, 2, 4, 8}; pass `Transforms { gcd, gcd_note: note, delta }`.

- [ ] **Step 6: Python tests** (`tests/test_clickhouse_recommend.py`, one-shot and streaming):
  - regular `pl.Datetime("ms")` 200 000 rows step 1 000 ms → `clickhouse_codecs` starts with `"DoubleDelta"`; candidates contain `Delta` rows too;
  - random walk Int64 (steps ±100, start 10¹²) → starts with `"Delta"` (if DoubleDelta wins, check the sizes and adjust the walk so that the expectation holds — the test pins the decision rule, not the data);
  - random Int64 → no Delta / DoubleDelta measured, `pruned` rows with "delta" evidence;
  - Float64, String, Decimal(30, 2) (Decimal128) columns → no Delta / DoubleDelta rows;
  - `transform_min_saving=0.99` → no transform chosen;
  - streaming parity: one batch, `block_rows` a multiple of 8 192 → same `clickhouse_codecs` and sizes as one-shot.
  Java: an existing params test round-trips `transformMinSaving`.

- [ ] **Step 7:** `$PY scripts/check.py ci` → all pass. Commit `"recommenders: Delta / DoubleDelta codec families, transform_min_saving (spec 2026-10-23 §4–5)"`.

### Task 6: `to_chbench` and calibration

**Files:** `services/analytics/python/turbo_parakeet/recommenders/chbench.py`, `tests/performance/calibrate_clickhouse_sizes.py`, tests for `to_chbench` in `tests/test_clickhouse_recommend.py`, the spec §6 / §7.

- [ ] **Step 1: `to_chbench`** (`_write`): the recommended codec is the chain's transform: `rec_codec = "LowCardinality" if … else chosen.split(", ")[0] if chosen.split(", ")[0] in ("GCD", "Delta", "DoubleDelta") else "none"`; `predict(k, c)` builds `f"{k}, {c}"` for any transform `k`; the original-type suite measures `f"{rec_codec}, {comp}"` for a transform `rec_codec` (generalising the GCD-only branch). Test: a regular-timestamp frame's suite YAML has codec axis `["DoubleDelta", "none"]` and `predicted.json` keys for both. Update the spec §6 sentence "with no change beyond the chain's DDL" to "the transform generalises `to_chbench`'s GCD handling".

- [ ] **Step 2: calibration frames** — add `delta_frames()` beside `gcd_frames()` (1 000 000 rows, seeded): `ts_regular` (Datetime("ms"), step 1 000), `ts_jitter` (step 1 000 ± 5 with 1% gaps of 1–60 s), `walk` (Int64 random walk ±1 000), `ids_sorted` (UInt64 cumulative sums of 1–10), `walk_i16` (Int16 walk ±3, clipped), `walk_i32` (Int32 walk ±100), `random_ints` (Int64 uniform, the control). Add them to `frames` in `main`.

- [ ] **Step 3: run** `$PY tests/performance/calibrate_clickhouse_sizes.py` (needs `chbench-ch`; ≈ 10–20 min). Expected: every Delta / DoubleDelta cell within max(2 % × measured, 256 bytes); existing cells unchanged. Record the summary table in the spec §7 ("Calibration" subsection). If a cell is out of tolerance, investigate (emulator vs chbench DDL) before going on; report if unresolved.

- [ ] **Step 4:** fmt, lint, `$PY -m pytest tests/test_clickhouse_recommend.py -q`; commit `"to_chbench and calibration: Delta / DoubleDelta (spec 2026-10-23 §6)"` (+ `git add -f` the spec).

### Task 7: Documentation

**Files:** `CLAUDE.md`, `ToDo.md`.

- [ ] CLAUDE.md: §1b gains a **Delta — `turbo_parakeet.techniques.delta`** paragraph (`DeltaRust`, ★`DeltaPolars`; spec 2026-10-23; eligibility, metrics, sig definition, exact baseline, no conclusions / keywords, Rust file, agreement tolerances, benchmark outcome from Task 2 Step 6); the OneShotRecommender paragraph's compression decisions mention the Delta / DoubleDelta families, their pruning rule and `transform_min_saving` (replacing every `gcd_min_saving` / `gcdMinSaving`, incl. the capi and Java lists); calibration line gets the Delta / DoubleDelta result; Project Structure: `techniques/single_column/ordered/{runs_test, delta, skip_index/}`, the Python `delta/` package, `tests/test_delta.py`, `benchmark_delta.py`, `spike_clickhouse_delta.py`; "Twelve techniques" → thirteen in Current Focus, and Current Focus says ToDo 3 is done (spec 2026-10-23).
- [ ] ToDo.md item 3: prefix `**Done** (spec 2026-10-23): \`delta\` technique; \`Delta\` / \`DoubleDelta\` codec families in both recommenders, pruned by mean significant bytes, calibrated.  ` and update the "Agreed starting order" line ("3 done; next: …" — leave the next item for the user: write "next: to be chosen"). Items 15–16 untouched (`git diff ToDo.md`).
- [ ] Commit `"Docs: delta technique and Delta / DoubleDelta codecs (CLAUDE.md, ToDo 3)"`.

### Task 8: Verification

- [ ] `$PY scripts/check.py ci` → "all checks passed"; report the pytest count (baseline + the new tests).
- [ ] `git grep -n -i "gcd_min_saving\|gcdMinSaving" -- ':!docs/superpowers'` prints nothing.
- [ ] Report.
