# Recommender Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the one-shot recommender (`describe_and_recommend`, Describe / Recommend) and the `StreamingRecommender` emit one shared set of columns with the same names, types and rules, adding HyperLogLog and a bottom-k distinct sample so streaming estimates cardinality for every dtype.

**Architecture:** Two new self-contained Rust structures (`hll.rs`, `distinct_sample.rs`) feed streaming's `LevelStats`. One estimator rule (`cardinality_estimators::pick_estimate`) and one class rule (`conclusions.rs`) are computed in Rust for both recommenders, so C and Java callers see the same table as Python. The Python Describe base keeps reference versions of both rules for the Polars / DataFusion implementations, fed by private per-level inputs (`INPUTS`) that never reach the output. `population_rows`, Duj1, entropy, top-5 and the separate Chao1 / Schnabel columns are removed everywhere.

**Tech Stack:** Rust (polars 0.51, arrow-rs 60, rayon, foldhash 0.1), pyo3 0.25, Python 3.12 + Polars + pytest, Java 25 (Panama FFM, Arrow Java) + JUnit 5.

**Spec:** `docs/superpowers/specs/2026-10-01-recommender-parity-design.md`. Read it first; Task 0 amends it.

---

## Conventions for every task

Rust tests, from `services/analytics/` in Git Bash:

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics
export PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe
export PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH"
cargo test --lib <module>::
```

Without the `PATH` line, tests fail with `STATUS_DLL_NOT_FOUND`. Also run `cargo test --lib --no-default-features <module>::` whenever a task touches `api.rs` or `capi.rs`.

Python build and tests (Rust changes need the rebuild; Python-only changes do not):

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/<file> -v
```

`slow`-marked tests run with `-m slow`; run them where a step says so.

Java (Task 3, 8, 11):

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo build --release --no-default-features --target-dir target/capi
cd bindings/java && ./mvnw -q test
```

Commit messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
`docs/superpowers/` is git-ignored but tracked: add its files with `git add -f`.
New Rust items unused until a later task produce dead-code warnings; that is expected until Task 12's clippy run.

## File map

| File | Change |
|---|---|
| `src/hll.rs` (new) | `Hll` (Ertl's estimator), `hash_key` |
| `src/distinct_sample.rs` (new) | `DistinctSample`, `sample_size` |
| `src/conclusions.rs` (new) | `whole_range`, `classify`, `Conclusions`, `conclude` |
| `src/cardinality_estimators.rs` | drop Duj1 / `Exact`; `Count`, `pick_estimate`; methods observed / hll / schnabel / chao1 |
| `src/describe.rs` | `Frequencies` without entropy / top-5, with `first_few`, `all_once`, `hll`; `Profile` gains `min`, `max`, `numeric`; new field lists; conclusions in the row |
| `src/recommend.rs` | `Params` without `population_rows`; `render_value`; `Level` from conclusions and `first_few` |
| `src/partial.rs` | `Distinct` → `DistinctSample` + `Hll`; every dtype tracked; text / boolean / enum / binary extremes (`Key::S`) |
| `src/streaming.rs` | conclusions, shared value columns, `distinct_overflowed` removed |
| `src/api.rs`, `src/capi.rs`, `src/python.rs` | `population_rows` removed; `describe_columns(…, categorical_threshold)`; `render` |
| `analytics/base.py` | `INPUTS` (private per-row columns) |
| `analytics/_plugin.py` | `render`, signatures |
| `analytics/describe/{base,estimators,_values,polars,datafusion,rust}.py` | new contract |
| `analytics/recommend/{base,rust}.py` | population removed; conclusions passed through |
| `bindings/java/.../{Params,Analytics}.java` | `populationRows` removed |
| tests | per task |

---

### Task 0: Spec amendments found while planning

**Files:**
- Modify: `docs/superpowers/specs/2026-10-01-recommender-parity-design.md`

- [ ] **Step 1: Append §13 "Amendments (planning)"**

Append exactly:

```markdown
## 13. Amendments (planning, 2026-10-01)

1. **Rendering is arrow-rs's, everywhere.** Arrow C++ (pyarrow) and arrow-rs format some
   types differently (timestamps `2024-01-02 03:04:05` vs `2024-01-02T03:04:05`, floats,
   durations), so "Arrow's cast" is not one format. The canonical rendering is arrow-rs's cast
   to Utf8 (`recommend::render_value`). The Python reference implementations render through
   a private Rust helper (`_plugin.render`), not `pyarrow.compute.cast`. This is a format
   convention, not a computation, so the references stay independent where it matters.
2. **Ordering of Enum extremes** is by category order (its physical code), as one-shot does
   today — not by string value (§6 corrected). Categorical orders by string value.
3. **Nested extremes are dropped.** List / Array / Struct columns have no `min` / `max` in either
   recommender (one-shot used to order them by Polars' row encoding).
4. **Sampling-phase scaling.** f1, f2 and the capture history are the sample's values scaled by
   (HLL estimate ÷ sample size) — the sample's fractions applied to the HLL count — rather than by
   1/τ. Both estimate the same quantity; this one is consistent with the reported `n_unique`.
5. **`n_unique` floor in the sampling phase**: max(HLL estimate, k + 1). At least k + 1 distinct
   values have been seen, so the dictionary gate always rejects (k ≥ `categorical_threshold`).
6. **No `DistinctSample::merge`.** Streaming absorbs batches in order; `absorb` keeps the k smallest
   hashes whatever the batch order, which the tests check instead.
7. **`classify` lives in `src/conclusions.rs`** with `whole_range` and `conclude`, not in recommend.rs.
8. **Python contract.** Describe implementations return private per-level `INPUTS` (`argmin`,
   `argmax`, `f1`, `f2`, `capture_history` and `inner_` twins) beside `METRICS`; the base derives
   the conclusions (`min`, `max`, `unique`, `est_*`, `estimates_agree`, `class`) and drops the
   inputs from the output. `describe_columns` (private) returns Rust's conclusions *and* the inputs,
   so a test compares Rust's conclusions with the Python base's on identical inputs (exact).
   `describe_and_recommend` returns Rust's conclusions only; `RecommendRust` passes them through.
   Agreement with the reference compares `min`, `max`, `unique`, `class` exactly and
   `est_cardinality` / `est_low` / `est_high` within the Schnabel tolerance (10%); `est_method` and
   `estimates_agree` depend on the seeded split and are not compared.
9. **`projected_population_bytes` stays** in `rec_candidates`; without `population_rows` it is the
   predicted size scaled by the estimated cardinality only.
```

- [ ] **Step 2: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -f docs/superpowers/specs/2026-10-01-recommender-parity-design.md
git commit -m "docs: recommender parity spec amendments from planning

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: HyperLogLog module

**Files:**
- Create: `services/analytics/src/hll.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod hll;` in alphabetical order, after `mod gcd;`)

- [ ] **Step 1: Write the module with its tests, `estimate` stubbed**

Create `src/hll.rs`:

```rust
//! HyperLogLog distinct counting (Flajolet et al. 2007) with Ertl's improved estimator
//! ("New cardinality estimation algorithms for HyperLogLog sketches", 2017), which needs
//! no bias tables or range switches. Callers pass one well-mixed 64-bit hash per value
//! (`hash_key`); registers merge by max, so sketches of separate batches or threads
//! combine exactly. No Polars or Arrow types: a later entry point can wrap it as is.

use std::hash::BuildHasher;

/// A value key (`shared::encode_series`) spread over 64 bits. Integer keys are raw
/// values, so they are always mixed; foldhash's quality variant finishes with a full
/// avalanche. Not stable across foldhash versions: never persist a sketch.
#[inline]
pub(crate) fn hash_key(key: u64) -> u64 {
    foldhash::quality::FixedState::default().hash_one(key)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hll {
    p: u8,
    registers: Vec<u8>,
}

impl Hll {
    /// 2^p one-byte registers; p in 4..=18.
    pub(crate) fn new(p: u8) -> Self {
        assert!((4..=18).contains(&p), "HyperLogLog precision {p} outside 4..=18");
        Hll {
            p,
            registers: vec![0; 1 << p],
        }
    }

    /// Bucket = the top p bits; rank = leading zeros of the other 64 − p bits + 1.
    #[inline]
    pub(crate) fn insert(&mut self, hash: u64) {
        let i = (hash >> (64 - self.p)) as usize;
        let w = hash << self.p;
        let rank = if w == 0 {
            65 - self.p as u32
        } else {
            w.leading_zeros() + 1
        } as u8;
        let r = &mut self.registers[i];
        if rank > *r {
            *r = rank;
        }
    }

    /// Register-wise max: exactly the sketch of both inputs' values.
    pub(crate) fn merge(&mut self, other: &Hll) {
        assert_eq!(self.p, other.p, "HyperLogLog precisions differ");
        for (a, &b) in self.registers.iter_mut().zip(&other.registers) {
            *a = (*a).max(b);
        }
    }

    pub(crate) fn estimate(&self) -> f64 {
        todo!()
    }

    /// Relative standard error of `estimate`: 1.04 / √m.
    pub(crate) fn std_error(&self) -> f64 {
        1.04 / (self.registers.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(n: u64, seed: u64) -> Hll {
        let mut h = Hll::new(14);
        for v in 0..n {
            h.insert(hash_key(v + (seed << 40)));
        }
        h
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(Hll::new(14).estimate(), 0.0);
    }

    #[test]
    fn estimates_within_three_standard_errors() {
        for n in [1u64, 100, 10_000, 1_000_000] {
            for seed in 0..3 {
                let h = sketch(n, seed);
                let e = h.estimate();
                let tolerance = (3.0 * h.std_error() * n as f64).max(1.0);
                assert!((e - n as f64).abs() <= tolerance, "n={n} seed={seed} estimate={e}");
            }
        }
    }

    #[test]
    fn duplicates_do_not_count() {
        let mut h = Hll::new(14);
        for _ in 0..10 {
            for v in 0..1_000u64 {
                h.insert(hash_key(v));
            }
        }
        assert!((h.estimate() - 1_000.0).abs() < 30.0, "{}", h.estimate());
    }

    #[test]
    fn merge_equals_one_sketch() {
        let (mut a, mut b, mut all) = (Hll::new(12), Hll::new(12), Hll::new(12));
        for v in 0..5_000u64 {
            a.insert(hash_key(v));
            all.insert(hash_key(v));
        }
        for v in 3_000..9_000u64 {
            b.insert(hash_key(v));
            all.insert(hash_key(v));
        }
        a.merge(&b);
        assert_eq!(a, all);
    }

    #[test]
    #[should_panic(expected = "outside 4..=18")]
    fn precision_is_bounded() {
        Hll::new(3);
    }
}
```

Add `mod hll;` to `src/lib.rs` after `mod gcd;`.

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --lib hll::`
Expected: `empty_is_zero`, `estimates_within_three_standard_errors`, `duplicates_do_not_count` FAIL (panic: not yet implemented); `merge_equals_one_sketch`, `precision_is_bounded` pass.

- [ ] **Step 3: Implement Ertl's estimator**

Replace `estimate` and add `sigma` / `tau` below the `impl` block:

```rust
    /// Ertl's improved estimator: α∞·m² / z, with z built from the register histogram
    /// (σ corrects for empty registers, τ for saturated ones). 0 for an empty sketch.
    pub(crate) fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let q = 64 - self.p as usize;
        let mut c = vec![0u64; q + 2];
        for &r in &self.registers {
            c[r as usize] += 1;
        }
        if c[0] == self.registers.len() as u64 {
            return 0.0;
        }
        let mut z = m * tau(1.0 - c[q + 1] as f64 / m);
        for k in (1..=q).rev() {
            z = 0.5 * (z + c[k] as f64);
        }
        z += m * sigma(c[0] as f64 / m);
        m * m / (2.0 * std::f64::consts::LN_2) / z
    }
```

```rust
fn sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let (mut y, mut z) = (1.0, x);
    loop {
        x *= x;
        let previous = z;
        z += x * y;
        y += y;
        if z == previous {
            return z;
        }
    }
}

fn tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let (mut y, mut z) = (1.0, 1.0 - x);
    loop {
        x = x.sqrt();
        let previous = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if z == previous {
            return z / 3.0;
        }
    }
}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cargo test --lib hll::`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add services/analytics/src/hll.rs services/analytics/src/lib.rs
git commit -m "hll: HyperLogLog with Ertl's improved estimator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Bottom-k distinct sample

**Files:**
- Create: `services/analytics/src/distinct_sample.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod distinct_sample;` after `mod describe;`)
- Modify: `services/analytics/src/partial.rs:238-247` (make `KeyStat` and its fields `pub(crate)`)

- [ ] **Step 1: Expose `KeyStat`'s fields**

In `src/partial.rs`, change the `KeyStat` struct to:

```rust
/// A distinct value's statistics in one batch.
pub(crate) struct KeyStat {
    pub key: u64,
    pub first: u64,
    pub count: u64,
    pub mask: u8,
    pub len: u64,
    /// The value, when the batch holds at most five distinct values.
    pub text: Option<String>,
}
```

- [ ] **Step 2: Write the module with failing tests (`absorb` stubbed)**

Create `src/distinct_sample.rs`:

```rust
//! Bottom-k sample of distinct values (spec 2026-10-01 §3.2): the k values with the
//! smallest `hll::hash_key`, each with its count (capped at 3: enough for f1 / f2), its
//! capture mask and its length. A value is admitted on its first occurrence when its
//! hash is below the largest kept one; that bound only falls, so every value still
//! held has been counted since its first row and its count and mask are exact.
//!
//! Until the first eviction the sample holds every distinct value (the exact phase) and
//! also keeps what the dictionary rules need: `few`, `views` and `sum_len_unique`.

use std::collections::{BinaryHeap, HashMap};

use foldhash::fast::FixedState;

use crate::hll::hash_key;
use crate::partial::{KeyStat, ViewSim};

/// Sample size for a `categorical_threshold`: at least 1000, so the estimators have
/// data however low the threshold.
pub(crate) fn sample_size(threshold: u64) -> usize {
    threshold.max(1_000) as usize
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    /// Bits 0–1: count capped at 3; bits 2–4: capture mask.
    v: u8,
    len: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct DistinctSample {
    k: usize,
    map: HashMap<u64, Slot, FixedState>,
    /// (hash, key) of every held value; the largest hash on top.
    heap: BinaryHeap<(u64, u64)>,
    sampling: bool,
    /// Exact phase: every distinct value, first-occurrence order, while there are ≤ 5.
    pub few: Vec<String>,
    /// Exact phase: the distinct values' view blocks (a dictionary's Polars values).
    pub views: ViewSim,
    /// Exact phase: total byte length of the distinct values.
    pub sum_len_unique: u64,
}

impl DistinctSample {
    pub(crate) fn new(k: usize) -> Self {
        DistinctSample {
            k,
            map: HashMap::default(),
            heap: BinaryHeap::new(),
            sampling: false,
            few: Vec::new(),
            views: ViewSim::default(),
            sum_len_unique: 0,
        }
    }

    /// `keys`: one batch's distinct values, first-occurrence order.
    pub(crate) fn absorb(&mut self, keys: Vec<KeyStat>) {
        let _ = keys;
        todo!()
    }

    /// Whether the sample still holds every distinct value seen.
    pub(crate) fn is_exact(&self) -> bool {
        !self.sampling
    }

    /// Values held: every distinct value while exact, else k.
    pub(crate) fn len(&self) -> u64 {
        self.map.len() as u64
    }

    /// (f1, f2, capture history) over the held values.
    pub(crate) fn counts(&self) -> (u64, u64, [u64; 7]) {
        let (mut f1, mut f2, mut h) = (0, 0, [0u64; 7]);
        for s in self.map.values() {
            f1 += (s.v & 3 == 1) as u64;
            f2 += (s.v & 3 == 2) as u64;
            h[(s.v >> 2) as usize - 1] += 1;
        }
        (f1, f2, h)
    }

    /// Every held value occurred exactly once.
    pub(crate) fn all_once(&self) -> bool {
        self.map.values().all(|s| s.v & 3 == 1)
    }

    /// Mean byte length of the held values (0 when empty).
    pub(crate) fn mean_len(&self) -> f64 {
        if self.map.is_empty() {
            0.0
        } else {
            self.map.values().map(|s| s.len).sum::<u64>() as f64 / self.map.len() as f64
        }
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<u64> {
        let mut k: Vec<u64> = self.map.keys().copied().collect();
        k.sort_unstable();
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ks(key: u64, count: u64, mask: u8, text: Option<&str>) -> KeyStat {
        KeyStat {
            key,
            first: 0,
            count,
            mask,
            len: 20,
            text: text.map(str::to_owned),
        }
    }

    fn batch(keys: std::ops::Range<u64>) -> Vec<KeyStat> {
        keys.map(|k| ks(k, 1, 1, None)).collect()
    }

    #[test]
    fn exact_phase_counts_every_value() {
        let mut d = DistinctSample::new(1_000);
        d.absorb((0..100).map(|k| ks(k, k % 3 + 1, 1, None)).collect());
        assert!(d.is_exact());
        assert_eq!(d.len(), 100);
        let (f1, f2, h) = d.counts();
        assert_eq!((f1, f2, h[0]), (34, 33, 100));
        assert_eq!(d.sum_len_unique, 2_000);
    }

    #[test]
    fn counts_cap_at_three_and_masks_accumulate() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(vec![ks(1, 1, 1, None)]);
        d.absorb(vec![ks(1, 5, 2, None)]);
        assert_eq!(d.counts(), (0, 0, [0, 0, 1, 0, 0, 0, 0]));
        assert!(!d.all_once());
    }

    #[test]
    fn few_values_kept_while_exact() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(vec![ks(1, 1, 1, Some("a")), ks(2, 1, 1, Some("b"))]);
        assert_eq!(d.few, ["a", "b"]);
        d.absorb(batch(10..20));
        assert!(d.few.is_empty());
    }

    #[test]
    fn bounded_and_sampling_past_k() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..50_000));
        assert_eq!(d.len(), 1_000);
        assert!(!d.is_exact() && d.all_once());
        assert_eq!((d.sum_len_unique, d.views.bytes()), (0, 0));
        assert!(d.few.is_empty());
    }

    #[test]
    fn keeps_the_smallest_hashes_whatever_the_order() {
        let (mut ab, mut ba) = (DistinctSample::new(1_000), DistinctSample::new(1_000));
        ab.absorb(batch(0..30_000));
        ab.absorb(batch(20_000..60_000));
        ba.absorb(batch(20_000..60_000));
        ba.absorb(batch(0..30_000));
        assert_eq!(ab.keys(), ba.keys());
        assert_eq!(ab.counts(), ba.counts());
        let mut smallest: Vec<u64> = (0..60_000).collect();
        smallest.sort_unstable_by_key(|&k| hash_key(k));
        smallest.truncate(1_000);
        smallest.sort_unstable();
        assert_eq!(ab.keys(), smallest);
    }

    #[test]
    fn sampled_fractions_estimate_the_population() {
        // 100 000 values seen once, 100 000 seen twice: half the population are singletons.
        let mut d = DistinctSample::new(10_000);
        d.absorb((0..100_000).map(|k| ks(k, 1, 1, None)).collect());
        d.absorb((100_000..200_000).map(|k| ks(k, 2, 1, None)).collect());
        let (f1, f2, _) = d.counts();
        let n = d.len() as f64;
        assert!((f1 as f64 / n - 0.5).abs() < 0.03, "f1={f1}");
        assert!((f2 as f64 / n - 0.5).abs() < 0.03, "f2={f2}");
    }
}
```

Add `mod distinct_sample;` to `src/lib.rs` after `mod describe;`.

- [ ] **Step 3: Run the tests to see them fail**

Run: `cargo test --lib distinct_sample::`
Expected: 6 FAIL (panic: not yet implemented).

- [ ] **Step 4: Implement `absorb`**

Replace the stub with:

```rust
    /// `keys`: one batch's distinct values, first-occurrence order.
    pub(crate) fn absorb(&mut self, keys: Vec<KeyStat>) {
        for k in keys {
            if let Some(s) = self.map.get_mut(&k.key) {
                let count = ((s.v & 3) as u64 + k.count).min(3) as u8;
                s.v = count | (s.v & !3) | (k.mask << 2);
                continue;
            }
            let h = hash_key(k.key);
            if self.map.len() == self.k {
                self.enter_sampling();
                if self.heap.peek().is_some_and(|&(top, _)| h >= top) {
                    continue;
                }
                let (_, evicted) = self.heap.pop().expect("a full sample is not empty");
                self.map.remove(&evicted);
            }
            self.map.insert(
                k.key,
                Slot {
                    v: k.count.min(3) as u8 | (k.mask << 2),
                    len: k.len,
                },
            );
            self.heap.push((h, k.key));
            if !self.sampling {
                self.sum_len_unique += k.len;
                self.views.push(k.len);
                match k.text {
                    Some(t) if self.map.len() <= 5 => self.few.push(t),
                    _ => self.few.clear(),
                }
            }
        }
    }

    /// The first value not held ends the exact phase and its exact-only statistics.
    fn enter_sampling(&mut self) {
        if !self.sampling {
            self.sampling = true;
            self.few.clear();
            self.views = ViewSim::default();
            self.sum_len_unique = 0;
        }
    }
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cargo test --lib distinct_sample::`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add services/analytics/src/distinct_sample.rs services/analytics/src/lib.rs services/analytics/src/partial.rs
git commit -m "distinct_sample: bottom-k sample of distinct values with exact counts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Remove `population_rows` and Duj1 everywhere

The estimator becomes Schnabel → Chao1 only (Task 4 adds the ratio rule). Every binding loses the parameter.

**Files:**
- Modify: `src/cardinality_estimators.rs`, `src/recommend.rs`, `src/streaming.rs`, `src/partial.rs`, `src/api.rs`, `src/capi.rs`, `src/python.rs`
- Modify: `analytics/_plugin.py`, `analytics/describe/base.py`, `analytics/describe/estimators.py`, `analytics/recommend/rust.py`
- Modify: `bindings/java/src/main/java/io/github/benssutton/analytics/{Params,Analytics}.java`
- Test: `tests/test_describe.py`, `tests/test_recommend.py`, Rust unit tests in the files above

- [ ] **Step 1: Rust estimator without `q`**

In `src/cardinality_estimators.rs`:
- delete `duj1`, the `Method::Exact` and `Method::Duj1` variants and their `name` arms;
- replace `estimate` with:

```rust
/// The estimate picked by rule: Schnabel when valid, else Chao1.
pub(crate) fn estimate(d: u64, n: u64, f1: u64, f2: u64, history: &[u64; 7]) -> Estimate {
    match schnabel(history, d, n) {
        Some((s, lo, hi)) => Estimate {
            est_cardinality: s,
            est_low: Some(lo),
            est_high: Some(hi),
            method: Method::Schnabel,
        },
        None => {
            let (c, lo, hi) = chao1(d, f1, f2);
            Estimate {
                est_cardinality: c,
                est_low: Some(lo),
                est_high: Some(hi),
                method: Method::Chao1,
            }
        }
    }
}
```

- in its tests delete `duj1_cases`, and replace `estimate_picks_exact_then_duj1_then_schnabel_then_chao1` with:

```rust
    #[test]
    fn estimate_picks_schnabel_then_chao1() {
        let h = [0, 0, 0, 0, 0, 0, 10];
        assert_eq!(estimate(10, 40, 4, 2, &h).method, Method::Schnabel);
        let chao = estimate(10, 15, 4, 2, &h);
        assert_eq!((chao.method, chao.est_cardinality), (Method::Chao1, 12.0));
    }
```

- [ ] **Step 2: Rust callers**

- `src/recommend.rs` `Params`: delete the `population_rows` field and its line in the test helper `params()` (≈ line 2879).
- `src/recommend.rs` `level_estimate`: drop the `q` parameter and pass none: `estimate(p.freq.n_unique, n, p.freq.f1, p.freq.f2, &p.freq.capture_history)`.
- `src/recommend.rs` `Level::of_block`: `est: estimate(0, 0, 0, 0, &[0; 7]),`.
- `src/recommend.rs` `recommend`: delete the `q` and `r` bindings; pass `1.0` where `r` was passed to both `Level::of_values` calls; drop the `q` argument of both `level_estimate` calls; replace the `enum_values` binding and its use with `Some(pl_name(&t, name, None, &key)),`.
- `src/recommend.rs` tests:
  - `dictionary_evidence_names_the_estimator`: `level_estimate(&d.outer, 5)`; keep only the `chao` assertion (`let chao = evidence(); assert!(chao.contains("method=chao1 est_low="), "{chao}");`), with `evidence` taking no argument.
  - `polars_types_of_results`: delete the `exact` block (from `let exact = Params {` through its `assert_eq!`), keep the rest.
  - `rec_with(s, population_rows)` → `rec_with(s)` using `&params()`; in `polars_sizes_match_polars_view_layout` drop the `Option<u64>` from the case tuples (`Vec<(Series, &str, u64)>`, remove every `None,` second element) and loop `for (s, polars_type, polars_size) in cases { let r = rec_with(s); … }`.
- `src/partial.rs` `LevelStats::estimate`: `estimate(d.n_unique(), self.n - self.n_null, f1, f2, &h)`.
- `src/streaming.rs`: `let fallback = || estimate(0, 0, 0, 0, &[0; 7]);`; delete `population_rows: None,` from the test `params()` (≈ line 878).
- `src/api.rs`: delete the `population_rows` parameter of `describe_and_recommend` and its field; delete `population_rows: None,` in `StreamingRecommender::new`.
- `src/python.rs`: signature `#[pyo3(signature = (data, *, seed, zstd_level, categorical_threshold, boolean_pairs))]`, delete the parameter and its argument.
- `src/capi.rs`: delete `population_rows: i64` from both `describe_and_recommend` and `analytics_describe_and_recommend` (signatures and the inner call), delete the `u64::try_from` line, and fix the doc comments: remove "`population_rows < 0` means none." and change "no population, " in the test-helper comment (≈ line 321) to nothing. Update every capi test call of `analytics_describe_and_recommend` / `describe_and_recommend` to drop that argument (search `capi.rs` for the calls; each passes `-1` or a value in that position).

- [ ] **Step 3: Run Rust tests**

Run: `cargo test --lib && cargo test --lib --no-default-features`
Expected: all pass (≈ 315 tests). Fix any remaining compile error by deleting the population argument at that call site.

- [ ] **Step 4: Python**

- `analytics/_plugin.py` `describe_and_recommend`: delete the `population_rows` parameter and argument.
- `analytics/recommend/rust.py`: delete the `pop = …` lines, the `ValueError` block and `population_rows=pop,`.
- `analytics/describe/estimators.py`: delete `duj1`; `estimate(d, n, f1, f2, history)` without `q`:

```python
def estimate(d: int, n: int, f1: int, f2: int, history: Sequence[int]) -> dict:
    """Every estimate plus the one picked by rule: Schnabel valid → Schnabel; else Chao1."""
    c, c_lo, c_hi = chao1(d, f1, f2)
    sch = schnabel(history, d, n)
    s, s_lo, s_hi = sch if sch else (None, None, None)
    if sch:
        method, est, lo, hi = "schnabel", s, s_lo, s_hi
    else:
        method, est, lo, hi = "chao1", c, c_lo, c_hi
    return {
        "unique": d == n and n > 0,
        "chao1": c,
        "chao1_low": c_lo,
        "chao1_high": c_hi,
        "schnabel": s,
        "schnabel_low": s_lo,
        "schnabel_high": s_hi,
        "est_cardinality": est,
        "est_method": method,
        "est_low": lo,
        "est_high": hi,
        "estimates_agree": None if sch is None else (c_lo <= s_hi and s_lo <= c_hi),
    }
```

- `analytics/describe/base.py`:
  - `METHOD = pl.Enum(["schnabel", "chao1"])`;
  - delete the `population_rows` keyword, its validation, `self.population_rows` and `_population`; delete the docstring paragraph about it and the sentence "`population_rows` … (0 ≤ min, max ≤ 2N)." (keep the `seed` sentence);
  - `_conclusions`:

```python
    def _conclusions(self, r: dict) -> dict:
        s = self._collected[r["df_a"]][r["col_a"]]
        out = self._one_level(s, r, "", r["n_rows"])
        if r["inner_n_values"] is not None:
            out |= self._one_level(flatten(s), r, "inner_", r["inner_n_values"])
        return out
```

  - `_one_level(self, s, r, p, n_values)`: drop `q` and `big_n`; `est = estimators.estimate(r[f"{p}n_unique"], n, r[f"{p}f1"], r[f"{p}f2"], r[f"{p}capture_history"])`; `self._classify(s, r, p, n_values, n, est["est_cardinality"])`;
  - `_classify(self, s, r, p, n_values, n, est)`: drop `big_n`; the ordinal test becomes `0 <= bounds[0] and bounds[1] <= 2 * n_values`.

- [ ] **Step 5: Python tests**

`tests/test_describe.py`:
- delete `test_duj1` and `test_estimate_branches_follow_population_rows`;
- delete the `2N_uses_population` case from `CLASS_CASES` and the two `population_rows` cases from `test_constructor_validates`;
- replace `test_estimate_picks_exact_then_duj1_then_schnabel_then_chao1` with:

```python
def test_estimate_picks_schnabel_then_chao1():
    sch = estimators.estimate(10, 40, 4, 2, HISTORY_10)
    assert sch["est_method"] == "schnabel" and sch["est_cardinality"] == approx(
        9.523809523809524
    )
    assert sch["estimates_agree"] is True  # [10.25, 26.0] overlaps [6.47, 16.38]
    chao = estimators.estimate(10, 15, 4, 2, HISTORY_10)  # d/n ≥ 0.5 → Schnabel invalid
    assert chao["est_method"] == "chao1" and chao["est_cardinality"] == 12.0
    assert chao["schnabel"] is None and chao["estimates_agree"] is None
```

- in `test_estimates_disagree_under_heavy_skew` and `test_unique_flag` drop the `q=None` / trailing `None` argument.

`tests/test_recommend.py`:
- delete `test_population_rows_below_frame_rows_raise`;
- `KNOWN` case `enum_keeps_dictionary`: params `{}` and polars type `'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))'` (without an exact population the Enum name is no longer produced); update its comment to "Enum keeps its dictionary with 32-bit value offsets (the original has 64-bit)."
- `test_dictionary_polars_types`: delete the `exact` part, keep the `rec(s)` Categorical assertion;
- `test_dictionary_key_widths`: data `[f"v{i:05d}" for i in range(d)] * 2` for every `d`, call `rec(s, categorical_threshold=100_000)`. At d/n = 0.5 Schnabel is invalid and Chao1 with f1 = 0 is exactly d, so the key width follows d as the old exact population did (after Task 4, d/n ≥ 0.5 gives the observed count — also d);
- `test_dictionary_gate_rejects_above_threshold`: drop `population_rows=1_000`;
- `test_rust_cardinality_matches_python_estimators`: remove the parametrize and the argument; `run(impl(), frames)`.

- [ ] **Step 6: Java**

`Params.java`:

```java
package io.github.benssutton.analytics;

import java.util.List;

/**
 * Keyword parameters of {@code describe_and_recommend}; {@link #defaults()} matches the
 * Python technique ({@code analytics.recommend}).
 */
public record Params(long seed, int zstdLevel, long categoricalThreshold, List<BooleanPair> booleanPairs) {

    /** Two strings that together mark a string column as boolean, e.g. ("true", "false"). */
    public record BooleanPair(String trueValue, String falseValue) {}

    public Params {
        booleanPairs = List.copyOf(booleanPairs);
    }

    /** seed 0, ZSTD level 1, categorical threshold 10 000, ("true", "false"). */
    public static Params defaults() {
        return new Params(0, 1, 10_000, List.of(new BooleanPair("true", "false")));
    }
}
```

`Analytics.java`: delete the `JAVA_LONG,  // int64_t population_rows (< 0 = none)` line from the descriptor and the `params.populationRows().orElse(-1),` argument.

- [ ] **Step 7: Build and run everything**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -q
cd services/analytics && cargo build --release --no-default-features --target-dir target/capi && cd bindings/java && ./mvnw -q test
```

Expected: pytest all pass (some skips); JUnit 9 pass. If `test_dictionary_key_widths` picks a wider key, print the dictionary candidate's evidence: it must read `method=chao1` with `c=` equal to d; anything else means the data is not `* 2`.

- [ ] **Step 8: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "Remove population_rows and Duj1 from both recommenders and every binding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The estimator rule (ratio, floor, observed / HLL)

**Files:**
- Modify: `src/cardinality_estimators.rs`, `src/recommend.rs` (`level_estimate`, `Level::of_block`), `src/partial.rs` (`LevelStats::estimate`), `src/streaming.rs` (fallback)
- Modify: `analytics/describe/estimators.py`, `analytics/describe/base.py`
- Test: `src/cardinality_estimators.rs` tests, `tests/test_describe.py`, `tests/test_recommend.py`

- [ ] **Step 1: Write the failing Rust tests**

Replace the tests `estimate_picks_schnabel_then_chao1` and `overflowed_method_name` in `src/cardinality_estimators.rs` with:

```rust
    const H10: [u64; 7] = [0, 0, 0, 0, 0, 0, 10];

    #[test]
    fn no_values_is_observed_zero() {
        let (e, agree) = pick_estimate(Count::Exact(0), 0, 0, 0, &[0; 7]);
        assert_eq!((e.method, e.est_cardinality, e.est_low, e.est_high), (Method::Observed, 0.0, Some(0.0), Some(0.0)));
        assert_eq!(agree, None);
    }

    #[test]
    fn high_ratio_is_the_count() {
        let (e, _) = pick_estimate(Count::Exact(10), 15, 4, 2, &H10);
        assert_eq!((e.method, e.est_cardinality, e.est_low, e.est_high), (Method::Observed, 10.0, Some(10.0), Some(10.0)));
        let (h, agree) = pick_estimate(Count::Hll { estimate: 1_000.0, std_error: 0.01 }, 1_500, 0, 0, &[0; 7]);
        assert_eq!((h.method, h.est_cardinality, agree), (Method::Hll, 1_000.0, None));
        assert!((h.est_low.unwrap() - 970.0).abs() < 1e-9 && (h.est_high.unwrap() - 1_030.0).abs() < 1e-9);
    }

    #[test]
    fn low_ratio_is_schnabel_floored_at_the_count() {
        // Schnabel 9.52 [6.47, 16.38] < d = 10: the estimate and low end are floored at 10.
        let (e, agree) = pick_estimate(Count::Exact(10), 40, 4, 2, &H10);
        assert_eq!((e.method, e.est_cardinality, e.est_low), (Method::Schnabel, 10.0, Some(10.0)));
        assert!((e.est_high.unwrap() - 16.378255262343956).abs() < 1e-9);
        assert_eq!(agree, Some(true));
    }

    #[test]
    fn low_ratio_without_recaptures_is_chao1() {
        let h = [3, 3, 0, 3, 0, 0, 0];
        let (e, agree) = pick_estimate(Count::Exact(9), 100, 4, 2, &h);
        let (c, lo, hi) = chao1(9, 4, 2);
        assert_eq!((e.method, e.est_cardinality, e.est_low, e.est_high), (Method::Chao1, c, Some(lo.max(9.0)), Some(hi)));
        assert_eq!(agree, None);
    }

    #[test]
    fn hll_floor_is_three_standard_errors_below() {
        let (e, _) = pick_estimate(Count::Hll { estimate: 100.0, std_error: 0.01 }, 1_000, 90, 5, &[0; 7]);
        assert_eq!(e.method, Method::Chao1);
        assert!(e.est_cardinality >= 100.0 && e.est_low.unwrap() >= 97.0 - 1e-9);
    }

    #[test]
    fn method_names() {
        let names: Vec<_> = [Method::Observed, Method::Hll, Method::Schnabel, Method::Chao1, Method::Overflowed]
            .iter().map(|m| m.name()).collect();
        assert_eq!(names, ["observed", "hll", "schnabel", "chao1", "overflowed"]);
    }
```

Run: `cargo test --lib cardinality_estimators::`
Expected: compile errors (`Count`, `pick_estimate`, `Method::Observed`, `Method::Hll` not found).

- [ ] **Step 2: Implement**

In `src/cardinality_estimators.rs`, replace the `Method` enum and its `name`, and replace `estimate` with `Count` + `pick_estimate` (keep `chao1`, `schnabel`, `Estimate` unchanged):

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Method {
    /// The exact distinct count (cardinality ratio ≥ 0.5, or nothing to estimate).
    Observed,
    /// The HyperLogLog count (streaming's sampling phase, ratio ≥ 0.5).
    Hll,
    Schnabel,
    Chao1,
    /// Streaming: distinct tracking stopped past `categorical_threshold`; removed in Task 9.
    Overflowed,
}

impl Method {
    /// Lower-case name, as `est_method` and the dictionary candidate's evidence report it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Method::Observed => "observed",
            Method::Hll => "hll",
            Method::Schnabel => "schnabel",
            Method::Chao1 => "chao1",
            Method::Overflowed => "overflowed",
        }
    }
}

/// A level's distinct count: exact, or a HyperLogLog estimate with its relative
/// standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Count {
    Exact(u64),
    Hll { estimate: f64, std_error: f64 },
}

/// The estimate picked by rule (spec 2026-10-01 §4), with `estimates_agree`.
/// n = non-null values; d = the count. d/n ≥ 0.5 → the count itself (observed, or
/// HLL ± 3σ). Below: Schnabel when valid, else Chao1, floored at d (the estimate) and
/// at d − 3σ for HLL (the low end): observed values bound the population from below.
pub(crate) fn pick_estimate(
    count: Count,
    n: u64,
    f1: u64,
    f2: u64,
    history: &[u64; 7],
) -> (Estimate, Option<bool>) {
    let (d, floor, high, method) = match count {
        Count::Exact(d) => (d as f64, d as f64, d as f64, Method::Observed),
        Count::Hll { estimate, std_error } => (
            estimate,
            estimate * (1.0 - 3.0 * std_error),
            estimate * (1.0 + 3.0 * std_error),
            Method::Hll,
        ),
    };
    if n == 0 {
        let zero = Estimate {
            est_cardinality: 0.0,
            est_low: Some(0.0),
            est_high: Some(0.0),
            method: Method::Observed,
        };
        return (zero, None);
    }
    if d / n as f64 >= 0.5 {
        let e = Estimate {
            est_cardinality: d,
            est_low: Some(floor),
            est_high: Some(high),
            method,
        };
        return (e, None);
    }
    let du = d.round() as u64;
    let (c, c_lo, c_hi) = chao1(du, f1, f2);
    let sch = schnabel(history, du, n);
    let (e, lo, hi, method) = match sch {
        Some((s, s_lo, s_hi)) => (s, s_lo, s_hi, Method::Schnabel),
        None => (c, c_lo, c_hi, Method::Chao1),
    };
    let est = e.max(d);
    (
        Estimate {
            est_cardinality: est,
            est_low: Some(lo.max(floor)),
            est_high: Some(hi.max(est)),
            method,
        },
        sch.map(|(_, s_lo, s_hi)| c_lo <= s_hi && s_lo <= c_hi),
    )
}
```

Update the callers to `pick_estimate(...).0` with `Count::Exact`:
- `recommend.rs` `level_estimate`: `pick_estimate(Count::Exact(p.freq.n_unique), n, p.freq.f1, p.freq.f2, &p.freq.capture_history).0`
- `recommend.rs` `Level::of_block`: `est: pick_estimate(Count::Exact(0), 0, 0, 0, &[0; 7]).0,`
- `partial.rs` `LevelStats::estimate` (non-overflowed branch): `pick_estimate(Count::Exact(d.n_unique()), self.n - self.n_null, f1, f2, &h).0`
- `streaming.rs`: `let fallback = || pick_estimate(Count::Exact(0), 0, 0, 0, &[0; 7]).0;`

Fix the imports (`use crate::cardinality_estimators::{pick_estimate, Count, Estimate, Method};` as each file needs).

- [ ] **Step 3: Run Rust tests**

Run: `cargo test --lib`
Expected: `cardinality_estimators::` passes. Fix other failures:
- `recommend::tests::dictionary_evidence_names_the_estimator` — the toy column ["a","b","a","b","c"] has d/n = 0.6 → now `method=observed`; assert `chao.contains("method=observed est_low=3")` and rename the binding to `evidence`.
- any other test asserting `chao1` / `schnabel` on a column with d/n ≥ 0.5: assert `observed` with est = d (the new rule's meaning). List each changed assertion in the commit message body.

- [ ] **Step 4: Python estimator rule**

Replace `estimate` in `analytics/describe/estimators.py`:

```python
def estimate(d: int, n: int, f1: int, f2: int, history: Sequence[int]) -> dict:
    """The estimate picked by rule (spec 2026-10-01 §4) for an exact count d of n
    non-null values: d/n ≥ 0.5 → d itself ("observed"); below, Schnabel when valid,
    else Chao1, floored at d. Same rule as cardinality_estimators::pick_estimate."""
    if n == 0:
        return _picked(False, 0.0, "observed", 0.0, 0.0, None)
    if d / n >= 0.5:
        return _picked(d == n, float(d), "observed", float(d), float(d), None)
    c, c_lo, c_hi = chao1(d, f1, f2)
    sch = schnabel(history, d, n)
    e, lo, hi, method = (*sch, "schnabel") if sch else (c, c_lo, c_hi, "chao1")
    est = max(e, d)
    agree = None if sch is None else (c_lo <= sch[2] and sch[1] <= c_hi)
    return _picked(False, est, method, max(lo, d), max(hi, est), agree)


def _picked(unique, est, method, lo, hi, agree) -> dict:
    return {
        "unique": unique,
        "est_cardinality": float(est),
        "est_method": method,
        "est_low": float(lo),
        "est_high": float(hi),
        "estimates_agree": agree,
    }
```

In `analytics/describe/base.py`:
- `METHOD = pl.Enum(["observed", "hll", "schnabel", "chao1"])` (Task 7 turns it into String);
- `ESTIMATES` becomes:

```python
ESTIMATES = {
    "unique": pl.Boolean,
    "est_cardinality": F64,
    "est_method": METHOD,
    "est_low": F64,
    "est_high": F64,
    "estimates_agree": pl.Boolean,
}
```

- in `agreement`, replace the `schnabel_cols` block with:

```python
        estimate_cols = [
            f"{p}{k}"
            for p in ("", "inner_")
            for k in ("est_cardinality", "est_low", "est_high")
        ]
        problems += metric_mismatches(
            as_float(result, estimate_cols),
            as_float(reference, estimate_cols),
            keys,
            estimate_cols,
            SCHNABEL_RTOL,
            0.0,
        )
```

- [ ] **Step 5: Python tests**

`tests/test_describe.py`:
- replace `test_estimate_picks_schnabel_then_chao1` with:

```python
def test_estimate_rule():
    assert estimators.estimate(0, 0, 0, 0, [0] * 7)["est_method"] == "observed"
    high = estimators.estimate(10, 15, 4, 2, HISTORY_10)  # d/n ≥ 0.5
    assert (high["est_method"], high["est_cardinality"], high["est_high"]) == (
        "observed",
        10.0,
        10.0,
    )
    sch = estimators.estimate(10, 40, 4, 2, HISTORY_10)  # Schnabel 9.52 floored at 10
    assert (sch["est_method"], sch["est_cardinality"], sch["est_low"]) == (
        "schnabel",
        10.0,
        10.0,
    )
    assert sch["est_high"] == approx(16.378255262343956)
    assert sch["estimates_agree"] is True
    chao = estimators.estimate(9, 100, 4, 2, [3, 3, 0, 3, 0, 0, 0])  # no recapture
    c, lo, hi = estimators.chao1(9, 4, 2)
    assert (chao["est_method"], chao["est_cardinality"], chao["est_high"]) == (
        "chao1",
        c,
        hi,
    )
    assert chao["est_low"] == max(lo, 9) and chao["estimates_agree"] is None
```

- `test_unique_flag` stays; `test_estimates_disagree_under_heavy_skew` stays.
- `test_classification` case `over_threshold`: params `{"categorical_threshold": 3}` (the estimate is now the observed 4, not Chao1's 10).
- `test_zero_row_column_conclusions`: `r["est_method"] == "observed"`.
- delete every remaining reference to `chao1` / `schnabel` conclusion columns (`r["chao1"]`, `r["schnabel"]`, …): search `grep -n '\["chao1\|\["schnabel\|chao1_low\|schnabel_low' tests/test_describe.py` and replace each with the equivalent `est_*` assertion or delete it when it only checked the separate column.

`tests/test_recommend.py`:
- replace `test_population_projection_chooses_plain_over_dictionary` with an invariant that does not depend on which side wins (d/n = 1 000 / 3 000 → Chao1 or Schnabel extrapolates):

```python
def test_projection_scales_the_dictionary_by_the_estimate():
    ids = [f"id-{i:027d}" for i in range(1_000)]
    s = pl.Series("x", ids[:600] + ids[600:800] * 2 + ids[800:] * 10)
    r = rec(s)
    dictionary = next(
        c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary")
    )
    assert r["est_method"] in ("schnabel", "chao1") and r["est_cardinality"] > 1_000
    assert dictionary["projected_population_bytes"] > dictionary["predicted_bytes"]
```

- [ ] **Step 6: Build and run**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_describe.py tests/test_recommend.py tests/test_streaming_recommend.py -q
```

Expected: all pass. A streaming parity failure on a dictionary candidate's evidence means one-shot and streaming now disagree on the method name for a text column — both call `pick_estimate`, so check the streaming caller passes `Count::Exact`.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "Estimator rule: observed count at ratio >= 0.5, else Schnabel/Chao1 floored at it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `class` and the whole-number range in Rust

**Files:**
- Create: `services/analytics/src/conclusions.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod conclusions;` after `mod chi_squared;`)

- [ ] **Step 1: Write the module with failing tests (`classify` stubbed)**

```rust
//! Describe's conclusions (spec 2026-10-01 §4, §5), shared by one-shot and streaming:
//! the whole-number range behind `ordinal` and the class rule. analytics/describe/base.py
//! keeps the reference implementation.

use polars::prelude::DataType as PT;

use crate::cardinality_estimators::Count;
use crate::describe::{FloatStats, StringStats};

/// (min, max) when every non-null value is a whole number: integers, Decimal with
/// scale 0, floats with no fraction / NaN / infinity, and strings that are all integers
/// without leading zeros. `numeric`: the level's numeric extremes (integer, decimal and
/// float dtypes; None otherwise). Temporal dtypes never qualify.
pub(crate) fn whole_range(
    dtype: &PT,
    n: u64,
    numeric: Option<(f64, f64)>,
    floats: Option<&FloatStats>,
    strings: Option<&StringStats>,
) -> Option<(f64, f64)> {
    match dtype {
        dt if dt.is_integer() => numeric,
        PT::Decimal(_, Some(0)) => numeric,
        PT::Float32 | PT::Float64 => floats
            .filter(|f| f.n_fractional == 0 && f.n_nan == 0 && f.n_inf == 0)
            .and(numeric),
        PT::String | PT::Categorical(..) | PT::Enum(..) => {
            let st = strings?;
            if st.n_numeric_int != n || st.n_leading_zero != 0 || st.int_overflow {
                return None;
            }
            Some((st.int_min? as f64, st.int_max? as f64))
        }
        _ => None,
    }
}

/// First match wins: null → constant → boolean → ordinal → categorical → discrete.
/// `n_values`: values at this level, nulls included.
pub(crate) fn classify(
    n_values: u64,
    n_null: u64,
    count: Count,
    whole: Option<(f64, f64)>,
    est: f64,
    threshold: u64,
) -> &'static str {
    let _ = (n_values, n_null, count, whole, est, threshold);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_in_order() {
        let c = |nv, nn, count, whole, est| classify(nv, nn, count, whole, est, 10);
        assert_eq!(c(4, 4, Count::Exact(0), None, 0.0), "null");
        assert_eq!(c(0, 0, Count::Exact(0), None, 0.0), "null");
        assert_eq!(c(4, 1, Count::Exact(1), Some((7.0, 7.0)), 1.0), "constant");
        assert_eq!(c(4, 0, Count::Exact(2), Some((1.0, 5.0)), 2.0), "boolean");
        assert_eq!(c(5, 0, Count::Exact(5), Some((0.0, 4.0)), 5.0), "ordinal");
        assert_eq!(c(4, 0, Count::Exact(4), Some((0.0, 9.0)), 4.0), "categorical"); // 9 > 2·4
        assert_eq!(c(4, 0, Count::Exact(4), Some((-3.0, 2.0)), 4.0), "categorical");
        assert_eq!(c(40, 0, Count::Exact(20), None, 20.0), "discrete");
        let hll = Count::Hll { estimate: 2.0, std_error: 0.01 };
        assert_eq!(c(4, 0, hll, None, 2.0), "categorical"); // HLL is never boolean
    }

    #[test]
    fn whole_ranges() {
        let r = Some((0.0, 3.0));
        assert_eq!(whole_range(&PT::Int64, 4, r, None, None), r);
        assert_eq!(whole_range(&PT::Decimal(Some(10), Some(0)), 4, r, None, None), r);
        assert_eq!(whole_range(&PT::Decimal(Some(10), Some(2)), 4, r, None, None), None);
        assert_eq!(whole_range(&PT::Date, 4, r, None, None), None);
        let whole = FloatStats { n_fractional: 0, ..Default::default() };
        let frac = FloatStats { n_fractional: 1, ..Default::default() };
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&whole), None), r);
        assert_eq!(whole_range(&PT::Float64, 4, r, Some(&frac), None), None);
        let ints = StringStats {
            n_numeric_int: 4,
            int_min: Some(0),
            int_max: Some(3),
            ..Default::default()
        };
        assert_eq!(whole_range(&PT::String, 4, None, None, Some(&ints)), r);
        let zero = StringStats { n_leading_zero: 1, ..ints.clone() };
        assert_eq!(whole_range(&PT::String, 4, None, None, Some(&zero)), None);
    }
}
```

If `FloatStats` / `StringStats` lack `Default` or `Clone`, add `#[derive(Default)]` / `Clone` to them in `describe.rs` (they are plain counters). If their counter fields are private, make the ones read here `pub(crate)`.

Run: `cargo test --lib conclusions::`
Expected: `whole_ranges` passes, `classes_in_order` FAILS (not yet implemented).

- [ ] **Step 2: Implement `classify`**

```rust
pub(crate) fn classify(
    n_values: u64,
    n_null: u64,
    count: Count,
    whole: Option<(f64, f64)>,
    est: f64,
    threshold: u64,
) -> &'static str {
    if n_null == n_values {
        return "null";
    }
    match count {
        Count::Exact(1) => return "constant",
        Count::Exact(2) => return "boolean",
        _ => {}
    }
    if whole.is_some_and(|(lo, hi)| 0.0 <= lo && hi <= 2.0 * n_values as f64) {
        return "ordinal";
    }
    if est <= threshold as f64 {
        "categorical"
    } else {
        "discrete"
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib conclusions::`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add services/analytics/src/conclusions.rs services/analytics/src/lib.rs services/analytics/src/describe.rs
git commit -m "conclusions: class rule and whole-number range in Rust

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: One rendering function (`render_value`, `_plugin.render`)

**Files:**
- Modify: `src/recommend.rs` (next to `fn render`, ≈ line 1488), `src/streaming.rs` (`fn render`, ≈ line 373), `src/api.rs`, `src/python.rs`, `analytics/_plugin.py`
- Test: `src/api.rs` tests, `tests/test_describe.py`

- [ ] **Step 1: Write the failing Rust test**

In `src/recommend.rs` tests add:

```rust
    #[test]
    fn render_value_is_arrow_rs_text() {
        use arrow_array::{BooleanArray, Date32Array, Decimal128Array, Float64Array, TimestampMicrosecondArray};
        let one = |a: ArrayRef| render_value(a.as_ref());
        assert_eq!(one(Arc::new(Float64Array::from(vec![2.5]))).as_deref(), Some("2.5"));
        assert_eq!(one(Arc::new(BooleanArray::from(vec![false]))).as_deref(), Some("false"));
        assert_eq!(one(Arc::new(Date32Array::from(vec![19_724]))).as_deref(), Some("2024-01-02"));
        let ts = TimestampMicrosecondArray::from(vec![1_704_164_645_000_000]);
        assert_eq!(one(Arc::new(ts)).as_deref(), Some("2024-01-02T03:04:05"));
        let dec = Decimal128Array::from(vec![150]).with_precision_and_scale(10, 2).unwrap();
        assert_eq!(one(Arc::new(dec)).as_deref(), Some("1.50"));
        assert_eq!(one(Arc::new(Float64Array::from(vec![None]))), None);
    }
```

Run: `cargo test --lib recommend::tests::render_value_is_arrow_rs_text`
Expected: compile error (`render_value` not found).

- [ ] **Step 2: Implement `render_value` and use it in streaming**

In `src/recommend.rs`, above `fn render`:

```rust
/// Row 0 of `a` as text: arrow-rs's cast to Utf8, the one rendering of `min` / `max`
/// in every recommender (spec 2026-10-01 §6, §13.1). None for a null, or a value with
/// no text form (e.g. non-UTF-8 binary).
pub(crate) fn render_value(a: &dyn Array) -> Option<String> {
    let s = arrow_cast(a.slice(0, 1).as_ref(), &AT::Utf8).ok()?;
    let s = s.as_string::<i32>();
    s.is_valid(0).then(|| s.value(0).to_string())
}
```

(`Array::slice` returns an `ArrayRef`; `arrow_cast` takes `&dyn Array`.)

In `src/streaming.rs`, replace the body of `fn render(e: &Option<Ext>)` with:

```rust
fn render(e: &Option<Ext>) -> AnyValue<'static> {
    e.as_ref()
        .and_then(|e| render_value(e.value.as_ref()))
        .map_or(AnyValue::Null, |s| AnyValue::StringOwned(s.into()))
}
```

- [ ] **Step 3: Add `api::render`, the pyo3 function and `_plugin.render`**

`src/api.rs`, after `column_sizes`:

```rust
/// Every value of every column as text (`recommend::render_value`): the canonical
/// rendering of `min` / `max`, for the Python reference implementations.
pub fn render(batch: &RecordBatch) -> Result<RecordBatch> {
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            Arc::new(arrow_array::StringArray::from_iter(
                (0..c.len()).map(|i| crate::recommend::render_value(c.slice(i, 1).as_ref())),
            )) as ArrayRef
        })
        .collect();
    let fields: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| arrow_schema::Field::new(f.name(), arrow_schema::DataType::Utf8, true))
        .collect();
    RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)
        .map_err(|e| Error::Compute(e.to_string()))
}
```

(Add `use std::sync::Arc;` / `ArrayRef` imports if absent.) Add a test in `api.rs`'s tests: a batch with an `Int64Array [3, null]` column → `render` gives `["3", null]` and the field is Utf8.

`src/python.rs`, beside `column_sizes`:

```rust
#[pyfunction]
fn render(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::render(&batch)).map(ArrowTable)
}
```

and register it with the module's other functions (`m.add_function(wrap_pyfunction!(render, m)?)?;`).

`analytics/_plugin.py`, after `column_sizes`:

```python
def render(s: pl.Series) -> list[str | None]:
    """`s`'s values as text, rendered by arrow-rs's cast to Utf8 — the one format of
    Describe's / Recommend's min and max (recommend.rs `render_value`)."""
    return pl.DataFrame(_rs.render(s.to_frame())).to_series().to_list()
```

- [ ] **Step 4: Python known answers for rendering**

In `tests/test_describe.py`, section 3, add:

```python
@pytest.mark.parametrize(
    "s, expected",
    [
        (pl.Series([2.5, 1.0]), ["2.5", "1.0"]),
        (pl.Series([False, True]), ["false", "true"]),
        (pl.Series([datetime(2024, 1, 2, 3, 4, 5)]), ["2024-01-02T03:04:05"]),
        (pl.Series([datetime(2024, 1, 2)]).dt.date(), ["2024-01-02"]),
        (pl.Series([Decimal("1.50")], dtype=pl.Decimal(10, 2)), ["1.50"]),
        (pl.Series(["b", "a"], dtype=pl.Categorical), ["b", "a"]),
        (pl.Series([1, None]), ["1", None]),
    ],
)
def test_render_is_arrow_rs_text(s, expected):
    from analytics import _plugin

    assert _plugin.render(s) == expected
```

- [ ] **Step 5: Build and run**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo test --lib recommend:: api:: && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_describe.py -k render -v && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_streaming_recommend.py -q
```

Expected: all pass. A literal in Step 1 or Step 4 that differs is arrow-rs's actual Display output: confirm it with `cargo test` output and use arrow-rs's text (it is the oracle by §13.1); note the change in the commit body.

- [ ] **Step 6: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "render_value: one arrow-rs rendering of extremes, exposed to Python

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The one-shot contract (Rust Describe output, Python Describe base, Recommend pass-through)

This is the largest task: Rust's describe output and Python's Describe contract change together, since `DescribeRust` and `RecommendRust` read Rust's columns by name.

**Files:**
- Modify: `src/describe.rs`, `src/recommend.rs`, `src/conclusions.rs`, `src/api.rs`, `src/python.rs`, `src/streaming.rs` (compile fixes only)
- Modify: `analytics/base.py`, `analytics/_plugin.py`, `analytics/describe/{base,_values,polars,datafusion,rust}.py`, `analytics/recommend/rust.py`
- Test: `src/describe.rs` tests, `tests/test_describe.py`, `tests/test_recommend.py`

**Target Rust column lists** (value block, in this order; the same block is used by streaming in Task 11):

```text
n_unique U64, unique Boolean, est_cardinality F64, est_low F64, est_high F64, est_method String,
estimates_agree Boolean, class String, min String, max String, min_len U64, max_len U64,
gcd Decimal(38,0), sum_len U64, sum_len_unique U64,
n_nan U64, n_inf U64, n_fractional U64, max_frac_digits U32, n_f32_inexact U64,
n_numeric U64, n_numeric_int U64, n_leading_zero U64, numeric_int_min D38, numeric_int_max D38,
numeric_max_int_digits U32, numeric_max_frac_digits U32, numeric_min_frac_digits U32,
numeric_max_sig_digits U32, n_iso_date U64, n_iso_time U64, n_iso_datetime U64,
n_iso_datetime_tz U64, iso_max_frac_digits U32, iso_max_sig_frac_digits U32, iso_n_offsets U64,
iso_n_midnight U64
```

`fields()` (public, describe_and_recommend): `column, n_rows, n_null, <value block>, n_midnight, inner_n_values, inner_n_null, inner_<value block>`.
`input_fields()` (private, describe_columns only, appended): `argmin U64, argmax U64, f1 U64, f2 U64, capture_history List(U64)`, then the same five with `inner_`.

- [ ] **Step 1: `Frequencies` and `Profile` (Rust)**

In `src/describe.rs`:

1. `Frequencies`: delete `entropy`, `top5_idx`, `top5_count`; add

```rust
    /// First rows of the distinct values, first-occurrence order, while there are ≤ 5
    /// (Recommend's boolean-pair rule reads them; streaming keeps the same list).
    pub first_few: Vec<u64>,
    /// Every distinct value occurred once (streaming's sampling phase: every sampled one).
    pub all_once: bool,
    /// Streaming's sampling phase: (HyperLogLog estimate, relative standard error); None
    /// while `n_unique` is exact.
    pub hll: Option<(f64, f64)>,
```

2. `frequencies()`: drop the entropy and top-5 code; build `first_few`:

```rust
    let mut first_few: Vec<u64> = if map.len() <= 5 {
        map.values().map(|e| e.first).collect()
    } else {
        Vec::new()
    };
    first_few.sort_unstable();
```

(before consuming the map), set `all_once: f1 == n_unique`, `hll: None`. Remove the now-unused `n_null` / `nf` code.

3. `Profile`: add

```rust
    /// Rendered extremes (`recommend::render_value`); None for nested dtypes.
    pub min: Option<String>,
    pub max: Option<String>,
    /// Numeric extremes as f64 (integer, decimal and float dtypes), for `ordinal`.
    pub numeric: Option<(f64, f64)>,
```

and fill them in `profile()`:

```rust
    let range = range(s, lengths.as_deref())?;
    let nested = matches!(s.dtype(), DataType::List(_) | DataType::Array(..) | DataType::Struct(_));
    let (min, max) = if nested {
        (None, None)
    } else {
        (render_at(s, range.argmin)?, render_at(s, range.argmax)?)
    };
    let numeric = numeric_extremes(s, range.argmin, range.argmax)?;
```

with helpers:

```rust
fn render_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<String>> {
    let Some(i) = i else { return Ok(None) };
    let one = crate::sizes::classic_layout(&s.slice(i as i64, 1))?;
    Ok(crate::recommend::render_value(one.as_ref()))
}

fn numeric_extremes(s: &Series, lo: Option<u64>, hi: Option<u64>) -> PolarsResult<Option<(f64, f64)>> {
    let dt = s.dtype();
    if !(dt.is_integer() || dt.is_float() || matches!(dt, DataType::Decimal(..))) {
        return Ok(None);
    }
    let at = |i: Option<u64>| -> PolarsResult<Option<f64>> {
        match i {
            None => Ok(None),
            Some(i) => Ok(s.slice(i as i64, 1).cast(&DataType::Float64)?.f64()?.get(0)),
        }
    };
    Ok(at(lo)?.zip(at(hi)?))
}
```

4. Rust compile fixes in other files:
- `recommend.rs` `Level::of_values`: `few_distinct` from `first_few`:

```rust
        let few_distinct = if is_text(dtype) && p.freq.n_unique <= 5 {
            p.freq
                .first_few
                .iter()
                .map(|&i| text_of(&values.slice(i as usize, 1)).map(|t| t.value(0).to_string()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
```

- `partial.rs` `LevelStats::profile`: the `Frequencies` literal drops `entropy`, `top5_idx`, `top5_count`, adds `first_few: Vec::new()`, `all_once: false`, `hll: None`; the `Profile` literal adds `min: None, max: None, numeric: None` (Task 9 fills them).

- [ ] **Step 2: `conclude` (Rust)**

Append to `src/conclusions.rs`:

```rust
use crate::cardinality_estimators::{pick_estimate, Estimate};
use crate::describe::Profile;

/// One level's conclusions: the picked estimate, `estimates_agree`, `unique`, `class`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conclusions {
    pub est: Estimate,
    pub agree: Option<bool>,
    pub unique: bool,
    pub class: &'static str,
}

/// `n_values`: values at this level, nulls included.
pub(crate) fn conclude(dtype: &PT, n_values: u64, n_null: u64, p: &Profile, threshold: u64) -> Conclusions {
    let n = n_values - n_null;
    let f = &p.freq;
    let count = match f.hll {
        Some((estimate, std_error)) => Count::Hll { estimate, std_error },
        None => Count::Exact(f.n_unique),
    };
    let (est, agree) = pick_estimate(count, n, f.f1, f.f2, &f.capture_history);
    let unique = n > 0
        && match count {
            Count::Exact(d) => d == n,
            Count::Hll { .. } => f.all_once,
        };
    let whole = whole_range(dtype, n, p.numeric, p.floats.as_ref(), p.strings.as_ref());
    Conclusions {
        est,
        agree,
        unique,
        class: classify(n_values, n_null, count, whole, est.est_cardinality, threshold),
    }
}
```

Add a test:

```rust
    #[test]
    fn conclude_on_a_profile() {
        use polars::prelude::*;
        let s = Series::new("x".into(), &[0i64, 4, 1, 2, 3]);
        let p = crate::describe::profile(&s, 0, false).unwrap();
        let c = conclude(s.dtype(), 5, 0, &p, 10_000);
        assert_eq!((c.class, c.unique, c.est.method.name()), ("ordinal", true, "observed"));
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("0"), Some("4")));
    }
```

- [ ] **Step 3: Rows and field lists (Rust)**

In `src/describe.rs`:
- `value_fields()` → the target value block above (types: `U64`, `DataType::Boolean`, `F64`, `DataType::String`, `d38`, `U32`).
- new `input_fields()`:

```rust
/// Private estimator inputs, appended by `describe_columns` only (Python's reference
/// conclusions read them; spec 2026-10-01 §13.8).
pub(crate) fn input_fields() -> Vec<(String, DataType)> {
    let level = [
        ("argmin", DataType::UInt64),
        ("argmax", DataType::UInt64),
        ("f1", DataType::UInt64),
        ("f2", DataType::UInt64),
        ("capture_history", DataType::List(Box::new(DataType::UInt64))),
    ];
    level
        .iter()
        .map(|(n, d)| (n.to_string(), d.clone()))
        .chain(level.iter().map(|(n, d)| (format!("inner_{n}"), d.clone())))
        .collect()
}
```

- `Profile::row(&self, c: &Conclusions) -> Row` in the value-block order:

```rust
    pub(crate) fn row(&self, c: &Conclusions) -> Row {
        let f = &self.freq;
        let text = |s: &Option<String>| s.clone().map_or(AnyValue::Null, |s| AnyValue::StringOwned(s.into()));
        let mut row: Row = vec![
            AnyValue::UInt64(f.n_unique),
            AnyValue::Boolean(c.unique),
            AnyValue::Float64(c.est.est_cardinality),
            c.est.est_low.map_or(AnyValue::Null, AnyValue::Float64),
            c.est.est_high.map_or(AnyValue::Null, AnyValue::Float64),
            AnyValue::StringOwned(c.est.method.name().into()),
            c.agree.map_or(AnyValue::Null, AnyValue::Boolean),
            AnyValue::StringOwned(c.class.into()),
            text(&self.min),
            text(&self.max),
            u64v(self.range.min_len),
            u64v(self.range.max_len),
            d38v(self.gcd),
            u64v(self.sum_len),
            u64v(f.sum_len_unique),
        ];
        // … the existing float (5) and string (17) blocks, unchanged …
        row
    }

    /// `input_fields()` values for this level.
    pub(crate) fn input_row(&self) -> Row {
        vec![
            u64v(self.range.argmin),
            u64v(self.range.argmax),
            AnyValue::UInt64(self.freq.f1),
            AnyValue::UInt64(self.freq.f2),
            listv(&self.freq.capture_history),
        ]
    }
```

(`Profile::row` becomes `pub(crate)`; streaming uses it in Task 11.)

- `Described`: add `pub dtype: DataType` (set in `describe_one` from `s.dtype().clone()`), and:

```rust
    /// The column's and its inner values' conclusions.
    pub(crate) fn conclusions(&self, threshold: u64) -> (Conclusions, Option<Conclusions>) {
        let outer = conclude(&self.dtype, self.n_rows, self.n_null, &self.outer, threshold);
        let inner = self.inner.as_ref().map(|i| {
            conclude(
                i.values.dtype(),
                i.values.len() as u64,
                i.values.null_count() as u64,
                &i.profile,
                threshold,
            )
        });
        (outer, inner)
    }

    /// One `fields()` row.
    pub(crate) fn row(&self, threshold: u64) -> Row {
        let (oc, ic) = self.conclusions(threshold);
        let mut row: Row = vec![
            AnyValue::StringOwned(self.name.clone()),
            AnyValue::UInt64(self.n_rows),
            AnyValue::UInt64(self.n_null),
        ];
        row.extend(self.outer.row(&oc));
        row.push(u64v(self.n_midnight));
        match (&self.inner, ic) {
            (Some(i), Some(ic)) => {
                row.push(AnyValue::UInt64(i.values.len() as u64));
                row.push(AnyValue::UInt64(i.values.null_count() as u64));
                row.extend(i.profile.row(&ic));
            }
            _ => row.extend(nulls(2 + value_fields().len())),
        }
        row
    }

    /// One `input_fields()` row.
    pub(crate) fn input_row(&self) -> Row {
        let mut row = self.outer.input_row();
        match &self.inner {
            Some(i) => row.extend(i.profile.input_row()),
            None => row.extend(nulls(5)),
        }
        row
    }
```

- `describe_columns_impl(inputs, seed, threshold)`: rows `d.row(threshold)` extended with `d.input_row()`; schema `fields()` + `input_fields()`.
- `api::describe_columns(batch, seed, categorical_threshold)`, `python.rs` `describe_columns(data, seed, categorical_threshold)`, `_plugin.describe_columns(df, seed, categorical_threshold)` — pass it through.
- `recommend.rs` `recommend()`: replace both `level_estimate(...)` arguments with the conclusions' estimates: at the top `let (oc, ic) = d.conclusions(params.categorical_threshold);`, pass `oc.est` for the outer level and `ic.map_or(oc.est, |c| c.est)` for the inner one. Delete `level_estimate` and fix its test users (`dictionary_evidence_names_the_estimator`: use `d.conclusions(10_000).0.est`).
- `recommend.rs` `describe_and_recommend_impl`: `let mut row = d.row(params.categorical_threshold);`.
- describe.rs tests: replace `counts_entropy_and_top5` with a test of `n_unique`, `f1`, `f2` and `first_few` on `["a","a","b",None]` (expect `(2, 1, 1, vec![0, 2])`); delete assertions on `entropy` / `top5` in the other tests (`z.entropy` → assert `z.n_unique == 0`).

Run: `cargo test --lib`
Expected: all pass, except `recommend::tests::output_matches_declared_schema` and other schema tests that compare against literal field lists — update those literals to the new lists.

- [ ] **Step 4: Python `INPUTS` in the technique base**

`analytics/base.py`:
- class attribute, after `CONCLUSIONS`:

```python
    # Private per-row values an implementation returns for the base's conclusions;
    # never in the result (e.g. Describe's argmin / f1 / capture_history).
    INPUTS: ClassVar[dict[str, pl.DataType]] = {}
```

- `result()`: `columns = [*keys, "status", *self.METRICS, *self.INPUTS]` (the final `select` stays without `INPUTS`).
- `metrics_frame`:

```python
        return cls.keys_frame(combos).with_columns(
            pl.Series("status", statuses, dtype=STATUS),
            *(
                _metric_series(name, metrics[name], dtype)
                for name, dtype in cls.METRICS.items()
            ),
            *(
                _metric_series(name, metrics.get(name, [None] * len(combos)), dtype)
                for name, dtype in cls.INPUTS.items()
            ),
        )
```

- [ ] **Step 5: Python Describe base**

`analytics/describe/base.py`, the column sets:

```python
GROUP_A = {  # whole values — every eligible dtype
    "n_unique": U64,
    "min_len": U64,
    "max_len": U64,
    "gcd": D38,
    "sum_len": U64,
    "sum_len_unique": U64,
}
# GROUP_B, GROUP_C, VALUE_METRICS, SIZE_METRICS, METRICS: unchanged definitions
LEVEL_INPUTS = {
    "argmin": U64,
    "argmax": U64,
    "f1": U64,
    "f2": U64,
    "capture_history": LU64,
}
INPUTS = {**LEVEL_INPUTS, **{f"inner_{k}": v for k, v in LEVEL_INPUTS.items()}}

ESTIMATES = {
    "unique": pl.Boolean,
    "est_cardinality": F64,
    "est_method": pl.String,
    "est_low": F64,
    "est_high": F64,
    "estimates_agree": pl.Boolean,
}
_LEVEL = {"min": pl.String, "max": pl.String, **ESTIMATES, "class": pl.String}
CONCLUSIONS = {**_LEVEL, **{f"inner_{k}": v for k, v in _LEVEL.items()}}
EXACT_CONCLUSIONS = [c for c in CONCLUSIONS if c.removeprefix("inner_") in ("min", "max", "unique", "class")]
ESTIMATE_CONCLUSIONS = [
    c for c in CONCLUSIONS if c.removeprefix("inner_") in ("est_cardinality", "est_low", "est_high")
]

TOLERANCES = {"size_zstd_bytes": 0.01, "size_polars_zstd_bytes": 0.01}
SCHNABEL_RTOL = 0.10
_NESTED = (pl.List, pl.Array, pl.Struct)
```

Delete `CLASS`, `METHOD`, `TOP5`, `_RENDERED`, `SPLIT_DEPENDENT`.

In the `Describe` class:
- `INPUTS = INPUTS`;
- docstring: replace "Min, max and top-5 values are reported as row indices…" in the module docstring with: "Implementations fill METRICS and the private INPUTS (first-occurrence argmin / argmax, f1, f2, capture history); the base renders min / max (arrow-rs text, `_plugin.render`), picks the estimate and classifies each column. Rust-backed implementations may supply the conclusions themselves (`_supplied`).";
- `__init__`: add `self._supplied: dict[tuple[str, str], dict] = {}`;
- add

```python
    def _on_result_end(self) -> None:
        self._supplied = {}
```

- `_conclusions`:

```python
    def _conclusions(self, r: dict) -> dict:
        supplied = self._supplied.get((r["df_a"], r["col_a"]))
        if supplied is not None:
            return supplied
        s = self._collected[r["df_a"]][r["col_a"]]
        out = self._one_level(s, r, "", r["n_rows"])
        if r["inner_n_values"] is not None:
            out |= self._one_level(flatten(s), r, "inner_", r["inner_n_values"])
        return out

    def _one_level(self, s: pl.Series, r: dict, p: str, n_values: int) -> dict:
        n = n_values - r[f"{p}n_null"]
        est = estimators.estimate(
            r[f"{p}n_unique"], n, r[f"{p}f1"], r[f"{p}f2"], r[f"{p}capture_history"]
        )
        lo, hi = _render(s, r[f"{p}argmin"], r[f"{p}argmax"])
        return {
            f"{p}min": lo,
            f"{p}max": hi,
            **{f"{p}{k}": v for k, v in est.items()},
            f"{p}class": self._classify(s, r, p, n_values, n, est["est_cardinality"]),
        }
```

- module-level helper:

```python
def _render(s: pl.Series, lo: int | None, hi: int | None) -> tuple[str | None, str | None]:
    """min / max as arrow-rs text (spec 2026-10-01 §13.1); None for nested dtypes."""
    if lo is None or isinstance(s.dtype, _NESTED):
        return None, None
    from analytics import _plugin

    lo_text, hi_text = _plugin.render(s.gather([lo, hi]))
    return lo_text, hi_text
```

- `agreement`:

```python
    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        keys = self.key_columns()
        exact = [m for m in self.METRICS if m not in TOLERANCES] + EXACT_CONCLUSIONS
        problems = metric_mismatches(result, reference, keys, exact, 0.0, 0.0)

        def as_float(df, cols):
            return df.with_columns(pl.col(c).cast(F64) for c in cols)

        for metric, rtol in TOLERANCES.items():
            problems += metric_mismatches(
                as_float(result, [metric]), as_float(reference, [metric]), keys, [metric], rtol, 0.0
            )
        problems += metric_mismatches(
            result, reference, keys, ESTIMATE_CONCLUSIONS, SCHNABEL_RTOL, 0.0
        )
        return list(dict.fromkeys(problems))
```

- [ ] **Step 6: Python implementations**

`analytics/describe/_values.py` `frequency_summary`:

```python
def frequency_summary(count, mask) -> dict:
    """n_unique, f1, f2 and the capture history from a frequency table of distinct
    non-null values: their counts and OR-ed split masks (numpy arrays)."""
    count = np.asarray(count, dtype=np.int64)
    return {
        "n_unique": len(count),
        "f1": int((count == 1).sum()),
        "f2": int((count == 2).sum()),
        "capture_history": np.bincount(np.asarray(mask, dtype=np.int64), minlength=8)[
            1:8
        ].tolist(),
    }
```

and update its docstring line in the module / the comment at `_values.py:27` ("what `inner_argmin` indexes into").

`analytics/describe/polars.py`:
- `profile()`: `summary = frequency_summary(freq["count"].to_numpy(), freq["mask"].to_numpy())`;
- `_compute`: `{m: [r[m] for r in rows] for m in {**self.METRICS, **self.INPUTS}}`;
- `_row`: the inner fallback `dict.fromkeys({**VALUE_METRICS, **LEVEL_INPUTS})` (import `LEVEL_INPUTS`).

`analytics/describe/datafusion.py`: the same three changes (`_frequencies` calls `frequency_summary(count, mask)`; the class docstring line "entropy / f1 / f2 / top-5 / capture history" → "f1 / f2 / capture history"; the `_compute` dict comprehension over `{**self.METRICS, **self.INPUTS}`; the inner fallback keys).

`analytics/describe/rust.py`:

```python
class DescribeRust(Describe):
    """Rust extension: `describe_columns` (one pass per column; rayon across columns and
    64K-row chunks) and `column_sizes`. Rust also returns its own conclusions; this class
    lets the base derive them from the same inputs, which tests compare."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            stats = _plugin.describe_columns(
                df, self.seed, self.categorical_threshold
            ).join(_plugin.column_sizes(df, self.zstd_level), on="column")
            for r in stats.iter_rows(named=True):
                rows[frame, r["column"]] = r
        wanted = {**self.METRICS, **self.INPUTS}
        return self.metrics_frame(
            combos, {m: [rows[k[0]][m] for k in combos] for m in wanted}
        )
```

`analytics/recommend/rust.py`: after computing `rows`, supply the conclusions:

```python
        for key, r in rows.items():
            self._supplied[key] = {c: r[c] for c in self.CONCLUSIONS}
        return self.metrics_frame(
            combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS}
        )
```

- [ ] **Step 7: Python tests**

`tests/test_describe.py`:
- `DEFAULTS`: delete `entropy`, `top5_idx`, `top5_count`;
- `conclude()`: `metrics = {m: None for m in {**Describe.METRICS, **Describe.INPUTS}} | DEFAULTS | overrides`;
- `test_constructor_validates`: `with_metrics(Describe, **{m: [None] for m in Describe.METRICS})` stays;
- `test_renders_min_max_and_top5_from_indices` → rename `test_renders_min_max_from_indices`, keep the min / max assertions (now arrow-rs text), delete the top-5 ones;
- `test_renders_inner_values_from_flattened_indices`: delete `inner_top5` assertions;
- `test_zero_row_column_conclusions`: drop the `entropy`, `top5_idx`, `top5_count` overrides and the `r["top5"] == []` check;
- `test_frequencies_entropy_and_top5` → `test_frequencies`: assert `(n_rows, n_null, n_unique)` and, through `profile`, the conclusions `min == "a"`, `max == "b"`;
- delete `test_top5_ties_break_by_first_occurrence`;
- `test_extremes_are_first_occurrences` and `test_enum_orders_by_category_and_categorical_by_string`, `test_nested_ordering_with_null_elements`, `test_list_whole_and_inner_values`, `test_nested_floats_and_enums`: replace `argmin` / `argmax` index checks by the rendered `min` / `max` text of the value at that index; for List / Array / Struct columns assert `min is None and max is None`.
- `test_agreement_tolerances`: adapt to the new `agreement` (delete the entropy / schnabel-column cases; add one where `est_cardinality` differs by 5% → no problem, by 20% → a problem).
- Any remaining hit of `grep -nE 'entropy|top5|chao1_|schnabel_' tests/test_describe.py` outside the estimator unit tests: delete it (the column is gone).

`tests/test_recommend.py`: no change expected beyond what Task 3–4 did; `assert_contract` checks the new schema.

- [ ] **Step 8: Build and run**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo test --lib && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -q && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_describe.py -m slow -q
```

Expected: all pass. If the slow large-dataset agreement reports a `class` mismatch (`categorical` vs `discrete`), it is an `est_cardinality` within 10% straddling `categorical_threshold` — compare the two rows' `est_method`; if both are Schnabel from different splits, exclude `class` from `EXACT_CONCLUSIONS` only for rows where `est_method == "schnabel"` and document it in the spec §13.8.

- [ ] **Step 9: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "One-shot contract: Rust conclusions and rendered extremes; entropy, top-5 and argmin dropped

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Rust conclusions equal the Python base's; Java one-shot checks

**Files:**
- Test: `tests/test_recommend.py`, `tests/test_describe.py`
- Test: `services/analytics/bindings/java/src/test/java/io/github/benssutton/analytics/AnalyticsTest.java`

- [ ] **Step 1: Write the Python tests**

`tests/test_describe.py`, section 2:

```python
def test_rust_conclusions_equal_the_reference_rule():
    """describe_columns returns Rust's conclusions and the inputs; the Python base
    derives its own from those inputs (DescribeRust). Same inputs, same rule."""
    from analytics import _plugin
    from analytics.describe import CONCLUSIONS, DescribeRust

    for frame in [describe_mixed(2_000), stringified(describe_mixed(500))]:
        raw = _plugin.describe_columns(frame, 0, 10_000)
        derived = DescribeRust().add({"t": frame}).result()
        for c in CONCLUSIONS:
            assert raw[c].to_list() == derived[c].to_list(), c
```

Make sure `analytics.describe` exports `CONCLUSIONS` (add it to `analytics/describe/__init__.py`'s imports from `analytics.describe.base` if absent).

`tests/test_recommend.py`:

```python
def test_rust_conclusions_match_describe():
    from analytics.describe import CONCLUSIONS, DescribeRust

    frame = describe_mixed(2_000)
    a = run(impl(), {"t": frame})
    b = DescribeRust().add({"t": frame}).result()
    for c in CONCLUSIONS:
        assert a[c].to_list() == b[c].to_list(), c
```

- [ ] **Step 2: Run them**

Run: `cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_describe.py tests/test_recommend.py -k "conclusions" -v`
Expected: PASS. A mismatch is a real disagreement between `conclusions.rs` / `pick_estimate` and `estimators.py` / `_classify`: fix the side that departs from the spec.

- [ ] **Step 3: Java one-shot checks**

In `AnalyticsTest.describesAndRecommendsToyData`, after the `rec_arrow_type` assertion:

```java
            assertEquals(List.of("categorical", "boolean"), strings(result, "class"));
            assertEquals(List.of("0", "x"), strings(result, "min"));
            assertEquals(List.of("7", "y"), strings(result, "max"));
            assertEquals(List.of("observed", "observed"), strings(result, "est_method"));
```

(`a = [0, 5, 7]`: 7 > 2·3, so not ordinal; `s` has two values, so boolean.)

Run:

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo build --release --no-default-features --target-dir target/capi && cd bindings/java && ./mvnw -q test
```

Expected: 9 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A tests services/analytics/analytics services/analytics/bindings
git commit -m "Tests: Rust conclusions equal the reference rule; Java sees class, min, est_method

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Streaming counts with HLL and the distinct sample

**Files:**
- Modify: `src/partial.rs` (`BatchStats::of`, `LevelStats`, `profile`, remove `Distinct`, `estimate`, `overflowed`)
- Modify: `src/streaming.rs` (`batch_stats`, `view_shape`, `row`)
- Modify: `src/cardinality_estimators.rs` (remove `Method::Overflowed`)
- Test: `src/partial.rs` tests, `src/streaming.rs` tests, `tests/test_streaming_recommend.py`

- [ ] **Step 1: Write the failing Rust tests**

In `src/partial.rs` tests add:

```rust
    #[test]
    fn every_dtype_is_counted() {
        let s = Series::new("i".into(), (0..2_000i64).map(|i| i % 300).collect::<Vec<_>>());
        let st = absorbed(&chunks(&s, 700), 10_000);
        let p = st.profile(s.dtype());
        assert_eq!((p.freq.n_unique, p.freq.hll), (300, None));
        assert_eq!((p.min.as_deref(), p.max.as_deref(), p.numeric), (Some("0"), Some("299"), Some((0.0, 299.0))));
    }

    #[test]
    fn past_k_the_count_is_hll() {
        let s = Series::new("i".into(), (0..50_000i64).collect::<Vec<_>>());
        let st = absorbed(&chunks(&s, 8_192), 1_000);
        let p = st.profile(s.dtype());
        let (e, se) = p.freq.hll.expect("sampling phase");
        assert!((e - 50_000.0).abs() <= 3.0 * se * 50_000.0, "{e}");
        assert!(p.freq.n_unique >= 1_001 && p.freq.all_once);
        assert_eq!(p.freq.sum_len_unique, None); // not a text level
    }
```

Change the test helper `absorbed(parts, track, threshold)` to `absorbed(parts, threshold)` (no `track`), and update its callers (drop the boolean argument). Update `summary()` to print `st.sample` instead of `st.distinct`:

```rust
        let d = st.sample.as_ref().map(|d| {
            (d.len(), d.counts(), d.sum_len_unique, d.few.clone(), d.views.bytes(), d.is_exact())
        });
```

Run: `cargo test --lib partial::`
Expected: compile errors (`profile` takes a dtype, no `sample` field).

- [ ] **Step 2: Implement in `partial.rs`**

1. Delete `Distinct` and its `impl` (the counting moves to `DistinctSample`).
2. `BatchStats::of(s, offset, seed)` — drop `track`; always build `keys`; capture text only for text dtypes:

```rust
        let map = frequency_map(&encode_series(s)?, seed, offset);
        let text = if map.len() <= 5 && crate::recommend::is_text(s.dtype()) {
            Some(s.cast(&DataType::String)?)
        } else {
            None
        };
        // … build `keys` exactly as before, sorted by first …
        let keys = Some(keys);
```

3. `LevelStats`: replace `pub distinct: Option<Distinct>` with

```rust
    /// Bottom-k sample of the distinct values (every eligible dtype).
    pub sample: Option<DistinctSample>,
    /// HyperLogLog of the distinct values (p = 14).
    pub hll: Option<Hll>,
```

4. `absorb`:

```rust
        if let Some(keys) = b.keys {
            let h = self.hll.get_or_insert_with(|| Hll::new(14));
            keys.iter().for_each(|k| h.insert(hash_key(k.key)));
            self.sample
                .get_or_insert_with(|| DistinctSample::new(sample_size(threshold)))
                .absorb(keys);
        }
```

5. Delete `overflowed()` and `estimate()`.
6. `profile(&self, dtype: &DataType) -> Profile` (the threshold now only sizes the sample, in `absorb`):

```rust
    pub(crate) fn profile(&self, dtype: &DataType) -> Profile {
        let text_len = self.sum_len.is_some();
        let (n_unique, hll, f1, f2, history, all_once, sum_len_unique) = match &self.sample {
            None => (0, None, 0, 0, [0; 7], false, None),
            Some(d) if d.is_exact() => {
                let (f1, f2, h) = d.counts();
                let unique = text_len.then_some(d.sum_len_unique);
                (d.len(), None, f1, f2, h, d.all_once(), unique)
            }
            Some(d) => {
                let sketch = self.hll.as_ref().expect("a sampled level has a sketch");
                let e = sketch.estimate().max((d.len() + 1) as f64);
                let scale = e / d.len() as f64;
                let sc = |x: u64| (x as f64 * scale).round() as u64;
                let (f1, f2, h) = d.counts();
                let unique = text_len.then(|| (d.mean_len() * e).round() as u64);
                (e.round() as u64, Some((e, sketch.std_error())), sc(f1), sc(f2), h.map(sc), d.all_once(), unique)
            }
        };
        let numeric_dtype = dtype.is_integer() || dtype.is_float() || matches!(dtype, DataType::Decimal(..));
        let numeric = if numeric_dtype {
            self.int_range()
                .map(|(a, b)| (a as f64, b as f64))
                .or_else(|| self.float_range())
        } else {
            None
        };
        Profile {
            freq: Frequencies {
                n_unique,
                f1,
                f2,
                capture_history: history,
                sum_len_unique,
                first_few: Vec::new(),
                all_once,
                hll,
            },
            range: Range { argmin: None, argmax: None, min_len: self.min_len, max_len: self.max_len },
            floats: self.floats,
            strings: self.strings.clone(),
            gcd: self.gcd,
            sum_len: self.sum_len,
            is_f32: self.is_f32,
            min: self.lo.as_ref().and_then(|e| render_value(e.value.as_ref())),
            max: self.hi.as_ref().and_then(|e| render_value(e.value.as_ref())),
            numeric,
        }
    }
```

Note on Decimal: `int_range` holds the unscaled value; `whole_range` only reads `numeric` for scale-0 decimals, where unscaled = value.

7. `few_distinct`:

```rust
    pub(crate) fn few_distinct(&self) -> Vec<String> {
        self.sample
            .as_ref()
            .filter(|d| d.is_exact() && d.len() <= 5)
            .map_or_else(Vec::new, |d| d.few.clone())
    }
```

Imports: `use crate::distinct_sample::{sample_size, DistinctSample}; use crate::hll::{hash_key, Hll}; use crate::recommend::render_value;`; drop `estimate, Estimate, Method` if unused.

- [ ] **Step 3: `streaming.rs`**

- `batch_stats`: delete `track`; `BatchStats::of(s, self.n_rows, seed)` and `BatchStats::of(&v, prev.map_or(0, |l| l.n), seed)`.
- `view_shape`: `distinct: st.sample.as_ref().filter(|d| d.is_exact()).map_or(0, |d| d.views.bytes()),`.
- `row`: compute conclusions and feed their estimates to the levels:

```rust
        let o = &c.outer;
        let p = o.profile(dtype);
        let oc = conclude(dtype, o.n, o.n_null, &p, t);
        let classic = o.classic.clone().ok_or("a typed column has absorbed no batch")?;
        let mut lvl = level(dtype, &classic, &p, o, oc.est, "");
        let inner = match (&c.inner, dtype) {
            (Some(i), PT::List(it) | PT::Array(it, _)) => Some((i, &**it)),
            _ => None,
        };
        let ip = inner.map(|(i, it)| i.profile(it));
        let ilvl = match (inner, &ip) {
            (Some((i, it)), Some(ip)) => {
                let iclassic = i.classic.clone().ok_or("an inner level has absorbed no batch")?;
                let ic = conclude(it, i.n, i.n_null, ip, t);
                let mut l = level(it, &iclassic, ip, i, ic.est, "inner: ");
                // … unchanged …
            }
            _ => None,
        };
```

Delete `fallback`. In `level()`, restrict `int_range` to the dtypes the rules read it for (Task 10 gives Boolean / Enum integer keys):

```rust
        int_range: (dtype.is_integer()
            || matches!(dtype, PT::Decimal(..) | PT::Date | PT::Datetime(..) | PT::Duration(_) | PT::Time))
            .then(|| st.int_range())
            .flatten(),
```

- The output columns stay as they are in this task; fill them from the new state:

```rust
        let exact = o.sample.as_ref().is_none_or(|d| d.is_exact());
        row.extend([
            render(&o.lo),
            render(&o.hi),
            p.gcd.map_or(AnyValue::Null, |g| AnyValue::Decimal(g, 0)),
            u(p.sum_len),
            u(o.min_len),
            u(o.max_len),
            AnyValue::UInt64(p.freq.n_unique),
            AnyValue::Boolean(!exact),
            f(Some(oc.est.est_cardinality)),
            f(oc.est.est_low),
            f(oc.est.est_high),
            text(oc.est.method.name()),
            // … the four size values, unchanged …
        ]);
```

- streaming tests: `overflow_rejects_the_dictionary` — `est_method` is now `"hll"` and `n_unique` is the HLL count (≥ threshold + 1), `distinct_overflowed` true. Update those two assertions.

- [ ] **Step 4: Remove `Method::Overflowed`**

Delete the variant and its `name` arm in `cardinality_estimators.rs`; drop it from the `method_names` test.

- [ ] **Step 5: Run Rust tests**

Run: `cargo test --lib && cargo test --lib --no-default-features`
Expected: all pass.

- [ ] **Step 6: Python streaming tests**

`tests/test_streaming_recommend.py`:
- `test_overflow_rejects_the_dictionary`: assert `est_method == "hll"` (was `"overflowed"`) and `n_unique >= categorical_threshold + 1` (was null).
- In `assert_parity`, `overflowed = bool(r["distinct_overflowed"])` stays for now (Task 11 replaces it).

Build and run:

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_streaming_recommend.py -q && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_streaming_recommend.py -m slow -q
```

Expected: all pass. Parity of a dictionary candidate's evidence on a text column in the exact phase must hold unchanged (same exact count, same rule); a mismatch on a non-text column's candidates means an estimate leaked into a non-dictionary rule's evidence — compare the two evidence strings.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "streaming: count every dtype with HyperLogLog and a bottom-k distinct sample

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Streaming extremes for text, boolean, enum and binary

**Files:**
- Modify: `src/partial.rs` (`Key`, `has_extremes`, `ext_at`, `int_range`, `float_range`, `absorb`)
- Test: `src/partial.rs` tests

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn text_boolean_and_enum_extremes() {
        let s = Series::new("s".into(), &[Some("m"), None, Some("b"), Some("z"), Some("c")]);
        let st = absorbed(&chunks(&s, 2), 10_000);
        let p = st.profile(s.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("b"), Some("z")));
        let b = Series::new("b".into(), &[true, true, false]);
        let p = absorbed(&chunks(&b, 1), 10_000).profile(b.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("false"), Some("true")));
        let cat = s.cast(&DataType::from_categories(Categories::global())).unwrap();
        let p = absorbed(&chunks(&cat, 2), 10_000).profile(cat.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("b"), Some("z")));
    }
```

(If `DataType::from_categories` / `Categories::global()` is spelled differently in polars 0.51, build the Categorical the way the existing partial.rs or streaming.rs tests do — search `Categorical` in `src/streaming.rs` tests.)

Run: `cargo test --lib partial::tests::text_boolean_and_enum_extremes`
Expected: FAIL (min / max are None).

- [ ] **Step 2: Implement**

1. `Key`:

```rust
/// Ordering key of an extreme: the physical integer (integers, decimals, temporals,
/// booleans, Enum codes), the float, or the bytes (strings, Categorical values, binary).
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub(crate) enum Key {
    I(i128),
    F(f64),
    S(Vec<u8>),
}
```

2. `has_extremes` adds `| DataType::String | DataType::Categorical(..) | DataType::Enum(..) | DataType::Boolean | DataType::Binary` to the `matches!`.
3. `ext_at` key:

```rust
    let key = match one.dtype() {
        DataType::String | DataType::Categorical(..) => one
            .cast(&DataType::String)?
            .str()?
            .get(0)
            .map(|v| Key::S(v.as_bytes().to_vec())),
        DataType::Binary => one.binary()?.get(0).map(|v| Key::S(v.to_vec())),
        DataType::Boolean => one.bool()?.get(0).map(|v| Key::I(v as i128)),
        dt if dt.is_float() => one.cast(&DataType::Float64)?.f64()?.get(0).map(Key::F),
        _ => one
            .to_physical_repr()
            .cast(&DataType::Int128)?
            .i128()?
            .get(0)
            .map(Key::I),
    };
```

4. `int_range` / `float_range` match by reference:

```rust
    pub(crate) fn int_range(&self) -> Option<(i128, i128)> {
        match (&self.lo.as_ref()?.key, &self.hi.as_ref()?.key) {
            (Key::I(a), Key::I(b)) => Some((*a, *b)),
            _ => None,
        }
    }
```

(and the same shape for `float_range` with `Key::F`). Fix any other `.key` copy the compiler flags (`Key` is no longer `Copy`) with `.clone()` or a reference.

5. `profile()`'s `numeric` already filters by dtype, so Boolean / Enum integer keys never reach `ordinal`; Task 9's `level()` filter keeps them out of the rules.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib`
Expected: all pass, including streaming's parity unit tests.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add services/analytics/src/partial.rs
git commit -m "streaming: min/max for text, categorical, enum, boolean and binary columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Streaming output = the shared columns; parity tests; Java streaming checks

**Files:**
- Modify: `src/streaming.rs` (`output_fields`, `row`)
- Test: `src/streaming.rs` tests, `tests/test_streaming_recommend.py`, `bindings/java/.../StreamingRecommenderTest.java`

- [ ] **Step 1: Output fields**

```rust
/// Output columns (spec 2026-10-01 §7), in order: the streaming head, Describe's value
/// block (describe.rs `value_fields`), n_midnight, the size columns, Recommend's, the sample.
fn output_fields() -> Vec<(String, PT)> {
    let mut f: Vec<(String, PT)> = [
        ("column", PT::String),
        ("status", PT::String),
        ("dtype", PT::String),
        ("first_row", PT::UInt64),
        ("n_rows", PT::UInt64),
        ("n_null", PT::UInt64),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), PT::UInt64));
    for n in ["size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes"] {
        f.push((n.into(), PT::UInt64));
    }
    f.extend(rec_fields());
    f.push(("n_sampled_rows".into(), PT::UInt64));
    f.push(("n_sampled_blocks".into(), PT::UInt64));
    f
}
```

- [ ] **Step 2: Row**

Replace the `row.extend([render(&o.lo), …])` block with:

```rust
        row.extend(p.row(&oc));
        row.push(u(o.n_midnight));
        row.extend([
            AnyValue::UInt64(lvl.size_bytes),
            u(scale(z[0])),
            AnyValue::UInt64(original_polars),
            u(scale(z[1])),
        ]);
```

Delete the now-unused `render`, `f` and `d` bindings in `row` and the old `fn render` if nothing else uses it.

- [ ] **Step 3: Rust streaming tests**

Update every test reading `distinct_overflowed` (`overflow_rejects_the_dictionary`: assert `est_method == "hll"` only). Add:

```rust
    #[test]
    fn shared_columns_are_filled() {
        let mut s = Streaming::new(params(), 0, 1);
        s.add(&batch(vec![("a", ints(&[Some(0), Some(5), Some(7), Some(0), Some(5), Some(7)]))]))
            .unwrap();
        let out = s.finish().unwrap();
        assert_eq!(texts(&out, "class"), some(&["ordinal"])); // 7 ≤ 2·6
        assert_eq!(texts(&out, "min"), some(&["0"]));
        assert_eq!(texts(&out, "est_method"), some(&["observed"]));
    }
```

(Use the existing test helpers `batch`, `ints`, `texts`, `some`; if `texts` cannot read a Boolean or Float column, read the String ones only.)

Run: `cargo test --lib streaming::`
Expected: all pass, including `output_matches_declared_schema`-style tests (update their expected column lists if they hard-code them).

- [ ] **Step 4: Python streaming tests**

`tests/test_streaming_recommend.py`:
- `SCHEMA`:

```python
VALUE_BLOCK = {
    "n_unique": pl.UInt64,
    "unique": pl.Boolean,
    "est_cardinality": pl.Float64,
    "est_low": pl.Float64,
    "est_high": pl.Float64,
    "est_method": pl.String,
    "estimates_agree": pl.Boolean,
    "class": pl.String,
    "min": pl.String,
    "max": pl.String,
    "min_len": pl.UInt64,
    "max_len": pl.UInt64,
    "gcd": pl.Decimal(38, 0),
    "sum_len": pl.UInt64,
    "sum_len_unique": pl.UInt64,
    "n_nan": pl.UInt64,
    "n_inf": pl.UInt64,
    "n_fractional": pl.UInt64,
    "max_frac_digits": pl.UInt32,
    "n_f32_inexact": pl.UInt64,
    "n_numeric": pl.UInt64,
    "n_numeric_int": pl.UInt64,
    "n_leading_zero": pl.UInt64,
    "numeric_int_min": pl.Decimal(38, 0),
    "numeric_int_max": pl.Decimal(38, 0),
    "numeric_max_int_digits": pl.UInt32,
    "numeric_max_frac_digits": pl.UInt32,
    "numeric_min_frac_digits": pl.UInt32,
    "numeric_max_sig_digits": pl.UInt32,
    "n_iso_date": pl.UInt64,
    "n_iso_time": pl.UInt64,
    "n_iso_datetime": pl.UInt64,
    "n_iso_datetime_tz": pl.UInt64,
    "iso_max_frac_digits": pl.UInt32,
    "iso_max_sig_frac_digits": pl.UInt32,
    "iso_n_offsets": pl.UInt64,
    "iso_n_midnight": pl.UInt64,
}
SCHEMA = {
    "column": pl.String,
    "status": pl.String,
    "dtype": pl.String,
    "first_row": pl.UInt64,
    "n_rows": pl.UInt64,
    "n_null": pl.UInt64,
    **VALUE_BLOCK,
    "n_midnight": pl.UInt64,
    "size_bytes": pl.UInt64,
    "size_zstd_bytes": pl.UInt64,
    "size_polars_bytes": pl.UInt64,
    "size_polars_zstd_bytes": pl.UInt64,
    # … the rec_* entries and n_sampled_rows / n_sampled_blocks, unchanged …
}
```

- `assert_parity`: `overflowed = r["est_method"] == "hll"`; and after the existing key loop, compare the shared value columns:

```python
        exact_cols = [
            "n_null", "min", "max", "min_len", "max_len", "sum_len", "gcd", "n_midnight",
            "n_nan", "n_inf", "n_fractional", "max_frac_digits", "n_f32_inexact",
            *[c for c in VALUE_BLOCK if c.startswith(("n_numeric", "numeric_", "n_leading", "n_iso", "iso_"))],
        ]
        for k in exact_cols:
            assert r[k] == o[k], (name, k, r[k], o[k])
        if overflowed:
            sigma = 3 * 1.04 / 128  # p = 14
            assert abs(r["n_unique"] - o["n_unique"]) <= sigma * o["n_unique"] + 1, name
        else:
            for k in ["n_unique", "unique", "class", "sum_len_unique"]:
                assert r[k] == o[k], (name, k, r[k], o[k])
        assert r["est_low"] <= r["est_cardinality"] <= r["est_high"], name
```

`one_shot(frame)` must return rows with the conclusion columns (`RecommendRust` result already has them — check the helper selects whole rows; if it selects a subset, add the shared columns).

- new high-cardinality test:

```python
def test_sampling_phase_estimates():
    n = 60_000
    frame = pl.DataFrame(
        {"key": pl.Series([f"id-{i:08d}" for i in range(n)]), "x": [i % 7 for i in range(n)]}
    )
    out = stream(frame, 8_192)
    r = {row["column"]: row for row in out.iter_rows(named=True)}
    key = r["key"]
    assert key["est_method"] == "hll" and key["unique"] is True
    assert abs(key["n_unique"] - n) <= 3 * 1.04 / 128 * n
    assert abs(key["sum_len_unique"] - 11 * n) <= 0.05 * 11 * n
    assert key["class"] == "discrete"
    assert r["x"]["class"] == "ordinal" and r["x"]["n_unique"] == 7
```

(`stream(frame, batch_rows)` is the file's existing helper.)

- [ ] **Step 5: Java streaming checks**

In `StreamingRecommenderTest.recommendsFromBatchesAddedOverTime`, inside the `result` lambda, add:

```java
                assertEquals(List.of("ordinal", "boolean"), column(root, "class"));
                assertEquals(List.of("0", "x"), column(root, "min"));
                assertEquals(List.of("observed", "observed"), column(root, "est_method"));
```

(`a = [0,5,7]` twice: 7 ≤ 2·6 → ordinal; `s` has two values → boolean.)

- [ ] **Step 6: Build and run everything**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo test --lib && cargo test --lib --no-default-features && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -q && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_streaming_recommend.py tests/test_describe.py -m slow -q
cd services/analytics && cargo build --release --no-default-features --target-dir target/capi && cd bindings/java && ./mvnw -q test
```

Expected: all pass; JUnit 9 pass.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add -A services/analytics tests
git commit -m "streaming: output the shared value columns; parity on estimates; Java checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Benchmarks, documentation, final verification

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-09-26-describe-technique-design.md`, `docs/superpowers/specs/2026-09-26-recommend-technique-design.md`, `docs/superpowers/specs/2026-09-29-streaming-recommender-design.md`, `docs/superpowers/specs/2026-10-01-recommender-parity-design.md` (status line)

- [ ] **Step 1: Benchmarks**

Before/after comparisons use `git stash`-free worktrees: run each script on this branch and on `main` (`git worktree add ../tp-main main`, build there with `maturin develop --release` into the same env only *after* recording this branch's numbers, then rebuild this branch).

```bash
cd /c/Users/Alexander/turbo-parakeet
/c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_streaming_recommend.py
/c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_describe.py
/c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_recommend.py
```

Record: streaming `add` throughput and `finish` time, DescribeRust and RecommendRust medians, before and after. Expected: one-shot within ±5% (top-5 removed, rendering added); streaming `add` slower on numeric-heavy frames (every dtype now counted) — report the number, it is the cost the spec accepted (§11).

- [ ] **Step 2: CLAUDE.md**

Update, keeping the file's style:
- **Describe** paragraph: remove entropy, top-5, Duj1, `population_rows`, the `exact` method; metrics now list `min` / `max` (arrow-rs text); conclusions: `unique`, `est_cardinality` / `est_low` / `est_high` / `est_method` ∈ {observed, hll, schnabel, chao1} by the ratio rule (d/n ≥ 0.5 → the count; else Schnabel → Chao1 floored at it), `estimates_agree`, `class`; keywords drop `population_rows`; agreement: exact except ZSTD sizes (1%) and the estimate columns (10%).
- **Recommend** paragraph: keywords are Describe's plus `boolean_pairs`; Rust supplies the conclusions.
- **Streaming Recommend** paragraph: output = head + Describe's value block + `n_midnight` + sizes + `rec_*` + sample columns; distinct counting by HyperLogLog (`src/hll.rs`, p = 14) and a bottom-k sample (`src/distinct_sample.rs`, k = max(`categorical_threshold`, 1000)); `distinct_overflowed` gone; estimates Schnabel → Chao1 from the sample, or HLL.
- Project structure: add `hll.rs`, `distinct_sample.rs`, `conclusions.rs` to the `src/` list.
- `capi.rs` bullet: `analytics_describe_and_recommend` has no `population_rows`.
- Private plugin list: add `render`.

- [ ] **Step 3: Spec cross-references**

- Parity spec: `Status: implemented (plan docs/superpowers/plans/2026-10-01-recommender-parity.md).`
- Describe, Recommend and streaming specs: add one line under their status, e.g. `Amended by 2026-10-01-recommender-parity-design.md (estimator rule, class, min/max values; entropy, top-5, population_rows and Duj1 removed).`

- [ ] **Step 4: Lint and full verification**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --lib && cargo test --lib --no-default-features
cd /c/Users/Alexander/turbo-parakeet
/c/Users/Alexander/miniconda3/envs/p312/python.exe -m black --check services/analytics/analytics tests
/c/Users/Alexander/miniconda3/envs/p312/python.exe -m ruff check services/analytics/analytics tests
/c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -q
/c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -m slow -q
cd services/analytics && cargo build --release --no-default-features --target-dir target/capi && cd bindings/java && ./mvnw -q test
```

Expected: everything clean and passing. Fix any clippy dead-code finding by deleting the unused item (e.g. the old `render` helper or `level_estimate`).

- [ ] **Step 5: Commit**

```bash
cd /c/Users/Alexander/turbo-parakeet
git add CLAUDE.md
git add -f docs/superpowers/specs docs/superpowers/plans/2026-10-01-recommender-parity.md
git commit -m "docs: recommender parity in CLAUDE.md and spec cross-references

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Do not push: report the benchmark numbers and the final test counts to the user and ask how to integrate (superpowers:finishing-a-development-branch).
