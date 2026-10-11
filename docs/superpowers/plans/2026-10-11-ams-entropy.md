# AMS-Sampling Entropy Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-pass, O(`num_samples`)-memory Shannon entropy sketch (Chakrabarti–Cormode–McGregor) as a third implementation of `analytics.pairwise_entropy` and `analytics.threeway_entropy`, and benchmark it against the exact Rust kernels and the Polars references for accuracy, speed and memory.

**Architecture:** A stand-alone Rust sketch `AmsEntropy` (`techniques/single_column/unordered/ams_entropy.rs`) consumes one u64 key per row. Marginal, pair and triplet kernels (`ams_entropy.rs`, `techniques/pairwise/ordered/ams_joint_entropy.rs`) feed it value-stable keys and run one sketch per column or combination, rayon-parallel. Three private Arrow-in/Arrow-out entry points reach Python through `bindings/api/techniques.rs`, `bindings/python.rs` and `analytics/_plugin.py`, and are wrapped by `PairwiseEntropyAmsRust` / `ThreewayEntropyAmsRust` on the uniform technique contract.

**Tech Stack:** Rust (polars 0.51, rayon, foldhash `FixedState`), pyo3 via maturin, Python 3.12 + Polars, pytest, the existing benchmark harness.

**Spec:** `docs/superpowers/specs/2026-10-11-ams-entropy-design.md`. Read sections 2.1–2.2 before Task 1.

---

## Environment (every command)

The Bash tool does not keep shell state, so start **every** shell command with this prefix (Git Bash):

```bash
export PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"; cd /c/Users/Alexander/turbo-parakeet
```

- Rust unit tests: `(cd services/analytics && cargo test --lib <filter>)`. The env dir must be on PATH or the test binary fails with `STATUS_DLL_NOT_FOUND`.
- Rebuild the Python extension after any Rust change Python must see: `$PY scripts/check.py build` (runs `maturin develop --release`, ~2.5 min warm).
- Python tests: `$PY -m pytest <file> -n 0 -q` from the repo root.
- Commits run a pre-commit hook (rustfmt + black) that needs the env's Python on PATH, which the prefix provides. Never pass `--no-verify`.
- Work on branch `ams-entropy`, which already exists and holds the spec.

## File map

| File | Status | Responsibility |
|---|---|---|
| `services/analytics/src/techniques/single_column/unordered/ams_entropy.rs` | create | `AmsEntropy` sketch, Misra–Gries, key helpers, marginal kernel, output builder |
| `services/analytics/src/techniques/pairwise/ordered/ams_joint_entropy.rs` | create | pair and triplet kernels |
| `services/analytics/src/techniques/single_column/unordered/mod.rs` | modify | register `ams_entropy` |
| `services/analytics/src/techniques/pairwise/ordered/mod.rs` | modify | register `ams_joint_entropy` |
| `services/analytics/src/techniques/mod.rs` | modify | flat re-exports |
| `services/analytics/src/bindings/api/techniques.rs` | modify | `ams_*` entry points, `threeway` kwargs helper, tests |
| `services/analytics/src/bindings/python.rs` | modify | three `#[pyfunction]`s + registration |
| `services/analytics/analytics/_plugin.py` | modify | three private wrappers |
| `services/analytics/analytics/pairwise_entropy/ams.py` | create | `PairwiseEntropyAmsRust`, `check_num_samples` |
| `services/analytics/analytics/threeway_entropy/ams.py` | create | `ThreewayEntropyAmsRust` |
| `services/analytics/analytics/{pairwise,threeway}_entropy/__init__.py` | modify | export + `IMPLEMENTATIONS` |
| `tests/test_pairwise_entropy.py`, `tests/test_threeway_entropy.py` | modify | tolerance-aware known answers, new known answers |
| `tests/datagen.py`, `tests/test_datagen.py` | modify | `near_unique` generator + test |
| `tests/performance/benchmark_{pairwise,threeway}_entropy.py` | modify | near-unique dataset |
| `tests/performance/benchmark_ams_entropy.py` | create | memory + error report |
| `CLAUDE.md` | modify | document the new implementations |

---

### Task 1: The `AmsEntropy` sketch — deterministic cases

**Files:**
- Create: `services/analytics/src/techniques/single_column/unordered/ams_entropy.rs`
- Modify: `services/analytics/src/techniques/single_column/unordered/mod.rs`
- Modify: `services/analytics/src/techniques/mod.rs`

- [ ] **Step 1: Register the module and write the failing tests**

In `services/analytics/src/techniques/single_column/unordered/mod.rs`, add `ams_entropy` (keep alphabetical order):

```rust
//! Single-column techniques that ignore row order: Describe, GCD, distinct-count sketches.

pub(crate) mod ams_entropy;
pub(crate) mod cardinality_estimators;
pub(crate) mod describe;
pub(crate) mod gcd;
pub(crate) mod hll;
pub(crate) mod marginal_entropy;
```

In `services/analytics/src/techniques/mod.rs`, extend the single-column re-export:

```rust
pub(crate) use single_column::unordered::{
    ams_entropy, cardinality_estimators, describe, gcd, hll, marginal_entropy,
};
```

Create `ams_entropy.rs` containing only the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(keys: &[u64], num_samples: usize, seed: u64) -> f64 {
        let mut s = AmsEntropy::new(num_samples, seed);
        for &k in keys {
            s.update(k);
        }
        s.estimate()
    }

    #[test]
    fn all_unique_stream_is_log2_m() {
        let keys: Vec<u64> = (0..1000).collect();
        let h = sketch(&keys, 256, 0);
        assert!((h - 1000f64.log2()).abs() < 1e-12, "{h}");
    }

    #[test]
    fn constant_stream_is_exactly_zero() {
        assert_eq!(sketch(&[7u64; 1000], 256, 0), 0.0);
    }

    #[test]
    fn empty_and_single_row_streams_are_zero() {
        assert_eq!(sketch(&[], 16, 0), 0.0);
        assert_eq!(sketch(&[42], 16, 0), 0.0);
    }

    #[test]
    fn same_seed_same_estimate() {
        let keys: Vec<u64> = (0..5000u64).map(|i| mix64(i) % 97).collect();
        assert_eq!(sketch(&keys, 128, 9), sketch(&keys, 128, 9));
        assert_ne!(sketch(&keys, 128, 9), sketch(&keys, 128, 10));
    }

    #[test]
    fn constant_stream_does_not_fire_every_estimator_every_row() {
        // Without the waiting set, an estimator with no backup has threshold 1 and fires on
        // every row: 64 × 100K events. With it, about 64 × (1 + ln 100K) ≈ 800.
        let mut s = AmsEntropy::new(64, 0);
        for _ in 0..100_000 {
            s.update(3);
        }
        assert!(s.fired < 64 * 100, "fired {} events", s.fired);
    }

    #[test]
    fn misra_gries_finds_the_dominant_item() {
        let mut mg = MisraGries::default();
        for i in 0..10_000u64 {
            mg.update(if i % 4 == 0 { i } else { 5 });
        }
        let (top, count) = mg.top();
        assert_eq!(top, 5);
        // Lower bound, short by at most (m − m_top) / k = 2500 / 256.
        assert!((7_490..=7_500).contains(&count), "{count}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `(cd services/analytics && cargo test --lib ams_entropy::)`
Expected: compile errors, `cannot find type AmsEntropy` / `cannot find function mix64`.

- [ ] **Step 3: Implement the sketch**

Put this above the test module in `ams_entropy.rs`:

```rust
//! Shannon entropy of a stream in O(`num_samples`) memory: the Chakrabarti–Cormode–McGregor
//! estimator (TALG 6(3), 2010). AMS sampling plus a backup sample, so a dominant item cannot
//! swamp the estimate. Spec: docs/superpowers/specs/2026-10-11-ams-entropy-design.md.

use foldhash::fast::FixedState;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Misra–Gries counters for the dominant item: p̂ is low by at most (1 − p)/k (CCM Lemma 2.3),
/// which biases H by about 1/k relative.
const MG_COUNTERS: usize = 256;

/// SplitMix64's finaliser on `z + γ`: a bijection on u64.
pub(crate) fn mix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64 generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in [0, 1).
    fn unit0(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in (0, 1].
    fn unit1(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Rows before the next one whose label is below `p`: geometric, failures before success.
    fn skip(&mut self, p: f64) -> u64 {
        if p >= 1.0 {
            return 0;
        }
        if p <= 0.0 {
            return u64::MAX;
        }
        (self.unit1().ln() / (-p).ln_1p()) as u64 // `as` saturates
    }
}

/// Misra–Gries frequent-items summary with `MG_COUNTERS` counters.
#[derive(Default)]
struct MisraGries(HashMap<u64, u64, FixedState>);

impl MisraGries {
    fn update(&mut self, x: u64) {
        if let Some(c) = self.0.get_mut(&x) {
            *c += 1;
        } else if self.0.len() < MG_COUNTERS {
            self.0.insert(x, 1);
        } else {
            // Amortised O(1): total decrements never exceed total increments.
            self.0.retain(|_, c| {
                *c -= 1;
                *c > 0
            });
        }
    }

    /// The item with the largest counter and that counter, a lower bound on its count.
    fn top(&self) -> (u64, u64) {
        self.0
            .iter()
            .max_by_key(|&(k, c)| (*c, *k))
            .map_or((0, 0), |(k, c)| (*k, *c))
    }
}

/// CCM Algorithm 2: H from m, the dominant item and each estimator's
/// (primary key, primary count r0, backup count r1).
fn ccm_estimate(
    m: u64,
    mg: &MisraGries,
    samples: impl Iterator<Item = (u64, u64, Option<u64>)>,
) -> f64 {
    if m == 0 {
        return 0.0;
    }
    let mf = m as f64;
    let lambda = |x: u64| {
        if x == 0 {
            0.0
        } else {
            x as f64 * (mf / x as f64).log2()
        }
    };
    let step = |r: u64| lambda(r) - lambda(r - 1); // r ≥ 1 always
    let (top, count) = mg.top();
    let p = count as f64 / mf;
    let (mut sum, mut n) = (0.0, 0usize);
    if p <= 0.5 {
        for (_, r0, _) in samples {
            sum += step(r0);
            n += 1;
        }
        return sum / n as f64;
    }
    // The dominant item's term is computed from p̂; the rest is estimated from samples of the
    // stream without it: the primary when it is not the dominant item, else the backup.
    for (s0, r0, r1) in samples {
        n += 1;
        if s0 != top {
            sum += step(r0);
        } else if let Some(r1) = r1 {
            sum += step(r1);
        }
    }
    (1.0 - p) * sum / n as f64 + p * (1.0 / p).log2()
}

/// A sampled row: its key, its label and the key's tracked count before that row.
#[derive(Clone, Copy)]
struct Sample {
    key: u64,
    label: f64,
    offset: u64,
}

/// One AMS estimator: the smallest-label row (primary), the smallest-label row of another
/// token (backup), and the row of its next label event.
#[derive(Clone, Copy)]
struct Estimator {
    primary: Option<Sample>,
    backup: Option<Sample>,
    next: u64,
}

impl Estimator {
    /// Labels below this change the estimator.
    fn threshold(&self) -> f64 {
        match (self.primary, self.backup) {
            (None, _) => 1.0,
            (Some(p), None) => p.label,
            (Some(_), Some(b)) => b.label,
        }
    }
}

/// Occurrences of a key since an estimator first held it, and how many samples hold it.
struct Tracked {
    count: u64,
    holders: u32,
}

/// Streaming Shannon-entropy sketch (bits) with `num_samples` estimators.
pub(crate) struct AmsEntropy {
    estimators: Vec<Estimator>,
    /// (next event row, estimator), smallest row first; entries not matching the
    /// estimator's `next` are stale and skipped.
    events: BinaryHeap<Reverse<(u64, u32)>>,
    tracked: HashMap<u64, Tracked, FixedState>,
    /// Estimators with a primary but no backup yet, by primary key.
    waiting: HashMap<u64, Vec<u32>, FixedState>,
    n_waiting: usize,
    mg: MisraGries,
    rng: Rng,
    m: u64,
    #[cfg(test)]
    fired: u64,
}

impl AmsEntropy {
    pub(crate) fn new(num_samples: usize, seed: u64) -> Self {
        let blank = Estimator {
            primary: None,
            backup: None,
            next: 0,
        };
        Self {
            estimators: vec![blank; num_samples],
            events: (0..num_samples as u32).map(|e| Reverse((0, e))).collect(),
            tracked: HashMap::default(),
            waiting: HashMap::default(),
            n_waiting: 0,
            mg: MisraGries::default(),
            rng: Rng(seed),
            m: 0,
            #[cfg(test)]
            fired: 0,
        }
    }

    pub(crate) fn update(&mut self, x: u64) {
        let row = self.m;
        self.m += 1;
        self.mg.update(x);
        if self.n_waiting > self.waiting.get(&x).map_or(0, Vec::len) {
            self.fire_waiting(row, x);
        }
        while let Some(&Reverse((at, e))) = self.events.peek() {
            if at > row {
                break;
            }
            self.events.pop();
            if self.estimators[e as usize].next == row {
                self.event(e as usize, row, x);
            }
        }
        if let Some(t) = self.tracked.get_mut(&x) {
            t.count += 1;
        }
    }

    /// Entropy estimate in bits; 0 for an empty stream.
    pub(crate) fn estimate(&self) -> f64 {
        let r = |s: &Sample| self.tracked[&s.key].count - s.offset;
        ccm_estimate(
            self.m,
            &self.mg,
            self.estimators.iter().filter_map(|e| {
                let p = e.primary?;
                Some((p.key, r(&p), e.backup.as_ref().map(r)))
            }),
        )
    }

    /// The first row of another token gives every estimator waiting on a different key its
    /// backup, or a new primary when its heap event falls on this row (label below t0).
    fn fire_waiting(&mut self, row: u64, x: u64) {
        let keys: Vec<u64> = self.waiting.keys().copied().filter(|&k| k != x).collect();
        for k in keys {
            for e in self.waiting.remove(&k).unwrap_or_default() {
                self.n_waiting -= 1;
                let e = e as usize;
                let p = self.estimators[e]
                    .primary
                    .expect("waiting estimators hold a primary");
                let offset = self.hold(x);
                if self.estimators[e].next == row {
                    let label = self.rng.unit0() * p.label;
                    self.estimators[e].backup = Some(p);
                    self.estimators[e].primary = Some(Sample { key: x, label, offset });
                } else {
                    let label = p.label + self.rng.unit0() * (1.0 - p.label);
                    self.estimators[e].backup = Some(Sample { key: x, label, offset });
                }
                #[cfg(test)]
                {
                    self.fired += 1;
                }
                self.schedule(e, row);
            }
        }
    }

    /// Estimator `e` draws a label below its threshold on this row, whose token is `x`.
    fn event(&mut self, e: usize, row: u64, x: u64) {
        #[cfg(test)]
        {
            self.fired += 1;
        }
        let est = self.estimators[e];
        let label = self.rng.unit0() * est.threshold();
        match (est.primary, est.backup) {
            (None, _) => {
                let offset = self.hold(x);
                self.estimators[e].primary = Some(Sample { key: x, label, offset });
                self.waiting.entry(x).or_default().push(e as u32);
                self.n_waiting += 1;
            }
            (Some(p), None) => {
                // Waiting: other tokens went through fire_waiting, so x is the primary's
                // token, and a smaller label restarts its count.
                debug_assert_eq!(x, p.key);
                let offset = self.tracked[&x].count;
                self.estimators[e].primary = Some(Sample { label, offset, ..p });
            }
            (Some(p), Some(b)) => {
                if label < p.label {
                    if x == p.key {
                        let offset = self.tracked[&x].count;
                        self.estimators[e].primary = Some(Sample { label, offset, ..p });
                    } else {
                        self.release(b.key);
                        let offset = self.hold(x);
                        self.estimators[e].backup = Some(p);
                        self.estimators[e].primary = Some(Sample { key: x, label, offset });
                    }
                } else if x != p.key {
                    self.release(b.key);
                    let offset = self.hold(x);
                    self.estimators[e].backup = Some(Sample { key: x, label, offset });
                }
            }
        }
        self.schedule(e, row);
    }

    /// Start (or share) tracking `key`; its count so far is the new sample's offset.
    fn hold(&mut self, key: u64) -> u64 {
        let t = self.tracked.entry(key).or_insert(Tracked {
            count: 0,
            holders: 0,
        });
        t.holders += 1;
        t.count
    }

    fn release(&mut self, key: u64) {
        if let Some(t) = self.tracked.get_mut(&key) {
            t.holders -= 1;
            if t.holders == 0 {
                self.tracked.remove(&key);
            }
        }
    }

    /// Draw the row of estimator `e`'s next event after `row`.
    fn schedule(&mut self, e: usize, row: u64) {
        let skip = self.rng.skip(self.estimators[e].threshold());
        let next = row.saturating_add(1).saturating_add(skip);
        self.estimators[e].next = next;
        if next != u64::MAX {
            self.events.push(Reverse((next, e as u32)));
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `(cd services/analytics && cargo test --lib ams_entropy::)`
Expected: 6 passed. If `all_unique_stream_is_log2_m` is off by more than 1e-12, the per-row ordering is wrong: offsets must be taken *before* the row's count increment, so every r is 1.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/techniques/single_column/unordered/ams_entropy.rs services/analytics/src/techniques/single_column/unordered/mod.rs services/analytics/src/techniques/mod.rs
git commit -m "AMS entropy sketch (CCM estimator) with skip-ahead and waiting estimators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Statistical tests — accuracy and agreement with CCM Algorithm 1

**Files:**
- Modify: `services/analytics/src/techniques/single_column/unordered/ams_entropy.rs` (test module only)

- [ ] **Step 1: Add the statistical tests**

Append inside `mod tests` in `ams_entropy.rs`:

```rust
    fn exact_entropy(keys: &[u64]) -> f64 {
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for &k in keys {
            *counts.entry(k).or_insert(0) += 1;
        }
        let m = keys.len() as f64;
        counts
            .values()
            .map(|&c| {
                let p = c as f64 / m;
                -p * p.log2()
            })
            .sum()
    }

    /// Pareto-rounded tokens: P(token ≥ k) = k^-1.1, a heavy head (~53% are 1) and long tail.
    fn zipf(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = Rng(seed);
        (0..n).map(|_| rng.unit1().powf(-1.0 / 1.1) as u64).collect()
    }

    /// 95% one value, the rest spread over 1,000 others: exercises the backup sample.
    fn skewed(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| {
                if rng.unit0() < 0.95 {
                    0
                } else {
                    1 + rng.next_u64() % 1000
                }
            })
            .collect()
    }

    fn mean_and_se(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    }

    /// CCM Algorithm 1 as written: a fresh label for every estimator on every row, O(c) per row.
    fn naive(keys: &[u64], c: usize, seed: u64) -> f64 {
        let mut rng = Rng(seed);
        let mut mg = MisraGries::default();
        // (key, label, r) per estimator; label 1.0 marks an empty slot.
        let mut p = vec![(0u64, 1.0f64, 0u64); c];
        let mut b = vec![(0u64, 1.0f64, 0u64); c];
        for &x in keys {
            mg.update(x);
            for (p, b) in p.iter_mut().zip(b.iter_mut()) {
                let t = rng.unit0();
                if t < p.1 {
                    if p.1 < 1.0 && x != p.0 {
                        *b = *p;
                    }
                    *p = (x, t, 1);
                } else if x == p.0 {
                    p.2 += 1;
                } else if t < b.1 {
                    *b = (x, t, 1);
                } else if b.1 < 1.0 && x == b.0 {
                    b.2 += 1;
                }
            }
        }
        ccm_estimate(
            keys.len() as u64,
            &mg,
            p.iter()
                .zip(&b)
                .map(|(p, b)| (p.0, p.2, (b.1 < 1.0).then_some(b.2))),
        )
    }

    #[test]
    fn mean_estimate_is_within_one_percent_of_exact() {
        for keys in [zipf(10_000, 1), skewed(10_000, 2)] {
            let h = exact_entropy(&keys);
            let est: Vec<f64> = (0..100).map(|seed| sketch(&keys, 4096, seed)).collect();
            let (mean, se) = mean_and_se(&est);
            assert!(
                (mean - h).abs() <= 0.01 * h,
                "mean {mean} (se {se}) vs exact {h}"
            );
        }
    }

    #[test]
    fn skip_ahead_matches_naive_algorithm_1() {
        for keys in [zipf(2_000, 3), skewed(2_000, 4)] {
            let fast: Vec<f64> = (0..400).map(|s| sketch(&keys, 64, s)).collect();
            let slow: Vec<f64> = (0..400).map(|s| naive(&keys, 64, 10_000 + s)).collect();
            let ((m1, se1), (m2, se2)) = (mean_and_se(&fast), mean_and_se(&slow));
            assert!(
                (m1 - m2).abs() <= 4.0 * (se1 * se1 + se2 * se2).sqrt(),
                "skip-ahead {m1} ± {se1}, naive {m2} ± {se2}"
            );
        }
    }
```

- [ ] **Step 2: Run the tests**

Run: `(cd services/analytics && cargo test --lib ams_entropy::)`
Expected: 8 passed, in under a minute in debug mode.

If `mean_estimate_is_within_one_percent_of_exact` fails, compare the failing stream's error with the reported `se`:
- **error several times `se`:** a bias. If the skewed stream fails and the naive comparison passes, the bias comes from p̂ (Misra–Gries); if the naive comparison also fails, the skip-ahead logic is wrong.
- **error within about 3 `se`:** variance, not a bug.

Either way, report the numbers rather than loosening the tolerance or changing seeds.

- [ ] **Step 3: Commit**

```bash
git add services/analytics/src/techniques/single_column/unordered/ams_entropy.rs
git commit -m "AMS entropy: accuracy and Algorithm 1 equivalence tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Marginal, pair and triplet kernels

**Files:**
- Modify: `services/analytics/src/techniques/single_column/unordered/ams_entropy.rs`
- Create: `services/analytics/src/techniques/pairwise/ordered/ams_joint_entropy.rs`
- Modify: `services/analytics/src/techniques/pairwise/ordered/mod.rs`
- Modify: `services/analytics/src/techniques/mod.rs`

- [ ] **Step 1: Register the module and write the failing kernel tests**

`services/analytics/src/techniques/pairwise/ordered/mod.rs`:

```rust
//! Pairwise techniques that compare row i of one column with row i of the other.

pub(crate) mod ams_joint_entropy;
pub(crate) mod ari;
pub(crate) mod chi_squared;
pub(crate) mod contingency;
pub(crate) mod joint_entropy;
```

`services/analytics/src/techniques/mod.rs`:

```rust
pub(crate) use pairwise::ordered::{ams_joint_entropy, ari, chi_squared, contingency, joint_entropy};
```

Append to the test module in `ams_entropy.rs`:

```rust
    fn entropies(out: Series, name: &str) -> Vec<f64> {
        let df = out.into_frame().unnest([name]).unwrap();
        df.column("entropy")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn marginal_kernel_unique_constant_and_null_columns() {
        let n = 1000i64;
        let u = Series::new("u".into(), (0..n).collect::<Vec<_>>());
        let k = Series::new("k".into(), vec![7i64; n as usize]);
        let nulls = Series::new("z".into(), vec![None::<i64>; n as usize]);
        let half = Series::new(
            "h".into(),
            (0..n).map(|i| (i % 2 == 0).then_some(1i64)).collect::<Vec<_>>(),
        );
        let out = ams_marginal_entropy_impl(&[u, k, nulls, half], 256, 0).unwrap();
        let h = entropies(out, "marginal_entropy");
        assert!((h[0] - (n as f64).log2()).abs() < 1e-12);
        assert_eq!(h[1], 0.0);
        assert_eq!(h[2], 0.0); // all null: one category
        assert!((h[3] - 1.0).abs() < 0.05, "{}", h[3]); // null vs 1, half each
    }

    #[test]
    fn marginal_kernel_rejects_empty_and_ragged_input() {
        assert!(ams_marginal_entropy_impl(&[], 16, 0).is_err());
        let empty = Series::new("a".into(), Vec::<i64>::new());
        assert!(ams_marginal_entropy_impl(&[empty], 16, 0).is_err());
        let a = Series::new("a".into(), &[1i64, 2, 3]);
        let b = Series::new("b".into(), &[1i64, 2]);
        assert!(ams_marginal_entropy_impl(&[a, b], 16, 0).is_err());
    }
```

Create `ams_joint_entropy.rs` with only its test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn frame(n: i64) -> Vec<Series> {
        vec![
            Series::new("hi".into(), (0..n).map(|i| i / 10).collect::<Vec<_>>()),
            Series::new("lo".into(), (0..n).map(|i| i % 10).collect::<Vec<_>>()),
            Series::new("k".into(), vec![7i64; n as usize]),
            Series::new("k2".into(), vec![1i64; n as usize]),
        ]
    }

    fn column<'a>(df: &'a DataFrame, name: &str) -> Vec<&'a str> {
        df.column(name).unwrap().str().unwrap().into_no_null_iter().collect()
    }

    #[test]
    fn composite_key_pair_is_log2_n_and_constant_pair_is_zero() {
        let out = ams_pairwise_joint_entropy_impl(
            &frame(1000),
            PairwiseKwargs { pairs: None },
            256,
            0,
        )
        .unwrap();
        let df = out.into_frame().unnest(["pairwise_entropy"]).unwrap();
        // (hi,lo) (hi,k) (hi,k2) (lo,k) (lo,k2) (k,k2)
        assert_eq!(column(&df, "col_a"), ["hi", "hi", "hi", "lo", "lo", "k"]);
        assert_eq!(column(&df, "col_b"), ["lo", "k", "k2", "k", "k2", "k2"]);
        let h: Vec<f64> = df.column("entropy").unwrap().f64().unwrap().into_no_null_iter().collect();
        assert!((h[0] - 1000f64.log2()).abs() < 1e-12);
        assert_eq!(h[5], 0.0);
    }

    #[test]
    fn explicit_pairs_keep_their_order() {
        let pairs = PairwiseKwargs {
            pairs: Some(vec![vec!["lo".into(), "hi".into()]]),
        };
        let out = ams_pairwise_joint_entropy_impl(&frame(100), pairs, 64, 0).unwrap();
        let df = out.into_frame().unnest(["pairwise_entropy"]).unwrap();
        assert_eq!(column(&df, "col_a"), ["lo"]);
        assert_eq!(column(&df, "col_b"), ["hi"]);
    }

    #[test]
    fn one_column_gives_no_pairs() {
        let out = ams_pairwise_joint_entropy_impl(
            &frame(10)[..1],
            PairwiseKwargs { pairs: None },
            16,
            0,
        )
        .unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn composite_key_triplet_is_log2_n() {
        let triplets = ThreewayKwargs {
            triplets: Some(vec![vec!["hi".into(), "lo".into(), "k".into()]]),
        };
        let out = ams_threeway_joint_entropy_impl(&frame(1000), triplets, 256, 0).unwrap();
        let df = out.into_frame().unnest(["threeway_entropy"]).unwrap();
        assert_eq!(column(&df, "col_c"), ["k"]);
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 1000f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn threeway_needs_three_columns() {
        let r = ams_threeway_joint_entropy_impl(
            &frame(10)[..2],
            ThreewayKwargs { triplets: None },
            16,
            0,
        );
        assert!(r.is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `(cd services/analytics && cargo test --lib ams_)`
Expected: compile errors, `cannot find function ams_marginal_entropy_impl` / `ams_pairwise_joint_entropy_impl`.

- [ ] **Step 3: Implement the kernels**

In `ams_entropy.rs`, replace the imports at the top with:

```rust
use foldhash::fast::FixedState;
use polars::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::common::*;
```

and add after `mix64` (before `struct Rng`):

```rust
/// Key of a null row: null is its own category, as in the exact kernels. (It collides with
/// the one value whose `mix64` equals it.)
const NULL_KEY: u64 = 0x6E75_6C6C_6E75_6C6C;

/// Seed domains, so a marginal and a combination with the same index draw different labels.
pub(crate) const MARGINAL: u64 = 1;
pub(crate) const PAIR: u64 = 2;
pub(crate) const TRIPLE: u64 = 3;

/// One key for two keys, order-sensitive; nest it for triplets.
pub(crate) fn combine(a: u64, b: u64) -> u64 {
    mix64(a ^ mix64(b))
}

/// The seed of sketch `index` in `domain`: independent sketches, whatever the thread schedule.
pub(crate) fn sketch_seed(seed: u64, domain: u64, index: usize) -> u64 {
    mix64(mix64(seed ^ domain) ^ index as u64)
}

/// Row count shared by every input column; errors on no columns, zero rows or ragged input.
pub(crate) fn row_count(inputs: &[Series], what: &str) -> PolarsResult<usize> {
    let Some(first) = inputs.first() else {
        return Err(PolarsError::ComputeError(
            format!("{what} requires at least one column").into(),
        ));
    };
    let r = first.len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate entropy on empty columns".into(),
        ));
    }
    if let Some((idx, s)) = inputs.iter().enumerate().find(|(_, s)| s.len() != r) {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "All columns must have the same length: column {idx} has length {} but expected {r}",
                s.len()
            )
            .into(),
        ));
    }
    Ok(r)
}

/// Per-row sketch keys of the columns in `needed` (value-stable `encode_series` keys, null
/// folded in); other slots stay empty.
pub(crate) fn keyed_columns(
    inputs: &[Series],
    needed: impl IntoIterator<Item = usize>,
) -> PolarsResult<Vec<Vec<u64>>> {
    let needed: HashSet<usize> = needed.into_iter().collect();
    let cache = build_column_cache_par(inputs, &needed)?;
    Ok(cache
        .par_iter()
        .map(|col| {
            (0..col.len())
                .map(|i| {
                    if col.is_null[i] {
                        NULL_KEY
                    } else {
                        mix64(col.values[i])
                    }
                })
                .collect()
        })
        .collect())
}

/// The kernels' output: struct `name` of string key columns, then `entropy`.
pub(crate) fn entropy_struct(
    name: &str,
    keys: Vec<(&str, Vec<&str>)>,
    entropy: Vec<f64>,
) -> PolarsResult<Series> {
    let n = entropy.len();
    let mut fields: Vec<Series> = keys
        .into_iter()
        .map(|(field, values)| {
            StringChunked::from_iter(values)
                .into_series()
                .with_name(field.into())
        })
        .collect();
    fields.push(Float64Chunked::from_vec("entropy".into(), entropy).into_series());
    Ok(StructChunked::from_series(name.into(), n, fields.iter())?.into_series())
}

/// H(col) of every column, one sketch each: `col_name`, `entropy`.
pub(crate) fn ams_marginal_entropy_impl(
    inputs: &[Series],
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    row_count(inputs, "ams_marginal_entropy")?;
    let keys = keyed_columns(inputs, 0..inputs.len())?;
    let entropy: Vec<f64> = keys
        .par_iter()
        .enumerate()
        .map(|(i, col)| {
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, MARGINAL, i));
            for &k in col {
                sketch.update(k);
            }
            sketch.estimate()
        })
        .collect();
    let names = inputs.iter().map(|s| s.name().as_str()).collect();
    entropy_struct("marginal_entropy", vec![("col_name", names)], entropy)
}
```

Fill `ams_joint_entropy.rs` above its test module:

```rust
//! Joint entropy of column pairs and triplets, estimated by one AMS (CCM) sketch each:
//! O(`num_samples`) memory per combination, independent of the number of distinct values.

use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::common::*;
use crate::techniques::ams_entropy::{
    combine, entropy_struct, keyed_columns, row_count, sketch_seed, AmsEntropy, PAIR, TRIPLE,
};

fn name_map(inputs: &[Series]) -> HashMap<String, usize> {
    inputs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name().to_string(), i))
        .collect()
}

/// H(A,B) per pair (every pair when `kwargs.pairs` is None): `col_a`, `col_b`, `entropy`.
pub(crate) fn ams_pairwise_joint_entropy_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    row_count(inputs, "ams_pairwise_joint_entropy")?;
    let n = inputs.len();
    let pairs: Vec<(usize, usize)> = match &kwargs.pairs {
        Some(raw) => resolve_pairs(raw, &name_map(inputs))?,
        None => (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect(),
    };
    let keys = keyed_columns(inputs, pairs.iter().flat_map(|&(i, j)| [i, j]))?;
    let entropy: Vec<f64> = pairs
        .par_iter()
        .enumerate()
        .map(|(p, &(i, j))| {
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, PAIR, p));
            for (&a, &b) in keys[i].iter().zip(&keys[j]) {
                sketch.update(combine(a, b));
            }
            sketch.estimate()
        })
        .collect();
    let name = |k: usize| inputs[k].name().as_str();
    entropy_struct(
        "pairwise_entropy",
        vec![
            ("col_a", pairs.iter().map(|p| name(p.0)).collect()),
            ("col_b", pairs.iter().map(|p| name(p.1)).collect()),
        ],
        entropy,
    )
}

/// H(A,B,C) per triplet (every triplet when `kwargs.triplets` is None):
/// `col_a`, `col_b`, `col_c`, `entropy`.
pub(crate) fn ams_threeway_joint_entropy_impl(
    inputs: &[Series],
    kwargs: ThreewayKwargs,
    num_samples: usize,
    seed: u64,
) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError(
            "ams_threeway_joint_entropy requires at least three columns".into(),
        ));
    }
    row_count(inputs, "ams_threeway_joint_entropy")?;
    let n = inputs.len();
    let triplets: Vec<(usize, usize, usize)> = match &kwargs.triplets {
        Some(raw) => resolve_triplets(raw, &name_map(inputs))?,
        None => (0..n)
            .flat_map(|i| ((i + 1)..n).flat_map(move |j| ((j + 1)..n).map(move |k| (i, j, k))))
            .collect(),
    };
    let keys = keyed_columns(inputs, triplets.iter().flat_map(|&(i, j, k)| [i, j, k]))?;
    let entropy: Vec<f64> = triplets
        .par_iter()
        .enumerate()
        .map(|(t, &(i, j, k))| {
            let mut sketch = AmsEntropy::new(num_samples, sketch_seed(seed, TRIPLE, t));
            for ((&a, &b), &c) in keys[i].iter().zip(&keys[j]).zip(&keys[k]) {
                sketch.update(combine(a, combine(b, c)));
            }
            sketch.estimate()
        })
        .collect();
    let name = |k: usize| inputs[k].name().as_str();
    entropy_struct(
        "threeway_entropy",
        vec![
            ("col_a", triplets.iter().map(|t| name(t.0)).collect()),
            ("col_b", triplets.iter().map(|t| name(t.1)).collect()),
            ("col_c", triplets.iter().map(|t| name(t.2)).collect()),
        ],
        entropy,
    )
}
```

- [ ] **Step 4: Run the tests, clippy and the layering check**

Run: `(cd services/analytics && cargo test --lib ams_) && $PY scripts/check.py fast`
Expected: all `ams_` tests pass (8 in `ams_entropy`, 5 in `ams_joint_entropy`), and `fast` (rustfmt, black, clippy, ruff, layering) is clean. If rustfmt complains, run `$PY scripts/check.py fmt` and re-run.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/techniques
git commit -m "AMS entropy: marginal, pair and triplet kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Entry points, Python binding and `_plugin`

**Files:**
- Modify: `services/analytics/src/bindings/api/techniques.rs`
- Modify: `services/analytics/src/bindings/python.rs`
- Modify: `services/analytics/analytics/_plugin.py`

- [ ] **Step 1: Write the failing API tests**

Append to `mod tests` in `services/analytics/src/bindings/api/techniques.rs`:

```rust
    #[test]
    fn ams_entropy_smoke() {
        let b = batch(vec![
            ("a", ints(&[1, 2, 3, 4])),
            ("b", ints(&[5, 5, 5, 5])),
            ("c", ints(&[1, 1, 2, 2])),
        ]);
        let m = ams_marginal_entropy(&b, 64, 0).unwrap();
        assert_eq!(m.column(1).as_primitive::<Float64Type>().value(0), 2.0); // unique
        assert_eq!(m.column(1).as_primitive::<Float64Type>().value(1), 0.0); // constant
        let pairs = [("a".to_string(), "b".to_string())];
        let p = ams_pairwise_joint_entropy(&b, Some(&pairs[..]), 64, 0).unwrap();
        assert_eq!(p.num_rows(), 1);
        assert_eq!(p.column(2).as_primitive::<Float64Type>().value(0), 2.0);
        let t = ams_threeway_joint_entropy(&b, None, 64, 0).unwrap();
        assert_eq!(t.column(3).as_primitive::<Float64Type>().value(0), 2.0);
    }

    #[test]
    fn ams_zero_samples_and_unknown_names_are_invalid_input() {
        let b = batch(vec![("a", ints(&[1, 2])), ("b", ints(&[1, 2]))]);
        assert!(matches!(ams_marginal_entropy(&b, 0, 0), Err(Error::InvalidInput(_))));
        assert!(matches!(
            ams_pairwise_joint_entropy(&b, None, 0, 0),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            ams_threeway_joint_entropy(&b, None, 0, 0),
            Err(Error::InvalidInput(_))
        ));
        let pairs = [("a".to_string(), "zz".to_string())];
        assert!(matches!(
            ams_pairwise_joint_entropy(&b, Some(&pairs[..]), 8, 0),
            Err(Error::InvalidInput(_))
        ));
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `(cd services/analytics && cargo test --lib bindings::api::techniques::tests::ams)`
Expected: compile errors, `cannot find function ams_marginal_entropy`.

- [ ] **Step 3: Implement the entry points**

In `techniques.rs` (`ThreewayKwargs` is already imported), replace the existing `threeway_joint_entropy` with the version below, which moves its kwargs building into a `threeway` helper the AMS entry point shares, and add the AMS functions after it:

```rust
/// `triplets` as kernel kwargs, every name checked.
fn threeway(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
) -> Result<ThreewayKwargs> {
    if let Some(t) = triplets {
        check_names(batch, t.iter().flat_map(|(a, b, c)| [a, b, c]))?;
    }
    Ok(ThreewayKwargs {
        triplets: triplets.map(|t| {
            t.iter()
                .map(|(a, b, c)| vec![a.clone(), b.clone(), c.clone()])
                .collect()
        }),
    })
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet; every triplet when None.
pub fn threeway_joint_entropy(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = threeway(batch, triplets)?;
    table(crate::techniques::joint_entropy::threeway_joint_entropy_impl(&columns(batch)?, kwargs))
}

/// The AMS sketches' size: 1 ..= u32::MAX estimators.
fn ams_samples(num_samples: usize) -> Result<usize> {
    if num_samples == 0 || num_samples > u32::MAX as usize {
        return Err(Error::InvalidInput(format!(
            "num_samples must be between 1 and {}, got {num_samples}",
            u32::MAX
        )));
    }
    Ok(num_samples)
}

/// `col_name`, `entropy` per column, estimated by the AMS (CCM) sketch.
pub fn ams_marginal_entropy(
    batch: &RecordBatch,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    table(crate::techniques::ams_entropy::ams_marginal_entropy_impl(
        &columns(batch)?,
        c,
        seed,
    ))
}

/// `col_a`, `col_b`, `entropy` per pair (every pair when None), estimated by the AMS sketch.
pub fn ams_pairwise_joint_entropy(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    let kwargs = pairwise(batch, pairs)?;
    table(
        crate::techniques::ams_joint_entropy::ams_pairwise_joint_entropy_impl(
            &columns(batch)?,
            kwargs,
            c,
            seed,
        ),
    )
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet (every triplet when None), estimated by
/// the AMS sketch.
pub fn ams_threeway_joint_entropy(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    let kwargs = threeway(batch, triplets)?;
    table(
        crate::techniques::ams_joint_entropy::ams_threeway_joint_entropy_impl(
            &columns(batch)?,
            kwargs,
            c,
            seed,
        ),
    )
}
```

Check `src/bindings/api/mod.rs` for how `techniques.rs`'s public functions are re-exported (for example `pub use techniques::*;`). If functions are listed by name, add the three `ams_*` names.

In `services/analytics/src/bindings/python.rs`, after `threeway_joint_entropy`:

```rust
#[pyfunction]
#[pyo3(signature = (data, num_samples, seed))]
fn ams_marginal_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    num_samples: usize,
    seed: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::ams_marginal_entropy(&batch, num_samples, seed)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs, num_samples, seed))]
fn ams_pairwise_joint_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    pairs: Option<Vec<(String, String)>>,
    num_samples: usize,
    seed: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || {
        api::ams_pairwise_joint_entropy(&batch, pairs.as_deref(), num_samples, seed)
    })
    .map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, triplets, num_samples, seed))]
fn ams_threeway_joint_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    triplets: Option<Vec<(String, String, String)>>,
    num_samples: usize,
    seed: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || {
        api::ams_threeway_joint_entropy(&batch, triplets.as_deref(), num_samples, seed)
    })
    .map(ArrowTable)
}
```

and register them after `threeway_joint_entropy` in the module function:

```rust
    m.add_function(wrap_pyfunction!(ams_marginal_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(ams_pairwise_joint_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(ams_threeway_joint_entropy, m)?)?;
```

In `services/analytics/analytics/_plugin.py`, after `marginal_entropy`:

```python
def ams_marginal_entropy(df: Frame, num_samples: int, seed: int) -> pl.DataFrame:
    """col_name, entropy per column, estimated by the AMS (CCM) sketch."""
    return pl.DataFrame(_rs.ams_marginal_entropy(_collected(df), num_samples, seed))


def ams_pairwise_joint_entropy(
    df: Frame,
    pairs: Sequence[tuple[str, str]] | None,
    num_samples: int,
    seed: int,
) -> pl.DataFrame:
    """col_a, col_b, entropy (H(A,B) in bits, AMS estimate); every pair when None."""
    return pl.DataFrame(
        _rs.ams_pairwise_joint_entropy(
            _collected(df), _tuples(pairs), num_samples, seed
        )
    )


def ams_threeway_joint_entropy(
    df: Frame,
    triplets: Sequence[tuple[str, str, str]] | None,
    num_samples: int,
    seed: int,
) -> pl.DataFrame:
    """col_a, col_b, col_c, entropy (H(A,B,C) in bits, AMS estimate); every triplet when None."""
    return pl.DataFrame(
        _rs.ams_threeway_joint_entropy(
            _collected(df), _tuples(triplets), num_samples, seed
        )
    )
```

- [ ] **Step 4: Run the Rust tests, build, and smoke-test from Python**

Run:
```bash
(cd services/analytics && cargo test --lib bindings::api::techniques::tests) && $PY scripts/check.py build && $PY -c "
import polars as pl
from analytics import _plugin
df = pl.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 5, 5, 5]})
print(_plugin.ams_marginal_entropy(df, 64, 0))
print(_plugin.ams_pairwise_joint_entropy(df, None, 64, 0))
"
```
Expected: the Rust tests pass, the build succeeds, and the script prints `a → 2.0`, `b → 0.0` and one pair with entropy `2.0`.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/bindings services/analytics/analytics/_plugin.py
git commit -m "AMS entropy: Arrow entry points, Python binding and _plugin wrappers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `PairwiseEntropyAmsRust` and `ThreewayEntropyAmsRust`

**Files:**
- Create: `services/analytics/analytics/pairwise_entropy/ams.py`
- Create: `services/analytics/analytics/threeway_entropy/ams.py`
- Modify: `services/analytics/analytics/pairwise_entropy/__init__.py`
- Modify: `services/analytics/analytics/threeway_entropy/__init__.py`
- Modify: `tests/test_pairwise_entropy.py`
- Modify: `tests/test_threeway_entropy.py`

- [ ] **Step 1: Make the existing known answers tolerance-aware and add new ones (failing)**

In `tests/test_pairwise_entropy.py`, add after the `NAN = ...` line:

```python
def _tol(impl) -> dict:
    """The implementation's own agreement tolerance: exact kernels 1e-5, sketches 5%."""
    cls = load(impl)
    return {"rel": cls.RTOL, "abs": cls.ATOL}
```

Replace `test_independent_columns` and `test_redundant_columns_and_null_category` with:

```python
@pytest.mark.parametrize("impl", ALL)
def test_independent_columns(impl):
    r = _row(impl, [0, 0, 1, 1], [0, 1, 0, 1])
    tol = _tol(impl)
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 2.0), **tol)
    # MI and NMI carry the entropies' errors: RTOL · (h_a + h_b + h_ab).
    mi_tol = tol["rel"] * 4 + 1e-12
    assert r["mi"] == pytest.approx(0.0, abs=mi_tol)
    assert r["nmi"] == pytest.approx(0.0, abs=mi_tol)
    assert r["redundant"] is False
    # 4 rows, 4 distinct pairs
    assert r["near_unique"] is True
    assert r["n_rows"] == 4


@pytest.mark.parametrize("impl", ALL)
def test_redundant_columns_and_null_category(impl):
    r = _row(
        impl, [None, None, 1, 1], ["x", "x", "y", "y"]
    )  # null is a category: a determines b
    tol = _tol(impl)
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 1.0), **tol)
    assert r["nmi"] == pytest.approx(1.0, abs=tol["rel"] * 3 + 1e-12)
    if tol["rel"] < 1e-3:  # exact kernels; an estimate may land either side of 0.9
        assert r["redundant"] is True
    assert r["near_unique"] is False
```

Append:

```python
@pytest.mark.parametrize("impl", ALL)
def test_unique_constant_and_composite_key(impl):
    n = 1_000
    df = pl.DataFrame(
        {
            "hi": [i // 10 for i in range(n)],
            "lo": [i % 10 for i in range(n)],
            "u": list(range(n)),
            "k": [7] * n,
        }
    )
    rows = {
        (r["col_a"], r["col_b"]): r
        for r in run(load(impl), {"t": df}).iter_rows(named=True)
    }
    log_n = math.log2(n)
    assert rows[("hi", "lo")]["h_ab"] == pytest.approx(log_n, rel=1e-9)  # composite key
    assert rows[("hi", "lo")]["near_unique"] is True
    assert rows[("u", "k")]["h_a"] == pytest.approx(log_n, rel=1e-9)  # unique column
    assert rows[("u", "k")]["h_b"] == pytest.approx(0.0, abs=1e-12)  # constant column
    assert rows[("u", "k")]["h_ab"] == pytest.approx(log_n, rel=1e-9)


def test_ams_seed_is_reproducible_and_num_samples_checked():
    from polars.testing import assert_frame_equal

    from analytics.pairwise_entropy import PairwiseEntropyAmsRust

    frames = {"t": mixed_dtypes(300)}
    first = run(PairwiseEntropyAmsRust, frames)
    assert_frame_equal(first, run(PairwiseEntropyAmsRust, frames))
    assert not first["h_ab"].equals(run(PairwiseEntropyAmsRust, frames, seed=1)["h_ab"])
    with pytest.raises(ValueError, match="num_samples"):
        PairwiseEntropyAmsRust(num_samples=0)
```

In `tests/test_threeway_entropy.py`, add `import math` to the imports and append:

```python
@pytest.mark.parametrize("impl", ALL)
def test_composite_key_and_constant_triplets(impl):
    n = 1_000
    keyed = pl.DataFrame(
        {"hi": [i // 10 for i in range(n)], "lo": [i % 10 for i in range(n)], "k": [7] * n}
    )
    assert run(load(impl), {"t": keyed})["h_abc"].item() == pytest.approx(
        math.log2(n), rel=1e-9
    )
    constant = pl.DataFrame({c: [1] * n for c in "abc"})
    assert run(load(impl), {"t": constant})["h_abc"].item() == pytest.approx(
        0.0, abs=1e-12
    )


def test_ams_num_samples_checked():
    from analytics.threeway_entropy import ThreewayEntropyAmsRust

    with pytest.raises(ValueError, match="num_samples"):
        ThreewayEntropyAmsRust(num_samples=0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_pairwise_entropy.py tests/test_threeway_entropy.py -n 0 -q`
Expected: the two `*num_samples*` tests fail with `ImportError: cannot import name 'PairwiseEntropyAmsRust'` / `'ThreewayEntropyAmsRust'`. Every other test passes, because the AMS classes aren't in `IMPLEMENTATIONS` yet.

- [ ] **Step 3: Implement the classes**

Create `services/analytics/analytics/pairwise_entropy/ams.py`:

```python
import math

import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame, metric_mismatches
from analytics.pairwise_entropy.base import PairwiseEntropy


def check_num_samples(num_samples: int) -> None:
    if not isinstance(num_samples, int) or not 1 <= num_samples <= 2**32 - 1:
        raise ValueError(
            f"num_samples must be an integer in [1, 2**32 - 1], got {num_samples!r}"
        )


class PairwiseEntropyAmsRust(PairwiseEntropy):
    """Rust extension functions `ams_marginal_entropy` + `ams_pairwise_joint_entropy`: one
    Chakrabarti–Cormode–McGregor sketch (`num_samples` estimators, O(num_samples) memory)
    per column and per pair, rayon-parallel. Entropies agree with the reference within
    RTOL; MI / NMI within the error the entropies propagate (see `agreement`)."""

    RTOL = 0.05
    ATOL = 1e-9

    def __init__(self, *, num_samples: int = 4096, seed: int = 0, **thresholds):
        super().__init__(**thresholds)
        check_num_samples(num_samples)
        self.num_samples = num_samples
        self.seed = seed

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(
                list(dict.fromkeys(c for k in group for _, c in k))
            )
            marginal = dict(
                _plugin.ams_marginal_entropy(
                    df, self.num_samples, self.seed
                ).iter_rows()
            )
            joint = _plugin.ams_pairwise_joint_entropy(
                df,
                [(a, b) for (_, a), (_, b) in group],
                self.num_samples,
                self.seed,
            )
            h_ab = {frozenset((a, b)): h for a, b, h in joint.iter_rows()}
            parts.append(
                self.entropy_rows(
                    group,
                    [marginal[a] for (_, a), _ in group],
                    [marginal[b] for _, (_, b) in group],
                    [h_ab[frozenset((a, b))] for (_, a), (_, b) in group],
                    [df.height] * len(group),
                )
            )
        return pl.concat(parts)

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """Entropies within RTOL. MI within RTOL·(h_a + h_b + h_ab) of the reference and NMI
        within that bound ÷ min(h_a, h_b): MI is a difference of estimates, so its relative
        error is unbounded. Conclusions are not compared: an estimate within tolerance can
        still land on the other side of a threshold."""
        keys = self.key_columns()
        problems = metric_mismatches(
            result,
            reference,
            keys,
            ["h_a", "h_b", "h_ab", "n_rows"],
            self.RTOL,
            self.ATOL,
        )
        if problems[:1] == ["key columns differ from the reference"]:
            return problems
        both = pl.concat(
            [
                reference.select(*keys, "h_a", "h_b", "h_ab", "mi", "nmi"),
                result.select(got_mi="mi", got_nmi="nmi"),
            ],
            how="horizontal",
        ).filter(pl.col("h_ab").is_not_null())
        for r in both.iter_rows(named=True):
            label = " ~ ".join(
                f"{r[keys[i]]}.{r[keys[i + 1]]}" for i in range(0, len(keys), 2)
            )
            bound = self.RTOL * (r["h_a"] + r["h_b"] + r["h_ab"]) + self.ATOL
            if r["got_mi"] is None or not abs(r["got_mi"] - r["mi"]) <= bound:
                problems.append(f"{label}: mi {r['got_mi']!r} != {r['mi']!r} ± {bound:.3g}")
            if math.isnan(r["nmi"]):
                continue
            nmi_bound = bound / min(r["h_a"], r["h_b"])
            if r["got_nmi"] is None or not abs(r["got_nmi"] - r["nmi"]) <= nmi_bound:
                problems.append(
                    f"{label}: nmi {r['got_nmi']!r} != {r['nmi']!r} ± {nmi_bound:.3g}"
                )
        return problems
```

Create `services/analytics/analytics/threeway_entropy/ams.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.pairwise_entropy.ams import check_num_samples
from analytics.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyAmsRust(ThreewayEntropy):
    """Rust extension `ams_threeway_joint_entropy`: one Chakrabarti–Cormode–McGregor sketch
    (`num_samples` estimators, O(num_samples) memory) per triplet, rayon-parallel; h_abc
    agrees with the reference within RTOL."""

    RTOL = 0.05
    ATOL = 1e-9

    def __init__(self, *, num_samples: int = 4096, seed: int = 0, **thresholds):
        super().__init__(**thresholds)
        check_num_samples(num_samples)
        self.num_samples = num_samples
        self.seed = seed

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(
                list(dict.fromkeys(c for k in group for _, c in k))
            )
            triplets = [tuple(c for _, c in k) for k in group]
            out = _plugin.ams_threeway_joint_entropy(
                df, triplets, self.num_samples, self.seed
            )
            h = {frozenset((a, b, c)): e for a, b, c, e in out.iter_rows()}
            parts.append(
                self.metrics_frame(
                    group,
                    {
                        "h_abc": [h[frozenset(t)] for t in triplets],
                        "n_rows": [df.height] * len(group),
                    },
                )
            )
        return pl.concat(parts)
```

Replace `services/analytics/analytics/pairwise_entropy/__init__.py` with:

```python
"""Pairwise joint entropy / mutual information (ordered scope). See PairwiseEntropy."""

from analytics.base import lazy_attributes
from analytics.pairwise_entropy.ams import PairwiseEntropyAmsRust
from analytics.pairwise_entropy.base import PairwiseEntropy
from analytics.pairwise_entropy.rust import PairwiseEntropyRust

REFERENCE = "PairwiseEntropyPolars"
IMPLEMENTATIONS = (
    "PairwiseEntropyRust",
    "PairwiseEntropyAmsRust",
    "PairwiseEntropyPolars",
)

__getattr__ = lazy_attributes(__name__, {"PairwiseEntropyPolars": ".polars"})
__all__ = [
    "PairwiseEntropy",
    "PairwiseEntropyRust",
    "PairwiseEntropyAmsRust",
    "PairwiseEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
```

Replace `services/analytics/analytics/threeway_entropy/__init__.py` with:

```python
"""Three-way joint entropy (ordered scope, arity 3). See ThreewayEntropy."""

from analytics.base import lazy_attributes
from analytics.threeway_entropy.ams import ThreewayEntropyAmsRust
from analytics.threeway_entropy.base import ThreewayEntropy
from analytics.threeway_entropy.rust import ThreewayEntropyRust

REFERENCE = "ThreewayEntropyPolars"
IMPLEMENTATIONS = (
    "ThreewayEntropyRust",
    "ThreewayEntropyAmsRust",
    "ThreewayEntropyPolars",
)

__getattr__ = lazy_attributes(__name__, {"ThreewayEntropyPolars": ".polars"})
__all__ = [
    "ThreewayEntropy",
    "ThreewayEntropyRust",
    "ThreewayEntropyAmsRust",
    "ThreewayEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
```

- [ ] **Step 4: Run the entropy tests, then the generic plugin tests**

Run: `$PY -m pytest tests/test_pairwise_entropy.py tests/test_threeway_entropy.py tests/test_plugin.py -q`
Expected: all pass. `test_plugin.py` now also runs the AMS classes through the 128-bit-ineligibility, Arrow-input-equality and sliced-Array/Struct checks.

If `test_agrees_with_reference[PairwiseEntropyAmsRust]` or `[ThreewayEntropyAmsRust]` fails, the 5% tolerance at `num_samples=4096` doesn't hold on `mixed_dtypes(1_000)`, which is a finding about the estimator. **Do not** change the seed, `RTOL` or the sample size to make it pass. Record the printed mismatches (worst entropies and their columns) and report them to the user.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/pairwise_entropy services/analytics/analytics/threeway_entropy tests/test_pairwise_entropy.py tests/test_threeway_entropy.py
git commit -m "PairwiseEntropyAmsRust / ThreewayEntropyAmsRust on the uniform contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `datagen.near_unique`

**Files:**
- Modify: `tests/datagen.py`
- Modify: `tests/test_datagen.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_datagen.py`, add `near_unique` to the `from datagen import (...)` list (keep alphabetical order, after `mixed_dtypes`) and append:

```python
def test_near_unique():
    df = near_unique(10_000)
    assert df.columns == ["u0", "u1", "u2", "key_hi", "key_lo", "skewed", "mid", "low"]
    assert df.select(pl.struct("key_hi", "key_lo").n_unique()).item() == 10_000
    assert 9_985 <= df["u0"].n_unique() < 10_000
    assert 0.93 < (df["skewed"] == 0).mean() < 0.97
    assert df["low"].n_unique() == 10
    assert near_unique(1_000).equals(near_unique(1_000))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_datagen.py -n 0 -q`
Expected: `ImportError: cannot import name 'near_unique'`.

- [ ] **Step 3: Implement the generator**

In `tests/datagen.py`, change the module docstring line `Ordered techniques:   mixed_dtypes, low_cardinality` to `Ordered techniques:   mixed_dtypes, low_cardinality, near_unique`, and add after `low_cardinality`:

```python
def near_unique(n_rows: int = 1_000_000, seed: int = 42) -> pl.DataFrame:
    """Int64 columns that stress entropy sketches and the exact kernels' hash-map path:
    u0–u2 near-unique (a permutation with ~0.1% of rows copying another row's value), a
    genuine composite key key_hi = i // 1000, key_lo = i % 1000 over shuffled i (unique
    together), `skewed` (95% one value, the rest over 50,000 others), and `mid` / `low`
    with ~1,000 / 10 values."""
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(3):
        values = rng.permutation(n_rows)
        dup = rng.choice(n_rows, n_rows // 1000, replace=False)
        values[dup] = values[rng.integers(0, n_rows, dup.size)]
        cols[f"u{i}"] = values
    perm = rng.permutation(n_rows)
    cols["key_hi"] = perm // 1000
    cols["key_lo"] = perm % 1000
    cols["skewed"] = np.where(
        rng.random(n_rows) < 0.95, 0, rng.integers(1, 50_001, n_rows)
    )
    cols["mid"] = rng.integers(0, 1_000, n_rows)
    cols["low"] = rng.integers(0, 10, n_rows)
    return pl.DataFrame({k: v.astype(np.int64) for k, v in cols.items()})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_datagen.py -n 0 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/datagen.py tests/test_datagen.py
git commit -m "datagen.near_unique: near-unique, composite-key and skewed columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Benchmarks — speed datasets and the memory / error report

**Files:**
- Modify: `tests/performance/benchmark_pairwise_entropy.py`
- Modify: `tests/performance/benchmark_threeway_entropy.py`
- Create: `tests/performance/benchmark_ams_entropy.py`

- [ ] **Step 1: Add the near-unique dataset to the speed benchmarks**

Replace `tests/performance/benchmark_pairwise_entropy.py` with:

```python
"""
Pairwise joint entropy speed benchmark: PairwiseEntropyRust and PairwiseEntropyAmsRust vs
PairwiseEntropyPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_pairwise_entropy.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, near_unique

if __name__ == "__main__":
    run(
        "analytics.pairwise_entropy",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
            Dataset("near-unique 1M x 8", lambda: {"t": near_unique()}),
        ],
    )
```

Replace `tests/performance/benchmark_threeway_entropy.py` with:

```python
"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust and ThreewayEntropyAmsRust vs
ThreewayEntropyPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_threeway_entropy.py

large_dataset.arrow is cut to its first 40 columns (9,880 triplets): all 101 columns
give 166,650 triplets (~65 s per Rust run), far past the reference's budget.
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, near_unique

if __name__ == "__main__":
    run(
        "analytics.threeway_entropy",
        [
            large_dataset(columns=40),
            # All 101 columns -> C(101, 3) = 166,650 triplets. Neither the one-triplet-at-a-
            # time Polars reference nor a 4096-estimator sketch per triplet finishes within
            # budget, so only the exact Rust implementation is timed here.
            large_dataset(exclude=("ThreewayEntropyPolars", "ThreewayEntropyAmsRust")),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
            Dataset("near-unique 1M x 8", lambda: {"t": near_unique()}),
        ],
        runs=3,
    )
```

- [ ] **Step 2: Create the memory / error report**

Create `tests/performance/benchmark_ams_entropy.py`:

```python
"""
AMS entropy memory and error report: every implementation of analytics.pairwise_entropy
and analytics.threeway_entropy on the speed benchmarks' datasets (near-unique 1M x 8
included).

Each (dataset, implementation) runs once in a fresh child process, which reports the peak
working set its result() added above the peak after loading the data (`peak_mb`; Windows
PeakWorkingSetSize, elsewhere ru_maxrss). The parent compares each result with the
reference's: max_rel_err / p95_rel_err over every computed entropy, max_abs_mi_err
(pairwise only).

Both Rust implementations hold the encoded columns (~9 B per row per column), so peak_mb
differences show the counting state only; the sketch's advantage of not keeping rows at
all is a streaming one and is not measured here.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_ams_entropy.py
Results -> tests/performance/results/ams_entropy.parquet
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
from harness import RESULTS_DIR, Dataset, large_dataset

from datagen import low_cardinality, near_unique

DATASETS = {
    "analytics.pairwise_entropy": [
        large_dataset(),
        Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
        Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        Dataset("near-unique 1M x 8", lambda: {"t": near_unique()}),
    ],
    "analytics.threeway_entropy": [
        large_dataset(columns=40),
        Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
        Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
        Dataset("near-unique 1M x 8", lambda: {"t": near_unique()}),
    ],
}
ENTROPIES = {
    "analytics.pairwise_entropy": ["h_a", "h_b", "h_ab"],
    "analytics.threeway_entropy": ["h_abc"],
}


def peak_mb() -> float:
    """This process's peak working set (resident set) so far, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if sys.platform == "darwin" else 2**10)  # bytes on macOS, KiB on Linux


def child(package: str, index: int, name: str, out: str) -> None:
    """Load the dataset, run the implementation once, write its result to `out` and print
    the peak working set it added."""
    frames = DATASETS[package][index].make()
    before = peak_mb()
    cls = getattr(importlib.import_module(package), name)
    result = cls().add(frames).result()
    after = peak_mb()
    result.write_parquet(out)
    print(json.dumps({"peak_mb": after - before}))


def errors(package: str, got: pl.DataFrame, want: pl.DataFrame) -> dict:
    """Relative error of every computed entropy; absolute MI error (pairwise only)."""
    computed = want["status"] == "computed"
    got, want = got.filter(computed), want.filter(computed)
    rel = pl.concat(
        [((got[c] - want[c]).abs() / want[c]).fill_nan(0.0) for c in ENTROPIES[package]]
    )  # 0/0 (both exactly 0) -> 0; x/0 -> inf
    mi = (got["mi"] - want["mi"]).abs().max() if "mi" in want.columns else None
    return {
        "max_rel_err": rel.max(),
        "p95_rel_err": rel.quantile(0.95),
        "max_abs_mi_err": mi,
    }


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for package, datasets in DATASETS.items():
            module = importlib.import_module(package)
            names = [module.REFERENCE] + [
                n for n in module.IMPLEMENTATIONS if n != module.REFERENCE
            ]
            for index, ds in enumerate(datasets):
                results = {}
                for name in names:
                    print(f"  {package} | {ds.name}: {name} …", file=sys.stderr, flush=True)
                    out = str(Path(tmp) / f"{name}.parquet")
                    proc = subprocess.run(
                        [sys.executable, __file__, "--child", package, str(index), name, out],
                        stdout=subprocess.PIPE,
                        text=True,
                        check=True,
                    )
                    peak = json.loads(proc.stdout.strip().splitlines()[-1])["peak_mb"]
                    results[name] = pl.read_parquet(out)
                    err = {"max_rel_err": None, "p95_rel_err": None, "max_abs_mi_err": None}
                    if name != module.REFERENCE:
                        err = errors(package, results[name], results[module.REFERENCE])
                    rows.append(
                        {
                            "package": package.split(".")[-1],
                            "dataset": ds.name,
                            "implementation": name,
                            "peak_mb": peak,
                            **err,
                        }
                    )
    df = pl.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_parquet(RESULTS_DIR / "ams_entropy.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        main()
```

- [ ] **Step 3: Lint**

Run: `$PY scripts/check.py fast`
Expected: clean. If black or ruff reformats or flags anything, run `$PY scripts/check.py fmt`, fix the remaining ruff findings and re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/performance/benchmark_pairwise_entropy.py tests/performance/benchmark_threeway_entropy.py tests/performance/benchmark_ams_entropy.py
git commit -m "Benchmarks: near-unique 1M x 8 dataset; AMS entropy memory and error report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Run the benchmarks, document, full check

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the three benchmarks (long-running; use `run_in_background`)**

Run each from the repo root and keep the printed tables:

```bash
$PY tests/performance/benchmark_pairwise_entropy.py
$PY tests/performance/benchmark_threeway_entropy.py
$PY tests/performance/benchmark_ams_entropy.py
```

Expected: each ends with a printed table and writes `tests/performance/results/*.parquet`. The speed tables include `PairwiseEntropyAmsRust` / `ThreewayEntropyAmsRust` rows at N and 1 thread, with algorithmic / parallel / total speedups and an `agrees` column. The memory report lists `peak_mb`, `max_rel_err`, `p95_rel_err` and `max_abs_mi_err` per dataset and implementation. An implementation reported as `skipped: over budget` is a result to report, not an error.

- [ ] **Step 2: Update CLAUDE.md**

In the **Joint entropy** paragraph under "## 3. Ordered", change the class lists to `(`PairwiseEntropyRust`, `PairwiseEntropyAmsRust`, ★`PairwiseEntropyPolars`)` and `(`ThreewayEntropyRust`, `ThreewayEntropyAmsRust`, ★`ThreewayEntropyPolars`)`, and append to that paragraph:

```markdown
The `*AmsRust` implementations estimate every entropy with a Chakrabarti–Cormode–McGregor AMS-sampling sketch (spec: docs/superpowers/specs/2026-10-11-ams-entropy-design.md; techniques/single_column/unordered/ams_entropy.rs, pairwise/ordered/ams_joint_entropy.rs): one pass, O(`num_samples`) memory per column / combination whatever the cardinality, a backup sample plus 256 Misra–Gries counters for a dominant item, skip-ahead label events and a waiting set so per-row work is O(1). Keywords `num_samples=4096`, `seed=0`. Agreement: entropies within 5%; MI within 5%·(h_a + h_b + h_ab), NMI within that ÷ min(h_a, h_b); conclusions not compared. Benchmarks: tests/performance/benchmark_ams_entropy.py (memory + error) and the two entropy speed benchmarks (near-unique 1M x 8 dataset).
```

In the project-structure tree, add `ams_entropy` to `single_column/unordered/{...}` and `ams_joint_entropy` to `pairwise/ordered/{...}`. In the "Private — reached only through `analytics._plugin`" list, add `ams_marginal_entropy`, `ams_pairwise_joint_entropy`, `ams_threeway_joint_entropy` after `threeway_joint_entropy (…)`.

- [ ] **Step 3: Run the full check**

Run: `$PY scripts/check.py check`
Expected: fmt-check, lint, layering, build, `cargo test --lib` and the full pytest suite all pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Document the AMS entropy implementations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report the results to the user**

Summarise from the three tables:
- **Accuracy:** per dataset, the AMS `max_rel_err`, `p95_rel_err` and `max_abs_mi_err`, and whether `agrees` held.
- **Speed:** AMS vs exact Rust vs Polars medians, and AMS's algorithmic / parallel / total speedups.
- **Memory:** `peak_mb` for each implementation.
- **The answer to the spec's question:** whether the sketch can stand in for exact counts when pruning primary-key candidates, given its MI error.

Quote the numbers, and don't round failures into successes.
