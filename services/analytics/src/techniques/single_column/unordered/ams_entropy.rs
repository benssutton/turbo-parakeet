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
                    self.estimators[e].primary = Some(Sample {
                        key: x,
                        label,
                        offset,
                    });
                } else {
                    let label = p.label + self.rng.unit0() * (1.0 - p.label);
                    self.estimators[e].backup = Some(Sample {
                        key: x,
                        label,
                        offset,
                    });
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
                self.estimators[e].primary = Some(Sample {
                    key: x,
                    label,
                    offset,
                });
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
                        self.estimators[e].primary = Some(Sample {
                            key: x,
                            label,
                            offset,
                        });
                    }
                } else if x != p.key {
                    self.release(b.key);
                    let offset = self.hold(x);
                    self.estimators[e].backup = Some(Sample {
                        key: x,
                        label,
                        offset,
                    });
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
