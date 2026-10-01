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
        debug_assert!(k > 0, "a sample holds at least one value");
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
            debug_assert!(k.mask != 0, "a counted value has a capture mask");
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
        let held = ab.keys();
        let f2 = held.iter().filter(|k| (20_000..30_000).contains(*k)).count() as u64;
        let (f1, f2_got, _) = ab.counts();
        assert_eq!((f1, f2_got), (held.len() as u64 - f2, f2));
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

    #[test]
    fn boundary_at_exactly_k() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..1_000));
        assert!(d.is_exact());
        assert_eq!(d.len(), 1_000);
        d.absorb(batch(1_000..1_001));
        assert!(!d.is_exact());
        assert_eq!(d.len(), 1_000);
    }

    #[test]
    fn rejected_key_is_not_readmitted() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..5_000));
        let held = d.keys();
        let gone = (0..5_000).find(|k| held.binary_search(k).is_err()).unwrap();
        let counts = d.counts();
        d.absorb(vec![ks(gone, 1, 1, None)]);
        assert_eq!(d.keys(), held);
        assert_eq!(d.counts(), counts);
    }

    #[test]
    fn mean_len_of_held_values() {
        let mut d = DistinctSample::new(1_000);
        assert_eq!(d.mean_len(), 0.0);
        let mut a = ks(1, 1, 1, None);
        a.len = 10;
        let mut b = ks(2, 1, 1, None);
        b.len = 30;
        d.absorb(vec![a, b]);
        assert_eq!(d.mean_len(), 20.0);
    }

    #[test]
    fn full_mask_lands_in_last_history_bucket() {
        let mut d = DistinctSample::new(1_000);
        for m in [1, 2, 4] {
            d.absorb(vec![ks(7, 1, m, None)]);
        }
        assert_eq!(d.counts().2[6], 1);
    }
}
