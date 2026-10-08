//! Bottom-k sample of distinct values (spec 2026-10-01 §3.2): the k values with the
//! smallest `hll::hash_key`, each with its count, first row, capture mask and length. A
//! value is admitted on its first occurrence when its hash is below the largest kept one;
//! that bound only falls, so every value still held has been counted since its first row
//! and its count, first row and mask are exact.
//!
//! Until the first eviction the sample holds every distinct value (the exact phase) and
//! also keeps what the dictionary rules need: `few`, `views`, `sum_len_unique` and, for text
//! levels, each value's text (`ranking`, spec 2026-10-07 §4).

use std::collections::{BinaryHeap, HashMap};

use foldhash::fast::FixedState;

use crate::recommenders::streaming::partial::{KeyStat, TextSource, ViewSim};
use crate::techniques::describe::{ranked_order, Ranking};

/// Sample size for a `categorical_threshold`: at least 1000, so the estimators have
/// data however low the threshold.
pub(crate) fn sample_size(threshold: u64) -> usize {
    threshold.max(1_000) as usize
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    count: u64,
    /// Global row of the value's first occurrence.
    first: u64,
    len: u64,
    /// Capture subsets the value occurred in (bits 0–2).
    mask: u8,
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
    /// Exact phase, text levels: every held value's text, read once on admission.
    texts: HashMap<u64, Box<str>, FixedState>,
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
            texts: HashMap::default(),
        }
    }

    /// `keys`: one batch's distinct values; in first-occurrence order while exact on a
    /// text / binary level (`few` and `views` read it), any order otherwise. `text` (a text
    /// level in the exact phase) gives each newly admitted value's text, read once at its
    /// first row.
    pub(crate) fn absorb(&mut self, keys: Vec<KeyStat>, text: Option<&TextSource>) {
        for k in keys {
            // Sampling: every held value hashes at or below the top, so one above it is
            // neither held nor admitted; skip it before the map lookup.
            if self.sampling && self.heap.peek().is_some_and(|&(top, _)| k.hash > top) {
                continue;
            }
            if let Some(s) = self.map.get_mut(&k.key) {
                s.count += k.count;
                s.mask |= k.mask;
                continue;
            }
            let h = k.hash;
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
                    count: k.count,
                    first: k.first,
                    len: k.len,
                    mask: k.mask,
                },
            );
            self.heap.push((h, k.key));
            if !self.sampling {
                self.admit_exact(k, text);
            }
        }
    }

    /// Exact phase: a newly admitted value's text, length and `few` entry.
    fn admit_exact(&mut self, k: KeyStat, text: Option<&TextSource>) {
        if let Some(src) = text {
            self.texts.insert(k.key, src.at(k.first).into_boxed_str());
        }
        self.sum_len_unique += k.len;
        self.views.push(k.len);
        match k.text {
            Some(t) if self.map.len() <= 5 => self.few.push(t),
            _ => self.few.clear(),
        }
    }

    /// The first value not held ends the exact phase and its exact-only statistics.
    fn enter_sampling(&mut self) {
        if !self.sampling {
            self.sampling = true;
            self.few.clear();
            self.views = ViewSim::default();
            self.sum_len_unique = 0;
            self.texts = HashMap::default();
        }
    }

    /// Sampling phase: the largest held hash; no value above it is held or admitted.
    pub(crate) fn top(&self) -> Option<u64> {
        if self.sampling {
            self.heap.peek().map(|&(h, _)| h)
        } else {
            None
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
            f1 += (s.count == 1) as u64;
            f2 += (s.count == 2) as u64;
            h[s.mask as usize - 1] += 1;
        }
        (f1, f2, h)
    }

    /// Every held value occurred exactly once.
    pub(crate) fn all_once(&self) -> bool {
        self.map.values().all(|s| s.count == 1)
    }

    /// Exact phase: every held value by frequency (count descending, ties to the earlier
    /// first row), as one-shot's `Profile.ranking`. None in the sampling phase, or when a held
    /// value has no text (not a text level).
    pub(crate) fn ranking(&self) -> Option<Ranking> {
        if self.sampling {
            return None;
        }
        let mut held: Vec<(u64, u64, u64)> = self
            .map
            .iter()
            .map(|(&key, s)| (s.count, s.first, key))
            .collect();
        held.sort_unstable_by(|a, b| ranked_order(&(a.0, a.1), &(b.0, b.1)));
        held.into_iter()
            .map(|(count, _, key)| self.texts.get(&key).map(|t| (t.to_string(), count)))
            .collect()
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
    use crate::recommenders::streaming::partial::TextSource;
    use crate::techniques::hll::hash_key;
    use polars::prelude::{NamedFrom, Series};

    fn ks(key: u64, count: u64, mask: u8, text: Option<&str>) -> KeyStat {
        KeyStat {
            key,
            hash: hash_key(key),
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
        d.absorb((0..100).map(|k| ks(k, k % 3 + 1, 1, None)).collect(), None);
        assert!(d.is_exact());
        assert_eq!(d.len(), 100);
        let (f1, f2, h) = d.counts();
        assert_eq!((f1, f2, h[0]), (34, 33, 100));
        assert_eq!(d.sum_len_unique, 2_000);
    }

    #[test]
    fn counts_accumulate_and_masks_merge() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(vec![ks(1, 1, 1, None)], None);
        d.absorb(vec![ks(1, 5, 2, None)], None);
        assert_eq!(d.counts(), (0, 0, [0, 0, 1, 0, 0, 0, 0]));
        assert!(!d.all_once());
    }

    #[test]
    fn counts_and_first_rows_are_exact_across_batches() {
        let at = |key, count, first| KeyStat {
            first,
            ..ks(key, count, 1, None)
        };
        let mut d = DistinctSample::new(1_000);
        d.absorb(vec![at(1, 2, 5), at(2, 1, 7)], None);
        d.absorb(vec![at(2, 4, 9), at(3, 1, 10)], None);
        let slot = |k: u64| (d.map[&k].count, d.map[&k].first);
        assert_eq!((slot(1), slot(2), slot(3)), ((2, 5), (5, 7), (1, 10)));
        assert_eq!(d.counts().0, 1); // f1: only key 3 occurred once
    }

    #[test]
    fn texts_are_read_once_on_admission_and_dropped_on_overflow() {
        // Global rows 100..103 hold "x", "y", "x".
        let src = TextSource::new(Series::new("s".into(), &["x", "y", "x"]), 100);
        let mut d = DistinctSample::new(1_000);
        d.absorb(
            vec![
                KeyStat {
                    first: 100,
                    ..ks(1, 2, 1, None)
                },
                KeyStat {
                    first: 101,
                    ..ks(2, 1, 1, None)
                },
            ],
            Some(&src),
        );
        assert_eq!(
            d.ranking(),
            Some(vec![("x".to_string(), 2), ("y".into(), 1)])
        );
        // Later occurrences only add counts: no text source is needed.
        d.absorb(
            vec![KeyStat {
                first: 500,
                ..ks(2, 3, 1, None)
            }],
            None,
        );
        assert_eq!(
            d.ranking(),
            Some(vec![("y".to_string(), 4), ("x".into(), 2)])
        );
        d.absorb(batch(10..2_000), None); // past k: sampling
        assert_eq!(d.ranking(), None);
        assert!(d.texts.is_empty());
    }

    #[test]
    fn a_held_value_without_text_means_no_ranking() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..3), None);
        assert_eq!(d.ranking(), None);
        assert_eq!(DistinctSample::new(1_000).ranking(), Some(vec![]));
    }

    #[test]
    fn few_values_kept_while_exact() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(vec![ks(1, 1, 1, Some("a")), ks(2, 1, 1, Some("b"))], None);
        assert_eq!(d.few, ["a", "b"]);
        d.absorb(batch(10..20), None);
        assert!(d.few.is_empty());
    }

    #[test]
    fn bounded_and_sampling_past_k() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..50_000), None);
        assert_eq!(d.len(), 1_000);
        assert!(!d.is_exact() && d.all_once());
        assert_eq!((d.sum_len_unique, d.views.bytes()), (0, 0));
        assert!(d.few.is_empty());
    }

    #[test]
    fn keeps_the_smallest_hashes_whatever_the_order() {
        let (mut ab, mut ba) = (DistinctSample::new(1_000), DistinctSample::new(1_000));
        ab.absorb(batch(0..30_000), None);
        ab.absorb(batch(20_000..60_000), None);
        ba.absorb(batch(20_000..60_000), None);
        ba.absorb(batch(0..30_000), None);
        assert_eq!(ab.keys(), ba.keys());
        assert_eq!(ab.counts(), ba.counts());
        let held = ab.keys();
        let f2 = held
            .iter()
            .filter(|k| (20_000..30_000).contains(*k))
            .count() as u64;
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
        d.absorb((0..100_000).map(|k| ks(k, 1, 1, None)).collect(), None);
        d.absorb(
            (100_000..200_000).map(|k| ks(k, 2, 1, None)).collect(),
            None,
        );
        let (f1, f2, _) = d.counts();
        let n = d.len() as f64;
        assert!((f1 as f64 / n - 0.5).abs() < 0.03, "f1={f1}");
        assert!((f2 as f64 / n - 0.5).abs() < 0.03, "f2={f2}");
    }

    #[test]
    fn boundary_at_exactly_k() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..1_000), None);
        assert!(d.is_exact());
        assert_eq!(d.len(), 1_000);
        d.absorb(batch(1_000..1_001), None);
        assert!(!d.is_exact());
        assert_eq!(d.len(), 1_000);
    }

    #[test]
    fn rejected_key_is_not_readmitted() {
        let mut d = DistinctSample::new(1_000);
        d.absorb(batch(0..5_000), None);
        let held = d.keys();
        let gone = (0..5_000).find(|k| held.binary_search(k).is_err()).unwrap();
        let counts = d.counts();
        d.absorb(vec![ks(gone, 1, 1, None)], None);
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
        d.absorb(vec![a, b], None);
        assert_eq!(d.mean_len(), 20.0);
    }

    #[test]
    fn full_mask_lands_in_last_history_bucket() {
        let mut d = DistinctSample::new(1_000);
        for m in [1, 2, 4] {
            d.absorb(vec![ks(7, 1, m, None)], None);
        }
        assert_eq!(d.counts().2[6], 1);
    }
}
