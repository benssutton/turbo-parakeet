// heavy_keeper_top_k_impl is called by bindings/api from Task 4; drop the allow then.
#![allow(dead_code)]

//! HeavyKeeper (Gong et al., "HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant
//! Flows", USENIX ATC 2018), final version: d = 2 arrays of (fingerprint, count) buckets with
//! count-with-exponential-decay (a different key decays a bucket with probability b^−C), plus a
//! min-heap of the top k with selective increment and fingerprint-collision detection. No
//! merge: a column is fed in row order. Spec 2026-10-12 §3.1.

use polars::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::techniques::ams_entropy::{mix64, name_key, sketch_seed};
use crate::techniques::top_k::{
    encoded, top_k_struct, ColumnTop, Counters, Entry, Item, SplitMix, MAX_MEMORY_BYTES,
};

/// The paper's decay base b.
const B: f64 = 1.08;
/// Arrays d.
const D: usize = 2;
/// Seed domain (the AMS sketches use 1–3).
const DOMAIN: u64 = 4;

#[derive(Clone, Copy, Default)]
struct Bucket {
    fp: u32,
    count: u32,
}

/// `t[C]` ≈ 2^64 · b^−C: a bucket of count C decays when a uniform u64 falls below it. The
/// table ends where b^−C < 2^−64 (C ≈ 577); larger counts never decay.
fn decay_thresholds() -> &'static [u64] {
    static T: OnceLock<Vec<u64>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![u64::MAX]; // C = 0: an empty bucket, never decayed
        for c in 1.. {
            let v = (B.powi(-c) * 2f64.powi(64)) as u64; // `as` saturates
            if v == 0 {
                break;
            }
            t.push(v);
        }
        t
    })
}

pub(crate) struct HeavyKeeper {
    width: usize,
    buckets: Vec<Bucket>,
    /// Hash salts: one per array, then the fingerprint's.
    salts: [u64; D + 1],
    rng: SplitMix,
    heap: Counters,
}

impl HeavyKeeper {
    /// The heap of `k` is paid for first; the rest of `memory_bytes` holds D × width buckets,
    /// width ≥ k.
    pub(crate) fn new(k: usize, memory_bytes: usize, seed: u64) -> Result<Self, String> {
        if k == 0 {
            return Err("k must be at least 1".into());
        }
        if memory_bytes > MAX_MEMORY_BYTES {
            return Err(format!(
                "memory_bytes = {memory_bytes} exceeds the 4 GiB ({MAX_MEMORY_BYTES} bytes) limit"
            ));
        }
        let rest = memory_bytes.saturating_sub(Counters::bytes_for(k));
        let width = rest / (D * std::mem::size_of::<Bucket>());
        if width < k {
            return Err(format!(
                "memory_bytes = {memory_bytes} leaves {width} HeavyKeeper buckets per array \
                 after the heap of k = {k}; at least k are needed"
            ));
        }
        let mut rng = SplitMix(seed);
        let salts = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
        Ok(HeavyKeeper {
            width,
            buckets: vec![Bucket::default(); D * width],
            salts,
            rng,
            heap: Counters::new(k),
        })
    }

    fn fingerprint(&self, key: u64) -> u32 {
        (mix64(key ^ self.salts[D]) >> 32) as u32
    }

    fn slot(&self, key: u64, j: usize) -> usize {
        j * self.width + (mix64(key ^ self.salts[j]) % self.width as u64) as usize
    }

    /// Algorithm 1 of the paper. A matching bucket increments only when the key is in the heap,
    /// the heap has room, or its count is ≤ n_min (selective increment); an outsider enters a
    /// full heap only at exactly n_min + 1 (fingerprint-collision detection).
    pub(crate) fn insert(&mut self, key: u64, row: u64) {
        let fp = self.fingerprint(key);
        let id = self.heap.find(key);
        let full = self.heap.is_full();
        let n_min = if full {
            self.heap.min().map_or(0, |e| e.count)
        } else {
            0
        };
        let free = id.is_some() || !full;
        let thresholds = decay_thresholds();
        let mut maxv = 0u64;
        for j in 0..D {
            let i = self.slot(key, j);
            let b = &mut self.buckets[i];
            if b.count == 0 {
                *b = Bucket { fp, count: 1 };
                maxv = maxv.max(1);
            } else if b.fp == fp {
                if free || u64::from(b.count) <= n_min {
                    b.count = b.count.saturating_add(1);
                }
                maxv = maxv.max(u64::from(b.count));
            } else if self.rng.next_u64() < thresholds.get(b.count as usize).copied().unwrap_or(0) {
                b.count -= 1;
                if b.count == 0 {
                    *b = Bucket { fp, count: 1 };
                    maxv = maxv.max(1);
                }
            }
        }
        let entry = Entry {
            key,
            count: maxv,
            error: 0,
            row,
        };
        match id {
            Some(id) => {
                if maxv > self.heap.get(id).count {
                    self.heap.raise(id, maxv, 0);
                }
            }
            None if maxv > 0 && !full => self.heap.push(entry),
            None if maxv > 0 && maxv == n_min + 1 => self.heap.replace_min(entry),
            None => {}
        }
    }

    /// The heap, in report order; HeavyKeeper has no per-item bound.
    pub(crate) fn top(&self) -> Vec<Item> {
        self.heap
            .ranked()
            .into_iter()
            .map(|e| Item {
                key: e.key,
                row: e.row,
                count: e.count,
                error: None,
            })
            .collect()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.heap.bytes() + self.buckets.capacity() * std::mem::size_of::<Bucket>()
    }
}

/// Each column's top `k` by HeavyKeeper, columns in parallel (rayon), each fed in row order.
/// A column's seed derives from `seed` and its name, as in the AMS sketches.
pub(crate) fn heavy_keeper_top_k_impl(
    inputs: &[Series],
    k: usize,
    memory_bytes: usize,
    seed: u64,
) -> PolarsResult<Series> {
    let keys = encoded(inputs)?;
    let tops = keys
        .par_iter()
        .zip(inputs)
        .map(|(col, s)| -> PolarsResult<ColumnTop> {
            let column_seed = sketch_seed(seed, DOMAIN, name_key(s.name()));
            let mut hk = HeavyKeeper::new(k, memory_bytes, column_seed)
                .map_err(|e| PolarsError::ComputeError(e.into()))?;
            let mut n = 0u64;
            for (i, (&v, &null)) in col.values.iter().zip(&col.is_null).enumerate() {
                if !null {
                    hk.insert(v, i as u64);
                    n += 1;
                }
            }
            Ok(ColumnTop {
                items: hk.top(),
                n_non_null: n,
                sketch_bytes: hk.bytes() as u64,
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    top_k_struct("heavy_keeper_top_k", inputs, &tops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techniques::top_k::streams::{exact_counts, zipf_stream};

    const BUDGET: usize = 16 * 1024;

    fn fed(k: usize, budget: usize, stream: &[u64], seed: u64) -> HeavyKeeper {
        let mut hk = HeavyKeeper::new(k, budget, seed).unwrap();
        for (i, &key) in stream.iter().enumerate() {
            hk.insert(key, i as u64);
        }
        hk
    }

    fn top(hk: &HeavyKeeper) -> Vec<(u64, u64)> {
        hk.top().iter().map(|i| (i.key, i.count)).collect()
    }

    /// Key i occurs 10·i times, interleaved; 20 keys in ~1,000 buckets per array rarely share a
    /// bucket in both arrays, so the top 5 are exact.
    #[test]
    fn exact_with_few_keys() {
        let mut stream: Vec<u64> = (1..=20u64)
            .flat_map(|k| std::iter::repeat_n(k, 10 * k as usize))
            .collect();
        let mut rng = SplitMix(1);
        for i in (1..stream.len()).rev() {
            stream.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let hk = fed(5, BUDGET, &stream, 0);
        assert_eq!(
            top(&hk),
            [(20, 200), (19, 190), (18, 180), (17, 170), (16, 160)]
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_top() {
        let stream = zipf_stream(100_000, 20_000, 0.9, 2);
        assert_eq!(
            top(&fed(5, BUDGET, &stream, 9)),
            top(&fed(5, BUDGET, &stream, 9))
        );
    }

    #[test]
    fn finds_the_top_5_of_a_zipf_stream() {
        let stream = zipf_stream(200_000, 10_000, 1.1, 4);
        let truth = exact_counts(&stream);
        let hk = fed(5, BUDGET, &stream, 0);
        let got: Vec<u64> = hk.top().iter().map(|i| i.key).collect();
        assert_eq!(got, [1000, 1001, 1002, 1003, 1004]);
        for i in hk.top() {
            let t = truth[&i.key] as f64;
            assert!((i.count as f64 - t).abs() / t < 0.05, "{i:?} true {t}");
        }
    }

    /// k = 1: A fills the heap at 10; C climbs while its bucket is ≤ n_min, enters at 11 and
    /// then counts freely.
    #[test]
    fn an_outsider_enters_at_n_min_plus_one() {
        let mut hk = HeavyKeeper::new(1, 1 << 20, 0).unwrap();
        for i in 0..10 {
            hk.insert(1, i);
        }
        assert_eq!(top(&hk), [(1, 10)]);
        for i in 0..30 {
            hk.insert(2, 10 + i);
        }
        assert_eq!(top(&hk), [(2, 30)]);
    }

    /// A key whose fingerprint collides with a large bucket it does not own (count > n_min)
    /// neither increments that bucket nor enters the heap.
    #[test]
    fn a_colliding_mouse_stays_out() {
        let mut hk = HeavyKeeper::new(1, 1 << 20, 0).unwrap();
        for i in 0..100 {
            hk.insert(1, i);
        }
        let mouse = 2;
        let fp = hk.fingerprint(mouse);
        for j in 0..D {
            let i = hk.slot(mouse, j);
            hk.buckets[i] = Bucket { fp, count: 500 };
        }
        hk.insert(mouse, 100);
        assert_eq!(top(&hk), [(1, 100)]);
        for j in 0..D {
            assert_eq!(hk.buckets[hk.slot(mouse, j)].count, 500);
        }
    }

    #[test]
    fn decay_is_exponential() {
        let t = decay_thresholds();
        assert_eq!(t[0], u64::MAX);
        let p1 = t[1] as f64 / 2f64.powi(64);
        assert!((p1 - 1.0 / 1.08).abs() < 1e-12);
        assert!(t.len() > 500 && t.len() < 600);
    }

    #[test]
    fn sizing() {
        let hk = HeavyKeeper::new(5, BUDGET, 0).unwrap();
        assert_eq!(hk.width, (BUDGET - Counters::bytes_for(5)) / 16);
        assert!(hk.bytes() <= BUDGET);
        assert!(HeavyKeeper::new(0, BUDGET, 0).is_err());
        assert!(HeavyKeeper::new(5, 300, 0).is_err());
        assert!(HeavyKeeper::new(5, MAX_MEMORY_BYTES + 1, 0).is_err());
    }

    #[test]
    fn the_kernel_counts_non_null_rows() {
        let s = Series::new("x".into(), [Some(1i64), None, Some(1), Some(2)]);
        let out = heavy_keeper_top_k_impl(&[s], 5, BUDGET, 0).unwrap();
        let out = out.struct_().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out.field_by_name("count").unwrap().u64().unwrap().get(0),
            Some(2)
        );
        assert_eq!(
            out.field_by_name("row").unwrap().u64().unwrap().get(0),
            Some(0)
        );
        assert_eq!(
            out.field_by_name("error").unwrap().u64().unwrap().get(0),
            None
        );
        assert_eq!(
            out.field_by_name("n_non_null")
                .unwrap()
                .u64()
                .unwrap()
                .get(0),
            Some(3)
        );
    }
}
