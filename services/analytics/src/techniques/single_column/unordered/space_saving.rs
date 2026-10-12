//! SpaceSaving (Metwally, Agrawal, El Abbadi, ICDT 2005): m counters; a new key replaces the
//! minimum counter and inherits its count as an over-count bound ε, so a tracked key's true
//! count lies in [count − ε, count] and every key above n/m is tracked. Mergeable with the
//! bound intact (Agarwal et al., "Mergeable Summaries", PODS 2012), so the kernel counts fixed
//! row chunks in parallel and merges them in order. Spec 2026-10-12 §3.2.

use polars::prelude::*;
use rayon::prelude::*;

use crate::techniques::top_k::{
    encoded, rank_order, top_k_struct, ColumnTop, Counters, Entry, Item, MAX_MEMORY_BYTES,
};

/// Rows per chunk of the parallel kernel: fixed, so results do not depend on thread count.
pub(crate) const CHUNK_ROWS: usize = 1 << 20;

pub(crate) struct SpaceSaving {
    k: usize,
    counters: Counters,
}

impl SpaceSaving {
    /// The most counters that fit `memory_bytes`; at least `k` (≥ 1) are required.
    pub(crate) fn new(k: usize, memory_bytes: usize) -> Result<Self, String> {
        if k == 0 {
            return Err("k must be at least 1".into());
        }
        if memory_bytes > MAX_MEMORY_BYTES {
            return Err(format!(
                "memory_bytes = {memory_bytes} exceeds the 4 GiB ({MAX_MEMORY_BYTES} bytes) limit"
            ));
        }
        let m = Counters::capacity_for(memory_bytes);
        if m < k {
            return Err(format!(
                "memory_bytes = {memory_bytes} holds {m} SpaceSaving counters, fewer than k = {k}"
            ));
        }
        Ok(SpaceSaving {
            k,
            counters: Counters::new(m),
        })
    }

    pub(crate) fn insert(&mut self, key: u64, row: u64, weight: u64) {
        let c = &mut self.counters;
        if let Some(id) = c.find(key) {
            let e = *c.get(id);
            c.raise(id, e.count + weight, e.error);
        } else if !c.is_full() {
            c.push(Entry {
                key,
                count: weight,
                error: 0,
                row,
            });
        } else {
            let min = c.min().expect("a full summary has a minimum").count;
            c.replace_min(Entry {
                key,
                count: min + weight,
                error: min,
                row,
            });
        }
    }

    /// The summary of both streams. An untracked key's true count in a full summary is at most
    /// its minimum (0 in a summary with room), so a side missing a key adds that minimum to both
    /// count and error; the m largest are kept.
    pub(crate) fn merge(&mut self, other: &SpaceSaving) {
        assert_eq!(
            self.counters.capacity(),
            other.counters.capacity(),
            "capacities differ"
        );
        let floor = |s: &SpaceSaving| match s.counters.min() {
            Some(e) if s.counters.is_full() => e.count,
            _ => 0,
        };
        let (fa, fb) = (floor(self), floor(other));
        let (a, b) = (&self.counters, &other.counters);
        let mut all: Vec<Entry> = a
            .entries()
            .iter()
            .map(|x| match b.find(x.key) {
                Some(id) => {
                    let y = b.get(id);
                    Entry {
                        key: x.key,
                        count: x.count + y.count,
                        error: x.error + y.error,
                        row: x.row.min(y.row),
                    }
                }
                None => Entry {
                    count: x.count + fb,
                    error: x.error + fb,
                    ..*x
                },
            })
            .collect();
        all.extend(
            b.entries()
                .iter()
                .filter(|y| a.find(y.key).is_none())
                .map(|y| Entry {
                    count: y.count + fa,
                    error: y.error + fa,
                    ..*y
                }),
        );
        all.sort_unstable_by(rank_order);
        all.truncate(a.capacity());
        let mut merged = Counters::new(a.capacity());
        for e in all.into_iter().rev() {
            merged.push(e);
        }
        self.counters = merged;
    }

    /// The k largest, in report order, with their bounds.
    pub(crate) fn top(&self) -> Vec<Item> {
        self.counters
            .ranked()
            .into_iter()
            .take(self.k)
            .map(|e| Item {
                key: e.key,
                row: e.row,
                count: e.count,
                error: Some(e.error),
            })
            .collect()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.counters.bytes()
    }
}

fn sketch(k: usize, memory_bytes: usize) -> PolarsResult<SpaceSaving> {
    SpaceSaving::new(k, memory_bytes).map_err(|e| PolarsError::ComputeError(e.into()))
}

/// Each column's top `k` by SpaceSaving: `CHUNK_ROWS` chunks summarised in parallel (rayon
/// across columns and chunks) and merged in chunk order. `sketch_bytes` is one summary's.
pub(crate) fn space_saving_top_k_impl(
    inputs: &[Series],
    k: usize,
    memory_bytes: usize,
) -> PolarsResult<Series> {
    sketch(k, memory_bytes)?; // validate the parameters once, before the parallel section
    let keys = encoded(inputs)?;
    let tops = keys
        .par_iter()
        .map(|col| -> PolarsResult<ColumnTop> {
            let n = col.len();
            let n_chunks = n.div_ceil(CHUNK_ROWS);
            let wave = rayon::current_num_threads().max(1);
            let mut s = sketch(k, memory_bytes)?;
            // Waves of chunks keep the peak at (wave × budget) while merging stays in chunk order.
            for first in (0..n_chunks).step_by(wave) {
                let parts = (first..(first + wave).min(n_chunks))
                    .into_par_iter()
                    .map(|c| {
                        let (lo, hi) = (c * CHUNK_ROWS, ((c + 1) * CHUNK_ROWS).min(n));
                        let mut p = sketch(k, memory_bytes)?;
                        let rows = col.values[lo..hi].iter().zip(&col.is_null[lo..hi]);
                        for (i, (&v, &null)) in rows.enumerate() {
                            if !null {
                                p.insert(v, (lo + i) as u64, 1);
                            }
                        }
                        Ok(p)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                for p in parts {
                    s.merge(&p);
                }
            }
            Ok(ColumnTop {
                items: s.top(),
                n_non_null: col.is_null.iter().filter(|&&null| !null).count() as u64,
                sketch_bytes: s.bytes() as u64,
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    top_k_struct("space_saving_top_k", inputs, &tops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techniques::top_k::streams::{exact_counts, zipf_stream};
    use std::collections::HashMap;

    const BUDGET: usize = 16 * 1024; // 307 counters

    fn fed(stream: &[u64], offset: u64) -> SpaceSaving {
        let mut s = SpaceSaving::new(5, BUDGET).unwrap();
        for (i, &k) in stream.iter().enumerate() {
            s.insert(k, offset + i as u64, 1);
        }
        s
    }

    /// Every tracked key's bound holds and every key above n/m is tracked.
    fn check_guarantees(s: &SpaceSaving, truth: &HashMap<u64, u64>, n: u64) {
        let m = s.counters.capacity() as u64;
        for e in s.counters.entries() {
            let t = truth.get(&e.key).copied().unwrap_or(0);
            assert!(e.count - e.error <= t && t <= e.count, "{e:?} true {t}");
        }
        for (&k, &t) in truth {
            if t > n / m {
                assert!(s.counters.find(k).is_some(), "key {k} ({t} > n/m) dropped");
            }
        }
    }

    #[test]
    fn exact_while_keys_fit() {
        let stream: Vec<u64> = (0..100u64)
            .flat_map(|k| std::iter::repeat_n(k, k as usize + 1))
            .collect();
        let s = fed(&stream, 0);
        let top = s.top();
        let counts: Vec<(u64, u64, Option<u64>)> =
            top.iter().map(|i| (i.key, i.count, i.error)).collect();
        assert_eq!(
            counts,
            [
                (99, 100, Some(0)),
                (98, 99, Some(0)),
                (97, 98, Some(0)),
                (96, 97, Some(0)),
                (95, 96, Some(0))
            ]
        );
    }

    #[test]
    fn bounds_hold_on_a_zipf_stream() {
        let stream = zipf_stream(200_000, 5_000, 1.1, 7);
        let s = fed(&stream, 0);
        let truth = exact_counts(&stream);
        check_guarantees(&s, &truth, stream.len() as u64);
        let mut want: Vec<(u64, u64)> = truth.iter().map(|(&k, &c)| (c, k)).collect();
        want.sort_unstable_by(|a, b| b.cmp(a));
        let got: Vec<u64> = s.top().iter().map(|i| i.key).collect();
        let want: Vec<u64> = want.iter().take(5).map(|&(_, k)| k).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn merge_keeps_the_guarantees() {
        let stream = zipf_stream(200_000, 5_000, 0.9, 11);
        let (lo, hi) = stream.split_at(stream.len() / 2);
        let mut a = fed(lo, 0);
        a.merge(&fed(hi, lo.len() as u64));
        check_guarantees(&a, &exact_counts(&stream), stream.len() as u64);
    }

    #[test]
    fn merge_of_three_parts_keeps_the_guarantees() {
        let stream = zipf_stream(150_000, 5_000, 1.0, 13);
        let third = stream.len() / 3;
        let mut a = fed(&stream[..third], 0);
        a.merge(&fed(&stream[third..2 * third], third as u64));
        a.merge(&fed(&stream[2 * third..], 2 * third as u64));
        check_guarantees(&a, &exact_counts(&stream), stream.len() as u64);
    }

    #[test]
    fn full_merged_with_room_keeps_the_guarantees() {
        let stream = zipf_stream(100_050, 5_000, 1.1, 17);
        let (big, small) = stream.split_at(100_000);
        let mut a = fed(big, 0);
        assert!(a.counters.is_full());
        a.merge(&fed(small, 100_000));
        check_guarantees(&a, &exact_counts(&stream), stream.len() as u64);
    }

    #[test]
    fn merge_with_room_is_exact() {
        let mut a = fed(&[1, 1, 2], 0);
        a.merge(&fed(&[2, 3], 3));
        let got: Vec<(u64, u64, u64)> = a
            .counters
            .ranked()
            .iter()
            .map(|e| (e.key, e.count, e.row))
            .collect();
        assert_eq!(got, [(1, 2, 0), (2, 2, 2), (3, 1, 4)]);
        assert!(a.counters.entries().iter().all(|e| e.error == 0));
    }

    #[test]
    fn a_weighted_insert_equals_repeated_inserts() {
        let stream = zipf_stream(20_000, 2_000, 1.0, 3);
        let (mut a, mut b) = (fed(&stream, 0), fed(&stream, 0));
        a.insert(424_242, 1, 5);
        for _ in 0..5 {
            b.insert(424_242, 1, 1);
        }
        assert_eq!(a.counters.ranked(), b.counters.ranked());
    }

    #[test]
    fn the_chunked_kernel_is_deterministic_and_bounded() {
        let n = CHUNK_ROWS + 50_000;
        let stream = zipf_stream(n, 10_000, 1.1, 5);
        let s = Series::new(
            "x".into(),
            stream.iter().map(|&k| k as i64).collect::<Vec<_>>(),
        );
        let run = || space_saving_top_k_impl(std::slice::from_ref(&s), 5, BUDGET).unwrap();
        let (a, b) = (run(), run());
        assert!(a.equals_missing(&b));
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        assert!(a.equals_missing(&one));
        let out = a.struct_().unwrap();
        let counts = out.field_by_name("count").unwrap();
        let errors = out.field_by_name("error").unwrap();
        let truth = exact_counts(&stream);
        let rows = out.field_by_name("row").unwrap();
        for i in 0..out.len() {
            let key = stream[rows.u64().unwrap().get(i).unwrap() as usize];
            let (c, e) = (
                counts.u64().unwrap().get(i).unwrap(),
                errors.u64().unwrap().get(i).unwrap(),
            );
            assert!(c - e <= truth[&key] && truth[&key] <= c);
        }
        let reported: Vec<u64> = (0..out.len())
            .map(|i| stream[rows.u64().unwrap().get(i).unwrap() as usize])
            .collect();
        let mut by_count: Vec<(u64, u64)> = truth.iter().map(|(&k, &c)| (c, k)).collect();
        by_count.sort_unstable_by(|a, b| b.cmp(a));
        for &(c, k) in by_count.iter().take(5) {
            if c > n as u64 / 307 {
                assert!(reported.contains(&k), "key {k} ({c}) not reported");
            }
        }
        assert_eq!(
            out.field_by_name("n_non_null")
                .unwrap()
                .u64()
                .unwrap()
                .get(0),
            Some(n as u64)
        );
    }

    #[test]
    fn sizing() {
        let s = SpaceSaving::new(5, BUDGET).unwrap();
        assert_eq!(s.counters.capacity(), 307);
        assert!(s.bytes() <= BUDGET);
        assert!(SpaceSaving::new(0, BUDGET).is_err());
        assert!(SpaceSaving::new(5, 64).is_err());
        assert!(SpaceSaving::new(5, MAX_MEMORY_BYTES + 1).is_err());
        assert!(SpaceSaving::new(usize::MAX, BUDGET).is_err());
    }
}
