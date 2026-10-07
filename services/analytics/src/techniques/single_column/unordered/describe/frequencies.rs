//! Value frequencies: counts, the frequency map, capture histories (describe group A).

use crate::common::EncodedColumn;
use crate::techniques::hll::{hash_key, Hll};
use foldhash::fast::FixedState;
use rayon::prelude::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// describe — value frequencies (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// One foldhash map `key → (count, first row, split-subset mask)` per 64K-row
// chunk, built in parallel and merged (counts summed, first = min, masks OR-ed).
// One O(distinct) sweep of the merged map yields n_unique, f1/f2, the capture
// history and, while there are ≤ 5 distinct values, their first rows. Keys come
// from `encode_series` (floats canonicalised; strings, nested and struct values
// hashed — collisions ~6e-11 per pair at 50K rows, accepted as documented in
// CLAUDE.md).

pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub count: u64,
    pub first: u64,
    pub mask: u8,
}

pub(crate) type Map = HashMap<u64, Entry, FixedState>;
/// (first row index, value) of the running extreme.
pub(crate) type Ext<T> = (u64, T);

/// Distinct values as (count, first row), count descending, ties to the earlier first row.
pub(crate) type Ranked = Vec<(u64, u64)>;

/// A text level's distinct non-null values as (text, count), in `Ranked` order: the
/// recommenders' `top_k` and frequency-ordered dictionary (spec 2026-10-07 §4).
pub(crate) type Ranking = Vec<(String, u64)>;

/// `Ranked` order: count descending, ties to the earlier first row.
pub(crate) fn ranked_order(a: &(u64, u64), b: &(u64, u64)) -> std::cmp::Ordering {
    b.0.cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Sorts (count, first row) pairs into `Ranked` order. First rows are unique per value, so the
/// unstable sort is deterministic.
pub(crate) fn sort_ranked(r: &mut [(u64, u64)]) {
    r.sort_unstable_by(ranked_order);
}

pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub f1: u64,
    pub f2: u64,
    /// First rows of the distinct values, first-occurrence order, while there are ≤ 5
    /// (Recommend's boolean-pair rule reads them; streaming keeps the same list).
    pub first_few: Vec<u64>,
    /// Every distinct value occurred once (streaming's sampling phase: every sampled one).
    pub all_once: bool,
    /// Streaming's sampling phase: (HyperLogLog estimate, relative standard error,
    /// distinct values proven seen); None while `n_unique` is exact.
    pub hll: Option<(f64, f64, u64)>,
    pub capture_history: [u64; 7],
    /// Total byte length of the distinct values (`lengths` given: string / binary columns).
    pub sum_len_unique: Option<u64>,
    /// Every distinct value in `Ranked` order, when `frequencies` was given a `rank_up_to`
    /// and `n_unique` is within it.
    pub ranked: Option<Ranked>,
}

/// Split subset (0, 1 or 2) of `row`: the SplitMix64 finaliser of `seed + row`.
/// Seeded and language-independent; the Python implementations use numpy's
/// generator instead, which is why capture histories are compared only through
/// the Schnabel estimate.
#[inline]
pub(crate) fn subset(seed: u64, row: u64) -> u8 {
    let mut z = seed.wrapping_add(row).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % 3) as u8
}

pub(crate) fn count_chunk(values: &[u64], is_null: &[bool], start: usize, seed: u64) -> Map {
    let mut map = Map::with_capacity_and_hasher(1024, FixedState::default());
    for (j, (&key, &null)) in values.iter().zip(is_null).enumerate() {
        if null {
            continue;
        }
        let row = (start + j) as u64;
        let e = map.entry(key).or_insert(Entry {
            count: 0,
            first: row,
            mask: 0,
        });
        e.count += 1;
        e.mask |= 1 << subset(seed, row);
    }
    map
}

pub(crate) fn merge(mut a: Map, mut b: Map) -> Map {
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    for (key, e) in b {
        a.entry(key)
            .and_modify(|x| {
                x.count += e.count;
                x.first = x.first.min(e.first);
                x.mask |= e.mask;
            })
            .or_insert(e);
    }
    a
}

/// The per-value map of `col`, rows counted from `offset` (streaming: the global row
/// of the batch's first value), so first rows and capture subsets are global.
pub(crate) fn frequency_map(col: &EncodedColumn, seed: u64, offset: u64) -> Map {
    col.values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, offset as usize + i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge)
}

/// Streaming's sampling phase: every non-null row's `hash_key` into a HyperLogLog
/// (precision `p`), and `frequency_map` restricted to the values hashing at or below
/// `top` (the only ones a bottom-k sample can still hold or admit). Each kept value's
/// count, first row and mask cover all of its rows, as in `frequency_map`.
pub(crate) fn frequency_map_below(
    col: &EncodedColumn,
    seed: u64,
    offset: u64,
    top: u64,
    p: u8,
) -> (Map, Hll) {
    col.values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| {
            let start = offset + (i * CHUNK) as u64;
            let mut map = Map::with_hasher(FixedState::default());
            let mut sketch = Hll::new(p);
            for (j, (&key, &null)) in values.iter().zip(nulls).enumerate() {
                if null {
                    continue;
                }
                let hash = hash_key(key);
                sketch.insert(hash);
                if hash > top {
                    continue;
                }
                let row = start + j as u64;
                let e = map.entry(key).or_insert(Entry {
                    count: 0,
                    first: row,
                    mask: 0,
                });
                e.count += 1;
                e.mask |= 1 << subset(seed, row);
            }
            (map, sketch)
        })
        .reduce(
            || (Map::with_hasher(FixedState::default()), Hll::new(p)),
            |(a, mut sa), (b, sb)| {
                sa.merge(&sb);
                (merge(a, b), sa)
            },
        )
}

/// `rank_up_to`: also rank the distinct values (`Frequencies.ranked`) when there are at most
/// that many (the recommenders' dictionary candidates; Describe passes None).
pub(crate) fn frequencies(
    col: &EncodedColumn,
    seed: u64,
    lengths: Option<&[u64]>,
    rank_up_to: Option<u64>,
) -> Frequencies {
    let map = frequency_map(col, seed, 0);
    let (mut f1, mut f2, mut history) = (0u64, 0u64, [0u64; 7]);
    let n_unique = map.len() as u64;
    let rank = rank_up_to.is_some_and(|t| n_unique <= t);
    let mut ranked: Ranked = Vec::with_capacity(if rank { map.len() } else { 0 });
    let mut first_few: Vec<u64> = if map.len() <= 5 {
        map.values().map(|e| e.first).collect()
    } else {
        Vec::new()
    };
    first_few.sort_unstable();
    let mut unique_len = 0u64;
    for e in map.into_values() {
        f1 += (e.count == 1) as u64;
        f2 += (e.count == 2) as u64;
        history[e.mask as usize - 1] += 1;
        if let Some(l) = lengths {
            unique_len += l[e.first as usize];
        }
        if rank {
            ranked.push((e.count, e.first));
        }
    }
    let ranked = rank.then(|| {
        sort_ranked(&mut ranked);
        ranked
    });
    Frequencies {
        n_unique,
        f1,
        f2,
        first_few,
        all_once: f1 == n_unique,
        hll: None,
        capture_history: history,
        sum_len_unique: lengths.map(|_| unique_len),
        ranked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::encode_series;
    use polars::prelude::*;

    fn freq(s: Series) -> Frequencies {
        frequencies(&encode_series(&s).unwrap(), 0, None, None)
    }

    #[test]
    fn ranking_orders_by_count_then_first_row() {
        let s = Series::new(
            "x".into(),
            &[
                Some("b"),
                Some("a"),
                Some("b"),
                None,
                Some("c"),
                Some("a"),
                Some("b"),
            ],
        );
        let enc = encode_series(&s).unwrap();
        // (count, first row): b ×3 from row 0, a ×2 from row 1, c ×1 at row 4; the null is not counted.
        assert_eq!(
            frequencies(&enc, 0, None, Some(10)).ranked,
            Some(vec![(3, 0), (2, 1), (1, 4)])
        );
        // 3 distinct values > 2: no ranking.
        assert_eq!(frequencies(&enc, 0, None, Some(2)).ranked, None);
        assert_eq!(frequencies(&enc, 0, None, None).ranked, None);
    }

    #[test]
    fn ranking_ties_go_to_the_value_seen_first() {
        let s = Series::new("x".into(), &["y", "x", "x", "y", "z"]);
        let f = frequencies(&encode_series(&s).unwrap(), 0, None, Some(10));
        assert_eq!(f.ranked, Some(vec![(2, 0), (2, 1), (1, 4)]));
    }

    #[test]
    fn frequency_map_below_is_the_full_map_restricted() {
        // Three chunks, repeated values, nulls; offset as in a later streaming batch.
        let n = 2 * CHUNK + 777;
        let s = Series::new(
            "x".into(),
            (0..n as i64)
                .map(|i| (i % 11 != 0).then_some(i % 40_000))
                .collect::<Vec<_>>(),
        );
        let enc = encode_series(&s).unwrap();
        let full = frequency_map(&enc, 3, 1_000);
        let mut hashes: Vec<u64> = full.keys().map(|&k| hash_key(k)).collect();
        hashes.sort_unstable();
        let top = hashes[hashes.len() / 10];
        let (below, sketch) = frequency_map_below(&enc, 3, 1_000, top, 14);
        let want: Vec<_> = full
            .iter()
            .filter(|(&k, _)| hash_key(k) <= top)
            .map(|(&k, e)| (k, e.count, e.first, e.mask))
            .collect();
        let mut got: Vec<_> = below
            .iter()
            .map(|(&k, e)| (k, e.count, e.first, e.mask))
            .collect();
        let mut want = want;
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);
        let mut all = Hll::new(14);
        hashes.iter().for_each(|&h| all.insert(h));
        assert_eq!(sketch, all);
    }

    #[test]
    fn frequency_map_offsets_rows() {
        let whole = encode_series(&Series::new("x".into(), &["a", "b", "a"])).unwrap();
        let tail = encode_series(&Series::new("x".into(), &["b", "a"])).unwrap();
        let (w, t) = (frequency_map(&whole, 7, 0), frequency_map(&tail, 7, 1));
        let b = whole.values[1];
        assert_eq!((w[&b].first, t[&b].first), (1, 1));
        assert_eq!(w[&b].mask, t[&b].mask); // capture subsets use the global row
        assert_eq!(frequencies(&whole, 7, None, None).n_unique, w.len() as u64);
    }

    #[test]
    fn counts_and_first_few() {
        let f = freq(Series::new(
            "a".into(),
            &[Some("a"), Some("a"), Some("b"), None],
        ));
        assert_eq!((f.n_unique, f.f1, f.f2, f.first_few), (2, 1, 1, vec![0, 2]));
        assert!(!f.all_once && f.hll.is_none());
        // More than five distinct values: no list.
        let g = freq(Series::new("a".into(), &[3i64, 1, 2, 1, 2, 3, 4, 5, 6]));
        assert!(g.first_few.is_empty());
    }

    #[test]
    fn merges_across_parallel_chunks() {
        let n = 3 * CHUNK + 5;
        // value 9 first appears in the third chunk and again in the fourth; every other value is 0.
        let mut v = vec![0i64; n];
        v[2 * CHUNK + 1] = 9;
        v[3 * CHUNK + 2] = 9;
        let f = freq(Series::new("a".into(), v));
        assert_eq!((f.n_unique, f.f1, f.f2), (2, 0, 1));
        assert_eq!(f.first_few, vec![0, (2 * CHUNK + 1) as u64]);
    }

    #[test]
    fn capture_history_sums_to_n_unique_and_fills_all_subsets() {
        let f = freq(Series::new(
            "a".into(),
            (0..10_000i64).map(|i| i % 10).collect::<Vec<_>>(),
        ));
        assert_eq!(f.capture_history, [0, 0, 0, 0, 0, 0, 10]);
        let g = freq(Series::new(
            "a".into(),
            (0..20_000i64)
                .map(|i| (i * 7919) % 5_003)
                .collect::<Vec<_>>(),
        ));
        assert_eq!(g.capture_history.iter().sum::<u64>(), g.n_unique);
    }

    #[test]
    fn subsets_are_roughly_uniform() {
        let mut counts = [0u64; 3];
        for row in 0..30_000 {
            counts[subset(0, row) as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn zero_rows_and_all_null() {
        let z = freq(Series::new_empty("a".into(), &DataType::Int32));
        assert_eq!((z.n_unique, z.capture_history), (0, [0; 7]));
        let a = freq(Series::new("a".into(), &[None::<i32>, None]));
        assert_eq!((a.n_unique, a.first_few), (0, vec![]));
    }

    #[test]
    fn struct_values_hash_whole_and_binary_encodes() {
        let a = Series::new("a".into(), &[1i32, 1, 1]);
        let b = Series::new("b".into(), &[Some("x"), Some("x"), None]);
        let s = StructChunked::from_series("s".into(), 3, [a, b].iter())
            .unwrap()
            .into_series();
        assert_eq!(freq(s).n_unique, 2);
        let bin = Series::new(
            "b".into(),
            &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), None],
        );
        assert_eq!(freq(bin).n_unique, 1);
    }
}
