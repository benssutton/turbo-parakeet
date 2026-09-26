// ─────────────────────────────────────────────────────────────────────────────
// describe — value frequencies (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// One foldhash map `key → (count, first row, split-subset mask)` per 64K-row
// chunk, built in parallel and merged (counts summed, first = min, masks OR-ed).
// One O(distinct) sweep of the merged map yields n_unique, entropy (null as its
// own category), f1/f2, the capture history and the top 5 (count desc, then
// first occurrence asc). Keys come from `encode_series` (floats canonicalised;
// strings, nested and struct values hashed — collisions ~6e-11 per pair at 50K
// rows, accepted as documented in CLAUDE.md).

use crate::shared::EncodedColumn;
use foldhash::fast::FixedState;
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Entry {
    count: u64,
    first: u64,
    mask: u8,
}

type Map = HashMap<u64, Entry, FixedState>;

pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub entropy: f64,
    pub f1: u64,
    pub f2: u64,
    pub top5_idx: Vec<u64>,
    pub top5_count: Vec<u64>,
    pub capture_history: [u64; 7],
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

fn count_chunk(values: &[u64], is_null: &[bool], start: usize, seed: u64) -> Map {
    let mut map = Map::with_capacity_and_hasher(1024, FixedState::default());
    for (j, (&key, &null)) in values.iter().zip(is_null).enumerate() {
        if null {
            continue;
        }
        let row = (start + j) as u64;
        let e = map.entry(key).or_insert(Entry { count: 0, first: row, mask: 0 });
        e.count += 1;
        e.mask |= 1 << subset(seed, row);
    }
    map
}

fn merge(mut a: Map, mut b: Map) -> Map {
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

pub(crate) fn frequencies(col: &EncodedColumn, seed: u64) -> Frequencies {
    let n = col.len();
    let map = col
        .values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge);

    let n_null = col.is_null.iter().filter(|&&x| x).count();
    let nf = n as f64;
    let mut entropy = if n == 0 { f64::NAN } else { 0.0 };
    let (mut f1, mut f2, mut history) = (0u64, 0u64, [0u64; 7]);
    let n_unique = map.len() as u64;
    let mut entries: Vec<Entry> = Vec::with_capacity(map.len());
    for e in map.into_values() {
        let p = e.count as f64 / nf;
        entropy -= p * p.log2();
        f1 += (e.count == 1) as u64;
        f2 += (e.count == 2) as u64;
        history[e.mask as usize - 1] += 1;
        entries.push(e);
    }
    if n_null > 0 {
        let p = n_null as f64 / nf;
        entropy -= p * p.log2();
    }
    let order = |a: &Entry, b: &Entry| b.count.cmp(&a.count).then(a.first.cmp(&b.first));
    if entries.len() > 5 {
        entries.select_nth_unstable_by(4, order);
        entries.truncate(5);
    }
    entries.sort_unstable_by(order);
    Frequencies {
        n_unique,
        entropy: entropy + 0.0, // -0.0 → 0.0 for an all-null column
        f1,
        f2,
        top5_idx: entries.iter().map(|e| e.first).collect(),
        top5_count: entries.iter().map(|e| e.count).collect(),
        capture_history: history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::encode_series;
    use polars::prelude::*;

    fn freq(s: Series) -> Frequencies {
        frequencies(&encode_series(&s).unwrap(), 0)
    }

    #[test]
    fn counts_entropy_and_top5() {
        let f = freq(Series::new("a".into(), &[Some("a"), Some("a"), Some("b"), None]));
        assert_eq!((f.n_unique, f.f1, f.f2), (2, 1, 1));
        assert!((f.entropy - 1.5).abs() < 1e-12);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 2], vec![2, 1]));
    }

    #[test]
    fn ties_break_by_first_occurrence() {
        let f = freq(Series::new("a".into(), &[3i64, 1, 2, 1, 2, 3, 4, 5, 6]));
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 1, 2, 6, 7], vec![2, 2, 2, 1, 1]));
    }

    #[test]
    fn merges_across_parallel_chunks() {
        let n = 3 * CHUNK + 5;
        // value 9 first appears in the third chunk and again in the fourth; every other value is 0.
        let mut v = vec![0i64; n];
        v[2 * CHUNK + 1] = 9;
        v[3 * CHUNK + 2] = 9;
        let f = freq(Series::new("a".into(), v));
        assert_eq!(f.n_unique, 2);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, (2 * CHUNK + 1) as u64], vec![(n - 2) as u64, 2]));
    }

    #[test]
    fn capture_history_sums_to_n_unique_and_fills_all_subsets() {
        let f = freq(Series::new("a".into(), (0..10_000i64).map(|i| i % 10).collect::<Vec<_>>()));
        assert_eq!(f.capture_history, [0, 0, 0, 0, 0, 0, 10]);
        let g = freq(Series::new("a".into(), (0..20_000i64).map(|i| (i * 7919) % 5_003).collect::<Vec<_>>()));
        assert_eq!(g.capture_history.iter().sum::<u64>(), g.n_unique);
    }

    #[test]
    fn subsets_are_roughly_uniform() {
        let mut counts = [0u64; 3];
        for row in 0..30_000 {
            counts[subset(0, row) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9_500..10_500).contains(&c)), "{counts:?}");
    }

    #[test]
    fn zero_rows_and_all_null() {
        let z = freq(Series::new_empty("a".into(), &DataType::Int32));
        assert!(z.entropy.is_nan());
        assert_eq!((z.n_unique, z.capture_history), (0, [0; 7]));
        let a = freq(Series::new("a".into(), &[None::<i32>, None]));
        assert_eq!((a.n_unique, a.entropy), (0, 0.0));
    }

    #[test]
    fn struct_values_hash_whole_and_binary_encodes() {
        let a = Series::new("a".into(), &[1i32, 1, 1]);
        let b = Series::new("b".into(), &[Some("x"), Some("x"), None]);
        let s = StructChunked::from_series("s".into(), 3, [a, b].iter()).unwrap().into_series();
        assert_eq!(freq(s).n_unique, 2);
        let bin = Series::new("b".into(), &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), None]);
        assert_eq!(freq(bin).n_unique, 1);
    }
}
