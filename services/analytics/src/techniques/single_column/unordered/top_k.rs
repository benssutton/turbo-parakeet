// Used by space_saving.rs / heavy_keeper.rs (next tasks); removed once they land.
#![allow(dead_code)]

//! Shared parts of the top-k sketches (spec docs/superpowers/specs/2026-10-12-heavy-hitters-
//! top-k-design.md): `Counters`, an indexed min-heap of tracked keys with an open-addressing
//! key index and exact byte accounting; the kernels' key extraction and output.

use polars::prelude::*;
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::common::*;
use crate::techniques::ams_entropy::mix64;

/// One tracked key: its count, SpaceSaving's over-count bound and a row holding its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub key: u64,
    pub count: u64,
    pub error: u64,
    pub row: u64,
}

/// A reported value; `error` is None when the sketch has no per-item bound (HeavyKeeper).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Item {
    /// The value's encoded key: read by the tests, kept for the recommender integration.
    #[cfg_attr(not(test), allow(dead_code))]
    pub key: u64,
    pub row: u64,
    pub count: u64,
    pub error: Option<u64>,
}

/// Report order: count desc, then row asc, then key asc.
pub(crate) fn rank_order(a: &Entry, b: &Entry) -> Ordering {
    b.count
        .cmp(&a.count)
        .then(a.row.cmp(&b.row))
        .then(a.key.cmp(&b.key))
}

/// 4 GiB per sketch: ids fit u32, sizing cannot overflow.
pub(crate) const MAX_MEMORY_BYTES: usize = 1 << 32;

const EMPTY: u32 = u32::MAX;

fn table_len(cap: usize) -> usize {
    (2 * cap).next_power_of_two().max(8)
}

/// Up to `cap` entries in a min-heap by (count, key), found by key through a linear-probing
/// table at most half full. An entry keeps its id (arena slot) for life: `replace_min`
/// overwrites the minimum in place, so ids are 0..len and every id is in the heap.
pub(crate) struct Counters {
    entries: Vec<Entry>,
    heap: Vec<u32>,
    pos: Vec<u32>,
    table: Vec<u32>,
    cap: usize,
}

impl Counters {
    pub(crate) fn new(cap: usize) -> Self {
        assert!(
            cap >= 1 && cap < EMPTY as usize,
            "capacity {cap} out of range"
        );
        Counters {
            entries: Vec::with_capacity(cap),
            heap: Vec::with_capacity(cap),
            pos: Vec::with_capacity(cap),
            table: vec![EMPTY; table_len(cap)],
            cap,
        }
    }

    /// Accounted bytes of `Counters::new(cap)`: entries, heap and positions at capacity, plus
    /// the key index.
    pub(crate) fn bytes_for(cap: usize) -> usize {
        cap * (std::mem::size_of::<Entry>() + 8) + table_len(cap) * 4
    }

    /// The most entries whose `bytes_for` fits `memory_bytes` (0 when none fits); the budget is
    /// clamped to `MAX_MEMORY_BYTES`.
    pub(crate) fn capacity_for(memory_bytes: usize) -> usize {
        let memory_bytes = memory_bytes.min(MAX_MEMORY_BYTES);
        let per = std::mem::size_of::<Entry>() + 8;
        let (mut best, mut t) = (0, 8usize);
        while 4 * t <= memory_bytes {
            best = best.max(((memory_bytes - 4 * t) / per).min(t / 2));
            t *= 2;
        }
        best
    }

    // `with_capacity` allocates exactly the capacity asked for these element types; the test pins it.
    /// Allocated bytes, by the same formula as `bytes_for`.
    pub(crate) fn bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<Entry>()
            + (self.heap.capacity() + self.pos.capacity() + self.table.capacity()) * 4
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    pub(crate) fn is_full(&self) -> bool {
        self.entries.len() == self.cap
    }

    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub(crate) fn get(&self, id: usize) -> &Entry {
        &self.entries[id]
    }

    /// The entry with the smallest (count, key).
    pub(crate) fn min(&self) -> Option<&Entry> {
        self.heap.first().map(|&id| &self.entries[id as usize])
    }

    pub(crate) fn find(&self, key: u64) -> Option<usize> {
        let mask = self.table.len() - 1;
        let mut i = mix64(key) as usize & mask;
        loop {
            let id = self.table[i];
            if id == EMPTY {
                return None;
            }
            if self.entries[id as usize].key == key {
                return Some(id as usize);
            }
            i = (i + 1) & mask;
        }
    }

    /// Track a key that is not tracked yet; the caller checks `!is_full()`.
    pub(crate) fn push(&mut self, e: Entry) {
        debug_assert!(!self.is_full() && self.find(e.key).is_none());
        let id = self.entries.len() as u32;
        self.entries.push(e);
        self.pos.push(self.heap.len() as u32);
        self.heap.push(id);
        self.index(id);
        self.sift_up(self.heap.len() - 1);
    }

    /// Set a tracked entry's count (never lower) and error.
    pub(crate) fn raise(&mut self, id: usize, count: u64, error: u64) {
        let e = &mut self.entries[id];
        debug_assert!(count >= e.count);
        e.count = count;
        e.error = error;
        self.sift_down(self.pos[id] as usize);
    }

    /// Replace the minimum entry with `e`, a key not tracked yet.
    pub(crate) fn replace_min(&mut self, e: Entry) {
        debug_assert!(self.find(e.key).is_none());
        let id = self.heap[0];
        self.unindex(self.entries[id as usize].key);
        self.entries[id as usize] = e;
        self.index(id);
        self.sift_down(0);
    }

    /// Every entry in report order (`rank_order`).
    pub(crate) fn ranked(&self) -> Vec<Entry> {
        let mut v = self.entries.clone();
        v.sort_unstable_by(rank_order);
        v
    }

    fn less(&self, a: u32, b: u32) -> bool {
        let (x, y) = (&self.entries[a as usize], &self.entries[b as usize]);
        (x.count, x.key) < (y.count, y.key)
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.pos[self.heap[i] as usize] = i as u32;
        self.pos[self.heap[j] as usize] = j as u32;
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if !self.less(self.heap[i], self.heap[p]) {
                break;
            }
            self.swap(i, p);
            i = p;
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && self.less(self.heap[l], self.heap[m]) {
                m = l;
            }
            if r < n && self.less(self.heap[r], self.heap[m]) {
                m = r;
            }
            if m == i {
                return;
            }
            self.swap(i, m);
            i = m;
        }
    }

    fn index(&mut self, id: u32) {
        let mask = self.table.len() - 1;
        let mut i = mix64(self.entries[id as usize].key) as usize & mask;
        while self.table[i] != EMPTY {
            i = (i + 1) & mask;
        }
        self.table[i] = id;
    }

    /// Remove a tracked `key` from the index by backward-shift deletion, so every other key's
    /// probe chain stays unbroken.
    fn unindex(&mut self, key: u64) {
        let mask = self.table.len() - 1;
        let mut i = mix64(key) as usize & mask;
        while self.entries[self.table[i] as usize].key != key {
            i = (i + 1) & mask;
        }
        let mut j = i;
        loop {
            j = (j + 1) & mask;
            let id = self.table[j];
            if id == EMPTY {
                break;
            }
            let home = mix64(self.entries[id as usize].key) as usize & mask;
            // `id` may move back to the hole at i when i lies cyclically in [home, j].
            if (j.wrapping_sub(home) & mask) >= (j.wrapping_sub(i) & mask) {
                self.table[i] = id;
                i = j;
            }
        }
        self.table[i] = EMPTY;
    }
}

/// SplitMix64 over `mix64` (which adds γ again): the sketches' seeded randomness. Not the reference stream.
pub(crate) struct SplitMix(pub u64);

impl SplitMix {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }
}

/// Every column's `encode_series` keys, nulls flagged, encoded in parallel.
pub(crate) fn encoded(inputs: &[Series]) -> PolarsResult<Vec<EncodedColumn>> {
    build_column_cache_par(inputs, &(0..inputs.len()).collect::<HashSet<_>>())
}

/// One column's result: ranked items, exact non-null count and the sketch's accounted bytes.
pub(crate) struct ColumnTop {
    pub items: Vec<Item>,
    pub n_non_null: u64,
    pub sketch_bytes: u64,
}

/// The kernels' output, one row per (column, rank): `col_name, rank, row, count, error,
/// n_non_null, sketch_bytes`. A column without items gets one row with null rank..error.
pub(crate) fn top_k_struct(
    name: &str,
    inputs: &[Series],
    tops: &[ColumnTop],
) -> PolarsResult<Series> {
    let (mut col, mut rank, mut row, mut count, mut error, mut nn, mut bytes) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for (s, t) in inputs.iter().zip(tops) {
        let mut push = |item: Option<(u32, &Item)>| {
            col.push(s.name().as_str());
            rank.push(item.map(|(r, _)| r));
            row.push(item.map(|(_, i)| i.row));
            count.push(item.map(|(_, i)| i.count));
            error.push(item.and_then(|(_, i)| i.error));
            nn.push(t.n_non_null);
            bytes.push(t.sketch_bytes);
        };
        if t.items.is_empty() {
            push(None);
        }
        for (r, item) in t.items.iter().enumerate() {
            push(Some((r as u32 + 1, item)));
        }
    }
    let n = col.len();
    let fields = [
        StringChunked::from_iter(col)
            .into_series()
            .with_name("col_name".into()),
        UInt32Chunked::from_iter(rank)
            .into_series()
            .with_name("rank".into()),
        UInt64Chunked::from_iter(row)
            .into_series()
            .with_name("row".into()),
        UInt64Chunked::from_iter(count)
            .into_series()
            .with_name("count".into()),
        UInt64Chunked::from_iter(error)
            .into_series()
            .with_name("error".into()),
        UInt64Chunked::from_vec("n_non_null".into(), nn).into_series(),
        UInt64Chunked::from_vec("sketch_bytes".into(), bytes).into_series(),
    ];
    Ok(StructChunked::from_series(name.into(), n, fields.iter())?.into_series())
}

/// Test streams shared by the sketches' tests.
#[cfg(test)]
pub(crate) mod streams {
    use super::SplitMix;
    use std::collections::HashMap;

    /// `n` draws of a Zipf(`alpha`) law over keys 1000..1000 + `n_keys` (rank 1 = key 1000).
    pub(crate) fn zipf_stream(n: usize, n_keys: usize, alpha: f64, seed: u64) -> Vec<u64> {
        let mut cdf: Vec<f64> = (1..=n_keys).map(|i| (i as f64).powf(-alpha)).collect();
        for i in 1..n_keys {
            cdf[i] += cdf[i - 1];
        }
        let total = cdf[n_keys - 1];
        let mut rng = SplitMix(seed);
        (0..n)
            .map(|_| {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
                1000 + cdf.partition_point(|&c| c <= u).min(n_keys - 1) as u64
            })
            .collect()
    }

    pub(crate) fn exact_counts(stream: &[u64]) -> HashMap<u64, u64> {
        let mut m = HashMap::new();
        for &k in stream {
            *m.entry(k).or_insert(0) += 1;
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn e(key: u64, count: u64) -> Entry {
        Entry {
            key,
            count,
            error: 0,
            row: key,
        }
    }

    /// Random push / raise / replace_min against a HashMap model, at a capacity small enough
    /// for long probe chains and many backward shifts.
    #[test]
    fn counters_match_a_model() {
        for cap in [1usize, 2, 5, 37] {
            let mut c = Counters::new(cap);
            let mut model: HashMap<u64, u64> = HashMap::new();
            let mut rng = SplitMix(cap as u64);
            for _ in 0..20_000 {
                let key = rng.next_u64() % (3 * cap as u64 + 3);
                match c.find(key) {
                    Some(id) => {
                        let n = c.get(id).count + 1 + rng.next_u64() % 3;
                        c.raise(id, n, 0);
                        model.insert(key, n);
                    }
                    None if !c.is_full() => {
                        c.push(e(key, 1));
                        model.insert(key, 1);
                    }
                    None => {
                        let m = *c.min().unwrap();
                        let want = model.iter().map(|(&k, &n)| (n, k)).min().unwrap();
                        assert_eq!((m.count, m.key), want);
                        model.remove(&m.key);
                        c.replace_min(e(key, m.count + 1));
                        model.insert(key, m.count + 1);
                    }
                }
                assert_eq!(c.entries().len(), model.len());
            }
            for (&k, &n) in &model {
                let id = c.find(k).expect("tracked key is found");
                assert_eq!(c.get(id).count, n);
            }
            assert!(c.find(u64::MAX).is_none());
        }
    }

    #[test]
    fn ranked_orders_by_count_then_row() {
        let mut c = Counters::new(4);
        for (key, count, row) in [(7, 2, 5), (8, 3, 9), (9, 2, 1)] {
            c.push(Entry {
                key,
                count,
                error: 0,
                row,
            });
        }
        let keys: Vec<u64> = c.ranked().iter().map(|e| e.key).collect();
        assert_eq!(keys, [8, 9, 7]);
    }

    #[test]
    fn bytes_are_accounted_exactly() {
        for cap in [1usize, 5, 307, 4915] {
            assert_eq!(Counters::new(cap).bytes(), Counters::bytes_for(cap));
        }
    }

    #[test]
    fn capacity_for_fits_the_budget() {
        assert_eq!(Counters::capacity_for(16 * 1024), 307);
        for budget in [0usize, 31, 100, 1_000, 16_384, 262_144, 10_000_000] {
            let m = Counters::capacity_for(budget);
            if m > 0 {
                assert!(Counters::bytes_for(m) <= budget, "{budget}: {m}");
                assert!(
                    Counters::bytes_for(m + 1) > budget,
                    "{budget}: {m} not maximal"
                );
            }
        }
        assert_eq!(Counters::capacity_for(31), 0);
        assert_eq!(
            Counters::capacity_for(usize::MAX),
            Counters::capacity_for(MAX_MEMORY_BYTES)
        );
    }

    #[test]
    fn output_has_one_null_row_for_an_empty_column() {
        let inputs = [
            Series::new("a".into(), [1i64]),
            Series::new("b".into(), [None::<i64>]),
        ];
        let item = Item {
            key: 1,
            row: 0,
            count: 1,
            error: Some(0),
        };
        let tops = [
            ColumnTop {
                items: vec![item],
                n_non_null: 1,
                sketch_bytes: 10,
            },
            ColumnTop {
                items: vec![],
                n_non_null: 0,
                sketch_bytes: 10,
            },
        ];
        let s = top_k_struct("t", &inputs, &tops).unwrap();
        let s = s.struct_().unwrap();
        assert_eq!(s.len(), 2);
        let rank = s.field_by_name("rank").unwrap();
        assert_eq!(rank.u32().unwrap().get(0), Some(1));
        assert_eq!(rank.u32().unwrap().get(1), None);
        assert_eq!(
            s.field_by_name("n_non_null").unwrap().u64().unwrap().get(1),
            Some(0)
        );
    }
}
