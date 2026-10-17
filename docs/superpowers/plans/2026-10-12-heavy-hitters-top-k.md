# Heavy Hitters (Top-k) Sketches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-column technique `analytics.top_k` that finds each column's k most frequent values with two Rust sketches (HeavyKeeper, SpaceSaving) and an exact Polars reference, and benchmark the sketches against the reference for accuracy, memory and speed.

**Architecture:** Three new Rust files in `src/techniques/single_column/unordered/`:
- `top_k.rs`: the indexed min-heap `Counters`, a SplitMix64 generator and the kernels' output builder, shared by both sketches;
- `heavy_keeper.rs` and `space_saving.rs`: the sketches and their kernels.

Two private API entry points are exposed through pyo3 and `_plugin.py`. A Python package `analytics/top_k/` holds the `TopK` base (eligibility, rendering values from rows, `dominated`, agreement) and three implementations. Two benchmark scripts measure the sketches: a speed run through the shared harness, and a child-process accuracy/memory report.

**Tech Stack:** Rust (polars, rayon, foldhash via `common`), pyo3, Python 3.12, Polars, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md` (read it first).

---

## Conventions for every task

- Python: `C:/Users/Alexander/miniconda3/envs/p312/python.exe` (written `$PY` below). In Git Bash, run
  `PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe` first.
- Rust unit tests for one module, from `services/analytics/`, in Git Bash:
  ```bash
  cd /c/Users/Alexander/turbo-parakeet/services/analytics && \
  PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib <module>::
  ```
- Rebuild the extension after any Rust change that Python needs: `$PY scripts/check.py build` from the repo root (about 2–3 min warm).
- pytest from the repo root: `$PY -m pytest tests/test_top_k.py -n 0 -q`.
- Commits run pre-commit hooks that call `python`. Prefix every `git commit` with
  `PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"`.
  End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- `docs/superpowers/` is git-ignored, but its files are tracked: add them with `git add -f`.
- Never stage `CLAUDE.md` except in Task 10. It has unrelated uncommitted user edits; stage only the hunks you add, with `git add -p CLAUDE.md`.

## File map

| File | Responsibility |
|---|---|
| Create `services/analytics/src/techniques/single_column/unordered/top_k.rs` | `Entry`, `Item`, `Counters` (indexed min-heap + open-addressing key index, byte accounting), `rank_order`, `SplitMix`, `encoded`, `ColumnTop`, `top_k_struct`; test helpers `zipf_stream`, `exact_counts` |
| Create `services/analytics/src/techniques/single_column/unordered/space_saving.rs` | `SpaceSaving` (insert / merge / top / bytes), `space_saving_top_k_impl` (chunked, merged) |
| Create `services/analytics/src/techniques/single_column/unordered/heavy_keeper.rs` | `HeavyKeeper` (insert / top / bytes), `heavy_keeper_top_k_impl` |
| Modify `services/analytics/src/techniques/single_column/unordered/mod.rs`, `src/techniques/mod.rs` | register the modules |
| Modify `services/analytics/src/bindings/api/techniques.rs` | `space_saving_top_k`, `heavy_keeper_top_k` + smoke tests |
| Modify `services/analytics/src/bindings/python.rs` | pyfunctions + registration |
| Modify `services/analytics/analytics/_plugin.py` | wrappers |
| Create `services/analytics/analytics/top_k/{__init__,base,polars,space_saving,heavy_keeper}.py` | technique package |
| Modify `tests/datagen.py`, `tests/test_datagen.py` | `top_k_streams` |
| Create `tests/test_top_k.py` | accuracy tests |
| Create `tests/performance/benchmark_top_k.py`, `tests/performance/benchmark_top_k_accuracy.py` | benchmarks |
| Modify `CLAUDE.md`, the spec (§8 Results) | docs |

---

### Task 1: Shared Rust core — `top_k.rs`

**Files:**
- Create: `services/analytics/src/techniques/single_column/unordered/top_k.rs`
- Modify: `services/analytics/src/techniques/single_column/unordered/mod.rs`
- Modify: `services/analytics/src/techniques/mod.rs`

- [ ] **Step 1: Register the module.** In `single_column/unordered/mod.rs` add, keeping alphabetical order:

```rust
pub(crate) mod heavy_keeper;
pub(crate) mod space_saving;
pub(crate) mod top_k;
```

To keep Task 1 compiling before Tasks 2–3, create `heavy_keeper.rs` and `space_saving.rs` now, each holding just one line:

```rust
//! Placeholder module, filled in by the next task.
```

In `src/techniques/mod.rs` extend the re-export:

```rust
pub(crate) use single_column::unordered::{
    ams_entropy, cardinality_estimators, describe, gcd, heavy_keeper, hll, marginal_entropy,
    space_saving, top_k,
};
```

- [ ] **Step 2: Write `top_k.rs` with its tests (tests first in the file's `mod tests`; implementation above).**

```rust
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
        assert!(cap >= 1 && cap < EMPTY as usize, "capacity {cap} out of range");
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

    /// The most entries whose `bytes_for` fits `memory_bytes` (0 when none fits).
    pub(crate) fn capacity_for(memory_bytes: usize) -> usize {
        let per = std::mem::size_of::<Entry>() + 8;
        let (mut best, mut t) = (0, 8usize);
        while 4 * t <= memory_bytes {
            best = best.max(((memory_bytes - 4 * t) / per).min(t / 2));
            t *= 2;
        }
        best
    }

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

/// SplitMix64: the sketches' seeded randomness.
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
        StringChunked::from_iter(col).into_series().with_name("col_name".into()),
        UInt32Chunked::from_iter(rank).into_series().with_name("rank".into()),
        UInt64Chunked::from_iter(row).into_series().with_name("row".into()),
        UInt64Chunked::from_iter(count).into_series().with_name("count".into()),
        UInt64Chunked::from_iter(error).into_series().with_name("error".into()),
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
        Entry { key, count, error: 0, row: key }
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
            c.push(Entry { key, count, error: 0, row });
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
            }
        }
        assert_eq!(Counters::capacity_for(31), 0);
    }

    #[test]
    fn output_has_one_null_row_for_an_empty_column() {
        let inputs = [Series::new("a".into(), [1i64]), Series::new("b".into(), [None::<i64>])];
        let item = Item { key: 1, row: 0, count: 1, error: Some(0) };
        let tops = [
            ColumnTop { items: vec![item], n_non_null: 1, sketch_bytes: 10 },
            ColumnTop { items: vec![], n_non_null: 0, sketch_bytes: 10 },
        ];
        let s = top_k_struct("t", &inputs, &tops).unwrap();
        let s = s.struct_().unwrap();
        assert_eq!(s.len(), 2);
        let rank = s.field_by_name("rank").unwrap();
        assert_eq!(rank.u32().unwrap().get(0), Some(1));
        assert_eq!(rank.u32().unwrap().get(1), None);
        assert_eq!(s.field_by_name("n_non_null").unwrap().u64().unwrap().get(1), Some(0));
    }
}
```

- [ ] **Step 3: Run the tests.**

Run: `cargo test --lib top_k::` (with the environment from Conventions).
Expected: 5 tests pass. If `capacity_for(16 * 1024)` is not 307, recheck that `size_of::<Entry>()` is 32. With 32, the formula gives 307 (13 640 + 4 × 1 024 > 16 384 at 341; 307 × 40 + 4 096 = 16 376).

- [ ] **Step 4: Commit.**

```bash
git add services/analytics/src/techniques
git commit -m "top_k.rs: indexed min-heap Counters and kernel output shared by the top-k sketches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: SpaceSaving — `space_saving.rs`

**Files:**
- Modify (replace placeholder): `services/analytics/src/techniques/single_column/unordered/space_saving.rs`

- [ ] **Step 1: Write the file: implementation and tests.**

```rust
//! SpaceSaving (Metwally, Agrawal, El Abbadi, ICDT 2005): m counters; a new key replaces the
//! minimum counter and inherits its count as an over-count bound ε, so a tracked key's true
//! count lies in [count − ε, count] and every key above n/m is tracked. Mergeable with the
//! bound intact (Agarwal et al., "Mergeable Summaries", PODS 2012), so the kernel counts fixed
//! row chunks in parallel and merges them in order. Spec 2026-10-12 §3.2.

use polars::prelude::*;
use rayon::prelude::*;

use crate::techniques::top_k::{encoded, rank_order, top_k_struct, ColumnTop, Counters, Entry, Item};

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
        let m = Counters::capacity_for(memory_bytes);
        if m < k {
            return Err(format!(
                "memory_bytes = {memory_bytes} holds {m} SpaceSaving counters, fewer than k = {k}"
            ));
        }
        Ok(SpaceSaving { k, counters: Counters::new(m) })
    }

    pub(crate) fn insert(&mut self, key: u64, row: u64, weight: u64) {
        let c = &mut self.counters;
        if let Some(id) = c.find(key) {
            let e = *c.get(id);
            c.raise(id, e.count + weight, e.error);
        } else if !c.is_full() {
            c.push(Entry { key, count: weight, error: 0, row });
        } else {
            let min = c.min().expect("a full summary has a minimum").count;
            c.replace_min(Entry { key, count: min + weight, error: min, row });
        }
    }

    /// The summary of both streams. An untracked key's true count in a full summary is at most
    /// its minimum (0 in a summary with room), so a side missing a key adds that minimum to both
    /// count and error; the m largest are kept.
    pub(crate) fn merge(&mut self, other: &SpaceSaving) {
        assert_eq!(self.counters.capacity(), other.counters.capacity(), "capacities differ");
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
                None => Entry { count: x.count + fb, error: x.error + fb, ..*x },
            })
            .collect();
        all.extend(
            b.entries()
                .iter()
                .filter(|y| a.find(y.key).is_none())
                .map(|y| Entry { count: y.count + fa, error: y.error + fa, ..*y }),
        );
        all.sort_unstable_by(rank_order);
        all.truncate(a.capacity());
        let mut merged = Counters::new(a.capacity());
        for e in all {
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
            .map(|e| Item { key: e.key, row: e.row, count: e.count, error: Some(e.error) })
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
    let keys = encoded(inputs)?;
    let tops = keys
        .par_iter()
        .map(|col| -> PolarsResult<ColumnTop> {
            let n = col.len();
            let parts = (0..n.div_ceil(CHUNK_ROWS))
                .into_par_iter()
                .map(|c| {
                    let mut s = sketch(k, memory_bytes)?;
                    for i in c * CHUNK_ROWS..((c + 1) * CHUNK_ROWS).min(n) {
                        if !col.is_null[i] {
                            s.insert(col.values[i], i as u64, 1);
                        }
                    }
                    Ok(s)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            let mut parts = parts.into_iter();
            let mut s = match parts.next() {
                Some(s) => s,
                None => sketch(k, memory_bytes)?,
            };
            for p in parts {
                s.merge(&p);
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
        let stream: Vec<u64> = (0..100u64).flat_map(|k| std::iter::repeat(k).take(k as usize + 1)).collect();
        let s = fed(&stream, 0);
        let top = s.top();
        let counts: Vec<(u64, u64, Option<u64>)> = top.iter().map(|i| (i.key, i.count, i.error)).collect();
        assert_eq!(counts, [(99, 100, Some(0)), (98, 99, Some(0)), (97, 98, Some(0)), (96, 97, Some(0)), (95, 96, Some(0))]);
    }

    #[test]
    fn bounds_hold_on_a_zipf_stream() {
        let stream = zipf_stream(200_000, 5_000, 1.1, 7);
        let s = fed(&stream, 0);
        check_guarantees(&s, &exact_counts(&stream), stream.len() as u64);
        let truth = exact_counts(&stream);
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
    fn merge_with_room_is_exact() {
        let mut a = fed(&[1, 1, 2], 0);
        a.merge(&fed(&[2, 3], 3));
        let got: Vec<(u64, u64, u64)> = a.counters.ranked().iter().map(|e| (e.key, e.count, e.row)).collect();
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
        let s = Series::new("x".into(), stream.iter().map(|&k| k as i64).collect::<Vec<_>>());
        let run = || space_saving_top_k_impl(std::slice::from_ref(&s), 5, BUDGET).unwrap();
        let (a, b) = (run(), run());
        assert!(a.equals_missing(&b));
        let out = a.struct_().unwrap();
        let counts = out.field_by_name("count").unwrap();
        let errors = out.field_by_name("error").unwrap();
        let truth = exact_counts(&stream);
        let rows = out.field_by_name("row").unwrap();
        for i in 0..out.len() {
            let key = stream[rows.u64().unwrap().get(i).unwrap() as usize];
            let (c, e) = (counts.u64().unwrap().get(i).unwrap(), errors.u64().unwrap().get(i).unwrap());
            assert!(c - e <= truth[&key] && truth[&key] <= c);
        }
        assert_eq!(out.field_by_name("n_non_null").unwrap().u64().unwrap().get(0), Some(n as u64));
    }

    #[test]
    fn sizing() {
        let s = SpaceSaving::new(5, BUDGET).unwrap();
        assert_eq!(s.counters.capacity(), 307);
        assert!(s.bytes() <= BUDGET);
        assert!(SpaceSaving::new(0, BUDGET).is_err());
        assert!(SpaceSaving::new(5, 64).is_err());
    }
}
```

Note: `Series::equals_missing` compares including nulls. If the polars version here names it differently, use `a == b` on the two `Series`' `struct_()` fields. Check with `grep -rn "equals_missing" services/analytics/src | head -3`.

- [ ] **Step 2: Run the tests.** `cargo test --lib space_saving::`. Expected: 7 pass. Then run rustfmt: `cargo fmt` (the test literals above are long; let rustfmt wrap them).

- [ ] **Step 3: Commit.**

```bash
git add services/analytics/src/techniques/single_column/unordered/space_saving.rs
git commit -m "SpaceSaving top-k sketch: weighted insert, bound-preserving merge, chunked kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: HeavyKeeper — `heavy_keeper.rs`

**Files:**
- Modify (replace placeholder): `services/analytics/src/techniques/single_column/unordered/heavy_keeper.rs`

- [ ] **Step 1: Write the file: implementation and tests.**

```rust
//! HeavyKeeper (Gong et al., "HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant
//! Flows", USENIX ATC 2018), final version: d = 2 arrays of (fingerprint, count) buckets with
//! count-with-exponential-decay (a different key decays a bucket with probability b^−C), plus a
//! min-heap of the top k with selective increment and fingerprint-collision detection. No
//! merge: a column is fed in row order. Spec 2026-10-12 §3.1.

use polars::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::techniques::ams_entropy::{mix64, name_key, sketch_seed};
use crate::techniques::top_k::{encoded, top_k_struct, ColumnTop, Counters, Entry, Item, SplitMix};

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
        let n_min = if full { self.heap.min().map_or(0, |e| e.count) } else { 0 };
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
        let entry = Entry { key, count: maxv, error: 0, row };
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
            .map(|e| Item { key: e.key, row: e.row, count: e.count, error: None })
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
            Ok(ColumnTop { items: hk.top(), n_non_null: n, sketch_bytes: hk.bytes() as u64 })
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
        let mut stream: Vec<u64> = (1..=20u64).flat_map(|k| std::iter::repeat(k).take(10 * k as usize)).collect();
        let mut rng = SplitMix(1);
        for i in (1..stream.len()).rev() {
            stream.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let hk = fed(5, BUDGET, &stream, 0);
        assert_eq!(top(&hk), [(20, 200), (19, 190), (18, 180), (17, 170), (16, 160)]);
    }

    #[test]
    fn the_same_seed_gives_the_same_top() {
        let stream = zipf_stream(100_000, 20_000, 0.9, 2);
        assert_eq!(top(&fed(5, BUDGET, &stream, 9)), top(&fed(5, BUDGET, &stream, 9)));
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
    }

    #[test]
    fn the_kernel_counts_non_null_rows() {
        let s = Series::new("x".into(), [Some(1i64), None, Some(1), Some(2)]);
        let out = heavy_keeper_top_k_impl(&[s], 5, BUDGET, 0).unwrap();
        let out = out.struct_().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out.field_by_name("count").unwrap().u64().unwrap().get(0), Some(2));
        assert_eq!(out.field_by_name("row").unwrap().u64().unwrap().get(0), Some(0));
        assert_eq!(out.field_by_name("error").unwrap().u64().unwrap().get(0), None);
        assert_eq!(out.field_by_name("n_non_null").unwrap().u64().unwrap().get(0), Some(3));
    }
}
```

- [ ] **Step 2: Run the tests.** `cargo test --lib heavy_keeper::`. Expected: 8 pass, then `cargo fmt`.
  - If `exact_with_few_keys` fails only because two keys share a bucket in both arrays, the shuffle seed is unlucky. Change the `HeavyKeeper` seed in `fed(5, BUDGET, &stream, 0)` to 1 and note it in the test comment. Do not loosen the assertion.
  - If `finds_the_top_5_of_a_zipf_stream` misses, report the observed counts rather than loosening anything. The spec expects HeavyKeeper to be exact on this skew.

- [ ] **Step 3: Commit.**

```bash
git add services/analytics/src/techniques/single_column/unordered/heavy_keeper.rs
git commit -m "HeavyKeeper top-k sketch: exponential decay, selective increment, collision detection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: API entry points, pyo3 binding, `_plugin` wrappers

**Files:**
- Modify: `services/analytics/src/bindings/api/techniques.rs`
- Modify: `services/analytics/src/bindings/python.rs`
- Modify: `services/analytics/analytics/_plugin.py`

- [ ] **Step 1: Write the failing API tests.** Append inside `mod tests` of `bindings/api/techniques.rs`, next to `ams_sample_count_is_capped`:

```rust
    #[test]
    fn top_k_smoke() {
        use arrow_array::types::{UInt32Type, UInt64Type};
        let b = batch(vec![(
            "a",
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), None])) as ArrayRef,
        )]);
        for out in [
            space_saving_top_k(&b, 5, 16 * 1024).unwrap(),
            heavy_keeper_top_k(&b, 5, 16 * 1024, 0).unwrap(),
        ] {
            let names: Vec<String> =
                out.schema().fields().iter().map(|f| f.name().clone()).collect();
            assert_eq!(
                names,
                ["col_name", "rank", "row", "count", "error", "n_non_null", "sketch_bytes"]
            );
            assert_eq!(out.num_rows(), 2);
            assert_eq!(out.column(1).as_primitive::<UInt32Type>().value(0), 1);
            assert_eq!(out.column(3).as_primitive::<UInt64Type>().value(0), 2);
            assert_eq!(out.column(5).as_primitive::<UInt64Type>().value(0), 3);
        }
    }

    #[test]
    fn top_k_parameters_are_checked() {
        let b = batch(vec![("a", ints(&[1, 2]))]);
        for r in [
            space_saving_top_k(&b, 0, 16 * 1024),
            space_saving_top_k(&b, 5, 64),
            heavy_keeper_top_k(&b, 0, 16 * 1024, 0),
            heavy_keeper_top_k(&b, 5, 64, 0),
        ] {
            assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
        }
    }
```

(`ints`, `batch` and `Arc` already exist in this test module. If `ints` takes a different argument type, build the column as in `top_k_smoke`.)

- [ ] **Step 2: Run them to see them fail.** `cargo test --lib bindings::api::techniques::tests::top_k`. Expected: compile error `cannot find function space_saving_top_k`.

- [ ] **Step 3: Add the entry points.** In `bindings/api/techniques.rs`, after `ams_threeway_joint_entropy`:

```rust
use crate::techniques::heavy_keeper::HeavyKeeper;
use crate::techniques::space_saving::SpaceSaving;

/// `col_name, rank, row, count, error, n_non_null, sketch_bytes`: each column's top `k` by
/// SpaceSaving in `memory_bytes` (spec 2026-10-12). k = 0 or a budget below k counters →
/// InvalidInput.
pub fn space_saving_top_k(batch: &RecordBatch, k: usize, memory_bytes: usize) -> Result<RecordBatch> {
    SpaceSaving::new(k, memory_bytes).map_err(Error::InvalidInput)?;
    table(crate::techniques::space_saving::space_saving_top_k_impl(
        &columns(batch)?,
        k,
        memory_bytes,
    ))
}

/// As `space_saving_top_k`, by HeavyKeeper (`error` null); a budget leaving fewer than k
/// buckets per array → InvalidInput.
pub fn heavy_keeper_top_k(
    batch: &RecordBatch,
    k: usize,
    memory_bytes: usize,
    seed: u64,
) -> Result<RecordBatch> {
    HeavyKeeper::new(k, memory_bytes, seed).map_err(Error::InvalidInput)?;
    table(crate::techniques::heavy_keeper::heavy_keeper_top_k_impl(
        &columns(batch)?,
        k,
        memory_bytes,
        seed,
    ))
}
```

Put the two `use` lines with the file's other `use` lines at the top.

- [ ] **Step 4: Run the API tests.** Same command. Expected: both pass.

- [ ] **Step 5: Expose them in `bindings/python.rs`.** After `ams_threeway_joint_entropy`'s pyfunction:

```rust
#[pyfunction]
#[pyo3(signature = (data, k, memory_bytes))]
fn space_saving_top_k(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    k: usize,
    memory_bytes: usize,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::space_saving_top_k(&batch, k, memory_bytes)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, k, memory_bytes, seed))]
fn heavy_keeper_top_k(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    k: usize,
    memory_bytes: usize,
    seed: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::heavy_keeper_top_k(&batch, k, memory_bytes, seed)).map(ArrowTable)
}
```

Register them after `ams_threeway_joint_entropy` in the module function:

```rust
    m.add_function(wrap_pyfunction!(space_saving_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(heavy_keeper_top_k, m)?)?;
```

- [ ] **Step 6: Add the `_plugin.py` wrappers** after `ams_threeway_joint_entropy`:

```python
def space_saving_top_k(df: Frame, k: int, memory_bytes: int) -> pl.DataFrame:
    """col_name, rank, row, count, error, n_non_null, sketch_bytes: each column's top k by
    SpaceSaving (one row per rank; one null-rank row for a column without values)."""
    return pl.DataFrame(_rs.space_saving_top_k(_collected(df), k, memory_bytes))


def heavy_keeper_top_k(df: Frame, k: int, memory_bytes: int, seed: int) -> pl.DataFrame:
    """As space_saving_top_k, by HeavyKeeper (error null)."""
    return pl.DataFrame(_rs.heavy_keeper_top_k(_collected(df), k, memory_bytes, seed))
```

- [ ] **Step 7: Build and smoke-test from Python.**

```bash
$PY scripts/check.py build
$PY -c "import polars as pl; from analytics import _plugin as p; print(p.space_saving_top_k(pl.DataFrame({'a':[1,1,2,None]}), 5, 16384)); print(p.heavy_keeper_top_k(pl.DataFrame({'a':[1,1,2,None]}), 5, 16384, 0))"
```

Expected: two 2-row frames. Rank 1 has row 0, count 2 and n_non_null 3. `error` is 0 in the first frame and null in the second.

- [ ] **Step 8: Commit.**

```bash
git add services/analytics/src/bindings services/analytics/analytics/_plugin.py
git commit -m "Top-k sketches: API entry points, pyo3 binding and _plugin wrappers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `top_k_streams` test data

**Files:**
- Modify: `tests/datagen.py` (new section after `near_constant`; add the name to the module docstring's per-column line)
- Modify: `tests/test_datagen.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_datagen.py`; add `top_k_streams` to its `from datagen import ...`):

```python
def test_top_k_streams_shape_and_determinism():
    df = top_k_streams(20_000)
    assert df.columns == ["zipf_15", "zipf_11", "zipf_08", "flat", "skewed", "zipf_str"]
    assert df.height == 20_000
    assert df.equals(top_k_streams(20_000))
    assert top_k_streams(20_000, columns=["zipf_08"])["zipf_08"].equals(df["zipf_08"])
    share = {c: df[c].value_counts(sort=True)["count"][0] / df.height for c in df.columns}
    assert share["zipf_15"] > share["zipf_11"] > share["zipf_08"] > share["flat"]
    assert 0.93 < share["skewed"] < 0.97
    assert df["zipf_str"][0] == f"item_{df['zipf_11'][0]}"
```

- [ ] **Step 2: Run it to see it fail.** `$PY -m pytest tests/test_datagen.py -n 0 -q -k top_k`. Expected: ImportError for `top_k_streams`.

- [ ] **Step 3: Implement** in `tests/datagen.py`:

```python
# ── per-column top-k ──────────────────────────────────────────────────────────

TOP_K_COLUMNS = ("zipf_15", "zipf_11", "zipf_08", "flat", "skewed", "zipf_str")


def _bounded_zipf(n_rows: int, n_values: int, alpha: float, rng) -> np.ndarray:
    """Zipf(alpha) truncated to n_values ranks (any alpha > 0, unlike numpy's zipf), mapped
    through a random permutation so the most frequent values are not 0, 1, 2, …"""
    cdf = np.cumsum(np.arange(1, n_values + 1, dtype=np.float64) ** -alpha)
    cdf /= cdf[-1]
    ranks = np.minimum(np.searchsorted(cdf, rng.random(n_rows), side="right"), n_values - 1)
    return rng.permutation(n_values)[ranks]


def top_k_streams(
    n_rows: int = 10_000_000, seed: int = 42, columns: list[str] | None = None
) -> pl.DataFrame:
    """Int64 streams for top-k sketches over 1M possible values: zipf_15 / zipf_11 / zipf_08
    (Zipf α = 1.5 / 1.1 / 0.8), flat (uniform), skewed (95% one value, the rest over 50,000
    others) and zipf_str (zipf_11 as "item_<n>" strings). Each column has its own seeded
    generator, so a column is the same whichever `columns` are asked for."""
    columns = list(TOP_K_COLUMNS) if columns is None else columns

    def rng(name: str):
        return np.random.default_rng([seed, TOP_K_COLUMNS.index(name)])

    def make(name: str) -> pl.Series:
        if name == "zipf_str":
            return ("item_" + make("zipf_11").cast(pl.String)).alias(name)
        r = rng(name)
        if name == "flat":
            values = r.integers(0, 1_000_000, n_rows)
        elif name == "skewed":
            values = np.where(r.random(n_rows) < 0.95, 0, r.integers(1, 50_001, n_rows))
        else:
            alpha = {"zipf_15": 1.5, "zipf_11": 1.1, "zipf_08": 0.8}[name]
            values = _bounded_zipf(n_rows, 1_000_000, alpha, r)
        return pl.Series(name, values.astype(np.int64))

    return pl.DataFrame([make(c) for c in columns])
```

`"item_" + series` (str + Series) is supported by Polars as a string concatenation. If not, use `make("zipf_11").cast(pl.String).str.replace(r"^", "item_")`.

- [ ] **Step 4: Run the test.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/datagen.py tests/test_datagen.py
git commit -m "datagen: top_k_streams (bounded Zipf, flat, skewed, string) for the top-k sketches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Technique base `TopK` and reference `TopKPolars`

**Files:**
- Create: `services/analytics/analytics/top_k/__init__.py`, `base.py`, `polars.py`
- Create: `tests/test_top_k.py`

- [ ] **Step 1: Write the package skeleton.** `__init__.py` (the Rust classes are added in Task 7):

```python
"""Top-k most frequent values per column (per-column scope). See TopK."""

from analytics.base import lazy_attributes
from analytics.top_k.base import TopK

REFERENCE = "TopKPolars"
IMPLEMENTATIONS = ("TopKPolars",)

__getattr__ = lazy_attributes(__name__, {"TopKPolars": ".polars"})
__all__ = ["TopK", "TopKPolars", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 2: Write the failing tests** — `tests/test_top_k.py`:

```python
"""
Top-k accuracy tests — every implementation in analytics.top_k.

Oracles: known answers and TopKPolars (exact group-by counts), the reference. Semantics
(spec docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md): each eligible
column's k most frequent non-null values as arrow-rs text, count desc then first row;
`error` is SpaceSaving's over-count bound (0 exact, null HeavyKeeper); `dominated` = top-1
share ≥ dominance_threshold.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from analytics.top_k import TopK
from datagen import mixed_dtypes, top_k_streams
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
    with_metrics,
)

PKG = "analytics.top_k"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
# Columns whose k-th count is not tied (agreement precision is a lower bound under ties).
UNTIED = ["zipf_15", "zipf_11", "zipf_08", "zipf_str"]


def tops(impl: str, frame: pl.DataFrame, **params) -> dict[str, list | None]:
    out = run(load(impl), {"t": frame}, **params)
    return {
        c: None if t is None else [(i["value"], i["count"]) for i in t]
        for c, t in zip(out["col_a"], out["top_k"].to_list())
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "zipf": top_k_streams(20_000),
        "empty": pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"zipf": top_k_streams(200_000, columns=UNTIED)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers


@pytest.mark.parametrize("impl", ALL)
def test_fewer_values_than_k_are_exact(impl):
    s = pl.Series("s", ["a", "a", "a", None, "b", "b", "c"])
    assert tops(impl, s.to_frame())["s"] == [("a", 3), ("b", 2), ("c", 1)]


@pytest.mark.parametrize("impl", ALL)
def test_zipf_top_5(impl):
    frame = top_k_streams(100_000, columns=["zipf_15"])
    want = (
        frame["zipf_15"].value_counts(sort=True).head(5)["zipf_15"].cast(pl.String).to_list()
    )
    assert [v for v, _ in tops(impl, frame)["zipf_15"]] == want


@pytest.mark.parametrize("impl", ALL)
def test_ties_rank_by_first_row(impl):
    s = pl.Series("x", [3, 1, 2, 2, 1, 3])
    assert tops(impl, s.to_frame())["x"] == [("3", 2), ("1", 2), ("2", 2)]


@pytest.mark.parametrize("impl", ALL)
def test_all_null_gives_an_empty_list(impl):
    out = run(load(impl), {"t": pl.DataFrame({"x": pl.Series([None] * 4, dtype=pl.Int64)})})
    assert out["top_k"].to_list() == [[]]
    assert out["n_non_null"].to_list() == [0]
    assert out["dominated"].to_list() == [None]


@pytest.mark.parametrize("impl", ALL)
def test_values_render_as_arrow_text(impl):
    frame = pl.DataFrame(
        {
            "cat": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "enum": pl.Series(["p", "p", "q"], dtype=pl.Enum(["q", "p"])),
            "date": [date(2024, 1, 2)] * 2 + [date(2024, 1, 3)],
            "bool": [True, True, False],
            "float": [2.5, 2.5, 1.0],
            "dec": pl.Series([Decimal("1.50"), Decimal("1.50"), Decimal("2.00")], dtype=pl.Decimal(10, 2)),
            "bin": [b"ab", b"ab", b"c"],
        }
    )
    got = {c: t[0][0] for c, t in tops(impl, frame).items()}
    assert got == {
        "cat": "x",
        "enum": "p",
        "date": "2024-01-02",
        "bool": "true",
        "float": "2.5",
        "dec": "1.50",
        "bin": "ab",
    }


@pytest.mark.parametrize("impl", ALL)
def test_nested_and_wide_columns_are_ineligible(impl):
    frame = pl.DataFrame(
        {
            "l": pl.Series([[1], [2]], dtype=pl.List(pl.Int64)),
            "w": pl.Series([1, 2], dtype=pl.Int128),
            "i": [1, 1],
        }
    )
    out = run(load(impl), {"t": frame})
    assert out["status"].to_list() == ["ineligible", "ineligible", "computed"]


@pytest.mark.parametrize("impl", ALL)
def test_parameters_are_checked(impl):
    cls = load(impl)
    for bad in ({"k": 0}, {"k": 1.5}, {"memory_bytes": -1}, {"dominance_threshold": 1.5}):
        with pytest.raises(ValueError):
            cls(**bad)


def test_default_budget_follows_k():
    cls = reference(PKG)
    assert cls().memory_bytes == 16 * 1024
    assert cls(k=256).memory_bytes == 256 * 1024
    assert cls(k=5, memory_bytes=1000).memory_bytes == 1000


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions


def test_dominated():
    item = lambda n: {"value": "a", "count": n, "error": 0}  # noqa: E731
    cls = with_metrics(
        TopK,
        n_non_null=[10, 10, 0],
        top_k=[[item(6)], [item(4)], []],
        sketch_bytes=[None, None, None],
    )
    frame = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = run(cls, {"t": frame})
    assert out["dominated"].to_list() == [True, False, None]
```

- [ ] **Step 3: Run them to see them fail.** `$PY -m pytest tests/test_top_k.py -n 0 -q`. Expected: ImportError, `analytics.top_k.base` does not exist.

- [ ] **Step 4: Write `base.py`.**

```python
"""Top-k most frequent values per column (spec
docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md).

Implementations rank items as rows of the RANKED schema (one per column and rank: the row
of an occurrence, the count and the error bound); the base renders each value from its row
(arrow-rs text, `_plugin.render`, as Describe's min / max), concludes `dominated` and
compares sketches with the exact reference.
"""

from __future__ import annotations

import polars as pl

from analytics._dtypes import INTEGERS_64, STRING_LIKE
from analytics.base import Technique, check_unit, computed, metric_mismatches

ITEM = pl.Struct({"value": pl.String, "count": pl.UInt64, "error": pl.UInt64})
TOP_K = pl.List(ITEM)
ELIGIBLE = (
    *INTEGERS_64,
    pl.Boolean,
    pl.Float32,
    pl.Float64,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
    *STRING_LIKE,
    pl.Decimal,
    pl.Binary,
)
# The Rust kernels' output (and TopKPolars'): one row per (column, rank); a column without
# values has one row with null rank / row / count / error.
RANKED = {
    "col_name": pl.String,
    "rank": pl.UInt32,
    "row": pl.UInt64,
    "count": pl.UInt64,
    "error": pl.UInt64,
    "n_non_null": pl.UInt64,
    "sketch_bytes": pl.UInt64,
}
MIN_BUDGET = 16 * 1024  # the per-column HyperLogLog's size (p = 14)


def default_budget(k: int) -> int:
    return max(MIN_BUDGET, k * 1024)


class TopK(Technique):
    """Each eligible column's k most frequent non-null values, ranked by count desc then the
    row of an occurrence (the first row, for the exact reference), with `n_non_null` and the
    sketch's accounted `sketch_bytes` (null for the reference). `top_k` items are
    {value (arrow-rs text; null for non-UTF-8 binary), count, error}: error is the
    implementation's over-count bound (true count ∈ [count − error, count]), 0 when exact and
    null without a bound. `dominated` = top-1 count / n_non_null ≥ dominance_threshold.
    Eligible: the scalar dtypes the Rust encoder keys; List / Array / Struct / Null / Object
    and 128-bit integers are not. `memory_bytes` (default max(16 KiB, k × 1 KiB)) sizes the
    sketches; the reference ignores it.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {"n_non_null": pl.UInt64, "top_k": TOP_K, "sketch_bytes": pl.UInt64}
    CONCLUSIONS = {"dominated": pl.Boolean}
    MIN_PRECISION = 0.9
    MAX_ARE: float | None = None  # mean relative count error over found values, if checked

    def __init__(
        self,
        *,
        k: int = 5,
        memory_bytes: int | None = None,
        dominance_threshold: float = 0.5,
    ):
        super().__init__()
        if type(k) is not int or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if memory_bytes is not None and (type(memory_bytes) is not int or memory_bytes < 0):
            raise ValueError(f"memory_bytes must be a non-negative integer, got {memory_bytes!r}")
        check_unit("dominance_threshold", dominance_threshold)
        self.k = k
        self.memory_bytes = default_budget(k) if memory_bytes is None else memory_bytes
        self.dominance_threshold = dominance_threshold

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, ELIGIBLE)

    def describe(self, frames, combos):
        return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    def top_k_rows(self, frames, combos, ranked: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """metrics_frame for `combos` from each frame's RANKED rows, values rendered from
        their rows."""
        from analytics import _plugin

        parts = {
            frame: r.cast(RANKED).sort("col_name", "rank").partition_by("col_name", as_dict=True)
            for frame, r in ranked.items()
        }
        n_non_null, top, sketch_bytes = [], [], []
        for ((frame, col),) in combos:
            r = parts[frame][(col,)]
            items = r.filter(pl.col("rank").is_not_null())
            values = []
            if items.height:
                s = frames[frame][col].gather(items["row"])
                if s.dtype == pl.Binary:  # as text; non-UTF-8 → null, as arrow-rs renders it
                    s = s.cast(pl.String, strict=False)
                values = _plugin.render(s)
            top.append(
                [
                    {"value": v, "count": c, "error": e}
                    for v, c, e in zip(values, items["count"], items["error"])
                ]
            )
            n_non_null.append(r["n_non_null"][0])
            sketch_bytes.append(r["sketch_bytes"][0])
        return self.metrics_frame(
            combos,
            {"n_non_null": n_non_null, "top_k": top, "sketch_bytes": sketch_bytes},
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        top1 = pl.col("top_k").list.first().struct.field("count").cast(pl.Float64)
        share = top1 / pl.col("n_non_null").cast(pl.Float64)
        return out.with_columns(dominated=computed(share >= self.dominance_threshold))

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """`n_non_null` exact; aggregate precision (reported values the reference lists ÷
        values it lists — a lower bound of tie-aware precision) ≥ MIN_PRECISION; mean
        relative count error ≤ MAX_ARE when set; per-item checks (`_item_problems`).
        `dominated` is not compared: an estimate can cross the threshold."""
        keys = self.key_columns()
        problems = metric_mismatches(result, reference, keys, ["n_non_null"], 0.0, 0.0)
        if problems:
            return problems
        found = wanted = 0
        rel: list[float] = []
        rows = zip(
            result.select(keys).rows(), result["top_k"].to_list(), reference["top_k"].to_list()
        )
        for (df, col), got, want in rows:
            if want is None:
                continue
            true = {w["value"]: w["count"] for w in want}
            floor = want[-1]["count"] if len(want) == self.k else 0
            wanted += len(want)
            for item in got:
                t = true.get(item["value"])
                if t is not None:
                    found += 1
                    rel.append(abs(item["count"] - t) / t)
                problems += self._item_problems(f"{df}.{col}", item, t, floor)
        precision = found / wanted if wanted else 1.0
        if precision < self.MIN_PRECISION:
            problems.append(f"precision {precision:.3f} < {self.MIN_PRECISION}")
        are = sum(rel) / len(rel) if rel else 0.0
        if self.MAX_ARE is not None and are > self.MAX_ARE:
            problems.append(f"mean relative count error {are:.4f} > {self.MAX_ARE}")
        return problems

    def _item_problems(self, label: str, item: dict, true: int | None, floor: int) -> list[str]:
        """Per-item checks; `true` is the exact count when the reference lists the value,
        else None (its true count is then at most `floor`, the reference's k-th count)."""
        return []
```

- [ ] **Step 5: Write `polars.py`.**

```python
import polars as pl

from analytics.base import group_by_frame
from analytics.top_k.base import RANKED, TopK


class TopKPolars(TopK):
    """Reference: exact counts per column (Polars group_by), ranked by count desc then first
    row. Holds every distinct value: memory grows with the column's cardinality."""

    def _compute(self, frames, combos):
        ranked = {
            frame: pl.concat([self._ranked(frames[frame][c]) for ((_, c),) in group])
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)

    def _ranked(self, s: pl.Series) -> pl.DataFrame:
        n = s.len() - s.null_count()
        rows = (
            pl.DataFrame({"v": s})
            .with_row_index("row")
            .drop_nulls("v")
            .group_by("v")
            .agg(count=pl.len(), row=pl.col("row").min())
            .sort(["count", "row"], descending=[True, False])
            .head(self.k)
            .select("row", "count")
            .rows()
        )
        if not rows:
            rows = [(None, None)]
        ranks = [None] if rows[0][0] is None else list(range(1, len(rows) + 1))
        return pl.DataFrame(
            {
                "col_name": [s.name] * len(rows),
                "rank": ranks,
                "row": [r for r, _ in rows],
                "count": [c for _, c in rows],
                "error": [None if r is None else 0 for r, _ in rows],
                "n_non_null": [n] * len(rows),
                "sketch_bytes": [None] * len(rows),
            },
            schema=RANKED,
        )
```

- [ ] **Step 6: Run the tests.** `$PY -m pytest tests/test_top_k.py -n 0 -q`. Expected: everything passes. Only TopKPolars is parametrised so far; the agreement test collects nothing.
  - `top_k_rows` casts Binary to String before `render`: arrow-rs's cast of Polars' `binary_view` export is not relied on. If `pl.Series([b"ÿ"]).cast(pl.String, strict=False)` raises rather than giving null in this Polars version, wrap the cast and fall back to `[None] * len(s)` for that column.

- [ ] **Step 7: Commit.**

```bash
git add services/analytics/analytics/top_k tests/test_top_k.py
git commit -m "analytics.top_k: TopK base (rendering, dominated, agreement) and exact TopKPolars reference

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Rust implementation classes

**Files:**
- Create: `services/analytics/analytics/top_k/space_saving.py`, `heavy_keeper.py`
- Modify: `services/analytics/analytics/top_k/__init__.py`
- Modify: `tests/test_top_k.py`

- [ ] **Step 1: Add Rust-only tests** to `tests/test_top_k.py`, in block 3:

```python
@pytest.mark.parametrize("impl", OTHERS)
def test_a_budget_below_k_entries_raises(impl):
    with pytest.raises(ValueError, match="memory_bytes"):
        run(load(impl), {"t": pl.DataFrame({"x": [1, 2]})}, memory_bytes=64)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("k", [5, 256])
def test_sketch_bytes_fit_the_budget(impl, k):
    out = run(load(impl), {"t": pl.DataFrame({"x": [1, 2]})}, k=k)
    assert 0 < out["sketch_bytes"][0] <= max(16 * 1024, k * 1024)


def test_space_saving_is_exact_while_values_fit():
    s = pl.Series("x", [i for i in range(100) for _ in range(i + 1)])
    out = run(load(f"{PKG}:TopKSpaceSavingRust"), {"t": s.to_frame()})
    assert out["top_k"].to_list()[0] == [
        {"value": str(v), "count": v + 1, "error": 0} for v in (99, 98, 97, 96, 95)
    ]


def test_heavy_keeper_is_deterministic_per_seed():
    frame = top_k_streams(50_000, columns=["zipf_08"])
    cls = load(f"{PKG}:TopKHeavyKeeperRust")
    assert run(cls, {"t": frame}, seed=3).equals(run(cls, {"t": frame}, seed=3))
    with pytest.raises(ValueError):
        cls(seed=-1)
```

- [ ] **Step 2: Run them to see them fail.** Expected: ImportError / `AttributeError: TopKSpaceSavingRust`, because the classes do not exist yet.

- [ ] **Step 3: Write `space_saving.py`.**

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.top_k.base import TopK


class TopKSpaceSavingRust(TopK):
    """Rust extension `space_saving_top_k`: SpaceSaving with as many counters as fit
    `memory_bytes` (~40–56 B each); 2^20-row chunks summarised in parallel and merged in
    order (rayon across columns and chunks). Deterministic; true count ∈ [count − error,
    count] for every item, and every value above n / counters is found."""

    EXACT = False

    def _compute(self, frames, combos):
        ranked = {
            frame: _plugin.space_saving_top_k(
                frames[frame].select([c for ((_, c),) in group]), self.k, self.memory_bytes
            )
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)

    def _item_problems(self, label, item, true, floor):
        lower, upper = item["count"] - item["error"], item["count"]
        if true is not None and not lower <= true <= upper:
            return [f"{label}: {item['value']!r} true count {true} outside [{lower}, {upper}]"]
        if true is None and lower > floor:
            return [f"{label}: {item['value']!r} lower bound {lower} above the k-th count {floor}"]
        return []
```

- [ ] **Step 4: Write `heavy_keeper.py`.**

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.top_k.base import TopK


class TopKHeavyKeeperRust(TopK):
    """Rust extension `heavy_keeper_top_k`: HeavyKeeper (Gong et al., ATC 2018) with a heap of
    k and 2 × width (fingerprint, count) buckets filling the rest of `memory_bytes`; columns in
    parallel (rayon), each fed in row order (the sketch does not merge). Seeded per column
    from `seed` and the column name; no per-item bound (error null)."""

    EXACT = False
    MAX_ARE = 0.05

    def __init__(self, *, seed: int = 0, **params):
        super().__init__(**params)
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError(f"seed must be an integer in [0, 2**64 - 1], got {seed!r}")
        self.seed = seed

    def _compute(self, frames, combos):
        ranked = {
            frame: _plugin.heavy_keeper_top_k(
                frames[frame].select([c for ((_, c),) in group]),
                self.k,
                self.memory_bytes,
                self.seed,
            )
            for frame, group in group_by_frame(combos).items()
        }
        return self.top_k_rows(frames, combos, ranked)
```

- [ ] **Step 5: Register them** in `__init__.py`:

```python
"""Top-k most frequent values per column (per-column scope). See TopK."""

from analytics.base import lazy_attributes
from analytics.top_k.base import TopK
from analytics.top_k.heavy_keeper import TopKHeavyKeeperRust
from analytics.top_k.space_saving import TopKSpaceSavingRust

REFERENCE = "TopKPolars"
IMPLEMENTATIONS = ("TopKHeavyKeeperRust", "TopKSpaceSavingRust", "TopKPolars")

__getattr__ = lazy_attributes(__name__, {"TopKPolars": ".polars"})
__all__ = [
    "TopK",
    "TopKHeavyKeeperRust",
    "TopKSpaceSavingRust",
    "TopKPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
```

- [ ] **Step 6: Run the whole file.** `$PY -m pytest tests/test_top_k.py -n 0 -q`. Expected: all pass, including contract, agreement and known answers for all three classes.
  - If `test_agrees_with_reference` fails, the message lists the precision / ARE / bound problems. A **bound violation** is a bug: fix the sketch, never the test. A **precision or ARE shortfall** on `zipf_08` is data, not a bug. Report the measured figures to the user before changing `UNTIED` or the thresholds; the spec fixes them at 0.9 / 0.05.
  - If `test_zipf_top_5` or `test_ties_rank_by_first_row` fails for a sketch, check the row each sketch records. SpaceSaving records the row where the key last entered a counter, and HeavyKeeper the row where it entered the heap. With no evictions, both are the first row.

- [ ] **Step 7: Run the full lint and test suite.** `$PY scripts/check.py check`. Expected: all steps pass, including layering, clippy for both feature sets and every pytest file. Fix clippy findings in the new files only.

- [ ] **Step 8: Commit.**

```bash
git add services/analytics/analytics/top_k tests/test_top_k.py
git commit -m "analytics.top_k: TopKSpaceSavingRust and TopKHeavyKeeperRust

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Speed benchmark

**Files:**
- Create: `tests/performance/benchmark_top_k.py`

- [ ] **Step 1: Write the script.**

```python
"""
Top-k speed benchmark: TopKHeavyKeeperRust vs TopKSpaceSavingRust vs TopKPolars (reference).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_top_k.py

Shapes: large_dataset.arrow (realistic mix); narrow / long (parallelism inside a column:
SpaceSaving merges 2^20-row chunks, HeavyKeeper cannot); wide (parallelism across columns);
zipf 10M x 4 (stream-like skew, 1M possible values). `agrees` on large_dataset.arrow can be
false from ties at the k-th count (agreement precision is a lower bound there).
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, top_k_streams

if __name__ == "__main__":
    run(
        "analytics.top_k",
        [
            large_dataset(),
            Dataset(
                "narrow 10M x 4",
                lambda: {"t": low_cardinality(10_000_000, 4, cardinality=1_000)},
            ),
            Dataset(
                "wide 200K x 100",
                lambda: {"t": low_cardinality(200_000, 100, cardinality=1_000)},
            ),
            Dataset(
                "zipf 10M x 4",
                lambda: {
                    "t": top_k_streams(
                        10_000_000, columns=["zipf_15", "zipf_11", "zipf_08", "flat"]
                    )
                },
            ),
        ],
    )
```

- [ ] **Step 2: Run it.** `$PY tests/performance/benchmark_top_k.py` (several minutes). Expected: a report table per dataset, and `tests/performance/results/top_k*.parquet` written. Keep the printed table for Task 10.

- [ ] **Step 3: Commit.**

```bash
git add tests/performance/benchmark_top_k.py
git commit -m "Benchmark: top-k speed (HeavyKeeper, SpaceSaving, Polars)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Accuracy and memory benchmark

**Files:**
- Create: `tests/performance/benchmark_top_k_accuracy.py`

- [ ] **Step 1: Write the script.**

```python
"""
Top-k accuracy and memory report: TopKHeavyKeeperRust and TopKSpaceSavingRust against the
exact TopKPolars on datagen.top_k_streams (10M rows: Zipf α = 1.5 / 1.1 / 0.8, flat, skewed,
string), k ∈ {5, 256} × memory_bytes ∈ {¼, 1, 4} × the default max(16 KiB, k × 1 KiB).

The parent writes the data once to a temporary Arrow IPC file and computes every column's
exact counts. Each (column, k, budget, implementation) then runs in a fresh child process
that reads its one column, notes its working set, runs result() once, and reports peak_mb
(peak working set minus the post-load working set, as benchmark_ams_entropy.py), time_s and
the reported items. The reference runs once per (column, k); its budget is null.

Scores, per run (`top` = the reported items, true = exact counts):
  precision            tie-aware: reported values whose true count ≥ the true k-th count,
                       ÷ min(k, distinct values)
  are, aae             mean relative / absolute count error over reported values (the paper's)
  max_bound_violation  SpaceSaving only: max over items of how far the true count falls
                       outside [count − error, count] (0 = bounds hold)
  sketch_bytes         the sketch's accounted bytes (null for the reference)

--rows N (e.g. --rows 100_000_000): the scale run instead — zipf_11 only, k = 5, 1× budget,
written to results/top_k_scale.parquet.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_top_k_accuracy.py
Results -> tests/performance/results/top_k_accuracy.parquet
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
from benchmark_ams_entropy import memory_mb
from harness import RESULTS_DIR

from datagen import TOP_K_COLUMNS, top_k_streams

SKETCHES = ("TopKHeavyKeeperRust", "TopKSpaceSavingRust")
REFERENCE = "TopKPolars"


def default_budget(k: int) -> int:
    return max(16 * 1024, k * 1024)


def child(data: str, column: str, name: str, k: int, budget: int, out: str) -> None:
    import analytics.top_k

    frame = pl.read_ipc(data, columns=[column], memory_map=False)
    cls = getattr(analytics.top_k, name)
    gc.collect()
    current_before, peak_before = memory_mb()
    start = time.perf_counter()
    result = cls(k=k, memory_bytes=budget).add({"t": frame}).result()
    elapsed = time.perf_counter() - start
    _, peak_after = memory_mb()
    result.select("top_k", "sketch_bytes").write_parquet(out)
    print(
        json.dumps(
            {
                "peak_mb": peak_after - current_before,
                "peak_is_lower_bound": peak_after == peak_before,
                "time_s": elapsed,
            }
        )
    )


def truth_of(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """value (text) → exact count, every distinct non-null value, count desc."""
    return (
        frame.select(pl.col(column).cast(pl.String).alias("value"))
        .drop_nulls()
        .group_by("value")
        .agg(true=pl.len())
        .sort("true", descending=True)
    )


def score(top: list[dict], truth: pl.DataFrame, k: int) -> dict:
    floor = truth["true"][k - 1] if truth.height >= k else 0
    reported = pl.DataFrame(
        top, schema={"value": pl.String, "count": pl.UInt64, "error": pl.UInt64}
    ).join(truth, on="value", how="left")
    count = reported["count"].cast(pl.Int64)
    true = reported["true"].cast(pl.Int64)
    err = (count - true).abs()
    violation = None
    if reported["error"].null_count() == 0 and reported.height:
        lower = count - reported["error"].cast(pl.Int64)
        violation = max(0, (lower - true).max(), (true - count).max())
    return {
        "precision": (true >= floor).sum() / min(k, truth.height),
        "are": (err / true).mean(),
        "aae": err.mean(),
        "max_bound_violation": violation,
    }


def run_one(data, column, name, k, budget, out) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--child", data, column, name, str(k), str(budget), out],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        timeout=1800,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(n_rows: int, scale: bool) -> None:
    columns = ["zipf_11"] if scale else list(TOP_K_COLUMNS)
    ks = (5,) if scale else (5, 256)
    multipliers = (1.0,) if scale else (0.25, 1.0, 4.0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        data = str(Path(tmp) / "streams.arrow")
        frame = top_k_streams(n_rows, columns=columns)
        frame.write_ipc(data)
        for column in columns:
            truth = truth_of(frame, column)
            for k in ks:
                runs = [(REFERENCE, None)] + [
                    (name, int(default_budget(k) * m)) for m in multipliers for name in SKETCHES
                ]
                for name, budget in runs:
                    row = {
                        "n_rows": n_rows,
                        "column": column,
                        "k": k,
                        "budget_bytes": budget,
                        "implementation": name,
                        "status": "ok",
                    }
                    rows.append(row)
                    print(f"  {column} k={k} budget={budget}: {name} ...", file=sys.stderr, flush=True)
                    out = str(Path(tmp) / "out.parquet")
                    try:
                        info = run_one(data, column, name, k, budget or default_budget(k), out)
                    except subprocess.CalledProcessError as exc:
                        row["status"] = "error"
                        print("\n".join((exc.stderr or "").splitlines()[-20:]), file=sys.stderr)
                        continue
                    except subprocess.TimeoutExpired:
                        row["status"] = "timeout"
                        continue
                    result = pl.read_parquet(out)
                    row.update(info)
                    row["sketch_bytes"] = result["sketch_bytes"][0]
                    row.update(score(result["top_k"][0].to_list(), truth, k))
    df = pl.DataFrame(rows, infer_schema_length=None)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_parquet(RESULTS_DIR / ("top_k_scale.parquet" if scale else "top_k_accuracy.parquet"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        a = sys.argv[2:]
        child(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--rows", type=int, default=None)
        args = parser.parse_args()
        main(args.rows or 10_000_000, scale=args.rows is not None)
```

- [ ] **Step 2: Smoke-run on a small stream.** `$PY tests/performance/benchmark_top_k_accuracy.py --rows 200000`. Expected: 3 rows (reference + 2 sketches) for zipf_11. SpaceSaving's `max_bound_violation` is 0, and precision is 1.0 for both sketches.

- [ ] **Step 3: Full run.** `$PY tests/performance/benchmark_top_k_accuracy.py` (about 15–30 min: 6 columns × 2 k × (1 + 6) runs). Then the scale run: `$PY tests/performance/benchmark_top_k_accuracy.py --rows 100000000` (needs about 3 GB free for generation). Keep both printed tables for Task 10.

- [ ] **Step 4: Commit.**

```bash
git add tests/performance/benchmark_top_k_accuracy.py
git commit -m "Benchmark: top-k accuracy, memory and scale report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Documentation and results

**Files:**
- Modify: `CLAUDE.md` (stage only your hunks: `git add -p CLAUDE.md`)
- Modify: `docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md`

- [ ] **Step 1: CLAUDE.md.** Under "## 1. Per-column", after the Describe entry, add:

```markdown
**Top-k — `analytics.top_k`** (`TopKHeavyKeeperRust`, `TopKSpaceSavingRust`, ★`TopKPolars`). Each column's k most frequent non-null values (spec: docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md). Metrics `n_non_null`, `top_k` (`List<Struct<value: Utf8, count: UInt64, error: UInt64>>`, count desc then row; values rendered by arrow-rs via `_plugin.render` in the base, from a row per item) and `sketch_bytes`; conclusion `dominated` = top-1 share ≥ `dominance_threshold=0.5`. `error` = SpaceSaving's over-count bound (true ∈ [count − error, count]), 0 exact, null HeavyKeeper. Keywords `k=5`, `memory_bytes=None` → max(16 KiB, k × 1 KiB) (both sketches sized to fit it; too small for k → ValueError), `seed=0` (HeavyKeeper). Rust: techniques/single_column/unordered/{top_k.rs (shared indexed min-heap `Counters`), space_saving.rs (mergeable; 2²⁰-row chunks merged in order), heavy_keeper.rs (d = 2, b = 1.08, 32-bit fingerprints; no merge)}. Agreement: `n_non_null` exact, precision vs the reference's list ≥ 0.9, SpaceSaving bounds hold, HeavyKeeper mean relative error ≤ 5%. Benchmarks: tests/performance/benchmark_top_k.py (speed), benchmark_top_k_accuracy.py (tie-aware precision, ARE/AAE, bounds, peak memory; `--rows N` scale run). **Outcome:** <one or two sentences from §8, written in Step 2>.
```

In "# Analytical Functions", also add `top_k` to the list of `_plugin` functions in the "Private" paragraph:

```
`space_saving_top_k`, `heavy_keeper_top_k`;
```

Add `analytics.top_k.base` beside `analytics.describe.base` as a `render` caller. In the project structure, add `top_k` to the `<technique>` list and `top_k, space_saving, heavy_keeper` to `single_column/unordered/{…}`. In "Current Focus", change "All eight techniques" to "All nine techniques" and add "Top-k".

- [ ] **Step 2: Spec §8 Results.** Append to the spec a "## 8. Results (2026-MM-DD)" section, in the style of the AMS spec §7, built from Tasks 8–9's tables:
  - **Accuracy:** a table of precision and ARE per column (zipf_15 … zipf_str) for both sketches at k = 5 and k = 256, 1× budget. Note where ¼× / 4× change the picture. Give SpaceSaving's `max_bound_violation` (expected 0).
  - **Memory:** `sketch_bytes` per k, and `peak_mb` of each implementation on zipf_11 at 10M rows and on the scale run.
  - **Speed:** median times on the four speed datasets, plus the parallel speedups. In particular, does SpaceSaving's chunk merge beat HeavyKeeper on the narrow 10M × 4 shape?
  - **Conclusion:** which sketch the recommender-integration spec should use, and why: accuracy at equal memory on light skew (zipf_08, flat), the guarantees, and mergeability.

  Write in the measured numbers. If a run errored or timed out, say so in the section.

- [ ] **Step 3: Run the full checks.** `$PY scripts/check.py check`. Expected: pass.

- [ ] **Step 4: Commit.**

```bash
git add -f docs/superpowers/specs/2026-10-12-heavy-hitters-top-k-design.md
git add -p CLAUDE.md
git commit -m "Document the top-k technique; spec results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
