//! Block reservoir (streaming spec §4.5): the stream is cut into contiguous blocks of
//! `block_rows` rows; a uniform random sample of `capacity` blocks is kept by Li's
//! Algorithm L (seeded). Blocks keep row order, so ZSTD sizes measured on them match
//! an IPC file written in `block_rows` batches. A kept block holds compact copies of
//! its rows, one piece per batch it spans, so it pins no batch buffers.

use arrow_array::ArrayRef;
use arrow_schema::FieldRef;

use crate::recommenders::streaming::partial::copy_rows;

/// Algorithm L's state, `Copy` so that `feed` can plan on a copy and commit only
/// once every copy of the batch's rows has succeeded.
#[derive(Clone, Copy, Debug)]
struct Cursor {
    rng: u64,
    w: f64,
    /// Index of the next block to keep once the reservoir is full.
    next: u64,
    /// Blocks completed so far (= the index of the block in progress).
    seen: u64,
    kept: usize,
    /// Rows in the block in progress.
    rows: u64,
    /// Whether the block in progress will be kept.
    keep: bool,
}

impl Cursor {
    /// SplitMix64, the same finaliser as describe.rs `subset`.
    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Next index to keep: the block just closed (`seen - 1`) plus a geometric skip.
    fn skip(&mut self) {
        let jump = (self.unit().ln() / (1.0 - self.w).ln()).floor();
        self.next = self.seen - 1 + jump as u64 + 1;
    }

    /// Closes the block in progress; returns its slot (None: dropped).
    fn close(&mut self, capacity: usize) -> Option<usize> {
        let keep = self.keep;
        self.seen += 1;
        self.rows = 0;
        let slot = if !keep {
            None
        } else if self.kept < capacity {
            self.kept += 1;
            if self.kept == capacity {
                self.w = (self.unit().ln() / capacity as f64).exp();
                self.skip();
            }
            Some(self.kept - 1)
        } else {
            let j = ((self.unit() * capacity as f64) as usize).min(capacity - 1);
            self.w *= (self.unit().ln() / capacity as f64).exp();
            self.skip();
            Some(j)
        };
        self.keep = capacity > 0 && (self.kept < capacity || self.seen == self.next);
        slot
    }
}

/// One batch's share of a block: `rows` rows of the columns that batch carried.
#[derive(Clone)]
pub(crate) struct Piece {
    pub rows: u64,
    pub cols: Vec<(FieldRef, ArrayRef)>,
}

#[derive(Clone, Default)]
pub(crate) struct Block {
    pub rows: u64,
    pub pieces: Vec<Piece>,
}

pub(crate) struct Reservoir {
    capacity: usize,
    block_rows: u64,
    cur: Cursor,
    kept: Vec<Block>,
    current: Block,
}

impl Reservoir {
    /// `capacity` blocks of `block_rows` (≥ 1) rows.
    pub(crate) fn new(capacity: usize, block_rows: u64, seed: u64) -> Self {
        Reservoir {
            capacity,
            block_rows,
            cur: Cursor {
                rng: seed ^ 0xB10C_B10C_B10C_B10C,
                w: 1.0,
                next: 0,
                seen: 0,
                kept: 0,
                rows: 0,
                keep: capacity > 0,
            },
            kept: Vec::new(),
            current: Block::default(),
        }
    }

    /// Adds `rows` rows whose columns are `cols`. On error nothing changes.
    pub(crate) fn feed(&mut self, cols: &[(FieldRef, ArrayRef)], rows: u64) -> Result<(), String> {
        // Plan on a copy of the cursor: (start, len, kept?, close → slot).
        let mut cur = self.cur;
        let mut segments = Vec::new();
        let mut start = 0;
        while start < rows {
            let len = (self.block_rows - cur.rows).min(rows - start);
            let keep = cur.keep;
            cur.rows += len;
            let close = (cur.rows == self.block_rows).then(|| cur.close(self.capacity));
            segments.push((start, len, keep, close));
            start += len;
        }
        // Copy the kept rows (the only fallible step), then commit.
        let mut pieces = Vec::with_capacity(segments.len());
        for &(start, len, keep, _) in &segments {
            pieces.push(if keep {
                let cols = cols
                    .iter()
                    .map(|(f, a)| Ok((f.clone(), copy_rows(a, start as usize, len as usize)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                Some(Piece { rows: len, cols })
            } else {
                None
            });
        }
        for ((_, len, _, close), piece) in segments.into_iter().zip(pieces) {
            self.current.rows += len;
            self.current.pieces.extend(piece);
            if let Some(slot) = close {
                let block = std::mem::take(&mut self.current);
                match slot {
                    Some(j) if j < self.kept.len() => self.kept[j] = block,
                    Some(_) => self.kept.push(block),
                    None => {}
                }
            }
        }
        self.cur = cur;
        Ok(())
    }

    /// The sampled blocks: the kept ones, plus the block in progress while the
    /// reservoir is not yet full (so a stream shorter than the reservoir is measured
    /// completely).
    pub(crate) fn blocks(&self) -> Vec<&Block> {
        let mut out: Vec<&Block> = self.kept.iter().collect();
        if self.kept.len() < self.capacity && self.current.rows > 0 {
            out.push(&self.current);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field};

    use super::*;

    fn col(v: std::ops::Range<i64>) -> Vec<(FieldRef, ArrayRef)> {
        vec![(
            Arc::new(Field::new("a", DataType::Int64, true)),
            Arc::new(Int64Array::from_iter_values(v)) as ArrayRef,
        )]
    }

    fn values(b: &Block) -> Vec<i64> {
        b.pieces
            .iter()
            .flat_map(|p| p.cols[0].1.as_primitive::<Int64Type>().values().to_vec())
            .collect()
    }

    #[test]
    fn keeps_every_block_until_full() {
        let mut r = Reservoir::new(4, 3, 0);
        r.feed(&col(0..4), 4).unwrap();
        r.feed(&col(4..10), 6).unwrap();
        let got: Vec<Vec<i64>> = r.blocks().iter().map(|b| values(b)).collect();
        assert_eq!(
            got,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
        );
        assert_eq!(r.blocks()[1].pieces.len(), 2); // spans both batches
    }

    #[test]
    fn capacity_zero_keeps_nothing() {
        let mut r = Reservoir::new(0, 2, 0);
        r.feed(&col(0..10), 10).unwrap();
        assert!(r.blocks().is_empty());
    }

    #[test]
    fn same_seed_same_sample() {
        let sample = |seed| {
            let mut r = Reservoir::new(3, 1, seed);
            for i in 0..100 {
                r.feed(&col(i..i + 1), 1).unwrap();
            }
            let mut v: Vec<i64> = r.blocks().iter().flat_map(|b| values(b)).collect();
            v.sort();
            v
        };
        assert_eq!(sample(1), sample(1));
        assert_ne!(sample(1), sample(2));
    }

    #[test]
    fn blocks_are_sampled_uniformly() {
        // 10 one-row blocks, 2 kept: each block is kept with probability 1/5.
        let mut hits = [0u32; 10];
        let trials = 20_000;
        for seed in 0..trials {
            let mut r = Reservoir::new(2, 1, seed);
            for i in 0..10 {
                r.feed(&col(i..i + 1), 1).unwrap();
            }
            for b in r.blocks() {
                hits[values(b)[0] as usize] += 1;
            }
        }
        let expected = trials as f64 * 2.0 / 10.0;
        for (i, &h) in hits.iter().enumerate() {
            assert!(
                (h as f64 - expected).abs() < 0.05 * expected,
                "block {i}: {h} vs {expected}"
            );
        }
    }
}
