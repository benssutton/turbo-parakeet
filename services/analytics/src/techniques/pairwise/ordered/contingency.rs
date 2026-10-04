// ─────────────────────────────────────────────────────────────────────────────
// Shared contingency-table builder — used by chi_squared.rs and ari.rs
// ─────────────────────────────────────────────────────────────────────────────
//
// Both chi-squared and ARI reduce a column pair to the same object: the joint
// distribution over rows where BOTH columns are non-null (pairwise deletion),
// plus the two marginals. This module builds that table once so the two
// consumers cannot drift apart on counting or null policy.
//
// Counting reuses the dense-id strategy from entropy.rs: the joint key is
// a_id·Kb + b_id (a single integer, no tuple hashing). Joint spaces ≤ FLAT_MAX
// are counted in a flat thread-local array with a touched-slot list for
// O(distinct) reset; larger spaces fall back to a u64-keyed hash map. The pair
// product always fits u64 because dense ids are u32.

use foldhash::fast::RandomState as FoldHashFast;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::common::DenseColumn;

pub(crate) struct ContingencyTable {
    /// Observed non-zero cells: (a_id, b_id, count).
    pub cells: Vec<(u32, u32, u64)>,
    /// Marginal counts over valid rows, indexed by dense id (0..card).
    /// Ids unseen among valid rows (including a null id) hold 0.
    pub marg_a: Vec<u64>,
    pub marg_b: Vec<u64>,
    /// Rows where both columns are non-null.
    pub n_valid: u64,
}

const FLAT_MAX: u64 = 1 << 20; // 4 MB of u32 counts per worker thread

thread_local! {
    static FLAT_COUNTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static TOUCHED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static FREQ: RefCell<HashMap<u64, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
}

/// Rows where neither column carries its null id: each adds to the marginals and
/// passes its joint key (`a·kb + b`) to `count_pair`. Returns `n_valid`. The closure
/// is inlined, so the hot loop is the same plain compare-and-count as a hand-written one.
#[inline]
fn tally(
    a: &DenseColumn,
    b: &DenseColumn,
    n_rows: usize,
    marg_a: &mut [u64],
    marg_b: &mut [u64],
    mut count_pair: impl FnMut(u64),
) -> u64 {
    let kb = b.card as u64;
    // Dense ids run 0..card-1, so u32::MAX can never be a real id — it is a
    // safe "no null id" sentinel that lets the hot loop use a plain compare.
    let a_null = a.null_id.unwrap_or(u32::MAX);
    let b_null = b.null_id.unwrap_or(u32::MAX);
    let mut n_valid = 0u64;
    for i in 0..n_rows {
        let (ai, bi) = (a.ids[i], b.ids[i]);
        if ai == a_null || bi == b_null {
            continue;
        }
        marg_a[ai as usize] += 1;
        marg_b[bi as usize] += 1;
        n_valid += 1;
        count_pair(ai as u64 * kb + bi as u64);
    }
    n_valid
}

/// Joint space small enough for flat-array counting (thread-local scratch array and a
/// touched-slot list, so the reset is O(distinct), not O(space)).
fn contingency_flat(
    a: &DenseColumn,
    b: &DenseColumn,
    n_rows: usize,
    space: usize,
) -> ContingencyTable {
    let kb = b.card as u64;
    let mut marg_a = vec![0u64; a.card as usize];
    let mut marg_b = vec![0u64; b.card as usize];
    FLAT_COUNTS.with(|counts_cell| {
        TOUCHED.with(|touched_cell| {
            let mut counts = counts_cell.borrow_mut();
            let mut touched = touched_cell.borrow_mut();
            if counts.len() < space {
                counts.resize(space, 0);
            }
            let n_valid = tally(a, b, n_rows, &mut marg_a, &mut marg_b, |key| {
                let slot = &mut counts[key as usize];
                if *slot == 0 {
                    touched.push(key as u32);
                }
                *slot += 1;
            });
            let mut cells = Vec::with_capacity(touched.len());
            for &t in touched.iter() {
                let cnt = counts[t as usize] as u64;
                cells.push(((t as u64 / kb) as u32, (t as u64 % kb) as u32, cnt));
                counts[t as usize] = 0;
            }
            touched.clear();
            ContingencyTable {
                cells,
                marg_a,
                marg_b,
                n_valid,
            }
        })
    })
}

/// Joint space too large for a flat array: counts in a hash map.
fn contingency_hashed(a: &DenseColumn, b: &DenseColumn, n_rows: usize) -> ContingencyTable {
    let kb = b.card as u64;
    let mut marg_a = vec![0u64; a.card as usize];
    let mut marg_b = vec![0u64; b.card as usize];
    FREQ.with(|freq_cell| {
        let mut freq = freq_cell.borrow_mut();
        freq.clear();
        let n_valid = tally(a, b, n_rows, &mut marg_a, &mut marg_b, |key| {
            *freq.entry(key).or_insert(0) += 1;
        });
        let cells = freq
            .iter()
            .map(|(&k, &c)| ((k / kb) as u32, (k % kb) as u32, c))
            .collect();
        ContingencyTable {
            cells,
            marg_a,
            marg_b,
            n_valid,
        }
    })
}

/// Build the drop-null contingency table for one column pair.
///
/// Rows where either column carries its null id are excluded from cells,
/// marginals, and `n_valid`.
pub(crate) fn build_contingency(
    a: &DenseColumn,
    b: &DenseColumn,
    n_rows: usize,
) -> ContingencyTable {
    let space = (a.card as u64) * (b.card as u64); // fits u64: both factors are u32
    if space <= FLAT_MAX {
        contingency_flat(a, b, n_rows, space as usize)
    } else {
        contingency_hashed(a, b, n_rows)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{densify, encode_series};
    use polars::prelude::*;
    use std::collections::HashMap as StdHashMap;

    fn dense(series: &Series) -> DenseColumn {
        densify(&encode_series(series).unwrap())
    }

    /// Collect cells into a map keyed by (a_id, b_id) since cell order is unspecified.
    fn cell_map(t: &ContingencyTable) -> StdHashMap<(u32, u32), u64> {
        t.cells.iter().map(|&(a, b, c)| ((a, b), c)).collect()
    }

    #[test]
    fn test_basic_counts() {
        // a=[0,0,1,2], b=[0,0,1,1] → cells {(0,0):2, (1,1):1, (2,1):1}, n=4.
        let a = dense(&Series::new("a".into(), &[0i32, 0, 1, 2]));
        let b = dense(&Series::new("b".into(), &[0i32, 0, 1, 1]));
        let t = build_contingency(&a, &b, 4);
        assert_eq!(t.n_valid, 4);
        assert_eq!(t.cells.len(), 3);
        let cells = cell_map(&t);
        // Dense ids follow first-appearance order: a → {0:0, 1:1, 2:2}, b → {0:0, 1:1}.
        assert_eq!(cells[&(0, 0)], 2);
        assert_eq!(cells[&(1, 1)], 1);
        assert_eq!(cells[&(2, 1)], 1);
        assert_eq!(t.marg_a, vec![2, 1, 1]);
        assert_eq!(t.marg_b, vec![2, 2]);
    }

    #[test]
    fn test_null_rows_dropped() {
        // Row 2 (a null) and row 3 (b null) are excluded everywhere.
        let a = dense(&Series::new(
            "a".into(),
            &[Some(1i32), Some(1), None, Some(2), Some(2)],
        ));
        let b = dense(&Series::new(
            "b".into(),
            &[Some(7i32), Some(7), Some(7), None, Some(8)],
        ));
        let t = build_contingency(&a, &b, 5);
        assert_eq!(t.n_valid, 3);
        let total_cells: u64 = t.cells.iter().map(|&(_, _, c)| c).sum();
        assert_eq!(total_cells, 3);
        // Null ids hold zero marginal counts.
        let a_null = a.null_id.unwrap() as usize;
        let b_null = b.null_id.unwrap() as usize;
        assert_eq!(t.marg_a[a_null], 0);
        assert_eq!(t.marg_b[b_null], 0);
    }

    #[test]
    fn test_sums_consistent() {
        // Σ cells == Σ marg_a == Σ marg_b == n_valid, on data with nulls.
        let a = dense(&Series::new(
            "a".into(),
            &(0..100)
                .map(|i| if i % 7 == 0 { None } else { Some(i % 5) })
                .collect::<Vec<Option<i32>>>(),
        ));
        let b = dense(&Series::new(
            "b".into(),
            &(0..100)
                .map(|i| if i % 11 == 0 { None } else { Some(i % 3) })
                .collect::<Vec<Option<i32>>>(),
        ));
        let t = build_contingency(&a, &b, 100);
        let s_cells: u64 = t.cells.iter().map(|&(_, _, c)| c).sum();
        let s_a: u64 = t.marg_a.iter().sum();
        let s_b: u64 = t.marg_b.iter().sum();
        assert_eq!(s_cells, t.n_valid);
        assert_eq!(s_a, t.n_valid);
        assert_eq!(s_b, t.n_valid);
    }

    #[test]
    fn test_flat_and_hash_paths_agree() {
        // 1500 distinct ids per column → space 2.25M > FLAT_MAX → hash path.
        // Verify the hash-map counting path against an independent brute-force
        // reference computed directly from the dense ids. The flat path is
        // covered separately by test_basic_counts, test_null_rows_dropped,
        // and test_sums_consistent, which all use small joint spaces.
        let va: Vec<i32> = (0..1500).collect();
        let vb: Vec<i32> = (0..1500).map(|i| (i * 7) % 1500).collect();
        let a = dense(&Series::new("a".into(), &va));
        let b = dense(&Series::new("b".into(), &vb));
        let t = build_contingency(&a, &b, 1500);
        assert_eq!(t.n_valid, 1500);

        let mut reference: StdHashMap<(u32, u32), u64> = StdHashMap::new();
        for i in 0..1500 {
            *reference.entry((a.ids[i], b.ids[i])).or_insert(0) += 1;
        }
        assert_eq!(cell_map(&t), reference);
    }

    #[test]
    fn test_all_null_column() {
        let a = dense(&Series::new("a".into(), &[None::<i32>, None, None]));
        let b = dense(&Series::new("b".into(), &[1i32, 2, 3]));
        let t = build_contingency(&a, &b, 3);
        assert_eq!(t.n_valid, 0);
        assert!(t.cells.is_empty());
        assert!(t.marg_b.iter().all(|&c| c == 0));
    }
}
