//! Entropy from counts: the SIMD reduction shared by marginal and joint entropy.

use foldhash::fast::RandomState as FoldHashFast;
use std::cell::RefCell;
use std::collections::HashMap;
use wide::f64x4;

thread_local! {
    static COC: RefCell<HashMap<u64, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
}

/// Reduce per-key counts to entropy via the shared count-of-counts scratch map.
pub(crate) fn entropy_from_counts_iter(
    counts: impl Iterator<Item = u64>,
    logr: f64,
    r_f: f64,
) -> f64 {
    COC.with(|coc_cell| {
        let mut coc_map = coc_cell.borrow_mut();
        coc_map.clear();
        for count in counts {
            *coc_map.entry(count).or_insert(0) += 1;
        }
        let coc: Vec<(u64, u64)> = coc_map.drain().collect();
        entropy_from_count_of_counts(&coc, logr, r_f)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD entropy from count-of-counts
// ─────────────────────────────────────────────────────────────────────────────

/// Compute entropy in bits from count-of-counts pairs.
///
/// Each element in `coc` is `(count_value, multiplicity)`:
///   - `count_value` (c): how many times a particular key appeared
///   - `multiplicity` (n): how many distinct keys share that count
///
/// Formula: H = log2(r) - Σ n * c * log2(c) / r
///   (logr hoisted out: Σ n·c = r, so the logr term reduces to a constant)
///
/// Uses SIMD f64x4 processing 4 (c, n) pairs per iteration with scalar tail.
pub(crate) fn entropy_from_count_of_counts(coc: &[(u64, u64)], logr: f64, r_f: f64) -> f64 {
    let mut acc = f64x4::ZERO;
    let full_chunks = coc.len() / 4;

    for chunk_idx in 0..full_chunks {
        let base = chunk_idx * 4;
        let c = f64x4::from([
            coc[base].0 as f64,
            coc[base + 1].0 as f64,
            coc[base + 2].0 as f64,
            coc[base + 3].0 as f64,
        ]);
        let n = f64x4::from([
            coc[base].1 as f64,
            coc[base + 1].1 as f64,
            coc[base + 2].1 as f64,
            coc[base + 3].1 as f64,
        ]);
        acc += n * c * c.log2();
    }

    let mut sum: f64 = acc.reduce_add();
    for &(c_val, n_val) in &coc[full_chunks * 4..] {
        let c = c_val as f64;
        let n = n_val as f64;
        sum += n * c * c.log2();
    }
    logr - sum / r_f
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Count-of-counts entropy ────────────────────────────────────────────

    #[test]
    fn test_coc_uniform_4() {
        // 4 unique values, each count=1, r=4 → H = log2(4) = 2.0
        let coc = vec![(1u64, 4u64)]; // count=1 appears 4 times
        let r_f: f64 = 4.0;
        let logr = r_f.log2();
        let h = entropy_from_count_of_counts(&coc, logr, r_f);
        assert!((h - 2.0).abs() < 1e-10, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_coc_deterministic() {
        // 1 unique value, count=100, r=100 → H = 0.0
        let coc = vec![(100u64, 1u64)]; // count=100 appears 1 time
        let r_f: f64 = 100.0;
        let logr = r_f.log2();
        let h = entropy_from_count_of_counts(&coc, logr, r_f);
        assert!(h.abs() < 1e-10, "Expected 0.0, got {}", h);
    }

    #[test]
    fn test_coc_mixed_counts() {
        // 2 values with count=2, 1 value with count=1 → r=5
        // H = -(2*2*(log2(2)-log2(5)) + 1*1*(log2(1)-log2(5))) / 5
        //   = -(4*(1-2.32193) + 1*(0-2.32193)) / 5
        //   = -(4*(-1.32193) + (-2.32193)) / 5
        //   = -(-5.28772 + -2.32193) / 5
        //   = 7.60965 / 5 = 1.52193
        let coc = vec![(2u64, 2u64), (1u64, 1u64)];
        let r_f: f64 = 5.0;
        let logr = r_f.log2();
        let h = entropy_from_count_of_counts(&coc, logr, r_f);
        let expected = 1.52193;
        assert!(
            (h - expected).abs() < 1e-4,
            "Expected ~{}, got {}",
            expected,
            h
        );
    }
}
