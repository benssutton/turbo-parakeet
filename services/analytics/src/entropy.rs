use foldhash::fast::RandomState as FoldHashFast;
use rayon::prelude::*;
use wide::f64x4;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::shared::*;

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local scratch buffers
// ─────────────────────────────────────────────────────────────────────────────
//
// Each pair/triplet counts joint frequencies. Under Rayon these closures run
// thousands of times per worker thread (C(101,3) = 166650 triplets), so scratch
// storage is kept per worker thread and reset between calls — capacity is
// retained, so steady-state cost is zero allocations.
//
// Columns are dictionary-encoded to dense ids 0..card (see DenseColumn), so a
// joint key is a single integer: (a·Kb + b)·Kc + c. Two counting strategies:
//   - joint space ≤ FLAT_MAX → flat array indexing, no hashing at all;
//     `touched` records used slots so reset is O(distinct), not O(space).
//   - larger → hash map keyed by the combined u64 (u128 when Ka·Kb·Kc
//     overflows u64 — only possible past ~2.6M rows).
const FLAT_MAX: u64 = 1 << 20; // 4 MB of u32 counts per worker thread

thread_local! {
    static FLAT_COUNTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static TOUCHED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static FREQ: RefCell<HashMap<u64, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
    static FREQ_WIDE: RefCell<HashMap<u128, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
    static COC: RefCell<HashMap<u64, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
}

/// Reduce per-key counts to entropy via the shared count-of-counts scratch map.
fn entropy_from_counts_iter(
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

/// Count joint frequencies of pre-combined keys in the flat scratch array.
/// Caller guarantees every key < `space` and `space` ≤ FLAT_MAX.
fn entropy_flat(
    keys: impl Iterator<Item = usize>,
    space: usize,
    logr: f64,
    r_f: f64,
) -> f64 {
    FLAT_COUNTS.with(|counts_cell| {
        TOUCHED.with(|touched_cell| {
            let mut counts = counts_cell.borrow_mut();
            let mut touched = touched_cell.borrow_mut();
            if counts.len() < space {
                counts.resize(space, 0);
            }
            for key in keys {
                let slot = &mut counts[key];
                if *slot == 0 {
                    touched.push(key as u32);
                }
                *slot += 1;
            }
            let h = entropy_from_counts_iter(
                touched.iter().map(|&t| counts[t as usize] as u64),
                logr,
                r_f,
            );
            for &t in touched.iter() {
                counts[t as usize] = 0;
            }
            touched.clear();
            h
        })
    })
}

/// Joint entropy of two dense-encoded columns.
fn joint_entropy_pair(a: &DenseColumn, b: &DenseColumn, r: usize, logr: f64, r_f: f64) -> f64 {
    let kb = b.card as u64;
    // Cards are u32, so the pair product always fits u64.
    let space = (a.card as u64) * kb;
    if space <= FLAT_MAX {
        entropy_flat(
            (0..r).map(|i| (a.ids[i] as u64 * kb + b.ids[i] as u64) as usize),
            space as usize,
            logr,
            r_f,
        )
    } else {
        FREQ.with(|freq_cell| {
            let mut freq = freq_cell.borrow_mut();
            freq.clear();
            for i in 0..r {
                let key = a.ids[i] as u64 * kb + b.ids[i] as u64;
                *freq.entry(key).or_insert(0) += 1;
            }
            entropy_from_counts_iter(freq.values().copied(), logr, r_f)
        })
    }
}

/// Joint entropy of three dense-encoded columns.
fn joint_entropy_triple(
    a: &DenseColumn,
    b: &DenseColumn,
    c: &DenseColumn,
    r: usize,
    logr: f64,
    r_f: f64,
) -> f64 {
    let kb = b.card as u64;
    let kc = c.card as u64;
    let space = a.card as u128 * kb as u128 * kc as u128;
    if space <= FLAT_MAX as u128 {
        entropy_flat(
            (0..r).map(|i| {
                ((a.ids[i] as u64 * kb + b.ids[i] as u64) * kc + c.ids[i] as u64) as usize
            }),
            space as usize,
            logr,
            r_f,
        )
    } else if space <= u64::MAX as u128 {
        FREQ.with(|freq_cell| {
            let mut freq = freq_cell.borrow_mut();
            freq.clear();
            for i in 0..r {
                let key = (a.ids[i] as u64 * kb + b.ids[i] as u64) * kc + c.ids[i] as u64;
                *freq.entry(key).or_insert(0) += 1;
            }
            entropy_from_counts_iter(freq.values().copied(), logr, r_f)
        })
    } else {
        // Ka·Kb·Kc overflows u64: pack the three ids into a u128 instead.
        FREQ_WIDE.with(|freq_cell| {
            let mut freq = freq_cell.borrow_mut();
            freq.clear();
            for i in 0..r {
                let key = ((a.ids[i] as u128) << 64)
                    | ((b.ids[i] as u128) << 32)
                    | (c.ids[i] as u128);
                *freq.entry(key).or_insert(0) += 1;
            }
            entropy_from_counts_iter(freq.values().copied(), logr, r_f)
        })
    }
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
fn entropy_from_count_of_counts(coc: &[(u64, u64)], logr: f64, r_f: f64) -> f64 {
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

// ─────────────────────────────────────────────────────────────────────────────
// Pairwise (2-way)
// ─────────────────────────────────────────────────────────────────────────────

fn pairwise_entropy_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("col_a".into(), DataType::String),
        Field::new("col_b".into(), DataType::String),
        Field::new("entropy".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "pairwise_entropy".into(),
        DataType::Struct(fields),
    ))
}

pub(crate) fn pairwise_joint_entropy_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "pairwise_joint_entropy requires at least one column".into(),
        ));
    }

    let n_cols = inputs.len();

    // Fewer than 2 columns → no pairs, return empty struct.
    if n_cols < 2 {
        let col_a_s = StringChunked::from_iter(std::iter::empty::<&str>())
            .into_series()
            .with_name("col_a".into());
        let col_b_s = StringChunked::from_iter(std::iter::empty::<&str>())
            .into_series()
            .with_name("col_b".into());
        let entropy_s = Float64Chunked::from_vec("entropy".into(), vec![]).into_series();
        let struct_ca = StructChunked::from_series(
            "pairwise_entropy".into(),
            0,
            [col_a_s, col_b_s, entropy_s].iter(),
        )?;
        return Ok(struct_ca.into_series());
    }

    // Step 1: row count and log2(r), computed once.
    let r = inputs[0].len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate entropy on empty columns".into(),
        ));
    }
    let r_f = r as f64;
    let logr = r_f.log2();

    // Validate all columns have the same length.
    for (idx, series) in inputs.iter().enumerate() {
        if series.len() != r {
            return Err(PolarsError::ShapeMismatch(
                format!(
                    "All columns must have the same length: column {} has length {} but expected {}",
                    idx,
                    series.len(),
                    r
                )
                .into(),
            ));
        }
    }

    // Step 2: resolve pairs from kwargs or generate all N-choose-2.
    let pairs: Vec<(usize, usize)> = match &kwargs.pairs {
        Some(raw_pairs) => {
            let name_map: HashMap<String, usize> = inputs
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name().to_string(), i))
                .collect();
            resolve_pairs(raw_pairs, &name_map)?
        }
        None => (0..n_cols)
            .flat_map(|i| ((i + 1)..n_cols).map(move |j| (i, j)))
            .collect(),
    };

    let n_pairs = pairs.len();

    // Step 3: parallel dense cache (encode + dictionary-encode needed columns).
    // Nulls are folded in as their own dense id, so the per-pair loop needs no
    // null mask; a null is still a distinct category from every real value.
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    // Pre-collect column names to avoid per-thread allocations inside par_iter.
    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Steps 4–5: parallel entropy calculation across all pairs.
    let results: Vec<_> = pairs
        .par_iter()
        .map(|(i, j)| {
            let entropy = joint_entropy_pair(&cache[*i], &cache[*j], r, logr, r_f);
            Ok((col_names[*i].clone(), col_names[*j].clone(), entropy))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // Step 6: build struct series.
    let mut col_a_names = Vec::with_capacity(n_pairs);
    let mut col_b_names = Vec::with_capacity(n_pairs);
    let mut entropy_values = Vec::with_capacity(n_pairs);

    for (a, b, e) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        entropy_values.push(e);
    }

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s: &String| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s: &String| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let entropy_s = Float64Chunked::from_vec("entropy".into(), entropy_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_entropy".into(),
        n_pairs,
        [col_a_s, col_b_s, entropy_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

#[polars_expr(output_type_func=pairwise_entropy_output_type)]
fn pairwise_joint_entropy(inputs: &[Series], kwargs: PairwiseKwargs) -> PolarsResult<Series> {
    pairwise_joint_entropy_impl(inputs, kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Threeway (3-way)
// ─────────────────────────────────────────────────────────────────────────────

fn threeway_entropy_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("col_a".into(), DataType::String),
        Field::new("col_b".into(), DataType::String),
        Field::new("col_c".into(), DataType::String),
        Field::new("entropy".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "threeway_entropy".into(),
        DataType::Struct(fields),
    ))
}

pub(crate) fn threeway_joint_entropy_impl(
    inputs: &[Series],
    kwargs: ThreewayKwargs,
) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError(
            "threeway_joint_entropy requires at least three columns".into(),
        ));
    }

    let n_cols = inputs.len();

    // Step 1: row count and log2(r), computed once.
    let r = inputs[0].len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate entropy on empty columns".into(),
        ));
    }
    let r_f = r as f64;
    let logr = r_f.log2();

    // Validate all columns have the same length.
    for (idx, series) in inputs.iter().enumerate() {
        if series.len() != r {
            return Err(PolarsError::ShapeMismatch(
                format!(
                    "All columns must have the same length: column {} has length {} but expected {}",
                    idx,
                    series.len(),
                    r
                )
                .into(),
            ));
        }
    }

    // Step 2: resolve triplets from kwargs or generate all N-choose-3.
    let triplets: Vec<(usize, usize, usize)> = match &kwargs.triplets {
        Some(raw_triplets) => {
            let name_map: HashMap<String, usize> = inputs
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name().to_string(), i))
                .collect();
            resolve_triplets(raw_triplets, &name_map)?
        }
        None => (0..n_cols)
            .flat_map(|i| {
                ((i + 1)..n_cols)
                    .flat_map(move |j| ((j + 1)..n_cols).map(move |k| (i, j, k)))
            })
            .collect(),
    };

    let n_triplets = triplets.len();

    // Step 3: parallel dense cache (encode + dictionary-encode needed columns).
    // Nulls are folded in as their own dense id, so the per-triplet loop needs
    // no null mask; a null is still a distinct category from every real value.
    let needed: HashSet<usize> = triplets
        .iter()
        .flat_map(|(i, j, k)| [*i, *j, *k])
        .collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    // Pre-collect column names to avoid per-thread allocations inside par_iter.
    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Steps 4–5: parallel entropy calculation across all triplets.
    let results: Vec<_> = triplets
        .par_iter()
        .map(|(i, j, k)| {
            let entropy = joint_entropy_triple(&cache[*i], &cache[*j], &cache[*k], r, logr, r_f);
            Ok((
                col_names[*i].clone(),
                col_names[*j].clone(),
                col_names[*k].clone(),
                entropy,
            ))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // Step 6: build struct series.
    let mut col_a_names = Vec::with_capacity(n_triplets);
    let mut col_b_names = Vec::with_capacity(n_triplets);
    let mut col_c_names = Vec::with_capacity(n_triplets);
    let mut entropy_values = Vec::with_capacity(n_triplets);

    for (a, b, c, e) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        col_c_names.push(c);
        entropy_values.push(e);
    }

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s: &String| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s: &String| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let col_c_s = StringChunked::from_iter(col_c_names.iter().map(|s: &String| s.as_str()))
        .into_series()
        .with_name("col_c".into());
    let entropy_s = Float64Chunked::from_vec("entropy".into(), entropy_values).into_series();

    let struct_ca = StructChunked::from_series(
        "threeway_entropy".into(),
        n_triplets,
        [col_a_s, col_b_s, col_c_s, entropy_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

#[polars_expr(output_type_func=threeway_entropy_output_type)]
fn threeway_joint_entropy(inputs: &[Series], kwargs: ThreewayKwargs) -> PolarsResult<Series> {
    threeway_joint_entropy_impl(inputs, kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Marginal (single-column)
// ─────────────────────────────────────────────────────────────────────────────
//
// H(col) for every input column independently — one row per column, no
// combinatorics. Deliberately built on the shared null-safe encoder
// (encode_series via build_column_cache_par) and a plain frequency HashMap,
// NOT the dense-id/flat-array machinery pairwise/threeway use above: this
// keeps the marginal path an independent cross-check of the SIMD entropy
// reduction (entropy_from_counts_iter / entropy_from_count_of_counts) and the
// encoder, uncoupled from the newer dense re-encoding counting strategy.

fn marginal_entropy_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("col_name".into(), DataType::String),
        Field::new("entropy".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "marginal_entropy".into(),
        DataType::Struct(fields),
    ))
}

thread_local! {
    static FREQ_MARGINAL: RefCell<HashMap<(u64, bool), u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
}

pub(crate) fn marginal_entropy_impl(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "marginal_entropy requires at least one column".into(),
        ));
    }

    let n_cols = inputs.len();

    // Step 1: row count and log2(r), computed once.
    let r = inputs[0].len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate entropy on empty columns".into(),
        ));
    }
    let r_f = r as f64;
    let logr = r_f.log2();

    // Validate all columns have the same length.
    for (idx, series) in inputs.iter().enumerate() {
        if series.len() != r {
            return Err(PolarsError::ShapeMismatch(
                format!(
                    "All columns must have the same length: column {} has length {} but expected {}",
                    idx,
                    series.len(),
                    r
                )
                .into(),
            ));
        }
    }

    // Step 2: parallel column cache (every column is needed).
    let needed: HashSet<usize> = (0..n_cols).collect();
    let cache = build_column_cache_par(inputs, &needed)?;
    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Step 3: parallel entropy calculation, one column at a time.
    let results: Vec<(String, f64)> = (0..n_cols)
        .into_par_iter()
        .map(|i| {
            let col = &cache[i];

            // The key folds is_null in directly so a null is a distinct
            // category from every real value, matching pairwise/threeway's
            // null policy.
            let entropy = FREQ_MARGINAL.with(|freq_cell| {
                let mut freq = freq_cell.borrow_mut();
                freq.clear();
                for idx in 0..r {
                    let key = (col.values[idx], col.is_null[idx]);
                    *freq.entry(key).or_insert(0) += 1;
                }
                entropy_from_counts_iter(freq.values().copied(), logr, r_f)
            });

            (col_names[i].clone(), entropy)
        })
        .collect();

    // Step 4: build struct series.
    let col_name_s = StringChunked::from_iter(results.iter().map(|(name, _)| name.as_str()))
        .into_series()
        .with_name("col_name".into());
    let entropy_s = Float64Chunked::from_vec(
        "entropy".into(),
        results.iter().map(|(_, e)| *e).collect(),
    )
    .into_series();

    let struct_ca = StructChunked::from_series(
        "marginal_entropy".into(),
        n_cols,
        [col_name_s, entropy_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

#[polars_expr(output_type_func=marginal_entropy_output_type)]
fn marginal_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    marginal_entropy_impl(inputs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn no_pairs() -> PairwiseKwargs {
        PairwiseKwargs { pairs: None }
    }

    fn no_triplets() -> ThreewayKwargs {
        ThreewayKwargs { triplets: None }
    }

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

    // ── Null encoding (out-of-band mask) ───────────────────────────────────

    #[test]
    fn test_null_marked_in_mask() {
        let s = Series::new("test".into(), &[Some(1i32), None, Some(3)]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values, vec![1u64, 0, 3]); // null value canonicalised to 0
        assert_eq!(enc.is_null, vec![false, true, false]);
    }

    #[test]
    fn test_null_does_not_collide_with_zero() {
        // A real 0 and a null both store value 0; only the mask distinguishes them.
        let s = Series::new("test".into(), &[Some(0i32), None]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], 0);
        assert!(!enc.is_null[0]);
        assert!(enc.is_null[1]);
    }

    #[test]
    fn test_negative_one_is_not_null() {
        // Regression: -1 at every signed width sign-extends to u64::MAX, which used
        // to collide with the old NULL_SENTINEL. It must now be a real, non-null value.
        for s in [
            Series::new("i8".into(), &[Some(-1i8), Some(0)]),
            Series::new("i16".into(), &[Some(-1i16), Some(0)]),
            Series::new("i32".into(), &[Some(-1i32), Some(0)]),
            Series::new("i64".into(), &[Some(-1i64), Some(0)]),
        ] {
            let dtype = s.dtype().clone();
            let enc = encode_series(&s).unwrap();
            assert!(!enc.is_null[0], "-1 must not be treated as null ({:?})", dtype);
            assert_eq!(enc.values[0], u64::MAX, "-1 sign-extends to u64::MAX ({:?})", dtype);
            assert_ne!(enc.values[0], enc.values[1]);
        }
    }

    #[test]
    fn test_u64_max_is_not_null() {
        let s = Series::new("test".into(), &[Some(u64::MAX), None, Some(7u64)]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], u64::MAX);
        assert!(!enc.is_null[0], "u64::MAX is a legal value, not null");
        assert!(enc.is_null[1]);
    }

    #[test]
    fn test_string_null_in_mask() {
        let s = Series::new("test".into(), &[Some("a"), None, Some("b")]);
        let enc = encode_series(&s).unwrap();
        assert!(enc.is_null[1]);
        assert!(!enc.is_null[0]);
        assert_ne!(enc.values[0], enc.values[2]);
    }

    #[test]
    fn test_float_value_and_null() {
        let s = Series::new("test".into(), &[Some(1.5f64), None]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], 1.5f64.to_bits());
        assert!(!enc.is_null[0]);
        assert!(enc.is_null[1]);
    }

    #[test]
    fn test_float_zero_canonicalised() {
        // +0.0 and -0.0 compare equal, so they must share one key.
        let s = Series::new("test".into(), &[Some(0.0f64), Some(-0.0f64)]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], enc.values[1], "+0.0 and -0.0 must share a key");
        assert_eq!(enc.values[0], 0);
    }

    #[test]
    fn test_nan_canonicalised() {
        // Distinct NaN bit patterns must collapse to a single key.
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0001);
        assert!(nan2.is_nan());
        let s = Series::new("test".into(), &[Some(f64::NAN), Some(nan2)]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], enc.values[1], "all NaNs must share a key");
    }

    // ── Pairwise ────────────────────────────────────────────────────────────

    #[test]
    fn test_pairwise_uniform() {
        // 4 unique pairs, each once → H = log2(4) = 2.0
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();
        assert_eq!(result.len(), 1);

        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 2.0).abs() < 1e-10, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_pairwise_deterministic() {
        let s1 = Series::new("a".into(), &[1i32; 100]);
        let s2 = Series::new("b".into(), &[1i32; 100]);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();

        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!(h.abs() < 1e-10, "Expected 0.0, got {}", h);
    }

    #[test]
    fn test_pairwise_3_columns() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let result = pairwise_joint_entropy_impl(&[s1, s2, s3], no_pairs()).unwrap();
        assert!(matches!(result.dtype(), DataType::Struct(_)));
        assert_eq!(result.len(), 3); // 3-choose-2 = 3 pairs
    }

    #[test]
    fn test_pairwise_single_column() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let result = pairwise_joint_entropy_impl(&[s1], no_pairs()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pairwise_empty_inputs() {
        let result = pairwise_joint_entropy_impl(&[], no_pairs());
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_length_mismatch() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs());
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_specific_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "c".to_string()]]),
        };
        let result = pairwise_joint_entropy_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_pairwise_specific_pairs_multiple() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["a".to_string(), "c".to_string()],
            ]),
        };
        let result = pairwise_joint_entropy_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_pairwise_invalid_column_name() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "nonexistent".to_string()]]),
        };
        let result = pairwise_joint_entropy_impl(&[s1, s2], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_invalid_pair_length() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string()]]),
        };
        let result = pairwise_joint_entropy_impl(&[s1, s2], kwargs);
        assert!(result.is_err());
    }

    // ── Threeway ────────────────────────────────────────────────────────────

    #[test]
    fn test_threeway_basic() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let result = threeway_joint_entropy_impl(&[s1, s2, s3], no_triplets()).unwrap();
        assert!(matches!(result.dtype(), DataType::Struct(_)));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_threeway_four_columns() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let s4 = Series::new("d".into(), &[1i32, 1, 2, 1]);

        let result = threeway_joint_entropy_impl(&[s1, s2, s3, s4], no_triplets()).unwrap();
        assert_eq!(result.len(), 4); // 4-choose-3 = 4
    }

    #[test]
    fn test_threeway_insufficient_columns() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2, 3]);
        let result = threeway_joint_entropy_impl(&[s1, s2], no_triplets());
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_length_mismatch() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let s3 = Series::new("c".into(), &[1i32, 2, 3]);
        let result = threeway_joint_entropy_impl(&[s1, s2, s3], no_triplets());
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_specific_triplets() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let s4 = Series::new("d".into(), &[1i32, 1, 2, 1]);

        let kwargs = ThreewayKwargs {
            triplets: Some(vec![vec![
                "a".to_string(),
                "b".to_string(),
                "d".to_string(),
            ]]),
        };
        let result = threeway_joint_entropy_impl(&[s1, s2, s3, s4], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_threeway_specific_triplets_multiple() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let s4 = Series::new("d".into(), &[1i32, 1, 2, 1]);

        let kwargs = ThreewayKwargs {
            triplets: Some(vec![
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec!["b".to_string(), "c".to_string(), "d".to_string()],
            ]),
        };
        let result = threeway_joint_entropy_impl(&[s1, s2, s3, s4], kwargs).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_threeway_invalid_triplet_column() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = ThreewayKwargs {
            triplets: Some(vec![vec![
                "a".to_string(),
                "b".to_string(),
                "x".to_string(),
            ]]),
        };
        let result = threeway_joint_entropy_impl(&[s1, s2, s3], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_invalid_triplet_length() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = ThreewayKwargs {
            triplets: Some(vec![vec!["a".to_string(), "b".to_string()]]),
        };
        let result = threeway_joint_entropy_impl(&[s1, s2, s3], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_all_triplets_generated() {
        // 40 columns → C(40,3) = 9880 triplets, no cap.
        let series: Vec<Series> = (0..40)
            .map(|i| Series::new(format!("col_{}", i).into(), &[1i32, 2, 3, 4]))
            .collect();
        let result = threeway_joint_entropy_impl(&series, no_triplets()).unwrap();
        assert_eq!(result.len(), 9880);
    }

    // ── Marginal (single-column) ───────────────────────────────────────────

    #[test]
    fn test_marginal_uniform() {
        // 4 distinct values, each once → H = log2(4) = 2.0
        let s = Series::new("a".into(), &[0i32, 1, 2, 3]);
        let result = marginal_entropy_impl(&[s]).unwrap();
        assert_eq!(result.len(), 1);

        let df = result.into_frame().unnest(["marginal_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 2.0).abs() < 1e-10, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_marginal_deterministic() {
        let s = Series::new("a".into(), &[7i32; 100]);
        let result = marginal_entropy_impl(&[s]).unwrap();

        let df = result.into_frame().unnest(["marginal_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!(h.abs() < 1e-10, "Expected 0.0, got {}", h);
    }

    #[test]
    fn test_marginal_multiple_columns() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3, 4]);
        let s2 = Series::new("b".into(), &[1i32, 1, 1, 1]);
        let s3 = Series::new("c".into(), &[1i32, 1, 2, 2]);
        let result = marginal_entropy_impl(&[s1, s2, s3]).unwrap();
        assert!(matches!(result.dtype(), DataType::Struct(_)));
        assert_eq!(result.len(), 3);

        let df = result.into_frame().unnest(["marginal_entropy"]).unwrap();
        let names: Vec<&str> = df
            .column("col_name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_marginal_null_is_distinct_category() {
        // [1, 1, null, null, 2]: 3 categories (1, null, 2) with counts (2,2,1)
        // over r=5 → H = -(2/5*log2(2/5)*2 + 1/5*log2(1/5)).
        let s = Series::new("a".into(), &[Some(1i32), Some(1), None, None, Some(2)]);
        let result = marginal_entropy_impl(&[s]).unwrap();
        let df = result.into_frame().unnest(["marginal_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();

        let p1: f64 = 2.0 / 5.0;
        let p2: f64 = 1.0 / 5.0;
        let expected = -(2.0 * p1 * p1.log2() + p2 * p2.log2());
        assert!((h - expected).abs() < 1e-10, "Expected {}, got {}", expected, h);
    }

    #[test]
    fn test_marginal_empty_inputs() {
        let result = marginal_entropy_impl(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_marginal_length_mismatch() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let result = marginal_entropy_impl(&[s1, s2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_marginal_boolean_column() {
        let s = Series::new("a".into(), &[true, false, true, false, true]);
        let result = marginal_entropy_impl(&[s]).unwrap();
        let df = result.into_frame().unnest(["marginal_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!(h >= 0.0 && h <= 1.0);
    }

    #[test]
    fn test_categorical_to_u64() {
        use polars::datatypes::Categories;
        let cats = Categories::global();
        let s = Series::new("cat".into(), &["x", "y", "x", "z"])
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], enc.values[2]); // "x" == "x"
        assert_ne!(enc.values[0], enc.values[1]); // "x" != "y"
    }

    #[test]
    fn test_categorical_null() {
        use polars::datatypes::Categories;
        let cats = Categories::global();
        let s = Series::new("cat".into(), &[Some("a"), None, Some("b")])
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        let enc = encode_series(&s).unwrap();
        assert!(enc.is_null[1]);
    }

    #[test]
    fn test_boolean_to_u64() {
        let s = Series::new("test".into(), &[Some(true), Some(false), None]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], 1u64);
        assert_eq!(enc.values[1], 0u64);
        assert!(!enc.is_null[0]);
        assert!(!enc.is_null[1]);
        assert!(enc.is_null[2]);
        assert_ne!(enc.values[0], enc.values[1]);
    }

    #[test]
    fn test_time_to_u64() {
        let s = Series::new("test".into(), &[Some(1_000_000i64), Some(2_000_000i64), None])
            .cast(&DataType::Time)
            .unwrap();
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], 1_000_000u64);
        assert_eq!(enc.values[1], 2_000_000u64);
        assert!(enc.is_null[2]);
        assert_ne!(enc.values[0], enc.values[1]);
    }

    #[test]
    fn test_decimal_to_u64() {
        let s = Series::new("test".into(), &[1i32, 2, 1])
            .cast(&DataType::Decimal(Some(10), Some(0)))
            .unwrap();
        let enc = encode_series(&s).unwrap();
        assert_ne!(enc.values[0], enc.values[1]); // 1 ≠ 2
        assert_eq!(enc.values[0], enc.values[2]); // 1 == 1 → same hash
    }

    #[test]
    fn test_list_same_contents_same_hash() {
        // Rows with identical element sequences must produce the same u64.
        let s = Series::from_any_values(
            "test".into(),
            &[
                AnyValue::List(Series::new("".into(), &[1i32, 2i32])),
                AnyValue::List(Series::new("".into(), &[1i32, 2i32])),
                AnyValue::List(Series::new("".into(), &[3i32])),
                AnyValue::Null,
            ],
            false,
        )
        .unwrap();
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], enc.values[1]); // [1,2] == [1,2]
        assert_ne!(enc.values[0], enc.values[2]); // [1,2] != [3]
        assert!(enc.is_null[3]);
    }

    #[test]
    fn test_list_order_matters() {
        // [1,2] and [2,1] are different lists and must produce different hashes.
        let s = Series::from_any_values(
            "test".into(),
            &[
                AnyValue::List(Series::new("".into(), &[1i32, 2i32])),
                AnyValue::List(Series::new("".into(), &[2i32, 1i32])),
            ],
            false,
        )
        .unwrap();
        let enc = encode_series(&s).unwrap();
        assert_ne!(enc.values[0], enc.values[1]);
    }

    // ── Dense re-encoding paths ────────────────────────────────────────────

    #[test]
    fn test_densify_nulls_get_own_id() {
        // [1, null, 1, 2] → 3 distinct categories (null is one of them).
        let s = Series::new("test".into(), &[Some(1i32), None, Some(1), Some(2)]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.card, 3);
        assert_eq!(dense.ids[0], dense.ids[2]); // 1 == 1
        assert_ne!(dense.ids[0], dense.ids[1]); // 1 != null
        assert_ne!(dense.ids[1], dense.ids[3]); // null != 2
        assert_eq!(dense.null_id, Some(dense.ids[1])); // null id recorded
    }

    #[test]
    fn test_densify_no_nulls_has_no_null_id() {
        let s = Series::new("test".into(), &[1i32, 2, 3]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.null_id, None);
    }

    #[test]
    fn test_pairwise_null_category_flat_path() {
        // Flat path (tiny joint space). Joint keys (0,1),(null,1),(0,2),(null,2)
        // are 4 distinct categories, each once → H = 2.0. Guards the null-as-
        // category policy through the dense encoding.
        let s1 = Series::new("a".into(), &[Some(0i32), None, Some(0), None]);
        let s2 = Series::new("b".into(), &[1i32, 1, 2, 2]);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 2.0).abs() < 1e-10, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_pairwise_high_cardinality_hash_path() {
        // 2000 distinct values per column → joint space 4M > FLAT_MAX,
        // exercising the u64 hash path. Rows are unique pairs → H = log2(2000).
        let v: Vec<i32> = (0..2000).collect();
        let s1 = Series::new("a".into(), &v);
        let s2 = Series::new("b".into(), &v);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        let expected = 2000f64.log2();
        assert!((h - expected).abs() < 1e-10, "Expected {}, got {}", expected, h);
    }

    #[test]
    fn test_threeway_high_cardinality_hash_path() {
        // 200 distinct values per column → joint space 8M > FLAT_MAX,
        // exercising the u64 hash path. Rows are unique triplets → H = log2(200).
        let v: Vec<i32> = (0..200).collect();
        let s1 = Series::new("a".into(), &v);
        let s2 = Series::new("b".into(), &v);
        let s3 = Series::new("c".into(), &v);
        let result = threeway_joint_entropy_impl(&[s1, s2, s3], no_triplets()).unwrap();
        let df = result.into_frame().unnest(["threeway_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        let expected = 200f64.log2();
        assert!((h - expected).abs() < 1e-10, "Expected {}, got {}", expected, h);
    }

    #[test]
    fn test_flat_and_hash_paths_agree() {
        // Same data pushed through both paths must give identical entropy.
        // 1500 rows, 40 distinct values per column: joint space 1600 → flat;
        // forcing the hash path via entropy_flat vs FREQ is implicit — instead
        // compare against the analytic value from a reference frequency count.
        let v1: Vec<i32> = (0..1500).map(|i| i % 40).collect();
        let v2: Vec<i32> = (0..1500).map(|i| (i / 3) % 40).collect();
        let s1 = Series::new("a".into(), &v1);
        let s2 = Series::new("b".into(), &v2);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();

        // Reference: brute-force count.
        let mut counts: HashMap<(i32, i32), u64> = HashMap::new();
        for i in 0..1500 {
            *counts.entry((v1[i], v2[i])).or_insert(0) += 1;
        }
        let r_f = 1500.0f64;
        let h_ref: f64 = counts
            .values()
            .map(|&c| {
                let p = c as f64 / r_f;
                -p * p.log2()
            })
            .sum();
        assert!((h - h_ref).abs() < 1e-10, "Expected {}, got {}", h_ref, h);
    }

    #[test]
    fn test_pairwise_boolean_column() {
        let s1 = Series::new("a".into(), &[true, false, true, false]);
        let s2 = Series::new("b".into(), &[true, true, false, false]);
        let result = pairwise_joint_entropy_impl(&[s1, s2], no_pairs()).unwrap();
        assert_eq!(result.len(), 1);
        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!(h >= 0.0 && h <= 2.0);
    }
}
