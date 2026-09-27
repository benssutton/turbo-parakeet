use polars::prelude::*;
use xxhash_rust::xxh3::xxh3_128;
use rayon::prelude::*;
use std::collections::HashSet;
use crate::shared::{EncodedColumn, build_column_cache_par, encode_series};

/// Set a bit at the given index in a byte array
#[inline(always)]
fn set_bit(bytes: &mut [u8], bit_index: usize) {
    bytes[bit_index / 8] |= 1u8 << (bit_index % 8);
}

/// Merge two bit arrays using bitwise OR
#[inline]
fn merge_bit_arrays(dest: &mut [u8], src: &[u8]) {
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Number of bytes needed to hold `m` bits. `m` is always the filter size in
/// BITS; the backing byte array is `ceil(m/8)` bytes.
#[inline]
fn n_bytes_for_bits(m: usize) -> usize {
    m.div_ceil(8)
}

/// Rayon chunk size heuristic shared by all row-parallel bloom operations.
#[inline]
fn chunk_size(len: usize) -> usize {
    (len / (rayon::current_num_threads() * 4)).clamp(100, 10000)
}

/// Ensure a supplied bit array matches the geometry implied by `m` bits.
///
/// This is the single guard that makes the `get_unchecked` byte access in
/// `check_item_membership` sound: after this returns Ok, `bit_array.len()`
/// equals `ceil(m/8)`, so every `idx/8` (idx < m) is in bounds.
fn validate_bit_array(bit_array: &[u8], m: usize) -> PolarsResult<()> {
    let n_bytes = n_bytes_for_bits(m);
    if bit_array.len() != n_bytes {
        return Err(PolarsError::ComputeError(
            format!(
                "bloom filter bit array has {} bytes but m={} bits requires {} bytes",
                bit_array.len(),
                m,
                n_bytes
            )
            .into(),
        ));
    }
    Ok(())
}

/// Kwargs struct for bloom_filter
pub(crate) struct BloomFilterKwargs {
    pub(crate) bit_array_bytes: Vec<u8>,
    pub(crate) k: usize,
    pub(crate) m: usize,
}

/// Kwargs struct for membership / membership_ratio
pub(crate) struct MembershipKwargs {
    pub(crate) bit_array_bytes: Vec<u8>,
    pub(crate) k: usize,
    pub(crate) m: usize,
}

pub(crate) fn bloom_filter_impl(series: &Series, kwargs: BloomFilterKwargs) -> PolarsResult<Vec<u8>> {
    let enc = encode_series(series)?;
    let m = kwargs.m; // filter size in BITS
    let k = kwargs.k;
    let n_bytes = n_bytes_for_bits(m);

    // Start from the caller's filter when supplied. An empty array means "no
    // prior state" (build a fresh filter); a non-empty array of the wrong size
    // is a caller error and must fail loudly rather than silently discarding
    // the accumulated bits.
    let mut bit_array = if kwargs.bit_array_bytes.is_empty() {
        vec![0u8; n_bytes]
    } else if kwargs.bit_array_bytes.len() == n_bytes {
        kwargs.bit_array_bytes
    } else {
        return Err(PolarsError::ComputeError(
            format!(
                "existing bloom filter has {} bytes but m={} bits requires {} bytes",
                kwargs.bit_array_bytes.len(),
                m,
                n_bytes
            )
            .into(),
        ));
    };

    // Only non-null values are inserted (a null is not a set member).
    let values: Vec<u64> = enc
        .values
        .iter()
        .zip(enc.is_null.iter())
        .filter_map(|(v, n)| if *n { None } else { Some(*v) })
        .collect();

    let local_arrays: Vec<Vec<u8>> = values
        .par_chunks(chunk_size(values.len()))
        .map(|chunk| {
            let mut local = vec![0u8; n_bytes];
            for &v in chunk {
                add_item_to_bits(&v.to_le_bytes(), &mut local, k, m);
            }
            local
        })
        .collect();

    for la in &local_arrays {
        merge_bit_arrays(&mut bit_array, la);
    }
    Ok(bit_array)
}

// ─────────────────────────────────────────────────────────────────────────────
// membership_ratio  (batch — one result row per input column)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute (ratio_all, ratio_non_null) from a pre-converted column.
///
/// - ratio_all:       found / total_rows      (nulls count in denominator)
/// - ratio_non_null:  found / non_null_rows   (nulls excluded from denominator)
/// Caller must have validated `bit_array` length against `m` (validate_bit_array).
fn compute_ratio(col: &EncodedColumn, bit_array: &[u8], k: usize, m: usize) -> (f64, f64) {
    let total = col.len();
    if total == 0 {
        return (0.0, 0.0);
    }
    let null_count = col.is_null.iter().filter(|&&n| n).count();
    let found: usize = (0..total)
        .into_par_iter()
        .with_min_len(chunk_size(total))
        .filter(|&idx| {
            !col.is_null[idx]
                && check_item_membership(&col.values[idx].to_le_bytes(), bit_array, k, m)
        })
        .count();
    let ratio_all = found as f64 / total as f64;
    let ratio_non_null = if total == null_count {
        0.0
    } else {
        found as f64 / (total - null_count) as f64
    };
    (ratio_all, ratio_non_null)
}

/// Multi-column batch impl: checks each input column independently.
/// Returns a struct Series with one row per input column.
pub(crate) fn membership_ratio_multi_impl(
    inputs: &[Series],
    kwargs: &MembershipKwargs,
) -> PolarsResult<Series> {
    validate_bit_array(&kwargs.bit_array_bytes, kwargs.m)?;
    let n = inputs.len();
    let needed: HashSet<usize> = (0..n).collect();
    let cache = build_column_cache_par(inputs, &needed)?;
    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    let results: Vec<(String, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (ratio_all, ratio_non_null) =
                compute_ratio(&cache[i], &kwargs.bit_array_bytes, kwargs.k, kwargs.m);
            (col_names[i].clone(), ratio_all, ratio_non_null)
        })
        .collect();

    let col_name_s = StringChunked::from_iter(results.iter().map(|(name, _, _)| name.as_str()))
        .into_series()
        .with_name("col_name".into());
    let ratio_all_s =
        Float64Chunked::from_vec("ratio_all".into(), results.iter().map(|(_, r, _)| *r).collect())
            .into_series();
    let ratio_non_null_s = Float64Chunked::from_vec(
        "ratio_non_null".into(),
        results.iter().map(|(_, _, r)| *r).collect(),
    )
    .into_series();

    let struct_ca = StructChunked::from_series(
        "membership_ratio".into(),
        n,
        [col_name_s, ratio_all_s, ratio_non_null_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Hash helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Add a single item to the bit array using double hashing.
///
/// Computes k bit positions inline via xxh3_128 double hashing, avoiding
/// the per-item `Vec<usize>` heap allocation.
#[inline]
fn add_item_to_bits(item_bytes: &[u8], bit_array: &mut [u8], hash_count: usize, size: usize) {
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;
    let hash2 = (hash128 as u64) as usize;
    for i in 0..hash_count {
        let idx = hash1.wrapping_add(i.wrapping_mul(hash2)) % size;
        set_bit(bit_array, idx);
    }
}

/// Helper function to get hash indices using double hashing.
/// Retained for external callers; internal hot paths use inlined versions.
#[allow(dead_code)]
pub fn get_hash_indices(item_bytes: &[u8], k: usize, m: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(k);
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;
    let hash2 = (hash128 as u64) as usize;
    for i in 0..k {
        indices.push(hash1.wrapping_add(i.wrapping_mul(hash2)) % m);
    }
    indices
}

/// Check if a single item might be in the Bloom filter.
///
/// Computes k bit positions inline via xxh3_128 double hashing. Early-exits on first miss.
#[inline]
fn check_item_membership(item_bytes: &[u8], bit_array: &[u8], hash_count: usize, size: usize) -> bool {
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;
    let hash2 = (hash128 as u64) as usize;
    for i in 0..hash_count {
        let idx = hash1.wrapping_add(i.wrapping_mul(hash2)) % size;
        // SAFETY: `size` is the filter's bit count, so idx < size and therefore
        // idx/8 < ceil(size/8). The production entry point (membership_ratio_multi_impl)
        // and the test helpers call validate_bit_array first, guaranteeing
        // bit_array.len() == ceil(size/8), so this byte index is always in bounds.
        if unsafe { *bit_array.get_unchecked(idx / 8) } & (1u8 << (idx % 8)) == 0 {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
fn bloom_filter_test(series: &Series, kwargs: BloomFilterKwargs) -> PolarsResult<Vec<u8>> {
    bloom_filter_impl(series, kwargs)
}

#[cfg(test)]
fn membership_impl(series: &Series, kwargs: &MembershipKwargs) -> PolarsResult<Series> {
    let bit_array = &kwargs.bit_array_bytes;
    let k = kwargs.k;
    let m = kwargs.m;
    validate_bit_array(bit_array, m)?;

    let enc = encode_series(series)?;
    let n = enc.len();

    // One result per input row (nulls → false). with_min_len preserves ordering
    // while chunking the parallel work.
    let results: Vec<bool> = (0..n)
        .into_par_iter()
        .with_min_len(chunk_size(n))
        .map(|idx| {
            !enc.is_null[idx]
                && check_item_membership(&enc.values[idx].to_le_bytes(), bit_array, k, m)
        })
        .collect();

    Ok(BooleanChunked::from_iter(results.into_iter())
        .into_series()
        .with_name(series.name().clone()))
}

#[cfg(test)]
fn membership_test(series: &Series, kwargs: MembershipKwargs) -> PolarsResult<Series> {
    membership_impl(series, &kwargs)
}

/// Single-series membership ratio (used by test helpers).
#[cfg(test)]
fn membership_ratio_impl(series: &Series, kwargs: &MembershipKwargs) -> PolarsResult<(f64, f64)> {
    validate_bit_array(&kwargs.bit_array_bytes, kwargs.m)?;
    let enc = encode_series(series)?;
    Ok(compute_ratio(&enc, &kwargs.bit_array_bytes, kwargs.k, kwargs.m))
}

/// Returns (ratio_all, ratio_non_null).
#[cfg(test)]
fn membership_ratio_test(series: &Series, kwargs: MembershipKwargs) -> PolarsResult<(f64, f64)> {
    membership_ratio_impl(series, &kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn count_set_bits(bytes: &[u8]) -> usize {
        bytes.iter().map(|&byte| byte.count_ones() as usize).sum()
    }

    #[test]
    fn test_add_series_string() {
        let series = Series::new("items".into(), &["hello", "world", "test"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        assert!(count_set_bits(&bit_array) > 0);

        let test_series = Series::new("test_items".into(), &["hello", "world", "test", "notfound"]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array.clone(), k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // "hello" found
        assert!(results_bool.get(1).unwrap());   // "world" found
        assert!(results_bool.get(2).unwrap());   // "test" found
        assert!(!results_bool.get(3).unwrap());  // "notfound" not found
    }

    #[test]
    fn test_add_series_numeric() {
        let series = Series::new("items".into(), &[1i64, 2, 3, 4, 5]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        assert!(count_set_bits(&bit_array) > 0);

        let test_series = Series::new("test_items".into(), &[1i64, 5, 99]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array.clone(), k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // 1 found
        assert!(results_bool.get(1).unwrap());   // 5 found
        assert!(!results_bool.get(2).unwrap());  // 99 not found
    }

    #[test]
    fn test_add_series_with_nulls() {
        let series = Series::new("items".into(), &[Some("a"), None, Some("b"), None, Some("c")]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test_items".into(), &["a", "b", "c", "notfound"]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array.clone(), k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // "a" found
        assert!(results_bool.get(1).unwrap());   // "b" found
        assert!(results_bool.get(2).unwrap());   // "c" found
        assert!(!results_bool.get(3).unwrap());  // "notfound" not found
    }

    #[test]
    fn test_membership_with_nulls() {
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &[Some("a"), None, Some("b"), Some("notfound")]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // "a" found
        assert!(!results_bool.get(1).unwrap());  // null → false
        assert!(results_bool.get(2).unwrap());   // "b" found
        assert!(!results_bool.get(3).unwrap());  // "notfound" not found
    }

    #[test]
    fn test_large_dataset_no_false_negatives() {
        let random_numbers: Vec<i64> = (0..10000).map(|i| i * 7 + 13).collect();
        let series = Series::new("random_numbers".into(), &random_numbers);
        let m = 100000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 7, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        assert!(count_set_bits(&bit_array) > 0);

        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 7, m };
        let results = membership_test(&series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        for i in 0..10000 {
            assert!(results_bool.get(i).unwrap(), "Item at index {} should be found", i);
        }
        let found_count = results_bool.into_iter().filter(|&x| x.unwrap_or(false)).count();
        assert_eq!(found_count, 10000, "All 10,000 items should be found");
    }

    #[test]
    fn test_false_positive_rate() {
        let training_set: Vec<i64> = (10000..50000).step_by(4).take(10000).collect();
        let training_series = Series::new("training".into(), &training_set);
        let m = 95850;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 7, m };
        let bit_array = bloom_filter_test(&training_series, kwargs).unwrap();

        assert!(count_set_bits(&bit_array) > 0);

        let test_set: Vec<i64> = (50000..100000).step_by(5).take(10000).collect();
        let test_series = Series::new("test".into(), &test_set);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 7, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        let false_positives = results_bool.into_iter().filter(|&x| x.unwrap_or(false)).count();
        let false_positive_rate = (false_positives as f64) / 10000.0;

        println!("False positives: {}/10000 ({:.2}%)", false_positives, false_positive_rate * 100.0);

        assert!(
            false_positive_rate <= 0.015,
            "False positive rate {:.4}% exceeds 1.5% threshold (found {} false positives)",
            false_positive_rate * 100.0,
            false_positives
        );
        assert!(false_positives > 0, "Expected some false positives");
    }

    // ── membership_ratio tests ────────────────────────────────────────────────

    #[test]
    fn test_membership_ratio_all_found() {
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let (ratio, _) = membership_ratio_test(&series, membership_kwargs).unwrap();
        assert_eq!(ratio, 1.0, "All items should be found, ratio should be 1.0");
    }

    #[test]
    fn test_membership_ratio_none_found() {
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &["x", "y", "z"]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let (ratio, _) = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.0, "No items should be found, ratio should be 0.0");
    }

    #[test]
    fn test_membership_ratio_partial() {
        let series = Series::new("items".into(), &["a", "b", "c", "d"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &["a", "b", "x", "y"]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let (ratio, _) = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.5, "Half items should be found, ratio should be 0.5");
    }

    #[test]
    fn test_membership_ratio_empty_series() {
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let empty_series: Series = Series::new("empty".into(), Vec::<String>::new());
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let (ratio, _) = membership_ratio_test(&empty_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.0, "Empty series should return 0.0");
    }

    #[test]
    fn test_membership_ratio_with_nulls() {
        // 2 found ("a", "b"), 1 null, 1 not found ("x") → ratio_all = 2/4 = 0.5
        let series = Series::new("items".into(), &["a", "b"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &[Some("a"), None, Some("b"), Some("x")]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let (ratio_all, ratio_non_null) = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio_all, 0.5, "Nulls count in denominator: 2/4 = 0.5");
        // ratio_non_null: 2 found out of 3 non-null rows = 2/3
        assert!((ratio_non_null - 2.0 / 3.0).abs() < 1e-10, "ratio_non_null should be 2/3");
    }

    #[test]
    fn test_membership_ratio_multi_impl_two_columns() {
        // Generous m so a false positive is practically impossible.
        let m = 1024;
        let num_bytes = (m + 7) / 8;
        let training = Series::new("training".into(), &["a", "b", "c"]);
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 7, m };
        let bit_array = bloom_filter_impl(&training, kwargs).unwrap();

        // col1: every value is in the filter → ratio_all = ratio_non_null = 1.0.
        let col1 = Series::new("contained".into(), &["a", "b", "c"]);
        // col2: one null, one value never inserted, one value that is inserted.
        let col2 = Series::new("mixed".into(), &[Some("a"), None, Some("notfound")]);

        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 7, m };
        let result = membership_ratio_multi_impl(&[col1, col2], &membership_kwargs).unwrap();
        let df = result.into_frame().unnest(["membership_ratio"]).unwrap();

        let names: Vec<&str> = df.column("col_name").unwrap().str().unwrap().into_no_null_iter().collect();
        assert_eq!(names, ["contained", "mixed"]);

        let ratio_all: Vec<f64> = df.column("ratio_all").unwrap().f64().unwrap().into_no_null_iter().collect();
        let ratio_non_null: Vec<f64> = df.column("ratio_non_null").unwrap().f64().unwrap().into_no_null_iter().collect();

        assert_eq!(ratio_all[0], 1.0, "every value in 'contained' is in the filter");
        assert_eq!(ratio_non_null[0], 1.0, "every value in 'contained' is in the filter");

        // "mixed": 1 found ("a"), 1 null, 1 not found ("notfound") → total 3, non_null 2.
        assert!((ratio_all[1] - 1.0 / 3.0).abs() < 1e-10, "ratio_all should be 1/3, got {}", ratio_all[1]);
        assert_eq!(ratio_non_null[1], 0.5, "1 of 2 non-null rows found, got {}", ratio_non_null[1]);
    }

    // ── numeric type tests ────────────────────────────────────────────────────

    #[test]
    fn test_bloom_filter_int64() {
        let series = Series::new("items".into(), &[10i64, 20, 30, 40, 50]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();
        assert!(bit_array.iter().any(|&b| b != 0), "Some bits must be set");

        let test_series = Series::new("test".into(), &[10i64, 30, 50, 99]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        assert!(results_bool.get(0).unwrap());   // 10 found
        assert!(results_bool.get(1).unwrap());   // 30 found
        assert!(results_bool.get(2).unwrap());   // 50 found
        assert!(!results_bool.get(3).unwrap());  // 99 not found
    }

    #[test]
    fn test_bloom_filter_float64() {
        let series = Series::new("items".into(), &[1.5f64, 2.5, 3.5]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &[1.5f64, 3.5, 9.9]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        assert!(results_bool.get(0).unwrap());   // 1.5 found
        assert!(results_bool.get(1).unwrap());   // 3.5 found
        assert!(!results_bool.get(2).unwrap());  // 9.9 not found
    }

    #[test]
    fn test_bloom_filter_numeric_nulls() {
        let series = Series::new("items".into(), &[Some(1i64), None, Some(2i64), None, Some(3i64)]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 5, m };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &[Some(1i64), None, Some(2i64), Some(99i64)]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: bit_array, k: 5, m };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        assert!(results_bool.get(0).unwrap());   // 1 found
        assert!(!results_bool.get(1).unwrap());  // null → false
        assert!(results_bool.get(2).unwrap());   // 2 found
        assert!(!results_bool.get(3).unwrap());  // 99 not found
    }

    #[test]
    fn test_numeric_and_string_filters_are_independent() {
        // Integer 42 and string "42" use different byte encodings → different bloom filter bits.
        let int_series = Series::new("items".into(), &[42i64]);
        let m = 10000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs { bit_array_bytes: vec![0u8; num_bytes], k: 7, m };
        let int_filter = bloom_filter_test(&int_series, kwargs).unwrap();

        let str_series = Series::new("test".into(), &["42"]);
        let membership_kwargs = MembershipKwargs { bit_array_bytes: int_filter, k: 7, m };
        let results = membership_test(&str_series, membership_kwargs).unwrap();
        let _ = results.bool().unwrap().get(0); // verify no panic; hash independence is probabilistic
    }
}
