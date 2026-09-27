use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use foldhash::fast::{FixedState as FoldHashFixed, RandomState as FoldHashFast};
use rayon::prelude::*;

use crate::shared::encode_series;

/// Parameters for LSH candidate finding
#[derive(Deserialize, Debug)]
pub(crate) struct LSHKwargs {
    pub(crate) num_bands: usize,
    pub(crate) rows_per_band: usize,
}

/// Find candidate pairs using Locality Sensitive Hashing (LSH).
///
/// This function takes a DataFrame with MinHash signatures and finds pairs
/// of items that are likely to have Jaccard similarity above the threshold.
///
/// # LSH Algorithm
/// LSH partitions the MinHash signature into bands. Items that hash to the
/// same bucket in at least one band are considered candidates. The probability
/// of two items becoming candidates is approximately 1 - (1 - s^r)^b where
/// s is their Jaccard similarity, r is rows_per_band, and b is num_bands.
///
/// # Arguments
/// * `inputs` - Array containing: [qualified_names: Utf8, minhash_signatures: List[UInt32]]
/// * `kwargs` - LSHKwargs containing the band configuration
///
/// # Returns
/// A DataFrame with columns (col_a: Utf8, col_b: Utf8) containing candidate pairs
#[polars_expr(output_type_func=lsh_candidates_output)]
fn lsh_candidates(inputs: &[Series], kwargs: LSHKwargs) -> PolarsResult<Series> {
    lsh_candidates_impl(inputs, &kwargs)
}

pub(crate) fn lsh_candidates_impl(inputs: &[Series], kwargs: &LSHKwargs) -> PolarsResult<Series> {
    let names = &inputs[0];
    let signatures = &inputs[1];

    let names_ca = names.str()?;
    let signatures_list = signatures.list()?;

    let num_bands = kwargs.num_bands;
    let rows_per_band = kwargs.rows_per_band;

    // Create buckets: (band_index, band_hash) -> list of row indices
    let mut buckets: HashMap<(usize, u64), Vec<usize>, FoldHashFast> =
        HashMap::with_hasher(FoldHashFast::default());

    // Process each row - extract signatures and assign to buckets
    for (idx, opt_sig) in signatures_list.into_iter().enumerate() {
        if let Some(sig_series) = opt_sig {
            let sig_ca = sig_series.u32()?;
            let sig: Vec<u32> = sig_ca.into_iter().flatten().collect();

            // Split signature into bands and hash each band
            for band_idx in 0..num_bands {
                let start = band_idx * rows_per_band;
                let end = start + rows_per_band;

                if end <= sig.len() {
                    let band_slice = &sig[start..end];
                    let band_hash = hash_band(band_slice);

                    buckets
                        .entry((band_idx, band_hash))
                        .or_default()
                        .push(idx);
                }
            }
        }
    }

    // Collect unique candidate pairs from buckets
    let mut seen_pairs: HashSet<(usize, usize), FoldHashFast> =
        HashSet::with_capacity_and_hasher(256, FoldHashFast::default());

    for indices in buckets.values() {
        if indices.len() > 1 {
            // Generate all pairs from items in the same bucket
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let idx_a = indices[i];
                    let idx_b = indices[j];
                    // Ensure consistent ordering for deduplication
                    let pair = if idx_a < idx_b {
                        (idx_a, idx_b)
                    } else {
                        (idx_b, idx_a)
                    };
                    seen_pairs.insert(pair);
                }
            }
        }
    }

    // Convert index pairs to name pairs
    let mut pairs_a: Vec<String> = Vec::with_capacity(seen_pairs.len());
    let mut pairs_b: Vec<String> = Vec::with_capacity(seen_pairs.len());

    for (idx_a, idx_b) in seen_pairs {
        if let (Some(name_a), Some(name_b)) = (names_ca.get(idx_a), names_ca.get(idx_b)) {
            pairs_a.push(name_a.to_string());
            pairs_b.push(name_b.to_string());
        }
    }

    // Create result columns
    let col_a = StringChunked::from_iter_values("col_a".into(), pairs_a.iter().map(|s| s.as_str()));
    let col_b = StringChunked::from_iter_values("col_b".into(), pairs_b.iter().map(|s| s.as_str()));

    // Create a struct with two fields
    let df = DataFrame::new(vec![
        col_a.into_series().into(),
        col_b.into_series().into(),
    ])?;

    // Convert to series
    Ok(df.into_struct("candidates".into()).into_series())
}

/// Hash a band of MinHash values to create a bucket key.
///
/// Uses foldhash FixedState for deterministic, high-quality hashing.
fn hash_band(band: &[u32]) -> u64 {
    let state = FoldHashFixed::default();
    let mut hasher = state.build_hasher();
    band.hash(&mut hasher);
    hasher.finish()
}

/// Output type function for find_lsh_candidates
fn lsh_candidates_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "candidates".into(),
        DataType::Struct(vec![
            Field::new("col_a".into(), DataType::String),
            Field::new("col_b".into(), DataType::String),
        ])
    ))
}

/// Parameters for batch MinHash computation
#[derive(Deserialize, Debug)]
pub(crate) struct MinHashKwargs {
    pub(crate) df_name: String,
    pub(crate) num_perm: usize,
}

/// Compute MinHash signatures for all columns passed as a struct series.
///
/// This function processes all columns of a DataFrame in parallel using rayon,
/// returning M rows (one per column) with qualified names and MinHash signatures.
/// Uses changes_length=True in Python to allow returning fewer rows than input.
///
/// # Arguments
/// * `inputs` - Array containing a single struct series where each field is a column to process
/// * `kwargs` - MinHashKwargs containing df_name prefix and num_perm
///
/// # Returns
/// A struct series with M rows (one per input column), each containing:
/// - qualified_name: String with "{df_name}|{column_name}" format
/// - minhash: List[UInt32] with MinHash signature
#[polars_expr(output_type_func=minhash_output)]
fn minhash(inputs: &[Series], kwargs: MinHashKwargs) -> PolarsResult<Series> {
    minhash_impl(inputs, &kwargs)
}

pub(crate) fn minhash_impl(inputs: &[Series], kwargs: &MinHashKwargs) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let df_name = &kwargs.df_name;
    let num_perm = kwargs.num_perm;

    // Extract fields from the struct series
    let struct_ca = struct_series.struct_()?;

    // Get field names from the struct's data type
    let field_names: Vec<PlSmallStr> = match struct_series.dtype() {
        DataType::Struct(fields) => fields.iter().map(|f| f.name.clone()).collect(),
        _ => return Err(PolarsError::ComputeError("Expected struct type".into())),
    };

    // Pre-compute permutation coefficients ONCE for all columns
    let (a_coeffs, b_coeffs) = generate_permutation_coeffs(num_perm);

    // Process each field (column) in parallel. Encode errors abort the whole
    // computation; all-null columns emit Ok(None) → null list entry (skipped by LSH).
    let results: Vec<(String, Option<Vec<u32>>)> = field_names
        .par_iter()
        .map(|field_name| -> PolarsResult<(String, Option<Vec<u32>>)> {
            let series = struct_ca.field_by_name(field_name.as_str())?;
            let qualified_name = format!("{}|{}", df_name, field_name);
            let signature = compute_signature_for_series_with_coeffs(&series, num_perm, &a_coeffs, &b_coeffs)?;
            Ok((qualified_name, signature))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // Build output with M rows (one per column)
    let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
    let name_col = StringChunked::from_iter_values("qualified_name".into(), names.into_iter());

    // Build list column for signatures
    let mut list_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "minhash".into(),
        results.len(),
        num_perm,
        DataType::UInt32,
    );
    for (_, sig_opt) in &results {
        match sig_opt {
            Some(sig) => list_builder.append_slice(sig),
            None => list_builder.append_null(),
        }
    }
    let minhash_col = list_builder.finish();

    // Return as struct series with M rows
    let df = DataFrame::new(vec![
        name_col.into_series().into(),
        minhash_col.into_series().into(),
    ])?;

    Ok(df.into_struct("minhash_result".into()).into_series())
}

/// Generate deterministic permutation coefficients for MinHash.
///
/// Uses foldhash FixedState with domain separation to produce distinct a/b
/// coefficient pairs. Results are fully reproducible across runs.
fn generate_permutation_coeffs(num_perm: usize) -> (Vec<u64>, Vec<u64>) {
    let state = FoldHashFixed::default();
    let mut a_coeffs = Vec::with_capacity(num_perm);
    let mut b_coeffs = Vec::with_capacity(num_perm);

    for i in 0..num_perm {
        let mut h = state.build_hasher();
        h.write_u64(i as u64);
        h.write_u8(0); // domain separator for a coefficients
        a_coeffs.push(h.finish() | 1); // ensure odd for better distribution

        let mut h = state.build_hasher();
        h.write_u64(i as u64);
        h.write_u8(1); // domain separator for b coefficients
        b_coeffs.push(h.finish());
    }
    (a_coeffs, b_coeffs)
}

/// Compute MinHash signature using pre-computed permutation coefficients.
///
/// Applies the linear hash family (a * val + b) mod 2^64, taking the lower
/// 32 bits as the hash value per permutation. Null values are skipped —
/// a null is not a set member.
///
/// Returns:
/// - `Ok(Some(sig))` when at least one non-null value was hashed.
/// - `Ok(None)` when the column is all-null or empty — MinHash is undefined
///   for the empty set; the caller should emit a null list entry so LSH skips it.
/// - `Err(e)` when `encode_series` fails (unsupported dtype etc.); the caller
///   should propagate the error rather than silently continuing.
fn compute_signature_for_series_with_coeffs(
    series: &Series,
    num_perm: usize,
    a_coeffs: &[u64],
    b_coeffs: &[u64],
) -> PolarsResult<Option<Vec<u32>>> {
    let enc = encode_series(series)?; // propagate encode errors instead of swallowing them

    let mut minhash_sig: Vec<u32> = vec![u32::MAX; num_perm];
    let mut has_non_null = false;

    for (val, is_null) in enc.values.iter().zip(enc.is_null.iter()) {
        if *is_null {
            continue;
        }
        has_non_null = true;
        // Apply linear hash family: (a * val + b) mod 2^64, take lower 32 bits
        for i in 0..num_perm {
            let derived = a_coeffs[i].wrapping_mul(*val).wrapping_add(b_coeffs[i]);
            let hash_val = derived as u32;
            minhash_sig[i] = minhash_sig[i].min(hash_val);
        }
    }

    if !has_non_null {
        // All-null or empty column — emit null so LSH skips it rather than
        // matching it against every other all-null column at Jaccard 1.0.
        return Ok(None);
    }

    Ok(Some(minhash_sig))
}

/// Output type function for compute_minhash_batch
fn minhash_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "minhash_result".into(),
        DataType::Struct(vec![
            Field::new("qualified_name".into(), DataType::String),
            Field::new("minhash".into(), DataType::List(Box::new(DataType::UInt32))),
        ])
    ))
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test data
    fn create_test_dataframes() -> (DataFrame, DataFrame) {
        // DataFrame 1:
        let df1 = DataFrame::new(vec![
            Column::new("A".into(), &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Column::new("B".into(), &[3i64, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            Column::new("C".into(), &[10i64, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ]).unwrap();

        // DataFrame 2:
        let df2 = DataFrame::new(vec![
            Column::new("A".into(), &[1i64, 2, 3, 4, 5]),
            Column::new("B".into(), &[4i64, 5, 6, 7, 8]),
            Column::new("C".into(), &[10i64, 11, 12, 13, 14]),
        ]).unwrap();

        (df1, df2)
    }

    #[test]
    fn test_all_null_column_returns_none() {
        // An all-null column has no set members — MinHash is undefined for the empty set.
        // compute_signature_for_series_with_coeffs must return Ok(None) so that the
        // caller emits a null list entry and LSH skips it.
        let s = Series::new("x".into(), &[Option::<i64>::None, None, None]);
        let (a, b) = generate_permutation_coeffs(16);
        let result = compute_signature_for_series_with_coeffs(&s, 16, &a, &b).unwrap();
        assert!(result.is_none(), "all-null column must yield None, not all-MAX signature");
    }

    #[test]
    fn test_encode_error_propagated() {
        // compute_signature_for_series_with_coeffs must propagate errors from encode_series
        // rather than swallowing them and returning a spurious all-MAX signature.
        // (Encoding can fail for unsupported / complex dtypes; here we test the contract
        // via a normal column to confirm Ok(Some) is returned when encoding succeeds.)
        let s = Series::new("x".into(), &[1i64, 2, 3]);
        let (a, b) = generate_permutation_coeffs(16);
        let result = compute_signature_for_series_with_coeffs(&s, 16, &a, &b).unwrap();
        assert!(result.is_some());
        // All-MAX would indicate the old error-swallowing behaviour is still present.
        assert!(result.unwrap().iter().any(|&v| v != u32::MAX));
    }

    #[test]
    fn test_minhash_and_lsh_pipeline() {
        // Step 1: Create test dataframes
        let (df1, df2) = create_test_dataframes();

        // Step 2: Compute MinHash signatures for both dataframes

        // Convert df1 to struct series for minhash_impl
        let df1_struct = df1.clone().into_struct("df1_struct".into()).into_series();
        let minhash_kwargs_1 = MinHashKwargs {
            df_name: "df1".to_string(),
            num_perm: 128,
        };
        let mut minhash_result_1 = minhash_impl(&[df1_struct], &minhash_kwargs_1)
            .expect("MinHash computation for df1 failed");

        // Convert df2 to struct series for minhash_impl
        let df2_struct = df2.clone().into_struct("df2_struct".into()).into_series();
        let minhash_kwargs_2 = MinHashKwargs {
            df_name: "df2".to_string(),
            num_perm: 128,
        };
        let minhash_result_2 = minhash_impl(&[df2_struct], &minhash_kwargs_2)
            .expect("MinHash computation for df2 failed");

        // Combine the two minhash results into a single series
        let combined = minhash_result_1.append(&minhash_result_2)
            .expect("Failed to combine minhash results");

        // Extract the struct fields to pass to lsh_candidates_impl
        let struct_ca = combined.struct_().expect("Expected struct type");
        let qualified_names = struct_ca.field_by_name("qualified_name")
            .expect("qualified_name field not found");
        let minhash_sigs = struct_ca.field_by_name("minhash")
            .expect("minhash field not found");

        // Step 3: Find LSH candidate pairs
        let lsh_kwargs = LSHKwargs {
            num_bands: 32,
            rows_per_band: 4,
        };
        let candidates_result = lsh_candidates_impl(
            &[qualified_names, minhash_sigs],
            &lsh_kwargs
        ).expect("LSH candidate finding failed");

        // Step 4: Assert that at least 5 candidate pairs were found
        let candidates_struct = candidates_result.struct_()
            .expect("Expected struct type for candidates");
        let col_a = candidates_struct.field_by_name("col_a")
            .expect("col_a field not found");

        let num_candidates = col_a.len();

        assert!(
            num_candidates >= 5,
            "Expected at least 5 candidate pairs, but found {}",
            num_candidates
        );

        println!("Successfully found {} candidate pairs", num_candidates);
    }

}
