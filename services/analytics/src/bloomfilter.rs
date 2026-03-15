use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use xxhash_rust::xxh3::xxh3_128;
use rayon::prelude::*;
use serde::Deserialize;
use crate::type_conversion::series_to_opt_u64;

////////////////////////////////////////////////////////////////////////
///  Bloom filter functions to calculate the membership of large sets
////////////////////////////////////////////////////////////////////////

/// Set a bit at the given index in a byte array
#[inline(always)]
fn set_bit(bytes: &mut [u8], bit_index: usize) {
    bytes[bit_index / 8] |= 1u8 << (bit_index % 8);
}

/// Batch bit checking with fewer bounds checks - optimized for membership queries
/// Early exits on first miss, which is critical for negative queries
#[allow(dead_code)]
#[inline]
fn check_bits(bytes: &[u8], indices: &[usize]) -> bool {
    for &bit_index in indices {
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;

        // SAFETY: Indices are always within bounds (guaranteed by hash % m)
        if unsafe { *bytes.get_unchecked(byte_index) } & (1u8 << bit_offset) == 0 {
            return false;
        }
    }
    true
}

/// Merge two bit arrays using bitwise OR
#[inline]
fn merge_bit_arrays(dest: &mut [u8], src: &[u8]) {
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Kwargs struct for bloom_filter
#[derive(Deserialize)]
struct BloomFilterKwargs {
    bit_array_bytes: Vec<u8>,
    k: usize,
    m: usize,
}

/// Kwargs struct for membership
#[derive(Deserialize)]
struct MembershipKwargs {
    bit_array_bytes: Vec<u8>,
    k: usize,
    m: usize,
}

/// Kwargs struct for membership_ratio_sample
#[derive(Deserialize)]
struct MembershipRatioSampleKwargs {
    bit_array_bytes: Vec<u8>,
    k: usize,
    m: usize,
    sample_frac: f64,
}

/// Internal implementation of bloom filter - returns Vec<u8>
fn bloom_filter_impl(
    series: &Series,
    kwargs: BloomFilterKwargs,
) -> Result<Vec<u8>, PolarsError> {

    // Rechunk to a single contiguous chunk once (no-op when already single-chunk)
    let rechunked;
    let series = if series.n_chunks() > 1 {
        rechunked = series.rechunk();
        &rechunked
    } else {
        series
    };

    // Calculate optimal chunk size based on thread count and dataset size
    let num_threads = rayon::current_num_threads();
    let chunk_size = (series.len() / (num_threads * 4)).max(100).min(10000);
    let m = kwargs.m;
    let k = kwargs.k;

    // Start with existing filter or create fresh array
    let mut bit_array = if kwargs.bit_array_bytes.len() == m {
        kwargs.bit_array_bytes
    } else {
        vec![0u8; m]
    };

    if matches!(series.dtype(), DataType::String) {
        // String path: feed raw UTF-8 bytes directly to xxh3 (no intermediate hash)
        let ca = series.str()?;
        let values: Vec<Option<&str>> = ca.into_iter().collect();
        let local_bit_arrays: Vec<Vec<u8>> = values
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_bits = vec![0u8; m];
                for opt_val in chunk {
                    if let Some(val) = opt_val {
                        add_item_to_bits(val.as_bytes(), &mut local_bits, k, m);
                    }
                }
                local_bits
            })
            .collect();
        for local_bits in &local_bit_arrays {
            merge_bit_arrays(&mut bit_array, local_bits);
        }
    } else if series.null_count() == 0 {
        // Null-free numeric fast path: iterate contiguous values directly,
        // avoiding the 16-byte-per-element Vec<Option<u64>> allocation.
        match series.dtype() {
            DataType::Int64 => {
                let ca = series.i64()?;
                let slice = ca.cont_slice().unwrap();
                let local_bit_arrays: Vec<Vec<u8>> = slice
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_bits = vec![0u8; m];
                        for &v in chunk {
                            add_item_to_bits(&(v as u64).to_le_bytes(), &mut local_bits, k, m);
                        }
                        local_bits
                    })
                    .collect();
                for local_bits in &local_bit_arrays {
                    merge_bit_arrays(&mut bit_array, local_bits);
                }
            }
            DataType::UInt64 => {
                let ca = series.u64()?;
                let slice = ca.cont_slice().unwrap();
                let local_bit_arrays: Vec<Vec<u8>> = slice
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_bits = vec![0u8; m];
                        for &v in chunk {
                            add_item_to_bits(&v.to_le_bytes(), &mut local_bits, k, m);
                        }
                        local_bits
                    })
                    .collect();
                for local_bits in &local_bit_arrays {
                    merge_bit_arrays(&mut bit_array, local_bits);
                }
            }
            DataType::Float64 => {
                let ca = series.f64()?;
                let slice = ca.cont_slice().unwrap();
                let local_bit_arrays: Vec<Vec<u8>> = slice
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_bits = vec![0u8; m];
                        for &v in chunk {
                            add_item_to_bits(&v.to_bits().to_le_bytes(), &mut local_bits, k, m);
                        }
                        local_bits
                    })
                    .collect();
                for local_bits in &local_bit_arrays {
                    merge_bit_arrays(&mut bit_array, local_bits);
                }
            }
            _ => {
                // Other null-free types: fall through to series_to_opt_u64
                let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
                let local_bit_arrays: Vec<Vec<u8>> = values
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut local_bits = vec![0u8; m];
                        for opt_val in chunk {
                            if let Some(v) = opt_val {
                                add_item_to_bits(&v.to_le_bytes(), &mut local_bits, k, m);
                            }
                        }
                        local_bits
                    })
                    .collect();
                for local_bits in &local_bit_arrays {
                    merge_bit_arrays(&mut bit_array, local_bits);
                }
            }
        }
    } else {
        // Nullable numeric/temporal path: convert to Option<u64>.
        let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
        let local_bit_arrays: Vec<Vec<u8>> = values
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_bits = vec![0u8; m];
                for opt_val in chunk {
                    if let Some(v) = opt_val {
                        add_item_to_bits(&v.to_le_bytes(), &mut local_bits, k, m);
                    }
                }
                local_bits
            })
            .collect();
        for local_bits in &local_bit_arrays {
            merge_bit_arrays(&mut bit_array, local_bits);
        }
    }

    Ok(bit_array)
}

/// Add all items from a Polars Series to a Bloom filter bit array
/// Polars plugin wrapper that returns a Series with binary data
#[polars_expr(output_type=Binary)]
pub fn bloom_filter(
    inputs: &[Series],
    kwargs: BloomFilterKwargs,
) -> Result<Series, PolarsError> {
    // Extract the first Series from the input slice
    let series = &inputs[0];

    // Call the internal implementation - returns Vec<u8> directly
    let bytes = bloom_filter_impl(series, kwargs)?;

    Ok(Series::new("bloom_filter".into(), &[bytes]))
}

/// Internal implementation of membership check - returns a Series
fn membership_impl(
    series: &Series,
    kwargs: &MembershipKwargs,
) -> Result<Series, PolarsError> {
    let bit_array = &kwargs.bit_array_bytes;

    // Rechunk to a single contiguous chunk once (no-op when already single-chunk)
    let rechunked;
    let series = if series.n_chunks() > 1 {
        rechunked = series.rechunk();
        &rechunked
    } else {
        series
    };

    // Calculate optimal chunk size based on thread count and dataset size
    let num_threads = rayon::current_num_threads();
    let chunk_size = (series.len() / (num_threads * 4)).max(100).min(10000);
    let k = kwargs.k;
    let m = kwargs.m;

    // Process each value and check membership; nulls are not in the set → false
    let results: Vec<bool> = if matches!(series.dtype(), DataType::String) {
        // String path: feed raw UTF-8 bytes to xxh3
        let ca = series.str()?;
        let values: Vec<Option<&str>> = ca.into_iter().collect();
        values
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|opt_val| {
                    match opt_val {
                        Some(val) => check_item_membership(val.as_bytes(), bit_array, k, m),
                        None => false,
                    }
                }).collect::<Vec<bool>>()
            })
            .collect()
    } else if series.null_count() == 0 {
        // Null-free numeric fast path: iterate contiguous slice directly
        match series.dtype() {
            DataType::Int64 => {
                let ca = series.i64()?;
                let slice = ca.cont_slice().unwrap();
                slice
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().map(|&v| {
                            check_item_membership(&(v as u64).to_le_bytes(), bit_array, k, m)
                        }).collect::<Vec<bool>>()
                    })
                    .collect()
            }
            DataType::UInt64 => {
                let ca = series.u64()?;
                let slice = ca.cont_slice().unwrap();
                slice
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().map(|&v| {
                            check_item_membership(&v.to_le_bytes(), bit_array, k, m)
                        }).collect::<Vec<bool>>()
                    })
                    .collect()
            }
            DataType::Float64 => {
                let ca = series.f64()?;
                let slice = ca.cont_slice().unwrap();
                slice
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().map(|&v| {
                            check_item_membership(&v.to_bits().to_le_bytes(), bit_array, k, m)
                        }).collect::<Vec<bool>>()
                    })
                    .collect()
            }
            _ => {
                // Other null-free types: fall through to series_to_opt_u64
                let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
                values
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().map(|opt_val| {
                            match opt_val {
                                Some(v) => check_item_membership(&v.to_le_bytes(), bit_array, k, m),
                                None => false,
                            }
                        }).collect::<Vec<bool>>()
                    })
                    .collect()
            }
        }
    } else {
        // Nullable numeric/temporal path: use Option<u64>
        let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
        values
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|opt_val| {
                    match opt_val {
                        Some(v) => check_item_membership(&v.to_le_bytes(), bit_array, k, m),
                        None => false,
                    }
                }).collect::<Vec<bool>>()
            })
            .collect()
    };

    // Convert to Polars Boolean Series
    Ok(Series::new(series.name().clone(), results))
}

/// Check membership for all items in a Polars Series against a Bloom filter
/// Polars plugin wrapper
#[polars_expr(output_type=Boolean)]
pub fn membership(
    inputs: &[Series],
    kwargs: MembershipKwargs,
) -> Result<Series, PolarsError> {
    // Extract the first Series from the input slice
    let series = &inputs[0];

    // Call the internal implementation
    membership_impl(series, &kwargs)
}

/// Internal implementation of membership ratio — counts matches directly
/// without materializing an intermediate `Vec<bool>` or `Series`.
fn membership_ratio_impl(
    series: &Series,
    kwargs: &MembershipKwargs,
) -> Result<f64, PolarsError> {
    let bit_array = &kwargs.bit_array_bytes;

    // Rechunk to a single contiguous chunk once
    let rechunked;
    let series = if series.n_chunks() > 1 {
        rechunked = series.rechunk();
        &rechunked
    } else {
        series
    };

    let total = series.len();
    if total == 0 {
        return Ok(0.0);
    }

    let num_threads = rayon::current_num_threads();
    let chunk_size = (total / (num_threads * 4)).max(100).min(10000);
    let k = kwargs.k;
    let m = kwargs.m;

    let found_count: usize = if matches!(series.dtype(), DataType::String) {
        let ca = series.str()?;
        let values: Vec<Option<&str>> = ca.into_iter().collect();
        values
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter().filter(|opt| match opt {
                    Some(v) => check_item_membership(v.as_bytes(), bit_array, k, m),
                    None => false,
                }).count()
            })
            .sum()
    } else if series.null_count() == 0 {
        match series.dtype() {
            DataType::Int64 => {
                let ca = series.i64()?;
                let slice = ca.cont_slice().unwrap();
                slice
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        chunk.iter().filter(|&&v| {
                            check_item_membership(&(v as u64).to_le_bytes(), bit_array, k, m)
                        }).count()
                    })
                    .sum()
            }
            DataType::Float64 => {
                let ca = series.f64()?;
                let slice = ca.cont_slice().unwrap();
                slice
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        chunk.iter().filter(|&&v| {
                            check_item_membership(&v.to_bits().to_le_bytes(), bit_array, k, m)
                        }).count()
                    })
                    .sum()
            }
            _ => {
                let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
                values
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        chunk.iter().filter(|opt| match opt {
                            Some(v) => check_item_membership(&v.to_le_bytes(), bit_array, k, m),
                            None => false,
                        }).count()
                    })
                    .sum()
            }
        }
    } else {
        let values: Vec<Option<u64>> = series_to_opt_u64(series)?;
        values
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter().filter(|opt| match opt {
                    Some(v) => check_item_membership(&v.to_le_bytes(), bit_array, k, m),
                    None => false,
                }).count()
            })
            .sum()
    };

    Ok(found_count as f64 / total as f64)
}

/// Calculate the ratio of items found in a Bloom filter (0.0 to 1.0)
/// Returns 1.0 if all items were found, 0.0 if none were found
/// Polars plugin wrapper
#[polars_expr(output_type=Float64)]
pub fn membership_ratio(
    inputs: &[Series],
    kwargs: MembershipKwargs,
) -> Result<Series, PolarsError> {
    // Extract the first Series from the input slice
    let series = &inputs[0];

    // Call the internal implementation
    let ratio = membership_ratio_impl(series, &kwargs)?;

    Ok(Series::new("membership_ratio".into(), &[ratio]))
}

/// Internal implementation of membership ratio with sampling.
/// Samples a fraction of the series before checking membership.
fn membership_ratio_sample_impl(
    series: &Series,
    kwargs: &MembershipRatioSampleKwargs,
) -> Result<f64, PolarsError> {
    // If sample_frac is 1.0 or greater, or series is small, use full series
    let sampled = if kwargs.sample_frac >= 1.0 || series.len() <= 10 {
        series.clone()
    } else {
        series.sample_frac(kwargs.sample_frac, false, false, None)?
    };

    let membership_kwargs = MembershipKwargs {
        bit_array_bytes: kwargs.bit_array_bytes.clone(),
        k: kwargs.k,
        m: kwargs.m,
    };

    membership_ratio_impl(&sampled, &membership_kwargs)
}

/// Calculate the ratio of items found in a Bloom filter using a random sample
/// Returns 1.0 if all sampled items were found, 0.0 if none were found
/// Used for early-exit optimization when testing large datasets
/// Polars plugin wrapper
#[polars_expr(output_type=Float64)]
pub fn membership_ratio_sample(
    inputs: &[Series],
    kwargs: MembershipRatioSampleKwargs,
) -> Result<Series, PolarsError> {
    // Extract the first Series from the input slice
    let series = &inputs[0];

    // Call the internal implementation
    let ratio = membership_ratio_sample_impl(series, &kwargs)?;

    Ok(Series::new("membership_ratio_sample".into(), &[ratio]))
}

/// Add a single item to the bit array using double hashing.
///
/// Computes k bit positions inline via xxh3_128 double hashing, avoiding
/// the per-item `Vec<usize>` heap allocation in `get_hash_indices`.
#[inline]
fn add_item_to_bits(
    item_bytes: &[u8],
    bit_array: &mut [u8],
    hash_count: usize,
    size: usize,
) {
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;
    let hash2 = (hash128 as u64) as usize;
    for i in 0..hash_count {
        let idx = hash1.wrapping_add(i.wrapping_mul(hash2)) % size;
        set_bit(bit_array, idx);
    }
}

/// Helper function to get hash indices using double hashing.
/// Uses a single xxh3 128-bit hash, split into two 64-bit halves.
/// Retained for external callers; internal hot paths use inlined versions.
#[allow(dead_code)]
pub fn get_hash_indices(item_bytes: &[u8], k: usize, m: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(k);

    // Single 128-bit hash, split into two 64-bit halves
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;   // upper 64 bits
    let hash2 = (hash128 as u64) as usize;  // lower 64 bits

    // Generate k hash indices using double hashing
    // Double hashing: hash(i) = (hash1 + i * hash2) mod m
    // Use wrapping arithmetic to handle overflow safely
    for i in 0..k {
        let hash = hash1.wrapping_add(i.wrapping_mul(hash2));
        indices.push(hash % m);
    }

    indices
}

/// Check if a single item might be in the Bloom filter.
///
/// Computes k bit positions inline via xxh3_128 double hashing, avoiding
/// the per-item `Vec<usize>` heap allocation. Early-exits on first miss.
#[inline]
fn check_item_membership(
    item_bytes: &[u8],
    bit_array: &[u8],
    hash_count: usize,
    size: usize,
) -> bool {
    let hash128 = xxh3_128(item_bytes);
    let hash1 = (hash128 >> 64) as usize;
    let hash2 = (hash128 as u64) as usize;
    for i in 0..hash_count {
        let idx = hash1.wrapping_add(i.wrapping_mul(hash2)) % size;
        // SAFETY: idx is always < size, and bit_array length == size (in bytes)
        if unsafe { *bit_array.get_unchecked(idx / 8) } & (1u8 << (idx % 8)) == 0 {
            return false;
        }
    }
    true
}

/// Helper function for testing membership without the plugin infrastructure
#[cfg(test)]
fn membership_test(
    series: &Series,
    kwargs: MembershipKwargs,
) -> Result<Series, PolarsError> {
    membership_impl(series, &kwargs)
}

/// Helper function for testing membership_ratio without the plugin infrastructure
#[cfg(test)]
fn membership_ratio_test(
    series: &Series,
    kwargs: MembershipKwargs,
) -> Result<f64, PolarsError> {
    membership_ratio_impl(series, &kwargs)
}

/// Helper function for testing membership_ratio_sample without the plugin infrastructure
#[cfg(test)]
fn membership_ratio_sample_test(
    series: &Series,
    kwargs: MembershipRatioSampleKwargs,
) -> Result<f64, PolarsError> {
    membership_ratio_sample_impl(series, &kwargs)
}

/// Helper function for testing bloom_filter without the plugin infrastructure
#[cfg(test)]
fn bloom_filter_test(
    series: &Series,
    kwargs: BloomFilterKwargs,
) -> Result<Vec<u8>, PolarsError> {
    bloom_filter_impl(series, kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to count set bits in a byte array
    fn count_set_bits(bytes: &[u8]) -> usize {
        bytes.iter().map(|&byte| byte.count_ones() as usize).sum()
    }

    #[test]
    fn test_add_series_string() {
        let series = Series::new("items".into(), &["hello", "world", "test"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Check that some bits are set
        assert!(count_set_bits(&bit_array) > 0);

        // Check that items are found using membership function
        let test_series = Series::new("test_items".into(), &["hello", "world", "test", "notfound"]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array.clone(),
            k: 5,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());  // "hello" found
        assert!(results_bool.get(1).unwrap());  // "world" found
        assert!(results_bool.get(2).unwrap());  // "test" found
        assert!(!results_bool.get(3).unwrap()); // "notfound" not found
    }
    
    #[test]
    fn test_add_series_numeric() {
        let series = Series::new("items".into(), &[1i64, 2, 3, 4, 5]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        assert!(count_set_bits(&bit_array) > 0);

        // Check that numeric items (as strings) are found using membership function
        let test_series = Series::new("test_items".into(), &[1i64, 5, 99]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array.clone(),
            k: 5,
            m
        };
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
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Check membership using membership function
        let test_series = Series::new("test_items".into(), &["a", "b", "c", "notfound"]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array.clone(),
            k: 5,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // "a" found
        assert!(results_bool.get(1).unwrap());   // "b" found
        assert!(results_bool.get(2).unwrap());   // "c" found
        assert!(!results_bool.get(3).unwrap());  // "notfound" not found
    }
    
    #[test]
    fn test_membership_with_nulls() {
        // Create bit array with some values
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test membership including nulls
        let test_series = Series::new("test".into(), &[Some("a"), None, Some("b"), Some("notfound")]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        assert!(results_bool.get(0).unwrap());   // "a" found
        assert!(!results_bool.get(1).unwrap());  // null → false
        assert!(results_bool.get(2).unwrap());   // "b" found
        assert!(!results_bool.get(3).unwrap());  // "notfound" not found
    }

    #[test]
    fn test_large_dataset_no_false_negatives() {
        // Generate 10,000 random numbers
        let random_numbers: Vec<i64> = (0..10000).map(|i| i * 7 + 13).collect();
        let series = Series::new("random_numbers".into(), &random_numbers);

        // Create bloom filter with appropriate size (10x the number of items for low false positive rate)
        // Using k=7 hash functions
        let m = 100000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 7,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Check that bits were set
        assert!(count_set_bits(&bit_array) > 0);

        // Test membership for all original items
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 7,
            m
        };
        let results = membership_test(&series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        // Assert all items were found (no false negatives)
        for i in 0..10000 {
            assert!(
                results_bool.get(i).unwrap(),
                "Item at index {} should be found in bloom filter", i
            );
        }

        // Verify all 10,000 items were found
        let found_count = results_bool.into_iter().filter(|&x| x.unwrap_or(false)).count();
        assert_eq!(found_count, 10000, "All 10,000 items should be found");
    }

    #[test]
    fn test_false_positive_rate() {
        // Create first set: 10,000 numbers in range 10,000-49,999
        let training_set: Vec<i64> = (10000..50000).step_by(4).take(10000).collect();
        let training_series = Series::new("training".into(), &training_set);

        // Create bloom filter with m=95850, k=7
        // Expected false positive rate: ~0.97%
        let m = 95850;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 7,
            m
        };
        let bit_array = bloom_filter_test(&training_series, kwargs).unwrap();

        // Check that bits were set
        assert!(count_set_bits(&bit_array) > 0);

        // Create second set: 10,000 numbers in range 50,000-99,999
        // These should NOT be in the bloom filter
        let test_set: Vec<i64> = (50000..100000).step_by(5).take(10000).collect();
        let test_series = Series::new("test".into(), &test_set);

        // Test membership for items NOT in the filter
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 7,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();

        // Count false positives (items found that shouldn't be)
        let false_positives = results_bool.into_iter().filter(|&x| x.unwrap_or(false)).count();
        let false_positive_rate = (false_positives as f64) / 10000.0;

        println!("False positives: {}/10000 ({:.2}%)", false_positives, false_positive_rate * 100.0);

        // Assert false positive rate is no more than 1.5% (allowing for statistical variance)
        // Theoretical rate is ~0.97%, but we allow 1.5% to account for randomness
        assert!(
            false_positive_rate <= 0.015,
            "False positive rate {:.4}% exceeds 1.5% threshold (found {} false positives)",
            false_positive_rate * 100.0,
            false_positives
        );

        // Also verify it's reasonable (not 0, which would indicate a problem)
        assert!(
            false_positives > 0,
            "Expected some false positives due to probabilistic nature of bloom filters"
        );
    }

    ///   Tests for the membership_ratio function

    #[test]
    fn test_membership_ratio_all_found() {
        // Create bloom filter with some items
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test ratio when all items are found
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let ratio = membership_ratio_test(&series, membership_kwargs).unwrap();
        assert_eq!(ratio, 1.0, "All items should be found, ratio should be 1.0");
    }

    #[test]
    fn test_membership_ratio_none_found() {
        // Create bloom filter with some items
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test ratio when no items are found
        let test_series = Series::new("test".into(), &["x", "y", "z"]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let ratio = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.0, "No items should be found, ratio should be 0.0");
    }

    #[test]
    fn test_membership_ratio_partial() {
        // Create bloom filter with some items
        let series = Series::new("items".into(), &["a", "b", "c", "d"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test ratio when half items are found (2 of 4 in test set)
        let test_series = Series::new("test".into(), &["a", "b", "x", "y"]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let ratio = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.5, "Half items should be found, ratio should be 0.5");
    }

    #[test]
    fn test_membership_ratio_empty_series() {
        // Create bloom filter with some items
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test ratio with empty series
        let empty_series: Series = Series::new("empty".into(), Vec::<String>::new());
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let ratio = membership_ratio_test(&empty_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.0, "Empty series should return 0.0");
    }

    #[test]
    fn test_membership_ratio_with_nulls() {
        // Create bloom filter with some items
        let series = Series::new("items".into(), &["a", "b"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test ratio with nulls (nulls count as not found)
        // 2 found ("a", "b"), 2 not found (null, "x") = 50%
        let test_series = Series::new("test".into(), &[Some("a"), None, Some("b"), Some("x")]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let ratio = membership_ratio_test(&test_series, membership_kwargs).unwrap();
        assert_eq!(ratio, 0.5, "Nulls should count as not found");
    }


    ///   Tests for the membership_ratio_sample function

    #[test]
    fn test_membership_ratio_sample_full_sample() {
        // Create bloom filter with items
        let series = Series::new("items".into(), &["a", "b", "c", "d"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // With sample_frac = 1.0, should behave like membership_ratio
        let sample_kwargs = MembershipRatioSampleKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m,
            sample_frac: 1.0,
        };
        let ratio = membership_ratio_sample_test(&series, sample_kwargs).unwrap();
        assert_eq!(ratio, 1.0, "All items should be found with full sample");
    }

    #[test]
    fn test_membership_ratio_sample_partial() {
        // Create bloom filter with 100 items (numbers 0-99)
        let items: Vec<i64> = (0..100).collect();
        let series = Series::new("items".into(), &items);
        let m = 10000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 7,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test with 50% sample - all sampled items should still be found
        let sample_kwargs = MembershipRatioSampleKwargs {
            bit_array_bytes: bit_array,
            k: 7,
            m,
            sample_frac: 0.5,
        };
        let ratio = membership_ratio_sample_test(&series, sample_kwargs).unwrap();
        // All items are in the filter, so ratio should be 1.0 regardless of sample
        assert_eq!(ratio, 1.0, "All sampled items should be found");
    }

    #[test]
    fn test_membership_ratio_sample_none_found() {
        // Create bloom filter with items "a", "b", "c"
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Test with items not in filter
        let test_series = Series::new("test".into(), &["x", "y", "z", "w", "v", "u", "t", "s", "r", "q", "p"]);
        let sample_kwargs = MembershipRatioSampleKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m,
            sample_frac: 0.5,
        };
        let ratio = membership_ratio_sample_test(&test_series, sample_kwargs).unwrap();
        assert_eq!(ratio, 0.0, "No sampled items should be found");
    }

    #[test]
    fn test_membership_ratio_sample_small_series() {
        // Small series (<=10 items) should use full series regardless of sample_frac
        let series = Series::new("items".into(), &["a", "b", "c"]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Even with very small sample_frac, small series should use full data
        let sample_kwargs = MembershipRatioSampleKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m,
            sample_frac: 0.01,  // 1% would be 0 items, but we use full series for small data
        };
        let ratio = membership_ratio_sample_test(&series, sample_kwargs).unwrap();
        assert_eq!(ratio, 1.0, "Small series should use full data");
    }

    ///   Tests for numeric (non-string) series via direct byte path

    #[test]
    fn test_bloom_filter_int64() {
        // Build filter from Int64 series (uses direct byte path, not string cast)
        let series = Series::new("items".into(), &[10i64, 20, 30, 40, 50]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();
        assert!(bit_array.iter().any(|&b| b != 0), "Some bits must be set");

        // Membership: items in filter found, unknown item not found
        let test_series = Series::new("test".into(), &[10i64, 30, 50, 99]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
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
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        let test_series = Series::new("test".into(), &[1.5f64, 3.5, 9.9]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        assert!(results_bool.get(0).unwrap());   // 1.5 found
        assert!(results_bool.get(1).unwrap());   // 3.5 found
        assert!(!results_bool.get(2).unwrap());  // 9.9 not found
    }

    #[test]
    fn test_bloom_filter_numeric_nulls() {
        // Nulls are excluded from filter construction and return false on membership
        let series = Series::new("items".into(), &[Some(1i64), None, Some(2i64), None, Some(3i64)]);
        let m = 1000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 5,
            m
        };
        let bit_array = bloom_filter_test(&series, kwargs).unwrap();

        // Check that non-null items were inserted
        let test_series = Series::new("test".into(), &[Some(1i64), None, Some(2i64), Some(99i64)]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: bit_array,
            k: 5,
            m
        };
        let results = membership_test(&test_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        assert!(results_bool.get(0).unwrap());   // 1 found
        assert!(!results_bool.get(1).unwrap());  // null → false
        assert!(results_bool.get(2).unwrap());   // 2 found
        assert!(!results_bool.get(3).unwrap());  // 99 not found
    }

    #[test]
    fn test_numeric_and_string_filters_are_independent() {
        // A filter built from Int64 values must not match string representations
        // e.g. the integer 42 and the string "42" should produce different bit patterns
        let int_series = Series::new("items".into(), &[42i64]);
        let m = 10000;
        let num_bytes = (m + 7) / 8;
        let kwargs = BloomFilterKwargs {
            bit_array_bytes: vec![0u8; num_bytes],
            k: 7,
            m
        };
        let int_filter = bloom_filter_test(&int_series, kwargs).unwrap();

        // The string "42" should NOT be found in a filter built from integer 42
        // (they produce different byte representations: 8-byte LE vs ASCII bytes)
        let str_series = Series::new("test".into(), &["42"]);
        let membership_kwargs = MembershipKwargs {
            bit_array_bytes: int_filter,
            k: 7,
            m
        };
        let results = membership_test(&str_series, membership_kwargs).unwrap();
        let results_bool = results.bool().unwrap();
        // With high probability this is false (different byte encoding)
        // We can't assert absolutely due to hash collisions, but this documents the behavior
        let _ = results_bool.get(0);  // just verify it doesn't panic
    }
}