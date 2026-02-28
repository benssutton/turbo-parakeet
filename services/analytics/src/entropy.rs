use rayon::prelude::*;
use foldhash::fast::{FixedState as FoldHashFixed, RandomState as FoldHashFast};
use scc::HashMap as SccHashMap;
use wide::f64x4;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};

type FoldHashFastSccMap<K, V> = SccHashMap<K, V, FoldHashFast>;

// ─────────────────────────────────────────────────────────────────────────────
// Kwargs (same wire format as entropy.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct PairwiseKwargs {
    pub pairs: Option<Vec<Vec<String>>>,
}

#[derive(Deserialize)]
pub struct ThreewayKwargs {
    pub triplets: Option<Vec<Vec<String>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Null-safe column conversion (nulls → u64::MAX)
// ─────────────────────────────────────────────────────────────────────────────

const NULL_SENTINEL: u64 = u64::MAX;

fn series_to_u64(series: &Series) -> PolarsResult<Vec<u64>> {
    match series.dtype() {
        DataType::Int8 => Ok(series
            .i8()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::Int16 => Ok(series
            .i16()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::Int32 => Ok(series
            .i32()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::Int64 => Ok(series
            .i64()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::UInt8 => Ok(series
            .u8()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::UInt16 => Ok(series
            .u16()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::UInt32 => Ok(series
            .u32()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::UInt64 => Ok(series
            .u64()?
            .iter()
            .map(|v| v.unwrap_or(NULL_SENTINEL))
            .collect()),
        DataType::Float32 => Ok(series
            .f32()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x.to_bits() as u64))
            .collect()),
        DataType::Float64 => Ok(series
            .f64()?
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x.to_bits()))
            .collect()),
        DataType::Date => Ok(series
            .date()?
            .phys
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::Datetime(_, _) => Ok(series
            .datetime()?
            .phys
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::Duration(_) => Ok(series
            .duration()?
            .phys
            .iter()
            .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
            .collect()),
        DataType::String => {
            let build_hasher = FoldHashFixed::default();
            Ok(series
                .str()?
                .iter()
                .map(|v| match v {
                    Some(s) => {
                        let mut hasher = build_hasher.build_hasher();
                        s.hash(&mut hasher);
                        hasher.finish()
                    }
                    None => NULL_SENTINEL,
                })
                .collect())
        }
        _ => Err(PolarsError::ComputeError(
            format!(
                "Unsupported data type for joint entropy: {:?}. \
                 Supported: Int8/16/32/64, UInt8/16/32/64, Float32/64, \
                 Date, Datetime, Duration, String",
                series.dtype()
            )
            .into(),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel column cache
// ─────────────────────────────────────────────────────────────────────────────

fn build_column_cache_par(
    inputs: &[Series],
    needed: &HashSet<usize>,
) -> PolarsResult<Vec<Vec<u64>>> {
    // Collect (index, series) pairs for needed columns, convert in parallel,
    // then scatter results back into positional Vec.
    let tasks: Vec<(usize, &Series)> = inputs
        .iter()
        .enumerate()
        .filter(|(idx, _)| needed.contains(idx))
        .collect();

    let converted: Vec<(usize, Vec<u64>)> = tasks
        .into_par_iter()
        .map(|(idx, series)| {
            let data = series_to_u64(series)?;
            Ok((idx, data))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut cache: Vec<Vec<u64>> = (0..inputs.len()).map(|_| Vec::new()).collect();
    for (idx, data) in converted {
        cache[idx] = data;
    }
    Ok(cache)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair / triplet resolution from kwargs
// ─────────────────────────────────────────────────────────────────────────────

fn resolve_pairs(
    raw_pairs: &[Vec<String>],
    name_map: &HashMap<String, usize>,
) -> PolarsResult<Vec<(usize, usize)>> {
    raw_pairs
        .iter()
        .map(|pair| {
            if pair.len() != 2 {
                return Err(PolarsError::ComputeError(
                    format!("Each pair must have exactly 2 column names, got {}", pair.len())
                        .into(),
                ));
            }
            let i = *name_map.get(&pair[0]).ok_or_else(|| {
                PolarsError::ColumnNotFound(pair[0].clone().into())
            })?;
            let j = *name_map.get(&pair[1]).ok_or_else(|| {
                PolarsError::ColumnNotFound(pair[1].clone().into())
            })?;
            Ok((i, j))
        })
        .collect()
}

fn resolve_triplets(
    raw_triplets: &[Vec<String>],
    name_map: &HashMap<String, usize>,
) -> PolarsResult<Vec<(usize, usize, usize)>> {
    raw_triplets
        .iter()
        .map(|triplet| {
            if triplet.len() != 3 {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Each triplet must have exactly 3 column names, got {}",
                        triplet.len()
                    )
                    .into(),
                ));
            }
            let i = *name_map.get(&triplet[0]).ok_or_else(|| {
                PolarsError::ColumnNotFound(triplet[0].clone().into())
            })?;
            let j = *name_map.get(&triplet[1]).ok_or_else(|| {
                PolarsError::ColumnNotFound(triplet[1].clone().into())
            })?;
            let k = *name_map.get(&triplet[2]).ok_or_else(|| {
                PolarsError::ColumnNotFound(triplet[2].clone().into())
            })?;
            Ok((i, j, k))
        })
        .collect()
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
/// Formula: H = -Σ n * c * (log2(c) - log2(r)) / r
///
/// Uses SIMD f64x4 processing 4 (c, n) pairs per iteration with scalar tail.
fn entropy_from_count_of_counts(coc: &[(u64, u64)], logr: f64, r_f: f64) -> f64 {
    let logr_vec = f64x4::from([logr; 4]);
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
        acc -= n * c * (c.log2() - logr_vec);
    }

    let mut entropy: f64 = acc.reduce_add();
    for &(c_val, n_val) in &coc[full_chunks * 4..] {
        let c = c_val as f64;
        let n = n_val as f64;
        entropy -= n * c * (c.log2() - logr);
    }
    entropy / r_f
}

// ─────────────────────────────────────────────────────────────────────────────
// Pairwise (2-way) v2
// ─────────────────────────────────────────────────────────────────────────────

fn pairwise_entropy_v2_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
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

pub(crate) fn pairwise_joint_entropy_v2_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "pairwise_joint_entropy_v2 requires at least one column".into(),
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

    // Step 3: parallel column cache (convert only needed columns, in parallel).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_column_cache_par(inputs, &needed)?;

    // Steps 4–5: parallel entropy calculation across all pairs.
    let results: Vec<_> = pairs
        .par_iter()
        .map(|(i, j)| {
            let col_a = &cache[*i];
            let col_b = &cache[*j];
            let col_a_name = inputs[*i].name().to_string();
            let col_b_name = inputs[*j].name().to_string();

            // 5a+5b: pack into u128 keys, build frequency map.
            let freq: FoldHashFastSccMap<u128, u64> =
                SccHashMap::with_capacity_and_hasher(r, FoldHashFast::default());
            col_a.iter().zip(col_b.iter()).for_each(|(a, b)| {
                let key = (*a as u128) << 64 | (*b as u128);
                let _ = freq.entry(key).and_modify(|v| *v += 1).or_insert(1);
            });

            // 5c: count-of-counts.
            let mut coc_map: HashMap<u64, u64, FoldHashFast> =
                HashMap::with_capacity_and_hasher(freq.len(), FoldHashFast::default());
            freq.scan(|_, count| {
                *coc_map.entry(*count).or_insert(0) += 1;
            });
            let coc: Vec<(u64, u64)> = coc_map.into_iter().collect();

            // 5d: SIMD entropy from count-of-counts.
            let entropy = entropy_from_count_of_counts(&coc, logr, r_f);

            Ok((col_a_name, col_b_name, entropy))
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

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s| s.as_str()))
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

#[polars_expr(output_type_func=pairwise_entropy_v2_output_type)]
fn pairwise_joint_entropy_v2(inputs: &[Series], kwargs: PairwiseKwargs) -> PolarsResult<Series> {
    pairwise_joint_entropy_v2_impl(inputs, kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Threeway (3-way) v2
// ─────────────────────────────────────────────────────────────────────────────

fn threeway_entropy_v2_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
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

pub(crate) fn threeway_joint_entropy_v2_impl(
    inputs: &[Series],
    kwargs: ThreewayKwargs,
) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return Err(PolarsError::ComputeError(
            "threeway_joint_entropy_v2 requires at least three columns".into(),
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
        None => {
            const MAX_TRIPLETS: usize = 5000;
            (0..n_cols)
                .flat_map(|i| {
                    ((i + 1)..n_cols)
                        .flat_map(move |j| ((j + 1)..n_cols).map(move |k| (i, j, k)))
                })
                .take(MAX_TRIPLETS)
                .collect()
        }
    };

    let n_triplets = triplets.len();

    // Step 3: parallel column cache.
    let needed: HashSet<usize> = triplets
        .iter()
        .flat_map(|(i, j, k)| [*i, *j, *k])
        .collect();
    let cache = build_column_cache_par(inputs, &needed)?;

    // Steps 4–5: parallel entropy calculation across all triplets.
    let results: Vec<_> = triplets
        .par_iter()
        .map(|(i, j, k)| {
            let col_a = &cache[*i];
            let col_b = &cache[*j];
            let col_c = &cache[*k];
            let col_a_name = inputs[*i].name().to_string();
            let col_b_name = inputs[*j].name().to_string();
            let col_c_name = inputs[*k].name().to_string();

            // 5a+5b: pack into [u64; 3] keys, build frequency map.
            let freq: FoldHashFastSccMap<[u64; 3], u64> =
                SccHashMap::with_capacity_and_hasher(r, FoldHashFast::default());
            col_a
                .iter()
                .zip(col_b.iter())
                .zip(col_c.iter())
                .for_each(|((a, b), c)| {
                    let key = [*a, *b, *c];
                    let _ = freq.entry(key).and_modify(|v| *v += 1).or_insert(1);
                });

            // 5c: count-of-counts.
            let mut coc_map: HashMap<u64, u64, FoldHashFast> =
                HashMap::with_capacity_and_hasher(freq.len(), FoldHashFast::default());
            freq.scan(|_, count| {
                *coc_map.entry(*count).or_insert(0) += 1;
            });
            let coc: Vec<(u64, u64)> = coc_map.into_iter().collect();

            // 5d: SIMD entropy from count-of-counts.
            let entropy = entropy_from_count_of_counts(&coc, logr, r_f);

            Ok((col_a_name, col_b_name, col_c_name, entropy))
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

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let col_c_s = StringChunked::from_iter(col_c_names.iter().map(|s| s.as_str()))
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

#[polars_expr(output_type_func=threeway_entropy_v2_output_type)]
fn threeway_joint_entropy_v2(inputs: &[Series], kwargs: ThreewayKwargs) -> PolarsResult<Series> {
    threeway_joint_entropy_v2_impl(inputs, kwargs)
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

    // ── Null-safe conversion ───────────────────────────────────────────────

    #[test]
    fn test_null_max_sentinel() {
        let s = Series::new("test".into(), &[Some(1i32), None, Some(3)]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result, vec![1u64, NULL_SENTINEL, 3]);
    }

    #[test]
    fn test_null_does_not_collide_with_zero() {
        let s = Series::new("test".into(), &[Some(0i32), None]);
        let result = series_to_u64(&s).unwrap();
        assert_ne!(result[0], result[1], "Null must not collide with zero");
        assert_eq!(result[0], 0);
        assert_eq!(result[1], NULL_SENTINEL);
    }

    #[test]
    fn test_string_null_max() {
        let s = Series::new("test".into(), &[Some("a"), None, Some("b")]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result[1], NULL_SENTINEL);
        assert_ne!(result[0], NULL_SENTINEL);
    }

    #[test]
    fn test_float_null_max() {
        let s = Series::new("test".into(), &[Some(1.5f64), None]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result[0], 1.5f64.to_bits());
        assert_eq!(result[1], NULL_SENTINEL);
    }

    // ── Pairwise v2 ───────────────────────────────────────────────────────

    #[test]
    fn test_pairwise_v2_uniform() {
        // 4 unique pairs, each once → H = log2(4) = 2.0
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2], no_pairs()).unwrap();
        assert_eq!(result.len(), 1);

        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!((h - 2.0).abs() < 1e-10, "Expected 2.0, got {}", h);
    }

    #[test]
    fn test_pairwise_v2_deterministic() {
        let s1 = Series::new("a".into(), &[1i32; 100]);
        let s2 = Series::new("b".into(), &[1i32; 100]);
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2], no_pairs()).unwrap();

        let df = result.into_frame().unnest(["pairwise_entropy"]).unwrap();
        let h = df.column("entropy").unwrap().f64().unwrap().get(0).unwrap();
        assert!(h.abs() < 1e-10, "Expected 0.0, got {}", h);
    }

    #[test]
    fn test_pairwise_v2_3_columns() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2, s3], no_pairs()).unwrap();
        assert!(matches!(result.dtype(), DataType::Struct(_)));
        assert_eq!(result.len(), 3); // 3-choose-2 = 3 pairs
    }

    #[test]
    fn test_pairwise_v2_single_column() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let result = pairwise_joint_entropy_v2_impl(&[s1], no_pairs()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pairwise_v2_empty_inputs() {
        let result = pairwise_joint_entropy_v2_impl(&[], no_pairs());
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_v2_length_mismatch() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2], no_pairs());
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_v2_specific_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "c".to_string()]]),
        };
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_pairwise_v2_specific_pairs_multiple() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["a".to_string(), "c".to_string()],
            ]),
        };
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_pairwise_v2_invalid_column_name() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "nonexistent".to_string()]]),
        };
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_pairwise_v2_invalid_pair_length() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);

        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string()]]),
        };
        let result = pairwise_joint_entropy_v2_impl(&[s1, s2], kwargs);
        assert!(result.is_err());
    }

    // ── Threeway v2 ────────────────────────────────────────────────────────

    #[test]
    fn test_threeway_v2_basic() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3], no_triplets()).unwrap();
        assert!(matches!(result.dtype(), DataType::Struct(_)));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_threeway_v2_four_columns() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let s4 = Series::new("d".into(), &[1i32, 1, 2, 1]);

        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3, s4], no_triplets()).unwrap();
        assert_eq!(result.len(), 4); // 4-choose-3 = 4
    }

    #[test]
    fn test_threeway_v2_insufficient_columns() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2, 3]);
        let result = threeway_joint_entropy_v2_impl(&[s1, s2], no_triplets());
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_v2_length_mismatch() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let s3 = Series::new("c".into(), &[1i32, 2, 3]);
        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3], no_triplets());
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_v2_specific_triplets() {
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
        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3, s4], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_threeway_v2_specific_triplets_multiple() {
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
        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3, s4], kwargs).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_threeway_v2_invalid_triplet_column() {
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
        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_v2_invalid_triplet_length() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);

        let kwargs = ThreewayKwargs {
            triplets: Some(vec![vec!["a".to_string(), "b".to_string()]]),
        };
        let result = threeway_joint_entropy_v2_impl(&[s1, s2, s3], kwargs);
        assert!(result.is_err());
    }

    #[test]
    fn test_threeway_v2_max_triplets_limit() {
        let series: Vec<Series> = (0..40)
            .map(|i| Series::new(format!("col_{}", i).into(), &[1i32, 2, 3, 4]))
            .collect();
        let result = threeway_joint_entropy_v2_impl(&series, no_triplets()).unwrap();
        assert_eq!(result.len(), 5000);
    }
}
