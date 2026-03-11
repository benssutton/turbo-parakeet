use crate::shared::{
    NULL_SENTINEL, PairwiseKwargs, build_column_cache_par, resolve_pairs,
};
use foldhash::fast::RandomState as FoldHashFast;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Output type
// ─────────────────────────────────────────────────────────────────────────────

fn chi_squared_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("col_a".into(), DataType::String),
        Field::new("col_b".into(), DataType::String),
        Field::new("chi2_stat".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("cramers_v".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "pairwise_chi_squared".into(),
        DataType::Struct(fields),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn pairwise_chi_squared_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "pairwise_chi_squared requires at least one column".into(),
        ));
    }

    let n_cols = inputs.len();

    // Fewer than 2 columns → no pairs, return empty struct.
    if n_cols < 2 {
        return build_empty_result();
    }

    let r = inputs[0].len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate chi-squared on empty columns".into(),
        ));
    }

    // Validate uniform length.
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

    // Resolve pairs from kwargs or generate all N-choose-2.
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

    // Build parallel column cache (only convert needed columns).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_column_cache_par(inputs, &needed)?;

    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Compute chi-squared stats in parallel across all pairs.
    let results: Vec<(String, String, f64, f64, f64)> = pairs
        .par_iter()
        .map(|(i, j)| {
            let col_a = &cache[*i];
            let col_b = &cache[*j];
            let col_a_name = col_names[*i].clone();
            let col_b_name = col_names[*j].clone();

            let (chi2, p, v) = compute_chi_squared(col_a, col_b);
            (col_a_name, col_b_name, chi2, p, v)
        })
        .collect();

    // Build output struct series.
    let mut col_a_names = Vec::with_capacity(n_pairs);
    let mut col_b_names = Vec::with_capacity(n_pairs);
    let mut chi2_values = Vec::with_capacity(n_pairs);
    let mut p_values = Vec::with_capacity(n_pairs);
    let mut cramers_v_values = Vec::with_capacity(n_pairs);

    for (a, b, chi2, p, v) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        chi2_values.push(chi2);
        p_values.push(p);
        cramers_v_values.push(v);
    }

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let chi2_s = Float64Chunked::from_vec("chi2_stat".into(), chi2_values).into_series();
    let p_s = Float64Chunked::from_vec("p_value".into(), p_values).into_series();
    let v_s = Float64Chunked::from_vec("cramers_v".into(), cramers_v_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        n_pairs,
        [col_a_s, col_b_s, chi2_s, p_s, v_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

/// Compute chi-squared statistic, p-value, and Cramer's V for one column pair.
///
/// Null rows (either column is NULL_SENTINEL) are dropped before computing.
/// Returns (chi2_stat, p_value, cramers_v); NaN for degenerate inputs.
fn compute_chi_squared(col_a: &[u64], col_b: &[u64]) -> (f64, f64, f64) {
    // Single pass: build joint freq map + marginals, skipping null rows.
    let mut joint: HashMap<u128, u64, FoldHashFast> =
        HashMap::with_capacity_and_hasher(64, FoldHashFast::default());
    let mut row_m: HashMap<u64, u64, FoldHashFast> =
        HashMap::with_capacity_and_hasher(32, FoldHashFast::default());
    let mut col_m: HashMap<u64, u64, FoldHashFast> =
        HashMap::with_capacity_and_hasher(32, FoldHashFast::default());
    let mut n_valid: u64 = 0;

    for (a, b) in col_a.iter().zip(col_b.iter()) {
        if *a == NULL_SENTINEL || *b == NULL_SENTINEL {
            continue;
        }
        let key = (*a as u128) << 64 | (*b as u128);
        *joint.entry(key).or_insert(0) += 1;
        *row_m.entry(*a).or_insert(0) += 1;
        *col_m.entry(*b).or_insert(0) += 1;
        n_valid += 1;
    }

    if n_valid == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let unique_a = row_m.len();
    let unique_b = col_m.len();

    // Degenerate: constant column — chi-squared is undefined.
    if unique_a < 2 || unique_b < 2 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let n_f = n_valid as f64;

    // Second pass over joint freq map: compute chi-squared statistic.
    let mut chi2_stat = 0.0f64;
    for (&key, &obs) in &joint {
        let a_key = (key >> 64) as u64;
        let b_key = (key & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        let row_total = *row_m.get(&a_key).unwrap() as f64;
        let col_total = *col_m.get(&b_key).unwrap() as f64;
        let expected = row_total * col_total / n_f;
        let diff = obs as f64 - expected;
        chi2_stat += diff * diff / expected;
    }

    // Degrees of freedom = (unique_a - 1) * (unique_b - 1).
    let df_val = ((unique_a - 1) * (unique_b - 1)) as f64;

    let p_value = match ChiSquared::new(df_val) {
        Ok(dist) => 1.0 - dist.cdf(chi2_stat),
        Err(_) => f64::NAN,
    };

    let min_dim = (unique_a - 1).min(unique_b - 1) as f64;
    let cramers_v = (chi2_stat / (n_f * min_dim)).sqrt();

    (chi2_stat, p_value, cramers_v)
}

fn build_empty_result() -> PolarsResult<Series> {
    let col_a_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_b".into());
    let chi2_s = Float64Chunked::from_vec("chi2_stat".into(), vec![]).into_series();
    let p_s = Float64Chunked::from_vec("p_value".into(), vec![]).into_series();
    let v_s = Float64Chunked::from_vec("cramers_v".into(), vec![]).into_series();
    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        0,
        [col_a_s, col_b_s, chi2_s, p_s, v_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin entry point
// ─────────────────────────────────────────────────────────────────────────────

#[polars_expr(output_type_func=chi_squared_output_type)]
fn pairwise_chi_squared(inputs: &[Series], kwargs: PairwiseKwargs) -> PolarsResult<Series> {
    pairwise_chi_squared_impl(inputs, kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::*;

    fn no_pairs() -> PairwiseKwargs {
        PairwiseKwargs { pairs: None }
    }

    // Known 2×2 table: independence (no association).
    // col_a: [0,0,1,1], col_b: [0,1,0,1] → all cells = 1, chi2 = 0, p = 1.
    #[test]
    fn test_independent_columns() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_chi_squared"]).unwrap();
        let chi2 = df.column("chi2_stat").unwrap().f64().unwrap().get(0).unwrap();
        let p = df.column("p_value").unwrap().f64().unwrap().get(0).unwrap();
        let v = df.column("cramers_v").unwrap().f64().unwrap().get(0).unwrap();
        assert!(chi2.abs() < 1e-10, "Expected chi2~0, got {}", chi2);
        assert!((p - 1.0).abs() < 1e-6, "Expected p~1, got {}", p);
        assert!(v.abs() < 1e-10, "Expected V~0, got {}", v);
    }

    // Perfect association: col_a == col_b → chi2 should be at its maximum for this table.
    #[test]
    fn test_perfectly_associated_columns() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 0, 1, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_chi_squared"]).unwrap();
        let chi2 = df.column("chi2_stat").unwrap().f64().unwrap().get(0).unwrap();
        let p = df.column("p_value").unwrap().f64().unwrap().get(0).unwrap();
        let v = df.column("cramers_v").unwrap().f64().unwrap().get(0).unwrap();
        assert!(chi2 > 0.0, "Expected chi2 > 0 for perfect association");
        assert!(p < 0.05, "Expected p < 0.05 for perfect association, got {}", p);
        assert!((v - 1.0).abs() < 1e-10, "Expected V=1 for perfect association, got {}", v);
    }

    // Constant column → NaN (test undefined).
    #[test]
    fn test_constant_column_produces_nan() {
        let s1 = Series::new("a".into(), &[1i32, 1, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_chi_squared"]).unwrap();
        let chi2 = df.column("chi2_stat").unwrap().f64().unwrap().get(0).unwrap();
        assert!(chi2.is_nan(), "Expected NaN for constant column, got {}", chi2);
    }

    // Null rows are dropped before computing.
    #[test]
    fn test_null_rows_dropped() {
        // With nulls: [0,null,1,1] × [0,1,0,1] → null row dropped → 3 valid rows
        let s1 = Series::new("a".into(), &[Some(0i32), None, Some(1), Some(1)]);
        let s2 = Series::new("b".into(), &[Some(0i32), Some(1), Some(0), Some(1)]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_chi_squared"]).unwrap();
        let chi2 = df.column("chi2_stat").unwrap().f64().unwrap().get(0).unwrap();
        assert!(!chi2.is_nan(), "Should produce a result with nulls dropped");
    }

    // Empty inputs → error.
    #[test]
    fn test_empty_inputs_error() {
        let result = pairwise_chi_squared_impl(&[], no_pairs());
        assert!(result.is_err());
    }

    // Single column → empty struct (no pairs).
    #[test]
    fn test_single_column_empty() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let result = pairwise_chi_squared_impl(&[s1], no_pairs()).unwrap();
        assert_eq!(result.len(), 0);
    }

    // 3 columns → 3 pairs.
    #[test]
    fn test_three_columns_three_pairs() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let s3 = Series::new("c".into(), &[0i32, 0, 0, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2, s3], no_pairs()).unwrap();
        assert_eq!(result.len(), 3);
    }

    // Specific pairs kwarg.
    #[test]
    fn test_specific_pairs() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let s3 = Series::new("c".into(), &[0i32, 0, 0, 1]);
        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "c".to_string()]]),
        };
        let result = pairwise_chi_squared_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }
}
