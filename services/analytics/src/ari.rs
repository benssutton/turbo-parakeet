// ─────────────────────────────────────────────────────────────────────────────
// Pairwise Adjusted Rand Index
// ─────────────────────────────────────────────────────────────────────────────
//
// ARI measures agreement between two partitions of the same rows, corrected
// for chance. Each column is a partition (rows sharing a value are one
// cluster). ARI = 1 → identical partitions; ≈ 0 → chance-level agreement;
// can go negative (floor −0.5) for worse-than-chance.
//
// Null policy: rows where either column is null are dropped (pairwise
// deletion, via the shared contingency builder) — a null row belongs to no
// cluster, so it must not vote on partition agreement.

use crate::contingency::{build_contingency, ContingencyTable};
use crate::shared::{build_dense_cache_par, resolve_pairs, PairwiseKwargs};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// ARI from a drop-null contingency table.
///
/// With index = Σᵢⱼ C(nᵢⱼ,2), A = Σᵢ C(aᵢ,2), B = Σⱼ C(bⱼ,2), total = C(n,2):
///   ARI = (index − A·B/total) / ((A+B)/2 − A·B/total)
///
/// Sums accumulate in u64 (each is ≤ C(n,2); the C(x,2) multiply overflows
/// only past ~4×10⁹ rows). The final expression is f64 because A·B overflows
/// u64. Edge conventions match sklearn.metrics.adjusted_rand_score:
///   - n_valid == 0 → NaN (no data, undefined)
///   - degenerate denominator (max_index == expected, e.g. both columns
///     constant, both all-singletons, or n_valid == 1) → 1.0: sklearn's
///     "trivially perfect agreement" convention (its fn == 0 && fp == 0 branch).
fn compute_ari(t: &ContingencyTable) -> f64 {
    if t.n_valid == 0 {
        return f64::NAN;
    }

    // Marginal entries can be zero (ids seen only in dropped rows, or a null
    // id — see build_contingency); C(0,2) and C(1,2) are both 0.
    #[inline]
    fn comb2(x: u64) -> u64 {
        x * x.saturating_sub(1) / 2
    }

    let index: u64 = t.cells.iter().map(|&(_, _, c)| comb2(c)).sum();
    let a_sum: u64 = t.marg_a.iter().map(|&c| comb2(c)).sum();
    let b_sum: u64 = t.marg_b.iter().map(|&c| comb2(c)).sum();
    let total = comb2(t.n_valid) as f64;

    // n_valid == 1: no pairs exist, both partitions trivially identical.
    if total == 0.0 {
        return 1.0;
    }

    let expected = a_sum as f64 * b_sum as f64 / total;
    let max_index = (a_sum as f64 + b_sum as f64) / 2.0;

    if max_index == expected {
        return 1.0;
    }

    (index as f64 - expected) / (max_index - expected)
}

pub(crate) fn pairwise_adjusted_rand_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "pairwise_adjusted_rand requires at least one column".into(),
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
            "Cannot calculate ARI on empty columns".into(),
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

    // Build parallel dense cache (encode + dictionary-encode needed columns).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Compute ARI in parallel across all pairs.
    let results: Vec<(String, String, f64, u32)> = pairs
        .par_iter()
        .map(|(i, j)| {
            let table = build_contingency(&cache[*i], &cache[*j], r);
            let ari = compute_ari(&table);
            (
                col_names[*i].clone(),
                col_names[*j].clone(),
                ari,
                table.n_valid as u32,
            )
        })
        .collect();

    // Build output struct series.
    let mut col_a_names = Vec::with_capacity(n_pairs);
    let mut col_b_names = Vec::with_capacity(n_pairs);
    let mut ari_values = Vec::with_capacity(n_pairs);
    let mut n_valid_values = Vec::with_capacity(n_pairs);

    for (a, b, ari, n_valid) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        ari_values.push(ari);
        n_valid_values.push(n_valid);
    }

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let ari_s = Float64Chunked::from_vec("ari".into(), ari_values).into_series();
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), n_valid_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_adjusted_rand".into(),
        n_pairs,
        [col_a_s, col_b_s, ari_s, n_valid_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

fn build_empty_result() -> PolarsResult<Series> {
    let col_a_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_b".into());
    let ari_s = Float64Chunked::from_vec("ari".into(), vec![]).into_series();
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), vec![]).into_series();
    let struct_ca = StructChunked::from_series(
        "pairwise_adjusted_rand".into(),
        0,
        [col_a_s, col_b_s, ari_s, n_valid_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn no_pairs() -> PairwiseKwargs {
        PairwiseKwargs { pairs: None }
    }

    fn ari_of(s1: Series, s2: Series) -> (f64, u32) {
        let result = pairwise_adjusted_rand_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_adjusted_rand"])
            .unwrap();
        let ari = df.column("ari").unwrap().f64().unwrap().get(0).unwrap();
        let n_valid = df.column("n_valid").unwrap().u32().unwrap().get(0).unwrap();
        (ari, n_valid)
    }

    // sklearn doc example: adjusted_rand_score([0,0,1,2], [0,0,1,1]) = 4/7 ≈ 0.5714.
    #[test]
    fn test_sklearn_doc_example() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 2]);
        let s2 = Series::new("b".into(), &[0i32, 0, 1, 1]);
        let (ari, n_valid) = ari_of(s1, s2);
        let expected = 4.0 / 7.0;
        assert!(
            (ari - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            ari
        );
        assert_eq!(n_valid, 4);
    }

    // Identical partitions → 1.0 (label values need not match, only the grouping).
    #[test]
    fn test_identical_partitions() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[5i32, 5, 9, 9, 7, 7]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari - 1.0).abs() < 1e-12, "Expected 1.0, got {}", ari);
    }

    // Maximal disagreement on a 2x2 crossing: sklearn gives exactly -0.5.
    #[test]
    fn test_crossed_partitions_negative() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari + 0.5).abs() < 1e-12, "Expected -0.5, got {}", ari);
    }

    // Null rows dropped: appending a null-containing row to the sklearn doc
    // example must not change the score.
    #[test]
    fn test_null_rows_dropped() {
        let s1 = Series::new("a".into(), &[Some(0i32), Some(0), Some(1), Some(2), None]);
        let s2 = Series::new(
            "b".into(),
            &[Some(0i32), Some(0), Some(1), Some(1), Some(9)],
        );
        let (ari, n_valid) = ari_of(s1, s2);
        let expected = 4.0 / 7.0;
        assert!(
            (ari - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            ari
        );
        assert_eq!(n_valid, 4);
    }

    // Both columns constant → degenerate denominator → 1.0 (sklearn convention).
    #[test]
    fn test_both_constant_is_one() {
        let s1 = Series::new("a".into(), &[1i32, 1, 1, 1]);
        let s2 = Series::new("b".into(), &[2i32, 2, 2, 2]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari - 1.0).abs() < 1e-12, "Expected 1.0, got {}", ari);
    }

    // No non-null overlap → NaN, n_valid 0.
    #[test]
    fn test_disjoint_nulls_nan() {
        let s1 = Series::new("a".into(), &[Some(1i32), Some(2), None, None]);
        let s2 = Series::new("b".into(), &[None::<i32>, None, Some(1), Some(2)]);
        let (ari, n_valid) = ari_of(s1, s2);
        assert!(ari.is_nan(), "Expected NaN, got {}", ari);
        assert_eq!(n_valid, 0);
    }

    #[test]
    fn test_three_columns_three_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let result = pairwise_adjusted_rand_impl(&[s1, s2, s3], no_pairs()).unwrap();
        assert_eq!(result.len(), 3); // 3-choose-2
    }

    #[test]
    fn test_specific_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "c".to_string()]]),
        };
        let result = pairwise_adjusted_rand_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_invalid_pair_column_errors() {
        let s1 = Series::new("a".into(), &[1i32, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "nope".to_string()]]),
        };
        assert!(pairwise_adjusted_rand_impl(&[s1, s2], kwargs).is_err());
    }

    #[test]
    fn test_empty_inputs_error() {
        assert!(pairwise_adjusted_rand_impl(&[], no_pairs()).is_err());
    }

    #[test]
    fn test_single_column_empty_result() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let result = pairwise_adjusted_rand_impl(&[s1], no_pairs()).unwrap();
        assert_eq!(result.len(), 0);
    }
}
