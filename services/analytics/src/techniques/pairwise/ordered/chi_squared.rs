use crate::common::{build_dense_cache_par, resolve_pairs, PairwiseKwargs};
use crate::techniques::contingency::{build_contingency, ContingencyTable};
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{HashMap, HashSet};

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

    // Build parallel dense cache (encode + dictionary-encode needed columns).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Compute chi-squared stats in parallel across all pairs.
    let results: Vec<(String, String, f64, f64, f64, bool, u32)> = pairs
        .par_iter()
        .map(|(i, j)| {
            let table = build_contingency(&cache[*i], &cache[*j], r);
            let (chi2, p, v, low_exp) = compute_chi_squared(&table);
            (
                col_names[*i].clone(),
                col_names[*j].clone(),
                chi2,
                p,
                v,
                low_exp,
                table.n_valid as u32,
            )
        })
        .collect();

    // Build output struct series.
    let mut col_a_names = Vec::with_capacity(n_pairs);
    let mut col_b_names = Vec::with_capacity(n_pairs);
    let mut chi2_values = Vec::with_capacity(n_pairs);
    let mut p_values = Vec::with_capacity(n_pairs);
    let mut cramers_v_values = Vec::with_capacity(n_pairs);
    let mut low_exp_values = Vec::with_capacity(n_pairs);
    let mut n_valid_values = Vec::with_capacity(n_pairs);

    for (a, b, chi2, p, v, low_exp, n_valid) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        chi2_values.push(chi2);
        p_values.push(p);
        cramers_v_values.push(v);
        low_exp_values.push(low_exp);
        n_valid_values.push(n_valid);
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
    let low_exp_s = BooleanChunked::from_iter(low_exp_values)
        .into_series()
        .with_name("low_expected_count".into());
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), n_valid_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        n_pairs,
        [col_a_s, col_b_s, chi2_s, p_s, v_s, low_exp_s, n_valid_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

/// Compute chi-squared statistic, p-value, Cramer's V, and low-expected-count flag
/// from a drop-null contingency table.
///
/// Null policy: rows where either column is null are dropped (pairwise deletion)
/// by build_contingency. Note this differs from the entropy kernels, which treat
/// null as its own category — keep that in mind when deriving mutual information
/// from the two outputs.
///
/// Returns (chi2_stat, p_value, cramers_v, low_expected_count); NaN for degenerate
/// inputs. `low_expected_count` is true when the minimum expected cell count (rarest
/// row marginal × rarest col marginal / N) is below 5, the standard threshold above
/// which the chi-squared approximation is reliable.
fn compute_chi_squared(t: &ContingencyTable) -> (f64, f64, f64, bool) {
    if t.n_valid == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, false);
    }

    // Marginal entries can be zero (ids seen only in dropped rows, or a null id);
    // uniqueness and minima consider non-zero entries only.
    let unique_a = t.marg_a.iter().filter(|&&c| c > 0).count();
    let unique_b = t.marg_b.iter().filter(|&&c| c > 0).count();

    // Degenerate: constant column — chi-squared is undefined.
    if unique_a < 2 || unique_b < 2 {
        return (f64::NAN, f64::NAN, f64::NAN, false);
    }

    let n_f = t.n_valid as f64;

    // χ² = Σ O²/E − N, summing only over observed (non-zero) cells since zero-obs
    // cells contribute 0 to Σ O²/E. This is O(observed pairs) rather than the
    // O(unique_a × unique_b) nested-loop form, which hangs on high-cardinality columns.
    let mut chi2_stat = 0.0f64;
    for &(ai, bi, obs) in &t.cells {
        let expected = t.marg_a[ai as usize] as f64 * t.marg_b[bi as usize] as f64 / n_f;
        chi2_stat += (obs as f64 * obs as f64) / expected;
    }
    chi2_stat -= n_f;

    // low_expected_count: true when the minimum possible expected cell count falls
    // below 5, the standard chi-squared validity threshold.
    let min_row = *t.marg_a.iter().filter(|&&c| c > 0).min().unwrap() as f64;
    let min_col = *t.marg_b.iter().filter(|&&c| c > 0).min().unwrap() as f64;
    let low_expected_count = min_row * min_col / n_f < 5.0;

    // Degrees of freedom = (unique_a - 1) * (unique_b - 1).
    let df_val = ((unique_a - 1) * (unique_b - 1)) as f64;

    // Use sf (survival function) rather than 1 - cdf to avoid catastrophic
    // cancellation in the far tail where χ² is most significant.
    let p_value = match ChiSquared::new(df_val) {
        Ok(dist) => dist.sf(chi2_stat),
        Err(_) => f64::NAN,
    };

    let min_dim = (unique_a - 1).min(unique_b - 1) as f64;
    let cramers_v = (chi2_stat / (n_f * min_dim)).sqrt();

    (chi2_stat, p_value, cramers_v, low_expected_count)
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
    let low_exp_s = BooleanChunked::from_iter(std::iter::empty::<bool>())
        .into_series()
        .with_name("low_expected_count".into());
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), vec![]).into_series();
    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        0,
        [col_a_s, col_b_s, chi2_s, p_s, v_s, low_exp_s, n_valid_s].iter(),
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

    // Known 2×2 table: independence (no association).
    // col_a: [0,0,1,1], col_b: [0,1,0,1] → all cells = 1, chi2 = 0, p = 1.
    #[test]
    fn test_independent_columns() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let chi2 = df
            .column("chi2_stat")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let p = df.column("p_value").unwrap().f64().unwrap().get(0).unwrap();
        let v = df
            .column("cramers_v")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
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
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let chi2 = df
            .column("chi2_stat")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let p = df.column("p_value").unwrap().f64().unwrap().get(0).unwrap();
        let v = df
            .column("cramers_v")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(chi2 > 0.0, "Expected chi2 > 0 for perfect association");
        assert!(
            p < 0.05,
            "Expected p < 0.05 for perfect association, got {}",
            p
        );
        assert!(
            (v - 1.0).abs() < 1e-10,
            "Expected V=1 for perfect association, got {}",
            v
        );
    }

    // Constant column → NaN (test undefined).
    #[test]
    fn test_constant_column_produces_nan() {
        let s1 = Series::new("a".into(), &[1i32, 1, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let chi2 = df
            .column("chi2_stat")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(
            chi2.is_nan(),
            "Expected NaN for constant column, got {}",
            chi2
        );
    }

    // Null rows are dropped before computing.
    #[test]
    fn test_null_rows_dropped() {
        // With nulls: [0,null,1,1] × [0,1,0,1] → null row dropped → 3 valid rows
        let s1 = Series::new("a".into(), &[Some(0i32), None, Some(1), Some(1)]);
        let s2 = Series::new("b".into(), &[Some(0i32), Some(1), Some(0), Some(1)]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let chi2 = df
            .column("chi2_stat")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
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

    // low_expected_count is true when rarest marginal pair expected count < 5.
    #[test]
    fn test_low_expected_count_sparse() {
        // 10 rows, 10 unique values in each column → most expected cells < 1.
        let a: Vec<i32> = (0..10).collect();
        let b: Vec<i32> = (0..10).collect();
        let s1 = Series::new("a".into(), &a);
        let s2 = Series::new("b".into(), &b);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let low = df
            .column("low_expected_count")
            .unwrap()
            .bool()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(low, "sparse contingency table must set low_expected_count");
    }

    #[test]
    fn test_low_expected_count_dense() {
        // 10000 rows split evenly across 2 categories → min expected = 2500, not low.
        let a: Vec<i32> = (0..10000).map(|i| i % 2).collect();
        let b: Vec<i32> = (0..10000).map(|i| i % 2).collect();
        let s1 = Series::new("a".into(), &a);
        let s2 = Series::new("b".into(), &b);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let low = df
            .column("low_expected_count")
            .unwrap()
            .bool()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(
            !low,
            "dense 2×2 table (n=10000) must not set low_expected_count"
        );
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

    #[test]
    fn test_n_valid_counts_non_null_overlap() {
        // Rows 2 (a null) and 3 (b null) dropped → n_valid = 4.
        let s1 = Series::new(
            "a".into(),
            &[Some(0i32), Some(0), None, Some(1), Some(1), Some(0)],
        );
        let s2 = Series::new(
            "b".into(),
            &[Some(0i32), Some(1), Some(0), None, Some(1), Some(0)],
        );
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result
            .into_frame()
            .unnest(["pairwise_chi_squared"])
            .unwrap();
        let n_valid = df.column("n_valid").unwrap().u32().unwrap().get(0).unwrap();
        assert_eq!(n_valid, 4);
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
