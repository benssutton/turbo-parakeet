//! Marginal entropy H(col) of every column: an independent cross-check of the joint-entropy counting.

use foldhash::fast::RandomState as FoldHashFast;
use polars::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::common::entropy_math::*;
use crate::common::*;

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
    let entropy_s =
        Float64Chunked::from_vec("entropy".into(), results.iter().map(|(_, e)| *e).collect())
            .into_series();

    let struct_ca = StructChunked::from_series(
        "marginal_entropy".into(),
        n_cols,
        [col_name_s, entropy_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            (h - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            h
        );
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
        assert!((0.0..=1.0).contains(&h));
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
        let s = Series::new(
            "test".into(),
            &[Some(1_000_000i64), Some(2_000_000i64), None],
        )
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
}
