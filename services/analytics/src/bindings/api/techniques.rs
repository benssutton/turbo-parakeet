//! Entry points of the techniques: one function per kernel, Arrow in, Arrow out.

use super::*;

use arrow_array::RecordBatch;
use polars::prelude::{IntoSeries, StructChunked};

use crate::common::ThreewayKwargs;
use crate::techniques::bloomfilter::{BloomFilterKwargs, MembershipKwargs};
use crate::techniques::minhash::{LSHKwargs, MinHashKwargs};

/// `column`, `dtype`, `gcd: decimal128(38, 0)` per column.
pub fn column_gcd(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::techniques::gcd::column_gcd_impl(&columns(batch)?))
}

/// `col_name`, `entropy` per column.
pub fn marginal_entropy(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::techniques::marginal_entropy::marginal_entropy_impl(
        &columns(batch)?,
    ))
}

/// `col_a`, `col_b`, `entropy` per pair; every pair when `pairs` is None.
pub fn pairwise_joint_entropy(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::techniques::joint_entropy::pairwise_joint_entropy_impl(&columns(batch)?, kwargs))
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet; every triplet when None.
pub fn threeway_joint_entropy(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
) -> Result<RecordBatch> {
    if let Some(t) = triplets {
        check_names(batch, t.iter().flat_map(|(a, b, c)| [a, b, c]))?;
    }
    let kwargs = ThreewayKwargs {
        triplets: triplets.map(|t| {
            t.iter()
                .map(|(a, b, c)| vec![a.clone(), b.clone(), c.clone()])
                .collect()
        }),
    };
    table(crate::techniques::joint_entropy::threeway_joint_entropy_impl(&columns(batch)?, kwargs))
}

/// `col_a`, `col_b`, `chi2_stat`, `p_value`, `cramers_v`, `low_expected_count`, `n_valid`.
pub fn pairwise_chi_squared(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::techniques::chi_squared::pairwise_chi_squared_impl(
        &columns(batch)?,
        kwargs,
    ))
}

/// `col_a`, `col_b`, `ari`, `n_valid` per pair.
pub fn pairwise_adjusted_rand(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::techniques::ari::pairwise_adjusted_rand_impl(
        &columns(batch)?,
        kwargs,
    ))
}

/// A fresh k-hash, m-bit Bloom filter over a one-column batch: ⌈m/8⌉ bytes.
pub fn bloom_filter(batch: &RecordBatch, k: usize, m: usize) -> Result<Vec<u8>> {
    if k == 0 || m == 0 {
        return Err(Error::InvalidInput(format!(
            "bloom_filter requires k > 0 and m > 0, got k={k}, m={m}"
        )));
    }
    let cols = columns(batch)?;
    let [s] = cols.as_slice() else {
        return Err(Error::InvalidInput(format!(
            "bloom_filter takes one column, got {}",
            cols.len()
        )));
    };
    crate::techniques::bloomfilter::bloom_filter_impl(
        s,
        BloomFilterKwargs {
            bit_array_bytes: Vec::new(),
            k,
            m,
        },
    )
    .map_err(compute)
}

/// `col_name`, `ratio_all`, `ratio_non_null` per column, against the filter `bits`.
pub fn membership_ratio(
    batch: &RecordBatch,
    bits: &[u8],
    k: usize,
    m: usize,
) -> Result<RecordBatch> {
    if k == 0 || m == 0 {
        return Err(Error::InvalidInput(format!(
            "membership_ratio requires k > 0 and m > 0, got k={k}, m={m}"
        )));
    }
    if bits.len() != m.div_ceil(8) {
        return Err(Error::InvalidInput(format!(
            "bloom filter bit array has {} bytes but m={m} bits requires {} bytes",
            bits.len(),
            m.div_ceil(8)
        )));
    }
    let kwargs = MembershipKwargs {
        bit_array_bytes: bits.to_vec(),
        k,
        m,
    };
    table(crate::techniques::bloomfilter::membership_ratio_multi_impl(
        &columns(batch)?,
        &kwargs,
    ))
}

/// `qualified_name` ("{df_name}|{column}"), `minhash: list<uint32>` per column.
pub fn minhash(batch: &RecordBatch, df_name: &str, num_perm: usize) -> Result<RecordBatch> {
    let cols = columns(batch)?;
    let packed = StructChunked::from_series("frame".into(), batch.num_rows(), cols.iter())
        .map_err(compute)?
        .into_series();
    table(crate::techniques::minhash::minhash_impl(
        &[packed],
        &MinHashKwargs {
            df_name: df_name.to_owned(),
            num_perm,
        },
    ))
}

/// `col_a`, `col_b` per candidate pair, from a batch of (names, signatures) — by position.
pub fn lsh_candidates(
    signatures: &RecordBatch,
    num_bands: usize,
    rows_per_band: usize,
) -> Result<RecordBatch> {
    if num_bands == 0 || rows_per_band == 0 {
        return Err(Error::InvalidInput(format!(
            "lsh_candidates requires num_bands > 0 and rows_per_band > 0, got num_bands={num_bands}, rows_per_band={rows_per_band}"
        )));
    }
    let cols = columns(signatures)?;
    if cols.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "lsh_candidates takes (names, signatures), got {} columns",
            cols.len()
        )));
    }
    table(crate::techniques::minhash::lsh_candidates_impl(
        &cols,
        &LSHKwargs {
            num_bands,
            rows_per_band,
        },
    ))
}

/// `column`, Describe's value metrics and conclusions, then the private estimator
/// inputs (argmin / argmax, f1, f2, capture history and their `inner_` twins).
pub fn describe_columns(
    batch: &RecordBatch,
    seed: u64,
    categorical_threshold: u64,
) -> Result<RecordBatch> {
    table(crate::techniques::describe::describe_columns_impl(
        &columns(batch)?,
        seed,
        categorical_threshold,
    ))
}

/// `column`, `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes`.
pub fn column_sizes(batch: &RecordBatch, zstd_level: i32) -> Result<RecordBatch> {
    table(crate::common::ipc_sizes::column_sizes_impl(
        &columns(batch)?,
        zstd_level,
    ))
}

/// Every value of every column as text (`recommend::render_value`): the canonical
/// rendering of `min` / `max`, for the Python reference implementations. Meant for a
/// few values per column: each value is cast on its own.
pub fn render(batch: &RecordBatch) -> Result<RecordBatch> {
    let columns: Vec<arrow_array::ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            std::sync::Arc::new(arrow_array::StringArray::from_iter(
                (0..c.len()).map(|i| crate::common::text::render_value(c.slice(i, 1).as_ref())),
            )) as arrow_array::ArrayRef
        })
        .collect();
    let fields: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| arrow_schema::Field::new(f.name(), arrow_schema::DataType::Utf8, true))
        .collect();
    RecordBatch::try_new(
        std::sync::Arc::new(arrow_schema::Schema::new(fields)),
        columns,
    )
    .map_err(|e| Error::Compute(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal128Type, Float64Type};
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::DataType as AT;
    use polars::prelude::PolarsError;

    use super::*;

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
    }

    #[test]
    fn render_gives_text_and_nulls() {
        use arrow_array::Array;
        let out = render(&batch(vec![(
            "a",
            Arc::new(Int64Array::from(vec![Some(3), None])) as ArrayRef,
        )]))
        .unwrap();
        assert_eq!(out.schema().field(0).data_type(), &AT::Utf8);
        let c = out.column(0).as_string::<i32>();
        assert_eq!((c.value(0), c.is_valid(1)), ("3", false));
    }

    #[test]
    fn gcd_of_plain_arrow_columns() {
        let out = column_gcd(&batch(vec![
            ("a", ints(&[12, 18, 30])),
            ("b", ints(&[7, 14, 21])),
        ]))
        .unwrap();
        assert_eq!(out.schema().field(2).data_type(), &AT::Decimal128(38, 0));
        let gcd = out.column(2).as_primitive::<Decimal128Type>();
        assert_eq!((gcd.value(0), gcd.value(1)), (6, 7));
        assert_eq!(out.column(0).as_string_view().value(1), "b");
    }

    #[test]
    fn pairwise_entropy_output_columns() {
        let out = pairwise_joint_entropy(
            &batch(vec![("a", ints(&[1, 1, 2, 2])), ("b", ints(&[1, 2, 1, 2]))]),
            None,
        )
        .unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["col_a", "col_b", "entropy"]);
        assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 2.0);
    }

    #[test]
    fn unknown_pair_column_is_invalid_input() {
        let b = batch(vec![("a", ints(&[1, 2]))]);
        let err = pairwise_chi_squared(&b, Some(&[("a".into(), "zz".into())])).unwrap_err();
        assert_eq!(err, Error::InvalidInput("unknown column \"zz\"".into()));
        assert!(matches!(
            threeway_joint_entropy(&b, Some(&[("a".into(), "a".into(), "q".into())])),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn bloom_filter_then_membership() {
        let b = batch(vec![("a", ints(&[1, 2, 3]))]);
        let bits = bloom_filter(&b, 3, 64).unwrap();
        assert_eq!(bits.len(), 8);
        let out = membership_ratio(&b, &bits, 3, 64).unwrap();
        assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 1.0);
    }

    #[test]
    fn bloom_errors_are_invalid_input() {
        let b = batch(vec![("a", ints(&[1])), ("b", ints(&[2]))]);
        assert!(matches!(
            bloom_filter(&b, 3, 64),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            membership_ratio(&b, &[0u8; 3], 3, 64),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn minhash_then_lsh() {
        let sigs = minhash(
            &batch(vec![("a", ints(&[1, 2, 3])), ("b", ints(&[1, 2, 3]))]),
            "0",
            16,
        )
        .unwrap();
        assert_eq!(sigs.num_rows(), 2);
        assert_eq!(lsh_candidates(&sigs, 4, 4).unwrap().num_rows(), 1); // identical columns always collide
    }

    #[test]
    fn lsh_on_a_non_list_signature_is_invalid_input() {
        let bad = batch(vec![
            (
                "qualified_name",
                Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
            ),
            ("minhash", ints(&[1])),
        ]);
        assert!(matches!(
            lsh_candidates(&bad, 1, 1),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn compute_error_maps_to_compute_variant() {
        assert!(matches!(
            compute(PolarsError::ComputeError("x".into())),
            Error::Compute(_)
        ));
    }

    #[test]
    fn bloom_filter_rejects_zero_k_or_m() {
        let b = batch(vec![("a", ints(&[1, 2, 3]))]);
        assert!(matches!(
            bloom_filter(&b, 0, 64),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            bloom_filter(&b, 3, 0),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn membership_ratio_rejects_zero_k_or_m() {
        let b = batch(vec![("a", ints(&[1, 2, 3]))]);
        assert!(matches!(
            membership_ratio(&b, &[], 0, 64),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            membership_ratio(&b, &[], 3, 0),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn lsh_candidates_rejects_zero_bands_or_rows_per_band() {
        let sigs = minhash(
            &batch(vec![("a", ints(&[1, 2, 3])), ("b", ints(&[1, 2, 3]))]),
            "0",
            16,
        )
        .unwrap();
        assert!(matches!(
            lsh_candidates(&sigs, 0, 4),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            lsh_candidates(&sigs, 4, 0),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn duplicate_column_names_are_invalid_input() {
        let b = batch(vec![("a", ints(&[1, 2])), ("a", ints(&[3, 4]))]);
        let err = column_gcd(&b).unwrap_err();
        assert_eq!(err, Error::InvalidInput("duplicate column \"a\"".into()));
    }

    #[test]
    fn marginal_entropy_smoke() {
        let out = marginal_entropy(&batch(vec![("a", ints(&[1, 1, 2, 2]))])).unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["col_name", "entropy"]);
        assert_eq!(out.column(1).as_primitive::<Float64Type>().value(0), 1.0);
    }

    #[test]
    fn pairwise_chi_squared_smoke() {
        let out = pairwise_chi_squared(
            &batch(vec![("a", ints(&[0, 0, 1, 1])), ("b", ints(&[0, 0, 1, 1]))]),
            None,
        )
        .unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            [
                "col_a",
                "col_b",
                "chi2_stat",
                "p_value",
                "cramers_v",
                "low_expected_count",
                "n_valid"
            ]
        );
        assert_eq!(out.column(4).as_primitive::<Float64Type>().value(0), 1.0);
    }

    #[test]
    fn pairwise_adjusted_rand_smoke() {
        let out = pairwise_adjusted_rand(
            &batch(vec![("a", ints(&[0, 0, 1, 1])), ("b", ints(&[0, 0, 1, 1]))]),
            None,
        )
        .unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["col_a", "col_b", "ari", "n_valid"]);
        assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 1.0);
    }

    #[test]
    fn threeway_entropy_smoke() {
        let out = threeway_joint_entropy(
            &batch(vec![
                ("a", ints(&[1, 1, 2, 2])),
                ("b", ints(&[1, 2, 1, 2])),
                ("c", ints(&[1, 2, 1, 2])),
            ]),
            Some(&[("a".into(), "b".into(), "c".into())]),
        )
        .unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["col_a", "col_b", "col_c", "entropy"]);
        assert_eq!(out.column(3).as_primitive::<Float64Type>().value(0), 2.0);
    }

    #[test]
    fn describe_and_sizes_one_row_per_column() {
        let b = batch(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ]);
        assert_eq!(describe_columns(&b, 0, 10_000).unwrap().num_rows(), 2);
        assert_eq!(column_sizes(&b, 1).unwrap().num_rows(), 2);
    }
}
