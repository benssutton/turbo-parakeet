//! The language-neutral core: every entry point takes an Arrow RecordBatch plus
//! plain parameters and returns an Arrow RecordBatch (Bloom: bytes). No pyo3 and no
//! Polars type appears in any signature — bindings (python.rs; later Java / C) wrap
//! exactly this module. Kernels compute on Polars Series behind arrow_io.
//!
//! Input batches must be valid Arrow: the entry points do not re-check them (an O(n)
//! pass over every value). A RecordBatch built with arrow-rs's safe constructors is
//! valid by construction; one imported through the C Data Interface is not, so callers
//! import with `arrow_io::read_stream` / `CheckedReader` (python.rs and capi.rs do), or
//! check with `arrow_io::validate_batch`, before calling in — arrow-rs and Polars may
//! panic or read out of bounds on a malformed batch.

use std::collections::HashSet;
use std::fmt;

use arrow_array::RecordBatch;
use polars::prelude::{IntoSeries, PolarsError, PolarsResult, Series, StructChunked};

use crate::arrow_io::{export_struct, import_batch};
use crate::bloomfilter::{BloomFilterKwargs, MembershipKwargs};
use crate::minhash::{LSHKwargs, MinHashKwargs};
use crate::recommend::Params;
use crate::shared::{PairwiseKwargs, ThreewayKwargs};

#[derive(Debug, PartialEq)]
pub enum Error {
    /// The caller's input: unknown or duplicate column names, a malformed Bloom
    /// array or invalid Bloom/LSH parameters, an Arrow type no kernel accepts, or
    /// a column of the wrong type for the kernel.
    InvalidInput(String),
    /// Any other failure inside a kernel.
    Compute(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) | Error::Compute(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn compute(e: PolarsError) -> Error {
    let msg = e.to_string();
    match e {
        PolarsError::ColumnNotFound(_)
        | PolarsError::SchemaMismatch(_)
        | PolarsError::InvalidOperation(_)
        | PolarsError::ShapeMismatch(_) => Error::InvalidInput(msg),
        _ => Error::Compute(msg),
    }
}

/// `batch`'s columns as Series. `batch` must be valid Arrow (see the module docs).
fn columns(batch: &RecordBatch) -> Result<Vec<Series>> {
    let schema = batch.schema();
    let mut seen: HashSet<&str> = HashSet::new();
    for f in schema.fields().iter() {
        if !seen.insert(f.name().as_str()) {
            return Err(Error::InvalidInput(format!(
                "duplicate column {:?}",
                f.name()
            )));
        }
    }
    import_batch(batch).map_err(|e| Error::InvalidInput(e.to_string()))
}

fn table(out: PolarsResult<Series>) -> Result<RecordBatch> {
    out.and_then(|s| export_struct(&s)).map_err(compute)
}

fn check_names<'a>(batch: &RecordBatch, names: impl IntoIterator<Item = &'a String>) -> Result<()> {
    let schema = batch.schema();
    let known: HashSet<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    match names.into_iter().find(|n| !known.contains(n.as_str())) {
        Some(n) => Err(Error::InvalidInput(format!("unknown column {n:?}"))),
        None => Ok(()),
    }
}

fn pairwise(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<PairwiseKwargs> {
    let Some(pairs) = pairs else {
        return Ok(PairwiseKwargs { pairs: None });
    };
    check_names(batch, pairs.iter().flat_map(|(a, b)| [a, b]))?;
    Ok(PairwiseKwargs {
        pairs: Some(
            pairs
                .iter()
                .map(|(a, b)| vec![a.clone(), b.clone()])
                .collect(),
        ),
    })
}

/// `column`, `dtype`, `gcd: decimal128(38, 0)` per column.
pub fn column_gcd(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::gcd::column_gcd_impl(&columns(batch)?))
}

/// `col_name`, `entropy` per column.
pub fn marginal_entropy(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::entropy::marginal_entropy_impl(&columns(batch)?))
}

/// `col_a`, `col_b`, `entropy` per pair; every pair when `pairs` is None.
pub fn pairwise_joint_entropy(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::entropy::pairwise_joint_entropy_impl(
        &columns(batch)?,
        kwargs,
    ))
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
    table(crate::entropy::threeway_joint_entropy_impl(
        &columns(batch)?,
        kwargs,
    ))
}

/// `col_a`, `col_b`, `chi2_stat`, `p_value`, `cramers_v`, `low_expected_count`, `n_valid`.
pub fn pairwise_chi_squared(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::chi_squared::pairwise_chi_squared_impl(
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
    table(crate::ari::pairwise_adjusted_rand_impl(
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
    crate::bloomfilter::bloom_filter_impl(
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
    table(crate::bloomfilter::membership_ratio_multi_impl(
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
    table(crate::minhash::minhash_impl(
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
    table(crate::minhash::lsh_candidates_impl(
        &cols,
        &LSHKwargs {
            num_bands,
            rows_per_band,
        },
    ))
}

/// `column` plus Describe's value metrics per column.
pub fn describe_columns(batch: &RecordBatch, seed: u64) -> Result<RecordBatch> {
    table(crate::describe::describe_columns_impl(
        &columns(batch)?,
        seed,
    ))
}

/// `column`, `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes`.
pub fn column_sizes(batch: &RecordBatch, zstd_level: i32) -> Result<RecordBatch> {
    table(crate::sizes::column_sizes_impl(
        &columns(batch)?,
        zstd_level,
    ))
}

/// Describe's table, the size columns and the `rec_*` columns per column.
pub fn describe_and_recommend(
    batch: &RecordBatch,
    seed: u64,
    zstd_level: i32,
    categorical_threshold: u64,
    boolean_pairs: Vec<(String, String)>,
) -> Result<RecordBatch> {
    let params = Params {
        seed,
        zstd_level,
        categorical_threshold,
        boolean_pairs,
    };
    table(crate::recommend::describe_and_recommend_impl(
        &columns(batch)?,
        &params,
    ))
}

/// Keywords of the streaming recommender (spec 2026-09-29 §7.1).
#[derive(Clone, Debug)]
pub struct StreamingParams {
    /// Rows of contiguous blocks kept for ZSTD sizes and the cross-check (0: none).
    pub reservoir_rows: u64,
    pub block_rows: u64,
    pub categorical_threshold: u64,
    pub zstd_level: i32,
    pub seed: u64,
    pub boolean_pairs: Vec<(String, String)>,
}

/// Recommends dtypes from record batches added over time; all state stays in Rust.
pub struct StreamingRecommender(crate::streaming::Streaming);

impl StreamingRecommender {
    pub fn new(p: StreamingParams) -> Result<Self> {
        if p.block_rows == 0 {
            return Err(Error::InvalidInput("block_rows must be at least 1".into()));
        }
        if p.reservoir_rows != 0 && p.reservoir_rows < p.block_rows {
            return Err(Error::InvalidInput(format!(
                "reservoir_rows {} is below block_rows {}: use 0 (no sample) or at least one block",
                p.reservoir_rows, p.block_rows
            )));
        }
        let levels = zstd::compression_level_range();
        if !levels.contains(&p.zstd_level) {
            return Err(Error::InvalidInput(format!(
                "zstd_level {} is outside {levels:?}",
                p.zstd_level
            )));
        }
        if let Some((t, f)) = p
            .boolean_pairs
            .iter()
            .find(|(t, f)| t.is_empty() || f.is_empty() || t.to_lowercase() == f.to_lowercase())
        {
            return Err(Error::InvalidInput(format!(
                "boolean_pairs must be pairs of distinct non-empty strings, got ({t:?}, {f:?})"
            )));
        }
        let params = Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
        };
        Ok(Self(crate::streaming::Streaming::new(
            params,
            p.reservoir_rows,
            p.block_rows,
        )))
    }

    /// Adds one batch. On error the state is unchanged.
    pub fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        self.0.add(batch)
    }

    /// Marks a column the caller cannot send (Int128 / UInt128, Object) as ineligible.
    pub fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        self.0.mark_ineligible(name, dtype)
    }

    /// The recommendation for every column seen so far; the state is kept.
    pub fn finish(&self) -> Result<RecordBatch> {
        self.0.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal128Type, Float64Type};
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::DataType as AT;

    use super::*;

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
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
    fn describe_sizes_and_recommend_one_row_per_column() {
        let b = batch(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ]);
        assert_eq!(describe_columns(&b, 0).unwrap().num_rows(), 2);
        assert_eq!(column_sizes(&b, 1).unwrap().num_rows(), 2);
        let rec = describe_and_recommend(
            &b,
            0,
            1,
            10_000,
            vec![("true".into(), "false".into())],
        )
        .unwrap();
        assert_eq!(
            rec.column_by_name("rec_arrow_type")
                .unwrap()
                .as_string_view()
                .value(0),
            "uint8"
        );
    }

    fn streaming_params() -> StreamingParams {
        StreamingParams {
            reservoir_rows: 1 << 20,
            block_rows: 1 << 16,
            categorical_threshold: 10_000,
            zstd_level: 1,
            seed: 0,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    #[test]
    fn streaming_parameters_are_validated() {
        let bad = [
            StreamingParams {
                block_rows: 0,
                ..streaming_params()
            },
            StreamingParams {
                reservoir_rows: 10,
                block_rows: 100,
                ..streaming_params()
            },
            StreamingParams {
                zstd_level: 99,
                ..streaming_params()
            },
            StreamingParams {
                boolean_pairs: vec![("Y".into(), "y".into())],
                ..streaming_params()
            },
            StreamingParams {
                boolean_pairs: vec![("".into(), "n".into())],
                ..streaming_params()
            },
        ];
        for p in bad {
            assert!(
                matches!(
                    StreamingRecommender::new(p.clone()),
                    Err(Error::InvalidInput(_))
                ),
                "{p:?}"
            );
        }
        assert!(StreamingRecommender::new(StreamingParams {
            reservoir_rows: 0,
            ..streaming_params()
        })
        .is_ok());
    }

    #[test]
    fn streaming_round_trip() {
        let batch = RecordBatch::try_from_iter(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ])
        .unwrap();
        let mut rec = StreamingRecommender::new(streaming_params()).unwrap();
        rec.add(&batch).unwrap();
        rec.add(&batch).unwrap();
        let out = rec.finish().unwrap();
        assert_eq!(out.num_rows(), 2);
        let types = out
            .column_by_name("rec_arrow_type")
            .unwrap()
            .as_string_view();
        assert_eq!(types.value(0), "uint8");
        let n_rows = out.column_by_name("n_rows").unwrap();
        assert_eq!(
            n_rows
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            6
        );
    }
}
