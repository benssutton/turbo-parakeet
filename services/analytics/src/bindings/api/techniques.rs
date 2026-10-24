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

/// `column`, `n_rows`, `n_runs`, `expected_runs`, `z`, `p_value`, `mean_run_length` per
/// column (k-category runs test, rows in batch order; p one-sided, lower tail).
pub fn runs_test(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::techniques::runs_test::runs_test_impl(&columns(
        batch,
    )?))
}

/// `column` + the delta technique's metrics per column (rows in batch order).
pub fn delta(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::techniques::delta::delta_impl(&columns(batch)?))
}

/// `window_rows >= 1`, else InvalidInput.
fn window(window_rows: u64) -> Result<u64> {
    if window_rows == 0 {
        return Err(Error::InvalidInput("window_rows must be >= 1".into()));
    }
    Ok(window_rows)
}

/// `column` + MinMaxIndex's metrics per column: windows of `window_rows` rows sliding over the
/// batch's row order.
pub fn minmax_index(batch: &RecordBatch, window_rows: u64) -> Result<RecordBatch> {
    let w = window(window_rows)?;
    table(crate::techniques::skip_index::minmax::minmax_index_impl(
        &columns(batch)?,
        w,
    ))
}

/// `column` + SetIndex's metrics per column; `max_size` 0 = unlimited.
pub fn set_index(batch: &RecordBatch, window_rows: u64, max_size: u64) -> Result<RecordBatch> {
    let w = window(window_rows)?;
    table(crate::techniques::skip_index::set::set_index_impl(
        &columns(batch)?,
        w,
        max_size,
    ))
}

/// `col_name`, `n_non_null`, `estimate`, `std_error`, `sketch_bytes` per column: a
/// HyperLogLog of 2^precision registers; `precision` in 4..=18, else InvalidInput.
pub fn hll_distinct(batch: &RecordBatch, precision: u8) -> Result<RecordBatch> {
    if !(4..=18).contains(&precision) {
        return Err(Error::InvalidInput(format!(
            "precision must be in 4..=18, got {precision}"
        )));
    }
    table(crate::techniques::distinct_count::hll_distinct_impl(
        &columns(batch)?,
        precision,
    ))
}

/// `col_name` + PopulationCardinality's metrics per column; `seed` sets the 3-way split.
pub fn population_cardinality(batch: &RecordBatch, seed: u64) -> Result<RecordBatch> {
    table(
        crate::techniques::population_cardinality::population_cardinality_impl(
            &columns(batch)?,
            seed,
        ),
    )
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

/// `triplets` as kernel kwargs, every name checked.
fn threeway(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
) -> Result<ThreewayKwargs> {
    if let Some(t) = triplets {
        check_names(batch, t.iter().flat_map(|(a, b, c)| [a, b, c]))?;
    }
    Ok(ThreewayKwargs {
        triplets: triplets.map(|t| {
            t.iter()
                .map(|(a, b, c)| vec![a.clone(), b.clone(), c.clone()])
                .collect()
        }),
    })
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet; every triplet when None.
pub fn threeway_joint_entropy(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
) -> Result<RecordBatch> {
    let kwargs = threeway(batch, triplets)?;
    table(crate::techniques::joint_entropy::threeway_joint_entropy_impl(&columns(batch)?, kwargs))
}

use crate::techniques::ams_entropy::MAX_SAMPLES;

/// The AMS sketches' size: 1 ..= `MAX_SAMPLES` estimators (a larger one would abort on allocation).
fn ams_samples(num_samples: usize) -> Result<usize> {
    if num_samples == 0 || num_samples > MAX_SAMPLES {
        return Err(Error::InvalidInput(format!(
            "num_samples must be between 1 and {}, got {num_samples}",
            MAX_SAMPLES
        )));
    }
    Ok(num_samples)
}

/// `col_name`, `entropy` per column, estimated by the AMS (CCM) sketch.
pub fn ams_marginal_entropy(
    batch: &RecordBatch,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    table(crate::techniques::ams_entropy::ams_marginal_entropy_impl(
        &columns(batch)?,
        c,
        seed,
    ))
}

/// `col_a`, `col_b`, `entropy` per pair (every pair when None), estimated by the AMS sketch.
pub fn ams_pairwise_joint_entropy(
    batch: &RecordBatch,
    pairs: Option<&[(String, String)]>,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    let kwargs = pairwise(batch, pairs)?;
    table(
        crate::techniques::ams_joint_entropy::ams_pairwise_joint_entropy_impl(
            &columns(batch)?,
            kwargs,
            c,
            seed,
        ),
    )
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet (every triplet when None), estimated by
/// the AMS sketch.
pub fn ams_threeway_joint_entropy(
    batch: &RecordBatch,
    triplets: Option<&[(String, String, String)]>,
    num_samples: usize,
    seed: u64,
) -> Result<RecordBatch> {
    let c = ams_samples(num_samples)?;
    let kwargs = threeway(batch, triplets)?;
    table(
        crate::techniques::ams_joint_entropy::ams_threeway_joint_entropy_impl(
            &columns(batch)?,
            kwargs,
            c,
            seed,
        ),
    )
}

use crate::techniques::heavy_keeper::HeavyKeeper;
use crate::techniques::space_saving::SpaceSaving;

/// `col_name, rank, row, count, error, n_non_null, sketch_bytes`: each column's top `k` by
/// SpaceSaving in `memory_bytes` (spec 2026-10-12). k = 0 or a budget below k counters ->
/// InvalidInput.
pub fn space_saving_top_k(
    batch: &RecordBatch,
    k: usize,
    memory_bytes: usize,
) -> Result<RecordBatch> {
    SpaceSaving::new(k, memory_bytes).map_err(Error::InvalidInput)?;
    table(crate::techniques::space_saving::space_saving_top_k_impl(
        &columns(batch)?,
        k,
        memory_bytes,
    ))
}

/// As `space_saving_top_k`, by HeavyKeeper (`error` null); a budget leaving fewer than k
/// buckets per array -> InvalidInput.
pub fn heavy_keeper_top_k(
    batch: &RecordBatch,
    k: usize,
    memory_bytes: usize,
    seed: u64,
) -> Result<RecordBatch> {
    HeavyKeeper::new(k, memory_bytes, seed).map_err(Error::InvalidInput)?;
    table(crate::techniques::heavy_keeper::heavy_keeper_top_k_impl(
        &columns(batch)?,
        k,
        memory_bytes,
        seed,
    ))
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

/// `column`, `bytes`: each column's Arrow IPC body size under `codec` (`"none"`, `"lz4"`,
/// `"zstd:<level>"`). Private: tests and the calibration script.
pub fn ipc_body_sizes(batch: &RecordBatch, codec: &str) -> Result<RecordBatch> {
    use crate::common::ipc_sizes::{ipc_body_size, Codec};
    let codec = match codec {
        "none" => Codec::None,
        "lz4" => Codec::Lz4Frame,
        c => match c.strip_prefix("zstd:").and_then(|l| l.parse().ok()) {
            Some(level) => Codec::Zstd(level),
            None => return Err(Error::InvalidInput(format!("unknown codec {c:?}"))),
        },
    };
    let schema = batch.schema();
    let bytes = batch
        .columns()
        .iter()
        .map(|a| ipc_body_size(a.as_ref(), codec))
        .collect::<PolarsResult<Vec<u64>>>()
        .map_err(compute)?;
    let names = arrow_array::StringArray::from_iter_values(
        schema.fields().iter().map(|f| f.name().as_str()),
    );
    RecordBatch::try_from_iter([
        (
            "column",
            std::sync::Arc::new(names) as arrow_array::ArrayRef,
        ),
        (
            "bytes",
            std::sync::Arc::new(arrow_array::UInt64Array::from(bytes)) as arrow_array::ArrayRef,
        ),
    ])
    .map_err(|e| Error::Compute(e.to_string()))
}

/// The codec chain written in DDL: `[GCD|Delta|DoubleDelta][, ][LZ4|ZSTD(n)]`, a lone
/// transform, or "NONE".
fn parse_chain(codecs: &str) -> Result<crate::common::clickhouse_sizes::Chain> {
    use crate::common::clickhouse_sizes::{Chain, Compression, Transform};
    let bad = || Error::InvalidInput(format!("unparsable codecs {codecs:?}"));
    let codecs = codecs.trim();
    if codecs == "NONE" {
        return Ok(Chain {
            transform: Transform::None,
            compression: Compression::None,
        });
    }
    let (transform, rest) = [
        ("GCD", Transform::Gcd),
        ("DoubleDelta", Transform::DoubleDelta),
        ("Delta", Transform::Delta),
    ]
    .into_iter()
    .find_map(|(name, t)| {
        codecs
            .strip_prefix(name)
            .filter(|r| r.is_empty() || r.starts_with(','))
            .map(|r| (t, r.strip_prefix(',').unwrap_or(r).trim()))
    })
    .unwrap_or((Transform::None, codecs));
    let compression = match rest {
        "" if transform != Transform::None && codecs.find(',').is_none() => Compression::None,
        "LZ4" => Compression::Lz4,
        z => match z
            .strip_prefix("ZSTD(")
            .and_then(|l| l.strip_suffix(')'))
            .and_then(|l| l.trim().parse().ok())
        {
            Some(level) => Compression::Zstd(level),
            None => return Err(bad()),
        },
    };
    Ok(Chain {
        transform,
        compression,
    })
}

/// One row: `clickhouse_type` (the type of `column`'s own Arrow type, nullable when it has
/// nulls) and `bytes` (its emulated ClickHouse size under the chain `codecs`); both null
/// when the type has no ClickHouse type, `bytes` null also when GCD does not apply.
/// Private: tests and `to_chbench`.
pub fn clickhouse_profile(batch: &RecordBatch, column: &str, codecs: &str) -> Result<RecordBatch> {
    use crate::common::clickhouse_sizes::{emulate, streams, ChLayout, Transform};
    use crate::recommenders::engine::{clickhouse_type, physical_stats};
    let chain = parse_chain(codecs)?;
    let arr = batch
        .column_by_name(column)
        .ok_or_else(|| Error::InvalidInput(format!("unknown column {column:?}")))?;
    let range = physical_stats(arr.as_ref()).map(|s| s.range);
    let mut ty = None;
    let mut bytes = None;
    if let Ok((t, layout)) = clickhouse_type(arr.data_type(), range, arr.null_count() > 0, None) {
        let applies = matches!(
            &layout,
            ChLayout::Fixed {
                int_codecs: true,
                ..
            }
        ) || matches!(
            &layout,
            ChLayout::Nullable(inner) if matches!(**inner, ChLayout::Fixed { int_codecs: true, .. })
        );
        if chain.transform == Transform::None || applies {
            let st = streams(arr.as_ref(), &layout).map_err(Error::Compute)?;
            bytes = Some(emulate(&st, chain).map_err(Error::Compute)?.bytes);
        }
        ty = Some(t);
    }
    RecordBatch::try_from_iter([
        (
            "clickhouse_type",
            std::sync::Arc::new(arrow_array::StringArray::from(vec![ty])) as arrow_array::ArrayRef,
        ),
        (
            "bytes",
            std::sync::Arc::new(arrow_array::UInt64Array::from(vec![bytes]))
                as arrow_array::ArrayRef,
        ),
    ])
    .map_err(|e| Error::Compute(e.to_string()))
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

    #[test]
    fn chain_parser() {
        use crate::common::clickhouse_sizes::{Compression, Transform};
        let c = parse_chain("GCD, ZSTD(3)").unwrap();
        assert!(c.transform == Transform::Gcd && c.compression == Compression::Zstd(3));
        let c = parse_chain("LZ4").unwrap();
        assert!(c.transform == Transform::None && c.compression == Compression::Lz4);
        for ok in [
            "GCD, LZ4",
            "Delta, LZ4",
            "DoubleDelta, ZSTD(1)",
            "GCD",
            "Delta",
            "DoubleDelta",
            "NONE",
            "ZSTD(3)",
        ] {
            assert_eq!(parse_chain(ok).unwrap().ddl(), ok);
        }
        for bad in ["", "ZSTD", "ZSTD(x)", "Delta,", "Deltas", "Gorilla, LZ4"] {
            assert!(parse_chain(bad).is_err(), "{bad}");
        }
    }

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

    #[test]
    fn ams_entropy_smoke() {
        let b = batch(vec![
            ("a", ints(&[1, 2, 3, 4])),
            ("b", ints(&[5, 5, 5, 5])),
            ("c", ints(&[1, 1, 2, 2])),
        ]);
        let m = ams_marginal_entropy(&b, 64, 0).unwrap();
        assert_eq!(m.column(1).as_primitive::<Float64Type>().value(0), 2.0); // unique
        assert_eq!(m.column(1).as_primitive::<Float64Type>().value(1), 0.0); // constant
        let pairs = [("a".to_string(), "b".to_string())];
        let p = ams_pairwise_joint_entropy(&b, Some(&pairs[..]), 64, 0).unwrap();
        assert_eq!(p.num_rows(), 1);
        assert_eq!(p.column(2).as_primitive::<Float64Type>().value(0), 2.0);
        let t = ams_threeway_joint_entropy(&b, None, 64, 0).unwrap();
        assert_eq!(t.column(3).as_primitive::<Float64Type>().value(0), 2.0);
    }

    #[test]
    fn top_k_smoke() {
        use arrow_array::types::{UInt32Type, UInt64Type};
        let b = batch(vec![(
            "a",
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), None])) as ArrayRef,
        )]);
        for out in [
            space_saving_top_k(&b, 5, 16 * 1024).unwrap(),
            heavy_keeper_top_k(&b, 5, 16 * 1024, 0).unwrap(),
        ] {
            let names: Vec<String> = out
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert_eq!(
                names,
                [
                    "col_name",
                    "rank",
                    "row",
                    "count",
                    "error",
                    "n_non_null",
                    "sketch_bytes"
                ]
            );
            assert_eq!(out.num_rows(), 2);
            assert_eq!(out.column(1).as_primitive::<UInt32Type>().value(0), 1);
            assert_eq!(out.column(3).as_primitive::<UInt64Type>().value(0), 2);
            assert_eq!(out.column(5).as_primitive::<UInt64Type>().value(0), 3);
        }
    }

    #[test]
    fn top_k_parameters_are_checked() {
        let b = batch(vec![("a", ints(&[1, 2]))]);
        for r in [
            space_saving_top_k(&b, 0, 16 * 1024),
            space_saving_top_k(&b, 5, 64),
            heavy_keeper_top_k(&b, 0, 16 * 1024, 0),
            heavy_keeper_top_k(&b, 5, 64, 0),
        ] {
            assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
        }
    }

    #[test]
    fn ams_sample_count_is_capped() {
        let b = batch(vec![("a", ints(&[1, 2]))]);
        assert!(matches!(
            ams_marginal_entropy(&b, MAX_SAMPLES + 1, 0),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn ams_zero_samples_and_unknown_names_are_invalid_input() {
        let b = batch(vec![("a", ints(&[1, 2])), ("b", ints(&[1, 2]))]);
        assert!(matches!(
            ams_marginal_entropy(&b, 0, 0),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            ams_pairwise_joint_entropy(&b, None, 0, 0),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            ams_threeway_joint_entropy(&b, None, 0, 0),
            Err(Error::InvalidInput(_))
        ));
        let pairs = [("a".to_string(), "zz".to_string())];
        assert!(matches!(
            ams_pairwise_joint_entropy(&b, Some(&pairs[..]), 8, 0),
            Err(Error::InvalidInput(_))
        ));
    }
}
