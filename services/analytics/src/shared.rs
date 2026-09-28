// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities — used by entropy.rs, chi_squared.rs, minhash.rs, bloomfilter.rs
// ─────────────────────────────────────────────────────────────────────────────

use foldhash::fast::FixedState as FoldHashFixed;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};

// ─────────────────────────────────────────────────────────────────────────────
// Kwargs structs
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct PairwiseKwargs {
    pub pairs: Option<Vec<Vec<String>>>,
}

pub(crate) struct ThreewayKwargs {
    pub triplets: Option<Vec<Vec<String>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Null representation
// ─────────────────────────────────────────────────────────────────────────────
//
// Nulls are tracked OUT OF BAND via a parallel validity mask, never with an
// in-band sentinel value. This is deliberate and load-bearing: every consumer
// works over the full u64 range — `-1i64` encodes to `u64::MAX`, and `u64::MAX`
// is itself a legal value — so no in-band sentinel can be reserved without
// aliasing real data. (The previous `NULL_SENTINEL = u64::MAX` scheme silently
// treated every `-1` and every `u64::MAX` as null, corrupting counts on real
// data while passing synthetic tests that never used those values.)
//
// Each consumer applies its own documented null policy on top of the mask:
//   - entropy:       null is a distinct category (the mask is folded into the
//                    group key, so (null, x), (x, null), (null, null) are all
//                    distinct joint keys).
//   - chi-squared:   null rows are dropped (pairwise deletion — the standard
//                    treatment for contingency tables).
//   - minhash/bloom: null rows are skipped (a null is not a set member).

/// A column encoded to `u64` keys plus a parallel null mask.
///
/// `values[i]` is only meaningful where `is_null[i]` is false; null positions
/// are canonicalised to `0` so a leftover payload can never alias a real value.
pub(crate) struct EncodedColumn {
    pub values: Vec<u64>,
    pub is_null: Vec<bool>,
}

impl EncodedColumn {
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Canonicalise an `f64` before taking its bit pattern:
///   - `+0.0` and `-0.0` collapse to the same key (they compare equal),
///   - every NaN (regardless of sign/payload) collapses to one key.
/// This matches how Polars `value_counts` groups floats and keeps the Rust
/// entropy/chi-squared consistent with a Polars reference on real float data.
#[inline]
fn canon_f64(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

#[inline]
fn canon_f32(x: f32) -> u64 {
    if x == 0.0 {
        0
    } else if x.is_nan() {
        f32::NAN.to_bits() as u64
    } else {
        x.to_bits() as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Null-safe column conversion (values + out-of-band null mask)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn encode_series(series: &Series) -> PolarsResult<EncodedColumn> {
    let (values, is_null): (Vec<u64>, Vec<bool>) = match series.dtype() {
        DataType::Int8 => series
            .i8()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Int16 => series
            .i16()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Int32 => series
            .i32()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Int64 => series
            .i64()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::UInt8 => series
            .u8()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::UInt16 => series
            .u16()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::UInt32 => series
            .u32()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::UInt64 => series
            .u64()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x, false)))
            .unzip(),
        DataType::Boolean => series
            .bool()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Float32 => series
            .f32()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (canon_f32(x), false)))
            .unzip(),
        DataType::Float64 => series
            .f64()?
            .iter()
            .map(|v| v.map_or((0, true), |x| (canon_f64(x), false)))
            .unzip(),
        DataType::Date => series
            .date()?
            .phys
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Datetime(_, _) => series
            .datetime()?
            .phys
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Duration(_) => series
            .duration()?
            .phys
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::Time => series
            .time()?
            .phys
            .iter()
            .map(|v| v.map_or((0, true), |x| (x as u64, false)))
            .unzip(),
        DataType::String => {
            let build_hasher = FoldHashFixed::default();
            series
                .str()?
                .iter()
                .map(|v| v.map_or((0, true), |s| (hash_one(&build_hasher, s), false)))
                .unzip()
        }
        // Resolve categories to their string values so a categorical "x" encodes
        // identically to the string "x" — and identically across frames, whatever
        // per-Series physical code each frame assigned. Casting to String is
        // version-stable across the Polars 0.51 generic-categorical rework and
        // preserves nulls; from here the logic is identical to the String arm.
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let str_series = series.cast(&DataType::String)?;
            str_series
                .str()?
                .iter()
                .map(|v| v.map_or((0, true), |s| (hash_one(&build_hasher, s), false)))
                .unzip()
        }
        // Decimal is physically i128 with a fixed scale per-Series; hash the raw integer.
        DataType::Decimal(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let phys = series.to_physical_repr();
            phys.i128()?
                .iter()
                .map(|v| v.map_or((0, true), |x| (hash_one(&build_hasher, x), false)))
                .unzip()
        }
        // List and Array: hash the ordered element sequence so [1,2] != [2,1].
        DataType::List(_) => {
            let build_hasher = FoldHashFixed::default();
            let list_ca = series.list()?;
            let mut values = Vec::with_capacity(list_ca.len());
            let mut is_null = Vec::with_capacity(list_ca.len());
            for opt_inner in list_ca.amortized_iter() {
                match opt_inner {
                    None => {
                        values.push(0);
                        is_null.push(true);
                    }
                    Some(inner) => {
                        values.push(hash_nested(&build_hasher, inner.as_ref())?);
                        is_null.push(false);
                    }
                }
            }
            (values, is_null)
        }
        DataType::Array(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let arr_ca = series.array()?;
            let mut values = Vec::with_capacity(arr_ca.len());
            let mut is_null = Vec::with_capacity(arr_ca.len());
            for opt_inner in arr_ca.amortized_iter() {
                match opt_inner {
                    None => {
                        values.push(0);
                        is_null.push(true);
                    }
                    Some(inner) => {
                        values.push(hash_nested(&build_hasher, inner.as_ref())?);
                        is_null.push(false);
                    }
                }
            }
            (values, is_null)
        }
        DataType::Binary => {
            let build_hasher = FoldHashFixed::default();
            series
                .binary()?
                .iter()
                .map(|v| v.map_or((0, true), |b| (hash_one(&build_hasher, b), false)))
                .unzip()
        }
        // Struct: the whole value is one key — a hash of every field's key and
        // null-ness — so equal structs match and a null struct never aliases a
        // struct whose fields are all null (the outer validity is checked first).
        DataType::Struct(_) => {
            let build_hasher = FoldHashFixed::default();
            let ca = series.struct_()?;
            let fields = ca
                .fields_as_series()
                .iter()
                .map(encode_series)
                .collect::<PolarsResult<Vec<_>>>()?;
            let validity = ca.rechunk_validity();
            (0..ca.len())
                .map(|i| {
                    if validity.as_ref().is_some_and(|bm| !bm.get_bit(i)) {
                        return (0, true);
                    }
                    let mut h = build_hasher.build_hasher();
                    fields.len().hash(&mut h);
                    for f in &fields {
                        f.is_null[i].hash(&mut h);
                        f.values[i].hash(&mut h);
                    }
                    (h.finish(), false)
                })
                .unzip()
        }
        _ => {
            return Err(PolarsError::ComputeError(
                format!("Unsupported data type: {:?}.", series.dtype()).into(),
            ));
        }
    };

    Ok(EncodedColumn { values, is_null })
}

/// Hash a single `Hash`-able value with the fixed-seed hasher.
#[inline]
fn hash_one<T: Hash>(build_hasher: &FoldHashFixed, value: T) -> u64 {
    let mut hasher = build_hasher.build_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Hash the ordered element sequence of a nested (List/Array) inner series,
/// folding in each element's null-ness so nulls never alias real elements.
fn hash_nested(build_hasher: &FoldHashFixed, inner: &Series) -> PolarsResult<u64> {
    let enc = encode_series(inner)?;
    let mut h = build_hasher.build_hasher();
    enc.len().hash(&mut h);
    for (v, n) in enc.values.iter().zip(enc.is_null.iter()) {
        n.hash(&mut h);
        v.hash(&mut h);
    }
    Ok(h.finish())
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense re-encoding (dictionary encoding to 0..card ids)
// ─────────────────────────────────────────────────────────────────────────────

/// A column dictionary-encoded to dense ids `0..card`.
///
/// Nulls are folded in as their own id (consistent with the entropy null
/// policy: null is a distinct category), so consumers need no separate null
/// mask. Dense ids are only meaningful *within* one column of one call —
/// they carry no cross-column or cross-frame identity, which is exactly why
/// they are fine for entropy/chi² (invariant under relabelling) and wrong
/// for bloom/minhash (which compare actual values across columns/frames).
pub(crate) struct DenseColumn {
    pub ids: Vec<u32>,
    pub card: u32,
    /// Dense id assigned to nulls, if the column has any. Consumers with a
    /// drop-null policy (contingency-based stats) skip rows carrying this id;
    /// consumers treating null as a category (entropy) ignore it.
    pub null_id: Option<u32>,
}

/// Dictionary-encode an `EncodedColumn` into dense ids.
pub(crate) fn densify(col: &EncodedColumn) -> DenseColumn {
    let mut map: HashMap<(u64, bool), u32, FoldHashFixed> =
        HashMap::with_capacity_and_hasher(col.len(), FoldHashFixed::default());
    let mut ids = Vec::with_capacity(col.len());
    let mut null_id: Option<u32> = None;
    for i in 0..col.len() {
        let next = map.len() as u32;
        let id = *map.entry((col.values[i], col.is_null[i])).or_insert(next);
        if col.is_null[i] && null_id.is_none() {
            null_id = Some(id);
        }
        ids.push(id);
    }
    DenseColumn {
        ids,
        card: map.len() as u32,
        null_id,
    }
}

/// Encode + densify the needed columns in parallel.
pub(crate) fn build_dense_cache_par(
    inputs: &[Series],
    needed: &HashSet<usize>,
) -> PolarsResult<Vec<DenseColumn>> {
    let tasks: Vec<(usize, &Series)> = inputs
        .iter()
        .enumerate()
        .filter(|(idx, _)| needed.contains(idx))
        .collect();

    let converted: Vec<(usize, DenseColumn)> = tasks
        .into_par_iter()
        .map(|(idx, series)| {
            let enc = encode_series(series)?;
            Ok((idx, densify(&enc)))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut cache: Vec<DenseColumn> = (0..inputs.len())
        .map(|_| DenseColumn {
            ids: Vec::new(),
            card: 0,
            null_id: None,
        })
        .collect();
    for (idx, data) in converted {
        cache[idx] = data;
    }
    Ok(cache)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel column cache
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn build_column_cache_par(
    inputs: &[Series],
    needed: &HashSet<usize>,
) -> PolarsResult<Vec<EncodedColumn>> {
    let tasks: Vec<(usize, &Series)> = inputs
        .iter()
        .enumerate()
        .filter(|(idx, _)| needed.contains(idx))
        .collect();

    let converted: Vec<(usize, EncodedColumn)> = tasks
        .into_par_iter()
        .map(|(idx, series)| {
            let data = encode_series(series)?;
            Ok((idx, data))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut cache: Vec<EncodedColumn> = (0..inputs.len())
        .map(|_| EncodedColumn {
            values: Vec::new(),
            is_null: Vec::new(),
        })
        .collect();
    for (idx, data) in converted {
        cache[idx] = data;
    }
    Ok(cache)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair / triplet resolution from kwargs
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn resolve_pairs(
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

pub(crate) fn resolve_triplets(
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

#[cfg(test)]
mod tests {
    use super::*;
    use polars_arrow::bitmap::Bitmap;

    // ── Float canonicalization ─────────────────────────────────────────────

    #[test]
    fn positive_and_negative_zero_canonicalise_to_same_key() {
        assert_eq!(canon_f64(0.0), canon_f64(-0.0));
        assert_eq!(canon_f32(0.0), canon_f32(-0.0));
    }

    #[test]
    fn every_nan_payload_canonicalises_to_one_key() {
        let a = f64::NAN;
        let b = -f64::NAN;
        let c = f64::from_bits(0x7ff8000000000001); // a different NaN payload
        assert_eq!(canon_f64(a), canon_f64(b));
        assert_eq!(canon_f64(a), canon_f64(c));

        let fa = f32::NAN;
        let fb = f32::from_bits(0x7fc00001);
        assert_eq!(canon_f32(fa), canon_f32(fb));
    }

    #[test]
    fn distinct_finite_values_get_distinct_keys() {
        assert_ne!(canon_f64(1.0), canon_f64(2.0));
        assert_ne!(canon_f32(1.0), canon_f32(2.0));
    }

    #[test]
    fn infinities_are_not_swept_into_zero_or_nan() {
        assert_ne!(canon_f64(f64::INFINITY), canon_f64(0.0));
        assert_ne!(canon_f64(f64::NEG_INFINITY), canon_f64(0.0));
        assert_ne!(canon_f64(f64::INFINITY), canon_f64(f64::NAN));
        assert_ne!(canon_f64(f64::INFINITY), canon_f64(f64::NEG_INFINITY));
    }

    // ── encode_series: arms with no coverage elsewhere ─────────────────────

    #[test]
    fn negative_signed_integer_sign_extends_to_u64_max() {
        let s = Series::new("a".into(), &[-1i32]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], u64::MAX);
    }

    #[test]
    fn uint64_passes_through_unchanged() {
        let s = Series::new("a".into(), &[Some(42u64), None]);
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], 42u64);
        assert!(enc.is_null[1]);
    }

    #[test]
    fn binary_hashes_by_content() {
        let s = Series::new(
            "a".into(),
            &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), Some(b"cd".as_ref()), None],
        );
        let enc = encode_series(&s).unwrap();
        assert_eq!(enc.values[0], enc.values[1]);
        assert_ne!(enc.values[0], enc.values[2]);
        assert!(enc.is_null[3]);
    }

    #[test]
    fn decimal_hashes_unscaled_physical_value_not_rendered_value() {
        // Same unscaled i128 (100) declared at different scales (2 vs 0, so they'd
        // render as different values, 1.00 vs 100) built directly via AnyValue::Decimal
        // (unscaled, scale) → encode_series hashes to_physical_repr()'s raw i128, so
        // these must hash equal regardless of the differing scale metadata.
        let a = Series::from_any_values("a".into(), &[AnyValue::Decimal(100, 2)], false).unwrap();
        let b = Series::from_any_values("b".into(), &[AnyValue::Decimal(100, 0)], false).unwrap();
        let ea = encode_series(&a).unwrap();
        let eb = encode_series(&b).unwrap();
        assert_eq!(ea.values[0], eb.values[0]);
    }

    #[test]
    fn null_struct_row_does_not_alias_all_null_fields_struct() {
        let a = Series::new("a".into(), &[Some(1i32), None, Some(1)]);
        let b = Series::new("b".into(), &[Some("x"), None, Some("x")]);
        let all_null_fields = StructChunked::from_series("s".into(), 3, [a, b].iter())
            .unwrap()
            .into_series();

        // Same shape, but row 1's outer validity is explicitly unset: the struct
        // itself is null there (not merely holding all-null fields).
        let a2 = Series::new("a".into(), &[Some(1i32), None, Some(1)]);
        let b2 = Series::new("b".into(), &[Some("x"), None, Some("x")]);
        let ca = StructChunked::from_series("s".into(), 3, [a2, b2].iter()).unwrap();
        let outer_validity: Bitmap = vec![true, false, true].into_iter().collect();
        let outer_null = ca.with_outer_validity(Some(outer_validity)).into_series();

        let enc_all_null_fields = encode_series(&all_null_fields).unwrap();
        let enc_outer_null = encode_series(&outer_null).unwrap();

        // Row 1: all-null fields, struct itself present → not null.
        assert!(!enc_all_null_fields.is_null[1]);
        // Row 1 with outer validity unset → the struct itself is null.
        assert!(enc_outer_null.is_null[1]);
    }

    #[test]
    fn unsupported_dtype_is_a_compute_error() {
        let s = Series::new_empty("a".into(), &DataType::Null);
        assert!(encode_series(&s).is_err());
    }

    // ── densify ──────────────────────────────────────────────────────────

    #[test]
    fn densify_all_distinct_gets_sequential_ids_in_encounter_order() {
        let s = Series::new("a".into(), &[30i32, 10, 20]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.card, 3);
        assert_eq!(dense.ids, vec![0, 1, 2]); // encounter order, not sorted by value
    }

    #[test]
    fn densify_repeated_values_get_id_of_first_occurrence() {
        let s = Series::new("a".into(), &[5i32, 9, 5, 9, 5]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.card, 2);
        assert_eq!(dense.ids, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn densify_empty_column_has_zero_cardinality() {
        let s = Series::new_empty("a".into(), &DataType::Int32);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.card, 0);
        assert!(dense.ids.is_empty());
        assert_eq!(dense.null_id, None);
    }

    // ── build_dense_cache_par / build_column_cache_par ──────────────────────

    #[test]
    fn build_dense_cache_par_only_processes_needed_indices() {
        let cols = vec![
            Series::new("a".into(), &[1i32, 2, 1]),
            Series::new("b".into(), &[9i32, 9, 9]),
            Series::new("c".into(), &[1i32, 1, 2]),
        ];
        let needed: HashSet<usize> = [0, 2].into_iter().collect();
        let cache = build_dense_cache_par(&cols, &needed).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].card, 2); // processed
        assert_eq!(cache[2].card, 2); // processed
        // index 1 untouched → placeholder
        assert_eq!(cache[1].card, 0);
        assert!(cache[1].ids.is_empty());
        assert_eq!(cache[1].null_id, None);
    }

    #[test]
    fn build_dense_cache_par_preserves_original_index_order_under_parallelism() {
        // Each column gets a distinct expected cardinality (i+1) so a scatter-back
        // bug (results landing at the wrong original index under rayon) would
        // very likely produce a mismatch somewhere in the 32 columns.
        let cols: Vec<Series> = (0..32)
            .map(|i| {
                let vals: Vec<i32> = (0..(i + 1)).collect();
                Series::new(format!("c{i}").into(), &vals)
            })
            .collect();
        let needed: HashSet<usize> = (0..32).collect();
        let cache = build_dense_cache_par(&cols, &needed).unwrap();
        for i in 0..32 {
            assert_eq!(cache[i].card as usize, i + 1, "column {i} should have cardinality {}", i + 1);
        }
    }

    #[test]
    fn build_dense_cache_par_propagates_encode_error() {
        let cols = vec![
            Series::new("a".into(), &[1i32]),
            Series::new_empty("bad".into(), &DataType::Null),
        ];
        let needed: HashSet<usize> = [0, 1].into_iter().collect();
        assert!(build_dense_cache_par(&cols, &needed).is_err());
    }

    #[test]
    fn build_column_cache_par_only_processes_needed_indices() {
        let cols = vec![
            Series::new("a".into(), &[Some(1i32), None]),
            Series::new("b".into(), &[7i32, 7]),
        ];
        let needed: HashSet<usize> = [0].into_iter().collect();
        let cache = build_column_cache_par(&cols, &needed).unwrap();
        assert_eq!(cache[0].values.len(), 2);
        assert!(cache[0].is_null[1]);
        // index 1 untouched → placeholder empty EncodedColumn
        assert!(cache[1].values.is_empty());
        assert!(cache[1].is_null.is_empty());
    }

    // ── resolve_pairs / resolve_triplets ────────────────────────────────────

    #[test]
    fn resolve_pairs_maps_names_to_indices_preserving_order() {
        let name_map: HashMap<String, usize> =
            [("a".to_string(), 0), ("b".to_string(), 1), ("c".to_string(), 2)].into_iter().collect();
        let raw = vec![
            vec!["c".to_string(), "a".to_string()],
            vec!["b".to_string(), "b".to_string()],
        ];
        let pairs = resolve_pairs(&raw, &name_map).unwrap();
        assert_eq!(pairs, vec![(2, 0), (1, 1)]);
    }

    #[test]
    fn resolve_pairs_rejects_wrong_arity() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0)].into_iter().collect();
        assert!(resolve_pairs(&[vec!["a".to_string()]], &name_map).is_err());
        let raw3 = vec![vec!["a".to_string(), "a".to_string(), "a".to_string()]];
        assert!(resolve_pairs(&raw3, &name_map).is_err());
    }

    #[test]
    fn resolve_pairs_rejects_unknown_column_name() {
        let name_map: HashMap<String, usize> = [("a".to_string(), 0)].into_iter().collect();
        let raw = vec![vec!["a".to_string(), "nope".to_string()]];
        assert!(resolve_pairs(&raw, &name_map).is_err());
    }

    #[test]
    fn resolve_pairs_does_not_deduplicate() {
        let name_map: HashMap<String, usize> =
            [("a".to_string(), 0), ("b".to_string(), 1)].into_iter().collect();
        let raw = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "b".to_string()],
        ];
        let pairs = resolve_pairs(&raw, &name_map).unwrap();
        assert_eq!(pairs, vec![(0, 1), (0, 1)]);
    }

    #[test]
    fn resolve_triplets_maps_names_to_indices_preserving_order() {
        let name_map: HashMap<String, usize> =
            [("a".to_string(), 0), ("b".to_string(), 1), ("c".to_string(), 2)].into_iter().collect();
        let raw = vec![vec!["c".to_string(), "a".to_string(), "b".to_string()]];
        let triplets = resolve_triplets(&raw, &name_map).unwrap();
        assert_eq!(triplets, vec![(2, 0, 1)]);
    }

    #[test]
    fn resolve_triplets_rejects_wrong_arity() {
        let name_map: HashMap<String, usize> =
            [("a".to_string(), 0), ("b".to_string(), 1)].into_iter().collect();
        let raw = vec![vec!["a".to_string(), "b".to_string()]];
        assert!(resolve_triplets(&raw, &name_map).is_err());
    }

    #[test]
    fn resolve_triplets_rejects_unknown_column_name() {
        let name_map: HashMap<String, usize> =
            [("a".to_string(), 0), ("b".to_string(), 1), ("c".to_string(), 2)].into_iter().collect();
        let raw = vec![vec!["a".to_string(), "b".to_string(), "nope".to_string()]];
        assert!(resolve_triplets(&raw, &name_map).is_err());
    }
}
