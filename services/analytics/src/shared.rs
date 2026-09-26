// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities — used by entropy.rs, chi_squared.rs, minhash.rs, bloomfilter.rs
// ─────────────────────────────────────────────────────────────────────────────

use foldhash::fast::FixedState as FoldHashFixed;
use polars::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};

// ─────────────────────────────────────────────────────────────────────────────
// Kwargs structs
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub(crate) struct PairwiseKwargs {
    pub pairs: Option<Vec<Vec<String>>>,
}

#[derive(Deserialize)]
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
/// This matches how Polars `value_counts` groups floats and keeps the plugin's
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
        DataType::Int128 => {
            let build_hasher = FoldHashFixed::default();
            series
                .i128()?
                .iter()
                .map(|v| v.map_or((0, true), |x| (hash_one(&build_hasher, x), false)))
                .unzip()
        }
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
