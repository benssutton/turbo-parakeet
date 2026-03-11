// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities — used by entropy.rs and chi_squared.rs
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
// Null sentinel — u64::MAX marks missing values
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) const NULL_SENTINEL: u64 = u64::MAX;

// ─────────────────────────────────────────────────────────────────────────────
// Null-safe column conversion (nulls → NULL_SENTINEL)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn series_to_u64(series: &Series) -> PolarsResult<Vec<u64>> {
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
        DataType::Int128 => {
            let build_hasher = FoldHashFixed::default();
            Ok(series
                .i128()?
                .iter()
                .map(|v| match v {
                    None => NULL_SENTINEL,
                    Some(x) => {
                        let mut hasher = build_hasher.build_hasher();
                        x.hash(&mut hasher);
                        hasher.finish()
                    }
                })
                .collect())
        }
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
        DataType::Boolean => Ok(series
            .bool()?
            .iter()
            .map(|v| match v {
                Some(true) => 1u64,
                Some(false) => 0u64,
                None => NULL_SENTINEL,
            })
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
        DataType::Time => Ok(series
            .time()?
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
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let phys = series.to_physical_repr();
            Ok(phys
                .u32()?
                .iter()
                .map(|v| v.map_or(NULL_SENTINEL, |x| x as u64))
                .collect())
        }
        // Decimal is physically i128 with a fixed scale per-Series; hash the raw integer.
        DataType::Decimal(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let phys = series.to_physical_repr();
            Ok(phys
                .i128()?
                .iter()
                .map(|v| match v {
                    None => NULL_SENTINEL,
                    Some(x) => {
                        let mut hasher = build_hasher.build_hasher();
                        x.hash(&mut hasher);
                        hasher.finish()
                    }
                })
                .collect())
        }
        // List and Array: hash ordered element sequence so [1,2] != [2,1].
        DataType::List(_) => {
            let build_hasher = FoldHashFixed::default();
            let list_ca = series.list()?;
            let mut out = Vec::with_capacity(list_ca.len());
            for opt_inner in list_ca.amortized_iter() {
                match opt_inner {
                    None => out.push(NULL_SENTINEL),
                    Some(inner) => {
                        let vals = series_to_u64(inner.as_ref())?;
                        let mut h = build_hasher.build_hasher();
                        vals.len().hash(&mut h);
                        for v in vals {
                            v.hash(&mut h);
                        }
                        out.push(h.finish());
                    }
                }
            }
            Ok(out)
        }
        DataType::Array(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let arr_ca = series.array()?;
            let mut out = Vec::with_capacity(arr_ca.len());
            for opt_inner in arr_ca.amortized_iter() {
                match opt_inner {
                    None => out.push(NULL_SENTINEL),
                    Some(inner) => {
                        let vals = series_to_u64(inner.as_ref())?;
                        let mut h = build_hasher.build_hasher();
                        vals.len().hash(&mut h);
                        for v in vals {
                            v.hash(&mut h);
                        }
                        out.push(h.finish());
                    }
                }
            }
            Ok(out)
        }
        _ => Err(PolarsError::ComputeError(
            format!(
                "Unsupported data type: {:?}.",
                series.dtype()
            )
            .into(),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel column cache
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn build_column_cache_par(
    inputs: &[Series],
    needed: &HashSet<usize>,
) -> PolarsResult<Vec<Vec<u64>>> {
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

