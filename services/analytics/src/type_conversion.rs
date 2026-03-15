use foldhash::fast::FixedState;
use polars::prelude::*;
use std::hash::{BuildHasher, Hash, Hasher};

/// Convert a Polars Series to Vec<u64> for frequency counting
///
/// Handles various Polars data types by converting them to u64 representation:
/// - Integer types: direct cast to u64
/// - Float types: IEEE 754 bit representation via to_bits()
/// - Date/Datetime/Duration: cast i64 to u64
/// - String: hash via FoldHash FixedState
/// - Nulls: mapped to 0u64
///
/// # Arguments
/// * `series` - Polars Series to convert
///
/// # Returns
/// Vec<u64> with same length as input Series
///
/// # Errors
/// Returns PolarsError for unsupported types (Boolean, List, Struct, etc.)
pub fn series_to_u64(series: &Series) -> PolarsResult<Vec<u64>> {
    match series.dtype() {
        DataType::Int8 => {
            Ok(series.i8()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::Int16 => {
            Ok(series.i16()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::Int32 => {
            Ok(series.i32()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::Int64 => {
            Ok(series.i64()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::UInt8 => {
            Ok(series.u8()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::UInt16 => {
            Ok(series.u16()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::UInt32 => {
            Ok(series.u32()?.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::UInt64 => {
            Ok(series.u64()?.iter().map(|v| v.unwrap_or(0)).collect())
        }
        DataType::Float32 => {
            Ok(series.f32()?.iter()
                .map(|v| v.unwrap_or(0.0).to_bits() as u64)
                .collect())
        }
        DataType::Float64 => {
            Ok(series.f64()?.iter()
                .map(|v| v.unwrap_or(0.0).to_bits())
                .collect())
        }
        DataType::Date => {
            Ok(series.date()?.phys.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::Datetime(_, _) => {
            Ok(series.datetime()?.phys.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::Duration(_) => {
            Ok(series.duration()?.phys.iter().map(|v| v.unwrap_or(0) as u64).collect())
        }
        DataType::String => {
            let build_hasher = FixedState::default();
            Ok(series.str()?.iter().map(|v| {
                let mut hasher = build_hasher.build_hasher();
                v.unwrap_or("").hash(&mut hasher);
                hasher.finish()
            }).collect())
        }
        _ => {
            Err(PolarsError::ComputeError(
                format!(
                    "Unsupported data type for joint entropy: {:?}. \
                     Supported types: Int8/16/32/64, UInt8/16/32/64, Float32/64, \
                     Date, Datetime, Duration, String",
                    series.dtype()
                ).into()
            ))
        }
    }
}

/// Convert a Polars Series to Vec<Option<u64>>, preserving nulls as None.
///
/// Same type handling as `series_to_u64` but retains null information
/// for callers that need to distinguish null from zero (e.g. bloom filters).
pub fn series_to_opt_u64(series: &Series) -> PolarsResult<Vec<Option<u64>>> {
    match series.dtype() {
        DataType::Int8 => Ok(series.i8()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::Int16 => Ok(series.i16()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::Int32 => Ok(series.i32()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::Int64 => Ok(series.i64()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::UInt8 => Ok(series.u8()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::UInt16 => Ok(series.u16()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::UInt32 => Ok(series.u32()?.iter().map(|v| v.map(|x| x as u64)).collect()),
        DataType::UInt64 => Ok(series.u64()?.iter().collect()),
        DataType::Float32 => Ok(series
            .f32()?
            .iter()
            .map(|v| v.map(|x| x.to_bits() as u64))
            .collect()),
        DataType::Float64 => Ok(series
            .f64()?
            .iter()
            .map(|v| v.map(|x| x.to_bits()))
            .collect()),
        DataType::Date => Ok(series
            .date()?
            .phys
            .iter()
            .map(|v| v.map(|x| x as u64))
            .collect()),
        DataType::Datetime(_, _) => Ok(series
            .datetime()?
            .phys
            .iter()
            .map(|v| v.map(|x| x as u64))
            .collect()),
        DataType::Duration(_) => Ok(series
            .duration()?
            .phys
            .iter()
            .map(|v| v.map(|x| x as u64))
            .collect()),
        DataType::String => {
            let build_hasher = FixedState::default();
            Ok(series
                .str()?
                .iter()
                .map(|v| {
                    v.map(|s| {
                        let mut hasher = build_hasher.build_hasher();
                        s.hash(&mut hasher);
                        hasher.finish()
                    })
                })
                .collect())
        }
        _ => Err(PolarsError::ComputeError(
            format!(
                "Unsupported data type: {:?}. \
                 Supported: Int8/16/32/64, UInt8/16/32/64, Float32/64, \
                 Date, Datetime, Duration, String",
                series.dtype()
            )
            .into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_types() {
        let s = Series::new("test".into(), &[1i32, 2, 3]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result, vec![1u64, 2, 3]);
    }

    #[test]
    fn test_uint_types() {
        let s = Series::new("test".into(), &[1u64, 2, 3]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result, vec![1u64, 2, 3]);
    }

    #[test]
    fn test_float_types() {
        let s = Series::new("test".into(), &[1.5f64, 2.5, 3.5]);
        let result = series_to_u64(&s).unwrap();
        // Should convert via to_bits()
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.5f64.to_bits());
    }

    #[test]
    fn test_string_types() {
        let s = Series::new("test".into(), &["a", "b", "c"]);
        let result = series_to_u64(&s).unwrap();
        // Should hash strings
        assert_eq!(result.len(), 3);
        // Same string should give same hash
        let s2 = Series::new("test".into(), &["a"]);
        let result2 = series_to_u64(&s2).unwrap();
        assert_eq!(result[0], result2[0]);
    }

    #[test]
    fn test_null_handling() {
        let s = Series::new("test".into(), &[Some(1i32), None, Some(3)]);
        let result = series_to_u64(&s).unwrap();
        assert_eq!(result, vec![1u64, 0, 3]);
    }

    #[test]
    fn test_unsupported_type() {
        let s = Series::new("test".into(), &[true, false, true]);
        let result = series_to_u64(&s);
        assert!(result.is_err());
    }
}
