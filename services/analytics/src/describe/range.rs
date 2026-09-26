// ─────────────────────────────────────────────────────────────────────────────
// describe — first-occurrence extremes and value lengths (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// Ordering matches Polars `sort()`: integers, Decimal and temporals by physical
// value; floats numerically with NaN excluded (-0.0 ties 0.0); strings and
// binary by bytes; Categorical by string value; Enum by category order (its
// physical code); List, Array and Struct by Polars' row encoding — the encoding
// its sort uses. Ties keep the lowest row index.

use polars::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars::prelude::*;
use polars_arrow::array::Array;

pub(crate) struct Range {
    pub argmin: Option<u64>,
    pub argmax: Option<u64>,
    pub min_len: Option<u64>,
    pub max_len: Option<u64>,
}

fn lt<T: PartialOrd>(a: T, b: T) -> bool {
    a < b
}

fn extremes<T: Copy>(values: impl Iterator<Item = Option<T>>, lt: impl Fn(T, T) -> bool) -> (Option<u64>, Option<u64>) {
    let (mut lo, mut hi): (Option<(u64, T)>, Option<(u64, T)>) = (None, None);
    for (i, v) in values.enumerate() {
        let Some(v) = v else { continue };
        if lo.is_none_or(|(_, m)| lt(v, m)) {
            lo = Some((i as u64, v));
        }
        if hi.is_none_or(|(_, m)| lt(m, v)) {
            hi = Some((i as u64, v));
        }
    }
    (lo.map(|x| x.0), hi.map(|x| x.0))
}

fn arg_extremes(s: &Series) -> PolarsResult<(Option<u64>, Option<u64>)> {
    Ok(match s.dtype() {
        DataType::Float32 => extremes(s.f32()?.iter().map(|v| v.filter(|x| !x.is_nan())), lt),
        DataType::Float64 => extremes(s.f64()?.iter().map(|v| v.filter(|x| !x.is_nan())), lt),
        DataType::String => extremes(s.str()?.iter().map(|v| v.map(str::as_bytes)), lt),
        DataType::Binary => extremes(s.binary()?.iter(), lt),
        DataType::Categorical(_, _) => return arg_extremes(&s.cast(&DataType::String)?),
        DataType::Boolean => extremes(s.bool()?.iter(), lt),
        DataType::List(_) | DataType::Array(_, _) | DataType::Struct(_) => {
            let rows = _get_rows_encoded_arr(&[s.clone().into_column()], &[false], &[false])?;
            let valid = s.is_not_null();
            extremes(rows.values_iter().zip(valid.iter()).map(|(r, ok)| (ok == Some(true)).then_some(r)), lt)
        }
        _ => {
            let p = s.to_physical_repr();
            match p.dtype() {
                DataType::Int8 => extremes(p.i8()?.iter(), lt),
                DataType::Int16 => extremes(p.i16()?.iter(), lt),
                DataType::Int32 => extremes(p.i32()?.iter(), lt),
                DataType::Int64 => extremes(p.i64()?.iter(), lt),
                DataType::Int128 => extremes(p.i128()?.iter(), lt),
                DataType::UInt8 => extremes(p.u8()?.iter(), lt),
                DataType::UInt16 => extremes(p.u16()?.iter(), lt),
                DataType::UInt32 => extremes(p.u32()?.iter(), lt),
                DataType::UInt64 => extremes(p.u64()?.iter(), lt),
                dt => polars_bail!(ComputeError: "describe: no ordering for {dt}"),
            }
        }
    })
}

fn min_max(values: impl Iterator<Item = Option<u64>>) -> (Option<u64>, Option<u64>) {
    values.flatten().fold((None, None), |(lo, hi), v| {
        (Some(lo.map_or(v, |x: u64| x.min(v))), Some(hi.map_or(v, |x: u64| x.max(v))))
    })
}

/// Per row of a **rechunked** ListChunked: `Some((start, len))` into its child
/// values (`get_inner()`), `None` for a null list.
pub(crate) fn list_ranges(ca: &ListChunked) -> Vec<Option<(usize, usize)>> {
    ca.downcast_iter()
        .flat_map(|arr| {
            let offsets = arr.offsets().as_slice();
            (0..arr.len()).map(move |i| arr.is_valid(i).then(|| (offsets[i] as usize, (offsets[i + 1] - offsets[i]) as usize)))
        })
        .collect()
}

fn lengths(s: &Series) -> PolarsResult<(Option<u64>, Option<u64>)> {
    Ok(match s.dtype() {
        DataType::String => min_max(s.str()?.iter().map(|v| v.map(|x| x.len() as u64))),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => return lengths(&s.cast(&DataType::String)?),
        DataType::Binary => min_max(s.binary()?.iter().map(|v| v.map(|x| x.len() as u64))),
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            min_max(list_ranges(&ca).into_iter().map(|r| r.map(|(_, len)| len as u64)))
        }
        DataType::Array(_, width) => min_max(s.is_not_null().iter().map(|ok| (ok == Some(true)).then_some(*width as u64))),
        _ => (None, None),
    })
}

pub(crate) fn range(s: &Series) -> PolarsResult<Range> {
    let (argmin, argmax) = arg_extremes(s)?;
    let (min_len, max_len) = lengths(s)?;
    Ok(Range { argmin, argmax, min_len, max_len })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(s: Series) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
        let x = range(&s).unwrap();
        (x.argmin, x.argmax, x.min_len, x.max_len)
    }

    #[test]
    fn first_occurrence_extremes() {
        assert_eq!(r(Series::new("x".into(), &[5i64, 1, 3, 1, 5])), (Some(1), Some(0), None, None));
        assert_eq!(r(Series::new("x".into(), &[0.0f64, -0.0, f64::NAN, 1.5])), (Some(0), Some(3), None, None));
        assert_eq!(r(Series::new("x".into(), &[None::<i32>, None])), (None, None, None, None));
    }

    #[test]
    fn strings_bytes_and_lengths() {
        assert_eq!(r(Series::new("x".into(), &[Some("ab"), Some(""), None, Some("héllo")])), (Some(1), Some(3), Some(0), Some(6)));
    }

    #[test]
    fn lists_use_polars_sort_order() {
        let s = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 5])), Some(Series::new("".into(), &[2i64, 1])), None, Some(Series::new_empty("".into(), &DataType::Int64))]);
        assert_eq!(r(s), (Some(3), Some(1), Some(0), Some(2)));
    }
}
