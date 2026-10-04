//! Casting a column to a candidate type (spec 2026-09-26 §5.4).

use super::*;
use crate::techniques::describe::{parse_decimal, parse_iso};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int16Array, LargeStringArray, StructArray, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType as AT, TimeUnit};
use std::sync::Arc;

// ── cast and verify (Spec B §5.4) ───────────────────────────────────────────

pub(crate) fn exact_div(ns: i128, u: &TimeUnit) -> Option<i64> {
    let f = unit_ns(u);
    (ns % f == 0)
        .then(|| ns / f)
        .and_then(|v| i64::try_from(v).ok())
}

pub(crate) fn time_array(u: TimeUnit, v: Vec<Option<i64>>) -> ArrayRef {
    let narrow = || v.iter().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>();
    match u {
        TimeUnit::Second => Arc::new(Time32SecondArray::from(narrow())),
        TimeUnit::Millisecond => Arc::new(Time32MillisecondArray::from(narrow())),
        TimeUnit::Microsecond => Arc::new(Time64MicrosecondArray::from(v)),
        TimeUnit::Nanosecond => Arc::new(Time64NanosecondArray::from(v)),
    }
}

pub(crate) fn timestamp_array(u: TimeUnit, v: Vec<Option<i64>>, tz: Option<Arc<str>>) -> ArrayRef {
    match u {
        TimeUnit::Second => Arc::new(TimestampSecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(v).with_timezone_opt(tz)),
    }
}

pub(crate) fn decimal_array(v: Vec<Option<i128>>, scale: i8) -> Result<ArrayRef, String> {
    Decimal128Array::from(v)
        .with_precision_and_scale(38, scale)
        .map(|a| Arc::new(a) as ArrayRef)
        .map_err(|e| e.to_string())
}

/// `s.to_lowercase() == lower` for an already lower-cased `lower`, allocating only for
/// non-ASCII `s` (whose Unicode lower case can map onto ASCII, e.g. the Kelvin sign).
pub(crate) fn lower_eq(s: &str, lower: &str) -> bool {
    if s.is_ascii() {
        s.eq_ignore_ascii_case(lower)
    } else {
        s.to_lowercase() == lower
    }
}

/// `f` over every non-null text value; the first value it rejects fails the cast.
pub(crate) fn parsed<T>(
    text: &LargeStringArray,
    f: impl Fn(&str) -> Option<T>,
) -> Result<Vec<Option<T>>, String> {
    text.iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|s| f(s).ok_or_else(|| format!("row {i}: {s:?} does not convert")))
                .transpose()
        })
        .collect()
}

/// A string column recast to `t`, built from describe.rs's exact parsers.
pub(crate) fn from_text(t: &Target, text: &LargeStringArray) -> Result<ArrayRef, String> {
    match t {
        Target::Boolean | Target::BoolPair(..) => {
            let (tt, ff) = match t {
                Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
                _ => ("1", "0"),
            };
            let v = parsed(text, |s| {
                if lower_eq(s, tt) {
                    Some(true)
                } else if lower_eq(s, ff) {
                    Some(false)
                } else {
                    None
                }
            })?;
            Ok(Arc::new(BooleanArray::from(v)))
        }
        Target::Fixed(to) => match to {
            AT::Int8
            | AT::Int16
            | AT::Int32
            | AT::Int64
            | AT::UInt8
            | AT::UInt16
            | AT::UInt32
            | AT::UInt64 => arrow_cast(
                decimal_array(parsed(text, |s| parse_decimal(s.as_bytes(), 0))?, 0)?.as_ref(),
                to,
            ),
            AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => arrow_cast(
                decimal_array(
                    parsed(text, |x| parse_decimal(x.as_bytes(), *s as u32))?,
                    *s,
                )?
                .as_ref(),
                to,
            ),
            AT::Float32 => Ok(Arc::new(Float32Array::from(parsed(text, |s| {
                s.parse::<f32>().ok()
            })?))),
            AT::Float64 => Ok(Arc::new(Float64Array::from(parsed(text, |s| {
                s.parse::<f64>().ok()
            })?))),
            AT::Date32 => Ok(Arc::new(Date32Array::from(parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                (v.nanos == 0 && v.offset_minutes.is_none()).then_some(())?;
                i32::try_from(v.days?).ok()
            })?))),
            AT::Time32(u) | AT::Time64(u) => Ok(time_array(
                *u,
                parsed(text, |s| {
                    let v = parse_iso(s.as_bytes())?;
                    v.days.is_none().then_some(())?;
                    exact_div(v.nanos as i128, u)
                })?,
            )),
            AT::Timestamp(u, tz) => Ok(timestamp_array(
                *u,
                parsed(text, |s| exact_div(parse_iso(s.as_bytes())?.epoch_ns(), u))?,
                tz.clone(),
            )),
            t => Err(format!("no string conversion to {}", pa_name(t))),
        },
        Target::TimestampWithOffset(u) => {
            let parts = parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                Some((
                    exact_div(v.epoch_ns(), u)?,
                    i16::try_from(v.offset_minutes?).ok()?,
                ))
            })?;
            let ts = timestamp_array(
                *u,
                parts.iter().map(|p| Some(p.map_or(0, |p| p.0))).collect(),
                Some("UTC".into()),
            );
            let off: ArrayRef = Arc::new(Int16Array::from(
                parts
                    .iter()
                    .map(|p| p.map_or(0, |p| p.1))
                    .collect::<Vec<i16>>(),
            ));
            let AT::Struct(fields) = timestamp_with_offset(*u) else {
                unreachable!()
            };
            StructArray::try_new(fields, vec![ts, off], text.logical_nulls())
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        Target::Dictionary(..) => {
            arrow_cast(arrow_cast(text, &AT::Utf8)?.as_ref(), &t.arrow_type())
        }
        Target::Plain(to) => arrow_cast(text, to),
        t => Err(format!(
            "no string conversion to {}",
            pa_name(&t.arrow_type())
        )),
    }
}

/// Float → Decimal through the exact digits of each value's shortest round-trip
/// representation (ryu), never multiply-and-round.
pub(crate) fn float_to_decimal(src: &ArrayRef, to: &AT) -> Result<ArrayRef, String> {
    let (AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s)) = to else {
        return Err("not a decimal".into());
    };
    let f32_src = src.data_type() == &AT::Float32;
    let values = arrow_cast(src.as_ref(), &AT::Float64)?;
    let mut buf = ryu::Buffer::new();
    let unscaled: Vec<Option<i128>> = values
        .as_primitive::<Float64Type>()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|x| {
                if !x.is_finite() {
                    return Err(format!("row {i}: {x} is not finite"));
                }
                let repr = if f32_src {
                    buf.format_finite(x as f32).to_string()
                } else {
                    buf.format_finite(x).to_string()
                };
                decimal_from_repr(&repr, *s as u32)
                    .ok_or_else(|| format!("row {i}: {repr} does not fit scale {s}"))
            })
            .transpose()
        })
        .collect::<Result<_, _>>()?;
    arrow_cast(decimal_array(unscaled, *s)?.as_ref(), to)
}

/// The column recast to `t` (lists are wrapped by `wrap`).
pub(crate) fn cast_to(t: &Target, lvl: &Level) -> Result<ArrayRef, String> {
    let src = &lvl.values;
    match t {
        Target::Original(_) => Ok(src.clone()),
        Target::Null => Ok(arrow_array::new_null_array(&AT::Null, src.len())),
        _ if is_text(lvl.dtype) => from_text(t, lvl.text()?),
        Target::Fixed(to @ (AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..)))
            if is_float(lvl.dtype) =>
        {
            float_to_decimal(src, to)
        }
        Target::Boolean | Target::Fixed(_) | Target::Plain(_) => {
            arrow_cast(src.as_ref(), &t.arrow_type())
        }
        t => Err(format!("no cast to {}", pa_name(&t.arrow_type()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Decimal128Type;

    #[test]
    fn lower_case_comparison() {
        assert!(lower_eq("TRUE", "true") && lower_eq("tRuE", "true") && !lower_eq("true ", "true"));
        assert!(lower_eq("\u{212A}", "k")); // Kelvin sign lower-cases to ASCII k
        assert!(!lower_eq("É", "e") && lower_eq("É", "é"));
    }

    use arrow_array::Float64Array;

    #[test]
    fn floats_become_exact_decimals() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.1), Some(123.45), None]));
        let d = float_to_decimal(&a, &decimal_type(5, 2)).unwrap();
        let d = arrow_cast(d.as_ref(), &AT::Decimal128(38, 2)).unwrap();
        let v = d.as_primitive::<Decimal128Type>();
        assert_eq!((v.value(0), v.value(1), v.is_null(2)), (10, 12_345, true));
        assert!(verify_float(&a, &arrow_cast(d.as_ref(), &decimal_type(5, 2)).unwrap()).is_ok());
    }

    #[test]
    fn timestamps_with_offsets_from_text() {
        let text = LargeStringArray::from(vec![Some("2024-01-05T10:00+05:00"), None]);
        let a = from_text(&Target::TimestampWithOffset(TimeUnit::Second), &text).unwrap();
        let s = a.as_struct();
        let ts = s
            .column(0)
            .as_primitive::<arrow_array::types::TimestampSecondType>();
        assert_eq!(
            (
                ts.value(0),
                s.column(1)
                    .as_primitive::<arrow_array::types::Int16Type>()
                    .value(0)
            ),
            (1_704_430_800, 300)
        );
        assert!(a.is_null(1));
        assert!(verify_text(&Target::TimestampWithOffset(TimeUnit::Second), &text, &a).is_ok());
    }

    #[test]
    fn time_targets_reject_text_carrying_a_date() {
        let text = LargeStringArray::from(vec!["2024-01-05T10:00"]);
        assert!(from_text(&Target::Fixed(AT::Time32(TimeUnit::Second)), &text).is_err());
        let bare = LargeStringArray::from(vec!["10:00:00"]);
        assert!(from_text(&Target::Fixed(AT::Time32(TimeUnit::Second)), &bare).is_ok());
    }

    #[test]
    fn float_to_decimal_rejects_non_finite() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(f64::INFINITY),
            Some(f64::NAN),
            None,
        ]));
        let err = float_to_decimal(&a, &decimal_type(5, 2)).unwrap_err();
        assert!(err.contains("not finite"), "{err}");
    }
}
