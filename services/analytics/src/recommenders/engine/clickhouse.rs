//! Arrow → ClickHouse column types (spec 2026-10-19 §2).

use super::*;
use crate::common::clickhouse_sizes::ChLayout;
use arrow_array::Array;
use arrow_schema::{DataType as AT, TimeUnit};

/// Last day number `Date` holds here (2149-06-06, UInt16 days).
const DATE_MAX_DAYS: i128 = 65_535;
/// Last second `DateTime` holds (2106-02-07 06:28:15).
const DATETIME_MAX_SECONDS: i128 = 4_294_967_295;

fn fixed(width: usize, signed: bool, gcd_applies: bool) -> ChLayout {
    ChLayout::Fixed {
        width,
        signed,
        gcd_applies,
    }
}

fn in_range(range: Option<(i128, i128)>, max: i128) -> bool {
    matches!(range, Some((lo, hi)) if lo >= 0 && hi <= max && lo <= hi)
}

fn quote_ident(name: &str) -> String {
    let mut chars = name.chars();
    let plain = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        name.to_string()
    } else {
        format!("`{}`", name.replace('\\', "\\\\").replace('`', "\\`"))
    }
}

fn enum_type(values: &[String]) -> (String, ChLayout) {
    let (name, width) = if values.len() <= 127 {
        ("Enum8", 1)
    } else {
        ("Enum16", 2)
    };
    let items: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            format!(
                "'{}' = {}",
                v.replace('\\', "\\\\").replace('\'', "\\'"),
                i + 1
            )
        })
        .collect();
    (
        format!("{name}({})", items.join(", ")),
        fixed(width, true, false),
    )
}

fn precision_digits(unit: &TimeUnit) -> u8 {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 3,
        TimeUnit::Microsecond => 6,
        TimeUnit::Nanosecond => 9,
    }
}

/// The ClickHouse type for a recommended Arrow type (spec 2026-10-19 §2), with its layout;
/// Err(reason) when there is none. `range`: the physical min / max (Date32 days, timestamp
/// values in their unit) when known; `nullable`: the recommended array holds nulls;
/// `enum_values`: a kept Polars Enum's categories.
pub(crate) fn clickhouse_type(
    t: &AT,
    range: Option<(i128, i128)>,
    nullable: bool,
    enum_values: Option<&[String]>,
) -> Result<(String, ChLayout), String> {
    let (ty, layout) = match t {
        AT::Dictionary(_, _) if enum_values.is_some() => enum_type(enum_values.unwrap_or(&[])),
        AT::Dictionary(_, v) if matches!(**v, AT::Utf8 | AT::LargeUtf8) => {
            // LowCardinality cannot sit inside Nullable: the nullable form is
            // LowCardinality(Nullable(String)). The layout only tells the emulator that nulls
            // occur: it stores no null map (null is dictionary key 0).
            return Ok(if nullable {
                (
                    "LowCardinality(Nullable(String))".to_string(),
                    ChLayout::Nullable(Box::new(ChLayout::LowCardinality)),
                )
            } else {
                (
                    "LowCardinality(String)".to_string(),
                    ChLayout::LowCardinality,
                )
            });
        }
        AT::Boolean => ("Bool".to_string(), fixed(1, false, false)),
        AT::Int8 => ("Int8".to_string(), fixed(1, true, true)),
        AT::Int16 => ("Int16".to_string(), fixed(2, true, true)),
        AT::Int32 => ("Int32".to_string(), fixed(4, true, true)),
        AT::Int64 => ("Int64".to_string(), fixed(8, true, true)),
        AT::UInt8 => ("UInt8".to_string(), fixed(1, false, true)),
        AT::UInt16 => ("UInt16".to_string(), fixed(2, false, true)),
        AT::UInt32 => ("UInt32".to_string(), fixed(4, false, true)),
        AT::UInt64 => ("UInt64".to_string(), fixed(8, false, true)),
        AT::Float32 => ("Float32".to_string(), fixed(4, true, false)),
        AT::Float64 => ("Float64".to_string(), fixed(8, true, false)),
        AT::Decimal32(p, s) | AT::Decimal64(p, s) | AT::Decimal128(p, s) => match *p {
            0..=9 => (format!("Decimal32({s})"), fixed(4, true, true)),
            10..=18 => (format!("Decimal64({s})"), fixed(8, true, true)),
            _ => (format!("Decimal128({s})"), fixed(16, true, true)),
        },
        AT::Date32 if in_range(range, DATE_MAX_DAYS) => ("Date".to_string(), fixed(2, false, true)),
        AT::Date32 => ("Date32".to_string(), fixed(4, true, true)),
        AT::Timestamp(unit, tz) => {
            if *unit == TimeUnit::Second && in_range(range, DATETIME_MAX_SECONDS) {
                let ty = match tz {
                    Some(tz) => format!("DateTime('{tz}')"),
                    None => "DateTime".to_string(),
                };
                (ty, fixed(4, false, true))
            } else {
                let p = precision_digits(unit);
                let ty = match tz {
                    Some(tz) => format!("DateTime64({p}, '{tz}')"),
                    None => format!("DateTime64({p})"),
                };
                (ty, fixed(8, true, true))
            }
        }
        AT::Time32(u) | AT::Time64(u) => (
            format!("Time64({})", precision_digits(u)),
            fixed(8, true, false),
        ),
        AT::Utf8 | AT::LargeUtf8 | AT::Utf8View | AT::Binary | AT::LargeBinary | AT::BinaryView => {
            ("String".to_string(), ChLayout::String)
        }
        AT::FixedSizeBinary(n) => (
            format!("FixedString({n})"),
            ChLayout::FixedString(*n as usize),
        ),
        AT::List(f) | AT::LargeList(f) | AT::FixedSizeList(f, _) => {
            let (ty, layout) = clickhouse_type(f.data_type(), None, f.is_nullable(), None)?;
            (format!("Array({ty})"), ChLayout::Array(Box::new(layout)))
        }
        AT::Struct(fields) => {
            if let Some(AT::Timestamp(u, _)) = fields.first().map(|f| f.data_type()) {
                if *t == timestamp_with_offset(*u) {
                    return Err("timestamp_with_offset has no ClickHouse type".to_string());
                }
            }
            if fields.is_empty() {
                return Err("empty struct has no ClickHouse type".to_string());
            }
            let mut names = Vec::with_capacity(fields.len());
            let mut layouts = Vec::with_capacity(fields.len());
            for f in fields.iter() {
                let (ty, layout) = clickhouse_type(f.data_type(), None, f.is_nullable(), None)?;
                names.push(format!("{} {ty}", quote_ident(f.name())));
                layouts.push(layout);
            }
            (
                format!("Tuple({})", names.join(", ")),
                ChLayout::Tuple(layouts),
            )
        }
        other => return Err(format!("{other} has no ClickHouse type")),
    };
    // ClickHouse cannot make Array or Tuple Nullable.
    if nullable && !matches!(layout, ChLayout::Array(_) | ChLayout::Tuple(_)) {
        Ok((
            format!("Nullable({ty})"),
            ChLayout::Nullable(Box::new(layout)),
        ))
    } else {
        Ok((ty, layout))
    }
}

/// The factor by which a column's original physical values (integers; decimal unscaled
/// values; Date days; Datetime values in their unit) divide to give the recommended type's
/// physical values, so the running GCD and range carry over. None when the recommended
/// values are not an exact rescaling of the original ones (text-derived numbers, float →
/// decimal, ...).
pub(crate) fn rescale(original: &polars::prelude::DataType, recommended: &AT) -> Option<i128> {
    use polars::prelude::{DataType as PT, TimeUnit as PU};
    let ns = |u: &PU| match u {
        PU::Nanoseconds => 1i128,
        PU::Microseconds => 1_000,
        PU::Milliseconds => 1_000_000,
    };
    match (original, recommended) {
        (
            PT::Int8
            | PT::Int16
            | PT::Int32
            | PT::Int64
            | PT::UInt8
            | PT::UInt16
            | PT::UInt32
            | PT::UInt64,
            AT::Int8
            | AT::Int16
            | AT::Int32
            | AT::Int64
            | AT::UInt8
            | AT::UInt16
            | AT::UInt32
            | AT::UInt64,
        )
        | (PT::Date, AT::Date32) => Some(1),
        (PT::Decimal(_, s), AT::Decimal32(_, r) | AT::Decimal64(_, r) | AT::Decimal128(_, r)) => {
            let d = i32::try_from(s.unwrap_or(0)).ok()? - i32::from(*r);
            u32::try_from(d).ok().and_then(|d| 10i128.checked_pow(d))
        }
        (PT::Datetime(u, _), AT::Timestamp(r, _)) => {
            let (o, r) = (ns(u), super::unit_ns(r));
            (r >= o).then_some(r / o)
        }
        (PT::Datetime(u, _), AT::Date32) => Some(86_400 * 1_000_000_000 / ns(u)),
        _ => None,
    }
}

/// Statistics of an integer-backed column's raw physical values (non-null ones).
pub(crate) struct PhysicalStats {
    pub gcd: u128,
    pub max_abs: u128,
    pub range: (i128, i128),
}

fn binary_gcd(mut a: u128, mut b: u128) -> u128 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

fn stats_of<T: arrow_array::ArrowPrimitiveType>(
    a: &dyn Array,
    wide: impl Fn(T::Native) -> i128,
) -> Option<PhysicalStats> {
    let a = a
        .as_any()
        .downcast_ref::<arrow_array::PrimitiveArray<T>>()?;
    let (mut gcd, mut max_abs, mut lo, mut hi) = (0u128, 0u128, i128::MAX, i128::MIN);
    for v in a.iter().flatten() {
        let v = wide(v);
        let m = v.unsigned_abs();
        gcd = binary_gcd(gcd, m);
        max_abs = max_abs.max(m);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo <= hi).then_some(PhysicalStats {
        gcd,
        max_abs,
        range: (lo, hi),
    })
}

/// GCD, largest magnitude and range of the non-null physical values of an
/// integer, decimal (unscaled), Date32 (days), Timestamp (unit values) or Time array; None
/// for other types or when there are no non-null values.
pub(crate) fn physical_stats(a: &dyn Array) -> Option<PhysicalStats> {
    use arrow_array::types::*;
    match a.data_type() {
        AT::Int8 => stats_of::<Int8Type>(a, i128::from),
        AT::Int16 => stats_of::<Int16Type>(a, i128::from),
        AT::Int32 => stats_of::<Int32Type>(a, i128::from),
        AT::Int64 => stats_of::<Int64Type>(a, i128::from),
        AT::UInt8 => stats_of::<UInt8Type>(a, i128::from),
        AT::UInt16 => stats_of::<UInt16Type>(a, i128::from),
        AT::UInt32 => stats_of::<UInt32Type>(a, i128::from),
        AT::UInt64 => stats_of::<UInt64Type>(a, i128::from),
        AT::Decimal32(..) => stats_of::<Decimal32Type>(a, i128::from),
        AT::Decimal64(..) => stats_of::<Decimal64Type>(a, i128::from),
        AT::Decimal128(..) => stats_of::<Decimal128Type>(a, |v| v),
        AT::Date32 => stats_of::<Date32Type>(a, i128::from),
        AT::Timestamp(TimeUnit::Second, _) => stats_of::<TimestampSecondType>(a, i128::from),
        AT::Timestamp(TimeUnit::Millisecond, _) => {
            stats_of::<TimestampMillisecondType>(a, i128::from)
        }
        AT::Timestamp(TimeUnit::Microsecond, _) => {
            stats_of::<TimestampMicrosecondType>(a, i128::from)
        }
        AT::Timestamp(TimeUnit::Nanosecond, _) => {
            stats_of::<TimestampNanosecondType>(a, i128::from)
        }
        AT::Time32(TimeUnit::Second) => stats_of::<Time32SecondType>(a, i128::from),
        AT::Time32(_) => stats_of::<Time32MillisecondType>(a, i128::from),
        AT::Time64(TimeUnit::Microsecond) => stats_of::<Time64MicrosecondType>(a, i128::from),
        AT::Time64(_) => stats_of::<Time64NanosecondType>(a, i128::from),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field as AField, Fields};

    #[test]
    fn rescale_cases() {
        use polars::prelude::{DataType as PT, TimeUnit as PU};
        assert_eq!(rescale(&PT::Int64, &AT::Int8), Some(1));
        assert_eq!(rescale(&PT::Date, &AT::Date32), Some(1));
        assert_eq!(
            rescale(&PT::Decimal(Some(10), Some(4)), &AT::Decimal32(7, 2)),
            Some(100)
        );
        assert_eq!(
            rescale(
                &PT::Datetime(PU::Nanoseconds, None),
                &AT::Timestamp(TimeUnit::Second, None)
            ),
            Some(1_000_000_000)
        );
        assert_eq!(
            rescale(&PT::Datetime(PU::Milliseconds, None), &AT::Date32),
            Some(86_400_000)
        );
        assert_eq!(rescale(&PT::String, &AT::Int32), None);
        assert_eq!(rescale(&PT::Float64, &AT::Decimal64(10, 2)), None);
    }

    #[test]
    fn physical_stats_cases() {
        use arrow_array::{Decimal128Array, Int64Array, NullArray, StringArray, UInt8Array};
        let s = physical_stats(&Int64Array::from(vec![Some(1000), Some(-3000), None])).unwrap();
        assert_eq!((s.gcd, s.max_abs, s.range), (1000, 3000, (-3000, 1000)));
        let s = physical_stats(&UInt8Array::from(vec![4, 8])).unwrap();
        assert_eq!(s.gcd, 4);
        let d = Decimal128Array::from(vec![150, 300])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(physical_stats(&d).unwrap().gcd, 150);
        assert!(physical_stats(&Int64Array::from(vec![None, None])).is_none());
        assert!(physical_stats(&StringArray::from(vec!["a"])).is_none());
        assert!(physical_stats(&NullArray::new(2)).is_none());
        assert_eq!(
            physical_stats(&UInt8Array::from(vec![0, 0])).unwrap().gcd,
            0
        );
    }

    use std::sync::Arc;

    fn ty(t: &AT) -> String {
        clickhouse_type(t, None, false, None).unwrap().0
    }

    fn dict() -> AT {
        AT::Dictionary(Box::new(AT::UInt16), Box::new(AT::Utf8))
    }

    #[test]
    fn bool_and_numbers() {
        let (s, l) = clickhouse_type(&AT::Boolean, None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Bool", fixed(1, false, false)));
        for (t, name, w, signed) in [
            (AT::Int8, "Int8", 1, true),
            (AT::Int16, "Int16", 2, true),
            (AT::Int32, "Int32", 4, true),
            (AT::Int64, "Int64", 8, true),
            (AT::UInt8, "UInt8", 1, false),
            (AT::UInt16, "UInt16", 2, false),
            (AT::UInt32, "UInt32", 4, false),
            (AT::UInt64, "UInt64", 8, false),
        ] {
            let (s, l) = clickhouse_type(&t, None, false, None).unwrap();
            assert_eq!((s.as_str(), l), (name, fixed(w, signed, true)));
        }
        let (s, l) = clickhouse_type(&AT::Float32, None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Float32", fixed(4, true, false)));
        let (s, l) = clickhouse_type(&AT::Float64, None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Float64", fixed(8, true, false)));
    }

    #[test]
    fn decimals() {
        let (s, l) = clickhouse_type(&AT::Decimal128(9, 2), None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Decimal32(2)", fixed(4, true, true)));
        let (s, l) = clickhouse_type(&AT::Decimal128(10, 2), None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Decimal64(2)", fixed(8, true, true)));
        let (s, l) = clickhouse_type(&AT::Decimal128(19, 2), None, false, None).unwrap();
        assert_eq!((s.as_str(), l), ("Decimal128(2)", fixed(16, true, true)));
        assert!(clickhouse_type(&AT::Decimal256(40, 2), None, false, None).is_err());
    }

    #[test]
    fn dates() {
        let d = |r| clickhouse_type(&AT::Date32, r, false, None).unwrap();
        assert_eq!(
            d(Some((0, 65_535))),
            ("Date".to_string(), fixed(2, false, true))
        );
        assert_eq!(
            d(Some((0, 65_536))),
            ("Date32".to_string(), fixed(4, true, true))
        );
        assert_eq!(d(Some((-1, 5))).0, "Date32");
        assert_eq!(d(None), ("Date32".to_string(), fixed(4, true, true)));
    }

    #[test]
    fn timestamps() {
        let utc: Option<Arc<str>> = Some(Arc::from("UTC"));
        let s = AT::Timestamp(TimeUnit::Second, utc.clone());
        let f = |t: &AT, r| clickhouse_type(t, r, false, None).unwrap();
        assert_eq!(
            f(&s, Some((0, 4_294_967_295))),
            ("DateTime('UTC')".to_string(), fixed(4, false, true))
        );
        assert_eq!(
            f(&s, Some((0, 4_294_967_296))),
            ("DateTime64(0, 'UTC')".to_string(), fixed(8, true, true))
        );
        assert_eq!(f(&s, None).0, "DateTime64(0, 'UTC')");
        assert_eq!(
            f(&AT::Timestamp(TimeUnit::Second, None), Some((0, 1))).0,
            "DateTime"
        );
        assert_eq!(
            f(&AT::Timestamp(TimeUnit::Second, None), None).0,
            "DateTime64(0)"
        );
        assert_eq!(
            f(
                &AT::Timestamp(TimeUnit::Millisecond, utc.clone()),
                Some((0, 1))
            )
            .0,
            "DateTime64(3, 'UTC')"
        );
        assert_eq!(
            f(&AT::Timestamp(TimeUnit::Microsecond, None), None).0,
            "DateTime64(6)"
        );
        assert_eq!(
            f(&AT::Timestamp(TimeUnit::Nanosecond, utc), None).0,
            "DateTime64(9, 'UTC')"
        );
    }

    #[test]
    fn time_maps_to_time64() {
        let f = |t: AT| clickhouse_type(&t, None, false, None).unwrap();
        for (t, name) in [
            (AT::Time32(TimeUnit::Second), "Time64(0)"),
            (AT::Time32(TimeUnit::Millisecond), "Time64(3)"),
            (AT::Time64(TimeUnit::Microsecond), "Time64(6)"),
            (AT::Time64(TimeUnit::Nanosecond), "Time64(9)"),
        ] {
            assert_eq!(f(t), (name.to_string(), fixed(8, true, false)));
        }
    }

    #[test]
    fn strings_and_binary() {
        for t in [
            AT::Utf8,
            AT::LargeUtf8,
            AT::Utf8View,
            AT::Binary,
            AT::LargeBinary,
            AT::BinaryView,
        ] {
            assert_eq!(
                clickhouse_type(&t, None, false, None).unwrap(),
                ("String".to_string(), ChLayout::String)
            );
        }
        assert_eq!(
            clickhouse_type(&AT::FixedSizeBinary(16), None, false, None).unwrap(),
            ("FixedString(16)".to_string(), ChLayout::FixedString(16))
        );
    }

    #[test]
    fn dictionaries() {
        assert_eq!(
            clickhouse_type(&dict(), None, false, None).unwrap(),
            (
                "LowCardinality(String)".to_string(),
                ChLayout::LowCardinality
            )
        );
        assert_eq!(
            clickhouse_type(&dict(), None, true, None).unwrap(),
            (
                "LowCardinality(Nullable(String))".to_string(),
                ChLayout::Nullable(Box::new(ChLayout::LowCardinality))
            )
        );
    }

    #[test]
    fn enums() {
        let vals: Vec<String> = (0..127).map(|i| format!("v{i}")).collect();
        let (s, l) = clickhouse_type(&dict(), None, false, Some(&vals)).unwrap();
        assert!(s.starts_with("Enum8('v0' = 1, 'v1' = 2"));
        assert!(s.ends_with("'v126' = 127)"));
        assert_eq!(l, fixed(1, true, false));
        let vals: Vec<String> = (0..128).map(|i| format!("v{i}")).collect();
        let (s, l) = clickhouse_type(&dict(), None, false, Some(&vals)).unwrap();
        assert!(s.starts_with("Enum16("));
        assert_eq!(l, fixed(2, true, false));
        let q = vec!["it's".to_string(), "a\\b".to_string()];
        assert_eq!(
            clickhouse_type(&dict(), None, false, Some(&q)).unwrap().0,
            "Enum8('it\\'s' = 1, 'a\\\\b' = 2)"
        );
    }

    #[test]
    fn nullable_wrappers() {
        let (s, l) = clickhouse_type(&AT::Int32, None, true, None).unwrap();
        assert_eq!(s, "Nullable(Int32)");
        assert_eq!(l, ChLayout::Nullable(Box::new(fixed(4, true, true))));
    }

    #[test]
    fn nested() {
        let list = AT::List(Arc::new(AField::new("item", AT::Int64, true)));
        let (s, l) = clickhouse_type(&list, None, false, None).unwrap();
        assert_eq!(s, "Array(Nullable(Int64))");
        assert_eq!(
            l,
            ChLayout::Array(Box::new(ChLayout::Nullable(Box::new(fixed(8, true, true)))))
        );
        // a nullable list stays Array(...)
        assert_eq!(
            clickhouse_type(&list, None, true, None).unwrap().0,
            "Array(Nullable(Int64))"
        );
        let fsl = AT::FixedSizeList(Arc::new(AField::new("item", AT::Utf8, false)), 3);
        assert_eq!(ty(&fsl), "Array(String)");
        let nested = AT::LargeList(Arc::new(AField::new("item", list, true)));
        assert_eq!(ty(&nested), "Array(Array(Nullable(Int64)))");
        let st = AT::Struct(Fields::from(vec![
            AField::new("a", AT::Int32, false),
            AField::new("b", AT::Utf8, false),
        ]));
        let (s, l) = clickhouse_type(&st, None, true, None).unwrap();
        assert_eq!(s, "Tuple(a Int32, b String)");
        assert_eq!(
            l,
            ChLayout::Tuple(vec![fixed(4, true, true), ChLayout::String])
        );
        let odd = AT::Struct(Fields::from(vec![AField::new("a b", AT::Int8, false)]));
        assert_eq!(ty(&odd), "Tuple(`a b` Int8)");
        let dl = AT::List(Arc::new(AField::new("item", dict(), true)));
        assert_eq!(ty(&dl), "Array(LowCardinality(Nullable(String)))");
    }

    #[test]
    fn no_clickhouse_type() {
        let two = timestamp_with_offset(TimeUnit::Second);
        let e = clickhouse_type(&two, None, false, None).unwrap_err();
        assert!(e.contains("timestamp_with_offset"));
        assert!(clickhouse_type(&AT::Null, None, false, None).is_err());
        assert!(clickhouse_type(&AT::Duration(TimeUnit::Second), None, false, None).is_err());
    }
}
