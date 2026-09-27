// ─────────────────────────────────────────────────────────────────────────────
// recommend — narrowest value-preserving Arrow type per column (Spec B)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-recommend-technique-design.md.
//
// Per column: Describe's profile (describe.rs) and Rust cardinality estimates
// (cardinality_estimators.rs) feed the step-1 type rules and the step-2
// dictionary rule, which emit candidate Arrow types with analytically predicted
// IPC sizes (§5.1). Candidates are tried smallest projected population size
// first, ties broken by hierarchy rank; each is cast, verified row by row
// against the original and measured with sizes.rs; the first that verifies is
// chosen. The original type is always a candidate and cannot fail.
//
// Everything below the plugin entry works on arrow-rs arrays (Series cross in
// through shared::to_arrow_rs), so moving the Python↔Rust boundary to Arrow
// tables later changes only the entry point.
//
// Leading-zero rule (Spec A §5.1): an integer-looking string with a leading zero
// ("007") must stay a String — identifiers such as UUID fragments, account
// numbers or zip codes can be all digits with significant leading zeros, and
// casting to an integer would lose them. A value with a decimal point
// ("007.50") is unlikely to be an identifier, so only numeric equivalence
// matters for it; differing leading or trailing zeros set rec_lossy_formatting.

use arrow_schema::{DataType as AT, Field as AField, Fields, TimeUnit};
use std::sync::Arc;

/// Hierarchy rank (Spec B §4.1): breaks ties between candidates of equal projected size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rank {
    Null,
    Boolean,
    UInt,
    Int,
    Decimal,
    Float,
    Date,
    Time,
    Timestamp,
    TimestampWithOffset,
    Dictionary,
    Plain,
    List,
    Original,
}

/// What a candidate casts a column (or a list's inner values) to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Target {
    Null,
    Boolean,
    /// A string column of two values: (true text, false text), lower-cased.
    BoolPair(String, String),
    /// A fixed-width type: integers, decimals, floats, Date32, Time32/64, Timestamp, Duration.
    Fixed(AT),
    TimestampWithOffset(TimeUnit),
    /// Dictionary-encoded strings: (Arrow key, Polars key — Polars reserves one code).
    Dictionary(AT, AT),
    /// Utf8 or Binary with 32-bit offsets.
    Plain(AT),
    /// The column's own type (Arrow classic layout): always a candidate, cannot fail.
    Original(AT),
    /// Every non-null list holds exactly one item → that item.
    Scalar(Box<Target>),
    List(Box<Target>),
    FixedList(Box<Target>, i32),
}

impl Target {
    pub(crate) fn arrow_type(&self) -> AT {
        let item = |t: &Target| Arc::new(AField::new("item", t.arrow_type(), true));
        match self {
            Target::Null => AT::Null,
            Target::Boolean | Target::BoolPair(..) => AT::Boolean,
            Target::Fixed(t) | Target::Plain(t) | Target::Original(t) => t.clone(),
            Target::TimestampWithOffset(u) => timestamp_with_offset(*u),
            Target::Dictionary(k, _) => AT::Dictionary(Box::new(k.clone()), Box::new(AT::Utf8)),
            Target::Scalar(t) => t.arrow_type(),
            Target::List(t) => AT::List(item(t)),
            Target::FixedList(t, w) => AT::FixedSizeList(item(t), *w),
        }
    }

    /// The Polars key of a dictionary anywhere in this target.
    pub(crate) fn polars_key(&self) -> Option<AT> {
        match self {
            Target::Dictionary(_, k) => Some(k.clone()),
            Target::Scalar(t) | Target::List(t) | Target::FixedList(t, _) => t.polars_key(),
            _ => None,
        }
    }
}

pub(crate) fn decimal_type(precision: u8, scale: i8) -> AT {
    match precision {
        0..=9 => AT::Decimal32(precision, scale),
        10..=18 => AT::Decimal64(precision, scale),
        _ => AT::Decimal128(precision, scale),
    }
}

/// Storage of Arrow's canonical extension `arrow.timestamp_with_offset`.
pub(crate) fn timestamp_with_offset(unit: TimeUnit) -> AT {
    AT::Struct(Fields::from(vec![
        AField::new("timestamp", AT::Timestamp(unit, Some("UTC".into())), false),
        AField::new("offset_minutes", AT::Int16, false),
    ]))
}

/// Dictionary key widths for cardinality `c`: (Arrow, Polars). Arrow indexes
/// 0..=255 with UInt8; Polars reserves one code (an Enum of 256 categories is UInt16).
pub(crate) fn dictionary_keys(c: f64) -> (AT, AT) {
    let arrow = if c <= 256.0 { AT::UInt8 } else if c <= 65_536.0 { AT::UInt16 } else { AT::UInt32 };
    let polars = if c <= 255.0 { AT::UInt8 } else if c <= 65_535.0 { AT::UInt16 } else { AT::UInt32 };
    (arrow, polars)
}

fn unit_name(u: &TimeUnit) -> &'static str {
    match u {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

/// pyarrow's `str(type)` spelling.
pub(crate) fn pa_name(t: &AT) -> String {
    match t {
        AT::Null => "null".into(),
        AT::Boolean => "bool".into(),
        AT::Int8 => "int8".into(),
        AT::Int16 => "int16".into(),
        AT::Int32 => "int32".into(),
        AT::Int64 => "int64".into(),
        AT::UInt8 => "uint8".into(),
        AT::UInt16 => "uint16".into(),
        AT::UInt32 => "uint32".into(),
        AT::UInt64 => "uint64".into(),
        AT::Float16 => "halffloat".into(),
        AT::Float32 => "float".into(),
        AT::Float64 => "double".into(),
        AT::Decimal32(p, s) => format!("decimal32({p}, {s})"),
        AT::Decimal64(p, s) => format!("decimal64({p}, {s})"),
        AT::Decimal128(p, s) => format!("decimal128({p}, {s})"),
        AT::Date32 => "date32[day]".into(),
        AT::Date64 => "date64[ms]".into(),
        AT::Time32(u) => format!("time32[{}]", unit_name(u)),
        AT::Time64(u) => format!("time64[{}]", unit_name(u)),
        AT::Timestamp(u, None) => format!("timestamp[{}]", unit_name(u)),
        AT::Timestamp(u, Some(tz)) => format!("timestamp[{}, tz={tz}]", unit_name(u)),
        AT::Duration(u) => format!("duration[{}]", unit_name(u)),
        AT::Utf8 => "string".into(),
        AT::LargeUtf8 => "large_string".into(),
        AT::Utf8View => "string_view".into(),
        AT::Binary => "binary".into(),
        AT::LargeBinary => "large_binary".into(),
        AT::BinaryView => "binary_view".into(),
        AT::List(f) => format!("list<{}: {}>", f.name(), pa_name(f.data_type())),
        AT::LargeList(f) => format!("large_list<{}: {}>", f.name(), pa_name(f.data_type())),
        AT::FixedSizeList(f, w) => format!("fixed_size_list<{}: {}>[{w}]", f.name(), pa_name(f.data_type())),
        AT::Dictionary(k, v) => format!("dictionary<values={}, indices={}, ordered=0>", pa_name(v), pa_name(k)),
        AT::Struct(fs) => format!(
            "struct<{}>",
            fs.iter()
                .map(|f| format!("{}: {}{}", f.name(), pa_name(f.data_type()), if f.is_nullable() { "" } else { " not null" }))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other}"),
    }
}

fn py_str(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Python `str(dtype)` of the Polars type that holds `t`. A dictionary is an Enum of
/// `enum_values` when given (exact cardinality), else a Categorical named after
/// `column` whose physical type is `key`.
pub(crate) fn pl_name(t: &AT, column: &str, enum_values: Option<&[String]>, key: &AT) -> String {
    let unit = |u: &TimeUnit| match u {
        TimeUnit::Second | TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    };
    let inner = |t: &AT| pl_name(t, column, enum_values, key);
    match t {
        AT::Null => "Null".into(),
        AT::Boolean => "Boolean".into(),
        AT::Int8 => "Int8".into(),
        AT::Int16 => "Int16".into(),
        AT::Int32 => "Int32".into(),
        AT::Int64 => "Int64".into(),
        AT::UInt8 => "UInt8".into(),
        AT::UInt16 => "UInt16".into(),
        AT::UInt32 => "UInt32".into(),
        AT::UInt64 => "UInt64".into(),
        AT::Float32 => "Float32".into(),
        AT::Float64 => "Float64".into(),
        AT::Decimal32(p, s) | AT::Decimal64(p, s) => format!("Decimal(precision={p}, scale={s})"),
        AT::Decimal128(p, s) => format!("Decimal(precision={p}, scale={s})"),
        AT::Date32 => "Date".into(),
        AT::Time32(_) | AT::Time64(_) => "Time".into(),
        AT::Timestamp(u, tz) => format!(
            "Datetime(time_unit='{}', time_zone={})",
            unit(u),
            tz.as_ref().map_or("None".to_string(), |z| format!("'{z}'"))
        ),
        AT::Duration(u) => format!("Duration(time_unit='{}')", unit(u)),
        AT::Utf8 | AT::LargeUtf8 | AT::Utf8View => "String".into(),
        AT::Binary | AT::LargeBinary | AT::BinaryView => "Binary".into(),
        AT::Dictionary(..) => match enum_values {
            Some(v) => format!("Enum(categories=[{}])", v.iter().map(|s| py_str(s)).collect::<Vec<_>>().join(", ")),
            None => format!("Categorical(Categories(name=\"{column}\", namespace=\"\", physical=pl.{}))", inner(key)),
        },
        AT::List(f) | AT::LargeList(f) => format!("List({})", inner(f.data_type())),
        AT::FixedSizeList(f, w) => format!("Array({}, shape=({w},))", inner(f.data_type())),
        AT::Struct(fs) => format!(
            "Struct({{{}}})",
            fs.iter().map(|f| format!("'{}': {}", f.name(), inner(f.data_type()))).collect::<Vec<_>>().join(", ")
        ),
        other => format!("{other}"),
    }
}

/// The Arrow type Polars exports (CompatLevel::newest) for a column Polars holds as
/// the type recommended by `t` (Spec B §5.5); `key` is a dictionary's Polars key.
pub(crate) fn polars_layout(t: &AT, key: &AT) -> AT {
    let field = |f: &Arc<AField>| Arc::new(AField::new(f.name(), polars_layout(f.data_type(), key), f.is_nullable()));
    match t {
        AT::Decimal32(p, s) | AT::Decimal64(p, s) => AT::Decimal128(*p, *s),
        AT::Timestamp(TimeUnit::Second, tz) => AT::Timestamp(TimeUnit::Millisecond, tz.clone()),
        AT::Time32(_) | AT::Time64(_) => AT::Time64(TimeUnit::Nanosecond),
        AT::Duration(TimeUnit::Second) => AT::Duration(TimeUnit::Millisecond),
        AT::Utf8 | AT::LargeUtf8 => AT::Utf8View,
        AT::Binary | AT::LargeBinary => AT::BinaryView,
        AT::Dictionary(_, _) => AT::Dictionary(Box::new(key.clone()), Box::new(AT::Utf8View)),
        AT::List(f) | AT::LargeList(f) => AT::LargeList(field(f)),
        AT::FixedSizeList(f, w) => AT::FixedSizeList(field(f), *w),
        AT::Struct(fs) => AT::Struct(fs.iter().map(field).collect()),
        other => other.clone(),
    }
}

// ── analytic sizes (Spec B §5.1, §5.3) ──────────────────────────────────────

/// What a predicted size depends on: rows, nulls, value bytes, distinct values and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Shape {
    pub n: f64,
    pub nulls: f64,
    pub sum_len: f64,
    pub d: f64,
    pub sum_len_unique: f64,
}

impl Shape {
    /// The population the frame samples: row-proportional terms scale by `r`; a
    /// dictionary holds `c` values of the observed mean length.
    pub(crate) fn project(&self, r: f64, c: f64) -> Shape {
        let per_value = if self.d > 0.0 { self.sum_len_unique / self.d } else { 0.0 };
        Shape { n: self.n * r, nulls: self.nulls * r, sum_len: self.sum_len * r, d: c, sum_len_unique: per_value * c }
    }
}

pub(crate) fn pad(x: f64) -> f64 {
    (x / 8.0).ceil() * 8.0
}

pub(crate) fn validity(n: f64, nulls: f64) -> f64 {
    if nulls > 0.0 { pad((n / 8.0).ceil()) } else { 0.0 }
}

/// Uncompressed Arrow IPC body bytes of a scalar type `t` holding values shaped
/// like `s`, exactly as sizes.rs measures it. Lists are sized by the caller.
pub(crate) fn body_size(t: &AT, s: &Shape) -> f64 {
    let v = validity(s.n, s.nulls);
    match t {
        AT::Null => 0.0,
        AT::Boolean => v + pad((s.n / 8.0).ceil()),
        AT::Utf8 | AT::Binary => v + pad(4.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::LargeUtf8 | AT::LargeBinary => v + pad(8.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::Dictionary(k, _) => {
            v + pad(s.n * k.primitive_width().unwrap() as f64) + pad(4.0 * (s.d + 1.0)) + pad(s.sum_len_unique)
        }
        AT::Struct(_) => v + pad(8.0 * s.n) + pad(2.0 * s.n), // timestamp_with_offset
        t => v + pad(s.n * t.primitive_width().unwrap_or(8) as f64),
    }
}

// ── numbers and units ────────────────────────────────────────────────────────

pub(crate) fn digits(v: u128) -> u8 {
    if v == 0 { 1 } else { (v.ilog10() + 1) as u8 }
}

pub(crate) fn narrowest_uint(hi: i128) -> Option<AT> {
    [(u8::MAX as i128, AT::UInt8), (u16::MAX as i128, AT::UInt16), (u32::MAX as i128, AT::UInt32), (u64::MAX as i128, AT::UInt64)]
        .into_iter()
        .find(|(max, _)| hi <= *max)
        .map(|(_, t)| t)
}

pub(crate) fn narrowest_int(lo: i128, hi: i128) -> Option<AT> {
    [
        (i8::MIN as i128, i8::MAX as i128, AT::Int8),
        (i16::MIN as i128, i16::MAX as i128, AT::Int16),
        (i32::MIN as i128, i32::MAX as i128, AT::Int32),
        (i64::MIN as i128, i64::MAX as i128, AT::Int64),
    ]
    .into_iter()
    .find(|(min, max, _)| *min <= lo && hi <= *max)
    .map(|(_, _, t)| t)
}

pub(crate) fn trailing_zeros10(mut g: i128) -> usize {
    let mut k = 0;
    while g != 0 && g % 10 == 0 {
        g /= 10;
        k += 1;
    }
    k
}

pub(crate) fn unit_ns(u: &TimeUnit) -> i128 {
    match u {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// The coarsest Arrow unit dividing `g_ns` nanoseconds (0 → seconds).
pub(crate) fn coarsest_unit(g_ns: i128) -> TimeUnit {
    match g_ns {
        0 => TimeUnit::Second,
        g if g % 1_000_000_000 == 0 => TimeUnit::Second,
        g if g % 1_000_000 == 0 => TimeUnit::Millisecond,
        g if g % 1_000 == 0 => TimeUnit::Microsecond,
        _ => TimeUnit::Nanosecond,
    }
}

/// Unit for ISO strings from their significant fractional-second digits.
pub(crate) fn iso_unit(sig: u32) -> TimeUnit {
    match sig {
        0 => TimeUnit::Second,
        1..=3 => TimeUnit::Millisecond,
        4..=6 => TimeUnit::Microsecond,
        _ => TimeUnit::Nanosecond,
    }
}

pub(crate) fn time_type(u: TimeUnit) -> AT {
    match u {
        TimeUnit::Second | TimeUnit::Millisecond => AT::Time32(u),
        _ => AT::Time64(u),
    }
}

/// Arrow time zone of a fixed offset in minutes: "UTC" or "±HH:MM".
pub(crate) fn offset_tz(minutes: i32) -> String {
    if minutes == 0 {
        return "UTC".into();
    }
    let sign = if minutes < 0 { '-' } else { '+' };
    format!("{sign}{:02}:{:02}", minutes.abs() / 60, minutes.abs() % 60)
}

/// A decimal or exponent string as (negative, significant digits, point) with
/// value = 0.DIGITS × 10^point; zero is (false, "", 0). Equal values give equal
/// tuples regardless of formatting ("1.50" ≡ "1.5", "0.00120" ≡ "1.2e-3").
pub(crate) fn canon(s: &str) -> (bool, String, i32) {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let (mantissa, exp) = s.split_once(['e', 'E']).map_or((s, 0), |(m, e)| (m, e.parse::<i32>().unwrap_or(0)));
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{int}{frac}");
    let lead = all.len() - all.trim_start_matches('0').len();
    let sig = all.trim_start_matches('0').trim_end_matches('0');
    if sig.is_empty() {
        return (false, String::new(), 0);
    }
    (neg, sig.to_string(), int.len() as i32 + exp - lead as i32)
}

/// Exact unscaled value of a decimal/exponent string at `scale`; None when it needs
/// more decimal places or more than 38 digits.
pub(crate) fn decimal_from_repr(repr: &str, scale: u32) -> Option<i128> {
    let (neg, sig, point) = canon(repr);
    if sig.is_empty() {
        return Some(0);
    }
    let places = sig.len() as i32 - point;
    if places > scale as i32 {
        return None;
    }
    let mut v: i128 = 0;
    for c in sig.bytes() {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow((scale as i32 - places) as u32)?)?;
    (v < 10i128.pow(38)).then_some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizes::ipc_body_bytes;
    use arrow_array::{builder::StringDictionaryBuilder, types::UInt8Type, DictionaryArray, StringArray};

    #[test]
    fn pyarrow_type_names() {
        assert_eq!(pa_name(&AT::Float32), "float");
        assert_eq!(pa_name(&decimal_type(6, 2)), "decimal32(6, 2)");
        assert_eq!(pa_name(&decimal_type(10, 3)), "decimal64(10, 3)");
        assert_eq!(pa_name(&AT::Timestamp(TimeUnit::Millisecond, Some("+05:00".into()))), "timestamp[ms, tz=+05:00]");
        assert_eq!(pa_name(&Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type()), "dictionary<values=string, indices=uint8, ordered=0>");
        assert_eq!(pa_name(&Target::List(Box::new(Target::Fixed(AT::UInt8))).arrow_type()), "list<item: uint8>");
        assert_eq!(pa_name(&timestamp_with_offset(TimeUnit::Second)), "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>");
    }

    #[test]
    fn polars_type_names() {
        let k = AT::UInt8;
        assert_eq!(pl_name(&decimal_type(6, 2), "x", None, &k), "Decimal(precision=6, scale=2)");
        assert_eq!(pl_name(&AT::Timestamp(TimeUnit::Second, None), "x", None, &k), "Datetime(time_unit='ms', time_zone=None)");
        assert_eq!(pl_name(&AT::Duration(TimeUnit::Second), "x", None, &k), "Duration(time_unit='ms')");
        let dict = Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type();
        assert_eq!(pl_name(&dict, "x", Some(&["a".into(), "b".into()]), &k), "Enum(categories=['a', 'b'])");
        assert_eq!(pl_name(&dict, "x", None, &k), "Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))");
        assert_eq!(
            pl_name(&timestamp_with_offset(TimeUnit::Second), "x", None, &k),
            "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})"
        );
    }

    #[test]
    fn polars_layouts_and_key_widths() {
        assert_eq!(polars_layout(&AT::Time32(TimeUnit::Second), &AT::UInt8), AT::Time64(TimeUnit::Nanosecond));
        assert_eq!(polars_layout(&decimal_type(6, 2), &AT::UInt8), AT::Decimal128(6, 2));
        assert_eq!(dictionary_keys(256.0), (AT::UInt8, AT::UInt16));
        assert_eq!(dictionary_keys(255.0), (AT::UInt8, AT::UInt8));
        assert_eq!(dictionary_keys(65_537.0).0, AT::UInt32);
    }

    #[test]
    fn predicted_sizes_match_sizes_rs() {
        let s = StringArray::from(vec![Some("ab"), None]);
        let shape = Shape { n: 2.0, nulls: 1.0, sum_len: 2.0, d: 1.0, sum_len_unique: 2.0 };
        assert_eq!(body_size(&AT::Utf8, &shape), ipc_body_bytes(&s, None).unwrap() as f64);
        let mut builder = StringDictionaryBuilder::<UInt8Type>::new();
        for v in ["a", "b", "a"] {
            builder.append_value(v);
        }
        let d: DictionaryArray<UInt8Type> = builder.finish();
        let shape = Shape { n: 3.0, nulls: 0.0, sum_len: 3.0, d: 2.0, sum_len_unique: 2.0 };
        assert_eq!(body_size(&Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type(), &shape), ipc_body_bytes(&d, None).unwrap() as f64);
    }

    #[test]
    fn canonical_decimals() {
        assert_eq!(canon("1.50"), canon("1.5"));
        assert_eq!(canon("0.00120"), canon("1.2e-3"));
        assert_eq!(canon("-0.0"), canon("0"));
        assert_ne!(canon("123"), canon("12.3"));
        assert_eq!(decimal_from_repr("1e-7", 7), Some(1));
        assert_eq!(decimal_from_repr("1.5e20", 0), Some(150_000_000_000_000_000_000));
        assert_eq!(decimal_from_repr("123.45", 1), None);
        assert_eq!(decimal_from_repr("-0.5", 2), Some(-50));
    }

    #[test]
    fn units_and_widths() {
        assert_eq!(coarsest_unit(86_400_000_000_000), TimeUnit::Second);
        assert_eq!(coarsest_unit(120_000_000), TimeUnit::Millisecond);
        assert_eq!(narrowest_uint(255), Some(AT::UInt8));
        assert_eq!(narrowest_int(-200, 5), Some(AT::Int16));
        assert_eq!(narrowest_int(0, i128::from(u64::MAX)), None);
        assert_eq!(trailing_zeros10(1_200), 2);
        assert_eq!(offset_tz(-210), "-03:30");
    }
}
