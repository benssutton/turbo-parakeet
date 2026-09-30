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
// Everything below the entry works on arrow-rs arrays (Series cross in
// through arrow_io::export_series), so the Python↔Rust boundary is Arrow tables.
//
// Leading-zero rule (Spec A §5.1): an integer-looking string with a leading zero
// ("007") must stay a String — identifiers such as UUID fragments, account
// numbers or zip codes can be all digits with significant leading zeros, and
// casting to an integer would lose them. A value with a decimal point
// ("007.50") is unlikely to be an identifier, so only numeric equivalence
// matters for it; differing leading or trailing zeros set rec_lossy_formatting.

use crate::cardinality_estimators::{estimate, Estimate};
use crate::describe::{
    assemble, describe_one, fields, parse_decimal, parse_iso, Described, Profile, Row,
};
use crate::sizes::{classic_layout, ipc_body_bytes, sizes_of, Sizes, SIZE_FIELDS};
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int16Type};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array,
    FixedSizeListArray, Float32Array, Float64Array, Int16Array, LargeListArray, LargeStringArray,
    ListArray, StringViewArray, StructArray, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_cast::cast::{cast_with_options, CastOptions};
use arrow_schema::{DataType as AT, Field as AField, Fields, TimeUnit};
use polars::prelude::{
    polars_err, AnyValue, DataType as PT, Field as PField, Float64Chunked, IntoSeries,
    NewChunkedArray, PolarsResult, Series, StringChunked, StructChunked, UInt64Chunked,
};
use rayon::prelude::*;
use std::cell::OnceCell;
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
    Duration,
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
    let arrow = if c <= 256.0 {
        AT::UInt8
    } else if c <= 65_536.0 {
        AT::UInt16
    } else {
        AT::UInt32
    };
    let polars = if c <= 255.0 {
        AT::UInt8
    } else if c <= 65_535.0 {
        AT::UInt16
    } else {
        AT::UInt32
    };
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
        AT::FixedSizeList(f, w) => format!(
            "fixed_size_list<{}: {}>[{w}]",
            f.name(),
            pa_name(f.data_type())
        ),
        AT::Dictionary(k, v) => format!(
            "dictionary<values={}, indices={}, ordered=0>",
            pa_name(v),
            pa_name(k)
        ),
        AT::Struct(fs) => format!(
            "struct<{}>",
            fs.iter()
                .map(|f| format!(
                    "{}: {}{}",
                    f.name(),
                    pa_name(f.data_type()),
                    if f.is_nullable() { "" } else { " not null" }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other}"),
    }
}

/// Python `repr()` of a str, as Polars' `str(pl.Enum([...]))` renders each category:
/// double-quoted when the string holds `'` and no `"` (avoids escaping the apostrophe),
/// else single-quoted with `'` escaped; `\`, `\n`, `\r`, `\t` get their short escapes,
/// other control characters (`< 0x20` or `0x7f`) become `\xNN`.
fn py_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
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
            Some(v) => format!(
                "Enum(categories=[{}])",
                v.iter().map(|s| py_str(s)).collect::<Vec<_>>().join(", ")
            ),
            None => format!(
                "Categorical(Categories(name=\"{column}\", namespace=\"\", physical=pl.{}))",
                inner(key)
            ),
        },
        AT::List(f) | AT::LargeList(f) => format!("List({})", inner(f.data_type())),
        AT::FixedSizeList(f, w) => format!("Array({}, shape=({w},))", inner(f.data_type())),
        AT::Struct(fs) => format!(
            "Struct({{{}}})",
            fs.iter()
                .map(|f| format!("'{}': {}", f.name(), inner(f.data_type())))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other}"),
    }
}

/// The Arrow type Polars exports (CompatLevel::newest) for a column Polars holds as
/// the type recommended by `t` (Spec B §5.5); `key` is a dictionary's Polars key.
pub(crate) fn polars_layout(t: &AT, key: &AT) -> AT {
    let field = |f: &Arc<AField>| {
        Arc::new(AField::new(
            f.name(),
            polars_layout(f.data_type(), key),
            f.is_nullable(),
        ))
    };
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
        let per_value = if self.d > 0.0 {
            self.sum_len_unique / self.d
        } else {
            0.0
        };
        Shape {
            n: self.n * r,
            nulls: self.nulls * r,
            sum_len: self.sum_len * r,
            d: c,
            sum_len_unique: per_value * c,
        }
    }
}

pub(crate) fn pad(x: f64) -> f64 {
    (x / 8.0).ceil() * 8.0
}

pub(crate) fn validity(n: f64, nulls: f64) -> f64 {
    if nulls > 0.0 {
        pad((n / 8.0).ceil())
    } else {
        0.0
    }
}

/// Uncompressed Arrow IPC body bytes of a scalar type `t` holding values shaped
/// like `s`, exactly as sizes.rs measures it. Lists are sized by the caller; a type
/// with no analytic size is an error (its candidate fails), never a wrong size.
pub(crate) fn body_size(t: &AT, s: &Shape) -> Result<f64, String> {
    let v = validity(s.n, s.nulls);
    let width = |t: &AT| {
        t.primitive_width()
            .ok_or_else(|| format!("no predicted size for {}", pa_name(t)))
    };
    Ok(match t {
        AT::Null => 0.0,
        AT::Boolean => v + pad((s.n / 8.0).ceil()),
        AT::Utf8 | AT::Binary => v + pad(4.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::LargeUtf8 | AT::LargeBinary => v + pad(8.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::Dictionary(k, values) if **values == AT::Utf8 => {
            v + pad(s.n * width(k)? as f64) + pad(4.0 * (s.d + 1.0)) + pad(s.sum_len_unique)
        }
        AT::Struct(f) if matches!(f.first().map(|x| x.data_type()), Some(AT::Timestamp(u, _)) if *t == timestamp_with_offset(*u)) => {
            v + pad(8.0 * s.n) + pad(2.0 * s.n)
        }
        t => v + pad(s.n * width(t)? as f64),
    })
}

// ── numbers and units ────────────────────────────────────────────────────────

pub(crate) fn digits(v: u128) -> u8 {
    if v == 0 {
        1
    } else {
        (v.ilog10() + 1) as u8
    }
}

pub(crate) fn narrowest_uint(hi: i128) -> Option<AT> {
    [
        (u8::MAX as i128, AT::UInt8),
        (u16::MAX as i128, AT::UInt16),
        (u32::MAX as i128, AT::UInt32),
        (u64::MAX as i128, AT::UInt64),
    ]
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
/// canonical forms regardless of formatting ("1.50" ≡ "1.5", "0.00120" ≡ "1.2e-3").
/// The digits borrow the input (`head` then `tail`, the dot skipped): no allocation.
///
/// Precondition: `s` is a numeric literal already validated by describe.rs's
/// `scan_numeric` (or produced by Rust's own float formatter, e.g. ryu) — an
/// optional leading `-`, ASCII digits, at most one `.`, and an optional
/// `e`/`E`-led exponent. `canon` does not re-validate this; a non-conforming
/// `s` (e.g. embedded non-digit bytes in the mantissa) can produce digits that
/// are not purely ASCII, which `decimal_from_repr` below guards against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Canon<'a> {
    neg: bool,
    head: &'a [u8],
    tail: &'a [u8],
    point: i32,
}

impl Canon<'_> {
    fn digits(&self) -> impl Iterator<Item = u8> + '_ {
        self.head.iter().chain(self.tail).copied()
    }

    fn n_digits(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

impl PartialEq for Canon<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.neg == other.neg
            && self.point == other.point
            && self.n_digits() == other.n_digits()
            && self.digits().eq(other.digits())
    }
}

pub(crate) fn canon(s: &str) -> Canon<'_> {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let (mantissa, exp) = s
        .split_once(['e', 'E'])
        .map_or((s, 0), |(m, e)| (m, e.parse::<i32>().unwrap_or(0)));
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let (int, frac) = (int.as_bytes(), frac.as_bytes());
    let zeros = |b: &[u8]| b.iter().take_while(|&&c| c == b'0').count();
    let trim_end =
        |b: &'_ [u8]| -> usize { b.len() - b.iter().rev().take_while(|&&c| c == b'0').count() };
    let lead_int = zeros(int);
    let (head, tail, lead) = if lead_int == int.len() {
        // No significant integer digit: the digits start inside the fraction.
        let lead_frac = zeros(frac);
        let tail = &frac[lead_frac..];
        (&int[..0], &tail[..trim_end(tail)], lead_int + lead_frac)
    } else {
        let head = &int[lead_int..];
        let t = trim_end(frac);
        if t > 0 {
            (head, &frac[..t], lead_int)
        } else {
            (&head[..trim_end(head)], &frac[..0], lead_int)
        }
    };
    if head.is_empty() && tail.is_empty() {
        return Canon {
            neg: false,
            head,
            tail,
            point: 0,
        };
    }
    Canon {
        neg,
        head,
        tail,
        point: int.len() as i32 + exp - lead as i32,
    }
}

/// Exact unscaled value of a decimal/exponent string at `scale`; None when it needs
/// more decimal places or more than 38 digits. Same precondition as `canon`; also
/// returns None (rather than underflowing the `c - b'0'` subtraction) if `canon`
/// ever hands back a non-digit byte, which a conforming input cannot do.
pub(crate) fn decimal_from_repr(repr: &str, scale: u32) -> Option<i128> {
    let c = canon(repr);
    if c.n_digits() == 0 {
        return Some(0);
    }
    let places = c.n_digits() as i32 - c.point;
    if places > scale as i32 {
        return None;
    }
    let mut v: i128 = 0;
    for d in c.digits() {
        if !d.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((d - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow((scale as i32 - places) as u32)?)?;
    (v < 10i128.pow(38)).then_some(if c.neg { -v } else { v })
}

// ── candidates (Spec B §4, §5.2) ────────────────────────────────────────────

/// Recommend's constructor keywords.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    pub seed: u64,
    pub zstd_level: i32,
    pub population_rows: Option<u64>,
    pub categorical_threshold: u64,
    pub boolean_pairs: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Outcome {
    Chosen,
    Failed,
    Rejected,
    NotTried,
}

impl Outcome {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Outcome::Chosen => "chosen",
            Outcome::Failed => "failed",
            Outcome::Rejected => "rejected",
            Outcome::NotTried => "not_tried",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub target: Target,
    pub rank: Rank,
    pub rule: String,
    /// The metric values the rule tested and what they implied.
    pub evidence: String,
    pub predicted: u64,
    pub projected: f64,
    pub outcome: Outcome,
    pub reason: Option<String>,
}

/// One level of a column: the column, or a list's inner values.
pub(crate) struct Level<'a> {
    /// Polars dtype of these values.
    pub dtype: &'a PT,
    /// The values in Arrow's classic layout (CompatLevel::oldest): the whole level
    /// (one-shot), a sample block, or none (streaming) — the rules read only its type.
    pub values: ArrayRef,
    pub p: &'a Profile,
    pub n_rows: u64,
    pub n_null: u64,
    /// Exact min / max: integers, or a decimal's unscaled values.
    pub int_range: Option<(i128, i128)>,
    /// Exact min / max of a float level, NaN excluded.
    pub float_range: Option<(f64, f64)>,
    /// Every distinct non-null value as text, when there are at most five (the
    /// boolean-pair rule); empty otherwise.
    pub few_distinct: Vec<String>,
    pub n_midnight: Option<u64>,
    /// Uncompressed size of the original type.
    pub size_bytes: u64,
    /// How `size_bytes` was obtained, for the original candidate's evidence.
    pub size_note: &'static str,
    pub est: Estimate,
    /// Population rows ÷ frame rows.
    pub r: f64,
    /// "" or "inner: " — prefixes every rule name.
    pub prefix: &'static str,
    /// Text sources: `values` as LargeUtf8, built once and shared by cast, verify and lossy.
    pub text: OnceCell<Result<LargeStringArray, String>>,
}

impl<'a> Level<'a> {
    /// A level over all its values (one-shot): counts, extremes and the few distinct
    /// values are read off `values` at Describe's argmin / argmax / top-5 rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of_values(
        dtype: &'a PT,
        values: ArrayRef,
        p: &'a Profile,
        n_midnight: Option<u64>,
        size_bytes: u64,
        est: Estimate,
        r: f64,
        prefix: &'static str,
    ) -> Result<Self, String> {
        let row = |i: Option<u64>| i.filter(|&i| (i as usize) < values.len());
        let (lo, hi) = (row(p.range.argmin), row(p.range.argmax));
        let int_range = match (dtype, lo, hi) {
            (PT::Decimal(..), Some(a), Some(b)) => values
                .as_primitive_opt::<Decimal128Type>()
                .map(|d| (d.value(a as usize), d.value(b as usize))),
            (dt, Some(a), Some(b)) if dt.is_integer() => {
                int_at(&values, a).zip(int_at(&values, b))
            }
            _ => None,
        };
        let float_range = match (is_float(dtype), lo, hi) {
            (true, Some(a), Some(b)) => Some((f64_at(&values, a), f64_at(&values, b))),
            _ => None,
        };
        let few_distinct = if is_text(dtype) && p.freq.n_unique <= 5 {
            p.freq
                .top5_idx
                .iter()
                .map(|&i| text_of(&values.slice(i as usize, 1)).map(|t| t.value(0).to_string()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        Ok(Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range,
            float_range,
            few_distinct,
            n_midnight,
            size_bytes,
            size_note: "measured",
            est,
            r,
            prefix,
            text: Default::default(),
        })
    }

    /// A level over one sample block, for `cast_to` / `verify` only (streaming's cross-check).
    pub(crate) fn of_block(dtype: &'a PT, values: ArrayRef, p: &'a Profile) -> Self {
        Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range: None,
            float_range: None,
            few_distinct: Vec::new(),
            n_midnight: None,
            size_bytes: 0,
            size_note: "",
            est: estimate(0, 0, 0, 0, &[0; 7], None),
            r: 1.0,
            prefix: "",
            text: Default::default(),
        }
    }
}

impl Level<'_> {
    /// The values as text (LargeUtf8), computed on first use.
    fn text(&self) -> Result<&LargeStringArray, String> {
        self.text
            .get_or_init(|| text_of(&self.values))
            .as_ref()
            .map_err(|e| e.clone())
    }

    fn n_rows(&self) -> u64 {
        self.n_rows
    }

    fn n_null(&self) -> u64 {
        self.n_null
    }

    fn n(&self) -> u64 {
        self.n_rows() - self.n_null()
    }

    /// Population cardinality for dictionaries: est_high where an interval exists,
    /// floored at the observed distinct count (an estimate must never claim fewer
    /// distinct values than were actually observed).
    fn cardinality(&self) -> (f64, &'static str) {
        let (c, source) = match self.est.est_high {
            Some(h) => (h, "est_high"),
            None => (self.est.est_cardinality, "est_cardinality"),
        };
        (c.max(self.p.freq.n_unique as f64), source)
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape {
            n: self.n_rows() as f64,
            nulls: self.n_null() as f64,
            sum_len: self.p.sum_len.unwrap_or(0) as f64,
            d: self.p.freq.n_unique as f64,
            sum_len_unique: self.p.freq.sum_len_unique.unwrap_or(0) as f64,
        }
    }
}

pub(crate) fn level_estimate(p: &Profile, n: u64, q: Option<f64>) -> Estimate {
    estimate(
        p.freq.n_unique,
        n,
        p.freq.f1,
        p.freq.f2,
        &p.freq.capture_history,
        q,
    )
}

/// arrow-cast with `safe: false`: a value that does not fit is an error, not a null.
pub(crate) fn arrow_cast(a: &dyn Array, to: &AT) -> Result<ArrayRef, String> {
    cast_with_options(
        a,
        to,
        &CastOptions {
            safe: false,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn is_text(dt: &PT) -> bool {
    matches!(dt, PT::String | PT::Categorical(..) | PT::Enum(..))
}

pub(crate) fn is_float(dt: &PT) -> bool {
    matches!(dt, PT::Float32 | PT::Float64)
}

pub(crate) fn text_of(values: &ArrayRef) -> Result<LargeStringArray, String> {
    Ok(arrow_cast(values.as_ref(), &AT::LargeUtf8)?
        .as_string::<i64>()
        .clone())
}

/// Row `i` as an exact integer (integer columns).
fn int_at(a: &ArrayRef, i: u64) -> Option<i128> {
    let v = arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Decimal128(38, 0)).ok()?;
    let v = v.as_primitive::<Decimal128Type>();
    v.is_valid(0).then(|| v.value(0))
}

fn f64_at(a: &ArrayRef, i: u64) -> f64 {
    arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Float64)
        .map_or(f64::NAN, |v| v.as_primitive::<Float64Type>().value(0))
}

struct Rules<'l, 'a> {
    lvl: &'l Level<'a>,
    shape: Shape,
    out: Vec<Candidate>,
}

impl Rules<'_, '_> {
    fn push(&mut self, target: Target, rank: Rank, rule: &str, evidence: String) {
        let t = target.arrow_type();
        let (c, _) = self.lvl.cardinality();
        let sizes = body_size(&t, &self.shape)
            .and_then(|p| Ok((p, body_size(&t, &self.shape.project(self.lvl.r, c))?)));
        self.out.push(candidate(
            target,
            rank,
            &format!("{}{rule}", self.lvl.prefix),
            evidence,
            sizes,
        ));
    }

    fn integers(&mut self, lo: i128, hi: i128, from: &str, evidence: &str) {
        let ev = format!("{evidence}min={lo} max={hi}");
        if lo >= 0 && hi <= 1 {
            self.push(
                Target::Boolean,
                Rank::Boolean,
                &format!("{from}→boolean"),
                ev.clone(),
            );
        }
        let uint = if lo >= 0 { narrowest_uint(hi) } else { None };
        let int = narrowest_int(lo, hi);
        if let Some(t) = uint.clone() {
            self.push(
                Target::Fixed(t),
                Rank::UInt,
                &format!("{from}→uint"),
                ev.clone(),
            );
        }
        if let Some(t) = int.clone() {
            self.push(
                Target::Fixed(t),
                Rank::Int,
                &format!("{from}→int"),
                ev.clone(),
            );
        }
        if uint.is_none() && int.is_none() {
            let p = digits(lo.unsigned_abs().max(hi.unsigned_abs()));
            if p <= 38 {
                self.push(
                    Target::Fixed(AT::Decimal128(p, 0)),
                    Rank::Decimal,
                    &format!("{from}→decimal128"),
                    format!("{ev} → p={p}"),
                );
            }
        }
    }

    fn decimal(&mut self, scale: usize) {
        let Some((lo, hi)) = self.lvl.int_range else {
            return;
        };
        let g = self.lvl.p.gcd.unwrap_or(1);
        let k = if g == 0 {
            scale
        } else {
            trailing_zeros10(g).min(scale)
        };
        let f = 10i128.pow(k as u32);
        let (lo, hi, s) = (lo / f, hi / f, scale - k);
        let ev = format!("gcd={g} → {k} trailing zeros, scale {scale}→{s}; ");
        if s == 0 {
            return self.integers(lo, hi, "decimal", &ev);
        }
        let prec = digits(lo.unsigned_abs().max(hi.unsigned_abs())).max(s as u8);
        if prec <= 38 {
            self.push(
                Target::Fixed(decimal_type(prec, s as i8)),
                Rank::Decimal,
                "decimal→decimal",
                format!("{ev}min={lo} max={hi} → p={prec} s={s}"),
            );
        }
    }

    fn float(&mut self) {
        let p = self.lvl.p;
        let Some(f) = p.floats else { return }; // Describe profiles every float column: never None
        let ev = format!(
            "n_nan={} n_inf={} n_fractional={} ",
            f.n_nan, f.n_inf, f.n_fractional
        );
        if let (0, 0, Some((lo, hi))) = (f.n_nan, f.n_inf, self.lvl.float_range) {
            let top = lo.abs().max(hi.abs());
            if f.n_fractional == 0 {
                if top < 1e38 {
                    self.integers(lo as i128, hi as i128, "float", &ev);
                }
            } else if let Some(s) = f.max_frac_digits {
                let int_digits = if top < 1.0 {
                    0
                } else {
                    digits(top.floor() as u128) as u32
                };
                let prec = int_digits + s;
                if prec <= 38 {
                    self.push(
                        Target::Fixed(decimal_type(prec as u8, s as i8)),
                        Rank::Decimal,
                        "float→decimal",
                        format!("{ev}max_frac_digits={s} int_digits={int_digits} → p={prec} s={s}"),
                    );
                }
            }
        }
        if self.lvl.dtype == &PT::Float64 && f.n_f32_inexact == 0 {
            self.push(
                Target::Fixed(AT::Float32),
                Rank::Float,
                "float64→float32",
                "n_f32_inexact=0".into(),
            );
        }
    }

    fn temporal(&mut self) {
        let lvl = self.lvl;
        let (unit, tz) = match lvl.values.data_type() {
            AT::Timestamp(u, tz) => (*u, tz.clone()),
            AT::Duration(u) | AT::Time64(u) | AT::Time32(u) => (*u, None),
            _ => return,
        };
        let g = lvl.p.gcd.unwrap_or(1);
        let coarse = coarsest_unit(g * unit_ns(&unit));
        let ev = format!("gcd={g} ({}) → {}", unit_name(&unit), unit_name(&coarse));
        match lvl.dtype {
            PT::Datetime(_, zone) => {
                if zone.is_none() && lvl.n_midnight == Some(lvl.n()) {
                    self.push(
                        Target::Fixed(AT::Date32),
                        Rank::Date,
                        "datetime→date32",
                        format!("n_midnight={}", lvl.n()),
                    );
                }
                if coarse != unit {
                    self.push(
                        Target::Fixed(AT::Timestamp(coarse, tz)),
                        Rank::Timestamp,
                        "datetime→timestamp",
                        ev,
                    );
                }
            }
            PT::Duration(_) if coarse != unit => self.push(
                Target::Fixed(AT::Duration(coarse)),
                Rank::Duration,
                "duration→duration",
                ev,
            ),
            PT::Time if coarse != unit => self.push(
                Target::Fixed(time_type(coarse)),
                Rank::Time,
                "time→time",
                ev,
            ),
            _ => {}
        }
    }

    fn text(&mut self, params: &Params) -> Result<(), String> {
        let lvl = self.lvl;
        let n = lvl.n();
        let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 {
            lvl.few_distinct.iter().map(|v| v.to_lowercase()).collect()
        } else {
            Vec::new()
        };
        let pair = params
            .boolean_pairs
            .iter()
            .map(|(t, f)| (t.to_lowercase(), f.to_lowercase()))
            .find(|(t, f)| !distinct.is_empty() && distinct.iter().all(|v| v == t || v == f));
        // Describe profiles every text column, so `strings` is always present; without
        // it only the plain / dictionary candidates apply.
        let Some(st) = lvl.p.strings.as_ref() else {
            self.plain_and_dictionary(params);
            return Ok(());
        };
        let sig = st.iso_max_sig_frac_digits.unwrap_or(0);
        if let Some((t, f)) = pair {
            self.push(
                Target::BoolPair(t.clone(), f.clone()),
                Rank::Boolean,
                "string→boolean",
                format!("distinct={distinct:?} pair=({t:?}, {f:?})"),
            );
        } else if let (true, Some(lo), Some(hi)) = (
            st.n_numeric_int == n && st.n_leading_zero == 0 && !st.int_overflow,
            st.int_min,
            st.int_max,
        ) {
            self.integers(
                lo,
                hi,
                "string",
                &format!("n_numeric_int={n} n_leading_zero=0 "),
            );
        } else if st.n_numeric == n && st.n_leading_zero == 0 {
            let (i, f) = (
                st.max_int_digits.unwrap_or(0),
                st.max_frac_digits.unwrap_or(0),
            );
            let (min_f, sig_d) = (
                st.min_frac_digits.unwrap_or(0),
                st.max_sig_digits.unwrap_or(0),
            );
            let prec = (i + f).max(1);
            let ev = format!(
                "n_numeric={n} n_leading_zero=0 numeric_max_int_digits={i} numeric_max_frac_digits={f} \
                 numeric_min_frac_digits={min_f} numeric_max_sig_digits={sig_d}"
            );
            if prec <= 38 {
                self.push(
                    Target::Fixed(decimal_type(prec as u8, f as i8)),
                    Rank::Decimal,
                    "string→decimal",
                    format!("{ev} → p={prec} s={f}"),
                );
            }
            if min_f < f && prec > 18 {
                let why = format!("{ev} → varying places, p={prec} > 18");
                if sig_d <= 6 {
                    self.push(
                        Target::Fixed(AT::Float32),
                        Rank::Float,
                        "string→float32",
                        why.clone(),
                    );
                }
                if sig_d <= 15 {
                    self.push(
                        Target::Fixed(AT::Float64),
                        Rank::Float,
                        "string→float64",
                        why,
                    );
                }
            }
        } else if st.n_iso_date == n {
            self.push(
                Target::Fixed(AT::Date32),
                Rank::Date,
                "string→date32",
                format!("n_iso_date={n}"),
            );
        } else if st.n_iso_time == n {
            self.push(
                Target::Fixed(time_type(iso_unit(sig))),
                Rank::Time,
                "string→time",
                format!("n_iso_time={n} iso_max_sig_frac_digits={sig}"),
            );
        } else if st.n_iso_datetime == n {
            if st.iso_n_midnight == n {
                self.push(
                    Target::Fixed(AT::Date32),
                    Rank::Date,
                    "string→date32",
                    format!("n_iso_datetime={n} iso_n_midnight={n}"),
                );
            } else {
                self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), None)),
                    Rank::Timestamp,
                    "string→timestamp",
                    format!("n_iso_datetime={n} iso_max_sig_frac_digits={sig}"),
                );
            }
        } else if st.n_iso_datetime_tz == n {
            let ev = format!(
                "n_iso_datetime_tz={n} iso_n_offsets={} iso_max_sig_frac_digits={sig}",
                st.offsets.len()
            );
            match st.offsets.iter().next() {
                Some(&m) if st.offsets.len() == 1 => self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), Some(offset_tz(m).into()))),
                    Rank::Timestamp,
                    "string→timestamp",
                    ev,
                ),
                _ => self.push(
                    Target::TimestampWithOffset(iso_unit(sig)),
                    Rank::TimestampWithOffset,
                    "string→timestamp_with_offset (arrow.timestamp_with_offset)",
                    ev,
                ),
            }
        }
        self.plain_and_dictionary(params);
        Ok(())
    }

    /// Step 2 for text (Spec B §4.3 "always", §5.2): Utf8 with 32-bit offsets, and the dictionary.
    fn plain_and_dictionary(&mut self, params: &Params) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(
                Target::Plain(AT::Utf8),
                Rank::Plain,
                "string→utf8",
                format!("sum_len={sum_len}"),
            );
        }
        self.dictionary(params);
    }

    fn dictionary(&mut self, params: &Params) {
        let (c, source) = self.lvl.cardinality();
        let (key, polars_key) = dictionary_keys(c);
        let threshold = params.categorical_threshold;
        let est = &self.lvl.est;
        let low = est
            .est_low
            .map_or(String::new(), |l| format!(" est_low={l:?}"));
        let ev = format!(
            "c={c:?} from {source} method={}{low} n_unique={} categorical_threshold={threshold}",
            est.method.name(),
            self.lvl.p.freq.n_unique
        );
        self.push(
            Target::Dictionary(key, polars_key),
            Rank::Dictionary,
            "string→dictionary",
            ev,
        );
        if c > threshold as f64 {
            let last = self.out.last_mut().unwrap();
            last.outcome = Outcome::Rejected;
            last.reason = Some(format!("c={c:?} > categorical_threshold={threshold}"));
        }
    }

    fn binary(&mut self) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(
                Target::Plain(AT::Binary),
                Rank::Plain,
                "binary→binary",
                format!("sum_len={sum_len}"),
            );
        }
    }

    fn original(&mut self) {
        let lvl = self.lvl;
        self.out.push(Candidate {
            target: Target::Original(lvl.values.data_type().clone()),
            rank: Rank::Original,
            rule: format!("{}original", lvl.prefix),
            evidence: format!("{} size_bytes={}", lvl.size_note, lvl.size_bytes),
            predicted: lvl.size_bytes,
            projected: lvl.size_bytes as f64 * lvl.r,
            outcome: Outcome::NotTried,
            reason: None,
        });
    }
}

/// Every candidate for one level, in rule order (Spec B §4.2–4.3, §5.2); the
/// original type last.
pub(crate) fn candidates(lvl: &Level, params: &Params) -> Result<Vec<Candidate>, String> {
    let mut r = Rules {
        lvl,
        shape: lvl.shape(),
        out: Vec::new(),
    };
    if lvl.n_rows() > 0 && lvl.n() == 0 {
        r.push(
            Target::Null,
            Rank::Null,
            "all-null→null",
            format!("n_null={} n_rows={}", lvl.n_null(), lvl.n_rows()),
        );
    } else if lvl.n() > 0 {
        match lvl.dtype {
            PT::Int8
            | PT::Int16
            | PT::Int32
            | PT::Int64
            | PT::UInt8
            | PT::UInt16
            | PT::UInt32
            | PT::UInt64 => {
                if let Some((lo, hi)) = lvl.int_range {
                    r.integers(lo, hi, "integer", "");
                }
            }
            PT::Decimal(_, s) => r.decimal(s.unwrap_or(0)),
            PT::Float32 | PT::Float64 => r.float(),
            PT::Datetime(..) | PT::Duration(_) | PT::Time => r.temporal(),
            dt if is_text(dt) => r.text(params)?,
            PT::Binary => r.binary(),
            _ => {}
        }
    }
    r.original();
    Ok(r.out)
}

// ── cast and verify (Spec B §5.4) ───────────────────────────────────────────

fn exact_div(ns: i128, u: &TimeUnit) -> Option<i64> {
    let f = unit_ns(u);
    (ns % f == 0)
        .then(|| ns / f)
        .and_then(|v| i64::try_from(v).ok())
}

fn time_array(u: TimeUnit, v: Vec<Option<i64>>) -> ArrayRef {
    let narrow = || v.iter().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>();
    match u {
        TimeUnit::Second => Arc::new(Time32SecondArray::from(narrow())),
        TimeUnit::Millisecond => Arc::new(Time32MillisecondArray::from(narrow())),
        TimeUnit::Microsecond => Arc::new(Time64MicrosecondArray::from(v)),
        TimeUnit::Nanosecond => Arc::new(Time64NanosecondArray::from(v)),
    }
}

fn timestamp_array(u: TimeUnit, v: Vec<Option<i64>>, tz: Option<Arc<str>>) -> ArrayRef {
    match u {
        TimeUnit::Second => Arc::new(TimestampSecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(v).with_timezone_opt(tz)),
    }
}

fn decimal_array(v: Vec<Option<i128>>, scale: i8) -> Result<ArrayRef, String> {
    Decimal128Array::from(v)
        .with_precision_and_scale(38, scale)
        .map(|a| Arc::new(a) as ArrayRef)
        .map_err(|e| e.to_string())
}

/// `s.to_lowercase() == lower` for an already lower-cased `lower`, allocating only for
/// non-ASCII `s` (whose Unicode lower case can map onto ASCII, e.g. the Kelvin sign).
fn lower_eq(s: &str, lower: &str) -> bool {
    if s.is_ascii() {
        s.eq_ignore_ascii_case(lower)
    } else {
        s.to_lowercase() == lower
    }
}

/// `f` over every non-null text value; the first value it rejects fails the cast.
fn parsed<T>(
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

fn render(a: &ArrayRef, i: usize) -> String {
    arrow_cast(a.slice(i, 1).as_ref(), &AT::Utf8)
        .ok()
        .and_then(|s| {
            let s = s.as_string::<i32>();
            s.is_valid(0).then(|| s.value(0).to_string())
        })
        .unwrap_or_else(|| "null".into())
}

/// Rows compared per ArrayData equality check in `first_mismatch`.
const MISMATCH_CHUNK: usize = 1 << 16;

/// Ok when `a` and `b` hold equal values; else the first differing row. Compares
/// 64K-row chunks, then bisects the first failing chunk on prefix equality (a
/// prefix that differs stays different when extended) — O(log) comparisons, no
/// per-row slicing.
pub(crate) fn first_mismatch(a: &ArrayRef, b: &ArrayRef) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length {} ≠ {}", b.len(), a.len()));
    }
    let same =
        |start: usize, len: usize| a.slice(start, len).to_data() == b.slice(start, len).to_data();
    let Some(start) = (0..a.len())
        .step_by(MISMATCH_CHUNK)
        .find(|&s| !same(s, MISMATCH_CHUNK.min(a.len() - s)))
    else {
        return Ok(());
    };
    // Invariant: rows start..start+lo are equal; start..start+hi differ somewhere.
    let (mut lo, mut hi) = (0, MISMATCH_CHUNK.min(a.len() - start));
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if same(start, mid) {
            lo = mid
        } else {
            hi = mid
        }
    }
    let i = start + lo;
    Err(format!(
        "row {i}: {} round-trips to {}",
        render(a, i),
        render(b, i)
    ))
}

/// A decimal's rendered text parsed back as a float; a failure to parse is an error,
/// never silently folded into NaN (NaN is only ever a *value*, from a genuine decimal
/// text like "nan" — which cannot occur here since decimals never render one).
fn parse_back(s: &str, f32_src: bool) -> Result<f64, String> {
    if f32_src {
        s.parse::<f32>().map(f64::from)
    } else {
        s.parse::<f64>()
    }
    .map_err(|_| format!("{s:?} does not parse as a float"))
}

/// Float sources: every recast value converts back to the original float (NaN = NaN,
/// -0.0 = 0.0). Decimals come back through their text (correctly rounded parse).
pub(crate) fn verify_float(src: &ArrayRef, recast: &ArrayRef) -> Result<(), String> {
    let f32_src = src.data_type() == &AT::Float32;
    let decimal = matches!(
        recast.data_type(),
        AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..)
    );
    let back: Vec<Option<f64>> = if decimal {
        let t = arrow_cast(recast.as_ref(), &AT::Utf8)?;
        t.as_string::<i32>()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.map(|s| parse_back(s, f32_src).map_err(|e| format!("row {i}: {e}")))
                    .transpose()
            })
            .collect::<Result<_, _>>()?
    } else if f32_src {
        arrow_cast(recast.as_ref(), &AT::Float32)?
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| v.map(f64::from))
            .collect()
    } else {
        arrow_cast(recast.as_ref(), &AT::Float64)?
            .as_primitive::<Float64Type>()
            .iter()
            .collect()
    };
    let orig = arrow_cast(src.as_ref(), &AT::Float64)?;
    for (i, (a, b)) in orig
        .as_primitive::<Float64Type>()
        .iter()
        .zip(back)
        .enumerate()
    {
        if let (Some(a), Some(b)) = (a, b) {
            if !(a == b || (a.is_nan() && b.is_nan())) {
                return Err(format!("row {i}: {a} round-trips to {b}"));
            }
        }
    }
    Ok(())
}

/// String sources: the recast values, rendered to text by arrow-cast, equal the
/// original text by value (canonical digits; parse_iso components). Returns the
/// LargeUtf8 rendering when it made one, so `lossy` need not render again.
pub(crate) fn verify_text(
    t: &Target,
    text: &LargeStringArray,
    recast: &ArrayRef,
) -> Result<Option<ArrayRef>, String> {
    let bad = |i: usize, got: &str| {
        Err(format!(
            "row {i}: {:?} round-trips to {got:?}",
            text.value(i)
        ))
    };
    let iso = |s: &str| parse_iso(s.as_bytes());
    match t {
        Target::Dictionary(..) | Target::Plain(_) => first_mismatch(
            &(Arc::new(text.clone()) as ArrayRef),
            &arrow_cast(recast.as_ref(), &AT::LargeUtf8)?,
        )
        .map(|_| None),
        Target::Boolean | Target::BoolPair(..) => {
            let (tt, ff) = match t {
                Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
                _ => ("1", "0"),
            };
            let b = recast.as_boolean_opt().ok_or("recast is not boolean")?;
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let want = if b.value(i) { tt } else { ff };
                if !lower_eq(text.value(i), want) {
                    return bad(i, want);
                }
            }
            Ok(None)
        }
        Target::TimestampWithOffset(_) => {
            let s = recast.as_struct_opt().ok_or("recast is not a struct")?;
            let back = arrow_cast(s.column(0).as_ref(), &AT::Utf8)?;
            let (back, off) = (
                back.as_string::<i32>(),
                s.column(1).as_primitive::<Int16Type>(),
            );
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (iso(text.value(i)), iso(back.value(i)));
                let same = matches!((a, b), (Some(a), Some(b)) if a.epoch_ns() == b.epoch_ns() && a.offset_minutes == Some(off.value(i) as i32));
                if !same {
                    return bad(i, back.value(i));
                }
            }
            Ok(None)
        }
        Target::Fixed(to) => {
            let rendered = arrow_cast(recast.as_ref(), &AT::LargeUtf8)?;
            let back = rendered.as_string::<i64>();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (text.value(i), back.value(i));
                let same = match to {
                    AT::Date32 => {
                        matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns())
                    }
                    // A fixed-offset target renders the same offset as the text only when
                    // every value actually had one; a naive target renders no offset at all
                    // ("Z" and "UTC" both parse back to offset_minutes = Some(0), so this
                    // must compare the *parsed* offsets, not the rendered strings).
                    AT::Timestamp(_, tz) => match (iso(a), iso(b)) {
                        (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns() => match tz {
                            Some(_) => {
                                x.offset_minutes.is_some() && x.offset_minutes == y.offset_minutes
                            }
                            None => x.offset_minutes.is_none(),
                        },
                        _ => false,
                    },
                    AT::Time32(_) | AT::Time64(_) => {
                        matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.nanos == y.nanos)
                    }
                    _ => canon(a) == canon(b),
                };
                if !same {
                    return bad(i, b);
                }
            }
            Ok(Some(rendered))
        }
        _ => Ok(None),
    }
}

/// Whether row `i` is null "logically" — i.e. as `Array::logical_nulls` sees it, not
/// `Array::is_null`'s raw physical null buffer. They differ for `NullArray` (no physical
/// buffer at all, so `is_null` is always false despite every value being null) and for
/// Dictionary/Run/Union arrays whose nullability can live in a child array; every array
/// this module builds keeps its own physical buffer, but checking logical nulls
/// throughout keeps that assumption from becoming a silent correctness trap.
/// `nulls` is the array's `logical_nulls()`, computed once per array, not per row.
fn logical_is_null(nulls: Option<&NullBuffer>, i: usize) -> bool {
    nulls.is_some_and(|n| n.is_null(i))
}

/// Row-by-row check that `recast` holds the original values (Spec B §5.4 step 3).
/// Text sources: returns the recast values' LargeUtf8 rendering when verification made one.
pub(crate) fn verify(
    t: &Target,
    lvl: &Level,
    recast: &ArrayRef,
) -> Result<Option<ArrayRef>, String> {
    if matches!(t, Target::Original(_)) {
        return Ok(None);
    }
    let src = &lvl.values;
    if recast.len() != src.len() {
        return Err(format!("length {} ≠ {}", recast.len(), src.len()));
    }
    if matches!(t, Target::Null) {
        // NullArray carries no physical null buffer of its own — `Array::is_null` is
        // always false for it — so the only real check is that every source value was
        // actually null (via the source's own *logical* nulls).
        let non_null = src.len() - src.logical_null_count();
        return (non_null == 0)
            .then_some(None)
            .ok_or_else(|| format!("{non_null} non-null source value(s)"));
    }
    let (recast_nulls, src_nulls) = (recast.logical_nulls(), src.logical_nulls());
    let null_count = |n: &Option<NullBuffer>| n.as_ref().map_or(0, |n| n.null_count());
    if recast_nulls != src_nulls || null_count(&recast_nulls) != null_count(&src_nulls) {
        if let Some(i) = (0..src.len()).find(|&i| {
            logical_is_null(recast_nulls.as_ref(), i) != logical_is_null(src_nulls.as_ref(), i)
        }) {
            return Err(format!("row {i}: null mismatch"));
        }
    }
    match t {
        _ if is_text(lvl.dtype) => verify_text(t, lvl.text()?, recast),
        _ if is_float(lvl.dtype) => verify_float(src, recast).map(|_| None),
        _ => {
            // arrow-cast's cast-back below forces `recast` into `src`'s exact data type
            // (tz included), which would silently paper over a timezone that changed
            // along the way — Timestamp's physical storage (epoch units) doesn't depend
            // on tz, so the value comparison alone can't catch it.
            if let (AT::Timestamp(_, a), AT::Timestamp(_, b)) =
                (src.data_type(), recast.data_type())
            {
                if a != b {
                    return Err(format!("timezone changed: {a:?} → {b:?}"));
                }
            }
            first_mismatch(src, &arrow_cast(recast.as_ref(), src.data_type())?).map(|_| None)
        }
    }
}

/// Some value's text changed although its value did not (Spec B §5.4). `rendered`:
/// `recast` as LargeUtf8 if `verify` already made it.
pub(crate) fn lossy(
    t: &Target,
    lvl: &Level,
    recast: &ArrayRef,
    rendered: Option<ArrayRef>,
) -> bool {
    match t {
        Target::Original(_) | Target::Null | Target::Dictionary(..) | Target::Plain(_) => false,
        _ if is_text(lvl.dtype) => match (
            lvl.text(),
            rendered.map_or_else(|| arrow_cast(recast.as_ref(), &AT::LargeUtf8), Ok),
        ) {
            (Ok(a), Ok(b)) => a
                .iter()
                .zip(b.as_string::<i64>().iter())
                .any(|(x, y)| x != y),
            _ => true, // no text rendering (timestamp_with_offset): the format necessarily changes
        },
        Target::Fixed(AT::Float32 | AT::Float64) => false,
        _ if is_float(lvl.dtype) => arrow_cast(lvl.values.as_ref(), &AT::Float64)
            .map(|f| {
                f.as_primitive::<Float64Type>()
                    .iter()
                    .flatten()
                    .any(|x| x == 0.0 && x.is_sign_negative())
            })
            .unwrap_or(false),
        _ => false,
    }
}

// ── choosing (Spec B §4.1, §4.4, §5.4) ──────────────────────────────────────

pub(crate) struct Chosen {
    pub target: Target,
    pub rank: Rank,
    pub array: ArrayRef,
    pub lossy: bool,
    pub predicted: u64,
    pub projected: f64,
    pub candidates: Vec<Candidate>,
}

/// Candidates in the order tried: rejected last, then smallest projected size, then rank.
fn order(c: &mut [Candidate]) {
    c.sort_by(|a, b| {
        (a.outcome == Outcome::Rejected)
            .cmp(&(b.outcome == Outcome::Rejected))
            .then(a.projected.total_cmp(&b.projected))
            .then(a.rank.cmp(&b.rank))
    });
}

/// Tries the candidates in order; the first success is chosen, failures keep their
/// reason. Only `not_tried` candidates are attempted (rejected ones, and ones that
/// could not be sized, already carry their outcome).
pub(crate) fn first_success<T>(
    mut cands: Vec<Candidate>,
    mut attempt: impl FnMut(&Target) -> Result<T, String>,
) -> (usize, T, Vec<Candidate>) {
    order(&mut cands);
    for i in 0..cands.len() {
        if cands[i].outcome != Outcome::NotTried {
            continue;
        }
        match attempt(&cands[i].target) {
            Ok(a) => {
                cands[i].outcome = Outcome::Chosen;
                return (i, a, cands);
            }
            Err(reason) => {
                cands[i].outcome = Outcome::Failed;
                cands[i].reason = Some(reason);
            }
        }
    }
    unreachable!("the original type is always a candidate and cannot fail")
}

fn chosen_from(i: usize, array: ArrayRef, cands: Vec<Candidate>, lossy: bool) -> Chosen {
    let c = &cands[i];
    Chosen {
        target: c.target.clone(),
        rank: c.rank,
        lossy,
        predicted: c.predicted,
        projected: c.projected,
        array,
        candidates: cands,
    }
}

pub(crate) fn choose(lvl: &Level, params: &Params) -> Result<Chosen, String> {
    let (i, (array, rendered), cands) = first_success(candidates(lvl, params)?, |t| {
        let a = cast_to(t, lvl)?;
        let rendered = verify(t, lvl, &a)?;
        Ok((a, rendered))
    });
    let lossy = lossy(&cands[i].target, lvl, &array, rendered);
    Ok(chosen_from(i, array, cands, lossy))
}

/// The original type only; `why` explains, in its evidence, why nothing else was tried.
fn choose_original(lvl: &Level, why: &str) -> Chosen {
    let mut r = Rules {
        lvl,
        shape: lvl.shape(),
        out: Vec::new(),
    };
    r.original();
    r.out[0].evidence = format!("{}; {why}", r.out[0].evidence);
    let (i, array, cands) = first_success(r.out, |_| Ok(lvl.values.clone()));
    chosen_from(i, array, cands, false)
}

/// Per row of a List/Array column: (start, len) in the child the inner level holds,
/// or None for a null row; that child; the fixed width (None for List). The child is
/// Describe's `flatten` — the values of the non-null rows only, in row order — so the
/// inner profile's argmin/top5 indices point into it: valid row k starts where valid
/// row k−1 ends. A null row can still span values (a List's offsets after
/// `pl.when(mask).then(list).otherwise(None)`; an Array's w slots); those are dropped
/// with a `take`, and `wrap` rebuilds the offsets from row lengths (a null row → empty).
/// (row spans, compacted values, fixed width).
pub(crate) type ListParts = (Vec<Option<(usize, usize)>>, ArrayRef, Option<i32>);

pub(crate) fn list_parts(values: &ArrayRef) -> Result<ListParts, String> {
    // (physical start, len) of every row, then the compacted rows.
    let (spans, child, width): (Vec<(usize, usize, bool)>, ArrayRef, Option<i32>) =
        match values.data_type() {
            AT::LargeList(_) => {
                let l = values.as_list::<i64>();
                let o = l.value_offsets();
                let first = o[0] as usize;
                let spans = (0..l.len())
                    .map(|i| {
                        (
                            o[i] as usize - first,
                            (o[i + 1] - o[i]) as usize,
                            l.is_valid(i),
                        )
                    })
                    .collect();
                (
                    spans,
                    l.values().slice(first, o[l.len()] as usize - first),
                    None,
                )
            }
            AT::FixedSizeList(_, width) => {
                let f = values.as_fixed_size_list();
                let w = *width as usize;
                let child = f.values().slice(f.value_offset(0) as usize, f.len() * w); // offset non-zero for a slice
                (
                    (0..f.len()).map(|i| (i * w, w, f.is_valid(i))).collect(),
                    child,
                    Some(*width),
                )
            }
            t => return Err(format!("not a list type: {}", pa_name(t))),
        };
    let mut k = 0;
    let rows = spans
        .iter()
        .map(|&(_, len, valid)| {
            valid.then(|| {
                k += len;
                (k - len, len)
            })
        })
        .collect();
    if !spans.iter().any(|&(_, len, valid)| !valid && len > 0) {
        return Ok((rows, child, width)); // no null row spans values: the child is already compact
    }
    let idx = UInt64Array::from_iter_values(
        spans
            .iter()
            .filter(|s| s.2)
            .flat_map(|&(start, len, _)| (start..start + len).map(|j| j as u64)),
    );
    let compact =
        arrow_select::take::take(child.as_ref(), &idx, None).map_err(|e| e.to_string())?;
    Ok((rows, compact, width))
}

/// `take` whose null indices give null rows; a struct keeps its nulls at struct
/// level only, its children null-free (as `from_text` builds timestamp_with_offset
/// and as `body_size` predicts it).
fn take_rows(a: &ArrayRef, idx: &UInt64Array) -> Result<ArrayRef, String> {
    let e = |e: arrow_schema::ArrowError| e.to_string();
    match a.data_type() {
        AT::Struct(fields) if !a.is_empty() => {
            let s = a.as_struct();
            let dense = UInt64Array::from_iter_values(idx.iter().map(|i| i.unwrap_or(0)));
            let cols = s
                .columns()
                .iter()
                .map(|c| arrow_select::take::take(c.as_ref(), &dense, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(e)?;
            let valid: Vec<bool> = idx
                .iter()
                .map(|i| i.is_some_and(|i| s.is_valid(i as usize)))
                .collect();
            let nulls = valid.iter().any(|v| !v).then(|| NullBuffer::from(valid));
            StructArray::try_new(fields.clone(), cols, nulls)
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(e)
        }
        _ => arrow_select::take::take(a.as_ref(), idx, None).map_err(e),
    }
}

/// A list column rebuilt around its recast inner values (`rows` from `list_parts`).
pub(crate) fn wrap(
    t: &Target,
    values: &ArrayRef,
    rows: &[Option<(usize, usize)>],
    inner: &ArrayRef,
) -> Result<ArrayRef, String> {
    let field = |c: &ArrayRef| Arc::new(AField::new("item", c.data_type().clone(), true));
    let nulls = values.logical_nulls();
    match t {
        Target::Original(_) => Ok(values.clone()),
        Target::Scalar(_) => take_rows(
            inner,
            &UInt64Array::from(
                rows.iter()
                    .map(|r| r.map(|(start, _)| start as u64))
                    .collect::<Vec<_>>(),
            ),
        ),
        Target::List(_) => {
            let mut offsets = vec![0i32];
            for r in rows {
                let len = i32::try_from(r.map_or(0, |(_, len)| len)).map_err(|e| e.to_string())?;
                offsets.push(offsets.last().unwrap() + len);
            }
            ListArray::try_new(
                field(inner),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                inner.clone(),
                nulls,
            )
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| e.to_string())
        }
        Target::FixedList(_, w) => {
            // Re-expand the compacted inner values: a null row gets w null slots.
            let child = if rows.iter().all(Option::is_some) {
                inner.clone()
            } else {
                let w = *w as usize;
                let idx: Vec<Option<u64>> = rows
                    .iter()
                    .flat_map(|r| (0..w).map(move |j| r.map(|(start, _)| (start + j) as u64)))
                    .collect();
                take_rows(inner, &UInt64Array::from(idx))?
            };
            FixedSizeListArray::try_new(field(&child), *w, child, nulls)
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        t => Err(format!("not a list target: {t:?}")),
    }
}

/// A candidate with its (predicted, projected) sizes; one that cannot be sized is
/// listed as failed with the reason, and never tried.
fn candidate(
    target: Target,
    rank: Rank,
    rule: &str,
    evidence: String,
    sizes: Result<(f64, f64), String>,
) -> Candidate {
    let (predicted, projected, outcome, reason) = match sizes {
        Ok((p, q)) => (p as u64, q, Outcome::NotTried, None),
        Err(e) => (0, f64::INFINITY, Outcome::Failed, Some(e)),
    };
    Candidate {
        target,
        rank,
        rule: rule.into(),
        evidence,
        predicted,
        projected,
        outcome,
        reason,
    }
}

/// Lists: choose the inner type first, then wrap it — as a scalar when every list
/// holds one item, else as a List with 32-bit offsets (Array keeps its width).
///
/// The reported candidates are the outer level's (scalar / list / array / original,
/// in the order tried) followed by the inner level's (rules prefixed "inner: "), so
/// a list column shows two `chosen` entries: the outer choice and the inner choice.
fn choose_list(
    lvl: &Level,
    inner: &Level,
    rows: &[Option<(usize, usize)>],
    width: Option<i32>,
    params: &Params,
) -> Result<Chosen, String> {
    let ic = choose(inner, params)?;
    let (n, nulls, r) = (lvl.n_rows() as f64, lvl.n_null() as f64, lvl.r);
    let inner_t = ic.target.clone();
    let kept = matches!(inner_t, Target::Original(_));
    let (c, _) = inner.cardinality();
    let mut outer = Vec::new();
    let nested = kept && matches!(inner.dtype, PT::List(_) | PT::Array(..) | PT::Struct(_));
    let single = lvl.p.range.min_len == Some(1) && lvl.p.range.max_len == Some(1);
    if single && !(lvl.n_null() > 0 && inner.n_null() > 0) && !nested {
        let shape = Shape {
            n,
            nulls: nulls + inner.n_null() as f64,
            ..inner.shape()
        };
        let t = inner_t.arrow_type();
        outer.push(candidate(
            Target::Scalar(Box::new(inner_t.clone())),
            ic.rank,
            "list→scalar",
            format!(
                "min_len=1 max_len=1 n_null={} inner_n_null={}",
                lvl.n_null(),
                inner.n_null()
            ),
            body_size(&t, &shape).and_then(|p| Ok((p, body_size(&t, &shape.project(r, c))?))),
        ));
    }
    match width {
        None if inner.n_rows() < 1 << 31 => outer.push(candidate(
            Target::List(Box::new(inner_t.clone())),
            Rank::List,
            "large_list→list",
            format!("inner_n_values={}", inner.n_rows()),
            Ok((
                validity(n, nulls) + pad(4.0 * (n + 1.0)) + ic.predicted as f64,
                validity(n * r, nulls * r) + pad(4.0 * (n * r + 1.0)) + ic.projected,
            )),
        )),
        // An Array whose inner type is kept is the original type: no candidate.
        Some(w) if !kept => {
            // The child holds w slots per row; a null row's slots are null.
            let wf = w as f64;
            let shape = Shape {
                n: n * wf,
                nulls: inner.n_null() as f64 + nulls * wf,
                ..inner.shape()
            };
            let t = inner_t.arrow_type();
            outer.push(candidate(
                Target::FixedList(Box::new(inner_t.clone()), w),
                Rank::List,
                "array→array",
                format!("width={w}"),
                body_size(&t, &shape).and_then(|p| {
                    Ok((
                        validity(n, nulls) + p,
                        validity(n * r, nulls * r) + body_size(&t, &shape.project(r, c))?,
                    ))
                }),
            ));
        }
        _ => {}
    }
    let mut rules = Rules {
        lvl,
        shape: lvl.shape(),
        out: outer,
    };
    rules.original();
    let (i, array, cands) = first_success(rules.out, |t| wrap(t, &lvl.values, rows, &ic.array));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    let mut chosen = chosen_from(i, array, cands, lossy);
    chosen.candidates.extend(ic.candidates);
    Ok(chosen)
}

// ── Polars layout of the result (Spec B §5.5) ───────────────────────────────

/// `a` converted to the Arrow layout Polars exports for it (`polars_layout`),
/// recursing through lists and structs; `key` is a dictionary's Polars key.
pub(crate) fn to_polars_layout(a: &ArrayRef, key: &AT) -> Result<ArrayRef, String> {
    let item = |c: &ArrayRef| Arc::new(AField::new("item", c.data_type().clone(), true));
    match a.data_type() {
        AT::Dictionary(..) => {
            let keyed = arrow_cast(
                a.as_ref(),
                &AT::Dictionary(Box::new(key.clone()), Box::new(AT::Utf8)),
            )?;
            let d = keyed.as_any_dictionary();
            Ok(d.with_values(polars_views(d.values().as_ref(), &AT::Utf8View)?))
        }
        AT::List(f) | AT::LargeList(f) => {
            let large = arrow_cast(a.as_ref(), &AT::LargeList(f.clone()))?;
            let l = large.as_list::<i64>();
            let child = to_polars_layout(l.values(), key)?;
            LargeListArray::try_new(item(&child), l.offsets().clone(), child, l.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::FixedSizeList(_, w) => {
            let f = a.as_fixed_size_list();
            let child = to_polars_layout(f.values(), key)?;
            FixedSizeListArray::try_new(item(&child), *w, child, f.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::Struct(fields) => {
            let s = a.as_struct();
            let cols = s
                .columns()
                .iter()
                .map(|c| to_polars_layout(c, key))
                .collect::<Result<Vec<_>, _>>()?;
            let fields: Fields = fields
                .iter()
                .zip(&cols)
                .map(|(f, c)| {
                    Arc::new(AField::new(
                        f.name(),
                        c.data_type().clone(),
                        f.is_nullable(),
                    ))
                })
                .collect();
            StructArray::try_new(fields, cols, s.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::Utf8 | AT::LargeUtf8 => polars_views(a.as_ref(), &AT::Utf8View),
        AT::Binary | AT::LargeBinary => polars_views(a.as_ref(), &AT::BinaryView),
        t => arrow_cast(a.as_ref(), &polars_layout(t, key)),
    }
}

/// Polars' view-array data blocks (polars-arrow `binview`): the first holds 8 KiB,
/// each next one doubles (capped at 16 MiB) and grows to fit a larger value.
pub(crate) const VIEW_BLOCK: usize = 8 * 1024;
pub(crate) const VIEW_MAX_BLOCK: usize = 16 * 1024 * 1024;

/// Strings / binaries as the Utf8View / BinaryView array Polars builds from them
/// (`MutableBinaryViewArray::push_value_into_buffer`): values of ≤ 12 bytes inline
/// in the view, longer ones appended to the current block, never straddling blocks —
/// so the measured sizes are Polars' own (arrow-cast would reuse the source buffer).
pub(crate) fn polars_views(a: &dyn Array, to: &AT) -> Result<ArrayRef, String> {
    let bytes = arrow_cast(a, &AT::LargeBinary)?;
    let bytes = bytes.as_binary::<i64>();
    let mut views = Vec::with_capacity(bytes.len());
    let (mut blocks, mut current, mut capacity) = (Vec::<Buffer>::new(), Vec::<u8>::new(), 0usize);
    for v in bytes.iter() {
        let Some(v) = v else {
            views.push(0u128);
            continue;
        };
        if v.len() <= 12 {
            views.push(make_view(v, 0, 0));
            continue;
        }
        if capacity < current.len() + v.len() {
            if !current.is_empty() {
                blocks.push(Buffer::from_vec(std::mem::take(&mut current)));
            }
            capacity = (capacity * 2)
                .clamp(VIEW_BLOCK, VIEW_MAX_BLOCK)
                .max(v.len());
        }
        views.push(make_view(v, blocks.len() as u32, current.len() as u32));
        current.extend_from_slice(v);
    }
    if !current.is_empty() {
        blocks.push(Buffer::from_vec(current));
    }
    let (views, nulls) = (ScalarBuffer::from(views), bytes.nulls().cloned());
    let e = |e: arrow_schema::ArrowError| e.to_string();
    match to {
        AT::Utf8View => StringViewArray::try_new(views, blocks, nulls)
            .map(|x| Arc::new(x) as ArrayRef)
            .map_err(e),
        _ => BinaryViewArray::try_new(views, blocks, nulls)
            .map(|x| Arc::new(x) as ArrayRef)
            .map_err(e),
    }
}

/// A dictionary's values (the Enum categories), looking through one list level.
fn dictionary_values(a: &ArrayRef) -> Option<Vec<String>> {
    match a.data_type() {
        AT::Dictionary(..) => {
            let v = arrow_cast(a.as_any_dictionary().values().as_ref(), &AT::Utf8).ok()?;
            Some(
                v.as_string::<i32>()
                    .iter()
                    .map(|x| x.unwrap_or_default().to_string())
                    .collect(),
            )
        }
        AT::List(_) => dictionary_values(a.as_list::<i32>().values()),
        AT::FixedSizeList(..) => dictionary_values(a.as_fixed_size_list().values()),
        _ => None,
    }
}

// ── one column ───────────────────────────────────────────────────────────────

pub(crate) struct Rec {
    /// The recommended array has nulls (a list → scalar recast turns `[null]` into a null row).
    pub nullable: bool,
    pub arrow_type: String,
    pub arrow_size: u64,
    /// None when nothing was sampled (streaming, reservoir_rows = 0).
    pub arrow_zstd: Option<u64>,
    /// None when the original type is kept: Python fills in `str(dtype)`.
    pub polars_type: Option<String>,
    pub polars_size: u64,
    /// None when nothing was sampled (streaming, reservoir_rows = 0).
    pub polars_zstd: Option<u64>,
    pub lossy: bool,
    pub candidates: Vec<Candidate>,
}

/// The recommendation for one column.
/// `values` is `s` in the classic layout (sizes.rs's `classic_layout`).
pub(crate) fn recommend(
    s: &Series,
    values: &ArrayRef,
    d: &Described,
    sz: &Sizes,
    params: &Params,
) -> PolarsResult<Rec> {
    let [size_bytes, _, polars_bytes, polars_zstd] = *sz;
    let name = s.name().as_str();
    let err = |e: String| polars_err!(ComputeError: "recommend {}: {}", name, e);
    let q = params.population_rows.map(|p| {
        if p == d.n_rows {
            1.0
        } else {
            d.n_rows as f64 / p as f64
        }
    });
    let r = match params.population_rows {
        Some(p) if d.n_rows > 0 => p as f64 / d.n_rows as f64,
        _ => 1.0,
    };
    let outer = Level::of_values(
        s.dtype(),
        values.clone(),
        &d.outer,
        d.n_midnight,
        size_bytes,
        level_estimate(&d.outer, d.n_rows - d.n_null, q),
        r,
        "",
    )
    .map_err(err)?;
    let chosen = match &d.inner {
        Some(inner) if matches!(s.dtype(), PT::List(_) | PT::Array(..)) => {
            let (rows, child, width) = list_parts(values).map_err(err)?;
            if child.len() == inner.values.len() {
                let child_size = ipc_body_bytes(child.as_ref(), None)?;
                let inner_lvl = Level::of_values(
                    inner.values.dtype(),
                    child,
                    &inner.profile,
                    None,
                    child_size,
                    level_estimate(
                        &inner.profile,
                        (inner.values.len() - inner.values.null_count()) as u64,
                        q,
                    ),
                    r,
                    "inner: ",
                )
                .map_err(err)?;
                choose_list(&outer, &inner_lvl, &rows, width, params).map_err(err)?
            } else {
                // Defensive: list_parts builds Describe's flatten, so the lengths agree.
                let why = format!(
                    "inner values {} ≠ Describe's inner_n_values {}: inner type kept",
                    child.len(),
                    inner.values.len()
                );
                choose_original(&outer, &why)
            }
        }
        _ => choose(&outer, params).map_err(err)?,
    };
    let t = chosen.array.data_type().clone();
    let (polars_type, polars_size, polars_zstd) = if matches!(chosen.target, Target::Original(_)) {
        (None, polars_bytes, Some(polars_zstd))
    } else {
        let key = chosen.target.polars_key().unwrap_or(AT::UInt32);
        let layout = to_polars_layout(&chosen.array, &key).map_err(err)?;
        // Spec B §5.4 step 6: the Polars layout only widens, so it must cast back exactly.
        first_mismatch(
            &chosen.array,
            &arrow_cast(layout.as_ref(), &t).map_err(err)?,
        )
        .map_err(|e| err(format!("Polars layout: {e}")))?;
        let enum_values = if q == Some(1.0) {
            dictionary_values(&chosen.array)
        } else {
            None
        };
        (
            Some(pl_name(&t, name, enum_values.as_deref(), &key)),
            ipc_body_bytes(layout.as_ref(), None)?,
            Some(ipc_body_bytes(layout.as_ref(), Some(params.zstd_level))?),
        )
    };
    Ok(Rec {
        nullable: chosen.array.logical_null_count() > 0,
        arrow_type: pa_name(&t),
        arrow_size: ipc_body_bytes(chosen.array.as_ref(), None)?,
        arrow_zstd: Some(ipc_body_bytes(chosen.array.as_ref(), Some(params.zstd_level))?),
        polars_type,
        polars_size,
        polars_zstd,
        lossy: chosen.lossy,
        candidates: chosen.candidates,
    })
}

// ── entry ────────────────────────────────────────────────────────────────────

fn candidate_type() -> PT {
    PT::Struct(vec![
        PField::new("arrow_type".into(), PT::String),
        PField::new("rule".into(), PT::String),
        PField::new("evidence".into(), PT::String),
        PField::new("predicted_bytes".into(), PT::UInt64),
        PField::new("projected_population_bytes".into(), PT::Float64),
        PField::new("outcome".into(), PT::String),
        PField::new("reason".into(), PT::String),
    ])
}

pub(crate) fn rec_fields() -> Vec<(String, PT)> {
    [
        ("rec_nullable", PT::Boolean),
        ("rec_arrow_type", PT::String),
        ("rec_arrow_size_bytes", PT::UInt64),
        ("rec_arrow_size_zstd_bytes", PT::UInt64),
        ("rec_polars_type", PT::String),
        ("rec_polars_size_bytes", PT::UInt64),
        ("rec_polars_size_zstd_bytes", PT::UInt64),
        ("rec_lossy_formatting", PT::Boolean),
        ("rec_candidates", PT::List(Box::new(candidate_type()))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}

fn output_fields() -> Vec<(String, PT)> {
    let mut f = fields();
    f.extend(SIZE_FIELDS.iter().map(|n| (n.to_string(), PT::UInt64)));
    f.extend(rec_fields());
    f
}

fn candidates_series(c: &[Candidate]) -> Series {
    let text = |name: &str, v: Vec<Option<String>>| {
        StringChunked::from_iter_options(name.into(), v.into_iter()).into_series()
    };
    let cols = [
        text(
            "arrow_type",
            c.iter()
                .map(|x| Some(pa_name(&x.target.arrow_type())))
                .collect(),
        ),
        text("rule", c.iter().map(|x| Some(x.rule.clone())).collect()),
        text(
            "evidence",
            c.iter().map(|x| Some(x.evidence.clone())).collect(),
        ),
        UInt64Chunked::from_iter_values("predicted_bytes".into(), c.iter().map(|x| x.predicted))
            .into_series(),
        Float64Chunked::from_iter_values(
            "projected_population_bytes".into(),
            c.iter().map(|x| x.projected),
        )
        .into_series(),
        text(
            "outcome",
            c.iter()
                .map(|x| Some(x.outcome.name().to_string()))
                .collect(),
        ),
        text("reason", c.iter().map(|x| x.reason.clone()).collect()),
    ];
    StructChunked::from_series("candidate".into(), c.len(), cols.iter())
        .expect("equal-length fields")
        .into_series()
}

pub(crate) fn rec_row(r: &Rec) -> Row {
    let text = |s: &str| AnyValue::StringOwned(s.into());
    vec![
        AnyValue::Boolean(r.nullable),
        text(&r.arrow_type),
        AnyValue::UInt64(r.arrow_size),
        r.arrow_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        r.polars_type.as_deref().map_or(AnyValue::Null, text),
        AnyValue::UInt64(r.polars_size),
        r.polars_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        AnyValue::Boolean(r.lossy),
        AnyValue::List(candidates_series(&r.candidates)),
    ]
}

pub(crate) fn describe_and_recommend_impl(
    inputs: &[Series],
    params: &Params,
) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs
        .par_iter()
        .map(|s| {
            let d = describe_one(s, params.seed)?;
            let classic = classic_layout(s)?;
            let sz = sizes_of(s, &classic, params.zstd_level)?;
            let rec = recommend(s, &classic, &d, &sz, params)?;
            let mut row = d.row();
            row.extend(sz.iter().map(|&v| AnyValue::UInt64(v)));
            row.extend(rec_row(&rec));
            Ok(row)
        })
        .collect::<PolarsResult<_>>()?;
    assemble("recommend", &output_fields(), &rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizes::ipc_body_bytes;
    use arrow_array::{
        builder::StringDictionaryBuilder, types::UInt8Type, DictionaryArray, StringArray,
    };

    #[test]
    fn pyarrow_type_names() {
        assert_eq!(pa_name(&AT::Float32), "float");
        assert_eq!(pa_name(&decimal_type(6, 2)), "decimal32(6, 2)");
        assert_eq!(pa_name(&decimal_type(10, 3)), "decimal64(10, 3)");
        assert_eq!(
            pa_name(&AT::Timestamp(TimeUnit::Millisecond, Some("+05:00".into()))),
            "timestamp[ms, tz=+05:00]"
        );
        assert_eq!(
            pa_name(&Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type()),
            "dictionary<values=string, indices=uint8, ordered=0>"
        );
        assert_eq!(
            pa_name(&Target::List(Box::new(Target::Fixed(AT::UInt8))).arrow_type()),
            "list<item: uint8>"
        );
        assert_eq!(
            pa_name(&timestamp_with_offset(TimeUnit::Second)),
            "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"
        );
    }

    #[test]
    fn polars_type_names() {
        let k = AT::UInt8;
        assert_eq!(
            pl_name(&decimal_type(6, 2), "x", None, &k),
            "Decimal(precision=6, scale=2)"
        );
        assert_eq!(
            pl_name(&AT::Timestamp(TimeUnit::Second, None), "x", None, &k),
            "Datetime(time_unit='ms', time_zone=None)"
        );
        assert_eq!(
            pl_name(&AT::Duration(TimeUnit::Second), "x", None, &k),
            "Duration(time_unit='ms')"
        );
        let dict = Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type();
        assert_eq!(
            pl_name(&dict, "x", Some(&["a".into(), "b".into()]), &k),
            "Enum(categories=['a', 'b'])"
        );
        assert_eq!(
            pl_name(&dict, "x", None, &k),
            "Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))"
        );
        assert_eq!(
            pl_name(&timestamp_with_offset(TimeUnit::Second), "x", None, &k),
            "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})"
        );
    }

    #[test]
    fn python_repr_for_enum_categories() {
        // Verified against: python -c "import polars as pl; print(str(pl.Enum([...])))"
        let k = AT::UInt8;
        let dict = Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type();
        let values = [
            "it's".to_string(),
            "a\"b".to_string(),
            "x\\y".to_string(),
            "n\nl".to_string(),
        ];
        assert_eq!(
            pl_name(&dict, "x", Some(&values), &k),
            "Enum(categories=[\"it's\", 'a\"b', 'x\\\\y', 'n\\nl'])"
        );
    }

    #[test]
    fn polars_layouts_and_key_widths() {
        assert_eq!(
            polars_layout(&AT::Time32(TimeUnit::Second), &AT::UInt8),
            AT::Time64(TimeUnit::Nanosecond)
        );
        assert_eq!(
            polars_layout(&decimal_type(6, 2), &AT::UInt8),
            AT::Decimal128(6, 2)
        );
        assert_eq!(dictionary_keys(256.0), (AT::UInt8, AT::UInt16));
        assert_eq!(dictionary_keys(255.0), (AT::UInt8, AT::UInt8));
        assert_eq!(dictionary_keys(65_537.0).0, AT::UInt32);
    }

    #[test]
    fn predicted_sizes_match_sizes_rs() {
        let s = StringArray::from(vec![Some("ab"), None]);
        let shape = Shape {
            n: 2.0,
            nulls: 1.0,
            sum_len: 2.0,
            d: 1.0,
            sum_len_unique: 2.0,
        };
        assert_eq!(
            body_size(&AT::Utf8, &shape),
            Ok(ipc_body_bytes(&s, None).unwrap() as f64)
        );
        let mut builder = StringDictionaryBuilder::<UInt8Type>::new();
        for v in ["a", "b", "a"] {
            builder.append_value(v);
        }
        let d: DictionaryArray<UInt8Type> = builder.finish();
        let shape = Shape {
            n: 3.0,
            nulls: 0.0,
            sum_len: 3.0,
            d: 2.0,
            sum_len_unique: 2.0,
        };
        assert_eq!(
            body_size(
                &Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type(),
                &shape
            ),
            Ok(ipc_body_bytes(&d, None).unwrap() as f64)
        );
    }

    #[test]
    fn lower_case_comparison() {
        assert!(lower_eq("TRUE", "true") && lower_eq("tRuE", "true") && !lower_eq("true ", "true"));
        assert!(lower_eq("\u{212A}", "k")); // Kelvin sign lower-cases to ASCII k
        assert!(!lower_eq("É", "e") && lower_eq("É", "é"));
    }

    #[test]
    fn unsized_types_are_errors_not_panics() {
        let shape = Shape {
            n: 3.0,
            ..Default::default()
        };
        assert!(body_size(&AT::Utf8View, &shape).is_err());
        assert!(body_size(
            &Target::List(Box::new(Target::Fixed(AT::UInt8))).arrow_type(),
            &shape
        )
        .is_err());
        assert!(body_size(
            &AT::Struct(Fields::from(vec![AField::new("a", AT::Int8, true)])),
            &shape
        )
        .is_err());
        assert!(body_size(
            &AT::Dictionary(Box::new(AT::Utf8), Box::new(AT::Utf8)),
            &shape
        )
        .is_err());
        assert_eq!(
            body_size(&timestamp_with_offset(TimeUnit::Millisecond), &shape),
            Ok(24.0 + 8.0)
        );
        let c = candidate(
            Target::Fixed(AT::Utf8View),
            Rank::Plain,
            "r",
            String::new(),
            Err("no size".into()),
        );
        assert_eq!(
            (c.outcome, c.reason.as_deref()),
            (Outcome::Failed, Some("no size"))
        );
        let a: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
        assert!(list_parts(&a).is_err());
    }

    #[test]
    fn canonical_decimals() {
        assert_eq!(canon("1.50"), canon("1.5"));
        assert_eq!(canon("0.00120"), canon("1.2e-3"));
        assert_eq!(canon("-0.0"), canon("0"));
        assert_ne!(canon("123"), canon("12.3"));
        assert_eq!(canon("120"), canon("1.2e2"));
        assert_eq!(canon("00.0100"), canon("1e-2"));
        assert_eq!(canon("10.05"), canon("1005e-2"));
        assert_ne!(canon("10.05"), canon("10.5"));
        assert_ne!(canon("-1.5"), canon("1.5"));
        assert_eq!(canon("0"), canon("-0.000"));
        assert_eq!(decimal_from_repr("1e-7", 7), Some(1));
        assert_eq!(
            decimal_from_repr("1.5e20", 0),
            Some(150_000_000_000_000_000_000)
        );
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

    use crate::arrow_io::export_series;
    use crate::describe::describe_one;
    use polars::prelude::{
        CompatLevel, DataType as PT, IntoSeries, NamedFrom, NewChunkedArray, Series,
        TimeUnit as PTimeUnit,
    };

    pub(super) fn params() -> Params {
        Params {
            seed: 0,
            zstd_level: 1,
            population_rows: None,
            categorical_threshold: 10_000,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    fn types(s: Series, p: &Params) -> Vec<(String, Outcome)> {
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            level_estimate(&d.outer, d.n_rows - d.n_null, None),
            1.0,
            "",
        )
        .unwrap();
        candidates(&lvl, p)
            .unwrap()
            .iter()
            .map(|c| (pa_name(&c.target.arrow_type()), c.outcome))
            .collect()
    }

    fn names(s: Series) -> Vec<String> {
        types(s, &params()).into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn integer_rules() {
        assert_eq!(
            names(Series::new("x".into(), &[0i64, 1])),
            ["bool", "uint8", "int8", "int64"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[-200i64, 5])),
            ["int16", "int64"]
        );
    }

    #[test]
    fn decimal_scale_reduced_by_gcd() {
        let dec = polars::prelude::Int128Chunked::from_slice("x".into(), &[120, 340])
            .into_decimal_unchecked(Some(10), 2)
            .into_series();
        assert_eq!(names(dec), ["decimal32(2, 1)", "decimal128(10, 2)"]);
    }

    #[test]
    fn float_rules() {
        assert_eq!(
            names(Series::new("x".into(), &[123.45f64, 99.99])),
            ["decimal32(5, 2)", "double"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[0.5f64, 0.25])),
            ["decimal32(2, 2)", "float", "double"]
        );
        assert_eq!(
            names(Series::new("x".into(), &[0.1f64, f64::NAN])),
            ["double"]
        );
    }

    #[test]
    fn string_rules() {
        let dict = "dictionary<values=string, indices=uint8, ordered=0>";
        assert_eq!(
            names(Series::new("x".into(), &["007", "12"])),
            ["string", dict, "large_string"]
        );
        assert_eq!(
            names(Series::new("x".into(), &["1234567890.1", "0.00000012345"])),
            [
                "decimal128(21, 11)",
                "double",
                "string",
                dict,
                "large_string"
            ]
        );
        let offsets = names(Series::new(
            "x".into(),
            &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
        ));
        assert_eq!(
            offsets[0],
            "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"
        );
        assert_eq!(
            names(Series::new("x".into(), &["True", "false"]))[0],
            "bool"
        );
    }

    #[test]
    fn temporal_rules() {
        let days = Series::new("x".into(), &[0i64, 86_400_000_000])
            .cast(&PT::Datetime(PTimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(
            names(days),
            ["date32[day]", "timestamp[s]", "timestamp[us]"]
        );
    }

    #[test]
    fn missing_profile_parts_give_no_type_candidates_not_panics() {
        // A profile whose parts do not match the dtype (Describe never builds one): the
        // rules bail instead of panicking — only the original (and text's plain/dictionary).
        let ints = Series::new("x".into(), &[1i64, 2]);
        let d = describe_one(&ints, 0).unwrap();
        let values = export_series(&ints, CompatLevel::oldest()).unwrap();
        for dtype in [PT::Float64, PT::Decimal(Some(10), Some(2)), PT::String] {
            let lvl = Level::of_values(
                &dtype,
                values.clone(),
                &d.outer,
                None,
                0,
                level_estimate(&d.outer, 2, None),
                1.0,
                "",
            )
            .unwrap();
            let got: Vec<String> = candidates(&lvl, &params())
                .unwrap()
                .iter()
                .map(|c| c.rule.clone())
                .collect();
            let expected: &[&str] = if dtype == PT::String {
                &["string→utf8", "string→dictionary", "original"]
            } else {
                &["original"]
            };
            assert_eq!(got, expected, "{dtype}");
        }
    }

    #[test]
    fn dictionary_evidence_names_the_estimator() {
        let s = Series::new("x".into(), &["a", "b", "a", "b", "c"]);
        let d = describe_one(&s, 0).unwrap();
        let evidence = |q: Option<f64>| {
            let lvl = Level::of_values(
                s.dtype(),
                export_series(&s, CompatLevel::oldest()).unwrap(),
                &d.outer,
                None,
                0,
                level_estimate(&d.outer, 5, q),
                1.0,
                "",
            )
            .unwrap();
            candidates(&lvl, &params())
                .unwrap()
                .into_iter()
                .find(|c| c.rule == "string→dictionary")
                .unwrap()
                .evidence
        };
        let chao = evidence(None);
        assert!(chao.contains("method=chao1 est_low="), "{chao}");
        let duj = evidence(Some(0.5));
        assert!(
            duj.contains("method=duj1") && !duj.contains("est_low"),
            "{duj}"
        );
        assert!(evidence(Some(1.0)).contains("c=3.0 from est_high method=exact est_low=3.0"));
    }

    #[test]
    fn dictionary_gate_rejects() {
        let p = Params {
            categorical_threshold: 1,
            ..params()
        };
        let got = types(Series::new("x".into(), &["a", "b", "a", "b"]), &p);
        assert!(got
            .iter()
            .any(|(t, o)| t.starts_with("dictionary") && *o == Outcome::Rejected));
    }

    #[test]
    fn dictionary_cardinality_floors_at_n_unique() {
        // 3 distinct values, but the estimate handed to Level is (artificially)
        // below that — cardinality() must still report at least n_unique.
        let s = Series::new("x".into(), &["a", "b", "c"]);
        let d = describe_one(&s, 0).unwrap();
        assert_eq!(d.outer.freq.n_unique, 3);
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            Estimate {
                    est_cardinality: 1.0,
                    est_low: Some(1.0),
                    est_high: Some(1.0),
                    method: crate::cardinality_estimators::Method::Chao1,
                },
            1.0,
            "",
        )
        .unwrap();
        assert_eq!(lvl.cardinality(), (3.0, "est_high"));
    }

    use arrow_array::{Float64Array, Int64Array};

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
    fn verification_reports_the_first_mismatch() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 3]));
        assert!(first_mismatch(&a, &b).unwrap_err().starts_with("row 1:"));
        assert!(first_mismatch(&a, &a.clone()).is_ok());
        let empty: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        assert!(first_mismatch(&empty, &empty.clone()).is_ok());
        // Mismatches around the 64K chunk boundaries, and a null against a value.
        let n = 3 * MISMATCH_CHUNK + 7;
        let base: Vec<Option<i64>> = (0..n as i64).map(Some).collect();
        let a: ArrayRef = Arc::new(Int64Array::from(base.clone()));
        for i in [
            0,
            MISMATCH_CHUNK - 1,
            MISMATCH_CHUNK,
            2 * MISMATCH_CHUNK + 1,
            n - 1,
        ] {
            for v in [Some(-1), None] {
                let mut other = base.clone();
                other[i] = v;
                other[n - 1] = if i == n - 1 { v } else { Some(-2) }; // a later mismatch too
                let b: ArrayRef = Arc::new(Int64Array::from(other));
                let err = first_mismatch(&a, &b).unwrap_err();
                assert!(err.starts_with(&format!("row {i}:")), "{i}: {err}");
            }
        }
        let text = LargeStringArray::from(vec!["2300-01-01T00:00:00.123456789"]);
        assert!(from_text(
            &Target::Fixed(AT::Timestamp(TimeUnit::Nanosecond, None)),
            &text
        )
        .is_err()); // beyond i64 ns
    }

    #[test]
    fn timestamp_with_offset_struct_children_have_no_nulls() {
        let text = LargeStringArray::from(vec![
            Some("2024-01-05T10:00+05:00"),
            None,
            Some("2024-01-06T00:00Z"),
        ]);
        let a = from_text(&Target::TimestampWithOffset(TimeUnit::Second), &text).unwrap();
        let s = a.as_struct();
        assert_eq!(s.column(0).null_count(), 0);
        assert_eq!(s.column(1).null_count(), 0);
        let shape = Shape {
            n: 3.0,
            nulls: 1.0,
            ..Default::default()
        };
        assert_eq!(
            Ok(ipc_body_bytes(a.as_ref(), None).unwrap() as f64),
            body_size(&timestamp_with_offset(TimeUnit::Second), &shape)
        );
    }

    #[test]
    fn null_target_verifies_all_null_columns() {
        // NullArray (Target::Null's recast) has no physical null buffer at all, so a
        // naive `is_null` per-row check against it is always false — verify() must
        // special-case Target::Null and check the *source*'s logical nulls instead.
        for s in [
            Series::new("x".into(), &[None::<i64>, None]),
            Series::new("x".into(), &[None::<&str>, None]),
        ] {
            let d = describe_one(&s, 0).unwrap();
            let lvl = Level::of_values(
                s.dtype(),
                export_series(&s, CompatLevel::oldest()).unwrap(),
                &d.outer,
                d.n_midnight,
                0,
                level_estimate(&d.outer, 0, None),
                1.0,
                "",
            )
            .unwrap();
            let recast = cast_to(&Target::Null, &lvl).unwrap();
            assert!(verify(&Target::Null, &lvl, &recast).is_ok());
        }
        // A column that is NOT all-null must not verify against Target::Null.
        let s = Series::new("x".into(), &[Some(1i64), None]);
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            level_estimate(&d.outer, 1, None),
            1.0,
            "",
        )
        .unwrap();
        let recast = cast_to(&Target::Null, &lvl).unwrap();
        assert!(verify(&Target::Null, &lvl, &recast).is_err());
    }

    #[test]
    fn timestamp_offset_consistency_is_verified() {
        // Same instant, but the two rows disagree on offset — a target fixed at +05:00
        // is only valid if every row actually carried +05:00; the previous epoch-only
        // check missed this because relabelling the tz doesn't change the stored instant.
        let text = LargeStringArray::from(vec!["2024-01-05T10:00+05:00", "2024-01-05T02:00-03:30"]);
        let target = Target::Fixed(AT::Timestamp(TimeUnit::Second, Some("+05:00".into())));
        let recast = from_text(&target, &text).unwrap();
        assert!(verify_text(&target, &text, &recast).is_err());

        // A single consistent offset does verify.
        let text = LargeStringArray::from(vec!["2024-01-05T10:00+05:00", "2024-01-06T10:00+05:00"]);
        let recast = from_text(&target, &text).unwrap();
        assert!(verify_text(&target, &text, &recast).is_ok());

        // An offset-bearing source cast to a naive Timestamp must fail: the offset was
        // silently dropped, not merely re-rendered.
        let text = LargeStringArray::from(vec!["2024-01-05T10:00+05:00"]);
        let naive = Target::Fixed(AT::Timestamp(TimeUnit::Second, None));
        let recast = from_text(&naive, &text).unwrap();
        assert!(verify_text(&naive, &text, &recast).is_err());
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

    #[test]
    fn verify_float_reports_parse_failure_instead_of_nan() {
        assert!(parse_back("not_a_number", false).is_err());
        assert_eq!(parse_back("1.5", false), Ok(1.5));
    }

    #[test]
    fn recast_timezone_change_fails_generic_verify() {
        // Two arrays holding the same instant but tagged with different tz: casting
        // `recast` up to `src`'s exact data type (tz included) makes their values equal
        // (Timestamp storage doesn't depend on tz), so only an explicit tz check catches
        // the mismatch — first_mismatch alone cannot.
        let src = timestamp_array(
            TimeUnit::Microsecond,
            vec![Some(0), Some(3_600_000_000)],
            Some("+05:00".into()),
        );
        let recast = timestamp_array(
            TimeUnit::Microsecond,
            vec![Some(0), Some(3_600_000_000)],
            Some("+00:00".into()),
        );
        let dummy = Series::new("x".into(), &[0i64, 1]);
        let d = describe_one(&dummy, 0).unwrap();
        let dtype = PT::Datetime(PTimeUnit::Microseconds, None);
        let lvl = Level::of_values(
            &dtype,
            src,
            &d.outer,
            None,
            0,
            level_estimate(&d.outer, 2, None),
            1.0,
            "",
        )
        .unwrap();
        let target = Target::Fixed(AT::Timestamp(TimeUnit::Microsecond, Some("+05:00".into())));
        assert!(verify(&target, &lvl, &recast).is_err());
    }

    use crate::sizes::sizes;

    fn rec(s: Series) -> Rec {
        let d = describe_one(&s, 0).unwrap();
        let sz = sizes(&s, 1).unwrap();
        recommend(&s, &classic_layout(&s).unwrap(), &d, &sz, &params()).unwrap()
    }

    fn chosen(r: &Rec) -> &Candidate {
        r.candidates
            .iter()
            .find(|c| c.outcome == Outcome::Chosen)
            .unwrap()
    }

    #[test]
    fn end_to_end_choices() {
        assert_eq!(
            rec(Series::new("x".into(), &[0i64, 5, 127])).arrow_type,
            "uint8"
        );
        let price = rec(Series::new("x".into(), &[123.45f64, 99.99]));
        assert_eq!(
            (price.arrow_type.as_str(), price.polars_type.as_deref()),
            ("decimal32(5, 2)", Some("Decimal(precision=5, scale=2)"))
        );
        assert_eq!(
            rec(Series::new(
                "x".into(),
                &["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"]
            ))
            .arrow_type,
            "timestamp[ms]"
        );
        let kept = rec(Series::new("x".into(), &[0.1f64, f64::NAN]));
        assert_eq!(
            (kept.arrow_type.as_str(), kept.polars_type.clone()),
            ("double", None)
        );
    }

    #[test]
    fn failed_cast_falls_back() {
        let r = rec(Series::new(
            "x".into(),
            &["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"],
        ));
        assert_eq!(r.arrow_type, "string");
        assert!(r
            .candidates
            .iter()
            .any(|c| c.outcome == Outcome::Failed
                && pa_name(&c.target.arrow_type()) == "timestamp[ns]"));
    }

    #[test]
    fn single_item_lists_become_scalars() {
        let s = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[1i64])),
                Some(Series::new("".into(), &[2i64])),
                None,
            ],
        );
        assert_eq!(rec(s).arrow_type, "uint8");
        // Null lists and null items both occur: the column stays a list (Spec B §4.4).
        let both = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(2i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                None,
            ],
        );
        assert_eq!(rec(both).arrow_type, "list<item: uint8>");
        // Inner values {1}: 0 ≤ min, max ≤ 1 → Boolean ties UInt8 in size and wins on rank (§4.1, §4.2).
        let ones = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(1i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                None,
            ],
        );
        assert_eq!(rec(ones).arrow_type, "list<item: bool>");
    }

    #[test]
    fn nullable_means_the_recommended_array_has_nulls() {
        // [null] items become null scalars: the recast column has a null although the source has none.
        let s = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[Some(1i64)])),
                Some(Series::new("".into(), &[None::<i64>])),
                Some(Series::new("".into(), &[Some(2i64)])),
            ],
        );
        let r = rec(s);
        assert_eq!((r.arrow_type.as_str(), r.nullable), ("uint8", true));
        assert!(!rec(Series::new("x".into(), &[1i64, 2])).nullable);
        assert!(rec(Series::new("x".into(), &[Some(1i64), None])).nullable);
    }

    /// A List(Int64) Series with offsets `offsets` over `values`, row `null` null — its
    /// values still behind the offsets (as `pl.when(mask).then(list).otherwise(None)` leaves them).
    fn null_list_holding_values(values: Vec<i64>, offsets: Vec<i64>, null: usize) -> Series {
        use polars_arrow::array::{ListArray as PlList, PrimitiveArray};
        use polars_arrow::bitmap::Bitmap;
        use polars_arrow::offset::Offsets;
        let values = PrimitiveArray::<i64>::from_vec(values).boxed();
        let rows = offsets.len() - 1;
        let arr = PlList::<i64>::new(
            PlList::<i64>::default_datatype(values.dtype().clone()),
            Offsets::try_from(offsets).unwrap().into(),
            values,
            Some(Bitmap::from_iter((0..rows).map(|i| i != null))),
        );
        Series::from_arrow("x".into(), arr.boxed()).unwrap()
    }

    #[test]
    fn null_lists_holding_values_are_narrowed() {
        let s = null_list_holding_values(vec![1, 2, 3, 4, 5], vec![0, 2, 4, 5], 1);
        assert_eq!(describe_one(&s, 0).unwrap().inner.unwrap().values.len(), 3); // Describe's flatten skips the null row
        let r = rec(s);
        assert_eq!(
            (r.arrow_type.as_str(), r.nullable),
            ("list<item: uint8>", true)
        );
        assert_eq!(chosen(&r).predicted, r.arrow_size);
        let inner = r
            .candidates
            .iter()
            .find(|c| c.outcome == Outcome::Chosen && c.rule.starts_with("inner: "))
            .unwrap();
        assert_eq!(pa_name(&inner.target.arrow_type()), "uint8");
        // Single-item rows: the null row (holding [9]) becomes a null scalar.
        let r = rec(null_list_holding_values(vec![1, 9, 3], vec![0, 1, 2, 3], 1));
        assert_eq!((r.arrow_type.as_str(), r.nullable), ("uint8", true));
        assert_eq!(chosen(&r).predicted, r.arrow_size);
    }

    #[test]
    fn predicted_equals_measured() {
        let cases = [
            Series::new("x".into(), &[Some(0i64), Some(5), None]),
            Series::new("x".into(), &[123.45f64, 99.99]),
            Series::new("x".into(), &["a", "b", "a", "b", "a", "b"]),
            Series::new(
                "x".into(),
                &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
            ),
            Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]),
            // list→scalar timestamp_with_offset: a null list is a null struct row, children null-free
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &["2024-01-05T10:00+05:00"])),
                    None,
                    Some(Series::new("".into(), &["2024-01-05T10:00-03:30"])),
                ],
            ),
        ];
        for s in cases {
            let r = rec(s);
            assert_eq!(chosen(&r).predicted, r.arrow_size, "{}", r.arrow_type);
        }
    }

    /// The inner level's choice for a list column, as `recommend` makes it.
    fn inner_chosen(s: &Series) -> Chosen {
        let d = describe_one(s, 0).unwrap();
        let inner = d.inner.as_ref().unwrap();
        let (_, child, _) = list_parts(&classic_layout(s).unwrap()).unwrap();
        let lvl = Level::of_values(
            inner.values.dtype(),
            child.clone(),
            &inner.profile,
            None,
            ipc_body_bytes(child.as_ref(), None).unwrap(),
            level_estimate(
                    &inner.profile,
                    (inner.values.len() - inner.values.null_count()) as u64,
                    None,
                ),
            1.0,
            "inner: ",
        )
        .unwrap();
        choose(&lvl, &params()).unwrap()
    }

    #[test]
    fn inner_predicted_equals_measured() {
        let many: Vec<&str> = (0..200)
            .map(|i| if i % 3 == 0 { "alpha" } else { "beta" })
            .collect();
        let cases = [
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[1i64, 2])),
                    None,
                    Some(Series::new("".into(), &[300i64])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[Some(1.5f64), None])),
                    Some(Series::new("".into(), &[2.25f64])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &many)),
                    None,
                    Some(Series::new("".into(), &["gamma"])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &["1.50", "2.2"])),
                    Some(Series::new("".into(), &["3"])),
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new(
                        "".into(),
                        &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"],
                    )),
                    None,
                ],
            ),
            Series::new(
                "x".into(),
                [
                    Some(Series::new("".into(), &[1i64, 2])),
                    None,
                    Some(Series::new("".into(), &[5i64, 6])),
                ],
            )
            .cast(&PT::Array(Box::new(PT::Int64), 2))
            .unwrap(),
            null_list_holding_values(vec![1, 2, 3, 4, 5], vec![0, 2, 4, 5], 1),
        ];
        for s in cases {
            let c = inner_chosen(&s);
            assert!(
                !matches!(c.target, Target::Original(_)),
                "{}",
                pa_name(c.array.data_type())
            ); // a recast, not the measured original
            assert_eq!(
                c.predicted,
                ipc_body_bytes(c.array.as_ref(), None).unwrap(),
                "{}",
                pa_name(c.array.data_type())
            );
        }
    }

    #[test]
    fn polars_types_of_results() {
        let s = Series::new("x".into(), &["a", "b", "a", "b", "a", "b"]);
        let (d, sz) = (describe_one(&s, 0).unwrap(), sizes(&s, 1).unwrap());
        let exact = Params {
            population_rows: Some(6),
            ..params()
        };
        let r = recommend(&s, &classic_layout(&s).unwrap(), &d, &sz, &exact).unwrap();
        assert_eq!(
            r.polars_type.as_deref(),
            Some("Enum(categories=['a', 'b'])")
        );
        let many: Vec<&str> = (0..200)
            .map(|i| if i % 2 == 0 { "alpha" } else { "beta" })
            .collect();
        let r = rec(Series::new("x".into(), &many));
        assert_eq!(
            r.polars_type.as_deref(),
            Some("Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))")
        );
        // List of dictionary strings: Polars layout is large_list<dictionary<string_view>>, sized without error.
        let words = Series::new("x".into(), [Some(Series::new("".into(), &many)), None]);
        let r = rec(words);
        assert_eq!(
            r.polars_type.as_deref(),
            Some("List(Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8)))")
        );
        assert!(r.polars_size > 0);
    }

    fn rec_with(s: Series, population_rows: Option<u64>) -> Rec {
        let (d, sz) = (describe_one(&s, 0).unwrap(), sizes(&s, 1).unwrap());
        recommend(
            &s,
            &classic_layout(&s).unwrap(),
            &d,
            &sz,
            &Params {
                population_rows,
                ..params()
            },
        )
        .unwrap()
    }

    #[test]
    fn polars_sizes_match_polars_view_layout() {
        // Oracle: the same data cast in Polars and measured by analytics/describe/_sizes.py
        // (size_polars_bytes). Views inline ≤ 12 bytes; longer values fill 8 KiB, 16 KiB, … blocks.
        let long = |i: usize| format!("long string number {i:06}");
        let cases: Vec<(Series, Option<u64>, &str, u64)> = vec![
            (
                Series::new("x".into(), &[Some("x"), Some("y"), Some("x"), None]),
                Some(4),
                "Enum(categories=['x', 'y'])",
                48,
            ),
            (
                Series::new("x".into(), ["a long category value 1", "b"].repeat(4)),
                Some(8),
                "Enum(categories=['a long category value 1', 'b'])",
                64,
            ),
            (
                Series::new(
                    "x".into(),
                    [
                        Some(Series::new("".into(), &["a", "bb"])),
                        None,
                        Some(Series::new("".into(), &["a"])),
                    ],
                ),
                None,
                "List(String)",
                88,
            ),
            (
                Series::new(
                    "x".into(),
                    [
                        Some(Series::new("".into(), &["a", "this is a long string!"])),
                        None,
                        Some(Series::new("".into(), &["a"])),
                    ],
                ),
                None,
                "List(String)",
                112,
            ),
            (
                Series::new(
                    "x".into(),
                    (0..256).map(|i| format!("s{i}")).collect::<Vec<_>>(),
                ),
                None,
                "String",
                4096,
            ),
            (
                Series::new("x".into(), (0..1000).map(long).collect::<Vec<_>>()),
                None,
                "String",
                41_008,
            ),
            (
                Series::new(
                    "x".into(),
                    (0..1000)
                        .map(|i| (i % 3 != 0).then(|| long(i)))
                        .collect::<Vec<_>>(),
                ),
                None,
                "String",
                32_784,
            ),
            (
                Series::new(
                    "x".into(),
                    &[Some(&b"ab"[..]), Some(&b"0123456789abcdefg"[..]), None],
                ),
                None,
                "Binary",
                80,
            ),
        ];
        for (s, population_rows, polars_type, polars_size) in cases {
            let r = rec_with(s, population_rows);
            assert_eq!(
                (r.polars_type.as_deref(), r.polars_size),
                (Some(polars_type), polars_size),
                "{}",
                r.arrow_type
            );
        }
    }

    #[test]
    fn nullable_arrays_narrow() {
        let lists = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[1i64, 2])),
                None,
                Some(Series::new("".into(), &[5i64, 6])),
                Some(Series::new("".into(), &[7i64, 8])),
            ],
        );
        let s = lists.cast(&PT::Array(Box::new(PT::Int64), 2)).unwrap();
        let r = rec(s);
        assert_eq!(r.arrow_type, "fixed_size_list<item: uint8>[2]");
        assert_eq!(chosen(&r).predicted, r.arrow_size);
        assert_eq!(r.polars_type.as_deref(), Some("Array(UInt8, shape=(2,))"));
    }

    #[test]
    fn output_matches_declared_schema() {
        let nested = Series::new(
            "n".into(),
            [
                Some(Series::new("".into(), &[1i64])),
                None,
                Some(Series::new("".into(), &[2i64])),
            ],
        );
        let inputs = [
            Series::new("a".into(), &[Some(0i64), Some(5), None]),
            Series::new("s".into(), &["x", "y", "x"]),
            nested,
        ];
        let out = describe_and_recommend_impl(&inputs, &params()).unwrap();
        let declared = PT::Struct(
            output_fields()
                .into_iter()
                .map(|(n, d)| PField::new(n.into(), d))
                .collect(),
        );
        assert_eq!(out.dtype(), &declared);
        assert_eq!(out.len(), 3);
        let ca = out.struct_().unwrap();
        let fields = ca.fields_as_series();
        let get = |n: &str| {
            fields
                .iter()
                .find(|f| f.name().as_str() == n)
                .unwrap()
                .clone()
        };
        let types = get("rec_arrow_type");
        assert_eq!(types.str().unwrap().get(0), Some("uint8"));
        assert_eq!(types.str().unwrap().get(2), Some("uint8"));
        assert_eq!(types.null_count(), 0);
        let cands = get("rec_candidates");
        let first = cands.list().unwrap().get_as_series(0).unwrap();
        assert!(first.len() >= 2);
        assert!(cands.list().unwrap().get_as_series(2).is_some());
    }
}
