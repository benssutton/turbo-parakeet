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

/// Python `repr()` of a str, as Polars' `str(pl.Enum([...]))` renders each category:
/// double-quoted when the string holds `'` and no `"` (avoids escaping the apostrophe),
/// else single-quoted with `'` escaped; `\`, `\n`, `\r`, `\t` get their short escapes,
/// other control characters (`< 0x20` or `0x7f`) become `\xNN`.
fn py_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
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
        t => v + pad(s.n * t.primitive_width().expect("body_size: fixed-width type") as f64),
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
///
/// Precondition: `s` is a numeric literal already validated by describe.rs's
/// `scan_numeric` (or produced by Rust's own float formatter, e.g. ryu) — an
/// optional leading `-`, ASCII digits, at most one `.`, and an optional
/// `e`/`E`-led exponent. `canon` does not re-validate this; a non-conforming
/// `s` (e.g. embedded non-digit bytes in the mantissa) can produce a `sig` that
/// is not purely ASCII digits, which `decimal_from_repr` below guards against.
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
/// more decimal places or more than 38 digits. Same precondition as `canon`; also
/// returns None (rather than underflowing the `c - b'0'` subtraction) if `canon`
/// ever hands back a non-digit byte in `sig`, which a conforming input cannot do.
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
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow((scale as i32 - places) as u32)?)?;
    (v < 10i128.pow(38)).then_some(if neg { -v } else { v })
}

// ── candidates (Spec B §4, §5.2) ────────────────────────────────────────────

use crate::cardinality_estimators::{estimate, Estimate};
use crate::describe::Profile;
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type};
use arrow_array::{Array, ArrayRef, LargeStringArray};
use arrow_cast::cast::{cast_with_options, CastOptions};
use polars::prelude::DataType as PT;
use serde::Deserialize;

/// Plugin keyword arguments (Recommend's constructor keywords).
#[derive(Deserialize, Clone, Debug)]
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
    pub column: &'a str,
    /// Polars dtype of these values.
    pub dtype: &'a PT,
    /// The values in Arrow's classic layout (CompatLevel::oldest).
    pub values: ArrayRef,
    pub p: &'a Profile,
    pub n_midnight: Option<u64>,
    /// Measured uncompressed size of `values`.
    pub size_bytes: u64,
    pub est: Estimate,
    /// Population rows ÷ frame rows.
    pub r: f64,
    /// "" or "inner: " — prefixes every rule name.
    pub prefix: &'static str,
}

impl Level<'_> {
    fn n_rows(&self) -> u64 {
        self.values.len() as u64
    }

    fn n_null(&self) -> u64 {
        self.values.null_count() as u64
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

    fn shape(&self) -> Shape {
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
    estimate(p.freq.n_unique, n, p.freq.f1, p.freq.f2, &p.freq.capture_history, q)
}

/// arrow-cast with `safe: false`: a value that does not fit is an error, not a null.
pub(crate) fn arrow_cast(a: &dyn Array, to: &AT) -> Result<ArrayRef, String> {
    cast_with_options(a, to, &CastOptions { safe: false, ..Default::default() }).map_err(|e| e.to_string())
}

pub(crate) fn is_text(dt: &PT) -> bool {
    matches!(dt, PT::String | PT::Categorical(..) | PT::Enum(..))
}

pub(crate) fn is_float(dt: &PT) -> bool {
    matches!(dt, PT::Float32 | PT::Float64)
}

pub(crate) fn text_of(values: &ArrayRef) -> Result<LargeStringArray, String> {
    Ok(arrow_cast(values.as_ref(), &AT::LargeUtf8)?.as_string::<i64>().clone())
}

/// Row `i` as an exact integer (integer columns; Int128 arrives as decimal128(38, 0)).
fn int_at(a: &ArrayRef, i: u64) -> Option<i128> {
    let v = arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Decimal128(38, 0)).ok()?;
    let v = v.as_primitive::<Decimal128Type>();
    v.is_valid(0).then(|| v.value(0))
}

fn f64_at(a: &ArrayRef, i: u64) -> f64 {
    arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Float64).map_or(f64::NAN, |v| v.as_primitive::<Float64Type>().value(0))
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
        self.out.push(Candidate {
            predicted: body_size(&t, &self.shape) as u64,
            projected: body_size(&t, &self.shape.project(self.lvl.r, c)),
            target,
            rank,
            rule: format!("{}{rule}", self.lvl.prefix),
            evidence,
            outcome: Outcome::NotTried,
            reason: None,
        });
    }

    fn integers(&mut self, lo: i128, hi: i128, from: &str, evidence: &str) {
        let ev = format!("{evidence}min={lo} max={hi}");
        if lo >= 0 && hi <= 1 {
            self.push(Target::Boolean, Rank::Boolean, &format!("{from}→boolean"), ev.clone());
        }
        let uint = if lo >= 0 { narrowest_uint(hi) } else { None };
        let int = narrowest_int(lo, hi);
        if let Some(t) = uint.clone() {
            self.push(Target::Fixed(t), Rank::UInt, &format!("{from}→uint"), ev.clone());
        }
        if let Some(t) = int.clone() {
            self.push(Target::Fixed(t), Rank::Int, &format!("{from}→int"), ev.clone());
        }
        if uint.is_none() && int.is_none() {
            let p = digits(lo.unsigned_abs().max(hi.unsigned_abs()));
            if p <= 38 {
                self.push(Target::Fixed(AT::Decimal128(p, 0)), Rank::Decimal, &format!("{from}→decimal128"), format!("{ev} → p={p}"));
            }
        }
    }

    fn decimal(&mut self, scale: usize) {
        let (p, v) = (self.lvl.p, &self.lvl.values);
        let (Some(lo), Some(hi)) = (p.range.argmin, p.range.argmax) else { return };
        let unscaled = |i: u64| v.as_primitive::<Decimal128Type>().value(i as usize);
        let g = p.gcd.unwrap_or(1);
        let k = if g == 0 { scale } else { trailing_zeros10(g).min(scale) };
        let f = 10i128.pow(k as u32);
        let (lo, hi, s) = (unscaled(lo) / f, unscaled(hi) / f, scale - k);
        let ev = format!("gcd={g} → {k} trailing zeros, scale {scale}→{s}; ");
        if s == 0 {
            return self.integers(lo, hi, "decimal", &ev);
        }
        let prec = digits(lo.unsigned_abs().max(hi.unsigned_abs())).max(s as u8);
        if prec <= 38 {
            self.push(Target::Fixed(decimal_type(prec, s as i8)), Rank::Decimal, "decimal→decimal", format!("{ev}min={lo} max={hi} → p={prec} s={s}"));
        }
    }

    fn float(&mut self) {
        let (p, v) = (self.lvl.p, &self.lvl.values);
        let f = p.floats.expect("float columns have float stats");
        let ev = format!("n_nan={} n_inf={} n_fractional={} ", f.n_nan, f.n_inf, f.n_fractional);
        if let (0, 0, Some(lo), Some(hi)) = (f.n_nan, f.n_inf, p.range.argmin, p.range.argmax) {
            let (lo, hi) = (f64_at(v, lo), f64_at(v, hi));
            let top = lo.abs().max(hi.abs());
            if f.n_fractional == 0 {
                if top < 1e38 {
                    self.integers(lo as i128, hi as i128, "float", &ev);
                }
            } else if let Some(s) = f.max_frac_digits {
                let int_digits = if top < 1.0 { 0 } else { digits(top.floor() as u128) as u32 };
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
            self.push(Target::Fixed(AT::Float32), Rank::Float, "float64→float32", "n_f32_inexact=0".into());
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
                    self.push(Target::Fixed(AT::Date32), Rank::Date, "datetime→date32", format!("n_midnight={}", lvl.n()));
                }
                if coarse != unit {
                    self.push(Target::Fixed(AT::Timestamp(coarse, tz)), Rank::Timestamp, "datetime→timestamp", ev);
                }
            }
            PT::Duration(_) if coarse != unit => self.push(Target::Fixed(AT::Duration(coarse)), Rank::Timestamp, "duration→duration", ev),
            PT::Time if coarse != unit => self.push(Target::Fixed(time_type(coarse)), Rank::Time, "time→time", ev),
            _ => {}
        }
    }

    fn text(&mut self, params: &Params) -> Result<(), String> {
        let lvl = self.lvl;
        let (st, n) = (lvl.p.strings.as_ref().expect("string columns have string stats"), lvl.n());
        // Only cast the (≤5) top5 rows to text, not the whole column — the rest of
        // this rule never needs the column's text form.
        let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 {
            lvl.p
                .freq
                .top5_idx
                .iter()
                .map(|&i| text_of(&lvl.values.slice(i as usize, 1)).map(|t| t.value(0).to_lowercase()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let pair = params
            .boolean_pairs
            .iter()
            .map(|(t, f)| (t.to_lowercase(), f.to_lowercase()))
            .find(|(t, f)| !distinct.is_empty() && distinct.iter().all(|v| v == t || v == f));
        let sig = st.iso_max_sig_frac_digits.unwrap_or(0);
        if let Some((t, f)) = pair {
            self.push(Target::BoolPair(t.clone(), f.clone()), Rank::Boolean, "string→boolean", format!("distinct={distinct:?} pair=({t:?}, {f:?})"));
        } else if st.n_numeric_int == n && st.n_leading_zero == 0 && !st.int_overflow && st.int_min.is_some() {
            self.integers(st.int_min.unwrap(), st.int_max.unwrap(), "string", &format!("n_numeric_int={n} n_leading_zero=0 "));
        } else if st.n_numeric == n && st.n_leading_zero == 0 {
            let (i, f) = (st.max_int_digits.unwrap_or(0), st.max_frac_digits.unwrap_or(0));
            let (min_f, sig_d) = (st.min_frac_digits.unwrap_or(0), st.max_sig_digits.unwrap_or(0));
            let prec = (i + f).max(1);
            let ev = format!(
                "n_numeric={n} n_leading_zero=0 numeric_max_int_digits={i} numeric_max_frac_digits={f} \
                 numeric_min_frac_digits={min_f} numeric_max_sig_digits={sig_d}"
            );
            if prec <= 38 {
                self.push(Target::Fixed(decimal_type(prec as u8, f as i8)), Rank::Decimal, "string→decimal", format!("{ev} → p={prec} s={f}"));
            }
            if min_f < f && prec > 18 {
                let why = format!("{ev} → varying places, p={prec} > 18");
                if sig_d <= 6 {
                    self.push(Target::Fixed(AT::Float32), Rank::Float, "string→float32", why.clone());
                }
                if sig_d <= 15 {
                    self.push(Target::Fixed(AT::Float64), Rank::Float, "string→float64", why);
                }
            }
        } else if st.n_iso_date == n {
            self.push(Target::Fixed(AT::Date32), Rank::Date, "string→date32", format!("n_iso_date={n}"));
        } else if st.n_iso_time == n {
            self.push(Target::Fixed(time_type(iso_unit(sig))), Rank::Time, "string→time", format!("n_iso_time={n} iso_max_sig_frac_digits={sig}"));
        } else if st.n_iso_datetime == n {
            if st.iso_n_midnight == n {
                self.push(Target::Fixed(AT::Date32), Rank::Date, "string→date32", format!("n_iso_datetime={n} iso_n_midnight={n}"));
            } else {
                self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), None)),
                    Rank::Timestamp,
                    "string→timestamp",
                    format!("n_iso_datetime={n} iso_max_sig_frac_digits={sig}"),
                );
            }
        } else if st.n_iso_datetime_tz == n {
            let ev = format!("n_iso_datetime_tz={n} iso_n_offsets={} iso_max_sig_frac_digits={sig}", st.offsets.len());
            match st.offsets.iter().next() {
                Some(&m) if st.offsets.len() == 1 => {
                    self.push(Target::Fixed(AT::Timestamp(iso_unit(sig), Some(offset_tz(m).into()))), Rank::Timestamp, "string→timestamp", ev)
                }
                _ => self.push(
                    Target::TimestampWithOffset(iso_unit(sig)),
                    Rank::TimestampWithOffset,
                    "string→timestamp_with_offset (arrow.timestamp_with_offset)",
                    ev,
                ),
            }
        }
        let sum_len = lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(Target::Plain(AT::Utf8), Rank::Plain, "string→utf8", format!("sum_len={sum_len}"));
        }
        self.dictionary(params);
        Ok(())
    }

    fn dictionary(&mut self, params: &Params) {
        let (c, source) = self.lvl.cardinality();
        let (key, polars_key) = dictionary_keys(c);
        let threshold = params.categorical_threshold;
        let ev = format!("c={c:?} from {source} n_unique={} categorical_threshold={threshold}", self.lvl.p.freq.n_unique);
        self.push(Target::Dictionary(key, polars_key), Rank::Dictionary, "string→dictionary", ev);
        if c > threshold as f64 {
            let last = self.out.last_mut().unwrap();
            last.outcome = Outcome::Rejected;
            last.reason = Some(format!("c={c:?} > categorical_threshold={threshold}"));
        }
    }

    fn binary(&mut self) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(Target::Plain(AT::Binary), Rank::Plain, "binary→binary", format!("sum_len={sum_len}"));
        }
    }

    fn original(&mut self) {
        let lvl = self.lvl;
        self.out.push(Candidate {
            target: Target::Original(lvl.values.data_type().clone()),
            rank: Rank::Original,
            rule: format!("{}original", lvl.prefix),
            evidence: format!("measured size_bytes={}", lvl.size_bytes),
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
    let mut r = Rules { lvl, shape: lvl.shape(), out: Vec::new() };
    if lvl.n_rows() > 0 && lvl.n() == 0 {
        r.push(Target::Null, Rank::Null, "all-null→null", format!("n_null={} n_rows={}", lvl.n_null(), lvl.n_rows()));
    } else if lvl.n() > 0 {
        match lvl.dtype {
            PT::Int8 | PT::Int16 | PT::Int32 | PT::Int64 | PT::Int128 | PT::UInt8 | PT::UInt16 | PT::UInt32 | PT::UInt64 => {
                if let (Some(a), Some(b)) = (lvl.p.range.argmin, lvl.p.range.argmax) {
                    if let (Some(lo), Some(hi)) = (int_at(&lvl.values, a), int_at(&lvl.values, b)) {
                        r.integers(lo, hi, "integer", "");
                    }
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

use crate::describe::{parse_decimal, parse_iso};
use arrow_array::types::{Float32Type, Int16Type};
use arrow_array::{
    BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int16Array, StructArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};

fn exact_div(ns: i128, u: &TimeUnit) -> Option<i64> {
    let f = unit_ns(u);
    (ns % f == 0).then(|| ns / f).and_then(|v| i64::try_from(v).ok())
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
    Decimal128Array::from(v).with_precision_and_scale(38, scale).map(|a| Arc::new(a) as ArrayRef).map_err(|e| e.to_string())
}

/// `f` over every non-null text value; the first value it rejects fails the cast.
fn parsed<T>(text: &LargeStringArray, f: impl Fn(&str) -> Option<T>) -> Result<Vec<Option<T>>, String> {
    text.iter()
        .enumerate()
        .map(|(i, v)| v.map(|s| f(s).ok_or_else(|| format!("row {i}: {s:?} does not convert"))).transpose())
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
                let s = s.to_lowercase();
                if s == tt { Some(true) } else if s == ff { Some(false) } else { None }
            })?;
            Ok(Arc::new(BooleanArray::from(v)))
        }
        Target::Fixed(to) => match to {
            AT::Int8 | AT::Int16 | AT::Int32 | AT::Int64 | AT::UInt8 | AT::UInt16 | AT::UInt32 | AT::UInt64 => {
                arrow_cast(decimal_array(parsed(text, |s| parse_decimal(s.as_bytes(), 0))?, 0)?.as_ref(), to)
            }
            AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => {
                arrow_cast(decimal_array(parsed(text, |x| parse_decimal(x.as_bytes(), *s as u32))?, *s)?.as_ref(), to)
            }
            AT::Float32 => Ok(Arc::new(Float32Array::from(parsed(text, |s| s.parse::<f32>().ok())?))),
            AT::Float64 => Ok(Arc::new(Float64Array::from(parsed(text, |s| s.parse::<f64>().ok())?))),
            AT::Date32 => Ok(Arc::new(Date32Array::from(parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                (v.nanos == 0 && v.offset_minutes.is_none()).then_some(())?;
                i32::try_from(v.days?).ok()
            })?))),
            AT::Time32(u) | AT::Time64(u) => Ok(time_array(*u, parsed(text, |s| exact_div(parse_iso(s.as_bytes())?.nanos as i128, u))?)),
            AT::Timestamp(u, tz) => Ok(timestamp_array(*u, parsed(text, |s| exact_div(parse_iso(s.as_bytes())?.epoch_ns(), u))?, tz.clone())),
            t => Err(format!("no string conversion to {}", pa_name(t))),
        },
        Target::TimestampWithOffset(u) => {
            let parts = parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                Some((exact_div(v.epoch_ns(), u)?, i16::try_from(v.offset_minutes?).ok()?))
            })?;
            let ts = timestamp_array(*u, parts.iter().map(|p| Some(p.map_or(0, |p| p.0))).collect(), Some("UTC".into()));
            let off: ArrayRef = Arc::new(Int16Array::from(parts.iter().map(|p| p.map_or(0, |p| p.1)).collect::<Vec<i16>>()));
            let AT::Struct(fields) = timestamp_with_offset(*u) else { unreachable!() };
            StructArray::try_new(fields, vec![ts, off], text.logical_nulls()).map(|a| Arc::new(a) as ArrayRef).map_err(|e| e.to_string())
        }
        Target::Dictionary(..) => arrow_cast(arrow_cast(text, &AT::Utf8)?.as_ref(), &t.arrow_type()),
        Target::Plain(to) => arrow_cast(text, to),
        t => Err(format!("no string conversion to {}", pa_name(&t.arrow_type()))),
    }
}

/// Float → Decimal through the exact digits of each value's shortest round-trip
/// representation (ryu), never multiply-and-round.
pub(crate) fn float_to_decimal(src: &ArrayRef, to: &AT) -> Result<ArrayRef, String> {
    let (AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s)) = to else { return Err("not a decimal".into()) };
    let f32_src = src.data_type() == &AT::Float32;
    let values = arrow_cast(src.as_ref(), &AT::Float64)?;
    let mut buf = ryu::Buffer::new();
    let unscaled: Vec<Option<i128>> = values
        .as_primitive::<Float64Type>()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|x| {
                let repr = if f32_src { buf.format_finite(x as f32).to_string() } else { buf.format_finite(x).to_string() };
                decimal_from_repr(&repr, *s as u32).ok_or_else(|| format!("row {i}: {repr} does not fit scale {s}"))
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
        _ if is_text(lvl.dtype) => from_text(t, &text_of(src)?),
        Target::Fixed(to @ (AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..))) if is_float(lvl.dtype) => float_to_decimal(src, to),
        Target::Boolean | Target::Fixed(_) | Target::Plain(_) => arrow_cast(src.as_ref(), &t.arrow_type()),
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

pub(crate) fn first_mismatch(a: &ArrayRef, b: &ArrayRef) -> Result<(), String> {
    if a.to_data() == b.to_data() {
        return Ok(());
    }
    let i = (0..a.len()).find(|&i| a.slice(i, 1).to_data() != b.slice(i, 1).to_data()).unwrap_or(0);
    Err(format!("row {i}: {} round-trips to {}", render(a, i), render(b, i)))
}

/// Float sources: every recast value converts back to the original float (NaN = NaN,
/// -0.0 = 0.0). Decimals come back through their text (correctly rounded parse).
pub(crate) fn verify_float(src: &ArrayRef, recast: &ArrayRef) -> Result<(), String> {
    let f32_src = src.data_type() == &AT::Float32;
    let decimal = matches!(recast.data_type(), AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..));
    let back: Vec<Option<f64>> = if decimal {
        let t = arrow_cast(recast.as_ref(), &AT::Utf8)?;
        t.as_string::<i32>()
            .iter()
            .map(|v| v.map(|s| if f32_src { s.parse::<f32>().map_or(f64::NAN, f64::from) } else { s.parse::<f64>().unwrap_or(f64::NAN) }))
            .collect()
    } else if f32_src {
        arrow_cast(recast.as_ref(), &AT::Float32)?.as_primitive::<Float32Type>().iter().map(|v| v.map(f64::from)).collect()
    } else {
        arrow_cast(recast.as_ref(), &AT::Float64)?.as_primitive::<Float64Type>().iter().collect()
    };
    let orig = arrow_cast(src.as_ref(), &AT::Float64)?;
    for (i, (a, b)) in orig.as_primitive::<Float64Type>().iter().zip(back).enumerate() {
        if let (Some(a), Some(b)) = (a, b) {
            if !(a == b || (a.is_nan() && b.is_nan())) {
                return Err(format!("row {i}: {a} round-trips to {b}"));
            }
        }
    }
    Ok(())
}

/// String sources: the recast values, rendered to text by arrow-cast, equal the
/// original text by value (canonical digits; parse_iso components).
pub(crate) fn verify_text(t: &Target, text: &LargeStringArray, recast: &ArrayRef) -> Result<(), String> {
    let bad = |i: usize, got: &str| Err(format!("row {i}: {:?} round-trips to {got:?}", text.value(i)));
    let iso = |s: &str| parse_iso(s.as_bytes());
    match t {
        Target::Dictionary(..) | Target::Plain(_) => {
            first_mismatch(&(Arc::new(text.clone()) as ArrayRef), &arrow_cast(recast.as_ref(), &AT::LargeUtf8)?)
        }
        Target::Boolean | Target::BoolPair(..) => {
            let (tt, ff) = match t {
                Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
                _ => ("1", "0"),
            };
            let b = recast.as_boolean();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let want = if b.value(i) { tt } else { ff };
                if text.value(i).to_lowercase() != want {
                    return bad(i, want);
                }
            }
            Ok(())
        }
        Target::TimestampWithOffset(_) => {
            let s = recast.as_struct();
            let back = arrow_cast(s.column(0).as_ref(), &AT::Utf8)?;
            let (back, off) = (back.as_string::<i32>(), s.column(1).as_primitive::<Int16Type>());
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (iso(text.value(i)), iso(back.value(i)));
                let same = matches!((a, b), (Some(a), Some(b)) if a.epoch_ns() == b.epoch_ns() && a.offset_minutes == Some(off.value(i) as i32));
                if !same {
                    return bad(i, back.value(i));
                }
            }
            Ok(())
        }
        Target::Fixed(to) => {
            let back = arrow_cast(recast.as_ref(), &AT::Utf8)?;
            let back = back.as_string::<i32>();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (text.value(i), back.value(i));
                let same = match to {
                    AT::Date32 | AT::Timestamp(..) => matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns()),
                    AT::Time32(_) | AT::Time64(_) => matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.nanos == y.nanos),
                    _ => canon(a) == canon(b),
                };
                if !same {
                    return bad(i, b);
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Row-by-row check that `recast` holds the original values (Spec B §5.4 step 3).
pub(crate) fn verify(t: &Target, lvl: &Level, recast: &ArrayRef) -> Result<(), String> {
    if matches!(t, Target::Original(_)) {
        return Ok(());
    }
    let src = &lvl.values;
    if recast.len() != src.len() {
        return Err(format!("length {} ≠ {}", recast.len(), src.len()));
    }
    if let Some(i) = (0..src.len()).find(|&i| recast.is_null(i) != src.is_null(i)) {
        return Err(format!("row {i}: null mismatch"));
    }
    match t {
        Target::Null => Ok(()),
        _ if is_text(lvl.dtype) => verify_text(t, &text_of(src)?, recast),
        _ if is_float(lvl.dtype) => verify_float(src, recast),
        _ => first_mismatch(src, &arrow_cast(recast.as_ref(), src.data_type())?),
    }
}

/// Some value's text changed although its value did not (Spec B §5.4).
pub(crate) fn lossy(t: &Target, lvl: &Level, recast: &ArrayRef) -> bool {
    match t {
        Target::Original(_) | Target::Null | Target::Dictionary(..) | Target::Plain(_) => false,
        _ if is_text(lvl.dtype) => match (text_of(&lvl.values), arrow_cast(recast.as_ref(), &AT::LargeUtf8)) {
            (Ok(a), Ok(b)) => a.iter().zip(b.as_string::<i64>().iter()).any(|(x, y)| x != y),
            _ => true, // no text rendering (timestamp_with_offset): the format necessarily changes
        },
        Target::Fixed(AT::Float32 | AT::Float64) => false,
        _ if is_float(lvl.dtype) => arrow_cast(lvl.values.as_ref(), &AT::Float64)
            .map(|f| f.as_primitive::<Float64Type>().iter().flatten().any(|x| x == 0.0 && x.is_sign_negative()))
            .unwrap_or(false),
        _ => false,
    }
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
    fn python_repr_for_enum_categories() {
        // Verified against: python -c "import polars as pl; print(str(pl.Enum([...])))"
        let k = AT::UInt8;
        let dict = Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type();
        let values = ["it's".to_string(), "a\"b".to_string(), "x\\y".to_string(), "n\nl".to_string()];
        assert_eq!(pl_name(&dict, "x", Some(&values), &k), "Enum(categories=[\"it's\", 'a\"b', 'x\\\\y', 'n\\nl'])");
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

    use crate::describe::describe_one;
    use crate::shared::to_arrow_rs;
    use polars::prelude::{CompatLevel, DataType as PT, IntoSeries, NamedFrom, NewChunkedArray, Series, TimeUnit as PTimeUnit};

    pub(super) fn params() -> Params {
        Params { seed: 0, zstd_level: 1, population_rows: None, categorical_threshold: 10_000, boolean_pairs: vec![("true".into(), "false".into())] }
    }

    fn types(s: Series, p: &Params) -> Vec<(String, Outcome)> {
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level {
            column: "x", dtype: s.dtype(), values: to_arrow_rs(&s, CompatLevel::oldest()).unwrap(), p: &d.outer,
            n_midnight: d.n_midnight, size_bytes: 0, est: level_estimate(&d.outer, d.n_rows - d.n_null, None), r: 1.0, prefix: "",
        };
        candidates(&lvl, p).unwrap().iter().map(|c| (pa_name(&c.target.arrow_type()), c.outcome)).collect()
    }

    fn names(s: Series) -> Vec<String> {
        types(s, &params()).into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn integer_rules() {
        assert_eq!(names(Series::new("x".into(), &[0i64, 1])), ["bool", "uint8", "int8", "int64"]);
        assert_eq!(names(Series::new("x".into(), &[-200i64, 5])), ["int16", "int64"]);
    }

    #[test]
    fn decimal_scale_reduced_by_gcd() {
        let dec = polars::prelude::Int128Chunked::from_slice("x".into(), &[120, 340]).into_decimal_unchecked(Some(10), 2).into_series();
        assert_eq!(names(dec), ["decimal32(2, 1)", "decimal128(10, 2)"]);
    }

    #[test]
    fn float_rules() {
        assert_eq!(names(Series::new("x".into(), &[123.45f64, 99.99])), ["decimal32(5, 2)", "double"]);
        assert_eq!(names(Series::new("x".into(), &[0.5f64, 0.25])), ["decimal32(2, 2)", "float", "double"]);
        assert_eq!(names(Series::new("x".into(), &[0.1f64, f64::NAN])), ["double"]);
    }

    #[test]
    fn string_rules() {
        let dict = "dictionary<values=string, indices=uint8, ordered=0>";
        assert_eq!(names(Series::new("x".into(), &["007", "12"])), ["string", dict, "large_string"]);
        assert_eq!(
            names(Series::new("x".into(), &["1234567890.1", "0.00000012345"])),
            ["decimal128(21, 11)", "double", "string", dict, "large_string"]
        );
        let offsets = names(Series::new("x".into(), &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]));
        assert_eq!(offsets[0], "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>");
        assert_eq!(names(Series::new("x".into(), &["True", "false"]))[0], "bool");
    }

    #[test]
    fn temporal_rules() {
        let days = Series::new("x".into(), &[0i64, 86_400_000_000]).cast(&PT::Datetime(PTimeUnit::Microseconds, None)).unwrap();
        assert_eq!(names(days), ["date32[day]", "timestamp[s]", "timestamp[us]"]);
    }

    #[test]
    fn dictionary_gate_rejects() {
        let p = Params { categorical_threshold: 1, ..params() };
        let got = types(Series::new("x".into(), &["a", "b", "a", "b"]), &p);
        assert!(got.iter().any(|(t, o)| t.starts_with("dictionary") && *o == Outcome::Rejected));
    }

    #[test]
    fn dictionary_cardinality_floors_at_n_unique() {
        // 3 distinct values, but the estimate handed to Level is (artificially)
        // below that — cardinality() must still report at least n_unique.
        let s = Series::new("x".into(), &["a", "b", "c"]);
        let d = describe_one(&s, 0).unwrap();
        assert_eq!(d.outer.freq.n_unique, 3);
        let lvl = Level {
            column: "x", dtype: s.dtype(), values: to_arrow_rs(&s, CompatLevel::oldest()).unwrap(), p: &d.outer,
            n_midnight: d.n_midnight, size_bytes: 0,
            est: Estimate {
                est_cardinality: 1.0,
                est_low: Some(1.0),
                est_high: Some(1.0),
                method: crate::cardinality_estimators::Method::Chao1,
            },
            r: 1.0, prefix: "",
        };
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
        let ts = s.column(0).as_primitive::<arrow_array::types::TimestampSecondType>();
        assert_eq!((ts.value(0), s.column(1).as_primitive::<arrow_array::types::Int16Type>().value(0)), (1_704_430_800, 300));
        assert!(a.is_null(1));
        assert!(verify_text(&Target::TimestampWithOffset(TimeUnit::Second), &text, &a).is_ok());
    }

    #[test]
    fn verification_reports_the_first_mismatch() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 3]));
        assert!(first_mismatch(&a, &b).unwrap_err().starts_with("row 1:"));
        let text = LargeStringArray::from(vec!["2300-01-01T00:00:00.123456789"]);
        assert!(from_text(&Target::Fixed(AT::Timestamp(TimeUnit::Nanosecond, None)), &text).is_err()); // beyond i64 ns
    }

    #[test]
    fn timestamp_with_offset_struct_children_have_no_nulls() {
        let text = LargeStringArray::from(vec![Some("2024-01-05T10:00+05:00"), None, Some("2024-01-06T00:00Z")]);
        let a = from_text(&Target::TimestampWithOffset(TimeUnit::Second), &text).unwrap();
        let s = a.as_struct();
        assert_eq!(s.column(0).null_count(), 0);
        assert_eq!(s.column(1).null_count(), 0);
        let shape = Shape { n: 3.0, nulls: 1.0, ..Default::default() };
        assert_eq!(ipc_body_bytes(a.as_ref(), None).unwrap() as f64, body_size(&timestamp_with_offset(TimeUnit::Second), &shape));
    }
}
