//! Row-by-row verification that a cast preserved every value (spec 2026-09-26 §5.4).

use super::*;
use crate::techniques::describe::{parse_iso, IsoValue};
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int16Type};
use arrow_array::{Array, ArrayRef, LargeStringArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as AT, TimeUnit};
use std::sync::Arc;

/// Rows compared per ArrayData equality check in `first_mismatch`.
pub(crate) const MISMATCH_CHUNK: usize = 1 << 16;

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
pub(crate) fn parse_back(s: &str, f32_src: bool) -> Result<f64, String> {
    if f32_src {
        s.parse::<f32>().map(f64::from)
    } else {
        s.parse::<f64>()
    }
    .map_err(|_| format!("{s:?} does not parse as a float"))
}

/// Exact powers of ten: 10^s is a float exactly while 5^s fits the mantissa —
/// s ≤ 22 in f64 (5^22 < 2^53), s ≤ 10 in f32 (5^10 < 2^24).
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];
const POW10_F32: [f32; 11] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];

/// The decimal `unscaled × 10^-scale` as the float its text parses to (`parse_back`),
/// without the text: when the unscaled value and 10^scale are both exact floats, one
/// IEEE division rounds the exact quotient correctly, as the parse does. None outside
/// that range (the caller falls back to the text).
fn decimal_to_float(unscaled: i128, scale: i8, f32_src: bool) -> Option<f64> {
    let s = usize::try_from(scale).ok()?;
    if f32_src {
        let p = *POW10_F32.get(s)?;
        (unscaled.unsigned_abs() <= 1 << 24).then(|| f64::from(unscaled as f32 / p))
    } else {
        let p = *POW10_F64.get(s)?;
        (unscaled.unsigned_abs() <= 1 << 53).then(|| unscaled as f64 / p)
    }
}

/// A decimal recast's values as floats: by `decimal_to_float` when every value is in
/// its exact range, else through the text (`parse_back`).
fn decimal_back(recast: &ArrayRef, f32_src: bool) -> Result<Vec<Option<f64>>, String> {
    let s = match recast.data_type() {
        AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => *s,
        t => return Err(format!("{t} is not a decimal")),
    };
    let wide = arrow_cast(recast.as_ref(), &AT::Decimal128(38, s))?;
    let fast: Option<Vec<Option<f64>>> = wide
        .as_primitive::<Decimal128Type>()
        .iter()
        .map(|v| match v {
            None => Some(None),
            Some(u) => decimal_to_float(u, s, f32_src).map(Some),
        })
        .collect();
    if let Some(back) = fast {
        return Ok(back);
    }
    let t = arrow_cast(recast.as_ref(), &AT::Utf8)?;
    t.as_string::<i32>()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|s| parse_back(s, f32_src).map_err(|e| format!("row {i}: {e}")))
                .transpose()
        })
        .collect()
}

/// Float sources: every recast value converts back to the original float (NaN = NaN,
/// -0.0 = 0.0). Decimals come back as the float their text parses to (`decimal_back`).
pub(crate) fn verify_float(src: &ArrayRef, recast: &ArrayRef) -> Result<(), String> {
    let f32_src = src.data_type() == &AT::Float32;
    let decimal = matches!(
        recast.data_type(),
        AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..)
    );
    let back: Vec<Option<f64>> = if decimal {
        decimal_back(recast, f32_src)?
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

/// The error for text row `i` whose recast value renders as `got`.
pub(crate) fn text_mismatch<T>(text: &LargeStringArray, i: usize, got: &str) -> Result<T, String> {
    Err(format!(
        "row {i}: {:?} round-trips to {got:?}",
        text.value(i)
    ))
}

pub(crate) fn iso(s: &str) -> Option<IsoValue> {
    parse_iso(s.as_bytes())
}

/// Text → Boolean: every value matches the text of the recast bool (`1` / `0`, or the pair).
pub(crate) fn verify_boolean_text(
    t: &Target,
    text: &LargeStringArray,
    recast: &ArrayRef,
) -> Result<(), String> {
    let (tt, ff) = match t {
        Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
        _ => ("1", "0"),
    };
    let b = recast.as_boolean_opt().ok_or("recast is not boolean")?;
    for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
        let want = if b.value(i) { tt } else { ff };
        if !lower_eq(text.value(i), want) {
            return text_mismatch(text, i, want);
        }
    }
    Ok(())
}

/// Text → timestamp_with_offset: same instant, and the offset column is the text's offset.
pub(crate) fn verify_offset_text(text: &LargeStringArray, recast: &ArrayRef) -> Result<(), String> {
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
            return text_mismatch(text, i, back.value(i));
        }
    }
    Ok(())
}

/// Whether the text `a` and its recast rendering `b` hold the same value, for a recast to `to`.
pub(crate) fn same_text_value(to: &AT, a: &str, b: &str) -> bool {
    match to {
        AT::Date32 => {
            matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns())
        }
        // A fixed-offset target renders the same offset as the text only when
        // every value actually had one; a naive target renders no offset at all
        // ("Z" and "UTC" both parse back to offset_minutes = Some(0), so this
        // must compare the *parsed* offsets, not the rendered strings).
        AT::Timestamp(_, tz) => match (iso(a), iso(b)) {
            (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns() => match tz {
                Some(_) => x.offset_minutes.is_some() && x.offset_minutes == y.offset_minutes,
                None => x.offset_minutes.is_none(),
            },
            _ => false,
        },
        AT::Time32(_) | AT::Time64(_) => {
            matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.nanos == y.nanos)
        }
        _ => canon(a) == canon(b),
    }
}

/// String sources: the recast values, rendered to text by arrow-cast, equal the
/// original text by value (canonical digits; parse_iso components). Returns the
/// LargeUtf8 rendering when it made one, so `lossy` need not render again.
pub(crate) fn verify_text(
    t: &Target,
    text: &LargeStringArray,
    recast: &ArrayRef,
) -> Result<Option<ArrayRef>, String> {
    match t {
        Target::Dictionary(..) | Target::Plain(_) => first_mismatch(
            &(Arc::new(text.clone()) as ArrayRef),
            &arrow_cast(recast.as_ref(), &AT::LargeUtf8)?,
        )
        .map(|_| None),
        Target::Boolean | Target::BoolPair(..) => {
            verify_boolean_text(t, text, recast).map(|_| None)
        }
        Target::TimestampWithOffset(_) => verify_offset_text(text, recast).map(|_| None),
        Target::Fixed(to) => {
            let rendered = arrow_cast(recast.as_ref(), &AT::LargeUtf8)?;
            let back = rendered.as_string::<i64>();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                if !same_text_value(to, text.value(i), back.value(i)) {
                    return text_mismatch(text, i, back.value(i));
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
pub(crate) fn logical_is_null(nulls: Option<&NullBuffer>, i: usize) -> bool {
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

/// Proof by statistics (streaming spec §5.1 step 4): Ok when the statistics show that
/// `t` holds every value of the level. Every rule's own condition already proves its
/// candidate except:
/// - Timestamp → Date32: arrow-cast converts through chrono, which errors on dates
///   outside its range (checked on the physical `int_range`);
/// - string → Boolean: "-0" passes the 0/1 integer rule but is not the text "0";
/// - string → Float32 / Float64: a value may not round-trip;
/// - string → Timestamp(ns) / timestamp_with_offset(ns): the int64 nanosecond range.
pub(crate) fn prove(t: &Target, lvl: &Level) -> Result<(), String> {
    if let (Target::Fixed(AT::Date32), AT::Timestamp(u, _), Some((lo, hi))) =
        (t, lvl.values.data_type(), lvl.int_range)
    {
        let per_s = 1_000_000_000 / unit_ns(u);
        let in_chrono = |v: i128| {
            i64::try_from(v.div_euclid(per_s))
                .is_ok_and(|s| chrono::DateTime::from_timestamp(s, 0).is_some())
        };
        if !(in_chrono(lo) && in_chrono(hi)) {
            return Err(format!(
                "timestamp range {lo}..{hi} is outside chrono's dates"
            ));
        }
    }
    let Some(st) = lvl.p.strings.as_ref().filter(|_| is_text(lvl.dtype)) else {
        return Ok(());
    };
    match t {
        Target::Boolean if st.n_neg_zero > 0 => Err(format!("n_neg_zero={}", st.n_neg_zero)),
        Target::Fixed(AT::Float32) if st.n_f32_roundtrip_fail > 0 => {
            Err(format!("n_f32_roundtrip_fail={}", st.n_f32_roundtrip_fail))
        }
        Target::Fixed(AT::Float64) if st.n_f64_roundtrip_fail > 0 => {
            Err(format!("n_f64_roundtrip_fail={}", st.n_f64_roundtrip_fail))
        }
        Target::Fixed(AT::Timestamp(TimeUnit::Nanosecond, _))
        | Target::TimestampWithOffset(TimeUnit::Nanosecond) => {
            match (st.iso_instant_min, st.iso_instant_max) {
                (Some(lo), Some(hi)) if lo < i64::MIN as i128 || hi > i64::MAX as i128 => {
                    Err(format!("iso_instant range {lo}..{hi} ns exceeds int64"))
                }
                _ => Ok(()),
            }
        }
        _ => Ok(()),
    }
}

/// `lossy` from statistics alone: arrow-cast renders each value from the value itself
/// (integers canonically; decimals with exactly `scale` places; floats with ryu; times
/// and timestamps with chrono's 0/3/6/9 fraction digits, `T`, and `Z` for UTC), so the
/// scanner's per-value counters decide it exactly.
pub(crate) fn lossy_by_stats(t: &Target, lvl: &Level) -> bool {
    match t {
        Target::Original(_) | Target::Null | Target::Dictionary(..) | Target::Plain(_) => false,
        _ if is_text(lvl.dtype) => {
            let Some(st) = lvl.p.strings.as_ref() else {
                return true;
            };
            match t {
                Target::Boolean | Target::TimestampWithOffset(_) => lvl.n() > 0,
                Target::BoolPair(..) => {
                    lvl.few_distinct.iter().any(|v| v != "true" && v != "false")
                }
                Target::Fixed(to) => match to {
                    AT::Float32 => st.n_f32_render_diff > 0,
                    AT::Float64 => st.n_f64_render_diff > 0,
                    AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => {
                        let s = Some(*s as u32);
                        st.raw_frac_min != s
                            || st.raw_frac_max != s
                            || st.n_int_lead0 > 0
                            || st.n_neg_zero > 0
                    }
                    AT::Date32 => st.n_iso_datetime > 0,
                    AT::Time32(_) | AT::Time64(_) => st.n_iso_time_noncanonical > 0,
                    AT::Timestamp(_, tz) => {
                        st.n_iso_time_noncanonical > 0
                            || st.n_iso_space_sep > 0
                            || (tz.is_some() && st.n_iso_offset_noncanonical > 0)
                    }
                    _ => st.n_neg_zero > 0, // integers: only "-0" renders differently
                },
                _ => false,
            }
        }
        Target::Fixed(AT::Float32 | AT::Float64) => false,
        _ if is_float(lvl.dtype) => lvl.p.floats.is_some_and(|f| f.n_neg_zero > 0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::arrow_io::export_series;
    use crate::recommenders::engine::choose::tests::describe_one;
    use arrow_array::Int64Array;
    use polars::prelude::{CompatLevel, DataType as PT, NamedFrom, Series, TimeUnit as PTimeUnit};

    use crate::common::ipc_sizes::ipc_body_bytes;
    use arrow_array::{Float32Array, Float64Array};

    /// A decimal's text as `verify_float`'s text path renders it.
    fn decimal_text(u: i128, scale: i8) -> String {
        let d = decimal_array(vec![Some(u)], scale).unwrap();
        arrow_cast(d.as_ref(), &AT::Utf8)
            .unwrap()
            .as_string::<i32>()
            .value(0)
            .to_owned()
    }

    #[test]
    fn decimal_to_float_matches_the_text_parse() {
        // Every in-range (unscaled, scale) gives exactly what its text parses to.
        let (b53, b24) = (1i128 << 53, 1i128 << 24);
        let fixed = [
            (b53, 0, false),
            (-b53, 0, false),
            (b53, 22, false),
            (-b53, 22, false),
            (b24, 0, true),
            (-b24, 0, true),
            (b24, 10, true),
            (-b24, 10, true),
            (1, 22, false),
            (1, 10, true),
            (0, 0, false),
            (0, 22, false),
            (0, 10, true),
        ];
        for (u, s, f32_src) in fixed {
            let want = parse_back(&decimal_text(u, s), f32_src).unwrap();
            assert_eq!(decimal_to_float(u, s, f32_src), Some(want), "u={u} s={s}");
        }
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17; // xorshift64
            let scale = (x % 23) as i8;
            for f32_src in [false, true] {
                let bound: i128 = if f32_src { 1 << 24 } else { 1 << 53 };
                let sign = if x & 1 == 0 { 1 } else { -1 };
                let u = ((x >> 8) as i128 % (bound + 1)) * sign;
                let Some(fast) = decimal_to_float(u, scale, f32_src) else {
                    assert!(f32_src && scale > 10, "u={u} scale={scale}");
                    continue;
                };
                let text = decimal_text(u, scale);
                assert_eq!(
                    fast,
                    parse_back(&text, f32_src).unwrap(),
                    "{text} f32={f32_src}"
                );
            }
        }
    }

    #[test]
    fn decimal_to_float_refuses_inexact_operands() {
        assert_eq!(decimal_to_float((1 << 53) + 1, 0, false), None);
        assert_eq!(decimal_to_float(1, 23, false), None);
        assert_eq!(decimal_to_float((1 << 24) + 1, 0, true), None);
        assert_eq!(decimal_to_float(1, 11, true), None);
        assert_eq!(decimal_to_float(1, -1, false), None);
        assert_eq!(decimal_to_float(-2_500, 3, false), Some(-2.5));
    }

    #[test]
    fn verify_float_checks_decimals_on_both_paths() {
        let dec = |v: Vec<Option<i128>>, p: u8, s: i8| {
            arrow_cast(decimal_array(v, s).unwrap().as_ref(), &decimal_type(p, s)).unwrap()
        };
        let src: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.1), Some(-2.5), None]));
        assert!(verify_float(&src, &dec(vec![Some(100), Some(-2_500), None], 9, 3)).is_ok());
        let err = verify_float(&src, &dec(vec![Some(100), Some(-2_501), None], 9, 3)).unwrap_err();
        assert!(err.starts_with("row 1"), "{err}");
        // Float32 source against a Decimal64 recast (precision 12).
        let src32: ArrayRef = Arc::new(Float32Array::from(vec![Some(0.1f32), Some(-2.5), None]));
        assert!(verify_float(&src32, &dec(vec![Some(100), Some(-2_500), None], 12, 3)).is_ok());
        let err =
            verify_float(&src32, &dec(vec![Some(100), Some(-2_501), None], 12, 3)).unwrap_err();
        assert!(err.starts_with("row 1"), "{err}");
        // Unscaled 10^19 > 2^53: the text path.
        let big: ArrayRef = Arc::new(Float64Array::from(vec![Some(1e17)]));
        assert!(verify_float(&big, &dec(vec![Some(10i128.pow(19))], 21, 2)).is_ok());
        // 1e17 + 100 rounds to 1e17 + 96 (ulp 16), not 1e17.
        let err = verify_float(&big, &dec(vec![Some(10i128.pow(19) + 10_000)], 21, 2)).unwrap_err();
        assert!(err.starts_with("row 0"), "{err}");
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
                d.conclusions(10_000).0.est,
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
            d.conclusions(10_000).0.est,
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
            d.conclusions(10_000).0.est,
            "",
        )
        .unwrap();
        let target = Target::Fixed(AT::Timestamp(TimeUnit::Microsecond, Some("+05:00".into())));
        assert!(verify(&target, &lvl, &recast).is_err());
    }
}
