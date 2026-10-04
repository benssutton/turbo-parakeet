//! Text forms of values: canonical decimal strings, arrow-rs casts and rendering.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_cast::cast::{cast_with_options, CastOptions};
use arrow_schema::DataType as AT;

/// A decimal or exponent string as (negative, significant digits, point) with
/// value = 0.DIGITS × 10^point; zero is (false, "", 0). Equal values give equal
/// canonical forms regardless of formatting ("1.50" ≡ "1.5", "0.00120" ≡ "1.2e-3").
/// The digits borrow the input (`head` then `tail`, the dot skipped): no allocation.
///
/// Precondition: `s` is a numeric literal already validated by describe/scanners.rs's
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

/// Row 0 of `a` as text: arrow-rs's cast to Utf8, the one rendering of `min` / `max`
/// in every recommender (spec 2026-10-01 §6, §13.1). None for a null, or a value with
/// no text form (e.g. non-UTF-8 binary).
pub(crate) fn render_value(a: &dyn Array) -> Option<String> {
    let s = arrow_cast(a.slice(0, 1).as_ref(), &AT::Utf8).ok()?;
    let s = s.as_string::<i32>();
    s.is_valid(0).then(|| s.value(0).to_string())
}

pub(crate) fn render(a: &ArrayRef, i: usize) -> String {
    render_value(a.slice(i, 1).as_ref()).unwrap_or_else(|| "null".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn render_value_is_arrow_rs_text() {
        use arrow_array::{
            BooleanArray, Date32Array, Decimal128Array, Float64Array, TimestampMicrosecondArray,
        };
        let one = |a: ArrayRef| render_value(a.as_ref());
        assert_eq!(
            one(Arc::new(Float64Array::from(vec![2.5]))).as_deref(),
            Some("2.5")
        );
        assert_eq!(
            one(Arc::new(BooleanArray::from(vec![false]))).as_deref(),
            Some("false")
        );
        assert_eq!(
            one(Arc::new(Date32Array::from(vec![19_724]))).as_deref(),
            Some("2024-01-02")
        );
        let ts = TimestampMicrosecondArray::from(vec![1_704_164_645_000_000]);
        assert_eq!(one(Arc::new(ts)).as_deref(), Some("2024-01-02T03:04:05"));
        let dec = Decimal128Array::from(vec![150])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(one(Arc::new(dec)).as_deref(), Some("1.50"));
        assert_eq!(one(Arc::new(Float64Array::from(vec![None]))), None);
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
}
