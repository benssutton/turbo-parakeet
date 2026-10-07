//! Numeric-string and ISO 8601 byte scanners and their statistics (group C).

use crate::common::text::canon;
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// describe — numeric-string and ISO 8601 scanners (group C)
// ─────────────────────────────────────────────────────────────────────────────
//
// Hand-written byte state machines: one forward pass per value, no backtracking
// and no regex, so run time is linear in the input for any string — adversarial
// input cannot trigger catastrophic backtracking. The Python implementations use
// the same grammar as anchored regexes on the Rust `regex` engine
// (analytics/describe/_values.py).
//
// Numeric grammar: -?[0-9]+(\.[0-9]+)?   (ASCII digits only)
// ISO grammar:     date     YYYY-MM-DD (calendar-valid, Gregorian leap years)
//                  time     HH:MM[:SS[.f{1,9}]]   (00–23, 00–59, 00–59; no leap second)
//                  datetime date ('T' | ' ') time
//                  offset   'Z' | ±HH:MM          (uppercase T and Z only)
//
// Leading-zero rule:
// An integer-looking string with a leading zero ("007") must stay a String:
// identifiers such as UUID fragments, account numbers or zip codes can be all
// digits with significant leading zeros, and casting to an integer would lose
// them. A value with a single decimal point ("007.50") is unlikely to be an
// identifier, so only numeric equivalence matters for it — differing leading or
// trailing zeros are acceptable.

pub(crate) struct Numeric {
    pub is_int: bool,
    /// Integer-looking with a leading zero ("007", "-012"; not "0" or "-0").
    pub leading_zero: bool,
    /// Significant integer-part digits (leading zeros ignored).
    pub int_digits: u32,
    /// Fraction digits with trailing zeros removed.
    pub frac_digits: u32,
    /// Significant digits of the whole value: leading zeros (across the dot) and
    /// trailing fraction zeros removed ("0.00120" → 2, "1200" → 4).
    pub sig_digits: u32,
    /// Fraction digits as written, trailing zeros kept ("1.50" → 2; an integer 0).
    pub raw_frac_digits: u32,
    /// The integer part has more than one digit and starts with 0 ("007", "00.5").
    pub int_lead0: bool,
}

pub(crate) fn scan_numeric(b: &[u8]) -> Option<Numeric> {
    let body = b.strip_prefix(b"-").unwrap_or(b);
    let int_len = body.iter().take_while(|c| c.is_ascii_digit()).count();
    if int_len == 0 {
        return None;
    }
    let (int, rest) = body.split_at(int_len);
    let frac = match rest {
        [] => &b""[..],
        [b'.', frac @ ..] if !frac.is_empty() && frac.iter().all(u8::is_ascii_digit) => frac,
        _ => return None,
    };
    let raw_frac_digits = frac.len() as u32;
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
    let significant = int
        .iter()
        .position(|&c| c != b'0')
        .map_or(0, |p| int.len() - p);
    let sig_digits = if significant > 0 {
        significant + frac.len()
    } else {
        frac.iter()
            .position(|&c| c != b'0')
            .map_or(0, |p| frac.len() - p)
    };
    Some(Numeric {
        is_int: rest.is_empty(),
        leading_zero: rest.is_empty() && int.len() > 1 && int[0] == b'0',
        int_digits: significant as u32,
        frac_digits: frac.len() as u32,
        sig_digits: sig_digits as u32,
        raw_frac_digits,
        int_lead0: int.len() > 1 && int[0] == b'0',
    })
}

/// Value of an integer-looking string with at most 38 significant digits
/// (every such value fits i128); None beyond that.
pub(crate) fn parse_i128(b: &[u8]) -> Option<i128> {
    let (neg, digits) = match b.strip_prefix(b"-") {
        Some(d) => (true, d),
        None => (false, b),
    };
    let digits = &digits[digits
        .iter()
        .position(|&c| c != b'0')
        .unwrap_or(digits.len())..];
    if digits.len() > 38 {
        return None;
    }
    let v = digits
        .iter()
        .fold(0i128, |acc, &c| acc * 10 + (c - b'0') as i128);
    Some(if neg { -v } else { v })
}

#[derive(Debug, PartialEq)]
pub(crate) enum Iso {
    Date,
    /// frac = fractional-second digits as written; sig = without trailing zeros.
    Time {
        frac: u32,
        sig: u32,
    },
    DateTime {
        frac: u32,
        sig: u32,
        midnight: bool,
    },
    DateTimeTz {
        frac: u32,
        sig: u32,
        midnight: bool,
        offset_minutes: i32,
    },
}

#[inline]
pub(crate) fn two(b: &[u8], i: usize) -> Option<u32> {
    let (x, y) = (*b.get(i)?, *b.get(i + 1)?);
    (x.is_ascii_digit() && y.is_ascii_digit()).then(|| ((x - b'0') * 10 + (y - b'0')) as u32)
}

pub(crate) fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400)) => 29,
        2 => 28,
        _ => 0,
    }
}

/// `YYYY-MM-DD` at the start of `b`; Some(10) when present and calendar-valid.
pub(crate) fn date(b: &[u8]) -> Option<usize> {
    let y = two(b, 0)? * 100 + two(b, 2)?;
    if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') {
        return None;
    }
    let (m, d) = (two(b, 5)?, two(b, 8)?);
    (1..=days_in_month(y, m)).contains(&d).then_some(10)
}

/// `HH:MM[:SS[.f{1,9}]]` from `i`: (end, fraction digits as written, without
/// trailing zeros, all-zero time).
pub(crate) fn time(b: &[u8], i: usize) -> Option<(usize, u32, u32, bool)> {
    let (h, m) = (two(b, i)?, two(b, i + 3)?);
    if b.get(i + 2) != Some(&b':') || h > 23 || m > 59 {
        return None;
    }
    let (mut end, mut frac, mut sig, mut zero) = (i + 5, 0, 0, h == 0 && m == 0);
    if b.get(end) == Some(&b':') {
        let s = two(b, end + 1)?;
        if s > 59 {
            return None;
        }
        zero &= s == 0;
        end += 3;
        if b.get(end) == Some(&b'.') {
            let digits = b[end + 1..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .count();
            if !(1..=9).contains(&digits) {
                return None;
            }
            let f = &b[end + 1..end + 1 + digits];
            zero &= f.iter().all(|&c| c == b'0');
            frac = digits as u32;
            sig = f.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1) as u32;
            end += 1 + digits;
        }
    }
    Some((end, frac, sig, zero))
}

/// `Z` or `±HH:MM` from `i` to the end of `b`, in minutes east of UTC.
pub(crate) fn offset(b: &[u8], i: usize) -> Option<i32> {
    match *b.get(i)? {
        b'Z' => (b.len() == i + 1).then_some(0),
        sign @ (b'+' | b'-') => {
            let (h, m) = (two(b, i + 1)?, two(b, i + 4)?);
            let ok = b.get(i + 3) == Some(&b':') && h <= 23 && m <= 59 && b.len() == i + 6;
            ok.then(|| (if sign == b'-' { -1 } else { 1 }) * (h * 60 + m) as i32)
        }
        _ => None,
    }
}

pub(crate) fn scan_iso(b: &[u8]) -> Option<Iso> {
    if let Some(d) = date(b) {
        if b.len() == d {
            return Some(Iso::Date);
        }
        if !matches!(b[d], b'T' | b' ') {
            return None;
        }
        let (end, frac, sig, midnight) = time(b, d + 1)?;
        if end == b.len() {
            return Some(Iso::DateTime {
                frac,
                sig,
                midnight,
            });
        }
        return offset(b, end).map(|offset_minutes| Iso::DateTimeTz {
            frac,
            sig,
            midnight,
            offset_minutes,
        });
    }
    let (end, frac, sig, _) = time(b, 0)?;
    (end == b.len()).then_some(Iso::Time { frac, sig })
}

/// Components of a value in the scan_iso grammar.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct IsoValue {
    /// Days since 1970-01-01; None for a bare time.
    pub days: Option<i64>,
    /// Nanoseconds since midnight.
    pub nanos: i64,
    /// Offset minutes east of UTC; None when the value carries no offset.
    pub offset_minutes: Option<i32>,
}

impl IsoValue {
    /// Nanoseconds since the Unix epoch, UTC (a bare time counts from 1970-01-01).
    pub(crate) fn epoch_ns(&self) -> i128 {
        self.days.unwrap_or(0) as i128 * 86_400_000_000_000 + self.nanos as i128
            - self.offset_minutes.unwrap_or(0) as i128 * 60_000_000_000
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's days_from_civil).
pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Exact components of an ISO value (validated by scan_iso first). scan_iso's
/// `midnight` flag isn't needed here — a midnight value simply has nanos == 0 —
/// and every field below has already been validated by scan_iso, so these `?`
/// reads (re-parsing the same bytes) cannot fail.
pub(crate) fn parse_iso(b: &[u8]) -> Option<IsoValue> {
    let kind = scan_iso(b)?;
    let num = |i: usize| two(b, i).map(|v| v as i64);
    let date_days = || Some(days_from_civil(num(0)? * 100 + num(2)?, num(5)?, num(8)?));
    let (days, t) = match kind {
        Iso::Date => {
            return Some(IsoValue {
                days: date_days(),
                nanos: 0,
                offset_minutes: None,
            })
        }
        Iso::Time { .. } => (None, 0),
        _ => (date_days(), 11),
    };
    let mut nanos = (num(t)? * 60 + num(t + 3)?) * 60 * 1_000_000_000;
    let end = t + 5;
    if b.get(end) == Some(&b':') {
        nanos += num(end + 1)? * 1_000_000_000;
        if b.get(end + 3) == Some(&b'.') {
            let (f, digits) = b[end + 4..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .fold((0i64, 0u32), |(a, n), &c| {
                    (a * 10 + (c - b'0') as i64, n + 1)
                });
            nanos += f * 10i64.pow(9 - digits);
        }
    }
    let offset_minutes = match kind {
        Iso::DateTimeTz { offset_minutes, .. } => Some(offset_minutes),
        _ => None,
    };
    Some(IsoValue {
        days,
        nanos,
        offset_minutes,
    })
}

/// Exact unscaled value of a numeric string (scan_numeric grammar) at `scale`:
/// None when it needs more decimal places or more than 38 digits.
pub(crate) fn parse_decimal(b: &[u8], scale: u32) -> Option<i128> {
    scan_numeric(b)?;
    let (neg, body) = match b.strip_prefix(b"-") {
        Some(r) => (true, r),
        None => (false, b),
    };
    let (int, frac) = match body.iter().position(|&c| c == b'.') {
        Some(p) => (&body[..p], &body[p + 1..]),
        None => (body, &b""[..]),
    };
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
    if frac.len() > scale as usize {
        return None;
    }
    let mut v: i128 = 0;
    for &c in int.iter().chain(frac) {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow(scale - frac.len() as u32)?)?;
    (v < 10i128.pow(38)).then_some(if neg { -v } else { v })
}

/// Group C accumulator over the non-null string values of one column.
#[derive(Default, Clone)]
pub(crate) struct StringStats {
    pub n_numeric: u64,
    pub n_numeric_int: u64,
    pub n_leading_zero: u64,
    pub int_min: Option<i128>,
    pub int_max: Option<i128>,
    /// Some integer-looking value has more than 38 significant digits → min/max null.
    pub int_overflow: bool,
    pub max_int_digits: Option<u32>,
    pub max_frac_digits: Option<u32>,
    pub min_frac_digits: Option<u32>,
    pub max_sig_digits: Option<u32>,
    pub n_iso_date: u64,
    pub n_iso_time: u64,
    pub n_iso_datetime: u64,
    pub n_iso_datetime_tz: u64,
    pub iso_max_frac_digits: Option<u32>,
    pub iso_max_sig_frac_digits: Option<u32>,
    pub offsets: HashSet<i32>,
    pub iso_n_midnight: u64,
    /// Collect the proof statistics below. Only streaming reads them (`recommend::prove` /
    /// `lossy_by_stats` on a statistics Level), so the one-shot path leaves this false and
    /// skips their per-value cost (float parse + render, ISO instant parse); they stay 0/None.
    pub proof: bool,
    /// Numeric values that are zero written with a minus sign ("-0", "-0.00").
    pub n_neg_zero: u64,
    /// Numeric values whose integer part has a leading zero ("007", "00.5").
    pub n_int_lead0: u64,
    /// Min / max fraction digits as written (trailing zeros kept; integers 0).
    pub raw_frac_min: Option<u32>,
    pub raw_frac_max: Option<u32>,
    /// Numeric values whose Float32 / Float64 parse is not canonically equal to the text
    /// (the one-shot `verify_text` float check, value by value).
    pub n_f32_roundtrip_fail: u64,
    pub n_f64_roundtrip_fail: u64,
    /// Numeric values whose Float32 / Float64 parse arrow-cast renders (ryu) as other text.
    pub n_f32_render_diff: u64,
    pub n_f64_render_diff: u64,
    /// UTC instants (ns) of ISO datetimes, both kinds.
    pub iso_instant_min: Option<i128>,
    pub iso_instant_max: Option<i128>,
    /// ISO times / datetimes whose time part arrow-cast renders differently: no seconds,
    /// or fraction digits other than chrono's 0 / 3 / 6 / 9 grouping of the significant ones.
    pub n_iso_time_noncanonical: u64,
    /// ISO datetimes written with a space instead of `T`.
    pub n_iso_space_sep: u64,
    /// ISO datetimes with a zero offset not written `Z` (arrow-cast renders UTC as `Z`).
    pub n_iso_offset_noncanonical: u64,
}

pub(crate) fn opt_min<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Fraction digits chrono prints for a time with `sig` significant ones: 0, 3, 6 or 9.
pub(crate) fn chrono_frac_digits(sig: u32) -> u32 {
    match sig {
        0 => 0,
        1..=3 => 3,
        4..=6 => 6,
        _ => 9,
    }
}

impl StringStats {
    pub fn add(&mut self, b: &[u8]) {
        if let Some(n) = scan_numeric(b) {
            self.n_numeric += 1;
            self.max_int_digits = self.max_int_digits.max(Some(n.int_digits));
            self.max_frac_digits = self.max_frac_digits.max(Some(n.frac_digits));
            self.min_frac_digits = opt_min(self.min_frac_digits, Some(n.frac_digits));
            self.max_sig_digits = self.max_sig_digits.max(Some(n.sig_digits));
            if self.proof {
                self.raw_frac_min = opt_min(self.raw_frac_min, Some(n.raw_frac_digits));
                self.raw_frac_max = self.raw_frac_max.max(Some(n.raw_frac_digits));
                self.n_int_lead0 += n.int_lead0 as u64;
                self.n_neg_zero += (n.sig_digits == 0 && b[0] == b'-') as u64;
                // The numeric grammar is ASCII, so the bytes are valid UTF-8.
                self.float_probe(std::str::from_utf8(b).unwrap_or_default());
            }
            if n.is_int {
                self.n_numeric_int += 1;
                self.n_leading_zero += n.leading_zero as u64;
                match parse_i128(b) {
                    Some(v) => {
                        self.int_min = opt_min(self.int_min, Some(v));
                        self.int_max = self.int_max.max(Some(v));
                    }
                    None => self.int_overflow = true,
                }
            }
            return; // a numeric string is never an ISO value
        }
        match scan_iso(b) {
            Some(Iso::Date) => self.n_iso_date += 1,
            Some(Iso::Time { frac, sig }) => {
                self.n_iso_time += 1;
                self.fraction(frac, sig);
                self.time_form(b, 0, frac, sig);
            }
            Some(Iso::DateTime {
                frac,
                sig,
                midnight,
            }) => {
                self.n_iso_datetime += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.datetime_form(b, frac, sig);
            }
            Some(Iso::DateTimeTz {
                frac,
                sig,
                midnight,
                offset_minutes,
            }) => {
                self.n_iso_datetime_tz += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
                self.datetime_form(b, frac, sig);
                if self.proof {
                    self.n_iso_offset_noncanonical +=
                        (offset_minutes == 0 && b.last() != Some(&b'Z')) as u64;
                }
            }
            None => {}
        }
    }

    /// The one-shot string → float verification and lossy check, value by value:
    /// parse, render as arrow-cast does (ryu), compare canonically and textually.
    fn float_probe(&mut self, s: &str) {
        let mut buf = ryu::Buffer::new();
        let r = buf.format(s.parse::<f64>().unwrap_or(f64::NAN));
        self.n_f64_roundtrip_fail += (canon(s) != canon(r)) as u64;
        self.n_f64_render_diff += (s != r) as u64;
        let r = buf.format(s.parse::<f32>().unwrap_or(f32::NAN));
        self.n_f32_roundtrip_fail += (canon(s) != canon(r)) as u64;
        self.n_f32_render_diff += (s != r) as u64;
    }

    /// `t0`: where the time part starts (0 for a bare time, 11 after a date).
    fn time_form(&mut self, b: &[u8], t0: usize, frac: u32, sig: u32) {
        if !self.proof {
            return;
        }
        let seconds = b.get(t0 + 5) == Some(&b':');
        self.n_iso_time_noncanonical += (!seconds || frac != chrono_frac_digits(sig)) as u64;
    }

    /// A datetime's separator, time part and UTC instant (both kinds).
    fn datetime_form(&mut self, b: &[u8], frac: u32, sig: u32) {
        if !self.proof {
            return;
        }
        self.n_iso_space_sep += (b[10] == b' ') as u64;
        self.time_form(b, 11, frac, sig);
        if let Some(v) = parse_iso(b) {
            let t = v.epoch_ns();
            self.iso_instant_min = opt_min(self.iso_instant_min, Some(t));
            self.iso_instant_max = self.iso_instant_max.max(Some(t));
        }
    }

    fn fraction(&mut self, frac: u32, sig: u32) {
        self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
        self.iso_max_sig_frac_digits = self.iso_max_sig_frac_digits.max(Some(sig));
    }

    pub fn merge(mut self, o: Self) -> Self {
        self.n_numeric += o.n_numeric;
        self.n_numeric_int += o.n_numeric_int;
        self.n_leading_zero += o.n_leading_zero;
        self.int_min = opt_min(self.int_min, o.int_min);
        self.int_max = self.int_max.max(o.int_max);
        self.int_overflow |= o.int_overflow;
        self.max_int_digits = self.max_int_digits.max(o.max_int_digits);
        self.max_frac_digits = self.max_frac_digits.max(o.max_frac_digits);
        self.min_frac_digits = opt_min(self.min_frac_digits, o.min_frac_digits);
        self.max_sig_digits = self.max_sig_digits.max(o.max_sig_digits);
        self.n_iso_date += o.n_iso_date;
        self.n_iso_time += o.n_iso_time;
        self.n_iso_datetime += o.n_iso_datetime;
        self.n_iso_datetime_tz += o.n_iso_datetime_tz;
        self.iso_max_frac_digits = self.iso_max_frac_digits.max(o.iso_max_frac_digits);
        self.iso_max_sig_frac_digits = self.iso_max_sig_frac_digits.max(o.iso_max_sig_frac_digits);
        self.offsets.extend(o.offsets);
        self.iso_n_midnight += o.iso_n_midnight;
        self.n_neg_zero += o.n_neg_zero;
        self.n_int_lead0 += o.n_int_lead0;
        self.raw_frac_min = opt_min(self.raw_frac_min, o.raw_frac_min);
        self.raw_frac_max = self.raw_frac_max.max(o.raw_frac_max);
        self.n_f32_roundtrip_fail += o.n_f32_roundtrip_fail;
        self.n_f64_roundtrip_fail += o.n_f64_roundtrip_fail;
        self.n_f32_render_diff += o.n_f32_render_diff;
        self.n_f64_render_diff += o.n_f64_render_diff;
        self.iso_instant_min = opt_min(self.iso_instant_min, o.iso_instant_min);
        self.iso_instant_max = self.iso_instant_max.max(o.iso_instant_max);
        self.n_iso_time_noncanonical += o.n_iso_time_noncanonical;
        self.n_iso_space_sep += o.n_iso_space_sep;
        self.n_iso_offset_noncanonical += o.n_iso_offset_noncanonical;
        self.proof |= o.proof;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::encode_series;
    use crate::techniques::describe::{byte_lengths, frequencies};
    use polars::prelude::*;

    fn num(s: &str) -> Option<(bool, bool, u32, u32)> {
        scan_numeric(s.as_bytes()).map(|n| (n.is_int, n.leading_zero, n.int_digits, n.frac_digits))
    }

    #[test]
    fn numeric_grammar() {
        for bad in [
            "5.", ".5", "1.2.3", "+5", "1e5", " 5", "5 ", "", "-", "\u{0663}",
        ] {
            assert_eq!(num(bad), None, "{bad:?}");
        }
        assert_eq!(num("007"), Some((true, true, 1, 0)));
        assert_eq!(num("-012"), Some((true, true, 2, 0)));
        assert_eq!(num("0"), Some((true, false, 0, 0)));
        assert_eq!(num("-0"), Some((true, false, 0, 0)));
        assert_eq!(num("007.50"), Some((false, false, 1, 1)));
        assert_eq!(num("0.25"), Some((false, false, 0, 2)));
    }

    #[test]
    fn i128_parse_limit() {
        assert_eq!(parse_i128(b"-012"), Some(-12));
        assert_eq!(
            parse_i128(format!("9{}", "0".repeat(37)).as_bytes()),
            Some(9 * 10i128.pow(37))
        );
        assert_eq!(parse_i128(format!("1{}", "0".repeat(38)).as_bytes()), None);
        assert_eq!(
            parse_i128(format!("{}1", "0".repeat(50)).as_bytes()),
            Some(1)
        );
    }

    fn iso(s: &str) -> Option<Iso> {
        scan_iso(s.as_bytes())
    }

    #[test]
    fn iso_grammar() {
        assert_eq!(iso("2024-02-29"), Some(Iso::Date));
        for bad in [
            "2023-02-29",
            "2024-13-01",
            "2024-1-05",
            "24:00",
            "23:59:60",
            "10:00:00.1234567890",
            "10:00:00.",
            "2024-01-05t10:00",
            "2024-01-05T10:00+0200",
            "2024-01-05T10:00Zx",
        ] {
            assert_eq!(iso(bad), None, "{bad:?}");
        }
        assert_eq!(iso("23:59"), Some(Iso::Time { frac: 0, sig: 0 }));
        assert_eq!(
            iso("10:00:00.123456789"),
            Some(Iso::Time { frac: 9, sig: 9 })
        );
        assert_eq!(
            iso("2024-01-05 10:00:00"),
            Some(Iso::DateTime {
                frac: 0,
                sig: 0,
                midnight: false
            })
        );
        assert_eq!(
            iso("2024-01-05T00:00:00.000"),
            Some(Iso::DateTime {
                frac: 3,
                sig: 0,
                midnight: true
            })
        );
        assert_eq!(
            iso("2024-01-05T00:00Z"),
            Some(Iso::DateTimeTz {
                frac: 0,
                sig: 0,
                midnight: true,
                offset_minutes: 0
            })
        );
        assert_eq!(
            iso("2024-01-05T10:00-00:00"),
            Some(Iso::DateTimeTz {
                frac: 0,
                sig: 0,
                midnight: false,
                offset_minutes: 0
            })
        );
        assert_eq!(
            iso("2024-01-05T10:00-05:30"),
            Some(Iso::DateTimeTz {
                frac: 0,
                sig: 0,
                midnight: false,
                offset_minutes: -330
            })
        );
    }

    #[test]
    fn stats_accumulate_and_merge() {
        let mut a = StringStats::default();
        for s in [
            "007",
            "12",
            "0.25",
            "2024-01-05T00:00Z",
            "2024-01-05T10:00+02:00",
        ] {
            a.add(s.as_bytes());
        }
        let mut b = StringStats::default();
        b.add(format!("1{}", "0".repeat(38)).as_bytes());
        let m = a.merge(b);
        assert_eq!((m.n_numeric, m.n_numeric_int, m.n_leading_zero), (4, 3, 1));
        assert!(m.int_overflow);
        assert_eq!((m.max_int_digits, m.max_frac_digits), (Some(39), Some(2)));
        assert_eq!((m.min_frac_digits, m.max_sig_digits), (Some(0), Some(39)));
        assert_eq!(
            (m.n_iso_datetime_tz, m.offsets.len(), m.iso_n_midnight),
            (2, 2, 1)
        );
        assert_eq!(m.iso_max_sig_frac_digits, Some(0));
    }

    #[test]
    fn adversarial_inputs_are_linear_and_correct() {
        let long = format!("{}.{}.", "0".repeat(1_000_000), "0".repeat(1_000_000));
        assert_eq!(num(&long), None);
        assert_eq!(iso(&format!("2024-01-05T{}", "0".repeat(1_000_000))), None);
    }

    #[test]
    fn significant_digits() {
        let sig = |s: &str| scan_numeric(s.as_bytes()).map(|n| (n.frac_digits, n.sig_digits));
        assert_eq!(sig("1.50"), Some((1, 2)));
        assert_eq!(sig("0.00120"), Some((4, 2)));
        assert_eq!(sig("1200"), Some((0, 4)));
        assert_eq!(sig("-0.0"), Some((0, 0)));
        assert_eq!(sig("12.50"), Some((1, 3)));
    }

    #[test]
    fn iso_significant_fraction() {
        assert_eq!(iso("10:00:00.120"), Some(Iso::Time { frac: 3, sig: 2 }));
        assert_eq!(
            iso("2024-01-05T00:00:00.000"),
            Some(Iso::DateTime {
                frac: 3,
                sig: 0,
                midnight: true
            })
        );
    }

    #[test]
    fn sum_len_unique_counts_each_distinct_value_once() {
        let s = Series::new("a".into(), &[Some("ab"), Some("ab"), Some("c"), None]);
        let lens = byte_lengths(&s).unwrap().unwrap();
        assert_eq!(lens.iter().sum::<u64>(), 5);
        assert_eq!(
            frequencies(&encode_series(&s).unwrap(), 0, Some(&lens), None).sum_len_unique,
            Some(3)
        );
    }

    #[test]
    fn exact_iso_components() {
        assert_eq!(
            parse_iso(b"1970-01-02"),
            Some(IsoValue {
                days: Some(1),
                nanos: 0,
                offset_minutes: None
            })
        );
        assert_eq!(parse_iso(b"2024-02-29").unwrap().days, Some(19_782));
        let t = parse_iso(b"10:00:00.12").unwrap();
        assert_eq!((t.days, t.nanos), (None, 36_000_120_000_000));
        let z = parse_iso(b"1970-01-01T05:30+05:30").unwrap();
        assert_eq!((z.offset_minutes, z.epoch_ns()), (Some(330), 0));
        assert_eq!(
            parse_iso(b"1969-12-31 23:59:59.999999999")
                .unwrap()
                .epoch_ns(),
            -1
        );
        assert_eq!(parse_iso(b"2023-02-29"), None);
    }

    #[test]
    fn exact_decimals() {
        assert_eq!(parse_decimal(b"007.50", 2), Some(750));
        assert_eq!(parse_decimal(b"-1.5", 3), Some(-1500));
        assert_eq!(parse_decimal(b"12", 0), Some(12));
        assert_eq!(parse_decimal(b"1.25", 1), None); // needs 2 places
        assert_eq!(
            parse_decimal(format!("1{}", "0".repeat(38)).as_bytes(), 0),
            None
        ); // 39 digits
    }

    #[test]
    fn scanner_streaming_statistics_numeric() {
        let mut st = StringStats {
            proof: true,
            ..Default::default()
        };
        for v in [
            "-0.00",
            "007.50",
            "1.5",
            "12",
            "0.30000000000000001",
            "16777217",
        ] {
            st.add(v.as_bytes());
        }
        assert_eq!(st.n_neg_zero, 1);
        assert_eq!(st.n_int_lead0, 1);
        assert_eq!((st.raw_frac_min, st.raw_frac_max), (Some(0), Some(17)));
        // f64: only "0.3…01" changes value (parses to 0.3); all but "1.5" render differently.
        assert_eq!((st.n_f64_roundtrip_fail, st.n_f64_render_diff), (1, 5));
        // f32: "0.3…01" and "16777217" (→ 16777216) change value.
        assert_eq!((st.n_f32_roundtrip_fail, st.n_f32_render_diff), (2, 5));
    }

    #[test]
    fn scanner_streaming_statistics_iso() {
        let mut st = StringStats {
            proof: true,
            ..Default::default()
        };
        for v in [
            "2024-01-01 10:00:00",
            "2024-01-01T10:00",
            "2024-01-01T10:00:00.5",
            "2024-01-01T10:00:00.500+00:00",
            "2024-01-01T10:00:00Z",
            "10:00:00.120000",
        ] {
            st.add(v.as_bytes());
        }
        assert_eq!(st.n_iso_space_sep, 1);
        // no seconds; ".5" (chrono prints .500); ".120000" (chrono prints .120)
        assert_eq!(st.n_iso_time_noncanonical, 3);
        assert_eq!(st.n_iso_offset_noncanonical, 1); // "+00:00" renders as "Z"
        assert_eq!(st.iso_instant_min, Some(1_704_103_200_000_000_000));
        assert_eq!(st.iso_instant_max, Some(1_704_103_200_500_000_000));
        let merged = StringStats::default().merge(st.clone());
        assert_eq!(merged.n_iso_time_noncanonical, 3);
    }
}
