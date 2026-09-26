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

use std::collections::HashSet;

pub(crate) struct Numeric {
    pub is_int: bool,
    /// Integer-looking with a leading zero ("007", "-012"; not "0" or "-0").
    pub leading_zero: bool,
    /// Significant integer-part digits (leading zeros ignored).
    pub int_digits: u32,
    /// Fraction digits with trailing zeros removed.
    pub frac_digits: u32,
}

pub(crate) fn scan_numeric(b: &[u8]) -> Option<Numeric> {
    let body = b.strip_prefix(b"-").unwrap_or(b);
    let int_len = body.iter().take_while(|c| c.is_ascii_digit()).count();
    if int_len == 0 {
        return None;
    }
    let (int, rest) = body.split_at(int_len);
    let frac = match rest {
        [] => None,
        [b'.', frac @ ..] if !frac.is_empty() && frac.iter().all(u8::is_ascii_digit) => Some(frac),
        _ => return None,
    };
    let significant = int.iter().position(|&c| c != b'0').map_or(0, |p| int.len() - p);
    Some(Numeric {
        is_int: frac.is_none(),
        leading_zero: frac.is_none() && int.len() > 1 && int[0] == b'0',
        int_digits: significant as u32,
        frac_digits: frac.map_or(0, |f| f.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)) as u32,
    })
}

/// Value of an integer-looking string with at most 38 significant digits
/// (every such value fits i128); None beyond that.
pub(crate) fn parse_i128(b: &[u8]) -> Option<i128> {
    let (neg, digits) = match b.strip_prefix(b"-") {
        Some(d) => (true, d),
        None => (false, b),
    };
    let digits = &digits[digits.iter().position(|&c| c != b'0').unwrap_or(digits.len())..];
    if digits.len() > 38 {
        return None;
    }
    let v = digits.iter().fold(0i128, |acc, &c| acc * 10 + (c - b'0') as i128);
    Some(if neg { -v } else { v })
}

#[derive(Debug, PartialEq)]
pub(crate) enum Iso {
    Date,
    Time { frac: u32 },
    DateTime { frac: u32, midnight: bool },
    DateTimeTz { frac: u32, midnight: bool, offset_minutes: i32 },
}

#[inline]
fn two(b: &[u8], i: usize) -> Option<u32> {
    let (x, y) = (*b.get(i)?, *b.get(i + 1)?);
    (x.is_ascii_digit() && y.is_ascii_digit()).then(|| ((x - b'0') * 10 + (y - b'0')) as u32)
}

fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    }
}

/// `YYYY-MM-DD` at the start of `b`; Some(10) when present and calendar-valid.
fn date(b: &[u8]) -> Option<usize> {
    let y = two(b, 0)? * 100 + two(b, 2)?;
    if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') {
        return None;
    }
    let (m, d) = (two(b, 5)?, two(b, 8)?);
    (1..=days_in_month(y, m)).contains(&d).then_some(10)
}

/// `HH:MM[:SS[.f{1,9}]]` from `i`: (end, fraction digits as written, all-zero time).
fn time(b: &[u8], i: usize) -> Option<(usize, u32, bool)> {
    let (h, m) = (two(b, i)?, two(b, i + 3)?);
    if b.get(i + 2) != Some(&b':') || h > 23 || m > 59 {
        return None;
    }
    let (mut end, mut frac, mut zero) = (i + 5, 0, h == 0 && m == 0);
    if b.get(end) == Some(&b':') {
        let s = two(b, end + 1)?;
        if s > 59 {
            return None;
        }
        zero &= s == 0;
        end += 3;
        if b.get(end) == Some(&b'.') {
            let digits = b[end + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            if !(1..=9).contains(&digits) {
                return None;
            }
            zero &= b[end + 1..end + 1 + digits].iter().all(|&c| c == b'0');
            frac = digits as u32;
            end += 1 + digits;
        }
    }
    Some((end, frac, zero))
}

/// `Z` or `±HH:MM` from `i` to the end of `b`, in minutes east of UTC.
fn offset(b: &[u8], i: usize) -> Option<i32> {
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
        let (end, frac, midnight) = time(b, d + 1)?;
        if end == b.len() {
            return Some(Iso::DateTime { frac, midnight });
        }
        return offset(b, end).map(|offset_minutes| Iso::DateTimeTz { frac, midnight, offset_minutes });
    }
    let (end, frac, _) = time(b, 0)?;
    (end == b.len()).then_some(Iso::Time { frac })
}

/// Group C accumulator over the non-null string values of one column.
#[derive(Default)]
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
    pub n_iso_date: u64,
    pub n_iso_time: u64,
    pub n_iso_datetime: u64,
    pub n_iso_datetime_tz: u64,
    pub iso_max_frac_digits: Option<u32>,
    pub offsets: HashSet<i32>,
    pub iso_n_midnight: u64,
}

fn opt_min<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl StringStats {
    pub fn add(&mut self, b: &[u8]) {
        if let Some(n) = scan_numeric(b) {
            self.n_numeric += 1;
            self.max_int_digits = self.max_int_digits.max(Some(n.int_digits));
            self.max_frac_digits = self.max_frac_digits.max(Some(n.frac_digits));
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
            Some(Iso::Time { frac }) => {
                self.n_iso_time += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
            }
            Some(Iso::DateTime { frac, midnight }) => {
                self.n_iso_datetime += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
                self.iso_n_midnight += midnight as u64;
            }
            Some(Iso::DateTimeTz { frac, midnight, offset_minutes }) => {
                self.n_iso_datetime_tz += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
            }
            None => {}
        }
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
        self.n_iso_date += o.n_iso_date;
        self.n_iso_time += o.n_iso_time;
        self.n_iso_datetime += o.n_iso_datetime;
        self.n_iso_datetime_tz += o.n_iso_datetime_tz;
        self.iso_max_frac_digits = self.iso_max_frac_digits.max(o.iso_max_frac_digits);
        self.offsets.extend(o.offsets);
        self.iso_n_midnight += o.iso_n_midnight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(s: &str) -> Option<(bool, bool, u32, u32)> {
        scan_numeric(s.as_bytes()).map(|n| (n.is_int, n.leading_zero, n.int_digits, n.frac_digits))
    }

    #[test]
    fn numeric_grammar() {
        for bad in ["5.", ".5", "1.2.3", "+5", "1e5", " 5", "5 ", "", "-", "\u{0663}"] {
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
        assert_eq!(parse_i128(format!("9{}", "0".repeat(37)).as_bytes()), Some(9 * 10i128.pow(37)));
        assert_eq!(parse_i128(format!("1{}", "0".repeat(38)).as_bytes()), None);
        assert_eq!(parse_i128(format!("{}1", "0".repeat(50)).as_bytes()), Some(1));
    }

    fn iso(s: &str) -> Option<Iso> {
        scan_iso(s.as_bytes())
    }

    #[test]
    fn iso_grammar() {
        assert_eq!(iso("2024-02-29"), Some(Iso::Date));
        for bad in ["2023-02-29", "2024-13-01", "2024-1-05", "24:00", "23:59:60", "10:00:00.1234567890", "10:00:00.", "2024-01-05t10:00", "2024-01-05T10:00+0200", "2024-01-05T10:00Zx"] {
            assert_eq!(iso(bad), None, "{bad:?}");
        }
        assert_eq!(iso("23:59"), Some(Iso::Time { frac: 0 }));
        assert_eq!(iso("10:00:00.123456789"), Some(Iso::Time { frac: 9 }));
        assert_eq!(iso("2024-01-05 10:00:00"), Some(Iso::DateTime { frac: 0, midnight: false }));
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, midnight: true }));
        assert_eq!(iso("2024-01-05T00:00Z"), Some(Iso::DateTimeTz { frac: 0, midnight: true, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-00:00"), Some(Iso::DateTimeTz { frac: 0, midnight: false, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-05:30"), Some(Iso::DateTimeTz { frac: 0, midnight: false, offset_minutes: -330 }));
    }

    #[test]
    fn stats_accumulate_and_merge() {
        let mut a = StringStats::default();
        for s in ["007", "12", "0.25", "2024-01-05T00:00Z", "2024-01-05T10:00+02:00"] {
            a.add(s.as_bytes());
        }
        let mut b = StringStats::default();
        b.add(format!("1{}", "0".repeat(38)).as_bytes());
        let m = a.merge(b);
        assert_eq!((m.n_numeric, m.n_numeric_int, m.n_leading_zero), (4, 3, 1));
        assert!(m.int_overflow);
        assert_eq!((m.max_int_digits, m.max_frac_digits), (Some(39), Some(2)));
        assert_eq!((m.n_iso_datetime_tz, m.offsets.len(), m.iso_n_midnight), (2, 2, 1));
    }

    #[test]
    fn adversarial_inputs_are_linear_and_correct() {
        let long = format!("{}.{}.", "0".repeat(1_000_000), "0".repeat(1_000_000));
        assert_eq!(num(&long), None);
        assert_eq!(iso(&format!("2024-01-05T{}", "0".repeat(1_000_000))), None);
    }
}
