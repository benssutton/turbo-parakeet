// ─────────────────────────────────────────────────────────────────────────────
// describe — per-column profile (plugin entry points `describe_columns`, `column_sizes`)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md. One pass
// per column, columns in parallel (rayon); inside a column the frequency, float
// and string work runs over 64K-row chunks in parallel. Every value metric is
// computed on the column and again on its inner values (List/Array flattened one
// level, null lists skipped). Field names and order match
// analytics/describe/base.py (DescribeRust maps them by name).

use crate::shared::{encode_series, EncodedColumn};
use foldhash::fast::FixedState;
use polars::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// describe — value frequencies (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// One foldhash map `key → (count, first row, split-subset mask)` per 64K-row
// chunk, built in parallel and merged (counts summed, first = min, masks OR-ed).
// One O(distinct) sweep of the merged map yields n_unique, entropy (null as its
// own category), f1/f2, the capture history and the top 5 (count desc, then
// first occurrence asc). Keys come from `encode_series` (floats canonicalised;
// strings, nested and struct values hashed — collisions ~6e-11 per pair at 50K
// rows, accepted as documented in CLAUDE.md).

pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Entry {
    count: u64,
    first: u64,
    mask: u8,
}

type Map = HashMap<u64, Entry, FixedState>;

pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub entropy: f64,
    pub f1: u64,
    pub f2: u64,
    pub top5_idx: Vec<u64>,
    pub top5_count: Vec<u64>,
    pub capture_history: [u64; 7],
    /// Total byte length of the distinct values (`lengths` given: string / binary columns).
    pub sum_len_unique: Option<u64>,
}

/// Split subset (0, 1 or 2) of `row`: the SplitMix64 finaliser of `seed + row`.
/// Seeded and language-independent; the Python implementations use numpy's
/// generator instead, which is why capture histories are compared only through
/// the Schnabel estimate.
#[inline]
pub(crate) fn subset(seed: u64, row: u64) -> u8 {
    let mut z = seed.wrapping_add(row).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % 3) as u8
}

fn count_chunk(values: &[u64], is_null: &[bool], start: usize, seed: u64) -> Map {
    let mut map = Map::with_capacity_and_hasher(1024, FixedState::default());
    for (j, (&key, &null)) in values.iter().zip(is_null).enumerate() {
        if null {
            continue;
        }
        let row = (start + j) as u64;
        let e = map.entry(key).or_insert(Entry { count: 0, first: row, mask: 0 });
        e.count += 1;
        e.mask |= 1 << subset(seed, row);
    }
    map
}

fn merge(mut a: Map, mut b: Map) -> Map {
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    for (key, e) in b {
        a.entry(key)
            .and_modify(|x| {
                x.count += e.count;
                x.first = x.first.min(e.first);
                x.mask |= e.mask;
            })
            .or_insert(e);
    }
    a
}

pub(crate) fn frequencies(col: &EncodedColumn, seed: u64, lengths: Option<&[u64]>) -> Frequencies {
    let n = col.len();
    let map = col
        .values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge);

    let n_null = col.is_null.iter().filter(|&&x| x).count();
    let nf = n as f64;
    let mut entropy = if n == 0 { f64::NAN } else { 0.0 };
    let (mut f1, mut f2, mut history) = (0u64, 0u64, [0u64; 7]);
    let n_unique = map.len() as u64;
    let mut entries: Vec<Entry> = Vec::with_capacity(map.len());
    let mut unique_len = 0u64;
    for e in map.into_values() {
        let p = e.count as f64 / nf;
        entropy -= p * p.log2();
        f1 += (e.count == 1) as u64;
        f2 += (e.count == 2) as u64;
        history[e.mask as usize - 1] += 1;
        if let Some(l) = lengths {
            unique_len += l[e.first as usize];
        }
        entries.push(e);
    }
    if n_null > 0 {
        let p = n_null as f64 / nf;
        entropy -= p * p.log2();
    }
    let order = |a: &Entry, b: &Entry| b.count.cmp(&a.count).then(a.first.cmp(&b.first));
    if entries.len() > 5 {
        entries.select_nth_unstable_by(4, order);
        entries.truncate(5);
    }
    entries.sort_unstable_by(order);
    Frequencies {
        n_unique,
        entropy: entropy + 0.0, // -0.0 → 0.0 for an all-null column
        f1,
        f2,
        top5_idx: entries.iter().map(|e| e.first).collect(),
        top5_count: entries.iter().map(|e| e.count).collect(),
        capture_history: history,
        sum_len_unique: lengths.map(|_| unique_len),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// describe — first-occurrence extremes and value lengths (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// Ordering matches Polars `sort()`: integers, Decimal and temporals by physical
// value; floats numerically with NaN excluded (-0.0 ties 0.0); strings and
// binary by bytes; Categorical by string value; Enum by category order (its
// physical code); List, Array and Struct by Polars' row encoding — the encoding
// its sort uses. Ties keep the lowest row index.

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

/// Byte length of every value (0 for nulls) of a String, Categorical, Enum or Binary series.
fn byte_lengths(s: &Series) -> PolarsResult<Option<Vec<u64>>> {
    Ok(match s.dtype() {
        DataType::String => Some(s.str()?.iter().map(|v| v.map_or(0, |x| x.len() as u64)).collect()),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => return byte_lengths(&s.cast(&DataType::String)?),
        DataType::Binary => Some(s.binary()?.iter().map(|v| v.map_or(0, |x| x.len() as u64)).collect()),
        _ => None,
    })
}

pub(crate) fn range(s: &Series) -> PolarsResult<Range> {
    let (argmin, argmax) = arg_extremes(s)?;
    let (min_len, max_len) = lengths(s)?;
    Ok(Range { argmin, argmax, min_len, max_len })
}

// ─────────────────────────────────────────────────────────────────────────────
// describe — float statistics (group B)
// ─────────────────────────────────────────────────────────────────────────────
//
// Decimal places come from the shortest round-trip representation (ryu) in the
// column's own width — f32 digits for Float32 — exponent-aware:
// max(0, fraction digits without trailing zeros − exponent).

#[derive(Default, Clone, Copy)]
pub(crate) struct FloatStats {
    pub n_nan: u64,
    pub n_inf: u64,
    pub n_fractional: u64,
    pub max_frac_digits: Option<u32>,
    /// Finite values that change under f64 → f32 → f64 (meaningless for Float32).
    pub n_f32_inexact: u64,
}

pub(crate) fn frac_digits(repr: &str) -> u32 {
    let repr = repr.trim_start_matches('-');
    let (mantissa, exp) = match repr.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (repr, 0),
    };
    let frac = mantissa.split_once('.').map_or("", |(_, f)| f).trim_end_matches('0');
    (frac.len() as i64 - exp).max(0) as u32
}

impl FloatStats {
    fn merge(self, o: Self) -> Self {
        Self {
            n_nan: self.n_nan + o.n_nan,
            n_inf: self.n_inf + o.n_inf,
            n_fractional: self.n_fractional + o.n_fractional,
            max_frac_digits: self.max_frac_digits.max(o.max_frac_digits),
            n_f32_inexact: self.n_f32_inexact + o.n_f32_inexact,
        }
    }

    fn add_f64(&mut self, x: f64, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self.max_frac_digits.max(Some(frac_digits(buf.format_finite(x))));
            self.n_f32_inexact += ((x as f32) as f64 != x) as u64;
        }
    }

    fn add_f32(&mut self, x: f32, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self.max_frac_digits.max(Some(frac_digits(buf.format_finite(x))));
        }
    }
}

/// Fold `add` over the valid values of one Arrow chunk, CHUNK values per task.
fn fold<T: Copy + Sync>(values: &[T], validity: Option<&Bitmap>, add: impl Fn(&mut FloatStats, T, &mut ryu::Buffer) + Sync) -> FloatStats {
    let validity = validity.filter(|bm| bm.unset_bits() > 0);
    values
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            let (mut st, mut buf) = (FloatStats::default(), ryu::Buffer::new());
            for (j, &x) in chunk.iter().enumerate() {
                if validity.is_some_and(|bm| !bm.get_bit(i * CHUNK + j)) {
                    continue;
                }
                add(&mut st, x, &mut buf);
            }
            st
        })
        .reduce(FloatStats::default, FloatStats::merge)
}

pub(crate) fn float_stats(s: &Series) -> PolarsResult<Option<FloatStats>> {
    Ok(Some(match s.dtype() {
        DataType::Float64 => s
            .f64()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f64))
            .fold(FloatStats::default(), FloatStats::merge),
        DataType::Float32 => s
            .f32()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f32))
            .fold(FloatStats::default(), FloatStats::merge),
        _ => return Ok(None),
    }))
}

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
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
    let significant = int.iter().position(|&c| c != b'0').map_or(0, |p| int.len() - p);
    let sig_digits = if significant > 0 {
        significant + frac.len()
    } else {
        frac.iter().position(|&c| c != b'0').map_or(0, |p| frac.len() - p)
    };
    Some(Numeric {
        is_int: rest.is_empty(),
        leading_zero: rest.is_empty() && int.len() > 1 && int[0] == b'0',
        int_digits: significant as u32,
        frac_digits: frac.len() as u32,
        sig_digits: sig_digits as u32,
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
    /// frac = fractional-second digits as written; sig = without trailing zeros.
    Time { frac: u32, sig: u32 },
    DateTime { frac: u32, sig: u32, midnight: bool },
    DateTimeTz { frac: u32, sig: u32, midnight: bool, offset_minutes: i32 },
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

/// `HH:MM[:SS[.f{1,9}]]` from `i`: (end, fraction digits as written, without
/// trailing zeros, all-zero time).
fn time(b: &[u8], i: usize) -> Option<(usize, u32, u32, bool)> {
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
            let digits = b[end + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
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
        let (end, frac, sig, midnight) = time(b, d + 1)?;
        if end == b.len() {
            return Some(Iso::DateTime { frac, sig, midnight });
        }
        return offset(b, end).map(|offset_minutes| Iso::DateTimeTz { frac, sig, midnight, offset_minutes });
    }
    let (end, frac, sig, _) = time(b, 0)?;
    (end == b.len()).then_some(Iso::Time { frac, sig })
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
            self.min_frac_digits = opt_min(self.min_frac_digits, Some(n.frac_digits));
            self.max_sig_digits = self.max_sig_digits.max(Some(n.sig_digits));
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
            }
            Some(Iso::DateTime { frac, sig, midnight }) => {
                self.n_iso_datetime += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
            }
            Some(Iso::DateTimeTz { frac, sig, midnight, offset_minutes }) => {
                self.n_iso_datetime_tz += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
            }
            None => {}
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
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// describe — column profile assembly and the `describe_columns` plugin entry
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) type Row = Vec<AnyValue<'static>>;

/// Metrics computed on any value series, in output order (base.py VALUE_METRICS).
pub(crate) fn value_fields() -> Vec<(&'static str, DataType)> {
    use DataType::{Float64 as F64, UInt32 as U32, UInt64 as U64};
    let list = DataType::List(Box::new(U64));
    // Arrow has no plain 128-bit integer; parse_i128 caps values at 38 digits.
    let d38 = DataType::Decimal(Some(38), Some(0));
    vec![
        ("n_unique", U64), ("entropy", F64), ("f1", U64), ("f2", U64), ("argmin", U64), ("argmax", U64),
        ("min_len", U64), ("max_len", U64), ("gcd", d38.clone()), ("sum_len", U64), ("sum_len_unique", U64),
        ("top5_idx", list.clone()), ("top5_count", list.clone()), ("capture_history", list),
        ("n_nan", U64), ("n_inf", U64), ("n_fractional", U64), ("max_frac_digits", U32), ("n_f32_inexact", U64),
        ("n_numeric", U64), ("n_numeric_int", U64), ("n_leading_zero", U64), ("numeric_int_min", d38.clone()), ("numeric_int_max", d38),
        ("numeric_max_int_digits", U32), ("numeric_max_frac_digits", U32), ("numeric_min_frac_digits", U32), ("numeric_max_sig_digits", U32),
        ("n_iso_date", U64), ("n_iso_time", U64), ("n_iso_datetime", U64), ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32), ("iso_max_sig_frac_digits", U32), ("iso_n_offsets", U64), ("iso_n_midnight", U64),
    ]
}

pub(crate) fn fields() -> Vec<(String, DataType)> {
    let mut f = vec![("column".to_string(), DataType::String), ("n_rows".into(), DataType::UInt64), ("n_null".into(), DataType::UInt64)];
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), DataType::UInt64));
    f.push(("inner_n_values".into(), DataType::UInt64));
    f.push(("inner_n_null".into(), DataType::UInt64));
    f.extend(value_fields().into_iter().map(|(n, d)| (format!("inner_{n}"), d)));
    f
}

fn describe_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = fields().into_iter().map(|(n, d)| Field::new(n.into(), d)).collect();
    Ok(Field::new("describe".into(), DataType::Struct(fields)))
}

fn u64v(v: Option<u64>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt64) }
fn u32v(v: Option<u32>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt32) }
fn d38v(v: Option<i128>) -> AnyValue<'static> { v.map_or(AnyValue::Null, |v| AnyValue::Decimal(v, 0)) }
fn listv(v: &[u64]) -> AnyValue<'static> { AnyValue::List(Series::new(PlSmallStr::EMPTY, v)) }
fn nulls(n: usize) -> Row { vec![AnyValue::Null; n] }

fn strings(s: &Series) -> PolarsResult<Option<StringStats>> {
    let st = match s.dtype() {
        DataType::String => s.clone(),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => s.cast(&DataType::String)?,
        _ => return Ok(None),
    };
    Ok(Some(
        st.str()?
            .downcast_iter()
            .map(|arr| {
                (0..arr.len())
                    .into_par_iter()
                    .with_min_len(CHUNK)
                    .fold(StringStats::default, |mut acc, i| {
                        if arr.is_valid(i) {
                            acc.add(arr.value(i).as_bytes());
                        }
                        acc
                    })
                    .reduce(StringStats::default, StringStats::merge)
            })
            .fold(StringStats::default(), StringStats::merge),
    ))
}

/// Every value metric of one series (the column, or its list's inner values).
pub(crate) struct Profile {
    pub freq: Frequencies,
    pub range: Range,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    /// GCD of the physical values (gcd.rs); None for non-integer dtypes or > 38 digits.
    pub gcd: Option<i128>,
    /// Total byte length of the non-null values (string / binary columns).
    pub sum_len: Option<u64>,
    /// Float32 series: `n_f32_inexact` does not apply.
    pub is_f32: bool,
}

pub(crate) fn profile(s: &Series, seed: u64) -> PolarsResult<Profile> {
    let lengths = byte_lengths(s)?;
    Ok(Profile {
        freq: frequencies(&encode_series(s)?, seed, lengths.as_deref()),
        range: range(s)?,
        floats: float_stats(s)?,
        strings: strings(s)?,
        gcd: crate::gcd::series_gcd(s)?,
        sum_len: lengths.map(|l| l.iter().sum()),
        is_f32: s.dtype() == &DataType::Float32,
    })
}

impl Profile {
    /// The metrics in `value_fields()` order.
    fn row(&self) -> Row {
        let (f, r) = (&self.freq, &self.range);
        let mut row: Row = vec![
            AnyValue::UInt64(f.n_unique), AnyValue::Float64(f.entropy), AnyValue::UInt64(f.f1), AnyValue::UInt64(f.f2),
            u64v(r.argmin), u64v(r.argmax), u64v(r.min_len), u64v(r.max_len),
            d38v(self.gcd), u64v(self.sum_len), u64v(f.sum_len_unique),
            listv(&f.top5_idx), listv(&f.top5_count), listv(&f.capture_history),
        ];
        match self.floats {
            Some(fl) => row.extend([
                AnyValue::UInt64(fl.n_nan), AnyValue::UInt64(fl.n_inf), AnyValue::UInt64(fl.n_fractional), u32v(fl.max_frac_digits),
                if self.is_f32 { AnyValue::Null } else { AnyValue::UInt64(fl.n_f32_inexact) },
            ]),
            None => row.extend(nulls(5)),
        }
        match &self.strings {
            Some(st) => {
                let (lo, hi) = if st.int_overflow { (None, None) } else { (st.int_min, st.int_max) };
                row.extend([
                    AnyValue::UInt64(st.n_numeric), AnyValue::UInt64(st.n_numeric_int), AnyValue::UInt64(st.n_leading_zero),
                    d38v(lo), d38v(hi), u32v(st.max_int_digits), u32v(st.max_frac_digits),
                    u32v(st.min_frac_digits), u32v(st.max_sig_digits),
                    AnyValue::UInt64(st.n_iso_date), AnyValue::UInt64(st.n_iso_time), AnyValue::UInt64(st.n_iso_datetime),
                    AnyValue::UInt64(st.n_iso_datetime_tz), u32v(st.iso_max_frac_digits), u32v(st.iso_max_sig_frac_digits),
                    AnyValue::UInt64(st.offsets.len() as u64), AnyValue::UInt64(st.iso_n_midnight),
                ])
            }
            None => row.extend(nulls(17)),
        }
        row
    }
}

/// A list column's values one level down (`flatten`) and their profile.
pub(crate) struct Inner {
    pub values: Series,
    pub profile: Profile,
}

/// Everything Describe measures on one column (sizes excepted — sizes.rs).
pub(crate) struct Described {
    pub name: PlSmallStr,
    pub n_rows: u64,
    pub n_null: u64,
    pub outer: Profile,
    pub n_midnight: Option<u64>,
    pub inner: Option<Inner>,
}

impl Described {
    /// One output row in `fields()` order.
    pub(crate) fn row(&self) -> Row {
        let mut row: Row = vec![AnyValue::StringOwned(self.name.clone()), AnyValue::UInt64(self.n_rows), AnyValue::UInt64(self.n_null)];
        row.extend(self.outer.row());
        row.push(u64v(self.n_midnight));
        match &self.inner {
            Some(i) => {
                row.push(AnyValue::UInt64(i.values.len() as u64));
                row.push(AnyValue::UInt64(i.values.null_count() as u64));
                row.extend(i.profile.row());
            }
            None => row.extend(nulls(2 + value_fields().len())),
        }
        row
    }
}

/// Datetime values at exactly 00:00:00 local time (column time zone, else naive).
fn n_midnight(s: &Series) -> PolarsResult<Option<u64>> {
    let DataType::Datetime(unit, tz) = s.dtype() else { return Ok(None) };
    let per_day: i64 = match unit {
        TimeUnit::Nanoseconds => 86_400_000_000_000,
        TimeUnit::Microseconds => 86_400_000_000,
        TimeUnit::Milliseconds => 86_400_000,
    };
    let phys = s.to_physical_repr();
    let values = phys.i64()?;
    let count = match tz {
        None => values.into_iter().flatten().filter(|v| v.rem_euclid(per_day) == 0).count(),
        Some(tz) => {
            let zone: chrono_tz::Tz = tz.as_str().parse().map_err(|_| polars_err!(ComputeError: "describe: unknown time zone {tz}"))?;
            let per_sec = per_day / 86_400;
            values
                .into_iter()
                .flatten()
                .filter(|&v| {
                    v.rem_euclid(per_sec) == 0
                        && chrono::DateTime::from_timestamp(v.div_euclid(per_sec), 0)
                            .is_some_and(|t| t.with_timezone(&zone).time() == chrono::NaiveTime::MIN)
                })
                .count()
        }
    };
    Ok(Some(count as u64))
}

/// Values one level down, skipping null lists — the same definition as the
/// Python `flatten` (drop_nulls, then explode the non-empty lists). Element i is
/// what `inner_argmin` / `inner_top5_idx` index into.
fn flatten(s: &Series) -> PolarsResult<Option<Series>> {
    let (inner, ranges): (Series, Vec<Option<(usize, usize)>>) = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            (ca.get_inner(), list_ranges(&ca))
        }
        DataType::Array(_, width) => {
            let ca = s.array()?.rechunk();
            let valid = ca.is_not_null();
            let ranges = valid.iter().enumerate().map(|(i, ok)| (ok == Some(true)).then_some((i * width, *width))).collect();
            (ca.get_inner(), ranges)
        }
        _ => return Ok(None),
    };
    let idx: Vec<IdxSize> = ranges.into_iter().flatten().flat_map(|(start, len)| (start..start + len).map(|i| i as IdxSize)).collect();
    Ok(Some(inner.take_slice(&idx)?))
}

pub(crate) fn describe_one(s: &Series, seed: u64) -> PolarsResult<Described> {
    let inner = match flatten(s)? {
        Some(values) => {
            let profile = profile(&values, seed)?;
            Some(Inner { values, profile })
        }
        None => None,
    };
    Ok(Described {
        name: s.name().clone(),
        n_rows: s.len() as u64,
        n_null: s.null_count() as u64,
        outer: profile(s, seed)?,
        n_midnight: n_midnight(s)?,
        inner,
    })
}

/// A Struct series `name` with one field per `fields` entry and one row per `rows` entry.
pub(crate) fn assemble(name: &str, fields: &[(String, DataType)], rows: &[Row]) -> PolarsResult<Series> {
    let columns = fields
        .iter()
        .enumerate()
        .map(|(j, (field, dtype))| {
            let values: Vec<AnyValue> = rows.iter().map(|r| r[j].clone()).collect();
            Series::from_any_values_and_dtype(field.as_str().into(), &values, dtype, true)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructChunked::from_series(name.into(), rows.len(), columns.iter())?.into_series())
}

pub(crate) fn describe_columns_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs.par_iter().map(|s| describe_one(s, seed).map(|d| d.row())).collect::<PolarsResult<_>>()?;
    assemble("describe", &fields(), &rows)
}

#[derive(Deserialize)]
struct DescribeKwargs {
    seed: u64,
}

#[polars_expr(output_type_func=describe_output_type)]
fn describe_columns(inputs: &[Series], kwargs: DescribeKwargs) -> PolarsResult<Series> {
    describe_columns_impl(inputs, kwargs.seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freq(s: Series) -> Frequencies {
        frequencies(&encode_series(&s).unwrap(), 0, None)
    }

    #[test]
    fn counts_entropy_and_top5() {
        let f = freq(Series::new("a".into(), &[Some("a"), Some("a"), Some("b"), None]));
        assert_eq!((f.n_unique, f.f1, f.f2), (2, 1, 1));
        assert!((f.entropy - 1.5).abs() < 1e-12);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 2], vec![2, 1]));
    }

    #[test]
    fn ties_break_by_first_occurrence() {
        let f = freq(Series::new("a".into(), &[3i64, 1, 2, 1, 2, 3, 4, 5, 6]));
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 1, 2, 6, 7], vec![2, 2, 2, 1, 1]));
    }

    #[test]
    fn merges_across_parallel_chunks() {
        let n = 3 * CHUNK + 5;
        // value 9 first appears in the third chunk and again in the fourth; every other value is 0.
        let mut v = vec![0i64; n];
        v[2 * CHUNK + 1] = 9;
        v[3 * CHUNK + 2] = 9;
        let f = freq(Series::new("a".into(), v));
        assert_eq!(f.n_unique, 2);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, (2 * CHUNK + 1) as u64], vec![(n - 2) as u64, 2]));
    }

    #[test]
    fn capture_history_sums_to_n_unique_and_fills_all_subsets() {
        let f = freq(Series::new("a".into(), (0..10_000i64).map(|i| i % 10).collect::<Vec<_>>()));
        assert_eq!(f.capture_history, [0, 0, 0, 0, 0, 0, 10]);
        let g = freq(Series::new("a".into(), (0..20_000i64).map(|i| (i * 7919) % 5_003).collect::<Vec<_>>()));
        assert_eq!(g.capture_history.iter().sum::<u64>(), g.n_unique);
    }

    #[test]
    fn subsets_are_roughly_uniform() {
        let mut counts = [0u64; 3];
        for row in 0..30_000 {
            counts[subset(0, row) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9_500..10_500).contains(&c)), "{counts:?}");
    }

    #[test]
    fn zero_rows_and_all_null() {
        let z = freq(Series::new_empty("a".into(), &DataType::Int32));
        assert!(z.entropy.is_nan());
        assert_eq!((z.n_unique, z.capture_history), (0, [0; 7]));
        let a = freq(Series::new("a".into(), &[None::<i32>, None]));
        assert_eq!((a.n_unique, a.entropy), (0, 0.0));
    }

    #[test]
    fn struct_values_hash_whole_and_binary_encodes() {
        let a = Series::new("a".into(), &[1i32, 1, 1]);
        let b = Series::new("b".into(), &[Some("x"), Some("x"), None]);
        let s = StructChunked::from_series("s".into(), 3, [a, b].iter()).unwrap().into_series();
        assert_eq!(freq(s).n_unique, 2);
        let bin = Series::new("b".into(), &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), None]);
        assert_eq!(freq(bin).n_unique, 1);
    }

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

    #[test]
    fn decimal_places_of_shortest_reprs() {
        for (repr, want) in [("0.1", 1), ("1e-7", 7), ("1.5e20", 0), ("1.5e+20", 0), ("3.0", 0), ("-0.0", 0), ("123.45", 2), ("1.25e-3", 5)] {
            assert_eq!(frac_digits(repr), want, "{repr}");
        }
    }

    #[test]
    fn f64_stats() {
        let s = Series::new("x".into(), &[Some(0.1), Some(1e-7), Some(1.5e20), Some(3.0), Some(f64::INFINITY), Some(f64::NAN), None]);
        let st = float_stats(&s).unwrap().unwrap();
        assert_eq!((st.n_nan, st.n_inf, st.n_fractional, st.max_frac_digits), (1, 1, 2, Some(7)));
        assert_eq!(st.n_f32_inexact, 3); // 0.1, 1e-7, 1.5e20
    }

    #[test]
    fn f32_uses_its_own_repr_and_non_floats_are_none() {
        let st = float_stats(&Series::new("x".into(), &[0.1f32, 0.25])).unwrap().unwrap();
        assert_eq!(st.max_frac_digits, Some(2));
        assert!(float_stats(&Series::new("x".into(), &[1i32])).unwrap().is_none());
        assert_eq!(float_stats(&Series::new("x".into(), &[f64::NAN])).unwrap().unwrap().max_frac_digits, None);
    }

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
        assert_eq!(iso("23:59"), Some(Iso::Time { frac: 0, sig: 0 }));
        assert_eq!(iso("10:00:00.123456789"), Some(Iso::Time { frac: 9, sig: 9 }));
        assert_eq!(iso("2024-01-05 10:00:00"), Some(Iso::DateTime { frac: 0, sig: 0, midnight: false }));
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, sig: 0, midnight: true }));
        assert_eq!(iso("2024-01-05T00:00Z"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: true, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-00:00"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: false, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-05:30"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: false, offset_minutes: -330 }));
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
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, sig: 0, midnight: true }));
    }

    #[test]
    fn sum_len_unique_counts_each_distinct_value_once() {
        let s = Series::new("a".into(), &[Some("ab"), Some("ab"), Some("c"), None]);
        let lens = byte_lengths(&s).unwrap().unwrap();
        assert_eq!(lens.iter().sum::<u64>(), 5);
        assert_eq!(frequencies(&encode_series(&s).unwrap(), 0, Some(&lens)).sum_len_unique, Some(3));
    }

    #[test]
    fn profile_gcd_and_lengths() {
        let p = profile(&Series::new("x".into(), &[Some(10i64), None, Some(30)]), 0).unwrap();
        assert_eq!((p.gcd, p.sum_len), (Some(10), None));
        let s = profile(&Series::new("x".into(), &["ab", "ab", "c"]), 0).unwrap();
        assert_eq!((s.gcd, s.sum_len, s.freq.sum_len_unique), (None, Some(5), Some(3)));
    }

    #[test]
    fn described_row_matches_fields() {
        let list = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]);
        let d = describe_one(&list, 0).unwrap();
        assert_eq!(d.row().len(), fields().len());
        assert_eq!(d.inner.as_ref().unwrap().values.len(), 2);
        let floats = describe_one(&Series::new("y".into(), &[1.5f64]), 0).unwrap();
        assert_eq!(floats.row().len(), fields().len());
        assert!(floats.inner.is_none() && floats.outer.floats.is_some());
    }
}
