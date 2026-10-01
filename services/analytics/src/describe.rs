// ─────────────────────────────────────────────────────────────────────────────
// describe — per-column profile (entry points `describe_columns`, `column_sizes`)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md. One pass
// per column, columns in parallel (rayon); inside a column the frequency, float
// and string work runs over 64K-row chunks in parallel. Every value metric is
// computed on the column and again on its inner values (List/Array flattened one
// level, null lists skipped), with the conclusions (conclusions.rs) and rendered
// min / max. Field names match analytics/describe/base.py (DescribeRust maps them
// by name).

use crate::conclusions::{conclude, Conclusions};
use crate::recommend::canon;
use crate::shared::{encode_series, EncodedColumn};
use foldhash::fast::FixedState;
use polars::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// describe — value frequencies (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// One foldhash map `key → (count, first row, split-subset mask)` per 64K-row
// chunk, built in parallel and merged (counts summed, first = min, masks OR-ed).
// One O(distinct) sweep of the merged map yields n_unique, f1/f2, the capture
// history and, while there are ≤ 5 distinct values, their first rows. Keys come from `encode_series` (floats canonicalised;
// strings, nested and struct values hashed — collisions ~6e-11 per pair at 50K
// rows, accepted as documented in CLAUDE.md).

pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub count: u64,
    pub first: u64,
    pub mask: u8,
}

pub(crate) type Map = HashMap<u64, Entry, FixedState>;
/// (first row index, value) of the running extreme.
type Ext<T> = (u64, T);

pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub f1: u64,
    pub f2: u64,
    /// First rows of the distinct values, first-occurrence order, while there are ≤ 5
    /// (Recommend's boolean-pair rule reads them; streaming keeps the same list).
    pub first_few: Vec<u64>,
    /// Every distinct value occurred once (streaming's sampling phase: every sampled one).
    pub all_once: bool,
    /// Streaming's sampling phase: (HyperLogLog estimate, relative standard error); None
    /// while `n_unique` is exact.
    pub hll: Option<(f64, f64)>,
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
        let e = map.entry(key).or_insert(Entry {
            count: 0,
            first: row,
            mask: 0,
        });
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

/// The per-value map of `col`, rows counted from `offset` (streaming: the global row
/// of the batch's first value), so first rows and capture subsets are global.
pub(crate) fn frequency_map(col: &EncodedColumn, seed: u64, offset: u64) -> Map {
    col.values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, offset as usize + i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge)
}

pub(crate) fn frequencies(col: &EncodedColumn, seed: u64, lengths: Option<&[u64]>) -> Frequencies {
    let map = frequency_map(col, seed, 0);
    let (mut f1, mut f2, mut history) = (0u64, 0u64, [0u64; 7]);
    let n_unique = map.len() as u64;
    let mut first_few: Vec<u64> = if map.len() <= 5 {
        map.values().map(|e| e.first).collect()
    } else {
        Vec::new()
    };
    first_few.sort_unstable();
    let mut unique_len = 0u64;
    for e in map.into_values() {
        f1 += (e.count == 1) as u64;
        f2 += (e.count == 2) as u64;
        history[e.mask as usize - 1] += 1;
        if let Some(l) = lengths {
            unique_len += l[e.first as usize];
        }
    }
    Frequencies {
        n_unique,
        f1,
        f2,
        first_few,
        all_once: f1 == n_unique,
        hll: None,
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

fn extremes<T: Copy>(
    values: impl Iterator<Item = Option<T>>,
    lt: impl Fn(T, T) -> bool,
) -> (Option<u64>, Option<u64>) {
    let (mut lo, mut hi): (Option<Ext<T>>, Option<Ext<T>>) = (None, None);
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

pub(crate) fn arg_extremes(s: &Series) -> PolarsResult<(Option<u64>, Option<u64>)> {
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
            extremes(
                rows.values_iter()
                    .zip(valid.iter())
                    .map(|(r, ok)| (ok == Some(true)).then_some(r)),
                lt,
            )
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
        (
            Some(lo.map_or(v, |x: u64| x.min(v))),
            Some(hi.map_or(v, |x: u64| x.max(v))),
        )
    })
}

/// Per row of a **rechunked** ListChunked: `Some((start, len))` into its child
/// values (`get_inner()`), `None` for a null list.
pub(crate) fn list_ranges(ca: &ListChunked) -> Vec<Option<(usize, usize)>> {
    ca.downcast_iter()
        .flat_map(|arr| {
            let offsets = arr.offsets().as_slice();
            (0..arr.len()).map(move |i| {
                arr.is_valid(i)
                    .then(|| (offsets[i] as usize, (offsets[i + 1] - offsets[i]) as usize))
            })
        })
        .collect()
}

/// `min_len`/`max_len` for String/Categorical/Enum/Binary from an already-computed
/// `byte_lengths` vector (0 at null rows — filtered out via the series' validity,
/// not the value, so a genuine zero-length string still counts).
fn min_max_bytes(s: &Series, byte_lens: &[u64]) -> (Option<u64>, Option<u64>) {
    min_max(
        s.is_not_null()
            .iter()
            .zip(byte_lens)
            .map(|(ok, &l)| (ok == Some(true)).then_some(l)),
    )
}

/// `byte_lens` must be `Some` (from `byte_lengths`) for String/Categorical/Enum/Binary.
pub(crate) fn lengths(
    s: &Series,
    byte_lens: Option<&[u64]>,
) -> PolarsResult<(Option<u64>, Option<u64>)> {
    Ok(match s.dtype() {
        DataType::String
        | DataType::Categorical(_, _)
        | DataType::Enum(_, _)
        | DataType::Binary => min_max_bytes(
            s,
            byte_lens.expect("byte_lengths precomputed for String/Categorical/Enum/Binary"),
        ),
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            min_max(
                list_ranges(&ca)
                    .into_iter()
                    .map(|r| r.map(|(_, len)| len as u64)),
            )
        }
        DataType::Array(_, width) => min_max(
            s.is_not_null()
                .iter()
                .map(|ok| (ok == Some(true)).then_some(*width as u64)),
        ),
        _ => (None, None),
    })
}

/// Byte length of every value (0 for nulls) of a String, Categorical, Enum or Binary series.
pub(crate) fn byte_lengths(s: &Series) -> PolarsResult<Option<Vec<u64>>> {
    Ok(match s.dtype() {
        DataType::String => Some(
            s.str()?
                .iter()
                .map(|v| v.map_or(0, |x| x.len() as u64))
                .collect(),
        ),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            return byte_lengths(&s.cast(&DataType::String)?)
        }
        DataType::Binary => Some(
            s.binary()?
                .iter()
                .map(|v| v.map_or(0, |x| x.len() as u64))
                .collect(),
        ),
        _ => None,
    })
}

pub(crate) fn range(s: &Series, byte_lens: Option<&[u64]>) -> PolarsResult<Range> {
    let (argmin, argmax) = arg_extremes(s)?;
    let (min_len, max_len) = lengths(s, byte_lens)?;
    Ok(Range {
        argmin,
        argmax,
        min_len,
        max_len,
    })
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
    /// Values equal to -0.0 (arrow-cast renders them "-0.0", a decimal "0").
    pub n_neg_zero: u64,
}

pub(crate) fn frac_digits(repr: &str) -> u32 {
    let repr = repr.trim_start_matches('-');
    let (mantissa, exp) = match repr.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (repr, 0),
    };
    let frac = mantissa
        .split_once('.')
        .map_or("", |(_, f)| f)
        .trim_end_matches('0');
    (frac.len() as i64 - exp).max(0) as u32
}

impl FloatStats {
    pub(crate) fn merge(self, o: Self) -> Self {
        Self {
            n_nan: self.n_nan + o.n_nan,
            n_inf: self.n_inf + o.n_inf,
            n_fractional: self.n_fractional + o.n_fractional,
            max_frac_digits: self.max_frac_digits.max(o.max_frac_digits),
            n_f32_inexact: self.n_f32_inexact + o.n_f32_inexact,
            n_neg_zero: self.n_neg_zero + o.n_neg_zero,
        }
    }

    fn add_f64(&mut self, x: f64, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_neg_zero += (x == 0.0 && x.is_sign_negative()) as u64;
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self
                .max_frac_digits
                .max(Some(frac_digits(buf.format_finite(x))));
            self.n_f32_inexact += ((x as f32) as f64 != x) as u64;
        }
    }

    fn add_f32(&mut self, x: f32, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_neg_zero += (x == 0.0 && x.is_sign_negative()) as u64;
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self
                .max_frac_digits
                .max(Some(frac_digits(buf.format_finite(x))));
        }
    }
}

/// Fold `add` over the valid values of one Arrow chunk, CHUNK values per task.
fn fold<T: Copy + Sync>(
    values: &[T],
    validity: Option<&Bitmap>,
    add: impl Fn(&mut FloatStats, T, &mut ryu::Buffer) + Sync,
) -> FloatStats {
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
fn two(b: &[u8], i: usize) -> Option<u32> {
    let (x, y) = (*b.get(i)?, *b.get(i + 1)?);
    (x.is_ascii_digit() && y.is_ascii_digit()).then(|| ((x - b'0') * 10 + (y - b'0')) as u32)
}

fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400)) => 29,
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
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
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

fn opt_min<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Fraction digits chrono prints for a time with `sig` significant ones: 0, 3, 6 or 9.
fn chrono_frac_digits(sig: u32) -> u32 {
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

// ─────────────────────────────────────────────────────────────────────────────
// describe — column profile assembly and the `describe_columns` entry
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) type Row = Vec<AnyValue<'static>>;

/// Metrics and conclusions of any value series, in output order (spec 2026-10-01 §7).
pub(crate) fn value_fields() -> Vec<(&'static str, DataType)> {
    use DataType::{Boolean, Float64 as F64, String as Str, UInt32 as U32, UInt64 as U64};
    // Arrow has no plain 128-bit integer; parse_i128 caps values at 38 digits.
    let d38 = DataType::Decimal(Some(38), Some(0));
    vec![
        ("n_unique", U64),
        ("unique", Boolean),
        ("est_cardinality", F64),
        ("est_low", F64),
        ("est_high", F64),
        ("est_method", Str),
        ("estimates_agree", Boolean),
        ("class", Str),
        ("min", Str),
        ("max", Str),
        ("min_len", U64),
        ("max_len", U64),
        ("gcd", d38.clone()),
        ("sum_len", U64),
        ("sum_len_unique", U64),
        ("n_nan", U64),
        ("n_inf", U64),
        ("n_fractional", U64),
        ("max_frac_digits", U32),
        ("n_f32_inexact", U64),
        ("n_numeric", U64),
        ("n_numeric_int", U64),
        ("n_leading_zero", U64),
        ("numeric_int_min", d38.clone()),
        ("numeric_int_max", d38),
        ("numeric_max_int_digits", U32),
        ("numeric_max_frac_digits", U32),
        ("numeric_min_frac_digits", U32),
        ("numeric_max_sig_digits", U32),
        ("n_iso_date", U64),
        ("n_iso_time", U64),
        ("n_iso_datetime", U64),
        ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32),
        ("iso_max_sig_frac_digits", U32),
        ("iso_n_offsets", U64),
        ("iso_n_midnight", U64),
    ]
}

pub(crate) fn fields() -> Vec<(String, DataType)> {
    let mut f = vec![
        ("column".to_string(), DataType::String),
        ("n_rows".into(), DataType::UInt64),
        ("n_null".into(), DataType::UInt64),
    ];
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), DataType::UInt64));
    f.push(("inner_n_values".into(), DataType::UInt64));
    f.push(("inner_n_null".into(), DataType::UInt64));
    f.extend(
        value_fields()
            .into_iter()
            .map(|(n, d)| (format!("inner_{n}"), d)),
    );
    f
}

/// Private estimator inputs, appended by `describe_columns` only (Python's reference
/// conclusions read them; spec 2026-10-01 §13.8).
pub(crate) fn input_fields() -> Vec<(String, DataType)> {
    let level = [
        ("argmin", DataType::UInt64),
        ("argmax", DataType::UInt64),
        ("f1", DataType::UInt64),
        ("f2", DataType::UInt64),
        (
            "capture_history",
            DataType::List(Box::new(DataType::UInt64)),
        ),
    ];
    level
        .iter()
        .map(|(n, d)| (n.to_string(), d.clone()))
        .chain(level.iter().map(|(n, d)| (format!("inner_{n}"), d.clone())))
        .collect()
}

fn u64v(v: Option<u64>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, AnyValue::UInt64)
}
fn u32v(v: Option<u32>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, AnyValue::UInt32)
}
fn d38v(v: Option<i128>) -> AnyValue<'static> {
    v.map_or(AnyValue::Null, |v| AnyValue::Decimal(v, 0))
}
fn listv(v: &[u64]) -> AnyValue<'static> {
    AnyValue::List(Series::new(PlSmallStr::EMPTY, v))
}
fn nulls(n: usize) -> Row {
    vec![AnyValue::Null; n]
}

/// `proof`: also collect the streaming proof statistics (`StringStats::proof`).
pub(crate) fn strings(s: &Series, proof: bool) -> PolarsResult<Option<StringStats>> {
    let empty = || StringStats {
        proof,
        ..Default::default()
    };
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
                    .fold(empty, |mut acc, i| {
                        if arr.is_valid(i) {
                            acc.add(arr.value(i).as_bytes());
                        }
                        acc
                    })
                    .reduce(empty, StringStats::merge)
            })
            .fold(empty(), StringStats::merge),
    ))
}

/// Every value metric of one series (the column, or its list's inner values).
pub(crate) struct Profile {
    pub freq: Frequencies,
    pub range: Range,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    /// GCD of the physical values (gcd.rs); None for non-integer dtypes.
    pub gcd: Option<i128>,
    /// Total byte length of the non-null values (String, Categorical, Enum or Binary columns).
    pub sum_len: Option<u64>,
    /// Float32 series: `n_f32_inexact` does not apply.
    pub is_f32: bool,
    /// Rendered extremes (`recommend::render_value`); None for nested dtypes.
    pub min: Option<String>,
    pub max: Option<String>,
    /// Numeric extremes as f64 (integer, decimal and float dtypes), for `ordinal`.
    pub numeric: Option<(f64, f64)>,
}

/// `proof`: see `StringStats::proof` (false on the one-shot path).
pub(crate) fn profile(s: &Series, seed: u64, proof: bool) -> PolarsResult<Profile> {
    let lengths = byte_lengths(s)?;
    let range = range(s, lengths.as_deref())?;
    let nested = matches!(
        s.dtype(),
        DataType::List(_) | DataType::Array(..) | DataType::Struct(_)
    );
    let (min, max) = if nested {
        (None, None)
    } else {
        (render_at(s, range.argmin)?, render_at(s, range.argmax)?)
    };
    let numeric = numeric_extremes(s, range.argmin, range.argmax)?;
    Ok(Profile {
        freq: frequencies(&encode_series(s)?, seed, lengths.as_deref()),
        range,
        floats: float_stats(s)?,
        strings: strings(s, proof)?,
        gcd: crate::gcd::series_gcd(s)?,
        sum_len: lengths.map(|l| l.iter().sum()),
        is_f32: s.dtype() == &DataType::Float32,
        min,
        max,
        numeric,
    })
}

/// Row `i` rendered as arrow-rs text (spec 2026-10-01 §13.1).
fn render_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<String>> {
    let Some(i) = i else { return Ok(None) };
    let one = crate::sizes::classic_layout(&s.slice(i as i64, 1))?;
    Ok(crate::recommend::render_value(one.as_ref()))
}

/// The values at rows `lo` / `hi` as f64, for integer, decimal and float dtypes.
fn numeric_extremes(
    s: &Series,
    lo: Option<u64>,
    hi: Option<u64>,
) -> PolarsResult<Option<(f64, f64)>> {
    let dt = s.dtype();
    if !(dt.is_integer() || dt.is_float() || matches!(dt, DataType::Decimal(..))) {
        return Ok(None);
    }
    let at = |i: Option<u64>| -> PolarsResult<Option<f64>> {
        match i {
            None => Ok(None),
            Some(i) => Ok(s.slice(i as i64, 1).cast(&DataType::Float64)?.f64()?.get(0)),
        }
    };
    Ok(at(lo)?.zip(at(hi)?))
}

impl Profile {
    /// The metrics and conclusions in `value_fields()` order.
    pub(crate) fn row(&self, c: &Conclusions) -> Row {
        let f = &self.freq;
        let text = |s: &Option<String>| {
            s.clone()
                .map_or(AnyValue::Null, |s| AnyValue::StringOwned(s.into()))
        };
        let mut row: Row = vec![
            AnyValue::UInt64(f.n_unique),
            AnyValue::Boolean(c.unique),
            AnyValue::Float64(c.est.est_cardinality),
            c.est.est_low.map_or(AnyValue::Null, AnyValue::Float64),
            c.est.est_high.map_or(AnyValue::Null, AnyValue::Float64),
            AnyValue::StringOwned(c.est.method.name().into()),
            c.agree.map_or(AnyValue::Null, AnyValue::Boolean),
            AnyValue::StringOwned(c.class.into()),
            text(&self.min),
            text(&self.max),
            u64v(self.range.min_len),
            u64v(self.range.max_len),
            d38v(self.gcd),
            u64v(self.sum_len),
            u64v(f.sum_len_unique),
        ];
        match self.floats {
            Some(fl) => row.extend([
                AnyValue::UInt64(fl.n_nan),
                AnyValue::UInt64(fl.n_inf),
                AnyValue::UInt64(fl.n_fractional),
                u32v(fl.max_frac_digits),
                if self.is_f32 {
                    AnyValue::Null
                } else {
                    AnyValue::UInt64(fl.n_f32_inexact)
                },
            ]),
            None => row.extend(nulls(5)),
        }
        match &self.strings {
            Some(st) => {
                let (lo, hi) = if st.int_overflow {
                    (None, None)
                } else {
                    (st.int_min, st.int_max)
                };
                row.extend([
                    AnyValue::UInt64(st.n_numeric),
                    AnyValue::UInt64(st.n_numeric_int),
                    AnyValue::UInt64(st.n_leading_zero),
                    d38v(lo),
                    d38v(hi),
                    u32v(st.max_int_digits),
                    u32v(st.max_frac_digits),
                    u32v(st.min_frac_digits),
                    u32v(st.max_sig_digits),
                    AnyValue::UInt64(st.n_iso_date),
                    AnyValue::UInt64(st.n_iso_time),
                    AnyValue::UInt64(st.n_iso_datetime),
                    AnyValue::UInt64(st.n_iso_datetime_tz),
                    u32v(st.iso_max_frac_digits),
                    u32v(st.iso_max_sig_frac_digits),
                    AnyValue::UInt64(st.offsets.len() as u64),
                    AnyValue::UInt64(st.iso_n_midnight),
                ])
            }
            None => row.extend(nulls(17)),
        }
        row
    }

    /// `input_fields()` values for this level.
    pub(crate) fn input_row(&self) -> Row {
        vec![
            u64v(self.range.argmin),
            u64v(self.range.argmax),
            AnyValue::UInt64(self.freq.f1),
            AnyValue::UInt64(self.freq.f2),
            listv(&self.freq.capture_history),
        ]
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
    pub dtype: DataType,
    pub n_rows: u64,
    pub n_null: u64,
    pub outer: Profile,
    pub n_midnight: Option<u64>,
    pub inner: Option<Inner>,
}

impl Described {
    /// The column's and its inner values' conclusions.
    pub(crate) fn conclusions(&self, threshold: u64) -> (Conclusions, Option<Conclusions>) {
        let outer = conclude(
            &self.dtype,
            self.n_rows,
            self.n_null,
            &self.outer,
            threshold,
        );
        let inner = self.inner.as_ref().map(|i| {
            conclude(
                i.values.dtype(),
                i.values.len() as u64,
                i.values.null_count() as u64,
                &i.profile,
                threshold,
            )
        });
        (outer, inner)
    }

    /// One `fields()` row.
    pub(crate) fn row(&self, threshold: u64) -> Row {
        let (oc, ic) = self.conclusions(threshold);
        let mut row: Row = vec![
            AnyValue::StringOwned(self.name.clone()),
            AnyValue::UInt64(self.n_rows),
            AnyValue::UInt64(self.n_null),
        ];
        row.extend(self.outer.row(&oc));
        row.push(u64v(self.n_midnight));
        match (&self.inner, ic) {
            (Some(i), Some(ic)) => {
                row.push(AnyValue::UInt64(i.values.len() as u64));
                row.push(AnyValue::UInt64(i.values.null_count() as u64));
                row.extend(i.profile.row(&ic));
            }
            _ => row.extend(nulls(2 + value_fields().len())),
        }
        row
    }

    /// One `input_fields()` row.
    pub(crate) fn input_row(&self) -> Row {
        let mut row = self.outer.input_row();
        match &self.inner {
            Some(i) => row.extend(i.profile.input_row()),
            None => row.extend(nulls(5)),
        }
        row
    }
}

/// Datetime values at exactly 00:00:00 local time (column time zone, else naive).
pub(crate) fn n_midnight(s: &Series) -> PolarsResult<Option<u64>> {
    let DataType::Datetime(unit, tz) = s.dtype() else {
        return Ok(None);
    };
    let per_day: i64 = match unit {
        TimeUnit::Nanoseconds => 86_400_000_000_000,
        TimeUnit::Microseconds => 86_400_000_000,
        TimeUnit::Milliseconds => 86_400_000,
    };
    let phys = s.to_physical_repr();
    let values = phys.i64()?;
    let count = match tz {
        None => values
            .into_iter()
            .flatten()
            .filter(|v| v.rem_euclid(per_day) == 0)
            .count(),
        Some(tz) => {
            let zone: chrono_tz::Tz = tz
                .as_str()
                .parse()
                .map_err(|_| polars_err!(ComputeError: "describe: unknown time zone {tz}"))?;
            let per_sec = per_day / 86_400;
            values
                .into_iter()
                .flatten()
                .filter(|&v| {
                    v.rem_euclid(per_sec) == 0
                        && chrono::DateTime::from_timestamp(v.div_euclid(per_sec), 0).is_some_and(
                            |t| t.with_timezone(&zone).time() == chrono::NaiveTime::MIN,
                        )
                })
                .count()
        }
    };
    Ok(Some(count as u64))
}

/// Values one level down, skipping null lists — the same definition as the
/// Python `flatten` (drop_nulls, then explode the non-empty lists). Element i is
/// what `inner_argmin` / `inner_argmax` index into.
pub(crate) fn flatten(s: &Series) -> PolarsResult<Option<Series>> {
    let (inner, ranges): (Series, Vec<Option<(usize, usize)>>) = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            (ca.get_inner(), list_ranges(&ca))
        }
        DataType::Array(_, width) => {
            let ca = s.array()?.rechunk();
            let valid = ca.is_not_null();
            let ranges = valid
                .iter()
                .enumerate()
                .map(|(i, ok)| (ok == Some(true)).then_some((i * width, *width)))
                .collect();
            (ca.get_inner(), ranges)
        }
        _ => return Ok(None),
    };
    let idx: Vec<IdxSize> = ranges
        .into_iter()
        .flatten()
        .flat_map(|(start, len)| (start..start + len).map(|i| i as IdxSize))
        .collect();
    Ok(Some(inner.take_slice(&idx)?))
}

/// `proof`: see `StringStats::proof` (false on the one-shot path).
pub(crate) fn describe_one(s: &Series, seed: u64, proof: bool) -> PolarsResult<Described> {
    let inner = match flatten(s)? {
        Some(values) => {
            let profile = profile(&values, seed, proof)?;
            Some(Inner { values, profile })
        }
        None => None,
    };
    Ok(Described {
        name: s.name().clone(),
        dtype: s.dtype().clone(),
        n_rows: s.len() as u64,
        n_null: s.null_count() as u64,
        outer: profile(s, seed, proof)?,
        n_midnight: n_midnight(s)?,
        inner,
    })
}

/// A Struct series `name` with one field per `fields` entry and one row per `rows` entry.
pub(crate) fn assemble(
    name: &str,
    fields: &[(String, DataType)],
    rows: &[Row],
) -> PolarsResult<Series> {
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

/// `fields()` then the private `input_fields()`, one row per input.
pub(crate) fn describe_columns_impl(
    inputs: &[Series],
    seed: u64,
    threshold: u64,
) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs
        .par_iter()
        .map(|s| {
            describe_one(s, seed, false).map(|d| {
                let mut row = d.row(threshold);
                row.extend(d.input_row());
                row
            })
        })
        .collect::<PolarsResult<_>>()?;
    let mut schema = fields();
    schema.extend(input_fields());
    assemble("describe", &schema, &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freq(s: Series) -> Frequencies {
        frequencies(&encode_series(&s).unwrap(), 0, None)
    }

    #[test]
    fn frequency_map_offsets_rows() {
        let whole = encode_series(&Series::new("x".into(), &["a", "b", "a"])).unwrap();
        let tail = encode_series(&Series::new("x".into(), &["b", "a"])).unwrap();
        let (w, t) = (frequency_map(&whole, 7, 0), frequency_map(&tail, 7, 1));
        let b = whole.values[1];
        assert_eq!((w[&b].first, t[&b].first), (1, 1));
        assert_eq!(w[&b].mask, t[&b].mask); // capture subsets use the global row
        assert_eq!(frequencies(&whole, 7, None).n_unique, w.len() as u64);
    }

    #[test]
    fn counts_and_first_few() {
        let f = freq(Series::new(
            "a".into(),
            &[Some("a"), Some("a"), Some("b"), None],
        ));
        assert_eq!((f.n_unique, f.f1, f.f2, f.first_few), (2, 1, 1, vec![0, 2]));
        assert!(!f.all_once && f.hll.is_none());
        // More than five distinct values: no list.
        let g = freq(Series::new("a".into(), &[3i64, 1, 2, 1, 2, 3, 4, 5, 6]));
        assert!(g.first_few.is_empty());
    }

    #[test]
    fn merges_across_parallel_chunks() {
        let n = 3 * CHUNK + 5;
        // value 9 first appears in the third chunk and again in the fourth; every other value is 0.
        let mut v = vec![0i64; n];
        v[2 * CHUNK + 1] = 9;
        v[3 * CHUNK + 2] = 9;
        let f = freq(Series::new("a".into(), v));
        assert_eq!((f.n_unique, f.f1, f.f2), (2, 0, 1));
        assert_eq!(f.first_few, vec![0, (2 * CHUNK + 1) as u64]);
    }

    #[test]
    fn capture_history_sums_to_n_unique_and_fills_all_subsets() {
        let f = freq(Series::new(
            "a".into(),
            (0..10_000i64).map(|i| i % 10).collect::<Vec<_>>(),
        ));
        assert_eq!(f.capture_history, [0, 0, 0, 0, 0, 0, 10]);
        let g = freq(Series::new(
            "a".into(),
            (0..20_000i64)
                .map(|i| (i * 7919) % 5_003)
                .collect::<Vec<_>>(),
        ));
        assert_eq!(g.capture_history.iter().sum::<u64>(), g.n_unique);
    }

    #[test]
    fn subsets_are_roughly_uniform() {
        let mut counts = [0u64; 3];
        for row in 0..30_000 {
            counts[subset(0, row) as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn zero_rows_and_all_null() {
        let z = freq(Series::new_empty("a".into(), &DataType::Int32));
        assert_eq!((z.n_unique, z.capture_history), (0, [0; 7]));
        let a = freq(Series::new("a".into(), &[None::<i32>, None]));
        assert_eq!((a.n_unique, a.first_few), (0, vec![]));
    }

    #[test]
    fn struct_values_hash_whole_and_binary_encodes() {
        let a = Series::new("a".into(), &[1i32, 1, 1]);
        let b = Series::new("b".into(), &[Some("x"), Some("x"), None]);
        let s = StructChunked::from_series("s".into(), 3, [a, b].iter())
            .unwrap()
            .into_series();
        assert_eq!(freq(s).n_unique, 2);
        let bin = Series::new(
            "b".into(),
            &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), None],
        );
        assert_eq!(freq(bin).n_unique, 1);
    }

    fn r(s: Series) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
        let lens = byte_lengths(&s).unwrap();
        let x = range(&s, lens.as_deref()).unwrap();
        (x.argmin, x.argmax, x.min_len, x.max_len)
    }

    #[test]
    fn first_occurrence_extremes() {
        assert_eq!(
            r(Series::new("x".into(), &[5i64, 1, 3, 1, 5])),
            (Some(1), Some(0), None, None)
        );
        assert_eq!(
            r(Series::new("x".into(), &[0.0f64, -0.0, f64::NAN, 1.5])),
            (Some(0), Some(3), None, None)
        );
        assert_eq!(
            r(Series::new("x".into(), &[None::<i32>, None])),
            (None, None, None, None)
        );
    }

    #[test]
    fn strings_bytes_and_lengths() {
        assert_eq!(
            r(Series::new(
                "x".into(),
                &[Some("ab"), Some(""), None, Some("héllo")]
            )),
            (Some(1), Some(3), Some(0), Some(6))
        );
    }

    #[test]
    fn lists_use_polars_sort_order() {
        let s = Series::new(
            "x".into(),
            [
                Some(Series::new("".into(), &[1i64, 5])),
                Some(Series::new("".into(), &[2i64, 1])),
                None,
                Some(Series::new_empty("".into(), &DataType::Int64)),
            ],
        );
        assert_eq!(r(s), (Some(3), Some(1), Some(0), Some(2)));
    }

    #[test]
    fn decimal_places_of_shortest_reprs() {
        for (repr, want) in [
            ("0.1", 1),
            ("1e-7", 7),
            ("1.5e20", 0),
            ("1.5e+20", 0),
            ("3.0", 0),
            ("-0.0", 0),
            ("123.45", 2),
            ("1.25e-3", 5),
        ] {
            assert_eq!(frac_digits(repr), want, "{repr}");
        }
    }

    #[test]
    fn f64_stats() {
        let s = Series::new(
            "x".into(),
            &[
                Some(0.1),
                Some(1e-7),
                Some(1.5e20),
                Some(3.0),
                Some(f64::INFINITY),
                Some(f64::NAN),
                None,
            ],
        );
        let st = float_stats(&s).unwrap().unwrap();
        assert_eq!(
            (st.n_nan, st.n_inf, st.n_fractional, st.max_frac_digits),
            (1, 1, 2, Some(7))
        );
        assert_eq!(st.n_f32_inexact, 3); // 0.1, 1e-7, 1.5e20
    }

    #[test]
    fn f32_uses_its_own_repr_and_non_floats_are_none() {
        let st = float_stats(&Series::new("x".into(), &[0.1f32, 0.25]))
            .unwrap()
            .unwrap();
        assert_eq!(st.max_frac_digits, Some(2));
        assert!(float_stats(&Series::new("x".into(), &[1i32]))
            .unwrap()
            .is_none());
        assert_eq!(
            float_stats(&Series::new("x".into(), &[f64::NAN]))
                .unwrap()
                .unwrap()
                .max_frac_digits,
            None
        );
    }

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
            frequencies(&encode_series(&s).unwrap(), 0, Some(&lens)).sum_len_unique,
            Some(3)
        );
    }

    #[test]
    fn profile_gcd_and_lengths() {
        let p = profile(
            &Series::new("x".into(), &[Some(10i64), None, Some(30)]),
            0,
            false,
        )
        .unwrap();
        assert_eq!((p.gcd, p.sum_len), (Some(10), None));
        let s = profile(&Series::new("x".into(), &["ab", "ab", "c"]), 0, false).unwrap();
        assert_eq!(
            (s.gcd, s.sum_len, s.freq.sum_len_unique),
            (None, Some(5), Some(3))
        );
    }

    #[test]
    fn described_row_matches_fields() {
        let list = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]);
        let d = describe_one(&list, 0, false).unwrap();
        assert_eq!(d.row(10_000).len(), fields().len());
        assert_eq!(d.input_row().len(), input_fields().len());
        assert_eq!(d.inner.as_ref().unwrap().values.len(), 2);
        // Nested extremes are dropped; the inner values keep theirs.
        assert_eq!(
            (d.outer.min.as_deref(), d.outer.max.as_deref()),
            (None, None)
        );
        let inner = &d.inner.as_ref().unwrap().profile;
        assert_eq!(
            (inner.min.as_deref(), inner.max.as_deref()),
            (Some("1"), Some("2"))
        );
        let floats = describe_one(&Series::new("y".into(), &[1.5f64]), 0, false).unwrap();
        assert_eq!(floats.row(10_000).len(), fields().len());
        assert!(floats.inner.is_none() && floats.outer.floats.is_some());
    }

    #[test]
    fn rendered_extremes_are_arrow_rs_text() {
        let ts = Series::new("t".into(), &[1_704_164_645_000_000i64, 0])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let p = profile(&ts, 0, false).unwrap();
        assert_eq!(
            (p.min.as_deref(), p.max.as_deref()),
            (Some("1970-01-01T00:00:00"), Some("2024-01-02T03:04:05"))
        );
        let f = profile(
            &Series::new("f".into(), &[1.0f64, f64::NAN, -2.5]),
            0,
            false,
        )
        .unwrap();
        assert_eq!(
            (f.min.as_deref(), f.max.as_deref()),
            (Some("-2.5"), Some("1.0"))
        );
        assert_eq!(f.numeric, Some((-2.5, 1.0)));
        let s = profile(&Series::new("s".into(), &["b", "a"]), 0, false).unwrap();
        assert_eq!(
            (s.min.as_deref(), s.max.as_deref(), s.numeric),
            (Some("a"), Some("b"), None)
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

    #[test]
    fn float_negative_zero() {
        let st = float_stats(&Series::new("x".into(), &[0.0f64, -0.0, 1.0]))
            .unwrap()
            .unwrap();
        assert_eq!(st.n_neg_zero, 1);
    }
}
