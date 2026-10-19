//! ClickHouse on-disk size emulation (spec 2026-10-19 §3).

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{make_array, Array, ArrayRef, BooleanArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as AT, TimeUnit};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

/// How a ClickHouse column is laid out on disk (MergeTree Wide parts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChLayout {
    /// Little-endian fixed-width values; `signed` for the GCD codec's magnitude;
    /// `gcd_applies` for integer, decimal, Date, Date32, DateTime and DateTime64 types.
    Fixed {
        width: usize,
        signed: bool,
        gcd_applies: bool,
    },
    /// Per row a varint length and the bytes.
    String,
    /// A dictionary stream (String layout, the distinct values) and a keys stream.
    LowCardinality,
    FixedString(usize),
    /// A UInt8 null-map stream plus the inner type with defaults in null slots.
    Nullable(Box<ChLayout>),
    /// A UInt64 offsets stream plus the inner streams.
    Array(Box<ChLayout>),
    Tuple(Vec<ChLayout>),
}

/// One on-disk stream of a column.
pub(crate) struct Stream {
    pub bytes: Vec<u8>,
    /// Byte end of each top-level row in this stream (cumulative), so granules of top-level
    /// rows map to byte ranges (an Array's inner streams: the end of each top-level row's
    /// items). A LowCardinality dictionary is written once per part, not per row: all its
    /// `row_ends` equal the stream length, and it is cut into blocks by bytes alone (an
    /// approximation of ClickHouse, where the dictionary is one unit per part).
    pub row_ends: Vec<u64>,
    /// (value width, signed) when the GCD codec applies to this stream.
    pub gcd_width: Option<(usize, bool)>,
}

fn stream(bytes: Vec<u8>, row_ends: Vec<u64>) -> Stream {
    Stream {
        bytes,
        row_ends,
        gcd_width: None,
    }
}

/// `n` rows of `width` bytes each.
fn fixed_ends(n: usize, width: usize) -> Vec<u64> {
    (1..=n as u64).map(|i| i * width as u64).collect()
}

fn put_varint(out: &mut Vec<u8>, mut v: usize) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Append each value's low `width` bytes (little-endian); null → zeros.
fn put_fixed(
    it: impl Iterator<Item = Option<i128>>,
    width: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    if width == 0 || width > 16 {
        return Err(format!("unsupported fixed width {width}"));
    }
    for v in it {
        out.extend_from_slice(&v.unwrap_or(0).to_le_bytes()[..width]);
    }
    Ok(())
}

fn fixed_stream(
    a: &dyn Array,
    width: usize,
    signed: bool,
    gcd_applies: bool,
) -> Result<Stream, String> {
    let mut out = Vec::with_capacity(a.len() * width);
    macro_rules! prim {
        ($t:ty, $conv:expr) => {
            put_fixed(
                a.as_primitive::<$t>().iter().map(|o| o.map($conv)),
                width,
                &mut out,
            )?
        };
    }
    match a.data_type() {
        AT::Int8 => prim!(Int8Type, |v| v as i128),
        AT::Int16 => prim!(Int16Type, |v| v as i128),
        AT::Int32 => prim!(Int32Type, |v| v as i128),
        AT::Int64 => prim!(Int64Type, |v| v as i128),
        AT::UInt8 => prim!(UInt8Type, |v| v as i128),
        AT::UInt16 => prim!(UInt16Type, |v| v as i128),
        AT::UInt32 => prim!(UInt32Type, |v| v as i128),
        AT::UInt64 => prim!(UInt64Type, |v| v as i128),
        AT::Float16 => prim!(Float16Type, |v| v.to_bits() as i128),
        AT::Float32 => prim!(Float32Type, |v| v.to_bits() as i128),
        AT::Float64 => prim!(Float64Type, |v| v.to_bits() as i128),
        AT::Date32 => prim!(Date32Type, |v| v as i128),
        AT::Date64 => prim!(Date64Type, |v| v as i128),
        AT::Time32(TimeUnit::Second) => prim!(Time32SecondType, |v| v as i128),
        AT::Time32(_) => prim!(Time32MillisecondType, |v| v as i128),
        AT::Time64(TimeUnit::Microsecond) => prim!(Time64MicrosecondType, |v| v as i128),
        AT::Time64(_) => prim!(Time64NanosecondType, |v| v as i128),
        AT::Timestamp(TimeUnit::Second, _) => prim!(TimestampSecondType, |v| v as i128),
        AT::Timestamp(TimeUnit::Millisecond, _) => prim!(TimestampMillisecondType, |v| v as i128),
        AT::Timestamp(TimeUnit::Microsecond, _) => prim!(TimestampMicrosecondType, |v| v as i128),
        AT::Timestamp(TimeUnit::Nanosecond, _) => prim!(TimestampNanosecondType, |v| v as i128),
        AT::Decimal32(..) => prim!(Decimal32Type, |v| v as i128),
        AT::Decimal64(..) => prim!(Decimal64Type, |v| v as i128),
        AT::Decimal128(..) => prim!(Decimal128Type, |v| v),
        AT::Boolean => {
            let b: &BooleanArray = a.as_boolean();
            put_fixed(b.iter().map(|o| o.map(i128::from)), width, &mut out)?
        }
        AT::Dictionary(..) => {
            // An Enum: the 1-based category code (key + 1).
            let keys = arrow_cast::cast(a.as_any_dictionary().keys(), &AT::UInt64)
                .map_err(|e| e.to_string())?;
            put_fixed(
                keys.as_primitive::<UInt64Type>()
                    .iter()
                    .map(|o| o.map(|k| k as i128 + 1)),
                width,
                &mut out,
            )?
        }
        t => return Err(format!("type {t} has no fixed-width layout")),
    }
    Ok(Stream {
        bytes: out,
        row_ends: fixed_ends(a.len(), width),
        gcd_width: gcd_applies.then_some((width, signed)),
    })
}

/// Calls `f` for each row's bytes (None = null) of a string / binary array.
fn for_each_bytes(a: &dyn Array, f: &mut dyn FnMut(Option<&[u8]>)) -> Result<(), String> {
    match a.data_type() {
        AT::Utf8 => a
            .as_string::<i32>()
            .iter()
            .for_each(|v| f(v.map(str::as_bytes))),
        AT::LargeUtf8 => a
            .as_string::<i64>()
            .iter()
            .for_each(|v| f(v.map(str::as_bytes))),
        AT::Utf8View => a
            .as_string_view()
            .iter()
            .for_each(|v| f(v.map(str::as_bytes))),
        AT::Binary => a.as_binary::<i32>().iter().for_each(f),
        AT::LargeBinary => a.as_binary::<i64>().iter().for_each(f),
        AT::BinaryView => a.as_binary_view().iter().for_each(f),
        AT::FixedSizeBinary(_) => a.as_fixed_size_binary().iter().for_each(f),
        AT::Dictionary(_, v) => {
            let c = arrow_cast::cast(a, v).map_err(|e| e.to_string())?;
            return for_each_bytes(c.as_ref(), f);
        }
        t => return Err(format!("type {t} has no string layout")),
    }
    Ok(())
}

fn string_stream(a: &dyn Array) -> Result<Stream, String> {
    let (mut bytes, mut ends) = (Vec::new(), Vec::with_capacity(a.len()));
    for_each_bytes(a, &mut |v| {
        let v = v.unwrap_or(&[]);
        put_varint(&mut bytes, v.len());
        bytes.extend_from_slice(v);
        ends.push(bytes.len() as u64);
    })?;
    Ok(stream(bytes, ends))
}

fn fixed_string_stream(a: &dyn Array, n: usize) -> Result<Stream, String> {
    let mut bytes = Vec::with_capacity(a.len() * n);
    let mut err = None;
    for_each_bytes(a, &mut |v| {
        let v = v.unwrap_or(&[]);
        if v.len() > n {
            err = Some(format!(
                "value of {} bytes exceeds FixedString({n})",
                v.len()
            ));
        } else {
            bytes.extend_from_slice(v);
            bytes.resize(bytes.len() + n - v.len(), 0);
        }
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    Ok(stream(bytes, fixed_ends(a.len(), n)))
}

/// Dictionary stream + keys stream; `nullable`: key 0 = null, value keys shift by one.
fn low_cardinality_streams(a: &dyn Array, nullable: bool) -> Result<Vec<Stream>, String> {
    let mut ids: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut dict = Vec::new();
    let mut keys = Vec::with_capacity(a.len());
    let mut err = None;
    for_each_bytes(a, &mut |v| match v {
        Some(v) => {
            let next = ids.len() as u32;
            let id = *ids.entry(v.to_vec()).or_insert_with(|| {
                put_varint(&mut dict, v.len());
                dict.extend_from_slice(v);
                next
            });
            keys.push(id + nullable as u32);
        }
        None if nullable => keys.push(0),
        None => err = Some("null in a non-nullable LowCardinality column".to_string()),
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    let entries = ids.len() + nullable as usize;
    let kw = if entries <= 256 {
        1
    } else if entries <= 65_536 {
        2
    } else {
        4
    };
    let mut key_bytes = Vec::with_capacity(keys.len() * kw);
    for k in &keys {
        key_bytes.extend_from_slice(&k.to_le_bytes()[..kw]);
    }
    let dict_ends = vec![dict.len() as u64; keys.len()];
    Ok(vec![
        stream(dict, dict_ends),
        stream(key_bytes, fixed_ends(keys.len(), kw)),
    ])
}

fn null_map(a: &dyn Array) -> Stream {
    let nulls = a.logical_nulls();
    let bytes = (0..a.len())
        .map(|i| nulls.as_ref().is_some_and(|n| n.is_null(i)) as u8)
        .collect();
    stream(bytes, fixed_ends(a.len(), 1))
}

/// Cumulative item counts (n + 1, from 0) and the flattened child of a list array.
fn list_parts(a: &dyn Array) -> Result<(Vec<u64>, ArrayRef), String> {
    let n = a.len();
    let (offsets, child): (Vec<u64>, ArrayRef) = match a.data_type() {
        AT::List(_) => {
            let l = a.as_list::<i32>();
            let o = l.offsets();
            let (s, e) = (o[0] as usize, o[n] as usize);
            (
                o.iter().map(|&x| (x as usize - s) as u64).collect(),
                l.values().slice(s, e - s),
            )
        }
        AT::LargeList(_) => {
            let l = a.as_list::<i64>();
            let o = l.offsets();
            let (s, e) = (o[0] as usize, o[n] as usize);
            (
                o.iter().map(|&x| (x as usize - s) as u64).collect(),
                l.values().slice(s, e - s),
            )
        }
        AT::FixedSizeList(..) => {
            let l = a.as_fixed_size_list();
            let size = l.value_length() as usize;
            (
                (0..=n as u64).map(|i| i * size as u64).collect(),
                l.values().slice(l.value_offset(0) as usize, n * size),
            )
        }
        t => return Err(format!("type {t} is not a list")),
    };
    // ClickHouse stores a null list as an empty array: skip its items, repeat the offset.
    if let Some(nb) = a.nulls() {
        if (0..n).any(|i| nb.is_null(i) && offsets[i + 1] != offsets[i]) {
            let (mut ends, mut keep, mut total) = (vec![0u64], Vec::new(), 0u64);
            for i in 0..n {
                if nb.is_valid(i) {
                    keep.extend(offsets[i]..offsets[i + 1]);
                    total += offsets[i + 1] - offsets[i];
                }
                ends.push(total);
            }
            let child = arrow_select::take::take(
                child.as_ref(),
                &arrow_array::UInt64Array::from(keep),
                None,
            )
            .map_err(|e| e.to_string())?;
            return Ok((ends, child));
        }
    }
    Ok((offsets, child))
}

/// `a` with `parent` nulls added (a null struct row → its fields' defaults).
fn with_nulls(a: &ArrayRef, parent: &NullBuffer) -> Result<ArrayRef, String> {
    let nulls = NullBuffer::union(a.nulls(), Some(parent));
    let data = a
        .to_data()
        .into_builder()
        .nulls(nulls)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(make_array(data))
}

/// The column's streams in ClickHouse's layout: `a` is the recommended array (classic Arrow
/// layout: Int*/UInt*/Float*, Decimal*, Date32, Timestamp, Boolean, Utf8/LargeUtf8/views,
/// Binary/LargeBinary, FixedSizeBinary, Dictionary<K, Utf8>, List/LargeList/FixedSizeList,
/// Struct), `layout` its `ChLayout`.
pub(crate) fn streams(a: &dyn Array, layout: &ChLayout) -> Result<Vec<Stream>, String> {
    match layout {
        ChLayout::Fixed {
            width,
            signed,
            gcd_applies,
        } => Ok(vec![fixed_stream(a, *width, *signed, *gcd_applies)?]),
        ChLayout::String => Ok(vec![string_stream(a)?]),
        ChLayout::FixedString(n) => Ok(vec![fixed_string_stream(a, *n)?]),
        ChLayout::LowCardinality => low_cardinality_streams(a, false),
        ChLayout::Nullable(inner) if **inner == ChLayout::LowCardinality => {
            low_cardinality_streams(a, true)
        }
        ChLayout::Nullable(inner) => {
            let mut out = vec![null_map(a)];
            out.extend(streams(a, inner)?);
            Ok(out)
        }
        ChLayout::Array(inner) => {
            let (ends, child) = list_parts(a)?;
            let offsets: Vec<u8> = ends[1..].iter().flat_map(|e| e.to_le_bytes()).collect();
            let mut out = vec![stream(offsets, fixed_ends(a.len(), 8))];
            for s in streams(child.as_ref(), inner)? {
                let whole = s
                    .row_ends
                    .first()
                    .is_some_and(|f| s.row_ends.iter().all(|e| e == f));
                let row_ends = if whole {
                    vec![s.bytes.len() as u64; a.len()]
                } else {
                    ends[1..]
                        .iter()
                        .map(|&e| {
                            if e == 0 {
                                0
                            } else {
                                s.row_ends[e as usize - 1]
                            }
                        })
                        .collect()
                };
                out.push(Stream { row_ends, ..s });
            }
            Ok(out)
        }
        ChLayout::Tuple(fields) => {
            let s = a
                .as_struct_opt()
                .ok_or_else(|| format!("type {} is not a struct", a.data_type()))?;
            if s.num_columns() != fields.len() {
                return Err("struct and tuple layout differ in field count".to_string());
            }
            let mut out = Vec::new();
            for (col, l) in s.columns().iter().zip(fields) {
                match s.nulls() {
                    Some(nb) => out.extend(streams(with_nulls(col, nb)?.as_ref(), l)?),
                    None => out.extend(streams(col.as_ref(), l)?),
                }
            }
            Ok(out)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Compression {
    Lz4,
    Zstd(i32),
}

/// A codec chain: optional GCD, then a compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Chain {
    pub gcd: bool,
    pub compression: Compression,
}

impl Chain {
    /// As written in DDL: "LZ4", "ZSTD(3)", "GCD, LZ4", "GCD, ZSTD(3)".
    pub(crate) fn ddl(&self) -> String {
        let c = match self.compression {
            Compression::Lz4 => "LZ4".to_string(),
            Compression::Zstd(l) => format!("ZSTD({l})"),
        };
        if self.gcd {
            format!("GCD, {c}")
        } else {
            c
        }
    }
}

/// Emulated size of the streams under `chain`, and the histogram of the uncompressed bytes
/// the compressor sees (after GCD).
pub(crate) struct Emulated {
    pub bytes: u64,
    pub histogram: [u64; 256],
}

pub(crate) const GRANULE_ROWS: u64 = 8_192;
pub(crate) const MIN_BLOCK: u64 = 65_536;
pub(crate) const MAX_BLOCK: u64 = 1_048_576;
/// 16-byte checksum + 9-byte header.
pub(crate) const BLOCK_HEADER: u64 = 25;

/// Byte ranges of a stream's compression blocks: granules of top-level rows are appended to
/// the open block, which closes once it holds at least `MIN_BLOCK` bytes; a granule over
/// `MAX_BLOCK` closes the open block and is split into `MAX_BLOCK` pieces, each a block.
/// A stream whose `row_ends` are all equal (a LowCardinality dictionary) is one granule.
fn block_ranges(s: &Stream) -> Vec<(usize, usize)> {
    let total = s.bytes.len() as u64;
    let n = s.row_ends.len();
    let whole = n == 0 || s.row_ends.iter().all(|&e| e == s.row_ends[0]);
    let mut granules: Vec<(u64, u64)> = Vec::new();
    if whole {
        granules.push((0, total));
    } else {
        let g = GRANULE_ROWS as usize;
        let mut start = 0;
        for k in 0..n.div_ceil(g) {
            let end = s.row_ends[((k + 1) * g).min(n) - 1];
            granules.push((start, end));
            start = end;
        }
    }
    let (mut blocks, mut open) = (Vec::new(), None::<(u64, u64)>);
    for (gs, ge) in granules {
        if ge == gs {
            continue;
        }
        if ge - gs > MAX_BLOCK {
            blocks.extend(open.take());
            let mut p = gs;
            while p < ge {
                let q = (p + MAX_BLOCK).min(ge);
                blocks.push((p, q));
                p = q;
            }
            continue;
        }
        let b = open.get_or_insert((gs, gs));
        b.1 = ge;
        if b.1 - b.0 >= MIN_BLOCK {
            blocks.extend(open.take());
        }
    }
    blocks.extend(open);
    blocks
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect()
}

/// Uncompressed byte length of each block of a stream.
#[cfg(test)]
fn block_sizes(s: &Stream) -> Vec<u64> {
    block_ranges(s)
        .iter()
        .map(|&(a, b)| (b - a) as u64)
        .collect()
}

/// The GCD codec on one block of `width`-byte values: `[gcd] + values / gcd`. The GCD is of
/// the values' magnitudes (zeros skipped); GCD 0 or 1 leaves the block unchanged. (An
/// approximation of the real layout, to be checked by a spike.)
fn gcd_transform(block: &[u8], width: usize, signed: bool) -> Cow<'_, [u8]> {
    if width == 0 || width > 16 || !block.len().is_multiple_of(width) {
        return Cow::Borrowed(block);
    }
    let read = |c: &[u8]| -> u128 {
        let mut buf = [0u8; 16];
        buf[..width].copy_from_slice(c);
        if signed && c[width - 1] & 0x80 != 0 {
            buf[width..].fill(0xFF);
        }
        u128::from_le_bytes(buf)
    };
    let magnitude = |v: u128| {
        if signed {
            (v as i128).unsigned_abs()
        } else {
            v
        }
    };
    let mut g = 0u128;
    for c in block.chunks_exact(width) {
        let m = magnitude(read(c));
        if m == 0 {
            continue;
        }
        g = if g == 0 { m } else { binary_gcd(g, m) };
        if g == 1 {
            return Cow::Borrowed(block);
        }
    }
    if g <= 1 || (signed && g > i128::MAX as u128) {
        return Cow::Borrowed(block);
    }
    let mut out = Vec::with_capacity(block.len() + width);
    out.extend_from_slice(&g.to_le_bytes()[..width]);
    for c in block.chunks_exact(width) {
        let v = read(c);
        let q = if signed {
            ((v as i128) / g as i128) as u128
        } else {
            v / g
        };
        out.extend_from_slice(&q.to_le_bytes()[..width]);
    }
    Cow::Owned(out)
}

fn binary_gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn compressed_len(bytes: &[u8], c: Compression) -> Result<u64, String> {
    match c {
        Compression::Lz4 => {
            lz4::block::compress(bytes, Some(lz4::block::CompressionMode::DEFAULT), false)
                .map(|v| v.len() as u64)
                .map_err(|e| e.to_string())
        }
        Compression::Zstd(l) => zstd::bulk::compress(bytes, l)
            .map(|v| v.len() as u64)
            .map_err(|e| e.to_string()),
    }
}

/// Emulated size of the streams under `chain` (every block costs `BLOCK_HEADER` + its
/// compressed payload) and the histogram of the bytes the compressor sees.
pub(crate) fn emulate(streams: &[Stream], chain: Chain) -> Result<Emulated, String> {
    let work: Vec<(&Stream, (usize, usize))> = streams
        .iter()
        .flat_map(|s| block_ranges(s).into_iter().map(move |r| (s, r)))
        .collect();
    let parts = work
        .par_iter()
        .map(|&(s, (a, b))| {
            let raw = &s.bytes[a..b];
            let data = match s.gcd_width {
                Some((w, signed)) if chain.gcd => gcd_transform(raw, w, signed),
                _ => Cow::Borrowed(raw),
            };
            let mut h = [0u64; 256];
            for &x in data.iter() {
                h[x as usize] += 1;
            }
            Ok((BLOCK_HEADER + compressed_len(&data, chain.compression)?, h))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut out = Emulated {
        bytes: 0,
        histogram: [0; 256],
    };
    for (n, h) in parts {
        out.bytes += n;
        for (t, v) in out.histogram.iter_mut().zip(h) {
            *t += v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::*;
    use arrow_schema::{Field, Fields};
    use std::sync::Arc;

    fn fx(width: usize, signed: bool) -> ChLayout {
        ChLayout::Fixed {
            width,
            signed,
            gcd_applies: true,
        }
    }
    fn nullable(l: ChLayout) -> ChLayout {
        ChLayout::Nullable(Box::new(l))
    }

    #[test]
    fn nullable_int32() {
        let a = Int32Array::from(vec![Some(1), Some(-1), None]);
        let s = streams(&a, &nullable(fx(4, true))).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].bytes, [0, 0, 1]);
        assert_eq!(s[1].bytes, [1, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0]);
        assert_eq!(s[1].gcd_width, Some((4, true)));
        assert_eq!(s[0].gcd_width, None);
    }

    #[test]
    fn strings() {
        let a = StringArray::from(vec![Some("a"), Some(""), None]);
        let s = streams(&a, &ChLayout::String).unwrap();
        assert_eq!(s[0].bytes, [1, b'a', 0, 0]);
        let a = StringArray::from(vec!["ab", "c"]);
        assert_eq!(streams(&a, &ChLayout::String).unwrap()[0].row_ends, [3, 5]);
        let long = "x".repeat(300);
        let a = LargeStringArray::from(vec![long.as_str()]);
        let s = streams(&a, &ChLayout::String).unwrap();
        assert_eq!(s[0].bytes[..2], [0xAC, 0x02]);
        assert_eq!(s[0].bytes.len(), 302);
    }

    #[test]
    fn fixed_string() {
        let a = StringArray::from(vec![Some("ab"), None]);
        let s = streams(&a, &ChLayout::FixedString(2)).unwrap();
        assert_eq!(s[0].bytes, [b'a', b'b', 0, 0]);
        assert_eq!(s[0].row_ends, [2, 4]);
    }

    #[test]
    fn low_cardinality() {
        let a = StringArray::from(vec!["x", "y", "x"]);
        let s = streams(&a, &ChLayout::LowCardinality).unwrap();
        assert_eq!(s[0].bytes, [1, b'x', 1, b'y']);
        assert_eq!(s[0].row_ends, [4, 4, 4]);
        assert_eq!(s[1].bytes, [0, 1, 0]);
        let a = StringArray::from(vec![Some("x"), None, Some("x")]);
        let s = streams(&a, &nullable(ChLayout::LowCardinality)).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].bytes, [1, b'x']);
        assert_eq!(s[1].bytes, [1, 0, 1]);
        // Key width 2 above 256 entries.
        let vals: Vec<String> = (0..300).map(|i| i.to_string()).collect();
        let a = StringArray::from(vals);
        let s = streams(&a, &ChLayout::LowCardinality).unwrap();
        assert_eq!(s[1].bytes.len(), 600);
        let d: DictionaryArray<Int8Type> = vec!["p", "q", "p"].into_iter().collect();
        let s = streams(&d, &ChLayout::LowCardinality).unwrap();
        assert_eq!(s[1].bytes, [0, 1, 0]);
    }

    #[test]
    fn array_int8() {
        let a = ListArray::from_iter_primitive::<Int8Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![]),
            Some(vec![Some(3)]),
        ]);
        let l = ChLayout::Array(Box::new(fx(1, true)));
        let s = streams(&a, &l).unwrap();
        let off: Vec<u8> = [2u64, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(s[0].bytes, off);
        assert_eq!(s[1].bytes, [1, 2, 3]);
        assert_eq!(s[1].row_ends, [2, 2, 3]);
    }

    #[test]
    fn null_list_is_stored_empty() {
        let a = ListArray::from_iter_primitive::<Int8Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(9), Some(9)]),
            Some(vec![Some(3)]),
        ]);
        let (f, o, v, _) = a.into_parts();
        let nulls = NullBuffer::from(vec![true, false, true]);
        let a = ListArray::new(f, o, v, Some(nulls));
        let l = ChLayout::Array(Box::new(fx(1, true)));
        let s = streams(&a, &l).unwrap();
        let off: Vec<u8> = [2u64, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(s[0].bytes, off);
        assert_eq!(s[1].bytes, [1, 2, 3]);
        assert_eq!(s[1].row_ends, [2, 2, 3]);
    }

    #[test]
    fn temporal_and_decimal() {
        let a = Date32Array::from(vec![1]);
        assert_eq!(streams(&a, &fx(2, false)).unwrap()[0].bytes, [1, 0]);
        let a = Decimal128Array::from(vec![150])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert_eq!(
            streams(&a, &fx(8, true)).unwrap()[0].bytes,
            [0x96, 0, 0, 0, 0, 0, 0, 0]
        );
        let a = TimestampMillisecondArray::from(vec![1000]);
        assert_eq!(
            streams(&a, &fx(8, true)).unwrap()[0].bytes,
            [0xE8, 3, 0, 0, 0, 0, 0, 0]
        );
        let a = TimestampSecondArray::from(vec![1]);
        assert_eq!(streams(&a, &fx(4, false)).unwrap()[0].bytes, [1, 0, 0, 0]);
        let a = BooleanArray::from(vec![true, false]);
        assert_eq!(streams(&a, &fx(1, false)).unwrap()[0].bytes, [1, 0]);
        let a = Float32Array::from(vec![1.0f32]);
        assert_eq!(
            streams(&a, &fx(4, true)).unwrap()[0].bytes,
            1.0f32.to_le_bytes()
        );
    }

    #[test]
    fn enum_codes_and_tuple() {
        let d: DictionaryArray<Int8Type> = vec!["p", "q", "p"].into_iter().collect();
        assert_eq!(streams(&d, &fx(1, false)).unwrap()[0].bytes, [1, 2, 1]);
        let fields = Fields::from(vec![
            Field::new("a", AT::Int8, true),
            Field::new("b", AT::Utf8, true),
        ]);
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![5, 6])),
            Arc::new(StringArray::from(vec!["u", "v"])),
        ];
        let st = StructArray::new(fields, cols, Some(NullBuffer::from(vec![true, false])));
        let l = ChLayout::Tuple(vec![fx(1, true), ChLayout::String]);
        let s = streams(&st, &l).unwrap();
        assert_eq!(s[0].bytes, [5, 0]);
        assert_eq!(s[1].bytes, [1, b'u', 0]);
    }

    fn plain(bytes: Vec<u8>, width: usize) -> Stream {
        let n = bytes.len() / width;
        stream(bytes, fixed_ends(n, width))
    }

    #[test]
    fn blocks_cut_on_granules() {
        let a = UInt64Array::from_iter_values(0..20_000u64);
        let s = streams(&a, &fx(8, false)).unwrap();
        assert_eq!(block_sizes(&s[0]), [65_536, 65_536, 3_616 * 8]);
        let a = UInt8Array::from(vec![7u8; 100_000]);
        let s = streams(&a, &fx(1, false)).unwrap();
        assert_eq!(block_sizes(&s[0]), [65_536, 100_000 - 65_536]);
    }

    #[test]
    fn oversize_granule_is_split() {
        // 3-byte varint + value = 2 MiB - 1 bytes.
        let v = "x".repeat((2 << 20) - 4);
        let a = StringArray::from(vec![v.as_str()]);
        let s = streams(&a, &ChLayout::String).unwrap();
        assert_eq!(block_sizes(&s[0]), [MAX_BLOCK, MAX_BLOCK - 1]);
        // A dictionary is cut by bytes alone.
        let s = streams(&a, &ChLayout::LowCardinality).unwrap();
        assert_eq!(block_sizes(&s[0]), [MAX_BLOCK, MAX_BLOCK - 1]);
    }

    fn i64s(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn gcd_transform_cases() {
        let input = i64s(&[1000, 2000, 3000]);
        let t = gcd_transform(&input, 8, true);
        assert_eq!(&t[..8], [0xE8, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(t.as_ref(), i64s(&[1000, 1, 2, 3]).as_slice());
        let input = i64s(&[-6, 9]);
        let t = gcd_transform(&input, 8, true);
        assert_eq!(t.as_ref(), i64s(&[3, -2, 3]).as_slice());
        let z = i64s(&[0, 0]);
        assert_eq!(gcd_transform(&z, 8, true).as_ref(), z.as_slice());
        let one = i64s(&[2, 3]);
        assert_eq!(gcd_transform(&one, 8, true).as_ref(), one.as_slice());
        // Zeros are skipped; unsigned narrow width.
        assert_eq!(
            gcd_transform(&[0, 10, 20], 1, false).as_ref(),
            [10, 0, 1, 2]
        );
        // Signed narrow width sign-extends: -6 and 9.
        assert_eq!(gcd_transform(&[0xFA, 9], 1, true).as_ref(), [3, 0xFE, 3]);
    }

    #[test]
    fn histogram_and_sizes() {
        let lz4 = Chain {
            gcd: false,
            compression: Compression::Lz4,
        };
        let e = emulate(&[plain(vec![0u8; 10], 1)], lz4).unwrap();
        assert_eq!(e.histogram[0], 10);
        assert!(e.histogram[1..].iter().all(|&c| c == 0));
        let payload = lz4::block::compress(
            &[0u8; 10],
            Some(lz4::block::CompressionMode::DEFAULT),
            false,
        )
        .unwrap()
        .len() as u64;
        assert_eq!(e.bytes, BLOCK_HEADER + payload);
        // The histogram counts the post-GCD bytes: header 1000 = E8 03, then 1, 2, 3.
        let mut s = plain(i64s(&[1000, 2000, 3000]), 8);
        s.gcd_width = Some((8, true));
        let e = emulate(&[s], Chain { gcd: true, ..lz4 }).unwrap();
        assert_eq!(e.histogram.iter().sum::<u64>(), 32);
        assert_eq!(
            (e.histogram[0xE8], e.histogram[3], e.histogram[1]),
            (1, 2, 1)
        );
        let z = Chain {
            gcd: false,
            compression: Compression::Zstd(3),
        };
        assert!(emulate(&[plain(vec![1; 100], 1)], z).unwrap().bytes > BLOCK_HEADER);
    }

    #[test]
    fn chain_ddl() {
        let c = |gcd, compression| Chain { gcd, compression }.ddl();
        assert_eq!(c(false, Compression::Lz4), "LZ4");
        assert_eq!(c(false, Compression::Zstd(3)), "ZSTD(3)");
        assert_eq!(c(true, Compression::Lz4), "GCD, LZ4");
        assert_eq!(c(true, Compression::Zstd(3)), "GCD, ZSTD(3)");
    }
}
