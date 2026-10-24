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
    /// `int_codecs` for integer, decimal, Date, Date32, DateTime and DateTime64 types.
    Fixed {
        width: usize,
        signed: bool,
        int_codecs: bool,
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
    /// (value width, signed) for integer streams: the GCD, Delta and DoubleDelta codecs apply.
    pub int_width: Option<(usize, bool)>,
}

fn stream(bytes: Vec<u8>, row_ends: Vec<u64>) -> Stream {
    Stream {
        bytes,
        row_ends,
        int_width: None,
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
    int_codecs: bool,
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
        int_width: int_codecs.then_some((width, signed)),
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
            int_codecs,
        } => Ok(vec![fixed_stream(a, *width, *signed, *int_codecs)?]),
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
        ChLayout::Array(inner) => array_streams(a, inner),
        ChLayout::Tuple(fields) => tuple_streams(a, fields),
    }
}

/// `Array(inner)`: the offsets stream, then the child's streams with their row ends lifted to
/// the list rows.
fn array_streams(a: &dyn Array, inner: &ChLayout) -> Result<Vec<Stream>, String> {
    let (ends, child) = list_parts(a)?;
    let offsets: Vec<u8> = ends[1..].iter().flat_map(|e| e.to_le_bytes()).collect();
    let mut out = vec![stream(offsets, fixed_ends(a.len(), 8))];
    for s in streams(child.as_ref(), inner)? {
        let row_ends = list_row_ends(&s, &ends[1..], a.len());
        out.push(Stream { row_ends, ..s });
    }
    Ok(out)
}

/// A child stream's row ends per list row: all rows alike when the stream is one whole block,
/// else the child's end after each list's last item (0 for an empty prefix).
fn list_row_ends(s: &Stream, list_ends: &[u64], rows: usize) -> Vec<u64> {
    let whole = s
        .row_ends
        .first()
        .is_some_and(|f| s.row_ends.iter().all(|e| e == f));
    if whole {
        return vec![s.bytes.len() as u64; rows];
    }
    list_ends
        .iter()
        .map(|&e| {
            if e == 0 {
                0
            } else {
                s.row_ends[e as usize - 1]
            }
        })
        .collect()
}

/// `Tuple(fields)`: each struct field's streams, a null struct row nulling its fields.
fn tuple_streams(a: &dyn Array, fields: &[ChLayout]) -> Result<Vec<Stream>, String> {
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

/// The transform codec before the compression (spec 2026-10-23 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Transform {
    None,
    Gcd,
    Delta,
    DoubleDelta,
}

impl Transform {
    #[cfg(test)]
    pub(crate) fn from_gcd(gcd: bool) -> Self {
        if gcd {
            Transform::Gcd
        } else {
            Transform::None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Compression {
    /// No generic compression: used by the spikes and `clickhouse_profile` only, never chosen.
    None,
    Lz4,
    Zstd(i32),
}

/// A codec chain: an optional transform, then a compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Chain {
    pub transform: Transform,
    pub compression: Compression,
}

impl Chain {
    /// As written in DDL: "LZ4", "ZSTD(3)", "GCD, LZ4", "Delta, ZSTD(3)", "DoubleDelta",
    /// "NONE" for no codec at all.
    pub(crate) fn ddl(&self) -> String {
        let t = match self.transform {
            Transform::None => None,
            Transform::Gcd => Some("GCD"),
            Transform::Delta => Some("Delta"),
            Transform::DoubleDelta => Some("DoubleDelta"),
        };
        let c = match self.compression {
            Compression::None => None,
            Compression::Lz4 => Some("LZ4".to_string()),
            Compression::Zstd(l) => Some(format!("ZSTD({l})")),
        };
        match (t, c) {
            (None, None) => "NONE".to_string(),
            (Some(t), None) => t.to_string(),
            (None, Some(c)) => c,
            (Some(t), Some(c)) => format!("{t}, {c}"),
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
/// the values' magnitudes (zeros skipped); GCD 0 or 1 leaves the block unchanged. (Layout
/// verified byte-exact against ClickHouse 26.8: spec 2026-10-23 §7.)
fn gcd_transform(block: &[u8], width: usize, signed: bool) -> Cow<'_, [u8]> {
    if width == 0 || width > 16 || !block.len().is_multiple_of(width) {
        return Cow::Borrowed(block);
    }
    let g = block_gcd(block, width, signed);
    if g <= 1 || (signed && g > i128::MAX as u128) {
        return Cow::Borrowed(block);
    }
    let mut out = Vec::with_capacity(block.len() + width);
    out.extend_from_slice(&g.to_le_bytes()[..width]);
    for c in block.chunks_exact(width) {
        let v = read_wide(c, width, signed);
        let q = if signed {
            ((v as i128) / g as i128) as u128
        } else {
            v / g
        };
        out.extend_from_slice(&q.to_le_bytes()[..width]);
    }
    Cow::Owned(out)
}

/// The `width`-byte little-endian value `c`, sign-extended to 128 bits when `signed`.
fn read_wide(c: &[u8], width: usize, signed: bool) -> u128 {
    let mut buf = [0u8; 16];
    buf[..width].copy_from_slice(c);
    if signed && c[width - 1] & 0x80 != 0 {
        buf[width..].fill(0xFF);
    }
    u128::from_le_bytes(buf)
}

/// GCD of the magnitudes of a block's `width`-byte values, zeros skipped (0 when all are
/// zero); stops at 1.
fn block_gcd(block: &[u8], width: usize, signed: bool) -> u128 {
    let mut g = 0u128;
    for c in block.chunks_exact(width) {
        let v = read_wide(c, width, signed);
        let m = if signed {
            (v as i128).unsigned_abs()
        } else {
            v
        };
        if m == 0 {
            continue;
        }
        g = if g == 0 { m } else { binary_gcd(g, m) };
        if g == 1 {
            break;
        }
    }
    g
}

/// Little-endian value of the first `w` bytes of `c`, zero-extended.
fn read_le(c: &[u8], w: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf[..w].copy_from_slice(&c[..w]);
    u64::from_le_bytes(buf)
}

/// ClickHouse `Delta(width)` on one block (CompressionCodecDelta.cpp): header
/// `[width, bytes_to_skip]`, the `len % width` leading bytes verbatim, then each value minus the
/// previous one (the first minus 0), wrapping at the width. Widths other than 1, 2, 4, 8 leave
/// the block unchanged.
fn delta_transform(block: &[u8], width: usize) -> Cow<'_, [u8]> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Cow::Borrowed(block);
    }
    let skip = block.len() % width;
    let mut out = Vec::with_capacity(block.len() + 2);
    out.extend_from_slice(&[width as u8, skip as u8]);
    out.extend_from_slice(&block[..skip]);
    let mut prev = 0u64;
    for c in block[skip..].chunks_exact(width) {
        let v = read_le(c, width);
        out.extend_from_slice(&v.wrapping_sub(prev).to_le_bytes()[..width]);
        prev = v;
    }
    Cow::Owned(out)
}

/// MSB-first bit stream (ClickHouse `BitWriter`: first bit written = top bit of the first
/// byte; `flush` pads the last byte with zeros).
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    /// The low `bits` (0..=64) of `v`.
    fn write(&mut self, bits: u32, v: u64) {
        for i in (0..bits).rev() {
            self.acc = self.acc << 1 | (v >> i & 1);
            self.n += 1;
            if self.n == 8 {
                self.out.push(self.acc as u8);
                (self.acc, self.n) = (0, 0);
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push((self.acc << (8 - self.n)) as u8);
        }
        self.out
    }
}

/// (prefix bits, prefix, data bits incl. the sign bit) of a double-delta, `getDeltaWriteSpec`.
fn dd_spec(v: i64) -> (u32, u64, u32) {
    if v > -63 && v < 64 {
        (2, 0b10, 7)
    } else if v > -255 && v < 256 {
        (3, 0b110, 9)
    } else if v > -2047 && v < 2048 {
        (4, 0b1110, 12)
    } else if v > i64::from(i32::MIN) && v < i64::from(i32::MAX) {
        (5, 0b11110, 32)
    } else {
        (5, 0b11111, 64)
    }
}

/// ClickHouse `DoubleDelta` on one block (CompressionCodecDoubleDelta.cpp): header
/// `[width, bytes_to_skip]`, the leading `len % width` bytes verbatim, a UInt32 item count, the
/// first value and the first delta (`width` bytes each; absent when the block has fewer than
/// 1 / 2 items), then per further item a bit-packed double-delta: `0` for zero, else a prefix
/// (`10`, `110`, `1110`, `11110`, `11111` for 7 / 9 / 12 / 32 / 64 data bits), a sign bit
/// (1 = negative) and `data_bits - 1` bits of `|dd| - 1` (the width's minimum: `2^(8w-1) - 1`),
/// MSB first, the last byte zero-padded.
fn double_delta_transform(block: &[u8], width: usize) -> Cow<'_, [u8]> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Cow::Borrowed(block);
    }
    let skip = block.len() % width;
    let items = (block.len() - skip) / width;
    let mask = if width == 8 {
        u64::MAX
    } else {
        (1u64 << (8 * width)) - 1
    };
    let mut out = Vec::with_capacity(block.len() / 4 + 16);
    out.extend_from_slice(&[width as u8, skip as u8]);
    out.extend_from_slice(&block[..skip]);
    out.extend_from_slice(&(items as u32).to_le_bytes());
    let mut vals = block[skip..].chunks_exact(width).map(|c| read_le(c, width));
    let Some(first) = vals.next() else {
        return Cow::Owned(out);
    };
    out.extend_from_slice(&first.to_le_bytes()[..width]);
    let Some(second) = vals.next() else {
        return Cow::Owned(out);
    };
    let mut prev_delta = second.wrapping_sub(first) & mask;
    out.extend_from_slice(&prev_delta.to_le_bytes()[..width]);
    let mut prev = second;
    let shift = 64 - 8 * width as u32;
    let mut w = BitWriter { out, acc: 0, n: 0 };
    for v in vals {
        let delta = v.wrapping_sub(prev) & mask;
        let dd = delta.wrapping_sub(prev_delta) & mask;
        (prev_delta, prev) = (delta, v);
        if dd == 0 {
            w.write(1, 0);
            continue;
        }
        let signed = ((dd << shift) as i64) >> shift;
        let abs = if dd == 1u64 << (8 * width as u32 - 1) {
            mask >> 1
        } else {
            signed.unsigned_abs() - 1
        };
        let (pb, prefix, data) = dd_spec(signed);
        w.write(pb, prefix);
        w.write(1, u64::from(signed < 0));
        w.write(data - 1, abs);
    }
    Cow::Owned(w.finish())
}

/// A compression block of a two-codec chain is a `CompressionCodecMultiple` payload: a codec
/// count byte, one method byte per codec, and the compressor's own 9-byte block header (kept
/// inside the payload): 12 bytes on top of the outer `BLOCK_HEADER`. The transform's output
/// carries its own 9-byte block header (method, compressed size, decompressed size) into the
/// compressor. Measured against ClickHouse 26.8 (spec 2026-10-23 §7).
const MULTIPLE_HEADER: u64 = 3 + 9;
// Method bytes (CompressionMethodByte.h); only their entropy matters here.
const DELTA_METHOD: u8 = 0x92;
const DOUBLE_DELTA_METHOD: u8 = 0x94;
const GCD_METHOD: u8 = 0x9a;
const INNER_BLOCK_HEADER: usize = 9;

/// The GCD codec's block body as written by ClickHouse: header `[width, bytes_to_skip]`, the
/// GCD (`width` bytes; 1 when the values have none), then the quotients. Widths other than
/// 1, 2, 4, 8 keep the bare `gcd_transform`.
fn gcd_body(block: &[u8], width: usize, signed: bool) -> Cow<'_, [u8]> {
    let t = gcd_transform(block, width, signed);
    if !matches!(width, 1 | 2 | 4 | 8) {
        return t;
    }
    let mut out = Vec::with_capacity(t.len() + 2 + width);
    out.extend_from_slice(&[width as u8, 0]);
    if matches!(t, Cow::Borrowed(_)) {
        out.extend_from_slice(&1u128.to_le_bytes()[..width]);
    }
    out.extend_from_slice(&t);
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
        Compression::None => Ok(bytes.len() as u64),
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
            let mut extra = 0;
            let data = match (s.int_width, chain.transform) {
                (
                    Some((w, signed)),
                    t @ (Transform::Gcd | Transform::Delta | Transform::DoubleDelta),
                ) => {
                    let (body, method) = match t {
                        Transform::Gcd => (gcd_body(raw, w, signed), GCD_METHOD),
                        Transform::Delta => (delta_transform(raw, w), DELTA_METHOD),
                        _ => (double_delta_transform(raw, w), DOUBLE_DELTA_METHOD),
                    };
                    if chain.compression == Compression::None {
                        body
                    } else {
                        extra = MULTIPLE_HEADER;
                        let mut v = Vec::with_capacity(INNER_BLOCK_HEADER + body.len());
                        v.push(method);
                        v.extend_from_slice(
                            &((INNER_BLOCK_HEADER + body.len()) as u32).to_le_bytes(),
                        );
                        v.extend_from_slice(&(raw.len() as u32).to_le_bytes());
                        v.extend_from_slice(&body);
                        Cow::Owned(v)
                    }
                }
                _ => Cow::Borrowed(raw),
            };
            let mut h = [0u64; 256];
            for &x in data.iter() {
                h[x as usize] += 1;
            }
            Ok((
                BLOCK_HEADER + extra + compressed_len(&data, chain.compression)?,
                h,
            ))
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

// ─────────────────────────────────────────────────────────────────────────────
// Skip indexes (chbench R37)
// ─────────────────────────────────────────────────────────────────────────────

/// Bloom filter bits per distinct value of an index block, by false-positive rate (chbench R37).
pub(crate) const BLOOM_BITS_PER_VALUE: [(f64, u64); 4] =
    [(0.001, 15), (0.01, 10), (0.025, 8), (0.05, 7)];

/// A skip index's kind for size emulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum IndexKind {
    MinMax,
    Set { max_size: u64 },
    Bloom { fpr: f64 },
}

/// How a value's bytes order (minmax).
#[derive(Clone, Copy)]
enum Order {
    Unsigned,
    Signed,
    Float,
    /// String bytes after the varint, or the FixedString bytes.
    Bytes,
}

/// The column's value stream and its row accessors, for the index builders.
struct Rows {
    st: Stream,
    order: Order,
    /// String layout: each row's slice starts with a varint length.
    varint: bool,
    default: Vec<u8>,
}

impl Rows {
    fn new(a: &dyn Array, layout: &ChLayout) -> Result<Self, String> {
        const NESTED: &str = "skip indexes on nested types are not modelled";
        let inner = match layout {
            ChLayout::Nullable(i) => i.as_ref(),
            l => l,
        };
        let (base, order, varint) = match inner {
            ChLayout::Fixed { signed, .. } => {
                let o = match a.data_type() {
                    AT::Float16 | AT::Float32 | AT::Float64 => Order::Float,
                    _ if *signed => Order::Signed,
                    _ => Order::Unsigned,
                };
                (inner.clone(), o, false)
            }
            ChLayout::String | ChLayout::LowCardinality => (ChLayout::String, Order::Bytes, true),
            ChLayout::FixedString(_) => (inner.clone(), Order::Bytes, false),
            _ => return Err(NESTED.to_string()),
        };
        let st = streams(a, &base)?.pop().ok_or(NESTED)?;
        let default = if varint {
            vec![0]
        } else {
            vec![0; st.row_ends.first().copied().unwrap_or(0) as usize]
        };
        Ok(Self {
            st,
            order,
            varint,
            default,
        })
    }

    /// Row `i`'s bytes as written to an index.
    fn written(&self, i: usize) -> &[u8] {
        let a = if i == 0 { 0 } else { self.st.row_ends[i - 1] } as usize;
        &self.st.bytes[a..self.st.row_ends[i] as usize]
    }

    /// Row `i`'s bytes as compared.
    fn value(&self, i: usize) -> &[u8] {
        let w = self.written(i);
        if self.varint {
            let skip = w
                .iter()
                .position(|b| b & 0x80 == 0)
                .map_or(w.len(), |p| p + 1);
            &w[skip..]
        } else {
            w
        }
    }

    fn key(&self, i: usize) -> u128 {
        let v = self.value(i);
        let bits = v.iter().rev().fold(0u128, |a, &b| a << 8 | u128::from(b));
        let sign = 1u128 << (8 * v.len() - 1);
        match self.order {
            Order::Unsigned | Order::Bytes => bits,
            Order::Signed => bits ^ sign,
            Order::Float if bits & sign != 0 => !bits & (sign | (sign - 1)),
            Order::Float => bits | sign,
        }
    }

    fn less(&self, i: usize, j: usize) -> bool {
        match self.order {
            Order::Bytes => self.value(i) < self.value(j),
            _ => self.key(i) < self.key(j),
        }
    }
}

fn index_blocks(n: usize, block_rows: u64) -> Result<Vec<std::ops::Range<usize>>, String> {
    if block_rows == 0 {
        return Err("index block of zero rows".to_string());
    }
    let b = block_rows as usize;
    Ok((0..n.div_ceil(b))
        .map(|k| k * b..((k + 1) * b).min(n))
        .collect())
}

/// One stream, a row per index block.
fn block_stream(parts: Vec<Vec<u8>>) -> Stream {
    let mut bytes = Vec::new();
    let mut ends = Vec::with_capacity(parts.len());
    for p in parts {
        bytes.extend_from_slice(&p);
        ends.push(bytes.len() as u64);
    }
    stream(bytes, ends)
}

/// The uncompressed minmax index: per block the min and max of the non-null values (defaults
/// for an all-null block), plus one byte per block when the column is nullable.
fn minmax_stream(a: &dyn Array, layout: &ChLayout, block_rows: u64) -> Result<Stream, String> {
    let rows = Rows::new(a, layout)?;
    let nullable = matches!(layout, ChLayout::Nullable(_));
    let parts = index_blocks(a.len(), block_rows)?
        .into_iter()
        .map(|r| {
            let mut ext: Option<(usize, usize)> = None;
            for i in r.filter(|&i| a.is_valid(i)) {
                let (lo, hi) = ext.get_or_insert((i, i));
                if rows.less(i, *lo) {
                    *lo = i;
                }
                if rows.less(*hi, i) {
                    *hi = i;
                }
            }
            let mut out = Vec::new();
            match ext {
                Some((lo, hi)) => {
                    out.extend_from_slice(rows.written(lo));
                    out.extend_from_slice(rows.written(hi));
                }
                None => {
                    out.extend_from_slice(&rows.default);
                    out.extend_from_slice(&rows.default);
                }
            }
            if nullable {
                out.push(0);
            }
            out
        })
        .collect();
    Ok(block_stream(parts))
}

/// A block's distinct non-null values (first-seen order, as written) and whether it holds a null.
fn block_distinct<'a>(
    rows: &'a Rows,
    a: &dyn Array,
    r: std::ops::Range<usize>,
) -> (Vec<&'a [u8]>, bool) {
    let mut seen = std::collections::HashSet::with_hasher(foldhash::fast::RandomState::default());
    let (mut out, mut null) = (Vec::new(), false);
    for i in r {
        if a.is_null(i) {
            null = true;
        } else if seen.insert(rows.value(i)) {
            out.push(rows.written(i));
        }
    }
    (out, null)
}

/// The uncompressed set(`max_size`) index: per block an 8-byte count and, unless the block
/// overflows (distinct values + NULL > `max_size`; 0 = unlimited), its distinct values.
fn set_stream(
    a: &dyn Array,
    layout: &ChLayout,
    max_size: u64,
    block_rows: u64,
) -> Result<Stream, String> {
    let rows = Rows::new(a, layout)?;
    let parts = index_blocks(a.len(), block_rows)?
        .into_par_iter()
        .map(|r| {
            let (vals, null) = block_distinct(&rows, a, r);
            let entries = vals.len() as u64 + u64::from(null);
            let mut out = entries.to_le_bytes().to_vec();
            if max_size == 0 || entries <= max_size {
                vals.iter().for_each(|v| out.extend_from_slice(v));
            }
            out
        })
        .collect();
    Ok(block_stream(parts))
}

/// Emulated bytes of a skip index on `a` (the column in its recommended Arrow type, ClickHouse
/// physical values per `layout`) with index blocks of `block_rows` rows (G × 8 192); Err for an
/// fpr not in `BLOOM_BITS_PER_VALUE`. minmax and set: the per-block entries concatenated and
/// LZ4-compressed like a column stream; bloom_filter: ⌈D·b/8⌉ + v(D) bytes per block,
/// uncompressed (random bits).
pub(crate) fn skip_index_bytes(
    a: &dyn Array,
    layout: &ChLayout,
    kind: IndexKind,
    block_rows: u64,
) -> Result<u64, String> {
    let lz4 = Chain {
        transform: Transform::None,
        compression: Compression::Lz4,
    };
    match kind {
        IndexKind::MinMax => Ok(emulate(&[minmax_stream(a, layout, block_rows)?], lz4)?.bytes),
        IndexKind::Set { max_size } => {
            Ok(emulate(&[set_stream(a, layout, max_size, block_rows)?], lz4)?.bytes)
        }
        IndexKind::Bloom { fpr } => {
            let bits = BLOOM_BITS_PER_VALUE
                .iter()
                .find(|(f, _)| *f == fpr)
                .ok_or_else(|| {
                    format!(
                        "bloom_filter fpr {fpr} not measured (chbench R37: 0.001, 0.01, 0.025, 0.05)"
                    )
                })?
                .1;
            let rows = Rows::new(a, layout)?;
            Ok(index_blocks(a.len(), block_rows)?
                .into_par_iter()
                .map(|r| {
                    let d = block_distinct(&rows, a, r).0.len() as u64;
                    let v = if d < 128 {
                        1
                    } else if d < 16_384 {
                        2
                    } else {
                        3
                    };
                    (d * bits).div_ceil(8) + v
                })
                .sum())
        }
    }
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
            int_codecs: true,
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
        assert_eq!(s[1].int_width, Some((4, true)));
        assert_eq!(s[0].int_width, None);
    }

    #[test]
    fn time32_widens_to_eight_bytes() {
        let a = Time32MillisecondArray::from(vec![1500, -1]);
        let l = ChLayout::Fixed {
            width: 8,
            signed: true,
            int_codecs: false,
        };
        let s = streams(&a, &l).unwrap();
        assert_eq!(s[0].bytes[..8], 1500i64.to_le_bytes());
        assert_eq!(s[0].bytes[8..], (-1i64).to_le_bytes());
        assert_eq!(s[0].int_width, None);
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
    fn gcd_body_header_and_chain_overhead() {
        // [width, skip], the GCD (1 when none), then the values
        assert_eq!(gcd_body(&[2, 3], 1, false).as_ref(), [1, 0, 1, 2, 3]);
        assert_eq!(gcd_body(&[10, 20], 1, false).as_ref(), [1, 0, 10, 1, 2]);
        // measured on ClickHouse 26.8 (spec 2026-10-23 §7): codec alone = outer header + body;
        // two codecs add the Multiple header (3) + the compressor's own 9-byte header
        let mut s = plain(vec![0u8; 8], 1);
        s.int_width = Some((1, false));
        let c = |transform, compression| Chain {
            transform,
            compression,
        };
        let one = emulate(&[s], c(Transform::Delta, Compression::None)).unwrap();
        assert_eq!(one.bytes, BLOCK_HEADER + 2 + 8);
        let mut s = plain(vec![0u8; 8], 1);
        s.int_width = Some((1, false));
        let two = emulate(&[s], c(Transform::Delta, Compression::Lz4)).unwrap();
        assert!(two.bytes >= BLOCK_HEADER + MULTIPLE_HEADER);
    }

    #[test]
    fn histogram_and_sizes() {
        let lz4 = Chain {
            transform: Transform::None,
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
        // The histogram counts what the compressor sees: 9 + 2 header bytes, GCD 1000 = E8 03, then 1, 2, 3.
        let mut s = plain(i64s(&[1000, 2000, 3000]), 8);
        s.int_width = Some((8, true));
        let e = emulate(
            &[s],
            Chain {
                transform: Transform::Gcd,
                ..lz4
            },
        )
        .unwrap();
        assert_eq!(e.histogram.iter().sum::<u64>(), 9 + 2 + 32);
        assert_eq!(
            (e.histogram[0xE8], e.histogram[3], e.histogram[1]),
            (1, 2, 1)
        );
        let z = Chain {
            transform: Transform::None,
            compression: Compression::Zstd(3),
        };
        assert!(emulate(&[plain(vec![1; 100], 1)], z).unwrap().bytes > BLOCK_HEADER);
    }

    #[test]
    fn chain_ddl() {
        use Compression as C;
        use Transform as T;
        let d = |transform, compression| {
            Chain {
                transform,
                compression,
            }
            .ddl()
        };
        assert_eq!(d(T::None, C::Lz4), "LZ4");
        assert_eq!(d(T::None, C::Zstd(3)), "ZSTD(3)");
        assert_eq!(d(T::Gcd, C::Lz4), "GCD, LZ4");
        assert_eq!(d(T::Gcd, C::Zstd(3)), "GCD, ZSTD(3)");
        assert_eq!(d(T::Delta, C::Lz4), "Delta, LZ4");
        assert_eq!(d(T::Delta, C::Zstd(1)), "Delta, ZSTD(1)");
        assert_eq!(d(T::DoubleDelta, C::Lz4), "DoubleDelta, LZ4");
        assert_eq!(d(T::DoubleDelta, C::Zstd(9)), "DoubleDelta, ZSTD(9)");
        assert_eq!(d(T::Gcd, C::None), "GCD");
        assert_eq!(d(T::Delta, C::None), "Delta");
        assert_eq!(d(T::DoubleDelta, C::None), "DoubleDelta");
        assert_eq!(d(T::None, C::None), "NONE");
    }

    #[test]
    fn delta_transform_cases() {
        let u32s = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let input = u32s(&[1, 3, 6]);
        let t = delta_transform(&input, 4);
        assert_eq!(t.as_ref(), [&[4u8, 0][..], &u32s(&[1, 2, 3])].concat());
        // a 2-byte remainder is copied right after the header
        let mut b = vec![9u8, 8];
        b.extend(u32s(&[1, 3]));
        b.truncate(10);
        assert_eq!(b.len() % 4, 2);
        let t = delta_transform(&b, 4);
        assert_eq!(t[..4], [4, 2, 9, 8]);
        assert_eq!(t[4..], u32s(&[1, 2])[..]);
        // wraps at the width
        assert_eq!(delta_transform(&[250, 5], 1).as_ref(), [1, 0, 250, 11]);
        // widths outside 1/2/4/8 pass through
        assert_eq!(delta_transform(&[1, 2, 3], 3).as_ref(), [1, 2, 3]);
    }

    /// The decoder of CompressionCodecDoubleDelta (decompressDataForType), bit by bit.
    fn double_delta_decode(enc: &[u8]) -> Vec<u8> {
        let width = enc[0] as usize;
        let skip = enc[1] as usize;
        let mut out = enc[2..2 + skip].to_vec();
        let mut pos = 2 + skip;
        let items = u32::from_le_bytes(enc[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mask = if width == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * width)) - 1
        };
        if items == 0 {
            return out;
        }
        let first = read_le(&enc[pos..], width);
        pos += width;
        out.extend_from_slice(&first.to_le_bytes()[..width]);
        if items == 1 {
            return out;
        }
        let mut delta = read_le(&enc[pos..], width);
        pos += width;
        let mut value = first.wrapping_add(delta) & mask;
        out.extend_from_slice(&value.to_le_bytes()[..width]);
        let bits = &enc[pos..];
        let mut at = 0usize;
        let mut take = |n: u32| -> u64 {
            let mut v = 0u64;
            for _ in 0..n {
                v = v << 1 | u64::from(bits[at / 8] >> (7 - at % 8) & 1);
                at += 1;
            }
            v
        };
        for _ in 2..items {
            let mut ones = 0;
            while ones < 5 && take(1) == 1 {
                ones += 1;
            }
            let data = [0, 7, 9, 12, 32, 64][ones];
            let dd = if data == 0 {
                0
            } else {
                let neg = take(1) == 1;
                let m = take(data - 1) + 1;
                (if neg { m.wrapping_neg() } else { m }) & mask
            };
            delta = delta.wrapping_add(dd) & mask;
            value = value.wrapping_add(delta) & mask;
            out.extend_from_slice(&value.to_le_bytes()[..width]);
        }
        out
    }

    #[test]
    fn double_delta_sizes_and_roundtrip() {
        let u64s = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let regular: Vec<u64> = (0..1000).map(|i| 1_000_000 + 60 * i).collect();
        let input = u64s(&regular);
        let t = double_delta_transform(&input, 8);
        assert_eq!(t.len(), 2 + 4 + 8 + 8 + 998usize.div_ceil(8));
        assert_eq!(double_delta_decode(&t), u64s(&regular));
        // degenerate blocks
        for n in 0..4usize {
            let v: Vec<u64> = (0..n as u64).map(|i| i * i).collect();
            let input = u64s(&v);
            let t = double_delta_transform(&input, 8);
            assert_eq!(double_delta_decode(&t), u64s(&v), "{n}");
        }
        // width 1, values 0, 1, 3, 3: d = 1, 2, 0 so dd = 1 ("10" 0 000000), -2 ("10" 1 000001)
        let t = double_delta_transform(&[0, 1, 3, 3], 1);
        assert_eq!(t[..], [1, 0, 4, 0, 0, 0, 0, 1, 0x80, 0x50, 0x40]);
        // pseudo-random blocks at every width, with extreme steps and unaligned tails
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for case in 0..200usize {
            let width = [1usize, 2, 4, 8][case % 4];
            let n = (next() % 300) as usize + (case % 7);
            let extra = (next() % 3) as usize % width;
            let top = 1u64 << (8 * width - 1);
            let mut b = Vec::new();
            for i in 0..n {
                let r = next();
                let v = match (case / 4) % 4 {
                    0 => r,
                    1 => (i as u64).wrapping_mul(r % 5),
                    2 => [0, u64::MAX, top, top - 1][(r % 4) as usize],
                    _ => (i as u64) * 7 + r % 3,
                };
                b.extend_from_slice(&v.to_le_bytes()[..width]);
            }
            b.extend((0..extra).map(|_| next() as u8));
            let t = double_delta_transform(&b, width);
            assert_eq!(double_delta_decode(&t), b, "case {case}");
        }
    }

    fn ints(f: impl Fn(usize) -> i64, n: usize) -> Int64Array {
        Int64Array::from((0..n).map(f).collect::<Vec<_>>())
    }

    #[test]
    fn minmax_index_stream_and_size() {
        let a = ints(|i| (i / 8192) as i64 * 100 + (i % 8192) as i64, 4 * 8192);
        let s = minmax_stream(&a, &fx(8, true), 8192).unwrap();
        assert_eq!(s.bytes.len(), 4 * 16);
        assert_eq!(s.bytes[..8], 0i64.to_le_bytes());
        assert_eq!(s.bytes[8..16], 8191i64.to_le_bytes());
        assert_eq!(s.bytes[16..24], 100i64.to_le_bytes());
        let b = skip_index_bytes(&a, &fx(8, true), IndexKind::MinMax, 8192).unwrap();
        assert!(b > BLOCK_HEADER && b < BLOCK_HEADER + 64 + 16, "{b}");
        // negatives order below positives; nullable adds a byte; all-null block = defaults
        let a = Int64Array::from(vec![Some(5), Some(-3), None, None]);
        let s = minmax_stream(&a, &nullable(fx(8, true)), 2).unwrap();
        assert_eq!(s.bytes.len(), 2 * 17);
        assert_eq!(s.bytes[..8], (-3i64).to_le_bytes());
        assert_eq!(s.bytes[8..16], 5i64.to_le_bytes());
        assert_eq!(s.bytes[17..33], [0u8; 16]);
        let f = Float64Array::from(vec![1.5, -2.5, 0.0]);
        let s = minmax_stream(&f, &fx(8, false), 8192).unwrap();
        assert_eq!(s.bytes[..8], (-2.5f64).to_le_bytes());
        assert_eq!(s.bytes[8..], 1.5f64.to_le_bytes());
        let t = StringArray::from(vec!["b", "abc", "c"]);
        let s = minmax_stream(&t, &ChLayout::String, 8192).unwrap();
        assert_eq!(s.bytes, [3, b'a', b'b', b'c', 1, b'c']);
    }

    #[test]
    fn set_index_stream() {
        let a = Int32Array::from((0..4 * 8192).map(|i| i % 3).collect::<Vec<_>>());
        let s = set_stream(&a, &fx(4, true), 125, 8192).unwrap();
        assert_eq!(s.bytes.len(), 4 * (8 + 3 * 4));
        assert_eq!(s.bytes[..8], 3u64.to_le_bytes());
        let a = Int32Array::from((0..4 * 8192).map(|i| i % 200).collect::<Vec<_>>());
        let s = set_stream(&a, &fx(4, true), 125, 8192).unwrap();
        assert_eq!(s.bytes.len(), 4 * 8);
        assert_eq!(s.bytes[..8], 200u64.to_le_bytes());
        // NULL counts toward the limit but is not written
        let a = Int32Array::from(vec![Some(1), Some(2), None]);
        let l = nullable(fx(4, true));
        assert_eq!(set_stream(&a, &l, 2, 8192).unwrap().bytes.len(), 8);
        assert_eq!(set_stream(&a, &l, 3, 8192).unwrap().bytes.len(), 16);
        let t = StringArray::from(vec!["x", "x", "yz"]);
        let s = set_stream(&t, &ChLayout::String, 0, 8192).unwrap();
        assert_eq!(s.bytes[8..], [1, b'x', 2, b'y', b'z']);
        let k = IndexKind::Set { max_size: 3 };
        assert!(skip_index_bytes(&a, &l, k, 8192).is_ok());
    }

    #[test]
    fn bloom_index_bytes() {
        let k = IndexKind::Bloom { fpr: 0.025 };
        let l = fx(8, true);
        let b = |d: i64, n: usize| skip_index_bytes(&ints(|i| i as i64 % d, n), &l, k, 8192);
        assert_eq!(b(10, 4 * 8192), Ok(4 * 11));
        assert_eq!(b(200, 4 * 8192), Ok(4 * 202));
        assert_eq!(b(1 << 40, 8192), Ok(8194));
        // partial last block counted with its own D
        assert_eq!(b(10, 8192 + 5), Ok(11 + 6));
        let e = skip_index_bytes(
            &ints(|i| i as i64, 4),
            &l,
            IndexKind::Bloom { fpr: 0.5 },
            8192,
        )
        .unwrap_err();
        assert!(e.contains("fpr 0.5 not measured"), "{e}");
    }

    #[test]
    fn skip_index_rejects_nested() {
        let a = ListArray::from_iter_primitive::<Int8Type, _, _>(vec![Some(vec![Some(1)])]);
        let l = ChLayout::Array(Box::new(fx(1, true)));
        let e = skip_index_bytes(&a, &l, IndexKind::MinMax, 8192).unwrap_err();
        assert_eq!(e, "skip indexes on nested types are not modelled");
    }
}
