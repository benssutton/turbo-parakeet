//! Mergeable per-level statistics for the streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md §4). Each batch
//! is profiled per level (a column, or a list's inner values) into a `BatchStats` with
//! the describe kernels; batches are absorbed in stream order into a `LevelStats`,
//! which finishes into the `Profile` the recommenders/engine rules read.

use std::collections::HashMap;

use arrow_array::{ArrayRef, UInt64Array};
use arrow_schema::DataType as AT;
use polars::prelude::*;

use crate::common::arrow_io::export_series;
use crate::common::encode_series;
use crate::common::ipc_sizes::{classic_layout, ipc_body_bytes};
use crate::common::text::render_value;
use crate::recommenders::engine::{
    body_size, is_text, to_polars_layout, Shape, VIEW_BLOCK, VIEW_MAX_BLOCK,
};
use crate::recommenders::streaming::distinct_sample::{sample_size, DistinctSample};
use crate::techniques::describe::{
    arg_extremes, byte_lengths, float_stats, frequency_map, frequency_map_below, lengths,
    n_midnight, strings, FloatStats, Frequencies, Profile, Range, StringStats,
};
use crate::techniques::hll::{hash_key, Hll};

/// HyperLogLog precision: 2^14 registers, 16 KB, relative standard error ≈ 0.8%.
const HLL_P: u8 = 14;

fn pad8(x: u64) -> u64 {
    x.next_multiple_of(8)
}

fn opt<T>(a: Option<T>, b: Option<T>, f: impl FnOnce(T, T) -> T) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b)),
        (a, b) => a.or(b),
    }
}

fn gcd128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i128
}

/// Copy of `len` rows from `start` that shares no buffer with `a`, so it pins none of
/// the source batch. `take` copies most types, but reuses every data buffer of a
/// Utf8View / BinaryView and the whole values array of a dictionary; those are
/// compacted here, at any depth (list, fixed-size list, map and struct children).
/// The copy has `a`'s data type at every depth: a dictionary keeps its key and value
/// types, holding only the values its rows use (first-use order, keys renumbered), so
/// the copy imports under `a`'s original field.
pub(crate) fn copy_rows(a: &ArrayRef, start: usize, len: usize) -> Result<ArrayRef, String> {
    let idx = UInt64Array::from_iter_values((start..start + len).map(|i| i as u64));
    arrow_select::take::take(a.as_ref(), &idx, None)
        .and_then(compact)
        .map_err(|e| e.to_string())
}

/// `a` (a `take` result) with no buffer shared with its source; see `copy_rows`.
fn compact(a: ArrayRef) -> Result<ArrayRef, arrow_schema::ArrowError> {
    use arrow_array::cast::AsArray;
    use arrow_array::{
        Array, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray,
    };
    use std::sync::Arc;
    Ok(match a.data_type() {
        AT::Utf8View => Arc::new(a.as_string_view().gc()),
        AT::BinaryView => Arc::new(a.as_binary_view().gc()),
        AT::Dictionary(key_type, _) => {
            // Keep only the values the rows use, first-use order; keys renumbered to
            // match. There are no more of them than before, so they fit the key type.
            use arrow_array::types::UInt64Type;
            let d = a.as_any_dictionary();
            let keys = arrow_cast::cast(d.keys(), &AT::UInt64)?;
            let mut slot: HashMap<u64, u64> = HashMap::new();
            let mut used = Vec::new();
            let renumbered: Vec<u64> = keys
                .as_primitive::<UInt64Type>()
                .iter()
                .map(|k| {
                    k.map_or(0, |k| {
                        *slot.entry(k).or_insert_with(|| {
                            used.push(k);
                            used.len() as u64 - 1
                        })
                    })
                })
                .collect();
            let keys = arrow_array::UInt64Array::new(renumbered.into(), keys.nulls().cloned());
            let keys = arrow_cast::cast(&keys, key_type)?;
            let values = arrow_select::take::take(
                d.values().as_ref(),
                &arrow_array::UInt64Array::from(used),
                None,
            )?;
            let data = keys
                .to_data()
                .into_builder()
                .data_type(a.data_type().clone())
                .child_data(vec![compact(values)?.to_data()])
                .build()?;
            arrow_array::make_array(data)
        }
        AT::List(f) => {
            let l = a.as_list::<i32>();
            let values = compact(l.values().clone())?;
            Arc::new(ListArray::try_new(
                f.clone(),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        AT::LargeList(f) => {
            let l = a.as_list::<i64>();
            let values = compact(l.values().clone())?;
            Arc::new(LargeListArray::try_new(
                f.clone(),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        AT::FixedSizeList(f, size) => {
            let l = a.as_fixed_size_list();
            let values = compact(l.values().clone())?;
            Arc::new(FixedSizeListArray::try_new(
                f.clone(),
                *size,
                values,
                l.nulls().cloned(),
            )?)
        }
        AT::Map(f, sorted) => {
            let m = a.as_map();
            let entries = compact(Arc::new(m.entries().clone()))?;
            Arc::new(MapArray::try_new(
                f.clone(),
                m.offsets().clone(),
                entries.as_struct().clone(),
                m.nulls().cloned(),
                *sorted,
            )?)
        }
        AT::Struct(fields) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .cloned()
                .map(compact)
                .collect::<Result<_, _>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                columns,
                s.nulls().cloned(),
            )?)
        }
        _ => a,
    })
}

/// Polars' view-array data blocks (recommenders/engine/polars_layout.rs `polars_views`) replayed on value
/// lengths alone: values of ≤ 12 bytes are inline; longer ones fill blocks whose
/// capacity doubles from 8 KiB to 16 MiB (or grows to fit one value). `bytes` is the
/// IPC body of those blocks, each padded to 8.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ViewSim {
    closed: u64,
    current: u64,
    capacity: u64,
}

impl ViewSim {
    pub(crate) fn push(&mut self, len: u64) {
        if len <= 12 {
            return;
        }
        if self.capacity < self.current + len {
            if self.current > 0 {
                self.closed += pad8(self.current);
                self.current = 0;
            }
            self.capacity = (self.capacity * 2)
                .clamp(VIEW_BLOCK as u64, VIEW_MAX_BLOCK as u64)
                .max(len);
        }
        self.current += len;
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.closed + pad8(self.current)
    }
}

/// Ordering key of an extreme: the physical integer (integers, decimals, temporals,
/// booleans, Enum codes), the float, or the bytes (strings, Categorical values, binary).
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub(crate) enum Key {
    I(i128),
    F(f64),
    S(Vec<u8>),
}

/// A running extreme: its key, and the value itself (one row, classic layout) to render.
#[derive(Clone, Debug)]
pub(crate) struct Ext {
    pub key: Key,
    pub value: ArrayRef,
}

/// Integer-backed numeric and temporal dtypes: those whose integer range (`int_range`)
/// the recommendation rules read.
pub(crate) fn has_int_range(dt: &DataType) -> bool {
    dt.is_integer()
        || matches!(
            dt,
            DataType::Decimal(..)
                | DataType::Date
                | DataType::Datetime(..)
                | DataType::Duration(_)
                | DataType::Time
        )
}

/// Numeric dtypes: those whose extremes feed the conclusions' whole-number range
/// (`Profile::numeric`). Temporal dtypes never do.
pub(crate) fn is_numeric(dt: &DataType) -> bool {
    dt.is_integer() || dt.is_float() || matches!(dt, DataType::Decimal(..))
}

/// Dtypes whose min / max the rules read (as integers or floats) and the output renders.
fn has_extremes(dt: &DataType) -> bool {
    dt.is_integer()
        || dt.is_float()
        || matches!(
            dt,
            DataType::Decimal(..)
                | DataType::Date
                | DataType::Datetime(..)
                | DataType::Duration(_)
                | DataType::Time
                | DataType::String
                | DataType::Categorical(..)
                | DataType::Enum(..)
                | DataType::Boolean
                | DataType::Binary
        )
}

fn ext_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<Ext>> {
    let Some(i) = i else { return Ok(None) };
    let one = s.slice(i as i64, 1);
    let key = match one.dtype() {
        DataType::String | DataType::Categorical(..) => one
            .cast(&DataType::String)?
            .str()?
            .get(0)
            .map(|v| Key::S(v.as_bytes().to_vec())),
        DataType::Binary => one.binary()?.get(0).map(|v| Key::S(v.to_vec())),
        DataType::Boolean => one.bool()?.get(0).map(|v| Key::I(v as i128)),
        dt if dt.is_float() => one.cast(&DataType::Float64)?.f64()?.get(0).map(Key::F),
        _ => one
            .to_physical_repr()
            .cast(&DataType::Int128)?
            .i128()?
            .get(0)
            .map(Key::I),
    };
    // A dictionary extreme would export (and retain) its whole category mapping.
    let value = match one.dtype() {
        DataType::Categorical(..) | DataType::Enum(..) => {
            classic_layout(&one.cast(&DataType::String)?)?
        }
        _ => classic_layout(&one)?,
    };
    Ok(key.map(|key| Ext { key, value }))
}

/// A distinct value's statistics in one batch.
pub(crate) struct KeyStat {
    pub key: u64,
    /// `hll::hash_key(key)`: the sample's order and the sketch's input.
    pub hash: u64,
    pub first: u64,
    pub count: u64,
    pub mask: u8,
    pub len: u64,
    /// The value, when the batch holds at most five distinct values.
    pub text: Option<String>,
}

/// A text level's values in one batch, read at a value's first row when the distinct sample
/// admits it (exact phase only; spec 2026-10-07 §4).
pub(crate) struct TextSource {
    values: Series,
    /// The level's global index of `values`' first row.
    offset: u64,
}

impl TextSource {
    pub(crate) fn new(values: Series, offset: u64) -> Self {
        TextSource { values, offset }
    }

    /// The value at global index `first`, as text (one-row slice; Categorical / Enum by label).
    pub(crate) fn at(&self, first: u64) -> Option<String> {
        let row = i64::try_from(first.checked_sub(self.offset)?).ok()?;
        let one = self.values.slice(row, 1).cast(&DataType::String).ok()?;
        one.str().ok()?.get(0).map(str::to_owned)
    }
}

/// One batch's statistics of one level.
pub(crate) struct BatchStats {
    n: u64,
    n_null: u64,
    lo: Option<Ext>,
    hi: Option<Ext>,
    min_len: Option<u64>,
    max_len: Option<u64>,
    gcd: Option<i128>,
    sum_len: Option<u64>,
    floats: Option<FloatStats>,
    strings: Option<StringStats>,
    n_midnight: Option<u64>,
    /// Every distinct value, first-occurrence order.
    keys: Option<Vec<KeyStat>>,
    /// Text levels in the exact phase: where the sample reads newly admitted values' text.
    text_source: Option<TextSource>,
    /// HyperLogLog of the batch's distinct values (merged into the level's).
    sketch: Hll,
    /// Text / binary levels: lengths of the values over 12 bytes, row order.
    long_lens: Option<Vec<u64>>,
    /// Measured classic size, when the classic type has no analytic size.
    size_bytes: u64,
    polars_bytes: u64,
    /// Nested levels: the Polars-layout size of the classic layout rebuilt as
    /// one-shot's `to_polars_layout` rebuilds a list's kept inner level.
    rebuilt_polars_bytes: u64,
    classic: AT,
    is_f32: bool,
}

impl BatchStats {
    /// `s`: the level's values in one batch; `offset`: the level's global index of its
    /// first value; `top`: the level's distinct-sample top hash in the sampling phase
    /// (None while exact: then the keys' first-occurrence order matters).
    pub(crate) fn of(s: &Series, offset: u64, seed: u64, top: Option<u64>) -> PolarsResult<Self> {
        let lens = byte_lengths(s)?;
        let classic = export_series(&s.slice(0, 0), CompatLevel::oldest())?
            .data_type()
            .clone();
        let (size_bytes, rebuilt_polars_bytes) = if body_size(&classic, &Shape::default()).is_ok() {
            (0, 0)
        } else {
            let c = classic_layout(s)?;
            let rebuilt = match classic {
                AT::Struct(_) | AT::List(_) | AT::LargeList(_) | AT::FixedSizeList(..) => {
                    let p = to_polars_layout(&c, &AT::UInt32)
                        .map_err(|e| polars_err!(ComputeError: "{e}"))?;
                    ipc_body_bytes(p.as_ref(), None)?
                }
                _ => 0,
            };
            (ipc_body_bytes(c.as_ref(), None)?, rebuilt)
        };
        let (lo, hi) = if has_extremes(s.dtype()) {
            let (a, b) = arg_extremes(s)?;
            (ext_at(s, a)?, ext_at(s, b)?)
        } else {
            (None, None)
        };
        let enc = encode_series(s)?;
        // Sampling phase: only values hashing at or below the sample's top can still be
        // held or admitted (the top only falls); the sketch takes every row's hash.
        let (map, mut sketch) = match top {
            Some(t) => frequency_map_below(&enc, seed, offset, t, HLL_P),
            None => (frequency_map(&enc, seed, offset), Hll::new(HLL_P)),
        };
        // The values themselves only for text in the exact phase (`few`).
        let text = if top.is_none() && map.len() <= 5 && is_text(s.dtype()) {
            Some(s.cast(&DataType::String)?)
        } else {
            None
        };
        let mut keys: Vec<KeyStat> = Vec::with_capacity(map.len());
        for (key, e) in map {
            let hash = hash_key(key);
            if top.is_none() {
                sketch.insert(hash);
            }
            let row = (e.first - offset) as usize;
            keys.push(KeyStat {
                key,
                hash,
                first: e.first,
                count: e.count,
                mask: e.mask,
                len: lens.as_ref().map_or(0, |l| l[row]),
                text: text
                    .as_ref()
                    .and_then(|t| t.str().ok()?.get(row).map(str::to_owned)),
            });
        }
        // First-occurrence order feeds the exact phase's `few` and view blocks (text /
        // binary levels); the sample itself keeps the k smallest hashes in any order.
        if top.is_none() && lens.is_some() {
            keys.sort_unstable_by_key(|k| k.first);
        }
        let keys = Some(keys);
        let (min_len, max_len) = lengths(s, lens.as_deref())?;
        Ok(BatchStats {
            n: s.len() as u64,
            n_null: s.null_count() as u64,
            lo,
            hi,
            min_len,
            max_len,
            gcd: crate::techniques::gcd::series_gcd(s)?,
            sum_len: lens.as_ref().map(|l| l.iter().sum()),
            floats: float_stats(s)?,
            strings: strings(s, true)?,
            n_midnight: n_midnight(s)?,
            keys,
            text_source: (top.is_none() && is_text(s.dtype()))
                .then(|| TextSource::new(s.clone(), offset)),
            sketch,
            long_lens: lens.map(|l| l.into_iter().filter(|&x| x > 12).collect()),
            size_bytes,
            polars_bytes: ipc_body_bytes(export_series(s, CompatLevel::newest())?.as_ref(), None)?,
            rebuilt_polars_bytes,
            classic,
            is_f32: s.dtype() == &DataType::Float32,
        })
    }
}

/// A level's statistics over the stream so far.
#[derive(Clone, Default)]
pub(crate) struct LevelStats {
    /// Values at this level, nulls included (the column: every row of the stream).
    pub n: u64,
    pub n_null: u64,
    pub lo: Option<Ext>,
    pub hi: Option<Ext>,
    pub min_len: Option<u64>,
    pub max_len: Option<u64>,
    pub gcd: Option<i128>,
    pub sum_len: Option<u64>,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    pub n_midnight: Option<u64>,
    /// Bottom-k sample of the distinct values (every eligible dtype).
    pub sample: Option<DistinctSample>,
    /// HyperLogLog of the distinct values (p = 14).
    pub hll: Option<Hll>,
    /// All values' view blocks (Utf8View / BinaryView results), row order.
    pub views: Option<ViewSim>,
    /// Per-batch sums of the measured classic size (types with no analytic size) and
    /// of the Polars-layout size.
    pub size_bytes: u64,
    pub polars_bytes: u64,
    /// Per-batch sum of the rebuilt Polars-layout size (nested levels).
    pub rebuilt_polars_bytes: u64,
    pub classic: Option<AT>,
    pub is_f32: bool,
}

impl LevelStats {
    /// `rows` null values: a column absent from a batch, or backfilled when it appears.
    pub(crate) fn nulls(&mut self, rows: u64) {
        self.n += rows;
        self.n_null += rows;
    }

    pub(crate) fn absorb(&mut self, b: BatchStats, threshold: u64) {
        self.n += b.n;
        self.n_null += b.n_null;
        if b.lo
            .as_ref()
            .is_some_and(|x| self.lo.as_ref().is_none_or(|y| x.key < y.key))
        {
            self.lo = b.lo;
        }
        if b.hi
            .as_ref()
            .is_some_and(|x| self.hi.as_ref().is_none_or(|y| y.key < x.key))
        {
            self.hi = b.hi;
        }
        self.min_len = opt(self.min_len, b.min_len, u64::min);
        self.max_len = opt(self.max_len, b.max_len, u64::max);
        self.gcd = opt(self.gcd, b.gcd, gcd128);
        self.sum_len = opt(self.sum_len, b.sum_len, |a, b| a + b);
        self.floats = opt(self.floats, b.floats, FloatStats::merge);
        self.strings = opt(self.strings.take(), b.strings, StringStats::merge);
        self.n_midnight = opt(self.n_midnight, b.n_midnight, |a, b| a + b);
        if let Some(keys) = b.keys {
            self.hll
                .get_or_insert_with(|| Hll::new(HLL_P))
                .merge(&b.sketch);
            self.sample
                .get_or_insert_with(|| DistinctSample::new(sample_size(threshold)))
                .absorb_with_text(keys, b.text_source.as_ref());
        }
        if let Some(lens) = b.long_lens {
            let v = self.views.get_or_insert_with(Default::default);
            lens.into_iter().for_each(|l| v.push(l));
        }
        self.size_bytes += b.size_bytes;
        self.polars_bytes += b.polars_bytes;
        self.rebuilt_polars_bytes += b.rebuilt_polars_bytes;
        self.classic.get_or_insert(b.classic);
        self.is_f32 = b.is_f32;
    }

    /// The Profile the rules read: exact counts while the sample holds every distinct
    /// value; past it, the HyperLogLog count with the sample's f1 / f2 / capture history
    /// scaled to it (spec 2026-10-01 §3.2).
    pub(crate) fn profile(&self, dtype: &DataType) -> Profile {
        let text_len = self.sum_len.is_some();
        let (n_unique, hll, f1, f2, history, all_once, sum_len_unique) = match &self.sample {
            None => (0, None, 0, 0, [0; 7], false, None),
            Some(d) if d.is_exact() => {
                let (f1, f2, h) = d.counts();
                let unique = text_len.then_some(d.sum_len_unique);
                (d.len(), None, f1, f2, h, d.all_once(), unique)
            }
            Some(d) => {
                let sketch = self.hll.as_ref().expect("a sampled level has a sketch");
                // Proven bounds: len + 1 distinct values seen (one was evicted), and no
                // more than the non-null values. A sample whose every value occurred once
                // reports the level unique (`conclude`), so its count is every value.
                let n_non_null = (self.n - self.n_null) as f64;
                let e = if d.all_once() {
                    n_non_null
                } else {
                    sketch.estimate().max((d.len() + 1) as f64).min(n_non_null)
                };
                let scale = e / d.len() as f64;
                let sc = |x: u64| (x as f64 * scale).round() as u64;
                let (f1, f2, h) = d.counts();
                let unique = text_len.then(|| (d.mean_len() * e).round() as u64);
                (
                    e.round() as u64,
                    // The sample proves len + 1 distinct values: one was evicted.
                    Some((e, sketch.std_error(), d.len() + 1)),
                    sc(f1),
                    sc(f2),
                    h.map(sc),
                    d.all_once(),
                    unique,
                )
            }
        };
        let numeric = if is_numeric(dtype) {
            self.int_range()
                .map(|(a, b)| (a as f64, b as f64))
                .or_else(|| self.float_range())
        } else {
            None
        };
        Profile {
            freq: Frequencies {
                n_unique,
                f1,
                f2,
                capture_history: history,
                sum_len_unique,
                first_few: Vec::new(),
                all_once,
                hll,
                ranked: None,
            },
            range: Range {
                argmin: None,
                argmax: None,
                min_len: self.min_len,
                max_len: self.max_len,
            },
            floats: self.floats,
            strings: self.strings.clone(),
            gcd: self.gcd,
            sum_len: self.sum_len,
            is_f32: self.is_f32,
            min: self
                .lo
                .as_ref()
                .and_then(|e| render_value(e.value.as_ref())),
            max: self
                .hi
                .as_ref()
                .and_then(|e| render_value(e.value.as_ref())),
            numeric,
            // Text levels: the exact phase's ranking (None once sampling); no sample yet
            // means no non-null value.
            ranking: if is_text(dtype) {
                self.sample
                    .as_ref()
                    .map_or(Some(Vec::new()), DistinctSample::ranking)
            } else {
                None
            },
        }
    }

    /// Sampling phase: the largest hash the distinct sample holds (None while exact).
    pub(crate) fn sample_top(&self) -> Option<u64> {
        self.sample.as_ref().and_then(DistinctSample::top)
    }

    pub(crate) fn int_range(&self) -> Option<(i128, i128)> {
        match (&self.lo.as_ref()?.key, &self.hi.as_ref()?.key) {
            (Key::I(a), Key::I(b)) => Some((*a, *b)),
            _ => None,
        }
    }

    pub(crate) fn float_range(&self) -> Option<(f64, f64)> {
        match (&self.lo.as_ref()?.key, &self.hi.as_ref()?.key) {
            (Key::F(a), Key::F(b)) => Some((*a, *b)),
            _ => None,
        }
    }

    pub(crate) fn few_distinct(&self) -> Vec<String> {
        self.sample
            .as_ref()
            .filter(|d| d.is_exact() && d.len() <= 5)
            .map_or_else(Vec::new, |d| d.few.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recommenders::engine::polars_views;
    use crate::techniques::describe::frequencies;

    fn absorbed(parts: &[Series], threshold: u64) -> LevelStats {
        let mut st = LevelStats::default();
        for p in parts {
            let b = BatchStats::of(p, st.n, 7, st.sample_top()).unwrap();
            st.absorb(b, threshold);
        }
        st
    }

    fn chunks(s: &Series, k: usize) -> Vec<Series> {
        (0..s.len())
            .step_by(k)
            .map(|o| s.slice(o as i64, k))
            .collect()
    }

    fn summary(st: &LevelStats) -> String {
        let key = |e: &Option<Ext>| e.as_ref().map(|e| format!("{:?}", e.key));
        let d = st.sample.as_ref().map(|d| {
            (
                d.len(),
                d.counts(),
                d.sum_len_unique,
                d.few.clone(),
                d.views.bytes(),
                d.is_exact(),
            )
        });
        let s = st.strings.as_ref().map(|s| {
            (
                s.n_numeric,
                s.n_iso_datetime,
                s.n_f64_roundtrip_fail,
                s.raw_frac_min,
                s.raw_frac_max,
                s.iso_instant_min,
                s.n_iso_time_noncanonical,
            )
        });
        let f = st
            .floats
            .map(|f| (f.n_nan, f.n_fractional, f.max_frac_digits, f.n_neg_zero));
        format!(
            "{} {} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            st.n,
            st.n_null,
            key(&st.lo),
            key(&st.hi),
            st.min_len,
            st.max_len,
            st.gcd,
            st.sum_len,
            d,
            s,
            f,
            st.views.as_ref().map(ViewSim::bytes),
            st.n_midnight,
        )
    }

    fn samples() -> Vec<Series> {
        vec![
            Series::new(
                "i".into(),
                &[
                    Some(30i64),
                    None,
                    Some(-6),
                    Some(12),
                    Some(30),
                    None,
                    Some(0),
                ],
            ),
            Series::new(
                "f".into(),
                &[
                    Some(1.5f64),
                    Some(-0.0),
                    None,
                    Some(f64::NAN),
                    Some(2.25),
                    Some(-7.0),
                ],
            ),
            Series::new(
                "s".into(),
                &[
                    Some("1.50"),
                    Some("a value longer than twelve"),
                    None,
                    Some("2024-01-01T10:00:00"),
                    Some("1.50"),
                    Some("another value longer than twelve"),
                    Some("x"),
                ],
            ),
        ]
    }

    #[test]
    fn splitting_the_stream_does_not_change_the_statistics() {
        for s in samples() {
            let whole = summary(&absorbed(std::slice::from_ref(&s), 10_000));
            for k in [1, 2, 3, 5] {
                let parts = chunks(&s, k);
                assert_eq!(
                    summary(&absorbed(&parts, 10_000)),
                    whole,
                    "{} k={k}",
                    s.name()
                );
            }
        }
    }

    #[test]
    fn distinct_counts_match_describe() {
        let s = Series::new("s".into(), &["a", "b", "a", "c", "c", "c", "d"]);
        let f = frequencies(&encode_series(&s).unwrap(), 7, None, None);
        let st = absorbed(&chunks(&s, 2), 10_000);
        let d = st.sample.as_ref().unwrap();
        assert_eq!(d.len(), f.n_unique);
        assert_eq!(d.counts(), (f.f1, f.f2, f.capture_history));
        assert_eq!(d.few, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn every_dtype_is_counted() {
        let s = Series::new(
            "i".into(),
            (0..2_000i64).map(|i| i % 300).collect::<Vec<_>>(),
        );
        let st = absorbed(&chunks(&s, 700), 10_000);
        let p = st.profile(s.dtype());
        assert_eq!((p.freq.n_unique, p.freq.hll), (300, None));
        assert_eq!(
            (p.min.as_deref(), p.max.as_deref(), p.numeric),
            (Some("0"), Some("299"), Some((0.0, 299.0)))
        );
    }

    #[test]
    fn past_k_the_count_is_hll() {
        let s = Series::new("i".into(), (0..50_000i64).collect::<Vec<_>>());
        let st = absorbed(&chunks(&s, 8_192), 1_000);
        let p = st.profile(s.dtype());
        let (e, se, seen) = p.freq.hll.expect("sampling phase");
        assert_eq!(seen, 1_001);
        assert!((e - 50_000.0).abs() <= 3.0 * se * 50_000.0, "{e}");
        assert!(p.freq.n_unique >= 1_001 && p.freq.all_once);
        assert_eq!(p.freq.sum_len_unique, None); // not a text level
    }

    #[test]
    fn sampled_counts_stay_within_the_proven_bounds() {
        use crate::techniques::describe::conclusions::conclude;
        for seed in 0..6u64 {
            // 50K rows, every 10th null; the rest distinct (seed 0..2), or each value
            // twice (seed 3..5).
            let s = Series::new(
                "i".into(),
                (0..50_000i64)
                    .map(|i| {
                        let v = if seed < 3 { i } else { i / 2 };
                        (i % 10 != 0).then_some(v + seed as i64 * 1_000_000)
                    })
                    .collect::<Vec<_>>(),
            );
            let mut st = LevelStats::default();
            for p in chunks(&s, 10_000) {
                let b = BatchStats::of(&p, st.n, seed, st.sample_top()).unwrap();
                st.absorb(b, 1_000);
            }
            let n = st.n - st.n_null;
            let p = st.profile(s.dtype());
            let c = conclude(s.dtype(), st.n, st.n_null, &p, 1_000);
            let (e, high) = (c.est.est_cardinality, c.est.est_high.unwrap());
            assert!(p.freq.hll.is_some(), "seed {seed}: sampling phase");
            assert!(
                p.freq.n_unique <= n,
                "seed {seed}: {} > {n}",
                p.freq.n_unique
            );
            assert!(high <= n as f64, "seed {seed}: est_high {high} > {n}");
            assert!(c.est.est_low.unwrap() <= e && e <= high, "seed {seed}");
            if c.unique {
                assert_eq!(p.freq.n_unique, n, "seed {seed}");
            }
            assert_eq!(c.unique, seed < 3, "seed {seed}");
        }
    }

    #[test]
    fn view_blocks_match_polars_views() {
        let values: Vec<String> = (0..1000).map(|i| "x".repeat(1 + (i * 37) % 9000)).collect();
        let arr =
            arrow_array::StringArray::from(values.iter().map(String::as_str).collect::<Vec<_>>());
        let views = polars_views(&arr, &AT::Utf8View).unwrap();
        let measured =
            ipc_body_bytes(views.as_ref(), None).unwrap() - pad8(16 * values.len() as u64);
        let mut sim = ViewSim::default();
        values.iter().for_each(|v| sim.push(v.len() as u64));
        assert_eq!(sim.bytes(), measured);
    }

    #[test]
    fn extremes_keep_values_to_render() {
        let s = Series::new("i".into(), &[5i64, -3, 9]);
        let st = absorbed(&chunks(&s, 1), 10_000);
        assert_eq!(st.int_range(), Some((-3, 9)));
        assert_eq!(st.lo.as_ref().unwrap().value.len(), 1);
    }

    #[test]
    fn text_boolean_and_enum_extremes() {
        let s = Series::new(
            "s".into(),
            &[Some("m"), None, Some("b"), Some("z"), Some("c")],
        );
        let st = absorbed(&chunks(&s, 2), 10_000);
        let p = st.profile(s.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("b"), Some("z")));
        let b = Series::new("b".into(), &[true, true, false]);
        let p = absorbed(&chunks(&b, 1), 10_000).profile(b.dtype());
        assert_eq!(
            (p.min.as_deref(), p.max.as_deref()),
            (Some("false"), Some("true"))
        );
        let cats = Categories::global();
        let cat = s
            .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
            .unwrap();
        let st = absorbed(&chunks(&cat, 2), 10_000);
        let p = st.profile(cat.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("b"), Some("z")));
        let plain = |e: &Option<Ext>| {
            let v = &e.as_ref().unwrap().value;
            v.len() == 1 && matches!(v.data_type(), AT::Utf8 | AT::LargeUtf8 | AT::Utf8View)
        };
        assert!(plain(&st.lo) && plain(&st.hi));
        // Enum: ordered by category code (["b", "a"]), not by string.
        let en = Series::new("e".into(), &["a", "b", "a"])
            .cast(&DataType::from_frozen_categories(
                polars::datatypes::FrozenCategories::new(["b", "a"]).unwrap(),
            ))
            .unwrap();
        let st = absorbed(&chunks(&en, 1), 10_000);
        let p = st.profile(en.dtype());
        assert_eq!((p.min.as_deref(), p.max.as_deref()), (Some("b"), Some("a")));
        assert!(plain(&st.lo) && plain(&st.hi));
        let bin = Series::new(
            "bin".into(),
            &[Some(&b"m"[..]), None, Some(b"b"), Some(b"z")],
        );
        let p = absorbed(&chunks(&bin, 2), 10_000).profile(bin.dtype());
        assert_eq!((p.min.is_some(), p.max.is_some()), (true, true));
        assert_ne!(p.min, p.max);
    }

    #[test]
    fn copy_rows_pins_none_of_the_source_buffers() {
        use arrow_array::types::UInt32Type;
        use arrow_array::{Array, DictionaryArray, LargeListArray, StringViewArray};
        use arrow_buffer::OffsetBuffer;
        use std::sync::Arc;

        let views: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..100_000).map(|i| format!("value number {i:>12} padded")),
        ));
        let n = views.len();
        let list: ArrayRef = Arc::new(LargeListArray::new(
            Arc::new(arrow_schema::Field::new_list_field(AT::Utf8View, true)),
            OffsetBuffer::from_lengths(std::iter::repeat_n(1, n)),
            views.clone(),
            None,
        ));
        // Repeated keys and null keys (every 10th row, from row 3).
        let keys = arrow_array::UInt32Array::from_iter(
            (0..n as u32).map(|i| (i % 10 != 3).then_some((n as u32 - 1 - i) / 2)),
        );
        let dict: ArrayRef =
            Arc::new(DictionaryArray::<UInt32Type>::try_new(keys, views.clone()).unwrap());
        let dict_list: ArrayRef = Arc::new(LargeListArray::new(
            Arc::new(arrow_schema::Field::new_list_field(
                dict.data_type().clone(),
                true,
            )),
            OffsetBuffer::from_lengths(std::iter::repeat_n(1, n)),
            dict.clone(),
            None,
        ));
        let dict_struct: ArrayRef = Arc::new(arrow_array::StructArray::from(vec![(
            Arc::new(arrow_schema::Field::new(
                "c",
                dict.data_type().clone(),
                true,
            )),
            dict.clone(),
        )]));
        // The same values with every dictionary decoded to Utf8.
        let decoded = |dt: &AT| match dt {
            AT::LargeList(_) => AT::LargeList(Arc::new(arrow_schema::Field::new_list_field(
                AT::Utf8,
                true,
            ))),
            AT::Struct(_) => AT::Struct(vec![arrow_schema::Field::new("c", AT::Utf8, true)].into()),
            _ => AT::Utf8,
        };
        for a in [&views, &list, &dict, &dict_list, &dict_struct] {
            assert!(a.to_data().get_buffer_memory_size() > 1 << 20);
            let copy = copy_rows(a, 500, 5).unwrap();
            assert_eq!(copy.len(), 5);
            assert_eq!(copy.data_type(), a.data_type());
            let retained = copy.to_data().get_buffer_memory_size();
            assert!(retained < 4096, "{}: {retained} bytes", a.data_type());
            let dt = decoded(a.data_type());
            assert_eq!(
                arrow_cast::cast(&copy, &dt).unwrap().as_ref(),
                arrow_cast::cast(&a.slice(500, 5), &dt).unwrap().as_ref(),
                "{}",
                a.data_type()
            );
        }
    }

    #[test]
    fn copied_categoricals_import_under_the_original_field() {
        use crate::common::arrow_io::import_array;
        let cats = Categories::global();
        let s = Series::new(
            "c".into(),
            &[Some("x"), Some("y"), None, Some("z"), Some("x")],
        )
        .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
        .unwrap();
        let a = export_series(&s, CompatLevel::newest()).unwrap();
        let md: std::collections::HashMap<String, String> = s
            .field()
            .to_arrow(CompatLevel::newest())
            .metadata
            .as_deref()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let field = arrow_schema::Field::new("c", a.data_type().clone(), true).with_metadata(md);
        let copy = copy_rows(&a, 1, 4).unwrap();
        assert_eq!(copy.data_type(), a.data_type());
        let back = import_array(&field, &copy).unwrap();
        assert_eq!(back.dtype(), s.dtype());
        assert!(back
            .cast(&DataType::String)
            .unwrap()
            .equals_missing(&s.slice(1, 4).cast(&DataType::String).unwrap()));
    }
}
