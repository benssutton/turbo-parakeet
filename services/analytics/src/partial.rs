//! Mergeable per-level statistics for the streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md §4). Each batch
//! is profiled per level (a column, or a list's inner values) into a `BatchStats` with
//! describe.rs's kernels; batches are absorbed in stream order into a `LevelStats`,
//! which finishes into the `Profile` recommend.rs's rules read.

use std::collections::HashMap;

use arrow_array::{ArrayRef, UInt64Array};
use arrow_schema::DataType as AT;
use foldhash::fast::FixedState;
use polars::prelude::*;

use crate::arrow_io::export_series;
use crate::cardinality_estimators::{estimate, Estimate, Method};
use crate::describe::{
    arg_extremes, byte_lengths, float_stats, frequency_map, lengths, n_midnight, strings,
    FloatStats, Frequencies, Profile, Range, StringStats,
};
use crate::recommend::{body_size, to_polars_layout, Shape, VIEW_BLOCK, VIEW_MAX_BLOCK};
use crate::shared::encode_series;
use crate::sizes::{classic_layout, ipc_body_bytes};

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

/// Polars' view-array data blocks (recommend.rs `polars_views`) replayed on value
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

/// Ordering key of an extreme: the physical integer, or the float.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) enum Key {
    I(i128),
    F(f64),
}

/// A running extreme: its key, and the value itself (one row, classic layout) to render.
#[derive(Clone, Debug)]
pub(crate) struct Ext {
    pub key: Key,
    pub value: ArrayRef,
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
        )
}

fn ext_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<Ext>> {
    let Some(i) = i else { return Ok(None) };
    let one = s.slice(i as i64, 1);
    let key = if one.dtype().is_float() {
        one.cast(&DataType::Float64)?.f64()?.get(0).map(Key::F)
    } else {
        one.to_physical_repr()
            .cast(&DataType::Int128)?
            .i128()?
            .get(0)
            .map(Key::I)
    };
    let value = classic_layout(&one)?;
    Ok(key.map(|key| Ext { key, value }))
}

/// A distinct value's statistics in one batch.
pub(crate) struct KeyStat {
    key: u64,
    first: u64,
    count: u64,
    mask: u8,
    len: u64,
    /// The value, when the batch holds at most five distinct values.
    text: Option<String>,
}

/// Distinct values of a text level, bounded by `categorical_threshold` (spec §4.2).
#[derive(Clone, Debug, Default)]
pub(crate) struct Distinct {
    /// key → bits 0–1: count capped at 3 (enough for f1 / f2); bits 2–4: capture mask.
    map: HashMap<u64, u8, FixedState>,
    pub sum_len_unique: u64,
    /// Every distinct value, first-occurrence order, while there are at most five.
    pub few: Vec<String>,
    /// The distinct values' view blocks, first-occurrence order (a dictionary's Polars values).
    pub views: ViewSim,
    pub overflowed: bool,
}

impl Distinct {
    /// `keys` in first-occurrence order.
    fn absorb(&mut self, keys: Vec<KeyStat>, threshold: u64) {
        if self.overflowed {
            return;
        }
        for k in keys {
            match self.map.get_mut(&k.key) {
                Some(v) => {
                    let count = ((*v & 3) as u64 + k.count).min(3) as u8;
                    *v = count | (*v & !3) | (k.mask << 2);
                }
                None => {
                    self.map.insert(k.key, k.count.min(3) as u8 | (k.mask << 2));
                    self.sum_len_unique += k.len;
                    self.views.push(k.len);
                    match k.text {
                        Some(t) if self.map.len() <= 5 => self.few.push(t),
                        _ => self.few.clear(),
                    }
                }
            }
            if self.map.len() as u64 > threshold {
                // Exact, not statistical: the estimate is floored at n_unique, so the
                // dictionary is rejected whatever follows. Stop hashing from here on.
                self.overflowed = true;
                self.map = HashMap::default();
                self.few.clear();
                return;
            }
        }
    }

    pub(crate) fn n_unique(&self) -> u64 {
        self.map.len() as u64
    }

    /// (f1, f2, capture history).
    pub(crate) fn counts(&self) -> (u64, u64, [u64; 7]) {
        let (mut f1, mut f2, mut h) = (0, 0, [0u64; 7]);
        for &v in self.map.values() {
            f1 += (v & 3 == 1) as u64;
            f2 += (v & 3 == 2) as u64;
            h[(v >> 2) as usize - 1] += 1;
        }
        (f1, f2, h)
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
    /// Text levels with distinct tracking on: every distinct value, first-occurrence order.
    keys: Option<Vec<KeyStat>>,
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
    /// first value; distinct values are tracked when `track`.
    pub(crate) fn of(s: &Series, offset: u64, seed: u64, track: bool) -> PolarsResult<Self> {
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
        let keys = if track {
            let map = frequency_map(&encode_series(s)?, seed, offset);
            let text = if map.len() <= 5 {
                Some(s.cast(&DataType::String)?)
            } else {
                None
            };
            let mut keys: Vec<KeyStat> = map
                .into_iter()
                .map(|(key, e)| {
                    let row = (e.first - offset) as usize;
                    KeyStat {
                        key,
                        first: e.first,
                        count: e.count,
                        mask: e.mask,
                        len: lens.as_ref().map_or(0, |l| l[row]),
                        text: text
                            .as_ref()
                            .and_then(|t| t.str().ok()?.get(row).map(str::to_owned)),
                    }
                })
                .collect();
            keys.sort_unstable_by_key(|k| k.first);
            Some(keys)
        } else {
            None
        };
        let (min_len, max_len) = lengths(s, lens.as_deref())?;
        Ok(BatchStats {
            n: s.len() as u64,
            n_null: s.null_count() as u64,
            lo,
            hi,
            min_len,
            max_len,
            gcd: crate::gcd::series_gcd(s)?,
            sum_len: lens.as_ref().map(|l| l.iter().sum()),
            floats: float_stats(s)?,
            strings: strings(s)?,
            n_midnight: n_midnight(s)?,
            keys,
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
    pub distinct: Option<Distinct>,
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
            self.distinct
                .get_or_insert_with(Default::default)
                .absorb(keys, threshold);
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

    pub(crate) fn overflowed(&self) -> bool {
        self.distinct.as_ref().is_some_and(|d| d.overflowed)
    }

    /// The Profile the rules read. An overflowed level reports `threshold + 1` distinct
    /// values, so the dictionary gate rejects it.
    pub(crate) fn profile(&self, threshold: u64) -> Profile {
        let d = self.distinct.as_ref();
        let (f1, f2, capture_history) = d.map_or((0, 0, [0; 7]), Distinct::counts);
        let n_unique = d.map_or(0, |d| {
            if d.overflowed {
                threshold + 1
            } else {
                d.n_unique()
            }
        });
        Profile {
            freq: Frequencies {
                n_unique,
                entropy: f64::NAN,
                f1,
                f2,
                top5_idx: Vec::new(),
                top5_count: Vec::new(),
                capture_history,
                sum_len_unique: d.map(|d| d.sum_len_unique),
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
        }
    }

    /// Text levels only: Schnabel → Chao1 (no population), or `overflowed`.
    pub(crate) fn estimate(&self, threshold: u64) -> Option<Estimate> {
        let d = self.distinct.as_ref()?;
        Some(if d.overflowed {
            Estimate {
                est_cardinality: (threshold + 1) as f64,
                est_low: None,
                est_high: None,
                method: Method::Overflowed,
            }
        } else {
            let (f1, f2, h) = d.counts();
            estimate(d.n_unique(), self.n - self.n_null, f1, f2, &h, None)
        })
    }

    pub(crate) fn int_range(&self) -> Option<(i128, i128)> {
        match (self.lo.as_ref()?.key, self.hi.as_ref()?.key) {
            (Key::I(a), Key::I(b)) => Some((a, b)),
            _ => None,
        }
    }

    pub(crate) fn float_range(&self) -> Option<(f64, f64)> {
        match (self.lo.as_ref()?.key, self.hi.as_ref()?.key) {
            (Key::F(a), Key::F(b)) => Some((a, b)),
            _ => None,
        }
    }

    pub(crate) fn few_distinct(&self) -> Vec<String> {
        self.distinct
            .as_ref()
            .filter(|d| !d.overflowed && d.n_unique() <= 5)
            .map_or_else(Vec::new, |d| d.few.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::describe::frequencies;
    use crate::recommend::polars_views;

    fn absorbed(parts: &[Series], track: bool, threshold: u64) -> LevelStats {
        let mut st = LevelStats::default();
        for p in parts {
            let b = BatchStats::of(p, st.n, 7, track).unwrap();
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
        let d = st.distinct.as_ref().map(|d| {
            (
                d.n_unique(),
                d.counts(),
                d.sum_len_unique,
                d.few.clone(),
                d.views.bytes(),
                d.overflowed,
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

    fn samples() -> Vec<(Series, bool)> {
        vec![
            (
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
                false,
            ),
            (
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
                false,
            ),
            (
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
                true,
            ),
        ]
    }

    #[test]
    fn splitting_the_stream_does_not_change_the_statistics() {
        for (s, track) in samples() {
            let whole = summary(&absorbed(&[s.clone()], track, 10_000));
            for k in [1, 2, 3, 5] {
                let parts = chunks(&s, k);
                assert_eq!(
                    summary(&absorbed(&parts, track, 10_000)),
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
        let f = frequencies(&encode_series(&s).unwrap(), 7, None);
        let st = absorbed(&chunks(&s, 2), true, 10_000);
        let d = st.distinct.as_ref().unwrap();
        assert_eq!(d.n_unique(), f.n_unique);
        assert_eq!(d.counts(), (f.f1, f.f2, f.capture_history));
        assert_eq!(d.few, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn distinct_tracking_stops_past_the_threshold() {
        let s = Series::new("s".into(), &["a", "b", "c", "d", "e"]);
        let st = absorbed(&chunks(&s, 2), true, 3);
        assert!(st.overflowed());
        assert_eq!(st.profile(3).freq.n_unique, 4);
        assert_eq!(st.estimate(3).unwrap().method, Method::Overflowed);
        assert!(st.few_distinct().is_empty());
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
        let st = absorbed(&chunks(&s, 1), false, 10_000);
        assert_eq!(st.int_range(), Some((-3, 9)));
        assert_eq!(st.lo.as_ref().unwrap().value.len(), 1);
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
        use crate::arrow_io::import_array;
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
