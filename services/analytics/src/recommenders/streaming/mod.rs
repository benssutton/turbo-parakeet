//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (partial.rs) and a block reservoir
//! (reservoir.rs) are kept; `result` recommends from them at any point. Its output
//! shares Describe's value columns and conclusions (parity spec
//! docs/superpowers/specs/2026-10-01-recommender-parity-design.md, §7).
pub(crate) mod distinct_sample;
pub(crate) mod partial;
pub(crate) mod reservoir;

use std::collections::{HashMap, HashSet};

use arrow_array::{new_empty_array, Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType as AT, Field};
use polars::prelude::{polars_err, AnyValue, DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::bindings::api::{Error, Result};
use crate::common::arrow_io::{export_struct, import_array, import_batch};
use crate::common::ipc_sizes::{classic_layout, ipc_body_bytes, sizes_of};
use crate::recommenders::engine::{
    body_size, cast_to, enum_categories, list_parts, pa_name, pad, pick_by_stats,
    pick_list_by_stats, pl_name, polars_layout, rec_row, recommender_fields, to_polars_layout,
    validity, verify, wrap, Level, Params, Pick, Rec, Shape, Target,
};
use crate::recommenders::streaming::partial::{has_int_range, BatchStats, LevelStats, ViewSim};
use crate::recommenders::streaming::reservoir::{Block, Reservoir};
use crate::techniques::cardinality_estimators::Estimate;
use crate::techniques::describe::conclusions::conclude;
use crate::techniques::describe::{assemble, flatten, value_fields, Profile, Row};

/// One column's state.
pub(crate) struct Column {
    pub name: String,
    /// None while only Null-typed batches have been seen.
    pub dtype: Option<PT>,
    /// pyarrow spelling of the first concrete input type ("null" until then); for an
    /// ineligible column, the dtype its caller reported.
    pub input_type: String,
    pub ineligible: bool,
    /// Global row where the column first appeared; earlier rows count as nulls.
    pub first_row: u64,
    pub outer: LevelStats,
    /// List / Array columns: the inner values.
    pub inner: Option<LevelStats>,
    /// Whether the first concrete batch held the (outer, inner) level as Utf8View /
    /// BinaryView — as a Polars frame exports through `__arrow_c_stream__`.
    pub views_input: (bool, bool),
}

fn is_view(t: &AT) -> bool {
    matches!(t, AT::Utf8View | AT::BinaryView)
}

/// A list type's item type.
fn item_type(t: &AT) -> Option<&AT> {
    match t {
        AT::List(f)
        | AT::LargeList(f)
        | AT::FixedSizeList(f, _)
        | AT::ListView(f)
        | AT::LargeListView(f) => Some(f.data_type()),
        _ => None,
    }
}

pub(crate) struct Streaming {
    params: Params,
    n_rows: u64,
    columns: Vec<Column>,
    index: HashMap<String, usize>,
    reservoir: Reservoir,
}

/// Whether a batch's dtype `b` continues a column of dtype `a`: Categoricals always do
/// (each batch may carry its own mapping); lists and arrays by their inner types;
/// anything else (Enums included, with their categories) only when equal.
fn same_type(a: &PT, b: &PT) -> bool {
    match (a, b) {
        (PT::Categorical(..), PT::Categorical(..)) => true,
        (PT::List(x), PT::List(y)) => same_type(x, y),
        (PT::Array(x, w), PT::Array(y, v)) => w == v && same_type(x, y),
        _ => a == b,
    }
}

/// A Level built from statistics: the rules read its counts, extremes and few distinct
/// values; `values` is empty (only its type is read). Its size is set by the caller.
fn level<'a>(
    dtype: &'a PT,
    classic: &AT,
    p: &'a Profile,
    st: &LevelStats,
    est: Estimate,
    prefix: &'static str,
) -> Level<'a> {
    Level {
        dtype,
        values: new_empty_array(classic),
        p,
        n_rows: st.n,
        n_null: st.n_null,
        // Boolean and Enum extremes also carry integer keys (the value, the category
        // code); the rules read an integer range only for integer-backed numeric and
        // temporal dtypes.
        int_range: has_int_range(dtype).then(|| st.int_range()).flatten(),
        float_range: st.float_range(),
        few_distinct: st.few_distinct(),
        n_midnight: st.n_midnight,
        size_bytes: 0,
        size_note: "",
        est,
        prefix,
        text: Default::default(),
    }
}

/// A Categorical / Enum level's dictionary values as Polars exports them: every
/// category of the dtype's mapping, in id order (`CategoricalMapping::to_arrow`) — for
/// an Enum its categories, for a Categorical every value its Categories object has
/// seen (other columns' included when it is shared, e.g. the global one). Read at
/// `result`: the mapping only grows.
#[derive(Clone, Copy, Debug, Default)]
struct Cats {
    n: f64,
    sum_len: f64,
    /// Polars' view blocks over the categories (the native layout).
    views: u64,
}

fn cats_of(dtype: &PT) -> Option<Cats> {
    let m = match dtype {
        PT::Categorical(_, m) | PT::Enum(_, m) => m,
        _ => return None,
    };
    let mut c = Cats::default();
    let mut views = ViewSim::default();
    for i in 0..m.num_cats_upper_bound() {
        let len = m.cat_to_str(i as _).map_or(0, str::len) as u64;
        c.n += 1.0;
        c.sum_len += len as f64;
        views.push(len);
    }
    c.views = views.bytes();
    Some(c)
}

/// A dictionary's keys: validity + keys of `key`'s width.
fn dictionary_keys(key: &AT, s: &Shape) -> std::result::Result<f64, String> {
    let w = key
        .primitive_width()
        .ok_or_else(|| format!("no predicted size for key {}", pa_name(key)))?;
    Ok(validity(s.n, s.nulls) + pad(s.n * w as f64))
}

/// The classic layout's IPC body (spec §5.3): `body_size`; a Categorical / Enum
/// dictionary (keys + every category, `Cats`); LargeList and FixedSizeList over an
/// analytic inner level (whose shape excludes a null row's slots). Structs have no
/// analytic form.
///
/// Known gap: an Array row that is null but holds values in its slots (hand-built
/// arrays only; Polars and pyarrow leave them empty or null) is counted as null
/// slots, so variable-width values under it are missed.
fn classic_body(
    classic: &AT,
    o: &ViewShape,
    inner: Option<&ViewShape>,
) -> std::result::Result<f64, String> {
    let s = &o.shape;
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    match classic {
        AT::Dictionary(k, values) => {
            let c = o.cats.ok_or("a dictionary without categories")?;
            let offsets = match **values {
                AT::Utf8 => 4.0,
                AT::LargeUtf8 => 8.0,
                _ => return Err(format!("no predicted size for {}", pa_name(classic))),
            };
            Ok(dictionary_keys(k, s)? + pad(offsets * (c.n + 1.0)) + pad(c.sum_len))
        }
        AT::LargeList(f) => Ok(validity(s.n, s.nulls)
            + pad(8.0 * (s.n + 1.0))
            + classic_body(f.data_type(), &inner()?, None)?),
        AT::FixedSizeList(f, w) => {
            // The child holds w slots per row; a null row's slots count as null.
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            Ok(validity(s.n, s.nulls)
                + classic_body(f.data_type(), &ViewShape { shape, ..i }, None)?)
        }
        AT::Struct(_) => Err("no analytic size".into()),
        t => body_size(t, s),
    }
}

/// The original type's uncompressed size: analytic where the classic layout has a
/// closed form (predicted = measured), else the per-batch measured sum.
fn original_size(
    classic: &AT,
    o: &ViewShape,
    inner: Option<&ViewShape>,
    st: &LevelStats,
) -> (u64, &'static str) {
    match classic_body(classic, o, inner) {
        Ok(b) => (b as u64, "analytic"),
        Err(_) => (st.size_bytes, "per-batch sum of"),
    }
}

/// The original type's IPC body in Polars' layout, as one-shot measures the Series
/// Polars imports; None where there is no analytic form (structs, nested lists).
///
/// The model: a freshly built Polars frame (view input), or one zero-copy import of a
/// compact string array (non-view input). A string_view input keeps its own buffers —
/// for a Polars-built frame, blocks of 8 KiB doubling that hold only values over 12
/// bytes (`ViewSim`). A Utf8/Binary input is converted zero-copy (polars-compute
/// `binary_to_binview`): when any value is over 12 bytes the whole values buffer
/// becomes one data buffer, else there is none. A sliced or filtered frame carries
/// arbitrary buffers, which no statistic predicts. A Categorical / Enum exports its
/// keys and every category as views Polars builds (`Cats`).
fn original_polars(classic: &AT, o: &ViewShape, inner: Option<&ViewShape>) -> Option<f64> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    Some(match classic {
        AT::Utf8 | AT::LargeUtf8 | AT::Binary | AT::LargeBinary => {
            let data = match (o.views_input, o.all > 0) {
                (true, _) => o.all as f64,
                (false, true) => pad(s.sum_len),
                (false, false) => 0.0,
            };
            v + pad(16.0 * s.n) + data
        }
        AT::Dictionary(k, _) => {
            let c = o.cats?;
            dictionary_keys(k, s).ok()? + pad(16.0 * c.n) + c.views as f64
        }
        AT::LargeList(f) => {
            v + pad(8.0 * (s.n + 1.0)) + original_polars(f.data_type(), inner?, None)?
        }
        AT::FixedSizeList(f, w) => {
            let (i, w) = (inner?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            v + original_polars(f.data_type(), &ViewShape { shape, ..*i }, None)?
        }
        AT::Struct(_) | AT::List(_) => return None,
        t => body_size(&polars_layout(t, &AT::UInt32), s).ok()?,
    })
}

/// A level's analytic inputs: its shape, Polars' view blocks over all values and
/// over the distinct ones, whether the input held it as views, and a Categorical /
/// Enum level's categories.
#[derive(Clone, Copy)]
struct ViewShape {
    shape: Shape,
    all: u64,
    distinct: u64,
    views_input: bool,
    cats: Option<Cats>,
    /// The per-batch measured Polars-layout size (where there is no closed form) of
    /// the classic layout rebuilt as `to_polars_layout` does.
    rebuilt_polars_bytes: u64,
}

fn view_shape(lvl: &Level, st: &LevelStats, views_input: bool) -> ViewShape {
    ViewShape {
        shape: lvl.shape(),
        all: st.views.as_ref().map_or(0, ViewSim::bytes),
        distinct: st
            .sample
            .as_ref()
            .filter(|d| d.is_exact())
            .map_or(0, |d| d.views.bytes()),
        views_input,
        cats: cats_of(lvl.dtype),
        rebuilt_polars_bytes: st.rebuilt_polars_bytes,
    }
}

/// Uncompressed IPC body of `t` in Polars' layout (Spec B §5.5), from statistics:
/// what `to_polars_layout` + `ipc_body_bytes` measure in one-shot.
fn polars_body(
    t: &Target,
    o: &ViewShape,
    inner: Option<&ViewShape>,
    key: &AT,
) -> std::result::Result<f64, String> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    Ok(match t {
        Target::Plain(_) => v + pad(16.0 * s.n) + o.all as f64,
        Target::Dictionary(..) => {
            let w = key.primitive_width().unwrap_or(4) as f64;
            v + pad(s.n * w) + pad(16.0 * s.d) + o.distinct as f64
        }
        Target::Scalar(it) => {
            let i = inner()?;
            let shape = Shape {
                n: s.n,
                nulls: s.nulls + i.shape.nulls,
                ..i.shape
            };
            polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        Target::List(it) => v + pad(8.0 * (s.n + 1.0)) + polars_body(it, &inner()?, None, key)?,
        Target::FixedList(it, w) => {
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            v + polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        // A list's kept inner level, laid out as `to_polars_layout` does: a dictionary
        // takes the Polars key `key` (one-shot widens an Enum's UInt8 keys to it).
        Target::Original(t) => {
            let t = match t {
                AT::Dictionary(_, v) => AT::Dictionary(Box::new(key.clone()), v.clone()),
                t => t.clone(),
            };
            // No closed form (a struct, say): as measured per batch, rebuilt as
            // one-shot's `to_polars_layout` rebuilds the recast array (exact for one
            // batch; a Struct's dictionary fields take Polars' UInt32 keys).
            original_polars(&t, o, inner().ok().as_ref()).unwrap_or(o.rebuilt_polars_bytes as f64)
        }
        t => body_size(&polars_layout(&t.arrow_type(), key), s)?,
    })
}

/// The recommended array has nulls (as one-shot's `logical_null_count() > 0`).
fn nullable(t: &Target, o: &LevelStats, inner: Option<&LevelStats>) -> bool {
    match t {
        Target::Null => o.n > 0,
        Target::Scalar(_) => o.n_null + inner.map_or(0, |i| i.n_null) > 0,
        _ => o.n_null > 0,
    }
}

/// Column `name`'s rows in block `b` as one Series of `dtype` (absent pieces: nulls),
/// with compact buffers: appending keeps each piece's view buffers, so the block is
/// rebuilt as one-shot's Series is built from its input — through its classic layout
/// (a zero-copy import), or, for view input, with views built as Polars builds them.
fn block_series(
    b: &Block,
    name: &str,
    dtype: &PT,
    views_input: (bool, bool),
) -> PolarsResult<Series> {
    let mut out = Series::new_empty(name.into(), dtype);
    for piece in &b.pieces {
        let s = match piece.cols.iter().find(|(f, _)| f.name() == name) {
            Some((f, a)) => {
                let s = import_array(f, a)?;
                // An Array's cast gives null rows' slots a validity buffer: cast only
                // when the type differs (a Categorical's mapping, say).
                if s.dtype() == dtype {
                    s
                } else {
                    s.cast(dtype)?
                }
            }
            None => Series::full_null(name.into(), piece.rows as usize, dtype),
        };
        out.append(&s)?;
    }
    let mut c = classic_layout(&out)?;
    if views_input.0 || views_input.1 {
        // View input keeps its buffers: rebuild them as Polars builds a frame's views.
        c = to_polars_layout(&c, &AT::UInt32).map_err(|e| polars_err!(ComputeError: "{e}"))?;
    }
    let s = import_array(&Field::new(name, c.data_type().clone(), true), &c)?;
    if s.dtype() == dtype {
        Ok(s)
    } else {
        s.cast(dtype)
    }
}

/// The block cast to the pick and verified against itself: the cross-check (spec §5.1
/// step 6). A failure means a statistic was wrong — never a silent fallback.
fn recast(
    pick: &Pick,
    dtype: &PT,
    classic: &ArrayRef,
    p: &Profile,
    inner: Option<(&PT, &Profile)>,
) -> std::result::Result<ArrayRef, String> {
    if let (Some(ip), Some((idt, iprof)), false) = (
        &pick.inner,
        inner,
        matches!(pick.target, Target::Original(_)),
    ) {
        let (rows, child, _) = list_parts(classic)?;
        let il = Level::of_block(idt, child, iprof);
        let ia = cast_to(&ip.target, &il)?;
        verify(&ip.target, &il, &ia)?;
        return wrap(&pick.target, classic, &rows, &ia);
    }
    let lvl = Level::of_block(dtype, classic.clone(), p);
    let a = cast_to(&pick.target, &lvl)?;
    verify(&pick.target, &lvl, &a)?;
    Ok(a)
}

/// `t` holds the Null type below its top level (a List(Null), a Struct with a Null
/// field, ...).
pub(crate) fn holds_nested_null(t: &PT) -> bool {
    let is_or_holds = |t: &PT| t == &PT::Null || holds_nested_null(t);
    match t {
        PT::List(i) | PT::Array(i, _) => is_or_holds(i),
        PT::Struct(fs) => fs.iter().any(|f| is_or_holds(f.dtype())),
        _ => false,
    }
}

/// A batch column's statistics: None for a Null-typed column; else the column's and,
/// for List / Array, its inner values'.
type Stats = Option<(BatchStats, Option<BatchStats>)>;

impl Streaming {
    pub(crate) fn new(params: Params, reservoir_rows: u64, block_rows: u64) -> Self {
        let block_rows = block_rows.max(1);
        Streaming {
            reservoir: Reservoir::new(
                (reservoir_rows / block_rows) as usize,
                block_rows,
                params.seed,
            ),
            params,
            n_rows: 0,
            columns: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// The column called `name`, created (its earlier rows backfilled as nulls) if new.
    fn column(&mut self, name: &str) -> &mut Column {
        let i = match self.index.get(name) {
            Some(&i) => i,
            None => {
                let mut outer = LevelStats::default();
                outer.nulls(self.n_rows);
                self.columns.push(Column {
                    name: name.to_string(),
                    dtype: None,
                    input_type: "null".into(),
                    ineligible: false,
                    first_row: self.n_rows,
                    outer,
                    inner: None,
                    views_input: (false, false),
                });
                self.index.insert(name.to_string(), self.columns.len() - 1);
                self.columns.len() - 1
            }
        };
        &mut self.columns[i]
    }

    /// A column the caller cannot send (Int128 / UInt128, Object). From here on its rows
    /// count as absent; its `n_null` is not reported.
    pub(crate) fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        if let Some(&i) = self.index.get(name) {
            let c = &self.columns[i];
            if let (Some(d), false) = (&c.dtype, c.ineligible) {
                return Err(Error::InvalidInput(format!(
                    "column {name:?}: type changed from {d} to {dtype}"
                )));
            }
        }
        let c = self.column(name);
        c.ineligible = true;
        c.input_type = dtype.to_string();
        Ok(())
    }

    fn check(&self, s: &Series) -> Result<()> {
        let Some(&i) = self.index.get(s.name().as_str()) else {
            return Ok(());
        };
        let c = &self.columns[i];
        if c.ineligible {
            return Err(Error::InvalidInput(format!(
                "column {:?} is ineligible ({})",
                s.name(),
                c.input_type
            )));
        }
        match &c.dtype {
            Some(d) if s.dtype() != &PT::Null && !same_type(d, s.dtype()) => {
                Err(Error::InvalidInput(format!(
                    "column {:?}: type changed from {d} to {}",
                    s.name(),
                    s.dtype()
                )))
            }
            _ => Ok(()),
        }
    }

    fn batch_stats(&self, s: &Series) -> PolarsResult<Stats> {
        if s.dtype() == &PT::Null {
            return Ok(None);
        }
        let c = self.index.get(s.name().as_str()).map(|&i| &self.columns[i]);
        let seed = self.params.seed;
        let top = c.and_then(|c| c.outer.sample_top());
        let outer = BatchStats::of(s, self.n_rows, seed, top)?;
        let inner = match flatten(s)? {
            Some(v) => {
                let prev = c.and_then(|c| c.inner.as_ref());
                let top = prev.and_then(LevelStats::sample_top);
                Some(BatchStats::of(&v, prev.map_or(0, |l| l.n), seed, top)?)
            }
            None => None,
        };
        Ok(Some((outer, inner)))
    }

    /// Adds one batch; on error the state is unchanged. `batch` must be valid Arrow
    /// (checked on import: `arrow_io::CheckedReader`; see the api module docs).
    pub(crate) fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let mut seen = HashSet::new();
        if let Some(f) = schema
            .fields()
            .iter()
            .find(|f| !seen.insert(f.name().as_str()))
        {
            return Err(Error::InvalidInput(format!(
                "duplicate column {:?}",
                f.name()
            )));
        }
        let series = import_batch(batch).map_err(|e| Error::InvalidInput(e.to_string()))?;
        // A column holding Null below the top is ineligible, as in one-shot (whose
        // Python eligibility refuses it); it is kept out of the statistics and sample.
        let (nested_null, rest): (Vec<_>, Vec<_>) = series
            .into_iter()
            .zip(schema.fields().iter().cloned())
            .zip(batch.columns().iter().cloned())
            .partition(|((s, _), _)| holds_nested_null(s.dtype()));
        for ((s, _), _) in &nested_null {
            // A typed column cannot turn into one (type changed); a Null-typed or
            // already ineligible one can.
            let c = self.index.get(s.name().as_str()).map(|&i| &self.columns[i]);
            if c.is_some_and(|c| !c.ineligible && c.dtype.is_some()) {
                self.check(s)?;
            }
        }
        let (series, cols): (Vec<(Series, arrow_schema::FieldRef)>, Vec<_>) = rest
            .into_iter()
            .map(|((s, f), a)| ((s, f.clone()), (f, a)))
            .unzip();
        series.iter().try_for_each(|(s, _)| self.check(s))?;
        let stats: Vec<Stats> = series
            .par_iter()
            .map(|(s, _)| self.batch_stats(s))
            .collect::<PolarsResult<_>>()
            .map_err(|e| Error::Compute(e.to_string()))?;
        let rows = batch.num_rows() as u64;
        self.reservoir.feed(&cols, rows).map_err(Error::Compute)?;
        // Commit: nothing below can fail.
        let threshold = self.params.categorical_threshold;
        for ((s, f), _) in &nested_null {
            let c = self.column(s.name().as_str());
            c.ineligible = true;
            c.input_type = pa_name(f.data_type());
        }
        // Serially: create new columns and adopt types; then absorb every column's
        // statistics in parallel (column names are unique, so each slot is one column).
        let mut placed = Vec::with_capacity(stats.len());
        for ((s, f), st) in series.iter().zip(stats) {
            let c = self.column(s.name().as_str());
            if st.is_some() && c.dtype.is_none() {
                c.dtype = Some(s.dtype().clone());
                c.input_type = pa_name(f.data_type());
                let t = f.data_type();
                c.views_input = (is_view(t), item_type(t).is_some_and(is_view));
            }
            placed.push((self.index[s.name().as_str()], st));
        }
        let mut slots: Vec<Stats> = (0..self.columns.len()).map(|_| None).collect();
        for (i, st) in placed {
            slots[i] = st;
        }
        // A column absent from the batch, or Null-typed in it, takes `rows` nulls.
        self.columns
            .par_iter_mut()
            .zip(slots)
            .for_each(|(c, st)| match st {
                None => c.outer.nulls(rows),
                Some((outer, inner)) => {
                    c.outer.absorb(outer, threshold);
                    if let Some(inner) = inner {
                        c.inner
                            .get_or_insert_with(Default::default)
                            .absorb(inner, threshold);
                    }
                }
            });
        self.n_rows += rows;
        Ok(())
    }

    /// One row per column, first-seen order (spec §6). The state is kept.
    pub(crate) fn result(&self) -> Result<RecordBatch> {
        let blocks = self.reservoir.blocks();
        let sampled: u64 = blocks.iter().map(|b| b.rows).sum();
        let rows = self
            .columns
            .par_iter()
            .map(|c| self.row(c, &blocks, sampled))
            .collect::<std::result::Result<Vec<Row>, String>>()
            .map_err(Error::Compute)?;
        assemble("streaming_recommend", &recommender_fields(true), &rows)
            .and_then(|s| export_struct(&s))
            .map_err(|e| Error::Compute(e.to_string()))
    }

    fn row(&self, c: &Column, blocks: &[&Block], sampled: u64) -> std::result::Result<Row, String> {
        let text = |s: &str| AnyValue::StringOwned(s.into());
        let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
        let dtype = c.dtype.as_ref().filter(|_| !c.ineligible);
        let mut row: Row = vec![
            text(&c.name),
            text(if dtype.is_some() {
                "computed"
            } else {
                "ineligible"
            }),
            text(&c.input_type),
            AnyValue::UInt64(c.first_row),
            AnyValue::UInt64(self.n_rows),
            if c.ineligible {
                AnyValue::Null
            } else {
                AnyValue::UInt64(c.outer.n_null)
            },
        ];
        let Some(dtype) = dtype else {
            row.resize(recommender_fields(true).len(), AnyValue::Null);
            return Ok(row);
        };
        let t = self.params.categorical_threshold;
        let err = |e: polars::prelude::PolarsError| e.to_string();

        // Levels from statistics.
        let o = &c.outer;
        let p = o.profile(dtype);
        let oc = conclude(dtype, o.n, o.n_null, &p, t);
        let classic = o
            .classic
            .clone()
            .ok_or("a typed column has absorbed no batch")?;
        let mut lvl = level(dtype, &classic, &p, o, oc.est, "");
        let inner = match (&c.inner, dtype) {
            (Some(i), PT::List(it) | PT::Array(it, _)) => Some((i, &**it)),
            _ => None,
        };
        let ip = inner.map(|(i, it)| i.profile(it));
        let ic = inner
            .zip(ip.as_ref())
            .map(|((i, it), ip)| conclude(it, i.n, i.n_null, ip, t));
        let ilvl = match (inner, &ip, &ic) {
            (Some((i, it)), Some(ip), Some(ic)) => {
                let iclassic = i
                    .classic
                    .clone()
                    .ok_or("an inner level has absorbed no batch")?;
                let mut l = level(it, &iclassic, ip, i, ic.est, "inner: ");
                let iv = view_shape(&l, i, c.views_input.1);
                (l.size_bytes, l.size_note) = original_size(&iclassic, &iv, None, i);
                Some((l, iv))
            }
            _ => None,
        };
        let (ilvl, iv) = ilvl.unzip();
        let ov = view_shape(&lvl, o, c.views_input.0);
        (lvl.size_bytes, lvl.size_note) = original_size(&classic, &ov, iv.as_ref(), o);
        // The original's Polars-layout size: analytic where it has a closed form.
        let original_polars =
            original_polars(&classic, &ov, iv.as_ref()).map_or(o.polars_bytes, |b| b as u64);

        // The choice, proven by statistics.
        let pick = match &ilvl {
            Some(il) => {
                let width = match dtype {
                    PT::Array(_, w) => Some(*w as i32),
                    _ => None,
                };
                pick_list_by_stats(&lvl, il, width, &self.params)?
            }
            None => pick_by_stats(&lvl, &self.params)?,
        };

        // Uncompressed sizes: analytic (the original's where it has a closed form).
        let key = pick.target.polars_key().unwrap_or(AT::UInt32);
        let original = matches!(pick.target, Target::Original(_));
        let (rec_size, rec_polars) = if original {
            (lvl.size_bytes, original_polars)
        } else {
            let polars = polars_body(&pick.target, &ov, iv.as_ref(), &key)?;
            (pick.predicted, polars as u64)
        };

        // ZSTD sizes and the cross-check on the sampled blocks.
        let z_level = Some(self.params.zstd_level);
        let mut z = [0u64; 4]; // original Arrow, original Polars, recommended Arrow, recommended Polars
        for b in blocks {
            let s = block_series(b, &c.name, dtype, c.views_input).map_err(err)?;
            let bc = classic_layout(&s).map_err(err)?;
            let sz = sizes_of(&s, &bc, self.params.zstd_level).map_err(err)?;
            z[0] += sz[1];
            z[1] += sz[3];
            if original {
                z[2] += sz[1];
                z[3] += sz[3];
                continue;
            }
            let a = recast(
                &pick,
                dtype,
                &bc,
                &p,
                inner.map(|(_, it)| it).zip(ip.as_ref()),
            )
            .map_err(|e| {
                format!(
                    "cross-check {:?} → {}: {e}",
                    c.name,
                    pa_name(&pick.target.arrow_type())
                )
            })?;
            z[2] += ipc_body_bytes(a.as_ref(), z_level).map_err(err)?;
            z[3] += ipc_body_bytes(to_polars_layout(&a, &key)?.as_ref(), z_level).map_err(err)?;
        }
        let scale = |x: u64| {
            (sampled > 0).then(|| (x as f64 * self.n_rows as f64 / sampled as f64).round() as u64)
        };

        let rec_t = pick.target.arrow_type();
        // A kept Enum (at any depth) is named with its categories.
        let enum_values = enum_categories(dtype);
        let rec = Rec {
            nullable: nullable(&pick.target, o, inner.map(|(i, _)| i)),
            arrow_type: pa_name(&rec_t),
            arrow_size: rec_size,
            arrow_zstd: scale(z[2]),
            polars_type: Some(pl_name(
                &rec_t,
                &c.name,
                enum_values.as_deref().filter(|_| original),
                &key,
            )),
            polars_size: rec_polars,
            polars_zstd: scale(z[3]),
            lossy: pick.lossy,
            candidates: pick.candidates,
        };
        row.extend(p.row(&oc));
        row.push(u(o.n_midnight));
        row.extend([
            AnyValue::UInt64(lvl.size_bytes),
            u(scale(z[0])),
            AnyValue::UInt64(original_polars),
            u(scale(z[1])),
        ]);
        match (inner, &ip, &ic) {
            (Some((i, _)), Some(ip), Some(ic)) => {
                row.push(AnyValue::UInt64(i.n));
                row.push(AnyValue::UInt64(i.n_null));
                row.extend(ip.row(ic));
            }
            _ => row.extend(vec![AnyValue::Null; 2 + value_fields().len()]),
        }
        row.extend(rec_row(&rec));
        row.extend([
            AnyValue::UInt64(sampled),
            AnyValue::UInt64(blocks.len() as u64),
        ]);
        Ok(row)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{FixedSizeListBuilder, Int64Builder};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{
        new_null_array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, ListArray,
        NullArray, StringArray,
    };
    use arrow_schema::DataType as AT;
    use arrow_select::concat::concat;

    use crate::common::arrow_io::export_series;
    use crate::recommenders::engine::arrow_cast;
    use polars::prelude::{CompatLevel, IntoSeries, NamedFrom};

    use super::*;

    pub(crate) fn params() -> Params {
        Params {
            seed: 0,
            zstd_level: 1,
            categorical_threshold: 10_000,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    pub(crate) fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(cols).unwrap()
    }

    pub(crate) fn ints(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    pub(crate) fn strs(v: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }

    pub(crate) fn streaming() -> Streaming {
        Streaming::new(params(), 1 << 20, 1 << 16)
    }

    fn col<'a>(s: &'a Streaming, name: &str) -> &'a Column {
        &s.columns[s.index[name]]
    }

    #[test]
    fn a_new_column_is_backfilled_with_nulls() {
        let mut s = streaming();
        s.add(&batch(vec![("a", ints(&[Some(1), Some(2)]))]))
            .unwrap();
        s.add(&batch(vec![
            ("a", ints(&[Some(3)])),
            ("b", strs(&[Some("x")])),
        ]))
        .unwrap();
        let b = col(&s, "b");
        assert_eq!((b.first_row, b.outer.n, b.outer.n_null), (2, 3, 2));
    }

    #[test]
    fn an_absent_column_counts_as_null() {
        let mut s = streaming();
        s.add(&batch(vec![
            ("a", ints(&[Some(1)])),
            ("b", ints(&[Some(1)])),
        ]))
        .unwrap();
        s.add(&batch(vec![("a", ints(&[Some(2), Some(3)]))]))
            .unwrap();
        let b = col(&s, "b");
        assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
    }

    #[test]
    fn a_null_typed_column_adopts_the_first_concrete_type() {
        let mut s = streaming();
        s.add(&batch(vec![("b", Arc::new(NullArray::new(2)) as ArrayRef)]))
            .unwrap();
        assert!(col(&s, "b").dtype.is_none());
        s.add(&batch(vec![("b", strs(&[Some("x")]))])).unwrap();
        let b = col(&s, "b");
        assert_eq!(b.dtype, Some(PT::String));
        assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
    }

    #[test]
    fn a_type_change_is_rejected_and_changes_nothing() {
        let mut s = streaming();
        s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
        let err = s
            .add(&batch(vec![
                ("a", strs(&[Some("x")])),
                ("c", ints(&[Some(1)])),
            ]))
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("type changed")),
            "{err:?}"
        );
        assert_eq!((s.n_rows, s.columns.len(), col(&s, "a").outer.n), (1, 1, 1));
    }

    #[test]
    fn a_duplicate_column_is_rejected() {
        let mut s = streaming();
        let err = s
            .add(&batch(vec![
                ("a", ints(&[Some(1)])),
                ("a", ints(&[Some(2)])),
            ]))
            .unwrap_err();
        assert_eq!(err, Error::InvalidInput("duplicate column \"a\"".into()));
    }

    #[test]
    fn ineligible_columns_are_kept_apart() {
        let mut s = streaming();
        s.mark_ineligible("w", "Int128").unwrap();
        s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
        assert!(col(&s, "w").ineligible);
        assert!(s.add(&batch(vec![("w", ints(&[Some(1)]))])).is_err());
        assert!(s.mark_ineligible("a", "Object").is_err());
    }

    #[test]
    fn a_nested_null_column_is_ineligible() {
        // One-shot lists a column holding Null below the top as ineligible; Polars
        // exports such a column's Null level with a buffer arrow-rs cannot import.
        let item = Arc::new(arrow_schema::Field::new("item", AT::Null, true));
        let nulls: ArrayRef = Arc::new(ListArray::new(
            item,
            arrow_buffer::OffsetBuffer::from_lengths([1, 0]),
            Arc::new(NullArray::new(1)),
            Some(arrow_buffer::NullBuffer::from(vec![true, false])),
        ));
        let b = || batch(vec![("n", nulls.clone()), ("a", ints(&[Some(1), Some(2)]))]);
        let mut s = streaming();
        s.add(&b()).unwrap();
        s.add(&b()).unwrap();
        let n = col(&s, "n");
        assert!(n.ineligible);
        assert_eq!(n.input_type, "list<item: null>");
        assert_eq!(col(&s, "a").outer.n, 4);
        let out = s.result().unwrap();
        assert_eq!(
            texts(&out, "status"),
            vec![Some("ineligible".into()), Some("computed".into())]
        );
        // A column already typed cannot turn into one.
        let mut s = streaming();
        s.add(&batch(vec![("n", ints(&[Some(1)]))])).unwrap();
        let err = s.add(&b()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("type changed")),
            "{err:?}"
        );
        assert_eq!(s.n_rows, 1);
    }

    fn one_shot(b: &RecordBatch) -> RecordBatch {
        let mut r = crate::recommenders::oneshot::OneShot::new(params());
        r.add(b).unwrap();
        r.result().unwrap()
    }

    pub(crate) fn texts(b: &RecordBatch, name: &str) -> Vec<Option<String>> {
        let a = arrow_cast(b.column_by_name(name).unwrap().as_ref(), &AT::Utf8).unwrap();
        a.as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    fn mixed() -> RecordBatch {
        let list = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2)]),
            None,
            Some(vec![Some(3)]),
            Some(vec![Some(4)]),
            Some(vec![Some(500)]),
        ]);
        batch(vec![
            (
                "i",
                ints(&[Some(1), None, Some(300), Some(-5), Some(7), Some(7)]),
            ),
            (
                "s_num",
                strs(&[
                    Some("1.50"),
                    Some("2.25"),
                    None,
                    Some("3"),
                    Some("4.5"),
                    Some("5"),
                ]),
            ),
            (
                "s_cat",
                strs(&[
                    Some("red"),
                    Some("green"),
                    Some("red"),
                    None,
                    Some("blue"),
                    Some("a long value over twelve bytes"),
                ]),
            ),
            (
                "s_dt",
                strs(&[
                    Some("2024-01-01T10:00:00"),
                    Some("2024-01-02T11:30:00"),
                    None,
                    Some("2024-01-03T00:00:00"),
                    Some("2024-01-04T12:00:00.250"),
                    Some("2024-01-05 13:00:00"),
                ]),
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(1.5),
                    Some(-0.0),
                    None,
                    Some(2.25),
                    Some(0.5),
                    Some(3.0),
                ])) as ArrayRef,
            ),
            ("l", Arc::new(list) as ArrayRef),
            // Kept in its original type by construction: beyond float32's range and
            // with ~200 integer digits, no narrower float or decimal holds it.
            (
                "f_orig",
                Arc::new(Float64Array::from(
                    (0..6)
                        .map(|i| (i % 3 != 1).then(|| 3.3e200 * (i + 1) as f64 / 7.0))
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            // Values over 12 bytes: Polars packs them into view blocks.
            (
                "free",
                strs(&[
                    Some("free text value number one xx"),
                    None,
                    Some("free text value number three xx"),
                    Some("free text value number four xx"),
                    Some("free text value number five xx"),
                    Some("free text value number six xx"),
                ]),
            ),
            ("arr", Arc::new(fixed_list()) as ArrayRef),
        ])
    }

    /// Array[Int64, 2] with a null row (its slots hold values) and a null item.
    fn fixed_list() -> FixedSizeListArray {
        let mut b = FixedSizeListBuilder::new(Int64Builder::new(), 2);
        for i in 0..6i64 {
            b.values().append_value(i * 1_000_000_007);
            if i == 4 {
                b.values().append_null();
            } else {
                b.values().append_value(i);
            }
            b.append(i != 2);
        }
        b.finish()
    }

    const REC: [&str; 12] = [
        "rec_arrow_type",
        "rec_nullable",
        "rec_lossy_formatting",
        "rec_arrow_size_bytes",
        "rec_polars_size_bytes",
        "rec_arrow_size_zstd_bytes",
        "rec_polars_size_zstd_bytes",
        "rec_polars_type",
        "size_bytes",
        "size_polars_bytes",
        "size_zstd_bytes",
        "size_polars_zstd_bytes",
    ];

    /// `batches` streamed recommend and size like `whole` in one shot.
    fn assert_like_one_shot(batches: &[RecordBatch], whole: &RecordBatch, label: &str) {
        assert_like_one_shot_but(batches, whole, label, &[]);
    }

    /// As `assert_like_one_shot`, the fields `skip` aside.
    fn assert_like_one_shot_but(
        batches: &[RecordBatch],
        whole: &RecordBatch,
        label: &str,
        skip: &[&str],
    ) {
        let reference = one_shot(whole);
        let mut s = streaming();
        batches.iter().for_each(|b| s.add(b).unwrap());
        let out = s.result().unwrap();
        let columns = texts(&out, "column");
        assert_eq!(columns, texts(&reference, "column"), "{label}");
        for name in REC.into_iter().filter(|n| !skip.contains(n)) {
            let (got, want) = (texts(&out, name), texts(&reference, name));
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{label} {:?} {name}", columns[row]);
            }
        }
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        assert_eq!(
            texts(&out, "n_sampled_rows")[0],
            Some(rows.to_string()),
            "{label}"
        );
    }

    #[test]
    fn output_is_the_shared_streaming_schema() {
        let mut s = streaming();
        s.add(&mixed()).unwrap();
        let out = s.result().unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let want: Vec<String> = crate::recommenders::engine::recommender_fields(true)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, want);
    }

    #[test]
    fn inner_columns_match_one_shot() {
        let whole = mixed();
        let reference = one_shot(&whole);
        let mut s = streaming();
        s.add(&whole.slice(0, 4)).unwrap();
        s.add(&whole.slice(4, 2)).unwrap();
        let out = s.result().unwrap();
        for name in [
            "inner_n_values",
            "inner_n_null",
            "inner_n_unique",
            "inner_unique",
            "inner_class",
            "inner_min",
            "inner_max",
            "inner_min_len",
            "inner_max_len",
            "inner_sum_len",
            "inner_gcd",
        ] {
            assert_eq!(texts(&out, name), texts(&reference, name), "{name}");
        }
        let columns = texts(&out, "column");
        let l = columns
            .iter()
            .position(|c| c.as_deref() == Some("l"))
            .unwrap();
        assert_eq!(texts(&out, "inner_n_values")[l].as_deref(), Some("5"));
    }

    #[test]
    fn streamed_batches_recommend_like_one_shot() {
        let whole = mixed();
        for k in [1, 2, 6] {
            let batches: Vec<_> = (0..whole.num_rows())
                .step_by(k)
                .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
                .collect();
            assert_like_one_shot(&batches, &whole, &format!("k={k}"));
        }
    }

    /// A frame as Polars exports it through `__arrow_c_stream__`: string_view columns
    /// whose data blocks Polars built (8 KiB, doubling; only values over 12 bytes).
    fn polars_views_frame() -> RecordBatch {
        let text = |i: usize| match i % 3 {
            0 => None,
            1 => Some(format!("s{}", i % 7)),
            _ => Some(format!("a value longer than twelve bytes {i}")),
        };
        let n = 900;
        let s = Series::new("s".into(), (0..n).map(text).collect::<Vec<_>>());
        let l = Series::new(
            "l".into(),
            (0..n)
                .map(|i| {
                    (!i.is_multiple_of(5)).then(|| {
                        Series::new("".into(), (i..i + i % 3).map(text).collect::<Vec<_>>())
                    })
                })
                .collect::<Vec<_>>(),
        );
        let cols: Vec<(&str, ArrayRef)> = [&s, &l]
            .into_iter()
            .map(|s| {
                (
                    s.name().as_str(),
                    export_series(s, CompatLevel::newest()).unwrap(),
                )
            })
            .collect();
        assert_eq!(cols[0].1.data_type(), &AT::Utf8View);
        batch(cols)
    }

    /// `columns` as Polars exports them: native layout plus Polars' field metadata
    /// (which restores Categorical / Enum on import).
    fn polars_frame(columns: &[Series]) -> RecordBatch {
        let fields: Vec<(arrow_schema::Field, ArrayRef)> = columns
            .iter()
            .map(|s| {
                let a = export_series(s, CompatLevel::newest()).unwrap();
                let md: HashMap<String, String> = s
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
                let f = arrow_schema::Field::new(s.name().as_str(), a.data_type().clone(), true)
                    .with_metadata(md);
                (f, a)
            })
            .collect();
        let (fields, arrays): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), arrays).unwrap()
    }

    /// The original candidate's predicted size, per column.
    fn original_predicted(b: &RecordBatch) -> Vec<Option<String>> {
        let col = b.column_by_name("rec_candidates").unwrap().as_list::<i64>();
        (0..b.num_rows())
            .map(|row| {
                let v = col.value(row);
                let s = v.as_struct();
                let text =
                    |f: &str| arrow_cast(s.column_by_name(f).unwrap().as_ref(), &AT::Utf8).unwrap();
                let (rule, predicted) = (text("rule"), text("predicted_bytes"));
                let (rule, predicted) = (rule.as_string::<i32>(), predicted.as_string::<i32>());
                (0..s.len())
                    .find(|&i| rule.value(i) == "original")
                    .map(|i| predicted.value(i).to_string())
            })
            .collect()
    }

    #[test]
    fn dictionary_sources_size_like_one_shot() {
        use polars::prelude::{CategoricalPhysical, Categories, FrozenCategories};
        let long = "a category name longer than twelve bytes";
        let values: Vec<Option<&str>> = (0..60)
            .map(|i| match i % 5 {
                0 => None,
                1 | 2 => Some("b"),
                3 => Some("a"),
                _ => Some(long),
            })
            .collect();
        let e = PT::from_frozen_categories(
            FrozenCategories::new(["b", "a", "unused", long, "another long unused category"])
                .unwrap(),
        );
        let cats = Categories::new(
            "streaming-dictionary-test".into(),
            "".into(),
            CategoricalPhysical::U32,
        );
        let c = PT::Categorical(cats.clone(), cats.mapping());
        let series = |name: &str, dt: &PT| Series::new(name.into(), &values).cast(dt).unwrap();
        let l = Series::new(
            "l".into(),
            (0..60)
                .map(|i| {
                    Series::new("".into(), &values[i..(i + i % 3).min(60)])
                        .cast(&e)
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        );
        let whole = polars_frame(&[series("enum", &e), series("cat", &c), l]);
        let reference = one_shot(&whole);
        for k in [60, 7, 1] {
            let batches: Vec<_> = (0..whole.num_rows())
                .step_by(k)
                .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
                .collect();
            assert_like_one_shot(&batches, &whole, &format!("dictionary k={k}"));
            let mut s = streaming();
            batches.iter().for_each(|b| s.add(b).unwrap());
            let out = s.result().unwrap();
            assert_eq!(
                original_predicted(&out),
                original_predicted(&reference),
                "k={k}"
            );
        }
    }

    #[test]
    fn a_list_of_structs_keeps_its_inner_type() {
        // Rows of 0-2 structs {a: f64, b: string (0-14 bytes, some over 12)}, every
        // 5th row null: the kept inner struct has no closed-form Polars size.
        let row = |i: usize| {
            (!i.is_multiple_of(5)).then(|| {
                let k = i % 3;
                let a = Series::new("a".into(), vec![i as f64 / 7.0; k]);
                let b = Series::new("b".into(), vec!["x".repeat(i % 15); k]);
                polars::prelude::StructChunked::from_series("".into(), k, [a, b].iter())
                    .unwrap()
                    .into_series()
            })
        };
        let s = Series::new("s".into(), (0..100).map(row).collect::<Vec<_>>());
        let whole = polars_frame(&[s]);
        let reference = one_shot(&whole);
        assert!(texts(&reference, "rec_arrow_type")[0]
            .as_deref()
            .is_some_and(|t| t.contains("struct")));
        // One batch: every recommendation and size as one-shot's, but the original's
        // Polars ZSTD size: the input's struct field holds its views in many small
        // variadic buffers (as py-polars builds them too), each compressed on its
        // own, which the sample's rebuilt block does not reproduce.
        assert_like_one_shot_but(
            std::slice::from_ref(&whole),
            &whole,
            "one batch",
            &["size_polars_zstd_bytes"],
        );
        // Several: the same recommendation.
        let mut st = streaming();
        st.add(&whole.slice(0, 40)).unwrap();
        st.add(&whole.slice(40, 60)).unwrap();
        let out = st.result().unwrap();
        assert_eq!(
            texts(&out, "rec_arrow_type"),
            texts(&reference, "rec_arrow_type")
        );
    }

    /// [[{a:1}]], None, [[{a:3}, None], None], [[{a:5}]], sliced to rows 1..3: a
    /// Polars Series over it holds structs whose validity starts mid-bitmap, which
    /// polars-arrow exports as invalid Arrow (arrow-rs panicked on it).
    fn sliced_lists_of_lists_of_structs() -> RecordBatch {
        use arrow_array::builder::{LargeListBuilder, StructBuilder};
        type B = LargeListBuilder<LargeListBuilder<StructBuilder>>;
        let sb =
            StructBuilder::from_fields(vec![arrow_schema::Field::new("a", AT::Int64, true)], 8);
        let mut b: B = LargeListBuilder::new(LargeListBuilder::new(sb));
        let item = |b: &mut B, v: Option<i64>| {
            let s = b.values().values();
            s.field_builder::<Int64Builder>(0).unwrap().append_option(v);
            s.append(v.is_some());
        };
        item(&mut b, Some(1));
        b.values().append(true);
        b.append(true);
        b.append(false);
        item(&mut b, Some(3));
        item(&mut b, None);
        b.values().append(true);
        b.values().append(false);
        b.append(true);
        item(&mut b, Some(5));
        b.values().append(true);
        b.append(true);
        let a: ArrayRef = Arc::new(b.finish());
        batch(vec![("c", a.slice(1, 2))])
    }

    #[test]
    fn sliced_nested_structs_with_nulls_size_like_one_shot() {
        let b = sliced_lists_of_lists_of_structs();
        assert_like_one_shot(std::slice::from_ref(&b), &b, "sliced list<list<struct>>");
    }

    #[test]
    fn view_input_sizes_like_one_shot() {
        let whole = polars_views_frame();
        for k in [900, 250, 7] {
            let batches: Vec<_> = (0..whole.num_rows())
                .step_by(k)
                .map(|off| whole.slice(off, k.min(whole.num_rows() - off)))
                .collect();
            assert_like_one_shot(&batches, &whole, &format!("views k={k}"));
        }
    }

    #[test]
    fn a_column_absent_from_the_first_batches_sizes_like_its_nulls() {
        // "i" in every batch; the rest only from row 3 on.
        let full = mixed();
        let b = full.slice(3, 3);
        let first = batch(vec![("i", full.slice(0, 3).column(0).clone())]);
        let cols: Vec<(String, ArrayRef)> = full
            .schema()
            .fields()
            .iter()
            .zip(full.columns())
            .map(|(f, c)| {
                let col = if f.name() == "i" {
                    c.clone()
                } else {
                    let nulls = new_null_array(c.data_type(), 3);
                    concat(&[nulls.as_ref(), c.slice(3, 3).as_ref()]).unwrap()
                };
                (f.name().clone(), col)
            })
            .collect();
        let whole = batch(cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect());
        assert_like_one_shot(&[first.clone(), b.clone()], &whole, "absent");
        assert_like_one_shot(
            &[first, b.slice(0, 1), b.slice(1, 2)],
            &whole,
            "absent, split",
        );
    }

    #[test]
    fn no_sample_means_no_zstd_sizes() {
        let mut s = Streaming::new(params(), 0, 1 << 16);
        s.add(&mixed()).unwrap();
        let out = s.result().unwrap();
        assert!(texts(&out, "rec_arrow_size_zstd_bytes")
            .iter()
            .all(Option::is_none));
        assert_eq!(texts(&out, "n_sampled_blocks")[0].as_deref(), Some("0"));
        assert!(texts(&out, "rec_arrow_type").iter().all(Option::is_some));
    }

    /// Column "s" streamed with `categorical_threshold` 3.
    fn threshold_3(v: &[Option<&str>]) -> RecordBatch {
        let mut p = params();
        p.categorical_threshold = 3;
        let mut s = Streaming::new(p, 1 << 20, 1 << 16);
        s.add(&batch(vec![("s", strs(v))])).unwrap();
        s.result().unwrap()
    }

    /// The first row's string→dictionary candidate outcome.
    fn dictionary_outcome(out: &RecordBatch) -> String {
        let list = out
            .column_by_name("rec_candidates")
            .unwrap()
            .as_list::<i64>()
            .value(0);
        let cands = list.as_struct();
        let text =
            |f: &str| arrow_cast(cands.column_by_name(f).unwrap().as_ref(), &AT::Utf8).unwrap();
        let (rules, outcomes) = (text("rule"), text("outcome"));
        let (rules, outcomes) = (rules.as_string::<i32>(), outcomes.as_string::<i32>());
        let i = (0..rules.len())
            .find(|&i| rules.value(i) == "string→dictionary")
            .unwrap();
        outcomes.value(i).to_string()
    }

    #[test]
    fn exact_count_past_the_threshold_rejects_the_dictionary() {
        let out = threshold_3(&[
            Some("a"),
            Some("b"),
            Some("c"),
            Some("d"),
            Some("e"),
            Some("a"),
        ]);
        assert_eq!(texts(&out, "n_unique")[0].as_deref(), Some("5"));
        assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("observed"));
        assert_eq!(dictionary_outcome(&out), "rejected");
    }

    #[test]
    fn overflow_rejects_the_dictionary() {
        // More distinct values than the sample holds (k = max(threshold, 1000)).
        let v: Vec<String> = (0..2_000).map(|i| format!("v{i}")).collect();
        let v: Vec<Option<&str>> = v.iter().map(|x| Some(x.as_str())).collect();
        let out = threshold_3(&v);
        let n_unique: u64 = texts(&out, "n_unique")[0]
            .as_deref()
            .unwrap()
            .parse()
            .unwrap();
        assert!(n_unique >= 1_001, "{n_unique}");
        assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("hll"));
        assert_eq!(dictionary_outcome(&out), "rejected");
    }

    #[test]
    fn shared_columns_are_filled() {
        let mut s = Streaming::new(params(), 0, 1);
        s.add(&batch(vec![(
            "a",
            ints(&[Some(0), Some(5), Some(7), Some(0), Some(5), Some(7)]),
        )]))
        .unwrap();
        let out = s.result().unwrap();
        let some = |v: &[&str]| v.iter().map(|x| Some(x.to_string())).collect::<Vec<_>>();
        assert_eq!(texts(&out, "class"), some(&["ordinal"])); // 7 ≤ 2·6
        assert_eq!(texts(&out, "min"), some(&["0"]));
        assert_eq!(texts(&out, "max"), some(&["7"]));
        assert_eq!(texts(&out, "unique"), some(&["false"]));
        assert_eq!(texts(&out, "est_method"), some(&["observed"]));
        assert_eq!(texts(&out, "n_midnight"), vec![None]);
    }

    #[test]
    fn ineligible_null_typed_and_empty() {
        assert_eq!(streaming().result().unwrap().num_rows(), 0);
        let mut s = streaming();
        s.mark_ineligible("w", "Int128").unwrap();
        s.add(&batch(vec![
            ("a", ints(&[Some(1), Some(2)])),
            ("n", Arc::new(NullArray::new(2)) as ArrayRef),
        ]))
        .unwrap();
        let out = s.result().unwrap();
        let some = |v: &[&str]| v.iter().map(|x| Some(x.to_string())).collect::<Vec<_>>();
        assert_eq!(texts(&out, "column"), some(&["w", "a", "n"]));
        assert_eq!(
            texts(&out, "status"),
            some(&["ineligible", "computed", "ineligible"])
        );
        assert_eq!(texts(&out, "dtype")[0].as_deref(), Some("Int128"));
        assert_eq!(texts(&out, "n_null")[0], None);
        assert_eq!(texts(&out, "n_rows")[0].as_deref(), Some("2"));
    }

    #[test]
    fn an_enum_is_named_by_its_categories() {
        use polars::prelude::FrozenCategories;
        let e = PT::from_frozen_categories(FrozenCategories::new(["b", "a"]).unwrap());
        let want = Some(vec!["b".to_string(), "a".to_string()]);
        assert_eq!(enum_categories(&e), want);
        assert_eq!(enum_categories(&PT::List(Box::new(e.clone()))), want);
        assert_eq!(enum_categories(&PT::String), None);
        let t = AT::Dictionary(Box::new(AT::UInt8), Box::new(AT::Utf8));
        assert_eq!(
            pl_name(&t, "x", enum_categories(&e).as_deref(), &AT::UInt8),
            "Enum(categories=['b', 'a'])"
        );
    }
}
