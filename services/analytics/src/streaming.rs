//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (partial.rs) and a block reservoir
//! (reservoir.rs) are kept; `finish` recommends from them at any point (Task 10).

use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use arrow_array::{new_empty_array, Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType as AT, Field};
use polars::prelude::{AnyValue, DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::api::{Error, Result};
use crate::arrow_io::{export_struct, import_array, import_batch};
use crate::cardinality_estimators::{estimate, Estimate};
use crate::describe::{assemble, flatten, Profile, Row};
use crate::partial::{BatchStats, Ext, LevelStats, ViewSim};
use crate::recommend::{
    arrow_cast, body_size, cast_to, is_text, list_parts, pa_name, pad, pick_by_stats,
    pick_list_by_stats, pl_name, polars_layout, rec_fields, rec_row, to_polars_layout, validity,
    verify, wrap, Level, Params, Pick, Rec, Shape, Target,
};
use crate::reservoir::{Block, Reservoir};
use crate::sizes::{classic_layout, ipc_body_bytes, sizes_of};

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

/// Output columns (spec §6), in order.
fn output_fields() -> Vec<(String, PT)> {
    let mut f: Vec<(String, PT)> = [
        ("column", PT::String),
        ("status", PT::String),
        ("dtype", PT::String),
        ("first_row", PT::UInt64),
        ("n_rows", PT::UInt64),
        ("n_null", PT::UInt64),
        ("min", PT::String),
        ("max", PT::String),
        ("gcd", PT::Decimal(Some(38), Some(0))),
        ("sum_len", PT::UInt64),
        ("min_len", PT::UInt64),
        ("max_len", PT::UInt64),
        ("n_unique", PT::UInt64),
        ("distinct_overflowed", PT::Boolean),
        ("est_cardinality", PT::Float64),
        ("est_low", PT::Float64),
        ("est_high", PT::Float64),
        ("est_method", PT::String),
        ("size_bytes", PT::UInt64),
        ("size_zstd_bytes", PT::UInt64),
        ("size_polars_bytes", PT::UInt64),
        ("size_polars_zstd_bytes", PT::UInt64),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    f.extend(rec_fields());
    f.push(("n_sampled_rows".into(), PT::UInt64));
    f.push(("n_sampled_blocks".into(), PT::UInt64));
    f
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
        int_range: st.int_range(),
        float_range: st.float_range(),
        few_distinct: st.few_distinct(),
        n_midnight: st.n_midnight,
        size_bytes: 0,
        size_note: "",
        est,
        r: 1.0,
        prefix,
        text: Default::default(),
    }
}

/// The classic layout's IPC body (spec §5.3): `body_size`, plus LargeList and
/// FixedSizeList over an analytic inner type (`inner`: the inner level's shape, which
/// excludes a null row's slots). Dictionaries (Categorical / Enum mappings) and
/// structs have no analytic form.
fn classic_body(
    classic: &AT,
    s: &Shape,
    inner: Option<&Shape>,
) -> std::result::Result<f64, String> {
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    match classic {
        AT::LargeList(f) => Ok(validity(s.n, s.nulls)
            + pad(8.0 * (s.n + 1.0))
            + body_size(f.data_type(), &inner()?)?),
        AT::FixedSizeList(f, w) => {
            // The child holds w slots per row; a null row's slots count as null.
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.nulls + s.nulls * w,
                ..i
            };
            Ok(validity(s.n, s.nulls) + body_size(f.data_type(), &shape)?)
        }
        AT::Dictionary(..) | AT::Struct(_) => Err("no analytic size".into()),
        t => body_size(t, s),
    }
}

/// The original type's uncompressed size: analytic where the classic layout has a
/// closed form (predicted = measured), else the per-batch measured sum.
fn original_size(
    classic: &AT,
    shape: &Shape,
    inner: Option<&Shape>,
    st: &LevelStats,
) -> (u64, &'static str) {
    match classic_body(classic, shape, inner) {
        Ok(b) => (b as u64, "analytic"),
        Err(_) => (st.size_bytes, "per-batch sum of"),
    }
}

/// The original type's IPC body in Polars' layout, as one-shot measures the Series
/// Polars imports: None where there is no analytic form (dictionaries, structs, nested
/// lists). Polars converts Utf8/Binary to views zero-copy (polars-compute
/// `binary_to_binview`): when any value is over 12 bytes (Polars' view blocks hold
/// bytes) the whole values buffer becomes one data buffer, else there is none.
fn original_polars(classic: &AT, o: &ViewShape, inner: Option<&ViewShape>) -> Option<f64> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    Some(match classic {
        AT::Utf8 | AT::LargeUtf8 | AT::Binary | AT::LargeBinary => {
            v + pad(16.0 * s.n) + if o.all > 0 { pad(s.sum_len) } else { 0.0 }
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
        AT::Dictionary(..) | AT::Struct(_) | AT::List(_) => return None,
        t => body_size(&polars_layout(t, &AT::UInt32), s).ok()?,
    })
}

/// A level's analytic Polars-layout inputs: its shape and Polars' view blocks over all
/// values and over the distinct ones.
#[derive(Clone, Copy)]
struct ViewShape {
    shape: Shape,
    all: u64,
    distinct: u64,
}

fn view_shape(lvl: &Level, st: &LevelStats) -> ViewShape {
    ViewShape {
        shape: lvl.shape(),
        all: st.views.as_ref().map_or(0, ViewSim::bytes),
        distinct: st.distinct.as_ref().map_or(0, |d| d.views.bytes()),
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
        Target::Original(_) => return Err("the original's Polars size is measured".into()),
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

fn render(e: &Option<Ext>) -> AnyValue<'static> {
    e.as_ref()
        .and_then(|e| arrow_cast(e.value.as_ref(), &AT::Utf8).ok())
        .and_then(|a| {
            let s = a.as_string::<i32>();
            s.is_valid(0)
                .then(|| AnyValue::StringOwned(s.value(0).into()))
        })
        .unwrap_or(AnyValue::Null)
}

/// An Enum's categories, found through lists and arrays.
fn enum_categories(dtype: &PT) -> Option<Vec<String>> {
    match dtype {
        PT::Enum(fc, _) => Some(fc.categories().values_iter().map(str::to_owned).collect()),
        PT::List(it) | PT::Array(it, _) => enum_categories(it),
        _ => None,
    }
}

/// Column `name`'s rows in block `b` as one Series of `dtype` (absent pieces: nulls),
/// with compact buffers: appending keeps each piece's view buffers, so the block is
/// rebuilt through its classic layout, as one-shot's Series is built from its input.
fn block_series(b: &Block, name: &str, dtype: &PT) -> PolarsResult<Series> {
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
    let c = classic_layout(&out)?;
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
        // Distinct values are hashed until a level overflows, never after.
        let track = |l: Option<&LevelStats>| l.is_none_or(|l| !l.overflowed());
        let seed = self.params.seed;
        let outer = BatchStats::of(
            s,
            self.n_rows,
            seed,
            is_text(s.dtype()) && track(c.map(|c| &c.outer)),
        )?;
        let inner = match flatten(s)? {
            Some(v) => {
                let prev = c.and_then(|c| c.inner.as_ref());
                Some(BatchStats::of(
                    &v,
                    prev.map_or(0, |l| l.n),
                    seed,
                    is_text(v.dtype()) && track(prev),
                )?)
            }
            None => None,
        };
        Ok(Some((outer, inner)))
    }

    /// Adds one batch; on error the state is unchanged.
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
        series.iter().try_for_each(|s| self.check(s))?;
        let stats: Vec<Stats> = series
            .par_iter()
            .map(|s| self.batch_stats(s))
            .collect::<PolarsResult<_>>()
            .map_err(|e| Error::Compute(e.to_string()))?;
        let rows = batch.num_rows() as u64;
        let cols: Vec<_> = schema
            .fields()
            .iter()
            .cloned()
            .zip(batch.columns().iter().cloned())
            .collect();
        self.reservoir.feed(&cols, rows).map_err(Error::Compute)?;
        // Commit: nothing below can fail.
        let threshold = self.params.categorical_threshold;
        let mut present = HashSet::new();
        for ((s, f), st) in series.iter().zip(schema.fields()).zip(stats) {
            present.insert(s.name().to_string());
            let c = self.column(s.name().as_str());
            match st {
                None => c.outer.nulls(rows),
                Some((outer, inner)) => {
                    if c.dtype.is_none() {
                        c.dtype = Some(s.dtype().clone());
                        c.input_type = pa_name(f.data_type());
                    }
                    c.outer.absorb(outer, threshold);
                    if let Some(inner) = inner {
                        c.inner
                            .get_or_insert_with(Default::default)
                            .absorb(inner, threshold);
                    }
                }
            }
        }
        for c in &mut self.columns {
            if !present.contains(&c.name) {
                c.outer.nulls(rows);
            }
        }
        self.n_rows += rows;
        Ok(())
    }

    /// One row per column, first-seen order (spec §6). The state is kept.
    pub(crate) fn finish(&self) -> Result<RecordBatch> {
        let blocks = self.reservoir.blocks();
        let sampled: u64 = blocks.iter().map(|b| b.rows).sum();
        let rows = self
            .columns
            .par_iter()
            .map(|c| self.row(c, &blocks, sampled))
            .collect::<std::result::Result<Vec<Row>, String>>()
            .map_err(Error::Compute)?;
        assemble("streaming_recommend", &output_fields(), &rows)
            .and_then(|s| export_struct(&s))
            .map_err(|e| Error::Compute(e.to_string()))
    }

    fn row(&self, c: &Column, blocks: &[&Block], sampled: u64) -> std::result::Result<Row, String> {
        let text = |s: &str| AnyValue::StringOwned(s.into());
        let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
        let f = |v: Option<f64>| v.map_or(AnyValue::Null, AnyValue::Float64);
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
            row.resize(output_fields().len(), AnyValue::Null);
            return Ok(row);
        };
        let t = self.params.categorical_threshold;
        let fallback = || estimate(0, 0, 0, 0, &[0; 7], None);
        let err = |e: polars::prelude::PolarsError| e.to_string();

        // Levels from statistics.
        let o = &c.outer;
        let p = o.profile(t);
        let est = o.estimate(t);
        let classic = o
            .classic
            .clone()
            .ok_or("a typed column has absorbed no batch")?;
        let mut lvl = level(dtype, &classic, &p, o, est.unwrap_or_else(fallback), "");
        let inner = match (&c.inner, dtype) {
            (Some(i), PT::List(it) | PT::Array(it, _)) => Some((i, &**it)),
            _ => None,
        };
        let ip = inner.map(|(i, _)| i.profile(t));
        let ilvl = match (inner, &ip) {
            (Some((i, it)), Some(ip)) => {
                let iclassic = i
                    .classic
                    .clone()
                    .ok_or("an inner level has absorbed no batch")?;
                let mut l = level(
                    it,
                    &iclassic,
                    ip,
                    i,
                    i.estimate(t).unwrap_or_else(fallback),
                    "inner: ",
                );
                (l.size_bytes, l.size_note) = original_size(&iclassic, &l.shape(), None, i);
                Some(l)
            }
            _ => None,
        };
        let ishape = ilvl.as_ref().map(Level::shape);
        (lvl.size_bytes, lvl.size_note) = original_size(&classic, &lvl.shape(), ishape.as_ref(), o);
        let iv = match (&ilvl, inner) {
            (Some(il), Some((i, _))) => Some(view_shape(il, i)),
            _ => None,
        };
        let ov = view_shape(&lvl, o);
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
            let s = block_series(b, &c.name, dtype).map_err(err)?;
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
        let d = o.distinct.as_ref();
        row.extend([
            render(&o.lo),
            render(&o.hi),
            p.gcd.map_or(AnyValue::Null, |g| AnyValue::Decimal(g, 0)),
            u(p.sum_len),
            u(o.min_len),
            u(o.max_len),
            u(d.filter(|d| !d.overflowed).map(|d| d.n_unique())),
            d.map_or(AnyValue::Null, |d| AnyValue::Boolean(d.overflowed)),
            f(est.map(|e| e.est_cardinality)),
            f(est.and_then(|e| e.est_low)),
            f(est.and_then(|e| e.est_high)),
            est.map_or(AnyValue::Null, |e| text(e.method.name())),
            AnyValue::UInt64(lvl.size_bytes),
            u(scale(z[0])),
            AnyValue::UInt64(original_polars),
            u(scale(z[1])),
        ]);
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

    use crate::arrow_io::export_struct;
    use crate::recommend::{arrow_cast, describe_and_recommend_impl};

    use super::*;

    pub(crate) fn params() -> Params {
        Params {
            seed: 0,
            zstd_level: 1,
            population_rows: None,
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

    fn one_shot(b: &RecordBatch) -> RecordBatch {
        let out = describe_and_recommend_impl(&import_batch(b).unwrap(), &params()).unwrap();
        export_struct(&out).unwrap()
    }

    fn texts(b: &RecordBatch, name: &str) -> Vec<Option<String>> {
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
            // Kept in its original type: i/7 needs every float64 digit.
            (
                "f_orig",
                Arc::new(Float64Array::from(
                    (0..6)
                        .map(|i| (i % 3 != 1).then(|| (i + 1) as f64 / 7.0))
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
        let reference = one_shot(whole);
        let mut s = streaming();
        batches.iter().for_each(|b| s.add(b).unwrap());
        let out = s.finish().unwrap();
        let columns = texts(&out, "column");
        assert_eq!(columns, texts(&reference, "column"), "{label}");
        for name in REC {
            let (got, want) = (texts(&out, name), texts(&reference, name));
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                // One-shot leaves rec_polars_type null when the original is kept.
                if name == "rec_polars_type" && w.is_none() {
                    continue;
                }
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
        let out = s.finish().unwrap();
        assert!(texts(&out, "rec_arrow_size_zstd_bytes")
            .iter()
            .all(Option::is_none));
        assert_eq!(texts(&out, "n_sampled_blocks")[0].as_deref(), Some("0"));
        assert!(texts(&out, "rec_arrow_type").iter().all(Option::is_some));
    }

    #[test]
    fn overflow_rejects_the_dictionary() {
        let mut p = params();
        p.categorical_threshold = 3;
        let mut s = Streaming::new(p, 1 << 20, 1 << 16);
        let v = [
            Some("a"),
            Some("b"),
            Some("c"),
            Some("d"),
            Some("e"),
            Some("a"),
        ];
        s.add(&batch(vec![("s", strs(&v))])).unwrap();
        let out = s.finish().unwrap();
        assert_eq!(texts(&out, "n_unique"), vec![None]);
        assert_eq!(
            texts(&out, "distinct_overflowed")[0].as_deref(),
            Some("true")
        );
        assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("overflowed"));
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
        assert_eq!(outcomes.value(i), "rejected");
    }

    #[test]
    fn ineligible_null_typed_and_empty() {
        assert_eq!(streaming().finish().unwrap().num_rows(), 0);
        let mut s = streaming();
        s.mark_ineligible("w", "Int128").unwrap();
        s.add(&batch(vec![
            ("a", ints(&[Some(1), Some(2)])),
            ("n", Arc::new(NullArray::new(2)) as ArrayRef),
        ]))
        .unwrap();
        let out = s.finish().unwrap();
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
