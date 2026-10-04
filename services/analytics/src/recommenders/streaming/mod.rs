//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (partial.rs) and a block reservoir
//! (reservoir.rs) are kept; `result` recommends from them at any point. Its output
//! shares Describe's value columns and conclusions (parity spec
//! docs/superpowers/specs/2026-10-01-recommender-parity-design.md, §7).

mod sizing;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) mod distinct_sample;
pub(crate) mod partial;
pub(crate) mod reservoir;

use sizing::*;

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use arrow_schema::DataType as AT;
use polars::prelude::{AnyValue, DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::common::arrow_io::{export_struct, import_batch};
use crate::common::error::{Error, Result};
use crate::common::ipc_sizes::{classic_layout, ipc_body_bytes, sizes_of};
use crate::recommenders::engine::{
    enum_categories, pa_name, pick_by_stats, pick_list_by_stats, pl_name, rec_row,
    to_polars_layout, Params, Rec, Target,
};
use crate::recommenders::schema::recommender_fields;
use crate::recommenders::streaming::partial::{BatchStats, LevelStats};
use crate::recommenders::streaming::reservoir::{Block, Reservoir};
use crate::techniques::describe::conclusions::conclude;
use crate::techniques::describe::{assemble, flatten, value_fields, Row};

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
