//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (streaming/partial.rs) and a block reservoir
//! (streaming/reservoir.rs) are kept; `result` recommends from them at any point. Its output
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

use arrow_array::cast::AsArray;
use arrow_array::{make_array, Array, ArrayRef, RecordBatch};
use arrow_data::ArrayData;
use arrow_schema::{DataType as AT, SchemaRef};
use polars::prelude::{AnyValue, CompatLevel, DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::common::arrow_io::{export_series, export_struct, import_batch};
use crate::common::clickhouse_sizes::ChLayout;
use crate::common::error::{Error, Result};
use crate::common::ipc_sizes::{classic_layout, layout_sizes, sizes_of};
use crate::recommenders::engine::{
    clickhouse_choice_over, entropy_cells, enum_categories, exemplar_cells, exemplar_value,
    ipc_choice, pa_name, pick_by_stats, pick_list_by_stats, pl_name, rec_row, rescale, skip_choice,
    to_polars_layout, top_k_cell, Entropy, EntropyEstimator, Exemplars, Kind, Level, Params, Pick,
    Rec, RunningStats, SkipCandidate, Target,
};
use crate::recommenders::schema::{id_schema, recommender_fields, with_ids, with_maps, IdCell};
use crate::recommenders::streaming::partial::{BatchStats, LevelStats};
use crate::recommenders::streaming::reservoir::{Block, Reservoir};
use crate::techniques::ams_entropy::{name_key, sketch_seed, MARGINAL};
use crate::techniques::describe::conclusions::{conclude, Conclusions};
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
    /// First non-null row (and its text) and first null row.
    pub exemplars: Exemplars,
    /// The id columns' values at the exemplar row and the null exemplar row (one-row copies;
    /// spec 2026-10-18 §4.2); None while the row is, and always without `id_columns`.
    pub ids: IdCell,
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
    block_rows: u64,
    /// The id columns' schema, from the first batch; None without `id_columns`.
    id_schema: Option<SchemaRef>,
    /// The id columns' values at global row 0 (a one-row copy): the null exemplar of a
    /// column that first appears later.
    first_ids: Option<RecordBatch>,
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

/// A batch column's first non-null row (with its text) and first null row.
type Firsts = (Option<(u64, Option<String>)>, Option<u64>);

/// The blocks' arrays as one array. Dictionary arrays sharing one dictionary keep it (the
/// library concat would offset the keys of copies it does not recognise as equal).
fn concat_blocks(blocks: &[ArrayRef]) -> std::result::Result<ArrayRef, String> {
    let err = |e: arrow_schema::ArrowError| e.to_string();
    if let Some(d0) = blocks[0].as_any_dictionary_opt() {
        let values = d0.values().to_data();
        let same = blocks.iter().all(|b| {
            b.data_type() == blocks[0].data_type()
                && b.as_any_dictionary_opt()
                    .is_some_and(|d| d.values().to_data() == values)
        });
        if same {
            let keys: Vec<&dyn Array> = blocks
                .iter()
                .filter_map(|b| b.as_any_dictionary_opt())
                .map(|d| d.keys() as &dyn Array)
                .collect();
            let keys = arrow_select::concat::concat(&keys).map_err(err)?.to_data();
            let data = ArrayData::builder(blocks[0].data_type().clone())
                .len(keys.len())
                .add_buffer(keys.buffers()[0].clone())
                .nulls(keys.nulls().cloned())
                .add_child_data(values)
                .build()
                .map_err(err)?;
            return Ok(make_array(data));
        }
    }
    let refs: Vec<&dyn Array> = blocks.iter().map(|a| a.as_ref()).collect();
    arrow_select::concat::concat(&refs).map_err(err)
}

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
            block_rows,
            n_rows: 0,
            columns: Vec::new(),
            index: HashMap::new(),
            id_schema: None,
            first_ids: None,
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
                    exemplars: Exemplars {
                        // A late column's earlier rows are nulls: its first null is row 0.
                        null_row: (self.n_rows > 0).then_some(0),
                        ..Exemplars::default()
                    },
                    ids: (None, self.first_ids.clone().filter(|_| self.n_rows > 0)),
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
        let ams_seed = (self.params.entropy_estimator == EntropyEstimator::Ams)
            .then(|| sketch_seed(seed, MARGINAL, name_key(s.name().as_str())));
        let outer = BatchStats::of(s, self.n_rows, seed, top, ams_seed)?;
        let inner = match flatten(s)? {
            Some(v) => {
                let prev = c.and_then(|c| c.inner.as_ref());
                let top = prev.and_then(LevelStats::sample_top);
                Some(BatchStats::of(
                    &v,
                    prev.map_or(0, |l| l.n),
                    seed,
                    top,
                    None,
                )?)
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
        // Exemplars (spec 2026-10-18 §4): per series, the first non-null row (and its text)
        // and first null row this batch adds, for columns still missing them.
        let start = self.n_rows;
        let firsts: Vec<Firsts> = series
            .par_iter()
            .map(|(s, _)| {
                let known = self
                    .index
                    .get(s.name().as_str())
                    .map(|&i| &self.columns[i].exemplars);
                let need_row = known.is_none_or(|e| e.row.is_none());
                // A column new at start > 0 already has null row 0.
                let need_null = match known {
                    None => start == 0,
                    Some(e) => e.null_row.is_none(),
                };
                let nulls = s.is_null();
                let row = if need_row {
                    nulls.iter().position(|v| v == Some(false))
                } else {
                    None
                };
                let row = row
                    .map(|i| exemplar_value(s, i as u64).map(|v| (start + i as u64, v)))
                    .transpose()?;
                let null = if need_null {
                    nulls
                        .iter()
                        .position(|v| v == Some(true))
                        .map(|i| start + i as u64)
                } else {
                    None
                };
                Ok((row, null))
            })
            .collect::<PolarsResult<_>>()
            .map_err(|e| Error::Compute(e.to_string()))?;
        let rows = batch.num_rows() as u64;
        // Id values (spec §4.2): one-row copies at the exemplar rows found above, and at
        // this batch's row 0 for columns absent from it. Fallible, so before the commit.
        let ids_schema = if self.params.id_columns.is_empty() {
            None
        } else {
            let s = id_schema(&schema, &self.params.id_columns).map_err(Error::InvalidInput)?;
            if let Some(prev) = &self.id_schema {
                if let Some((a, b)) = prev
                    .fields()
                    .iter()
                    .zip(s.fields())
                    .find(|(a, b)| a.data_type() != b.data_type())
                {
                    return Err(Error::InvalidInput(format!(
                        "id column {:?}: type changed from {} to {}",
                        a.name(),
                        pa_name(a.data_type()),
                        pa_name(b.data_type())
                    )));
                }
            }
            Some(s)
        };
        let ids_at = |row: u64| -> Result<RecordBatch> {
            let s = ids_schema.as_ref().expect("id columns are set");
            let cols = s
                .fields()
                .iter()
                .map(|f| {
                    let c = batch
                        .column_by_name(f.name())
                        .expect("checked by id_schema");
                    partial::copy_rows(c, (row - start) as usize, 1)
                })
                .collect::<std::result::Result<Vec<_>, String>>()
                .map_err(Error::Compute)?;
            RecordBatch::try_new(s.clone(), cols).map_err(|e| Error::Compute(e.to_string()))
        };
        let (new_ids, row0_ids): (Vec<IdCell>, Option<RecordBatch>) = if ids_schema.is_some() {
            let cells = firsts
                .iter()
                .map(|(row, null)| {
                    Ok((
                        row.as_ref().map(|(r, _)| ids_at(*r)).transpose()?,
                        null.map(&ids_at).transpose()?,
                    ))
                })
                .collect::<Result<Vec<IdCell>>>()?;
            (cells, (rows > 0).then(|| ids_at(start)).transpose()?)
        } else {
            (vec![(None, None); firsts.len()], None)
        };
        self.reservoir.feed(&cols, rows).map_err(Error::Compute)?;
        // Commit: nothing below can fail.
        if ids_schema.is_some() {
            self.id_schema = ids_schema.clone();
            if self.first_ids.is_none() {
                self.first_ids = row0_ids.clone();
            }
        }
        let threshold = self.params.categorical_threshold;
        for ((s, f), _) in &nested_null {
            let c = self.column(s.name().as_str());
            c.ineligible = true;
            c.input_type = pa_name(f.data_type());
        }
        // Serially: create new columns and adopt types; then absorb every column's
        // statistics in parallel (column names are unique, so each slot is one column).
        let mut placed = Vec::with_capacity(stats.len());
        for ((((s, f), st), (row, null)), nid) in series.iter().zip(stats).zip(firsts).zip(new_ids)
        {
            let c = self.column(s.name().as_str());
            if let (None, Some((r, v))) = (c.exemplars.row, row) {
                c.exemplars.row = Some(r);
                c.exemplars.value = v;
                c.ids.0 = nid.0;
            }
            if c.exemplars.null_row.is_none() && null.is_some() {
                c.exemplars.null_row = null;
                c.ids.1 = nid.1;
            }
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
        if rows > 0 {
            // Columns absent from the batch: their first null may be here.
            let present: HashSet<&str> = series.iter().map(|(s, _)| s.name().as_str()).collect();
            for c in self
                .columns
                .iter_mut()
                .filter(|c| !present.contains(c.name.as_str()))
            {
                if c.exemplars.null_row.is_none() {
                    c.exemplars.null_row = Some(start);
                    c.ids.1 = row0_ids.clone();
                }
            }
        }
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
        let batch = assemble("streaming_recommend", &recommender_fields(true), &rows)
            .and_then(|s| export_struct(&s))
            .map_err(|e| Error::Compute(e.to_string()))
            .and_then(|b| with_maps(b).map_err(|e| Error::Compute(e.to_string())))?;
        let Some(ids_schema) = &self.id_schema else {
            return Ok(batch);
        };
        let cells: Vec<IdCell> = self
            .columns
            .iter()
            .map(|c| {
                if c.ineligible || c.dtype.is_none() {
                    (None, None)
                } else {
                    c.ids.clone()
                }
            })
            .collect();
        with_ids(batch, ids_schema, &cells).map_err(|e| Error::Compute(e.to_string()))
    }

    fn row(&self, c: &Column, blocks: &[&Block], sampled: u64) -> std::result::Result<Row, String> {
        let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
        let dtype = c.dtype.as_ref().filter(|_| !c.ineligible);
        let mut row = self.row_head(c, dtype.is_some());
        let Some(dtype) = dtype else {
            row.resize(recommender_fields(true).len(), AnyValue::Null);
            return Ok(row);
        };
        let t = self.params.categorical_threshold;

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
                Some(inner_level(i, it, ip, ic, c.views_input.1)?)
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
        let pick = self.pick_of(dtype, &lvl, ilvl.as_ref())?;

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
        let Sampled {
            z,
            arrow_blocks,
            polars_blocks,
        } = self.sampled_blocks(
            c,
            dtype,
            blocks,
            &pick,
            &key,
            &p,
            inner.map(|(_, it)| it).zip(ip.as_ref()),
        )?;
        let scale = |x: u64| {
            (sampled > 0).then(|| (x as f64 * self.n_rows as f64 / sampled as f64).round() as u64)
        };
        let scale_up = |x: u64| scale(x).unwrap_or(x);

        let rec_t = pick.target.arrow_type();
        // A kept Enum (at any depth) is named with its categories.
        let enum_values = enum_categories(dtype);
        let nullable = nullable(&pick.target, o, inner.map(|(i, _)| i));

        // Arrow IPC compression and the ClickHouse column, from the sampled blocks.
        let distinct_ratio = distinct_ratio(o, &p);
        let has_sample = sampled > 0 && !arrow_blocks.is_empty();
        let mut whole: Option<ArrayRef> = None;
        let (arrow_lz4, arrow_ipc, polars_lz4, polars_ipc) = if has_sample {
            // The sampled blocks as one array: the IPC buffers compress as the column's do.
            let (a, p_) = (
                concat_blocks(&arrow_blocks)?,
                concat_blocks(&polars_blocks)?,
            );
            let (al, ai) = ipc_choice(&[a.as_ref()], &scale_up, distinct_ratio, &self.params)?;
            whole = Some(a);
            let (pl, pi) = ipc_choice(&[p_.as_ref()], &scale_up, distinct_ratio, &self.params)?;
            (Some(al), Some(ai), Some(pl), Some(pi))
        } else {
            (None, None, None, None)
        };
        let (running, gcd_note) = running_stats(o, dtype, inner, &rec_t);
        // A LowCardinality dictionary is written once per part: emulate it on the whole sample
        // (one dictionary), not per block, which would count it once per block.
        let low_card = enum_values.is_none()
            && matches!(&rec_t, AT::Dictionary(_, v) if matches!(**v, AT::Utf8 | AT::LargeUtf8));
        let a: Vec<&dyn Array> = match (&whole, low_card) {
            (Some(w), true) => vec![w.as_ref()],
            _ => arrow_blocks.iter().map(|a| a.as_ref()).collect(),
        };
        let ch = clickhouse_choice_over(
            &rec_t,
            nullable,
            enum_values.as_deref(),
            running.as_ref(),
            &gcd_note,
            has_sample.then_some(a.as_slice()),
            &scale_up,
            distinct_ratio,
            &self.params,
        )?;
        let (skip_index, skip_candidates) = self.stream_skip_index(
            whole.as_ref(),
            ch.ty.as_deref().zip(ch.layout.as_ref()),
            enum_values.as_deref(),
            ch.size,
            &arrow_blocks,
            sampled,
        );
        let rec = Rec {
            nullable,
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
            arrow_lz4,
            arrow_ipc,
            polars_lz4,
            polars_ipc,
            ch_type: ch.ty,
            ch_chain: ch.chain,
            ch_size: ch.size,
            ch_candidates: ch.candidates,
            skip_index,
            skip_candidates,
        };
        row.extend(p.row(&oc));
        row.push(u(o.n_midnight));
        let entropy = self.level_entropy(o, &p, &oc);
        row.extend(entropy_cells(Some(&entropy)));
        row.extend(exemplar_cells(Some(&c.exemplars)));
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
        let k = self.params.top_k;
        row.push(top_k_cell(p.ranking.as_ref(), &rec.candidates, "", k));
        let inner_ranking = ip.as_ref().and_then(|ip| ip.ranking.as_ref());
        row.push(top_k_cell(inner_ranking, &rec.candidates, "inner: ", k));
        row.extend(rec_row(&rec));
        row.extend([
            AnyValue::UInt64(sampled),
            AnyValue::UInt64(blocks.len() as u64),
        ]);
        Ok(row)
    }

    /// The leading cells: `column, status, dtype, first_row, n_rows, n_null`.
    fn row_head(&self, c: &Column, computed: bool) -> Row {
        let text = |s: &str| AnyValue::StringOwned(s.into());
        vec![
            text(&c.name),
            text(if computed { "computed" } else { "ineligible" }),
            text(&c.input_type),
            AnyValue::UInt64(c.first_row),
            AnyValue::UInt64(self.n_rows),
            if c.ineligible {
                AnyValue::Null
            } else {
                AnyValue::UInt64(c.outer.n_null)
            },
        ]
    }

    /// The choice, proven by statistics: a list's inner level is picked with it.
    fn pick_of(
        &self,
        dtype: &PT,
        lvl: &Level,
        ilvl: Option<&Level>,
    ) -> std::result::Result<Pick, String> {
        let Some(il) = ilvl else {
            return pick_by_stats(lvl, &self.params);
        };
        let width = match dtype {
            PT::Array(_, w) => Some(*w as i32),
            _ => None,
        };
        pick_list_by_stats(lvl, il, width, &self.params)
    }

    /// ZSTD sizes of the sampled blocks (original Arrow, original Polars, recommended Arrow,
    /// recommended Polars) and each block's recommended Arrow array and Polars layout, kept
    /// for the compression and ClickHouse choices (the sample is bounded by `reservoir_rows`).
    /// The cross-check: every block is recast to the pick.
    #[allow(clippy::too_many_arguments)]
    fn sampled_blocks(
        &self,
        c: &Column,
        dtype: &PT,
        blocks: &[&Block],
        pick: &Pick,
        key: &AT,
        p: &Profile,
        inner: Option<(&PT, &Profile)>,
    ) -> std::result::Result<Sampled, String> {
        let err = |e: polars::prelude::PolarsError| e.to_string();
        let original = matches!(pick.target, Target::Original(_));
        let mut z = [0u64; 4];
        let mut arrow_blocks: Vec<ArrayRef> = Vec::with_capacity(blocks.len());
        let mut polars_blocks: Vec<ArrayRef> = Vec::with_capacity(blocks.len());
        for b in blocks {
            let s = block_series(b, &c.name, dtype, c.views_input).map_err(err)?;
            let bc = classic_layout(&s).map_err(err)?;
            let sz = sizes_of(&s, &bc, self.params.zstd_level).map_err(err)?;
            z[0] += sz[1];
            z[1] += sz[3];
            if original {
                z[2] += sz[1];
                z[3] += sz[3];
                polars_blocks.push(export_series(&s, CompatLevel::newest()).map_err(err)?);
                arrow_blocks.push(bc);
                continue;
            }
            let a = recast(pick, dtype, &bc, p, inner).map_err(|e| {
                format!(
                    "cross-check {:?} → {}: {e}",
                    c.name,
                    pa_name(&pick.target.arrow_type())
                )
            })?;
            let polars = to_polars_layout(&a, key)?;
            let rz =
                layout_sizes(a.as_ref(), polars.as_ref(), self.params.zstd_level).map_err(err)?;
            z[2] += rz[1];
            z[3] += rz[3];
            arrow_blocks.push(a);
            polars_blocks.push(polars);
        }
        Ok(Sampled {
            z,
            arrow_blocks,
            polars_blocks,
        })
    }

    /// Skip indexes: windows are scanned inside each sampled block, so a granularity whose
    /// window exceeds `block_rows` is pruned (one row per granularity, kind MinMax, naming
    /// the reason for every kind). Index sizes scale to the stream like `ch_size`.
    fn stream_skip_index(
        &self,
        whole: Option<&ArrayRef>,
        ch: Option<(&str, &ChLayout)>,
        enum_values: Option<&[String]>,
        ch_size: Option<u64>,
        arrow_blocks: &[ArrayRef],
        sampled: u64,
    ) -> (Option<String>, Vec<SkipCandidate>) {
        let (Some(w), Some(_)) = (whole, ch) else {
            return (None, Vec::new());
        };
        let mut sp = self.params.skip.clone();
        let mut extra = Vec::new();
        sp.granularities.retain(|&g| {
            let rows = u64::from(g) * 8_192;
            if rows <= self.block_rows {
                return true;
            }
            extra.push(SkipCandidate {
                measured: None,
                kind: Kind::MinMax,
                granularity: g,
                outcome: "pruned",
                evidence: format!(
                    "window of {rows} rows exceeds block_rows {}",
                    self.block_rows
                ),
            });
            false
        });
        let mut at = 0;
        let segments: Vec<(usize, usize)> = arrow_blocks
            .iter()
            .map(|b| {
                at += b.len();
                (at - b.len(), at)
            })
            .collect();
        skip_choice(
            w.as_ref(),
            ch,
            enum_values,
            ch_size,
            &segments,
            &sp,
            self.n_rows as f64 / sampled as f64,
            extra,
        )
    }

    /// The level's entropy: exact while the sample is, else AMS or the sample's estimate.
    fn level_entropy(&self, o: &LevelStats, p: &Profile, oc: &Conclusions) -> Entropy {
        match &o.sample {
            None => Entropy::from_counts(std::iter::empty(), o.n_null, o.n),
            Some(d) if d.is_exact() => {
                Entropy::from_counts(d.held_counts().into_iter(), o.n_null, o.n)
            }
            Some(d) => match (&o.ams, self.params.entropy_estimator) {
                (Some(a), EntropyEstimator::Ams) => {
                    let (h, se) = a.estimate_with_se();
                    Entropy::from_ams(h, se, o.n)
                }
                _ => {
                    let dd = sample_distinct(o, p, oc);
                    Entropy::from_sample(&d.held_counts(), dd, o.n_null, o.n)
                }
            },
        }
    }
}

/// A list's inner level and its original size (the view shape it measures with).
fn inner_level<'a>(
    i: &LevelStats,
    it: &'a PT,
    ip: &'a Profile,
    ic: &Conclusions,
    views_input: bool,
) -> std::result::Result<(Level<'a>, ViewShape), String> {
    let iclassic = i
        .classic
        .clone()
        .ok_or("an inner level has absorbed no batch")?;
    let mut l = level(it, &iclassic, ip, i, ic.est, "inner: ");
    let iv = view_shape(&l, i, views_input);
    (l.size_bytes, l.size_note) = original_size(&iclassic, &iv, None, i);
    Ok((l, iv))
}

/// Distinct values over non-null values (1 when there are none).
fn distinct_ratio(o: &LevelStats, p: &Profile) -> f64 {
    let non_null = o.n - o.n_null;
    if non_null == 0 {
        1.0
    } else {
        p.freq.n_unique as f64 / non_null as f64
    }
}

/// The recommended values' running statistics (a list recast to scalars reads its inner
/// level) and the note for a GCD unknown from them.
fn running_stats(
    o: &LevelStats,
    dtype: &PT,
    inner: Option<(&LevelStats, &PT)>,
    rec_t: &AT,
) -> (Option<RunningStats>, String) {
    let (stats_level, src) = match (inner, rec_t) {
        (Some((i, it)), t)
            if !matches!(t, AT::List(_) | AT::LargeList(_) | AT::FixedSizeList(..)) =>
        {
            (i, it)
        }
        _ => (o, dtype),
    };
    let running = rescale(src, rec_t).map(|f| RunningStats {
        range: stats_level.int_range().map(|(a, b)| (a / f, b / f)),
        gcd: stats_level.gcd.map(|g| g.unsigned_abs() / f.unsigned_abs()),
        max_abs: stats_level.int_range().map_or(0, |(a, b)| {
            (a / f).unsigned_abs().max((b / f).unsigned_abs())
        }),
    });
    (
        running,
        format!("GCD unknown in streaming for values derived from {src}"),
    )
}

/// The sampled blocks' sizes and recommended arrays (`Streaming::sampled_blocks`).
struct Sampled {
    z: [u64; 4],
    arrow_blocks: Vec<ArrayRef>,
    polars_blocks: Vec<ArrayRef>,
}

/// The distinct count of a sampled level with its +-3 sigma interval: HyperLogLog's
/// (accurate; `est_cardinality`'s Schnabel overshoots at d/n < 0.5), else the conclusions'.
fn sample_distinct(o: &LevelStats, p: &Profile, oc: &Conclusions) -> (f64, f64, f64) {
    let non_null = (o.n - o.n_null) as f64;
    match p.freq.hll {
        Some((e, se, seen)) => {
            let seen = seen as f64;
            (
                e.max(seen).min(non_null),
                (e * (1.0 - 3.0 * se)).max(seen).min(non_null),
                (e * (1.0 + 3.0 * se)).max(seen).min(non_null),
            )
        }
        None => (
            oc.est.est_cardinality,
            oc.est.est_low.unwrap_or(oc.est.est_cardinality),
            oc.est.est_high.unwrap_or(oc.est.est_cardinality),
        ),
    }
}
