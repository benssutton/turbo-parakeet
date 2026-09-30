//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (partial.rs) and a block reservoir
//! (reservoir.rs) are kept; `finish` recommends from them at any point (Task 10).

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use polars::prelude::{DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::api::{Error, Result};
use crate::arrow_io::import_batch;
use crate::describe::flatten;
use crate::partial::{BatchStats, LevelStats};
use crate::recommend::{is_text, pa_name, Params};
use crate::reservoir::Reservoir;

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
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, NullArray, StringArray};

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
}
