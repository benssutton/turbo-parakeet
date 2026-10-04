//! One-shot recommender (spec
//! docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md): one frame. `add`
//! collects each column's exact statistics and sizes (recommend.rs `prepare`); `result`
//! recommends from them — every candidate cast, verified on the actual rows and
//! measured — once, and caches the table. Same output as the streaming recommender,
//! without its `first_row` and sample columns (recommend.rs `recommender_fields`).

use std::collections::HashSet;
use std::sync::OnceLock;

use arrow_array::RecordBatch;
use arrow_schema::DataType as AT;
use polars::prelude::{AnyValue, DataType as PT, PolarsResult};
use rayon::prelude::*;

use crate::bindings::api::{compute, Error, Result};
use crate::common::arrow_io::{export_struct, import_batch};
use crate::recommenders::engine::{pa_name, prepare, recommender_fields, Params, Prepared};
use crate::recommenders::streaming::holds_nested_null;
use crate::techniques::describe::{assemble, Row};

const ADDED: &str = "a frame has already been added: create a new OneShotRecommender";

/// One column of the added frame.
enum Entry {
    Computed(Box<Prepared>),
    /// Null-typed (`n_null` = the rows) or holding Null below its top level (`n_null`
    /// None), as the streaming recommender reports them.
    Ineligible {
        name: String,
        dtype: String,
        n_null: Option<u64>,
    },
}

pub(crate) struct OneShot {
    params: Params,
    n_rows: u64,
    /// (name, dtype) of columns the caller cannot send (Int128 / UInt128, Object).
    marked: Vec<(String, String)>,
    /// None until a frame is added.
    entries: Option<Vec<Entry>>,
    result: OnceLock<RecordBatch>,
}

impl OneShot {
    pub(crate) fn new(params: Params) -> Self {
        OneShot {
            params,
            n_rows: 0,
            marked: Vec::new(),
            entries: None,
            result: OnceLock::new(),
        }
    }

    /// Lists a column the caller cannot send as ineligible; before `add` only. Marking
    /// the same (name, dtype) again is a no-op, so a failed `add` can be retried.
    pub(crate) fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        if self.entries.is_some() {
            return Err(Error::InvalidInput(ADDED.into()));
        }
        match self.marked.iter().find(|(n, _)| n == name) {
            Some((_, d)) if d == dtype => return Ok(()),
            Some(_) => return Err(Error::InvalidInput(format!("duplicate column {name:?}"))),
            None => {}
        }
        self.marked.push((name.to_string(), dtype.to_string()));
        self.result = OnceLock::new();
        Ok(())
    }

    /// Adds the frame and collects every column's statistics (in parallel). A second
    /// call is InvalidInput. On error the state is unchanged. `batch` must be valid
    /// Arrow (see the api module docs).
    pub(crate) fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.entries.is_some() {
            return Err(Error::InvalidInput(ADDED.into()));
        }
        let schema = batch.schema();
        let mut seen: HashSet<&str> = self.marked.iter().map(|(n, _)| n.as_str()).collect();
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
        let types: Vec<&AT> = schema.fields().iter().map(|f| f.data_type()).collect();
        let rows = batch.num_rows() as u64;
        let entries = series
            .into_par_iter()
            .zip(types)
            .map(|(s, t)| {
                let name = s.name().to_string();
                if s.dtype() == &PT::Null {
                    return Ok(Entry::Ineligible {
                        name,
                        dtype: pa_name(t),
                        n_null: Some(rows),
                    });
                }
                if holds_nested_null(s.dtype()) {
                    return Ok(Entry::Ineligible {
                        name,
                        dtype: pa_name(t),
                        n_null: None,
                    });
                }
                prepare(s, t, &self.params).map(|p| Entry::Computed(Box::new(p)))
            })
            .collect::<PolarsResult<Vec<_>>>()
            .map_err(compute)?;
        self.n_rows = rows;
        self.entries = Some(entries);
        self.result = OnceLock::new();
        Ok(())
    }

    /// One row per column — marked columns first, then the frame's in order; an empty
    /// table before `add`. Computed on the first call, then cached.
    pub(crate) fn result(&self) -> Result<RecordBatch> {
        if let Some(b) = self.result.get() {
            return Ok(b.clone());
        }
        let batch = self.build()?;
        Ok(self.result.get_or_init(|| batch).clone())
    }

    fn build(&self) -> Result<RecordBatch> {
        let fields = recommender_fields(false);
        let ineligible = |name: &str, dtype: &str, n_null: Option<u64>| -> Row {
            let mut row: Row = vec![
                AnyValue::StringOwned(name.into()),
                AnyValue::StringOwned("ineligible".into()),
                AnyValue::StringOwned(dtype.into()),
                AnyValue::UInt64(self.n_rows),
                n_null.map_or(AnyValue::Null, AnyValue::UInt64),
            ];
            row.resize(fields.len(), AnyValue::Null);
            row
        };
        let mut rows: Vec<Row> = self
            .marked
            .iter()
            .map(|(n, d)| ineligible(n, d, None))
            .collect();
        let entries = self.entries.as_deref().unwrap_or(&[]);
        rows.extend(
            entries
                .par_iter()
                .map(|e| match e {
                    Entry::Computed(p) => p.row(&self.params),
                    Entry::Ineligible {
                        name,
                        dtype,
                        n_null,
                    } => Ok(ineligible(name, dtype, *n_null)),
                })
                .collect::<PolarsResult<Vec<Row>>>()
                .map_err(compute)?,
        );
        assemble("recommend", &fields, &rows)
            .and_then(|s| export_struct(&s))
            .map_err(compute)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int64Type;
    use arrow_array::{new_null_array, ArrayRef, Float64Array, ListArray};
    use arrow_schema::DataType as AT;

    use super::*;
    use crate::recommenders::streaming::tests::{batch, ints, params, strs, texts};

    fn toy() -> RecordBatch {
        let list = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            None,
            Some(vec![Some(2), None]),
        ]);
        batch(vec![
            ("a", ints(&[Some(0), Some(5), None])),
            ("s", strs(&[Some("x"), Some("y"), Some("x")])),
            ("l", Arc::new(list) as ArrayRef),
        ])
    }

    fn added(b: &RecordBatch) -> OneShot {
        let mut r = OneShot::new(params());
        r.add(b).unwrap();
        r
    }

    fn text(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    #[test]
    fn output_is_the_shared_one_shot_schema() {
        let out = added(&toy()).result().unwrap();
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let want: Vec<String> = recommender_fields(false)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, want);
        assert_eq!(texts(&out, "column"), [text("a"), text("s"), text("l")]);
        assert_eq!(texts(&out, "status"), vec![text("computed"); 3]);
        assert_eq!(texts(&out, "dtype")[..2], [text("int64"), text("string")]);
        assert_eq!(
            texts(&out, "rec_arrow_type")[..2],
            [text("uint8"), text("string")]
        );
        assert_eq!(texts(&out, "n_rows"), vec![text("3"); 3]);
        assert_eq!(texts(&out, "inner_n_values")[2], text("3"));
        assert_eq!(texts(&out, "rec_polars_type").iter().flatten().count(), 3);
    }

    #[test]
    fn result_is_cached_and_repeatable() {
        let r = added(&toy());
        let first = r.result().unwrap();
        assert!(r.result.get().is_some());
        assert_eq!(r.result().unwrap(), first);
    }

    #[test]
    fn result_before_add_is_empty_then_follows_the_frame() {
        let mut r = OneShot::new(params());
        let empty = r.result().unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_eq!(empty.num_columns(), recommender_fields(false).len());
        r.add(&toy()).unwrap();
        assert_eq!(r.result().unwrap().num_rows(), 3);
    }

    #[test]
    fn a_second_add_is_invalid_and_keeps_the_first() {
        let mut r = added(&toy());
        let err = r.add(&batch(vec![("z", ints(&[Some(1)]))])).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("already been added")),
            "{err:?}"
        );
        assert_eq!(texts(&r.result().unwrap(), "column").len(), 3);
        assert!(matches!(
            r.mark_ineligible("w", "Int128"),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn a_failed_add_can_be_retried() {
        let mut r = OneShot::new(params());
        r.mark_ineligible("w", "Int128").unwrap();
        let dup = batch(vec![("a", ints(&[Some(1)])), ("a", ints(&[Some(2)]))]);
        assert!(matches!(r.add(&dup), Err(Error::InvalidInput(_))));
        // Python re-marks before retrying: the same mark again is a no-op.
        r.mark_ineligible("w", "Int128").unwrap();
        assert!(matches!(
            r.mark_ineligible("w", "Object"),
            Err(Error::InvalidInput(_))
        ));
        r.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
        assert_eq!(
            texts(&r.result().unwrap(), "column"),
            [text("w"), text("a")]
        );
    }

    #[test]
    fn ineligible_rows_marked_first_null_typed_in_place() {
        let mut r = OneShot::new(params());
        r.mark_ineligible("w", "Int128").unwrap();
        r.add(&batch(vec![
            ("n", new_null_array(&AT::Null, 2)),
            ("a", ints(&[Some(1), Some(2)])),
        ]))
        .unwrap();
        let out = r.result().unwrap();
        assert_eq!(texts(&out, "column"), [text("w"), text("n"), text("a")]);
        assert_eq!(
            texts(&out, "status"),
            [text("ineligible"), text("ineligible"), text("computed")]
        );
        assert_eq!(texts(&out, "dtype")[..2], [text("Int128"), text("null")]);
        assert_eq!(texts(&out, "n_rows"), vec![text("2"); 3]);
        assert_eq!(texts(&out, "n_null")[..2], [None, text("2")]);
        assert_eq!(texts(&out, "rec_arrow_type")[..2], [None, None]);
    }

    #[test]
    fn a_column_named_twice_is_invalid() {
        let mut r = OneShot::new(params());
        r.mark_ineligible("a", "Int128").unwrap();
        assert!(matches!(
            r.add(&batch(vec![("a", ints(&[Some(1)]))])),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn kept_original_names_its_polars_type() {
        // Beyond float32's range with ~200 integer digits: no narrower type holds it.
        let f: ArrayRef = Arc::new(Float64Array::from(vec![3.3e200 / 7.0, 9.9e200 / 7.0]));
        let out = added(&batch(vec![("f", f)])).result().unwrap();
        assert_eq!(texts(&out, "rec_arrow_type"), [text("double")]);
        assert_eq!(texts(&out, "rec_polars_type"), [text("Float64")]);
    }
}
