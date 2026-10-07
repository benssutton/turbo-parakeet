//! The recommenders' output columns (spec 2026-10-04 §4).

use crate::common::ipc_sizes::SIZE_FIELDS;
use crate::techniques::describe::value_fields;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, MapArray, RecordBatch, RecordBatchOptions, StructArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType as AT, Field, Fields, Schema};
use polars::prelude::{DataType as PT, Field as PField};

pub(crate) fn candidate_type() -> PT {
    PT::Struct(vec![
        PField::new("arrow_type".into(), PT::String),
        PField::new("rule".into(), PT::String),
        PField::new("evidence".into(), PT::String),
        PField::new("predicted_bytes".into(), PT::UInt64),
        PField::new("projected_population_bytes".into(), PT::Float64),
        PField::new("outcome".into(), PT::String),
        PField::new("reason".into(), PT::String),
    ])
}

pub(crate) fn rec_fields() -> Vec<(String, PT)> {
    [
        ("rec_nullable", PT::Boolean),
        ("rec_arrow_type", PT::String),
        ("rec_arrow_size_bytes", PT::UInt64),
        ("rec_arrow_size_zstd_bytes", PT::UInt64),
        ("rec_polars_type", PT::String),
        ("rec_polars_size_bytes", PT::UInt64),
        ("rec_polars_size_zstd_bytes", PT::UInt64),
        ("rec_lossy_formatting", PT::Boolean),
        ("rec_candidates", PT::List(Box::new(candidate_type()))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}

/// The recommenders' output columns (spec 2026-10-04 §4), one definition for both:
/// `streaming` adds `first_row` after `dtype` and the sample counts at the end.
pub(crate) fn recommender_fields(streaming: bool) -> Vec<(String, PT)> {
    let mut f: Vec<(String, PT)> = vec![
        ("column".into(), PT::String),
        ("status".into(), PT::String),
        ("dtype".into(), PT::String),
    ];
    if streaming {
        f.push(("first_row".into(), PT::UInt64));
    }
    f.push(("n_rows".into(), PT::UInt64));
    f.push(("n_null".into(), PT::UInt64));
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), PT::UInt64));
    f.extend(SIZE_FIELDS.iter().map(|n| (n.to_string(), PT::UInt64)));
    f.push(("inner_n_values".into(), PT::UInt64));
    f.push(("inner_n_null".into(), PT::UInt64));
    f.extend(
        value_fields()
            .into_iter()
            .map(|(n, d)| (format!("inner_{n}"), d)),
    );
    f.extend(TOP_K_FIELDS.iter().map(|n| (n.to_string(), top_k_type())));
    f.extend(rec_fields());
    if streaming {
        f.push(("n_sampled_rows".into(), PT::UInt64));
        f.push(("n_sampled_blocks".into(), PT::UInt64));
    }
    f
}

/// The map columns (spec 2026-10-07 §3), in output order.
pub(crate) const TOP_K_FIELDS: [&str; 2] = ["top_k", "inner_top_k"];

/// `top_k` / `inner_top_k` as assembled through Polars, which has no map type: lists of
/// (key, value) structs; `with_maps` makes them `Map<Utf8, UInt64>`.
pub(crate) fn top_k_type() -> PT {
    PT::List(Box::new(PT::Struct(vec![
        PField::new("key".into(), PT::String),
        PField::new("value".into(), PT::UInt64),
    ])))
}

/// The finished result with `TOP_K_FIELDS` as Arrow `Map<Utf8, UInt64>`. Polars exports
/// them as lists of structs with view-string keys: offsets (to i32) and keys (to Utf8)
/// are copied, at most K entries per level.
pub(crate) fn with_maps(b: RecordBatch) -> Result<RecordBatch, ArrowError> {
    let schema = b.schema();
    let mut fields = Vec::with_capacity(b.num_columns());
    let mut columns = Vec::with_capacity(b.num_columns());
    for (f, c) in schema.fields().iter().zip(b.columns()) {
        if TOP_K_FIELDS.contains(&f.name().as_str()) {
            let m = to_map(c)?;
            fields.push(Field::new(f.name(), m.data_type().clone(), true));
            columns.push(m);
        } else {
            fields.push(f.as_ref().clone());
            columns.push(c.clone());
        }
    }
    let options = RecordBatchOptions::new().with_row_count(Some(b.num_rows()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
}

fn to_map(c: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let (entries, offsets, nulls) = match c.data_type() {
        AT::LargeList(_) => {
            let l = c.as_list::<i64>();
            let o = l
                .offsets()
                .iter()
                .map(|&o| i32::try_from(o))
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|_| ArrowError::ComputeError("top_k: over 2^31 entries".into()))?;
            (l.values().as_struct().clone(), o, l.nulls().cloned())
        }
        _ => {
            let l = c.as_list::<i32>();
            (
                l.values().as_struct().clone(),
                l.offsets().to_vec(),
                l.nulls().cloned(),
            )
        }
    };
    let keys = ::arrow_cast::cast(entries.column(0).as_ref(), &AT::Utf8)?;
    let entry_fields = Fields::from(vec![
        Field::new("key", AT::Utf8, false),
        Field::new("value", AT::UInt64, false),
    ]);
    let entries = StructArray::try_new(
        entry_fields.clone(),
        vec![keys, entries.column(1).clone()],
        None,
    )?;
    let field = Arc::new(Field::new("entries", AT::Struct(entry_fields), false));
    let map = MapArray::try_new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        entries,
        nulls,
        false,
    )?;
    Ok(Arc::new(map))
}
