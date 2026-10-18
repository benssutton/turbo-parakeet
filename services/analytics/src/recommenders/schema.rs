//! The recommenders' output columns (spec 2026-10-04 §4).

use crate::common::ipc_sizes::SIZE_FIELDS;
use crate::techniques::describe::value_fields;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    new_empty_array, new_null_array, Array, ArrayRef, MapArray, RecordBatch, RecordBatchOptions,
    StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType as AT, Field, Fields, Schema, SchemaRef};
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
        ("nullable", PT::Boolean),
        ("lossy_formatting", PT::Boolean),
        ("datafusion_type", PT::String),
        ("datafusion_size_bytes", PT::UInt64),
        ("datafusion_size_zstd_bytes", PT::UInt64),
        ("polars_type", PT::String),
        ("polars_size_bytes", PT::UInt64),
        ("polars_size_zstd_bytes", PT::UInt64),
        ("candidates", PT::List(Box::new(candidate_type()))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}

/// Entropy and exemplar columns (spec 2026-10-18 §3, §4), after `n_midnight`. The two
/// `*_id` columns are Null-typed here; `with_ids` replaces them when there are id columns.
pub(crate) fn profile_fields() -> Vec<(String, PT)> {
    [
        ("entropy", PT::Float64),
        ("entropy_low", PT::Float64),
        ("entropy_high", PT::Float64),
        ("entropy_method", PT::String),
        ("exemplar_row", PT::UInt64),
        ("exemplar_id", PT::Null),
        ("exemplar_value", PT::String),
        ("null_exemplar_row", PT::UInt64),
        ("null_exemplar_id", PT::Null),
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
    f.extend(profile_fields());
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
        t => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "top_k: expected a large list, got {t}"
            )))
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

/// One output row's id values: (exemplar, null exemplar) one-row batches, None = no such row.
pub(crate) type IdCell = (Option<RecordBatch>, Option<RecordBatch>);

/// Replaces the Null-typed `exemplar_id` / `null_exemplar_id` columns by Structs of the id
/// columns (spec 2026-10-18 §4.2). `ids[i]` = (exemplar, null exemplar) one-row batches of
/// output row i, all with `schema`; None gives a null struct.
pub(crate) fn with_ids(
    b: RecordBatch,
    schema: &SchemaRef,
    ids: &[IdCell],
) -> Result<RecordBatch, ArrowError> {
    let fields = schema.fields().clone();
    let struct_type = AT::Struct(fields.clone());
    let column = |pick: &dyn Fn(&IdCell) -> Option<&RecordBatch>| -> Result<ArrayRef, ArrowError> {
        let null_row = new_null_array(&struct_type, 1);
        let parts: Vec<ArrayRef> = ids
            .iter()
            .map(|cell| match pick(cell) {
                Some(r) => StructArray::try_new(fields.clone(), r.columns().to_vec(), None)
                    .map(|s| Arc::new(s) as ArrayRef),
                None => Ok(null_row.clone()),
            })
            .collect::<Result<_, _>>()?;
        if parts.is_empty() {
            return Ok(new_empty_array(&struct_type));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
        arrow_select::concat::concat(&refs)
    };
    let exemplar = column(&|c| c.0.as_ref())?;
    let null_exemplar = column(&|c| c.1.as_ref())?;
    let out_schema = b.schema();
    let mut out_fields = Vec::with_capacity(b.num_columns());
    let mut out_columns = Vec::with_capacity(b.num_columns());
    for (f, c) in out_schema.fields().iter().zip(b.columns()) {
        let replaced = match f.name().as_str() {
            "exemplar_id" => Some(exemplar.clone()),
            "null_exemplar_id" => Some(null_exemplar.clone()),
            _ => None,
        };
        match replaced {
            Some(a) => {
                out_fields.push(Field::new(f.name(), struct_type.clone(), true));
                out_columns.push(a);
            }
            None => {
                out_fields.push(f.as_ref().clone());
                out_columns.push(c.clone());
            }
        }
    }
    let options = RecordBatchOptions::new().with_row_count(Some(b.num_rows()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(out_fields)), out_columns, &options)
}

/// The id columns' sub-schema of `schema` (fields made nullable), checked (spec §4.2):
/// duplicates and missing names are an Err message the caller maps to InvalidInput.
pub(crate) fn id_schema(schema: &Schema, id_columns: &[String]) -> Result<SchemaRef, String> {
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::with_capacity(id_columns.len());
    for name in id_columns {
        if !seen.insert(name.as_str()) {
            return Err(format!("duplicate id column {name:?}"));
        }
        let f = schema
            .field_with_name(name)
            .map_err(|_| format!("id column {name:?} is not in the input"))?;
        fields.push(f.clone().with_nullable(true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, LargeListArray, StringViewArray, UInt64Array};

    #[test]
    fn with_maps_rewraps_a_sliced_list_with_nulls() {
        let entry = Fields::from(vec![
            Field::new("key", AT::Utf8View, true),
            Field::new("value", AT::UInt64, true),
        ]);
        let entries = StructArray::new(
            entry.clone(),
            vec![
                Arc::new(StringViewArray::from(vec!["a", "b", "c"])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
            None,
        );
        // Rows: [a, b], null, [c]; sliced to the last two (first offset 2).
        let list = LargeListArray::new(
            Arc::new(Field::new("item", AT::Struct(entry), true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 2, 2, 3])),
            Arc::new(entries),
            Some(vec![true, false, true].into()),
        )
        .slice(1, 2);
        let other: ArrayRef = Arc::new(Int64Array::from(vec![7, 8]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("top_k", list.data_type().clone(), true),
            Field::new("x", AT::Int64, false),
        ]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(list), other.clone()]).unwrap();
        let out = with_maps(b).unwrap();
        let kv = Fields::from(vec![
            Field::new("key", AT::Utf8, false),
            Field::new("value", AT::UInt64, false),
        ]);
        let want = AT::Map(
            Arc::new(Field::new("entries", AT::Struct(kv), false)),
            false,
        );
        assert_eq!(out.schema().field(0).data_type(), &want);
        let m = out.column(0).as_map();
        assert!(m.is_null(0));
        let e = m.value(1);
        assert_eq!(e.len(), 1);
        assert_eq!(e.column(0).as_string::<i32>().value(0), "c");
        assert_eq!(
            e.column(1)
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            3
        );
        assert_eq!(out.column(1), &other);
        assert_eq!(out.schema().field(1).name(), "x");
    }

    #[test]
    fn with_maps_refuses_other_types() {
        let b = RecordBatch::try_from_iter([(
            "top_k",
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        )])
        .unwrap();
        assert!(with_maps(b).is_err());
    }

    #[test]
    fn with_ids_builds_typed_structs_and_null_cells() {
        let ids_schema = Arc::new(Schema::new(vec![
            Field::new("id", AT::Int64, true),
            Field::new("tag", AT::Utf8, true),
        ]));
        let one = |id: i64, tag: &str| {
            RecordBatch::try_new(
                ids_schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![id])) as ArrayRef,
                    Arc::new(arrow_array::StringArray::from(vec![tag])) as ArrayRef,
                ],
            )
            .unwrap()
        };
        let other: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", AT::Int64, false),
            Field::new("exemplar_id", AT::Null, true),
            Field::new("null_exemplar_id", AT::Null, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                other.clone(),
                new_null_array(&AT::Null, 2),
                new_null_array(&AT::Null, 2),
            ],
        )
        .unwrap();
        let cells = vec![(Some(one(7, "a")), None), (None, Some(one(9, "b")))];
        let out = with_ids(b, &ids_schema, &cells).unwrap();
        assert_eq!(out.column(0), &other);
        let e = out.column(1).as_struct();
        assert!(e.is_valid(0) && e.is_null(1));
        assert_eq!(
            e.column(0)
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(0),
            7
        );
        assert_eq!(e.column(1).as_string::<i32>().value(0), "a");
        let n = out.column(2).as_struct();
        assert!(n.is_null(0) && n.is_valid(1));
        assert_eq!(n.column(1).as_string::<i32>().value(1), "b");
        assert_eq!(
            out.schema().field(1).data_type(),
            &AT::Struct(ids_schema.fields().clone())
        );
    }

    #[test]
    fn with_ids_handles_zero_rows() {
        let ids_schema = Arc::new(Schema::new(vec![Field::new("id", AT::Int64, true)]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("exemplar_id", AT::Null, true),
            Field::new("null_exemplar_id", AT::Null, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![new_null_array(&AT::Null, 0), new_null_array(&AT::Null, 0)],
        )
        .unwrap();
        let out = with_ids(b, &ids_schema, &[]).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert!(matches!(out.schema().field(0).data_type(), AT::Struct(_)));
    }

    #[test]
    fn id_schema_checks_names_and_keeps_order() {
        let schema = Schema::new(vec![
            Field::new("a", AT::Int64, false),
            Field::new("b", AT::Utf8, false),
        ]);
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let got = id_schema(&schema, &names(&["b", "a"])).unwrap();
        assert_eq!(got.field(0).name(), "b");
        assert_eq!(got.field(1).name(), "a");
        assert!(got.fields().iter().all(|f| f.is_nullable()));
        assert!(id_schema(&schema, &names(&["a", "a"]))
            .unwrap_err()
            .contains("duplicate id column"));
        assert!(id_schema(&schema, &names(&["z"]))
            .unwrap_err()
            .contains("not in the input"));
    }
}
