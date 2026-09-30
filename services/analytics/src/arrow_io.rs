//! Arrow ↔ Polars for the core (api.rs). Both directions are zero-copy through the
//! Arrow C Data Interface: arrow-rs and polars-arrow bind the same C ABI structs.

use std::sync::Arc;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, RecordBatchReader};
use arrow_schema::{Field, Schema};
use polars::prelude::*;

/// Every column of `batch` as a Series. Field metadata travels with each column, so
/// Polars' own `_PL_CATEGORICAL2` / `_PL_ENUM_VALUES2` keys restore Categorical / Enum.
pub(crate) fn import_batch(batch: &RecordBatch) -> PolarsResult<Vec<Series>> {
    let schema = batch.schema();
    schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(f, a)| import_array(f, a))
        .collect()
}

pub(crate) fn import_array(field: &Field, array: &ArrayRef) -> PolarsResult<Series> {
    let schema = arrow_schema::ffi::FFI_ArrowSchema::try_from(field)
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    let array = arrow_data::ffi::FFI_ArrowArray::new(&array.to_data());
    // SAFETY: the reverse of `export_series`: the same two bindings of the same C ABI
    // structs (see the SAFETY note there), transmuted by value so each `release`
    // callback moves exactly once. `import_array_from_c` wraps the arrow-rs buffers
    // without copying and releases them when the Series drops.
    let (array, schema): (
        polars_arrow::ffi::ArrowArray,
        polars_arrow::ffi::ArrowSchema,
    ) = unsafe {
        (
            std::mem::transmute::<arrow_data::ffi::FFI_ArrowArray, polars_arrow::ffi::ArrowArray>(
                array,
            ),
            std::mem::transmute::<arrow_schema::ffi::FFI_ArrowSchema, polars_arrow::ffi::ArrowSchema>(
                schema,
            ),
        )
    };
    let field = unsafe { polars_arrow::ffi::import_field_from_c(&schema) }?;
    let array = unsafe { polars_arrow::ffi::import_array_from_c(array, field.dtype.clone()) }?;
    Series::try_from((&field, array))
}

/// A Series as one arrow-rs array in the given Polars layout (oldest: LargeUtf8 /
/// LargeList — Arrow's classic layout; newest: Utf8View / BinaryView — Polars'
/// native one). The hand-over is zero-copy; reaching one contiguous array in the
/// requested layout may copy (Utf8View → LargeUtf8, or a multi-chunk Series).
pub(crate) fn export_series(s: &Series, compat: CompatLevel) -> PolarsResult<ArrayRef> {
    // A zero-chunk Series (e.g. Series::new_empty) must be checked before any
    // rechunk(): rechunk concatenates the existing chunks, which panics on an
    // empty chunk list. Rechunk exactly once otherwise (ChunkedArray::rechunk is
    // a Cow, so this is a no-op when `s` is already single-chunk).
    let arr: Box<dyn polars_arrow::array::Array> = if s.n_chunks() == 0 {
        polars_arrow::array::new_empty_array(s.dtype().to_arrow(compat))
    } else {
        s.rechunk().to_arrow(0, compat)
    };
    let field = polars_arrow::datatypes::Field::new(s.name().clone(), arr.dtype().clone(), true);
    let schema = polars_arrow::ffi::export_field_to_c(&field);
    let array = polars_arrow::ffi::export_array_to_c(arr);
    // SAFETY: `polars_arrow::ffi::{ArrowSchema, ArrowArray}` (bindgen-generated, in
    // ffi/generated.rs) and `arrow_schema::ffi::FFI_ArrowSchema` /
    // `arrow_data::ffi::FFI_ArrowArray` are independent bindings of the *same* C ABI
    // structs defined by the Arrow C Data Interface spec. Both sides declare the same
    // fields, in the same order, with the same sizes (format/name/metadata pointers,
    // flags, n_children, children/dictionary pointers, release fn-pointer,
    // private_data pointer for the schema; length/null_count/offset/n_buffers/
    // n_children, buffers/children/dictionary pointers, release fn-pointer,
    // private_data for the array) — a pointer's pointee type never affects its own
    // size/alignment, so the two struct layouts are bit-for-bit identical and
    // `size_of` matches (transmute is a compile error otherwise, which is why no
    // pointer-based fallback was needed here). `transmute` takes each value *by
    // value*: the source binding (`array`/`schema`, both polars-arrow types with a
    // `Drop` that invokes `release`) is consumed without running its destructor, and
    // the destination binding is what carries the `release` callback onward — so
    // ownership (and the single eventual `release` call that frees the boxed
    // polars-arrow `Array`/`Field` behind `private_data`) moves exactly once, with no
    // double-release and no leak. `from_ffi` below takes the transmuted
    // `FFI_ArrowArray` by value and wraps it in an `Arc`, deriving every buffer via
    // `Buffer::from_custom_allocation(ptr, len, owner)` — so the imported arrow-rs
    // array keeps the polars-arrow-owned buffers alive (and calls `release` exactly
    // once, on last-Arc-drop) instead of copying them: this is the zero-copy
    // hand-over. `schema` is only borrowed by `from_ffi` and is dropped normally at
    // the end of this function, releasing the exported `Field` once.
    let (array, schema): (
        arrow_data::ffi::FFI_ArrowArray,
        arrow_schema::ffi::FFI_ArrowSchema,
    ) = unsafe {
        (
            std::mem::transmute::<polars_arrow::ffi::ArrowArray, arrow_data::ffi::FFI_ArrowArray>(
                array,
            ),
            std::mem::transmute::<polars_arrow::ffi::ArrowSchema, arrow_schema::ffi::FFI_ArrowSchema>(
                schema,
            ),
        )
    };
    let data = unsafe { arrow_array::ffi::from_ffi(array, &schema) }
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    Ok(arrow_array::make_array(data))
}

/// A kernel's one-struct-column result as a RecordBatch of its fields (native layout).
pub(crate) fn export_struct(out: &Series) -> PolarsResult<RecordBatch> {
    debug_assert_eq!(out.null_count(), 0);
    let fields = out.struct_()?.fields_as_series();
    let columns = fields
        .iter()
        .map(|s| export_series(s, CompatLevel::newest()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let schema = Schema::new(
        fields
            .iter()
            .zip(&columns)
            .map(|(s, a)| Field::new(s.name().as_str(), a.data_type().clone(), true))
            .collect::<Vec<_>>(),
    );
    let options = RecordBatchOptions::new().with_row_count(Some(out.len()));
    RecordBatch::try_new_with_options(Arc::new(schema), columns, &options)
        .map_err(|e| polars_err!(ComputeError: "result batch: {e}"))
}

/// A whole Arrow C stream as one RecordBatch (kernels expect one chunk per column).
/// The stream is consumed: `*stream` is left released whatever the outcome.
///
/// SAFETY: `stream` points to a valid ArrowArrayStream struct (released or not).
pub(crate) unsafe fn read_stream(
    stream: *mut FFI_ArrowArrayStream,
) -> std::result::Result<RecordBatch, arrow_schema::ArrowError> {
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream) }?;
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    arrow_select::concat::concat_batches(&schema, &batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use arrow_schema::DataType as AT;
    use polars::datatypes::{Categories, FrozenCategories};

    #[test]
    fn export_series_layouts() {
        let a = export_series(
            &Series::new("x".into(), &[Some(1i32), None]),
            CompatLevel::oldest(),
        )
        .unwrap();
        assert_eq!(
            (a.len(), a.null_count(), a.data_type().clone()),
            (2, 1, AT::Int32)
        );
        assert_eq!(
            export_series(&Series::new("x".into(), &["a"]), CompatLevel::oldest())
                .unwrap()
                .data_type(),
            &AT::LargeUtf8
        );
        assert_eq!(
            export_series(&Series::new("x".into(), &["a"]), CompatLevel::newest())
                .unwrap()
                .data_type(),
            &AT::Utf8View
        );
        assert_eq!(
            export_series(
                &Series::new_empty("x".into(), &DataType::String),
                CompatLevel::oldest()
            )
            .unwrap()
            .len(),
            0
        );
        let sliced = export_series(
            &Series::new("x".into(), &[1i64, 2, 3]).slice(1, 2),
            CompatLevel::oldest(),
        )
        .unwrap();
        let ints = sliced
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(ints.values().to_vec(), vec![2, 3]);
    }

    #[test]
    fn sliced_int32_with_nulls_at_unaligned_offset() {
        // 20 values, null every 3rd; slice(3, 10) starts at a validity-bitmap
        // offset (3) that is not a multiple of 8, exercising bit-level slicing.
        let vals: Vec<Option<i32>> = (0..20i32)
            .map(|i| if i % 3 == 0 { None } else { Some(i) })
            .collect();
        let sliced = Series::new("x".into(), vals.as_slice()).slice(3, 10);
        let a = export_series(&sliced, CompatLevel::oldest()).unwrap();
        assert_eq!(a.len(), 10);
        assert_eq!(a.null_count(), 4);
        let ints = a
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        let expected: Vec<Option<i32>> = (3..13i32)
            .map(|i| if i % 3 == 0 { None } else { Some(i) })
            .collect();
        assert_eq!(ints.iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn export_boolean_with_nulls() {
        let a = export_series(
            &Series::new("x".into(), &[Some(true), None, Some(false)]),
            CompatLevel::oldest(),
        )
        .unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        let bools = a
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .unwrap();
        assert_eq!(
            bools.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(false)]
        );
    }

    #[test]
    fn multi_chunk_series_is_rechunked() {
        let mut s = Series::new("x".into(), &[1i64, 2, 3]);
        let s2 = Series::new("x".into(), &[4i64, 5]);
        s.append(&s2).unwrap();
        assert!(s.n_chunks() > 1);
        let a = export_series(&s, CompatLevel::oldest()).unwrap();
        assert_eq!(a.len(), 5);
        let ints = a
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(ints.values().to_vec(), vec![1, 2, 3, 4, 5]);
    }

    /// `columns` as Polars itself exports them: native layout plus Polars' field metadata.
    fn polars_batch(columns: &[Series]) -> RecordBatch {
        let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = columns
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
                (
                    Field::new(s.name().as_str(), a.data_type().clone(), true).with_metadata(md),
                    a,
                )
            })
            .unzip();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn import_restores_every_dtype() {
        let cats = Categories::global();
        let columns = vec![
            Series::new("i".into(), &[Some(1i64), None, Some(-3)]),
            Series::new("u8".into(), &[1u8, 2, 3]),
            Series::new("f".into(), &[1.5f64, -0.0, 2.0]),
            Series::new(
                "s".into(),
                &[Some("a"), None, Some("a string longer than twelve bytes")],
            ),
            Series::new("b".into(), &[true, false, true]),
            Series::new("c".into(), &["x", "y", "x"])
                .cast(&DataType::Categorical(cats.clone(), cats.mapping()))
                .unwrap(),
            Series::new("e".into(), &["b", "a", "b"])
                .cast(&DataType::from_frozen_categories(
                    FrozenCategories::new(["b", "a"]).unwrap(),
                ))
                .unwrap(),
            Series::new(
                "l".into(),
                [
                    Some(Series::new("".into(), &[1i32, 2])),
                    None,
                    Some(Series::new("".into(), &[3i32])),
                ],
            ),
            Int128Chunked::from_slice("d".into(), &[120, -5, 0])
                .into_decimal_unchecked(Some(10), 2)
                .into_series(),
            Series::new("t".into(), &[0i64, 1, 2])
                .cast(&DataType::Datetime(
                    TimeUnit::Microseconds,
                    Some(TimeZone::UTC),
                ))
                .unwrap(),
        ];
        let back = import_batch(&polars_batch(&columns)).unwrap();
        assert_eq!(back.len(), columns.len());
        for (a, b) in columns.iter().zip(&back) {
            assert_eq!(
                (a.name(), a.dtype().to_string()),
                (b.name(), b.dtype().to_string())
            );
            let text = |s: &Series| match s.dtype() {
                DataType::Categorical(..) | DataType::Enum(..) => {
                    s.cast(&DataType::String).unwrap()
                }
                _ => s.clone(),
            };
            assert!(text(a).equals_missing(&text(b)), "{}", a.name());
        }
    }

    #[test]
    fn import_of_zero_rows_and_zero_columns() {
        let back = import_batch(&polars_batch(&[Series::new_empty(
            "x".into(),
            &DataType::Int64,
        )]))
        .unwrap();
        assert_eq!((back[0].len(), back[0].dtype()), (0, &DataType::Int64));
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(0)),
        )
        .unwrap();
        assert!(import_batch(&empty).unwrap().is_empty());
    }

    #[test]
    fn plain_arrow_imports_without_polars_metadata() {
        let batch = RecordBatch::try_from_iter([
            (
                "n",
                Arc::new(arrow_array::Int32Array::from(vec![1, 2])) as ArrayRef,
            ),
            (
                "s",
                Arc::new(arrow_array::StringArray::from(vec!["a", "b"])) as ArrayRef,
            ),
        ])
        .unwrap();
        let back = import_batch(&batch).unwrap();
        assert_eq!(
            (back[0].dtype(), back[1].dtype()),
            (&DataType::Int32, &DataType::String)
        );
    }

    #[test]
    fn import_of_sliced_arrays_with_nulls_at_unaligned_offset() {
        // Int32Array: 20 values, null every 3rd; slice(3, 3) starts at a
        // validity-bitmap offset (3) that is not a multiple of 8. StringArray:
        // slice(1, 3), offset 1. Both plain arrow-rs arrays, no Polars metadata.
        let vals: Vec<Option<i32>> = (0..20i32)
            .map(|i| if i % 3 == 0 { None } else { Some(i) })
            .collect();
        let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vals));
        let ints = ints.slice(3, 3);
        let strs: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
            Some("a"),
            None,
            Some("c"),
            Some("d"),
        ]));
        let strs = strs.slice(1, 3);
        let batch = RecordBatch::try_from_iter([("n", ints), ("s", strs)]).unwrap();
        let back = import_batch(&batch).unwrap();
        assert_eq!(
            back[0].i32().unwrap().iter().collect::<Vec<_>>(),
            vec![None, Some(4), Some(5)]
        );
        assert_eq!(
            back[1].str().unwrap().iter().collect::<Vec<_>>(),
            vec![None, Some("c"), Some("d")]
        );
    }

    #[test]
    fn plain_dictionary_arrays_import_as_categorical() {
        // No Polars metadata (no `_PL_CATEGORICAL2` key): polars 0.51 still recognises
        // a plain arrow-rs Dictionary<_, Utf8> array (any integer key width) and
        // imports it as Categorical, keyed to a fresh, ad hoc category set.
        use arrow_array::types::{Int32Type, Int8Type};
        use arrow_array::DictionaryArray;
        let d32: DictionaryArray<Int32Type> =
            vec![Some("a"), Some("b"), Some("a")].into_iter().collect();
        let d8: DictionaryArray<Int8Type> = vec![Some("x"), None, Some("y")].into_iter().collect();
        let batch = RecordBatch::try_from_iter([
            ("d32", Arc::new(d32) as ArrayRef),
            ("d8", Arc::new(d8) as ArrayRef),
        ])
        .unwrap();
        let back = import_batch(&batch).unwrap();
        for s in &back {
            assert!(
                matches!(s.dtype(), DataType::Categorical(..)),
                "{}: {}",
                s.name(),
                s.dtype()
            );
        }
        let text = |s: &Series| -> Vec<Option<String>> {
            s.cast(&DataType::String)
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .map(|v| v.map(str::to_string))
                .collect()
        };
        assert_eq!(
            text(&back[0]),
            vec![Some("a".into()), Some("b".into()), Some("a".into())]
        );
        assert_eq!(
            text(&back[1]),
            vec![Some("x".into()), None, Some("y".into())]
        );
    }

    #[test]
    fn struct_result_becomes_a_flat_batch() {
        let fields = [
            Series::new("col".into(), &["a", "b"]),
            Series::new("v".into(), &[1.0f64, 2.0]),
        ];
        let out = StructChunked::from_series("r".into(), 2, fields.iter())
            .unwrap()
            .into_series();
        let batch = export_struct(&out).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().field(0).name(), "col");
        assert_eq!(batch.schema().field(1).data_type(), &AT::Float64);
        assert!(import_batch(&batch).unwrap()[1].equals(&fields[1]));
    }

    #[test]
    fn read_stream_concatenates_batches_and_consumes_the_stream() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;
        use arrow_array::{Int64Array, RecordBatchIterator};

        let batch = |v: Vec<i64>| {
            RecordBatch::try_from_iter([("a", Arc::new(Int64Array::from(v)) as ArrayRef)]).unwrap()
        };
        let (b1, b2) = (batch(vec![1, 2]), batch(vec![3]));
        let schema = b1.schema();
        let mut stream =
            FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(b1), Ok(b2)], schema)));
        let out = unsafe { read_stream(&mut stream) }.unwrap();
        assert_eq!(out.num_rows(), 3);
        // The stream was moved out and left released, so a second read fails.
        assert!(unsafe { read_stream(&mut stream) }.is_err());
    }
}
