//! Arrow ↔ Polars for the core (api.rs). Both directions are zero-copy through the
//! Arrow C Data Interface: arrow-rs and polars-arrow bind the same C ABI structs.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use polars::prelude::*;

/// Every column of `batch` as a Series. Field metadata travels with each column, so
/// Polars' own `_PL_CATEGORICAL2` / `_PL_ENUM_VALUES2` keys restore Categorical / Enum.
pub(crate) fn import_batch(batch: &RecordBatch) -> PolarsResult<Vec<Series>> {
    let schema = batch.schema();
    schema.fields().iter().zip(batch.columns()).map(|(f, a)| import_array(f, a)).collect()
}

fn import_array(field: &Field, array: &ArrayRef) -> PolarsResult<Series> {
    let schema = arrow_schema::ffi::FFI_ArrowSchema::try_from(field)
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    let array = arrow_data::ffi::FFI_ArrowArray::new(&array.to_data());
    // SAFETY: the reverse of `export_series`: the same two bindings of the same C ABI
    // structs (see the SAFETY note there), transmuted by value so each `release`
    // callback moves exactly once. `import_array_from_c` wraps the arrow-rs buffers
    // without copying and releases them when the Series drops.
    let (array, schema): (polars_arrow::ffi::ArrowArray, polars_arrow::ffi::ArrowSchema) =
        unsafe { (std::mem::transmute(array), std::mem::transmute(schema)) };
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
    let (array, schema): (arrow_data::ffi::FFI_ArrowArray, arrow_schema::ffi::FFI_ArrowSchema) =
        unsafe { (std::mem::transmute(array), std::mem::transmute(schema)) };
    let data = unsafe { arrow_array::ffi::from_ffi(array, &schema) }
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    Ok(arrow_array::make_array(data))
}

/// A kernel's one-struct-column result as a RecordBatch of its fields (native layout).
pub(crate) fn export_struct(out: &Series) -> PolarsResult<RecordBatch> {
    let fields = out.struct_()?.fields_as_series();
    let columns = fields.iter().map(|s| export_series(s, CompatLevel::newest())).collect::<PolarsResult<Vec<_>>>()?;
    let schema = Schema::new(
        fields.iter().zip(&columns).map(|(s, a)| Field::new(s.name().as_str(), a.data_type().clone(), true)).collect::<Vec<_>>(),
    );
    let options = RecordBatchOptions::new().with_row_count(Some(out.len()));
    RecordBatch::try_new_with_options(Arc::new(schema), columns, &options)
        .map_err(|e| polars_err!(ComputeError: "result batch: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use arrow_schema::DataType as AT;
    use polars::datatypes::{Categories, FrozenCategories};

    #[test]
    fn series_cross_to_arrow_rs() {
        let a = export_series(&Series::new("x".into(), &[Some(1i32), None]), CompatLevel::oldest()).unwrap();
        assert_eq!((a.len(), a.null_count(), a.data_type().clone()), (2, 1, AT::Int32));
        assert_eq!(export_series(&Series::new("x".into(), &["a"]), CompatLevel::oldest()).unwrap().data_type(), &AT::LargeUtf8);
        assert_eq!(export_series(&Series::new("x".into(), &["a"]), CompatLevel::newest()).unwrap().data_type(), &AT::Utf8View);
        assert_eq!(export_series(&Series::new_empty("x".into(), &DataType::String), CompatLevel::oldest()).unwrap().len(), 0);
        let sliced = export_series(&Series::new("x".into(), &[1i64, 2, 3]).slice(1, 2), CompatLevel::oldest()).unwrap();
        let ints = sliced.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        assert_eq!(ints.values().to_vec(), vec![2, 3]);
    }

    #[test]
    fn sliced_int32_with_nulls_at_unaligned_offset() {
        // 20 values, null every 3rd; slice(3, 10) starts at a validity-bitmap
        // offset (3) that is not a multiple of 8, exercising bit-level slicing.
        let vals: Vec<Option<i32>> = (0..20i32).map(|i| if i % 3 == 0 { None } else { Some(i) }).collect();
        let sliced = Series::new("x".into(), vals.as_slice()).slice(3, 10);
        let a = export_series(&sliced, CompatLevel::oldest()).unwrap();
        assert_eq!(a.len(), 10);
        assert_eq!(a.null_count(), 4);
        let ints = a.as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let expected: Vec<Option<i32>> = (3..13i32).map(|i| if i % 3 == 0 { None } else { Some(i) }).collect();
        assert_eq!(ints.iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn boolean_series_round_trips() {
        let a = export_series(
            &Series::new("x".into(), &[Some(true), None, Some(false)]),
            CompatLevel::oldest(),
        )
        .unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        let bools = a.as_any().downcast_ref::<arrow_array::BooleanArray>().unwrap();
        assert_eq!(bools.iter().collect::<Vec<_>>(), vec![Some(true), None, Some(false)]);
    }

    #[test]
    fn multi_chunk_series_is_rechunked() {
        let mut s = Series::new("x".into(), &[1i64, 2, 3]);
        let s2 = Series::new("x".into(), &[4i64, 5]);
        s.append(&s2).unwrap();
        assert!(s.n_chunks() > 1);
        let a = export_series(&s, CompatLevel::oldest()).unwrap();
        assert_eq!(a.len(), 5);
        let ints = a.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
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
                    .map(|m| m.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
                    .unwrap_or_default();
                (Field::new(s.name().as_str(), a.data_type().clone(), true).with_metadata(md), a)
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
            Series::new("s".into(), &[Some("a"), None, Some("a string longer than twelve bytes")]),
            Series::new("b".into(), &[true, false, true]),
            Series::new("c".into(), &["x", "y", "x"]).cast(&DataType::Categorical(cats.clone(), cats.mapping())).unwrap(),
            Series::new("e".into(), &["b", "a", "b"])
                .cast(&DataType::from_frozen_categories(FrozenCategories::new(["b", "a"]).unwrap()))
                .unwrap(),
            Series::new("l".into(), [Some(Series::new("".into(), &[1i32, 2])), None, Some(Series::new("".into(), &[3i32]))]),
            Int128Chunked::from_slice("d".into(), &[120, -5, 0]).into_decimal_unchecked(Some(10), 2).into_series(),
            Series::new("t".into(), &[0i64, 1, 2]).cast(&DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))).unwrap(),
        ];
        let back = import_batch(&polars_batch(&columns)).unwrap();
        assert_eq!(back.len(), columns.len());
        for (a, b) in columns.iter().zip(&back) {
            assert_eq!((a.name(), a.dtype().to_string()), (b.name(), b.dtype().to_string()));
            let text = |s: &Series| match s.dtype() {
                DataType::Categorical(..) | DataType::Enum(..) => s.cast(&DataType::String).unwrap(),
                _ => s.clone(),
            };
            assert!(text(a).equals_missing(&text(b)), "{}", a.name());
        }
    }

    #[test]
    fn import_of_zero_rows_and_zero_columns() {
        let back = import_batch(&polars_batch(&[Series::new_empty("x".into(), &DataType::Int64)])).unwrap();
        assert_eq!((back[0].len(), back[0].dtype()), (0, &DataType::Int64));
        let empty = RecordBatch::try_new_with_options(Arc::new(Schema::empty()), vec![], &RecordBatchOptions::new().with_row_count(Some(0))).unwrap();
        assert!(import_batch(&empty).unwrap().is_empty());
    }

    #[test]
    fn plain_arrow_imports_without_polars_metadata() {
        let batch = RecordBatch::try_from_iter([
            ("n", Arc::new(arrow_array::Int32Array::from(vec![1, 2])) as ArrayRef),
            ("s", Arc::new(arrow_array::StringArray::from(vec!["a", "b"])) as ArrayRef),
        ])
        .unwrap();
        let back = import_batch(&batch).unwrap();
        assert_eq!((back[0].dtype(), back[1].dtype()), (&DataType::Int32, &DataType::String));
    }

    #[test]
    fn struct_result_becomes_a_flat_batch() {
        let fields = [Series::new("col".into(), &["a", "b"]), Series::new("v".into(), &[1.0f64, 2.0])];
        let out = StructChunked::from_series("r".into(), 2, fields.iter()).unwrap().into_series();
        let batch = export_struct(&out).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().field(0).name(), "col");
        assert_eq!(batch.schema().field(1).data_type(), &AT::Float64);
        assert!(import_batch(&batch).unwrap()[1].equals(&fields[1]));
    }
}
