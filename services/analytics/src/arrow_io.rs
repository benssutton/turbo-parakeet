//! Arrow ↔ Polars for the core (api.rs). Both directions are zero-copy through the
//! Arrow C Data Interface: arrow-rs and polars-arrow bind the same C ABI structs.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Arc;

use arrow_array::ffi::{from_ffi_and_data_type, FFI_ArrowArray};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, StructArray};
use arrow_data::{layout, ArrayData};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, DataType as AT, Field, Schema, SchemaRef};
use polars::prelude::*;
use rayon::prelude::*;

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
        ffi_safe(s.rechunk().to_arrow(0, compat))
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
        .and_then(|d| validate_tree(&d, false).map(|()| d))
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    Ok(arrow_array::make_array(data))
}

/// `a` with every Struct / FixedSizeList level (at any depth) exportable: polars-arrow
/// slices their children eagerly but exports them with `offset = validity.offset()`,
/// so a sliced one with nulls exports as invalid Arrow (child too short), which
/// arrow-rs panics on. Such a level gets its validity rebuilt from bit 0 (O(len)
/// bits, whatever its offset); every other buffer is shared.
fn ffi_safe(a: Box<dyn polars_arrow::array::Array>) -> Box<dyn polars_arrow::array::Array> {
    use polars_arrow::array::{Array as _, FixedSizeListArray, ListArray, StructArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::PhysicalType as P;
    let fresh = |v: Option<&Bitmap>| {
        v.map(|b| {
            // The absolute bit offset is not public, and `as_slice().1` is only that
            // offset mod 8, so always copy: bytes when byte-aligned, bits otherwise.
            match b.as_slice() {
                (bytes, 0, len) => Bitmap::from_u8_slice(bytes, len),
                _ => b.iter().collect(),
            }
        })
    };
    match a.dtype().to_physical_type() {
        P::Struct => {
            let s = a.as_any().downcast_ref::<StructArray>().unwrap();
            let values = s.values().iter().map(|v| ffi_safe(v.clone())).collect();
            Box::new(StructArray::new(
                s.dtype().clone(),
                s.len(),
                values,
                fresh(s.validity()),
            ))
        }
        P::FixedSizeList => {
            let l = a.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            Box::new(FixedSizeListArray::new(
                l.dtype().clone(),
                l.len(),
                ffi_safe(l.values().clone()),
                fresh(l.validity()),
            ))
        }
        P::List => {
            let l = a.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            Box::new(ListArray::new(
                l.dtype().clone(),
                l.offsets().clone(),
                ffi_safe(l.values().clone()),
                l.validity().cloned(),
            ))
        }
        P::LargeList => {
            let l = a.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            Box::new(ListArray::new(
                l.dtype().clone(),
                l.offsets().clone(),
                ffi_safe(l.values().clone()),
                l.validity().cloned(),
            ))
        }
        _ => a,
    }
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
) -> std::result::Result<RecordBatch, ArrowError> {
    let reader = unsafe { CheckedReader::from_raw(stream) }?;
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    arrow_select::concat::concat_batches(&schema, &batches)
}

// ── checked import ───────────────────────────────────────────────────────────
//
// arrow-rs imports C data without validation (`from_ffi` skips it) and then builds
// arrays that assert their structure: a malformed input panics, and the release
// profile aborts on panic. So each batch is checked before arrow-rs builds an array
// over it: first the raw C structs (buffer and child counts, null pointers, negative
// lengths, a view array's variadic buffers), then, once imported, every column
// (`check_column`): a type Polars cannot import is refused (Decimal256 aborts it), and
// arrow-rs's full validation runs at every level — structure (buffer sizes, child
// lengths against the parent's offset + length) and values (null counts, every offset,
// UTF-8, dictionary keys in range: Polars asserts on the keys and reads out of bounds
// on invalid UTF-8). The values pass is O(n) (~3 ms for a 1M-row string column).

/// The C Stream Interface struct, field for field: arrow-rs keeps its copy's
/// callbacks private.
#[repr(C)]
pub(crate) struct RawStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut RawStream, *mut FFI_ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut RawStream, *mut FFI_ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut RawStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut RawStream)>,
    pub private_data: *mut c_void,
}

/// The C Data Interface array struct, field for field (arrow-rs keeps its fields
/// private), for the checks arrow-rs would otherwise assert.
#[repr(C)]
struct RawArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut RawArray,
    dictionary: *mut RawArray,
    release: Option<unsafe extern "C" fn(*mut RawArray)>,
    private_data: *mut c_void,
}

/// The stream's last error message, else `what` failed.
///
/// SAFETY: `s` points to a valid, unreleased ArrowArrayStream.
unsafe fn last_error(s: *mut RawStream, what: &str) -> ArrowError {
    let detail = unsafe { (*s).get_last_error }.and_then(|get_last_error| {
        let msg = unsafe { get_last_error(s) };
        // SAFETY: a non-null `get_last_error` result is a NUL-terminated C string
        // owned by the stream, live until the next call on it.
        (!msg.is_null()).then(|| {
            unsafe { CStr::from_ptr(msg) }
                .to_string_lossy()
                .into_owned()
        })
    });
    ArrowError::CDataInterface(match detail {
        Some(msg) => format!("arrow stream: {what} failed: {msg}"),
        None => format!("arrow stream: {what} failed"),
    })
}

/// An Arrow C stream read batch by batch, each batch checked before arrow-rs builds
/// arrays over it (see above). Drops (releases) the stream when dropped.
pub(crate) struct CheckedReader {
    stream: FFI_ArrowArrayStream,
    schema: SchemaRef,
}

impl CheckedReader {
    /// SAFETY: `raw` points to a valid ArrowArrayStream struct (released or not); it is
    /// moved out, leaving a released one behind.
    pub(crate) unsafe fn from_raw(
        raw: *mut FFI_ArrowArrayStream,
    ) -> std::result::Result<Self, ArrowError> {
        let mut stream = unsafe { std::ptr::replace(raw, FFI_ArrowArrayStream::empty()) };
        let s = (&mut stream as *mut FFI_ArrowArrayStream).cast::<RawStream>();
        let released = || ArrowError::CDataInterface("arrow stream already released".into());
        // A released stream has `release == NULL`; its other members are undefined.
        if unsafe { (*s).release }.is_none() {
            return Err(released());
        }
        let get_schema = unsafe { (*s).get_schema }.ok_or_else(|| {
            ArrowError::CDataInterface("arrow stream: no get_schema callback".into())
        })?;
        let mut schema = FFI_ArrowSchema::empty();
        if unsafe { get_schema(s, &mut schema) } != 0 {
            return Err(unsafe { last_error(s, "get_schema") });
        }
        let schema = Arc::new(Schema::try_from(&schema)?);
        // A stream with no batches never reaches the per-batch check.
        check_schema(&schema)?;
        Ok(CheckedReader { stream, schema })
    }

    pub(crate) fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for CheckedReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let s = (&mut self.stream as *mut FFI_ArrowArrayStream).cast::<RawStream>();
        let Some(get_next) = (unsafe { (*s).get_next }) else {
            // A NULL callback is a broken producer, not the end of the stream.
            return Some(Err(ArrowError::CDataInterface(
                "arrow stream: no get_next callback".into(),
            )));
        };
        let mut array = FFI_ArrowArray::empty();
        if unsafe { get_next(s, &mut array) } != 0 {
            return Some(Err(unsafe { last_error(s, "get_next") }));
        }
        if array.is_released() {
            return None;
        }
        // SAFETY: the stream hands out arrays of its schema's struct type.
        Some(unsafe { import_checked(array, &self.schema) })
    }
}

/// One stream batch, checked (see above), as a RecordBatch of `schema`.
///
/// SAFETY: `array` is a C Data Interface array of `schema`'s struct type whose buffers
/// are as long as its lengths and offsets say (the one thing that cannot be checked).
pub(crate) unsafe fn import_checked(
    array: FFI_ArrowArray,
    schema: &SchemaRef,
) -> std::result::Result<RecordBatch, ArrowError> {
    let dt = AT::Struct(schema.fields().clone());
    check_raw(&array as *const FFI_ArrowArray as *const RawArray, &dt)?;
    let data = unsafe { from_ffi_and_data_type(array, dt) }?;
    data.validate_data()?;
    check_columns(schema.fields(), data.child_data())?;
    let len = data.len();
    RecordBatch::try_new_with_options(
        schema.clone(),
        StructArray::from(data).into_parts().1,
        &RecordBatchOptions::new().with_row_count(Some(len)),
    )
}

fn malformed(dt: &AT, what: String) -> ArrowError {
    ArrowError::CDataInterface(format!("malformed {dt} array: {what}"))
}

/// The raw C array `a` against its type `dt`, recursively: what arrow-rs's import
/// asserts or trusts before it validates anything.
fn check_raw(a: *const RawArray, dt: &AT) -> std::result::Result<(), ArrowError> {
    // SAFETY: `a` is non-null (checked by the caller for children) and points to a C
    // Data Interface array struct.
    let a = unsafe { &*a };
    if a.length < 0 || a.offset < 0 || a.n_buffers < 0 || a.n_children < 0 {
        return Err(malformed(dt, "negative length, offset or count".into()));
    }
    let l = layout(dt);
    let fixed = l.buffers.len() + usize::from(l.can_contain_null_mask);
    let n = a.n_buffers as usize;
    // A view type also carries its variadic buffers and their lengths.
    let ok = if l.variadic { n > fixed } else { n == fixed };
    if !ok {
        return Err(malformed(
            dt,
            format!(
                "{n} buffers, expected {}{fixed}",
                if l.variadic { "more than " } else { "" }
            ),
        ));
    }
    if n > 0 && a.buffers.is_null() {
        return Err(malformed(dt, "null buffer array".into()));
    }
    if l.variadic {
        // arrow-rs reads the last buffer as the variadic buffers' i64 lengths without
        // a null check, then trusts each length. With no variadic buffer it reads
        // nothing, and a zero-length buffer may be null.
        let variadic = n - fixed - 1;
        // SAFETY: `buffers` holds `n` pointers (non-null: checked above, as n > 0).
        let bufs = unsafe { std::slice::from_raw_parts(a.buffers, n) };
        let sizes = bufs[n - 1].cast::<i64>();
        if variadic > 0 && sizes.is_null() {
            return Err(malformed(dt, "null variadic sizes buffer".into()));
        }
        for i in 0..variadic {
            // SAFETY: the sizes buffer holds one i64 per variadic buffer.
            let len = unsafe { sizes.add(i).read_unaligned() };
            if len < 0 {
                return Err(malformed(
                    dt,
                    format!("variadic buffer {i} has length {len}"),
                ));
            }
            if len > 0 && bufs[fixed + i].is_null() {
                return Err(malformed(dt, format!("variadic buffer {i} is null")));
            }
        }
    }
    let children: Vec<&AT> = match dt {
        AT::List(f)
        | AT::LargeList(f)
        | AT::FixedSizeList(f, _)
        | AT::ListView(f)
        | AT::LargeListView(f)
        | AT::Map(f, _) => vec![f.data_type()],
        AT::Struct(fs) => fs.iter().map(|f| f.data_type()).collect(),
        AT::Union(fs, _) => fs.iter().map(|(_, f)| f.data_type()).collect(),
        AT::RunEndEncoded(r, v) => vec![r.data_type(), v.data_type()],
        _ => vec![],
    };
    if a.n_children as usize != children.len() {
        return Err(malformed(
            dt,
            format!("{} children, expected {}", a.n_children, children.len()),
        ));
    }
    if !children.is_empty() && a.children.is_null() {
        return Err(malformed(dt, "null child array".into()));
    }
    for (i, cdt) in children.into_iter().enumerate() {
        // SAFETY: `children` holds `n_children` pointers.
        let c = unsafe { *a.children.add(i) };
        if c.is_null() {
            return Err(malformed(dt, format!("child {i} is null")));
        }
        check_raw(c, cdt)?;
    }
    match (dt, a.dictionary.is_null()) {
        (AT::Dictionary(_, v), false) => check_raw(a.dictionary, v),
        (AT::Dictionary(..), true) => Err(malformed(dt, "no dictionary".into())),
        _ => Ok(()),
    }
}

/// arrow-rs's validation at every level: structure (`ArrayData::validate`) plus the
/// one structural check it lacks, made before anything reads a FixedSizeList's child
/// (the child must cover the parent's offset: arrow-rs checks `len · size` only, then
/// slices from `offset · size`); with `values`, also null counts and values (as
/// `ArrayData::validate_full`).
fn validate_tree(d: &ArrayData, values: bool) -> std::result::Result<(), ArrowError> {
    d.validate()?;
    if let AT::FixedSizeList(_, size) = d.data_type() {
        let need = (d.offset() + d.len()).saturating_mul(*size as usize);
        let have = d.child_data().first().map_or(0, ArrayData::len);
        if have < need {
            return Err(malformed(
                d.data_type(),
                format!("{have} child values, offset + length need {need}"),
            ));
        }
    }
    if values {
        d.validate_nulls()?;
        match d.data_type() {
            AT::Utf8View => validate_utf8_view(d)?,
            _ => d.validate_values()?,
        }
    }
    d.child_data()
        .iter()
        .try_for_each(|c| validate_tree(c, values))
}

/// `ArrayData::validate_values` for a (structurally valid) Utf8View, without decoding
/// every value: an inline ASCII value is valid as is; when the data buffers are valid
/// UTF-8 as a whole, a long view's bytes are valid iff they start and end on char
/// boundaries. The whole-buffer check is used only when the views reference at least
/// half the buffers' bytes (a sliced batch shares its parent's buffers: checking them
/// whole per batch would cost the parent's size each time); otherwise long values are
/// decoded one by one. A failure falls back to arrow-rs's check, which words the error.
fn validate_utf8_view(d: &ArrayData) -> std::result::Result<(), ArrowError> {
    const ASCII: u128 = 0x8080_8080_8080_8080_8080_8080;
    let views = &d.buffer::<u128>(0)[..d.len()];
    let bufs = &d.buffers()[1..];
    // Buffer indices, bounds, inline padding and prefixes.
    arrow_data::validate_binary_view(views, bufs)?;
    let long = |v: u128| (v as u32 > 12).then_some(v as u32 as usize);
    let referenced: usize = views.iter().filter_map(|&v| long(v)).sum();
    let total: usize = bufs.iter().map(|b| b.len()).sum();
    let whole = total <= 2 * referenced && bufs.iter().all(|b| std::str::from_utf8(b).is_ok());
    // Not a UTF-8 continuation byte (or the buffer's end).
    let boundary = |b: &[u8], i: usize| b.get(i).is_none_or(|&c| (c as i8) >= -0x40);
    let ok = views.iter().all(|&v| match long(v) {
        None => {
            let len = v as u32 as usize;
            (v >> 32) & ASCII == 0 || std::str::from_utf8(&v.to_le_bytes()[4..4 + len]).is_ok()
        }
        Some(len) => {
            let b = &bufs[(v >> 64) as u32 as usize];
            let start = (v >> 96) as u32 as usize;
            if whole {
                boundary(b, start) && boundary(b, start + len)
            } else {
                std::str::from_utf8(&b[start..start + len]).is_ok()
            }
        }
    });
    if ok {
        Ok(())
    } else {
        d.validate_values()
    }
}

/// The first type within `t` (itself or nested) that Polars cannot import without
/// aborting: Decimal256 (polars-core reads it as i128, and polars-arrow panics on
/// Int256). Every other type Polars cannot import is refused with an error by Polars.
fn polars_unsupported(t: &AT) -> Option<&AT> {
    match t {
        AT::Decimal256(..) => Some(t),
        AT::List(f)
        | AT::LargeList(f)
        | AT::FixedSizeList(f, _)
        | AT::ListView(f)
        | AT::LargeListView(f)
        | AT::Map(f, _) => polars_unsupported(f.data_type()),
        AT::Struct(fs) => fs.iter().find_map(|f| polars_unsupported(f.data_type())),
        AT::Union(fs, _) => fs
            .iter()
            .find_map(|(_, f)| polars_unsupported(f.data_type())),
        AT::Dictionary(_, v) => polars_unsupported(v),
        AT::RunEndEncoded(_, v) => polars_unsupported(v.data_type()),
        _ => None,
    }
}

/// `f`'s type refused if Polars cannot import it, naming the column.
fn check_field(f: &Field) -> std::result::Result<(), ArrowError> {
    match polars_unsupported(f.data_type()) {
        Some(t) => Err(ArrowError::InvalidArgumentError(format!(
            "column {:?}: {t} is not supported (Polars cannot import it)",
            f.name()
        ))),
        None => Ok(()),
    }
}

/// Every field of `schema` type-checked (`check_field`), so an input with no rows or
/// no batches is refused like one with data.
pub(crate) fn check_schema(schema: &Schema) -> std::result::Result<(), ArrowError> {
    schema.fields().iter().try_for_each(|f| check_field(f))
}

/// One column checked before arrow-rs or Polars reads it (see "checked import"),
/// errors naming it.
fn check_column(f: &Field, d: &ArrayData) -> std::result::Result<(), ArrowError> {
    check_field(f)?;
    validate_tree(d, true)
        .map_err(|e| ArrowError::InvalidArgumentError(format!("column {:?}: {e}", f.name())))
}

/// `check_column` over the columns in parallel (value validation is a linear pass
/// per column); the first failing column's error, in column order.
fn check_columns(
    fields: &arrow_schema::Fields,
    data: &[ArrayData],
) -> std::result::Result<(), ArrowError> {
    let checked: Vec<_> = fields
        .par_iter()
        .zip(data.par_iter())
        .map(|(f, c)| check_column(f, c))
        .collect();
    checked.into_iter().collect()
}

/// Every column of `batch` checked (`check_column`): a RecordBatch built without
/// validation can hold arrays whose buffers are too short or whose values are invalid.
/// The api entry points do not check (see the api module docs): batches read from a C
/// stream are checked on import (`import_checked`, which must check before it builds
/// arrays or concatenates batches), and safely built ones are valid by construction.
/// This is for a caller holding a RecordBatch from any other unchecked source.
#[allow(dead_code)] // every current binding imports through `import_checked`
pub(crate) fn validate_batch(batch: &RecordBatch) -> std::result::Result<(), ArrowError> {
    let data: Vec<ArrayData> = batch.columns().iter().map(|c| c.to_data()).collect();
    check_columns(batch.schema().fields(), &data)
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
    fn sliced_nested_columns_with_nulls_export_at_any_offset() {
        // polars-arrow exports Struct / FixedSizeList with `offset = validity.offset()`
        // (absolute), so offsets 8, 16, ... (a multiple of 8) need a rebuilt validity too.
        use polars_arrow::bitmap::Bitmap;
        let n = 40usize;
        let a: Vec<Option<i64>> = (0..n as i64)
            .map(|i| if i % 5 == 0 { None } else { Some(i) })
            .collect();
        let outer: Bitmap = (0..n).map(|i| i % 3 != 0).collect();
        let strukt = StructChunked::from_series(
            "s".into(),
            n,
            [Series::new("a".into(), a.as_slice())].iter(),
        )
        .unwrap()
        .with_outer_validity(Some(outer))
        .into_series();
        let lists = Series::new(
            "l".into(),
            (0..n as i64)
                .map(|i| (i % 4 != 0).then(|| Series::new("".into(), &[i])))
                .collect::<Vec<_>>(),
        );
        let array = lists
            .cast(&DataType::Array(Box::new(DataType::Int64), 1))
            .unwrap();
        let list_struct = Series::new(
            "ls".into(),
            (0..n as i64)
                .map(|i| {
                    (i % 4 != 0).then(|| {
                        StructChunked::from_series(
                            "".into(),
                            2,
                            [Series::new("a".into(), &[i, i + 1])].iter(),
                        )
                        .unwrap()
                        .with_outer_validity(Some([true, i % 3 != 0].into_iter().collect()))
                        .into_series()
                    })
                })
                .collect::<Vec<_>>(),
        );
        for (name, s) in [
            ("struct", &strukt),
            ("array", &array),
            ("list_struct", &list_struct),
        ] {
            for offset in [0i64, 1, 3, 8, 16] {
                let sliced = s.slice(offset, 16);
                let a = export_series(&sliced, CompatLevel::newest())
                    .unwrap_or_else(|e| panic!("{name} at offset {offset}: {e}"));
                assert_eq!(a.len(), 16, "{name} at offset {offset}");
                assert_eq!(
                    a.null_count(),
                    sliced.null_count(),
                    "{name} at offset {offset}"
                );
            }
        }
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

    fn int_stream() -> FFI_ArrowArrayStream {
        use arrow_array::{Int64Array, RecordBatchIterator};
        let b =
            RecordBatch::try_from_iter([("a", Arc::new(Int64Array::from(vec![1i64])) as ArrayRef)])
                .unwrap();
        let schema = b.schema();
        FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(b)], schema)))
    }

    static GET_SCHEMA_CALLED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    unsafe extern "C" fn flag_get_schema(_: *mut RawStream, _: *mut FFI_ArrowSchema) -> c_int {
        GET_SCHEMA_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
        1
    }

    #[test]
    fn a_stream_without_release_is_released() {
        // The C Stream Interface marks a released stream by `release == NULL`; its
        // other members are then undefined and must not be called.
        let mut stream = int_stream();
        let raw = (&mut stream as *mut FFI_ArrowArrayStream).cast::<RawStream>();
        // (Leaks the producer's state.)
        unsafe { (*raw).release = None };
        unsafe { (*raw).get_schema = Some(flag_get_schema) };
        let err = unsafe { CheckedReader::from_raw(&mut stream) }
            .err()
            .unwrap();
        assert!(err.to_string().contains("already released"), "{err}");
        assert!(!GET_SCHEMA_CALLED.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn a_stream_with_an_unsupported_schema_is_refused_even_with_no_batches() {
        use arrow_array::RecordBatchIterator;
        let item = Arc::new(Field::new("item", AT::Decimal256(10, 2), true));
        for dt in [AT::Decimal256(10, 2), AT::List(item)] {
            let schema = Arc::new(Schema::new(vec![Field::new("c", dt, true)]));
            let mut stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new(
                std::iter::empty(),
                schema,
            )));
            let err = unsafe { CheckedReader::from_raw(&mut stream) }
                .err()
                .expect("refused");
            let m = err.to_string();
            assert!(
                m.contains("column \"c\"") && m.contains("Decimal256(10, 2) is not supported"),
                "{m}"
            );
        }
    }

    #[test]
    fn a_stream_without_get_next_is_an_error() {
        let mut stream = int_stream();
        let raw = (&mut stream as *mut FFI_ArrowArrayStream).cast::<RawStream>();
        unsafe { (*raw).get_next = None };
        let mut reader = unsafe { CheckedReader::from_raw(&mut stream) }.unwrap();
        let err = reader.next().expect("an error, not the end").unwrap_err();
        assert!(err.to_string().contains("get_next"), "{err}");
    }

    /// Structurally invalid C data, as a producer might send it.
    mod malformed {
        use super::*;
        use arrow_buffer::Buffer;

        fn i64s(n: usize) -> ArrayData {
            ArrayData::builder(AT::Int64)
                .len(n)
                .add_buffer(Buffer::from_vec((0..n as i64).collect::<Vec<_>>()))
                .build()
                .unwrap()
        }

        /// `column` (possibly invalid) as the one column of an exported batch.
        fn exported(column: ArrayData) -> (FFI_ArrowArray, SchemaRef) {
            let field = Field::new("x", column.data_type().clone(), true);
            let schema = Arc::new(Schema::new(vec![field.clone()]));
            let data = unsafe {
                ArrayData::builder(AT::Struct(vec![field].into()))
                    .len(column.len())
                    .child_data(vec![column])
                    .build_unchecked()
            };
            (FFI_ArrowArray::new(&data), schema)
        }

        fn rejected(column: ArrayData) -> String {
            rejected_after(column, |_| {})
        }

        /// Rejected once `edit` has changed the exported column's raw C struct.
        fn rejected_after(column: ArrayData, edit: impl FnOnce(&mut RawArray)) -> String {
            let (mut array, schema) = exported(column);
            // SAFETY: the batch struct has one child, an exported C array.
            let raw = (&mut array as *mut FFI_ArrowArray).cast::<RawArray>();
            edit(unsafe { &mut **(*raw).children });
            match unsafe { import_checked(array, &schema) } {
                Ok(_) => panic!("accepted"),
                Err(e) => e.to_string(),
            }
        }

        #[test]
        fn a_valid_batch_imports() {
            let (array, schema) = exported(i64s(3));
            assert_eq!(
                unsafe { import_checked(array, &schema) }
                    .unwrap()
                    .num_rows(),
                3
            );
        }

        #[test]
        fn a_sliced_array_over_sliced_children() {
            // Offset 2, length 3 over a child with 3 rows' slots: py-polars' export of
            // a sliced Array with nulls.
            let item = Arc::new(Field::new("item", AT::Int64, true));
            let a = unsafe {
                ArrayData::builder(AT::FixedSizeList(item, 2))
                    .len(3)
                    .offset(2)
                    .child_data(vec![i64s(6)])
                    .build_unchecked()
            };
            let msg = rejected(a);
            assert!(msg.contains("offset + length need 10"), "{msg}");
        }

        #[test]
        fn a_sliced_struct_over_sliced_children() {
            let s = unsafe {
                ArrayData::builder(AT::Struct(vec![Field::new("a", AT::Int64, true)].into()))
                    .len(3)
                    .offset(2)
                    .child_data(vec![i64s(3)])
                    .build_unchecked()
            };
            assert!(rejected(s).contains("smaller than expected"));
        }

        #[test]
        fn a_null_column_with_a_buffer() {
            // py-polars exports a Null column with one buffer; the C Data Interface
            // gives it none.
            let msg = rejected_after(ArrayData::new_null(&AT::Null, 3), |a| a.n_buffers = 1);
            assert!(msg.contains("1 buffers, expected 0"), "{msg}");
        }

        /// A string view with one variadic data buffer: C buffers [validity, views,
        /// data, sizes].
        fn a_view_array() -> ArrayData {
            arrow_array::StringViewArray::from(vec!["a string longer than twelve bytes"])
                .into_data()
        }

        /// `a`'s C buffer `i` (negative: from the end).
        fn buffer_slot(a: &mut RawArray, i: i64) -> &mut *const c_void {
            let i = if i < 0 { a.n_buffers + i } else { i };
            // SAFETY: `buffers` holds `n_buffers` pointers, owned by the export.
            unsafe { &mut *a.buffers.add(i as usize) }
        }

        #[test]
        fn a_view_array_without_its_variadic_sizes() {
            // arrow-rs reads the sizes buffer (the last) without a null check.
            let msg = rejected_after(a_view_array(), |a| {
                *buffer_slot(a, -1) = std::ptr::null();
            });
            assert!(msg.contains("null variadic sizes buffer"), "{msg}");
        }

        #[test]
        fn a_view_array_with_a_negative_variadic_length() {
            let msg = rejected_after(a_view_array(), |a| {
                // SAFETY: the sizes buffer holds one i64, in memory the export owns.
                unsafe { *(*buffer_slot(a, -1) as *mut i64) = -1 };
            });
            assert!(msg.contains("variadic buffer 0 has length -1"), "{msg}");
        }

        #[test]
        fn a_view_array_without_its_data_buffer() {
            let msg = rejected_after(a_view_array(), |a| {
                *buffer_slot(a, 2) = std::ptr::null();
            });
            assert!(msg.contains("variadic buffer 0 is null"), "{msg}");
        }

        #[test]
        fn a_view_array_without_variadic_buffers_may_omit_its_sizes() {
            // Inline values only: no variadic buffer, so a null (zero-length) sizes
            // buffer is allowed.
            let (mut array, schema) =
                exported(arrow_array::StringViewArray::from(vec!["short"]).into_data());
            let raw = (&mut array as *mut FFI_ArrowArray).cast::<RawArray>();
            let a = unsafe { &mut **(*raw).children };
            assert_eq!(a.n_buffers, 3);
            *buffer_slot(a, -1) = std::ptr::null();
            assert_eq!(
                unsafe { import_checked(array, &schema) }
                    .unwrap()
                    .num_rows(),
                1
            );
        }

        #[test]
        fn a_missing_buffer() {
            let msg = rejected_after(i64s(3), |a| a.n_buffers = 1);
            assert!(msg.contains("1 buffers, expected 2"), "{msg}");
        }

        #[test]
        fn a_negative_length() {
            let msg = rejected_after(i64s(3), |a| a.length = -1);
            assert!(msg.contains("negative"), "{msg}");
        }

        #[test]
        fn a_missing_child() {
            let s = unsafe {
                ArrayData::builder(AT::Struct(vec![Field::new("a", AT::Int64, true)].into()))
                    .len(3)
                    .build_unchecked()
            };
            assert!(rejected(s).contains("0 children, expected 1"));
        }

        #[test]
        fn offsets_past_the_values() {
            let s = unsafe {
                ArrayData::builder(AT::Utf8)
                    .len(2)
                    .add_buffer(Buffer::from_vec(vec![0i32, 3, 100]))
                    .add_buffer(Buffer::from_vec(b"abcdef".to_vec()))
                    .build_unchecked()
            };
            let a = arrow_array::make_array(s);
            let batch = RecordBatch::try_from_iter([("s", a)]).unwrap();
            assert!(validate_batch(&batch)
                .unwrap_err()
                .to_string()
                .contains("\"s\""));
        }

        // Value-level malformations: arrow-rs's structural checks pass them, and Polars
        // asserts (dictionary keys), reads out of bounds (UTF-8) or aborts (Int256).

        fn keys_out_of_range() -> ArrayData {
            let values = arrow_array::StringArray::from(vec!["x", "y"]).into_data();
            unsafe {
                ArrayData::builder(AT::Dictionary(Box::new(AT::Int32), Box::new(AT::Utf8)))
                    .len(4)
                    .add_buffer(Buffer::from_vec(vec![0i32, 1, 5, 1]))
                    .child_data(vec![values])
                    .build_unchecked()
            }
        }

        fn invalid_utf8() -> ArrayData {
            unsafe {
                ArrayData::builder(AT::Utf8)
                    .len(2)
                    .add_buffer(Buffer::from_vec(vec![0i32, 2, 4]))
                    .add_buffer(Buffer::from_vec(vec![0xffu8, 0xfe, 0xc3, 0x28]))
                    .build_unchecked()
            }
        }

        fn decimal256() -> ArrayData {
            arrow_array::Decimal256Array::from(vec![arrow_buffer::i256::from(1)])
                .with_precision_and_scale(10, 2)
                .unwrap()
                .into_data()
        }

        /// `column`'s rejection, as the one column "x" of a C stream batch and of a
        /// RecordBatch (api callers).
        fn rejected_both_ways(column: ArrayData) -> [String; 2] {
            let batch =
                RecordBatch::try_from_iter([("x", arrow_array::make_array(column.clone()))])
                    .unwrap();
            let direct = validate_batch(&batch).unwrap_err().to_string();
            let msgs = [rejected(column), direct];
            for m in &msgs {
                assert!(m.contains("column \"x\""), "{m}");
            }
            msgs
        }

        #[test]
        fn dictionary_keys_out_of_range() {
            for m in rejected_both_ways(keys_out_of_range()) {
                assert!(m.contains("out of bounds"), "{m}");
            }
        }

        #[test]
        fn invalid_utf8_in_a_string_array() {
            for m in rejected_both_ways(invalid_utf8()) {
                assert!(m.to_lowercase().contains("utf8"), "{m}");
            }
        }

        /// A Utf8View over one data buffer `data`: `(offset, len)` views into it, or
        /// inline values given as bytes.
        fn utf8_view(data: &[u8], views: &[Result<(u32, u32), &[u8]>]) -> ArrayData {
            let views: Vec<u128> = views
                .iter()
                .map(|v| match *v {
                    Ok((offset, len)) => {
                        let s = offset as usize;
                        let mut prefix = [0u8; 4];
                        prefix.copy_from_slice(&data[s..s + 4]);
                        len as u128
                            | (u32::from_le_bytes(prefix) as u128) << 32
                            | (offset as u128) << 96
                    }
                    Err(bytes) => {
                        let mut b = [0u8; 16];
                        b[..4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
                        b[4..4 + bytes.len()].copy_from_slice(bytes);
                        u128::from_le_bytes(b)
                    }
                })
                .collect();
            unsafe {
                ArrayData::builder(AT::Utf8View)
                    .len(views.len())
                    .add_buffer(Buffer::from_vec(views))
                    .add_buffer(Buffer::from_vec(data.to_vec()))
                    .build_unchecked()
            }
        }

        fn view_batch(d: ArrayData) -> RecordBatch {
            RecordBatch::try_from_iter([("x", arrow_array::make_array(d))]).unwrap()
        }

        #[test]
        fn utf8_view_values() {
            let data = "ééééééééééé-abcdefghijklmn".as_bytes(); // 22 bytes of é, then ASCII
                                                                // Valid: whole chars, a non-ASCII inline value, an ASCII one.
            let ok = utf8_view(
                data,
                &[Ok((0, 14)), Ok((22, 13)), Err("né".as_bytes()), Err(b"ab")],
            );
            validate_batch(&view_batch(ok)).unwrap();
            // A view starting or ending inside a char; an invalid inline value.
            for bad in [
                utf8_view(data, &[Ok((1, 14))]),
                utf8_view(data, &[Ok((1, 14)), Ok((22, 13))]), // whole-buffer path
                utf8_view(data, &[Ok((0, 13))]),
                utf8_view(data, &[Ok((0, 13)), Ok((22, 13))]),
                utf8_view(data, &[Err(&[0xc3, b'a'])]),
            ] {
                let m = validate_batch(&view_batch(bad)).unwrap_err().to_string();
                assert!(m.contains("column \"x\"") && m.contains("UTF-8"), "{m}");
            }
            // Invalid bytes no view references are allowed.
            let mut junk = data.to_vec();
            junk.push(0xff);
            validate_batch(&view_batch(utf8_view(&junk, &[Ok((0, 14))]))).unwrap();
            let bad = utf8_view(&junk, &[Ok((1, 14))]);
            assert!(validate_batch(&view_batch(bad)).is_err());
        }

        #[test]
        fn a_decimal256_column_polars_cannot_import() {
            for m in rejected_both_ways(decimal256()) {
                assert!(m.contains("Decimal256(10, 2) is not supported"), "{m}");
            }
            // At any depth.
            let item = Arc::new(Field::new("item", AT::Decimal256(10, 2), true));
            let list = arrow_array::ListArray::new(
                item,
                arrow_buffer::OffsetBuffer::from_lengths([1]),
                arrow_array::make_array(decimal256()),
                None,
            );
            for m in rejected_both_ways(list.into_data()) {
                assert!(m.contains("Decimal256(10, 2) is not supported"), "{m}");
            }
        }
    }
}
