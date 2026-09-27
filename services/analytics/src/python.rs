//! Python binding over api.rs. Tables arrive as any object implementing the Arrow
//! PyCapsule interface (`__arrow_c_stream__`: Polars and pyarrow frames and readers)
//! and leave as `ArrowTable`, which implements it too. Errors: invalid input →
//! ValueError, kernel failure → RuntimeError.

use std::ffi::{c_char, c_int, c_void, CStr};

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::ffi::FFI_ArrowSchema;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};

use crate::api;

/// The C Stream Interface struct, field for field. arrow-rs keeps its copy's
/// callbacks private; this mirror lets `reject_wide_integers` call `get_schema`
/// before arrow-rs imports the stream (a stream may be asked for its schema repeatedly).
#[repr(C)]
struct RawStream {
    get_schema: Option<unsafe extern "C" fn(*mut RawStream, *mut FFI_ArrowSchema) -> c_int>,
    get_next: Option<unsafe extern "C" fn(*mut RawStream, *mut c_void) -> c_int>,
    get_last_error: Option<unsafe extern "C" fn(*mut RawStream) -> *const c_char>,
    release: Option<unsafe extern "C" fn(*mut RawStream)>,
    private_data: *mut c_void,
}

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A whole Arrow stream as one RecordBatch (kernels expect one chunk per column).
fn read_batch(data: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if !data.hasattr("__arrow_c_stream__")? {
        return Err(PyTypeError::new_err(format!(
            "expected Arrow tabular data (an object with __arrow_c_stream__), got {}",
            data.get_type().name()?
        )));
    }
    let capsule = data.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.downcast::<PyCapsule>()?;
    if capsule.name()? != Some(c"arrow_array_stream") {
        return Err(value_error("__arrow_c_stream__ did not return an arrow_array_stream capsule"));
    }
    let stream = capsule.pointer() as *mut FFI_ArrowArrayStream;
    // SAFETY: an "arrow_array_stream" capsule holds a valid, unreleased
    // ArrowArrayStream (Arrow PyCapsule interface). `from_raw` moves it out and leaves
    // a released stream behind, so the capsule's destructor releases nothing twice.
    unsafe { reject_wide_integers(stream.cast())? };
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream) }.map_err(value_error)?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(value_error)?;
    arrow_select::concat::concat_batches(&schema, &batches).map_err(value_error)
}

/// Arrow has no 128-bit integer type. Polars exports Int128 / UInt128 in its private
/// formats `_pli128` / `_plu128`, which arrow-rs (and every non-Polars consumer)
/// rejects, so they are refused here, naming the column. The Python classes list
/// such columns as ineligible and never send them (analytics._dtypes.WIDE_INTEGERS).
///
/// SAFETY: `stream` points to a valid, unreleased ArrowArrayStream.
unsafe fn reject_wide_integers(stream: *mut RawStream) -> PyResult<()> {
    let get_schema = unsafe { (*stream).get_schema }.ok_or_else(|| value_error("arrow stream already released"))?;
    let mut schema = FFI_ArrowSchema::empty();
    if unsafe { get_schema(stream, &mut schema) } != 0 {
        let detail = unsafe { (*stream).get_last_error }.and_then(|get_last_error| {
            let msg = unsafe { get_last_error(stream) };
            if msg.is_null() {
                None
            } else {
                // SAFETY: a non-null `get_last_error` result is a valid, NUL-terminated
                // C string owned by the stream, live at least until the next call on it.
                Some(unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned())
            }
        });
        return Err(value_error(match detail {
            Some(msg) => format!("arrow stream: get_schema failed: {msg}"),
            None => "arrow stream: get_schema failed".to_owned(),
        }));
    }
    for column in schema.children() {
        if let Some(kind) = wide_integer(column) {
            return Err(value_error(format!(
                "column {:?} holds {kind}: Arrow has no 128-bit integer type; cast it to Decimal(38, 0) or Int64",
                column.name().unwrap_or_default()
            )));
        }
    }
    Ok(())
}

fn wide_integer(s: &FFI_ArrowSchema) -> Option<&'static str> {
    match s.format() {
        "_pli128" => Some("Int128"),
        "_plu128" => Some("UInt128"),
        _ => s.children().find_map(wide_integer).or_else(|| s.dictionary().and_then(wide_integer)),
    }
}

/// A result table, handed to Python through the Arrow PyCapsule interface
/// (`pl.DataFrame(table)`, `pa.table(table)`).
#[pyclass(frozen, module = "analytics.analytics")]
struct ArrowTable(RecordBatch);

#[pymethods]
impl ArrowTable {
    /// `requested_schema` is not supported: the table is exported as it is.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(&self, py: Python<'py>, requested_schema: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let reader = RecordBatchIterator::new([Ok(self.0.clone())], self.0.schema());
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        PyCapsule::new(py, stream, Some(c"arrow_array_stream".to_owned()))
    }
}

fn run<T: Send>(py: Python<'_>, f: impl FnOnce() -> api::Result<T> + Send) -> PyResult<T> {
    py.allow_threads(f).map_err(|e| match e {
        api::Error::InvalidInput(m) => PyValueError::new_err(m),
        api::Error::Compute(m) => PyRuntimeError::new_err(m),
    })
}

#[pyfunction]
fn column_gcd(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::column_gcd(&batch)).map(ArrowTable)
}

#[pyfunction]
fn marginal_entropy(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::marginal_entropy(&batch)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs=None))]
fn pairwise_joint_entropy(py: Python<'_>, data: &Bound<'_, PyAny>, pairs: Option<Vec<(String, String)>>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::pairwise_joint_entropy(&batch, pairs.as_deref())).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, triplets=None))]
fn threeway_joint_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    triplets: Option<Vec<(String, String, String)>>,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::threeway_joint_entropy(&batch, triplets.as_deref())).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs=None))]
fn pairwise_chi_squared(py: Python<'_>, data: &Bound<'_, PyAny>, pairs: Option<Vec<(String, String)>>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::pairwise_chi_squared(&batch, pairs.as_deref())).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs=None))]
fn pairwise_adjusted_rand(py: Python<'_>, data: &Bound<'_, PyAny>, pairs: Option<Vec<(String, String)>>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::pairwise_adjusted_rand(&batch, pairs.as_deref())).map(ArrowTable)
}

#[pyfunction]
fn bloom_filter<'py>(py: Python<'py>, data: &Bound<'py, PyAny>, k: usize, m: usize) -> PyResult<Bound<'py, PyBytes>> {
    let batch = read_batch(data)?;
    let bits = run(py, || api::bloom_filter(&batch, k, m))?;
    Ok(PyBytes::new(py, &bits))
}

#[pyfunction]
fn membership_ratio(py: Python<'_>, data: &Bound<'_, PyAny>, bits: &[u8], k: usize, m: usize) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::membership_ratio(&batch, bits, k, m)).map(ArrowTable)
}

#[pyfunction]
fn minhash(py: Python<'_>, data: &Bound<'_, PyAny>, df_name: String, num_perm: usize) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::minhash(&batch, &df_name, num_perm)).map(ArrowTable)
}

#[pyfunction]
fn lsh_candidates(py: Python<'_>, signatures: &Bound<'_, PyAny>, num_bands: usize, rows_per_band: usize) -> PyResult<ArrowTable> {
    let batch = read_batch(signatures)?;
    run(py, || api::lsh_candidates(&batch, num_bands, rows_per_band)).map(ArrowTable)
}

#[pyfunction]
fn describe_columns(py: Python<'_>, data: &Bound<'_, PyAny>, seed: u64) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::describe_columns(&batch, seed)).map(ArrowTable)
}

#[pyfunction]
fn column_sizes(py: Python<'_>, data: &Bound<'_, PyAny>, zstd_level: i32) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::column_sizes(&batch, zstd_level)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, *, seed, zstd_level, population_rows, categorical_threshold, boolean_pairs))]
fn describe_and_recommend(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    seed: u64,
    zstd_level: i32,
    population_rows: Option<u64>,
    categorical_threshold: u64,
    boolean_pairs: Vec<(String, String)>,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::describe_and_recommend(&batch, seed, zstd_level, population_rows, categorical_threshold, boolean_pairs))
        .map(ArrowTable)
}

#[pymodule]
fn analytics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowTable>()?;
    m.add_function(wrap_pyfunction!(column_gcd, m)?)?;
    m.add_function(wrap_pyfunction!(marginal_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_joint_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(threeway_joint_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_chi_squared, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_adjusted_rand, m)?)?;
    m.add_function(wrap_pyfunction!(bloom_filter, m)?)?;
    m.add_function(wrap_pyfunction!(membership_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(minhash, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(describe_columns, m)?)?;
    m.add_function(wrap_pyfunction!(column_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(describe_and_recommend, m)?)?;
    Ok(())
}
