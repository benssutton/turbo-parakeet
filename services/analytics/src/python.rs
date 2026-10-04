//! Python binding over api.rs. Tables arrive as any object implementing the Arrow
//! PyCapsule interface (`__arrow_c_stream__`: Polars and pyarrow frames and readers)
//! and leave as `ArrowTable`, which implements it too. Errors: invalid input →
//! ValueError, kernel failure → RuntimeError.

use std::ffi::CStr;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::ffi::FFI_ArrowSchema;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};

use crate::api;
// `RawStream` lets `reject_wide_integers` call `get_schema` before the stream is
// imported (a stream may be asked for its schema repeatedly).
use crate::arrow_io::{read_stream, CheckedReader, RawStream};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Runs `f` on the ArrowArrayStream inside `data`'s `__arrow_c_stream__` capsule
/// (wide-integer columns refused). The capsule stays alive until `f` returns: `f`
/// must move the stream out (`from_raw`), leaving a released one behind, so the
/// capsule's destructor releases nothing twice.
fn with_stream<T>(
    data: &Bound<'_, PyAny>,
    f: impl FnOnce(*mut FFI_ArrowArrayStream) -> PyResult<T>,
) -> PyResult<T> {
    if !data.hasattr("__arrow_c_stream__")? {
        return Err(PyTypeError::new_err(format!(
            "expected Arrow tabular data (an object with __arrow_c_stream__), got {}",
            data.get_type().name()?
        )));
    }
    let capsule = data.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    if !capsule.is_valid_checked(Some(c"arrow_array_stream")) {
        return Err(value_error(
            "__arrow_c_stream__ did not return an arrow_array_stream capsule",
        ));
    }
    let stream = capsule
        .pointer_checked(Some(c"arrow_array_stream"))?
        .as_ptr() as *mut FFI_ArrowArrayStream;
    // SAFETY: an "arrow_array_stream" capsule holds a valid, unreleased ArrowArrayStream.
    unsafe { reject_wide_integers(stream.cast())? };
    f(stream)
}

/// A whole Arrow stream as one RecordBatch (kernels expect one chunk per column).
fn read_batch(data: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    // SAFETY: the stream is valid and unreleased (with_stream); read_stream moves it out.
    with_stream(data, |s| unsafe { read_stream(s) }.map_err(value_error))
}

/// An Arrow stream read batch by batch, each batch checked (the streaming
/// recommender never concatenates).
fn read_reader(data: &Bound<'_, PyAny>) -> PyResult<CheckedReader> {
    // SAFETY: as for read_batch; from_raw moves the stream out.
    with_stream(data, |s| {
        unsafe { CheckedReader::from_raw(s) }.map_err(value_error)
    })
}

/// Arrow has no 128-bit integer type. Polars exports Int128 / UInt128 in its private
/// formats `_pli128` / `_plu128`, which arrow-rs (and every non-Polars consumer)
/// rejects, so they are refused here, naming the column. The Python classes list
/// such columns as ineligible and never send them (analytics._dtypes.WIDE_INTEGERS).
///
/// SAFETY: `stream` points to a valid, unreleased ArrowArrayStream.
unsafe fn reject_wide_integers(stream: *mut RawStream) -> PyResult<()> {
    // A released stream has `release == NULL`; its other members are undefined.
    if unsafe { (*stream).release }.is_none() {
        return Err(value_error("arrow stream already released"));
    }
    let get_schema = unsafe { (*stream).get_schema }
        .ok_or_else(|| value_error("arrow stream: no get_schema callback"))?;
    let mut schema = FFI_ArrowSchema::empty();
    if unsafe { get_schema(stream, &mut schema) } != 0 {
        let detail = unsafe { (*stream).get_last_error }.and_then(|get_last_error| {
            let msg = unsafe { get_last_error(stream) };
            if msg.is_null() {
                None
            } else {
                // SAFETY: a non-null `get_last_error` result is a valid, NUL-terminated
                // C string owned by the stream, live at least until the next call on it.
                Some(
                    unsafe { CStr::from_ptr(msg) }
                        .to_string_lossy()
                        .into_owned(),
                )
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
        _ => s
            .children()
            .find_map(wide_integer)
            .or_else(|| s.dictionary().and_then(wide_integer)),
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
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let reader = RecordBatchIterator::new([Ok(self.0.clone())], self.0.schema());
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
    }
}

fn run<T: Send>(py: Python<'_>, f: impl FnOnce() -> api::Result<T> + Send) -> PyResult<T> {
    py.detach(f).map_err(|e| match e {
        api::Error::InvalidInput(m) => PyValueError::new_err(m),
        api::Error::Compute(m) => PyRuntimeError::new_err(m),
    })
}

/// The streaming recommender (api::StreamingRecommender): add batches over time,
/// `result` at any point. A mutex serialises callers; the work runs without the GIL.
#[pyclass(frozen, module = "analytics.analytics")]
struct StreamingRecommender(Mutex<api::StreamingRecommender>);

fn locked(
    m: &Mutex<api::StreamingRecommender>,
) -> api::Result<std::sync::MutexGuard<'_, api::StreamingRecommender>> {
    m.lock()
        .map_err(|_| api::Error::Compute("recommender poisoned by an earlier panic".into()))
}

#[pymethods]
impl StreamingRecommender {
    #[new]
    #[pyo3(signature = (*, reservoir_rows, block_rows, categorical_threshold, zstd_level, seed, boolean_pairs))]
    fn new(
        py: Python<'_>,
        reservoir_rows: u64,
        block_rows: u64,
        categorical_threshold: u64,
        zstd_level: i32,
        seed: u64,
        boolean_pairs: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let p = api::StreamingParams {
            reservoir_rows,
            block_rows,
            categorical_threshold,
            zstd_level,
            seed,
            boolean_pairs,
        };
        run(py, || api::StreamingRecommender::new(p)).map(|r| Self(Mutex::new(r)))
    }

    /// Adds every batch of `data`, in order. `ineligible`: (name, dtype) of columns the
    /// caller dropped (Int128 / UInt128, Object); they are marked first. Each batch is
    /// atomic, but the call is not: if a later batch (or the producer) fails, the
    /// earlier batches and the ineligible marks stay applied.
    #[pyo3(signature = (data, ineligible=Vec::new()))]
    fn add(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        ineligible: Vec<(String, String)>,
    ) -> PyResult<()> {
        let reader = read_reader(data)?;
        let rec = &self.0;
        run(py, move || {
            let mut rec = locked(rec)?;
            for (name, dtype) in &ineligible {
                rec.mark_ineligible(name, dtype)?;
            }
            for batch in reader {
                rec.add(&batch.map_err(|e| api::Error::InvalidInput(e.to_string()))?)?;
            }
            Ok(())
        })
    }

    fn result(&self, py: Python<'_>) -> PyResult<ArrowTable> {
        let rec = &self.0;
        run(py, move || locked(rec)?.result()).map(ArrowTable)
    }
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
fn pairwise_joint_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    pairs: Option<Vec<(String, String)>>,
) -> PyResult<ArrowTable> {
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
    run(py, || {
        api::threeway_joint_entropy(&batch, triplets.as_deref())
    })
    .map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs=None))]
fn pairwise_chi_squared(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    pairs: Option<Vec<(String, String)>>,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::pairwise_chi_squared(&batch, pairs.as_deref())).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, pairs=None))]
fn pairwise_adjusted_rand(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    pairs: Option<Vec<(String, String)>>,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::pairwise_adjusted_rand(&batch, pairs.as_deref())).map(ArrowTable)
}

#[pyfunction]
fn bloom_filter<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    k: usize,
    m: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = read_batch(data)?;
    let bits = run(py, || api::bloom_filter(&batch, k, m))?;
    Ok(PyBytes::new(py, &bits))
}

#[pyfunction]
fn membership_ratio(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    bits: &[u8],
    k: usize,
    m: usize,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::membership_ratio(&batch, bits, k, m)).map(ArrowTable)
}

#[pyfunction]
fn minhash(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    df_name: String,
    num_perm: usize,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::minhash(&batch, &df_name, num_perm)).map(ArrowTable)
}

#[pyfunction]
fn lsh_candidates(
    py: Python<'_>,
    signatures: &Bound<'_, PyAny>,
    num_bands: usize,
    rows_per_band: usize,
) -> PyResult<ArrowTable> {
    let batch = read_batch(signatures)?;
    run(py, || api::lsh_candidates(&batch, num_bands, rows_per_band)).map(ArrowTable)
}

#[pyfunction]
fn describe_columns(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    seed: u64,
    categorical_threshold: u64,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || {
        api::describe_columns(&batch, seed, categorical_threshold)
    })
    .map(ArrowTable)
}

#[pyfunction]
fn column_sizes(py: Python<'_>, data: &Bound<'_, PyAny>, zstd_level: i32) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::column_sizes(&batch, zstd_level)).map(ArrowTable)
}

#[pyfunction]
fn render(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::render(&batch)).map(ArrowTable)
}

#[pyfunction]
#[pyo3(signature = (data, *, seed, zstd_level, categorical_threshold, boolean_pairs))]
fn describe_and_recommend(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    seed: u64,
    zstd_level: i32,
    categorical_threshold: u64,
    boolean_pairs: Vec<(String, String)>,
) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || {
        api::describe_and_recommend(
            &batch,
            seed,
            zstd_level,
            categorical_threshold,
            boolean_pairs,
        )
    })
    .map(ArrowTable)
}

/// Arrow tabular data read and checked (arrow_io's checked import) into one table,
/// for a caller about to hand it to Polars: py-polars trusts Arrow input and panics
/// on malformed values (dictionary keys out of range) or a Decimal256.
#[pyfunction]
fn checked_table(data: &Bound<'_, PyAny>) -> PyResult<ArrowTable> {
    read_batch(data).map(ArrowTable)
}

#[pymodule]
fn analytics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowTable>()?;
    m.add_class::<StreamingRecommender>()?;
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
    m.add_function(wrap_pyfunction!(render, m)?)?;
    m.add_function(wrap_pyfunction!(describe_and_recommend, m)?)?;
    m.add_function(wrap_pyfunction!(checked_table, m)?)?;
    Ok(())
}
