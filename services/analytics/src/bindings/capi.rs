//! C ABI over api.rs, for Java (Panama FFM) and any other language that can call C.
//! Tables cross as Arrow C Streams; parameters as plain C values. Every entry point
//! returns 0 on success, 1 on invalid input or 2 on a compute failure (mirroring
//! `api::Error`); on failure `*error` holds a message the caller frees with
//! `analytics_free_error`. Built without Python by
//! `cargo build --profile ci --no-default-features --target-dir target/capi`.
//! The recommenders' `mark_ineligible` is intentionally not exposed: it exists
//! for Polars' Int128 / UInt128 columns, which Arrow input cannot carry.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchIterator};

use crate::bindings::api::{self, Error};
use crate::common::arrow_io::{read_stream, CheckedReader};

const OK: c_int = 0;
const INVALID_INPUT: c_int = 1;
const COMPUTE: c_int = 2;

/// Writes `result` as a return code, and on failure its message into `*error`.
///
/// SAFETY: `error` is null or valid for one pointer write.
unsafe fn report(result: api::Result<()>, error: *mut *mut c_char) -> c_int {
    let (code, msg) = match result {
        Ok(()) => return OK,
        Err(Error::InvalidInput(m)) => (INVALID_INPUT, m),
        Err(Error::Compute(m)) => (COMPUTE, m),
    };
    if !error.is_null() {
        let msg = CString::new(msg.replace('\0', " ")).unwrap_or_default();
        unsafe { *error = msg.into_raw() };
    }
    code
}

/// `n` UTF-8 C strings.
///
/// SAFETY: when `n > 0`, `ptrs` is null or points to `n` pointers, each null or a
/// NUL-terminated string.
unsafe fn strings(ptrs: *const *const c_char, n: usize) -> api::Result<Vec<String>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptrs.is_null() {
        return Err(Error::InvalidInput("boolean-pair array is null".into()));
    }
    unsafe { std::slice::from_raw_parts(ptrs, n) }
        .iter()
        .map(|&p| {
            if p.is_null() {
                return Err(Error::InvalidInput("boolean-pair string is null".into()));
            }
            unsafe { CStr::from_ptr(p) }
                .to_str()
                .map(str::to_owned)
                .map_err(|e| Error::InvalidInput(format!("boolean-pair string is not UTF-8: {e}")))
        })
        .collect()
}

/// The boolean pairs of a `_new` call.
///
/// SAFETY: as for `strings`, for both arrays.
unsafe fn boolean_pairs(
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n: usize,
) -> api::Result<Vec<(String, String)>> {
    let trues = unsafe { strings(bool_true, n) }?;
    let falses = unsafe { strings(bool_false, n) }?;
    Ok(trues.into_iter().zip(falses).collect())
}

/// Frees a message written to `*error`. Null is a no-op.
///
/// # Safety
/// `error` is null or a pointer this library wrote to `*error`, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn analytics_free_error(error: *mut c_char) {
    if !error.is_null() {
        drop(unsafe { CString::from_raw(error) });
    }
}

// Handles cross as `void *` pointing to a `Mutex<R>` (R: api::StreamingRecommender or
// api::OneShotRecommender): a pointer to a Rust type in an `extern "C"` signature trips
// `improper_ctypes_definitions`. Concurrent `_add` / `_result` calls on one handle are
// serialised by the lock; `_free` is not (see the `_free` functions). Passing a handle of one
// recommender kind to the other kind's functions is undefined behaviour.

// Handles are shared across caller threads: checked in the C-only build too.
const _: () = {
    fn thread_safe<T: Send + Sync>() {}
    let _ = thread_safe::<Mutex<api::StreamingRecommender>>;
    let _ = thread_safe::<Mutex<api::OneShotRecommender>>;
};

/// SAFETY: `out` is valid for one pointer write.
unsafe fn new_handle<R>(r: R, out: *mut *mut c_void) {
    unsafe { *out = Box::into_raw(Box::new(Mutex::new(r))) as *mut c_void };
}

/// SAFETY: `h` is null or a live handle to a `Mutex<R>`.
unsafe fn handle<'a, R>(h: *mut c_void) -> api::Result<&'a Mutex<R>> {
    if h.is_null() {
        return Err(Error::InvalidInput("recommender handle is null".into()));
    }
    Ok(unsafe { &*(h as *const Mutex<R>) })
}

fn locked<R>(r: &Mutex<R>) -> api::Result<std::sync::MutexGuard<'_, R>> {
    r.lock()
        .map_err(|_| Error::Compute("recommender poisoned by an earlier panic".into()))
}

/// `result(&R)` written to `*out` as a one-batch stream.
///
/// SAFETY: `h` as for `handle`; `out` is null or valid for writing one ArrowArrayStream.
unsafe fn write_result<R>(
    h: *mut c_void,
    out: *mut FFI_ArrowArrayStream,
    result: fn(&R) -> api::Result<RecordBatch>,
) -> api::Result<()> {
    if out.is_null() {
        return Err(Error::InvalidInput("output stream is null".into()));
    }
    let batch = result(&*locked(unsafe { handle::<R>(h) }?)?)?;
    let schema = batch.schema();
    let stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(batch)], schema)));
    // `*out` may be uninitialised or released: overwrite it without dropping.
    unsafe { ptr::write(out, stream) };
    Ok(())
}

/// SAFETY: `h` is null or a live handle to a `Mutex<R>`, not used afterwards.
unsafe fn free_handle<R>(h: *mut c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h as *mut Mutex<R>) });
    }
}

// ── streaming recommender ────────────────────────────────────────────────────

/// A streaming recommender (see `api::StreamingRecommender`). On success `*out` holds a
/// handle the caller frees with `analytics_streaming_recommender_free`.
///
/// # Safety
/// `out` is null or valid for one pointer write. When `n_bool_pairs > 0`, `bool_true`
/// and `bool_false` each point to `n_bool_pairs` NUL-terminated UTF-8 strings. `error`
/// is null or valid for one pointer write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn analytics_streaming_recommender_new(
    reservoir_rows: u64,
    block_rows: u64,
    categorical_threshold: u64,
    zstd_level: i32,
    seed: u64,
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n_bool_pairs: usize,
    out: *mut *mut c_void,
    error: *mut *mut c_char,
) -> c_int {
    let result = (|| {
        if out.is_null() {
            return Err(Error::InvalidInput("out is null".into()));
        }
        let r = api::StreamingRecommender::new(api::StreamingParams {
            reservoir_rows,
            block_rows,
            categorical_threshold,
            zstd_level,
            seed,
            boolean_pairs: unsafe { boolean_pairs(bool_true, bool_false, n_bool_pairs) }?,
        })?;
        unsafe { new_handle(r, out) };
        Ok(())
    })();
    unsafe { report(result, error) }
}

/// Adds every batch of `batches`, in order; each batch is checked on import (see
/// `arrow_io::CheckedReader`) and added atomically (on failure the batches before it
/// stay added).
///
/// The handle's lock is held while the whole stream is pulled: a slow producer blocks
/// `result` (and other `add`s) on other threads, and a producer whose callbacks
/// re-enter `add` / `result` on the same handle deadlocks.
///
/// # Safety
/// `h` is null or a live streaming handle. `batches` is null or a valid
/// ArrowArrayStream; it is consumed (left released) whatever the outcome. `error` is
/// null or valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_streaming_recommender_add(
    h: *mut c_void,
    batches: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = (|| {
        if batches.is_null() {
            return Err(Error::InvalidInput("batch stream is null".into()));
        }
        // Take the stream first: it is consumed whatever the outcome.
        let reader = unsafe { CheckedReader::from_raw(batches) }
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let mut rec = locked(unsafe { handle::<api::StreamingRecommender>(h) }?)?;
        for batch in reader {
            rec.add(&batch.map_err(|e| Error::InvalidInput(e.to_string()))?)?;
        }
        Ok(())
    })();
    unsafe { report(result, error) }
}

/// The recommendation so far (see `api::StreamingRecommender::result`); the state is
/// kept. On success `*out` holds a one-batch stream the caller owns and must release.
///
/// # Safety
/// `h` is null or a live streaming handle. `out` is null or valid for writing one
/// ArrowArrayStream. `error` is null or valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_streaming_recommender_result(
    h: *mut c_void,
    out: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = unsafe { write_result(h, out, api::StreamingRecommender::result) };
    unsafe { report(result, error) }
}

/// Frees a streaming handle. Null is a no-op.
///
/// # Safety
/// `h` is null or a live streaming handle, not used afterwards. Freeing a handle twice,
/// or while an `add` / `result` on it is running on another thread, is undefined
/// behaviour: the caller must ensure every other call has returned.
#[no_mangle]
pub unsafe extern "C" fn analytics_streaming_recommender_free(h: *mut c_void) {
    unsafe { free_handle::<api::StreamingRecommender>(h) }
}

// ── one-shot recommender ─────────────────────────────────────────────────────

/// A one-shot recommender (see `api::OneShotRecommender`). On success `*out` holds a
/// handle the caller frees with `analytics_oneshot_recommender_free`.
///
/// # Safety
/// As for `analytics_streaming_recommender_new`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn analytics_oneshot_recommender_new(
    categorical_threshold: u64,
    zstd_level: i32,
    seed: u64,
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n_bool_pairs: usize,
    out: *mut *mut c_void,
    error: *mut *mut c_char,
) -> c_int {
    let result = (|| {
        if out.is_null() {
            return Err(Error::InvalidInput("out is null".into()));
        }
        let r = api::OneShotRecommender::new(api::OneShotParams {
            categorical_threshold,
            zstd_level,
            seed,
            boolean_pairs: unsafe { boolean_pairs(bool_true, bool_false, n_bool_pairs) }?,
        })?;
        unsafe { new_handle(r, out) };
        Ok(())
    })();
    unsafe { report(result, error) }
}

/// Adds the frame `input` (its batches concatenated, checked on import) and collects
/// its statistics. A second call returns 1 (invalid input).
///
/// # Safety
/// `h` is null or a live one-shot handle. `input` is null or a valid ArrowArrayStream;
/// it is consumed (left released) whatever the outcome. `error` is null or valid for
/// one pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_oneshot_recommender_add(
    h: *mut c_void,
    input: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = (|| {
        if input.is_null() {
            return Err(Error::InvalidInput("input stream is null".into()));
        }
        // Read (and so release) the input first: it is consumed whatever the outcome.
        let batch =
            unsafe { read_stream(input) }.map_err(|e| Error::InvalidInput(e.to_string()))?;
        locked(unsafe { handle::<api::OneShotRecommender>(h) }?)?.add(&batch)
    })();
    unsafe { report(result, error) }
}

/// The recommendation (see `api::OneShotRecommender::result`): the same table on every
/// call. On success `*out` holds a one-batch stream the caller owns and must release.
///
/// # Safety
/// `h` is null or a live one-shot handle. `out` is null or valid for writing one
/// ArrowArrayStream. `error` is null or valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_oneshot_recommender_result(
    h: *mut c_void,
    out: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = unsafe { write_result(h, out, api::OneShotRecommender::result) };
    unsafe { report(result, error) }
}

/// Frees a one-shot handle. Null is a no-op.
///
/// # Safety
/// As for `analytics_streaming_recommender_free`, for a one-shot handle.
#[no_mangle]
pub unsafe extern "C" fn analytics_oneshot_recommender_free(h: *mut c_void) {
    unsafe { free_handle::<api::OneShotRecommender>(h) }
}

#[cfg(test)]
mod tests {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_buffer::Buffer;
    use arrow_data::ArrayData;
    use arrow_schema::DataType;

    use super::*;

    fn stream(columns: Vec<(&str, ArrayRef)>) -> FFI_ArrowArrayStream {
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let schema = batch.schema();
        FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
    }

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn released(s: &mut FFI_ArrowArrayStream) -> bool {
        let raw = (s as *mut FFI_ArrowArrayStream).cast::<crate::common::arrow_io::RawStream>();
        unsafe { (*raw).release }.is_none()
    }

    fn message(error: *mut c_char) -> String {
        let text = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { analytics_free_error(error) };
        text
    }

    unsafe fn new_oneshot(zstd_level: i32, error: *mut *mut c_char) -> (c_int, *mut c_void) {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        let mut h = ptr::null_mut();
        let code = unsafe {
            analytics_oneshot_recommender_new(
                10_000,
                zstd_level,
                0,
                trues.as_ptr(),
                falses.as_ptr(),
                1,
                &mut h,
                error,
            )
        };
        (code, h)
    }

    /// The one-batch result of `h`.
    unsafe fn oneshot_result(h: *mut c_void) -> RecordBatch {
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { analytics_oneshot_recommender_result(h, &mut output, &mut error) },
            OK
        );
        let mut batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        batches.remove(0)
    }

    #[test]
    fn oneshot_recommender_lifecycle() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_oneshot(1, &mut error) };
        assert_eq!(code, OK);
        let mut input = stream(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ]);
        assert_eq!(
            unsafe { analytics_oneshot_recommender_add(h, &mut input, &mut error) },
            OK
        );
        assert!(released(&mut input));
        let out = unsafe { oneshot_result(h) };
        assert_eq!(out.num_rows(), 2);
        let rec = out
            .column_by_name("rec_arrow_type")
            .unwrap()
            .as_string_view();
        assert_eq!((rec.value(0), rec.value(1)), ("uint8", "string"));
        assert_eq!(unsafe { oneshot_result(h) }, out);
        let mut again = stream(vec![("a", ints(&[1]))]);
        assert_eq!(
            unsafe { analytics_oneshot_recommender_add(h, &mut again, &mut error) },
            INVALID_INPUT
        );
        assert!(message(error).contains("already been added"));
        assert!(released(&mut again));
        unsafe { analytics_oneshot_recommender_free(h) };
        unsafe { analytics_oneshot_recommender_free(ptr::null_mut()) };
    }

    #[test]
    fn oneshot_recommender_errors() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_oneshot(99, &mut error) };
        assert_eq!((code, h.is_null()), (INVALID_INPUT, true));
        assert!(message(error).contains("zstd_level"));

        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_oneshot(1, &mut error) };
        assert_eq!(code, OK);
        let mut dup = stream(vec![("a", ints(&[1, 2])), ("a", ints(&[3, 4]))]);
        let code = unsafe { analytics_oneshot_recommender_add(h, &mut dup, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "duplicate column \"a\"");

        let mut error = ptr::null_mut();
        let code = unsafe { analytics_oneshot_recommender_add(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "input stream is null");

        let mut error = ptr::null_mut();
        let code = unsafe { analytics_oneshot_recommender_result(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "output stream is null");

        let mut output = FFI_ArrowArrayStream::empty();
        let code = unsafe {
            analytics_oneshot_recommender_result(ptr::null_mut(), &mut output, ptr::null_mut())
        };
        assert_eq!(code, INVALID_INPUT); // a null error slot is allowed
        unsafe { analytics_oneshot_recommender_free(h) };
    }

    unsafe fn new_recommender(block_rows: u64, error: *mut *mut c_char) -> (c_int, *mut c_void) {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        let mut h = ptr::null_mut();
        let code = unsafe {
            analytics_streaming_recommender_new(
                1 << 20,
                block_rows,
                10_000,
                1,
                0,
                trues.as_ptr(),
                falses.as_ptr(),
                1,
                &mut h,
                error,
            )
        };
        (code, h)
    }

    #[test]
    fn streaming_recommender_lifecycle() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(1 << 16, &mut error) };
        assert_eq!(code, OK);
        for _ in 0..2 {
            let mut input = stream(vec![("a", ints(&[0, 5, 7]))]);
            assert_eq!(
                unsafe { analytics_streaming_recommender_add(h, &mut input, &mut error) },
                OK
            );
        }
        let mut output = FFI_ArrowArrayStream::empty();
        assert_eq!(
            unsafe { analytics_streaming_recommender_result(h, &mut output, &mut error) },
            OK
        );
        let batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let rec = batches[0]
            .column_by_name("rec_arrow_type")
            .unwrap()
            .as_string_view();
        assert_eq!(rec.value(0), "uint8");
        unsafe { analytics_streaming_recommender_free(h) };
        unsafe { analytics_streaming_recommender_free(ptr::null_mut()) };
    }

    #[test]
    fn streaming_recommender_errors() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(0, &mut error) };
        assert_eq!((code, h.is_null()), (INVALID_INPUT, true));
        assert!(message(error).contains("block_rows"));
        let mut input = stream(vec![("a", ints(&[1]))]);
        let mut error = ptr::null_mut();
        let code =
            unsafe { analytics_streaming_recommender_add(ptr::null_mut(), &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert!(message(error).contains("handle"));
        // The stream was consumed even though the handle was null.
        assert!(released(&mut input));
    }

    /// `result`'s `n_rows` for the first column.
    unsafe fn result_rows(h: *mut c_void) -> u64 {
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { analytics_streaming_recommender_result(h, &mut output, &mut error) },
            OK
        );
        let batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        batches[0]
            .column_by_name("n_rows")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>()
            .value(0)
    }

    #[test]
    fn streaming_recommender_null_arguments_are_invalid_input() {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        let mut error = ptr::null_mut();
        let code = unsafe {
            analytics_streaming_recommender_new(
                1 << 20,
                1 << 16,
                10_000,
                1,
                0,
                trues.as_ptr(),
                falses.as_ptr(),
                1,
                ptr::null_mut(),
                &mut error,
            )
        };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "out is null");

        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(1 << 16, &mut error) };
        assert_eq!(code, OK);
        let code = unsafe { analytics_streaming_recommender_add(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "batch stream is null");

        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        let code = unsafe {
            analytics_streaming_recommender_result(ptr::null_mut(), &mut output, &mut error)
        };
        assert_eq!(code, INVALID_INPUT);
        assert!(message(error).contains("handle"));

        let mut error = ptr::null_mut();
        let code =
            unsafe { analytics_streaming_recommender_result(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "output stream is null");
        unsafe { analytics_streaming_recommender_free(h) };
    }

    #[test]
    fn streaming_recommender_results_repeatedly_and_adds_after() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(1 << 16, &mut error) };
        assert_eq!(code, OK);
        let mut input = stream(vec![("a", ints(&[0, 5, 7]))]);
        assert_eq!(
            unsafe { analytics_streaming_recommender_add(h, &mut input, &mut error) },
            OK
        );
        assert_eq!(unsafe { result_rows(h) }, 3);
        assert_eq!(unsafe { result_rows(h) }, 3);
        let mut input = stream(vec![("a", ints(&[1, 2]))]);
        assert_eq!(
            unsafe { analytics_streaming_recommender_add(h, &mut input, &mut error) },
            OK
        );
        assert_eq!(unsafe { result_rows(h) }, 5);
        assert!(error.is_null());
        unsafe { analytics_streaming_recommender_free(h) };
    }

    #[test]
    fn streaming_recommender_rejects_a_malformed_stream() {
        // A string column whose offsets decrease: without the checked import arrow-rs
        // would build it and a kernel would slice out of bounds (an abort in release).
        let data = unsafe {
            ArrayData::builder(DataType::Utf8)
                .len(2)
                .add_buffer(Buffer::from_vec(vec![0i32, 2, 1]))
                .add_buffer(Buffer::from_vec(b"ab".to_vec()))
                .build_unchecked()
        };
        let mut input = stream(vec![("s", Arc::new(StringArray::from(data)) as ArrayRef)]);
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(1 << 16, &mut error) };
        assert_eq!(code, OK);
        let code = unsafe { analytics_streaming_recommender_add(h, &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        let m = message(error);
        assert!(m.contains("offset"), "{m}");
        assert!(released(&mut input));
        unsafe { analytics_streaming_recommender_free(h) };
    }

    #[test]
    fn oneshot_recommender_rejects_a_malformed_stream() {
        let data = unsafe {
            ArrayData::builder(DataType::Utf8)
                .len(2)
                .add_buffer(Buffer::from_vec(vec![0i32, 2, 1]))
                .add_buffer(Buffer::from_vec(b"ab".to_vec()))
                .build_unchecked()
        };
        let mut input = stream(vec![("s", Arc::new(StringArray::from(data)) as ArrayRef)]);
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_oneshot(1, &mut error) };
        assert_eq!(code, OK);
        let code = unsafe { analytics_oneshot_recommender_add(h, &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        let m = message(error);
        assert!(m.contains("offset"), "{m}");
        assert!(released(&mut input));
        unsafe { analytics_oneshot_recommender_free(h) };
    }

    #[test]
    fn oneshot_recommender_null_handle_is_invalid_input() {
        let mut input = stream(vec![("a", ints(&[1]))]);
        let mut error = ptr::null_mut();
        let code =
            unsafe { analytics_oneshot_recommender_add(ptr::null_mut(), &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert!(message(error).contains("handle"));
        assert!(released(&mut input));
    }

    #[test]
    fn compute_errors_map_to_code_two() {
        assert_eq!(
            unsafe {
                report(
                    Err(crate::bindings::api::Error::Compute("boom".into())),
                    ptr::null_mut(),
                )
            },
            COMPUTE
        );
    }
}
