//! C ABI over api.rs, for Java (Panama FFM) and any other language that can call C.
//! Tables cross as Arrow C Streams; parameters as plain C values. Every entry point
//! returns 0 on success, 1 on invalid input or 2 on a compute failure (mirroring
//! `api::Error`); on failure `*error` holds a message the caller frees with
//! `analytics_free_error`. Built without Python by
//! `cargo build --release --no-default-features --target-dir target/capi`.
//! The streaming recommender's `mark_ineligible` is intentionally not exposed: it exists
//! for Polars' Int128 / UInt128 columns, which Arrow input cannot carry.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::RecordBatchIterator;

use crate::api::{self, Error};
use crate::arrow_io::{read_stream, CheckedReader};

const OK: c_int = 0;
const INVALID_INPUT: c_int = 1;
const COMPUTE: c_int = 2;

/// Writes `result` as a return code, and on failure its message into `*error`.
///
/// SAFETY: `error` is null or valid for one pointer write.
unsafe fn finish(result: api::Result<()>, error: *mut *mut c_char) -> c_int {
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

/// SAFETY: as for `analytics_describe_and_recommend`.
#[allow(clippy::too_many_arguments)]
unsafe fn describe_and_recommend(
    input: *mut FFI_ArrowArrayStream,
    seed: u64,
    zstd_level: i32,
    categorical_threshold: u64,
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n_bool_pairs: usize,
    output: *mut FFI_ArrowArrayStream,
) -> api::Result<()> {
    if input.is_null() {
        return Err(Error::InvalidInput("input stream is null".into()));
    }
    // Read (and so release) the input first: it is consumed whatever the outcome.
    let batch = unsafe { read_stream(input) }.map_err(|e| Error::InvalidInput(e.to_string()))?;
    if output.is_null() {
        return Err(Error::InvalidInput("output stream is null".into()));
    }
    let trues = unsafe { strings(bool_true, n_bool_pairs) }?;
    let falses = unsafe { strings(bool_false, n_bool_pairs) }?;
    let out = api::describe_and_recommend(
        &batch,
        seed,
        zstd_level,
        categorical_threshold,
        trues.into_iter().zip(falses).collect(),
    )?;
    let schema = out.schema();
    let stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(out)], schema)));
    // `*output` may be uninitialised or released: overwrite it without dropping.
    unsafe { ptr::write(output, stream) };
    Ok(())
}

/// Describe's table, the size columns and the `rec_*` columns per column of `input`
/// (see `api::describe_and_recommend`). On success
/// `*output` holds a one-batch stream the caller owns and must release.
///
/// # Safety
/// `input` is null or a valid ArrowArrayStream; it is consumed (left released).
/// `output` is null or valid for writing one ArrowArrayStream. When
/// `n_bool_pairs > 0`, `bool_true` and `bool_false` each point to `n_bool_pairs`
/// NUL-terminated UTF-8 strings. `error` is null or valid for one pointer write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn analytics_describe_and_recommend(
    input: *mut FFI_ArrowArrayStream,
    seed: u64,
    zstd_level: i32,
    categorical_threshold: u64,
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n_bool_pairs: usize,
    output: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = unsafe {
        describe_and_recommend(
            input,
            seed,
            zstd_level,
            categorical_threshold,
            bool_true,
            bool_false,
            n_bool_pairs,
            output,
        )
    };
    unsafe { finish(result, error) }
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

/// What an `AnalyticsRecommender *` points to. The handle crosses as `void *`: a
/// pointer to a Rust type in an `extern "C"` signature trips
/// `improper_ctypes_definitions`.
type Recommender = Mutex<api::StreamingRecommender>;

// The handle is shared across caller threads: checked in the C-only build too.
const _: () = {
    fn thread_safe<T: Send + Sync>() {}
    let _ = thread_safe::<Recommender>;
};

/// SAFETY: `h` is null or a live handle from `analytics_recommender_new`.
unsafe fn recommender<'a>(h: *mut c_void) -> api::Result<&'a Recommender> {
    if h.is_null() {
        return Err(Error::InvalidInput("recommender handle is null".into()));
    }
    Ok(unsafe { &*(h as *const Recommender) })
}

fn locked(r: &Recommender) -> api::Result<std::sync::MutexGuard<'_, api::StreamingRecommender>> {
    r.lock()
        .map_err(|_| Error::Compute("recommender poisoned by an earlier panic".into()))
}

/// A streaming recommender (see `api::StreamingRecommender`). On success `*out` holds a
/// handle the caller frees with `analytics_recommender_free`. Concurrent
/// `analytics_recommender_add` / `analytics_recommender_finish` calls on one handle
/// from different threads are safe (serialised by a lock); freeing is not (see
/// `analytics_recommender_free`).
///
/// # Safety
/// `out` is null or valid for one pointer write. When `n_bool_pairs > 0`, `bool_true`
/// and `bool_false` each point to `n_bool_pairs` NUL-terminated UTF-8 strings. `error`
/// is null or valid for one pointer write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn analytics_recommender_new(
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
        let trues = unsafe { strings(bool_true, n_bool_pairs) }?;
        let falses = unsafe { strings(bool_false, n_bool_pairs) }?;
        let r = api::StreamingRecommender::new(api::StreamingParams {
            reservoir_rows,
            block_rows,
            categorical_threshold,
            zstd_level,
            seed,
            boolean_pairs: trues.into_iter().zip(falses).collect(),
        })?;
        unsafe { *out = Box::into_raw(Box::new(Mutex::new(r))) as *mut c_void };
        Ok(())
    })();
    unsafe { finish(result, error) }
}

/// Adds every batch of `batches`, in order; each batch is checked on import (see
/// `arrow_io::CheckedReader`) and added atomically (on failure the batches before it
/// stay added).
///
/// The handle's lock is held while the whole stream is pulled: a slow producer blocks
/// `finish` (and other `add`s) on other threads, and a producer whose callbacks
/// re-enter `add` / `finish` on the same handle deadlocks.
///
/// # Safety
/// `h` is null or a live handle. `batches` is null or a valid ArrowArrayStream; it is
/// consumed (left released) whatever the outcome. `error` is null or valid for one
/// pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_recommender_add(
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
        let mut rec = locked(unsafe { recommender(h) }?)?;
        for batch in reader {
            rec.add(&batch.map_err(|e| Error::InvalidInput(e.to_string()))?)?;
        }
        Ok(())
    })();
    unsafe { finish(result, error) }
}

/// The recommendation so far (see `api::StreamingRecommender::finish`); the state is
/// kept. On success `*out` holds a one-batch stream the caller owns and must release.
///
/// # Safety
/// `h` is null or a live handle. `out` is null or valid for writing one
/// ArrowArrayStream. `error` is null or valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn analytics_recommender_finish(
    h: *mut c_void,
    out: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = (|| {
        if out.is_null() {
            return Err(Error::InvalidInput("output stream is null".into()));
        }
        let batch = locked(unsafe { recommender(h) }?)?.finish()?;
        let schema = batch.schema();
        let stream =
            FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(batch)], schema)));
        // `*out` may be uninitialised or released: overwrite it without dropping.
        unsafe { ptr::write(out, stream) };
        Ok(())
    })();
    unsafe { finish(result, error) }
}

/// Frees a handle. Null is a no-op.
///
/// # Safety
/// `h` is null or a live handle, not used afterwards. Freeing a handle twice, or
/// while an `add` / `finish` on it is running on another thread, is undefined
/// behaviour: the caller must ensure every other call has returned.
#[no_mangle]
pub unsafe extern "C" fn analytics_recommender_free(h: *mut c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h as *mut Recommender) });
    }
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

    /// Python's defaults: seed 0, ZSTD level 1, threshold 10 000, ("true", "false").
    unsafe fn call(
        input: *mut FFI_ArrowArrayStream,
        output: *mut FFI_ArrowArrayStream,
        error: *mut *mut c_char,
    ) -> c_int {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        unsafe {
            analytics_describe_and_recommend(
                input,
                0,
                1,
                10_000,
                trues.as_ptr(),
                falses.as_ptr(),
                1,
                output,
                error,
            )
        }
    }

    fn released(s: &mut FFI_ArrowArrayStream) -> bool {
        let raw = (s as *mut FFI_ArrowArrayStream).cast::<crate::arrow_io::RawStream>();
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

    #[test]
    fn toy_data_round_trips_through_the_c_abi() {
        let mut input = stream(vec![
            ("a", ints(&[0, 5, 7])),
            (
                "s",
                Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef,
            ),
        ]);
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { call(&mut input, &mut output, &mut error) }, OK);
        assert!(error.is_null());
        let batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        let out = &batches[0];
        assert_eq!(out.num_rows(), 2);
        let rec = out
            .column_by_name("rec_arrow_type")
            .unwrap()
            .as_string_view();
        assert_eq!((rec.value(0), rec.value(1)), ("uint8", "string"));
    }

    #[test]
    fn duplicate_columns_are_invalid_input_with_a_message() {
        let mut input = stream(vec![("a", ints(&[1, 2])), ("a", ints(&[3, 4]))]);
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { call(&mut input, &mut output, &mut error) },
            INVALID_INPUT
        );
        assert_eq!(message(error), "duplicate column \"a\"");
    }

    #[test]
    fn null_streams_are_invalid_input() {
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { call(ptr::null_mut(), &mut output, &mut error) },
            INVALID_INPUT
        );
        assert_eq!(message(error), "input stream is null");

        let mut input = stream(vec![("a", ints(&[1]))]);
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { call(&mut input, ptr::null_mut(), &mut error) },
            INVALID_INPUT
        );
        assert_eq!(message(error), "output stream is null");
    }

    #[test]
    fn null_error_slot_is_allowed() {
        let mut output = FFI_ArrowArrayStream::empty();
        assert_eq!(
            unsafe { call(ptr::null_mut(), &mut output, ptr::null_mut()) },
            INVALID_INPUT
        );
        unsafe { analytics_free_error(ptr::null_mut()) };
    }

    unsafe fn new_recommender(block_rows: u64, error: *mut *mut c_char) -> (c_int, *mut c_void) {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        let mut h = ptr::null_mut();
        let code = unsafe {
            analytics_recommender_new(
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
                unsafe { analytics_recommender_add(h, &mut input, &mut error) },
                OK
            );
        }
        let mut output = FFI_ArrowArrayStream::empty();
        assert_eq!(
            unsafe { analytics_recommender_finish(h, &mut output, &mut error) },
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
        unsafe { analytics_recommender_free(h) };
        unsafe { analytics_recommender_free(ptr::null_mut()) };
    }

    #[test]
    fn streaming_recommender_errors() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(0, &mut error) };
        assert_eq!((code, h.is_null()), (INVALID_INPUT, true));
        assert!(message(error).contains("block_rows"));
        let mut input = stream(vec![("a", ints(&[1]))]);
        let mut error = ptr::null_mut();
        let code = unsafe { analytics_recommender_add(ptr::null_mut(), &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert!(message(error).contains("handle"));
        // The stream was consumed even though the handle was null.
        assert!(released(&mut input));
    }

    /// `finish`'s `n_rows` for the first column.
    unsafe fn finished_rows(h: *mut c_void) -> u64 {
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(
            unsafe { analytics_recommender_finish(h, &mut output, &mut error) },
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
            analytics_recommender_new(
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
        let code = unsafe { analytics_recommender_add(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "batch stream is null");

        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        let code =
            unsafe { analytics_recommender_finish(ptr::null_mut(), &mut output, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert!(message(error).contains("handle"));

        let mut error = ptr::null_mut();
        let code = unsafe { analytics_recommender_finish(h, ptr::null_mut(), &mut error) };
        assert_eq!(code, INVALID_INPUT);
        assert_eq!(message(error), "output stream is null");
        unsafe { analytics_recommender_free(h) };
    }

    #[test]
    fn streaming_recommender_finishes_repeatedly_and_adds_after_finishing() {
        let mut error = ptr::null_mut();
        let (code, h) = unsafe { new_recommender(1 << 16, &mut error) };
        assert_eq!(code, OK);
        let mut input = stream(vec![("a", ints(&[0, 5, 7]))]);
        assert_eq!(
            unsafe { analytics_recommender_add(h, &mut input, &mut error) },
            OK
        );
        assert_eq!(unsafe { finished_rows(h) }, 3);
        assert_eq!(unsafe { finished_rows(h) }, 3);
        let mut input = stream(vec![("a", ints(&[1, 2]))]);
        assert_eq!(
            unsafe { analytics_recommender_add(h, &mut input, &mut error) },
            OK
        );
        assert_eq!(unsafe { finished_rows(h) }, 5);
        assert!(error.is_null());
        unsafe { analytics_recommender_free(h) };
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
        let code = unsafe { analytics_recommender_add(h, &mut input, &mut error) };
        assert_eq!(code, INVALID_INPUT);
        let m = message(error);
        assert!(m.contains("offset"), "{m}");
        assert!(released(&mut input));
        unsafe { analytics_recommender_free(h) };
    }

    #[test]
    fn compute_errors_map_to_code_two() {
        assert_eq!(
            unsafe {
                finish(
                    Err(crate::api::Error::Compute("boom".into())),
                    ptr::null_mut(),
                )
            },
            COMPUTE
        );
    }
}
