# Java Binding for `describe_and_recommend` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Call `api::describe_and_recommend` from Java through a C ABI, proven by JUnit tests on toy data and run in CI.

**Architecture:** pyo3 moves behind a default cargo feature `python`, so a Python-free library can be built. A new `src/capi.rs` exports `extern "C"` functions: Arrow C Streams go in and out, and parameters are plain C values. A Maven project at `services/analytics/bindings/java/` calls those functions with Panama FFM (`java.lang.foreign`). Tables cross with Arrow Java's `arrow-c-data` (`Data.exportArrayStream` / `Data.importArrayStream`).

**Tech Stack:** Rust (arrow-rs 60, polars 0.51), JDK 25 (Temurin), Maven 3.9.15 (wrapper), Arrow Java 18.1.0 (`arrow-vector`, `arrow-c-data`, `arrow-memory-netty`), JUnit Jupiter 6.0.3, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-28-java-binding-design.md`

---

## Facts the engineer needs

- **Repo root:** `C:\Users\Alexander\turbo-parakeet`. The Rust crate is at `services/analytics/`. All shell commands below are Git Bash.
- **Python env:** `/c/Users/Alexander/miniconda3/envs/p312` (CLAUDE.md shows a stale `C:\Users\Ben\...` path — ignore it).
- **Rust tests without Python:** `cargo test --lib --no-default-features <filter>` needs no Python at all, once Task 1 has landed. Running with the default features (pyo3) on Windows needs `PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe` and that directory on `PATH`. Otherwise you get `STATUS_DLL_NOT_FOUND`.
- **Two library builds, two target dirs:**
  - `maturin develop --release` writes the **Python-linked** `analytics.dll` to `target/release/`.
  - The Java build goes to `target/capi/release/` via `--target-dir target/capi`, so the two never overwrite each other.
- **Release builds are slow.** `lto = "fat"` means a cold release build takes several minutes. Give those commands a 10-minute timeout.
- **`cargo fmt --check` already fails on `main`,** before any of this work, and there is no `rustfmt.toml`. **Do not run `cargo fmt` on the whole crate.** Format only the new file, with `rustfmt --edition 2021 src/capi.rs`.
- **Arrow Java's C Data module uses JNI internally.** The `arrow-c-data-18.1.0.jar` bundles `arrow_cdata_jni/x86_64/arrow_cdata_jni.dll` and the Linux `.so`, which has been checked. JDK 25 therefore needs `--enable-native-access=ALL-UNNAMED`, and Arrow's memory module needs `--add-opens=java.base/java.nio=ALL-UNNAMED`.
- **Output schema of `describe_and_recommend`,** checked through the Python binding: 57 `uint64`, 14 `uint32`, 6 `decimal128(38,0)`, 6 `large_list<uint64>`, 3 `string_view` (`column`, `rec_arrow_type`, `rec_polars_type`), 2 `double`, 2 `bool`, and `rec_candidates: large_list<struct<… string_view …>>`.
  - For the toy data `a: Int64 [0,5,7]`, `s: Utf8 ["x","y","x"]`: `column = ["a","s"]` and `rec_arrow_type = ["uint8","string"]`.
- **Existing tooling:**
  - The Maven 3.9.15 binary is at `/c/Users/Alexander/.m2/wrapper/dists/apache-maven-3.9.15-bin/4faaaa08/apache-maven-3.9.15/bin/mvn`.
  - `maven-compiler-plugin` 3.15.0 and `maven-surefire-plugin` 3.5.6 are cached.
  - JDK 25 is on `PATH`.
- **Git ignore rules:** `.gitignore` already ignores every `target/` directory, including the Maven one. It also ignores `docs/superpowers/`, so the spec and this plan are local files and are never committed.

## File map

| File | Change | Responsibility |
|---|---|---|
| `services/analytics/Cargo.toml` | modify | `python` feature; pyo3 optional |
| `services/analytics/src/lib.rs` | modify | gate `mod python`, add `mod capi` |
| `services/analytics/src/arrow_io.rs` | modify | new `read_stream` (C stream → one RecordBatch) + test |
| `services/analytics/src/python.rs` | modify | `read_batch` delegates to `read_stream` |
| `services/analytics/src/capi.rs` | create | C ABI: `analytics_describe_and_recommend`, `analytics_free_error` + tests |
| `services/analytics/bindings/java/pom.xml` | create | Maven build, deps, surefire JVM flags + library dir |
| `services/analytics/bindings/java/mvnw`, `mvnw.cmd`, `.mvn/wrapper/maven-wrapper.properties` | create (generated) | Maven wrapper 3.9.15 |
| `services/analytics/bindings/java/src/main/java/io/github/benssutton/analytics/Params.java` | create | parameter record + defaults |
| `services/analytics/bindings/java/src/main/java/io/github/benssutton/analytics/Analytics.java` | create | library loading, FFM downcall, error mapping |
| `services/analytics/bindings/java/src/test/java/io/github/benssutton/analytics/AnalyticsTest.java` | create | JUnit tests |
| `.github/workflows/ci-cd.yml` | modify | `java-tests` job, clippy without default features, CodeQL Java, status needs |
| `CLAUDE.md` | modify | document the C ABI layer, the feature, the Java commands |

---

### Task 1: Make pyo3 optional behind a default `python` feature

**Files:**
- Modify: `services/analytics/Cargo.toml` (the `[lib]` block and the `pyo3` line)
- Modify: `services/analytics/src/lib.rs:6`

- [ ] **Step 1: Confirm the Python-free build fails today**

Run (from `services/analytics/`):
```bash
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -3
```
Expected: this build *succeeds*, but the DLL still links Python, because pyo3 is not optional yet. Check:
```bash
grep -a -c "python3" target/capi/release/analytics.dll
```
Expected: a count ≥ 1, because the DLL imports `python3.dll`. This is the failing condition that the task fixes.

- [ ] **Step 2: Add the feature to `Cargo.toml`**

Insert this block directly after the `[lib]` block (after the `crate-type = ["cdylib"] ...` line):

```toml

[features]
default = ["python"]
# The pyo3 binding (python.rs). Off for the C ABI build used by Java:
# cargo build --release --no-default-features --target-dir target/capi
python = ["dep:pyo3"]
```

Change the pyo3 dependency line from:
```toml
pyo3 = { version = "0.25.0", features = ["extension-module", "abi3-py39"] }
```
to:
```toml
pyo3 = { version = "0.25.0", features = ["extension-module", "abi3-py39"], optional = true }
```

- [ ] **Step 3: Gate the module in `lib.rs`**

In `services/analytics/src/lib.rs`, replace line 6:
```rust
mod python;
```
with:
```rust
#[cfg(feature = "python")]
mod python;
```

- [ ] **Step 4: Verify the Python-free build no longer imports Python**

```bash
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -3
grep -a -c "python3" target/capi/release/analytics.dll
```
Expected: the build finishes with `Finished`. `grep` prints `0` and exits 1.

- [ ] **Step 5: Verify the default (Python) build still compiles**

```bash
PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe cargo check --all-targets 2>&1 | tail -3
```
Expected: `Finished` with no errors.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/Cargo.toml services/analytics/src/lib.rs
git commit -m "Make pyo3 optional behind a default python feature

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `arrow_io::read_stream`, shared by both bindings

**Files:**
- Modify: `services/analytics/src/arrow_io.rs` (imports at the top, new function after `import_batch`, new test in the existing `mod tests` at line ~94)
- Modify: `services/analytics/src/python.rs:8-9` (imports), `:34-51` (`read_batch`)

- [ ] **Step 1: Write the failing test**

Add this test inside the existing `#[cfg(test)] mod tests { ... }` block in `services/analytics/src/arrow_io.rs`. Put it at the end of the block, before its closing `}`:

```rust
    #[test]
    fn read_stream_concatenates_batches_and_consumes_the_stream() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;
        use arrow_array::{Int64Array, RecordBatchIterator};

        let batch = |v: Vec<i64>| {
            RecordBatch::try_from_iter([("a", Arc::new(Int64Array::from(v)) as ArrayRef)]).unwrap()
        };
        let (b1, b2) = (batch(vec![1, 2]), batch(vec![3]));
        let schema = b1.schema();
        let mut stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(b1), Ok(b2)], schema)));
        let out = unsafe { read_stream(&mut stream) }.unwrap();
        assert_eq!(out.num_rows(), 3);
        // The stream was moved out and left released, so a second read fails.
        assert!(unsafe { read_stream(&mut stream) }.is_err());
    }
```

The existing test module starts with `use super::*;`, which already brings in `Arc`, `ArrayRef` and `RecordBatch`, so it needs no other imports.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test --lib --no-default-features arrow_io::tests::read_stream 2>&1 | tail -5
```
Expected: a compile error, `cannot find function 'read_stream'`.

- [ ] **Step 3: Implement `read_stream`**

In `services/analytics/src/arrow_io.rs`, change the import line:
```rust
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
```
to:
```rust
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, RecordBatchReader};
```

Add this function directly after `import_batch`:

```rust
/// A whole Arrow C stream as one RecordBatch (kernels expect one chunk per column).
/// The stream is consumed: `*stream` is left released whatever the outcome.
///
/// SAFETY: `stream` points to a valid ArrowArrayStream struct (released or not).
pub(crate) unsafe fn read_stream(stream: *mut FFI_ArrowArrayStream) -> std::result::Result<RecordBatch, arrow_schema::ArrowError> {
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream) }?;
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    arrow_select::concat::concat_batches(&schema, &batches)
}
```
(The `std::result::` prefixes guard against a `Result` alias coming in through `use polars::prelude::*`.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --lib --no-default-features arrow_io::tests::read_stream 2>&1 | tail -5
```
Expected: `test result: ok. 1 passed`.

- [ ] **Step 5: Make `python.rs` delegate to it**

In `services/analytics/src/python.rs`, replace lines 8-9:
```rust
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
```
with:
```rust
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchIterator};
```
and add, next to the existing `use crate::api;`:
```rust
use crate::arrow_io::read_stream;
```

In `read_batch`, replace these last lines:
```rust
    unsafe { reject_wide_integers(stream.cast())? };
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream) }.map_err(value_error)?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(value_error)?;
    arrow_select::concat::concat_batches(&schema, &batches).map_err(value_error)
}
```
with:
```rust
    unsafe { reject_wide_integers(stream.cast())? };
    unsafe { read_stream(stream) }.map_err(value_error)
}
```
Keep the SAFETY comment above these lines unchanged.

- [ ] **Step 6: Verify both builds compile and the full Rust suite passes**

```bash
export PATH=/c/Users/Alexander/miniconda3/envs/p312:$PATH
PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe cargo clippy --all-features --all-targets -- -D warnings 2>&1 | tail -3
cargo clippy --no-default-features --all-targets -- -D warnings 2>&1 | tail -3
PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe cargo test --lib 2>&1 | tail -3
```
Expected: both clippy runs finish with `Finished` and no warnings, and the tests report `test result: ok`.

If clippy reports warnings in files this plan does not touch, record them and move on. Only warnings in `arrow_io.rs`, `python.rs`, `lib.rs` or `capi.rs` are this plan's responsibility.

- [ ] **Step 7: Verify the Python binding still works end to end**

From `services/analytics/`:
```bash
export PATH=/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH
CONDA_PREFIX=/c/Users/Alexander/miniconda3/envs/p312 maturin develop --release 2>&1 | tail -2
cd ../.. && python -m pytest tests/test_recommend.py tests/test_describe.py -q 2>&1 | tail -3
```
Expected: maturin reports `Installed analytics`, and pytest reports all tests passed (skips are fine).

- [ ] **Step 8: Commit**

```bash
git add services/analytics/src/arrow_io.rs services/analytics/src/python.rs
git commit -m "Move Arrow C stream reading into arrow_io::read_stream

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The C ABI — `src/capi.rs`

**Files:**
- Create: `services/analytics/src/capi.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod capi;`)

- [ ] **Step 1: Write the failing tests**

Create `services/analytics/src/capi.rs` containing only the module doc, constants and tests for now:

```rust
//! C ABI over api.rs, for Java (Panama FFM) and any other language that can call C.
//! Tables cross as Arrow C Streams; parameters as plain C values. Every entry point
//! returns 0 on success, 1 on invalid input or 2 on a compute failure (mirroring
//! `api::Error`); on failure `*error` holds a message the caller frees with
//! `analytics_free_error`. Built without Python by
//! `cargo build --release --no-default-features --target-dir target/capi`.

use std::ffi::c_int;

const OK: c_int = 0;
const INVALID_INPUT: c_int = 1;
const COMPUTE: c_int = 2;

#[cfg(test)]
mod tests {
    use std::ffi::{c_char, CStr};
    use std::ptr;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray};

    use super::*;

    fn stream(columns: Vec<(&str, ArrayRef)>) -> FFI_ArrowArrayStream {
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let schema = batch.schema();
        FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
    }

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// Python's defaults: seed 0, ZSTD level 1, no population, threshold 10 000, ("true", "false").
    unsafe fn call(input: *mut FFI_ArrowArrayStream, output: *mut FFI_ArrowArrayStream, error: *mut *mut c_char) -> c_int {
        let (trues, falses) = ([c"true".as_ptr()], [c"false".as_ptr()]);
        unsafe { analytics_describe_and_recommend(input, 0, 1, -1, 10_000, trues.as_ptr(), falses.as_ptr(), 1, output, error) }
    }

    fn message(error: *mut c_char) -> String {
        let text = unsafe { CStr::from_ptr(error) }.to_str().unwrap().to_owned();
        unsafe { analytics_free_error(error) };
        text
    }

    #[test]
    fn toy_data_round_trips_through_the_c_abi() {
        let mut input = stream(vec![("a", ints(&[0, 5, 7])), ("s", Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef)]);
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { call(&mut input, &mut output, &mut error) }, OK);
        assert!(error.is_null());
        let batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output).unwrap().collect::<Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let out = &batches[0];
        assert_eq!(out.num_rows(), 2);
        let rec = out.column_by_name("rec_arrow_type").unwrap().as_string_view();
        assert_eq!((rec.value(0), rec.value(1)), ("uint8", "string"));
    }

    #[test]
    fn duplicate_columns_are_invalid_input_with_a_message() {
        let mut input = stream(vec![("a", ints(&[1, 2])), ("a", ints(&[3, 4]))]);
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { call(&mut input, &mut output, &mut error) }, INVALID_INPUT);
        assert_eq!(message(error), "duplicate column \"a\"");
    }

    #[test]
    fn null_streams_are_invalid_input() {
        let mut output = FFI_ArrowArrayStream::empty();
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { call(ptr::null_mut(), &mut output, &mut error) }, INVALID_INPUT);
        assert_eq!(message(error), "input stream is null");

        let mut input = stream(vec![("a", ints(&[1]))]);
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { call(&mut input, ptr::null_mut(), &mut error) }, INVALID_INPUT);
        assert_eq!(message(error), "output stream is null");
    }

    #[test]
    fn null_error_slot_is_allowed() {
        let mut output = FFI_ArrowArrayStream::empty();
        assert_eq!(unsafe { call(ptr::null_mut(), &mut output, ptr::null_mut()) }, INVALID_INPUT);
        unsafe { analytics_free_error(ptr::null_mut()) };
    }

    #[test]
    fn compute_errors_map_to_code_two() {
        assert_eq!(unsafe { finish(Err(crate::api::Error::Compute("boom".into())), ptr::null_mut()) }, COMPUTE);
    }
}
```

In `services/analytics/src/lib.rs`, add `mod capi;` on the line after `mod api;`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test --lib --no-default-features capi:: 2>&1 | tail -5
```
Expected: compile errors, `cannot find function 'analytics_describe_and_recommend'` (and `analytics_free_error`, `finish`).

- [ ] **Step 3: Implement the C ABI**

In `services/analytics/src/capi.rs`, replace the line `use std::ffi::c_int;` with the imports below. Then insert the implementation between the constants and the `#[cfg(test)]` line:

```rust
use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::RecordBatchIterator;

use crate::api::{self, Error};
use crate::arrow_io::read_stream;
```

```rust
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
    population_rows: i64,
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
    let population_rows = u64::try_from(population_rows).ok();
    let out = api::describe_and_recommend(&batch, seed, zstd_level, population_rows, categorical_threshold, trues.into_iter().zip(falses).collect())?;
    let schema = out.schema();
    let stream = FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(out)], schema)));
    // `*output` may be uninitialised or released: overwrite it without dropping.
    unsafe { ptr::write(output, stream) };
    Ok(())
}

/// Describe's table, the size columns and the `rec_*` columns per column of `input`
/// (see `api::describe_and_recommend`). `population_rows < 0` means none. On success
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
    population_rows: i64,
    categorical_threshold: u64,
    bool_true: *const *const c_char,
    bool_false: *const *const c_char,
    n_bool_pairs: usize,
    output: *mut FFI_ArrowArrayStream,
    error: *mut *mut c_char,
) -> c_int {
    let result = unsafe {
        describe_and_recommend(input, seed, zstd_level, population_rows, categorical_threshold, bool_true, bool_false, n_bool_pairs, output)
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
```

Leave the test module's own imports as they are. An explicit `use` that duplicates a `super::*` glob import is not a warning.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --lib --no-default-features capi:: 2>&1 | tail -5
```
Expected: `test result: ok. 5 passed`.

- [ ] **Step 5: Format the new file and lint both builds**

```bash
rustfmt --edition 2021 src/capi.rs
cargo clippy --no-default-features --all-targets -- -D warnings 2>&1 | tail -3
PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe cargo clippy --all-features --all-targets -- -D warnings 2>&1 | tail -3
```
Expected: no warnings from `capi.rs`. Re-run the Step 4 tests after `rustfmt` and confirm they still pass.

- [ ] **Step 6: Build the Java library and confirm the exports**

```bash
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -1
grep -a -c "analytics_describe_and_recommend" target/capi/release/analytics.dll
grep -a -c "analytics_free_error" target/capi/release/analytics.dll
```
Expected: `Finished`, then two counts ≥ 1, showing the symbols are in the export table.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/src/capi.rs services/analytics/src/lib.rs
git commit -m "Add C ABI for describe_and_recommend (capi.rs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Maven project skeleton + wrapper

**Files:**
- Create: `services/analytics/bindings/java/pom.xml`
- Create (generated): `services/analytics/bindings/java/mvnw`, `mvnw.cmd`, `.mvn/wrapper/maven-wrapper.properties`

- [ ] **Step 1: Write `pom.xml`**

Create `services/analytics/bindings/java/pom.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <!-- Java binding over the analytics C ABI (services/analytics/src/capi.rs).
       Build the native library first, from services/analytics:
       cargo build -\-release -\-no-default-features -\-target-dir target/capi -->
  <groupId>io.github.benssutton</groupId>
  <artifactId>analytics</artifactId>
  <version>0.1.0-SNAPSHOT</version>
  <packaging>jar</packaging>

  <properties>
    <maven.compiler.release>25</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <arrow.version>18.1.0</arrow.version>
    <junit.version>6.0.3</junit.version>
    <analytics.library.dir>${project.basedir}/../../target/capi/release</analytics.library.dir>
  </properties>

  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>org.apache.arrow</groupId>
        <artifactId>arrow-bom</artifactId>
        <version>${arrow.version}</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
      <dependency>
        <groupId>org.junit</groupId>
        <artifactId>junit-bom</artifactId>
        <version>${junit.version}</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>

  <dependencies>
    <dependency>
      <groupId>org.apache.arrow</groupId>
      <artifactId>arrow-vector</artifactId>
    </dependency>
    <dependency>
      <groupId>org.apache.arrow</groupId>
      <artifactId>arrow-c-data</artifactId>
    </dependency>
    <dependency>
      <groupId>org.apache.arrow</groupId>
      <artifactId>arrow-memory-netty</artifactId>
      <scope>runtime</scope>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-compiler-plugin</artifactId>
        <version>3.15.0</version>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.5.6</version>
        <configuration>
          <!-- FFM downcalls and Arrow's C Data JNI need native access; Arrow's memory
               module reflects into java.nio. -->
          <argLine>--enable-native-access=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow</argLine>
          <systemPropertyVariables>
            <analytics.library.dir>${analytics.library.dir}</analytics.library.dir>
          </systemPropertyVariables>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```

(In the XML comment, `-\-` stands for `--`, because `--` is illegal inside an XML comment. Keep it exactly as written.)

- [ ] **Step 2: Generate the Maven wrapper**

From `services/analytics/bindings/java/`:
```bash
MVN=/c/Users/Alexander/.m2/wrapper/dists/apache-maven-3.9.15-bin/4faaaa08/apache-maven-3.9.15/bin/mvn
"$MVN" -q -N wrapper:wrapper -Dmaven=3.9.15
ls mvnw mvnw.cmd .mvn/wrapper/maven-wrapper.properties
grep distributionUrl .mvn/wrapper/maven-wrapper.properties
```
Expected: all three files exist, and `distributionUrl` ends with `apache-maven-3.9.15-bin.zip`.

- [ ] **Step 3: Verify the build resolves dependencies**

```bash
./mvnw -B -q dependency:resolve; echo exit=$?
```
Expected: `exit=0`. Then confirm the C Data jar is there and bundles the Windows JNI DLL:
```bash
unzip -l ~/.m2/repository/org/apache/arrow/arrow-c-data/18.1.0/arrow-c-data-18.1.0.jar | grep arrow_cdata_jni.dll
```
Expected: one line, `arrow_cdata_jni/x86_64/arrow_cdata_jni.dll`.

- [ ] **Step 4: Commit (the wrapper script must be executable for Linux CI)**

From the repo root:
```bash
git add services/analytics/bindings/java/pom.xml services/analytics/bindings/java/mvnw services/analytics/bindings/java/mvnw.cmd services/analytics/bindings/java/.mvn
git update-index --chmod=+x services/analytics/bindings/java/mvnw
git commit -m "Add Maven project for the Java binding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git ls-files -s services/analytics/bindings/java/mvnw
```
Expected: the last command shows mode `100755`.

---

### Task 5: `Params`, `Analytics` and the JUnit tests

**Files:**
- Create: `services/analytics/bindings/java/src/test/java/io/github/benssutton/analytics/AnalyticsTest.java`
- Create: `services/analytics/bindings/java/src/main/java/io/github/benssutton/analytics/Params.java`
- Create: `services/analytics/bindings/java/src/main/java/io/github/benssutton/analytics/Analytics.java`

- [ ] **Step 1: Write the failing tests**

Create `AnalyticsTest.java`:

```java
package io.github.benssutton.analytics;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.junit.jupiter.api.Test;

class AnalyticsTest {

    /** `a: Int64 [0, 5, 7]` and `s: Utf8 ["x", "y", "x"]`, under the given names. */
    private static VectorSchemaRoot toyData(BufferAllocator allocator, String intName, String stringName) {
        BigIntVector a = new BigIntVector(intName, allocator);
        a.allocateNew(3);
        a.set(0, 0);
        a.set(1, 5);
        a.set(2, 7);
        a.setValueCount(3);
        VarCharVector s = new VarCharVector(stringName, allocator);
        s.allocateNew(3);
        s.set(0, "x".getBytes(UTF_8));
        s.set(1, "y".getBytes(UTF_8));
        s.set(2, "x".getBytes(UTF_8));
        s.setValueCount(3);
        VectorSchemaRoot root = VectorSchemaRoot.of(a, s);
        root.setRowCount(3);
        return root;
    }

    /** An in-memory ArrowReader over `root`'s one batch, via an IPC stream round trip. */
    private static ArrowReader reader(VectorSchemaRoot root, BufferAllocator allocator) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
            writer.start();
            writer.writeBatch();
            writer.end();
        }
        return new ArrowStreamReader(new ByteArrayInputStream(bytes.toByteArray()), allocator);
    }

    private static List<String> strings(VectorSchemaRoot root, String column) {
        FieldVector vector = root.getVector(column);
        return IntStream.range(0, root.getRowCount()).mapToObj(i -> String.valueOf(vector.getObject(i))).toList();
    }

    @Test
    void describesAndRecommendsToyData() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             VectorSchemaRoot data = toyData(allocator, "a", "s");
             ArrowReader input = reader(data, allocator);
             ArrowReader output = Analytics.describeAndRecommend(input, Params.defaults(), allocator)) {
            assertTrue(output.loadNextBatch());
            VectorSchemaRoot result = output.getVectorSchemaRoot();
            assertEquals(2, result.getRowCount());
            assertEquals(List.of("a", "s"), strings(result, "column"));
            assertEquals(List.of("uint8", "string"), strings(result, "rec_arrow_type"));
            assertFalse(output.loadNextBatch());
        }
    }

    @Test
    void duplicateColumnIsIllegalArgument() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             VectorSchemaRoot data = toyData(allocator, "a", "a");
             ArrowReader input = reader(data, allocator)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Analytics.describeAndRecommend(input, Params.defaults(), allocator));
            assertTrue(e.getMessage().contains("duplicate column \"a\""), e.getMessage());
        }
    }
}
```

Closing the `RootAllocator` last, in try-with-resources order, makes the test fail on any leaked Arrow buffer.

- [ ] **Step 2: Run the tests to verify they fail**

From `services/analytics/bindings/java/`:
```bash
./mvnw -B -q test 2>&1 | grep -m3 "ERROR"
```
Expected: compilation errors, `cannot find symbol ... class Analytics` / `Params`.

- [ ] **Step 3: Write `Params.java`**

```java
package io.github.benssutton.analytics;

import java.util.List;
import java.util.OptionalLong;

/**
 * Keyword parameters of {@code describe_and_recommend}; {@link #defaults()} matches the
 * Python technique ({@code analytics.recommend}).
 */
public record Params(long seed, int zstdLevel, OptionalLong populationRows,
                     long categoricalThreshold, List<BooleanPair> booleanPairs) {

    /** Two strings that together mark a string column as boolean, e.g. ("true", "false"). */
    public record BooleanPair(String trueValue, String falseValue) {}

    public Params {
        booleanPairs = List.copyOf(booleanPairs);
    }

    /** seed 0, ZSTD level 1, no population, categorical threshold 10 000, ("true", "false"). */
    public static Params defaults() {
        return new Params(0, 1, OptionalLong.empty(), 10_000, List.of(new BooleanPair("true", "false")));
    }
}
```

- [ ] **Step 4: Write `Analytics.java`**

```java
package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * Java binding over the analytics C ABI (services/analytics/src/capi.rs). Tables cross as
 * Arrow C Streams, without copying. The native library is found in the directory named by
 * the system property {@code analytics.library.dir}.
 */
public final class Analytics {

    private static final int INVALID_INPUT = 1;

    private static final MethodHandle DESCRIBE_AND_RECOMMEND;
    private static final MethodHandle FREE_ERROR;

    static {
        SymbolLookup library = SymbolLookup.libraryLookup(libraryPath(), Arena.global());
        Linker linker = Linker.nativeLinker();
        DESCRIBE_AND_RECOMMEND = linker.downcallHandle(
            library.findOrThrow("analytics_describe_and_recommend"),
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS,    // ArrowArrayStream *input (consumed)
                JAVA_LONG,  // uint64_t seed
                JAVA_INT,   // int32_t zstd_level
                JAVA_LONG,  // int64_t population_rows (< 0 = none)
                JAVA_LONG,  // uint64_t categorical_threshold
                ADDRESS,    // const char *const *bool_true
                ADDRESS,    // const char *const *bool_false
                JAVA_LONG,  // size_t n_bool_pairs
                ADDRESS,    // ArrowArrayStream *output
                ADDRESS));  // char **error
        FREE_ERROR = linker.downcallHandle(
            library.findOrThrow("analytics_free_error"), FunctionDescriptor.ofVoid(ADDRESS));
    }

    private Analytics() {}

    private static Path libraryPath() {
        String dir = System.getProperty("analytics.library.dir");
        if (dir == null) {
            throw new IllegalStateException("system property analytics.library.dir is not set");
        }
        Path path = Path.of(dir).resolve(System.mapLibraryName("analytics")).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException(path + " not found: in services/analytics run "
                + "`cargo build --release --no-default-features --target-dir target/capi`");
        }
        return path;
    }

    /**
     * Describe's table, the size columns and the {@code rec_*} columns, one row per column of
     * {@code input}. {@code input} is consumed. The returned reader holds one batch; the
     * caller closes it.
     *
     * @throws IllegalArgumentException for invalid input (e.g. duplicate column names)
     * @throws RuntimeException for a failure inside the Rust kernels
     */
    public static ArrowReader describeAndRecommend(ArrowReader input, Params params, BufferAllocator allocator) {
        try (ArrowArrayStream in = ArrowArrayStream.allocateNew(allocator);
             ArrowArrayStream out = ArrowArrayStream.allocateNew(allocator);
             Arena arena = Arena.ofConfined()) {
            Data.exportArrayStream(allocator, input, in);
            List<Params.BooleanPair> pairs = params.booleanPairs();
            MemorySegment trues = cStrings(arena, pairs.stream().map(Params.BooleanPair::trueValue).toList());
            MemorySegment falses = cStrings(arena, pairs.stream().map(Params.BooleanPair::falseValue).toList());
            MemorySegment error = arena.allocate(ADDRESS);  // zero-initialised: null
            int code = (int) DESCRIBE_AND_RECOMMEND.invokeExact(
                MemorySegment.ofAddress(in.memoryAddress()),
                params.seed(),
                params.zstdLevel(),
                params.populationRows().orElse(-1),
                params.categoricalThreshold(),
                trues,
                falses,
                (long) pairs.size(),
                MemorySegment.ofAddress(out.memoryAddress()),
                error);
            if (code != 0) {
                throw failure(code, error.get(ADDRESS, 0));
            }
            return Data.importArrayStream(allocator, out);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** The exception for a non-zero return code; frees the native message. */
    private static RuntimeException failure(int code, MemorySegment message) throws Throwable {
        String text = message.address() == 0
            ? "analytics error code " + code
            : message.reinterpret(Long.MAX_VALUE).getString(0);
        FREE_ERROR.invokeExact(message);
        return code == INVALID_INPUT ? new IllegalArgumentException(text) : new RuntimeException(text);
    }

    /** A native array of NUL-terminated UTF-8 strings, or NULL when empty. */
    private static MemorySegment cStrings(Arena arena, List<String> strings) {
        if (strings.isEmpty()) {
            return MemorySegment.NULL;
        }
        MemorySegment array = arena.allocate(ADDRESS, strings.size());
        for (int i = 0; i < strings.size(); i++) {
            array.setAtIndex(ADDRESS, i, arena.allocateFrom(strings.get(i)));
        }
        return array;
    }
}
```

- [ ] **Step 5: Make sure the native library is built, then run the tests**

From `services/analytics/`:
```bash
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -1
cd bindings/java && ./mvnw -B test 2>&1 | grep -E "Tests run|ERROR|FAIL|BUILD" | head -20
```
Expected: `Tests run: 2, Failures: 0, Errors: 0, Skipped: 0` and `BUILD SUCCESS`.

**If `describesAndRecommendsToyData` fails with an import error** (e.g. `Unsupported ... format 'vu'` / `ListView` / `unknown format` while importing the output stream), Arrow Java cannot import one of the output types. Go to Task 6 before continuing. Do **not** weaken the assertions.

**If it fails with `ExceptionInInitializerError`,** read the cause. An `IllegalStateException ... not found` means the Step 5 cargo build did not run or used the wrong target dir.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/bindings/java/src
git commit -m "Add Java Analytics binding and JUnit tests for describe_and_recommend

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6 (only if Task 5 Step 5 fails on output import): make the output importable

Do this task only when Arrow Java rejects an output type. Try the two fixes in order and stop at the first one that works.

- [ ] **Step 1: Bump Arrow Java to 19.0.0**

In `pom.xml`, change `<arrow.version>18.1.0</arrow.version>` to `<arrow.version>19.0.0</arrow.version>`. Re-run:
```bash
./mvnw -B test 2>&1 | grep -E "Tests run|ERROR|BUILD" | head -10
```
If you see `BUILD SUCCESS`, commit (`git add services/analytics/bindings/java/pom.xml && git commit -m "Bump Arrow Java to 19.0.0 for view-type import" ...` with the Co-Authored-By trailer), update the spec's version mention, and skip Step 2.

- [ ] **Step 2: Otherwise, export classic layouts from the C ABI only**

Revert the pom to 18.1.0. In `services/analytics/src/capi.rs`, add these imports:
```rust
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
```
and this function above `describe_and_recommend`:
```rust
/// `dt` with view types replaced by their classic layouts (Utf8View → Utf8,
/// BinaryView → Binary), recursively. Arrow Java cannot import view types.
fn classic(dt: &DataType) -> DataType {
    let field = |f: &Field| f.clone().with_data_type(classic(f.data_type()));
    match dt {
        DataType::Utf8View => DataType::Utf8,
        DataType::BinaryView => DataType::Binary,
        DataType::List(f) => DataType::List(Arc::new(field(f))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(field(f))),
        DataType::Struct(fs) => DataType::Struct(fs.iter().map(|f| field(f)).collect()),
        other => other.clone(),
    }
}

fn to_classic(batch: RecordBatch) -> api::Result<RecordBatch> {
    let fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone().with_data_type(classic(f.data_type()))).collect();
    let columns = batch
        .columns()
        .iter()
        .zip(&fields)
        .map(|(c, f)| arrow_cast::cast(c, f.data_type()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Error::Compute(e.to_string()))?;
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| Error::Compute(e.to_string()))
}
```
In `describe_and_recommend` (capi.rs), change:
```rust
    let out = api::describe_and_recommend(&batch, seed, zstd_level, population_rows, categorical_threshold, trues.into_iter().zip(falses).collect())?;
```
to:
```rust
    let out = to_classic(api::describe_and_recommend(&batch, seed, zstd_level, population_rows, categorical_threshold, trues.into_iter().zip(falses).collect())?)?;
```
In the capi test `toy_data_round_trips_through_the_c_abi`, change `.as_string_view()` to `.as_string::<i32>()`.

Re-run:
```bash
cargo test --lib --no-default-features capi:: 2>&1 | tail -3
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -1
cd bindings/java && ./mvnw -B test 2>&1 | grep -E "Tests run|ERROR|BUILD" | head -10
```
Expected: the capi tests pass, and Maven shows `Tests run: 2, Failures: 0` and `BUILD SUCCESS`. Commit `capi.rs` with the message `"Export classic Arrow layouts from the C ABI for Arrow Java"` plus the trailer.

---

### Task 7: CI — `java-tests` job, Python-free clippy, CodeQL Java

**Files:**
- Modify: `.github/workflows/ci-cd.yml`

- [ ] **Step 1: Add the `java-tests` job**

Insert this block directly before the `# CODEQL` banner comment (the `# ===...` line above `# CODEQL`):

```yaml
  # ==========================================================================
  # JAVA BINDING (C ABI + PANAMA FFM) TESTS
  # ==========================================================================
  java-tests:
    name: Java Tests
    runs-on: ubuntu-latest
    needs: rust-coverage

    steps:
      - uses: actions/checkout@v4

      - name: Set up Java
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '25'
          cache: maven

      - name: Install Rust (stable)
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Rust build artifacts
        uses: Swatinem/rust-cache@v2
        with:
          workspaces: services/analytics

      # Own target dir: maturin writes the Python-linked library to target/release.
      - name: Build the C ABI library (no Python)
        working-directory: services/analytics
        run: cargo build --release --no-default-features --target-dir target/capi

      - name: Run Java tests
        working-directory: services/analytics/bindings/java
        run: ./mvnw -B test

      - name: Upload Surefire reports
        uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: surefire-reports
          path: services/analytics/bindings/java/target/surefire-reports

```

- [ ] **Step 2: Add Java to CodeQL**

Change:
```yaml
          languages: python, rust
```
to:
```yaml
          languages: python, rust, java-kotlin
```

- [ ] **Step 3: Lint the Python-free build**

After the existing `Run Rust clippy` step in the `lint` job, add:
```yaml

      - name: Run Rust clippy (no Python, the C ABI build)
        working-directory: services/analytics
        run: cargo clippy --no-default-features --all-targets -- -D warnings
```

- [ ] **Step 4: Gate the pipeline status on it**

In the `status` job's `needs:` list, add `- java-tests` after `- python-tests`.

- [ ] **Step 5: Validate the YAML**

```bash
/c/Users/Alexander/miniconda3/envs/p312/python.exe -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/ci-cd.yml')); j=d['jobs']; print(sorted(j)); print(j['status']['needs']); print(j['codeql']['steps'][1]['with']['languages'])"
```
Expected output:
```
['codeql', 'java-tests', 'lint', 'python-tests', 'rust-coverage', 'semgrep', 'status']
['rust-coverage', 'python-tests', 'java-tests', 'codeql', 'semgrep', 'lint']
python, rust, java-kotlin
```
(If `yaml` is not installed in the env, run `pip install pyyaml` there first.)

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci-cd.yml
git commit -m "CI: run Java binding tests, lint the Python-free build, scan Java

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

The job itself is verified when the branch is pushed. That is outside this plan unless the user asks for it, but mention it in the final report.

---

### Task 8: Documentation

**Files:**
- Modify: `CLAUDE.md` (the "Project Structure" tree and the "Rust Extension (analytics)" section)
- Modify (local only, git-ignored): `docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md`, `docs/superpowers/specs/2026-09-28-java-binding-design.md`

- [ ] **Step 1: Update the CLAUDE.md "Project Structure" tree**

In the tree, change the `src/` comment lines:
```
│   ├── src/                                # Rust extension — lib.rs, shared.rs, entropy.rs, chi_squared.rs,
│   │                                       #   contingency.rs, ari.rs, gcd.rs, bloomfilter.rs, minhash.rs,
│   │                                       #   describe.rs, sizes.rs, cardinality_estimators.rs, recommend.rs,
│   │                                       #   api.rs, arrow_io.rs, python.rs
```
to:
```
│   ├── src/                                # Rust extension — lib.rs, shared.rs, entropy.rs, chi_squared.rs,
│   │                                       #   contingency.rs, ari.rs, gcd.rs, bloomfilter.rs, minhash.rs,
│   │                                       #   describe.rs, sizes.rs, cardinality_estimators.rs, recommend.rs,
│   │                                       #   api.rs, arrow_io.rs, python.rs, capi.rs
│   ├── bindings/java/                      # Maven project: Panama FFM binding over capi.rs + JUnit tests
```

- [ ] **Step 2: Update the CLAUDE.md "Rust Extension (analytics)" section**

Replace:
```
Build: `maturin develop --release` from `services/analytics/`. Python changes need no rebuild (editable install).

Three layers (spec: docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md):
```
with:
```
Build: `maturin develop --release` from `services/analytics/`. Python changes need no rebuild (editable install).
Cargo feature `python` (default) gates pyo3 + python.rs; the Java build is Python-free:
`cargo build --release --no-default-features --target-dir target/capi` (own target dir — maturin writes the
Python-linked library to `target/release`), then `./mvnw test` in `services/analytics/bindings/java/` (JDK 25).

Four layers (specs: docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md, 2026-09-28-java-binding-design.md):
```

After the `src/python.rs` bullet (the one ending `non-Arrow input → TypeError.`), add:
```
- `src/capi.rs` — C ABI (`extern "C"`, no pyo3): `analytics_describe_and_recommend` (Arrow C Stream in, one-batch
  stream out, plain C parameters; `population_rows < 0` = None) and `analytics_free_error`. Returns 0 / 1 InvalidInput /
  2 Compute with a message in `*error`. Wrapped by `io.github.benssutton.analytics.Analytics` (Java 25 FFM + Arrow Java
  `arrow-c-data`), which maps 1 → IllegalArgumentException, 2 → RuntimeException. Only describe_and_recommend is bound so far.
```

- [ ] **Step 3: Update the FFI spec's out-of-scope note (local file)**

In `docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md`, change:
```
out) and a thin Python binding over it. A Java / C-ABI binding is **out of scope**; it
will wrap the same core later with no kernel changes.
```
to:
```
out) and a thin Python binding over it. A Java / C-ABI binding is **out of scope**; it
will wrap the same core later with no kernel changes (started in
2026-09-28-java-binding-design.md).
```
In `docs/superpowers/specs/2026-09-28-java-binding-design.md`, change `Status: approved, not implemented.` to `Status: implemented.` If Task 6 ran, also note which fix was used, in §4.4.

- [ ] **Step 4: Commit (CLAUDE.md only; the specs are git-ignored)**

```bash
git add CLAUDE.md
git commit -m "Document the C ABI layer and the Java binding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run everything from a clean library build**

From `services/analytics/`:
```bash
export PATH=/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH
PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe cargo test --lib 2>&1 | tail -2
cargo test --lib --no-default-features 2>&1 | tail -2
cargo build --release --no-default-features --target-dir target/capi 2>&1 | tail -1
grep -a -c "python3" target/capi/release/analytics.dll
(cd bindings/java && ./mvnw -B test 2>&1 | grep -E "Tests run:|BUILD")
CONDA_PREFIX=/c/Users/Alexander/miniconda3/envs/p312 maturin develop --release 2>&1 | tail -1
(cd ../.. && python -m pytest -q 2>&1 | tail -2)
```
Expected:
- both cargo test runs end in `test result: ok`;
- `grep` prints `0`;
- Maven shows `Tests run: 2, Failures: 0, Errors: 0` and `BUILD SUCCESS`;
- maturin shows `Installed analytics`;
- pytest reports no failures.

- [ ] **Step 2: Report**

Report which of the spec's "Done when" items are verified locally. Say that the `java-tests` CI job is unverified until the branch is pushed. Also report whether Task 6 was needed.
