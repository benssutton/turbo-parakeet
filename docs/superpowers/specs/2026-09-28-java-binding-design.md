# Java binding for `describe_and_recommend` — design

Status: approved, not implemented.

## 1. Goal

Call `api::describe_and_recommend` from Java, and prove it works with one JUnit test on
toy data. This is the first step of the "Java / C-ABI binding" that the Arrow FFI
spec (2026-09-27-arrow-ffi-interface-design.md) left out of scope. Its core is
`api.rs`, and this design leaves `api.rs` unchanged.

In scope:
- a C ABI for `describe_and_recommend`;
- a thin Java wrapper over that ABI (Panama FFM + Arrow Java C Data);
- JUnit tests;
- a CI job that runs them.

Out of scope:
- binding the other `api.rs` functions (they follow the same pattern later);
- building the Rust library from Maven;
- publishing Java artifacts.

## 2. Architecture

```
AnalyticsTest (JUnit)
  └─► Analytics.describeAndRecommend(ArrowReader, Params, BufferAllocator) -> ArrowReader
        │  Arrow Java c-data: Data.exportArrayStream / Data.importArrayStream (zero copy)
        │  FFM downcall (java.lang.foreign)
        ▼
src/capi.rs   analytics_describe_and_recommend(...)  — C ABI, no pyo3, no Polars
        │  arrow_io::read_stream: stream → one RecordBatch
        ▼
src/api.rs    describe_and_recommend(&batch, ...)    — unchanged
```

**Boundary rule** (from the FFI spec, unchanged): tables cross as Arrow C Streams, and
parameters cross as plain C values.

## 3. Rust

### 3.1 Cargo feature `python`

pyo3 becomes optional, so the library can load without `python3.dll` /
`libpython`:

```toml
[features]
default = ["python"]
python = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.25.0", features = ["extension-module", "abi3-py39"], optional = true }
```

In `lib.rs`, `mod python;` becomes `#[cfg(feature = "python")] mod python;`.
`capi.rs` is always compiled and adds no dependencies.

- Python: maturin uses the default features, so nothing changes for it.
- Java: `cargo build --release --no-default-features --target-dir target/capi` produces
  `target/capi/release/analytics.dll` (`libanalytics.so` on Linux,
  `libanalytics.dylib` on macOS) with no Python import. It uses its own target
  directory because `maturin develop` writes the Python-linked library to
  `target/release`, which would overwrite it.

### 3.2 `src/arrow_io.rs`: `read_stream`

The part of `python.rs::read_batch` that turns a stream into one batch moves to
`arrow_io.rs`:

```rust
/// A whole Arrow C stream as one RecordBatch (kernels expect one chunk per column).
pub unsafe fn read_stream(stream: *mut FFI_ArrowArrayStream) -> Result<RecordBatch, ArrowError>
```

It calls `ArrowArrayStreamReader::from_raw`, collects the batches and runs
`concat_batches`. `python.rs::read_batch` keeps its capsule checks and
`reject_wide_integers`, then calls `read_stream`. The C ABI needs no wide-integer
guard: Arrow Java never produces Polars' `_pli128` / `_plu128`, and arrow-rs rejects
any unknown format as `InvalidInput` anyway.

### 3.3 `src/capi.rs`: the C ABI

```c
/* Returns 0 on success, 1 on invalid input, 2 on a compute failure. */
int analytics_describe_and_recommend(
    struct ArrowArrayStream *input,        /* consumed: released by Rust */
    uint64_t seed,
    int32_t zstd_level,
    int64_t population_rows,               /* < 0 means None */
    uint64_t categorical_threshold,
    const char *const *bool_true,          /* n_bool_pairs UTF-8 strings */
    const char *const *bool_false,         /* n_bool_pairs UTF-8 strings */
    size_t n_bool_pairs,
    struct ArrowArrayStream *output,       /* on success: a one-batch stream owned by the caller */
    char **error);                         /* on failure: message, free with analytics_free_error */

void analytics_free_error(char *error);
```

- Functions are `#[no_mangle] pub unsafe extern "C"`. The output is written with
  `FFI_ArrowArrayStream::new(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))`
  and `ptr::write`.
- Return codes mirror `api::Error`: `InvalidInput` → 1, `Compute` → 2. A failed
  `read_stream`, or a boolean-pair string that is not valid UTF-8, is `InvalidInput`.
- `*error` is a `CString::into_raw`, and `analytics_free_error` reclaims it with
  `CString::from_raw`. A null `input`, `output`, or pair array (when
  `n_bool_pairs > 0`) returns 1, with a message written to `*error` unless `error`
  is itself null. Null is safe to pass to `analytics_free_error`.
- Panics: the release profile uses `panic = "abort"`, so a kernel panic ends the
  JVM process. This matches Python and is left unchanged.

## 4. Java — `services/analytics/bindings/java/`

```
bindings/java/
├── pom.xml
├── mvnw, mvnw.cmd, .mvn/wrapper/maven-wrapper.properties   # Maven 3.9.15
└── src/
    ├── main/java/io/github/benssutton/analytics/
    │   ├── Analytics.java
    │   └── Params.java
    └── test/java/io/github/benssutton/analytics/
        └── AnalyticsTest.java
```

### 4.1 `pom.xml`

- `maven.compiler.release` 25.
- Dependencies:
  - `org.apache.arrow:arrow-vector`, `arrow-c-data` and `arrow-memory-netty` 18.1.0,
    through `arrow-bom`;
  - `org.junit.jupiter:junit-jupiter` 6.0.3 (test scope), through `junit-bom`.
- Surefire `argLine`: `--enable-native-access=ALL-UNNAMED
  --add-opens=java.base/java.nio=ALL-UNNAMED`.
- Surefire system property: `analytics.library.dir =
  ${project.basedir}/../../target/capi/release`.

### 4.2 `Params`

```java
public record Params(long seed, int zstdLevel, OptionalLong populationRows,
                     long categoricalThreshold, List<BooleanPair> booleanPairs) {
    public record BooleanPair(String trueValue, String falseValue) {}
    public static Params defaults();   // 0, 1, empty, 10_000, [("true", "false")]
}
```

The defaults match the Python keywords.

### 4.3 `Analytics`

`Analytics` is a final class with static methods only.

- **Loading (once).** The library is at
  `Path.of(System.getProperty("analytics.library.dir")).resolve(System.mapLibraryName("analytics"))`.
  It is opened with `SymbolLookup.libraryLookup(path, Arena.global())`. The
  downcall handles for both C functions are cached in static fields. If the
  property is missing or the file does not exist, loading throws
  `IllegalStateException` with the path and the build command
  `cargo build --release --no-default-features --target-dir target/capi`.
- **`ArrowReader describeAndRecommend(ArrowReader input, Params params, BufferAllocator allocator)`**
  1. Allocate `ArrowArrayStream in` and `out`. `Data.exportArrayStream(allocator, input, in)`.
  2. In a confined `Arena`, allocate the boolean-pair strings as C strings, two
     pointer arrays and an `error` pointer slot.
  3. Invoke the downcall with the addresses of `in` and `out`. `populationRows`
     empty → `-1`.
  4. If the code is non-zero, read the message, call `analytics_free_error`, and
     throw: `1` → `IllegalArgumentException`, `2` → `RuntimeException`.
  5. Otherwise return `Data.importArrayStream(allocator, out)`. The caller calls
     `loadNextBatch()` once and closes the reader.
  6. Close the `ArrowArrayStream` structs in `finally`. The Rust side has already
     moved their contents out, so closing them only frees the struct memory.

### 4.4 Output types — check first

The output uses Arrow view types (`Utf8View` for `column`, `rec_arrow_type`, …) and
nested columns such as `rec_candidates`. The first implementation task checks
that Arrow Java 18.1.0 imports every output column.

- If it fails, bump Arrow Java to the newest release.
- If a type is still unsupported after that, `capi.rs` alone casts view types to
  `Utf8` / `List` before export.

## 5. Tests

### 5.1 JUnit (`AnalyticsTest`)

Both tests build their input as a `VectorSchemaRoot` wrapped in an `ArrowReader`,
using a `RootAllocator` that is closed at the end of the test. Closing the allocator
detects buffer leaks.

1. **`describesAndRecommendsToyData`**
   - Input: `a: Int64 [0, 5, 7]` and `s: Utf8 ["x", "y", "x"]`, the same data as
     the Rust test in api.rs.
   - Expected: 2 rows, and `column` is `["a", "s"]`.
   - Expected: `a`'s `rec_arrow_type` is `"uint8"`.
   - Expected: `s`'s `rec_arrow_type` is `"string"` (checked against the Python binding).
2. **`duplicateColumnIsIllegalArgument`**
   - Input: two columns named `a`.
   - Expected: `IllegalArgumentException` whose message contains
     `duplicate column "a"`.

### 5.2 Rust unit test (`capi::tests`)

This test calls `analytics_describe_and_recommend` directly with an
`FFI_ArrowArrayStream` built from the same toy batch.

- Success case: it asserts return code 0 and imports `output`. `a`'s
  `rec_arrow_type` must be `"uint8"`.
- Error case: duplicate columns return 1, with a message that
  `analytics_free_error` frees.

`cargo test` therefore covers the ABI without a JVM.

## 6. CI — `.github/workflows/ci-cd.yml`

- **New job `java-tests`** ("Java Tests"), on `ubuntu-latest`, with
  `needs: rust-coverage`, the same as `python-tests`. Steps:
  1. `actions/checkout@v4`.
  2. `actions/setup-java@v4` with `distribution: temurin`, `java-version: '25'`,
     `cache: maven`.
  3. `dtolnay/rust-toolchain@stable` and `Swatinem/rust-cache@v2`, with
     `workspaces: services/analytics`.
  4. `cargo build --release --no-default-features --target-dir target/capi` in
     `services/analytics`.
  5. `./mvnw -B test` in `services/analytics/bindings/java`.
  6. `actions/upload-artifact@v4` of `target/surefire-reports`, with
     `if: failure()`.
- **`status`**: add `java-tests` to `needs`.
- **`lint`**: add `cargo clippy --no-default-features --all-targets -- -D warnings`
  after the existing clippy step. This catches `cfg(feature = "python")` mistakes
  in the Python-free build.
- **`codeql`**: `languages: python, rust, java-kotlin`. `build-mode: none`
  supports Java.
- `rust-coverage` keeps `--all-features`, so the `capi::tests` test runs there.
- `mvnw` must be committed executable (`git update-index --chmod=+x`), because
  the repository is developed on Windows.

## 7. Documentation

- Update the CLAUDE.md "Rust Extension" section to describe a fourth layer, `src/capi.rs`
  (the C ABI).
- Add to CLAUDE.md the `python` feature and the Java build/test commands:
  `cargo build --release --no-default-features --target-dir target/capi`, then
  `mvnw test` in
  `bindings/java`.
- Update the FFI spec's out-of-scope note to point to this spec.

## 8. Done when

- `cargo test --lib capi::` passes.
- `cargo build --release --no-default-features --target-dir target/capi` produces
  a library without a
  Python import.
- `maturin develop --release` and pytest are unchanged and pass.
- `mvnw test` passes locally on Windows.
- The `java-tests` CI job passes on Linux.
