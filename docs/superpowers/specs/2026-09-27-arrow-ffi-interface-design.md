# Arrow FFI interface for the Rust extension — design

Status: approved design, not yet implemented.

## 1. Goal

Every Rust entry point takes Arrow and returns Arrow, so the library is not tied to
Polars at its boundary. Today all 15 entry points are Polars *expression plugins*
(`#[polars_expr]`, called with `df.select(register_plugin_function(...))`): only
Polars can invoke them, whatever the data format.

In scope: a language-neutral Rust core (arrow-rs `RecordBatch` in, `RecordBatch`
out) and a thin Python binding over it. A Java / C-ABI binding is **out of scope**; it
will wrap the same core later with no kernel changes.

Not a goal: removing Polars from the kernels. They keep computing on Polars `Series`
internally; Polars is removed only where Arrow makes it redundant.

## 2. Architecture

```
Python classes (analytics/<technique>/rust.py)
        │  pass the frame itself (every accepted frame implements __arrow_c_stream__)
        ▼
src/python.rs    #[pymodule] analytics — one #[pyfunction] per entry point
        ▼
src/api.rs       language-neutral core: pub fn <entry>(&RecordBatch, params) -> Result<RecordBatch, Error>
        ▼
src/arrow_io.rs  RecordBatch → Vec<Series> (zero-copy import); DataFrame → RecordBatch
        ▼
kernels          entropy.rs, gcd.rs, describe.rs, … keep their *_impl functions on Series
                 recommend.rs, sizes.rs read arrow-rs directly
```

Dependencies point strictly downward. `api.rs` signatures use only arrow-rs and std
types: no pyo3 and no Polars. A future Java layer wraps exactly `api.rs`.

**Boundary rule:** tables cross as Arrow; parameters cross as plain typed values
(ints, strings, lists of names, `bytes`).

### Removed

- The `pyo3-polars` dependency, every `#[polars_expr]` wrapper and `*_output_type`
  function, `register_plugin_function`, and the one-struct-column-then-`.unnest(...)`
  output shape.
- `shared::to_arrow_rs`. `recommend.rs` and `sizes.rs` receive arrow-rs arrays
  directly instead of converting a `Series` back to Arrow.
- The unused compiled functions `membership` and `membership_ratio_sample`.
- `lsh_candidates`' `threshold` parameter, which is read and discarded today.

pyo3 stays at 0.25. Plain `#[pyfunction]`s already accept `bytes`; the old
`list[int]` workaround came from pyo3-polars' kwargs serialisation. Removing
pyo3-polars lifts its `pyo3 < 0.27` pin, but upgrading is not part of this work.

## 3. Components

### 3.1 `src/api.rs`: the core

Each function takes a `RecordBatch` plus typed parameters and returns a
`RecordBatch`. Output columns are exactly today's struct fields after `.unnest`.

| Function | Parameters | Output columns |
|---|---|---|
| `column_gcd` | none | `column`, `dtype`, `gcd: decimal128(38,0)` |
| `marginal_entropy` | none | `col_name`, `entropy` |
| `pairwise_joint_entropy` | `pairs: Option<&[(String, String)]>` | `col_a`, `col_b`, `entropy` |
| `threeway_joint_entropy` | `triplets: Option<&[(String, String, String)]>` | `col_a`, `col_b`, `col_c`, `entropy` |
| `pairwise_chi_squared` | `pairs` | `col_a`, `col_b`, `chi2_stat`, `p_value`, `cramers_v`, `low_expected_count`, `n_valid` |
| `pairwise_adjusted_rand` | `pairs` | `col_a`, `col_b`, `ari`, `n_valid` |
| `bloom_filter` | one-column batch, `k`, `m` | `Vec<u8>`, the ⌈m/8⌉-byte bit array (a parameter-like blob, not a table) |
| `membership_ratio` | `bits: &[u8]`, `k`, `m` | `col_name`, `ratio_all`, `ratio_non_null` |
| `minhash` | `df_name`, `num_perm` | `qualified_name`, `minhash: list<uint32>` |
| `lsh_candidates` | batch of `names: string`, `signatures: list<uint32>`; `num_bands`, `rows_per_band` | `col_a`, `col_b` |
| `describe_columns` | `seed` | Describe's value-metric columns (today's `describe` struct fields) |
| `column_sizes` | `zstd_level` | `column`, `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes` |
| `describe_and_recommend` | `Params { seed, zstd_level, population_rows, categorical_threshold, boolean_pairs }` | Describe's table plus the `rec_*` columns (today's `recommend` struct fields) |

Output dtypes and the default `pairs` / `triplets` behaviour (`None` means all
combinations) are unchanged.

**Errors.** `api::Error` has two kinds:

- `InvalidInput`: unknown column in `pairs` / `triplets`, a Bloom bit array of the
  wrong length, UInt128, or an Arrow type no kernel accepts.
- `Compute`: any other kernel failure.

`python.rs` raises them as `ValueError` and `RuntimeError` respectively. No test
depends on Polars' `ComputeError`.

### 3.2 `src/arrow_io.rs`: Arrow ↔ Polars

- **Import:** `RecordBatch → Vec<Series>`, zero-copy through the C Data Interface
  (arrow-rs FFI export, polars-arrow FFI import, `Series::from_arrow`). This is the
  reverse of today's `to_arrow_rs` bridge, and it reuses that bridge's documented
  `unsafe` layout equivalence between the two crates' FFI structs. Field metadata
  passes through, so Polars' `_PL_CATEGORICAL2` and `_PL_ENUM_VALUES2` restore
  Categorical and Enum. A field tagged `analytics.int128 = "true"` (§4) is restored
  to an Int128 `Series`.
- **Export:** a kernel's output columns (`Vec<Series>` / `DataFrame`) become one
  `RecordBatch`, using the newest compat level.

### 3.3 `src/python.rs`: the Python binding

- Accepts any object with `__arrow_c_stream__` (Polars DataFrame, pyarrow Table,
  RecordBatch, RecordBatchReader) through the Arrow PyCapsule interface.
- Reads the stream with **polars-arrow's** importer, which understands Polars'
  private formats, applies the Int128 relabelling (§4), and hands the result to
  arrow-rs over the C Data Interface.
- A multi-batch stream is concatenated into one `RecordBatch`, so kernels see one
  chunk per column, as they do after `rechunk` today.
- Releases the GIL (`py.allow_threads`) around each `api.rs` call. Rayon
  parallelism is unchanged.
- Returns a small `#[pyclass]` that exposes `__arrow_c_stream__` over the result
  batch. `bloom_filter` returns `bytes`.
- Maps `api::Error` to `ValueError` / `RuntimeError`.

### 3.4 Kernels

`entropy.rs`, `chi_squared.rs`, `contingency.rs`, `ari.rs`, `bloomfilter.rs`,
`minhash.rs`, `gcd.rs` and `describe.rs` keep their `*_impl` logic on `Series`
unchanged; only the plugin wrappers are removed. `recommend.rs` and `sizes.rs` take
the batch's arrow-rs arrays directly for their Arrow-native work, and the `Series`
(from `arrow_io`) only where they already use Polars: Describe's profile and
Polars-type naming.

### 3.5 Python

- **`Technique.add`** also accepts any object with `__arrow_c_stream__` or
  `__arrow_c_array__`, converted once with `pl.DataFrame(obj)` (zero-copy).
  `base.py` does not import pyarrow. Any other type still raises `TypeError`, with
  the message listing the accepted kinds. The Python-side logic (eligibility,
  descriptors, non-Rust implementations) stays Polars-based.
- **`_plugin.py`** wrappers keep their Python signatures and return shapes. Each
  becomes `pl.DataFrame(_rs.<fn>(df, ...))`, with LazyFrames collected first as
  today. Two callers change:
  - `similarity/rust.py`: `lsh_candidates` returns a frame, no longer a `pl.Expr`.
  - `membership/rust.py`: passes and receives `bytes` instead of `list[int]`.

## 4. Polars-only dtypes

Verified with py-polars 1.41 and pyarrow 24.

- **Categorical / Enum:** exported as `dictionary<uint32|uint8, string_view>` with
  `_PL_CATEGORICAL2` / `_PL_ENUM_VALUES2` field metadata. Both round-trip exactly
  (§3.2). A plain Arrow dictionary without that metadata imports as Categorical.
- **Int128:** Polars exports the private format `_pli128`, which arrow-rs and
  pyarrow reject. `python.rs` reads it with polars-arrow and, at any nesting depth:
  1. relabels it `decimal128(38, 0)` over the same 16-byte buffers (zero-copy);
     precision is only a label, so values beyond 38 digits are carried bitwise, as
     `to_arrow_rs` does today;
  2. tags the field `analytics.int128 = "true"`.

  `arrow_io` restores tagged fields to Int128, so every kernel behaves exactly as
  today. An untagged `decimal128(38, 0)` from any other caller stays Decimal(38, 0).
  The tag is optional and documented; non-Polars callers never set it.

  Rejected alternative: rewriting the format string inside the C schema struct. The
  producer owns that pointer and frees it in its release callback, so the rewrite is
  unsound.
- **UInt128** (`_plu128`): `InvalidInput` → `ValueError`. The Python classes
  already keep UInt128 away from the plugin (it is ineligible), so this only guards
  the boundary.
- **Outputs** never contain Int128 (`gcd` is `decimal128(38, 0)`), so results are
  native Arrow at the FFI itself.

## 5. Testing

The existing suites are the regression oracle, with unchanged expectations: 193 Rust
unit tests and 745 pytest. Kernel `*_impl` tests keep working because the kernels
are untouched.

New tests:

- **Rust `arrow_io`:** import/export round trip for every dtype in
  `tests/data/large_dataset.arrow`; Categorical and Enum with metadata kept; Int128
  including a value beyond 38 digits (tag restores it); nested Int128; zero-row and
  zero-column batches.
- **Rust `api`:** each error kind: unknown pair column, wrong-length Bloom array,
  UInt128.
- **Python `tests/test_plugin.py`:**
  - for every technique package, the reference and Rust implementations return
    identical `result()` from a Polars DataFrame, a pyarrow Table and a pyarrow
    RecordBatch;
  - `ValueError` / `RuntimeError` mapping;
  - GCD known answer on an Int128 column holding a value beyond 38 digits.

## 6. Done when

- Rust and pytest suites pass, new tests included.
- No `polars_expr`, `pyo3_polars`, `register_plugin_function` or `to_arrow_rs`
  remains.
- `api.rs` signatures mention only arrow-rs and std types.
- Benchmarks on `large_dataset.arrow`: every Rust implementation's median is
  within 10% of the pre-change run. The boundary is zero-copy, so equal or faster
  is expected.
- `CLAUDE.md` is updated:
  - the Rust Plugin section describes the core / binding split;
  - the structure listing gains `api.rs`, `arrow_io.rs` and `python.rs`;
  - two follow-ups are closed: Bloom arrays as `list[int]`, and the dead LSH
    `threshold`.
