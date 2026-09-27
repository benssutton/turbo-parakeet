# Arrow FFI Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Rust entry point takes Arrow and returns Arrow. A language-neutral core (`api.rs`, arrow-rs `RecordBatch` in and out) sits under a thin pyo3 binding that replaces the Polars expression plugins. Int128 and UInt128 are rejected everywhere.

**Architecture:** `python.rs` (pyo3 module `analytics.analytics`) reads any `__arrow_c_stream__` object into one `RecordBatch`, rejects Polars' private 128-bit formats, and calls `api.rs`. `api.rs` converts the batch to Polars `Series` through `arrow_io.rs` (zero-copy, C Data Interface), runs the existing `*_impl` kernels unchanged, and exports their one-struct-column result as a flat `RecordBatch`. `pyo3-polars` and every `#[polars_expr]` are removed. The Python wrappers in `analytics/_plugin.py` pass Polars frames straight to the binding and wrap results with `pl.DataFrame(...)`. `Technique.add` also accepts Arrow tabular objects.

**Tech Stack:** Rust (polars 0.51, polars-arrow 0.51, pyo3 0.25, arrow-rs 60: arrow-array with `ffi`, arrow-data, arrow-schema, arrow-select), Python 3.12 (polars 1.41, pyarrow 24), pytest, maturin.

**Spec:** `docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md`.

---

## Before you start

- **Branch.** Work on a feature branch: `git checkout -b arrow-ffi` from `main`.
- **Clean tree.** Run `git status`. Unrelated local changes exist and must never be committed: `.claude/*`, deleted old docs under `docs/superpowers/`, `notebooks/Examples.ipynb`, `results.csv`. If anything under `services/analytics/` or `tests/` is modified, stop and ask the user. Commit only the files each task names.
- **Commands** (Git Bash, repo root `c:/Users/Alexander/turbo-parakeet`):

```bash
PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe
# Rust unit tests (from services/analytics):
cd services/analytics && PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib <filter>; cd -
# Build the extension (needed before any pytest run after a Rust change; several minutes — fat LTO):
cd services/analytics && unset VIRTUAL_ENV && CONDA_PREFIX=/c/Users/Alexander/miniconda3/envs/p312 /c/Users/Alexander/miniconda3/envs/p312/Scripts/maturin.exe develop --release; cd -
# Python tests (from the repo root):
$PY -m pytest tests -q -m "not slow"
```

Python-only changes need no rebuild (editable install).

- **Baseline timings (do this first, before any change).** Save this script as `tests/performance/results/boundary_timing.py`. The `results/` directory is git-ignored, so the script is a local tool and is never committed:

```python
"""Median wall time (ms) of every *Rust technique class on large_dataset.arrow.

Usage: python boundary_timing.py save <file.json> | compare <file.json>
1 warm-up + 5 timed runs per class, fresh instance per run. `compare` exits 1 if any
class is more than 10% slower than the saved run."""

import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[2] / "data" / "large_dataset.arrow"
PACKAGES = ("gcd", "describe", "recommend", "membership", "similarity", "chi_squared",
            "pairwise_entropy", "threeway_entropy", "adjusted_rand")


def timings() -> dict[str, float]:
    df = pl.read_ipc(DATA)
    out = {}
    for pkg in PACKAGES:
        module = importlib.import_module(f"analytics.{pkg}")
        frame = df.select(df.columns[:40]) if pkg == "threeway_entropy" else df
        for name in (n for n in module.IMPLEMENTATIONS if n.endswith("Rust")):
            cls = getattr(module, name)
            runs = []
            for i in range(6):
                start = time.perf_counter()
                cls().add({"large": frame}).result()
                if i:
                    runs.append((time.perf_counter() - start) * 1000)
            out[name] = statistics.median(runs)
            print(f"{name:24} {out[name]:10.1f} ms", flush=True)
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    now = timings()
    if mode == "save":
        path.write_text(json.dumps(now, indent=1))
    else:
        before = json.loads(path.read_text())
        for k in before:
            print(f"{k:24} {before[k]:10.1f} -> {now[k]:10.1f} ms  ({now[k] / before[k]:.2f}x)")
        sys.exit(0 if max(now[k] / before[k] for k in before) <= 1.10 else 1)
```

Run: `$PY tests/performance/results/boundary_timing.py save tests/performance/results/boundary_baseline.json`
Expected: nine lines of timings, one per `*Rust` class, and the JSON file written.

## File structure

| File | Responsibility |
|---|---|
| `services/analytics/src/arrow_io.rs` (new) | `import_batch` (RecordBatch → Vec<Series>), `export_series` (Series → arrow-rs array; moved from `shared::to_arrow_rs`), `export_struct` (one-struct-column result → flat RecordBatch) |
| `services/analytics/src/api.rs` (new) | The language-neutral core: `Error`, `Result`, one `pub fn` per entry point |
| `services/analytics/src/python.rs` (new) | pyo3 module `analytics`: stream reading, 128-bit rejection, `ArrowTable`, 13 `#[pyfunction]`s |
| `services/analytics/src/lib.rs` | registers the three new modules |
| kernels (`gcd.rs`, `entropy.rs`, `chi_squared.rs`, `ari.rs`, `bloomfilter.rs`, `minhash.rs`, `describe.rs`, `sizes.rs`, `recommend.rs`, `shared.rs`) | `#[polars_expr]` wrappers, output-type functions and serde kwargs removed; dead Int128 branches removed; `*_impl` logic unchanged |
| `services/analytics/Cargo.toml` | drop `pyo3-polars` and `serde` |
| `services/analytics/analytics/_dtypes.py` | `WIDE_INTEGERS`, `holds_wide_integer`; `encodable` rejects them |
| `services/analytics/analytics/{gcd,chi_squared,describe}/base.py` | Int128 out of the eligibility sets |
| `services/analytics/analytics/{gcd/math,gcd/numpy,describe/datafusion,describe/_sizes}.py` | dead Int128 / 38-digit handling removed |
| `services/analytics/analytics/_plugin.py` | rewritten over the binding; flat frames |
| `services/analytics/analytics/*/rust.py` | drop `.unnest(...)`; Bloom `bytes`; LSH as a function |
| `services/analytics/analytics/base.py` | `Technique.add` accepts Arrow tabular objects |
| `tests/test_plugin.py` (new) | 128-bit ineligibility, binding errors, Arrow-input parity |
| `tests/test_{gcd,describe,recommend,base}.py` | Int128 cases updated |
| `CLAUDE.md`, the spec | documentation |

---

### Task 1: 128-bit integers are ineligible in every technique (Python)

**Files:**
- Create: `tests/test_plugin.py`
- Modify: `services/analytics/analytics/_dtypes.py`, `services/analytics/analytics/gcd/base.py`, `services/analytics/analytics/gcd/math.py`, `services/analytics/analytics/gcd/numpy.py`, `services/analytics/analytics/chi_squared/base.py`, `services/analytics/analytics/describe/base.py`, `services/analytics/analytics/describe/datafusion.py`, `services/analytics/analytics/describe/_sizes.py`
- Test: `tests/test_gcd.py`, `tests/test_describe.py`, `tests/test_recommend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plugin.py`:

```python
"""The Arrow boundary: 128-bit integers, the Python binding, and Arrow inputs."""

import importlib

import polars as pl
import pytest

from harness import load, run

PACKAGES = (
    "analytics.gcd", "analytics.describe", "analytics.recommend", "analytics.membership",
    "analytics.similarity", "analytics.chi_squared", "analytics.pairwise_entropy",
    "analytics.threeway_entropy", "analytics.adjusted_rand",
)
EVERY_IMPLEMENTATION = [
    pytest.param(f"{p}:{n}", id=n) for p in PACKAGES for n in importlib.import_module(p).IMPLEMENTATIONS
]


def wide_integer_frame() -> tuple[pl.DataFrame, list[str]]:
    cols = {
        "ok_a": [1, 2, 3],
        "ok_b": [3, 2, 1],
        "i128": pl.Series([1, 2, 3], dtype=pl.Int128),
        "list_i128": pl.Series([[1], [2], [3]], dtype=pl.List(pl.Int128)),
    }
    if hasattr(pl, "UInt128"):
        cols["u128"] = pl.Series([1, 2, 3], dtype=pl.UInt128)
    return pl.DataFrame(cols), [c for c in cols if not c.startswith("ok_")]


@pytest.mark.parametrize("spec", EVERY_IMPLEMENTATION)
def test_128_bit_integer_columns_are_ineligible(spec):
    frame, wide = wide_integer_frame()
    out = run(load(spec), {"t": frame})
    names = [c for c in out.columns if c.startswith("col_")]
    touches_wide = out.filter(pl.any_horizontal(pl.col(c).is_in(wide) for c in names))
    assert touches_wide.height > 0
    assert touches_wide["status"].unique().to_list() == ["ineligible"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_plugin.py -q`
Expected: FAIL for GCD, chi², entropy, ARI, Membership, Similarity, Describe and Recommend implementations: rows touching `i128` or `list_i128` have status `computed`.

- [ ] **Step 3: Add `WIDE_INTEGERS` and `holds_wide_integer`; make `encodable` reject them**

In `services/analytics/analytics/_dtypes.py`, replace the `ENCODABLE` block and `encodable` with:

```python
# Arrow has no 128-bit integer type. Polars exports Int128 / UInt128 in its private
# formats `_pli128` / `_plu128`, which the Rust extension's Arrow boundary (and every
# non-Polars Arrow consumer) rejects, so columns holding them — at any nesting depth —
# are ineligible in every technique. Cast to Decimal(38, 0) or Int64 to analyse them.
WIDE_INTEGERS = tuple(t for t in (pl.Int128, getattr(pl, "UInt128", None)) if t is not None)


def holds_wide_integer(dtype: pl.DataType) -> bool:
    if isinstance(dtype, WIDE_INTEGERS):
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return holds_wide_integer(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(holds_wide_integer(f.dtype) for f in dtype.fields)
    return False


# Dtypes the Rust encoder (src/shared.rs::encode_series) accepts. Anything else
# (Struct, Binary, Null, Object, 128-bit integers) makes the extension raise.
ENCODABLE = (
    *INTEGERS_64, pl.Boolean, pl.Float32, pl.Float64,
    pl.Date, pl.Datetime, pl.Duration, pl.Time, *STRING_LIKE, pl.Decimal, *NESTED,
)


def encodable(dtype: pl.DataType) -> bool:
    return isinstance(dtype, ENCODABLE) and not holds_wide_integer(dtype)
```

- [ ] **Step 4: GCD: Int128 out of `INTEGER_BACKED`; drop the 38-digit limit**

The 38-digit limit was reachable only from Int128: a Decimal's unscaled values always fit in 38 digits.

In `services/analytics/analytics/gcd/base.py`, replace everything from the line `GCD_LIMIT = 10**38 ...` down to the end of the class docstring with:

```python
INTEGER_BACKED = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Decimal, pl.Date, pl.Datetime, pl.Duration, pl.Time,
)


class Gcd(Technique):
    """GCD of the magnitudes of each column's raw physical integer values.

    Results are in physical units: Decimal → unscaled integer, Date → days,
    Datetime/Duration → their time unit, Time → ns. Nulls are skipped; all-null,
    all-zero and zero-row columns → 0. `gcd` is Decimal(38, 0): Arrow has no plain
    128-bit integer, and a Decimal's unscaled values fit in 38 digits. Other dtypes
    — including Categorical/Enum and 128-bit integers (analytics._dtypes.WIDE_INTEGERS)
    — are ineligible. `dtype` (Python's str(dtype)) is reported on every row.
    """
```

In `services/analytics/analytics/gcd/math.py`:
- change the import to `from analytics.gcd.base import Gcd`;
- change `math_gcd`'s body to `return math.gcd(*series.to_physical().drop_nulls().to_list())`;
- change its return annotation to `-> int`;
- in the class docstring, replace `so Int128, wide Decimal and MIN magnitudes are exact` with `so wide Decimal and MIN magnitudes are exact`.

In `services/analytics/analytics/gcd/numpy.py`:
- change the import to `from analytics.gcd.base import Gcd`;
- replace the last two lines of `numpy_gcd` with:

```python
    # reduce() of one element returns it unchanged (possibly negative): take abs.
    return abs(int(np.gcd.reduce(values)))
```

- change its return annotation to `-> int`.

- [ ] **Step 5: Chi-squared and Describe eligibility**

In `services/analytics/analytics/chi_squared/base.py`: `CATEGORICAL = (pl.Boolean, *STRING_LIKE, *INTEGERS_64)`.

In `services/analytics/analytics/describe/base.py`:
- add `from analytics._dtypes import holds_wide_integer` to the imports;
- delete the line `_UINT128 = getattr(pl, "UInt128", None)`;
- make `_unsupported`'s first test:

```python
    if isinstance(dtype, (pl.Object, pl.Null)) or holds_wide_integer(dtype):
        return True
```

- [ ] **Step 6: Remove dead Int128 handling in the Python Describe implementations**

In `services/analytics/analytics/describe/_sizes.py`:
- in the module docstring, delete the sentences from `pyarrow cannot import Polars Int128` to `(in every implementation).`, so the paragraph ends at `writes.`;
- delete `_nests_int128` and `_to_arrow`;
- replace `column_sizes` with:

```python
def column_sizes(s: pl.Series, zstd_level: int) -> dict[str, int]:
    s = s.rechunk()
    classic = s.to_arrow(compat_level=pl.CompatLevel.oldest())
    native = s.to_arrow(compat_level=pl.CompatLevel.newest())
    return {
        "size_bytes": ipc_body_bytes(classic, None),
        "size_zstd_bytes": ipc_body_bytes(classic, zstd_level),
        "size_polars_bytes": ipc_body_bytes(native, None),
        "size_polars_zstd_bytes": ipc_body_bytes(native, zstd_level),
    }
```

In `services/analytics/analytics/describe/datafusion.py`:
- In the class docstring, replace:

```
      - every group A metric of List/Array/Struct columns whose values hold floats,
        Enum, Categorical or Int128: the Polars reference helpers (SQL cannot key
        nested -0.0/NaN, list-of-dictionary children, Enum order or Int128 exactly);
```

with:

```
      - every group A metric of List/Array/Struct columns whose values hold floats,
        Enum or Categorical: the Polars reference helpers (SQL cannot key nested
        -0.0/NaN, list-of-dictionary children or Enum order exactly);
```

  and delete the line `    Int128 is registered as Decimal(38, 0); values beyond 38 digits are not supported.`
- `_INEXACT_IN_SQL = (pl.Float32, pl.Float64, pl.Enum, pl.Categorical)`.
- Make `_needs_polars`'s docstring:

```python
    """True for List/Array/Struct values that hold floats, Enum or Categorical:
    DataFusion cannot key them exactly (-0.0 and NaN payloads inside nested values,
    dictionary children in lists, Enum category order)."""
```

- Replace `_arrow` with:

```python
def _arrow(s: pl.Series) -> pa.Array:
    return s.rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())
```

- [ ] **Step 7: Update the Int128 cases in the existing tests**

`tests/test_gcd.py`:
- In `CASES`, delete `pytest.param(pl.Int128(), 10**20, -(10**15), 10**15, id="Int128"),`.
- In `test_dtype_extremes`' parameters, replace the six `pl.Int128()` rows (`decimal38_max` … `i128_min_with_null_zero`) with:

```python
        pytest.param(pl.Decimal(38, 0), [10**38 - 1], 10**38 - 1, id="decimal38_max"),
```

- In `_FUZZ_INT_RANGES`, delete the `pl.Int128: ...` line.
- In `test_conclusions`, change `"big": pl.Series([0], dtype=pl.Int128)` to `"big": pl.Series([0], dtype=pl.Int64)`.
- In `test_ineligible_dtypes_are_reported_with_their_dtype`, add `"i128": pl.Series([12, 18], dtype=pl.Int128),` to `cols`, and change the comment on the UInt128 line to `# 128-bit integers are ineligible (analytics._dtypes.WIDE_INTEGERS)`.

`tests/test_describe.py`:
- Delete `test_column_sizes_int128_and_nested_int128`.
- In `ineligible_frame`, after the `cols = [...]` line, add:

```python
    cols += [pl.Series("i128", [1, 2], dtype=pl.Int128), pl.Series("list_i128", [[1], [2]], dtype=pl.List(pl.Int128))]
```

- In `NESTED_CASES`, delete the `id="list_int128"` row, and rename `test_nested_floats_enums_and_int128` to `test_nested_floats_and_enums`.

`tests/test_recommend.py`:
- In `KNOWN`, delete the `id="int128_beyond_64_bits"` row.
- In `test_contract`, delete the trailing comment `# describe_mixed nests no Int128`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `$PY -m pytest tests -q -m "not slow"`
Expected: all pass. The skip count is unchanged apart from `test_plugin.py`'s optional-library skips.

- [ ] **Step 9: Commit**

```bash
git add tests/test_plugin.py tests/test_gcd.py tests/test_describe.py tests/test_recommend.py \
  services/analytics/analytics/_dtypes.py services/analytics/analytics/gcd/base.py \
  services/analytics/analytics/gcd/math.py services/analytics/analytics/gcd/numpy.py \
  services/analytics/analytics/chi_squared/base.py services/analytics/analytics/describe/base.py \
  services/analytics/analytics/describe/datafusion.py services/analytics/analytics/describe/_sizes.py
git commit -m "feat: Int128 and UInt128 are ineligible in every technique

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Remove the dead logical-Int128 branches in Rust

Polars stores Decimal **physically** as Int128. Keep the physical arms: `DataType::Int128 => ca_gcd(phys.i128()?, ...)` in `gcd.rs::series_gcd`, `DataType::Int128 => extremes(p.i128()?...)` in `describe.rs`, and the `DataType::Decimal(_, _)` arm of `encode_series`. Remove only the branches that match the **logical** Int128 dtype.

**Files:**
- Modify: `services/analytics/src/gcd.rs`, `services/analytics/src/shared.rs`, `services/analytics/src/entropy.rs` (test only), `services/analytics/src/sizes.rs`, `services/analytics/src/recommend.rs`

- [ ] **Step 1: gcd.rs**

- Delete `| DataType::Int128` from `is_integer_backed`.
- Delete `const GCD_LIMIT: u128 = 10u128.pow(38);`.
- In `series_gcd`, replace `Ok((g < GCD_LIMIT).then_some(g as i128))` with `Ok(Some(g as i128))`, and replace its doc comment with:

```rust
/// Whole-column GCD of `s`; `None` for non-integer-backed dtypes. The Int128 arm
/// reads Decimal's physical values, whose magnitudes fit in 38 digits.
```

- Change `/// u128 twin of [`step_u64`] (Int128 / Decimal).` to `/// u128 twin of [`step_u64`] (Decimal).`
- In the test `signed_and_unsigned_extremes`, replace the six lines from `assert_eq!(gcd_of(Series::new("a".into(), &[i128::MIN, 1i128 << 126])), ...` through `assert_eq!(gcd_of(Series::new("a".into(), &[i128::MIN])), None);` (including the comment between them) with:

```rust
        let dec = |v: &[i128]| Int128Chunked::from_slice("a".into(), v).into_decimal_unchecked(Some(38), 0).into_series();
        assert_eq!(gcd_of(dec(&[10i128.pow(38) - 1])), Some(10i128.pow(38) - 1));
        assert_eq!(gcd_of(dec(&[-(10i128.pow(37)), 10i128.pow(36)])), Some(10i128.pow(36)));
        assert_eq!(gcd_of(Series::new("a".into(), &[12i128])), None); // Int128 is not integer-backed
```

- [ ] **Step 2: shared.rs and entropy.rs**

- In `shared.rs::encode_series`, delete the whole `DataType::Int128 => { ... }` arm: the block that hashes `series.i128()?` values. Keep the `DataType::Decimal(_, _)` arm further down.
- In `entropy.rs`, delete the whole `#[test] fn test_int128_to_u64() { ... }`.

- [ ] **Step 3: sizes.rs — sizes are always present**

- In the module comment, replace the sentences `Columns nesting Int128 inside List/Array/Struct get null sizes: pyarrow cannot import them, so there is no oracle to agree with.` with nothing, so the comment ends at `(view types).`
- Delete `nests_int128`.
- Replace `Sizes`, `classic_layout`, `sizes` and `sizes_of` with:

```rust
pub(crate) type Sizes = [u64; 4];
pub(crate) const SIZE_FIELDS: [&str; 4] = ["size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes"];

/// `s`'s classic layout (CompatLevel::oldest).
pub(crate) fn classic_layout(s: &Series) -> PolarsResult<arrow_array::ArrayRef> {
    to_arrow_rs(s, CompatLevel::oldest())
}

pub(crate) fn sizes(s: &Series, level: i32) -> PolarsResult<Sizes> {
    sizes_of(s, &classic_layout(s)?, level)
}

/// `sizes` given `s`'s `classic_layout` (exported once by callers that reuse it).
pub(crate) fn sizes_of(s: &Series, classic: &arrow_array::ArrayRef, level: i32) -> PolarsResult<Sizes> {
    let native = to_arrow_rs(s, CompatLevel::newest())?;
    Ok([
        ipc_body_bytes(classic.as_ref(), None)?,
        ipc_body_bytes(classic.as_ref(), Some(level))?,
        ipc_body_bytes(native.as_ref(), None)?,
        ipc_body_bytes(native.as_ref(), Some(level))?,
    ])
}
```

- In `column_sizes_impl`, change the per-field line to:

```rust
        columns.push(UInt64Chunked::from_iter_values((*name).into(), rows.iter().map(|r| r[j])).into_series());
```

- Tests: delete `nested_int128_sizes_are_null`. Wherever a test compares an element of `sizes(...)` with `Some(n)`, compare with `n`. For example, `assert_eq!(sizes(&cat, 1).unwrap()[0], 48);` and `assert_eq!(sizes(&Series::new("x".into(), &[Some("ab"), None]), 1).unwrap()[2], 40);`.

- [ ] **Step 4: recommend.rs — `recommend` returns `Rec`**

- In `Rules`' candidates match, change `PT::Int8 | PT::Int16 | PT::Int32 | PT::Int64 | PT::Int128 | PT::UInt8 ...` to `PT::Int8 | PT::Int16 | PT::Int32 | PT::Int64 | PT::UInt8 ...`, dropping `PT::Int128`.
- Change the doc comment of `int_at` to `/// Row `i` as an exact integer (integer columns).`
- Replace `recommend`'s doc comment, signature and first line with:

```rust
/// The recommendation for one column.
/// `values` is `s` in the classic layout (sizes.rs's `classic_layout`).
pub(crate) fn recommend(s: &Series, values: &ArrayRef, d: &Described, sz: &Sizes, params: &Params) -> PolarsResult<Rec> {
    let [size_bytes, _, polars_bytes, polars_zstd] = *sz;
```

  At its end, replace `Ok(Some(Rec {` with `Ok(Rec {`, and the closing `}))` with `})`.
- Replace `rec_row` with:

```rust
fn rec_row(r: &Rec) -> Row {
    let text = |s: &str| AnyValue::StringOwned(s.into());
    vec![
        AnyValue::Boolean(r.nullable),
        text(&r.arrow_type),
        AnyValue::UInt64(r.arrow_size),
        AnyValue::UInt64(r.arrow_zstd),
        r.polars_type.as_deref().map_or(AnyValue::Null, text),
        AnyValue::UInt64(r.polars_size),
        AnyValue::UInt64(r.polars_zstd),
        AnyValue::Boolean(r.lossy),
        AnyValue::List(candidates_series(&r.candidates)),
    ]
}
```

- In `describe_and_recommend_impl`, replace the closure body with:

```rust
        .map(|s| {
            let d = describe_one(s, params.seed)?;
            let classic = classic_layout(s)?;
            let sz = sizes_of(s, &classic, params.zstd_level)?;
            let rec = recommend(s, &classic, &d, &sz, params)?;
            let mut row = d.row();
            row.extend(sz.iter().map(|&v| AnyValue::UInt64(v)));
            row.extend(rec_row(&rec));
            Ok(row)
        })
```

- Tests:
  - replace `classic_layout(&s).unwrap().unwrap()` with `classic_layout(&s).unwrap()` in `rec`, `polars_types_of_results` and `rec_with`;
  - replace `classic_layout(s).unwrap().unwrap()` with `classic_layout(s).unwrap()` in `inner_chosen`;
  - drop the final `.unwrap()` of each `recommend(...).unwrap().unwrap()`.
- In `plugin_output_matches_declared_schema`:
  - replace the `nested_int128` line with `let nested = Series::new("n".into(), [Some(Series::new("".into(), &[1i64])), None, Some(Series::new("".into(), &[2i64]))]);`;
  - use `nested` in `inputs`;
  - replace the two assertions about row 2 with `assert_eq!(types.null_count(), 0);` and `assert!(cands.list().unwrap().get_as_series(2).is_some());`.

- [ ] **Step 5: Run the Rust tests**

Run: `cargo test --lib` (see Commands)
Expected: all pass (about 190 tests, a few fewer than the 193 before).

- [ ] **Step 6: Rebuild and run the Python suite**

Run the maturin build, then `$PY -m pytest tests -q -m "not slow"`.
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/src/gcd.rs services/analytics/src/shared.rs services/analytics/src/entropy.rs \
  services/analytics/src/sizes.rs services/analytics/src/recommend.rs
git commit -m "refactor: drop Rust branches for logical Int128, now unreachable

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `arrow_io.rs` — RecordBatch ↔ Series

**Files:**
- Create: `services/analytics/src/arrow_io.rs`
- Modify: `services/analytics/src/lib.rs`, `services/analytics/src/shared.rs` (remove `to_arrow_rs` and its tests), `services/analytics/src/sizes.rs`, `services/analytics/src/recommend.rs` (use `export_series`)

- [ ] **Step 1: Write `arrow_io.rs` with its tests**

The body of `export_series` is `shared::to_arrow_rs` moved verbatim, minus its Int128 relabelling (Task 2 made that unreachable). The tests from `shared.rs`'s `mod arrow_rs_tests` move here with `to_arrow_rs` renamed to `export_series`, minus the `1i128` assertion.

```rust
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
    // SAFETY: <copy the full SAFETY comment from shared::to_arrow_rs unchanged>
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

    // ── moved from shared.rs (to_arrow_rs → export_series; the 1i128 line dropped) ──
    // <paste series_cross_to_arrow_rs, sliced_int32_with_nulls_at_unaligned_offset,
    //  boolean_series_round_trips and multi_chunk_series_is_rechunked here verbatim,
    //  replacing `to_arrow_rs(` with `export_series(` and deleting the line
    //  `assert_eq!(to_arrow_rs(&Series::new("x".into(), &[1i128]), ...Decimal128(38, 0));`>

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
```

In `export_series`, replace the `// SAFETY: <copy …>` placeholder line with the full SAFETY comment block from `shared::to_arrow_rs` (from `// SAFETY: \`polars_arrow::ffi::{ArrowSchema, ArrowArray}\`` through `releasing the exported \`Field\` once.`), unchanged. Likewise, replace the `// <paste …>` comment in the test module with the four moved tests.

- [ ] **Step 2: Register the module and switch callers**

- `lib.rs`: add `mod arrow_io;` after `mod shared;`.
- `shared.rs`: delete `to_arrow_rs`, its doc comment and `mod arrow_rs_tests`. Then remove any `use` that the compiler now reports unused (for example `CompatLevel` or `polars_arrow` items).
- `sizes.rs`: replace `use crate::shared::to_arrow_rs;` with `use crate::arrow_io::export_series;`, and every `to_arrow_rs(` with `export_series(`. In the test helper `arrow`, replace `crate::shared::to_arrow_rs(s, CompatLevel::oldest())` with `crate::arrow_io::export_series(s, CompatLevel::oldest())`. Update the module comment `Series arrive through shared::to_arrow_rs` to `Series arrive through arrow_io::export_series`.
- `recommend.rs`: run `grep -n "to_arrow_rs" services/analytics/src/recommend.rs`. For each hit, import `crate::arrow_io::export_series` and rename the call.

- [ ] **Step 3: Run the Rust tests**

Run: `cargo test --lib arrow_io::` then `cargo test --lib`
Expected: the 8 `arrow_io` tests pass (4 moved, 4 new), and so does the whole suite. `grep -rn to_arrow_rs services/analytics/src` prints nothing.

If `import_restores_every_dtype` fails only on the Categorical or Enum dtype string, check `polars_batch`: its field must carry Polars' metadata. The Python-side Categorical and Enum round trip is covered again in Task 6.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/arrow_io.rs services/analytics/src/lib.rs services/analytics/src/shared.rs \
  services/analytics/src/sizes.rs services/analytics/src/recommend.rs
git commit -m "feat: arrow_io — RecordBatch to Series and back, zero-copy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `api.rs` — the language-neutral core

**Files:**
- Create: `services/analytics/src/api.rs`
- Modify: `services/analytics/src/lib.rs`, `services/analytics/src/bloomfilter.rs`, `services/analytics/src/minhash.rs`

- [ ] **Step 1: Open up the kernel parameters `api.rs` builds**

- `bloomfilter.rs`:
  - make `struct BloomFilterKwargs` and `struct MembershipKwargs` `pub(crate)`, with `pub(crate)` fields;
  - make `fn bloom_filter_impl` and `fn membership_ratio_multi_impl` `pub(crate)`.
- `minhash.rs`:
  - in `LSHKwargs`, delete the `threshold: f64,` field and make the other fields `pub(crate)`;
  - in `lsh_candidates_impl`, delete the line `let _ = kwargs.threshold; // Available for future filtering if needed`;
  - in the doc comment above `lsh_candidates`, change `LSHKwargs containing threshold and band configuration` to `LSHKwargs containing the band configuration`;
  - make `MinHashKwargs`' fields `pub(crate)`.

  The Python wrapper still sends `threshold` until Task 6; serde ignores unknown kwargs fields.

- [ ] **Step 2: Write `api.rs` with its tests**

```rust
//! The language-neutral core: every entry point takes an Arrow RecordBatch plus
//! plain parameters and returns an Arrow RecordBatch (Bloom: bytes). No pyo3 and no
//! Polars type appears in any signature — bindings (python.rs; later Java / C) wrap
//! exactly this module. Kernels compute on Polars Series behind arrow_io.

use std::collections::HashSet;
use std::fmt;

use arrow_array::RecordBatch;
use polars::prelude::{IntoSeries, PolarsError, PolarsResult, Series, StructChunked};

use crate::arrow_io::{export_struct, import_batch};
use crate::bloomfilter::{BloomFilterKwargs, MembershipKwargs};
use crate::minhash::{LSHKwargs, MinHashKwargs};
use crate::recommend::Params;
use crate::shared::{PairwiseKwargs, ThreewayKwargs};

#[derive(Debug, PartialEq)]
pub enum Error {
    /// The caller's input: unknown column names, a malformed Bloom array, an Arrow
    /// type no kernel accepts.
    InvalidInput(String),
    /// Any other failure inside a kernel.
    Compute(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) | Error::Compute(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn compute(e: PolarsError) -> Error {
    Error::Compute(e.to_string())
}

fn columns(batch: &RecordBatch) -> Result<Vec<Series>> {
    import_batch(batch).map_err(|e| Error::InvalidInput(e.to_string()))
}

fn table(out: PolarsResult<Series>) -> Result<RecordBatch> {
    out.and_then(|s| export_struct(&s)).map_err(compute)
}

fn check_names<'a>(batch: &RecordBatch, names: impl IntoIterator<Item = &'a String>) -> Result<()> {
    let schema = batch.schema();
    let known: HashSet<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    match names.into_iter().find(|n| !known.contains(n.as_str())) {
        Some(n) => Err(Error::InvalidInput(format!("unknown column {n:?}"))),
        None => Ok(()),
    }
}

fn pairwise(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<PairwiseKwargs> {
    let Some(pairs) = pairs else { return Ok(PairwiseKwargs { pairs: None }) };
    check_names(batch, pairs.iter().flat_map(|(a, b)| [a, b]))?;
    Ok(PairwiseKwargs { pairs: Some(pairs.iter().map(|(a, b)| vec![a.clone(), b.clone()]).collect()) })
}

/// `column`, `dtype`, `gcd: decimal128(38, 0)` per column.
pub fn column_gcd(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::gcd::column_gcd_impl(&columns(batch)?))
}

/// `col_name`, `entropy` per column.
pub fn marginal_entropy(batch: &RecordBatch) -> Result<RecordBatch> {
    table(crate::entropy::marginal_entropy_impl(&columns(batch)?))
}

/// `col_a`, `col_b`, `entropy` per pair; every pair when `pairs` is None.
pub fn pairwise_joint_entropy(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::entropy::pairwise_joint_entropy_impl(&columns(batch)?, kwargs))
}

/// `col_a`, `col_b`, `col_c`, `entropy` per triplet; every triplet when None.
pub fn threeway_joint_entropy(batch: &RecordBatch, triplets: Option<&[(String, String, String)]>) -> Result<RecordBatch> {
    if let Some(t) = triplets {
        check_names(batch, t.iter().flat_map(|(a, b, c)| [a, b, c]))?;
    }
    let kwargs = ThreewayKwargs {
        triplets: triplets.map(|t| t.iter().map(|(a, b, c)| vec![a.clone(), b.clone(), c.clone()]).collect()),
    };
    table(crate::entropy::threeway_joint_entropy_impl(&columns(batch)?, kwargs))
}

/// `col_a`, `col_b`, `chi2_stat`, `p_value`, `cramers_v`, `low_expected_count`, `n_valid`.
pub fn pairwise_chi_squared(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::chi_squared::pairwise_chi_squared_impl(&columns(batch)?, kwargs))
}

/// `col_a`, `col_b`, `ari`, `n_valid` per pair.
pub fn pairwise_adjusted_rand(batch: &RecordBatch, pairs: Option<&[(String, String)]>) -> Result<RecordBatch> {
    let kwargs = pairwise(batch, pairs)?;
    table(crate::ari::pairwise_adjusted_rand_impl(&columns(batch)?, kwargs))
}

/// A fresh k-hash, m-bit Bloom filter over a one-column batch: ⌈m/8⌉ bytes.
pub fn bloom_filter(batch: &RecordBatch, k: usize, m: usize) -> Result<Vec<u8>> {
    let cols = columns(batch)?;
    let [s] = cols.as_slice() else {
        return Err(Error::InvalidInput(format!("bloom_filter takes one column, got {}", cols.len())));
    };
    crate::bloomfilter::bloom_filter_impl(s, BloomFilterKwargs { bit_array_bytes: Vec::new(), k, m }).map_err(compute)
}

/// `col_name`, `ratio_all`, `ratio_non_null` per column, against the filter `bits`.
pub fn membership_ratio(batch: &RecordBatch, bits: &[u8], k: usize, m: usize) -> Result<RecordBatch> {
    if bits.len() != m.div_ceil(8) {
        return Err(Error::InvalidInput(format!(
            "bloom filter bit array has {} bytes but m={m} bits requires {} bytes",
            bits.len(),
            m.div_ceil(8)
        )));
    }
    let kwargs = MembershipKwargs { bit_array_bytes: bits.to_vec(), k, m };
    table(crate::bloomfilter::membership_ratio_multi_impl(&columns(batch)?, &kwargs))
}

/// `qualified_name` ("{df_name}|{column}"), `minhash: list<uint32>` per column.
pub fn minhash(batch: &RecordBatch, df_name: &str, num_perm: usize) -> Result<RecordBatch> {
    let cols = columns(batch)?;
    let packed = StructChunked::from_series("frame".into(), batch.num_rows(), cols.iter()).map_err(compute)?.into_series();
    table(crate::minhash::minhash_impl(&[packed], &MinHashKwargs { df_name: df_name.to_owned(), num_perm }))
}

/// `col_a`, `col_b` per candidate pair, from a batch of (names, signatures) — by position.
pub fn lsh_candidates(signatures: &RecordBatch, num_bands: usize, rows_per_band: usize) -> Result<RecordBatch> {
    let cols = columns(signatures)?;
    if cols.len() != 2 {
        return Err(Error::InvalidInput(format!("lsh_candidates takes (names, signatures), got {} columns", cols.len())));
    }
    table(crate::minhash::lsh_candidates_impl(&cols, &LSHKwargs { num_bands, rows_per_band }))
}

/// `column` plus Describe's value metrics per column.
pub fn describe_columns(batch: &RecordBatch, seed: u64) -> Result<RecordBatch> {
    table(crate::describe::describe_columns_impl(&columns(batch)?, seed))
}

/// `column`, `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes`.
pub fn column_sizes(batch: &RecordBatch, zstd_level: i32) -> Result<RecordBatch> {
    table(crate::sizes::column_sizes_impl(&columns(batch)?, zstd_level))
}

/// Describe's table, the size columns and the `rec_*` columns per column.
pub fn describe_and_recommend(
    batch: &RecordBatch,
    seed: u64,
    zstd_level: i32,
    population_rows: Option<u64>,
    categorical_threshold: u64,
    boolean_pairs: Vec<(String, String)>,
) -> Result<RecordBatch> {
    let params = Params { seed, zstd_level, population_rows, categorical_threshold, boolean_pairs };
    table(crate::recommend::describe_and_recommend_impl(&columns(batch)?, &params))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal128Type, Float64Type};
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::DataType as AT;

    use super::*;

    fn ints(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
    }

    #[test]
    fn gcd_of_plain_arrow_columns() {
        let out = column_gcd(&batch(vec![("a", ints(&[12, 18, 30])), ("b", ints(&[7, 14, 21]))])).unwrap();
        assert_eq!(out.schema().field(2).data_type(), &AT::Decimal128(38, 0));
        let gcd = out.column(2).as_primitive::<Decimal128Type>();
        assert_eq!((gcd.value(0), gcd.value(1)), (6, 7));
        assert_eq!(out.column(0).as_string_view().value(1), "b");
    }

    #[test]
    fn pairwise_entropy_output_columns() {
        let out = pairwise_joint_entropy(&batch(vec![("a", ints(&[1, 1, 2, 2])), ("b", ints(&[1, 2, 1, 2]))]), None).unwrap();
        let names: Vec<String> = out.schema().fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, ["col_a", "col_b", "entropy"]);
        assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 2.0);
    }

    #[test]
    fn unknown_pair_column_is_invalid_input() {
        let b = batch(vec![("a", ints(&[1, 2]))]);
        let err = pairwise_chi_squared(&b, Some(&[("a".into(), "zz".into())])).unwrap_err();
        assert_eq!(err, Error::InvalidInput("unknown column \"zz\"".into()));
        assert!(matches!(threeway_joint_entropy(&b, Some(&[("a".into(), "a".into(), "q".into())])), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn bloom_filter_then_membership() {
        let b = batch(vec![("a", ints(&[1, 2, 3]))]);
        let bits = bloom_filter(&b, 3, 64).unwrap();
        assert_eq!(bits.len(), 8);
        let out = membership_ratio(&b, &bits, 3, 64).unwrap();
        assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 1.0);
    }

    #[test]
    fn bloom_errors_are_invalid_input() {
        let b = batch(vec![("a", ints(&[1])), ("b", ints(&[2]))]);
        assert!(matches!(bloom_filter(&b, 3, 64), Err(Error::InvalidInput(_))));
        assert!(matches!(membership_ratio(&b, &[0u8; 3], 3, 64), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn minhash_then_lsh() {
        let sigs = minhash(&batch(vec![("a", ints(&[1, 2, 3])), ("b", ints(&[1, 2, 3]))]), "0", 16).unwrap();
        assert_eq!(sigs.num_rows(), 2);
        assert_eq!(lsh_candidates(&sigs, 4, 4).unwrap().num_rows(), 1); // identical columns always collide
    }

    #[test]
    fn lsh_on_a_non_list_signature_is_a_compute_error() {
        let bad = batch(vec![("qualified_name", Arc::new(StringArray::from(vec!["a"])) as ArrayRef), ("minhash", ints(&[1]))]);
        assert!(matches!(lsh_candidates(&bad, 1, 1), Err(Error::Compute(_))));
    }

    #[test]
    fn describe_sizes_and_recommend_one_row_per_column() {
        let b = batch(vec![("a", ints(&[0, 5, 7])), ("s", Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef)]);
        assert_eq!(describe_columns(&b, 0).unwrap().num_rows(), 2);
        assert_eq!(column_sizes(&b, 1).unwrap().num_rows(), 2);
        let rec = describe_and_recommend(&b, 0, 1, None, 10_000, vec![("true".into(), "false".into())]).unwrap();
        assert_eq!(rec.column_by_name("rec_arrow_type").unwrap().as_string_view().value(0), "uint8");
    }
}
```

- [ ] **Step 3: Register the module**

`lib.rs`: add `mod api;` after `mod arrow_io;`.

- [ ] **Step 4: Run the Rust tests**

Run: `cargo test --lib api::` then `cargo test --lib`
Expected: all pass. Until Task 5 uses them, the compiler warns that the `api` functions are unused; that is expected.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/api.rs services/analytics/src/lib.rs services/analytics/src/bloomfilter.rs services/analytics/src/minhash.rs
git commit -m "feat: api.rs — Arrow-in, Arrow-out core over the kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `python.rs` — the pyo3 binding

The binding lives alongside the old plugins until Task 7. Polars still loads the same library as a plugin, and Python can now also import it as the module `analytics.analytics`.

**Files:**
- Create: `services/analytics/src/python.rs`
- Modify: `services/analytics/src/lib.rs`
- Test: `tests/test_plugin.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plugin.py`:

```python
# ── the binding (analytics.analytics) ─────────────────────────────────────────

import pyarrow as pa

from analytics import analytics as rs


def test_binding_takes_and_returns_arrow():
    out = pl.DataFrame(rs.column_gcd(pl.DataFrame({"a": [12, 18], "b": [7, 14]})))
    assert out.columns == ["column", "dtype", "gcd"]
    assert out["gcd"].to_list() == [6, 7]
    assert pa.table(rs.column_gcd(pa.table({"a": [12, 18]}))).num_rows == 1


def test_bloom_bits_are_bytes():
    bits = rs.bloom_filter(pl.DataFrame({"a": [1, 2, 3]}), 3, 64)
    assert isinstance(bits, bytes) and len(bits) == 8
    ratios = pl.DataFrame(rs.membership_ratio(pl.DataFrame({"a": [1, 2, 3]}), bits, 3, 64))
    assert ratios["ratio_non_null"].to_list() == [1.0]


def test_unknown_pair_column_is_value_error():
    with pytest.raises(ValueError, match='unknown column "nope"'):
        rs.pairwise_joint_entropy(pl.DataFrame({"a": [1, 2]}), [("a", "nope")])


def test_wrong_bloom_length_is_value_error():
    with pytest.raises(ValueError, match="requires 8 bytes"):
        rs.membership_ratio(pl.DataFrame({"a": [1]}), b"\x00", 3, 64)


def test_kernel_failure_is_runtime_error():
    with pytest.raises(RuntimeError):
        rs.lsh_candidates(pl.DataFrame({"qualified_name": ["a"], "minhash": [1]}), 1, 1)


def test_non_arrow_input_is_type_error():
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        rs.column_gcd(42)


WIDE = [pl.Series("x", [1], dtype=pl.Int128), pl.Series("x", [[1]], dtype=pl.List(pl.Int128))]
if hasattr(pl, "UInt128"):
    WIDE.append(pl.Series("x", [1], dtype=pl.UInt128))


@pytest.mark.parametrize("s", WIDE, ids=lambda s: str(s.dtype))
def test_128_bit_integers_are_rejected_at_the_binding(s):
    with pytest.raises(ValueError, match='column "x" holds .*128-bit'):
        rs.column_gcd(s.to_frame())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_plugin.py -q -k "binding or bloom_bits or error or rejected"`
Expected: FAIL at collection with `ImportError: cannot import name 'analytics' from 'analytics'`, or a missing module-init error, because there is no `#[pymodule]` yet.

- [ ] **Step 3: Write `python.rs`**

```rust
//! Python binding over api.rs. Tables arrive as any object implementing the Arrow
//! PyCapsule interface (`__arrow_c_stream__`: Polars and pyarrow frames and readers)
//! and leave as `ArrowTable`, which implements it too. Errors: invalid input →
//! ValueError, kernel failure → RuntimeError.

use std::ffi::{c_char, c_int, c_void};

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
        return Err(value_error("arrow stream: get_schema failed"));
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
```

- [ ] **Step 4: Register the module, build, and run the tests**

- `lib.rs`: add `mod python;` after `mod api;`.
- Run `cargo test --lib`: expected pass, with no unused-`api` warnings left.
- Run the maturin build.
- Run `$PY -m pytest tests/test_plugin.py -q`: expected all pass.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/python.rs services/analytics/src/lib.rs tests/test_plugin.py
git commit -m "feat: python.rs — Arrow PyCapsule binding over api.rs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Python wrappers call the binding

**Files:**
- Modify: `services/analytics/analytics/_plugin.py` (rewrite), `services/analytics/analytics/{adjusted_rand,chi_squared,gcd,membership,pairwise_entropy,threeway_entropy,similarity}/rust.py`

- [ ] **Step 1: Rewrite `_plugin.py`**

Replace the whole file with:

```python
"""Private thin wrappers around the compiled Rust extension (analytics.pyd).

Tables cross the boundary as Arrow: the Polars frames given here are passed as they
are (they implement the Arrow PyCapsule interface, zero-copy) and results come back
as Arrow tables, read into flat Polars frames. Only the *Rust implementation classes
call these; the public API is the technique classes in analytics.<technique>.
"""

import polars as pl

from analytics import analytics as _rs


def _collected(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _tuples(combos):
    return None if combos is None else [tuple(c) for c in combos]


def bloom_filter_bits(series: pl.Series, k: int, m: int) -> bytes:
    """A fresh k-hash, m-bit Bloom filter over `series`: its ceil(m/8) bytes."""
    return _rs.bloom_filter(series.to_frame(), k, m)


def membership_ratio(df: pl.DataFrame | pl.LazyFrame, bits: bytes, k: int, m: int) -> pl.DataFrame:
    """Per column: col_name, ratio_all (nulls in the denominator), ratio_non_null."""
    return pl.DataFrame(_rs.membership_ratio(_collected(df), bits, k, m))


def pairwise_joint_entropy(df, pairs=None) -> pl.DataFrame:
    """col_a, col_b, entropy (H(A,B) in bits); every pair when `pairs` is None."""
    return pl.DataFrame(_rs.pairwise_joint_entropy(_collected(df), _tuples(pairs)))


def threeway_joint_entropy(df, triplets=None) -> pl.DataFrame:
    """col_a, col_b, col_c, entropy (H(A,B,C) in bits); every triplet when None —
    no cap (C(101, 3) = 166,650 at 101 columns, ~65 s at 50K rows)."""
    return pl.DataFrame(_rs.threeway_joint_entropy(_collected(df), _tuples(triplets)))


def marginal_entropy(df) -> pl.DataFrame:
    """col_name, entropy per column: the shared encoder and SIMD reduction, not the
    dense-id counting the joint entropies use (an independent cross-check)."""
    return pl.DataFrame(_rs.marginal_entropy(_collected(df)))


def pairwise_chi_squared(df, pairs=None) -> pl.DataFrame:
    """col_a, col_b, chi2_stat, p_value, cramers_v, low_expected_count, n_valid."""
    return pl.DataFrame(_rs.pairwise_chi_squared(_collected(df), _tuples(pairs)))


def pairwise_adjusted_rand(df, pairs=None) -> pl.DataFrame:
    """col_a, col_b, ari, n_valid; null rows dropped pairwise (sklearn conventions)."""
    return pl.DataFrame(_rs.pairwise_adjusted_rand(_collected(df), _tuples(pairs)))


def column_gcd(df) -> pl.DataFrame:
    """column, dtype (Rust display form), gcd: Decimal(38, 0) per column."""
    return pl.DataFrame(_rs.column_gcd(_collected(df)))


def minhash(df, name: str, num_perm: int = 128) -> pl.DataFrame:
    """qualified_name ("{name}|{column}"), minhash: List(UInt32) per column."""
    return pl.DataFrame(_rs.minhash(_collected(df), name, num_perm))


def lsh_candidates(signatures: pl.DataFrame, num_bands: int, rows_per_band: int) -> pl.DataFrame:
    """col_a, col_b per LSH candidate pair of `signatures` (as `minhash` returns them)."""
    return pl.DataFrame(_rs.lsh_candidates(signatures.select("qualified_name", "minhash"), num_bands, rows_per_band))


def describe_columns(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """column + Describe's value metrics per column (see analytics.describe.base)."""
    return pl.DataFrame(_rs.describe_columns(df, seed))


def column_sizes(df: pl.DataFrame, zstd_level: int) -> pl.DataFrame:
    """column + Arrow (classic layout) and Polars (native layout) IPC body sizes."""
    return pl.DataFrame(_rs.column_sizes(df, zstd_level))


def describe_and_recommend(
    df: pl.DataFrame,
    *,
    seed: int,
    zstd_level: int,
    population_rows: int | None,
    categorical_threshold: int,
    boolean_pairs: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    """column, every Describe metric, the size metrics and the rec_* columns."""
    return pl.DataFrame(
        _rs.describe_and_recommend(
            df,
            seed=seed,
            zstd_level=zstd_level,
            population_rows=population_rows,
            categorical_threshold=categorical_threshold,
            boolean_pairs=[tuple(p) for p in boolean_pairs],
        )
    )
```

- [ ] **Step 2: Drop `.unnest(...)` in the callers**

- `adjusted_rand/rust.py`: `out = _plugin.pairwise_adjusted_rand(df, pairs)`
- `chi_squared/rust.py`: `out = _plugin.pairwise_chi_squared(df, pairs)`
- `gcd/rust.py`: `out = _plugin.column_gcd(frames[frame].select(columns))`
- `threeway_entropy/rust.py`: `out = _plugin.threeway_joint_entropy(df, triplets)`
- `pairwise_entropy/rust.py`:
  - `marginal = dict(_plugin.marginal_entropy(df).iter_rows())`
  - `h_ab = {frozenset((a, b)): h for a, b, h in joint.iter_rows()}`
- `membership/rust.py`: `ratios = _plugin.membership_ratio(padded, bits, k=k, m=m)`
- `similarity/rust.py`: replace the `pairs = sigs.select(...).unnest("c")` statement with:

```python
                pairs = _plugin.lsh_candidates(sigs, num_bands=self.bands, rows_per_band=self.rows_per_band)
```

- [ ] **Step 3: Run the full Python suite**

No rebuild is needed.

Run: `$PY -m pytest tests -q -m "not slow"`
Expected: all pass. `grep -rn "register_plugin_function\|unnest(\"" services/analytics/analytics` prints nothing.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/analytics/_plugin.py services/analytics/analytics/*/rust.py
git commit -m "refactor: Python wrappers call the Arrow binding; flat result frames

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Remove the Polars plugin machinery from Rust

**Files:**
- Modify: `services/analytics/Cargo.toml` and every kernel file listed below

- [ ] **Step 1: Delete the plugin entry points and their output types**

In each file, delete the `use pyo3_polars::derive::polars_expr;` line, every `#[polars_expr(...)]` function, and the output-type function its macro named:

| File | Delete |
|---|---|
| `ari.rs` | `ari_output_type`, `fn pairwise_adjusted_rand` |
| `chi_squared.rs` | `chi_squared_output_type`, `fn pairwise_chi_squared` |
| `entropy.rs` | `pairwise_entropy_output_type`, `threeway_entropy_output_type`, `marginal_entropy_output_type`, `fn pairwise_joint_entropy`, `fn threeway_joint_entropy`, `fn marginal_entropy` |
| `gcd.rs` | `gcd_output_type`, `fn column_gcd` |
| `minhash.rs` | `lsh_candidates_output`, `minhash_output`, `fn lsh_candidates`, `fn minhash` |
| `bloomfilter.rs` | `membership_ratio_output_type`, `pub fn bloom_filter`, `pub fn membership`, `pub fn membership_ratio`, `pub fn membership_ratio_sample` |
| `describe.rs` | `describe_output_type`, `struct DescribeKwargs`, `fn describe_columns` |
| `sizes.rs` | `sizes_output_type`, `struct SizesKwargs`, `fn column_sizes` |
| `recommend.rs` | `recommend_output_type`, `fn describe_and_recommend` |

Keep every `*_impl` function, and keep the `// ── plugin entry ──` banner's contents except those functions. Rename that banner to `// ── entry ──`.

- [ ] **Step 2: Remove what became unused in `bloomfilter.rs`**

`membership_impl` and `membership_ratio_impl` now serve only tests. Move them, unchanged, into the `#[cfg(test)]` region. The existing test helpers at the end of the file already wrap them, so put them beside those helpers. Delete `membership_ratio_sample_impl`, `MembershipRatioSampleKwargs`, and the test helper and tests that call `membership_ratio_sample_impl`.

- [ ] **Step 3: Drop serde**

- Delete `#[derive(Deserialize)]` (or `Deserialize` from a derive list) on:
  - `PairwiseKwargs` and `ThreewayKwargs` in `shared.rs`;
  - `BloomFilterKwargs` and `MembershipKwargs` in `bloomfilter.rs`;
  - `LSHKwargs` and `MinHashKwargs` in `minhash.rs` (keep `Debug`);
  - `Params` in `recommend.rs` (keep `Clone, Debug`).
- Delete every `use serde::Deserialize;`.
- In `recommend.rs`'s test `plugin_output_matches_declared_schema`, rename it `output_matches_declared_schema` and replace `assert_eq!(out.dtype(), recommend_output_type(&[]).unwrap().dtype());` with:

```rust
        let declared = PT::Struct(output_fields().into_iter().map(|(n, d)| PField::new(n.into(), d)).collect());
        assert_eq!(out.dtype(), &declared);
```

- In `Cargo.toml`, delete the `serde = ...` line and the `pyo3-polars = ...` line. Change the comment `#Polars and pyo3` to `#Polars (kernels) and pyo3 (python.rs)`.

- [ ] **Step 4: Build, test, check**

- Run `cargo test --lib`. Expected: all pass, with no warnings about unused items. Fix any unused import it reports.
- Run `grep -rn "polars_expr\|pyo3_polars\|serde\|to_arrow_rs" services/analytics/src services/analytics/Cargo.toml`. Expected: no output.
- Rebuild with maturin, then run `$PY -m pytest tests -q -m "not slow"`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/Cargo.toml services/analytics/src
git commit -m "refactor: remove the Polars plugin entry points, pyo3-polars and serde

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `Technique.add` accepts Arrow tabular data

**Files:**
- Modify: `services/analytics/analytics/base.py`
- Test: `tests/test_plugin.py`, `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plugin.py`:

```python
# ── Arrow inputs to the technique classes ─────────────────────────────────────

from polars.testing import assert_frame_equal

from datagen import mixed_dtypes

RUST_AND_REFERENCE = [
    pytest.param(f"{p}:{n}", id=n)
    for p in PACKAGES
    for n in importlib.import_module(p).IMPLEMENTATIONS
    if n.endswith("Rust") or n == importlib.import_module(p).REFERENCE
]


@pytest.mark.parametrize("kind", ["table", "record_batch", "reader"])
@pytest.mark.parametrize("spec", RUST_AND_REFERENCE)
def test_arrow_inputs_match_polars(spec, kind):
    df = mixed_dtypes(200, seed=1)
    table = pa.table(df)  # through __arrow_c_stream__: keeps Polars' Categorical metadata
    arrow = {"table": table, "record_batch": table.combine_chunks().to_batches()[0], "reader": table.to_reader()}[kind]
    cls = load(spec)
    assert_frame_equal(run(cls, {"t": arrow}), run(cls, {"t": df}))
```

In `tests/test_base.py`, extend `test_non_frame_raises`:

```python
def test_non_frame_raises():
    with pytest.raises(TypeError, match="DataFrame or LazyFrame, or Arrow tabular data"):
        Toy().add({"f": {"a": [1]}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_plugin.py tests/test_base.py -q -k "arrow_inputs or non_frame"`
Expected: FAIL. The Arrow inputs raise `TypeError: frame 't' must be a polars DataFrame or LazyFrame`, and the message match fails.

- [ ] **Step 3: Implement**

In `services/analytics/analytics/base.py`, replace `add` with:

```python
    def add(self, frames: dict[str, Any]) -> Self:
        """Register named frames: Polars DataFrames / LazyFrames, or any Arrow tabular
        object implementing the Arrow PyCapsule interface (pyarrow Table, RecordBatch,
        RecordBatchReader, …), read once into a Polars DataFrame. Names are unique for
        the life of the instance."""
        accepted = {}
        for name, frame in frames.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"frame names must be non-empty strings, got {name!r}")
            if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
                if not (hasattr(frame, "__arrow_c_stream__") or hasattr(frame, "__arrow_c_array__")):
                    raise TypeError(
                        f"frame {name!r} must be a polars DataFrame or LazyFrame, or Arrow tabular data "
                        f"(an object with __arrow_c_stream__), got {type(frame).__name__}"
                    )
                frame = pl.DataFrame(frame)
            if name in self._frames:
                raise ValueError(f"frame {name!r} already added")
            accepted[name] = frame
        self._frames.update(accepted)
        self._on_add()
        return self
```

Add `Any` to the `typing` import at the top of `base.py` if it is not already imported. Change `self._frames`' annotation in `__init__` only if a type checker complains. It still holds only `pl.DataFrame | pl.LazyFrame`.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests -q -m "not slow"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/base.py tests/test_plugin.py tests/test_base.py
git commit -m "feat: techniques accept Arrow tables, record batches and readers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Timings, documentation, final verification

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md`

- [ ] **Step 1: Compare timings with the baseline**

Run: `$PY tests/performance/results/boundary_timing.py compare tests/performance/results/boundary_baseline.json`
Expected: exit 0, with every class at ≤ 1.10x its baseline median.

If one class exceeds 1.10x, run the comparison once more, since timings are noisy. If it still exceeds 1.10x, stop and report the numbers to the user; do not tune code in this task.

Classes that lost their Int128 columns to `ineligible` may get faster; that is expected.

- [ ] **Step 2: Update `CLAUDE.md`**

- In **# Analytical Functions**:
  - After the sentence ending `use Decimal(38, 0) for 128-bit integers.`, add: `` `add()` accepts Polars DataFrames / LazyFrames and any Arrow tabular object (pyarrow Table, RecordBatch, RecordBatchReader — anything with `__arrow_c_stream__`). Int128 / UInt128 columns, at any depth, are ineligible in every technique: Arrow has no 128-bit integer (`analytics._dtypes.WIDE_INTEGERS`). ``
  - In the GCD paragraph, delete `Int128, ` from the physical-values list, and replace `` `gcd` is Decimal(38, 0), so a GCD over 38 digits (Int128 only) → null; other dtypes (incl. Categorical/Enum, UInt128) → ineligible `` with `` `gcd` is Decimal(38, 0); other dtypes (incl. Categorical/Enum and Int128/UInt128) → ineligible ``.
- In **# Project Structure**:
  - add `api.rs, arrow_io.rs, python.rs` to the `src/` comment;
  - change `_plugin.py  # PRIVATE plugin wrappers (called only by *Rust classes)` to `_plugin.py  # PRIVATE wrappers over the Arrow binding (called only by *Rust classes)`;
  - change `analytics.pyd  # compiled plugin` to `analytics.pyd  # compiled extension (module analytics.analytics)`.
- Replace the **# Rust Plugin (analytics)** section, up to but not including **# Testing Convention**, with:

```markdown
# Rust Extension (analytics)
Build: `maturin develop --release` from `services/analytics/`. Python changes need no rebuild (editable install).

Three layers (spec: docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md):
- `src/api.rs` — the language-neutral core: one `pub fn` per entry point, arrow-rs `RecordBatch` (+ plain parameters) in, `RecordBatch` out (Bloom: bytes). No pyo3 or Polars type in any signature; a future Java / C-ABI binding wraps exactly this file. Errors: `InvalidInput` (unknown column, bad Bloom array, unimportable type) / `Compute`.
- `src/arrow_io.rs` — RecordBatch ↔ Polars Series, zero-copy through the C Data Interface (Polars' `_PL_CATEGORICAL2` / `_PL_ENUM_VALUES2` field metadata restores Categorical / Enum). Kernels still compute on Series; `sizes.rs` / `recommend.rs` measure layouts derived with `export_series`.
- `src/python.rs` — pyo3 module `analytics.analytics`: reads any `__arrow_c_stream__` object into one batch, rejects Polars' private `_pli128` / `_plu128` (Int128 / UInt128) with ValueError naming the column, releases the GIL, returns `ArrowTable` (itself `__arrow_c_stream__`). InvalidInput → ValueError, Compute → RuntimeError, non-Arrow input → TypeError.

Private — reached only through `analytics._plugin`, only by the `*Rust` classes:
`column_gcd`, `pairwise_chi_squared`, `pairwise_adjusted_rand`, `marginal_entropy`,
`pairwise_joint_entropy`, `threeway_joint_entropy` (the classes always pass explicit triplets; `triplets=None` means every triplet, uncapped — C(101,3) = 166,650 at 101 cols, ~65 s at 50K rows),
`bloom_filter` + `membership_ratio` (the bit array crosses as `bytes`), `minhash` + `lsh_candidates`,
`describe_columns` + `column_sizes`, `describe_and_recommend`.
```

- In **# Next Steps**:
  - delete the `membership/rust.py`: bit arrays cross the FFI as `list[int]` bullet;
  - delete the `[minhash.rs:50]` `threshold` bullet;
  - in the `membership/rust.py` BloomRust bullet, replace `(`bloom_filter_bits`, one `register_plugin_function` call each)` with `(`bloom_filter_bits`, one binding call each)`.
- In **# Current Focus**, replace the first sentence's `share one class-based contract` with `share one class-based contract and one Arrow-in/Arrow-out Rust core`.

- [ ] **Step 3: Mark the spec implemented**

In `docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md`, change `Status: approved design, not yet implemented.` to `Status: implemented.`

- [ ] **Step 4: Final verification**

- `cargo test --lib`: all pass.
- `$PY -m pytest tests -q`: all pass, including slow tests.
- `grep -rn "polars_expr\|pyo3_polars\|register_plugin_function\|to_arrow_rs" services/analytics`: no output apart from compiled artifacts.
- `grep -rn "Int128" services/analytics/src`: only the physical-Int128 arms in `gcd.rs::series_gcd` and `describe.rs` extremes, `Int128Chunked` in Decimal construction and tests, and the rejection in `python.rs`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md
git commit -m "docs: Arrow FFI interface — CLAUDE.md and spec status

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
