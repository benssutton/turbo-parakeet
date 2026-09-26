# Recommend Technique Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `analytics.recommend` — for every column, the narrowest value-preserving Arrow type (plus dictionary encoding for strings), cast, verified row by row and measured (Arrow and Polars layouts, plain and ZSTD) — together with the six new Describe metrics it relies on.

**Architecture:** One Rust plugin entry `describe_and_recommend` (in `src/recommend.rs`) runs Describe's profile, ports of the cardinality estimators, the step-1 type rules and step-2 dictionary rule, then casts / verifies / measures with arrow-rs (Series cross zero-copy through the C Data Interface). `src/describe/` is flattened into `src/describe.rs`; `sizes.rs` moves to `src/` and is ported to arrow-rs. The Python package `analytics.recommend` (`Recommend(Describe)`, `RecommendRust`) returns Describe's table plus `rec_*` columns.

**Tech Stack:** Rust (polars 0.51, pyo3-polars 0.24, arrow-rs 60: arrow-array/-buffer/-cast/-data/-schema/-select, zstd, ryu), Python 3.12 (polars 1.41, pyarrow 24, datafusion), pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-recommend-technique-design.md` (Spec B). Spec A: `docs/superpowers/specs/2026-09-26-describe-technique-design.md`.

---

## Before you start

- **Clean tree.** The working tree may hold uncommitted edits in files this plan changes (e.g. `analytics/describe/base.py`, `src/describe/mod.rs`, `analytics/_plugin.py` — the Int128 → Decimal(38, 0) change). Run `git status`; if anything under `services/analytics/` or `tests/` is modified, stop and ask the user to commit or stash it. Never commit files a task does not name.
- **Commands** (Git Bash, repo root `c:/Users/Alexander/turbo-parakeet`):

```bash
PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe
# Rust unit tests (from services/analytics):
cd services/analytics && PYO3_PYTHON=$PY PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH" cargo test --lib <filter>; cd -
# Build the plugin (needed before any pytest run that touches Rust; several minutes — fat LTO):
cd services/analytics && CONDA_PREFIX=/c/Users/Alexander/miniconda3/envs/p312 /c/Users/Alexander/miniconda3/envs/p312/Scripts/maturin.exe develop --release; cd -
# Python tests (from the repo root):
$PY -m pytest tests/test_describe.py -q -m "not slow"
```

Python-only changes need no rebuild (editable install).

## File structure

| File | Responsibility |
|---|---|
| `services/analytics/src/describe.rs` (new; replaces `src/describe/`) | Describe kernels (frequency, range, float, string scanners), typed `Profile` / `Described`, exact parsers `parse_iso` / `parse_decimal`, `assemble`, entry `describe_columns` |
| `services/analytics/src/sizes.rs` (moved) | Arrow IPC body sizes over arrow-rs `ArrayData`; `sizes()` per Series; entry `column_sizes` |
| `services/analytics/src/shared.rs` | + `to_arrow_rs` (Series → arrow-rs, zero-copy) |
| `services/analytics/src/cardinality_estimators.rs` (new) | Chao1, Schnabel, Duj1, `estimate` — ports of `analytics/describe/estimators.py` |
| `services/analytics/src/recommend.rs` (new) | Type names, Polars layout, size formulas, rules → candidates, cast / verify, choose loop, entry `describe_and_recommend` |
| `services/analytics/src/lib.rs` | module list |
| `services/analytics/Cargo.toml` | arrow-rs dependencies |
| `services/analytics/analytics/describe/{base,polars,datafusion,rust,_values,_sizes}.py` | six new metrics; `size_polars_bytes` redefined |
| `services/analytics/analytics/recommend/{__init__,base,rust}.py` (new) | `Recommend`, `RecommendRust` |
| `services/analytics/analytics/_plugin.py` | + `describe_and_recommend` wrapper |
| `tests/test_describe.py` | tests for the new metrics and `size_polars_bytes` |
| `tests/test_recommend.py` (new) | contract, known answers, oracles |
| `tests/performance/benchmark_recommend.py` (new) | parallel-speedup benchmark |
| `CLAUDE.md`, Spec A | documentation |

---

### Task 1: Flatten `src/describe/` into `src/describe.rs`; move `sizes.rs` to `src/`

Pure restructuring — no behaviour change. The existing tests are the safety net.

**Files:**
- Create: `services/analytics/src/describe.rs`
- Move: `services/analytics/src/describe/sizes.rs` → `services/analytics/src/sizes.rs`
- Delete: `services/analytics/src/describe/{mod,frequency,range,numeric,patterns}.rs`
- Modify: `services/analytics/src/lib.rs`, doc comments in `analytics/describe/_sizes.py` and `analytics/describe/_values.py`

- [ ] **Step 1: Move sizes.rs**

```bash
git mv services/analytics/src/describe/sizes.rs services/analytics/src/sizes.rs
```

- [ ] **Step 2: Merge the five files with this script** (save as `$TMP/merge_describe.py` in your scratch directory, run from the repo root with `$PY`)

```python
"""One-off: merge src/describe/{mod,frequency,range,numeric,patterns}.rs into src/describe.rs."""
import re
from pathlib import Path

SRC = Path("services/analytics/src")
D = SRC / "describe"
USES = """use crate::shared::{encode_series, EncodedColumn};
use foldhash::fast::FixedState;
use polars::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
"""


def split(name: str) -> tuple[str, str]:
    text = (D / f"{name}.rs").read_text(encoding="utf-8").replace("\r\n", "\n")
    code, marker, tests = text.partition("#[cfg(test)]\nmod tests {\n")
    keep = [l for l in code.splitlines() if not re.match(r"(pub\(crate\) )?(use|mod) ", l)]
    body = "\n".join(keep).strip("\n")
    if not marker:
        return body, ""
    inner = tests.rstrip().removesuffix("}").rstrip("\n")
    return body, "\n".join(l for l in inner.splitlines() if not l.strip().startswith("use "))


mod_body, _ = split("mod")
banner, _, mod_rest = mod_body.partition("\ntype Row")
parts = [split(n) for n in ("frequency", "range", "numeric", "patterns")]
bodies = "\n\n".join(b for b, _ in parts)
bodies = bodies.replace("\nconst CHUNK: usize = 1 << 16;\n", "\n", 1)  # numeric's copy; frequency's pub(crate) one stays
rest = ("type Row" + mod_rest).replace("frequency::", "").replace("range::", "").replace("numeric::", "")
tests = "\n\n".join(t for _, t in parts if t)
out = f"{banner.rstrip()}\n\n{USES}\n{bodies}\n\n{rest}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n\n{tests}\n}}\n"
(SRC / "describe.rs").write_text(out, encoding="utf-8")
for n in ("mod", "frequency", "range", "numeric", "patterns"):
    (D / f"{n}.rs").unlink()
D.rmdir()
```

- [ ] **Step 3: Register the modules** — `services/analytics/src/lib.rs` becomes:

```rust
mod bloomfilter;
mod minhash;
mod shared;
mod entropy;
mod chi_squared;
mod contingency;
mod ari;
mod gcd;
mod describe;
mod sizes;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

- [ ] **Step 4: Fix doc references**

```bash
sed -i 's#src/describe/sizes.rs#src/sizes.rs#' services/analytics/analytics/describe/_sizes.py
sed -i 's#describe/mod.rs::flatten#describe.rs::flatten#; s#src/describe/patterns.rs#src/describe.rs#g' services/analytics/analytics/describe/_values.py
grep -rn "src/describe/" services/analytics docs/superpowers/specs/2026-09-26-describe-technique-design.md CLAUDE.md
```

The final grep may list Spec A §2 and CLAUDE.md; those are updated in Task 19 — leave them.

- [ ] **Step 5: Run the Rust tests**

Run: `cargo test --lib describe::` then `cargo test --lib sizes::` (see Commands).
Expected: PASS — 18 describe tests (7 frequency + 3 range + 3 numeric + 5 patterns), 4 sizes tests. If the merge left an unused-import warning or a duplicate item, fix it in `describe.rs` (e.g. remove the duplicate `CHUNK`).

- [ ] **Step 6: Build and run the Python tests**

Run: build, then `$PY -m pytest tests/test_describe.py -q -m "not slow"`
Expected: PASS (same count as before the task).

- [ ] **Step 7: Commit**

```bash
git add -A services/analytics/src services/analytics/analytics/describe/_sizes.py services/analytics/analytics/describe/_values.py
git commit -m "refactor: flatten src/describe into describe.rs; sizes.rs to src/"
```

---

### Task 2: Typed `Profile` and `Described`

`profile()` returns a struct instead of a row, so the recommender can read fields by name.

**Files:**
- Modify: `services/analytics/src/describe.rs`

- [ ] **Step 1: Write the failing test** — append inside `mod tests` of `describe.rs`:

```rust
    #[test]
    fn described_row_matches_fields() {
        let list = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]);
        let d = describe_one(&list, 0).unwrap();
        assert_eq!(d.row().len(), fields().len());
        assert_eq!(d.inner.as_ref().unwrap().values.len(), 2);
        let floats = describe_one(&Series::new("y".into(), &[1.5f64]), 0).unwrap();
        assert_eq!(floats.row().len(), fields().len());
        assert!(floats.inner.is_none() && floats.outer.floats.is_some());
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib describe::tests::described_row_matches_fields`
Expected: FAIL to compile (`describe_one` returns `Row`; no field `inner`).

- [ ] **Step 3: Replace `profile`, `describe_one` and `describe_columns_impl`** in `describe.rs` with:

```rust
/// Every value metric of one series (the column, or its list's inner values).
pub(crate) struct Profile {
    pub freq: Frequencies,
    pub range: Range,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    /// Float32 series: `n_f32_inexact` does not apply.
    pub is_f32: bool,
}

pub(crate) fn profile(s: &Series, seed: u64) -> PolarsResult<Profile> {
    Ok(Profile {
        freq: frequencies(&encode_series(s)?, seed),
        range: range(s)?,
        floats: float_stats(s)?,
        strings: strings(s)?,
        is_f32: s.dtype() == &DataType::Float32,
    })
}

impl Profile {
    /// The metrics in `value_fields()` order.
    fn row(&self) -> Row {
        let (f, r) = (&self.freq, &self.range);
        let mut row: Row = vec![
            AnyValue::UInt64(f.n_unique), AnyValue::Float64(f.entropy), AnyValue::UInt64(f.f1), AnyValue::UInt64(f.f2),
            u64v(r.argmin), u64v(r.argmax), u64v(r.min_len), u64v(r.max_len),
            listv(&f.top5_idx), listv(&f.top5_count), listv(&f.capture_history),
        ];
        match self.floats {
            Some(fl) => row.extend([
                AnyValue::UInt64(fl.n_nan), AnyValue::UInt64(fl.n_inf), AnyValue::UInt64(fl.n_fractional), u32v(fl.max_frac_digits),
                if self.is_f32 { AnyValue::Null } else { AnyValue::UInt64(fl.n_f32_inexact) },
            ]),
            None => row.extend(nulls(5)),
        }
        match &self.strings {
            Some(st) => {
                let (lo, hi) = if st.int_overflow { (None, None) } else { (st.int_min, st.int_max) };
                row.extend([
                    AnyValue::UInt64(st.n_numeric), AnyValue::UInt64(st.n_numeric_int), AnyValue::UInt64(st.n_leading_zero),
                    d38v(lo), d38v(hi), u32v(st.max_int_digits), u32v(st.max_frac_digits),
                    AnyValue::UInt64(st.n_iso_date), AnyValue::UInt64(st.n_iso_time), AnyValue::UInt64(st.n_iso_datetime),
                    AnyValue::UInt64(st.n_iso_datetime_tz), u32v(st.iso_max_frac_digits),
                    AnyValue::UInt64(st.offsets.len() as u64), AnyValue::UInt64(st.iso_n_midnight),
                ])
            }
            None => row.extend(nulls(14)),
        }
        row
    }
}

/// A list column's values one level down (`flatten`) and their profile.
pub(crate) struct Inner {
    pub values: Series,
    pub profile: Profile,
}

/// Everything Describe measures on one column (sizes excepted — sizes.rs).
pub(crate) struct Described {
    pub name: PlSmallStr,
    pub n_rows: u64,
    pub n_null: u64,
    pub outer: Profile,
    pub n_midnight: Option<u64>,
    pub inner: Option<Inner>,
}

impl Described {
    /// One output row in `fields()` order.
    pub(crate) fn row(&self) -> Row {
        let mut row: Row = vec![AnyValue::StringOwned(self.name.clone()), AnyValue::UInt64(self.n_rows), AnyValue::UInt64(self.n_null)];
        row.extend(self.outer.row());
        row.push(u64v(self.n_midnight));
        match &self.inner {
            Some(i) => {
                row.push(AnyValue::UInt64(i.values.len() as u64));
                row.push(AnyValue::UInt64(i.values.null_count() as u64));
                row.extend(i.profile.row());
            }
            None => row.extend(nulls(2 + value_fields().len())),
        }
        row
    }
}

pub(crate) fn describe_one(s: &Series, seed: u64) -> PolarsResult<Described> {
    let inner = match flatten(s)? {
        Some(values) => {
            let profile = profile(&values, seed)?;
            Some(Inner { values, profile })
        }
        None => None,
    };
    Ok(Described {
        name: s.name().clone(),
        n_rows: s.len() as u64,
        n_null: s.null_count() as u64,
        outer: profile(s, seed)?,
        n_midnight: n_midnight(s)?,
        inner,
    })
}

/// A Struct series `name` with one field per `fields` entry and one row per `rows` entry.
pub(crate) fn assemble(name: &str, fields: &[(String, DataType)], rows: &[Row]) -> PolarsResult<Series> {
    let columns = fields
        .iter()
        .enumerate()
        .map(|(j, (field, dtype))| {
            let values: Vec<AnyValue> = rows.iter().map(|r| r[j].clone()).collect();
            Series::from_any_values_and_dtype(field.as_str().into(), &values, dtype, true)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructChunked::from_series(name.into(), rows.len(), columns.iter())?.into_series())
}

pub(crate) fn describe_columns_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs.par_iter().map(|s| describe_one(s, seed).map(|d| d.row())).collect::<PolarsResult<_>>()?;
    assemble("describe", &fields(), &rows)
}
```

Also make these `pub(crate)`: `type Row`, `fn fields`, `fn value_fields`, and the structs `Frequencies`, `Range`, `FloatStats`, `StringStats` (already `pub(crate)`; check).

- [ ] **Step 4: Run the Rust tests**

Run: `cargo test --lib describe::`
Expected: PASS (19 tests).

- [ ] **Step 5: Build and run the Python describe tests**

Run: build; `$PY -m pytest tests/test_describe.py -q -m "not slow"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/src/describe.rs
git commit -m "refactor: typed Profile/Described in describe.rs; shared assemble()"
```

---

### Task 3: New Describe metrics — Python base and the Polars reference

Six metrics (Spec B §3): `gcd`, `sum_len`, `sum_len_unique` (group A); `numeric_min_frac_digits`, `numeric_max_sig_digits`, `iso_max_sig_frac_digits` (group C). Tests are parametrized over every implementation; in this task run them for `DescribePolars` only (Rust: Task 4, DataFusion: Task 5).

**Files:**
- Modify: `services/analytics/analytics/describe/base.py` (GROUP_A, GROUP_C)
- Modify: `services/analytics/analytics/describe/_values.py` (`byte_lengths`, `sig_digits`)
- Modify: `services/analytics/analytics/describe/polars.py`
- Test: `tests/test_describe.py`

- [ ] **Step 1: Write the failing tests** — add to `tests/test_describe.py` after `test_sizes_through_the_technique` (add `from decimal import Decimal` and `timedelta` to the imports at the top):

```python
@pytest.mark.parametrize("impl", ALL)
def test_gcd_metric(impl):
    assert profile(impl, pl.Series("x", [10, None, 20, 30]))["gcd"] == 10
    assert profile(impl, pl.Series("x", [None, None], dtype=pl.Int32))["gcd"] == 0
    assert profile(impl, pl.Series("x", [Decimal("1.20"), Decimal("3.40")], dtype=pl.Decimal(10, 2)))["gcd"] == 20
    days = pl.Series("x", [datetime(2024, 1, 1), datetime(2024, 1, 2)], dtype=pl.Datetime("us"))
    assert profile(impl, days)["gcd"] == 86_400_000_000
    assert profile(impl, pl.Series("x", [1.5, 2.5]))["gcd"] is None
    assert profile(impl, pl.Series("x", ["a"], dtype=pl.Categorical))["gcd"] is None
    lists = profile(impl, pl.Series("x", [[4, 8], None, [12]]))
    assert (lists["gcd"], lists["inner_gcd"]) == (None, 4)


@pytest.mark.parametrize("impl", ALL)
def test_sum_len_metrics(impl):
    r = profile(impl, pl.Series("x", ["ab", "ab", "c", None, "héllo"]))
    assert (r["sum_len"], r["sum_len_unique"]) == (11, 9)  # 2+2+1+6, 2+1+6
    c = profile(impl, pl.Series("x", ["ab", "ab", "c"], dtype=pl.Categorical))
    assert (c["sum_len"], c["sum_len_unique"]) == (5, 3)
    b = profile(impl, pl.Series("x", [b"ab", b"ab", None]))
    assert (b["sum_len"], b["sum_len_unique"]) == (4, 2)
    assert profile(impl, pl.Series("x", [1, 2]))["sum_len"] is None
    lists = profile(impl, pl.Series("x", [["ab", "c"], ["ab"]]))
    assert (lists["sum_len"], lists["inner_sum_len"], lists["inner_sum_len_unique"]) == (None, 5, 3)
    empty = profile(impl, pl.Series("x", [], dtype=pl.String))
    assert (empty["sum_len"], empty["sum_len_unique"]) == (0, 0)


@pytest.mark.parametrize("impl", ALL)
def test_numeric_fraction_and_significant_digits(impl):
    r = profile(impl, pl.Series("x", ["1.50", "0.00120", "7", "abc", None]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_frac_digits"], r["numeric_max_sig_digits"]) == (0, 4, 2)
    r = profile(impl, pl.Series("x", ["1200", "-0.0", "12.50"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (0, 4)
    r = profile(impl, pl.Series("x", ["0.25", "1.125"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (2, 4)
    r = profile(impl, pl.Series("x", ["abc"]))
    assert (r["numeric_min_frac_digits"], r["numeric_max_sig_digits"]) == (None, None)


@pytest.mark.parametrize("impl", ALL)
def test_iso_significant_fraction_digits(impl):
    r = profile(impl, pl.Series("x", ["10:00:00.120", "2024-01-05T10:00:00.000", "2024-01-05", None]))
    assert (r["iso_max_frac_digits"], r["iso_max_sig_frac_digits"]) == (3, 2)
    r = profile(impl, pl.Series("x", ["2024-01-05 10:00", "2024-01-05T10:00:00.000000+02:00"]))
    assert (r["iso_max_frac_digits"], r["iso_max_sig_frac_digits"]) == (6, 0)
    assert profile(impl, pl.Series("x", ["2024-01-05"]))["iso_max_sig_frac_digits"] is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_describe.py -q -k "DescribePolars and (gcd_metric or sum_len or significant)"`
Expected: FAIL with `KeyError: 'gcd'` (and the other new names).

- [ ] **Step 3: Add the metrics to the base** — in `analytics/describe/base.py` replace GROUP_A and GROUP_C with:

```python
GROUP_A = {  # whole values — every eligible dtype
    "n_unique": U64, "entropy": F64, "f1": U64, "f2": U64, "argmin": U64, "argmax": U64,
    "min_len": U64, "max_len": U64, "gcd": D38, "sum_len": U64, "sum_len_unique": U64,
    "top5_idx": LU64, "top5_count": LU64, "capture_history": LU64,
}
GROUP_B = {  # Float32 / Float64 only
    "n_nan": U64, "n_inf": U64, "n_fractional": U64, "max_frac_digits": U32, "n_f32_inexact": U64,
}
GROUP_C = {  # String / Categorical / Enum only
    "n_numeric": U64, "n_numeric_int": U64, "n_leading_zero": U64,
    "numeric_int_min": D38, "numeric_int_max": D38,
    "numeric_max_int_digits": U32, "numeric_max_frac_digits": U32,
    "numeric_min_frac_digits": U32, "numeric_max_sig_digits": U32,
    "n_iso_date": U64, "n_iso_time": U64, "n_iso_datetime": U64, "n_iso_datetime_tz": U64,
    "iso_max_frac_digits": U32, "iso_max_sig_frac_digits": U32, "iso_n_offsets": U64, "iso_n_midnight": U64,
}
```

(`gcd` applies to integer-backed dtypes; `sum_len` / `sum_len_unique` to String, Categorical, Enum and Binary; both are null elsewhere — Spec B §3.)

- [ ] **Step 4: Add the helpers** — append to `analytics/describe/_values.py`:

```python
def byte_lengths(s: pl.Series) -> pl.Series | None:
    """UTF-8 / binary byte length of each value (String, Categorical, Enum, Binary);
    None for every other dtype."""
    if isinstance(s.dtype, STRING_LIKE):
        return s.cast(pl.String).str.len_bytes()
    if s.dtype == pl.Binary:
        return s.bin.size()
    return None


def sig_digits(numeric: pl.Series) -> pl.Series:
    """Significant digits of numeric strings: leading zeros (across the dot) and
    trailing fraction zeros removed ("0.00120" → 2, "1200" → 4, "-0.0" → 0)."""
    int_part = numeric.str.extract(INT_DIGITS, 1)
    frac = numeric.str.extract(FRAC_DIGITS, 1).fill_null("")
    return (int_part + frac).str.strip_chars_start("0").str.len_bytes()
```

`INT_DIGITS` and `FRAC_DIGITS` are defined further down the same module; move these two functions below the `# ── string grammar` block so the names exist at import time.

- [ ] **Step 5: Compute them in the reference** — in `analytics/describe/polars.py`:

Add imports:

```python
from analytics.describe._values import byte_lengths, sig_digits
from analytics.gcd.base import INTEGER_BACKED
from analytics.gcd.math import math_gcd
```

Change `profile` to include the totals:

```python
def profile(s: pl.Series, seed: int) -> dict:
    """Every VALUE_METRICS entry for one series (outer column or flattened inner values)."""
    freq = frequencies(s, seed)
    summary = frequency_summary(freq["count"].to_numpy(), freq["first"].to_numpy(), freq["mask"].to_numpy(), s.len(), s.null_count())
    return {**summary, **extremes(s, freq), **lengths(s), **totals(s, freq), **float_stats(s), **string_stats(s)}


def totals(s: pl.Series, freq: pl.DataFrame) -> dict:
    """gcd of the physical values (as the Gcd technique) and the byte totals of all /
    distinct string or binary values."""
    lens = byte_lengths(s)
    return {
        "gcd": math_gcd(s) if isinstance(s.dtype, INTEGER_BACKED) else None,
        "sum_len": None if lens is None else int(lens.sum()),
        "sum_len_unique": None if lens is None else int(byte_lengths(freq["v"]).sum()),
    }
```

In `string_stats`, add three entries to the returned dict:

```python
        "numeric_min_frac_digits": numeric.str.extract(FRAC_DIGITS, 1).str.len_bytes().fill_null(0).min(),
        "numeric_max_sig_digits": sig_digits(numeric).max(),
        "iso_max_sig_frac_digits": timed.str.extract(ISO_FRACTION, 1).str.strip_chars_end("0").str.len_bytes().fill_null(0).max(),
```

- [ ] **Step 6: Run the new tests for the reference**

Run: `$PY -m pytest tests/test_describe.py -q -k "DescribePolars"`
Expected: PASS for every `DescribePolars` test (including contract and the four new ones).

- [ ] **Step 7: Commit**

```bash
git add services/analytics/analytics/describe/base.py services/analytics/analytics/describe/_values.py services/analytics/analytics/describe/polars.py tests/test_describe.py
git commit -m "feat: describe gcd, sum_len, sum_len_unique, numeric/iso significant digits (reference)"
```

---

### Task 4: New Describe metrics — Rust

**Files:**
- Modify: `services/analytics/src/describe.rs`

- [ ] **Step 1: Write the failing Rust tests** — append inside `mod tests`:

```rust
    #[test]
    fn significant_digits() {
        let sig = |s: &str| scan_numeric(s.as_bytes()).map(|n| (n.frac_digits, n.sig_digits));
        assert_eq!(sig("1.50"), Some((1, 2)));
        assert_eq!(sig("0.00120"), Some((4, 2)));
        assert_eq!(sig("1200"), Some((0, 4)));
        assert_eq!(sig("-0.0"), Some((0, 0)));
        assert_eq!(sig("12.50"), Some((1, 3)));
    }

    #[test]
    fn iso_significant_fraction() {
        assert_eq!(iso("10:00:00.120"), Some(Iso::Time { frac: 3, sig: 2 }));
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, sig: 0, midnight: true }));
    }

    #[test]
    fn sum_len_unique_counts_each_distinct_value_once() {
        let s = Series::new("a".into(), &[Some("ab"), Some("ab"), Some("c"), None]);
        let lens = byte_lengths(&s).unwrap().unwrap();
        assert_eq!(lens.iter().sum::<u64>(), 5);
        assert_eq!(frequencies(&encode_series(&s).unwrap(), 0, Some(&lens)).sum_len_unique, Some(3));
    }

    #[test]
    fn profile_gcd_and_lengths() {
        let p = profile(&Series::new("x".into(), &[Some(10i64), None, Some(30)]), 0).unwrap();
        assert_eq!((p.gcd, p.sum_len), (Some(10), None));
        let s = profile(&Series::new("x".into(), &["ab", "ab", "c"]), 0).unwrap();
        assert_eq!((s.gcd, s.sum_len, s.freq.sum_len_unique), (None, Some(5), Some(3)));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib describe::`
Expected: FAIL to compile (`sig_digits`, `sig`, `byte_lengths`, three-argument `frequencies` do not exist).

- [ ] **Step 3: Scanner changes** — replace `Numeric` and `scan_numeric`:

```rust
pub(crate) struct Numeric {
    pub is_int: bool,
    /// Integer-looking with a leading zero ("007", "-012"; not "0" or "-0").
    pub leading_zero: bool,
    /// Significant integer-part digits (leading zeros ignored).
    pub int_digits: u32,
    /// Fraction digits with trailing zeros removed.
    pub frac_digits: u32,
    /// Significant digits of the whole value: leading zeros (across the dot) and
    /// trailing fraction zeros removed ("0.00120" → 2, "1200" → 4).
    pub sig_digits: u32,
}

pub(crate) fn scan_numeric(b: &[u8]) -> Option<Numeric> {
    let body = b.strip_prefix(b"-").unwrap_or(b);
    let int_len = body.iter().take_while(|c| c.is_ascii_digit()).count();
    if int_len == 0 {
        return None;
    }
    let (int, rest) = body.split_at(int_len);
    let frac = match rest {
        [] => &b""[..],
        [b'.', frac @ ..] if !frac.is_empty() && frac.iter().all(u8::is_ascii_digit) => frac,
        _ => return None,
    };
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
    let significant = int.iter().position(|&c| c != b'0').map_or(0, |p| int.len() - p);
    let sig_digits = if significant > 0 {
        significant + frac.len()
    } else {
        frac.iter().position(|&c| c != b'0').map_or(0, |p| frac.len() - p)
    };
    Some(Numeric {
        is_int: rest.is_empty(),
        leading_zero: rest.is_empty() && int.len() > 1 && int[0] == b'0',
        int_digits: significant as u32,
        frac_digits: frac.len() as u32,
        sig_digits: sig_digits as u32,
    })
}
```

Replace `Iso`, `time` and `scan_iso`:

```rust
#[derive(Debug, PartialEq)]
pub(crate) enum Iso {
    Date,
    /// frac = fractional-second digits as written; sig = without trailing zeros.
    Time { frac: u32, sig: u32 },
    DateTime { frac: u32, sig: u32, midnight: bool },
    DateTimeTz { frac: u32, sig: u32, midnight: bool, offset_minutes: i32 },
}

/// `HH:MM[:SS[.f{1,9}]]` from `i`: (end, fraction digits as written, without
/// trailing zeros, all-zero time).
fn time(b: &[u8], i: usize) -> Option<(usize, u32, u32, bool)> {
    let (h, m) = (two(b, i)?, two(b, i + 3)?);
    if b.get(i + 2) != Some(&b':') || h > 23 || m > 59 {
        return None;
    }
    let (mut end, mut frac, mut sig, mut zero) = (i + 5, 0, 0, h == 0 && m == 0);
    if b.get(end) == Some(&b':') {
        let s = two(b, end + 1)?;
        if s > 59 {
            return None;
        }
        zero &= s == 0;
        end += 3;
        if b.get(end) == Some(&b'.') {
            let digits = b[end + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            if !(1..=9).contains(&digits) {
                return None;
            }
            let f = &b[end + 1..end + 1 + digits];
            zero &= f.iter().all(|&c| c == b'0');
            frac = digits as u32;
            sig = f.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1) as u32;
            end += 1 + digits;
        }
    }
    Some((end, frac, sig, zero))
}

pub(crate) fn scan_iso(b: &[u8]) -> Option<Iso> {
    if let Some(d) = date(b) {
        if b.len() == d {
            return Some(Iso::Date);
        }
        if !matches!(b[d], b'T' | b' ') {
            return None;
        }
        let (end, frac, sig, midnight) = time(b, d + 1)?;
        if end == b.len() {
            return Some(Iso::DateTime { frac, sig, midnight });
        }
        return offset(b, end).map(|offset_minutes| Iso::DateTimeTz { frac, sig, midnight, offset_minutes });
    }
    let (end, frac, sig, _) = time(b, 0)?;
    (end == b.len()).then_some(Iso::Time { frac, sig })
}
```

Update the existing `iso_grammar` test's expected values to the new variants:

```rust
        assert_eq!(iso("23:59"), Some(Iso::Time { frac: 0, sig: 0 }));
        assert_eq!(iso("10:00:00.123456789"), Some(Iso::Time { frac: 9, sig: 9 }));
        assert_eq!(iso("2024-01-05 10:00:00"), Some(Iso::DateTime { frac: 0, sig: 0, midnight: false }));
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, sig: 0, midnight: true }));
        assert_eq!(iso("2024-01-05T00:00Z"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: true, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-00:00"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: false, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-05:30"), Some(Iso::DateTimeTz { frac: 0, sig: 0, midnight: false, offset_minutes: -330 }));
```

- [ ] **Step 4: StringStats** — add three fields and maintain them:

```rust
    pub min_frac_digits: Option<u32>,
    pub max_sig_digits: Option<u32>,
    pub iso_max_sig_frac_digits: Option<u32>,
```

In `add`, inside `if let Some(n) = scan_numeric(b)` after the `max_frac_digits` line:

```rust
            self.min_frac_digits = opt_min(self.min_frac_digits, Some(n.frac_digits));
            self.max_sig_digits = self.max_sig_digits.max(Some(n.sig_digits));
```

and replace the ISO `match` with:

```rust
        match scan_iso(b) {
            Some(Iso::Date) => self.n_iso_date += 1,
            Some(Iso::Time { frac, sig }) => {
                self.n_iso_time += 1;
                self.fraction(frac, sig);
            }
            Some(Iso::DateTime { frac, sig, midnight }) => {
                self.n_iso_datetime += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
            }
            Some(Iso::DateTimeTz { frac, sig, midnight, offset_minutes }) => {
                self.n_iso_datetime_tz += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
            }
            None => {}
        }
```

with the helper (inside `impl StringStats`):

```rust
    fn fraction(&mut self, frac: u32, sig: u32) {
        self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
        self.iso_max_sig_frac_digits = self.iso_max_sig_frac_digits.max(Some(sig));
    }
```

In `merge`, add:

```rust
        self.min_frac_digits = opt_min(self.min_frac_digits, o.min_frac_digits);
        self.max_sig_digits = self.max_sig_digits.max(o.max_sig_digits);
        self.iso_max_sig_frac_digits = self.iso_max_sig_frac_digits.max(o.iso_max_sig_frac_digits);
```

- [ ] **Step 5: sum_len_unique in the frequency sweep** — change `Frequencies` and `frequencies`:

```rust
pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub entropy: f64,
    pub f1: u64,
    pub f2: u64,
    pub top5_idx: Vec<u64>,
    pub top5_count: Vec<u64>,
    pub capture_history: [u64; 7],
    /// Total byte length of the distinct values (`lengths` given: string / binary columns).
    pub sum_len_unique: Option<u64>,
}
```

Signature `pub(crate) fn frequencies(col: &EncodedColumn, seed: u64, lengths: Option<&[u64]>) -> Frequencies`; before the sweep add `let mut unique_len = 0u64;`, inside the `for e in map.into_values()` loop add `if let Some(l) = lengths { unique_len += l[e.first as usize]; }`, and set `sum_len_unique: lengths.map(|_| unique_len)` in the returned struct. Update the test helper `freq` to call `frequencies(&encode_series(&s).unwrap(), 0, None)`.

- [ ] **Step 6: Profile fields** — add `byte_lengths` and extend `Profile`:

```rust
/// Byte length of every value (0 for nulls) of a String, Categorical, Enum or Binary series.
fn byte_lengths(s: &Series) -> PolarsResult<Option<Vec<u64>>> {
    Ok(match s.dtype() {
        DataType::String => Some(s.str()?.iter().map(|v| v.map_or(0, |x| x.len() as u64)).collect()),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => return byte_lengths(&s.cast(&DataType::String)?),
        DataType::Binary => Some(s.binary()?.iter().map(|v| v.map_or(0, |x| x.len() as u64)).collect()),
        _ => None,
    })
}
```

Add to `Profile`:

```rust
    /// GCD of the physical values (gcd.rs); None for non-integer dtypes or > 38 digits.
    pub gcd: Option<i128>,
    /// Total byte length of the non-null values (string / binary columns).
    pub sum_len: Option<u64>,
```

and build them in `profile`:

```rust
pub(crate) fn profile(s: &Series, seed: u64) -> PolarsResult<Profile> {
    let lengths = byte_lengths(s)?;
    Ok(Profile {
        freq: frequencies(&encode_series(s)?, seed, lengths.as_deref()),
        range: range(s)?,
        floats: float_stats(s)?,
        strings: strings(s)?,
        gcd: crate::gcd::series_gcd(s)?,
        sum_len: lengths.map(|l| l.iter().sum()),
        is_f32: s.dtype() == &DataType::Float32,
    })
}
```

- [ ] **Step 7: Output order** — in `value_fields()` make group A read

```rust
        ("n_unique", U64), ("entropy", F64), ("f1", U64), ("f2", U64), ("argmin", U64), ("argmax", U64),
        ("min_len", U64), ("max_len", U64), ("gcd", d38.clone()), ("sum_len", U64), ("sum_len_unique", U64),
        ("top5_idx", list.clone()), ("top5_count", list.clone()), ("capture_history", list),
```

and group C read

```rust
        ("n_numeric", U64), ("n_numeric_int", U64), ("n_leading_zero", U64), ("numeric_int_min", d38.clone()), ("numeric_int_max", d38),
        ("numeric_max_int_digits", U32), ("numeric_max_frac_digits", U32), ("numeric_min_frac_digits", U32), ("numeric_max_sig_digits", U32),
        ("n_iso_date", U64), ("n_iso_time", U64), ("n_iso_datetime", U64), ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32), ("iso_max_sig_frac_digits", U32), ("iso_n_offsets", U64), ("iso_n_midnight", U64),
```

In `Profile::row`, after `u64v(r.max_len)` insert `d38v(self.gcd), u64v(self.sum_len), u64v(f.sum_len_unique),`; in the strings branch insert `u32v(st.min_frac_digits), u32v(st.max_sig_digits),` after `u32v(st.max_frac_digits)` and `u32v(st.iso_max_sig_frac_digits),` after `u32v(st.iso_max_frac_digits)`; change `nulls(14)` to `nulls(17)`.

- [ ] **Step 8: Run the Rust tests**

Run: `cargo test --lib describe::`
Expected: PASS (23 tests).

- [ ] **Step 9: Build and run the Python tests for Rust and the reference**

Run: build; `$PY -m pytest tests/test_describe.py -q -m "not slow" -k "not DataFusion"`
Expected: PASS, including `test_agrees_with_reference[DescribeRust]`.

- [ ] **Step 10: Commit**

```bash
git add services/analytics/src/describe.rs
git commit -m "feat: describe gcd, sum_len, sum_len_unique, significant digits (Rust)"
```

---

### Task 5: New Describe metrics — DataFusion

**Files:**
- Modify: `services/analytics/analytics/describe/datafusion.py`

- [ ] **Step 1: Confirm the failure**

Run: `$PY -m pytest tests/test_describe.py -q -m "not slow" -k DataFusion`
Expected: FAIL (`KeyError: 'gcd'` in `metrics_frame`).

- [ ] **Step 2: Add the totals** — imports:

```python
from analytics.gcd.base import INTEGER_BACKED
from analytics.gcd.math import math_gcd
```

New function:

```python
def _totals(ctx: SessionContext, s: pl.Series) -> dict:
    out = {"gcd": math_gcd(s) if isinstance(s.dtype, INTEGER_BACKED) else None, "sum_len": None, "sum_len_unique": None}
    if s.dtype == pl.Binary:  # octet_length() accepts only strings in DataFusion SQL
        arr = _arrow(s)
        out["sum_len"] = pc.sum(pc.binary_length(arr)).as_py() or 0
        out["sum_len_unique"] = pc.sum(pc.binary_length(pc.unique(arr.drop_null()))).as_py() or 0
    elif isinstance(s.dtype, STRING_LIKE):
        r = _one(
            ctx,
            "SELECT SUM(octet_length(v)) AS total, "
            "(SELECT SUM(octet_length(u.v)) FROM (SELECT DISTINCT v FROM t WHERE v IS NOT NULL) u) AS uniq FROM t",
        )
        out["sum_len"], out["sum_len_unique"] = r["total"] or 0, r["uniq"] or 0
    return out
```

In `_profile`, add `**_totals(ctx, s)` after `**_lengths(ctx, s)`. In the class docstring's "Computed outside SQL" list add:

```
      - gcd: Python math.gcd over the physical values (as GcdMath; SQL has no exact GCD aggregate);
      - sum_len / sum_len_unique of Binary: pyarrow binary_length;
```

- [ ] **Step 3: String metrics in SQL** — in `_strings`, add to the SELECT list:

```sql
              MIN(CASE WHEN num THEN COALESCE(length(regexp_match(v, '{FRAC_DIGITS}')[1]), 0) END) AS min_frac,
              MAX(CASE WHEN num THEN length(ltrim(concat(COALESCE(regexp_match(v, '{INT_DIGITS}')[1], ''), COALESCE(regexp_match(v, '{FRAC_DIGITS}')[1], '')), '0')) END) AS sig,
              MAX(CASE WHEN tm OR ((dt OR tz) AND dok) THEN COALESCE(length(rtrim(regexp_match(v, '{ISO_FRACTION}')[1], '0')), 0) END) AS iso_sig,
```

and to the returned dict:

```python
        "numeric_min_frac_digits": r["min_frac"],
        "numeric_max_sig_digits": r["sig"],
        "iso_max_sig_frac_digits": r["iso_sig"],
```

- [ ] **Step 4: Run the whole describe suite**

Run: `$PY -m pytest tests/test_describe.py -q` (includes `slow`)
Expected: PASS for all three implementations.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe/datafusion.py
git commit -m "feat: describe new metrics in DescribeDataFusion"
```

---

### Task 6: Redefine `size_polars_bytes` as the uncompressed native-layout IPC body

**Files:**
- Modify: `services/analytics/analytics/describe/_sizes.py`, `analytics/describe/rust.py`, `src/sizes.rs`
- Test: `tests/test_describe.py`

- [ ] **Step 1: Write the failing tests** — in `tests/test_describe.py` add after `test_column_sizes_arrow_and_polars`:

```python
def test_polars_size_is_the_native_ipc_body():
    assert _sizes.column_sizes(pl.Series("x", ["ab", None]), 1)["size_polars_bytes"] == 40  # validity 8 + 2 views × 16
    assert _sizes.column_sizes(pl.Series("x", ["a" * 20, "b"]), 1)["size_polars_bytes"] == 56  # views 32 + 20-byte buffer → 24
```

and inside `test_sizes_through_the_technique` add:

```python
    assert profile(impl, pl.Series("x", ["ab", None]))["size_polars_bytes"] == 40
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_describe.py -q -k "native_ipc_body or sizes_through"`
Expected: FAIL (`2 != 40` — the estimated size).

- [ ] **Step 3: Python oracle** — in `_sizes.column_sizes` replace `"size_polars_bytes": s.estimated_size(),` with `"size_polars_bytes": ipc_body_bytes(native, None),`, and in the module docstring replace "Polars sizes are `estimated_size()` and the ZSTD body of its native layout" with "Polars sizes are the IPC body of its native layout, plain and ZSTD".

- [ ] **Step 4: Rust** — in `src/sizes.rs` function `sizes`, replace the `polars_bytes` line and the zero-chunk early return and third element so it reads:

```rust
    let s = s.rechunk();
    if s.n_chunks() == 0 {
        return Ok([Some(0); 4]);
    }
    let classic = s.to_arrow(0, CompatLevel::oldest());
    let native = s.to_arrow(0, CompatLevel::newest());
    Ok([
        Some(ipc_body_bytes(classic.as_ref(), None)?),
        Some(ipc_body_bytes(classic.as_ref(), Some(level))?),
        Some(ipc_body_bytes(native.as_ref(), None)?),
        Some(ipc_body_bytes(native.as_ref(), Some(level))?),
    ])
```

Update the header comment: "Polars sizes are the plain and ZSTD body of CompatLevel::newest() (view types)."

- [ ] **Step 5: Remove the Python override** — `analytics/describe/rust.py` becomes:

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.describe.base import Describe


class DescribeRust(Describe):
    """Rust plugin: `describe_columns` (one pass per column; rayon across columns and
    64K-row chunks; hash-map frequencies, byte scanners, row-encoded extremes) and
    `column_sizes` (Arrow buffer walk + zstd, mirroring pyarrow's IPC writer)."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            stats = _plugin.describe_columns(df, self.seed).join(_plugin.column_sizes(df, self.zstd_level), on="column")
            for r in stats.iter_rows(named=True):
                rows[frame, r["column"]] = r
        return self.metrics_frame(combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS})
```

Also update `_plugin.column_sizes`'s docstring: "Polars sizes (IPC body of the native layout, plain and ZSTD)".

- [ ] **Step 6: Build and run**

Run: build; `$PY -m pytest tests/test_describe.py -q -m "not slow"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/analytics/describe/_sizes.py services/analytics/analytics/describe/rust.py services/analytics/analytics/_plugin.py services/analytics/src/sizes.rs tests/test_describe.py
git commit -m "feat: size_polars_bytes is the uncompressed native-layout IPC body"
```

---

### Task 7: arrow-rs dependencies and the zero-copy adapter

**Files:**
- Modify: `services/analytics/Cargo.toml`, `services/analytics/src/shared.rs`

- [ ] **Step 1: Write the failing test** — append to `src/shared.rs`:

```rust
#[cfg(test)]
mod arrow_rs_tests {
    use super::*;
    use arrow_array::Array as _;
    use arrow_schema::DataType as AT;

    #[test]
    fn series_cross_to_arrow_rs() {
        let a = to_arrow_rs(&Series::new("x".into(), &[Some(1i32), None]), CompatLevel::oldest()).unwrap();
        assert_eq!((a.len(), a.null_count(), a.data_type().clone()), (2, 1, AT::Int32));
        assert_eq!(to_arrow_rs(&Series::new("x".into(), &["a"]), CompatLevel::oldest()).unwrap().data_type(), &AT::LargeUtf8);
        assert_eq!(to_arrow_rs(&Series::new("x".into(), &["a"]), CompatLevel::newest()).unwrap().data_type(), &AT::Utf8View);
        assert_eq!(to_arrow_rs(&Series::new("x".into(), &[1i128]), CompatLevel::oldest()).unwrap().data_type(), &AT::Decimal128(38, 0));
        assert_eq!(to_arrow_rs(&Series::new_empty("x".into(), &DataType::String), CompatLevel::oldest()).unwrap().len(), 0);
        let sliced = to_arrow_rs(&Series::new("x".into(), &[1i64, 2, 3]).slice(1, 2), CompatLevel::oldest()).unwrap();
        let ints = sliced.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        assert_eq!(ints.values().to_vec(), vec![2, 3]);
    }
}
```

- [ ] **Step 2: Add the dependencies** — in `Cargo.toml` after the `bytemuck` line:

```toml
#Arrow-native core (recommend.rs, sizes.rs): arrow-rs, fed zero-copy through the C Data Interface
arrow-array = { version = "60", features = ["ffi"] }
arrow-buffer = "60"
arrow-cast = "60"
arrow-data = { version = "60", features = ["ffi"] }
arrow-schema = { version = "60", features = ["ffi"] }
arrow-select = "60"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib shared::arrow_rs_tests`
Expected: FAIL to compile (`to_arrow_rs` not found). If Cargo reports a missing `ffi` feature on a crate, drop that crate's `features` entry and retry.

- [ ] **Step 4: Implement the adapter** — append to `src/shared.rs` (above the test module):

```rust
/// A Series handed to arrow-rs zero-copy through the Arrow C Data Interface, in the
/// given Polars layout (oldest: LargeUtf8/LargeList — Arrow's classic layout;
/// newest: Utf8View/BinaryView — Polars' native one). Int128, which arrow-rs cannot
/// import, crosses as decimal128(38, 0) over the same 16-byte values (as pyarrow
/// receives it in analytics/describe/_sizes.py).
pub(crate) fn to_arrow_rs(s: &Series, compat: CompatLevel) -> PolarsResult<arrow_array::ArrayRef> {
    let s = match s.dtype() {
        DataType::Int128 => s.i128()?.clone().into_decimal_unchecked(Some(38), 0).into_series(),
        _ => s.rechunk(),
    };
    let arr: Box<dyn polars_arrow::array::Array> = if s.n_chunks() == 0 {
        polars_arrow::array::new_empty_array(s.dtype().to_arrow(compat))
    } else {
        s.rechunk().to_arrow(0, compat)
    };
    let field = polars_arrow::datatypes::Field::new(s.name().clone(), arr.dtype().clone(), true);
    let schema = polars_arrow::ffi::export_field_to_c(&field);
    let array = polars_arrow::ffi::export_array_to_c(arr);
    // SAFETY: both pairs are the #[repr(C)] ArrowSchema / ArrowArray structs of the
    // Arrow C Data Interface; ownership (their release callbacks) moves to arrow-rs.
    let (array, schema): (arrow_data::ffi::FFI_ArrowArray, arrow_schema::ffi::FFI_ArrowSchema) =
        unsafe { (std::mem::transmute(array), std::mem::transmute(schema)) };
    let data = unsafe { arrow_array::ffi::from_ffi(array, &schema) }
        .map_err(|e| polars_err!(ComputeError: "arrow C data interface: {e}"))?;
    Ok(arrow_array::make_array(data))
}
```

(`shared.rs` already imports `polars::prelude::*`; if `into_decimal_unchecked` or `to_arrow` need a trait import, the compiler names it.)

- [ ] **Step 5: Run the test**

Run: `cargo test --lib shared::arrow_rs_tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/Cargo.toml services/analytics/Cargo.lock services/analytics/src/shared.rs
git commit -m "feat: arrow-rs dependencies and zero-copy Series → arrow-rs adapter"
```

---

### Task 8: Port `sizes.rs` to arrow-rs

**Files:**
- Modify: `services/analytics/src/sizes.rs`

- [ ] **Step 1: Point the tests at arrow-rs** — in `sizes.rs`'s test module replace the `arrow` helper with:

```rust
    fn arrow(s: &Series) -> arrow_array::ArrayRef {
        crate::shared::to_arrow_rs(s, CompatLevel::oldest()).unwrap()
    }
```

and add:

```rust
    #[test]
    fn polars_size_is_native_ipc_body() {
        assert_eq!(sizes(&Series::new("x".into(), &[Some("ab"), None]), 1).unwrap()[2], Some(40));
    }

    #[test]
    fn dictionary_keys_and_values() {
        let cat = Series::new("x".into(), &["a", "b", "a"]).cast(&DataType::from_categories(Categories::global())).unwrap();
        assert_eq!(sizes(&cat, 1).unwrap()[0], Some(48)); // pyarrow: keys 16 + dictionary 32 (see test_describe.py)
    }
```

If `DataType::from_categories(Categories::global())` does not exist in polars 0.51, build the Categorical the way other tests in the crate do (`grep -rn "Categorical" services/analytics/src`).

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib sizes::`
Expected: FAIL to compile (`ipc_body_bytes` takes a polars-arrow array).

- [ ] **Step 3: Replace everything above `nests_int128` in `sizes.rs`** with:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// sizes — Arrow IPC body sizes (Describe group E; recast sizes for recommend.rs)
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors pyarrow's IPC writer (checked against pyarrow 24; the Python oracle is
// analytics/describe/_sizes.py): a column's size is the body length of the IPC
// messages that carry it (dictionary batches + record batch). Every buffer is
// padded to 8 bytes; a validity buffer is written only when the array has nulls;
// an empty buffer takes no space; List/Utf8 offsets are rebased to 0 and their
// child/values sliced to the referenced range. With ZSTD each non-empty buffer is
// an 8-byte uncompressed-length prefix plus one ZSTD frame (content size
// included) — pyarrow never falls back to raw bytes. arrow-rs's own IPC writer is
// not used because it does not expose the ZSTD level.
//
// Works on arrow-rs ArrayData; Series arrive through shared::to_arrow_rs. Arrow
// sizes use CompatLevel::oldest() (LargeUtf8, LargeList); Polars sizes the plain
// and ZSTD body of CompatLevel::newest() (view types). Columns nesting Int128
// inside List/Array/Struct get null sizes: pyarrow cannot import them, so there
// is no oracle to agree with.

use crate::shared::to_arrow_rs;
use arrow_array::Array;
use arrow_buffer::{ArrowNativeType, BooleanBuffer, ToByteSlice};
use arrow_data::ArrayData;
use arrow_schema::DataType as AT;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

struct Body {
    level: Option<i32>,
    bytes: u64,
}

impl Body {
    fn buffer(&mut self, data: &[u8]) -> PolarsResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let len = match self.level {
            None => data.len(),
            Some(level) => 8 + zstd::bulk::compress(data, level).map_err(|e| polars_err!(ComputeError: "zstd: {e}"))?.len(),
        };
        self.bytes += len.next_multiple_of(8) as u64;
        Ok(())
    }

    fn bits(&mut self, b: &BooleanBuffer) -> PolarsResult<()> {
        let packed = b.sliced(); // re-aligned to bit 0, as pyarrow writes it
        self.buffer(&packed.as_slice()[..b.len().div_ceil(8)])
    }

    fn validity(&mut self, d: &ArrayData) -> PolarsResult<()> {
        match d.nulls() {
            Some(n) if n.null_count() > 0 => self.bits(n.inner()),
            _ => Ok(()),
        }
    }

    fn fixed(&mut self, d: &ArrayData, width: usize) -> PolarsResult<()> {
        self.validity(d)?;
        let start = d.offset() * width;
        self.buffer(&d.buffers()[0].as_slice()[start..start + d.len() * width])
    }

    /// Writes the offsets (rebased to 0); returns the referenced child/values range.
    fn offsets<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<(usize, usize)> {
        if d.len() == 0 {
            self.buffer(&vec![0u8; std::mem::size_of::<O>()])?; // pyarrow writes the single offset [0]
            return Ok((0, 0));
        }
        let o = &d.buffers()[0].typed_data::<O>()[d.offset()..=d.offset() + d.len()];
        let (first, last) = (o[0].as_usize(), o[d.len()].as_usize());
        if first == 0 {
            self.buffer(o.to_byte_slice())?;
        } else {
            let rebased: Vec<O> = o.iter().map(|x| O::usize_as(x.as_usize() - first)).collect();
            self.buffer(rebased.as_slice().to_byte_slice())?;
        }
        Ok((first, last))
    }

    fn var_size<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<()> {
        self.validity(d)?;
        let (first, last) = self.offsets::<O>(d)?;
        self.buffer(&d.buffers()[1].as_slice()[first..last])
    }

    fn list<O: ArrowNativeType>(&mut self, d: &ArrayData) -> PolarsResult<()> {
        self.validity(d)?;
        let (first, last) = self.offsets::<O>(d)?;
        self.array(&d.child_data()[0].slice(first, last - first))
    }

    fn array(&mut self, d: &ArrayData) -> PolarsResult<()> {
        match d.data_type() {
            AT::Null => Ok(()),
            AT::Boolean => {
                self.validity(d)?;
                self.bits(&BooleanBuffer::new(d.buffers()[0].clone(), d.offset(), d.len()))
            }
            AT::Utf8 | AT::Binary => self.var_size::<i32>(d),
            AT::LargeUtf8 | AT::LargeBinary => self.var_size::<i64>(d),
            AT::Utf8View | AT::BinaryView => {
                self.validity(d)?;
                self.buffer(&d.buffers()[0].as_slice()[d.offset() * 16..(d.offset() + d.len()) * 16])?;
                d.buffers()[1..].iter().try_for_each(|b| self.buffer(b.as_slice()))
            }
            AT::List(_) => self.list::<i32>(d),
            AT::LargeList(_) => self.list::<i64>(d),
            AT::FixedSizeList(_, w) => {
                self.validity(d)?;
                let w = *w as usize;
                self.array(&d.child_data()[0].slice(d.offset() * w, d.len() * w))
            }
            AT::Struct(_) => {
                self.validity(d)?;
                d.child_data().iter().try_for_each(|c| self.array(&c.slice(d.offset(), d.len())))
            }
            AT::Dictionary(k, _) => {
                self.fixed(d, k.primitive_width().expect("integer dictionary key"))?; // the keys
                self.array(&d.child_data()[0]) // the dictionary batch
            }
            t => match t.primitive_width() {
                Some(w) => self.fixed(d, w),
                None => polars_bail!(ComputeError: "sizes: unsupported Arrow type {t}"),
            },
        }
    }
}

/// Arrow IPC body bytes of `arr`: uncompressed (`level` None) or ZSTD at `level`.
pub(crate) fn ipc_body_bytes(arr: &dyn Array, level: Option<i32>) -> PolarsResult<u64> {
    let mut body = Body { level, bytes: 0 };
    body.array(&arr.to_data())?;
    Ok(body.bytes)
}
```

and replace `Sizes`, `SIZE_FIELDS` and `sizes` with:

```rust
pub(crate) type Sizes = [Option<u64>; 4];
pub(crate) const SIZE_FIELDS: [&str; 4] = ["size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes"];

pub(crate) fn sizes(s: &Series, level: i32) -> PolarsResult<Sizes> {
    if nests_int128(s.dtype()) {
        return Ok([None; 4]);
    }
    let classic = to_arrow_rs(s, CompatLevel::oldest())?;
    let native = to_arrow_rs(s, CompatLevel::newest())?;
    Ok([
        Some(ipc_body_bytes(classic.as_ref(), None)?),
        Some(ipc_body_bytes(classic.as_ref(), Some(level))?),
        Some(ipc_body_bytes(native.as_ref(), None)?),
        Some(ipc_body_bytes(native.as_ref(), Some(level))?),
    ])
}
```

Remove the old `use bytemuck::Pod;` and polars-arrow imports; drop `bytemuck` from Cargo.toml only if nothing else uses it (`grep -rn bytemuck services/analytics/src`).

- [ ] **Step 4: Run the Rust tests**

Run: `cargo test --lib sizes::`
Expected: PASS (6 tests).

- [ ] **Step 5: Build and run the size oracles**

Run: build; `$PY -m pytest tests/test_describe.py -q` (includes `slow`)
Expected: PASS — `DescribeRust` sizes still agree with pyarrow.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/src/sizes.rs services/analytics/Cargo.toml services/analytics/Cargo.lock
git commit -m "refactor: sizes.rs walks arrow-rs ArrayData"
```

---

### Task 9: `cardinality_estimators.rs`

Ports of `analytics/describe/estimators.py`; the expected values are the same hand-worked numbers `tests/test_describe.py` asserts for the Python module.

**Files:**
- Create: `services/analytics/src/cardinality_estimators.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod cardinality_estimators;`)

- [ ] **Step 1: Write the file with its tests first**

```rust
// ─────────────────────────────────────────────────────────────────────────────
// cardinality_estimators — number of distinct values in a population
// ─────────────────────────────────────────────────────────────────────────────
//
// Ports of analytics/describe/estimators.py (the Python reference that feeds
// Describe's conclusions); recommend.rs uses them to size dictionaries. Picked by
// rule, never averaged. 95% closed-form intervals (z = 1.96).

pub(crate) const Z: f64 = 1.96;

/// Bias-corrected Chao1 with Chao's (1987) log-normal interval; variance as in the
/// EstimateS user guide. (estimate, low, high).
pub(crate) fn chao1(d: u64, f1: u64, f2: u64) -> (f64, f64, f64) {
    let (d, f1, f2) = (d as f64, f1 as f64, f2 as f64);
    let s = d + f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0));
    let t = s - d;
    if t <= 0.0 {
        return (s, d, d);
    }
    let var = if f2 > 0.0 {
        f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
            + f1 * (2.0 * f1 - 1.0).powi(2) / (4.0 * (f2 + 1.0).powi(2))
            + f1.powi(2) * f2 * (f1 - 1.0).powi(2) / (4.0 * (f2 + 1.0).powi(4))
    } else {
        f1 * (f1 - 1.0) / 2.0 + f1 * (2.0 * f1 - 1.0).powi(2) / 4.0 - f1.powi(4) / (4.0 * s)
    };
    let k = (Z * (1.0 + var.max(0.0) / (t * t)).ln().sqrt()).exp();
    (s, d + t / k, d + t * k)
}

/// Schnabel (multi-sample Lincoln–Petersen) over the three split subsets;
/// `history[k-1]` = distinct values whose subset mask is k. None unless n > 0,
/// d/n < 0.5 and there is at least one recapture.
pub(crate) fn schnabel(history: &[u64; 7], d: u64, n: u64) -> Option<(f64, f64, f64)> {
    if n == 0 || d as f64 / n as f64 >= 0.5 {
        return None;
    }
    let h = |masks: &[usize]| masks.iter().map(|&m| history[m - 1]).sum::<u64>() as f64;
    let (s1, s2, s3) = (h(&[1, 3, 5, 7]), h(&[2, 3, 6, 7]), h(&[4, 5, 6, 7]));
    let union12 = h(&[1, 2, 3, 5, 6, 7]);
    let r = h(&[3, 7]) + h(&[5, 6, 7]);
    if r < 1.0 {
        return None;
    }
    let a = s2 * s1 + s3 * union12;
    let r_lo = r * (1.0 - 1.0 / (9.0 * r) - Z / (3.0 * r.sqrt())).powi(3);
    let r_hi = (r + 1.0) * (1.0 - 1.0 / (9.0 * (r + 1.0)) + Z / (3.0 * (r + 1.0).sqrt())).powi(3);
    Some((a / (r + 1.0), a / r_hi, a / r_lo))
}

/// Haas–Stokes Duj1: a sample of n non-null values at sampling fraction q.
pub(crate) fn duj1(d: u64, f1: u64, n: u64, q: f64) -> f64 {
    if n == 0 {
        0.0
    } else {
        d as f64 / (1.0 - (1.0 - q) * f1 as f64 / n as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Method {
    Exact,
    Duj1,
    Schnabel,
    Chao1,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Estimate {
    pub est_cardinality: f64,
    pub est_low: Option<f64>,
    pub est_high: Option<f64>,
    pub method: Method,
}

/// The estimate picked by rule: q == 1 → exact; q < 1 → Duj1; Schnabel valid →
/// Schnabel; else Chao1. q = frame rows / population rows (None: unknown).
pub(crate) fn estimate(d: u64, n: u64, f1: u64, f2: u64, history: &[u64; 7], q: Option<f64>) -> Estimate {
    let (c, c_lo, c_hi) = chao1(d, f1, f2);
    match (q, schnabel(history, d, n)) {
        (Some(q), _) if q == 1.0 => {
            let d = d as f64;
            Estimate { est_cardinality: d, est_low: Some(d), est_high: Some(d), method: Method::Exact }
        }
        (Some(q), _) => Estimate { est_cardinality: duj1(d, f1, n, q), est_low: None, est_high: None, method: Method::Duj1 },
        (None, Some((s, lo, hi))) => Estimate { est_cardinality: s, est_low: Some(lo), est_high: Some(hi), method: Method::Schnabel },
        (None, None) => Estimate { est_cardinality: c, est_low: Some(c_lo), est_high: Some(c_hi), method: Method::Chao1 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
        [(a.0, b.0), (a.1, b.1), (a.2, b.2)].iter().all(|(x, y)| (x - y).abs() <= 1e-9 * y.abs().max(1.0))
    }

    #[test]
    fn chao1_cases() {
        assert!(close(chao1(10, 4, 2), (12.0, 10.249903382590167, 26.00618590489368)));
        assert!(close(chao1(5, 3, 0), (8.0, 5.369121767830802, 29.38219792045809)));
        assert_eq!(chao1(7, 0, 3), (7.0, 7.0, 7.0));
        assert_eq!(chao1(7, 1, 0), (7.0, 7.0, 7.0));
    }

    #[test]
    fn schnabel_cases() {
        assert!(close(schnabel(&[0, 0, 0, 0, 0, 0, 10], 10, 30_000).unwrap(), (9.523809523809524, 6.474567576908709, 16.378255262343956)));
        assert!(close(schnabel(&[5, 5, 5, 2, 2, 2, 1], 22, 1_000).unwrap(), (25.75, 15.69849888622768, 56.349688010901595)));
        assert!(schnabel(&[5, 5, 5, 2, 2, 2, 1], 22, 44).is_none());
        assert!(schnabel(&[3, 3, 0, 3, 0, 0, 0], 9, 100).is_none());
        assert!(schnabel(&[0; 7], 0, 0).is_none());
    }

    #[test]
    fn duj1_cases() {
        assert!((duj1(10, 4, 40, 0.5) - 10.526315789473685).abs() < 1e-12);
        assert_eq!(duj1(0, 0, 0, 0.5), 0.0);
    }

    #[test]
    fn estimate_picks_exact_then_duj1_then_schnabel_then_chao1() {
        let h = [0, 0, 0, 0, 0, 0, 10];
        let exact = estimate(10, 40, 4, 2, &h, Some(1.0));
        assert_eq!((exact.method, exact.est_cardinality, exact.est_high), (Method::Exact, 10.0, Some(10.0)));
        let duj = estimate(10, 40, 4, 2, &h, Some(0.5));
        assert_eq!((duj.method, duj.est_high), (Method::Duj1, None));
        assert!((duj.est_cardinality - 10.526315789473685).abs() < 1e-12);
        assert_eq!(estimate(10, 40, 4, 2, &h, None).method, Method::Schnabel);
        let chao = estimate(10, 15, 4, 2, &h, None);
        assert_eq!((chao.method, chao.est_cardinality), (Method::Chao1, 12.0));
    }
}
```

- [ ] **Step 2: Register the module** — add `mod cardinality_estimators;` to `lib.rs` after `mod sizes;`.

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib cardinality_estimators::`
Expected: PASS (4 tests). (Dead-code warnings are expected until Task 14.)

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/cardinality_estimators.rs services/analytics/src/lib.rs
git commit -m "feat: Rust cardinality estimators (Chao1, Schnabel, Duj1)"
```

---

### Task 10: Exact parsers in `describe.rs` — `parse_iso`, `parse_decimal`

The recommender builds recast arrays from these, never from `arrow-cast`'s string parsing (Spec B §5.4).

**Files:**
- Modify: `services/analytics/src/describe.rs`

- [ ] **Step 1: Write the failing tests** — append inside `mod tests`:

```rust
    #[test]
    fn exact_iso_components() {
        assert_eq!(parse_iso(b"1970-01-02"), Some(IsoValue { days: Some(1), nanos: 0, offset_minutes: None }));
        assert_eq!(parse_iso(b"2024-02-29").unwrap().days, Some(19_782));
        let t = parse_iso(b"10:00:00.12").unwrap();
        assert_eq!((t.days, t.nanos), (None, 36_000_120_000_000));
        let z = parse_iso(b"1970-01-01T05:30+05:30").unwrap();
        assert_eq!((z.offset_minutes, z.epoch_ns()), (Some(330), 0));
        assert_eq!(parse_iso(b"1969-12-31 23:59:59.999999999").unwrap().epoch_ns(), -1);
        assert_eq!(parse_iso(b"2023-02-29"), None);
    }

    #[test]
    fn exact_decimals() {
        assert_eq!(parse_decimal(b"007.50", 2), Some(750));
        assert_eq!(parse_decimal(b"-1.5", 3), Some(-1500));
        assert_eq!(parse_decimal(b"12", 0), Some(12));
        assert_eq!(parse_decimal(b"1.25", 1), None); // needs 2 places
        assert_eq!(parse_decimal(format!("1{}", "0".repeat(38)).as_bytes(), 0), None); // 39 digits
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib describe::tests::exact`
Expected: FAIL to compile.

- [ ] **Step 3: Implement** — add after `scan_iso`:

```rust
/// Components of a value in the scan_iso grammar: days since 1970-01-01 (None for
/// a bare time), nanoseconds since midnight, offset minutes east of UTC.
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct IsoValue {
    pub days: Option<i64>,
    pub nanos: i64,
    pub offset_minutes: Option<i32>,
}

impl IsoValue {
    /// Nanoseconds since the Unix epoch, UTC (a bare time counts from 1970-01-01).
    pub(crate) fn epoch_ns(&self) -> i128 {
        self.days.unwrap_or(0) as i128 * 86_400_000_000_000 + self.nanos as i128
            - self.offset_minutes.unwrap_or(0) as i128 * 60_000_000_000
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's days_from_civil).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Exact components of an ISO value (validated by scan_iso first).
pub(crate) fn parse_iso(b: &[u8]) -> Option<IsoValue> {
    let kind = scan_iso(b)?;
    let num = |i: usize| two(b, i).map(|v| v as i64);
    let date_days = || Some(days_from_civil(num(0)? * 100 + num(2)?, num(5)?, num(8)?));
    let (days, t) = match kind {
        Iso::Date => return Some(IsoValue { days: date_days(), nanos: 0, offset_minutes: None }),
        Iso::Time { .. } => (None, 0),
        _ => (date_days(), 11),
    };
    let mut nanos = (num(t)? * 60 + num(t + 3)?) * 60 * 1_000_000_000;
    let end = t + 5;
    if b.get(end) == Some(&b':') {
        nanos += num(end + 1)? * 1_000_000_000;
        if b.get(end + 3) == Some(&b'.') {
            let frac: Vec<u8> = b[end + 4..].iter().take_while(|c| c.is_ascii_digit()).copied().collect();
            let f = frac.iter().fold(0i64, |a, &c| a * 10 + (c - b'0') as i64);
            nanos += f * 10i64.pow(9 - frac.len() as u32);
        }
    }
    let offset_minutes = match kind {
        Iso::DateTimeTz { offset_minutes, .. } => Some(offset_minutes),
        _ => None,
    };
    Some(IsoValue { days, nanos, offset_minutes })
}

/// Exact unscaled value of a numeric string (scan_numeric grammar) at `scale`:
/// None when it needs more decimal places or more than 38 digits.
pub(crate) fn parse_decimal(b: &[u8], scale: u32) -> Option<i128> {
    scan_numeric(b)?;
    let (neg, body) = match b.strip_prefix(b"-") {
        Some(r) => (true, r),
        None => (false, b),
    };
    let (int, frac) = match body.iter().position(|&c| c == b'.') {
        Some(p) => (&body[..p], &body[p + 1..]),
        None => (body, &b""[..]),
    };
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
    if frac.len() > scale as usize {
        return None;
    }
    let mut v: i128 = 0;
    for &c in int.iter().chain(frac) {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow(scale - frac.len() as u32)?)?;
    (v < 10i128.pow(38)).then_some(if neg { -v } else { v })
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib describe::`
Expected: PASS (25 tests).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe.rs
git commit -m "feat: exact ISO 8601 and decimal parsers in describe.rs"
```

---

### Task 11: `recommend.rs` part 1 — type names, Polars layout, size formulas

Pure functions; no Series involved.

**Files:**
- Create: `services/analytics/src/recommend.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod recommend;`)

- [ ] **Step 1: Create the file with part 1 and its tests**

```rust
// ─────────────────────────────────────────────────────────────────────────────
// recommend — narrowest value-preserving Arrow type per column (Spec B)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-recommend-technique-design.md.
//
// Per column: Describe's profile (describe.rs) and Rust cardinality estimates
// (cardinality_estimators.rs) feed the step-1 type rules and the step-2
// dictionary rule, which emit candidate Arrow types with analytically predicted
// IPC sizes (§5.1). Candidates are tried smallest projected population size
// first, ties broken by hierarchy rank; each is cast, verified row by row
// against the original and measured with sizes.rs; the first that verifies is
// chosen. The original type is always a candidate and cannot fail.
//
// Everything below the plugin entry works on arrow-rs arrays (Series cross in
// through shared::to_arrow_rs), so moving the Python↔Rust boundary to Arrow
// tables later changes only the entry point.
//
// Leading-zero rule (Spec A §5.1): an integer-looking string with a leading zero
// ("007") must stay a String — identifiers such as UUID fragments, account
// numbers or zip codes can be all digits with significant leading zeros, and
// casting to an integer would lose them. A value with a decimal point
// ("007.50") is unlikely to be an identifier, so only numeric equivalence
// matters for it; differing leading or trailing zeros set rec_lossy_formatting.

use arrow_schema::{DataType as AT, Field as AField, Fields, TimeUnit};
use std::sync::Arc;

/// Hierarchy rank (Spec B §4.1): breaks ties between candidates of equal projected size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rank {
    Null,
    Boolean,
    UInt,
    Int,
    Decimal,
    Float,
    Date,
    Time,
    Timestamp,
    TimestampWithOffset,
    Dictionary,
    Plain,
    List,
    Original,
}

/// What a candidate casts a column (or a list's inner values) to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Target {
    Null,
    Boolean,
    /// A string column of two values: (true text, false text), lower-cased.
    BoolPair(String, String),
    /// A fixed-width type: integers, decimals, floats, Date32, Time32/64, Timestamp, Duration.
    Fixed(AT),
    TimestampWithOffset(TimeUnit),
    /// Dictionary-encoded strings: (Arrow key, Polars key — Polars reserves one code).
    Dictionary(AT, AT),
    /// Utf8 or Binary with 32-bit offsets.
    Plain(AT),
    /// The column's own type (Arrow classic layout): always a candidate, cannot fail.
    Original(AT),
    /// Every non-null list holds exactly one item → that item.
    Scalar(Box<Target>),
    List(Box<Target>),
    FixedList(Box<Target>, i32),
}

impl Target {
    pub(crate) fn arrow_type(&self) -> AT {
        let item = |t: &Target| Arc::new(AField::new("item", t.arrow_type(), true));
        match self {
            Target::Null => AT::Null,
            Target::Boolean | Target::BoolPair(..) => AT::Boolean,
            Target::Fixed(t) | Target::Plain(t) | Target::Original(t) => t.clone(),
            Target::TimestampWithOffset(u) => timestamp_with_offset(*u),
            Target::Dictionary(k, _) => AT::Dictionary(Box::new(k.clone()), Box::new(AT::Utf8)),
            Target::Scalar(t) => t.arrow_type(),
            Target::List(t) => AT::List(item(t)),
            Target::FixedList(t, w) => AT::FixedSizeList(item(t), *w),
        }
    }

    /// The Polars key of a dictionary anywhere in this target.
    pub(crate) fn polars_key(&self) -> Option<AT> {
        match self {
            Target::Dictionary(_, k) => Some(k.clone()),
            Target::Scalar(t) | Target::List(t) | Target::FixedList(t, _) => t.polars_key(),
            _ => None,
        }
    }
}

pub(crate) fn decimal_type(precision: u8, scale: i8) -> AT {
    match precision {
        0..=9 => AT::Decimal32(precision, scale),
        10..=18 => AT::Decimal64(precision, scale),
        _ => AT::Decimal128(precision, scale),
    }
}

/// Storage of Arrow's canonical extension `arrow.timestamp_with_offset`.
pub(crate) fn timestamp_with_offset(unit: TimeUnit) -> AT {
    AT::Struct(Fields::from(vec![
        AField::new("timestamp", AT::Timestamp(unit, Some("UTC".into())), false),
        AField::new("offset_minutes", AT::Int16, false),
    ]))
}

/// Dictionary key widths for cardinality `c`: (Arrow, Polars). Arrow indexes
/// 0..=255 with UInt8; Polars reserves one code (an Enum of 256 categories is UInt16).
pub(crate) fn dictionary_keys(c: f64) -> (AT, AT) {
    let arrow = if c <= 256.0 { AT::UInt8 } else if c <= 65_536.0 { AT::UInt16 } else { AT::UInt32 };
    let polars = if c <= 255.0 { AT::UInt8 } else if c <= 65_535.0 { AT::UInt16 } else { AT::UInt32 };
    (arrow, polars)
}

fn unit_name(u: &TimeUnit) -> &'static str {
    match u {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

/// pyarrow's `str(type)` spelling.
pub(crate) fn pa_name(t: &AT) -> String {
    match t {
        AT::Null => "null".into(),
        AT::Boolean => "bool".into(),
        AT::Int8 => "int8".into(),
        AT::Int16 => "int16".into(),
        AT::Int32 => "int32".into(),
        AT::Int64 => "int64".into(),
        AT::UInt8 => "uint8".into(),
        AT::UInt16 => "uint16".into(),
        AT::UInt32 => "uint32".into(),
        AT::UInt64 => "uint64".into(),
        AT::Float16 => "halffloat".into(),
        AT::Float32 => "float".into(),
        AT::Float64 => "double".into(),
        AT::Decimal32(p, s) => format!("decimal32({p}, {s})"),
        AT::Decimal64(p, s) => format!("decimal64({p}, {s})"),
        AT::Decimal128(p, s) => format!("decimal128({p}, {s})"),
        AT::Date32 => "date32[day]".into(),
        AT::Date64 => "date64[ms]".into(),
        AT::Time32(u) => format!("time32[{}]", unit_name(u)),
        AT::Time64(u) => format!("time64[{}]", unit_name(u)),
        AT::Timestamp(u, None) => format!("timestamp[{}]", unit_name(u)),
        AT::Timestamp(u, Some(tz)) => format!("timestamp[{}, tz={tz}]", unit_name(u)),
        AT::Duration(u) => format!("duration[{}]", unit_name(u)),
        AT::Utf8 => "string".into(),
        AT::LargeUtf8 => "large_string".into(),
        AT::Utf8View => "string_view".into(),
        AT::Binary => "binary".into(),
        AT::LargeBinary => "large_binary".into(),
        AT::BinaryView => "binary_view".into(),
        AT::List(f) => format!("list<{}: {}>", f.name(), pa_name(f.data_type())),
        AT::LargeList(f) => format!("large_list<{}: {}>", f.name(), pa_name(f.data_type())),
        AT::FixedSizeList(f, w) => format!("fixed_size_list<{}: {}>[{w}]", f.name(), pa_name(f.data_type())),
        AT::Dictionary(k, v) => format!("dictionary<values={}, indices={}, ordered=0>", pa_name(v), pa_name(k)),
        AT::Struct(fs) => format!(
            "struct<{}>",
            fs.iter()
                .map(|f| format!("{}: {}{}", f.name(), pa_name(f.data_type()), if f.is_nullable() { "" } else { " not null" }))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other}"),
    }
}

fn py_str(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Python `str(dtype)` of the Polars type that holds `t`. A dictionary is an Enum of
/// `enum_values` when given (exact cardinality), else a Categorical named after
/// `column` whose physical type is `key`.
pub(crate) fn pl_name(t: &AT, column: &str, enum_values: Option<&[String]>, key: &AT) -> String {
    let unit = |u: &TimeUnit| match u {
        TimeUnit::Second | TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    };
    let inner = |t: &AT| pl_name(t, column, enum_values, key);
    match t {
        AT::Null => "Null".into(),
        AT::Boolean => "Boolean".into(),
        AT::Int8 => "Int8".into(),
        AT::Int16 => "Int16".into(),
        AT::Int32 => "Int32".into(),
        AT::Int64 => "Int64".into(),
        AT::UInt8 => "UInt8".into(),
        AT::UInt16 => "UInt16".into(),
        AT::UInt32 => "UInt32".into(),
        AT::UInt64 => "UInt64".into(),
        AT::Float32 => "Float32".into(),
        AT::Float64 => "Float64".into(),
        AT::Decimal32(p, s) | AT::Decimal64(p, s) => format!("Decimal(precision={p}, scale={s})"),
        AT::Decimal128(p, s) => format!("Decimal(precision={p}, scale={s})"),
        AT::Date32 => "Date".into(),
        AT::Time32(_) | AT::Time64(_) => "Time".into(),
        AT::Timestamp(u, tz) => format!(
            "Datetime(time_unit='{}', time_zone={})",
            unit(u),
            tz.as_ref().map_or("None".to_string(), |z| format!("'{z}'"))
        ),
        AT::Duration(u) => format!("Duration(time_unit='{}')", unit(u)),
        AT::Utf8 | AT::LargeUtf8 | AT::Utf8View => "String".into(),
        AT::Binary | AT::LargeBinary | AT::BinaryView => "Binary".into(),
        AT::Dictionary(..) => match enum_values {
            Some(v) => format!("Enum(categories=[{}])", v.iter().map(|s| py_str(s)).collect::<Vec<_>>().join(", ")),
            None => format!("Categorical(Categories(name=\"{column}\", namespace=\"\", physical=pl.{}))", inner(key)),
        },
        AT::List(f) | AT::LargeList(f) => format!("List({})", inner(f.data_type())),
        AT::FixedSizeList(f, w) => format!("Array({}, shape=({w},))", inner(f.data_type())),
        AT::Struct(fs) => format!(
            "Struct({{{}}})",
            fs.iter().map(|f| format!("'{}': {}", f.name(), inner(f.data_type()))).collect::<Vec<_>>().join(", ")
        ),
        other => format!("{other}"),
    }
}

/// The Arrow type Polars exports (CompatLevel::newest) for a column Polars holds as
/// the type recommended by `t` (Spec B §5.5); `key` is a dictionary's Polars key.
pub(crate) fn polars_layout(t: &AT, key: &AT) -> AT {
    let field = |f: &Arc<AField>| Arc::new(AField::new(f.name(), polars_layout(f.data_type(), key), f.is_nullable()));
    match t {
        AT::Decimal32(p, s) | AT::Decimal64(p, s) => AT::Decimal128(*p, *s),
        AT::Timestamp(TimeUnit::Second, tz) => AT::Timestamp(TimeUnit::Millisecond, tz.clone()),
        AT::Time32(_) | AT::Time64(_) => AT::Time64(TimeUnit::Nanosecond),
        AT::Duration(TimeUnit::Second) => AT::Duration(TimeUnit::Millisecond),
        AT::Utf8 | AT::LargeUtf8 => AT::Utf8View,
        AT::Binary | AT::LargeBinary => AT::BinaryView,
        AT::Dictionary(_, _) => AT::Dictionary(Box::new(key.clone()), Box::new(AT::Utf8View)),
        AT::List(f) | AT::LargeList(f) => AT::LargeList(field(f)),
        AT::FixedSizeList(f, w) => AT::FixedSizeList(field(f), *w),
        AT::Struct(fs) => AT::Struct(fs.iter().map(field).collect()),
        other => other.clone(),
    }
}

// ── analytic sizes (Spec B §5.1, §5.3) ──────────────────────────────────────

/// What a predicted size depends on: rows, nulls, value bytes, distinct values and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Shape {
    pub n: f64,
    pub nulls: f64,
    pub sum_len: f64,
    pub d: f64,
    pub sum_len_unique: f64,
}

impl Shape {
    /// The population the frame samples: row-proportional terms scale by `r`; a
    /// dictionary holds `c` values of the observed mean length.
    pub(crate) fn project(&self, r: f64, c: f64) -> Shape {
        let per_value = if self.d > 0.0 { self.sum_len_unique / self.d } else { 0.0 };
        Shape { n: self.n * r, nulls: self.nulls * r, sum_len: self.sum_len * r, d: c, sum_len_unique: per_value * c }
    }
}

pub(crate) fn pad(x: f64) -> f64 {
    (x / 8.0).ceil() * 8.0
}

pub(crate) fn validity(n: f64, nulls: f64) -> f64 {
    if nulls > 0.0 { pad((n / 8.0).ceil()) } else { 0.0 }
}

/// Uncompressed Arrow IPC body bytes of a scalar type `t` holding values shaped
/// like `s`, exactly as sizes.rs measures it. Lists are sized by the caller.
pub(crate) fn body_size(t: &AT, s: &Shape) -> f64 {
    let v = validity(s.n, s.nulls);
    match t {
        AT::Null => 0.0,
        AT::Boolean => v + pad((s.n / 8.0).ceil()),
        AT::Utf8 | AT::Binary => v + pad(4.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::LargeUtf8 | AT::LargeBinary => v + pad(8.0 * (s.n + 1.0)) + pad(s.sum_len),
        AT::Dictionary(k, _) => {
            v + pad(s.n * k.primitive_width().unwrap() as f64) + pad(4.0 * (s.d + 1.0)) + pad(s.sum_len_unique)
        }
        AT::Struct(_) => v + pad(8.0 * s.n) + pad(2.0 * s.n), // timestamp_with_offset
        t => v + pad(s.n * t.primitive_width().unwrap_or(8) as f64),
    }
}

// ── numbers and units ────────────────────────────────────────────────────────

pub(crate) fn digits(v: u128) -> u8 {
    if v == 0 { 1 } else { (v.ilog10() + 1) as u8 }
}

pub(crate) fn narrowest_uint(hi: i128) -> Option<AT> {
    [(u8::MAX as i128, AT::UInt8), (u16::MAX as i128, AT::UInt16), (u32::MAX as i128, AT::UInt32), (u64::MAX as i128, AT::UInt64)]
        .into_iter()
        .find(|(max, _)| hi <= *max)
        .map(|(_, t)| t)
}

pub(crate) fn narrowest_int(lo: i128, hi: i128) -> Option<AT> {
    [
        (i8::MIN as i128, i8::MAX as i128, AT::Int8),
        (i16::MIN as i128, i16::MAX as i128, AT::Int16),
        (i32::MIN as i128, i32::MAX as i128, AT::Int32),
        (i64::MIN as i128, i64::MAX as i128, AT::Int64),
    ]
    .into_iter()
    .find(|(min, max, _)| *min <= lo && hi <= *max)
    .map(|(_, _, t)| t)
}

pub(crate) fn trailing_zeros10(mut g: i128) -> usize {
    let mut k = 0;
    while g != 0 && g % 10 == 0 {
        g /= 10;
        k += 1;
    }
    k
}

pub(crate) fn unit_ns(u: &TimeUnit) -> i128 {
    match u {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// The coarsest Arrow unit dividing `g_ns` nanoseconds (0 → seconds).
pub(crate) fn coarsest_unit(g_ns: i128) -> TimeUnit {
    match g_ns {
        0 => TimeUnit::Second,
        g if g % 1_000_000_000 == 0 => TimeUnit::Second,
        g if g % 1_000_000 == 0 => TimeUnit::Millisecond,
        g if g % 1_000 == 0 => TimeUnit::Microsecond,
        _ => TimeUnit::Nanosecond,
    }
}

/// Unit for ISO strings from their significant fractional-second digits.
pub(crate) fn iso_unit(sig: u32) -> TimeUnit {
    match sig {
        0 => TimeUnit::Second,
        1..=3 => TimeUnit::Millisecond,
        4..=6 => TimeUnit::Microsecond,
        _ => TimeUnit::Nanosecond,
    }
}

pub(crate) fn time_type(u: TimeUnit) -> AT {
    match u {
        TimeUnit::Second | TimeUnit::Millisecond => AT::Time32(u),
        _ => AT::Time64(u),
    }
}

/// Arrow time zone of a fixed offset in minutes: "UTC" or "±HH:MM".
pub(crate) fn offset_tz(minutes: i32) -> String {
    if minutes == 0 {
        return "UTC".into();
    }
    let sign = if minutes < 0 { '-' } else { '+' };
    format!("{sign}{:02}:{:02}", minutes.abs() / 60, minutes.abs() % 60)
}

/// A decimal or exponent string as (negative, significant digits, point) with
/// value = 0.DIGITS × 10^point; zero is (false, "", 0). Equal values give equal
/// tuples regardless of formatting ("1.50" ≡ "1.5", "0.00120" ≡ "1.2e-3").
pub(crate) fn canon(s: &str) -> (bool, String, i32) {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let (mantissa, exp) = s.split_once(['e', 'E']).map_or((s, 0), |(m, e)| (m, e.parse::<i32>().unwrap_or(0)));
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{int}{frac}");
    let lead = all.len() - all.trim_start_matches('0').len();
    let sig = all.trim_start_matches('0').trim_end_matches('0');
    if sig.is_empty() {
        return (false, String::new(), 0);
    }
    (neg, sig.to_string(), int.len() as i32 + exp - lead as i32)
}

/// Exact unscaled value of a decimal/exponent string at `scale`; None when it needs
/// more decimal places or more than 38 digits.
pub(crate) fn decimal_from_repr(repr: &str, scale: u32) -> Option<i128> {
    let (neg, sig, point) = canon(repr);
    if sig.is_empty() {
        return Some(0);
    }
    let places = sig.len() as i32 - point;
    if places > scale as i32 {
        return None;
    }
    let mut v: i128 = 0;
    for c in sig.bytes() {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
    }
    v = v.checked_mul(10i128.checked_pow((scale as i32 - places) as u32)?)?;
    (v < 10i128.pow(38)).then_some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizes::ipc_body_bytes;
    use arrow_array::{types::UInt8Type, DictionaryArray, StringArray};

    #[test]
    fn pyarrow_type_names() {
        assert_eq!(pa_name(&AT::Float32), "float");
        assert_eq!(pa_name(&decimal_type(6, 2)), "decimal32(6, 2)");
        assert_eq!(pa_name(&decimal_type(10, 3)), "decimal64(10, 3)");
        assert_eq!(pa_name(&AT::Timestamp(TimeUnit::Millisecond, Some("+05:00".into()))), "timestamp[ms, tz=+05:00]");
        assert_eq!(pa_name(&Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type()), "dictionary<values=string, indices=uint8, ordered=0>");
        assert_eq!(pa_name(&Target::List(Box::new(Target::Fixed(AT::UInt8))).arrow_type()), "list<item: uint8>");
        assert_eq!(pa_name(&timestamp_with_offset(TimeUnit::Second)), "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>");
    }

    #[test]
    fn polars_type_names() {
        let k = AT::UInt8;
        assert_eq!(pl_name(&decimal_type(6, 2), "x", None, &k), "Decimal(precision=6, scale=2)");
        assert_eq!(pl_name(&AT::Timestamp(TimeUnit::Second, None), "x", None, &k), "Datetime(time_unit='ms', time_zone=None)");
        assert_eq!(pl_name(&AT::Duration(TimeUnit::Second), "x", None, &k), "Duration(time_unit='ms')");
        let dict = Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type();
        assert_eq!(pl_name(&dict, "x", Some(&["a".into(), "b".into()]), &k), "Enum(categories=['a', 'b'])");
        assert_eq!(pl_name(&dict, "x", None, &k), "Categorical(Categories(name=\"x\", namespace=\"\", physical=pl.UInt8))");
        assert_eq!(
            pl_name(&timestamp_with_offset(TimeUnit::Second), "x", None, &k),
            "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})"
        );
    }

    #[test]
    fn polars_layouts_and_key_widths() {
        assert_eq!(polars_layout(&AT::Time32(TimeUnit::Second), &AT::UInt8), AT::Time64(TimeUnit::Nanosecond));
        assert_eq!(polars_layout(&decimal_type(6, 2), &AT::UInt8), AT::Decimal128(6, 2));
        assert_eq!(dictionary_keys(256.0), (AT::UInt8, AT::UInt16));
        assert_eq!(dictionary_keys(255.0), (AT::UInt8, AT::UInt8));
        assert_eq!(dictionary_keys(65_537.0).0, AT::UInt32);
    }

    #[test]
    fn predicted_sizes_match_sizes_rs() {
        let s = StringArray::from(vec![Some("ab"), None]);
        let shape = Shape { n: 2.0, nulls: 1.0, sum_len: 2.0, d: 1.0, sum_len_unique: 2.0 };
        assert_eq!(body_size(&AT::Utf8, &shape), ipc_body_bytes(&s, None).unwrap() as f64);
        let d: DictionaryArray<UInt8Type> = vec!["a", "b", "a"].into_iter().collect();
        let shape = Shape { n: 3.0, nulls: 0.0, sum_len: 3.0, d: 2.0, sum_len_unique: 2.0 };
        assert_eq!(body_size(&Target::Dictionary(AT::UInt8, AT::UInt8).arrow_type(), &shape), ipc_body_bytes(&d, None).unwrap() as f64);
    }

    #[test]
    fn canonical_decimals() {
        assert_eq!(canon("1.50"), canon("1.5"));
        assert_eq!(canon("0.00120"), canon("1.2e-3"));
        assert_eq!(canon("-0.0"), canon("0"));
        assert_ne!(canon("123"), canon("12.3"));
        assert_eq!(decimal_from_repr("1e-7", 7), Some(1));
        assert_eq!(decimal_from_repr("1.5e20", 0), Some(150_000_000_000_000_000_000));
        assert_eq!(decimal_from_repr("123.45", 1), None);
        assert_eq!(decimal_from_repr("-0.5", 2), Some(-50));
    }

    #[test]
    fn units_and_widths() {
        assert_eq!(coarsest_unit(86_400_000_000_000), TimeUnit::Second);
        assert_eq!(coarsest_unit(120_000_000), TimeUnit::Millisecond);
        assert_eq!(narrowest_uint(255), Some(AT::UInt8));
        assert_eq!(narrowest_int(-200, 5), Some(AT::Int16));
        assert_eq!(narrowest_int(0, i128::from(u64::MAX)), None);
        assert_eq!(trailing_zeros10(1_200), 2);
        assert_eq!(offset_tz(-210), "-03:30");
    }
}
```

- [ ] **Step 2: Register the module** — add `mod recommend;` to `lib.rs`.

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib recommend::`
Expected: PASS (6 tests). If `vec!["a","b","a"].into_iter().collect::<DictionaryArray<UInt8Type>>()` does not compile, build it with `StringDictionaryBuilder::<UInt8Type>` from `arrow_array::builder`.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/recommend.rs services/analytics/src/lib.rs
git commit -m "feat: recommend.rs type names, Polars layout and predicted sizes"
```

---

### Task 12: `recommend.rs` part 2 — rules → candidates

**Files:**
- Modify: `services/analytics/src/recommend.rs` (append below part 1, above `mod tests`)
- Modify: `services/analytics/src/describe.rs` (make `Profile`'s sub-structs' fields `pub` if any are not)

- [ ] **Step 1: Write the failing tests** — append inside `mod tests`:

```rust
    use crate::describe::describe_one;
    use crate::shared::to_arrow_rs;
    use polars::prelude::{CompatLevel, DataType as PT, IntoSeries, NamedFrom, Series, TimeUnit as PTimeUnit};

    pub(super) fn params() -> Params {
        Params { seed: 0, zstd_level: 1, population_rows: None, categorical_threshold: 10_000, boolean_pairs: vec![("true".into(), "false".into())] }
    }

    fn types(s: Series, p: &Params) -> Vec<(String, Outcome)> {
        let d = describe_one(&s, 0).unwrap();
        let lvl = Level {
            column: "x", dtype: s.dtype(), values: to_arrow_rs(&s, CompatLevel::oldest()).unwrap(), p: &d.outer,
            n_midnight: d.n_midnight, size_bytes: 0, est: level_estimate(&d.outer, d.n_rows - d.n_null, None), r: 1.0, prefix: "",
        };
        candidates(&lvl, p).unwrap().iter().map(|c| (pa_name(&c.target.arrow_type()), c.outcome)).collect()
    }

    fn names(s: Series) -> Vec<String> {
        types(s, &params()).into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn integer_rules() {
        assert_eq!(names(Series::new("x".into(), &[0i64, 1])), ["bool", "uint8", "int8", "int64"]);
        assert_eq!(names(Series::new("x".into(), &[-200i64, 5])), ["int16", "int64"]);
    }

    #[test]
    fn decimal_scale_reduced_by_gcd() {
        let dec = polars::prelude::Int128Chunked::from_slice("x".into(), &[120, 340]).into_decimal_unchecked(Some(10), 2).into_series();
        assert_eq!(names(dec), ["decimal32(2, 1)", "decimal128(10, 2)"]);
    }

    #[test]
    fn float_rules() {
        assert_eq!(names(Series::new("x".into(), &[123.45f64, 99.99])), ["decimal32(5, 2)", "double"]);
        assert_eq!(names(Series::new("x".into(), &[0.5f64, 0.25])), ["decimal32(2, 2)", "float", "double"]);
        assert_eq!(names(Series::new("x".into(), &[0.1f64, f64::NAN])), ["double"]);
    }

    #[test]
    fn string_rules() {
        let dict = "dictionary<values=string, indices=uint8, ordered=0>";
        assert_eq!(names(Series::new("x".into(), &["007", "12"])), ["string", dict, "large_string"]);
        assert_eq!(
            names(Series::new("x".into(), &["1234567890.1", "0.00000012345"])),
            ["decimal128(21, 11)", "double", "string", dict, "large_string"]
        );
        let offsets = names(Series::new("x".into(), &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]));
        assert_eq!(offsets[0], "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>");
        assert_eq!(names(Series::new("x".into(), &["True", "false"]))[0], "bool");
    }

    #[test]
    fn temporal_rules() {
        let days = Series::new("x".into(), &[0i64, 86_400_000_000]).cast(&PT::Datetime(PTimeUnit::Microseconds, None)).unwrap();
        assert_eq!(names(days), ["date32[day]", "timestamp[s]", "timestamp[us]"]);
    }

    #[test]
    fn dictionary_gate_rejects() {
        let p = Params { categorical_threshold: 1, ..params() };
        let got = types(Series::new("x".into(), &["a", "b", "a", "b"]), &p);
        assert!(got.iter().any(|(t, o)| t.starts_with("dictionary") && *o == Outcome::Rejected));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib recommend::`
Expected: FAIL to compile (`Params`, `Level`, `candidates` undefined).

- [ ] **Step 3: Implement** — append to `recommend.rs` (before `mod tests`):

```rust
// ── candidates (Spec B §4, §5.2) ────────────────────────────────────────────

use crate::cardinality_estimators::{estimate, Estimate};
use crate::describe::Profile;
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type};
use arrow_array::{Array, ArrayRef, LargeStringArray};
use arrow_cast::cast::{cast_with_options, CastOptions};
use polars::prelude::DataType as PT;
use serde::Deserialize;

/// Plugin keyword arguments (Recommend's constructor keywords).
#[derive(Deserialize, Clone, Debug)]
pub(crate) struct Params {
    pub seed: u64,
    pub zstd_level: i32,
    pub population_rows: Option<u64>,
    pub categorical_threshold: u64,
    pub boolean_pairs: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Outcome {
    Chosen,
    Failed,
    Rejected,
    NotTried,
}

impl Outcome {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Outcome::Chosen => "chosen",
            Outcome::Failed => "failed",
            Outcome::Rejected => "rejected",
            Outcome::NotTried => "not_tried",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub target: Target,
    pub rank: Rank,
    pub rule: String,
    /// The metric values the rule tested and what they implied.
    pub evidence: String,
    pub predicted: u64,
    pub projected: f64,
    pub outcome: Outcome,
    pub reason: Option<String>,
}

/// One level of a column: the column, or a list's inner values.
pub(crate) struct Level<'a> {
    pub column: &'a str,
    /// Polars dtype of these values.
    pub dtype: &'a PT,
    /// The values in Arrow's classic layout (CompatLevel::oldest).
    pub values: ArrayRef,
    pub p: &'a Profile,
    pub n_midnight: Option<u64>,
    /// Measured uncompressed size of `values`.
    pub size_bytes: u64,
    pub est: Estimate,
    /// Population rows ÷ frame rows.
    pub r: f64,
    /// "" or "inner: " — prefixes every rule name.
    pub prefix: &'static str,
}

impl Level<'_> {
    fn n_rows(&self) -> u64 {
        self.values.len() as u64
    }

    fn n_null(&self) -> u64 {
        self.values.null_count() as u64
    }

    fn n(&self) -> u64 {
        self.n_rows() - self.n_null()
    }

    /// Population cardinality for dictionaries: est_high where an interval exists.
    fn cardinality(&self) -> (f64, &'static str) {
        match self.est.est_high {
            Some(h) => (h, "est_high"),
            None => (self.est.est_cardinality, "est_cardinality"),
        }
    }

    fn shape(&self) -> Shape {
        Shape {
            n: self.n_rows() as f64,
            nulls: self.n_null() as f64,
            sum_len: self.p.sum_len.unwrap_or(0) as f64,
            d: self.p.freq.n_unique as f64,
            sum_len_unique: self.p.freq.sum_len_unique.unwrap_or(0) as f64,
        }
    }
}

pub(crate) fn level_estimate(p: &Profile, n: u64, q: Option<f64>) -> Estimate {
    estimate(p.freq.n_unique, n, p.freq.f1, p.freq.f2, &p.freq.capture_history, q)
}

/// arrow-cast with `safe: false`: a value that does not fit is an error, not a null.
pub(crate) fn arrow_cast(a: &dyn Array, to: &AT) -> Result<ArrayRef, String> {
    cast_with_options(a, to, &CastOptions { safe: false, ..Default::default() }).map_err(|e| e.to_string())
}

pub(crate) fn is_text(dt: &PT) -> bool {
    matches!(dt, PT::String | PT::Categorical(..) | PT::Enum(..))
}

pub(crate) fn is_float(dt: &PT) -> bool {
    matches!(dt, PT::Float32 | PT::Float64)
}

pub(crate) fn text_of(values: &ArrayRef) -> Result<LargeStringArray, String> {
    Ok(arrow_cast(values.as_ref(), &AT::LargeUtf8)?.as_string::<i64>().clone())
}

/// Row `i` as an exact integer (integer columns; Int128 arrives as decimal128(38, 0)).
fn int_at(a: &ArrayRef, i: u64) -> Option<i128> {
    let v = arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Decimal128(38, 0)).ok()?;
    let v = v.as_primitive::<Decimal128Type>();
    v.is_valid(0).then(|| v.value(0))
}

fn f64_at(a: &ArrayRef, i: u64) -> f64 {
    arrow_cast(a.slice(i as usize, 1).as_ref(), &AT::Float64).map_or(f64::NAN, |v| v.as_primitive::<Float64Type>().value(0))
}

struct Rules<'l, 'a> {
    lvl: &'l Level<'a>,
    shape: Shape,
    out: Vec<Candidate>,
}

impl Rules<'_, '_> {
    fn push(&mut self, target: Target, rank: Rank, rule: &str, evidence: String) {
        let t = target.arrow_type();
        let (c, _) = self.lvl.cardinality();
        self.out.push(Candidate {
            predicted: body_size(&t, &self.shape) as u64,
            projected: body_size(&t, &self.shape.project(self.lvl.r, c)),
            target,
            rank,
            rule: format!("{}{rule}", self.lvl.prefix),
            evidence,
            outcome: Outcome::NotTried,
            reason: None,
        });
    }

    fn integers(&mut self, lo: i128, hi: i128, from: &str, evidence: &str) {
        let ev = format!("{evidence}min={lo} max={hi}");
        if lo >= 0 && hi <= 1 {
            self.push(Target::Boolean, Rank::Boolean, &format!("{from}→boolean"), ev.clone());
        }
        let uint = if lo >= 0 { narrowest_uint(hi) } else { None };
        let int = narrowest_int(lo, hi);
        if let Some(t) = uint.clone() {
            self.push(Target::Fixed(t), Rank::UInt, &format!("{from}→uint"), ev.clone());
        }
        if let Some(t) = int.clone() {
            self.push(Target::Fixed(t), Rank::Int, &format!("{from}→int"), ev.clone());
        }
        if uint.is_none() && int.is_none() {
            let p = digits(lo.unsigned_abs().max(hi.unsigned_abs()));
            if p <= 38 {
                self.push(Target::Fixed(AT::Decimal128(p, 0)), Rank::Decimal, &format!("{from}→decimal128"), format!("{ev} → p={p}"));
            }
        }
    }

    fn decimal(&mut self, scale: usize) {
        let (p, v) = (self.lvl.p, &self.lvl.values);
        let (Some(lo), Some(hi)) = (p.range.argmin, p.range.argmax) else { return };
        let unscaled = |i: u64| v.as_primitive::<Decimal128Type>().value(i as usize);
        let g = p.gcd.unwrap_or(1);
        let k = if g == 0 { scale } else { trailing_zeros10(g).min(scale) };
        let f = 10i128.pow(k as u32);
        let (lo, hi, s) = (unscaled(lo) / f, unscaled(hi) / f, scale - k);
        let ev = format!("gcd={g} → {k} trailing zeros, scale {scale}→{s}; ");
        if s == 0 {
            return self.integers(lo, hi, "decimal", &ev);
        }
        let prec = digits(lo.unsigned_abs().max(hi.unsigned_abs())).max(s as u8);
        if prec <= 38 {
            self.push(Target::Fixed(decimal_type(prec, s as i8)), Rank::Decimal, "decimal→decimal", format!("{ev}min={lo} max={hi} → p={prec} s={s}"));
        }
    }

    fn float(&mut self) {
        let (p, v) = (self.lvl.p, &self.lvl.values);
        let f = p.floats.expect("float columns have float stats");
        let ev = format!("n_nan={} n_inf={} n_fractional={} ", f.n_nan, f.n_inf, f.n_fractional);
        if let (0, 0, Some(lo), Some(hi)) = (f.n_nan, f.n_inf, p.range.argmin, p.range.argmax) {
            let (lo, hi) = (f64_at(v, lo), f64_at(v, hi));
            let top = lo.abs().max(hi.abs());
            if f.n_fractional == 0 {
                if top < 1e38 {
                    self.integers(lo as i128, hi as i128, "float", &ev);
                }
            } else if let Some(s) = f.max_frac_digits {
                let int_digits = if top < 1.0 { 0 } else { digits(top.floor() as u128) as u32 };
                let prec = int_digits + s;
                if prec <= 38 {
                    self.push(
                        Target::Fixed(decimal_type(prec as u8, s as i8)),
                        Rank::Decimal,
                        "float→decimal",
                        format!("{ev}max_frac_digits={s} int_digits={int_digits} → p={prec} s={s}"),
                    );
                }
            }
        }
        if self.lvl.dtype == &PT::Float64 && f.n_f32_inexact == 0 {
            self.push(Target::Fixed(AT::Float32), Rank::Float, "float64→float32", "n_f32_inexact=0".into());
        }
    }

    fn temporal(&mut self) {
        let lvl = self.lvl;
        let (unit, tz) = match lvl.values.data_type() {
            AT::Timestamp(u, tz) => (*u, tz.clone()),
            AT::Duration(u) | AT::Time64(u) | AT::Time32(u) => (*u, None),
            _ => return,
        };
        let g = lvl.p.gcd.unwrap_or(1);
        let coarse = coarsest_unit(g * unit_ns(&unit));
        let ev = format!("gcd={g} ({}) → {}", unit_name(&unit), unit_name(&coarse));
        match lvl.dtype {
            PT::Datetime(_, zone) => {
                if zone.is_none() && lvl.n_midnight == Some(lvl.n()) {
                    self.push(Target::Fixed(AT::Date32), Rank::Date, "datetime→date32", format!("n_midnight={}", lvl.n()));
                }
                if coarse != unit {
                    self.push(Target::Fixed(AT::Timestamp(coarse, tz)), Rank::Timestamp, "datetime→timestamp", ev);
                }
            }
            PT::Duration(_) if coarse != unit => self.push(Target::Fixed(AT::Duration(coarse)), Rank::Timestamp, "duration→duration", ev),
            PT::Time if coarse != unit => self.push(Target::Fixed(time_type(coarse)), Rank::Time, "time→time", ev),
            _ => {}
        }
    }

    fn text(&mut self, params: &Params) -> Result<(), String> {
        let lvl = self.lvl;
        let (st, n) = (lvl.p.strings.as_ref().expect("string columns have string stats"), lvl.n());
        let text = text_of(&lvl.values)?;
        let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 {
            lvl.p.freq.top5_idx.iter().map(|&i| text.value(i as usize).to_lowercase()).collect()
        } else {
            Vec::new()
        };
        let pair = params
            .boolean_pairs
            .iter()
            .map(|(t, f)| (t.to_lowercase(), f.to_lowercase()))
            .find(|(t, f)| !distinct.is_empty() && distinct.iter().all(|v| v == t || v == f));
        let sig = st.iso_max_sig_frac_digits.unwrap_or(0);
        if let Some((t, f)) = pair {
            self.push(Target::BoolPair(t.clone(), f.clone()), Rank::Boolean, "string→boolean", format!("distinct={distinct:?} pair=({t:?}, {f:?})"));
        } else if st.n_numeric_int == n && st.n_leading_zero == 0 && !st.int_overflow && st.int_min.is_some() {
            self.integers(st.int_min.unwrap(), st.int_max.unwrap(), "string", &format!("n_numeric_int={n} n_leading_zero=0 "));
        } else if st.n_numeric == n && st.n_leading_zero == 0 {
            let (i, f) = (st.max_int_digits.unwrap_or(0), st.max_frac_digits.unwrap_or(0));
            let (min_f, sig_d) = (st.min_frac_digits.unwrap_or(0), st.max_sig_digits.unwrap_or(0));
            let prec = (i + f).max(1);
            let ev = format!(
                "n_numeric={n} n_leading_zero=0 numeric_max_int_digits={i} numeric_max_frac_digits={f} \
                 numeric_min_frac_digits={min_f} numeric_max_sig_digits={sig_d}"
            );
            if prec <= 38 {
                self.push(Target::Fixed(decimal_type(prec as u8, f as i8)), Rank::Decimal, "string→decimal", format!("{ev} → p={prec} s={f}"));
            }
            if min_f < f && prec > 18 {
                let why = format!("{ev} → varying places, p={prec} > 18");
                if sig_d <= 6 {
                    self.push(Target::Fixed(AT::Float32), Rank::Float, "string→float32", why.clone());
                }
                if sig_d <= 15 {
                    self.push(Target::Fixed(AT::Float64), Rank::Float, "string→float64", why);
                }
            }
        } else if st.n_iso_date == n {
            self.push(Target::Fixed(AT::Date32), Rank::Date, "string→date32", format!("n_iso_date={n}"));
        } else if st.n_iso_time == n {
            self.push(Target::Fixed(time_type(iso_unit(sig))), Rank::Time, "string→time", format!("n_iso_time={n} iso_max_sig_frac_digits={sig}"));
        } else if st.n_iso_datetime == n {
            if st.iso_n_midnight == n {
                self.push(Target::Fixed(AT::Date32), Rank::Date, "string→date32", format!("n_iso_datetime={n} iso_n_midnight={n}"));
            } else {
                self.push(
                    Target::Fixed(AT::Timestamp(iso_unit(sig), None)),
                    Rank::Timestamp,
                    "string→timestamp",
                    format!("n_iso_datetime={n} iso_max_sig_frac_digits={sig}"),
                );
            }
        } else if st.n_iso_datetime_tz == n {
            let ev = format!("n_iso_datetime_tz={n} iso_n_offsets={} iso_max_sig_frac_digits={sig}", st.offsets.len());
            match st.offsets.iter().next() {
                Some(&m) if st.offsets.len() == 1 => {
                    self.push(Target::Fixed(AT::Timestamp(iso_unit(sig), Some(offset_tz(m).into()))), Rank::Timestamp, "string→timestamp", ev)
                }
                _ => self.push(
                    Target::TimestampWithOffset(iso_unit(sig)),
                    Rank::TimestampWithOffset,
                    "string→timestamp_with_offset (arrow.timestamp_with_offset)",
                    ev,
                ),
            }
        }
        let sum_len = lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(Target::Plain(AT::Utf8), Rank::Plain, "string→utf8", format!("sum_len={sum_len}"));
        }
        self.dictionary(params);
        Ok(())
    }

    fn dictionary(&mut self, params: &Params) {
        let (c, source) = self.lvl.cardinality();
        let (key, polars_key) = dictionary_keys(c);
        let threshold = params.categorical_threshold;
        let ev = format!("c={c:?} from {source} n_unique={} categorical_threshold={threshold}", self.lvl.p.freq.n_unique);
        self.push(Target::Dictionary(key, polars_key), Rank::Dictionary, "string→dictionary", ev);
        if c > threshold as f64 {
            let last = self.out.last_mut().unwrap();
            last.outcome = Outcome::Rejected;
            last.reason = Some(format!("c={c:?} > categorical_threshold={threshold}"));
        }
    }

    fn binary(&mut self) {
        let sum_len = self.lvl.p.sum_len.unwrap_or(0);
        if sum_len < 1 << 31 {
            self.push(Target::Plain(AT::Binary), Rank::Plain, "binary→binary", format!("sum_len={sum_len}"));
        }
    }

    fn original(&mut self) {
        let lvl = self.lvl;
        self.out.push(Candidate {
            target: Target::Original(lvl.values.data_type().clone()),
            rank: Rank::Original,
            rule: format!("{}original", lvl.prefix),
            evidence: format!("measured size_bytes={}", lvl.size_bytes),
            predicted: lvl.size_bytes,
            projected: lvl.size_bytes as f64 * lvl.r,
            outcome: Outcome::NotTried,
            reason: None,
        });
    }
}

/// Every candidate for one level, in rule order (Spec B §4.2–4.3, §5.2); the
/// original type last.
pub(crate) fn candidates(lvl: &Level, params: &Params) -> Result<Vec<Candidate>, String> {
    let mut r = Rules { lvl, shape: lvl.shape(), out: Vec::new() };
    if lvl.n_rows() > 0 && lvl.n() == 0 {
        r.push(Target::Null, Rank::Null, "all-null→null", format!("n_null={} n_rows={}", lvl.n_null(), lvl.n_rows()));
    } else if lvl.n() > 0 {
        match lvl.dtype {
            PT::Int8 | PT::Int16 | PT::Int32 | PT::Int64 | PT::Int128 | PT::UInt8 | PT::UInt16 | PT::UInt32 | PT::UInt64 => {
                if let (Some(a), Some(b)) = (lvl.p.range.argmin, lvl.p.range.argmax) {
                    if let (Some(lo), Some(hi)) = (int_at(&lvl.values, a), int_at(&lvl.values, b)) {
                        r.integers(lo, hi, "integer", "");
                    }
                }
            }
            PT::Decimal(_, s) => r.decimal(s.unwrap_or(0)),
            PT::Float32 | PT::Float64 => r.float(),
            PT::Datetime(..) | PT::Duration(_) | PT::Time => r.temporal(),
            dt if is_text(dt) => r.text(params)?,
            PT::Binary => r.binary(),
            _ => {}
        }
    }
    r.original();
    Ok(r.out)
}
```

If `PT::Decimal`'s arity differs in polars 0.51, match it the way `describe.rs` constructs `DataType::Decimal(Some(38), Some(0))`.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib recommend::`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/recommend.rs services/analytics/src/describe.rs
git commit -m "feat: recommend.rs step-1 type rules and step-2 dictionary rule"
```

---

### Task 13: `recommend.rs` part 3 — cast and verify

**Files:**
- Modify: `services/analytics/src/recommend.rs`

- [ ] **Step 1: Write the failing tests** — append inside `mod tests`:

```rust
    use arrow_array::{Float64Array, Int64Array};

    #[test]
    fn floats_become_exact_decimals() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.1), Some(123.45), None]));
        let d = float_to_decimal(&a, &decimal_type(5, 2)).unwrap();
        let d = arrow_cast(d.as_ref(), &AT::Decimal128(38, 2)).unwrap();
        let v = d.as_primitive::<Decimal128Type>();
        assert_eq!((v.value(0), v.value(1), v.is_null(2)), (10, 12_345, true));
        assert!(verify_float(&a, &arrow_cast(d.as_ref(), &decimal_type(5, 2)).unwrap()).is_ok());
    }

    #[test]
    fn timestamps_with_offsets_from_text() {
        let text = LargeStringArray::from(vec![Some("2024-01-05T10:00+05:00"), None]);
        let a = from_text(&Target::TimestampWithOffset(TimeUnit::Second), &text).unwrap();
        let s = a.as_struct();
        let ts = s.column(0).as_primitive::<arrow_array::types::TimestampSecondType>();
        assert_eq!((ts.value(0), s.column(1).as_primitive::<arrow_array::types::Int16Type>().value(0)), (1_704_430_800, 300));
        assert!(a.is_null(1));
        assert!(verify_text(&Target::TimestampWithOffset(TimeUnit::Second), &text, &a).is_ok());
    }

    #[test]
    fn verification_reports_the_first_mismatch() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 3]));
        assert!(first_mismatch(&a, &b).unwrap_err().starts_with("row 1:"));
        let text = LargeStringArray::from(vec!["2300-01-01T00:00:00.123456789"]);
        assert!(from_text(&Target::Fixed(AT::Timestamp(TimeUnit::Nanosecond, None)), &text).is_err()); // beyond i64 ns
    }
```

(2024-01-05T05:00Z = 19,727 days × 86,400 + 18,000 s = 1,704,430,800.)

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib recommend::`
Expected: FAIL to compile.

- [ ] **Step 3: Implement** — append (before `mod tests`):

```rust
// ── cast and verify (Spec B §5.4) ───────────────────────────────────────────

use crate::describe::{parse_decimal, parse_iso};
use arrow_array::types::{Float32Type, Int16Type};
use arrow_array::{
    BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int16Array, StructArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};

fn exact_div(ns: i128, u: &TimeUnit) -> Option<i64> {
    let f = unit_ns(u);
    (ns % f == 0).then(|| ns / f).and_then(|v| i64::try_from(v).ok())
}

fn time_array(u: TimeUnit, v: Vec<Option<i64>>) -> ArrayRef {
    let narrow = || v.iter().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>();
    match u {
        TimeUnit::Second => Arc::new(Time32SecondArray::from(narrow())),
        TimeUnit::Millisecond => Arc::new(Time32MillisecondArray::from(narrow())),
        TimeUnit::Microsecond => Arc::new(Time64MicrosecondArray::from(v)),
        TimeUnit::Nanosecond => Arc::new(Time64NanosecondArray::from(v)),
    }
}

fn timestamp_array(u: TimeUnit, v: Vec<Option<i64>>, tz: Option<Arc<str>>) -> ArrayRef {
    match u {
        TimeUnit::Second => Arc::new(TimestampSecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(v).with_timezone_opt(tz)),
        TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(v).with_timezone_opt(tz)),
    }
}

fn decimal_array(v: Vec<Option<i128>>, scale: i8) -> Result<ArrayRef, String> {
    Decimal128Array::from(v).with_precision_and_scale(38, scale).map(|a| Arc::new(a) as ArrayRef).map_err(|e| e.to_string())
}

/// `f` over every non-null text value; the first value it rejects fails the cast.
fn parsed<T>(text: &LargeStringArray, f: impl Fn(&str) -> Option<T>) -> Result<Vec<Option<T>>, String> {
    text.iter()
        .enumerate()
        .map(|(i, v)| v.map(|s| f(s).ok_or_else(|| format!("row {i}: {s:?} does not convert"))).transpose())
        .collect()
}

/// A string column recast to `t`, built from describe.rs's exact parsers.
pub(crate) fn from_text(t: &Target, text: &LargeStringArray) -> Result<ArrayRef, String> {
    match t {
        Target::Boolean | Target::BoolPair(..) => {
            let (tt, ff) = match t {
                Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
                _ => ("1", "0"),
            };
            let v = parsed(text, |s| {
                let s = s.to_lowercase();
                if s == tt { Some(true) } else if s == ff { Some(false) } else { None }
            })?;
            Ok(Arc::new(BooleanArray::from(v)))
        }
        Target::Fixed(to) => match to {
            AT::Int8 | AT::Int16 | AT::Int32 | AT::Int64 | AT::UInt8 | AT::UInt16 | AT::UInt32 | AT::UInt64 => {
                arrow_cast(decimal_array(parsed(text, |s| parse_decimal(s.as_bytes(), 0))?, 0)?.as_ref(), to)
            }
            AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => {
                arrow_cast(decimal_array(parsed(text, |x| parse_decimal(x.as_bytes(), *s as u32))?, *s)?.as_ref(), to)
            }
            AT::Float32 => Ok(Arc::new(Float32Array::from(parsed(text, |s| s.parse::<f32>().ok())?))),
            AT::Float64 => Ok(Arc::new(Float64Array::from(parsed(text, |s| s.parse::<f64>().ok())?))),
            AT::Date32 => Ok(Arc::new(Date32Array::from(parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                (v.nanos == 0 && v.offset_minutes.is_none()).then_some(())?;
                i32::try_from(v.days?).ok()
            })?))),
            AT::Time32(u) | AT::Time64(u) => Ok(time_array(*u, parsed(text, |s| exact_div(parse_iso(s.as_bytes())?.nanos as i128, u))?)),
            AT::Timestamp(u, tz) => Ok(timestamp_array(*u, parsed(text, |s| exact_div(parse_iso(s.as_bytes())?.epoch_ns(), u))?, tz.clone())),
            t => Err(format!("no string conversion to {}", pa_name(t))),
        },
        Target::TimestampWithOffset(u) => {
            let parts = parsed(text, |s| {
                let v = parse_iso(s.as_bytes())?;
                Some((exact_div(v.epoch_ns(), u)?, i16::try_from(v.offset_minutes?).ok()?))
            })?;
            let ts = timestamp_array(*u, parts.iter().map(|p| Some(p.map_or(0, |p| p.0))).collect(), Some("UTC".into()));
            let off: ArrayRef = Arc::new(Int16Array::from(parts.iter().map(|p| p.map_or(0, |p| p.1)).collect::<Vec<i16>>()));
            let AT::Struct(fields) = timestamp_with_offset(*u) else { unreachable!() };
            StructArray::try_new(fields, vec![ts, off], text.logical_nulls()).map(|a| Arc::new(a) as ArrayRef).map_err(|e| e.to_string())
        }
        Target::Dictionary(..) => arrow_cast(arrow_cast(text, &AT::Utf8)?.as_ref(), &t.arrow_type()),
        Target::Plain(to) => arrow_cast(text, to),
        t => Err(format!("no string conversion to {}", pa_name(&t.arrow_type()))),
    }
}

/// Float → Decimal through the exact digits of each value's shortest round-trip
/// representation (ryu), never multiply-and-round.
pub(crate) fn float_to_decimal(src: &ArrayRef, to: &AT) -> Result<ArrayRef, String> {
    let (AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s)) = to else { return Err("not a decimal".into()) };
    let f32_src = src.data_type() == &AT::Float32;
    let values = arrow_cast(src.as_ref(), &AT::Float64)?;
    let mut buf = ryu::Buffer::new();
    let unscaled: Vec<Option<i128>> = values
        .as_primitive::<Float64Type>()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.map(|x| {
                let repr = if f32_src { buf.format_finite(x as f32).to_string() } else { buf.format_finite(x).to_string() };
                decimal_from_repr(&repr, *s as u32).ok_or_else(|| format!("row {i}: {repr} does not fit scale {s}"))
            })
            .transpose()
        })
        .collect::<Result<_, _>>()?;
    arrow_cast(decimal_array(unscaled, *s)?.as_ref(), to)
}

/// The column recast to `t` (lists are wrapped by `wrap`).
pub(crate) fn cast_to(t: &Target, lvl: &Level) -> Result<ArrayRef, String> {
    let src = &lvl.values;
    match t {
        Target::Original(_) => Ok(src.clone()),
        Target::Null => Ok(arrow_array::new_null_array(&AT::Null, src.len())),
        _ if is_text(lvl.dtype) => from_text(t, &text_of(src)?),
        Target::Fixed(to @ (AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..))) if is_float(lvl.dtype) => float_to_decimal(src, to),
        Target::Boolean | Target::Fixed(_) | Target::Plain(_) => arrow_cast(src.as_ref(), &t.arrow_type()),
        t => Err(format!("no cast to {}", pa_name(&t.arrow_type()))),
    }
}

fn render(a: &ArrayRef, i: usize) -> String {
    arrow_cast(a.slice(i, 1).as_ref(), &AT::Utf8)
        .ok()
        .and_then(|s| {
            let s = s.as_string::<i32>();
            s.is_valid(0).then(|| s.value(0).to_string())
        })
        .unwrap_or_else(|| "null".into())
}

pub(crate) fn first_mismatch(a: &ArrayRef, b: &ArrayRef) -> Result<(), String> {
    if a.to_data() == b.to_data() {
        return Ok(());
    }
    let i = (0..a.len()).find(|&i| a.slice(i, 1).to_data() != b.slice(i, 1).to_data()).unwrap_or(0);
    Err(format!("row {i}: {} round-trips to {}", render(a, i), render(b, i)))
}

/// Float sources: every recast value converts back to the original float (NaN = NaN,
/// -0.0 = 0.0). Decimals come back through their text (correctly rounded parse).
pub(crate) fn verify_float(src: &ArrayRef, recast: &ArrayRef) -> Result<(), String> {
    let f32_src = src.data_type() == &AT::Float32;
    let decimal = matches!(recast.data_type(), AT::Decimal32(..) | AT::Decimal64(..) | AT::Decimal128(..));
    let back: Vec<Option<f64>> = if decimal {
        let t = arrow_cast(recast.as_ref(), &AT::Utf8)?;
        t.as_string::<i32>()
            .iter()
            .map(|v| v.map(|s| if f32_src { s.parse::<f32>().map_or(f64::NAN, f64::from) } else { s.parse::<f64>().unwrap_or(f64::NAN) }))
            .collect()
    } else if f32_src {
        arrow_cast(recast.as_ref(), &AT::Float32)?.as_primitive::<Float32Type>().iter().map(|v| v.map(f64::from)).collect()
    } else {
        arrow_cast(recast.as_ref(), &AT::Float64)?.as_primitive::<Float64Type>().iter().collect()
    };
    let orig = arrow_cast(src.as_ref(), &AT::Float64)?;
    for (i, (a, b)) in orig.as_primitive::<Float64Type>().iter().zip(back).enumerate() {
        if let (Some(a), Some(b)) = (a, b) {
            if !(a == b || (a.is_nan() && b.is_nan())) {
                return Err(format!("row {i}: {a} round-trips to {b}"));
            }
        }
    }
    Ok(())
}

/// String sources: the recast values, rendered to text by arrow-cast, equal the
/// original text by value (canonical digits; parse_iso components).
pub(crate) fn verify_text(t: &Target, text: &LargeStringArray, recast: &ArrayRef) -> Result<(), String> {
    let bad = |i: usize, got: &str| Err(format!("row {i}: {:?} round-trips to {got:?}", text.value(i)));
    let iso = |s: &str| parse_iso(s.as_bytes());
    match t {
        Target::Dictionary(..) | Target::Plain(_) => {
            first_mismatch(&(Arc::new(text.clone()) as ArrayRef), &arrow_cast(recast.as_ref(), &AT::LargeUtf8)?)
        }
        Target::Boolean | Target::BoolPair(..) => {
            let (tt, ff) = match t {
                Target::BoolPair(a, b) => (a.as_str(), b.as_str()),
                _ => ("1", "0"),
            };
            let b = recast.as_boolean();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let want = if b.value(i) { tt } else { ff };
                if text.value(i).to_lowercase() != want {
                    return bad(i, want);
                }
            }
            Ok(())
        }
        Target::TimestampWithOffset(_) => {
            let s = recast.as_struct();
            let back = arrow_cast(s.column(0).as_ref(), &AT::Utf8)?;
            let (back, off) = (back.as_string::<i32>(), s.column(1).as_primitive::<Int16Type>());
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (iso(text.value(i)), iso(back.value(i)));
                let same = matches!((a, b), (Some(a), Some(b)) if a.epoch_ns() == b.epoch_ns() && a.offset_minutes == Some(off.value(i) as i32));
                if !same {
                    return bad(i, back.value(i));
                }
            }
            Ok(())
        }
        Target::Fixed(to) => {
            let back = arrow_cast(recast.as_ref(), &AT::Utf8)?;
            let back = back.as_string::<i32>();
            for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
                let (a, b) = (text.value(i), back.value(i));
                let same = match to {
                    AT::Date32 | AT::Timestamp(..) => matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.epoch_ns() == y.epoch_ns()),
                    AT::Time32(_) | AT::Time64(_) => matches!((iso(a), iso(b)), (Some(x), Some(y)) if x.nanos == y.nanos),
                    _ => canon(a) == canon(b),
                };
                if !same {
                    return bad(i, b);
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Row-by-row check that `recast` holds the original values (Spec B §5.4 step 3).
pub(crate) fn verify(t: &Target, lvl: &Level, recast: &ArrayRef) -> Result<(), String> {
    if matches!(t, Target::Original(_)) {
        return Ok(());
    }
    let src = &lvl.values;
    if recast.len() != src.len() {
        return Err(format!("length {} ≠ {}", recast.len(), src.len()));
    }
    if let Some(i) = (0..src.len()).find(|&i| recast.is_null(i) != src.is_null(i)) {
        return Err(format!("row {i}: null mismatch"));
    }
    match t {
        Target::Null => Ok(()),
        _ if is_text(lvl.dtype) => verify_text(t, &text_of(src)?, recast),
        _ if is_float(lvl.dtype) => verify_float(src, recast),
        _ => first_mismatch(src, &arrow_cast(recast.as_ref(), src.data_type())?),
    }
}

/// Some value's text changed although its value did not (Spec B §5.4).
pub(crate) fn lossy(t: &Target, lvl: &Level, recast: &ArrayRef) -> bool {
    match t {
        Target::Original(_) | Target::Null | Target::Dictionary(..) | Target::Plain(_) => false,
        _ if is_text(lvl.dtype) => match (text_of(&lvl.values), arrow_cast(recast.as_ref(), &AT::LargeUtf8)) {
            (Ok(a), Ok(b)) => a.iter().zip(b.as_string::<i64>().iter()).any(|(x, y)| x != y),
            _ => true, // no text rendering (timestamp_with_offset): the format necessarily changes
        },
        Target::Fixed(AT::Float32 | AT::Float64) => false,
        _ if is_float(lvl.dtype) => arrow_cast(lvl.values.as_ref(), &AT::Float64)
            .map(|f| f.as_primitive::<Float64Type>().iter().flatten().any(|x| x == 0.0 && x.is_sign_negative()))
            .unwrap_or(false),
        _ => false,
    }
}
```

Move the `use` lines of parts 2 and 3 to the top of the file with part 1's if you prefer one import block (either compiles).

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib recommend::`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/recommend.rs
git commit -m "feat: recommend.rs cast from exact parsers and row-by-row verification"
```

---

### Task 14: `recommend.rs` part 4 — choose loop, lists, Polars sizes, plugin entry

**Files:**
- Modify: `services/analytics/src/recommend.rs`

- [ ] **Step 1: Write the failing tests** — append inside `mod tests`:

```rust
    use crate::sizes::sizes;

    fn rec(s: Series) -> Rec {
        let d = describe_one(&s, 0).unwrap();
        let sz = sizes(&s, 1).unwrap();
        recommend(&s, &d, &sz, &params()).unwrap().unwrap()
    }

    fn chosen(r: &Rec) -> &Candidate {
        r.candidates.iter().find(|c| c.outcome == Outcome::Chosen).unwrap()
    }

    #[test]
    fn end_to_end_choices() {
        assert_eq!(rec(Series::new("x".into(), &[0i64, 5, 127])).arrow_type, "uint8");
        let price = rec(Series::new("x".into(), &[123.45f64, 99.99]));
        assert_eq!((price.arrow_type.as_str(), price.polars_type.as_deref()), ("decimal32(5, 2)", Some("Decimal(precision=5, scale=2)")));
        assert_eq!(rec(Series::new("x".into(), &["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"])).arrow_type, "timestamp[ms]");
        let kept = rec(Series::new("x".into(), &[0.1f64, f64::NAN]));
        assert_eq!((kept.arrow_type.as_str(), kept.polars_type.clone()), ("double", None));
    }

    #[test]
    fn failed_cast_falls_back() {
        let r = rec(Series::new("x".into(), &["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"]));
        assert_eq!(r.arrow_type, "string");
        assert!(r.candidates.iter().any(|c| c.outcome == Outcome::Failed && pa_name(&c.target.arrow_type()) == "timestamp[ns]"));
    }

    #[test]
    fn single_item_lists_become_scalars() {
        let s = Series::new("x".into(), [Some(Series::new("".into(), &[1i64])), Some(Series::new("".into(), &[2i64])), None]);
        assert_eq!(rec(s).arrow_type, "uint8");
        let both = Series::new("x".into(), [Some(Series::new("".into(), &[Some(1i64)])), Some(Series::new("".into(), &[None::<i64>])), None]);
        assert_eq!(rec(both).arrow_type, "list<item: uint8>");
    }

    #[test]
    fn predicted_equals_measured() {
        let cases = [
            Series::new("x".into(), &[Some(0i64), Some(5), None]),
            Series::new("x".into(), &[123.45f64, 99.99]),
            Series::new("x".into(), &["a", "b", "a", "b", "a", "b"]),
            Series::new("x".into(), &["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]),
            Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 2])), None]),
        ];
        for s in cases {
            let r = rec(s);
            assert_eq!(chosen(&r).predicted, r.arrow_size, "{}", r.arrow_type);
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib recommend::`
Expected: FAIL to compile (`Rec`, `recommend` undefined).

- [ ] **Step 3: Implement** — append (before `mod tests`):

```rust
// ── choosing (Spec B §4.1, §4.4, §5.4) ──────────────────────────────────────

use crate::describe::{assemble, describe_one, fields, Described, Row};
use crate::shared::to_arrow_rs;
use crate::sizes::{ipc_body_bytes, sizes, Sizes, SIZE_FIELDS};
use arrow_array::{FixedSizeListArray, LargeListArray, ListArray, UInt64Array};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use polars::prelude::{
    polars_err, AnyValue, CompatLevel, Field as PField, Float64Chunked, IntoSeries, NewChunkedArray, PolarsResult, Series, StringChunked,
    StructChunked, UInt64Chunked,
};
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;

pub(crate) struct Chosen {
    pub target: Target,
    pub rank: Rank,
    pub array: ArrayRef,
    pub lossy: bool,
    pub predicted: u64,
    pub projected: f64,
    pub candidates: Vec<Candidate>,
}

/// Candidates in the order tried: rejected last, then smallest projected size, then rank.
fn order(c: &mut [Candidate]) {
    c.sort_by(|a, b| {
        (a.outcome == Outcome::Rejected)
            .cmp(&(b.outcome == Outcome::Rejected))
            .then(a.projected.total_cmp(&b.projected))
            .then(a.rank.cmp(&b.rank))
    });
}

/// Tries the candidates in order; the first success is chosen, failures keep their reason.
fn first_success(mut cands: Vec<Candidate>, mut attempt: impl FnMut(&Target) -> Result<ArrayRef, String>) -> (usize, ArrayRef, Vec<Candidate>) {
    order(&mut cands);
    for i in 0..cands.len() {
        if cands[i].outcome == Outcome::Rejected {
            continue;
        }
        match attempt(&cands[i].target) {
            Ok(a) => {
                cands[i].outcome = Outcome::Chosen;
                return (i, a, cands);
            }
            Err(reason) => {
                cands[i].outcome = Outcome::Failed;
                cands[i].reason = Some(reason);
            }
        }
    }
    unreachable!("the original type is always a candidate and cannot fail")
}

fn chosen_from(i: usize, array: ArrayRef, cands: Vec<Candidate>, lossy: bool) -> Chosen {
    let c = &cands[i];
    Chosen { target: c.target.clone(), rank: c.rank, lossy, predicted: c.predicted, projected: c.projected, array, candidates: cands }
}

pub(crate) fn choose(lvl: &Level, params: &Params) -> Result<Chosen, String> {
    let (i, array, cands) = first_success(candidates(lvl, params)?, |t| {
        let a = cast_to(t, lvl)?;
        verify(t, lvl, &a)?;
        Ok(a)
    });
    let lossy = lossy(&cands[i].target, lvl, &array);
    Ok(chosen_from(i, array, cands, lossy))
}

fn choose_original(lvl: &Level) -> Chosen {
    let mut r = Rules { lvl, shape: lvl.shape(), out: Vec::new() };
    r.original();
    let (i, array, cands) = first_success(r.out, |_| Ok(lvl.values.clone()));
    chosen_from(i, array, cands, false)
}

/// Per row of a List/Array column: (start, len) in its child, or None for a null
/// list; the child sliced to the referenced range; the fixed width (0 for List).
fn list_parts(values: &ArrayRef) -> (Vec<Option<(usize, usize)>>, ArrayRef, i32) {
    match values.data_type() {
        AT::LargeList(_) => {
            let l = values.as_list::<i64>();
            let o = l.value_offsets();
            let first = o[0] as usize;
            let rows = (0..l.len()).map(|i| l.is_valid(i).then(|| (o[i] as usize - first, (o[i + 1] - o[i]) as usize))).collect();
            (rows, l.values().slice(first, o[l.len()] as usize - first), 0)
        }
        AT::FixedSizeList(_, w) => {
            let f = values.as_fixed_size_list();
            let w = *w as usize;
            let rows = (0..f.len()).map(|i| f.is_valid(i).then(|| (i * w, w))).collect();
            (rows, f.values().slice(0, f.len() * w), w as i32)
        }
        t => unreachable!("not a list: {t}"),
    }
}

/// A list column rebuilt around its recast inner values.
fn wrap(t: &Target, values: &ArrayRef, rows: &[Option<(usize, usize)>], inner: &ArrayRef) -> Result<ArrayRef, String> {
    let field = || Arc::new(AField::new("item", inner.data_type().clone(), true));
    let nulls = values.logical_nulls();
    match t {
        Target::Original(_) => Ok(values.clone()),
        Target::Scalar(_) => {
            let idx = UInt64Array::from(rows.iter().map(|r| r.map(|(start, _)| start as u64)).collect::<Vec<_>>());
            arrow_select::take::take(inner.as_ref(), &idx, None).map_err(|e| e.to_string())
        }
        Target::List(_) => {
            let mut offsets = vec![0i32];
            for r in rows {
                let len = i32::try_from(r.map_or(0, |(_, len)| len)).map_err(|e| e.to_string())?;
                offsets.push(offsets.last().unwrap() + len);
            }
            ListArray::try_new(field(), OffsetBuffer::new(ScalarBuffer::from(offsets)), inner.clone(), nulls)
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        Target::FixedList(_, w) => {
            FixedSizeListArray::try_new(field(), *w, inner.clone(), nulls).map(|a| Arc::new(a) as ArrayRef).map_err(|e| e.to_string())
        }
        t => Err(format!("not a list target: {t:?}")),
    }
}

fn candidate(target: Target, rank: Rank, rule: &str, evidence: String, predicted: f64, projected: f64) -> Candidate {
    Candidate { target, rank, rule: rule.into(), evidence, predicted: predicted as u64, projected, outcome: Outcome::NotTried, reason: None }
}

/// Lists: choose the inner type first, then wrap it — as a scalar when every list
/// holds one item, else as a List with 32-bit offsets (Array keeps its width).
fn choose_list(lvl: &Level, inner: &Level, params: &Params) -> Result<Chosen, String> {
    let (rows, _, width) = list_parts(&lvl.values);
    let ic = choose(inner, params)?;
    let (n, nulls, r) = (lvl.n_rows() as f64, lvl.n_null() as f64, lvl.r);
    let inner_t = ic.target.clone();
    let mut outer = Vec::new();
    let nested = matches!(inner_t, Target::Original(_)) && matches!(inner.dtype, PT::List(_) | PT::Array(..) | PT::Struct(_));
    let single = lvl.p.range.min_len == Some(1) && lvl.p.range.max_len == Some(1);
    if single && !(lvl.n_null() > 0 && inner.n_null() > 0) && !nested {
        let shape = Shape { n, nulls: nulls + inner.n_null() as f64, ..inner.shape() };
        let (c, _) = inner.cardinality();
        let t = inner_t.arrow_type();
        outer.push(candidate(
            Target::Scalar(Box::new(inner_t.clone())),
            ic.rank,
            "list→scalar",
            format!("min_len=1 max_len=1 n_null={} inner_n_null={}", lvl.n_null(), inner.n_null()),
            body_size(&t, &shape),
            body_size(&t, &shape.project(r, c)),
        ));
    }
    if width == 0 {
        if inner.n_rows() < 1 << 31 {
            outer.push(candidate(
                Target::List(Box::new(inner_t.clone())),
                Rank::List,
                "large_list→list",
                format!("inner_n_values={}", inner.n_rows()),
                validity(n, nulls) + pad(4.0 * (n + 1.0)) + ic.predicted as f64,
                validity(n * r, nulls * r) + pad(4.0 * (n * r + 1.0)) + ic.projected,
            ));
        }
    } else {
        outer.push(candidate(
            Target::FixedList(Box::new(inner_t.clone()), width),
            Rank::List,
            "array→array",
            format!("width={width}"),
            validity(n, nulls) + ic.predicted as f64,
            validity(n * r, nulls * r) + ic.projected,
        ));
    }
    let mut rules = Rules { lvl, shape: lvl.shape(), out: outer };
    rules.original();
    let (i, array, mut cands) = first_success(rules.out, |t| wrap(t, &lvl.values, &rows, &ic.array));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    let mut chosen = chosen_from(i, array, std::mem::take(&mut cands), lossy);
    chosen.candidates.extend(ic.candidates);
    Ok(chosen)
}

// ── Polars layout of the result (Spec B §5.5) ───────────────────────────────

fn to_polars_layout(a: &ArrayRef, key: &AT) -> Result<ArrayRef, String> {
    let item = |c: &ArrayRef| Arc::new(AField::new("item", c.data_type().clone(), true));
    match a.data_type() {
        AT::Dictionary(..) => {
            let keyed = arrow_cast(a.as_ref(), &AT::Dictionary(Box::new(key.clone()), Box::new(AT::Utf8)))?;
            let d = keyed.as_any_dictionary();
            Ok(d.with_values(arrow_cast(d.values().as_ref(), &AT::Utf8View)?))
        }
        AT::List(f) | AT::LargeList(f) => {
            let large = arrow_cast(a.as_ref(), &AT::LargeList(f.clone()))?;
            let l = large.as_list::<i64>();
            let child = to_polars_layout(l.values(), key)?;
            LargeListArray::try_new(item(&child), l.offsets().clone(), child, l.nulls().cloned())
                .map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| e.to_string())
        }
        AT::FixedSizeList(_, w) => {
            let f = a.as_fixed_size_list();
            let child = to_polars_layout(f.values(), key)?;
            FixedSizeListArray::try_new(item(&child), *w, child, f.nulls().cloned()).map(|x| Arc::new(x) as ArrayRef).map_err(|e| e.to_string())
        }
        AT::Struct(fields) => {
            let s = a.as_struct();
            let cols = s.columns().iter().map(|c| to_polars_layout(c, key)).collect::<Result<Vec<_>, _>>()?;
            let fields: Fields = fields.iter().zip(&cols).map(|(f, c)| Arc::new(AField::new(f.name(), c.data_type().clone(), f.is_nullable()))).collect();
            StructArray::try_new(fields, cols, s.nulls().cloned()).map(|x| Arc::new(x) as ArrayRef).map_err(|e| e.to_string())
        }
        t => arrow_cast(a.as_ref(), &polars_layout(t, key)),
    }
}

fn dictionary_values(a: &ArrayRef) -> Option<Vec<String>> {
    match a.data_type() {
        AT::Dictionary(..) => {
            let v = arrow_cast(a.as_any_dictionary().values().as_ref(), &AT::Utf8).ok()?;
            Some(v.as_string::<i32>().iter().map(|x| x.unwrap_or_default().to_string()).collect())
        }
        AT::List(_) => dictionary_values(a.as_list::<i32>().values()),
        AT::FixedSizeList(..) => dictionary_values(a.as_fixed_size_list().values()),
        _ => None,
    }
}

// ── one column ───────────────────────────────────────────────────────────────

pub(crate) struct Rec {
    pub nullable: bool,
    pub arrow_type: String,
    pub arrow_size: u64,
    pub arrow_zstd: u64,
    /// None when the original type is kept: Python fills in `str(dtype)`.
    pub polars_type: Option<String>,
    pub polars_size: u64,
    pub polars_zstd: u64,
    pub lossy: bool,
    pub candidates: Vec<Candidate>,
}

/// The recommendation for one column; None when its sizes are null (nested Int128).
pub(crate) fn recommend(s: &Series, d: &Described, sz: &Sizes, params: &Params) -> PolarsResult<Option<Rec>> {
    let (Some(size_bytes), Some(polars_bytes), Some(polars_zstd)) = (sz[0], sz[2], sz[3]) else { return Ok(None) };
    let name = s.name().as_str();
    let err = |e: String| polars_err!(ComputeError: "recommend {name}: {e}");
    let values = to_arrow_rs(s, CompatLevel::oldest())?;
    let q = params.population_rows.map(|p| if p == d.n_rows { 1.0 } else { d.n_rows as f64 / p as f64 });
    let r = match params.population_rows {
        Some(p) if d.n_rows > 0 => p as f64 / d.n_rows as f64,
        _ => 1.0,
    };
    let outer = Level {
        column: name, dtype: s.dtype(), values: values.clone(), p: &d.outer, n_midnight: d.n_midnight, size_bytes,
        est: level_estimate(&d.outer, d.n_rows - d.n_null, q), r, prefix: "",
    };
    let chosen = match &d.inner {
        Some(inner) if matches!(s.dtype(), PT::List(_) | PT::Array(..)) => {
            let (_, child, _) = list_parts(&values);
            if child.len() == inner.values.len() {
                let inner_lvl = Level {
                    column: name, dtype: inner.values.dtype(), size_bytes: ipc_body_bytes(child.as_ref(), None)?, values: child,
                    p: &inner.profile, n_midnight: None,
                    est: level_estimate(&inner.profile, (inner.values.len() - inner.values.null_count()) as u64, q), r, prefix: "inner: ",
                };
                choose_list(&outer, &inner_lvl, params).map_err(err)?
            } else {
                choose_original(&outer) // null lists hold values: keep the column as it is
            }
        }
        _ => choose(&outer, params).map_err(err)?,
    };
    let t = chosen.array.data_type().clone();
    let (polars_type, polars_size, polars_zstd) = if matches!(chosen.target, Target::Original(_)) {
        (None, polars_bytes, polars_zstd)
    } else {
        let key = chosen.target.polars_key().unwrap_or(AT::UInt32);
        let layout = to_polars_layout(&chosen.array, &key).map_err(err)?;
        let enum_values = if q == Some(1.0) { dictionary_values(&chosen.array) } else { None };
        (
            Some(pl_name(&t, name, enum_values.as_deref(), &key)),
            ipc_body_bytes(layout.as_ref(), None)?,
            ipc_body_bytes(layout.as_ref(), Some(params.zstd_level))?,
        )
    };
    Ok(Some(Rec {
        nullable: d.n_null > 0,
        arrow_type: pa_name(&t),
        arrow_size: ipc_body_bytes(chosen.array.as_ref(), None)?,
        arrow_zstd: ipc_body_bytes(chosen.array.as_ref(), Some(params.zstd_level))?,
        polars_type,
        polars_size,
        polars_zstd,
        lossy: chosen.lossy,
        candidates: chosen.candidates,
    }))
}

// ── plugin entry ─────────────────────────────────────────────────────────────

fn rec_fields() -> Vec<(String, PT)> {
    let candidate = PT::Struct(vec![
        PField::new("arrow_type".into(), PT::String),
        PField::new("rule".into(), PT::String),
        PField::new("evidence".into(), PT::String),
        PField::new("predicted_bytes".into(), PT::UInt64),
        PField::new("projected_population_bytes".into(), PT::Float64),
        PField::new("outcome".into(), PT::String),
        PField::new("reason".into(), PT::String),
    ]);
    [
        ("rec_nullable", PT::Boolean),
        ("rec_arrow_type", PT::String),
        ("rec_arrow_size_bytes", PT::UInt64),
        ("rec_arrow_size_zstd_bytes", PT::UInt64),
        ("rec_polars_type", PT::String),
        ("rec_polars_size_bytes", PT::UInt64),
        ("rec_polars_size_zstd_bytes", PT::UInt64),
        ("rec_lossy_formatting", PT::Boolean),
        ("rec_candidates", PT::List(Box::new(candidate))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}

fn output_fields() -> Vec<(String, PT)> {
    let mut f = fields();
    f.extend(SIZE_FIELDS.iter().map(|n| (n.to_string(), PT::UInt64)));
    f.extend(rec_fields());
    f
}

fn recommend_output_type(_input_fields: &[PField]) -> PolarsResult<PField> {
    let fields = output_fields().into_iter().map(|(n, d)| PField::new(n.into(), d)).collect();
    Ok(PField::new("recommend".into(), PT::Struct(fields)))
}

fn candidates_series(c: &[Candidate]) -> Series {
    let text = |name: &str, v: Vec<Option<String>>| StringChunked::from_iter_options(name.into(), v.into_iter()).into_series();
    let cols = [
        text("arrow_type", c.iter().map(|x| Some(pa_name(&x.target.arrow_type()))).collect()),
        text("rule", c.iter().map(|x| Some(x.rule.clone())).collect()),
        text("evidence", c.iter().map(|x| Some(x.evidence.clone())).collect()),
        UInt64Chunked::from_iter_values("predicted_bytes".into(), c.iter().map(|x| x.predicted)).into_series(),
        Float64Chunked::from_iter_values("projected_population_bytes".into(), c.iter().map(|x| x.projected)).into_series(),
        text("outcome", c.iter().map(|x| Some(x.outcome.name().to_string())).collect()),
        text("reason", c.iter().map(|x| x.reason.clone()).collect()),
    ];
    StructChunked::from_series("candidate".into(), c.len(), cols.iter()).expect("equal-length fields").into_series()
}

fn rec_row(rec: Option<&Rec>) -> Row {
    let Some(r) = rec else { return vec![AnyValue::Null; 9] };
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

pub(crate) fn describe_and_recommend_impl(inputs: &[Series], params: &Params) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs
        .par_iter()
        .map(|s| {
            let d = describe_one(s, params.seed)?;
            let sz = sizes(s, params.zstd_level)?;
            let rec = recommend(s, &d, &sz, params)?;
            let mut row = d.row();
            row.extend(sz.iter().map(|v| v.map_or(AnyValue::Null, AnyValue::UInt64)));
            row.extend(rec_row(rec.as_ref()));
            Ok(row)
        })
        .collect::<PolarsResult<_>>()?;
    assemble("recommend", &output_fields(), &rows)
}

#[polars_expr(output_type_func=recommend_output_type)]
fn describe_and_recommend(inputs: &[Series], kwargs: Params) -> PolarsResult<Series> {
    describe_and_recommend_impl(inputs, &kwargs)
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib recommend::`
Expected: PASS (19 tests). Then run the whole library: `cargo test --lib` — Expected: PASS, no warnings about unused items in `recommend.rs` / `cardinality_estimators.rs`.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/recommend.rs
git commit -m "feat: describe_and_recommend plugin - choose loop, lists, Polars sizes"
```

---

### Task 15: Python package `analytics.recommend` and the plugin wrapper

**Files:**
- Create: `services/analytics/analytics/recommend/__init__.py`, `base.py`, `rust.py`
- Modify: `services/analytics/analytics/_plugin.py`
- Test: `tests/test_recommend.py`

- [ ] **Step 1: Write the failing contract test** — create `tests/test_recommend.py`:

```python
"""
recommend accuracy tests — RecommendRust, the only implementation.

Oracles: hand-worked known answers; the pyarrow and Polars casts of each column to
the recommended types, measured by the pyarrow IPC oracle; predicted = measured;
the Python cardinality estimators. Accuracy only — nothing here is timed.
"""

import re
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from pytest import approx

from analytics.describe import _sizes
from datagen import describe_mixed, stringified
from harness import assert_contract, load, run

PKG = "analytics.recommend"
LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


def impl():
    return load(f"{PKG}:RecommendRust")


def rec(s: pl.Series, **params) -> dict:
    """recommend one Series; its result row as a dict."""
    return run(impl(), {"t": s.to_frame()}, **params).row(0, named=True)


def by_type(r: dict) -> dict:
    return {c["arrow_type"]: c for c in r["rec_candidates"]}


def ineligible_frame() -> pl.DataFrame:
    cols = [pl.Series("obj", [object(), object()], dtype=pl.Object), pl.Series("nul", [None, None], dtype=pl.Null)]
    return pl.DataFrame(cols)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract

def test_contract():
    frames = {"mixed": describe_mixed(300), "empty": describe_mixed(50).clear(), "bad": ineligible_frame()}
    cls = impl()
    result = run(cls, frames)
    assert_contract(cls, result, frames)
    computed = result.filter(pl.col("status") == "computed")
    assert computed["rec_arrow_type"].null_count() == 0  # describe_mixed nests no Int128
    assert computed["rec_polars_type"].null_count() == 0
    assert set(result.filter(pl.col("df_a") == "bad")["status"]) == {"ineligible"}


def test_constructor_validates_boolean_pairs():
    cls = impl()
    with pytest.raises(ValueError):
        cls(boolean_pairs=(("yes", "YES"),))
    with pytest.raises(ValueError):
        cls(boolean_pairs=(("y",),))


def test_population_rows_below_frame_rows_raise():
    with pytest.raises(ValueError):
        rec(pl.Series("x", [1, 2, 3]), population_rows=2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_recommend.py -q`
Expected: FAIL (`ModuleNotFoundError: analytics.recommend`, shown as a skip-with-reason by `load` → make sure the three tests FAIL or SKIP naming the missing module).

- [ ] **Step 3: Plugin wrapper** — append to `analytics/_plugin.py`:

```python
"""
Recommend
"""


def describe_and_recommend(
    df: pl.DataFrame,
    *,
    seed: int,
    zstd_level: int,
    population_rows: int | None,
    categorical_threshold: int,
    boolean_pairs: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    """One row per column of `df`: `column`, every Describe metric, the size metrics
    and the recommendation columns (see analytics.recommend.base). Private — called
    only by RecommendRust."""
    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="describe_and_recommend",
            args=df.get_columns(),
            kwargs={
                "seed": seed,
                "zstd_level": zstd_level,
                "population_rows": population_rows,
                "categorical_threshold": categorical_threshold,
                "boolean_pairs": [list(p) for p in boolean_pairs],
            },
            is_elementwise=False,
            changes_length=True,
        ).alias("recommend")
    ).unnest("recommend")
```

- [ ] **Step 4: The package** — `analytics/recommend/base.py`:

```python
"""Narrowest value-preserving Arrow type per column, cast, verified and measured
(Spec B: docs/superpowers/specs/2026-09-26-recommend-technique-design.md).

Recommend is Describe plus recommendation metrics: its implementations fill
Describe's METRICS and REC_METRICS; Describe's conclusions are unchanged.
"""

from __future__ import annotations

import polars as pl

from analytics.describe.base import Describe

OUTCOME = pl.Enum(["chosen", "failed", "rejected", "not_tried"])
CANDIDATE = pl.Struct(
    {
        "arrow_type": pl.String,
        "rule": pl.String,
        "evidence": pl.String,
        "predicted_bytes": pl.UInt64,
        "projected_population_bytes": pl.Float64,
        "outcome": OUTCOME,
        "reason": pl.String,
    }
)
REC_METRICS = {
    "rec_nullable": pl.Boolean,
    "rec_arrow_type": pl.String,
    "rec_arrow_size_bytes": pl.UInt64,
    "rec_arrow_size_zstd_bytes": pl.UInt64,
    "rec_polars_type": pl.String,
    "rec_polars_size_bytes": pl.UInt64,
    "rec_polars_size_zstd_bytes": pl.UInt64,
    "rec_lossy_formatting": pl.Boolean,
    "rec_candidates": pl.List(CANDIDATE),
}


class Recommend(Describe):
    """Describe, then per column: candidate Arrow types from the type hierarchy
    (Spec B §4) and dictionary encoding for strings (§5.2), each with a predicted
    IPC size; tried smallest projected population size first (ties: hierarchy
    rank), cast, verified row by row and measured. `rec_arrow_type` is pyarrow's
    spelling, `rec_polars_type` Python's `str(dtype)` of the equivalent Polars type;
    sizes are Arrow IPC bodies (Polars: its native layout), plain and ZSTD.
    `rec_candidates` lists every candidate with the rule, the metric values it
    tested, its predicted / projected size and its outcome.

    `boolean_pairs`: (true text, false text) pairs a two-valued string column may
    map to Boolean, compared case-insensitively.
    """

    METRICS = {**Describe.METRICS, **REC_METRICS}

    def __init__(self, *, boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),), **describe_params) -> None:
        super().__init__(**describe_params)
        pairs = tuple(tuple(p) for p in boolean_pairs)
        if any(len(p) != 2 or not all(isinstance(v, str) and v for v in p) or p[0].lower() == p[1].lower() for p in pairs):
            raise ValueError(f"boolean_pairs must be pairs of distinct non-empty strings, got {boolean_pairs!r}")
        self.boolean_pairs = pairs
```

`analytics/recommend/rust.py`:

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.recommend.base import Recommend


class RecommendRust(Recommend):
    """Rust plugin `describe_and_recommend`: Describe's metrics and sizes, then the
    candidate types cast, verified and measured with arrow-rs (src/recommend.rs)."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            pop = self._population(frame)
            if pop is not None and pop < df.height:
                raise ValueError(f"population_rows {pop} < {df.height} rows in frame {frame!r}")
            out = _plugin.describe_and_recommend(
                df,
                seed=self.seed,
                zstd_level=self.zstd_level,
                population_rows=pop,
                categorical_threshold=self.categorical_threshold,
                boolean_pairs=self.boolean_pairs,
            )
            for r in out.iter_rows(named=True):
                if r["rec_arrow_type"] is not None and r["rec_polars_type"] is None:  # the original type was kept
                    r["rec_polars_type"] = str(df.schema[r["column"]])
                rows[frame, r["column"]] = r
        return self.metrics_frame(combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS})
```

`analytics/recommend/__init__.py`:

```python
"""Narrowest value-preserving Arrow type per column, cast, verified and measured. See Recommend."""

from analytics.recommend.base import Recommend
from analytics.recommend.rust import RecommendRust

REFERENCE = "RecommendRust"
IMPLEMENTATIONS = ("RecommendRust",)

__all__ = ["Recommend", "RecommendRust", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 5: Build and run**

Run: build; `$PY -m pytest tests/test_recommend.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add services/analytics/analytics/recommend services/analytics/analytics/_plugin.py tests/test_recommend.py
git commit -m "feat: analytics.recommend - Recommend, RecommendRust"
```

---

### Task 16: Known-answer tests

**Files:**
- Modify: `tests/test_recommend.py`

- [ ] **Step 1: Add the tests** (they exercise code already built, so they should pass; any failure is a real bug — fix it in `recommend.rs` with a Rust unit test reproducing it first)

```python
# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers

LONDON = [datetime(2024, 1, 5), datetime(2024, 1, 6)]
OFFSETS = "struct<timestamp: timestamp[s, tz=UTC] not null, offset_minutes: int16 not null>"

KNOWN = [
    pytest.param(pl.Series("x", [0, 1, 1, None]), {}, "bool", "Boolean", id="int_0_1"),
    pytest.param(pl.Series("x", [0, 5, 127]), {}, "uint8", "UInt8", id="uint_before_int_on_tie"),
    pytest.param(pl.Series("x", [-5, 100]), {}, "int8", "Int8", id="int8"),
    pytest.param(pl.Series("x", [-200, 5]), {}, "int16", "Int16", id="int16"),
    pytest.param(pl.Series("x", [10**20, -1], dtype=pl.Int128), {}, "decimal128(21, 0)", "Decimal(precision=21, scale=0)", id="int128_beyond_64_bits"),
    pytest.param(pl.Series("x", [Decimal("1.20"), Decimal("3.40")], dtype=pl.Decimal(10, 2)), {}, "decimal32(2, 1)", "Decimal(precision=2, scale=1)", id="decimal_scale_by_gcd"),
    pytest.param(pl.Series("x", [Decimal("1.00"), Decimal("300.00")], dtype=pl.Decimal(10, 2)), {}, "uint16", "UInt16", id="decimal_to_integer"),
    pytest.param(pl.Series("x", [123.45, 99.99]), {}, "decimal32(5, 2)", "Decimal(precision=5, scale=2)", id="float_to_decimal32"),
    pytest.param(pl.Series("x", [1234567.891, 2.5]), {}, "decimal64(10, 3)", "Decimal(precision=10, scale=3)", id="float_to_decimal64"),
    pytest.param(pl.Series("x", [1.0, 2.0, 300.0]), {}, "uint16", "UInt16", id="whole_floats"),
    pytest.param(pl.Series("x", [0.0009765625, 0.5]), {}, "float", "Float32", id="float32_beats_decimal64"),
    pytest.param(pl.Series("x", [0.5, 0.25]), {}, "decimal32(2, 2)", "Decimal(precision=2, scale=2)", id="decimal32_beats_float32_on_tie"),
    pytest.param(pl.Series("x", [1.5, float("nan")]), {}, "float", "Float32", id="nan_keeps_a_float"),
    pytest.param(pl.Series("x", [0.1, float("nan")]), {}, "double", "Float64", id="original_float64"),
    pytest.param(pl.Series("x", [1e10, 1e-9]), {}, "double", "Float64", id="float64_needing_decimal128_kept"),
    pytest.param(pl.Series("x", ["007", "12"]), {}, "string", "String", id="leading_zero_stays_string"),
    pytest.param(pl.Series("x", ["1.50", "2.2"]), {}, "decimal32(2, 1)", "Decimal(precision=2, scale=1)", id="string_decimal"),
    pytest.param(pl.Series("x", ["1234567890.1", "0.00000012345"]), {}, "double", "Float64", id="varying_places_float64"),
    pytest.param(pl.Series("x", ["1.5", "0.00000000000000000012"]), {}, "float", "Float32", id="varying_places_float32"),
    pytest.param(pl.Series("x", ["12345678901234567.8", "0.12"]), {}, "decimal128(19, 2)", "Decimal(precision=19, scale=2)", id="over_15_significant_digits"),
    pytest.param(pl.Series("x", ["12345678901234567.89", "0.12"]), {}, "decimal128(19, 2)", "Decimal(precision=19, scale=2)", id="fixed_places_decimal128"),
    pytest.param(pl.Series("x", ["true", "False", None]), {}, "bool", "Boolean", id="boolean_pair_default"),
    pytest.param(pl.Series("x", ["Y", "n", "y"]), {"boolean_pairs": (("true", "false"), ("y", "n"))}, "bool", "Boolean", id="boolean_pair_y_n"),
    pytest.param(pl.Series("x", ["2024-01-05", "2024-02-29"]), {}, "date32[day]", "Date", id="iso_date"),
    pytest.param(pl.Series("x", ["10:00", "23:59:30"]), {}, "time32[s]", "Time", id="iso_time"),
    pytest.param(pl.Series("x", ["2024-01-05 10:00:00.120", "2024-01-06T11:00:00"]), {}, "timestamp[ms]", "Datetime(time_unit='ms', time_zone=None)", id="iso_naive"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00:00.000", "2024-01-05T11:30:00.000"]), {}, "timestamp[s]", "Datetime(time_unit='ms', time_zone=None)", id="iso_zero_fraction"),
    pytest.param(pl.Series("x", ["2024-01-05T00:00:00", "2024-01-06 00:00"]), {}, "date32[day]", "Date", id="iso_midnights"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-06T11:00:00+05:00"]), {}, "timestamp[s, tz=+05:00]", "Datetime(time_unit='ms', time_zone='+05:00')", id="iso_fixed_offset"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00Z", "2024-01-05T11:00+00:00"]), {}, "timestamp[s, tz=UTC]", "Datetime(time_unit='ms', time_zone='UTC')", id="iso_utc"),
    pytest.param(pl.Series("x", ["2024-01-05T10:00+05:00", "2024-01-05T10:00-03:30"]), {}, OFFSETS, "Struct({'timestamp': Datetime(time_unit='ms', time_zone='UTC'), 'offset_minutes': Int16})", id="iso_varying_offsets"),
    pytest.param(pl.Series("x", LONDON, dtype=pl.Datetime("us")), {}, "date32[day]", "Date", id="naive_midnights"),
    pytest.param(pl.Series("x", LONDON, dtype=pl.Datetime("us")).dt.replace_time_zone("Europe/London"), {}, "timestamp[s, tz=Europe/London]", "Datetime(time_unit='ms', time_zone='Europe/London')", id="tz_aware_midnights"),
    pytest.param(pl.Series("x", [datetime(2024, 1, 5, 10, 0, 0, 120_000)], dtype=pl.Datetime("us")), {}, "timestamp[ms]", "Datetime(time_unit='ms', time_zone=None)", id="datetime_unit_by_gcd"),
    pytest.param(pl.Series("x", [timedelta(seconds=5), timedelta(minutes=1)], dtype=pl.Duration("us")), {}, "duration[s]", "Duration(time_unit='ms')", id="duration_unit_by_gcd"),
    pytest.param(pl.Series("x", [time(10, 0), time(11, 30, 15)]), {}, "time32[s]", "Time", id="time_unit_by_gcd"),
    pytest.param(pl.Series("x", [[1], [2], None]), {}, "uint8", "UInt8", id="single_item_lists"),
    pytest.param(pl.Series("x", [[1], [None], None]), {}, "list<item: uint8>", "List(UInt8)", id="null_lists_and_elements_stay_lists"),
    pytest.param(pl.Series("x", [[1, 2], [3]]), {}, "list<item: uint8>", "List(UInt8)", id="large_list_to_list"),
    pytest.param(pl.Series("x", [None, None], dtype=pl.String), {}, "null", "Null", id="all_null"),
]


@pytest.mark.parametrize("s, params, arrow_type, polars_type", KNOWN)
def test_known_answers(s, params, arrow_type, polars_type):
    r = rec(s, **params)
    assert (r["rec_arrow_type"], r["rec_polars_type"]) == (arrow_type, polars_type)
    assert r["rec_nullable"] == (s.null_count() > 0)


def test_lossy_formatting():
    assert rec(pl.Series("x", ["1.50", "2.2"]))["rec_lossy_formatting"] is True
    assert rec(pl.Series("x", ["1.5", "2.2"]))["rec_lossy_formatting"] is False
    negative_zero = rec(pl.Series("x", [-0.0, 1.0, 1.0]))
    assert (negative_zero["rec_arrow_type"], negative_zero["rec_lossy_formatting"]) == ("bool", True)
    assert rec(pl.Series("x", [0, 1]))["rec_lossy_formatting"] is False
    assert rec(pl.Series("x", ["true", "False"]))["rec_lossy_formatting"] is True


def test_candidates_report_rule_evidence_and_outcomes():
    r = rec(pl.Series("x", [1e10, 1e-9]))
    c = by_type(r)
    assert c["double"]["outcome"] == "chosen"
    assert c["decimal128(20, 9)"]["outcome"] == "not_tried"
    assert "max_frac_digits=9" in c["decimal128(20, 9)"]["evidence"]
    tried = [x["projected_population_bytes"] for x in r["rec_candidates"] if x["outcome"] != "rejected"]
    assert tried == sorted(tried)


def test_failed_cast_falls_back_to_the_next_candidate():
    r = rec(pl.Series("x", ["2300-01-01T00:00:00.123456789", "2024-01-05T10:00:00"]))
    assert r["rec_arrow_type"] == "string"
    failed = by_type(r)["timestamp[ns]"]
    assert failed["outcome"] == "failed" and failed["reason"]


def test_dictionary_polars_types():
    s = pl.Series("x", ["a", "b"] * 50)
    exact = rec(s, population_rows=100)
    assert exact["rec_arrow_type"] == "dictionary<values=string, indices=uint8, ordered=0>"
    assert exact["rec_polars_type"] == "Enum(categories=['a', 'b'])"
    assert rec(s)["rec_polars_type"] == 'Categorical(Categories(name="x", namespace="", physical=pl.UInt8))'


@pytest.mark.parametrize(
    "d, arrow_key",
    [
        pytest.param(255, "uint8", id="255"),
        pytest.param(256, "uint8", id="256"),
        pytest.param(257, "uint16", id="257"),
        pytest.param(65_536, "uint16", id="65536", marks=pytest.mark.slow),
        pytest.param(65_537, "uint32", id="65537", marks=pytest.mark.slow),
    ],
)
def test_dictionary_key_widths(d, arrow_key):
    s = pl.Series("x", [f"v{i:05d}" for i in range(d)] * 2 if d > 1_000 else [f"v{i:05d}" for i in range(d)] * 40)
    r = rec(s, population_rows=s.len(), categorical_threshold=100_000)
    assert r["rec_arrow_type"] == f"dictionary<values=string, indices={arrow_key}, ordered=0>"
    # Polars reserves one key code: its own width (checked by casting) must match the measured layout.
    polars = s.cast(pl.Enum(s.unique(maintain_order=True).to_list()))
    assert r["rec_polars_size_bytes"] == _sizes.column_sizes(polars, 1)["size_polars_bytes"]


def test_dictionary_gate_rejects_above_threshold():
    r = rec(pl.Series("x", ["a", "b"] * 500), population_rows=1_000, categorical_threshold=1)
    assert r["rec_arrow_type"] == "string"
    dictionary = next(c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary"))
    assert dictionary["outcome"] == "rejected" and "categorical_threshold=1" in dictionary["reason"]


def test_population_projection_chooses_plain_over_dictionary():
    # 600 singletons, 200 doubletons, 200 values × 5: n = 2,000, d = 1,000 (d/n = 0.5, so
    # Chao1 — est_high ≈ 2,115 — sizes the dictionary); 30-byte values.
    ids = [f"id-{i:027d}" for i in range(1_000)]
    s = pl.Series("x", ids[:600] + ids[600:800] * 2 + ids[800:] * 5)
    r = rec(s)
    plain = by_type(r)["string"]
    dictionary = next(c for c in r["rec_candidates"] if c["arrow_type"].startswith("dictionary"))
    assert dictionary["predicted_bytes"] < plain["predicted_bytes"]  # smaller on the frame …
    assert dictionary["projected_population_bytes"] > plain["projected_population_bytes"]  # … larger in the population
    assert r["rec_arrow_type"] == "string"
```

- [ ] **Step 2: Run them**

Run: `$PY -m pytest tests/test_recommend.py -q` (and once with `-m slow -k key_widths`)
Expected: PASS. A mismatch in a type string is a real finding: check the Rust rule against Spec B §4 before touching the expectation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_recommend.py
git commit -m "test: recommend known answers"
```

---

### Task 17: Oracle tests — pyarrow, Polars, predicted = measured, estimators

**Files:**
- Modify: `tests/test_recommend.py`

- [ ] **Step 1: Add the oracles**

```python
# ─────────────────────────────────────────────────────────────────────────────
# 2. Oracles (single implementation: no reference agreement)

_SIMPLE = {
    "null": pa.null(), "bool": pa.bool_(), "float": pa.float32(), "double": pa.float64(),
    "string": pa.string(), "large_string": pa.large_string(), "binary": pa.binary(), "large_binary": pa.large_binary(),
    "date32[day]": pa.date32(),
    **{n: getattr(pa, n)() for n in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")},
}


def pa_type(name: str) -> pa.DataType:
    """pyarrow type from the `str(type)` spellings recommend emits."""
    if name in _SIMPLE:
        return _SIMPLE[name]
    if m := re.fullmatch(r"decimal(32|64|128)\((\d+), (\d+)\)", name):
        return getattr(pa, f"decimal{m[1]}")(int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(time32|time64|duration)\[(\w+)\]", name):
        return getattr(pa, m[1])(m[2])
    if m := re.fullmatch(r"timestamp\[(\w+)(?:, tz=(.+))?\]", name):
        return pa.timestamp(m[1], m[2])
    if m := re.fullmatch(r"large_list<item: (.+)>", name):
        return pa.large_list(pa_type(m[1]))
    if m := re.fullmatch(r"list<item: (.+)>", name):
        return pa.list_(pa_type(m[1]))
    if m := re.fullmatch(r"fixed_size_list<item: (.+)>\[(\d+)\]", name):
        return pa.list_(pa_type(m[1]), int(m[2]))
    if m := re.fullmatch(r"dictionary<values=(.+), indices=(\w+), ordered=0>", name):
        return pa.dictionary(pa_type(m[2]), pa_type(m[1]))
    raise NotImplementedError(name)


def pl_dtype(name: str) -> pl.DataType:
    """Polars dtype from its Python `str(dtype)` spelling."""
    return eval(name, {"__builtins__": {}}, {"pl": pl, **{k: getattr(pl, k) for k in dir(pl) if k[:1].isupper()}})


def _string_source(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.List, pl.Array)):
        return _string_source(dtype.inner)
    return isinstance(dtype, (pl.String, pl.Categorical, pl.Enum))


ORACLE_FRAMES = [
    pytest.param(lambda: {"mixed": describe_mixed(2_000)}, id="mixed"),
    pytest.param(lambda: {"strings": stringified(describe_mixed(500))}, id="stringified"),
    pytest.param(lambda: {"large": pl.read_ipc(LARGE)}, id="large", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("make", ORACLE_FRAMES)
def test_sizes_match_pyarrow_and_polars_casts(make):
    frames = make()
    result = run(impl(), frames)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        s = frames[r["df_a"]][r["col_a"]]
        chosen = next(c for c in r["rec_candidates"] if c["outcome"] == "chosen")
        assert chosen["predicted_bytes"] == r["rec_arrow_size_bytes"], r["col_a"]
        if chosen["rule"].endswith("original"):
            assert (r["rec_arrow_size_bytes"], r["rec_polars_size_bytes"]) == (r["size_bytes"], r["size_polars_bytes"])
            continue
        skippable = _string_source(s.dtype) or (isinstance(s.dtype, (pl.List, pl.Array)) and "list" not in r["rec_arrow_type"])
        try:
            arrow = _sizes._to_arrow(s.rechunk(), pl.CompatLevel.oldest()).cast(pa_type(r["rec_arrow_type"]), safe=False)
            polars = s.cast(pl_dtype(r["rec_polars_type"]))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pl.exceptions.PolarsError, NotImplementedError):
            assert skippable, f"{r['col_a']}: pyarrow/Polars cannot cast a non-string column to {r['rec_arrow_type']}"
            continue
        if arrow.null_count != s.null_count() or polars.null_count() != s.null_count():
            assert skippable, f"{r['col_a']}: the library cast lost values"
            continue
        level = 1
        assert _sizes.ipc_body_bytes(arrow, None) == r["rec_arrow_size_bytes"], r["col_a"]
        assert _sizes.ipc_body_bytes(arrow, level) == approx(r["rec_arrow_size_zstd_bytes"], rel=0.01, abs=16), r["col_a"]
        native = _sizes.column_sizes(polars, level)
        assert native["size_polars_bytes"] == r["rec_polars_size_bytes"], r["col_a"]
        assert native["size_polars_zstd_bytes"] == approx(r["rec_polars_size_zstd_bytes"], rel=0.01, abs=16), r["col_a"]
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("population_rows", [None, 1_000_000])
def test_rust_cardinality_matches_python_estimators(population_rows):
    frames = {"mixed": describe_mixed(2_000), "strings": stringified(describe_mixed(500))}
    result = run(impl(), frames, population_rows=population_rows)
    checked = 0
    for r in result.filter(pl.col("status") == "computed").iter_rows(named=True):
        for c in r["rec_candidates"]:
            m = re.search(r"c=(\S+) from (est_high|est_cardinality)", c["evidence"] or "")
            if m is None:
                continue
            prefix = "inner_" if c["rule"].startswith("inner: ") else ""
            assert float(m[1]) == approx(r[f"{prefix}{m[2]}"], rel=1e-9), (r["col_a"], c["rule"])
            checked += 1
    assert checked > 0
```

(`population_rows=1_000_000` applies to both frames, each smaller than a million rows, so Duj1 is exercised — `est_high` is null and `est_cardinality` is used.)

- [ ] **Step 2: Run them**

Run: `$PY -m pytest tests/test_recommend.py -q` then `$PY -m pytest tests/test_recommend.py -q -m slow`
Expected: PASS. A size mismatch on a non-string column means `recommend.rs` and pyarrow disagree on the recast data or its layout — reproduce it with one Series and a Rust unit test before fixing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_recommend.py
git commit -m "test: recommend oracles - pyarrow/Polars casts, predicted = measured, estimators"
```

---

### Task 18: Benchmark

**Files:**
- Create: `tests/performance/benchmark_recommend.py`
- Modify (only if needed): `tests/performance/harness.py`

- [ ] **Step 1: Write the benchmark**

```python
"""
recommend speed benchmark: RecommendRust only — there is no Python implementation,
so the report shows the parallel speedup (Rust@1 thread ÷ Rust@N threads).

Run: /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_recommend.py

Shapes as benchmark_describe.py: large_dataset.arrow; narrow 10M x 4; wide 1M x 100; nested 1M x 3.
"""

from harness import Dataset, large_dataset, run

from datagen import describe_narrow, describe_nested, describe_wide

if __name__ == "__main__":
    run(
        "analytics.recommend",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", lambda: {"t": describe_narrow(10_000_000)}),
            Dataset("wide 1M x 100", lambda: {"t": describe_wide(1_000_000, 100)}),
            Dataset("nested 1M x 3", lambda: {"t": describe_nested(1_000_000)}),
        ],
    )
```

- [ ] **Step 2: Run it**

Run: `$PY tests/performance/benchmark_recommend.py`
Expected: a report row per dataset for `RecommendRust` with `parallel` filled and `algorithmic` / `total` blank (`harness.speedups` leaves them None when no non-Rust implementation ran). If the harness raises because the package has a single implementation, fix `tests/performance/harness.py` minimally (e.g. skip the agreement comparison when `name == module.REFERENCE` is the only class) and add a case to `tests/test_benchmark_harness.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/performance/benchmark_recommend.py
git commit -m "test: recommend benchmark (parallel speedup)"
```

(Include `tests/performance/harness.py` and `tests/test_benchmark_harness.py` only if Step 2 required changes.)

---

### Task 19: Documentation

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-09-26-describe-technique-design.md`

- [ ] **Step 1: CLAUDE.md**

1. Under "## 1. Per-column", after the Describe paragraph, add:

```markdown
**Recommend — `analytics.recommend`** (★`RecommendRust`, the only implementation). Narrowest value-preserving Arrow type per column (spec: docs/superpowers/specs/2026-09-26-recommend-technique-design.md): Describe's table plus `rec_*` columns. Step 1 type rules (null → boolean → uint → int → decimal → float → date → time → timestamp → timestamp_with_offset → string; lists → scalar when every list holds one item), step 2 dictionary encoding for strings (key width from `est_high`, Polars key one code narrower). Candidates carry a predicted IPC size and are tried smallest projected population size first (ties: hierarchy rank); each is cast, verified row by row and measured (Arrow and Polars layouts, plain and ZSTD); the original type is always the last resort. `rec_candidates` lists rule, evidence (the metric values tested), predicted/projected size and outcome for every candidate. Keywords: Describe's plus `boolean_pairs=(("true", "false"),)`. No Python implementation, so no reference agreement or algorithmic speedup; oracles are pyarrow/Polars casts and predicted = measured.
```

2. In the Describe paragraph's metric list, add "`gcd`, byte totals (`sum_len`, `sum_len_unique`), significant-digit counts" and change "Polars sizes" to "Polars native-layout IPC sizes".

3. Project Structure: replace the `src/` comment with

```
│   ├── src/                                # Rust plugin — lib.rs, shared.rs, entropy.rs, chi_squared.rs,
│   │                                       #   contingency.rs, ari.rs, gcd.rs, bloomfilter.rs, minhash.rs,
│   │                                       #   describe.rs, sizes.rs, cardinality_estimators.rs, recommend.rs
```

and add `recommend` to the `<technique>/` list.

4. Rust Plugin list: add `describe_and_recommend` after `describe_columns` + `column_sizes`, and a line: "`recommend.rs` and `sizes.rs` are Arrow-native (arrow-rs 60); Series cross in through `shared::to_arrow_rs` (C Data Interface, zero-copy)."

- [ ] **Step 2: Spec A** — in `2026-09-26-describe-technique-design.md`:
- §2 layout: replace the `src/describe/` tree with `src/describe.rs` and `src/sizes.rs`;
- §4.1: add rows for `gcd`, `sum_len`, `sum_len_unique`; §4.3: rows for `numeric_min_frac_digits`, `numeric_max_sig_digits`, `iso_max_sig_frac_digits` (definitions from Spec B §3);
- §4.5: `size_polars_bytes` = "IPC body length of the column in Polars' native layout (`CompatLevel.newest()`), uncompressed";
- §10: replace "Out of scope (Spec B)" body with "Delivered by Spec B: docs/superpowers/specs/2026-09-26-recommend-technique-design.md."

- [ ] **Step 3: Full test run**

Run: `cargo test --lib` (Commands), then `$PY -m pytest tests -q` (includes slow)
Expected: PASS everywhere.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-09-26-describe-technique-design.md
git commit -m "docs: recommend technique; describe metrics and layout"
```
