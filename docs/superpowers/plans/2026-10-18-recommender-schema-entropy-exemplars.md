# Recommender Schema, Entropy and Exemplars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape both recommenders' output once (profile + per-engine blocks, `rec_*` renamed) and add each column's entropy (with interval and method) and its exemplar / null-exemplar rows (with optional typed id columns), through Rust, Python, the C ABI and Java.

**Architecture:** Output columns stay one definition (`recommenders/schema.rs::recommender_fields`) assembled as Polars rows and exported to Arrow; the typed `exemplar_id` Structs are spliced in afterwards in Arrow (like `with_maps`). One-shot reads entropy and first rows from Describe's frequency sweep (`frequencies.rs`) and reduces counts with the shared SIMD `entropy_from_counts_iter`. Streaming keeps O(1) exemplar state per column, computes exact entropy from its distinct sample while exact, and after overflow uses the bottom-k sample estimator or the AMS sketch (`entropy_estimator`).

**Tech Stack:** Rust (polars 0.51, arrow-rs 60, rayon, pyo3), Python 3.12 (Polars, pytest + xdist), Java 25 (FFM, Arrow Java), maturin.

**Spec:** `docs/superpowers/specs/2026-10-18-recommender-schema-entropy-exemplars-design.md` (read §2–§6 before starting any task).

---

## Conventions for every task

- Shell: Git Bash on Windows. Repo root `c:/Users/Alexander/turbo-parakeet`. Work on branch `schema-entropy-exemplars` (Task 0 creates it).
- Environment for every Python / Rust / commit command:
  ```bash
  E=/c/Users/Alexander/miniconda3/envs/p312
  export PATH="$E:$E/Scripts:$E/Library/bin:/c/Users/Alexander/.cargo/bin:$PATH" PYO3_PYTHON="$E/python.exe"
  PY=$E/python.exe
  ```
- Rust unit tests: `cd services/analytics && cargo test --lib <filter>`; the extension: `$PY scripts/check.py build` (from the repo root; ≈ 3 min warm) — needed before Python tests see Rust changes.
- pytest: `$PY -m pytest <files> -q` (4 xdist workers; `-n 0` serial).
- Before each commit: `$PY scripts/check.py fmt && $PY scripts/check.py lint`. Commit without `--no-verify`; message ends with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `docs/superpowers/` is git-ignored but tracked: `git add -f` there.
- The layering rule (`scripts/check_layering.py`): `common` ← `techniques` ← `recommenders` ← `bindings`; never import upwards.
- If a step's expected result does not happen, stop and report with the output; do not loosen tolerances or change behaviour outside the task.

## File map

| File | Responsibility / change |
|---|---|
| `src/recommenders/schema.rs` | field list: renames, order, new profile columns; `with_ids` (Struct splice) |
| `src/techniques/single_column/unordered/describe/frequencies.rs` | `Frequencies.entropy`, `first_row`, `first_null` from the existing sweep |
| `src/common/text.rs` | `display_value` (arrow-rs display text, every type) |
| `src/recommenders/engine/entropy.rs` (new) | `EntropyEstimator`, `Entropy`, exact / sample / AMS constructors, the 4 cells |
| `src/recommenders/engine/exemplar.rs` (new) | `Exemplar`, `exemplar_value`, the 3 non-id cells |
| `src/recommenders/engine/candidates.rs` | `Params` gains `entropy_estimator`, `id_columns` |
| `src/recommenders/engine/column.rs` | one-shot row: entropy + exemplar cells; `rec_row` order |
| `src/recommenders/oneshot.rs` | keeps the id columns; `with_ids` |
| `src/recommenders/streaming/mod.rs` | exemplar state, id slices, entropy at `result`, `with_ids` |
| `src/recommenders/streaming/partial.rs` | AMS per outer level (row keys in `BatchStats`) |
| `src/recommenders/streaming/distinct_sample.rs` | `held_counts()` |
| `src/techniques/single_column/unordered/ams_entropy.rs` | `estimate_with_se` (estimate unchanged), `Clone` |
| `src/bindings/api/recommenders.rs`, `python.rs`, `capi.rs` | new parameters |
| `analytics/_plugin.py`, `analytics/recommend/{oneshot,streaming}.py` | keywords `id_columns`, `entropy_estimator` |
| `bindings/java/.../{OneShotParams,StreamingParams,OneShotRecommender,StreamingRecommender,EntropyEstimator}.java` | new parameters |
| `tests/test_recommend.py`, `tests/test_streaming_recommend.py`, `tests/test_recommender_profile.py` (new) | renames; entropy / exemplar tests |
| `tests/performance/benchmark_streaming_entropy.py` (new) | estimator benchmark |

(`src/` = `services/analytics/src/`.)

---

### Task 0: Branch

- [ ] `git switch -c schema-entropy-exemplars` (from an up-to-date `main`). Nothing to commit.

---

### Task 1: Rename and reorder the recommendation columns

Spec §2.2 and §2.1 (shared recommendation before the engine blocks, `candidates` after). Mechanical, but it touches every layer.

**Files:** `src/recommenders/schema.rs`, `src/recommenders/engine/column.rs` (`rec_row`), Rust tests that name columns (`src/recommenders/oneshot.rs`, `src/recommenders/streaming/tests.rs`, `src/recommenders/streaming/mod.rs`, `src/bindings/capi.rs`, `src/bindings/api/recommenders.rs`), Python `services/analytics/analytics/recommend/oneshot.py` (docstring), `tests/test_recommend.py`, `tests/test_streaming_recommend.py`, `tests/performance/benchmark_recommend.py`, Java tests `bindings/java/src/test/java/io/github/benssutton/analytics/{OneShotRecommenderTest,StreamingRecommenderTest}.java`.

- [ ] **Step 1: `rec_fields` in `src/recommenders/schema.rs`** — replace the function body's list with the new names and order:

```rust
pub(crate) fn rec_fields() -> Vec<(String, PT)> {
    [
        ("nullable", PT::Boolean),
        ("lossy_formatting", PT::Boolean),
        ("datafusion_type", PT::String),
        ("datafusion_size_bytes", PT::UInt64),
        ("datafusion_size_zstd_bytes", PT::UInt64),
        ("polars_type", PT::String),
        ("polars_size_bytes", PT::UInt64),
        ("polars_size_zstd_bytes", PT::UInt64),
        ("candidates", PT::List(Box::new(candidate_type()))),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}
```

- [ ] **Step 2: `rec_row` in `src/recommenders/engine/column.rs`** — same order:

```rust
pub(crate) fn rec_row(r: &Rec) -> Row {
    let text = |s: &str| AnyValue::StringOwned(s.into());
    vec![
        AnyValue::Boolean(r.nullable),
        AnyValue::Boolean(r.lossy),
        text(&r.arrow_type),
        AnyValue::UInt64(r.arrow_size),
        r.arrow_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        r.polars_type.as_deref().map_or(AnyValue::Null, text),
        AnyValue::UInt64(r.polars_size),
        r.polars_zstd.map_or(AnyValue::Null, AnyValue::UInt64),
        AnyValue::List(candidates_series(&r.candidates)),
    ]
}
```

- [ ] **Step 3: Rename every use of the old names** (string literals and Python / Java keys only; Rust identifiers such as `rec_fields`, `rec_row`, `Rec` stay):

```bash
cd c:/Users/Alexander/turbo-parakeet
FILES=$(git grep -l -E 'rec_(arrow|polars)_|rec_nullable|rec_lossy_formatting|rec_candidates' -- services/analytics/src services/analytics/analytics services/analytics/bindings/java/src tests)
sed -i -E 's/\brec_arrow_(type|size_bytes|size_zstd_bytes)\b/datafusion_\1/g; s/\brec_polars_(type|size_bytes|size_zstd_bytes)\b/polars_\1/g; s/\brec_nullable\b/nullable/g; s/\brec_lossy_formatting\b/lossy_formatting/g; s/\brec_candidates\b/candidates/g' $FILES
git grep -n -E 'rec_(arrow|polars)_|rec_nullable|rec_lossy|rec_candidates' -- services tests
```

Expected: the last grep prints nothing. Review `git diff` for any Rust identifier or Python variable that was renamed by accident (e.g. a local `rec_candidates` variable) and restore it if so.

- [ ] **Step 4: Fix the schema dict order in `tests/test_streaming_recommend.py`** — in `SCHEMA`, the renamed entries must follow the Step 1 order: `"nullable"`, `"lossy_formatting"`, `"datafusion_type"`, `"datafusion_size_bytes"`, `"datafusion_size_zstd_bytes"`, `"polars_type"`, `"polars_size_bytes"`, `"polars_size_zstd_bytes"`, `"candidates"`. Do the same for any equivalent ordered list in `tests/test_recommend.py` (search for `"nullable"`).

- [ ] **Step 5: Verify**

```bash
cd services/analytics && cargo test --lib recommenders:: bindings:: 2>&1 | tail -3 && cd ../..
$PY scripts/check.py build && $PY -m pytest tests/test_recommend.py tests/test_streaming_recommend.py -q
$PY scripts/check.py test-java
```

Expected: all pass.

- [ ] **Step 6: Commit** — `git add -A services/analytics tests && git commit -m "Recommenders: datafusion_ / polars_ blocks, rec_ prefix dropped (spec 2026-10-18 §2.2)"`

---

### Task 2: New profile columns (emitted null), contract tests

Spec §2.1. Adds the 9 new columns after `n_midnight` in both recommenders, all null for now, so later tasks only fill cells.

**Files:** `src/recommenders/schema.rs`, `src/recommenders/engine/column.rs` (`Prepared::row`), `src/recommenders/streaming/mod.rs` (`Streaming::row`), tests.

- [ ] **Step 1: Failing Python contract test** — in `tests/test_streaming_recommend.py`, in `SCHEMA` after `"n_midnight": pl.UInt64,` insert:

```python
    "entropy": pl.Float64,
    "entropy_low": pl.Float64,
    "entropy_high": pl.Float64,
    "entropy_method": pl.String,
    "exemplar_row": pl.UInt64,
    "exemplar_id": pl.Null,
    "exemplar_value": pl.String,
    "null_exemplar_row": pl.UInt64,
    "null_exemplar_id": pl.Null,
```

Run `$PY -m pytest tests/test_streaming_recommend.py -q -k schema` → FAIL (columns missing). (Use the test(s) that compare against `SCHEMA`; if `-k schema` selects none, run the file.)

- [ ] **Step 2: Fields** — in `recommender_fields` (`src/recommenders/schema.rs`), after `f.push(("n_midnight".into(), PT::UInt64));` add:

```rust
    f.extend(profile_fields());
```

and add above `recommender_fields`:

```rust
/// Entropy and exemplar columns (spec 2026-10-18 §3, §4), after `n_midnight`. The two
/// `*_id` columns are Null-typed here; `with_ids` replaces them when there are id columns.
pub(crate) fn profile_fields() -> Vec<(String, PT)> {
    [
        ("entropy", PT::Float64),
        ("entropy_low", PT::Float64),
        ("entropy_high", PT::Float64),
        ("entropy_method", PT::String),
        ("exemplar_row", PT::UInt64),
        ("exemplar_id", PT::Null),
        ("exemplar_value", PT::String),
        ("null_exemplar_row", PT::UInt64),
        ("null_exemplar_id", PT::Null),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect()
}
```

- [ ] **Step 3: Null cells in both row builders** — `Prepared::row` (`engine/column.rs`): right after `row.extend(base.by_ref().take(3 + value_fields().len())); // through n_midnight` add
  `row.extend(vec![AnyValue::Null; profile_fields().len()]); // filled in Tasks 4–5`
  `Streaming::row` (`streaming/mod.rs`): right after `row.push(u(o.n_midnight));` add the same line. Import `profile_fields` from `crate::recommenders::schema` in both files. Ineligible rows are already padded with nulls.

- [ ] **Step 4: Rust contract tests still pass** (they compare against `recommender_fields`): `cargo test --lib recommenders::` → pass.

- [ ] **Step 5: Python** — `$PY scripts/check.py build && $PY -m pytest tests/test_recommend.py tests/test_streaming_recommend.py -q` → pass (the schema test now matches). Also check one-shot's schema test in `tests/test_recommend.py` (if it lists columns, add the same 9 entries there).

- [ ] **Step 6: Commit** — `"Recommenders: entropy and exemplar columns in the schema (null for now)"`.

---

### Task 3: Describe's frequency sweep reports entropy and first rows

Spec §3.2, §4.1. The sweep already visits every distinct value; add three fields. Streaming builds `Frequencies` itself and leaves them None.

**Files:** `src/techniques/single_column/unordered/describe/frequencies.rs`, `src/recommenders/streaming/partial.rs:579` (the `Frequencies { .. }` literal).

- [ ] **Step 1: Failing Rust tests** — append to the `tests` module of `frequencies.rs`:

```rust
    #[test]
    fn entropy_counts_null_as_a_category() {
        // [1, 1, null, null, 2]: counts (2, 2, 1) over 5 rows.
        let f = freq(Series::new("a".into(), &[Some(1i32), Some(1), None, None, Some(2)]));
        let (p1, p2): (f64, f64) = (0.4, 0.2);
        let want = -(2.0 * p1 * p1.log2() + p2 * p2.log2());
        assert!((f.entropy.unwrap() - want).abs() < 1e-12);
        assert_eq!((f.first_row, f.first_null), (Some(0), Some(2)));
    }

    #[test]
    fn entropy_edge_cases() {
        let constant = freq(Series::new("a".into(), &[7i64; 10]));
        assert_eq!(constant.entropy, Some(0.0));
        assert_eq!(constant.first_null, None);
        let distinct = freq(Series::new("a".into(), &[1i64, 2, 3, 4]));
        assert!((distinct.entropy.unwrap() - 2.0).abs() < 1e-12);
        let empty = freq(Series::new("a".into(), Vec::<i64>::new()));
        assert_eq!((empty.entropy, empty.first_row, empty.first_null), (Some(0.0), None, None));
        let all_null = freq(Series::new("a".into(), &[None::<i64>, None]));
        assert_eq!((all_null.entropy, all_null.first_row, all_null.first_null), (Some(0.0), None, Some(0)));
    }
```

Run `cargo test --lib frequencies` → compile error (no field `entropy`).

- [ ] **Step 2: Fields** — add to `pub(crate) struct Frequencies`:

```rust
    /// Shannon entropy in bits, null one category (spec 2026-10-18 §3.1); None when built by
    /// streaming's `LevelStats::profile` (it estimates its own).
    pub entropy: Option<f64>,
    /// Row of the first non-null / first null value (one-shot exemplars, spec §4); None from
    /// streaming's profile, and when there is no such row.
    pub first_row: Option<u64>,
    pub first_null: Option<u64>,
```

- [ ] **Step 3: Compute them in `frequencies()`** — add `use crate::common::entropy_math::entropy_from_counts_iter;` at the top; right after `let map = frequency_map(col, seed, 0);` insert:

```rust
    let n = col.is_null.len() as u64;
    let n_null = col.is_null.iter().filter(|&&b| b).count() as u64;
    // The shared SIMD reduction over the counts the sweep already holds (no second pass).
    let entropy = if n == 0 {
        0.0
    } else {
        entropy_from_counts_iter(
            map.values()
                .map(|e| e.count)
                .chain((n_null > 0).then_some(n_null)),
            (n as f64).log2(),
            n as f64,
        )
    };
    let first_row = map.values().map(|e| e.first).min();
    let first_null = col.is_null.iter().position(|&b| b).map(|i| i as u64);
```

and add `entropy: Some(entropy), first_row, first_null,` to the returned `Frequencies { .. }`. (`col.is_null.len()` is the row count; `encode_series` fills `is_null` for every row.)

- [ ] **Step 4: Streaming literal** — in `src/recommenders/streaming/partial.rs` `LevelStats::profile`, add `entropy: None, first_row: None, first_null: None,` to the `Frequencies { .. }` literal.

- [ ] **Step 5:** `cargo test --lib frequencies describe recommenders` → pass. Commit `"Describe frequencies: entropy and first rows from the existing sweep"`.

---

### Task 4: One-shot entropy and exemplar cells

Spec §3.1–3.2, §4.1 (rows and value; ids come in Task 8).

**Files:** create `src/recommenders/engine/entropy.rs`, `src/recommenders/engine/exemplar.rs`; modify `src/recommenders/engine/mod.rs`, `src/common/text.rs`, `src/recommenders/engine/column.rs`; create `tests/test_recommender_profile.py`.

- [ ] **Step 1: `display_value` in `src/common/text.rs`** (every type; binary as hex):

```rust
/// Row 0 of `a` as arrow-rs display text (`ArrayFormatter`): every type, nested included;
/// binary as hex. None when the row is null.
pub(crate) fn display_value(a: &dyn Array) -> Option<String> {
    if a.is_empty() || a.is_null(0) {
        return None;
    }
    let f = arrow_cast::display::ArrayFormatter::try_new(a, &Default::default()).ok()?;
    Some(f.value(0).to_string())
}
```

with a unit test in the same file's `tests` module:

```rust
    #[test]
    fn display_value_renders_every_type() {
        use arrow_array::{BinaryArray, Int64Array, ListArray, types::Int64Type};
        let list = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1), Some(2)])]);
        assert_eq!(display_value(&list).as_deref(), Some("[1, 2]"));
        let bin = BinaryArray::from(vec![&b"\x01\xff"[..]]);
        assert_eq!(display_value(&bin).as_deref(), Some("01ff"));
        assert_eq!(display_value(&Int64Array::from(vec![None::<i64>])), None);
    }
```

(If arrow-rs renders binary differently from `"01ff"`, keep its rendering and fix the expectation — the spec only requires hex.)

- [ ] **Step 2: `src/recommenders/engine/entropy.rs`**:

```rust
//! Entropy cells (spec 2026-10-18 §3): exact, the bottom-k sample estimate, the AMS estimate.

use polars::prelude::AnyValue;

use crate::common::entropy_math::entropy_from_counts_iter;
use crate::techniques::describe::Row;

/// Streaming's estimator once the distinct sample overflows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EntropyEstimator {
    #[default]
    Sample,
    Ams,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Entropy {
    pub value: f64,
    pub low: f64,
    pub high: f64,
    pub method: &'static str,
}

/// Bounds clamped to [0, log2 n].
fn clamp(x: f64, n: u64) -> f64 {
    let top = if n > 1 { (n as f64).log2() } else { 0.0 };
    x.clamp(0.0, top)
}

impl Entropy {
    pub(crate) fn exact(value: f64) -> Self {
        Entropy { value, low: value, high: value, method: "exact" }
    }

    /// Exact entropy from every non-null value's count plus `n_null` (n = all rows).
    pub(crate) fn from_counts(counts: impl Iterator<Item = u64>, n_null: u64, n: u64) -> Self {
        if n == 0 {
            return Self::exact(0.0);
        }
        let h = entropy_from_counts_iter(
            counts.chain((n_null > 0).then_some(n_null)),
            (n as f64).log2(),
            n as f64,
        );
        Self::exact(h)
    }

    /// Bottom-k sample estimate (spec §3.3): `counts` are the k' held values' exact counts,
    /// `d` = (est_cardinality, est_low, est_high) of the non-null distinct values.
    pub(crate) fn from_sample(counts: &[u64], d: (f64, f64, f64), n_null: u64, n: u64) -> Self {
        let k = counts.len() as f64;
        if n == 0 || counts.is_empty() {
            return Self::from_counts(counts.iter().copied(), n_null, n);
        }
        let nf = n as f64;
        let term = |c: u64| if c > 1 { c as f64 * (c as f64).log2() } else { 0.0 };
        let terms: Vec<f64> = counts.iter().map(|&c| term(c)).collect();
        let mean = terms.iter().sum::<f64>() / k;
        let s2 = if counts.len() > 1 {
            terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (k - 1.0)
        } else {
            0.0
        };
        let null_term = term(n_null);
        let at = |dd: f64| {
            let dd = dd.max(k);
            let h = nf.log2() - (dd * mean + null_term) / nf;
            let sd = (dd * dd * (1.0 - k / dd).max(0.0) * s2 / k).sqrt() / nf;
            (h, sd)
        };
        let (h, _) = at(d.0);
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for dd in [d.0, d.1, d.2] {
            let (hh, sd) = at(dd);
            low = low.min(hh - 3.0 * sd);
            high = high.max(hh + 3.0 * sd);
        }
        Entropy { value: clamp(h, n), low: clamp(low, n), high: clamp(high, n), method: "sample" }
    }

    /// AMS estimate with its standard error (spec §3.3), interval ±3σ.
    pub(crate) fn from_ams(value: f64, se: f64, n: u64) -> Self {
        Entropy {
            value: clamp(value, n),
            low: clamp(value - 3.0 * se, n),
            high: clamp(value + 3.0 * se, n),
            method: "ams",
        }
    }
}

/// The four `entropy*` cells; all null for None.
pub(crate) fn entropy_cells(e: Option<&Entropy>) -> Row {
    match e {
        Some(e) => vec![
            AnyValue::Float64(e.value),
            AnyValue::Float64(e.low),
            AnyValue::Float64(e.high),
            AnyValue::StringOwned(e.method.into()),
        ],
        None => vec![AnyValue::Null; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_from_counts() {
        let e = Entropy::from_counts([2u64, 1].into_iter(), 2, 5);
        let want = -(2.0 * 0.4f64 * 0.4f64.log2() + 0.2 * 0.2f64.log2());
        assert!((e.value - want).abs() < 1e-12);
        assert_eq!((e.low, e.high, e.method), (e.value, e.value, "exact"));
        assert_eq!(Entropy::from_counts(std::iter::empty(), 0, 0).value, 0.0);
    }

    #[test]
    fn sample_with_equal_counts_is_exact_and_tight() {
        // D = 1000 values of count 4 (n = 4000), sampled 100 of them: T = 1000·4·2 = 8000,
        // H = log2(4000) − 8000/4000 = log2(1000); every term equal, so s² = 0.
        let e = Entropy::from_sample(&[4; 100], (1000.0, 1000.0, 1000.0), 0, 4000);
        assert!((e.value - 1000f64.log2()).abs() < 1e-9);
        assert!((e.high - e.low).abs() < 1e-9);
        assert_eq!(e.method, "sample");
    }

    #[test]
    fn sample_interval_widens_with_spread_and_d_uncertainty() {
        let counts: Vec<u64> = (1..=100).collect();
        let e = Entropy::from_sample(&counts, (1000.0, 900.0, 1100.0), 10, 60_000);
        assert!(e.low < e.value && e.value < e.high);
        assert!(e.high <= 60_000f64.log2() && e.low >= 0.0);
    }

    #[test]
    fn ams_interval_is_three_sigma_clamped() {
        let e = Entropy::from_ams(3.0, 0.1, 1 << 10);
        assert!((e.low - 2.7).abs() < 1e-12 && (e.high - 3.3).abs() < 1e-12);
        assert_eq!(Entropy::from_ams(0.1, 1.0, 16).low, 0.0);
    }
}
```

- [ ] **Step 3: `src/recommenders/engine/exemplar.rs`**:

```rust
//! Exemplar cells (spec 2026-10-18 §4): first non-null and first null rows, and the value.

use polars::prelude::{AnyValue, PolarsResult, Series};

use crate::common::ipc_sizes::classic_layout;
use crate::common::text::display_value;
use crate::techniques::describe::Row;

/// A column's exemplar rows and the value at the first non-null one.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Exemplars {
    pub row: Option<u64>,
    pub value: Option<String>,
    pub null_row: Option<u64>,
}

/// Row `i` of `s` as display text (classic Arrow layout, as `min` / `max`).
pub(crate) fn exemplar_value(s: &Series, i: u64) -> PolarsResult<Option<String>> {
    let one = classic_layout(&s.slice(i as i64, 1))?;
    Ok(display_value(one.as_ref()))
}

/// The five exemplar cells in schema order; the `*_id` cells are null here (`with_ids`).
pub(crate) fn exemplar_cells(e: Option<&Exemplars>) -> Row {
    let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
    match e {
        Some(e) => vec![
            u(e.row),
            AnyValue::Null,
            e.value.clone().map_or(AnyValue::Null, |v| AnyValue::StringOwned(v.into())),
            u(e.null_row),
            AnyValue::Null,
        ],
        None => vec![AnyValue::Null; 5],
    }
}
```

Register both in `src/recommenders/engine/mod.rs` (`mod entropy; mod exemplar;` and `pub(crate) use entropy::*; pub(crate) use exemplar::*;`).

- [ ] **Step 4: One-shot cells** — in `Prepared::row` (`engine/column.rs`) replace the Task 2 null line with:

```rust
        let f = &d.outer.freq;
        let entropy = f.entropy.map(Entropy::exact);
        row.extend(entropy_cells(entropy.as_ref()));
        let ex = Exemplars {
            row: f.first_row,
            value: f.first_row.map(|i| exemplar_value(&self.series, i)).transpose()?.flatten(),
            null_row: f.first_null,
        };
        row.extend(exemplar_cells(Some(&ex)));
```

- [ ] **Step 5: Rust unit tests pass** — `cargo test --lib engine:: text::` → pass.

- [ ] **Step 6: Python tests** — create `tests/test_recommender_profile.py`:

```python
"""Entropy and exemplar columns of both recommenders (spec 2026-10-18 §3, §4)."""

import math

import polars as pl
import pytest

from analytics import _plugin
from analytics.recommend import OneShotRecommender, StreamingRecommender
from datagen import describe_mixed, large_dataset, large_dataset_mixed

FRAMES = [
    pytest.param(lambda: describe_mixed(2_000), id="describe_mixed"),
    pytest.param(large_dataset, id="large"),
    pytest.param(large_dataset_mixed, id="mixed"),
]


def one_shot(frame: pl.DataFrame, **params) -> dict[str, dict]:
    out = OneShotRecommender(**params).add(frame).result()
    return {r["column"]: r for r in out.iter_rows(named=True)}


@pytest.mark.parametrize("make", FRAMES)
def test_one_shot_entropy_matches_marginal_entropy(make):
    frame = make()
    rows = one_shot(frame)
    computed = [c for c, r in rows.items() if r["status"] == "computed"]
    want = dict(_plugin.marginal_entropy(frame.select(computed)).iter_rows())
    for c in computed:
        r = rows[c]
        assert r["entropy"] == pytest.approx(want[c], rel=1e-12, abs=1e-12), c
        assert (r["entropy_low"], r["entropy_high"]) == (r["entropy"], r["entropy"])
        assert r["entropy_method"] == "exact"


def test_one_shot_entropy_known_answers():
    frame = pl.DataFrame(
        {
            "constant": [7] * 8,
            "distinct": list(range(8)),
            "half_null": [1, 1, 1, 1, None, None, None, None],
        }
    )
    rows = one_shot(frame)
    assert rows["constant"]["entropy"] == 0.0
    assert rows["distinct"]["entropy"] == pytest.approx(3.0)
    assert rows["half_null"]["entropy"] == pytest.approx(1.0)


def test_one_shot_exemplars_known_answers():
    frame = pl.DataFrame(
        {
            "a": [None, 5, None, 6],
            "s": ["x", "y", "z", "w"],
            "l": [[1, 2], None, [None], [3]],
            "n": pl.Series([None, None, None, None], dtype=pl.Int64),
        }
    )
    rows = one_shot(frame)
    assert (rows["a"]["exemplar_row"], rows["a"]["exemplar_value"]) == (1, "5")
    assert rows["a"]["null_exemplar_row"] == 0
    assert (rows["s"]["exemplar_row"], rows["s"]["null_exemplar_row"]) == (0, None)
    assert rows["l"]["exemplar_value"] == "[1, 2]"
    assert rows["l"]["null_exemplar_row"] == 1  # top-level nulls only
    assert (rows["n"]["exemplar_row"], rows["n"]["exemplar_value"]) == (None, None)
    assert rows["n"]["null_exemplar_row"] == 0


def test_exemplar_value_matches_min_max_rendering_for_scalars():
    frame = pl.DataFrame({"x": [2.5], "d": [date(2024, 1, 2)], "s": ["hi"]})
    for r in one_shot(frame).values():
        assert r["exemplar_value"] == r["min"], r["column"]
```

(Add `from datetime import date` to the file's imports.)

- [ ] **Step 7: Run** — `$PY scripts/check.py build && $PY -m pytest tests/test_recommender_profile.py tests/test_recommend.py -q` → pass. Commit `"One-shot recommender: entropy (exact) and exemplar rows / values"`.

---

### Task 5: Streaming exact entropy and exemplars (rows and values)

Spec §3.3 (exact phase), §4.1. First-seen state per column; rows count from the stream start (`first_row`'s counter). Rows a column misses (absent from a batch, Null-typed in it, or before it appeared) are nulls.

**Files:** `src/recommenders/streaming/mod.rs`, `src/recommenders/streaming/distinct_sample.rs`, `src/recommenders/streaming/tests.rs`, `tests/test_streaming_recommend.py`, `tests/test_recommender_profile.py`.

- [ ] **Step 1: `held_counts`** in `distinct_sample.rs` (`impl DistinctSample`):

```rust
    /// Every held value's count (all values while exact; the sample after).
    pub(crate) fn held_counts(&self) -> Vec<u64> {
        self.map.values().map(|s| s.count).collect()
    }
```

- [ ] **Step 2: Exemplar state on `Column`** — add `pub exemplars: Exemplars,` (from `crate::recommenders::engine::Exemplars`) to `struct Column`, initialised `Exemplars::default()` in `Streaming::column`; and when a column is created while `self.n_rows > 0` set `exemplars.null_row = Some(0)` (its earlier rows are nulls).

- [ ] **Step 3: Find first rows before the commit** (rendering can fail, so it happens before "Commit: nothing below can fail"). In `Streaming::add`, after `let stats: Vec<Stats> = ...?;` and before `self.reservoir.feed(...)`, add:

```rust
        // Exemplars (spec 2026-10-18 §4): per series, the first non-null row (and its text)
        // and first null row this batch adds, for columns still missing them.
        let start = self.n_rows;
        let firsts: Vec<(Option<(u64, Option<String>)>, Option<u64>)> = series
            .par_iter()
            .map(|(s, _)| {
                let known = self
                    .index
                    .get(s.name().as_str())
                    .map(|&i| &self.columns[i].exemplars);
                let need_row = known.is_none_or(|e| e.row.is_none());
                // A column new at start > 0 already has null row 0 (Step 2).
                let need_null = match known {
                    None => start == 0,
                    Some(e) => e.null_row.is_none(),
                };
                let nulls = s.is_null();
                let row = match need_row {
                    true => nulls.iter().position(|v| v == Some(false)),
                    false => None,
                };
                let row = row
                    .map(|i| exemplar_value(s, i as u64).map(|v| (start + i as u64, v)))
                    .transpose()?;
                let null = match need_null {
                    true => nulls.iter().position(|v| v == Some(true)).map(|i| start + i as u64),
                    false => None,
                };
                Ok((row, null))
            })
            .collect::<PolarsResult<_>>()
            .map_err(|e| Error::Compute(e.to_string()))?;
```

(A Null-typed series has `is_null` all true, so its first null is `start`.)

- [ ] **Step 4: Commit the exemplars** — in the serial loop that creates columns (`for ((s, f), st) in series.iter().zip(stats)`), zip in `firsts` and, after `let c = self.column(...)`, apply:

```rust
            let (row, null) = first;
            if let (None, Some((r, v))) = (c.exemplars.row, row) {
                c.exemplars.row = Some(r);
                c.exemplars.value = v;
            }
            if c.exemplars.null_row.is_none() {
                c.exemplars.null_row = null;
            }
```

and for columns absent from the batch, before `self.n_rows += rows;`:

```rust
        if rows > 0 {
            let present: HashSet<&str> = series.iter().map(|(s, _)| s.name().as_str()).collect();
            for c in self.columns.iter_mut().filter(|c| !present.contains(c.name.as_str())) {
                c.exemplars.null_row.get_or_insert(start);
            }
        }
```

(Nested-Null columns are ineligible and keep null exemplars: their cells are null via the ineligible row.)

- [ ] **Step 5: Cells in `Streaming::row`** — replace the Task 2 null line with:

```rust
        let entropy = match &o.sample {
            None => Some(Entropy::from_counts(std::iter::empty(), o.n_null, o.n)),
            Some(d) if d.is_exact() => {
                Some(Entropy::from_counts(d.held_counts().into_iter(), o.n_null, o.n))
            }
            Some(_) => None, // Task 6 (sample) and Task 7 (AMS)
        };
        row.extend(entropy_cells(entropy.as_ref()));
        row.extend(exemplar_cells(Some(&c.exemplars)));
```

- [ ] **Step 6: Rust test** in `streaming/tests.rs`:

```rust
#[test]
fn exemplars_are_first_seen_across_batches() {
    let mut r = Streaming::new(params(), 0, 4);
    r.add(&batch(vec![("a", ints(&[None, None]))])).unwrap();
    r.add(&batch(vec![("a", ints(&[Some(3), None])), ("b", strs(&[Some("x"), Some("y")]))])).unwrap();
    r.add(&batch(vec![("a", ints(&[Some(4)]))])).unwrap();
    let out = r.result().unwrap();
    assert_eq!(texts(&out, "exemplar_row"), [Some("2".into()), Some("2".into())]);
    assert_eq!(texts(&out, "exemplar_value"), [Some("3".into()), Some("x".into())]);
    // a: first null row 0; b: absent from batch 1 (rows 0–1) and from batch 3 (row 4)
    assert_eq!(texts(&out, "null_exemplar_row"), [Some("0".into()), Some("0".into())]);
}
```

(`texts` renders a column as strings; adjust the helper call if its signature differs.) `cargo test --lib streaming::` → pass.

- [ ] **Step 7: Python parity** — in `tests/test_streaming_recommend.py::assert_parity`, inside the per-row loop after the `exact_cols` assertions add:

```python
        for k in ["exemplar_row", "exemplar_value", "null_exemplar_row"]:
            assert r[k] == o[k], (name, k, r[k], o[k])
        if not overflowed:
            assert r["entropy_method"] == "exact", name
            assert r["entropy"] == pytest.approx(o["entropy"], rel=1e-12, abs=1e-12), name
```

and in `tests/test_recommender_profile.py` add:

```python
@pytest.mark.parametrize("batch_rows", [1, 3, 1_000])
def test_streaming_exemplars_and_exact_entropy_equal_one_shot(batch_rows):
    frame = pl.DataFrame(
        {
            "a": [None, None, None, 5, None, 6] * 3,
            "s": ["x", "y", None, "z", "w", "v"] * 3,
            "l": [[1], None, [2, None], [3], [], None] * 3,
        }
    )
    want = one_shot(frame)
    rec = StreamingRecommender()
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    for r in rec.result().iter_rows(named=True):
        o = want[r["column"]]
        for k in ["exemplar_row", "exemplar_value", "null_exemplar_row", "entropy_method"]:
            assert r[k] == o[k], (r["column"], k)
        assert r["entropy"] == pytest.approx(o["entropy"], rel=1e-12, abs=1e-12)


def test_streaming_late_and_missing_columns_count_as_null():
    rec = StreamingRecommender()
    rec.add(pl.DataFrame({"a": [1, 2]}))
    rec.add(pl.DataFrame({"a": [3], "b": ["x"]}))
    rows = {r["column"]: r for r in rec.result().iter_rows(named=True)}
    assert (rows["b"]["exemplar_row"], rows["b"]["null_exemplar_row"]) == (2, 0)
    assert rows["b"]["entropy"] == pytest.approx(-(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3))
```

- [ ] **Step 8: Run** — build, then `$PY -m pytest tests/test_recommender_profile.py tests/test_streaming_recommend.py -q` → pass (overflowed rows in the large-dataset parity tests still have null entropy; Task 6 fills them). Commit `"Streaming recommender: exact entropy and first-seen exemplars"`.

---

### Task 6: Streaming sample estimator after overflow

Spec §3.3 (`sample`). No extra per-row work: the held counts and the cardinality estimate exist already.

**Files:** `src/recommenders/streaming/mod.rs`, `tests/test_recommender_profile.py`, `tests/test_streaming_recommend.py`.

- [ ] **Step 1: Failing Python test** — add to `tests/test_recommender_profile.py`:

```python
import numpy as np


def zipf_frame(n: int, d: int, alpha: float, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, d + 1) ** alpha
    return pl.DataFrame({"x": rng.choice(d, size=n, p=w / w.sum())})


STREAMS = [
    pytest.param(lambda: pl.DataFrame({"x": np.random.default_rng(0).integers(0, 200_000, 400_000)}), id="uniform"),
    pytest.param(lambda: zipf_frame(400_000, 200_000, 1.1), id="zipf1.1"),
]


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["sample"])  # Task 7 adds "ams"
@pytest.mark.parametrize("make", STREAMS)
def test_streaming_entropy_after_overflow(make, estimator):
    frame = make()
    exact = one_shot(frame)["x"]["entropy"]
    rec = StreamingRecommender()  # Task 7: StreamingRecommender(entropy_estimator=estimator)
    for off in range(0, frame.height, 50_000):
        rec.add(frame.slice(off, 50_000))
    r = rec.result().row(0, named=True)
    assert r["entropy_method"] == estimator
    assert r["entropy"] == pytest.approx(exact, rel=0.05)
    assert r["entropy_low"] <= exact <= r["entropy_high"]
```

Run `$PY -m pytest tests/test_recommender_profile.py -k overflow -q` → FAIL (`entropy_method` None).

- [ ] **Step 2: The estimate** — in `Streaming::row`, replace `Some(_) => None, // Task 6 ...` with:

```rust
            Some(d) => Some(Entropy::from_sample(
                &d.held_counts(),
                (
                    oc.est.est_cardinality,
                    oc.est.est_low.unwrap_or(oc.est.est_cardinality),
                    oc.est.est_high.unwrap_or(oc.est.est_cardinality),
                ),
                o.n_null,
                o.n,
            )),
```

(`oc` is computed above from `conclude`; `Estimate` is `Copy`. If the entropy block sits before `oc` is defined, move it after.)

- [ ] **Step 3: Parity after overflow** — in `assert_parity` (`tests/test_streaming_recommend.py`) extend the Task 5 block:

```python
        else:
            assert r["entropy_method"] in ("sample", "ams"), name
            assert r["entropy_low"] - 1e-9 <= o["entropy"] <= r["entropy_high"] + 1e-9, (
                name, r["entropy_low"], o["entropy"], r["entropy_high"])
```

- [ ] **Step 4: Run** — build; `$PY -m pytest tests/test_recommender_profile.py -k "not ams" tests/test_streaming_recommend.py -q` → pass. If the interval misses the exact value on a large-dataset column, stop and report the column, its counts summary and the three D values — do not widen the interval. Commit `"Streaming recommender: bottom-k sample entropy estimate past the exact phase"`.

---

### Task 7: AMS estimator and the `entropy_estimator` keyword

Spec §3.3 (`ams`), §5. Plumbing through engine `Params`, the API and Python; C / Java in Task 9.

**Files:** `src/techniques/single_column/unordered/ams_entropy.rs`, `src/recommenders/engine/candidates.rs` (`Params`), all `Params { .. }` literals (`bindings/api/recommenders.rs`, `engine/candidates.rs` tests, `engine/choose.rs` tests, `streaming/tests.rs`), `src/recommenders/streaming/{mod.rs,partial.rs}`, `src/bindings/{api/recommenders.rs,python.rs,capi.rs}`, `analytics/recommend/streaming.py`, `analytics/_plugin.py` (if it forwards keywords explicitly), tests.

- [ ] **Step 1: `estimate_with_se`** — in `ams_entropy.rs` change `ccm_estimate` to return `(estimate, standard_error)` without changing the estimate's arithmetic:

```rust
fn ccm_estimate(
    m: u64,
    mg: &MisraGries,
    samples: impl Iterator<Item = (u64, u64, Option<u64>)>,
) -> (f64, f64) {
    if m == 0 {
        return (0.0, 0.0);
    }
    let mf = m as f64;
    let lambda = |x: u64| if x == 0 { 0.0 } else { x as f64 * (mf / x as f64).log2() };
    let step = |r: u64| lambda(r) - lambda(r - 1); // r ≥ 1 always
    let (top, count) = mg.top();
    let p = count as f64 / mf;
    let (mut sum, mut sq, mut n) = (0.0, 0.0, 0usize);
    let mut push = |x: f64| {
        sum += x;
        sq += x * x;
        n += 1;
    };
    let dominant = p > 0.5;
    for (s0, r0, r1) in samples {
        // Below 0.5 every primary counts; above, the dominant item's samples use the backup.
        push(if !dominant || s0 != top {
            step(r0)
        } else {
            r1.map_or(0.0, step)
        });
    }
    let nf = n as f64;
    let mean = sum / nf;
    let se = if n > 1 { ((sq - sum * sum / nf) / (nf - 1.0)).max(0.0).sqrt() / nf.sqrt() } else { 0.0 };
    if !dominant {
        return (mean, se);
    }
    // The dominant item's term from p̂; the rest estimated from samples of the stream without it.
    ((1.0 - p) * mean + p * (1.0 / p).log2(), (1.0 - p) * se)
}
```

and in `impl AmsEntropy`:

```rust
    /// Entropy estimate in bits; 0 for an empty stream.
    pub(crate) fn estimate(&self) -> f64 {
        self.estimate_with_se().0
    }

    /// The estimate and its standard error (sample standard deviation of the per-estimator
    /// terms over √estimators; the recommenders' interval, spec 2026-10-18 §3.3).
    pub(crate) fn estimate_with_se(&self) -> (f64, f64) {
        let r = |s: &Sample| self.tracked[&s.key].count - s.offset;
        ccm_estimate(
            self.m,
            &self.mg,
            self.estimators.iter().filter_map(|e| {
                let p = e.primary?;
                Some((p.key, r(&p), e.backup.as_ref().map(r)))
            }),
        )
    }
```

Add `#[derive(Clone)]` to `AmsEntropy` and to any of its field types lacking `Clone` (`MisraGries`, `Rng`, `Tracked`, …) — `LevelStats` derives `Clone`. Add a unit test:

```rust
    #[test]
    fn standard_error_shrinks_with_more_estimators() {
        let keys = zipf(200_000, 3);
        let se = |s: usize| {
            let mut a = AmsEntropy::new(s, 9);
            keys.iter().for_each(|&k| a.update(k));
            a.estimate_with_se().1
        };
        assert!(se(4_096) < se(256));
        assert!(se(4_096) > 0.0);
    }
```

Run `cargo test --lib ams` → every existing AMS test still passes (estimates unchanged) plus the new one.

- [ ] **Step 2: `Params`** — in `engine/candidates.rs` add to `Params`:

```rust
    /// Streaming: the entropy estimator past the exact phase (spec 2026-10-18 §3.3).
    pub entropy_estimator: EntropyEstimator,
    /// Id columns for `exemplar_id` / `null_exemplar_id` (spec §4.2); empty: none.
    pub id_columns: Vec<String>,
```

Update every `Params { .. }` literal: tests get `entropy_estimator: EntropyEstimator::Sample, id_columns: Vec::new(),`; the API constructors get the values from their params (Step 4).

- [ ] **Step 3: AMS in the streaming levels** — `partial.rs`:
  - `BatchStats` gains `/// AMS: (sketch seed, every row's key in order: null → NULL_KEY). pub row_keys: Option<(u64, Vec<u64>)>,`; `BatchStats::of` gains a parameter `ams_seed: Option<u64>` and fills it from the `enc` it already computes:
    ```rust
            row_keys: ams_seed.map(|seed| {
                let keys = enc.values.iter().zip(&enc.is_null)
                    .map(|(&v, &null)| if null { NULL_KEY } else { hash_key(v) })
                    .collect();
                (seed, keys)
            }),
    ```
    with `const NULL_KEY: u64 = 0x5EED_0000_0000_0001; // nulls' AMS key (a 64-bit hash collides with it at ~2⁻⁶⁴)`.
  - `LevelStats` gains `pub ams: Option<AmsEntropy>` (`use crate::techniques::ams_entropy::AmsEntropy;`).
  - `LevelStats::nulls(rows)`: also `if let Some(a) = &mut self.ams { (0..rows).for_each(|_| a.update(NULL_KEY)); }`.
  - `LevelStats::absorb`: before `self.n += b.n;` add
    ```rust
        if let Some((seed, keys)) = b.row_keys {
            let prior = self.n; // every earlier row of this level was null (first typed batch) or fed already
            let a = self.ams.get_or_insert_with(|| {
                let mut a = AmsEntropy::new(AMS_SAMPLES, seed);
                (0..prior).for_each(|_| a.update(NULL_KEY));
                a
            });
            keys.into_iter().for_each(|k| a.update(k));
        }
    ```
    with `pub(crate) const AMS_SAMPLES: usize = 4_096;`. (Once created, `nulls` keeps it in step, so `prior` only matters on creation.)
  - Other `BatchStats::of` callers (inner levels, tests) pass `None`.
- [ ] **Step 4: Streaming wiring** — `Streaming::batch_stats`: the outer call becomes
  ```rust
        let ams_seed = (self.params.entropy_estimator == EntropyEstimator::Ams).then(|| {
            sketch_seed(seed, MARGINAL, name_key(s.name().as_str()))
        });
        let outer = BatchStats::of(s, self.n_rows, seed, top, ams_seed)?;
  ```
  (`use crate::techniques::ams_entropy::{name_key, sketch_seed, MARGINAL};`). In `Streaming::row`, the overflow arm:
  ```rust
            Some(d) => Some(match (&o.ams, self.params.entropy_estimator) {
                (Some(a), EntropyEstimator::Ams) => {
                    let (h, se) = a.estimate_with_se();
                    Entropy::from_ams(h, se, o.n)
                }
                _ => Entropy::from_sample(/* as in Task 6 */),
            }),
  ```
- [ ] **Step 5: API, Python binding, wrapper** —
  - `bindings/api/recommenders.rs`: `pub use crate::recommenders::engine::EntropyEstimator;`; `StreamingParams` gains `pub entropy_estimator: EntropyEstimator` and passes it into `Params`; `OneShotParams` is unchanged here (id columns in Task 8); fix the test helpers' literals.
  - `bindings/python.rs` `StreamingRecommender::new`: add `entropy_estimator` to the signature (`#[pyo3(signature = (*, reservoir_rows, block_rows, categorical_threshold, top_k, zstd_level, seed, boolean_pairs, entropy_estimator="sample"))]`, parameter `entropy_estimator: &str`) and map `"sample"` / `"ams"` to the enum, anything else → `api::Error::InvalidInput(format!("entropy_estimator must be \"sample\" or \"ams\", got {other:?}"))`.
  - `bindings/capi.rs`: pass `entropy_estimator: api::EntropyEstimator::Sample` for now (Task 9 adds the parameter).
  - `analytics/_plugin.py` `streaming_recommender(...)`: add the keyword `entropy_estimator: str` (it lists its keywords explicitly) and pass it to the Rust class; `analytics/recommend/streaming.py`: keyword `entropy_estimator: str = "sample"` forwarded to it; document it in the class docstring: "`entropy_estimator`: past the exact phase, `"sample"` (bottom-k sample, no extra cost) or `"ams"` (AMS sketch fed every row: ≈ 4 096 estimators per column)".
- [ ] **Step 6: Tests** — in `test_streaming_entropy_after_overflow` parametrize `estimator` over `["sample", "ams"]` and construct `StreamingRecommender(entropy_estimator=estimator)`; add to `tests/test_recommender_profile.py`:
  ```python
  def test_entropy_estimator_is_validated():
      with pytest.raises(ValueError, match="entropy_estimator"):
          StreamingRecommender(entropy_estimator="exact")
  ```
  `cargo test --lib` → pass; build; `$PY -m pytest tests/test_recommender_profile.py tests/test_streaming_recommend.py tests/test_pairwise_entropy.py tests/test_threeway_entropy.py -q` → pass (the AMS techniques must be unchanged). Commit `"Streaming recommender: AMS entropy estimator (entropy_estimator keyword)"`.

---

### Task 8: Id columns (`exemplar_id`, `null_exemplar_id`)

Spec §4.2. One mechanism for both recommenders: per output row an optional one-row `RecordBatch` of the id columns, spliced in as a Struct column after `with_maps`.

**Files:** `src/recommenders/schema.rs`, `src/recommenders/oneshot.rs`, `src/recommenders/streaming/mod.rs`, `src/bindings/{api/recommenders.rs,python.rs,capi.rs}`, `analytics/recommend/{oneshot,streaming,_input}.py`, tests.

- [ ] **Step 1: `with_ids`** in `schema.rs`:

```rust
/// Replaces the Null-typed `exemplar_id` / `null_exemplar_id` columns by Structs of the id
/// columns (spec 2026-10-18 §4.2). `ids[i]` = (exemplar, null exemplar) one-row batches of
/// output row i, all with `schema`; None → a null struct.
pub(crate) fn with_ids(
    b: RecordBatch,
    schema: &SchemaRef,
    ids: &[(Option<RecordBatch>, Option<RecordBatch>)],
) -> Result<RecordBatch, ArrowError> {
    let fields = schema.fields().clone();
    let struct_type = AT::Struct(fields.clone());
    let column = |pick: &dyn Fn(&(Option<RecordBatch>, Option<RecordBatch>)) -> Option<&RecordBatch>| {
        let null_row = new_null_array(&struct_type, 1);
        let parts: Vec<ArrayRef> = ids
            .iter()
            .map(|cell| match pick(cell) {
                Some(r) => StructArray::try_new(fields.clone(), r.columns().to_vec(), None)
                    .map(|s| Arc::new(s) as ArrayRef),
                None => Ok(null_row.clone()),
            })
            .collect::<Result<_, _>>()?;
        if parts.is_empty() {
            return Ok(new_empty_array(&struct_type));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
        arrow_select::concat::concat(&refs)
    };
    let exemplar = column(&|c| c.0.as_ref())?;
    let null_exemplar = column(&|c| c.1.as_ref())?;
    let out_schema = b.schema();
    let mut out_fields = Vec::with_capacity(b.num_columns());
    let mut out_columns = Vec::with_capacity(b.num_columns());
    for (f, c) in out_schema.fields().iter().zip(b.columns()) {
        let replaced = match f.name().as_str() {
            "exemplar_id" => Some(exemplar.clone()),
            "null_exemplar_id" => Some(null_exemplar.clone()),
            _ => None,
        };
        match replaced {
            Some(a) => {
                out_fields.push(Field::new(f.name(), struct_type.clone(), true));
                out_columns.push(a);
            }
            None => {
                out_fields.push(f.as_ref().clone());
                out_columns.push(c.clone());
            }
        }
    }
    let options = RecordBatchOptions::new().with_row_count(Some(b.num_rows()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(out_fields)), out_columns, &options)
}
```

(imports: `arrow_array::{new_empty_array, new_null_array, Array, StructArray}`, `arrow_schema::SchemaRef`.) Unit test: a 2-row batch with Null `exemplar_id`/`null_exemplar_id` columns, ids `[(Some(row(7, "a")), None), (None, Some(row(9, "b")))]` with a schema `{id: Int64, tag: Utf8}` → the struct columns hold `{7, a}`, null / null, `{9, b}`.

- [ ] **Step 2: Validation helper** (shared) in `schema.rs`:

```rust
/// The id columns' sub-schema of `schema`, checked (spec §4.2): duplicates and missing
/// names are InvalidInput (the caller maps the message).
pub(crate) fn id_schema(schema: &Schema, id_columns: &[String]) -> Result<SchemaRef, String> {
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::with_capacity(id_columns.len());
    for name in id_columns {
        if !seen.insert(name.as_str()) {
            return Err(format!("duplicate id column {name:?}"));
        }
        let f = schema
            .field_with_name(name)
            .map_err(|_| format!("id column {name:?} is not in the input"))?;
        fields.push(f.as_ref().clone().with_nullable(true));
    }
    Ok(Arc::new(Schema::new(fields)))
}
```

- [ ] **Step 3: One-shot** — `OneShot` keeps `ids: Option<(SchemaRef, RecordBatch)>` (the id columns of the added frame, set in `add` after the duplicate check: `id_schema(&schema, &self.params.id_columns).map_err(Error::InvalidInput)?` then `batch.project(&indices)` with the matching column indices). In `build`, when `ids` is Some: per output row, the exemplar row index → `Some(id_batch.slice(row, 1))` (`with_ids` copies them), marked / ineligible rows → `(None, None)`; then `with_ids(batch, &schema, &cells)` after `with_maps`. The exemplar rows are read from the computed `Prepared` rows: keep them alongside (`Prepared::exemplars()` returning `(Option<u64>, Option<u64>)` from `described.outer.freq`).

- [ ] **Step 4: Streaming** — `Streaming` keeps `id_schema: Option<SchemaRef>` (from the first batch) and `first_ids: Option<RecordBatch>` (row 0's ids); `Exemplars` stays as is, and `Column` gains `ids: (Option<RecordBatch>, Option<RecordBatch>)`. In `add`, before the commit:
  - if `params.id_columns` is non-empty: `let ids_schema = id_schema(&schema, &self.params.id_columns).map_err(Error::InvalidInput)?;` and, if `self.id_schema` is Some and its field types differ from this batch's, InvalidInput `"id column {name:?}: type changed from {a} to {b}"`;
  - a closure `slice(i) -> Result<RecordBatch>` copying row `i` of each id column with `partial::copy_rows(&batch.column(idx), i, 1)` (copied, so no batch buffer stays pinned) into a `RecordBatch` with the id schema;
  - compute the id slices for the `firsts` found in Task 5 Step 3 (exemplar row and null row, batch-relative index = global − start), the batch's row-0 slice for absent / Null-typed columns, and `first_ids` when `start == 0 && rows > 0`.
  Commit them next to `exemplars` (same conditions as Task 5 Step 4; a column created with `start > 0` takes `first_ids` as its null-exemplar ids). In `result`, when `self.id_schema` is Some: `with_ids(batch, schema, &columns.iter().map(|c| c.ids.clone()).collect::<Vec<_>>())`.
- [ ] **Step 5: API, Python** — `OneShotParams` and `StreamingParams` gain `pub id_columns: Vec<String>` (into `Params.id_columns`); `python.rs` both constructors take `id_columns: Vec<String>` (signature default `id_columns=Vec::new()`); `capi.rs` passes `Vec::new()` for now (Task 9); `analytics/_plugin.py` `oneshot_recommender` / `streaming_recommender` gain the keyword `id_columns: list[str]` passed through. Python wrappers: keyword `id_columns: Sequence[str] | None = None` on both, forwarded as `list(id_columns or ())`; in `add`, after `_input.prepare(frame)`, if any id column is in `ineligible` raise `ValueError(f"id column {name!r} is ineligible ({dtype}): it cannot identify rows")`. Docstrings: describe the five exemplar columns and `id_columns`.
- [ ] **Step 6: Tests** — add to `tests/test_recommender_profile.py`:

```python
def test_exemplar_ids_keep_their_types_and_nulls():
    frame = pl.DataFrame(
        {
            "order": [10, 11, None, 13],
            "line": ["a", "b", "c", "d"],
            "price": [None, 1.5, None, 2.5],
        }
    )
    for rec in (OneShotRecommender(id_columns=["order", "line"]), StreamingRecommender(id_columns=["order", "line"])):
        out = rec.add(frame).result()
        assert out.schema["exemplar_id"] == pl.Struct({"order": pl.Int64, "line": pl.String})
        price = out.filter(pl.col("column") == "price").row(0, named=True)
        assert price["exemplar_id"] == {"order": 11, "line": "b"}
        assert price["null_exemplar_id"] == {"order": 10, "line": "a"}
        order = out.filter(pl.col("column") == "order").row(0, named=True)
        assert order["null_exemplar_id"] == {"order": None, "line": "c"}
        line = out.filter(pl.col("column") == "line").row(0, named=True)
        assert line["null_exemplar_id"] is None


def test_without_id_columns_the_id_columns_are_null_typed():
    out = OneShotRecommender().add(pl.DataFrame({"a": [1, None]})).result()
    assert out.schema["exemplar_id"] == pl.Null and out.schema["null_exemplar_id"] == pl.Null


@pytest.mark.parametrize("batch_rows", [1, 2, 5])
def test_streaming_ids_equal_one_shot(batch_rows):
    frame = pl.DataFrame({"k": list(range(10)), "a": [None, None, 1, None, 2] * 2, "s": ["x"] * 3 + [None] * 7})
    want = {r["column"]: r for r in OneShotRecommender(id_columns=["k"]).add(frame).result().iter_rows(named=True)}
    rec = StreamingRecommender(id_columns=["k"])
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    for r in rec.result().iter_rows(named=True):
        for k in ["exemplar_id", "null_exemplar_id"]:
            assert r[k] == want[r["column"]][k], (r["column"], k)


def test_streaming_late_column_null_exemplar_takes_row_zero_ids():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [100, 101]}))
    rec.add(pl.DataFrame({"k": [102], "b": ["x"]}))
    b = rec.result().filter(pl.col("column") == "b").row(0, named=True)
    assert (b["exemplar_id"], b["null_exemplar_id"]) == ({"k": 102}, {"k": 100})


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
def test_id_column_errors(cls):
    with pytest.raises(ValueError, match="not in the input"):
        cls(id_columns=["missing"]).add(pl.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="duplicate id column"):
        cls(id_columns=["a", "a"]).add(pl.DataFrame({"a": [1]}))


def test_streaming_id_column_missing_from_a_later_batch():
    rec = StreamingRecommender(id_columns=["k"])
    rec.add(pl.DataFrame({"k": [1], "a": [2]}))
    with pytest.raises(ValueError, match="not in the input"):
        rec.add(pl.DataFrame({"a": [3]}))


def test_ineligible_id_column_is_rejected():
    frame = pl.DataFrame({"k": pl.Series([1, 2], dtype=pl.Int128), "a": [1, 2]})
    with pytest.raises(ValueError, match="ineligible"):
        OneShotRecommender(id_columns=["k"]).add(frame)
```

Run `cargo test --lib` → pass; build; `$PY -m pytest tests/test_recommender_profile.py tests/test_recommend.py tests/test_streaming_recommend.py -q` → pass. Commit `"Recommenders: id_columns — typed exemplar_id / null_exemplar_id"`.

---

### Task 9: C ABI and Java

Spec §5. New trailing parameters before `out` / `error`.

**Files:** `src/bindings/capi.rs`, `bindings/java/src/main/java/io/github/benssutton/analytics/{OneShotParams,StreamingParams,OneShotRecommender,StreamingRecommender}.java`, new `EntropyEstimator.java`, Java tests.

- [ ] **Step 1: C ABI** — `analytics_oneshot_recommender_new(..., n_bool_pairs, const char *const *id_columns, size_t n_id_columns, out, error)`; `analytics_streaming_recommender_new(..., n_bool_pairs, const char *const *id_columns, size_t n_id_columns, uint8_t entropy_estimator, out, error)` with `0` → Sample, `1` → Ams, other → InvalidInput `"entropy_estimator must be 0 (sample) or 1 (ams), got {v}"`. Reuse `strings(ptrs, n)` for the names, but give it a `what: &str` parameter so its messages say "id-column" vs "boolean-pair" (update the two `boolean_pairs` calls). Update the doc comments and the capi unit tests' calls (pass `ptr::null(), 0` and `0`); add a capi test that `entropy_estimator = 2` returns InvalidInput.
- [ ] **Step 2: Java** — new `public enum EntropyEstimator { SAMPLE, AMS; byte nativeValue() { return (byte) ordinal(); } }`. `OneShotParams` gains a trailing component `List<String> idColumns` (copied with `List.copyOf`; the existing shorter constructors pass `List.of()`), `StreamingParams` gains `List<String> idColumns, EntropyEstimator entropyEstimator` (shorter constructors pass `List.of()`, `EntropyEstimator.SAMPLE`; `Objects.requireNonNull`). The `NEW` descriptors add `ADDRESS, JAVA_LONG` (and `JAVA_BYTE` for streaming) before `out`; `create` passes `Native.cStrings(arena, params.idColumns())`, `(long) params.idColumns().size()` (and `params.entropyEstimator().nativeValue()`).
- [ ] **Step 3: Java tests** — in `OneShotRecommenderTest`: a frame with `k` (BigInt) and `a` (nullable BigInt) added with `idColumns = List.of("k")`; assert `exemplar_row` / `null_exemplar_row` values and that the `exemplar_id` vector is a `StructVector` whose `k` child holds the expected value. In `StreamingRecommenderTest`: the same with `EntropyEstimator.AMS`, asserting `entropy_method` is `"exact"` on a small frame (no overflow) and the constructor accepts the parameters; plus `idColumns` with a missing name → `IllegalArgumentException` on `add`.
- [ ] **Step 4: Run** — `cargo test --lib capi` → pass; `$PY scripts/check.py test-java` → BUILD SUCCESS. Commit `"C ABI and Java: id_columns and entropy_estimator"`.

---

### Task 10: Estimator benchmark and the default

Spec §3.4, §8.

**Files:** create `tests/performance/benchmark_streaming_entropy.py`; modify the spec §8 and, if the rule picks `ams`, the defaults (Python wrapper, `python.rs` signature default, `api` doc, Java `StreamingParams` shorter constructors, C ABI docs).

- [ ] **Step 1: The script**:

```python
"""
Streaming entropy past the exact phase: the bottom-k sample estimate vs the AMS sketch
(spec docs/superpowers/specs/2026-10-18-recommender-schema-entropy-exemplars-design.md §3.4).

Per shape and estimator, in a child process (peak memory is per process): one column of
10^6 rows streamed in 100 000-row batches through StreamingRecommender; reported: estimate,
interval, relative error and coverage against the exact entropy (one-shot), add() seconds,
peak working set above the loaded data.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_streaming_entropy.py
Results -> tests/performance/results/streaming_entropy.arrow
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/: datagen
sys.path.insert(0, str(HERE))  # memory_mb

from benchmark_ams_entropy import memory_mb  # noqa: E402
from datagen import large_dataset_mixed  # noqa: E402

from analytics.recommend import OneShotRecommender, StreamingRecommender  # noqa: E402

RESULTS_DIR = HERE / "results"
ROWS = 1_000_000
BATCH = 100_000


def zipf(alpha: float, d: int = 1_000_000) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    w = 1.0 / np.arange(1, d + 1) ** alpha
    return pl.DataFrame({"x": rng.choice(d, size=ROWS, p=w / w.sum())})


SHAPES = {
    "uniform 1e4": lambda: pl.DataFrame({"x": np.random.default_rng(0).integers(0, 10_000, ROWS)}),
    "uniform 1e5": lambda: pl.DataFrame({"x": np.random.default_rng(0).integers(0, 100_000, ROWS)}),
    "uniform 1e6": lambda: pl.DataFrame({"x": np.random.default_rng(0).integers(0, 1_000_000, ROWS)}),
    "zipf 0.8": lambda: zipf(0.8),
    "zipf 1.1": lambda: zipf(1.1),
    "zipf 1.5": lambda: zipf(1.5),
    "near-constant": lambda: pl.DataFrame(
        {"x": np.where(np.random.default_rng(2).random(ROWS) < 0.999, 0, np.arange(ROWS))}
    ),
    "large_dataset_mixed": large_dataset_mixed,
}


def child(shape: str, estimator: str) -> None:
    frame = SHAPES[shape]()
    gc.collect()
    current_before, _ = memory_mb()
    rec = StreamingRecommender(entropy_estimator=estimator, reservoir_rows=0)
    t0 = time.perf_counter()
    for off in range(0, frame.height, BATCH):
        rec.add(frame.slice(off, BATCH))
    add_s = time.perf_counter() - t0
    out = rec.result().select("column", "entropy", "entropy_low", "entropy_high", "entropy_method")
    _, peak = memory_mb()
    print(json.dumps({"add_s": add_s, "peak_mb": peak - current_before, "rows": out.to_dicts()}))


def main() -> None:
    rows = []
    for shape, make in SHAPES.items():
        frame = make()
        exact = {
            r["column"]: r["entropy"]
            for r in OneShotRecommender().add(frame).result().iter_rows(named=True)
        }
        for estimator in ("sample", "ams"):
            print(f"  {shape}: {estimator} ...", file=sys.stderr, flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", shape, estimator],
                capture_output=True, text=True, check=True, timeout=3600,
            )
            info = json.loads(proc.stdout.strip().splitlines()[-1])
            for r in info["rows"]:
                h = exact[r["column"]]
                if r["entropy_method"] == "exact":
                    continue  # never overflowed: nothing estimated
                rows.append(
                    {
                        "shape": shape,
                        "column": r["column"],
                        "estimator": estimator,
                        "exact": h,
                        "estimate": r["entropy"],
                        "low": r["entropy_low"],
                        "high": r["entropy_high"],
                        "rel_error": abs(r["entropy"] - h) / h if h else 0.0,
                        "covered": r["entropy_low"] <= h <= r["entropy_high"],
                        "add_s": info["add_s"],
                        "peak_mb": info["peak_mb"],
                    }
                )
    df = pl.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.write_ipc(RESULTS_DIR / "streaming_entropy.arrow", compression="zstd")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df.group_by("shape", "estimator").agg(
            pl.col("rel_error").max().alias("max_rel_error"),
            pl.col("covered").mean().alias("coverage"),
            pl.col("add_s").first(),
            pl.col("peak_mb").first(),
        ).sort("shape", "estimator"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
    else:
        main()
```

- [ ] **Step 2: Run it** (background; it can take tens of minutes) and keep the printed summary.
- [ ] **Step 3: Apply the rule** (spec §3.4): default `sample` unless its relative error exceeds 5% on a shape where `ams` stays within 5%. If the default changes to `ams`, change every default listed above and the docstrings.
- [ ] **Step 4: Spec §8** — replace "To be filled in by the implementation …" with the summary table (shape × estimator: max relative error, coverage, add s, peak MB), the chosen default and one sentence on why. Commit `"Benchmark: streaming entropy estimators; default <name> (spec §8)"`.

---

### Task 11: Documentation

**Files:** `CLAUDE.md`, `ToDo.md`.

- [ ] **Step 1: CLAUDE.md** — in the OneShotRecommender and Streaming Recommend paragraphs: the output columns now read `nullable, lossy_formatting, datafusion_* , polars_*, candidates` (replace every `rec_*` mention); the new profile columns (entropy with interval and method; exemplar rows, value and typed ids) and keywords (`id_columns` both; `entropy_estimator` streaming, with the chosen default); streaming memory: AMS adds ≈ 4 096 estimators per eligible column when selected; the spec pointer `docs/superpowers/specs/2026-10-18-recommender-schema-entropy-exemplars-design.md`; the benchmark `tests/performance/benchmark_streaming_entropy.py` with its outcome in one sentence; Project Structure: `engine/entropy.rs`, `engine/exemplar.rs`, `tests/test_recommender_profile.py`; Java: the new params. Current Focus: next is ToDo 1a–g.
- [ ] **Step 2: ToDo.md** — ToDo 1.0: "**Skeleton done** (spec 2026-10-18): layout, renames, reserved columns. The ClickHouse type mapping, thresholds, cost model, REE / LZ4 sizes and the `CREATE TABLE` handoff move to ToDo 1's specs."; ToDo 2 and 4: "**Done** (spec 2026-10-18)" with one line each (inner-level entropy not done); the agreed starting order line → `1a–g next (1.0 skeleton, 4, 2 done).`
- [ ] **Step 3:** Commit `"Docs: recommender schema, entropy and exemplars"`.

---

### Task 12: Full verification

- [ ] `$PY scripts/check.py ci` → all checks pass (fmt-check, lint, layering, build, test-rust, test-py, test-java).
- [ ] `$PY -m pytest -m slow -q` → pass.
- [ ] Check against the spec: §2 (order exactly as §2.1 in both recommenders; reserved columns absent), §3 (exact, sample, ams; intervals; method), §4 (rows, value, ids, edge cases), §5 (all bindings), §6 (tests), §7 (docs), §8 (filled). Report deviations to the user rather than fixing silently.
