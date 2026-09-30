# Streaming Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stateful `StreamingRecommender` that takes record batches over time, keeps exact running statistics plus a bounded sample of row blocks in Rust, and produces Recommend's dtype recommendations on demand. It is exposed through `api.rs`, pyo3 (plus a Python wrapper) and the C ABI.

**Architecture:**
- Each batch is profiled per column with describe.rs's existing kernels into a `BatchStats` (`partial.rs`). That is absorbed, in stream order, into a mergeable `LevelStats`, and blocks of rows are offered to a seeded Algorithm-L reservoir (`reservoir.rs`).
- `finish()` (`streaming.rs`) turns each `LevelStats` into the `Profile`/`Level` recommend.rs's unchanged rules read, and chooses candidates by *proof from statistics* (`prove`) instead of cast-and-verify. Uncompressed sizes are analytic.
- The sampled blocks are cast and verified as a cross-check, and measured for ZSTD.

**Tech Stack:** Rust (polars 0.51, arrow-rs 60, rayon, zstd, ryu), pyo3 0.25, Python 3.12 + Polars, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-streaming-recommender-design.md`. Read it first.

---

## Conventions for every task

Rust tests, from `services/analytics/` in Git Bash:

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics
export PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe
export PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH"
cargo test --lib <module>::
```

Without the `PATH` line, tests fail with `STATUS_DLL_NOT_FOUND`.

Python build and tests:

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/<file> -v
```

Commit messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

`docs/superpowers/` is git-ignored, but its files are tracked, so add them with `git add -f`.

## Spec amendments (found while planning; Task 16 writes them into the spec)

1. **Polars `string_view` sizes are simulated, not a formula.** Polars packs values longer than 12 bytes into data blocks whose capacity doubles (8 KiB → 16 MiB, `polars_views` in recommend.rs), and each block is padded separately. `ViewSim` replays that builder on value lengths in stream order:
   - over all values, for Utf8/Binary results;
   - over distinct values in first-occurrence order, for a dictionary's values.

   The size is then exact. `sum_len_gt12` is not needed.
2. **Lossy counters.** The exact per-value set is:
   - `n_neg_zero` and `n_int_lead0`;
   - `raw_frac_min` / `raw_frac_max`;
   - `n_f32_render_diff` / `n_f64_render_diff`;
   - `n_iso_time_noncanonical` (no seconds, or fraction digits ≠ chrono's 0/3/6/9 grouping);
   - `n_iso_space_sep`;
   - `n_iso_offset_noncanonical` (a zero offset not written `Z`);
   - `FloatStats::n_neg_zero`.

   arrow-cast renders from the value alone: floats with ryu, timestamps with `NaiveDateTime` Debug or RFC 3339 `AutoSi` with `Z`, decimals with exactly `scale` places. So these counters reproduce one-shot `lossy()`.
3. **Overflowed distinct tracking** gives the estimate method `overflowed` (a new `Method::Overflowed`), with `est_cardinality = categorical_threshold + 1`. The dictionary is rejected by the existing `c > categorical_threshold` reason.
4. **Ineligible columns** are marked with `mark_ineligible(name, dtype)` before the batches that omit them. Their `n_null` is reported as null, because the caller never sends their values.
5. **pyo3 stays at 0.25**, so the GIL is released with the existing `run` / `allow_threads` helper, not `detach`.
6. **Parity exclusions.** When the original type is kept, `rec_polars_type` isn't compared: one-shot fills it in Python. Original candidates' `predicted_bytes`, and `rec_*_size_bytes` of a kept original whose size is a per-batch sum, are compared only for single-batch streams.
7. **`min` / `max`** are rendered by arrow-cast from the kept extreme values.

## File map

| File | Change |
|---|---|
| `services/analytics/src/describe.rs` | New scanner and float statistics, `frequency_map` with a row offset; kernels made `pub(crate)` |
| `services/analytics/src/cardinality_estimators.rs` | `Method::Overflowed` |
| `services/analytics/src/recommend.rs` | `Level` carries counts, extremes and the few distinct values (constructor `of_values` / `of_block`); `prove`, `lossy_by_stats`, `list_candidates`, `Pick`, `pick_by_stats`, `pick_list_by_stats`; `Rec` ZSTD sizes become `Option`; helpers made `pub(crate)` |
| `services/analytics/src/partial.rs` (new) | `ViewSim`, `Key`/`Ext`, `KeyStat`, `Distinct`, `BatchStats`, `LevelStats` |
| `services/analytics/src/reservoir.rs` (new) | `Cursor` (Algorithm L), `Piece`, `Block`, `Reservoir` |
| `services/analytics/src/streaming.rs` (new) | `Column`, `Streaming::{new, add, mark_ineligible, finish}`, output schema |
| `services/analytics/src/lib.rs` | `mod partial; mod reservoir; mod streaming;` |
| `services/analytics/src/arrow_io.rs` | `import_array` made `pub(crate)` |
| `services/analytics/src/api.rs` | `StreamingParams`, `StreamingRecommender` |
| `services/analytics/src/python.rs` | `#[pyclass] StreamingRecommender`; stream reader helper |
| `services/analytics/src/capi.rs` | `analytics_recommender_{new,add,finish,free}` |
| `services/analytics/analytics/_plugin.py` | `streaming_recommender(...)` |
| `services/analytics/analytics/recommend/streaming.py` (new) | Python `StreamingRecommender` |
| `services/analytics/analytics/recommend/__init__.py` | export `StreamingRecommender` |
| `tests/test_streaming_recommend.py` (new) | accuracy tests |
| `tests/performance/benchmark_streaming_recommend.py` (new) | speed script |
| `CLAUDE.md`, the spec | documentation |

---

### Task 0: Branch

- [ ] **Step 1: Create the branch**

```bash
cd /c/Users/Alexander/turbo-parakeet
git checkout main && git pull --ff-only
git checkout -b streaming-recommender
```

Expected: `Switched to a new branch 'streaming-recommender'`.

---

### Task 1: New scanner and float statistics (describe.rs)

These statistics prove string → Float and string → Timestamp(ns) candidates, and reproduce `rec_lossy_formatting` without the data. They are internal: `value_fields()` and Describe's output are unchanged.

**Files:**
- Modify: `services/analytics/src/describe.rs`

- [ ] **Step 1: Write the failing tests** at the end of `describe.rs`'s `mod tests`:

```rust
    #[test]
    fn scanner_streaming_statistics_numeric() {
        let mut st = StringStats::default();
        for v in ["-0.00", "007.50", "1.5", "12", "0.30000000000000001", "16777217"] {
            st.add(v.as_bytes());
        }
        assert_eq!(st.n_neg_zero, 1);
        assert_eq!(st.n_int_lead0, 1);
        assert_eq!((st.raw_frac_min, st.raw_frac_max), (Some(0), Some(17)));
        // f64: only "0.3…01" changes value (parses to 0.3); all but "1.5" render differently.
        assert_eq!((st.n_f64_roundtrip_fail, st.n_f64_render_diff), (1, 5));
        // f32: "0.3…01" and "16777217" (→ 16777216) change value.
        assert_eq!((st.n_f32_roundtrip_fail, st.n_f32_render_diff), (2, 5));
    }

    #[test]
    fn scanner_streaming_statistics_iso() {
        let mut st = StringStats::default();
        for v in [
            "2024-01-01 10:00:00",
            "2024-01-01T10:00",
            "2024-01-01T10:00:00.5",
            "2024-01-01T10:00:00.500+00:00",
            "2024-01-01T10:00:00Z",
            "10:00:00.120000",
        ] {
            st.add(v.as_bytes());
        }
        assert_eq!(st.n_iso_space_sep, 1);
        // no seconds; ".5" (chrono prints .500); ".120000" (chrono prints .120)
        assert_eq!(st.n_iso_time_noncanonical, 3);
        assert_eq!(st.n_iso_offset_noncanonical, 1); // "+00:00" renders as "Z"
        assert_eq!(st.iso_instant_min, Some(1_704_103_200_000_000_000));
        assert_eq!(st.iso_instant_max, Some(1_704_103_200_500_000_000));
        let merged = StringStats::default().merge(st.clone());
        assert_eq!(merged.n_iso_time_noncanonical, 3);
    }

    #[test]
    fn float_negative_zero() {
        let st = float_stats(&Series::new("x".into(), &[0.0f64, -0.0, 1.0]))
            .unwrap()
            .unwrap();
        assert_eq!(st.n_neg_zero, 1);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib describe::`
Expected: compile errors `no field n_neg_zero on type StringStats` and similar.

- [ ] **Step 3: Implement.**

(a) Add `use crate::recommend::canon;` to the `use` block at the top of `describe.rs`.

(b) In `struct Numeric`, after `sig_digits`, add:

```rust
    /// Fraction digits as written, trailing zeros kept ("1.50" → 2; an integer 0).
    pub raw_frac_digits: u32,
    /// The integer part has more than one digit and starts with 0 ("007", "00.5").
    pub int_lead0: bool,
```

(c) In `scan_numeric`, replace

```rust
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
```

with

```rust
    let raw_frac_digits = frac.len() as u32;
    let frac = &frac[..frac.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)];
```

and add two fields to the returned `Numeric { … }`:

```rust
        raw_frac_digits,
        int_lead0: int.len() > 1 && int[0] == b'0',
```

(d) `FloatStats`:
- add the field `pub n_neg_zero: u64,` with the doc `/// Values equal to -0.0 (arrow-cast renders them "-0.0", a decimal "0").`;
- make `fn merge` `pub(crate) fn merge`, and add `n_neg_zero: self.n_neg_zero + o.n_neg_zero,` to it;
- in both `add_f64` and `add_f32`, as the first statement of the final `else { … }` branch, add:

```rust
            self.n_neg_zero += (x == 0.0 && x.is_sign_negative()) as u64;
```

(e) `StringStats`:
- change `#[derive(Default)]` to `#[derive(Default, Clone)]`;
- add these fields after `iso_n_midnight`:

```rust
    /// Numeric values that are zero written with a minus sign ("-0", "-0.00").
    pub n_neg_zero: u64,
    /// Numeric values whose integer part has a leading zero ("007", "00.5").
    pub n_int_lead0: u64,
    /// Min / max fraction digits as written (trailing zeros kept; integers 0).
    pub raw_frac_min: Option<u32>,
    pub raw_frac_max: Option<u32>,
    /// Numeric values whose Float32 / Float64 parse is not canonically equal to the text
    /// (the one-shot `verify_text` float check, value by value).
    pub n_f32_roundtrip_fail: u64,
    pub n_f64_roundtrip_fail: u64,
    /// Numeric values whose Float32 / Float64 parse arrow-cast renders (ryu) as other text.
    pub n_f32_render_diff: u64,
    pub n_f64_render_diff: u64,
    /// UTC instants (ns) of ISO datetimes, both kinds.
    pub iso_instant_min: Option<i128>,
    pub iso_instant_max: Option<i128>,
    /// ISO times / datetimes whose time part arrow-cast renders differently: no seconds,
    /// or fraction digits other than chrono's 0 / 3 / 6 / 9 grouping of the significant ones.
    pub n_iso_time_noncanonical: u64,
    /// ISO datetimes written with a space instead of `T`.
    pub n_iso_space_sep: u64,
    /// ISO datetimes with a zero offset not written `Z` (arrow-cast renders UTC as `Z`).
    pub n_iso_offset_noncanonical: u64,
```

(f) Add this free function above `impl StringStats`:

```rust
/// Fraction digits chrono prints for a time with `sig` significant ones: 0, 3, 6 or 9.
fn chrono_frac_digits(sig: u32) -> u32 {
    match sig {
        0 => 0,
        1..=3 => 3,
        4..=6 => 6,
        _ => 9,
    }
}
```

(g) Replace `StringStats::add` with:

```rust
    pub fn add(&mut self, b: &[u8]) {
        if let Some(n) = scan_numeric(b) {
            self.n_numeric += 1;
            self.max_int_digits = self.max_int_digits.max(Some(n.int_digits));
            self.max_frac_digits = self.max_frac_digits.max(Some(n.frac_digits));
            self.min_frac_digits = opt_min(self.min_frac_digits, Some(n.frac_digits));
            self.max_sig_digits = self.max_sig_digits.max(Some(n.sig_digits));
            self.raw_frac_min = opt_min(self.raw_frac_min, Some(n.raw_frac_digits));
            self.raw_frac_max = self.raw_frac_max.max(Some(n.raw_frac_digits));
            self.n_int_lead0 += n.int_lead0 as u64;
            self.n_neg_zero += (n.sig_digits == 0 && b[0] == b'-') as u64;
            // The numeric grammar is ASCII, so the bytes are valid UTF-8.
            self.float_probe(std::str::from_utf8(b).unwrap_or_default());
            if n.is_int {
                self.n_numeric_int += 1;
                self.n_leading_zero += n.leading_zero as u64;
                match parse_i128(b) {
                    Some(v) => {
                        self.int_min = opt_min(self.int_min, Some(v));
                        self.int_max = self.int_max.max(Some(v));
                    }
                    None => self.int_overflow = true,
                }
            }
            return; // a numeric string is never an ISO value
        }
        match scan_iso(b) {
            Some(Iso::Date) => self.n_iso_date += 1,
            Some(Iso::Time { frac, sig }) => {
                self.n_iso_time += 1;
                self.fraction(frac, sig);
                self.time_form(b, 0, frac, sig);
            }
            Some(Iso::DateTime {
                frac,
                sig,
                midnight,
            }) => {
                self.n_iso_datetime += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.datetime_form(b, frac, sig);
            }
            Some(Iso::DateTimeTz {
                frac,
                sig,
                midnight,
                offset_minutes,
            }) => {
                self.n_iso_datetime_tz += 1;
                self.fraction(frac, sig);
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
                self.datetime_form(b, frac, sig);
                self.n_iso_offset_noncanonical +=
                    (offset_minutes == 0 && b.last() != Some(&b'Z')) as u64;
            }
            None => {}
        }
    }

    /// The one-shot string → float verification and lossy check, value by value:
    /// parse, render as arrow-cast does (ryu), compare canonically and textually.
    fn float_probe(&mut self, s: &str) {
        let mut buf = ryu::Buffer::new();
        let r = buf.format(s.parse::<f64>().unwrap_or(f64::NAN));
        self.n_f64_roundtrip_fail += (canon(s) != canon(r)) as u64;
        self.n_f64_render_diff += (s != r) as u64;
        let r = buf.format(s.parse::<f32>().unwrap_or(f32::NAN));
        self.n_f32_roundtrip_fail += (canon(s) != canon(r)) as u64;
        self.n_f32_render_diff += (s != r) as u64;
    }

    /// `t0`: where the time part starts (0 for a bare time, 11 after a date).
    fn time_form(&mut self, b: &[u8], t0: usize, frac: u32, sig: u32) {
        let seconds = b.get(t0 + 5) == Some(&b':');
        self.n_iso_time_noncanonical += (!seconds || frac != chrono_frac_digits(sig)) as u64;
    }

    /// A datetime's separator, time part and UTC instant (both kinds).
    fn datetime_form(&mut self, b: &[u8], frac: u32, sig: u32) {
        self.n_iso_space_sep += (b[10] == b' ') as u64;
        self.time_form(b, 11, frac, sig);
        if let Some(v) = parse_iso(b) {
            let t = v.epoch_ns();
            self.iso_instant_min = opt_min(self.iso_instant_min, Some(t));
            self.iso_instant_max = self.iso_instant_max.max(Some(t));
        }
    }
```

(h) At the end of `StringStats::merge`, before `self`, add:

```rust
        self.n_neg_zero += o.n_neg_zero;
        self.n_int_lead0 += o.n_int_lead0;
        self.raw_frac_min = opt_min(self.raw_frac_min, o.raw_frac_min);
        self.raw_frac_max = self.raw_frac_max.max(o.raw_frac_max);
        self.n_f32_roundtrip_fail += o.n_f32_roundtrip_fail;
        self.n_f64_roundtrip_fail += o.n_f64_roundtrip_fail;
        self.n_f32_render_diff += o.n_f32_render_diff;
        self.n_f64_render_diff += o.n_f64_render_diff;
        self.iso_instant_min = opt_min(self.iso_instant_min, o.iso_instant_min);
        self.iso_instant_max = self.iso_instant_max.max(o.iso_instant_max);
        self.n_iso_time_noncanonical += o.n_iso_time_noncanonical;
        self.n_iso_space_sep += o.n_iso_space_sep;
        self.n_iso_offset_noncanonical += o.n_iso_offset_noncanonical;
```

(i) `canon` is `pub(crate)` in recommend.rs already. No change is needed there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib describe::`
Expected: all `describe::` tests pass, old and new.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe.rs
git commit -m "describe: scanner statistics that prove float/ns casts and lossy formatting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Frequency map with a row offset; kernels visible to the crate (describe.rs)

**Files:**
- Modify: `services/analytics/src/describe.rs`

- [ ] **Step 1: Write the failing test** in `describe.rs`'s `mod tests`:

```rust
    #[test]
    fn frequency_map_offsets_rows() {
        let whole = encode_series(&Series::new("x".into(), &["a", "b", "a"])).unwrap();
        let tail = encode_series(&Series::new("x".into(), &["b", "a"])).unwrap();
        let (w, t) = (frequency_map(&whole, 7, 0), frequency_map(&tail, 7, 1));
        let b = whole.values[1];
        assert_eq!((w[&b].first, t[&b].first), (1, 1));
        assert_eq!(w[&b].mask, t[&b].mask); // capture subsets use the global row
        assert_eq!(frequencies(&whole, 7, None).n_unique, w.len() as u64);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib describe::tests::frequency_map_offsets_rows`
Expected: `cannot find function frequency_map`.

- [ ] **Step 3: Implement.**

(a) Change `struct Entry { count: u64, first: u64, mask: u8 }` to:

```rust
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub count: u64,
    pub first: u64,
    pub mask: u8,
}
```

and `type Map = …` to `pub(crate) type Map = HashMap<u64, Entry, FixedState>;`.

(b) Add, above `pub(crate) fn frequencies`:

```rust
/// The per-value map of `col`, rows counted from `offset` (streaming: the global row
/// of the batch's first value), so first rows and capture subsets are global.
pub(crate) fn frequency_map(col: &EncodedColumn, seed: u64, offset: u64) -> Map {
    col.values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, offset as usize + i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge)
}
```

(c) In `frequencies`, replace the `let map = col.values.par_chunks(…) … .reduce(…);` statement with `let map = frequency_map(col, seed, 0);`.

(d) Make these functions `pub(crate)`: `arg_extremes`, `lengths`, `byte_lengths`, `strings`, `n_midnight` and `flatten`.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib describe::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe.rs
git commit -m "describe: frequency_map with a global row offset; kernels pub(crate)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Method::Overflowed` (cardinality_estimators.rs)

**Files:**
- Modify: `services/analytics/src/cardinality_estimators.rs`

- [ ] **Step 1: Write the failing test** in that file's `mod tests`:

```rust
    #[test]
    fn overflowed_method_name() {
        assert_eq!(Method::Overflowed.name(), "overflowed");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib cardinality_estimators::`
Expected: `no variant named Overflowed`.

- [ ] **Step 3: Implement.** Add the variant to `enum Method`:

```rust
    /// Streaming: distinct tracking stopped past `categorical_threshold`; no estimate.
    Overflowed,
```

and add the arm `Method::Overflowed => "overflowed",` to `name()`.

Check that no other exhaustive `match` on `Method` exists: `grep -n "Method::" src/*.rs`. Only `cardinality_estimators.rs` should match on it.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib cardinality_estimators::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/cardinality_estimators.rs
git commit -m "cardinality_estimators: Method::Overflowed for streaming

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `Level` carries counts, extremes and few distinct values (recommend.rs)

This is a pure refactor, so the existing tests are its tests. After it, the rules no longer read row values: they read `n_rows`, `n_null`, `int_range`, `float_range` and `few_distinct`, which streaming can supply without data.

**Files:**
- Modify: `services/analytics/src/recommend.rs`

- [ ] **Step 1: Replace `struct Level` and the start of `impl Level`.** The fields become:

```rust
/// One level of a column: the column, or a list's inner values.
pub(crate) struct Level<'a> {
    /// Polars dtype of these values.
    pub dtype: &'a PT,
    /// The values in Arrow's classic layout (CompatLevel::oldest): the whole level
    /// (one-shot), a sample block, or none (streaming) — the rules read only its type.
    pub values: ArrayRef,
    pub p: &'a Profile,
    pub n_rows: u64,
    pub n_null: u64,
    /// Exact min / max: integers, or a decimal's unscaled values.
    pub int_range: Option<(i128, i128)>,
    /// Exact min / max of a float level, NaN excluded.
    pub float_range: Option<(f64, f64)>,
    /// Every distinct non-null value as text, when there are at most five (the
    /// boolean-pair rule); empty otherwise.
    pub few_distinct: Vec<String>,
    pub n_midnight: Option<u64>,
    /// Uncompressed size of the original type.
    pub size_bytes: u64,
    /// How `size_bytes` was obtained, for the original candidate's evidence.
    pub size_note: &'static str,
    pub est: Estimate,
    /// Population rows ÷ frame rows.
    pub r: f64,
    /// "" or "inner: " — prefixes every rule name.
    pub prefix: &'static str,
    /// Text sources: `values` as LargeUtf8, built once and shared by cast, verify and lossy.
    pub text: OnceCell<Result<LargeStringArray, String>>,
}
```

- [ ] **Step 2: Add constructors and switch the count methods to fields.** In `impl Level<'_>`:
- replace `fn n_rows(&self) -> u64 { self.values.len() as u64 }` with `fn n_rows(&self) -> u64 { self.n_rows }`;
- replace `fn n_null` with `fn n_null(&self) -> u64 { self.n_null }`;
- make `fn shape` `pub(crate) fn shape`.

Then add a new `impl<'a> Level<'a>` block:

```rust
impl<'a> Level<'a> {
    /// A level over all its values (one-shot): counts, extremes and the few distinct
    /// values are read off `values` at Describe's argmin / argmax / top-5 rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of_values(
        dtype: &'a PT,
        values: ArrayRef,
        p: &'a Profile,
        n_midnight: Option<u64>,
        size_bytes: u64,
        est: Estimate,
        r: f64,
        prefix: &'static str,
    ) -> Result<Self, String> {
        let row = |i: Option<u64>| i.filter(|&i| (i as usize) < values.len());
        let (lo, hi) = (row(p.range.argmin), row(p.range.argmax));
        let int_range = match (dtype, lo, hi) {
            (PT::Decimal(..), Some(a), Some(b)) => values
                .as_primitive_opt::<Decimal128Type>()
                .map(|d| (d.value(a as usize), d.value(b as usize))),
            (dt, Some(a), Some(b)) if dt.is_integer() => {
                int_at(&values, a).zip(int_at(&values, b))
            }
            _ => None,
        };
        let float_range = match (is_float(dtype), lo, hi) {
            (true, Some(a), Some(b)) => Some((f64_at(&values, a), f64_at(&values, b))),
            _ => None,
        };
        let few_distinct = if is_text(dtype) && p.freq.n_unique <= 5 {
            p.freq
                .top5_idx
                .iter()
                .map(|&i| text_of(&values.slice(i as usize, 1)).map(|t| t.value(0).to_string()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        Ok(Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range,
            float_range,
            few_distinct,
            n_midnight,
            size_bytes,
            size_note: "measured",
            est,
            r,
            prefix,
            text: Default::default(),
        })
    }

    /// A level over one sample block, for `cast_to` / `verify` only (streaming's cross-check).
    pub(crate) fn of_block(dtype: &'a PT, values: ArrayRef, p: &'a Profile) -> Self {
        Level {
            dtype,
            n_rows: values.len() as u64,
            n_null: values.logical_null_count() as u64,
            values,
            p,
            int_range: None,
            float_range: None,
            few_distinct: Vec::new(),
            n_midnight: None,
            size_bytes: 0,
            size_note: "",
            est: estimate(0, 0, 0, 0, &[0; 7], None),
            r: 1.0,
            prefix: "",
            text: Default::default(),
        }
    }
}
```

- [ ] **Step 3: Switch the rules to the new fields.**

(a) In `candidates`, replace the integer arm's body (the `if let (Some(a), Some(b)) = (lvl.p.range.argmin, …) { if let (Some(lo), Some(hi)) = (int_at(…), int_at(…)) { r.integers(lo, hi, "integer", ""); } }` block) with:

```rust
                if let Some((lo, hi)) = lvl.int_range {
                    r.integers(lo, hi, "integer", "");
                }
```

(b) In `Rules::decimal`, replace everything from `let (p, v) = (self.lvl.p, &self.lvl.values);` through `let (lo, hi, s) = (unscaled(lo) / f, unscaled(hi) / f, scale - k);` with:

```rust
        let Some((lo, hi)) = self.lvl.int_range else {
            return;
        };
        let g = self.lvl.p.gcd.unwrap_or(1);
        let k = if g == 0 {
            scale
        } else {
            trailing_zeros10(g).min(scale)
        };
        let f = 10i128.pow(k as u32);
        let (lo, hi, s) = (lo / f, hi / f, scale - k);
```

(c) In `Rules::float`:
- replace `let (p, v) = (self.lvl.p, &self.lvl.values);` with `let p = self.lvl.p;`;
- replace the condition `if let (0, 0, Some(lo), Some(hi)) = (f.n_nan, f.n_inf, p.range.argmin, p.range.argmax) {` with `if let (0, 0, Some((lo, hi))) = (f.n_nan, f.n_inf, self.lvl.float_range) {`;
- delete the next line, `let (lo, hi) = (f64_at(v, lo), f64_at(v, hi));`.

(d) In `Rules::text`, replace the `let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 { … } else { Vec::new() };` statement, together with the comment above it, with:

```rust
        let distinct: Vec<String> = if lvl.p.freq.n_unique <= 5 {
            lvl.few_distinct.iter().map(|v| v.to_lowercase()).collect()
        } else {
            Vec::new()
        };
```

`Rules::text` no longer returns an error from this part, but keep its `Result` signature.

(e) In `Rules::original`, change the evidence to `evidence: format!("{} size_bytes={}", lvl.size_note, lvl.size_bytes),`.

- [ ] **Step 4: Construct levels with `of_values` in `recommend()`.** Replace `let outer = Level { … };` with:

```rust
    let outer = Level::of_values(
        s.dtype(),
        values.clone(),
        &d.outer,
        d.n_midnight,
        size_bytes,
        level_estimate(&d.outer, d.n_rows - d.n_null, q),
        r,
        "",
    )
    .map_err(err)?;
```

Replace `let inner_lvl = Level { … };` with:

```rust
                let child_size = ipc_body_bytes(child.as_ref(), None)?;
                let inner_lvl = Level::of_values(
                    inner.values.dtype(),
                    child,
                    &inner.profile,
                    None,
                    child_size,
                    level_estimate(
                        &inner.profile,
                        (inner.values.len() - inner.values.null_count()) as u64,
                        q,
                    ),
                    r,
                    "inner: ",
                )
                .map_err(err)?;
```

- [ ] **Step 5: Update the test-module literals.** Find them with `grep -n "Level {" src/recommend.rs`: every hit inside `mod tests`. Each literal has the fields `dtype, values, p, n_midnight, size_bytes, est, r, prefix, text`. Rewrite each as an `of_values` call with the same values in that order, dropping `text`. For example:

```rust
        let lvl = Level {
            dtype: s.dtype(),
            values: export_series(&s, CompatLevel::oldest()).unwrap(),
            p: &d.outer,
            n_midnight: d.n_midnight,
            size_bytes: 0,
            est: level_estimate(&d.outer, d.n_rows - d.n_null, None),
            r: 1.0,
            prefix: "",
            text: Default::default(),
        };
```

becomes

```rust
        let lvl = Level::of_values(
            s.dtype(),
            export_series(&s, CompatLevel::oldest()).unwrap(),
            &d.outer,
            d.n_midnight,
            0,
            level_estimate(&d.outer, d.n_rows - d.n_null, None),
            1.0,
            "",
        )
        .unwrap();
```

Afterwards `grep -n "Level {" src/recommend.rs` shows only the struct definition, the `impl` headers and the two constructor bodies.

- [ ] **Step 6: Run the tests**

Run: `cargo test --lib recommend::`
Expected: every existing `recommend::` test passes unchanged.

- [ ] **Step 7: Run the Python recommend tests (behaviour unchanged)**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_recommend.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add services/analytics/src/recommend.rs
git commit -m "recommend: Level carries counts, extremes and few distinct values

The rules read these fields instead of row values, so a Level can be built
from statistics alone (streaming).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `Rec` ZSTD sizes become optional; helpers visible to the crate (recommend.rs, arrow_io.rs)

**Files:**
- Modify: `services/analytics/src/recommend.rs`, `services/analytics/src/arrow_io.rs`

- [ ] **Step 1: Make `Rec`'s ZSTD fields optional.** In `pub(crate) struct Rec`, change `pub arrow_zstd: u64,` to `pub arrow_zstd: Option<u64>,` and `pub polars_zstd: u64,` to `pub polars_zstd: Option<u64>,`, with the doc comment `/// None when nothing was sampled (streaming, reservoir_rows = 0).` on each.

In `recommend()`:
- the tuple `(None, polars_bytes, polars_zstd)` becomes `(None, polars_bytes, Some(polars_zstd))`;
- the non-original branch's last element becomes `Some(ipc_body_bytes(layout.as_ref(), Some(params.zstd_level))?)`;
- `arrow_zstd: ipc_body_bytes(…, Some(params.zstd_level))?,` becomes `arrow_zstd: Some(ipc_body_bytes(chosen.array.as_ref(), Some(params.zstd_level))?),`.

In `rec_row`, replace `AnyValue::UInt64(r.arrow_zstd),` with `r.arrow_zstd.map_or(AnyValue::Null, AnyValue::UInt64),` and `AnyValue::UInt64(r.polars_zstd),` with `r.polars_zstd.map_or(AnyValue::Null, AnyValue::UInt64),`.

- [ ] **Step 2: Make these items `pub(crate)`:**
- in `recommend.rs`: `const VIEW_BLOCK`, `const VIEW_MAX_BLOCK`, `fn list_parts`, `type ListParts`, `fn wrap`, `fn to_polars_layout`, `fn polars_views`, `fn rec_fields`, `fn rec_row`, `fn first_success`, `fn verify` (already `pub(crate)`) and `fn cast_to` (already);
- in `arrow_io.rs`: `fn import_array`.

- [ ] **Step 3: Build and test**

Run: `cargo test --lib recommend:: && cargo test --lib arrow_io::`
Expected: PASS. `cargo build --release --no-default-features --target-dir target/capi` also succeeds; dead-code warnings are allowed there by `lib.rs`.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/recommend.rs services/analytics/src/arrow_io.rs
git commit -m "recommend: optional ZSTD sizes in Rec; helpers pub(crate) for streaming

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 6: Choosing from statistics — `prove`, `lossy_by_stats`, `list_candidates`, `Pick` (recommend.rs)

**Files:**
- Modify: `services/analytics/src/recommend.rs`

- [ ] **Step 1: Write the failing tests** in `recommend.rs`'s `mod tests`. The helpers `params()`, `describe_one`, `export_series` and `CompatLevel` are already imported there, and `sizes_of` and `ipc_body_bytes` reach the tests through `use super::*`.

```rust
    /// The one-shot choice (cast + verify) and the statistics-only pick for one series.
    fn both(s: Series, p: &Params) -> (Chosen, Pick) {
        let d = describe_one(&s, 0).unwrap();
        let values = export_series(&s, CompatLevel::oldest()).unwrap();
        let size = ipc_body_bytes(values.as_ref(), None).unwrap();
        let lvl = Level::of_values(
            s.dtype(),
            values,
            &d.outer,
            d.n_midnight,
            size,
            level_estimate(&d.outer, d.n_rows - d.n_null, None),
            1.0,
            "",
        )
        .unwrap();
        (choose(&lvl, p).unwrap(), pick_by_stats(&lvl, p).unwrap())
    }

    fn outcomes(c: &[Candidate]) -> Vec<(String, Outcome)> {
        c.iter().map(|c| (c.rule.clone(), c.outcome)).collect()
    }

    #[test]
    fn statistics_choose_like_cast_and_verify() {
        let strs = |v: &[Option<&str>]| Series::new("x".into(), v);
        let mut yes_no = params();
        yes_no.boolean_pairs = vec![("yes".into(), "no".into())];
        let cases: Vec<(Series, Params)> = vec![
            (strs(&[Some("1.50"), Some("2.25"), None]), params()),
            (strs(&[Some("1.5"), Some("2.25")]), params()),
            (strs(&[Some("-0"), Some("5")]), params()),
            (strs(&[Some("2024-01-01 10:00:00"), Some("2024-01-02 11:00:00")]), params()),
            (strs(&[Some("2024-01-01T10:00:00Z"), Some("2024-01-01T11:00:00+00:00")]), params()),
            (strs(&[Some("10:00:00.5"), Some("11:00:00")]), params()),
            (strs(&[Some("Yes"), Some("no"), Some("yes")]), yes_no),
            (strs(&[Some("true"), Some("false")]), params()),
            (strs(&[Some("1234567890.1"), Some("0.00000012345")]), params()),
            (strs(&[Some("2024-01-01"), Some("2024-02-01")]), params()),
            (strs(&[Some("2024-01-01T00:00:00"), Some("2024-02-01T00:00:00")]), params()),
            (strs(&[Some("0"), Some("1"), Some("1")]), params()),
            (Series::new("x".into(), &[0.0f64, -0.0, 1.5]), params()),
            (Series::new("x".into(), &[1.5f64, 2.25]), params()),
            (Series::new("x".into(), &[300i64, -2, 7]), params()),
        ];
        for (s, p) in cases {
            let (chosen, pick) = both(s.clone(), &p);
            assert_eq!(pick.target, chosen.target, "{s:?}");
            assert_eq!(pick.lossy, chosen.lossy, "{s:?}");
            assert_eq!(outcomes(&pick.candidates), outcomes(&chosen.candidates), "{s:?}");
        }
    }

    #[test]
    fn statistics_reject_a_float_that_underflows() {
        let tiny = format!("0.{}1", "0".repeat(400));
        let (chosen, pick) = both(Series::new("x".into(), &[tiny.as_str(), "1"]), &params());
        assert_eq!(pick.target, chosen.target);
        let f64c = pick
            .candidates
            .iter()
            .find(|c| c.rule == "string→float64")
            .unwrap();
        assert_eq!(f64c.outcome, Outcome::Failed);
        assert!(f64c
            .reason
            .as_deref()
            .unwrap()
            .starts_with("n_f64_roundtrip_fail=1"));
    }

    #[test]
    fn statistics_reject_nanoseconds_out_of_range() {
        let s = Series::new(
            "x".into(),
            &["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"],
        );
        let (chosen, pick) = both(s, &params());
        assert_eq!(pick.target, chosen.target);
        let ns = pick
            .candidates
            .iter()
            .find(|c| c.rule == "string→timestamp")
            .unwrap();
        assert_eq!(ns.outcome, Outcome::Failed);
        assert!(ns.reason.as_deref().unwrap().contains("iso_instant"));
    }

    #[test]
    fn list_statistics_choose_like_cast_and_verify() {
        let item = |v: &[i64]| Series::new("".into(), v);
        for s in [
            Series::new("x".into(), &[item(&[1]), item(&[2]), item(&[300])]),
            Series::new("x".into(), &[item(&[1, 2]), item(&[3]), item(&[])]),
        ] {
            let d = describe_one(&s, 0).unwrap();
            let classic = export_series(&s, CompatLevel::oldest()).unwrap();
            let sz = sizes_of(&s, &classic, 1).unwrap();
            let rec = recommend(&s, &classic, &d, &sz, &params()).unwrap();
            let inner = d.inner.as_ref().unwrap();
            let (_, child, width) = list_parts(&classic).unwrap();
            let outer = Level::of_values(
                s.dtype(),
                classic.clone(),
                &d.outer,
                None,
                sz[0],
                level_estimate(&d.outer, d.n_rows - d.n_null, None),
                1.0,
                "",
            )
            .unwrap();
            let child_size = ipc_body_bytes(child.as_ref(), None).unwrap();
            let il = Level::of_values(
                inner.values.dtype(),
                child,
                &inner.profile,
                None,
                child_size,
                level_estimate(
                    &inner.profile,
                    (inner.values.len() - inner.values.null_count()) as u64,
                    None,
                ),
                1.0,
                "inner: ",
            )
            .unwrap();
            let pick = pick_list_by_stats(&outer, &il, width, &params()).unwrap();
            assert_eq!(pa_name(&pick.target.arrow_type()), rec.arrow_type, "{s:?}");
            assert_eq!(outcomes(&pick.candidates), outcomes(&rec.candidates), "{s:?}");
            assert_eq!(pick.lossy, rec.lossy);
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib recommend::tests::statistics`
Expected: `cannot find function pick_by_stats` / `cannot find type Pick`.

- [ ] **Step 3: Implement `prove` and `lossy_by_stats`.** Add both after `pub(crate) fn lossy(…)`:

```rust
/// Proof by statistics (streaming spec §5.1 step 4): Ok when the statistics show that
/// `t` holds every value of the level. Every rule's own condition already proves its
/// candidate except string → Float (a value may not round-trip) and string →
/// Timestamp(ns) (the int64 nanosecond range).
pub(crate) fn prove(t: &Target, lvl: &Level) -> Result<(), String> {
    let Some(st) = lvl.p.strings.as_ref().filter(|_| is_text(lvl.dtype)) else {
        return Ok(());
    };
    match t {
        Target::Fixed(AT::Float32) if st.n_f32_roundtrip_fail > 0 => Err(format!(
            "n_f32_roundtrip_fail={}",
            st.n_f32_roundtrip_fail
        )),
        Target::Fixed(AT::Float64) if st.n_f64_roundtrip_fail > 0 => Err(format!(
            "n_f64_roundtrip_fail={}",
            st.n_f64_roundtrip_fail
        )),
        Target::Fixed(AT::Timestamp(TimeUnit::Nanosecond, _)) => {
            match (st.iso_instant_min, st.iso_instant_max) {
                (Some(lo), Some(hi)) if lo < i64::MIN as i128 || hi > i64::MAX as i128 => Err(
                    format!("iso_instant range {lo}..{hi} ns exceeds int64"),
                ),
                _ => Ok(()),
            }
        }
        _ => Ok(()),
    }
}

/// `lossy` from statistics alone: arrow-cast renders each value from the value itself
/// (integers canonically; decimals with exactly `scale` places; floats with ryu; times
/// and timestamps with chrono's 0/3/6/9 fraction digits, `T`, and `Z` for UTC), so the
/// scanner's per-value counters decide it exactly.
pub(crate) fn lossy_by_stats(t: &Target, lvl: &Level) -> bool {
    match t {
        Target::Original(_) | Target::Null | Target::Dictionary(..) | Target::Plain(_) => false,
        _ if is_text(lvl.dtype) => {
            let Some(st) = lvl.p.strings.as_ref() else {
                return true;
            };
            match t {
                Target::Boolean | Target::TimestampWithOffset(_) => lvl.n() > 0,
                Target::BoolPair(..) => lvl.few_distinct.iter().any(|v| v != "true" && v != "false"),
                Target::Fixed(to) => match to {
                    AT::Float32 => st.n_f32_render_diff > 0,
                    AT::Float64 => st.n_f64_render_diff > 0,
                    AT::Decimal32(_, s) | AT::Decimal64(_, s) | AT::Decimal128(_, s) => {
                        let s = Some(*s as u32);
                        st.raw_frac_min != s
                            || st.raw_frac_max != s
                            || st.n_int_lead0 > 0
                            || st.n_neg_zero > 0
                    }
                    AT::Date32 => st.n_iso_datetime > 0,
                    AT::Time32(_) | AT::Time64(_) => st.n_iso_time_noncanonical > 0,
                    AT::Timestamp(_, tz) => {
                        st.n_iso_time_noncanonical > 0
                            || st.n_iso_space_sep > 0
                            || (tz.is_some() && st.n_iso_offset_noncanonical > 0)
                    }
                    _ => st.n_neg_zero > 0, // integers: only "-0" renders differently
                },
                _ => false,
            }
        }
        Target::Fixed(AT::Float32 | AT::Float64) => false,
        _ if is_float(lvl.dtype) => lvl.p.floats.is_some_and(|f| f.n_neg_zero > 0),
        _ => false,
    }
}
```

- [ ] **Step 4: Split `choose_list` into `list_candidates` plus the attempt loop.** Replace the whole `fn choose_list` with:

```rust
/// A list level's outer candidates (scalar / list / array, then the original) around
/// the inner choice `it` (with its rank and sizes), unordered.
#[allow(clippy::too_many_arguments)]
fn list_candidates(
    lvl: &Level,
    inner: &Level,
    it: &Target,
    rank: Rank,
    predicted: u64,
    projected: f64,
    width: Option<i32>,
) -> Vec<Candidate> {
    let (n, nulls, r) = (lvl.n_rows() as f64, lvl.n_null() as f64, lvl.r);
    let kept = matches!(it, Target::Original(_));
    let (c, _) = inner.cardinality();
    let mut outer = Vec::new();
    let nested = kept && matches!(inner.dtype, PT::List(_) | PT::Array(..) | PT::Struct(_));
    let single = lvl.p.range.min_len == Some(1) && lvl.p.range.max_len == Some(1);
    if single && !(lvl.n_null() > 0 && inner.n_null() > 0) && !nested {
        let shape = Shape {
            n,
            nulls: nulls + inner.n_null() as f64,
            ..inner.shape()
        };
        let t = it.arrow_type();
        outer.push(candidate(
            Target::Scalar(Box::new(it.clone())),
            rank,
            "list→scalar",
            format!(
                "min_len=1 max_len=1 n_null={} inner_n_null={}",
                lvl.n_null(),
                inner.n_null()
            ),
            body_size(&t, &shape).and_then(|p| Ok((p, body_size(&t, &shape.project(r, c))?))),
        ));
    }
    match width {
        None if inner.n_rows() < 1 << 31 => outer.push(candidate(
            Target::List(Box::new(it.clone())),
            Rank::List,
            "large_list→list",
            format!("inner_n_values={}", inner.n_rows()),
            Ok((
                validity(n, nulls) + pad(4.0 * (n + 1.0)) + predicted as f64,
                validity(n * r, nulls * r) + pad(4.0 * (n * r + 1.0)) + projected,
            )),
        )),
        // An Array whose inner type is kept is the original type: no candidate.
        Some(w) if !kept => {
            // The child holds w slots per row; a null row's slots are null.
            let wf = w as f64;
            let shape = Shape {
                n: n * wf,
                nulls: inner.n_null() as f64 + nulls * wf,
                ..inner.shape()
            };
            let t = it.arrow_type();
            outer.push(candidate(
                Target::FixedList(Box::new(it.clone()), w),
                Rank::List,
                "array→array",
                format!("width={w}"),
                body_size(&t, &shape).and_then(|p| {
                    Ok((
                        validity(n, nulls) + p,
                        validity(n * r, nulls * r) + body_size(&t, &shape.project(r, c))?,
                    ))
                }),
            ));
        }
        _ => {}
    }
    let mut rules = Rules {
        lvl,
        shape: lvl.shape(),
        out: outer,
    };
    rules.original();
    rules.out
}

/// Lists: choose the inner type first, then wrap it — as a scalar when every list
/// holds one item, else as a List with 32-bit offsets (Array keeps its width).
///
/// The reported candidates are the outer level's (scalar / list / array / original,
/// in the order tried) followed by the inner level's (rules prefixed "inner: "), so
/// a list column shows two `chosen` entries: the outer choice and the inner choice.
fn choose_list(
    lvl: &Level,
    inner: &Level,
    rows: &[Option<(usize, usize)>],
    width: Option<i32>,
    params: &Params,
) -> Result<Chosen, String> {
    let ic = choose(inner, params)?;
    let cands = list_candidates(
        lvl,
        inner,
        &ic.target,
        ic.rank,
        ic.predicted,
        ic.projected,
        width,
    );
    let (i, array, cands) = first_success(cands, |t| wrap(t, &lvl.values, rows, &ic.array));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    let mut chosen = chosen_from(i, array, cands, lossy);
    chosen.candidates.extend(ic.candidates);
    Ok(chosen)
}
```

- [ ] **Step 5: Add `Pick` and the pick functions** after `choose_list`:

```rust
/// A choice made from statistics alone (streaming): the target and its sizes, no array.
pub(crate) struct Pick {
    pub target: Target,
    pub rank: Rank,
    pub predicted: u64,
    pub projected: f64,
    pub lossy: bool,
    /// In the order tried; a list's outer candidates, then its inner ones.
    pub candidates: Vec<Candidate>,
    /// A list's inner choice.
    pub inner: Option<Box<Pick>>,
}

fn pick_at(i: usize, cands: Vec<Candidate>, lossy: bool, inner: Option<Box<Pick>>) -> Pick {
    let c = &cands[i];
    let (target, rank, predicted, projected) = (c.target.clone(), c.rank, c.predicted, c.projected);
    Pick {
        target,
        rank,
        predicted,
        projected,
        lossy,
        candidates: cands,
        inner,
    }
}

/// `choose` with `prove` in place of cast + verify.
pub(crate) fn pick_by_stats(lvl: &Level, params: &Params) -> Result<Pick, String> {
    let (i, (), cands) = first_success(candidates(lvl, params)?, |t| prove(t, lvl));
    let lossy = lossy_by_stats(&cands[i].target, lvl);
    Ok(pick_at(i, cands, lossy, None))
}

/// `choose_list` from statistics: wrapping a proven inner type cannot fail.
pub(crate) fn pick_list_by_stats(
    lvl: &Level,
    inner: &Level,
    width: Option<i32>,
    params: &Params,
) -> Result<Pick, String> {
    let mut ic = pick_by_stats(inner, params)?;
    let cands = list_candidates(
        lvl,
        inner,
        &ic.target,
        ic.rank,
        ic.predicted,
        ic.projected,
        width,
    );
    let (i, (), mut cands) = first_success(cands, |_| Ok(()));
    let lossy = !matches!(cands[i].target, Target::Original(_)) && ic.lossy;
    cands.extend(std::mem::take(&mut ic.candidates));
    Ok(pick_at(i, cands, lossy, Some(Box::new(ic))))
}
```

- [ ] **Step 6: Run the tests**

Run: `cargo test --lib recommend::`
Expected: PASS, including the four new tests. If `statistics_choose_like_cast_and_verify` fails on a case, the message names the series. One-shot `lossy()` is the reference: fix the counter in describe.rs, or the arm in `lossy_by_stats`, until they agree.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/src/recommend.rs
git commit -m "recommend: choose from statistics (prove, lossy_by_stats, Pick)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Mergeable level statistics — `partial.rs`

**Files:**
- Create: `services/analytics/src/partial.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod partial;` after `mod minhash;`)

- [ ] **Step 1: Create `partial.rs`** with the implementation and its tests. Everything it uses already exists: the describe.rs kernels (`pub(crate)` since Task 2), `polars_views` and `VIEW_BLOCK` / `VIEW_MAX_BLOCK` (Task 5), `Shape` (which derives `Default`), `body_size` and `ipc_body_bytes`.

```rust
//! Mergeable per-level statistics for the streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md §4). Each batch
//! is profiled per level (a column, or a list's inner values) into a `BatchStats` with
//! describe.rs's kernels; batches are absorbed in stream order into a `LevelStats`,
//! which finishes into the `Profile` recommend.rs's rules read.

use std::collections::HashMap;

use arrow_array::{ArrayRef, UInt64Array};
use arrow_schema::DataType as AT;
use foldhash::fast::FixedState;
use polars::prelude::*;

use crate::arrow_io::export_series;
use crate::cardinality_estimators::{estimate, Estimate, Method};
use crate::describe::{
    arg_extremes, byte_lengths, float_stats, frequency_map, lengths, n_midnight, strings,
    FloatStats, Frequencies, Profile, Range, StringStats,
};
use crate::recommend::{body_size, Shape, VIEW_BLOCK, VIEW_MAX_BLOCK};
use crate::shared::encode_series;
use crate::sizes::{classic_layout, ipc_body_bytes};

fn pad8(x: u64) -> u64 {
    x.next_multiple_of(8)
}

fn opt<T>(a: Option<T>, b: Option<T>, f: impl FnOnce(T, T) -> T) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b)),
        (a, b) => a.or(b),
    }
}

fn gcd128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i128
}

/// Compact copy of `len` rows from `start` (pins none of `a`'s buffers).
pub(crate) fn copy_rows(a: &ArrayRef, start: usize, len: usize) -> Result<ArrayRef, String> {
    let idx = UInt64Array::from_iter_values((start..start + len).map(|i| i as u64));
    arrow_select::take::take(a.as_ref(), &idx, None).map_err(|e| e.to_string())
}

/// Polars' view-array data blocks (recommend.rs `polars_views`) replayed on value
/// lengths alone: values of ≤ 12 bytes are inline; longer ones fill blocks whose
/// capacity doubles from 8 KiB to 16 MiB (or grows to fit one value). `bytes` is the
/// IPC body of those blocks, each padded to 8.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ViewSim {
    closed: u64,
    current: u64,
    capacity: u64,
}

impl ViewSim {
    pub(crate) fn push(&mut self, len: u64) {
        if len <= 12 {
            return;
        }
        if self.capacity < self.current + len {
            if self.current > 0 {
                self.closed += pad8(self.current);
                self.current = 0;
            }
            self.capacity = (self.capacity * 2)
                .clamp(VIEW_BLOCK as u64, VIEW_MAX_BLOCK as u64)
                .max(len);
        }
        self.current += len;
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.closed + pad8(self.current)
    }
}

/// Ordering key of an extreme: the physical integer, or the float.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) enum Key {
    I(i128),
    F(f64),
}

/// A running extreme: its key, and the value itself (one row, classic layout) to render.
#[derive(Clone, Debug)]
pub(crate) struct Ext {
    pub key: Key,
    pub value: ArrayRef,
}

/// Dtypes whose min / max the rules read (as integers or floats) and the output renders.
fn has_extremes(dt: &DataType) -> bool {
    dt.is_integer()
        || dt.is_float()
        || matches!(
            dt,
            DataType::Decimal(..)
                | DataType::Date
                | DataType::Datetime(..)
                | DataType::Duration(_)
                | DataType::Time
        )
}

fn ext_at(s: &Series, i: Option<u64>) -> PolarsResult<Option<Ext>> {
    let Some(i) = i else { return Ok(None) };
    let one = s.slice(i as i64, 1);
    let key = if one.dtype().is_float() {
        one.cast(&DataType::Float64)?.f64()?.get(0).map(Key::F)
    } else {
        one.to_physical_repr()
            .cast(&DataType::Int128)?
            .i128()?
            .get(0)
            .map(Key::I)
    };
    let value = classic_layout(&one)?;
    Ok(key.map(|key| Ext { key, value }))
}

/// A distinct value's statistics in one batch.
pub(crate) struct KeyStat {
    key: u64,
    first: u64,
    count: u64,
    mask: u8,
    len: u64,
    /// The value, when the batch holds at most five distinct values.
    text: Option<String>,
}

/// Distinct values of a text level, bounded by `categorical_threshold` (spec §4.2).
#[derive(Clone, Debug, Default)]
pub(crate) struct Distinct {
    /// key → bits 0–1: count capped at 3 (enough for f1 / f2); bits 2–4: capture mask.
    map: HashMap<u64, u8, FixedState>,
    pub sum_len_unique: u64,
    /// Every distinct value, first-occurrence order, while there are at most five.
    pub few: Vec<String>,
    /// The distinct values' view blocks, first-occurrence order (a dictionary's Polars values).
    pub views: ViewSim,
    pub overflowed: bool,
}

impl Distinct {
    /// `keys` in first-occurrence order.
    fn absorb(&mut self, keys: Vec<KeyStat>, threshold: u64) {
        if self.overflowed {
            return;
        }
        for k in keys {
            match self.map.get_mut(&k.key) {
                Some(v) => {
                    let count = ((*v & 3) as u64 + k.count).min(3) as u8;
                    *v = count | (*v & !3) | (k.mask << 2);
                }
                None => {
                    self.map.insert(k.key, k.count.min(3) as u8 | (k.mask << 2));
                    self.sum_len_unique += k.len;
                    self.views.push(k.len);
                    match k.text {
                        Some(t) if self.map.len() <= 5 => self.few.push(t),
                        _ => self.few.clear(),
                    }
                }
            }
            if self.map.len() as u64 > threshold {
                // Exact, not statistical: the estimate is floored at n_unique, so the
                // dictionary is rejected whatever follows. Stop hashing from here on.
                self.overflowed = true;
                self.map = HashMap::default();
                self.few.clear();
                return;
            }
        }
    }

    pub(crate) fn n_unique(&self) -> u64 {
        self.map.len() as u64
    }

    /// (f1, f2, capture history).
    pub(crate) fn counts(&self) -> (u64, u64, [u64; 7]) {
        let (mut f1, mut f2, mut h) = (0, 0, [0u64; 7]);
        for &v in self.map.values() {
            f1 += (v & 3 == 1) as u64;
            f2 += (v & 3 == 2) as u64;
            h[(v >> 2) as usize - 1] += 1;
        }
        (f1, f2, h)
    }
}

/// One batch's statistics of one level.
pub(crate) struct BatchStats {
    n: u64,
    n_null: u64,
    lo: Option<Ext>,
    hi: Option<Ext>,
    min_len: Option<u64>,
    max_len: Option<u64>,
    gcd: Option<i128>,
    sum_len: Option<u64>,
    floats: Option<FloatStats>,
    strings: Option<StringStats>,
    n_midnight: Option<u64>,
    /// Text levels with distinct tracking on: every distinct value, first-occurrence order.
    keys: Option<Vec<KeyStat>>,
    /// Text / binary levels: lengths of the values over 12 bytes, row order.
    long_lens: Option<Vec<u64>>,
    /// Measured classic size, when the classic type has no analytic size.
    size_bytes: u64,
    polars_bytes: u64,
    classic: AT,
    is_f32: bool,
}

impl BatchStats {
    /// `s`: the level's values in one batch; `offset`: the level's global index of its
    /// first value; distinct values are tracked when `track`.
    pub(crate) fn of(s: &Series, offset: u64, seed: u64, track: bool) -> PolarsResult<Self> {
        let lens = byte_lengths(s)?;
        let classic = export_series(&s.slice(0, 0), CompatLevel::oldest())?
            .data_type()
            .clone();
        let size_bytes = if body_size(&classic, &Shape::default()).is_ok() {
            0
        } else {
            ipc_body_bytes(classic_layout(s)?.as_ref(), None)?
        };
        let (lo, hi) = if has_extremes(s.dtype()) {
            let (a, b) = arg_extremes(s)?;
            (ext_at(s, a)?, ext_at(s, b)?)
        } else {
            (None, None)
        };
        let keys = if track {
            let map = frequency_map(&encode_series(s)?, seed, offset);
            let text = if map.len() <= 5 {
                Some(s.cast(&DataType::String)?)
            } else {
                None
            };
            let mut keys: Vec<KeyStat> = map
                .into_iter()
                .map(|(key, e)| {
                    let row = (e.first - offset) as usize;
                    KeyStat {
                        key,
                        first: e.first,
                        count: e.count,
                        mask: e.mask,
                        len: lens.as_ref().map_or(0, |l| l[row]),
                        text: text
                            .as_ref()
                            .and_then(|t| t.str().ok()?.get(row).map(str::to_owned)),
                    }
                })
                .collect();
            keys.sort_unstable_by_key(|k| k.first);
            Some(keys)
        } else {
            None
        };
        let (min_len, max_len) = lengths(s, lens.as_deref())?;
        Ok(BatchStats {
            n: s.len() as u64,
            n_null: s.null_count() as u64,
            lo,
            hi,
            min_len,
            max_len,
            gcd: crate::gcd::series_gcd(s)?,
            sum_len: lens.as_ref().map(|l| l.iter().sum()),
            floats: float_stats(s)?,
            strings: strings(s)?,
            n_midnight: n_midnight(s)?,
            keys,
            long_lens: lens.map(|l| l.into_iter().filter(|&x| x > 12).collect()),
            size_bytes,
            polars_bytes: ipc_body_bytes(export_series(s, CompatLevel::newest())?.as_ref(), None)?,
            classic,
            is_f32: s.dtype() == &DataType::Float32,
        })
    }
}

/// A level's statistics over the stream so far.
#[derive(Clone, Default)]
pub(crate) struct LevelStats {
    /// Values at this level, nulls included (the column: every row of the stream).
    pub n: u64,
    pub n_null: u64,
    pub lo: Option<Ext>,
    pub hi: Option<Ext>,
    pub min_len: Option<u64>,
    pub max_len: Option<u64>,
    pub gcd: Option<i128>,
    pub sum_len: Option<u64>,
    pub floats: Option<FloatStats>,
    pub strings: Option<StringStats>,
    pub n_midnight: Option<u64>,
    pub distinct: Option<Distinct>,
    /// All values' view blocks (Utf8View / BinaryView results), row order.
    pub views: Option<ViewSim>,
    /// Per-batch sums of the measured classic size (types with no analytic size) and
    /// of the Polars-layout size.
    pub size_bytes: u64,
    pub polars_bytes: u64,
    pub classic: Option<AT>,
    pub is_f32: bool,
}

impl LevelStats {
    /// `rows` null values: a column absent from a batch, or backfilled when it appears.
    pub(crate) fn nulls(&mut self, rows: u64) {
        self.n += rows;
        self.n_null += rows;
    }

    pub(crate) fn absorb(&mut self, b: BatchStats, threshold: u64) {
        self.n += b.n;
        self.n_null += b.n_null;
        if b.lo.as_ref().is_some_and(|x| self.lo.as_ref().is_none_or(|y| x.key < y.key)) {
            self.lo = b.lo;
        }
        if b.hi.as_ref().is_some_and(|x| self.hi.as_ref().is_none_or(|y| y.key < x.key)) {
            self.hi = b.hi;
        }
        self.min_len = opt(self.min_len, b.min_len, u64::min);
        self.max_len = opt(self.max_len, b.max_len, u64::max);
        self.gcd = opt(self.gcd, b.gcd, gcd128);
        self.sum_len = opt(self.sum_len, b.sum_len, |a, b| a + b);
        self.floats = opt(self.floats, b.floats, FloatStats::merge);
        self.strings = opt(self.strings.take(), b.strings, StringStats::merge);
        self.n_midnight = opt(self.n_midnight, b.n_midnight, |a, b| a + b);
        if let Some(keys) = b.keys {
            self.distinct
                .get_or_insert_with(Default::default)
                .absorb(keys, threshold);
        }
        if let Some(lens) = b.long_lens {
            let v = self.views.get_or_insert_with(Default::default);
            lens.into_iter().for_each(|l| v.push(l));
        }
        self.size_bytes += b.size_bytes;
        self.polars_bytes += b.polars_bytes;
        self.classic.get_or_insert(b.classic);
        self.is_f32 = b.is_f32;
    }

    pub(crate) fn overflowed(&self) -> bool {
        self.distinct.as_ref().is_some_and(|d| d.overflowed)
    }

    /// The Profile the rules read. An overflowed level reports `threshold + 1` distinct
    /// values, so the dictionary gate rejects it.
    pub(crate) fn profile(&self, threshold: u64) -> Profile {
        let d = self.distinct.as_ref();
        let (f1, f2, capture_history) = d.map_or((0, 0, [0; 7]), Distinct::counts);
        let n_unique = d.map_or(0, |d| {
            if d.overflowed {
                threshold + 1
            } else {
                d.n_unique()
            }
        });
        Profile {
            freq: Frequencies {
                n_unique,
                entropy: f64::NAN,
                f1,
                f2,
                top5_idx: Vec::new(),
                top5_count: Vec::new(),
                capture_history,
                sum_len_unique: d.map(|d| d.sum_len_unique),
            },
            range: Range {
                argmin: None,
                argmax: None,
                min_len: self.min_len,
                max_len: self.max_len,
            },
            floats: self.floats,
            strings: self.strings.clone(),
            gcd: self.gcd,
            sum_len: self.sum_len,
            is_f32: self.is_f32,
        }
    }

    /// Text levels only: Schnabel → Chao1 (no population), or `overflowed`.
    pub(crate) fn estimate(&self, threshold: u64) -> Option<Estimate> {
        let d = self.distinct.as_ref()?;
        Some(if d.overflowed {
            Estimate {
                est_cardinality: (threshold + 1) as f64,
                est_low: None,
                est_high: None,
                method: Method::Overflowed,
            }
        } else {
            let (f1, f2, h) = d.counts();
            estimate(d.n_unique(), self.n - self.n_null, f1, f2, &h, None)
        })
    }

    pub(crate) fn int_range(&self) -> Option<(i128, i128)> {
        match (self.lo.as_ref()?.key, self.hi.as_ref()?.key) {
            (Key::I(a), Key::I(b)) => Some((a, b)),
            _ => None,
        }
    }

    pub(crate) fn float_range(&self) -> Option<(f64, f64)> {
        match (self.lo.as_ref()?.key, self.hi.as_ref()?.key) {
            (Key::F(a), Key::F(b)) => Some((a, b)),
            _ => None,
        }
    }

    pub(crate) fn few_distinct(&self) -> Vec<String> {
        self.distinct
            .as_ref()
            .filter(|d| !d.overflowed && d.n_unique() <= 5)
            .map_or_else(Vec::new, |d| d.few.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::describe::frequencies;
    use crate::recommend::polars_views;

    fn absorbed(parts: &[Series], track: bool, threshold: u64) -> LevelStats {
        let mut st = LevelStats::default();
        for p in parts {
            let b = BatchStats::of(p, st.n, 7, track).unwrap();
            st.absorb(b, threshold);
        }
        st
    }

    fn chunks(s: &Series, k: usize) -> Vec<Series> {
        (0..s.len())
            .step_by(k)
            .map(|o| s.slice(o as i64, k))
            .collect()
    }

    fn summary(st: &LevelStats) -> String {
        let key = |e: &Option<Ext>| e.as_ref().map(|e| format!("{:?}", e.key));
        let d = st.distinct.as_ref().map(|d| {
            (
                d.n_unique(),
                d.counts(),
                d.sum_len_unique,
                d.few.clone(),
                d.views.bytes(),
                d.overflowed,
            )
        });
        let s = st.strings.as_ref().map(|s| {
            (
                s.n_numeric,
                s.n_iso_datetime,
                s.n_f64_roundtrip_fail,
                s.raw_frac_min,
                s.raw_frac_max,
                s.iso_instant_min,
                s.n_iso_time_noncanonical,
            )
        });
        let f = st
            .floats
            .map(|f| (f.n_nan, f.n_fractional, f.max_frac_digits, f.n_neg_zero));
        format!(
            "{} {} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            st.n,
            st.n_null,
            key(&st.lo),
            key(&st.hi),
            st.min_len,
            st.max_len,
            st.gcd,
            st.sum_len,
            d,
            s,
            f,
            st.views.as_ref().map(ViewSim::bytes),
            st.n_midnight,
        )
    }

    fn samples() -> Vec<(Series, bool)> {
        vec![
            (
                Series::new(
                    "i".into(),
                    &[Some(30i64), None, Some(-6), Some(12), Some(30), None, Some(0)],
                ),
                false,
            ),
            (
                Series::new(
                    "f".into(),
                    &[Some(1.5f64), Some(-0.0), None, Some(f64::NAN), Some(2.25), Some(-7.0)],
                ),
                false,
            ),
            (
                Series::new(
                    "s".into(),
                    &[
                        Some("1.50"),
                        Some("a value longer than twelve"),
                        None,
                        Some("2024-01-01T10:00:00"),
                        Some("1.50"),
                        Some("another value longer than twelve"),
                        Some("x"),
                    ],
                ),
                true,
            ),
        ]
    }

    #[test]
    fn splitting_the_stream_does_not_change_the_statistics() {
        for (s, track) in samples() {
            let whole = summary(&absorbed(&[s.clone()], track, 10_000));
            for k in [1, 2, 3, 5] {
                let parts = chunks(&s, k);
                assert_eq!(summary(&absorbed(&parts, track, 10_000)), whole, "{} k={k}", s.name());
            }
        }
    }

    #[test]
    fn distinct_counts_match_describe() {
        let s = Series::new("s".into(), &["a", "b", "a", "c", "c", "c", "d"]);
        let f = frequencies(&encode_series(&s).unwrap(), 7, None);
        let st = absorbed(&chunks(&s, 2), true, 10_000);
        let d = st.distinct.as_ref().unwrap();
        assert_eq!(d.n_unique(), f.n_unique);
        assert_eq!(d.counts(), (f.f1, f.f2, f.capture_history));
        assert_eq!(d.few, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn distinct_tracking_stops_past_the_threshold() {
        let s = Series::new("s".into(), &["a", "b", "c", "d", "e"]);
        let st = absorbed(&chunks(&s, 2), true, 3);
        assert!(st.overflowed());
        assert_eq!(st.profile(3).freq.n_unique, 4);
        assert_eq!(st.estimate(3).unwrap().method, Method::Overflowed);
        assert!(st.few_distinct().is_empty());
    }

    #[test]
    fn view_blocks_match_polars_views() {
        let values: Vec<String> = (0..1000).map(|i| "x".repeat(1 + (i * 37) % 9000)).collect();
        let arr = arrow_array::StringArray::from(values.iter().map(String::as_str).collect::<Vec<_>>());
        let views = polars_views(&arr, &AT::Utf8View).unwrap();
        let measured =
            ipc_body_bytes(views.as_ref(), None).unwrap() - pad8(16 * values.len() as u64);
        let mut sim = ViewSim::default();
        values.iter().for_each(|v| sim.push(v.len() as u64));
        assert_eq!(sim.bytes(), measured);
    }

    #[test]
    fn extremes_keep_values_to_render() {
        let s = Series::new("i".into(), &[5i64, -3, 9]);
        let st = absorbed(&chunks(&s, 1), false, 10_000);
        assert_eq!(st.int_range(), Some((-3, 9)));
        assert_eq!(st.lo.as_ref().unwrap().value.len(), 1);
    }
}
```

- [ ] **Step 2: Register the module.** In `lib.rs`, add `mod partial;` after `mod minhash;`.

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib partial::`
Expected: 5 tests PASS. If `splitting_the_stream…` fails for `f`, check the `-0.0` / `0.0` extremes, which are compared with `<`: ties must keep the earlier value in both `arg_extremes` and `absorb`.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/partial.rs services/analytics/src/lib.rs
git commit -m "partial: mergeable per-level statistics for streaming

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Block reservoir — `reservoir.rs`

**Files:**
- Create: `services/analytics/src/reservoir.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod reservoir;` after `mod recommend;`)

- [ ] **Step 1: Create `reservoir.rs`:**

```rust
//! Block reservoir (streaming spec §4.5): the stream is cut into contiguous blocks of
//! `block_rows` rows; a uniform random sample of `capacity` blocks is kept by Li's
//! Algorithm L (seeded). Blocks keep row order, so ZSTD sizes measured on them match
//! an IPC file written in `block_rows` batches. A kept block holds compact copies of
//! its rows, one piece per batch it spans, so it pins no batch buffers.

use arrow_array::ArrayRef;
use arrow_schema::FieldRef;

use crate::partial::copy_rows;

/// Algorithm L's state, `Copy` so that `feed` can plan on a copy and commit only
/// once every copy of the batch's rows has succeeded.
#[derive(Clone, Copy, Debug)]
struct Cursor {
    rng: u64,
    w: f64,
    /// Index of the next block to keep once the reservoir is full.
    next: u64,
    /// Blocks completed so far (= the index of the block in progress).
    seen: u64,
    kept: usize,
    /// Rows in the block in progress.
    rows: u64,
    /// Whether the block in progress will be kept.
    keep: bool,
}

impl Cursor {
    /// SplitMix64, the same finaliser as describe.rs `subset`.
    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Next index to keep: the block just closed (`seen - 1`) plus a geometric skip.
    fn skip(&mut self) {
        let jump = (self.unit().ln() / (1.0 - self.w).ln()).floor();
        self.next = self.seen - 1 + jump as u64 + 1;
    }

    /// Closes the block in progress; returns its slot (None: dropped).
    fn close(&mut self, capacity: usize) -> Option<usize> {
        let keep = self.keep;
        self.seen += 1;
        self.rows = 0;
        let slot = if !keep {
            None
        } else if self.kept < capacity {
            self.kept += 1;
            if self.kept == capacity {
                self.w = (self.unit().ln() / capacity as f64).exp();
                self.skip();
            }
            Some(self.kept - 1)
        } else {
            let j = ((self.unit() * capacity as f64) as usize).min(capacity - 1);
            self.w *= (self.unit().ln() / capacity as f64).exp();
            self.skip();
            Some(j)
        };
        self.keep = capacity > 0 && (self.kept < capacity || self.seen == self.next);
        slot
    }
}

/// One batch's share of a block: `rows` rows of the columns that batch carried.
#[derive(Clone)]
pub(crate) struct Piece {
    pub rows: u64,
    pub cols: Vec<(FieldRef, ArrayRef)>,
}

#[derive(Clone, Default)]
pub(crate) struct Block {
    pub rows: u64,
    pub pieces: Vec<Piece>,
}

pub(crate) struct Reservoir {
    capacity: usize,
    block_rows: u64,
    cur: Cursor,
    kept: Vec<Block>,
    current: Block,
}

impl Reservoir {
    /// `capacity` blocks of `block_rows` (≥ 1) rows.
    pub(crate) fn new(capacity: usize, block_rows: u64, seed: u64) -> Self {
        Reservoir {
            capacity,
            block_rows,
            cur: Cursor {
                rng: seed ^ 0xB10C_B10C_B10C_B10C,
                w: 1.0,
                next: 0,
                seen: 0,
                kept: 0,
                rows: 0,
                keep: capacity > 0,
            },
            kept: Vec::new(),
            current: Block::default(),
        }
    }

    /// Adds `rows` rows whose columns are `cols`. On error nothing changes.
    pub(crate) fn feed(&mut self, cols: &[(FieldRef, ArrayRef)], rows: u64) -> Result<(), String> {
        // Plan on a copy of the cursor: (start, len, kept?, close → slot).
        let mut cur = self.cur;
        let mut segments = Vec::new();
        let mut start = 0;
        while start < rows {
            let len = (self.block_rows - cur.rows).min(rows - start);
            let keep = cur.keep;
            cur.rows += len;
            let close = (cur.rows == self.block_rows).then(|| cur.close(self.capacity));
            segments.push((start, len, keep, close));
            start += len;
        }
        // Copy the kept rows (the only fallible step), then commit.
        let mut pieces = Vec::with_capacity(segments.len());
        for &(start, len, keep, _) in &segments {
            pieces.push(if keep {
                let cols = cols
                    .iter()
                    .map(|(f, a)| Ok((f.clone(), copy_rows(a, start as usize, len as usize)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                Some(Piece { rows: len, cols })
            } else {
                None
            });
        }
        for ((_, len, _, close), piece) in segments.into_iter().zip(pieces) {
            self.current.rows += len;
            self.current.pieces.extend(piece);
            if let Some(slot) = close {
                let block = std::mem::take(&mut self.current);
                match slot {
                    Some(j) if j < self.kept.len() => self.kept[j] = block,
                    Some(_) => self.kept.push(block),
                    None => {}
                }
            }
        }
        self.cur = cur;
        Ok(())
    }

    /// The sampled blocks: the kept ones, plus the block in progress while the
    /// reservoir is not yet full (so a stream shorter than the reservoir is measured
    /// completely).
    pub(crate) fn blocks(&self) -> Vec<&Block> {
        let mut out: Vec<&Block> = self.kept.iter().collect();
        if self.kept.len() < self.capacity && self.current.rows > 0 {
            out.push(&self.current);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field};

    use super::*;

    fn col(v: std::ops::Range<i64>) -> Vec<(FieldRef, ArrayRef)> {
        vec![(
            Arc::new(Field::new("a", DataType::Int64, true)),
            Arc::new(Int64Array::from_iter_values(v)) as ArrayRef,
        )]
    }

    fn values(b: &Block) -> Vec<i64> {
        b.pieces
            .iter()
            .flat_map(|p| p.cols[0].1.as_primitive::<Int64Type>().values().to_vec())
            .collect()
    }

    #[test]
    fn keeps_every_block_until_full() {
        let mut r = Reservoir::new(4, 3, 0);
        r.feed(&col(0..4), 4).unwrap();
        r.feed(&col(4..10), 6).unwrap();
        let got: Vec<Vec<i64>> = r.blocks().iter().map(|b| values(b)).collect();
        assert_eq!(got, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]);
        assert_eq!(r.blocks()[1].pieces.len(), 2); // spans both batches
    }

    #[test]
    fn capacity_zero_keeps_nothing() {
        let mut r = Reservoir::new(0, 2, 0);
        r.feed(&col(0..10), 10).unwrap();
        assert!(r.blocks().is_empty());
    }

    #[test]
    fn same_seed_same_sample() {
        let sample = |seed| {
            let mut r = Reservoir::new(3, 1, seed);
            for i in 0..100 {
                r.feed(&col(i..i + 1), 1).unwrap();
            }
            let mut v: Vec<i64> = r.blocks().iter().flat_map(|b| values(b)).collect();
            v.sort();
            v
        };
        assert_eq!(sample(1), sample(1));
        assert_ne!(sample(1), sample(2));
    }

    #[test]
    fn blocks_are_sampled_uniformly() {
        // 10 one-row blocks, 2 kept: each block is kept with probability 1/5.
        let mut hits = [0u32; 10];
        let trials = 20_000;
        for seed in 0..trials {
            let mut r = Reservoir::new(2, 1, seed);
            for i in 0..10 {
                r.feed(&col(i..i + 1), 1).unwrap();
            }
            for b in r.blocks() {
                hits[values(b)[0] as usize] += 1;
            }
        }
        let expected = trials as f64 * 2.0 / 10.0;
        for (i, &h) in hits.iter().enumerate() {
            assert!(
                (h as f64 - expected).abs() < 0.05 * expected,
                "block {i}: {h} vs {expected}"
            );
        }
    }
}
```

- [ ] **Step 2: Register the module.** In `lib.rs`, add `mod reservoir;` after `mod recommend;`.

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib reservoir::`
Expected: 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/reservoir.rs services/analytics/src/lib.rs
git commit -m "reservoir: seeded Algorithm L over contiguous row blocks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Streaming state and `add` — `streaming.rs`

**Files:**
- Create: `services/analytics/src/streaming.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod streaming;` after `mod sizes;`)

- [ ] **Step 1: Create `streaming.rs`** with the state, `add`, `mark_ineligible` and their tests. `finish` comes in Task 10.

```rust
//! Streaming recommender (spec
//! docs/superpowers/specs/2026-09-29-streaming-recommender-design.md): record batches
//! are added over time; per-column statistics (partial.rs) and a block reservoir
//! (reservoir.rs) are kept; `finish` recommends from them at any point (Task 10).

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use polars::prelude::{DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::api::{Error, Result};
use crate::arrow_io::import_batch;
use crate::describe::flatten;
use crate::partial::{BatchStats, LevelStats};
use crate::recommend::{is_text, pa_name, Params};
use crate::reservoir::Reservoir;

/// One column's state.
pub(crate) struct Column {
    pub name: String,
    /// None while only Null-typed batches have been seen.
    pub dtype: Option<PT>,
    /// pyarrow spelling of the first concrete input type ("null" until then); for an
    /// ineligible column, the dtype its caller reported.
    pub input_type: String,
    pub ineligible: bool,
    /// Global row where the column first appeared; earlier rows count as nulls.
    pub first_row: u64,
    pub outer: LevelStats,
    /// List / Array columns: the inner values.
    pub inner: Option<LevelStats>,
}

pub(crate) struct Streaming {
    params: Params,
    n_rows: u64,
    columns: Vec<Column>,
    index: HashMap<String, usize>,
    reservoir: Reservoir,
}

/// Whether a batch's dtype `b` continues a column of dtype `a`: Categoricals always do
/// (each batch may carry its own mapping); lists and arrays by their inner types;
/// anything else (Enums included, with their categories) only when equal.
fn same_type(a: &PT, b: &PT) -> bool {
    match (a, b) {
        (PT::Categorical(..), PT::Categorical(..)) => true,
        (PT::List(x), PT::List(y)) => same_type(x, y),
        (PT::Array(x, w), PT::Array(y, v)) => w == v && same_type(x, y),
        _ => a == b,
    }
}

/// A batch column's statistics: None for a Null-typed column; else the column's and,
/// for List / Array, its inner values'.
type Stats = Option<(BatchStats, Option<BatchStats>)>;

impl Streaming {
    pub(crate) fn new(params: Params, reservoir_rows: u64, block_rows: u64) -> Self {
        let block_rows = block_rows.max(1);
        Streaming {
            reservoir: Reservoir::new((reservoir_rows / block_rows) as usize, block_rows, params.seed),
            params,
            n_rows: 0,
            columns: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// The column called `name`, created (its earlier rows backfilled as nulls) if new.
    fn column(&mut self, name: &str) -> &mut Column {
        let i = match self.index.get(name) {
            Some(&i) => i,
            None => {
                let mut outer = LevelStats::default();
                outer.nulls(self.n_rows);
                self.columns.push(Column {
                    name: name.to_string(),
                    dtype: None,
                    input_type: "null".into(),
                    ineligible: false,
                    first_row: self.n_rows,
                    outer,
                    inner: None,
                });
                self.index.insert(name.to_string(), self.columns.len() - 1);
                self.columns.len() - 1
            }
        };
        &mut self.columns[i]
    }

    /// A column the caller cannot send (Int128 / UInt128, Object). From here on its rows
    /// count as absent; its `n_null` is not reported.
    pub(crate) fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        if let Some(&i) = self.index.get(name) {
            let c = &self.columns[i];
            if let (Some(d), false) = (&c.dtype, c.ineligible) {
                return Err(Error::InvalidInput(format!(
                    "column {name:?}: type changed from {d} to {dtype}"
                )));
            }
        }
        let c = self.column(name);
        c.ineligible = true;
        c.input_type = dtype.to_string();
        Ok(())
    }

    fn check(&self, s: &Series) -> Result<()> {
        let Some(&i) = self.index.get(s.name().as_str()) else {
            return Ok(());
        };
        let c = &self.columns[i];
        if c.ineligible {
            return Err(Error::InvalidInput(format!(
                "column {:?} is ineligible ({})",
                s.name(),
                c.input_type
            )));
        }
        match &c.dtype {
            Some(d) if s.dtype() != &PT::Null && !same_type(d, s.dtype()) => {
                Err(Error::InvalidInput(format!(
                    "column {:?}: type changed from {d} to {}",
                    s.name(),
                    s.dtype()
                )))
            }
            _ => Ok(()),
        }
    }

    fn batch_stats(&self, s: &Series) -> PolarsResult<Stats> {
        if s.dtype() == &PT::Null {
            return Ok(None);
        }
        let c = self.index.get(s.name().as_str()).map(|&i| &self.columns[i]);
        // Distinct values are hashed until a level overflows, never after.
        let track = |l: Option<&LevelStats>| l.is_none_or(|l| !l.overflowed());
        let seed = self.params.seed;
        let outer = BatchStats::of(
            s,
            self.n_rows,
            seed,
            is_text(s.dtype()) && track(c.map(|c| &c.outer)),
        )?;
        let inner = match flatten(s)? {
            Some(v) => {
                let prev = c.and_then(|c| c.inner.as_ref());
                Some(BatchStats::of(
                    &v,
                    prev.map_or(0, |l| l.n),
                    seed,
                    is_text(v.dtype()) && track(prev),
                )?)
            }
            None => None,
        };
        Ok(Some((outer, inner)))
    }

    /// Adds one batch; on error the state is unchanged.
    pub(crate) fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let mut seen = HashSet::new();
        if let Some(f) = schema
            .fields()
            .iter()
            .find(|f| !seen.insert(f.name().as_str()))
        {
            return Err(Error::InvalidInput(format!("duplicate column {:?}", f.name())));
        }
        let series = import_batch(batch).map_err(|e| Error::InvalidInput(e.to_string()))?;
        series.iter().try_for_each(|s| self.check(s))?;
        let stats: Vec<Stats> = series
            .par_iter()
            .map(|s| self.batch_stats(s))
            .collect::<PolarsResult<_>>()
            .map_err(|e| Error::Compute(e.to_string()))?;
        let rows = batch.num_rows() as u64;
        let cols: Vec<_> = schema
            .fields()
            .iter()
            .cloned()
            .zip(batch.columns().iter().cloned())
            .collect();
        self.reservoir.feed(&cols, rows).map_err(Error::Compute)?;
        // Commit: nothing below can fail.
        let threshold = self.params.categorical_threshold;
        let mut present = HashSet::new();
        for ((s, f), st) in series.iter().zip(schema.fields()).zip(stats) {
            present.insert(s.name().to_string());
            let c = self.column(s.name().as_str());
            match st {
                None => c.outer.nulls(rows),
                Some((outer, inner)) => {
                    if c.dtype.is_none() {
                        c.dtype = Some(s.dtype().clone());
                        c.input_type = pa_name(f.data_type());
                    }
                    c.outer.absorb(outer, threshold);
                    if let Some(inner) = inner {
                        c.inner
                            .get_or_insert_with(Default::default)
                            .absorb(inner, threshold);
                    }
                }
            }
        }
        for c in &mut self.columns {
            if !present.contains(&c.name) {
                c.outer.nulls(rows);
            }
        }
        self.n_rows += rows;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, NullArray, StringArray};

    use super::*;

    pub(crate) fn params() -> Params {
        Params {
            seed: 0,
            zstd_level: 1,
            population_rows: None,
            categorical_threshold: 10_000,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    pub(crate) fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(cols).unwrap()
    }

    pub(crate) fn ints(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    pub(crate) fn strs(v: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }

    pub(crate) fn streaming() -> Streaming {
        Streaming::new(params(), 1 << 20, 1 << 16)
    }

    fn col<'a>(s: &'a Streaming, name: &str) -> &'a Column {
        &s.columns[s.index[name]]
    }

    #[test]
    fn a_new_column_is_backfilled_with_nulls() {
        let mut s = streaming();
        s.add(&batch(vec![("a", ints(&[Some(1), Some(2)]))])).unwrap();
        s.add(&batch(vec![("a", ints(&[Some(3)])), ("b", strs(&[Some("x")]))]))
            .unwrap();
        let b = col(&s, "b");
        assert_eq!((b.first_row, b.outer.n, b.outer.n_null), (2, 3, 2));
    }

    #[test]
    fn an_absent_column_counts_as_null() {
        let mut s = streaming();
        s.add(&batch(vec![("a", ints(&[Some(1)])), ("b", ints(&[Some(1)]))]))
            .unwrap();
        s.add(&batch(vec![("a", ints(&[Some(2), Some(3)]))])).unwrap();
        let b = col(&s, "b");
        assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
    }

    #[test]
    fn a_null_typed_column_adopts_the_first_concrete_type() {
        let mut s = streaming();
        s.add(&batch(vec![("b", Arc::new(NullArray::new(2)) as ArrayRef)]))
            .unwrap();
        assert!(col(&s, "b").dtype.is_none());
        s.add(&batch(vec![("b", strs(&[Some("x")]))])).unwrap();
        let b = col(&s, "b");
        assert_eq!(b.dtype, Some(PT::String));
        assert_eq!((b.outer.n, b.outer.n_null), (3, 2));
    }

    #[test]
    fn a_type_change_is_rejected_and_changes_nothing() {
        let mut s = streaming();
        s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
        let err = s
            .add(&batch(vec![("a", strs(&[Some("x")])), ("c", ints(&[Some(1)]))]))
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref m) if m.contains("type changed")),
            "{err:?}"
        );
        assert_eq!((s.n_rows, s.columns.len(), col(&s, "a").outer.n), (1, 1, 1));
    }

    #[test]
    fn a_duplicate_column_is_rejected() {
        let mut s = streaming();
        let err = s
            .add(&batch(vec![("a", ints(&[Some(1)])), ("a", ints(&[Some(2)]))]))
            .unwrap_err();
        assert_eq!(err, Error::InvalidInput("duplicate column \"a\"".into()));
    }

    #[test]
    fn ineligible_columns_are_kept_apart() {
        let mut s = streaming();
        s.mark_ineligible("w", "Int128").unwrap();
        s.add(&batch(vec![("a", ints(&[Some(1)]))])).unwrap();
        assert!(col(&s, "w").ineligible);
        assert!(s.add(&batch(vec![("w", ints(&[Some(1)]))])).is_err());
        assert!(s.mark_ineligible("a", "Object").is_err());
    }
}
```

- [ ] **Step 2: Register the module.** In `lib.rs`, add `mod streaming;` after `mod sizes;`.

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib streaming::`
Expected: 6 tests PASS. `Streaming::finish` doesn't exist yet, and dead-code warnings are acceptable until Task 11 uses these items.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/streaming.rs services/analytics/src/lib.rs
git commit -m "streaming: per-column state, dynamic schema, atomic add

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `finish` — recommend from statistics, measure ZSTD on sampled blocks (streaming.rs)

**Files:**
- Modify: `services/analytics/src/streaming.rs`

- [ ] **Step 1: Write the failing tests.** Append them to `streaming.rs`'s `mod tests`, and extend that module's imports to:

```rust
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Float64Array, Int64Array, ListArray, NullArray, StringArray};
    use arrow_schema::DataType as AT;

    use crate::arrow_io::export_struct;
    use crate::recommend::{arrow_cast, describe_and_recommend_impl};

    use super::*;
```

Tests:

```rust
    fn one_shot(b: &RecordBatch) -> RecordBatch {
        let out = describe_and_recommend_impl(&import_batch(b).unwrap(), &params()).unwrap();
        export_struct(&out).unwrap()
    }

    fn texts(b: &RecordBatch, name: &str) -> Vec<Option<String>> {
        let a = arrow_cast(b.column_by_name(name).unwrap().as_ref(), &AT::Utf8).unwrap();
        a.as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    fn mixed() -> RecordBatch {
        let list = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2)]),
            None,
            Some(vec![Some(3)]),
            Some(vec![Some(4)]),
            Some(vec![Some(500)]),
        ]);
        batch(vec![
            ("i", ints(&[Some(1), None, Some(300), Some(-5), Some(7), Some(7)])),
            (
                "s_num",
                strs(&[Some("1.50"), Some("2.25"), None, Some("3"), Some("4.5"), Some("5")]),
            ),
            (
                "s_cat",
                strs(&[
                    Some("red"),
                    Some("green"),
                    Some("red"),
                    None,
                    Some("blue"),
                    Some("a long value over twelve bytes"),
                ]),
            ),
            (
                "s_dt",
                strs(&[
                    Some("2024-01-01T10:00:00"),
                    Some("2024-01-02T11:30:00"),
                    None,
                    Some("2024-01-03T00:00:00"),
                    Some("2024-01-04T12:00:00.250"),
                    Some("2024-01-05 13:00:00"),
                ]),
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(1.5),
                    Some(-0.0),
                    None,
                    Some(2.25),
                    Some(0.5),
                    Some(3.0),
                ])) as ArrayRef,
            ),
            ("l", Arc::new(list) as ArrayRef),
        ])
    }

    const REC: [&str; 8] = [
        "rec_arrow_type",
        "rec_nullable",
        "rec_lossy_formatting",
        "rec_arrow_size_bytes",
        "rec_polars_size_bytes",
        "rec_arrow_size_zstd_bytes",
        "rec_polars_size_zstd_bytes",
        "rec_polars_type",
    ];

    #[test]
    fn streamed_batches_recommend_like_one_shot() {
        let whole = mixed();
        let reference = one_shot(&whole);
        for k in [1, 2, 6] {
            let mut s = streaming();
            for off in (0..whole.num_rows()).step_by(k) {
                s.add(&whole.slice(off, k.min(whole.num_rows() - off)))
                    .unwrap();
            }
            let out = s.finish().unwrap();
            let columns = texts(&out, "column");
            for name in REC {
                let (got, want) = (texts(&out, name), texts(&reference, name));
                for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                    // One-shot leaves rec_polars_type null when the original is kept.
                    if name == "rec_polars_type" && w.is_none() {
                        continue;
                    }
                    assert_eq!(g, w, "k={k} {:?} {name}", columns[row]);
                }
            }
            assert_eq!(texts(&out, "n_sampled_rows")[0].as_deref(), Some("6"));
        }
    }

    #[test]
    fn no_sample_means_no_zstd_sizes() {
        let mut s = Streaming::new(params(), 0, 1 << 16);
        s.add(&mixed()).unwrap();
        let out = s.finish().unwrap();
        assert!(texts(&out, "rec_arrow_size_zstd_bytes")
            .iter()
            .all(Option::is_none));
        assert_eq!(texts(&out, "n_sampled_blocks")[0].as_deref(), Some("0"));
        assert!(texts(&out, "rec_arrow_type").iter().all(Option::is_some));
    }

    #[test]
    fn overflow_rejects_the_dictionary() {
        let mut p = params();
        p.categorical_threshold = 3;
        let mut s = Streaming::new(p, 1 << 20, 1 << 16);
        let v = [Some("a"), Some("b"), Some("c"), Some("d"), Some("e"), Some("a")];
        s.add(&batch(vec![("s", strs(&v))])).unwrap();
        let out = s.finish().unwrap();
        assert_eq!(texts(&out, "n_unique"), vec![None]);
        assert_eq!(texts(&out, "distinct_overflowed")[0].as_deref(), Some("true"));
        assert_eq!(texts(&out, "est_method")[0].as_deref(), Some("overflowed"));
        let list = out
            .column_by_name("rec_candidates")
            .unwrap()
            .as_list::<i64>()
            .value(0);
        let cands = list.as_struct();
        let text = |f: &str| arrow_cast(cands.column_by_name(f).unwrap().as_ref(), &AT::Utf8).unwrap();
        let (rules, outcomes) = (text("rule"), text("outcome"));
        let (rules, outcomes) = (rules.as_string::<i32>(), outcomes.as_string::<i32>());
        let i = (0..rules.len())
            .find(|&i| rules.value(i) == "string→dictionary")
            .unwrap();
        assert_eq!(outcomes.value(i), "rejected");
    }

    #[test]
    fn ineligible_null_typed_and_empty() {
        assert_eq!(streaming().finish().unwrap().num_rows(), 0);
        let mut s = streaming();
        s.mark_ineligible("w", "Int128").unwrap();
        s.add(&batch(vec![
            ("a", ints(&[Some(1), Some(2)])),
            ("n", Arc::new(NullArray::new(2)) as ArrayRef),
        ]))
        .unwrap();
        let out = s.finish().unwrap();
        let some = |v: &[&str]| v.iter().map(|x| Some(x.to_string())).collect::<Vec<_>>();
        assert_eq!(texts(&out, "column"), some(&["w", "a", "n"]));
        assert_eq!(texts(&out, "status"), some(&["ineligible", "computed", "ineligible"]));
        assert_eq!(texts(&out, "dtype")[0].as_deref(), Some("Int128"));
        assert_eq!(texts(&out, "n_null")[0], None);
        assert_eq!(texts(&out, "n_rows")[0].as_deref(), Some("2"));
    }
```

If the list case returns a `List` array rather than a `LargeList` (the export layout decides), use `as_list::<i32>()`. Check it with `out.column_by_name("rec_candidates").unwrap().data_type()`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib streaming::`
Expected: `no method named finish`.

- [ ] **Step 3: Implement `finish`.** Extend the `use` block at the top of `streaming.rs` to:

```rust
use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use arrow_array::{new_empty_array, ArrayRef, RecordBatch};
use arrow_schema::DataType as AT;
use polars::prelude::{AnyValue, DataType as PT, PolarsResult, Series};
use rayon::prelude::*;

use crate::api::{Error, Result};
use crate::arrow_io::{export_struct, import_array, import_batch};
use crate::cardinality_estimators::{estimate, Estimate};
use crate::describe::{assemble, flatten, Profile, Row};
use crate::partial::{BatchStats, Ext, LevelStats, ViewSim};
use crate::recommend::{
    arrow_cast, body_size, cast_to, is_text, list_parts, pa_name, pad, pick_by_stats,
    pick_list_by_stats, pl_name, polars_layout, rec_fields, rec_row, to_polars_layout, validity,
    verify, wrap, Level, Params, Pick, Rec, Shape, Target,
};
use crate::reservoir::{Block, Reservoir};
use crate::sizes::{classic_layout, ipc_body_bytes, sizes_of};
```

Add these free functions, after `same_type`:

```rust
/// Output columns (spec §6), in order.
fn output_fields() -> Vec<(String, PT)> {
    let mut f: Vec<(String, PT)> = [
        ("column", PT::String),
        ("status", PT::String),
        ("dtype", PT::String),
        ("first_row", PT::UInt64),
        ("n_rows", PT::UInt64),
        ("n_null", PT::UInt64),
        ("min", PT::String),
        ("max", PT::String),
        ("gcd", PT::Decimal(Some(38), Some(0))),
        ("sum_len", PT::UInt64),
        ("min_len", PT::UInt64),
        ("max_len", PT::UInt64),
        ("n_unique", PT::UInt64),
        ("distinct_overflowed", PT::Boolean),
        ("est_cardinality", PT::Float64),
        ("est_low", PT::Float64),
        ("est_high", PT::Float64),
        ("est_method", PT::String),
        ("size_bytes", PT::UInt64),
        ("size_zstd_bytes", PT::UInt64),
        ("size_polars_bytes", PT::UInt64),
        ("size_polars_zstd_bytes", PT::UInt64),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    f.extend(rec_fields());
    f.push(("n_sampled_rows".into(), PT::UInt64));
    f.push(("n_sampled_blocks".into(), PT::UInt64));
    f
}

/// A Level built from statistics: the rules read its counts, extremes and few distinct
/// values; `values` is empty (only its type is read). Its size is set by the caller.
fn level<'a>(
    dtype: &'a PT,
    classic: &AT,
    p: &'a Profile,
    st: &LevelStats,
    est: Estimate,
    prefix: &'static str,
) -> Level<'a> {
    Level {
        dtype,
        values: new_empty_array(classic),
        p,
        n_rows: st.n,
        n_null: st.n_null,
        int_range: st.int_range(),
        float_range: st.float_range(),
        few_distinct: st.few_distinct(),
        n_midnight: st.n_midnight,
        size_bytes: 0,
        size_note: "",
        est,
        r: 1.0,
        prefix,
        text: Default::default(),
    }
}

/// The original type's uncompressed size: analytic where `body_size` covers the
/// classic type (predicted = measured), else the per-batch measured sum.
fn original_size(classic: &AT, shape: &Shape, st: &LevelStats) -> (u64, &'static str) {
    match body_size(classic, shape) {
        Ok(b) => (b as u64, "analytic"),
        Err(_) => (st.size_bytes, "per-batch sum of"),
    }
}

/// A level's analytic Polars-layout inputs: its shape and Polars' view blocks over all
/// values and over the distinct ones.
#[derive(Clone, Copy)]
struct ViewShape {
    shape: Shape,
    all: u64,
    distinct: u64,
}

fn view_shape(lvl: &Level, st: &LevelStats) -> ViewShape {
    ViewShape {
        shape: lvl.shape(),
        all: st.views.as_ref().map_or(0, ViewSim::bytes),
        distinct: st.distinct.as_ref().map_or(0, |d| d.views.bytes()),
    }
}

/// Uncompressed IPC body of `t` in Polars' layout (Spec B §5.5), from statistics:
/// what `to_polars_layout` + `ipc_body_bytes` measure in one-shot.
fn polars_body(
    t: &Target,
    o: &ViewShape,
    inner: Option<&ViewShape>,
    key: &AT,
) -> std::result::Result<f64, String> {
    let s = &o.shape;
    let v = validity(s.n, s.nulls);
    let inner = || inner.copied().ok_or_else(|| "no inner level".to_string());
    Ok(match t {
        Target::Plain(_) => v + pad(16.0 * s.n) + o.all as f64,
        Target::Dictionary(..) => {
            let w = key.primitive_width().unwrap_or(4) as f64;
            v + pad(s.n * w) + pad(16.0 * s.d) + o.distinct as f64
        }
        Target::Scalar(it) => {
            let i = inner()?;
            let shape = Shape {
                n: s.n,
                nulls: s.nulls + i.shape.nulls,
                ..i.shape
            };
            polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        Target::List(it) => v + pad(8.0 * (s.n + 1.0)) + polars_body(it, &inner()?, None, key)?,
        Target::FixedList(it, w) => {
            let (i, w) = (inner()?, *w as f64);
            let shape = Shape {
                n: s.n * w,
                nulls: i.shape.nulls + s.nulls * w,
                ..i.shape
            };
            v + polars_body(it, &ViewShape { shape, ..i }, None, key)?
        }
        Target::Original(_) => return Err("the original's Polars size is measured".into()),
        t => body_size(&polars_layout(&t.arrow_type(), key), s)?,
    })
}

/// The recommended array has nulls (as one-shot's `logical_null_count() > 0`).
fn nullable(t: &Target, o: &LevelStats, inner: Option<&LevelStats>) -> bool {
    match t {
        Target::Null => o.n > 0,
        Target::Scalar(_) => o.n_null + inner.map_or(0, |i| i.n_null) > 0,
        _ => o.n_null > 0,
    }
}

fn render(e: &Option<Ext>) -> AnyValue<'static> {
    e.as_ref()
        .and_then(|e| arrow_cast(e.value.as_ref(), &AT::Utf8).ok())
        .and_then(|a| {
            let s = a.as_string::<i32>();
            s.is_valid(0)
                .then(|| AnyValue::StringOwned(s.value(0).into()))
        })
        .unwrap_or(AnyValue::Null)
}

/// Column `name`'s rows in block `b` as one Series of `dtype` (absent pieces: nulls).
fn block_series(b: &Block, name: &str, dtype: &PT) -> PolarsResult<Series> {
    let mut out = Series::new_empty(name.into(), dtype);
    for piece in &b.pieces {
        let s = match piece.cols.iter().find(|(f, _)| f.name() == name) {
            Some((f, a)) => import_array(f, a)?.cast(dtype)?,
            None => Series::full_null(name.into(), piece.rows as usize, dtype),
        };
        out.append(&s)?;
    }
    Ok(out.rechunk())
}

/// The block cast to the pick and verified against itself: the cross-check (spec §5.1
/// step 6). A failure means a statistic was wrong — never a silent fallback.
fn recast(
    pick: &Pick,
    dtype: &PT,
    classic: &ArrayRef,
    p: &Profile,
    inner: Option<(&PT, &Profile)>,
) -> std::result::Result<ArrayRef, String> {
    if let (Some(ip), Some((idt, iprof)), false) = (
        &pick.inner,
        inner,
        matches!(pick.target, Target::Original(_)),
    ) {
        let (rows, child, _) = list_parts(classic)?;
        let il = Level::of_block(idt, child, iprof);
        let ia = cast_to(&ip.target, &il)?;
        verify(&ip.target, &il, &ia)?;
        return wrap(&pick.target, classic, &rows, &ia);
    }
    let lvl = Level::of_block(dtype, classic.clone(), p);
    let a = cast_to(&pick.target, &lvl)?;
    verify(&pick.target, &lvl, &a)?;
    Ok(a)
}
```

Add these methods to `impl Streaming`:

```rust
    /// One row per column, first-seen order (spec §6). The state is kept.
    pub(crate) fn finish(&self) -> Result<RecordBatch> {
        let blocks = self.reservoir.blocks();
        let sampled: u64 = blocks.iter().map(|b| b.rows).sum();
        let rows = self
            .columns
            .par_iter()
            .map(|c| self.row(c, &blocks, sampled))
            .collect::<std::result::Result<Vec<Row>, String>>()
            .map_err(Error::Compute)?;
        assemble("streaming_recommend", &output_fields(), &rows)
            .and_then(|s| export_struct(&s))
            .map_err(|e| Error::Compute(e.to_string()))
    }

    fn row(&self, c: &Column, blocks: &[&Block], sampled: u64) -> std::result::Result<Row, String> {
        let text = |s: &str| AnyValue::StringOwned(s.into());
        let u = |v: Option<u64>| v.map_or(AnyValue::Null, AnyValue::UInt64);
        let f = |v: Option<f64>| v.map_or(AnyValue::Null, AnyValue::Float64);
        let dtype = c.dtype.as_ref().filter(|_| !c.ineligible);
        let mut row: Row = vec![
            text(&c.name),
            text(if dtype.is_some() { "computed" } else { "ineligible" }),
            text(&c.input_type),
            AnyValue::UInt64(c.first_row),
            AnyValue::UInt64(self.n_rows),
            if c.ineligible {
                AnyValue::Null
            } else {
                AnyValue::UInt64(c.outer.n_null)
            },
        ];
        let Some(dtype) = dtype else {
            row.resize(output_fields().len(), AnyValue::Null);
            return Ok(row);
        };
        let t = self.params.categorical_threshold;
        let fallback = || estimate(0, 0, 0, 0, &[0; 7], None);
        let err = |e: polars::prelude::PolarsError| e.to_string();

        // Levels from statistics.
        let o = &c.outer;
        let p = o.profile(t);
        let est = o.estimate(t);
        let classic = o.classic.clone().ok_or("a typed column has absorbed no batch")?;
        let mut lvl = level(dtype, &classic, &p, o, est.unwrap_or_else(fallback), "");
        (lvl.size_bytes, lvl.size_note) = original_size(&classic, &lvl.shape(), o);
        let inner = match (&c.inner, dtype) {
            (Some(i), PT::List(it) | PT::Array(it, _)) => Some((i, &**it)),
            _ => None,
        };
        let ip = inner.map(|(i, _)| i.profile(t));
        let ilvl = match (inner, &ip) {
            (Some((i, it)), Some(ip)) => {
                let iclassic = i.classic.clone().ok_or("an inner level has absorbed no batch")?;
                let mut l = level(it, &iclassic, ip, i, i.estimate(t).unwrap_or_else(fallback), "inner: ");
                (l.size_bytes, l.size_note) = original_size(&iclassic, &l.shape(), i);
                Some(l)
            }
            _ => None,
        };

        // The choice, proven by statistics.
        let pick = match &ilvl {
            Some(il) => {
                let width = match dtype {
                    PT::Array(_, w) => Some(*w as i32),
                    _ => None,
                };
                pick_list_by_stats(&lvl, il, width, &self.params)?
            }
            None => pick_by_stats(&lvl, &self.params)?,
        };

        // Uncompressed sizes: analytic (the original: as measured per batch).
        let key = pick.target.polars_key().unwrap_or(AT::UInt32);
        let original = matches!(pick.target, Target::Original(_));
        let (rec_size, rec_polars) = if original {
            (lvl.size_bytes, o.polars_bytes)
        } else {
            let iv = match (&ilvl, inner) {
                (Some(il), Some((i, _))) => Some(view_shape(il, i)),
                _ => None,
            };
            let polars = polars_body(&pick.target, &view_shape(&lvl, o), iv.as_ref(), &key)?;
            (pick.predicted, polars as u64)
        };

        // ZSTD sizes and the cross-check on the sampled blocks.
        let z_level = Some(self.params.zstd_level);
        let mut z = [0u64; 4]; // original Arrow, original Polars, recommended Arrow, recommended Polars
        for b in blocks {
            let s = block_series(b, &c.name, dtype).map_err(err)?;
            let bc = classic_layout(&s).map_err(err)?;
            let sz = sizes_of(&s, &bc, self.params.zstd_level).map_err(err)?;
            z[0] += sz[1];
            z[1] += sz[3];
            if original {
                z[2] += sz[1];
                z[3] += sz[3];
                continue;
            }
            let a = recast(&pick, dtype, &bc, &p, inner.map(|(_, it)| it).zip(ip.as_ref()))
                .map_err(|e| {
                    format!(
                        "cross-check {:?} → {}: {e}",
                        c.name,
                        pa_name(&pick.target.arrow_type())
                    )
                })?;
            z[2] += ipc_body_bytes(a.as_ref(), z_level).map_err(err)?;
            z[3] += ipc_body_bytes(to_polars_layout(&a, &key)?.as_ref(), z_level).map_err(err)?;
        }
        let scale = |x: u64| {
            (sampled > 0).then(|| (x as f64 * self.n_rows as f64 / sampled as f64).round() as u64)
        };

        let rec_t = pick.target.arrow_type();
        let rec = Rec {
            nullable: nullable(&pick.target, o, inner.map(|(i, _)| i)),
            arrow_type: pa_name(&rec_t),
            arrow_size: rec_size,
            arrow_zstd: scale(z[2]),
            polars_type: Some(pl_name(&rec_t, &c.name, None, &key)),
            polars_size: rec_polars,
            polars_zstd: scale(z[3]),
            lossy: pick.lossy,
            candidates: pick.candidates,
        };
        let d = o.distinct.as_ref();
        row.extend([
            render(&o.lo),
            render(&o.hi),
            p.gcd.map_or(AnyValue::Null, |g| AnyValue::Decimal(g, 0)),
            u(p.sum_len),
            u(o.min_len),
            u(o.max_len),
            u(d.filter(|d| !d.overflowed).map(|d| d.n_unique())),
            d.map_or(AnyValue::Null, |d| AnyValue::Boolean(d.overflowed)),
            f(est.map(|e| e.est_cardinality)),
            f(est.and_then(|e| e.est_low)),
            f(est.and_then(|e| e.est_high)),
            est.map_or(AnyValue::Null, |e| text(e.method.name())),
            AnyValue::UInt64(lvl.size_bytes),
            u(scale(z[0])),
            AnyValue::UInt64(o.polars_bytes),
            u(scale(z[1])),
        ]);
        row.extend(rec_row(&rec));
        row.extend([AnyValue::UInt64(sampled), AnyValue::UInt64(blocks.len() as u64)]);
        Ok(row)
    }
```

Visibility this needs from recommend.rs: `pad`, `validity`, `polars_layout`, `pl_name`, `pa_name`, `arrow_cast`, `is_text`, `cast_to`, `verify`, `body_size` and `Shape` are already `pub(crate)`. `rec_fields`, `rec_row`, `list_parts`, `wrap` and `to_polars_layout` became so in Task 5, and `Level::of_block`, `Pick` and the pick functions came in Tasks 4 and 6. `Rec` and its fields must be `pub(crate)`, which they already are. `describe_and_recommend_impl` is `pub(crate)`.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib streaming::`
Expected: 10 tests PASS.

If `streamed_batches_recommend_like_one_shot` fails, the message names the column and field:
- A size difference points to `polars_body` or `original_size`. Compare the formula against `sizes.rs` for that type.
- A type or outcome difference points to a statistic. Compare `LevelStats::profile` with `describe_one` on the same data.

- [ ] **Step 5: Run the whole Rust suite**

Run: `cargo test --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/analytics/src/streaming.rs
git commit -m "streaming: finish — proven picks, analytic sizes, ZSTD on sampled blocks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The language-neutral API — `api.rs`

**Files:**
- Modify: `services/analytics/src/api.rs`

- [ ] **Step 1: Write the failing tests** in `api.rs`'s `mod tests`:

```rust
    fn streaming_params() -> StreamingParams {
        StreamingParams {
            reservoir_rows: 1 << 20,
            block_rows: 1 << 16,
            categorical_threshold: 10_000,
            zstd_level: 1,
            seed: 0,
            boolean_pairs: vec![("true".into(), "false".into())],
        }
    }

    #[test]
    fn streaming_parameters_are_validated() {
        let bad = [
            StreamingParams { block_rows: 0, ..streaming_params() },
            StreamingParams { reservoir_rows: 10, block_rows: 100, ..streaming_params() },
            StreamingParams { zstd_level: 99, ..streaming_params() },
            StreamingParams { boolean_pairs: vec![("Y".into(), "y".into())], ..streaming_params() },
            StreamingParams { boolean_pairs: vec![("".into(), "n".into())], ..streaming_params() },
        ];
        for p in bad {
            assert!(
                matches!(StreamingRecommender::new(p.clone()), Err(Error::InvalidInput(_))),
                "{p:?}"
            );
        }
        assert!(StreamingRecommender::new(StreamingParams { reservoir_rows: 0, ..streaming_params() }).is_ok());
    }

    #[test]
    fn streaming_round_trip() {
        let batch = RecordBatch::try_from_iter(vec![
            ("a", ints(&[0, 5, 7])),
            ("s", Arc::new(StringArray::from(vec!["x", "y", "x"])) as ArrayRef),
        ])
        .unwrap();
        let mut rec = StreamingRecommender::new(streaming_params()).unwrap();
        rec.add(&batch).unwrap();
        rec.add(&batch).unwrap();
        let out = rec.finish().unwrap();
        assert_eq!(out.num_rows(), 2);
        let types = out.column_by_name("rec_arrow_type").unwrap().as_string_view();
        assert_eq!(types.value(0), "uint8");
        let n_rows = out.column_by_name("n_rows").unwrap();
        assert_eq!(n_rows.as_primitive::<arrow_array::types::UInt64Type>().value(0), 6);
    }
```

The test module already imports `Arc`, `ArrayRef`, `StringArray`, `AsArray` and an `ints` helper. If `RecordBatch` isn't in scope there, add `use arrow_array::RecordBatch;`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib api::tests::streaming`
Expected: `cannot find struct StreamingParams`.

- [ ] **Step 3: Implement.** Append to `api.rs`, before `#[cfg(test)]`:

```rust
/// Keywords of the streaming recommender (spec 2026-09-29 §7.1).
#[derive(Clone, Debug)]
pub struct StreamingParams {
    /// Rows of contiguous blocks kept for ZSTD sizes and the cross-check (0: none).
    pub reservoir_rows: u64,
    pub block_rows: u64,
    pub categorical_threshold: u64,
    pub zstd_level: i32,
    pub seed: u64,
    pub boolean_pairs: Vec<(String, String)>,
}

/// Recommends dtypes from record batches added over time; all state stays in Rust.
pub struct StreamingRecommender(crate::streaming::Streaming);

impl StreamingRecommender {
    pub fn new(p: StreamingParams) -> Result<Self> {
        if p.block_rows == 0 {
            return Err(Error::InvalidInput("block_rows must be at least 1".into()));
        }
        if p.reservoir_rows != 0 && p.reservoir_rows < p.block_rows {
            return Err(Error::InvalidInput(format!(
                "reservoir_rows {} is below block_rows {}: use 0 (no sample) or at least one block",
                p.reservoir_rows, p.block_rows
            )));
        }
        let levels = zstd::compression_level_range();
        if !levels.contains(&p.zstd_level) {
            return Err(Error::InvalidInput(format!(
                "zstd_level {} is outside {levels:?}",
                p.zstd_level
            )));
        }
        if let Some((t, f)) = p
            .boolean_pairs
            .iter()
            .find(|(t, f)| t.is_empty() || f.is_empty() || t.to_lowercase() == f.to_lowercase())
        {
            return Err(Error::InvalidInput(format!(
                "boolean_pairs must be pairs of distinct non-empty strings, got ({t:?}, {f:?})"
            )));
        }
        let params = Params {
            seed: p.seed,
            zstd_level: p.zstd_level,
            population_rows: None,
            categorical_threshold: p.categorical_threshold,
            boolean_pairs: p.boolean_pairs,
        };
        Ok(Self(crate::streaming::Streaming::new(
            params,
            p.reservoir_rows,
            p.block_rows,
        )))
    }

    /// Adds one batch. On error the state is unchanged.
    pub fn add(&mut self, batch: &RecordBatch) -> Result<()> {
        self.0.add(batch)
    }

    /// Marks a column the caller cannot send (Int128 / UInt128, Object) as ineligible.
    pub fn mark_ineligible(&mut self, name: &str, dtype: &str) -> Result<()> {
        self.0.mark_ineligible(name, dtype)
    }

    /// The recommendation for every column seen so far; the state is kept.
    pub fn finish(&self) -> Result<RecordBatch> {
        self.0.finish()
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib api::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/api.rs
git commit -m "api: StreamingRecommender (new / add / mark_ineligible / finish)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Python binding and wrapper

**Files:**
- Modify: `services/analytics/src/python.rs`, `services/analytics/analytics/_plugin.py`, `services/analytics/analytics/recommend/__init__.py`
- Create: `services/analytics/analytics/recommend/streaming.py`

- [ ] **Step 1: Factor the capsule checks out of `read_batch`.** In `python.rs`, add `use std::sync::Mutex;` and extend the ffi import to `use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};`. Then replace `fn read_batch` with:

```rust
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
    let capsule = capsule.downcast::<PyCapsule>()?;
    if capsule.name()? != Some(c"arrow_array_stream") {
        return Err(value_error(
            "__arrow_c_stream__ did not return an arrow_array_stream capsule",
        ));
    }
    let stream = capsule.pointer() as *mut FFI_ArrowArrayStream;
    // SAFETY: an "arrow_array_stream" capsule holds a valid, unreleased ArrowArrayStream.
    unsafe { reject_wide_integers(stream.cast())? };
    f(stream)
}

/// A whole Arrow stream as one RecordBatch (kernels expect one chunk per column).
fn read_batch(data: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    // SAFETY: the stream is valid and unreleased (with_stream); read_stream moves it out.
    with_stream(data, |s| unsafe { read_stream(s) }.map_err(value_error))
}

/// An Arrow stream read batch by batch (the streaming recommender never concatenates).
fn read_reader(data: &Bound<'_, PyAny>) -> PyResult<ArrowArrayStreamReader> {
    // SAFETY: as for read_batch; from_raw moves the stream out.
    with_stream(data, |s| {
        unsafe { ArrowArrayStreamReader::from_raw(s) }.map_err(value_error)
    })
}
```

- [ ] **Step 2: Add the pyclass.** Add it after `fn run`:

```rust
/// The streaming recommender (api::StreamingRecommender): add batches over time,
/// `finish` at any point. A mutex serialises callers; the work runs without the GIL.
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
    /// caller dropped (Int128 / UInt128, Object); they are marked first.
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

    fn finish(&self, py: Python<'_>) -> PyResult<ArrowTable> {
        let rec = &self.0;
        run(py, move || locked(rec)?.finish()).map(ArrowTable)
    }
}
```

In `#[pymodule] fn analytics`, add `m.add_class::<StreamingRecommender>()?;` after `m.add_class::<ArrowTable>()?;`.

- [ ] **Step 3: Add the `_plugin.py` wrapper** at the end of `analytics/_plugin.py`:

```python
def streaming_recommender(
    *,
    reservoir_rows: int,
    block_rows: int,
    categorical_threshold: int,
    zstd_level: int,
    seed: int,
    boolean_pairs: tuple[tuple[str, str], ...],
):
    """The Rust streaming recommender (src/streaming.rs): .add(data, ineligible), .finish()."""
    return _rs.StreamingRecommender(
        reservoir_rows=reservoir_rows,
        block_rows=block_rows,
        categorical_threshold=categorical_threshold,
        zstd_level=zstd_level,
        seed=seed,
        boolean_pairs=[tuple(p) for p in boolean_pairs],
    )
```

- [ ] **Step 4: Create `analytics/recommend/streaming.py`:**

```python
"""StreamingRecommender: Recommend's dtype recommendations from batches added over time.

Spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md. All state lives
in Rust (src/streaming.rs): exact running statistics prove each recommendation on every
row; a sample of contiguous row blocks (`reservoir_rows`, in blocks of `block_rows`)
gives ZSTD sizes — those of an IPC file written in `block_rows` batches — and a
cross-check. Not a technique on the uniform contract: add batches, finish at any time.
"""

from __future__ import annotations

import polars as pl

from analytics import _plugin
from analytics._dtypes import holds_wide_integer


class StreamingRecommender:
    """Narrowest value-preserving Arrow type per column, from a stream of batches.

    `add(frame)` takes a Polars DataFrame or any Arrow tabular object
    (`__arrow_c_stream__`); its batches are processed one at a time. Columns may
    appear, disappear (their rows count as null) or start as the Null type. Int128 /
    UInt128 and Object columns are listed as ineligible. `finish()` returns one row
    per column and keeps the state, so adding can continue.
    """

    def __init__(
        self,
        *,
        reservoir_rows: int = 524_288,
        block_rows: int = 65_536,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
    ) -> None:
        self._rs = _plugin.streaming_recommender(
            reservoir_rows=reservoir_rows,
            block_rows=block_rows,
            categorical_threshold=categorical_threshold,
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=tuple(tuple(p) for p in boolean_pairs),
        )

    def add(self, frame) -> StreamingRecommender:
        if isinstance(frame, pl.LazyFrame):
            raise TypeError(
                "a LazyFrame is not supported: collect it, or add its batches"
            )
        ineligible: list[tuple[str, str]] = []
        if isinstance(frame, pl.DataFrame):
            bad = [
                name
                for name, dtype in frame.schema.items()
                if holds_wide_integer(dtype) or isinstance(dtype, pl.Object)
            ]
            ineligible = [(name, str(frame.schema[name])) for name in bad]
            frame = frame.drop(bad)
        self._rs.add(frame, ineligible)
        return self

    def finish(self) -> pl.DataFrame:
        return pl.DataFrame(self._rs.finish())
```

- [ ] **Step 5: Export it.** In `analytics/recommend/__init__.py`:
- add `from analytics.recommend.streaming import StreamingRecommender`;
- extend `__all__` to `["Recommend", "RecommendRust", "StreamingRecommender", "REFERENCE", "IMPLEMENTATIONS"]`;
- leave `IMPLEMENTATIONS` unchanged.

- [ ] **Step 6: Build and smoke-test**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m maturin develop --release
/c/Users/Alexander/miniconda3/envs/p312/python.exe -c "
import polars as pl
from analytics.recommend import StreamingRecommender
r = StreamingRecommender().add(pl.DataFrame({'a': [0, 5, 7]})).add(pl.DataFrame({'a': [1], 'b': ['x']}))
print(r.finish().select('column', 'status', 'n_rows', 'n_null', 'rec_arrow_type'))
"
```

Expected: two rows, `a` → `uint8` and `b` → a string type, with `n_rows` 4 and `b` `n_null` 3.

- [ ] **Step 7: Commit**

```bash
git add services/analytics/src/python.rs services/analytics/analytics/_plugin.py services/analytics/analytics/recommend/streaming.py services/analytics/analytics/recommend/__init__.py
git commit -m "python: StreamingRecommender pyclass and wrapper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Python accuracy tests — `tests/test_streaming_recommend.py`

**Files:**
- Create: `tests/test_streaming_recommend.py`

- [ ] **Step 1: Write the tests:**

```python
"""
StreamingRecommender accuracy tests. Oracles: one-shot RecommendRust on the batches
concatenated diagonally (parity, spec §6); the pyarrow IPC oracle (_sizes) for ZSTD
sizes; a complete sample for the sampled ZSTD estimate; hand-worked known answers.
Accuracy only — nothing here is timed.
"""

from pathlib import Path

import polars as pl
import pytest

from analytics.describe import _sizes
from analytics.recommend import RecommendRust, StreamingRecommender
from datagen import describe_mixed, stringified

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"

CANDIDATE = pl.Struct(
    {
        "arrow_type": pl.String,
        "rule": pl.String,
        "evidence": pl.String,
        "predicted_bytes": pl.UInt64,
        "projected_population_bytes": pl.Float64,
        "outcome": pl.String,
        "reason": pl.String,
    }
)
SCHEMA = {
    "column": pl.String,
    "status": pl.String,
    "dtype": pl.String,
    "first_row": pl.UInt64,
    "n_rows": pl.UInt64,
    "n_null": pl.UInt64,
    "min": pl.String,
    "max": pl.String,
    "gcd": pl.Decimal(38, 0),
    "sum_len": pl.UInt64,
    "min_len": pl.UInt64,
    "max_len": pl.UInt64,
    "n_unique": pl.UInt64,
    "distinct_overflowed": pl.Boolean,
    "est_cardinality": pl.Float64,
    "est_low": pl.Float64,
    "est_high": pl.Float64,
    "est_method": pl.String,
    "size_bytes": pl.UInt64,
    "size_zstd_bytes": pl.UInt64,
    "size_polars_bytes": pl.UInt64,
    "size_polars_zstd_bytes": pl.UInt64,
    "rec_nullable": pl.Boolean,
    "rec_arrow_type": pl.String,
    "rec_arrow_size_bytes": pl.UInt64,
    "rec_arrow_size_zstd_bytes": pl.UInt64,
    "rec_polars_type": pl.String,
    "rec_polars_size_bytes": pl.UInt64,
    "rec_polars_size_zstd_bytes": pl.UInt64,
    "rec_lossy_formatting": pl.Boolean,
    "rec_candidates": pl.List(CANDIDATE),
    "n_sampled_rows": pl.UInt64,
    "n_sampled_blocks": pl.UInt64,
}


def stream(frame: pl.DataFrame, batch_rows: int, **params) -> pl.DataFrame:
    rec = StreamingRecommender(**params)
    for off in range(0, max(frame.height, 1), batch_rows):
        rec.add(frame.slice(off, batch_rows))
    return rec.finish()


def one_shot(frame: pl.DataFrame) -> dict[str, dict]:
    out = RecommendRust().add({"t": frame}).result()
    return {r["col_a"]: r for r in out.iter_rows(named=True)}


def kept_original(r: dict) -> bool:
    return any(
        c["outcome"] == "chosen" and c["rule"] == "original" for c in r["rec_candidates"]
    )


def candidates(r: dict, with_original_sizes: bool) -> list:
    return [
        (
            c["arrow_type"],
            c["rule"],
            str(c["outcome"]),
            c["predicted_bytes"]
            if with_original_sizes or not c["rule"].endswith("original")
            else None,
        )
        for c in r["rec_candidates"]
    ]


def assert_parity(streamed: pl.DataFrame, frame: pl.DataFrame, single_batch: bool):
    """Spec §6: equal recommendations, sizes and candidates; ZSTD when N ≤ block_rows."""
    ref = one_shot(frame)
    checked = 0
    for r in streamed.filter(pl.col("status") == "computed").iter_rows(named=True):
        o = ref[r["column"]]
        keys = ["rec_arrow_type", "rec_nullable", "rec_lossy_formatting"]
        if single_batch or not kept_original(o):
            keys += ["rec_arrow_size_bytes", "rec_polars_size_bytes"]
            keys += ["rec_arrow_size_zstd_bytes", "rec_polars_size_zstd_bytes"]
        for k in keys:
            assert r[k] == o[k], (r["column"], k, r[k], o[k])
        if not kept_original(o):
            assert r["rec_polars_type"] == o["rec_polars_type"], r["column"]
        assert candidates(r, single_batch) == candidates(o, single_batch), r["column"]
        checked += 1
    assert checked > 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


def test_contract():
    out = stream(describe_mixed(200), 50)
    assert list(out.schema.items()) == list(SCHEMA.items())
    assert out.to_arrow().num_rows == out.height
    assert out["column"].to_list() == describe_mixed(10).columns
    empty = StreamingRecommender().finish()
    assert empty.height == 0 and list(empty.schema) == list(SCHEMA)


def test_ineligible_columns_are_listed():
    frame = pl.DataFrame(
        {"w": pl.Series([1, 2], dtype=pl.Int128), "a": [1, 2], "n": [None, None]}
    )
    out = StreamingRecommender().add(frame).finish()
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert rows["w"]["status"] == "ineligible" and rows["w"]["dtype"] == "Int128"
    assert rows["w"]["n_null"] is None and rows["w"]["rec_arrow_type"] is None
    assert rows["n"]["status"] == "ineligible"  # still the Null type
    assert rows["a"]["status"] == "computed"


def test_finish_keeps_the_state():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    assert rec.finish()["n_rows"].to_list() == [2]
    assert rec.add(pl.DataFrame({"a": [3]})).finish()["n_rows"].to_list() == [3]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parity with one-shot RecommendRust (spec §6)


@pytest.mark.parametrize("batch_rows", [1, 7, 60, 200])
@pytest.mark.parametrize(
    "make",
    [lambda: describe_mixed(200), lambda: stringified(describe_mixed(200))],
    ids=["mixed", "stringified"],
)
def test_parity_with_one_shot(make, batch_rows):
    frame = make()
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)


@pytest.mark.slow
@pytest.mark.parametrize("batch_rows", [7_919, 50_000])
def test_parity_on_the_large_dataset(batch_rows):
    frame = pl.read_ipc(LARGE)
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)


def test_parity_with_columns_appearing_and_disappearing():
    parts = [
        pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}),
        pl.DataFrame({"a": [4], "c": [0.5]}),
        pl.DataFrame({"b": ["z", None], "c": [1.25, None]}),
    ]
    rec = StreamingRecommender()
    for p in parts:
        rec.add(p)
    assert_parity(rec.finish(), pl.concat(parts, how="diagonal"), single_batch=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZSTD sizes against the IPC oracle, and the sampled estimate


def test_zstd_sizes_are_those_of_a_file_written_in_blocks():
    frame = describe_mixed(3_000).select("i32", "u16", "f64_price", "str_free", "date")
    block = 1_000
    out = stream(frame, 700, block_rows=block, reservoir_rows=4 * block)
    for r in out.iter_rows(named=True):
        s = frame[r["column"]]
        want = sum(
            _sizes.ipc_body_bytes(
                s.slice(o, block).to_arrow(compat_level=pl.CompatLevel.oldest()), 1
            )
            for o in range(0, frame.height, block)
        )
        assert r["size_zstd_bytes"] == want, r["column"]
        assert r["n_sampled_rows"] == frame.height


@pytest.mark.slow
def test_sampled_zstd_estimate_is_within_ten_percent():
    frame = pl.read_ipc(LARGE)
    full = stream(frame, 10_000, block_rows=4_096, reservoir_rows=frame.height + 4_096)
    sampled = stream(frame, 10_000, block_rows=4_096, reservoir_rows=16_384)
    for a, b in zip(full.iter_rows(named=True), sampled.iter_rows(named=True)):
        if a["status"] != "computed" or (a["rec_arrow_size_zstd_bytes"] or 0) < 1_000:
            continue
        assert b["rec_arrow_size_zstd_bytes"] == pytest.approx(
            a["rec_arrow_size_zstd_bytes"], rel=0.10
        ), a["column"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Known answers


def row(out: pl.DataFrame, column: str) -> dict:
    return out.filter(pl.col("column") == column).row(0, named=True)


def by_rule(r: dict) -> dict:
    return {c["rule"]: c for c in r["rec_candidates"]}


def test_a_new_column_is_backfilled():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    out = rec.add(pl.DataFrame({"a": [3], "b": ["x"]})).finish()
    b = row(out, "b")
    assert (b["first_row"], b["n_rows"], b["n_null"], b["rec_nullable"]) == (2, 3, 2, True)


def test_an_absent_column_counts_as_null():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1], "b": [1]}))
    out = rec.add(pl.DataFrame({"a": [2, 3]})).finish()
    assert row(out, "b")["n_null"] == 2


def test_a_null_typed_column_adopts_a_type():
    rec = StreamingRecommender().add(pl.DataFrame({"b": pl.Series([None, None], dtype=pl.Null)}))
    b = row(rec.add(pl.DataFrame({"b": ["x", "y"]})).finish(), "b")
    assert b["status"] == "computed" and b["n_null"] == 2


def test_a_type_change_is_rejected():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="type changed"):
        rec.add(pl.DataFrame({"a": ["x"]}))
    assert row(rec.finish(), "a")["n_rows"] == 1


def test_overflow_rejects_the_dictionary():
    frame = pl.DataFrame({"s": ["a", "b", "c", "d", "e", "a"]})
    s = row(stream(frame, 2, categorical_threshold=3), "s")
    assert s["n_unique"] is None and s["distinct_overflowed"] is True
    assert s["est_method"] == "overflowed"
    assert by_rule(s)["string→dictionary"]["outcome"] == "rejected"


def test_no_sample_means_no_zstd_sizes():
    out = stream(pl.DataFrame({"a": [1, 2, 3]}), 2, reservoir_rows=0)
    a = row(out, "a")
    assert a["size_zstd_bytes"] is None and a["rec_arrow_size_zstd_bytes"] is None
    assert (a["n_sampled_rows"], a["rec_arrow_type"]) == (0, "uint8")


def test_a_float_that_does_not_round_trip_fails_by_statistic():
    tiny = "0." + "0" * 400 + "1"
    s = row(stream(pl.DataFrame({"s": [tiny, "1"]}), 1), "s")
    c = by_rule(s)["string→float64"]
    assert c["outcome"] == "failed" and c["reason"].startswith("n_f64_roundtrip_fail=")


def test_nanoseconds_out_of_range_fail_by_statistic():
    frame = pl.DataFrame({"s": ["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"]})
    c = by_rule(row(stream(frame, 1), "s"))["string→timestamp"]
    assert c["outcome"] == "failed" and "iso_instant" in c["reason"]


@pytest.mark.parametrize(
    "params",
    [
        {"block_rows": 0},
        {"reservoir_rows": 10, "block_rows": 100},
        {"boolean_pairs": (("a", "A"),)},
        {"zstd_level": 99},
    ],
)
def test_parameters_are_validated(params):
    with pytest.raises(ValueError):
        StreamingRecommender(**params)


def test_lazyframe_is_refused():
    with pytest.raises(TypeError, match="LazyFrame"):
        StreamingRecommender().add(pl.LazyFrame({"a": [1]}))
```

- [ ] **Step 2: Run the tests**

Run: `cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests/test_streaming_recommend.py -v`
Expected: all non-slow tests PASS.

Then run the slow ones: `… -m pytest tests/test_streaming_recommend.py -v -m slow` (check `conftest.py` for how `slow` is enabled, and use its flag if it differs). Expected: PASS.

If a parity assertion fails, the tuple names the column and field. Parity differences are bugs in `partial.rs` or `streaming.rs`, not in the oracle. The parity rules are documented in spec §6 and in this plan's amendment 6.

- [ ] **Step 3: Commit**

```bash
git add tests/test_streaming_recommend.py
git commit -m "tests: streaming recommender accuracy (contract, parity, ZSTD, known answers)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: C ABI — `capi.rs`

**Files:**
- Modify: `services/analytics/src/capi.rs`

The handle crosses as `void *`, because a pointer to a Rust type in an `extern "C"` signature trips `improper_ctypes_definitions`. The C header still names it `AnalyticsRecommender *`.

- [ ] **Step 1: Write the failing tests** in `capi.rs`'s `mod tests`:

```rust
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
            assert_eq!(unsafe { analytics_recommender_add(h, &mut input, &mut error) }, OK);
        }
        let mut output = FFI_ArrowArrayStream::empty();
        assert_eq!(unsafe { analytics_recommender_finish(h, &mut output, &mut error) }, OK);
        let batches: Vec<RecordBatch> = ArrowArrayStreamReader::try_new(output)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let rec = batches[0].column_by_name("rec_arrow_type").unwrap().as_string_view();
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
    }
```

Add `use std::ffi::c_void;` to the test imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib capi::`
Expected: `cannot find function analytics_recommender_new`.

- [ ] **Step 3: Implement.** In `capi.rs`:
- extend the imports: `use std::ffi::{c_char, c_int, c_void, CStr, CString};`, `use std::sync::Mutex;` and `use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};`;
- append before `#[cfg(test)]`:

```rust
/// What an `AnalyticsRecommender *` points to.
type Recommender = Mutex<api::StreamingRecommender>;

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
/// handle the caller frees with `analytics_recommender_free`; it is thread-safe.
///
/// # Safety
/// `out` is valid for one pointer write. When `n_bool_pairs > 0`, `bool_true` and
/// `bool_false` each point to `n_bool_pairs` NUL-terminated UTF-8 strings. `error` is
/// null or valid for one pointer write.
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

/// Adds every batch of `batches`, in order; each batch is atomic (on failure the
/// batches before it stay added).
///
/// # Safety
/// `h` is null or a live handle. `batches` is null or a valid ArrowArrayStream; it is
/// consumed (left released). `error` is null or valid for one pointer write.
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
        let reader = unsafe { ArrowArrayStreamReader::from_raw(batches) }
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
/// kept. On success `*out` holds a one-batch stream the caller owns.
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
        unsafe { ptr::write(out, stream) };
        Ok(())
    })();
    unsafe { finish(result, error) }
}

/// Frees a handle. Null is a no-op.
///
/// # Safety
/// `h` is null or a live handle, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn analytics_recommender_free(h: *mut c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h as *mut Recommender) });
    }
}
```

- [ ] **Step 4: Run the tests, both feature sets, and clippy**

```bash
cargo test --lib capi::
cargo build --release --no-default-features --target-dir target/capi
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings
```

Expected: all clean. Fix any clippy findings in the new files (typically `too_many_arguments` allowances, or `needless_range_loop`).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/capi.rs
git commit -m "capi: analytics_recommender_{new,add,finish,free}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Benchmark script

**Files:**
- Create: `tests/performance/benchmark_streaming_recommend.py`

- [ ] **Step 1: Write the script:**

```python
"""
Streaming recommender speed: add throughput (rows/s) and finish time, beside one-shot
RecommendRust on the same data. Standalone — the shared harness assumes IMPLEMENTATIONS.

Run: /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_streaming_recommend.py

Results → tests/performance/results/streaming_recommend.parquet (git-ignored).
`peak_mb` is the process's peak memory so far (dataset included), where the OS reports it.
"""

import statistics
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/: datagen

from datagen import describe_narrow, describe_wide  # noqa: E402

from analytics.recommend import RecommendRust, StreamingRecommender  # noqa: E402

DATASETS = {
    "large_dataset": lambda: pl.read_ipc(HERE.parent / "data" / "large_dataset.arrow"),
    "narrow 1M x 4": lambda: describe_narrow(1_000_000),
    "wide 50K x 100": lambda: describe_wide(50_000, 100),
}
BATCH_ROWS = (10_000, 100_000)
RUNS = 3


def peak_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
    except ImportError:
        try:
            import psutil
        except ImportError:
            return None
        return getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20


def stream(frame: pl.DataFrame, batch_rows: int) -> tuple[float, float]:
    rec = StreamingRecommender()
    t0 = time.perf_counter()
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    t1 = time.perf_counter()
    rec.finish()
    return t1 - t0, time.perf_counter() - t1


def main() -> None:
    rows = []
    for name, make in DATASETS.items():
        frame = make()
        one_shot = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            RecommendRust().add({"t": frame}).result()
            one_shot.append(time.perf_counter() - t0)
        for batch_rows in BATCH_ROWS:
            runs = [stream(frame, batch_rows) for _ in range(RUNS)]
            add_s = statistics.median(r[0] for r in runs)
            rows.append(
                {
                    "dataset": name,
                    "rows": frame.height,
                    "cols": frame.width,
                    "batch_rows": batch_rows,
                    "add_s": add_s,
                    "add_rows_per_s": frame.height / add_s,
                    "finish_s": statistics.median(r[1] for r in runs),
                    "one_shot_s": statistics.median(one_shot),
                    "peak_mb": peak_mb(),
                }
            )
            print(rows[-1])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "streaming_recommend.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_streaming_recommend.py`
Expected: six printed rows, then `tests/performance/results/streaming_recommend.parquet` written.

- [ ] **Step 3: Commit**

```bash
git add tests/performance/benchmark_streaming_recommend.py
git commit -m "bench: streaming recommender throughput beside one-shot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Documentation — CLAUDE.md and the spec

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-09-29-streaming-recommender-design.md`, `docs/superpowers/specs/2026-09-26-recommend-technique-design.md`

- [ ] **Step 1: CLAUDE.md.**

(a) Under "## 1. Per-column", after the Recommend paragraph, add:

```markdown
**Streaming Recommend — `analytics.recommend.StreamingRecommender`** (Rust only; not on the uniform contract). Recommend's dtype recommendations from batches added over time (spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md). `add(frame)` any number of times (columns may appear, disappear or start as Null; other type changes raise), `finish()` at any point → one row per column: `column, status, dtype, first_row, n_rows, n_null, min, max, gcd, sum_len, min_len, max_len, n_unique, distinct_overflowed, est_*`, the size columns, Recommend's `rec_*` columns, `n_sampled_rows, n_sampled_blocks`. Exact running statistics (src/partial.rs) prove each recommendation on every row (`prove`, `lossy_by_stats` in recommend.rs); a seeded Algorithm-L sample of contiguous row blocks (src/reservoir.rs) gives ZSTD sizes — those of an IPC file written in `block_rows` batches — and cross-checks the chosen type. Distinct values tracked only for text, up to `categorical_threshold`. No `population_rows` (estimates: Schnabel → Chao1). Keywords: `reservoir_rows=524_288` (0: no sample, ZSTD null), `block_rows=65_536`, `categorical_threshold=10_000`, `zstd_level=1`, `seed=0`, `boolean_pairs`.
```

(b) In "# Project Structure", add `partial.rs, reservoir.rs, streaming.rs` to the `src/` list, and `streaming.py` under `recommend/`.

(c) In "# Rust Extension":
- extend the `api.rs` bullet with: `` `StreamingRecommender` (new / add / mark_ineligible / finish) is the one stateful entry point. ``;
- extend the `python.rs` bullet with: `` `StreamingRecommender` pyclass (frozen, mutex-guarded; `add` reads its stream batch by batch). ``;
- extend the `capi.rs` bullet with: `` and `analytics_recommender_{new,add,finish,free}` (opaque `void *` handle). ``.

(d) In the "Private" list, add `StreamingRecommender (via _plugin.streaming_recommender, used only by analytics.recommend.StreamingRecommender)`.

- [ ] **Step 2: The spec.** In `2026-09-29-streaming-recommender-design.md`:
- change `Status: approved, not implemented.` to `Status: implemented (plan docs/superpowers/plans/2026-09-30-streaming-recommender.md).`;
- append a section `## 11. Amendments made during planning`, containing the seven numbered items from this plan's "Spec amendments" section, verbatim.

In `2026-09-26-recommend-technique-design.md`, add under the title: `> See also: [the streaming recommender](2026-09-29-streaming-recommender-design.md), which recommends from batches added over time.`

- [ ] **Step 3: Final verification**

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics && cargo test --lib && cargo clippy --all-targets -- -D warnings
cd /c/Users/Alexander/turbo-parakeet && /c/Users/Alexander/miniconda3/envs/p312/python.exe -m pytest tests -q
```

Expected: everything passes, the existing techniques included.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git add -f docs/superpowers/specs/2026-09-29-streaming-recommender-design.md docs/superpowers/specs/2026-09-26-recommend-technique-design.md
git commit -m "docs: streaming recommender in CLAUDE.md; spec amendments

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
