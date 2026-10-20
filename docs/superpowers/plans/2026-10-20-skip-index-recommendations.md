# Skip-Index Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recommend at most one ClickHouse skip index per column (`minmax`, `set(N)`, `bloom_filter(fpr)` with GRANULARITY) for point queries, in both recommenders and every binding, with every candidate's skip shares, index size and reason; validate skip shares and index sizes with chbench.

**Architecture:** The existing skip-index kernels (`techniques/single_column/ordered/skip_index/`) gain a multi-segment entry point; a new pure decision module (`recommenders/engine/skip_index.rs`) applies the gate / overflow / bloom-economics / granularity / cross-kind rules to per-candidate metrics; index sizes are emulated in `common/clickhouse_sizes.rs` (bloom bits per row from a chbench measurement). One-shot builds the population from the recommended array; streaming from the reservoir's sampled blocks as segments. `to_chbench` adds the skip-index axis and point queries; a calibration compares predicted and measured skipping and index bytes.

**Tech Stack:** Rust (polars 0.51, arrow-rs 60, rayon), Python 3.12, Java 25 FFM, ClickHouse 26.8 (container `chbench-ch`), chbench.

**Spec:** `docs/superpowers/specs/2026-10-20-skip-index-recommendations-design.md` (read fully). Background: skip-index technique spec `docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md` (§2 model, §5 streaming readiness), ClickHouse spec `docs/superpowers/specs/2026-10-19-clickhouse-recommendations-design.md`.

---

## Conventions

turbo-parakeet (`c:/Users/Alexander/turbo-parakeet`, branch `skip-index-recommendations` from Task 0):
```bash
E=/c/Users/Alexander/miniconda3/envs/p312
export PATH="$E:$E/Scripts:$E/Library/bin:/c/Users/Alexander/.cargo/bin:$PATH" PYO3_PYTHON="$E/python.exe"
PY=$E/python.exe
```
- Rust tests `cd services/analytics && cargo test --lib <filter>`; build `$PY scripts/check.py build` (≈ 3 min; background or timeout 600000); pytest `$PY -m pytest <files> -q` (`-m "slow or not slow"` for slow ones); Java `$PY scripts/check.py test-java`.
- Before each commit `$PY scripts/check.py fmt && $PY scripts/check.py lint`; revert files `fmt` touched that the task did not change. Commit without `--no-verify` (env on PATH); messages end with a blank line and `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; stage by explicit path; `docs/superpowers/` needs `git add -f`; never push.
- Clippy `-D warnings`: `#[expect(dead_code)]` on items does not work in this toolchain (1.96); use a module-level `#![cfg_attr(not(test), allow(dead_code))]` for a module nothing calls yet, removed when used.
- Layering: `common` ← `techniques` ← `recommenders` ← `bindings`.

chbench (`c:/Users/Alexander/clickhouse_benchmark`, branch `skip-index-sizes` from Task 1): its CLAUDE.md conventions (TDD, spikes + FINDINGS, rulings; `clickhouse` marker); same Python env; identity is repo-local. Server: container `chbench-ch` (ClickHouse 26.8.31.2), localhost:8123, `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD`, database `bench` (`docker start chbench-ch` if stopped).

Use the Write / Edit tools for files (heredocs were unreliable in this environment).

## File map

| Repo / file | Change |
|---|---|
| tp `src/techniques/single_column/ordered/skip_index/mod.rs` | `over_segments` (multi-segment entry point) |
| tp `src/common/clickhouse_sizes.rs` | `skip_index_bytes` (minmax / set / bloom), `BLOOM_BITS_PER_ROW` table |
| tp `src/recommenders/engine/skip_index.rs` (new) | candidate metrics → decision (pure) |
| tp `src/recommenders/engine/{candidates.rs,column.rs}` | `Params.skip`; one-shot wiring |
| tp `src/recommenders/schema.rs` | `clickhouse_skip_indexes` (final type), `clickhouse_skip_index` |
| tp `src/recommenders/streaming/mod.rs` | streaming wiring |
| tp `src/bindings/{api/recommenders.rs,python.rs,capi.rs}`, Java params | six keywords |
| tp `analytics/_plugin.py`, `analytics/recommend/{oneshot,streaming,chbench}.py` | keywords; skip axis + point queries |
| tp `tests/test_skip_index_recommend.py` (new), `tests/test_streaming_recommend.py`, `tests/test_clickhouse_recommend.py` | tests |
| tp `tests/performance/calibrate_skip_indexes.py` (new) | calibration |
| ch `chbench/harness/{loader,store,report,orchestrate}.py`, `spikes/spike_ch_bloom_size.py`, `spikes/FINDINGS.md` | index sizes; bloom size spike |
| both | CLAUDE.md, ToDo.md, specs |

---

### Task 0: Branch and spec amendment

- [ ] turbo-parakeet: `git switch -c skip-index-recommendations` (from `main`).
- [ ] Spec §7: replace "(b) the denominators (non-null rows, distinct values) injected from the running statistics instead of `Population::counts`; (c) `Population::from_ids` given counts that describe exactly the sampled rows.  The random-order baselines use the running counts (exact while the distinct sample is exact; else its estimates)." with "(b) the population — ids and counts — is that of the sampled rows, so `Population::from_ids`' counts describe exactly the rows scanned, and the denominators and the random-order baselines are the sample's (consistent with the windows; exact when every row is sampled).  Injecting the running counts instead is a later refinement."
- [ ] Commit `"Spec: skip-index recommendations — streaming population from the sampled rows"` (`git add -f`).

---

### Task 1: chbench — index sizes in the store, bloom size spike (chbench repo)

Spec §8.1, §5.

- [ ] `git switch -c skip-index-sizes` in chbench (from `main`).
- [ ] TDD: `loader.storage_stats` (or a sibling) also reads the table's skip-index sizes from `system.data_skipping_indices` (`data_compressed_bytes`, `data_uncompressed_bytes`, `marks` where available; filter by database / table; 0 when no index); `store` persists them per table (a migration-free addition consistent with the store's style); `report` CSV gains `index_compressed_bytes`, `index_uncompressed_bytes`. A `clickhouse`-marked test: a tiny table with `minmax` and one with `set(100)` record non-zero index bytes; `none` records 0.
- [ ] Spike `spikes/spike_ch_bloom_size.py`: for fpr ∈ {0.001, 0.01, 0.025, 0.05} and GRANULARITY ∈ {1, 4}, Int64 columns of 1 000 000 rows with (a) all distinct values, (b) 100 distinct values per 8 192-row granule (clustered), (c) 10 distinct values overall; `bloom_filter(fpr) GRANULARITY G`, Wide parts, `OPTIMIZE FINAL`; record index compressed / uncompressed bytes, and derive bytes per row and per (block × distinct-in-block) — establish whether ClickHouse sizes the filter by rows or by distinct values per block, and the bits per row (or per distinct value) for each fpr; also record `minmax` and `set(125)` index bytes for (a)–(c) at G = 1 for the size model's calibration. FINDINGS §10 + a ruling (next after R36). Report the table.
- [ ] Full chbench suite passes (server up). Commit `"harness: skip-index sizes in the store; spike: bloom_filter size by fpr"`.

---

### Task 2: Multi-segment window scans (turbo-parakeet techniques)

Spec §7 (a).

**Files:** `src/techniques/single_column/ordered/skip_index/mod.rs`.

- [ ] Add:
```rust
/// Window starts over several segments (e.g. a streaming recommender's sampled blocks laid end to
/// end in one population): for each segment `[start, end)` with `end − start ≥ w`, the starts
/// `start ..= end − w` (no window crosses a segment boundary), scanned with `over_ranges`'
/// parallel split and merged. Segments shorter than `w` contribute nothing.
pub(crate) fn over_segments<F>(segments: &[(usize, usize)], w: u64, min_len: u64, scan: F) -> WindowSums
where
    F: Fn(Range<usize>) -> WindowSums + Sync,
```
  Implement by mapping each segment to its start range and splitting long ranges as `over_ranges` does (reuse its range-length rule; offset each sub-range by the segment's start).
- [ ] Tests: one segment `(0, n)` equals `over_ranges(windows(n, w).1, …)` for both `minmax::scan` and `set::scan` on a seeded population; two segments `(0, a), (a, n)` equal the sum of scanning each half separately and differ from one segment when a window would straddle `a`; a segment shorter than `w` contributes 0 windows; empty segment list → default sums.
- [ ] `cargo test --lib skip_index` → pass. Commit `"skip_index: multi-segment window scans"`.

---

### Task 3: Index-size emulation (turbo-parakeet common)

Spec §5 (with Task 1's findings).

**Files:** `src/common/clickhouse_sizes.rs`.

- [ ] Add a `pub(crate) const BLOOM_BITS_PER_ROW: [(f64, f64); 4]` (fpr → bits per row, or per distinct value — whichever Task 1 established; name it accordingly and document the source: chbench FINDINGS §10 / ruling) and:
```rust
/// A skip index's kind for size emulation.
pub(crate) enum IndexKind { MinMax, Set { max_size: u64 }, Bloom { fpr: f64 } }

/// Emulated bytes of a skip index over `streams`' value stream (the column's ClickHouse layout,
/// spec 2026-10-19 §3) with index blocks of `block_rows` rows; Err for an fpr not in the table.
pub(crate) fn skip_index_bytes(
    a: &dyn Array,
    layout: &ChLayout,
    kind: &IndexKind,
    block_rows: u64,
) -> Result<u64, String>
```
  minmax: per block the min and max non-null physical values at the layout's width (plus a byte per block when nullable), the block stream compressed with LZ4 + 25-byte headers (reuse `emulate`'s block machinery on a synthetic stream); set: per block a UInt64 count + the distinct values (String layout for strings; ≤ max_size else count 0 and no values), LZ4-compressed; bloom: per block ⌈(rows or distinct, per Task 1) × bits / 8⌉ bytes, uncompressed. Calibrate the structure (headers, per-block counts) to Task 1's minmax / set measurements where they show a clear constant (e.g. a fixed per-block overhead); otherwise keep the model and let Task 10 measure the error.
- [ ] Unit tests: minmax over 4 blocks of an Int32 column ≈ small and independent of the values' spread; set over a 3-value column ≈ 3 values per block; set overflowing → only counts; bloom bytes = table value × rows / 8 per block; unknown fpr → Err.
- [ ] `cargo test --lib clickhouse_sizes` → pass. Commit `"ClickHouse emulator: skip-index sizes"`.

---

### Task 4: The decision module (pure)

Spec §4.

**Files:** create `src/recommenders/engine/skip_index.rs` (register in `engine/mod.rs`; module-level dead-code allow until Task 6).

- [ ] Types:
```rust
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkipParams {
    pub granularities: Vec<u32>,   // (1, 2, 4, 8)
    pub set_max_sizes: Vec<u64>,   // (125, 250)
    pub bloom_fprs: Vec<f64>,      // (0.025,)
    pub min_skip: f64,             // 0.3
    pub min_gain: f64,             // 0.1
    pub max_overflow: f64,         // 0.1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind { MinMax, Set(u64), Bloom(f64) }

/// One evaluated candidate's measurements (from the techniques and the size emulator).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Measured {
    pub kind: Kind,
    pub granularity: u32,
    pub skip_rows: f64,
    pub skip_values: f64,
    pub expected_skip_rows: f64,
    pub bytes: Option<u64>,
    /// Set only: (window_distinct_mean, window_distinct_max, overflow_fraction).
    pub overflow: Option<(f64, u64, f64)>,
}

/// One `clickhouse_skip_indexes` row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkipCandidate {
    pub measured: Option<Measured>, // None for a pruned row with no measurement (e.g. G too large)
    pub kind: Kind,
    pub granularity: u32,
    pub outcome: &'static str,      // chosen | rejected | pruned
    pub evidence: String,
}

impl Kind {
    /// "minmax", "set(125)", "bloom_filter(0.025)".
    pub(crate) fn ddl(&self) -> String
}

/// Spec §4 steps 1–5 over the minmax / set candidates; `bloom(...)` is called only when none of
/// them passed, and returns the bloom candidates' measurements. `column_bytes` = the column's
/// predicted ClickHouse compressed bytes. Returns all rows, chosen first then by skip_rows desc.
pub(crate) fn decide_skip(
    cheap: Vec<Measured>,
    pruned: Vec<SkipCandidate>,
    bloom: impl FnOnce() -> Vec<Measured>,
    column_bytes: Option<u64>,
    p: &SkipParams,
) -> Vec<SkipCandidate>
```
  Rules: gate (`skip_rows ≥ min_skip` and `skip_rows − expected_skip_rows ≥ min_gain`, else `rejected` "skip 0.12 < 0.30" / "gain 0.04 over random order < 0.10"); set overflow `pruned` when `overflow_fraction ≥ max_overflow` ("window distinct mean 812, max 8 192; overflow 0.97 ≥ 0.10"); bloom only when no cheap candidate passed, gated, and economics `skip_rows × column_bytes > bytes` (else `rejected` "saves 1 200 B/query < index 9 800 B"; `column_bytes` None → bloom `pruned` "no ClickHouse size"); per kind (per N / fpr) the largest G within 0.02 of that kind's best skip_rows; across kinds the highest skip_rows, ties within 0.02 to the smaller `bytes` (then minmax < set < bloom); exactly one `chosen` or none; others `rejected` with "lower skip than <chosen ddl> G<g>" / "larger index than …" / "coarser G <g> chosen".
- [ ] Unit tests for each rule with hand-built `Measured` lists (a closure counter asserts `bloom` is not called when a cheap candidate passes).
- [ ] `cargo test --lib skip_index` → pass. Commit `"Decisions: skip-index gate, overflow, bloom economics, granularity and kind choice"`.

---

### Task 5: Keywords and columns (null for now)

Spec §4 (keywords), §6.

**Files:** `engine/candidates.rs` (`Params.skip: SkipParams`), all `Params { .. }` literals, `bindings/api/recommenders.rs` (both params structs gain `skip_granularities: Vec<u32>, set_max_sizes: Vec<u64>, bloom_fprs: Vec<f64>, min_skip: f64, min_gain: f64, max_overflow: f64`; validation: granularities non-empty, distinct, ≥ 1; max sizes non-empty, distinct, ≥ 1; fprs non-empty, each in (0, 1); fractions finite in [0, 1] — messages naming the keyword), `python.rs` (defaults), `capi.rs` (defaults for now), `_plugin.py`, wrappers (keywords + docstrings), `schema.rs`, `engine/column.rs` (`Rec` gains `skip_index: Option<String>`, `skip_candidates: Vec<SkipCandidate>`; `rec_row`), streaming `Rec`.

- [ ] Columns: `clickhouse_skip_index` String right after `clickhouse_codecs`; `clickhouse_skip_indexes` (`List<Struct{kind: Utf8, granularity: UInt32, max_size: UInt32, fpr: Float64, bytes: UInt64, skip_rows: Float64, skip_values: Float64, expected_skip_rows: Float64, outcome: Utf8, evidence: Utf8}>`) right after `clickhouse_candidates`; `kind` is "minmax" / "set" / "bloom_filter" with `max_size` / `fpr` in their fields; `clickhouse_skip_index` renders `"<ddl> GRANULARITY <g>"`.
- [ ] Tests: SCHEMA dict in `tests/test_streaming_recommend.py`; a keyword-validation test for both recommenders in new `tests/test_skip_index_recommend.py` (empty / repeated / zero granularities, fpr 0 or 1, `min_skip` 1.5 → ValueError naming the keyword).
- [ ] cargo, build, pytest (recommend, streaming, clickhouse, profile, skip-index files), Java → pass. Commit `"Recommenders: skip-index keywords and columns (null for now)"`.

---

### Task 6: One-shot wiring

Spec §2–§6.

**Files:** `engine/column.rs` (or a new `engine/skip_choice.rs` next to `clickhouse_choice`), tests.

- [ ] For a computed column with a ClickHouse type and an eligible dtype (`_dtypes.ORDERED_SCALARS` equivalent in Rust — the skip-index techniques' eligibility; check how `techniques/.../skip_index` / its Python base decide eligibility and reuse it): build the `Population` from the **recommended** array (imported as a Polars Series via `arrow_io` — the order of the recommended values is what ClickHouse compares); for each G: `w = G × 8 192` (`windows(n, w)`), `minmax::scan` + `minmax::baseline` → `Measured{MinMax}`; for each N: `set::scan` + `set::baseline` → `Measured{Set(N)}` with overflow figures; the bloom closure: `set` at `max_size = 0` → shares × (1 − fpr) for each fpr, bytes from `skip_index_bytes`; minmax / set bytes from `skip_index_bytes`. Kinds rejected for the ClickHouse type by chbench R36 (`bloom_filter` on `Decimal*`, `Date32`, `Time64`) → `pruned` rows. A column shorter than one window (n < w) → the techniques' one-window rule applies (spec 2026-10-16 §2.1), as the techniques do. `column_bytes` = `ch_size`.
- [ ] Fill `Rec.skip_index` / `skip_candidates` via `decide_skip`.
- [ ] Python known-answer tests (`tests/test_skip_index_recommend.py`), 200 000 rows each, seeded:
  - sorted unique int → `clickhouse_skip_index` starts with `minmax`;
  - high-cardinality ids clustered in runs of ~1 000 rows each (values random across runs) → `set(125)` (window distinct ≈ 9 per 8 192-row window);
  - a 50-value column in random order → null, and every candidate `rejected` by the gain rule;
  - grouped random 64-bit hashes (runs of 10 rows of the same random hash: high local cardinality ~820 per window, each value in one window, unordered within the block) → `bloom_filter(0.025)`;
  - a Decimal column of the same grouped shape → no bloom candidate (a `pruned` row naming R36);
  - keywords: `min_skip=0.99` → null for the sorted column; `max_overflow=1.0` lets an overflowing set pass the overflow rule (then the gate decides); `skip_granularities=(1,)` → only G = 1 rows.
- [ ] build; tests pass. Commit `"One-shot recommender: skip-index recommendation"`.

---

### Task 7: Streaming wiring

Spec §7 (+ Task 0 amendment).

**Files:** `streaming/mod.rs`, `tests/test_streaming_recommend.py`, `tests/test_skip_index_recommend.py`.

- [ ] Concatenate the sampled blocks' recommended arrays (the IPC path already concatenates — reuse), build the `Population` over them, segments = the blocks' row ranges in that concatenation; scans via `over_segments`; baselines from that population; sizes from `skip_index_bytes` on the concatenated sample scaled by `n_rows / sampled`; `column_bytes` = the streamed `ch_size`. G whose w > `block_rows` → `pruned` "window of <w> rows exceeds block_rows <b>". No sample → no candidates, null index.
- [ ] Parity in `assert_parity`: when everything is sampled in one reservoir block (as for the other ClickHouse columns), `clickhouse_skip_index` and every candidate's shares equal one-shot's (RTOL 1e-9); otherwise `clickhouse_skip_index` equal unless the best two candidates are within 0.05 skip (then either). Add `test_streaming_skip_index_matches_one_shot` (the four known-answer frames, 50 000-row batches, everything sampled: equal choices; shares within 0.02).
- [ ] build; the four pytest files with `-m "slow or not slow"` → pass. Commit `"Streaming recommender: skip indexes over the sampled blocks"`.

---

### Task 8: C ABI and Java keywords

Spec §4 keywords.

- [ ] C ABI `_new` (both) gain before `out`: `const uint32_t *skip_granularities, size_t n_skip_granularities, const uint64_t *set_max_sizes, size_t n_set_max_sizes, const double *bloom_fprs, size_t n_bloom_fprs, double min_skip, double min_gain, double max_overflow` (null with n > 0 → InvalidInput naming it); capi tests updated + one invalid case.
- [ ] Java `OneShotParams` / `StreamingParams` gain trailing `List<Integer> skipGranularities, List<Long> setMaxSizes, List<Double> bloomFprs, double minSkip, double minGain, double maxOverflow` (old overloads keep working with the defaults); descriptors and `create`; tests: `clickhouse_skip_index` of a sorted column starts with `minmax`; an invalid `minSkip` throws.
- [ ] cargo + Java → pass. Commit `"C ABI and Java: skip-index keywords"`.

---

### Task 9: `to_chbench` skip axis and point queries

Spec §8.2.

**Files:** `analytics/recommend/chbench.py`, tests.

- [ ] The recommended suite's `skip_index` axis = `[<chosen ddl without GRANULARITY>, none]` when an index is chosen (chbench's vocabulary: `minmax`, `set(N)`, `bloom_filter` — check chbench's `is_known_skip_index`; if `bloom_filter(fpr)` with an fpr is not accepted, write `bloom_filter` only when fpr = 0.025 (ClickHouse's default) and otherwise skip the axis value with a note in the returned skipped list); the GRANULARITY: chbench's DDL fixes `GRANULARITY 1` — if the chosen G ≠ 1, either extend chbench's axis to accept a granularity (preferred — coordinate in Task 10's chbench step) or record the cell as not like-for-like; queries gain `{kind: filter, predicate: in, values: 1, target_fraction: f}` for f in (0.001, 0.01, 0.05). `predicted.json` gains, per skip-index cell, the predicted index bytes.
- [ ] Tests: the sorted / clustered frames produce the axis values; the YAML loads with chbench's `load_suite`.
- [ ] Commit `"to_chbench: skip-index axis and point queries"`.

---

### Task 10: Calibration (turbo-parakeet + chbench)

Spec §8.3, §10.

- [ ] chbench (branch `skip-index-sizes`): if Task 9 needs it, a `skip_index` axis value with a granularity (e.g. `minmax:4` or a separate `skip_granularity` axis — follow chbench's suite style; TDD; measured: the spike in Task 1 already exercised G ∈ {1, 4}); commit.
- [ ] turbo-parakeet: `tests/performance/calibrate_skip_indexes.py`: frames = the four known-answer shapes at 1 000 000 rows + `datagen.large_dataset_mixed()`; recommender → `to_chbench` → `chbench run` (queries as written by Task 9, `samples: 1`) → for each point query, the chosen value (from chbench's stored `predicate_sql`) → the exact predicted skip for that value computed in Python from the dataset file (share of index blocks of G × 8 192 rows not containing v for set / bloom, or with v outside [min, max] for minmax, × (1 − fpr) for bloom) vs the measured `1 − selected_marks / total_marks` (total marks from the table's rows / 8 192); and per table the predicted index bytes vs the measured `index_compressed_bytes`. Write `tests/performance/results/skip_index_calibration.arrow` and print summaries by kind × G.
- [ ] Fill spec §10: Task 1's bloom table, the skip-share and index-size calibration tables, the adopted tolerances (from the evidence; name exceptions). Commit both repos.

---

### Task 11: Documentation

- [ ] turbo-parakeet CLAUDE.md (recommender paragraphs: candidates, metrics, decision rules and keywords, the two columns, streaming segments, calibration tolerances; Project Structure), ToDo.md (1c, 1d, 1e **Done**, spec 2026-10-20; the streaming sub-bullets under 1d/1e done; follow-ups found), chbench CLAUDE.md (index sizes; the bloom spike; the granularity axis if added). The user's ToDo items 15–16 stay byte-for-byte intact. Commit each repo.

---

### Task 12: Full verification

- [ ] turbo-parakeet `$PY scripts/check.py ci` + `-m slow` → pass; chbench `$PY -m pytest -q` (server up) → pass. Check the spec section by section; report deviations.
