# ClickHouse Recommendations and the chbench Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recommend a ClickHouse type, codec chain (GCD, LZ4 / ZSTD + level) with a predicted size, and the Arrow IPC compression for DataFusion / Polars, in both recommenders; hand each recommendation to `clickhouse_benchmark` as a runnable suite over the real column; calibrate the predictions against measured bytes.

**Architecture:** turbo-parakeet gains a ClickHouse on-disk size emulator (`common/clickhouse_sizes.rs`), a type mapping (`recommenders/engine/clickhouse.rs`) and an analytic-then-empirical decision module (`recommenders/engine/compression.rs`: GCD byte-width pruning, the near-incompressible shortcut, level / LZ4-vs-ZSTD choice) used for ClickHouse and for Arrow IPC; one-shot runs it on the recommended array, streaming on the reservoir blocks. A Python `to_chbench` helper writes `<col>.arrow` + `<col>.yaml` + `<col>.predicted.json`. `clickhouse_benchmark` gains a `dataset: {path}` source and, from a live-server spike, the types and the GCD codec in its vocabulary.

**Tech Stack:** Rust (arrow-rs 60, polars 0.51, `zstd`, new `lz4` = liblz4), Python 3.12 (pyarrow, Polars, PyYAML), Java 25 FFM, ClickHouse 26.8 in Docker, `clickhouse_connect`.

**Spec:** `docs/superpowers/specs/2026-10-19-clickhouse-recommendations-design.md` — every task's implementer reads it (and the sections it names) first.

---

## Conventions

**turbo-parakeet** (`c:/Users/Alexander/turbo-parakeet`, branch `clickhouse-recommendations` from Task 0):
```bash
E=/c/Users/Alexander/miniconda3/envs/p312
export PATH="$E:$E/Scripts:$E/Library/bin:/c/Users/Alexander/.cargo/bin:$PATH" PYO3_PYTHON="$E/python.exe"
PY=$E/python.exe
```
- Rust tests `cd services/analytics && cargo test --lib <filter>`; extension `$PY scripts/check.py build` (≈ 3 min — background or timeout 600000); pytest `$PY -m pytest <files> -q`; Java `$PY scripts/check.py test-java`.
- Before each commit: `$PY scripts/check.py fmt && $PY scripts/check.py lint`; revert files `fmt` touched that the task did not (e.g. `notebooks/Examples.ipynb`).
- Commit without `--no-verify`, env on PATH; message ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage by explicit path only. `docs/superpowers/` is git-ignored but tracked: `git add -f`.
- Layering: `common` ← `techniques` ← `recommenders` ← `bindings`. Clippy runs with `-D warnings`: items unused until a later task get `#[cfg_attr(not(test), expect(dead_code, reason = "..."))]`, removed when used.

**clickhouse_benchmark** (`c:/Users/Alexander/clickhouse_benchmark`, branch `real-data-and-vocabulary` from Task 11): identity `benssutton <benssutton@hotmail.com>` (repo-local, set). Read its `CLAUDE.md`, `docs/superpowers/specs/2026-09-13-clickhouse-benchmark-framework-design.md` (§4, §5) and `docs/superpowers/notes/2026-09-14-deferred-findings.md` before any task there. Its conventions: TDD; `pytest` with the `clickhouse` marker for server tests and `slow` for long ones; spikes in `spikes/` with findings in `spikes/FINDINGS.md`; rulings recorded in its spec; record only what was measured. Same Python env (`$PY`); commits end with the same Co-Authored-By line.

**ClickHouse server** (Tasks 11–14): Docker container `chbench-ch` from `clickhouse/clickhouse-server:26.8`, HTTP 8123, user `CLICKHOUSE_USER` / password `CLICKHOUSE_PASSWORD` (chbench's defaults), env `CH_HOST=localhost CH_PORT=8123`.

## File map

| Repo / file | Change |
|---|---|
| tp `services/analytics/Cargo.toml` | dependency `lz4` |
| tp `src/common/ipc_sizes.rs` | `Codec { None, Zstd(i32), Lz4Frame }`; `ipc_body(arr, codec) -> Body`(bytes + byte histogram) |
| tp `src/common/clickhouse_sizes.rs` (new) | `ChLayout`, `streams`, `Chain`, `emulate` (blocks, GCD, LZ4/ZSTD, headers, histogram) |
| tp `src/recommenders/engine/clickhouse.rs` (new) | Arrow → ClickHouse type + `ChLayout`; the physical GCD / magnitude of the recommended values |
| tp `src/recommenders/engine/compression.rs` (new) | `bytes_needed`, GCD pruning, the shortcut, level and LZ4/ZSTD choice, candidates |
| tp `src/recommenders/engine/candidates.rs` | `Params` gains the four keywords |
| tp `src/recommenders/schema.rs` | new DataFusion / Polars / ClickHouse columns |
| tp `src/recommenders/engine/column.rs`, `oneshot.rs` | one-shot cells |
| tp `src/recommenders/streaming/mod.rs` | streaming cells (per sampled block) |
| tp `src/bindings/{api/recommenders.rs,python.rs,capi.rs}`, Java params | keywords |
| tp `analytics/_plugin.py`, `analytics/recommend/{oneshot,streaming,_input}.py`, `analytics/recommend/chbench.py` (new) | keywords; `to_chbench` |
| tp `tests/test_clickhouse_recommend.py` (new), `tests/test_recommend.py`, `tests/test_streaming_recommend.py` | tests |
| tp `tests/performance/calibrate_clickhouse_sizes.py` (new) | calibration |
| ch `chbench/datagen/spec.py`, `cache.py`, `chbench/harness/suite.py` | `path` / `column` dataset source |
| ch `spikes/spike_ch_vocabulary.py` (new), `spikes/FINDINGS.md`, `chbench/harness/validate.py` | vocabulary |
| both | CLAUDE.md, ToDo.md, specs |

---

### Task 0: Branch and spec amendments

- [ ] `git switch -c clickhouse-recommendations` (turbo-parakeet, from `main`).
- [ ] Amend the spec (`docs/superpowers/specs/2026-10-19-clickhouse-recommendations-design.md`):
  - §4.1 **g** bullet, append: "In streaming, a numeric type derived from text (a numeric-string column) has no running GCD: its GCD candidate is not offered (`clickhouse_candidates` says why); one-shot computes it. Parity tests exclude that case."
  - §7, after the `<name>.arrow` bullet: "Types `chbench` cannot parse are normalised losslessly before writing: string_view / large_string → string, binary_view / large_binary → binary, Polars Categorical → its strings (dictionary dropped; the recommended `LowCardinality` is a type-axis value, not the stored type)."
  - §9.1, append: "Implementation: a path dataset is a `DatasetSpec` whose fields are **measured from the file** (`rows`, `arrow_type`, `null_count`, `decimal_places`, `value_range`), plus `path` and `column`; `load_or_generate` loads the file instead of generating when `path` is set, and the manifest holds the measured `cardinality_ratio` / `entropy_bits`, no targets and no warnings. The harness, which reads `DatasetSpec`, is otherwise unchanged."
- [ ] Commit `"Spec: ClickHouse recommendations — streaming text GCD, chbench type normalisation, path datasets"` (`git add -f` the spec).

---

### Task 1: LZ4_FRAME Arrow IPC sizes (exact against pyarrow)

Spec §4.2 (last bullet), §5.

**Files:** `services/analytics/Cargo.toml`, `src/common/ipc_sizes.rs`, `tests/test_clickhouse_recommend.py` (new).

- [ ] **Step 1: dependency** — `lz4 = "1.28"` under `[dependencies]` with a comment `#ClickHouse / Arrow IPC LZ4 sizes: liblz4 (the library ClickHouse and pyarrow use)`. `cargo build --lib` must succeed (liblz4 builds with the C toolchain zstd already uses).
- [ ] **Step 2: failing Python oracle test** — create `tests/test_clickhouse_recommend.py`:
```python
"""ClickHouse recommendations, Arrow IPC compression choice and the chbench handoff (spec 2026-10-19)."""

import io

import polars as pl
import pyarrow as pa
import pytest

from analytics import _plugin
from datagen import describe_mixed, large_dataset_mixed


def pyarrow_body(arr: pa.Array, compression: str | None) -> int:
    """Body bytes of a one-column IPC stream, as the recommenders measure them."""
    batch = pa.record_batch([arr], names=["x"])
    sink = io.BytesIO()
    opts = pa.ipc.IpcWriteOptions(compression=compression)
    with pa.ipc.new_stream(sink, batch.schema, options=opts) as w:
        w.write_batch(batch)
    msg = None
    stream = pa.ipc.MessageReader.open_stream(sink.getvalue())
    for m in stream:
        if m.type == "record batch":
            msg = m
    return msg.body.size


@pytest.mark.parametrize("make", [lambda: describe_mixed(2_000), large_dataset_mixed])
def test_lz4_frame_body_sizes_equal_pyarrow(make):
    frame = make()
    for name in frame.columns:
        arr = frame[name].to_arrow(compat_level=pl.CompatLevel.oldest())
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if pa.types.is_dictionary(arr.type) or pa.types.is_null(arr.type):
            continue  # dictionary batches are separate messages; the Rust test covers them
        want = pyarrow_body(arr, "lz4")
        got = _plugin.ipc_body_sizes(pa.table({"x": arr}), "lz4")["x"]
        assert got == want, name
```
  (Adapt `pyarrow_body` if the message API differs — the oracle is pyarrow's own record-batch body length with `compression="lz4"`, which is LZ4_FRAME. The existing ZSTD oracle lives in `analytics/describe/_sizes.py` (`ipc_body_bytes`); mirror it, and if it already has a codec parameter, extend that instead of writing a new helper.) The test needs a private binding `_plugin.ipc_body_sizes(table, codec)` → dict column → bytes, codec ∈ {"none", "zstd:<level>", "lz4"}: add it in Step 4 (`bindings/python.rs` + `_plugin.py`, private like `column_sizes`). Run → fails (no binding).
- [ ] **Step 3: Rust** — in `ipc_sizes.rs`, replace `level: Option<i32>` in `Body` by `codec: Codec`:
```rust
/// An Arrow IPC buffer compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    None,
    Zstd(i32),
    /// LZ4_FRAME: per non-empty buffer an 8-byte uncompressed length + one LZ4 frame, as
    /// pyarrow writes it.
    Lz4Frame,
}
```
  and add a byte histogram of the uncompressed buffer bytes (needed by the shortcut, spec §4.2): `Body { codec, bytes: u64, histogram: [u64; 256] }`; in `buffer`, `for &b in data { self.histogram[b as usize] += 1 }` before compressing. The LZ4 frame: `lz4::EncoderBuilder::new()` with the frame options pyarrow's Arrow C++ `Lz4FrameCodec` uses (check pyarrow's output bytes: block size, block independence, content checksum / content size flags) — the test in Step 2 is the oracle; match it byte-count-exact. New API:
```rust
/// Body bytes and the uncompressed bytes' histogram of `arr` under `codec`.
pub(crate) fn ipc_body(arr: &dyn Array, codec: Codec) -> PolarsResult<(u64, [u64; 256])>
```
  Keep `ipc_body_bytes(arr, level: Option<i32>)` as a thin wrapper (`None` → `Codec::None`, `Some(l)` → `Codec::Zstd(l)`) so existing callers do not change. Unit tests: an all-zero buffer and a random buffer under LZ4 are smaller / about equal to raw; the histogram of a 3-element Int8 array `[0, 1, 1]` counts 1×0, 2×1 (+ zero padding not counted).
- [ ] **Step 4: binding** — `bindings/api` + `python.rs`: `ipc_body_sizes(data, codec: &str)` returning one row per column (`column`, `bytes`); `_plugin.ipc_body_sizes(table, codec) -> dict[str, int]`. Private: used only by tests and the calibration script.
- [ ] **Step 5:** build; the test passes; `cargo test --lib ipc_sizes` passes. Commit `"IPC sizes: LZ4_FRAME bodies (liblz4), byte histograms"`.

---

### Task 2: Arrow → ClickHouse type mapping

Spec §2.

**Files:** create `src/common/clickhouse_sizes.rs` (only the `ChLayout` enum in this task) and `src/recommenders/engine/clickhouse.rs`; register both (`common/mod.rs`, `engine/mod.rs`).

- [ ] **Step 1: the physical layout enum** (`common/clickhouse_sizes.rs`):
```rust
//! ClickHouse on-disk size emulation (spec 2026-10-19 §3).

/// How a ClickHouse column is laid out on disk (MergeTree Wide parts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChLayout {
    /// Little-endian fixed-width values; `signed` for the GCD codec's magnitude.
    Fixed { width: usize, signed: bool, gcd_applies: bool },
    /// Per row a varint length and the bytes.
    String,
    /// A dictionary stream (String layout, the distinct values) and a keys stream.
    LowCardinality,
    FixedString(usize),
    /// A UInt8 null-map stream plus the inner type with defaults in null slots.
    Nullable(Box<ChLayout>),
    /// A UInt64 offsets stream plus the inner streams.
    Array(Box<ChLayout>),
    Tuple(Vec<ChLayout>),
}
```
- [ ] **Step 2: the mapping** (`engine/clickhouse.rs`):
```rust
/// The ClickHouse type for a recommended Arrow type (spec 2026-10-19 §2), with its layout;
/// Err(reason) when there is none. `range`: the physical min / max (Date32 days, timestamp
/// values in their unit) when known; `nullable`: the recommended array holds nulls;
/// `enum_values`: a kept Polars Enum's categories.
pub(crate) fn clickhouse_type(
    t: &AT,
    range: Option<(i128, i128)>,
    nullable: bool,
    enum_values: Option<&[String]>,
) -> Result<(String, ChLayout), String>
```
  covering every row of spec §2 (`Date` when the days lie in 0 … 49 710, i.e. 1970-01-01 … 2149-06-06; `DateTime('tz')` when unit is seconds and 0 ≤ values ≤ 4 294 967 295; decimal precision → width 4 / 8 / 16 bytes; `Bool` / `Enum8` width 1, `Enum16` 2; `Time64(p)` behind a `TIME64_SUPPORTED` const set to `false` until Task 13 confirms the server accepts it (then flip it and its test); `Nullable` wrapping rules: never around `Array` / `Tuple` / `LowCardinality(String)` itself — `LowCardinality(Nullable(String))` — and inside `Array` around the item when the item holds nulls). `gcd_applies` is true for integer, decimal, `Date`, `Date32`, `DateTime`, `DateTime64` layouts only.
- [ ] **Step 3: tests** — one `#[test]` per row of spec §2's table (type string and layout), plus: `Date` at days 49 710 vs 49 711 (→ `Date32`); `DateTime` at 4 294 967 295 vs 4 294 967 296 (→ `DateTime64(0)`); `timestamp[ms, UTC]` → `DateTime64(3, 'UTC')`; `decimal(9,2)` / `decimal(10,2)` / `decimal(19,2)` → `Decimal32(2)` / `Decimal64(2)` / `Decimal128(2)`; nullable int32 → `Nullable(Int32)`; nullable dictionary → `LowCardinality(Nullable(String))`; list<int64 with nulls> → `Array(Nullable(Int64))`; Enum with 127 / 128 categories → `Enum8(...)` / `Enum16(...)` (quote and escape `'` in names); `timestamp_with_offset` and `null` → Err. `cargo test --lib clickhouse` → pass. Commit `"ClickHouse type mapping (spec §2)"`.

---

### Task 3: Emulator — streams

Spec §3 (Streams).

**Files:** `src/common/clickhouse_sizes.rs`.

- [ ] **Step 1: the stream type and builder**:
```rust
/// One on-disk stream of a column: its bytes and where each granule-able row ends.
pub(crate) struct Stream {
    pub bytes: Vec<u8>,
    /// Cumulative byte end of each row (rows of this stream: values for value streams,
    /// list items for an array's inner streams).
    pub row_ends: Vec<u64>,
    /// Value width for the GCD codec, when it applies to this stream.
    pub gcd_width: Option<(usize, bool)>,
    /// Rows of the column's top level per stream row group, for granule cutting: the
    /// top-level row index at which each stream row starts (identity for top-level streams).
    pub top_rows: Vec<u64>,
}

/// The column's streams in ClickHouse's layout: `a` is the recommended array (classic Arrow
/// layout), `layout` its `ChLayout`.
pub(crate) fn streams(a: &dyn Array, layout: &ChLayout) -> Result<Vec<Stream>, String>
```
  Layouts (byte-exact; little-endian): `Fixed{width}` — each value at `width` bytes: integers / `Date` (days as u16) / `Date32` (i32 days) / `DateTime` (u32 s) / `DateTime64` and `Decimal*` (their physical i32/i64/i128 values at the layout's width), `Bool` 0/1, `Enum8/16` code (1-based index in the categories), floats as IEEE; null slots → 0 bytes of the width; `String` — LEB128 varint length + bytes per row (null → length 0); `FixedString(n)` — n bytes per row, null → n zero bytes; `LowCardinality` — the dictionary stream holds every distinct value once in first-seen order (String layout) and the keys stream holds one key per row, width 1 if dictionary ≤ 256, 2 if ≤ 65 536, else 4 (key 0 reserved for null when nullable — keys shift by one); `Nullable(T)` — a UInt8 null map (1 = null) stream then T's streams; `Array(T)` — a UInt64 stream of cumulative end offsets per row, then T's streams over the flattened items; `Tuple` — each field's streams in order.
- [ ] **Step 2: tests** — hand-built arrays with their exact expected bytes: Int32 `[1, -1, null]` nullable → null map `[0,0,1]`, values `01 00 00 00 FF FF FF FF 00 00 00 00`; String `["a", "", null]` → `01 61 00 00`; a 300-byte string → varint `AC 02`; FixedString(2) `["ab", null]` → `61 62 00 00`; LowCardinality of `["x","y","x"]` → dictionary `01 78 01 79`, keys `00 01 00`; Array(Int8) `[[1,2],[],[3]]` → offsets `02…, 02…, 03…` (u64), values `01 02 03`; Date `1970-01-02` → `01 00`; Decimal64 `1.50` (scale 2) → `96 00 00 00 00 00 00 00`; DateTime64(3) of 1 000 ms → `E8 03 00 00 00 00 00 00`. `cargo test --lib clickhouse_sizes` → pass. Commit `"ClickHouse emulator: column streams"`.

---

### Task 4: Emulator — blocks, codecs, sizes, histogram

Spec §3 (Blocks, Codecs, Byte entropy).

**Files:** `src/common/clickhouse_sizes.rs`.

- [ ] **Step 1: API**:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compression {
    Lz4,
    Zstd(i32),
}

/// A codec chain: optional GCD, then a compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Chain {
    pub gcd: bool,
    pub compression: Compression,
}

impl Chain {
    /// As written in DDL: "LZ4", "ZSTD(3)", "GCD, LZ4", "GCD, ZSTD(3)".
    pub(crate) fn ddl(&self) -> String
}

/// Emulated size of the streams under `chain`, and the histogram of the uncompressed bytes
/// the compressor sees (after GCD).
pub(crate) struct Emulated {
    pub bytes: u64,
    pub histogram: [u64; 256],
}

pub(crate) const GRANULE_ROWS: u64 = 8_192;
pub(crate) const MIN_BLOCK: u64 = 65_536;
pub(crate) const MAX_BLOCK: u64 = 1_048_576;
pub(crate) const BLOCK_HEADER: u64 = 25; // 16-byte checksum + 9-byte header

pub(crate) fn emulate(streams: &[Stream], chain: Chain) -> Result<Emulated, String>
```
- [ ] **Step 2: blocks** — per stream: walk top-level granules (rows `[k·8192, (k+1)·8192)` mapped to stream byte ranges through `top_rows` / `row_ends`); append each granule's bytes to the open block; close the block when it reaches ≥ `MIN_BLOCK`; a granule whose bytes exceed `MAX_BLOCK` is split into `MAX_BLOCK` pieces; close the last block at the end. Each block costs `BLOCK_HEADER + payload`.
- [ ] **Step 3: codecs** — GCD (only when `chain.gcd` and the stream has `gcd_width`): per block, the GCD of the block's values' magnitudes (as `i128`, signed per the width; zero values skipped; GCD 0 or 1 → values unchanged); the block becomes `[gcd as width bytes] + values / gcd` at the same width. Then `Lz4` → `lz4::block::compress(bytes, Some(CompressionMode::DEFAULT), false)` length; `Zstd(l)` → `zstd::bulk::compress(bytes, l)` length. The histogram counts the post-GCD uncompressed bytes of every block.
- [ ] **Step 4: tests** — block cutting: a UInt64 column of 20 000 rows (160 000 bytes, granules of 65 536 bytes) → 3 blocks (8 192, 8 192, 3 616 rows); a UInt8 column of 100 000 rows → granules of 8 192 bytes merge until ≥ 65 536 → blocks of 8 granules; a String column with one 2 MiB value → that granule split into 2 + blocks; GCD: a block of `[1000, 2000, 3000]` (Int64) emulates the same bytes as `[1, 2, 3]` plus 8 header bytes before compression (assert via histogram counts and by emulating with LZ4 on both and comparing payload lengths within ±8); negatives keep sign (`[-6, 9]` → gcd 3 → `[-2, 3]`); the histogram of `[0u8; 10]` UInt8 under LZ4 is `[10, 0, …]`; `Chain::ddl` strings. `cargo test --lib clickhouse_sizes` → pass. Commit `"ClickHouse emulator: blocks, GCD, LZ4 / ZSTD, histogram"`.

---

### Task 5: The decision module

Spec §4.

**Files:** create `src/recommenders/engine/compression.rs`; register in `engine/mod.rs`.

- [ ] **Step 1: pure functions with full code**:
```rust
//! Analytic pruning, then emulation: GCD and compression choices (spec 2026-10-19 §4).

use crate::common::clickhouse_sizes::{Chain, Compression};

/// The four keywords (spec §4.3).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompressionParams {
    pub zstd_levels: Vec<i32>,
    pub zstd_level_tolerance: f64,
    pub zstd_min_saving: f64,
    pub gcd_min_saving: f64,
}

impl Default for CompressionParams {
    fn default() -> Self {
        CompressionParams {
            zstd_levels: vec![1, 3, 9],
            zstd_level_tolerance: 0.01,
            zstd_min_saving: 0.10,
            gcd_min_saving: 0.01,
        }
    }
}

/// Bytes holding the magnitude `m` (plus a sign bit when `signed`); 0 → 1.
pub(crate) fn bytes_needed(m: u128, signed: bool) -> u32 {
    let bits = 128 - m.leading_zeros() + signed as u32;
    bits.div_ceil(8).max(1)
}

/// Spec §4.1: GCD is a candidate only if dividing by `g` frees a byte at the maximum
/// magnitude. Returns (survives, evidence).
pub(crate) fn gcd_survives(g: u128, max_abs: u128, signed: bool) -> (bool, String) {
    if g <= 1 {
        return (false, format!("gcd {g}"));
    }
    let (before, after) = (bytes_needed(max_abs, signed), bytes_needed(max_abs / g, signed));
    (
        after < before,
        format!("gcd {g}: {before} → {after} bytes at max |value| {max_abs}"),
    )
}

/// Shannon entropy in bits per byte of a byte histogram.
pub(crate) fn byte_entropy(h: &[u64; 256]) -> f64 {
    let n: u64 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    h.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.log2()
        })
        .sum()
}

/// Spec §4.2's shortcut: ZSTD cannot save `min_saving` over LZ4. Returns (fires, evidence).
pub(crate) fn incompressible(h: &[u64; 256], distinct_ratio: f64, min_saving: f64) -> (bool, String) {
    let hb = byte_entropy(h);
    let bound = 1.0 - hb / 8.0;
    (
        bound < min_saving && distinct_ratio >= 0.5,
        format!("byte entropy {hb:.3} bits (ZSTD gain ≤ {:.1}%), distinct ratio {distinct_ratio:.3}", bound * 100.0),
    )
}

/// Spec §4.2: the lowest level within tolerance of the best ZSTD size, then ZSTD over LZ4
/// only if it saves `min_saving`. `zstd`: (level, bytes), any order, may be empty.
pub(crate) fn choose_compression(lz4: u64, zstd: &[(i32, u64)], p: &CompressionParams) -> (Compression, u64) {
    let Some(best) = zstd.iter().map(|&(_, b)| b).min() else {
        return (Compression::Lz4, lz4);
    };
    let limit = best as f64 * (1.0 + p.zstd_level_tolerance);
    let (level, bytes) = zstd
        .iter()
        .filter(|&&(_, b)| b as f64 <= limit)
        .min_by_key(|&&(l, _)| l)
        .copied()
        .expect("the best level is within its own tolerance");
    if bytes as f64 <= (1.0 - p.zstd_min_saving) * lz4 as f64 {
        (Compression::Zstd(level), bytes)
    } else {
        (Compression::Lz4, lz4)
    }
}
```
- [ ] **Step 2: the generic decision** — one function used for ClickHouse (with GCD) and Arrow IPC (no GCD):
```rust
/// One candidate row of `clickhouse_candidates` (and the Arrow IPC evidence).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CodecCandidate {
    pub codecs: String,
    pub predicted_bytes: Option<u64>,
    pub outcome: &'static str, // "chosen" | "rejected" | "pruned"
    pub evidence: String,
}

pub(crate) struct Decision {
    pub chain: Chain,
    pub bytes: u64,
    pub candidates: Vec<CodecCandidate>,
}

/// Spec §4.1–4.2. `measure(chain)` emulates one chain (bytes + histogram); `gcd` is
/// Some((g, max_abs, signed)) when GCD applies to the column, None otherwise or unknown
/// (`gcd_note` then says why, or "" when GCD does not apply).
pub(crate) fn decide(
    mut measure: impl FnMut(Chain) -> Result<(u64, [u64; 256]), String>,
    gcd: Option<(u128, u128, bool)>,
    gcd_note: &str,
    distinct_ratio: f64,
    p: &CompressionParams,
) -> Result<Decision, String>
```
  Behaviour: for `gcd_flag` in [false] + [true if `gcd_survives`]: measure `Chain{gcd_flag, Lz4}`; if `incompressible(histogram, distinct_ratio, p.zstd_min_saving)` fires → LZ4 for this chain, every `ZSTD(l)` candidate recorded `pruned` with that evidence; else measure each `Zstd(l)` and `choose_compression`. Then pick between the GCD and no-GCD chains by `gcd_min_saving`. Candidate rows: every measured or pruned chain (`codecs` = `Chain::ddl`), outcome `chosen` for the winner, `rejected` for measured losers (evidence: e.g. "ZSTD(9) 1 210 vs ZSTD(3) 1 205: within 1%", "ZSTD 980 vs LZ4 1 020: saves < 10%"), `pruned` for the analytic cuts (a pruned GCD row: `codecs` = "GCD", evidence from `gcd_survives`, or `gcd_note`).
- [ ] **Step 3: tests** — `bytes_needed`: (0,false)→1, (255,false)→1, (256,false)→2, (127,true)→1, (128,true)→2, (u64::MAX as u128,false)→8; `gcd_survives`: g = 1 → false; g = 2 with max 2³² (5 bytes → 4) → true; g = 2 with max 2³²−2 (4 → 4) → false; g = 3, max 3·2⁴⁰ (6 → 6) → false; g = 4, max 1 020 (2 → 1) → true; g = 256, max 65 536 (3 → 2) → true; signed g = 2, max 128 (2 → 1) → true; `byte_entropy`: uniform 256 bins → 8.0, one bin → 0.0; `incompressible`: uniform histogram + ratio 0.9 → fires, + ratio 0.3 → not, zero-heavy histogram → not; `choose_compression`: levels {1: 1000, 3: 995, 9: 990} with tolerance 1% → the limit is 990 × 1.01 = 999.9, level 1 (1 000) is above it and level 3 (995) is the lowest within → `Zstd(3)` with 995 when LZ4 is ≥ 1 106 (995 ≤ 0.9 × LZ4); LZ4 1 050 vs best ZSTD 1 000 → LZ4 (saves < 10%); LZ4 2 000 → ZSTD; `decide` with a fake `measure` (a closure returning canned sizes / histograms) covering: GCD pruned, GCD survives but saves < 1% → no GCD chosen, GCD saves 5% → chosen, shortcut fires → ZSTD rows pruned, no ZSTD levels → LZ4. `cargo test --lib compression` → pass. Commit `"Decisions: GCD byte-width pruning, near-incompressible shortcut, compression choice"`.

---

### Task 6: Keywords, Params and the new columns (null for now)

Spec §4.3, §5, §8 (Python side; C / Java in Task 9).

**Files:** `engine/candidates.rs` (`Params` gains `compression: CompressionParams`; fix every `Params { .. }` literal with `CompressionParams::default()`), `schema.rs`, `engine/column.rs` (`Rec` gains the new fields, `rec_row` emits them), `streaming/mod.rs`, `bindings/api/recommenders.rs` (both params structs gain `zstd_levels: Vec<i32>, zstd_level_tolerance: f64, zstd_min_saving: f64, gcd_min_saving: f64`; validation: levels non-empty, distinct, each in `zstd::compression_level_range()`; fractions in [0, 1) — InvalidInput naming the keyword), `bindings/python.rs` (signatures with defaults), `capi.rs` (pass `CompressionParams::default()` values for now), `analytics/_plugin.py`, `analytics/recommend/{oneshot,streaming}.py` (keywords with the spec's defaults, docstrings), tests.

- [ ] **Step 1: columns** — in `rec_fields()` (schema.rs) insert, keeping the skeleton order:
  after `datafusion_size_zstd_bytes`: `("datafusion_size_lz4_bytes", UInt64), ("datafusion_ipc_compression", String), ("datafusion_ipc_zstd_level", Int32), ("datafusion_ipc_size_bytes", UInt64)`;
  after `polars_size_zstd_bytes`: the same four with `polars_`;
  then (before `candidates`): `("clickhouse_type", String), ("clickhouse_codecs", String), ("clickhouse_size_bytes", UInt64), ("clickhouse_candidates", List(Struct{codecs: String, predicted_bytes: UInt64, outcome: String, evidence: String}))`.
  `Rec` gains `arrow_lz4: Option<u64>, arrow_ipc: Option<(Compression, u64)>, polars_lz4: Option<u64>, polars_ipc: Option<(Compression, u64)>, ch_type: Option<String>, ch_chain: Option<Chain>, ch_size: Option<u64>, ch_candidates: Vec<CodecCandidate>` (all None / empty for now); `rec_row` emits them in order (`Compression::Lz4` → "LZ4_FRAME" and level null; `Zstd(l)` → "ZSTD" and `l`).
- [ ] **Step 2: tests** — update the `SCHEMA` dict in `tests/test_streaming_recommend.py` (and any ordered list in `tests/test_recommend.py`) with the new columns at those positions; add to `tests/test_clickhouse_recommend.py` a keyword-validation test for both recommenders: `zstd_levels=()` / `(1, 1)` / `(99,)`, `zstd_min_saving=1.0`, `gcd_min_saving=-0.1` → `ValueError` naming the keyword.
- [ ] **Step 3:** cargo tests, build, `pytest tests/test_recommend.py tests/test_streaming_recommend.py tests/test_clickhouse_recommend.py tests/test_recommender_profile.py -q`, Java tests → pass. Commit `"Recommenders: compression keywords and ClickHouse / IPC columns (null for now)"`.

---

### Task 7: One-shot cells

Spec §2, §3, §4, §5.

**Files:** `engine/clickhouse.rs` (physical GCD / magnitude helper), `engine/column.rs` (`recommend`), tests.

- [ ] **Step 1: physical values** — `pub(crate) fn physical_gcd_and_max(a: &ArrayRef) -> Option<(u128, u128, bool)>`: for an integer / decimal / Date32 / timestamp array (the recommended one), the GCD of the non-null physical values' magnitudes (`gcd` crate or `crate::techniques::gcd` helpers — reuse what `techniques/gcd.rs` exposes), the max magnitude and whether the type is signed; None for other types or no non-null values. For a `Date` layout the physical value is the day number; for `DateTime` the seconds.
- [ ] **Step 2: `recommend`** — after the recommended `chosen.array` and the Polars `layout` are known:
  - Arrow IPC: `decide` with `measure = |c| ipc_body(chosen.array, codec_of(c))` (`Chain{gcd: false, Lz4}` → `Codec::Lz4Frame`, `Zstd(l)` → `Codec::Zstd(l)`), `gcd` None, distinct ratio `n_unique / (n_rows − n_null)` from Describe (1.0 when no non-null); `arrow_lz4` = the LZ4 measurement; `arrow_ipc` = (chosen compression, bytes). The same for the Polars layout.
  - ClickHouse: `clickhouse_type(t, range, nullable, enum_values)`; Ok → `streams(chosen.array, &layout)` once, `decide` with `measure = |c| emulate(&streams, c)`, `gcd` = `physical_gcd_and_max` when the layout has `gcd_applies`; Err(reason) → `ch_type` None and one candidate `{codecs: "", predicted_bytes: None, outcome: "pruned", evidence: reason}`.
- [ ] **Step 3: Python tests** in `tests/test_clickhouse_recommend.py`:
```python
def one_shot(frame: pl.DataFrame, **kw) -> dict[str, dict]:
    from analytics.recommend import OneShotRecommender

    out = OneShotRecommender(**kw).add(frame).result()
    return {r["column"]: r for r in out.iter_rows(named=True)}


def test_clickhouse_types_and_codecs():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200_000
    frame = pl.DataFrame(
        {
            "small": rng.integers(0, 200, n).astype(np.int64),            # → UInt8
            "thousands": (rng.integers(1, 10**9, n) * 1_000).astype(np.int64),  # GCD 1000: 6 → 5 bytes at max
            "threes": (rng.integers(1, 2**38, n) * 3).astype(np.int64),   # GCD 3: no byte freed → pruned
            "noise": rng.random(n),                                       # near-incompressible
            "text": rng.choice(["alpha", "beta", "gamma"], n),            # → LowCardinality(String)
            "maybe": pl.Series([None, 1, 2] * (n // 3) + [1] * (n % 3), dtype=pl.Int64),
        }
    )
    rows = one_shot(frame)
    assert rows["small"]["clickhouse_type"] == "UInt8"
    assert rows["text"]["clickhouse_type"] == "LowCardinality(String)"
    assert rows["maybe"]["clickhouse_type"] == "Nullable(UInt8)"
    assert rows["thousands"]["clickhouse_codecs"].startswith("GCD")
    gcd_rows = [c for c in rows["threes"]["clickhouse_candidates"] if c["codecs"] == "GCD"]
    assert gcd_rows and gcd_rows[0]["outcome"] == "pruned"
    assert rows["noise"]["clickhouse_codecs"] == "LZ4"
    assert any(c["outcome"] == "pruned" and c["codecs"].startswith("ZSTD")
               for c in rows["noise"]["clickhouse_candidates"])
    for r in rows.values():
        assert r["clickhouse_size_bytes"] > 0
        assert r["datafusion_ipc_compression"] in ("LZ4_FRAME", "ZSTD")
```
  Exactness of the LZ4 size: extend the existing oracle `test_sizes_match_pyarrow_and_polars_casts` in `tests/test_recommend.py` (it already rebuilds each recommended array with pyarrow / Polars casts and compares `datafusion_size_zstd_bytes`): wherever it compares the ZSTD size, also assert `datafusion_size_lz4_bytes == pyarrow_body(recommended_array, "lz4")` (import `pyarrow_body` from `test_clickhouse_recommend`), and that `datafusion_ipc_size_bytes` equals the LZ4 size when `datafusion_ipc_compression == "LZ4_FRAME"`. Plus keyword effects: `zstd_min_saving=0.0` on `small` (no shortcut: low-entropy bytes) makes ZSTD win; `gcd_min_saving=0.99` makes `thousands` drop GCD; `zstd_levels=(3,)` → ZSTD level 3 whenever ZSTD is chosen.
- [ ] **Step 4:** build, run the ClickHouse, recommend and profile tests → pass. Commit `"One-shot recommender: ClickHouse type, codec chain and IPC compression"`.

---

### Task 8: Streaming cells

Spec §6.

**Files:** `streaming/mod.rs` (`Streaming::row`), `tests/test_streaming_recommend.py`, `tests/test_clickhouse_recommend.py`.

- [ ] **Step 1:** in the per-sampled-block loop of `Streaming::row` (where `layout_sizes` already accumulates `z[..]`), for the recommended block `a` (and its Polars layout): accumulate per chain the LZ4 / ZSTD-level sizes and histograms — but chains are only known after the shortcut, which needs the histogram: do two passes over the blocks — pass 1 measures LZ4 (Arrow body, Polars body, ClickHouse no-GCD and GCD chains when GCD survived) and collects histograms; then decide which ZSTD levels to measure; pass 2 measures those. Express this as `decide` with a `measure` closure that sums `emulate` / `ipc_body` over the sampled blocks (each call iterates the blocks; `decide` only calls `measure` for the surviving chains). Scale every summed size by `n_rows / sampled` (as `scale` does now). The ClickHouse type uses the running min / max (`o.int_range` mapped to the recommended unit) and the recommended `rec_t`; GCD: the running `o.gcd` divided by the type rule's rescale factor (decimal: 10^(original scale − recommended scale); timestamp: recommended unit / original unit in ns; integer → integer: 1); a text-derived numeric target → GCD not offered, note "GCD unknown for text-derived values in streaming" (spec §4.1 amendment). `reservoir_rows = 0` (no blocks) → sizes, compression and `clickhouse_codecs` null, `clickhouse_type` still set.
- [ ] **Step 2: parity** — in `assert_parity` (tests/test_streaming_recommend.py) add: `clickhouse_type` equal always; when `batch_rows >= frame.height` (one batch, all sampled) and the column is fixed-width: `clickhouse_codecs`, `datafusion_ipc_compression`, `polars_ipc_compression` equal and the sizes equal; otherwise sizes within `rel=0.02` and the codecs equal when both chains' sizes differ by more than 2% (else either may win). Exclude text-derived numeric columns from the GCD comparison.
- [ ] **Step 3:** build; `pytest tests/test_streaming_recommend.py tests/test_clickhouse_recommend.py tests/test_recommender_profile.py -q -m "slow or not slow"` → pass. If parity fails beyond the stated tolerance, report the column, both values and the batch split — do not loosen. Commit `"Streaming recommender: ClickHouse and IPC compression from the reservoir blocks"`.

---

### Task 9: C ABI and Java keywords

Spec §8.

**Files:** `bindings/capi.rs`, Java `OneShotParams`, `StreamingParams`, `OneShotRecommender`, `StreamingRecommender` (+ tests).

- [ ] C ABI: both `_new` gain, before `out`: `const int32_t *zstd_levels, size_t n_zstd_levels, double zstd_level_tolerance, double zstd_min_saving, double gcd_min_saving`; `n_zstd_levels = 0` is InvalidInput ("zstd_levels is empty"), matching the Rust API's validation (there is no implicit default at the C level); update docs and the capi tests (pass `[1, 3, 9]`, 0.01, 0.10, 0.01); add a test that `n_zstd_levels = 0` returns 1.
- [ ] Java: `OneShotParams` / `StreamingParams` gain trailing components `List<Integer> zstdLevels, double zstdLevelTolerance, double zstdMinSaving, double gcdMinSaving` (copied; the existing shorter constructors and `defaults()` pass `List.of(1, 3, 9), 0.01, 0.10, 0.01`); descriptors add `ADDRESS, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE`; `create` passes an `int[]` segment (`arena.allocateFrom(JAVA_INT, ...)`). Tests: a recommender with `zstdLevels = List.of(3)` returns `clickhouse_codecs` that is `LZ4` or ends with `ZSTD(3)`; `clickhouse_type` is read for a small frame.
- [ ] `cargo test --lib capi`, `check.py test-java` → pass. Commit `"C ABI and Java: compression keywords"`.

---

### Task 10: `to_chbench`

Spec §7 (+ Task 0's normalisation amendment).

**Files:** create `services/analytics/analytics/recommend/chbench.py`; export `to_chbench` from `analytics/recommend/__init__.py`; tests in `tests/test_clickhouse_recommend.py`.

- [ ] **Step 1: the helper** — `to_chbench(result: pl.DataFrame, frame, out_dir, *, columns=None) -> list[tuple[str, str]]` (returns skipped `(column, reason)`):
  - `chbench` base type of a ClickHouse type: strip `Nullable(...)` / `LowCardinality(...)` wrappers and parameters (`Decimal64(2)` → `Decimal64`, `DateTime64(3, 'UTC')` → `DateTime64`, `Enum8(...)` → `Enum8`, `FixedString(4)` → `FixedString`); the codec axis value is `LowCardinality` when the type is `LowCardinality(...)`, else `GCD` when `clickhouse_codecs` starts with `GCD`, else `none`; the compression axis value is the chain's last element (`LZ4` or `ZSTD(l)`).
  - The original's base type: the ClickHouse type of the original column's Arrow type (reuse the result's `dtype` — map it with a small Python table consistent with spec §2 for the types `chbench` handles; anything else → skipped with "original type <dtype> has no chbench type").
  - Skipped: status ≠ computed; `clickhouse_type` null; nested (`Array(`, `Tuple(`); Time64 while `chbench` lacks it.
  - Write `<name>.arrow` (the original column, normalised per Task 0: views / large types → plain, Categorical → string; one uncompressed record batch, column named as the original), `<name>.yaml` exactly as spec §7 (YAML via `yaml.safe_dump(..., sort_keys=False)`; `decimal_places` = the recommended decimal scale when a type axis value is a Decimal; the `range` query only for numeric / temporal), and `<name>.predicted.json`: `{"<type>|<codec>|<compression>": predicted_bytes}` for every grid cell — computed by calling the Rust emulator through a private binding `_plugin.clickhouse_size(table, column, clickhouse_type, codecs)` (add it: `bindings/python.rs` + `bindings/api`, returns the emulated bytes for one column under one type + chain; the type string parsed back to a `ChLayout` by a `parse_clickhouse_type` in `engine/clickhouse.rs` — unit-test the round trip on every §2 type).
  - `name` sanitised: lower-case, non `[a-z0-9_]` → `_`, a leading digit prefixed with `c_`, de-duplicated with `_2`, `_3`.
- [ ] **Step 2: tests** — on a small frame (int, decimal-like float, strings, nullable, list): files exist; YAML loads and has exactly the spec's keys and the expected axis values; the skipped list names the list column; `predicted.json` has one key per grid cell; if `chbench` is importable (`pytest.importorskip("chbench")` after inserting `c:/Users/Alexander/clickhouse_benchmark` on `sys.path` when that directory exists), `chbench.harness.suite.load_suite(path)` parses every written suite — this part only runs once Tasks 12–13 have landed in chbench (mark it `skip` with reason until then if it fails only for the not-yet-supported `path` source; Task 14 removes the skip).
- [ ] **Step 3:** build; tests pass. Commit `"to_chbench: real-data suites with per-cell predicted sizes"`.

---

### Task 11: ClickHouse server and chbench baseline (clickhouse_benchmark repo)

- [ ] `cd c:/Users/Alexander/clickhouse_benchmark && git switch -c real-data-and-vocabulary`.
- [ ] Start the server: `docker run -d --name chbench-ch -p 8123:8123 -e CLICKHOUSE_USER=CLICKHOUSE_USER -e CLICKHOUSE_PASSWORD=CLICKHOUSE_PASSWORD clickhouse/clickhouse-server:26.8` (or the image tag that `chbench`'s spec names; record the exact `SELECT version()`); wait until `curl -s localhost:8123/ping` answers `Ok.`.
- [ ] Install chbench's requirements into the p312 env if missing (`$PY -m pip install -r requirements.txt` — it pins `clickhouse-connect`; note the installed version before and after in the report; do not downgrade anything turbo-parakeet's `requirements*.txt` pin).
- [ ] Run chbench's whole test suite: `$PY -m pytest -q` (server up) → record pass / fail counts. Failures caused by this machine (paths in CLAUDE.md pointing at `C:\Users\Ben`, timing) are reported, not "fixed", unless trivially environmental (an env var). Nothing to commit unless an environment fix is needed and agreed.

---

### Task 12: chbench real-data dataset source (TDD, in chbench)

Spec §9.1 (+ Task 0 amendment).

- [ ] Tests first (`tests/datagen/test_path_dataset.py`, `tests/harness/test_suite_path.py`): a suite YAML with `dataset: {path: x.arrow, column: v}` next to a written Arrow file parses; the resulting `DatasetSpec` has `rows`, `arrow_type`, `null_count`, `decimal_places` (from a decimal column's scale, or the suite's explicit value), `value_range` = the file's min / max measured; `dataset_id` changes when the file content changes; `load_or_generate` returns the file's array unchanged (type and values) and a `Manifest` with measured `cardinality_ratio` / `entropy_bits`, empty `targets` / `warnings`; unknown keys with `path` (e.g. `distribution`) are rejected; a missing file / missing column is a `SuiteError` naming it; `_check_dataset_against_type_axis` works from the file's type (a Decimal axis on a float file without `decimal_places` is rejected at parse time).
- [ ] Implement: `DatasetSpec` gains `path: str | None = None`, `column: str | None = None`; `validate()` accepts a path spec with only the measured fields; `suite.py` resolves `path` relative to the suite file and fills the measured fields by reading the file (pyarrow `ipc.open_file`); `cache.load_or_generate` loads the column when `path` is set (no generation, no caching copy). Keep `chbench`'s style (docstrings citing spec sections, rulings).
- [ ] An integration test (`clickhouse` marker): a tiny path suite (1 000 rows, Int32 file, grid type [Int32, UInt32], codec [none], compression [LZ4]) runs end to end through `run_suite` against the container.
- [ ] Record a ruling in chbench's spec (§4 / §5: "R35: real-data datasets") and a line in its CLAUDE.md Status. Commit (in chbench) `"datagen, harness: real-data dataset source (dataset: {path, column})"`.

---

### Task 13: Vocabulary spike (in chbench)

Spec §9.2.

- [ ] Write `spikes/spike_ch_vocabulary.py` in the style of `spikes/spike_ch_roundtrip.py`: for each type in {`UInt8`, `UInt16`, `UInt64`, `Int8`, `Int16`, `Int64`, `Float32`, `Bool`, `Decimal128(4)`, `Date32`, `DateTime`, `DateTime64(0)`, `DateTime64(6)`, `DateTime64(9)`, `FixedString(4)`, `Enum8('a' = 1, 'b' = 2)`, `Enum16(...)` with 200 values, `Time64(3)` (with and without `enable_time_time64_type`)} × nullable {false, true} × codec {none, `GCD`, `Delta`, `DoubleDelta`, `Gorilla`, `T64`, `LowCardinality`} × compression {`LZ4`, `ZSTD(3)`} × skip index {none, minmax, set(100), bloom_filter}: run the DDL and a small insert; record accepted / rejected and the server's message; plus the GCD codec's layout measurement (known inputs → `system.parts_columns` compressed / uncompressed bytes for GCD vs none, to confirm "same width + header").
- [ ] Run it against the container; write `spikes/FINDINGS.md` §9 (matrices, messages, the GCD layout finding); amend chbench's spec §5.2 with the measured rules.
- [ ] Extend `chbench/harness/validate.py` from the findings only: `KNOWN_TYPES`, `KNOWN_CODECS` (`GCD`), `resolve_type` (Decimal128 / DateTime64 precision / FixedString width / Enum categories — taken from the dataset file: add the needed measured fields to the path `DatasetSpec` in Task 12's style if the type needs them), `_STORAGE_WIDTH`, the skip rules and messages. TDD: tests for each new rule with the measured messages.
- [ ] If the server accepts `Time64` (with the setting), report it so turbo-parakeet's `TIME64_SUPPORTED` (Task 2) is flipped in Task 14; if not, leave it false.
- [ ] chbench's full test suite passes with the server up. Commit (in chbench) `"spike + validate: measured vocabulary for the narrow types, Bool, Decimal128, Date32, DateTime, FixedString, Enum and the GCD codec"`.

---

### Task 14: Calibration (turbo-parakeet + chbench)

Spec §10, §11.

- [ ] In turbo-parakeet: remove Task 10's temporary skip; flip `TIME64_SUPPORTED` per Task 13; rerun those tests.
- [ ] Write `tests/performance/calibrate_clickhouse_sizes.py`: build the frames of spec §10 (`datagen.large_dataset_mixed()` plus the four GCD cases at 1 000 000 rows), `OneShotRecommender().add(frame).result()`, `to_chbench(...)` into a temp directory, then for each suite `chbench.cli run <suite> --store <tmp>/results.db --cache-dir <tmp>/cache` (subprocess, chbench's CLI), then `chbench.cli report --suite <name> --csv <tmp>/<name>.csv`; join each CSV's measured `compressed_bytes` per (type, codec, compression) with `<name>.predicted.json`; write `tests/performance/results/clickhouse_calibration.arrow` (column, type, codec, compression, predicted, measured, rel_error) and print the summary by type × codec × compression (max / median relative error).
- [ ] Run it (background; it loads many tables). If any fixed-width type's median error exceeds 10%, stop and report the table — the emulator approximation (spec §3) must be revised in a follow-up task before setting a tolerance (spec §10). Otherwise write spec §11 (findings of Task 13 in one paragraph + the calibration table + the adopted tolerance = the max relative error observed, rounded up to the next percent) and commit `"Calibration: ClickHouse size emulator against chbench (spec §11)"`.

---

### Task 15: Documentation

- [ ] turbo-parakeet `CLAUDE.md`: OneShot / Streaming paragraphs — the ClickHouse block (`clickhouse_type`, `clickhouse_codecs`, `clickhouse_size_bytes`, `clickhouse_candidates`), the IPC compression columns, the four keywords and the analytic pruning (byte-width GCD rule, near-incompressible shortcut), the calibration tolerance; `to_chbench` (Analytical Functions or a new "Handoff" paragraph); Project Structure (`common/clickhouse_sizes.rs`, `engine/{clickhouse,compression}.rs`, `analytics/recommend/chbench.py`, the calibration script); the `lz4` dependency; the C / Java parameters; Current Focus → sub-project B (skip indexes).
- [ ] `ToDo.md`: 1b and 1f **Done** (spec 2026-10-19); 1.0's remainder (type mapping, handoff) done, thresholds/cost model for skip indexes stay with 1c–e; new item: "Consolidate the synthetic data generators (`tests/datagen.py` and `chbench.datagen`) when the two projects merge." Preserve the user's items 15–16.
- [ ] chbench `CLAUDE.md` Status: the path source, the vocabulary spike, the calibration run (date, counts).
- [ ] Commit in each repository.

---

### Task 16: Full verification

- [ ] turbo-parakeet: `$PY scripts/check.py ci` → all checks pass; `$PY -m pytest -m slow -q` → pass.
- [ ] chbench: `$PY -m pytest -q` with the server up → pass (record counts).
- [ ] Check against the spec section by section (§2–§12); report deviations rather than fixing silently. Leave the container running unless the user asks otherwise; report its name.
