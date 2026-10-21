# Delta and DoubleDelta codecs (ToDo 3)

Status: approved 2026-10-23.  One spec, two increments: §2 the technique, §3–§5 the recommender integration.

## 1. Goal

Identify columns that suit ClickHouse's `Delta` and `DoubleDelta` codecs and recommend them in both recommenders.
These are ClickHouse codecs only (Arrow IPC has no delta encoding).  The codec chains considered become: none,
`GCD`, `Delta` or `DoubleDelta`, each followed by the compression choice (LZ4 / ZSTD + level).  Combined chains
(`GCD, Delta`, `GCD, DoubleDelta`) are out of scope, as are Gorilla, T64 and FPC.

## 2. Technique — `turbo_parakeet.techniques.delta`

Order-dependent, per column (CLAUDE.md §1b, beside the runs test).  Implementations `DeltaRust` and ★`DeltaPolars`
(reference: Polars `diff`, exact Python-int arithmetic where Int64 could overflow).  Same contract as every technique:
`DeltaRust().add({"name": frame}).result()`, one row per column, ineligible columns listed.

**Eligible:** integer-backed columns — Int/UInt 8–64, Decimal (unscaled value), Date (days), Datetime / Duration (time
unit), Time (ns).  **Ineligible:** floats (Gorilla's territory, out of scope), Boolean, strings, Categorical / Enum,
List / Array / Struct / Null / Object, Int128 / UInt128, zero-row columns.

Rows are read in frame order.  Nulls are skipped: deltas are taken between consecutive non-null values (d₁ = x₂ − x₁,
…; double-deltas dd₁ = d₂ − d₁, …).  The emulator, not the technique, models ClickHouse's layout (nulls stored as 0).

**Significant bytes** of an integer v: `sig(0) = 0`, else the smallest k ≥ 1 with −2^(8k−1) ≤ v < 2^(8k−1).  (So a
regular series has double-deltas of 0 bytes.)

**Metrics** (no conclusions, no keywords):

| column | type | meaning |
|---|---|---|
| `n_rows`, `n_non_null` | UInt64 | |
| `n_deltas` | UInt64 | n_non_null − 1 (0 if fewer than two values) |
| `delta_min`, `delta_max` | Decimal(38, 0) | range of deltas (an Int64 delta can overflow Int64); null if no delta |
| `double_delta_min`, `double_delta_max` | Decimal(38, 0) | range of double-deltas; null if none |
| `value_bytes_mean` | Float64 | mean `sig` of the non-null values |
| `delta_bytes_mean`, `double_delta_bytes_mean` | Float64 | mean `sig` of deltas / double-deltas; NaN if none |
| `zero_delta_fraction`, `zero_double_delta_fraction` | Float64 | share of zeros; NaN if none |
| `delta_entropy`, `double_delta_entropy` | Float64 | exact Shannon entropy (bits) of the delta / double-delta values; NaN if none |
| `expected_delta_bytes_mean` | Float64 | `delta_bytes_mean` under a uniformly random order of the non-null values |

**Random-order baseline.**  Under a random permutation each delta is x_j − x_i for a uniformly random ordered pair of
distinct positions, so E[sig(Δ)] = Σ_{k=1..9} P(sig(Δ) ≥ k), where sig(Δ) ≥ 1 iff Δ ≠ 0 and, for k ≥ 2,
sig(Δ) ≥ k iff Δ lies outside [−2^(8(k−1)−1), 2^(8(k−1)−1)).  Each probability is a count of ordered pairs over
n(n − 1), computed exactly from the sorted values: equal-value groups give the Δ = 0 pairs, and two pointers per
threshold give the pairs inside each interval (the interval is asymmetric, so both orders of a pair are counted
separately), O(n log n + 9n).  Exact in integers; NaN if fewer than two values.  No double-delta baseline
(no cheap exact form).  Observed − expected = gain from row order, as in the skip-index techniques.

**Rust:** `techniques/single_column/ordered/delta.rs` — columns in parallel (rayon), long columns in 2¹⁶-row chunks
merged in row order (each chunk keeps its first / last two non-null values so the boundary deltas are added back),
i128 arithmetic, entropy by hash counting of delta / double-delta values (`common/entropy_math`), baseline from a
sorted copy.  Binding: `bindings/api/techniques.rs` `delta`, `python.rs`, `_plugin.delta`.

**Testing** (`tests/test_delta.py`, the four blocks): contract; Rust vs reference — counts, ranges, bytes and
fractions exact, entropies RTOL 1e-9, baseline exact; known answers (regular series: double-delta bytes 0 and zero
fraction 1; constant column; overflow at Int64 bounds; nulls between values; one value; Decimal, Date, Datetime);
the baseline against a brute-force O(n²) oracle on small random columns.  `tests/performance/benchmark_delta.py`
(speed, the standard harness).

## 3. Emulator — `common/clickhouse_sizes.rs`

`Chain { gcd: bool, compression }` becomes `Chain { transform: Transform, compression }`, `Transform` ∈ {None, Gcd,
Delta, DoubleDelta}; `ddl()` renders `"LZ4"`, `"GCD, ZSTD(3)"`, `"Delta, LZ4"`, `"DoubleDelta, ZSTD(1)"`.  Transforms
apply per compression block to the same integer streams GCD uses (`Stream::gcd_width`: Int/UInt, Date, Date32,
DateTime, DateTime64, Time64, Decimal32/64, array inner streams), nulls as 0:

- **Delta:** ClickHouse's `CompressionCodecDelta` layout — its header, the first value, then wrapping differences at the
  type's width.
- **DoubleDelta:** `CompressionCodecDoubleDelta` — header, first value, first delta, then double-deltas bit-packed with
  its variable-length prefix codes.
- Two codecs in a chain add `CompressionCodecMultiple`'s header bytes.

**Spike first** (as for GCD, chbench R37's pattern): reproduce both layouts from ClickHouse's source, then a chbench
run on a handful of columns (regular timestamps, random walk, random ints; widths 2, 4, 8) compares emulated with
measured bytes, codec by codec, before the decision code lands.  Findings recorded in this spec's §7.

## 4. Decision — `recommenders/engine/compression.rs`

Families: plain; GCD (if the byte-width rule passes, unchanged); Delta; DoubleDelta.  The delta families apply only
to integer streams of a fixed width of 1, 2, 4 or 8 bytes (not Decimal128, floats, strings or LowCardinality).

**Pruning** (analytic, before emulation): the delta kernel (§2) runs on the emulator's integer stream values — the
recommended ClickHouse type's unit, nulls as 0:

- Delta survives iff `delta_bytes_mean < value_bytes_mean`;
- DoubleDelta survives iff `double_delta_bytes_mean < min(value_bytes_mean, delta_bytes_mean)`.

A pruned family gets `clickhouse_candidates` rows (outcome `pruned`) with evidence such as "delta 3.9 B ≥ value 3.1 B".
Each surviving family is measured under LZ4 and `zstd_levels`, with the near-incompressible shortcut per family.

**Choice:** a transform family wins only if its best chain saves `transform_min_saving` over the plain family's best;
among winners, the smallest.  `transform_min_saving` (default 0.01) **replaces** `gcd_min_saving` in Python (both
recommenders), the Rust API, the C ABI and Java (`OneShotParams` / `StreamingParams.transformMinSaving`) — no alias.

**Output:** no new columns.  `clickhouse_codecs` (e.g. "DoubleDelta, ZSTD(1)") and `clickhouse_candidates`
(chain, predicted bytes, outcome, evidence) carry the result; `clickhouse_size_bytes` is the chosen chain's.

## 5. Streaming

The pruning statistics come from the reservoir's sampled blocks — the blocks the emulator already measures — with
deltas within each block; sizes scaled by `n_rows / sampled_rows` as today.  Parity with one-shot: exact for one
batch with `block_rows` a multiple of 8 192, otherwise within about 2 %.

## 6. Calibration and handoff

`tests/performance/calibrate_clickhouse_sizes.py` gains Delta and DoubleDelta cells on five shapes — regular
timestamps, timestamps with jitter and gaps, a random walk, sorted ids, random ints (control) — at widths 2, 4 and 8,
under LZ4 and ZSTD(1/3).  Target: the fixed-width tolerance, within max(2 % × measured, 256 bytes).  `to_chbench`
already writes the codec axis in the R36 vocabulary, which includes `Delta` and `DoubleDelta`: the chosen transform
appears there the transform generalises `to_chbench`'s GCD handling.

Recommender tests (`tests/test_clickhouse_recommend.py`): a regular timestamp column chooses `DoubleDelta`; a
random walk chooses `Delta`; random ints prune both with evidence; Decimal128 / float / string columns get no delta
rows; `transform_min_saving` respected; streaming parity.  Java: the renamed parameter round-trips.

## 7. Spike findings

Spike: `tests/performance/spike_clickhouse_delta.py` (ClickHouse 26.8, one-column Wide MergeTree, one active part,
`data_compressed_bytes` after `OPTIMIZE FINAL`).  16 columns of 1 000 000 rows (regular UInt32, jittered Int64 ms,
random walks Int16 / Int32 / Int64, sorted UInt64 ids, random UInt16 / Int8, alternating-extreme Int8 / Int16 / Int32 /
Int64 that exercise the most negative double-delta, and four columns with GCD > 1) times 11 chains (`NONE`, `Delta`,
`DoubleDelta`, `GCD`, `LZ4`, and each transform with `LZ4` and `ZSTD(1)`): **all 176 cells are byte-exact**
(emulated = measured), codec-only and chained, for widths 1, 2, 4 and 8.  Examples (emulated = measured):

| column | NONE | Delta | DoubleDelta | GCD | Delta, LZ4 | DoubleDelta, LZ4 | DoubleDelta, ZSTD(1) |
|---|---|---|---|---|---|---|---|
| regular UInt32 | 4 001 550 | 4 001 674 | 127 418 | 4 001 922 | 19 778 | 4 829 | 4 648 |
| extreme Int8 | 1 000 400 | 1 000 432 | 1 063 012 | 1 000 448 | 4 914 | 5 415 | 1 426 |

Layout facts (source: CompressionCodec{Delta,DoubleDelta,Multiple}.cpp, BitHelpers.h, confirmed by the measurements):

- **Delta**: `[width, bytes_to_skip]`, the `len % width` leading bytes, then wrapping differences (first minus 0).
- **DoubleDelta**: the same 2-byte header and skipped bytes, UInt32 item count, first value, first delta (`width` bytes
  each), then per item `0` (double-delta 0) or prefix `10` / `110` / `1110` / `11110` / `11111` + sign bit (1 =
  negative) + `|dd| - 1` in 6 / 8 / 11 / 31 / 63 bits (ranges -63 < dd < 64, ±255, ±2047, Int32 interior, else); the
  width's most negative value is stored as `2^(8w-1) - 1`; MSB first, last byte zero-padded.  Same table for every width.
- **GCD** (changed from Task 3): the codec always writes `[width, 0]` and the GCD (`width` bytes, 1 when the values
  have none) before the quotients, so a GCD-1 column also grows by `2 + width` bytes per block.  Chains were
  previously off by that and by the Multiple header (up to 11% on LZ4-friendly Int16 data).
- **Two codecs** (`CompressionCodecMultiple`): the outer 25-byte block header, then 12 bytes (codec count + 2 method
  bytes + the compressor's own 9-byte header) before the compressed data; the transform's output enters the compressor
  behind its own 9-byte block header (method byte, compressed size, decompressed size).  A single codec has neither.
  Applied to GCD, Delta and DoubleDelta alike.
- `Compression::None` (codec-only) = outer header + transform body, uncompressed.
- Unaligned tails are never produced by fixed-width streams (all blocks are whole values), so the tail placement
  (modelled at the front, as in the source) is untested.

### Calibration

`tests/performance/calibrate_clickhouse_sizes.py` (456 cells, ClickHouse 26.8 via chbench, 3 workers) now also covers
seven 1 000 000-row delta frames (`ts_regular`, `ts_jitter`, `walk` Int64, `ids_sorted`, `walk_i16`, `walk_i32`,
`random_ints`).  `widen()` measures every transform on each of them (codec axis none / GCD / Delta / DoubleDelta,
compression LZ4 / ZSTD(1) / ZSTD(3), predictions from the emulator on the suite's own dataset), so the control and
the unrecommended families are covered as well as the recommended chain; `to_chbench` writes the recommended
transform on the axis (tested separately).

| family | cells | median error | max error | within max(2 %, 256 B) |
|---|---|---|---|---|
| Delta | 36 | 0 | 0 | yes (byte-exact) |
| DoubleDelta | 30 | 0 | 0 | yes (byte-exact) |
| GCD | 32 | 0 | 0 | yes (byte-exact) |
| none (codec-only: LZ4 / ZSTD) | 328 | 0 | 19.1 (six constant-looking `large` Int32 / Bool columns) | all but those |
| LowCardinality | 30 | 0.9 % | 2.5 % | yes (2.5 % rule) |

GCD before the spike's model change (271 cells): 2 cells, median 0.97 %, max 1.86 % (`gcd_1000` LZ4 -1.86 %);
now 32 cells, all byte-exact.  The "none" outliers (`large` col_60 / 68 / 72 / 80 / 88 / 92, original-type LZ4
cells: 925 predicted vs 46 measured Int32, 231 vs 46 Bool) are unchanged from before this work; they are
columns of the 50K-row mixed frame (cause not investigated here; not a codec effect, no transform is involved).  String columns keep their known error (ToDo 7).


## 8. Documentation

CLAUDE.md: the technique in §1b, the codec families and `transform_min_saving` in the recommender paragraphs, the
project tree; ToDo.md 3 marked **Done** (spec 2026-10-23).
