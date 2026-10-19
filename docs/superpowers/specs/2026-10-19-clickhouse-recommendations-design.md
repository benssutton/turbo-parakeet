# ClickHouse recommendations and the chbench handoff — design (ToDo 1b, 1f, rest of 1.0)

Status: approved 2026-10-19.  Sub-project A of ToDo 1a–g (B: skip indexes 1c–e; C: run-end encoding 1a; D: range
queries 1g — each its own spec).  Builds on the schema skeleton (spec 2026-10-18 §2), whose reserved ClickHouse and
compression columns this spec fills.

## 1. Goal and scope

Recommend, per column, a ClickHouse column type, a codec chain (GCD, then LZ4 or ZSTD + level) with a predicted
compressed size, and the Arrow IPC buffer compression (LZ4_FRAME or ZSTD + level) for the DataFusion and Polars
targets; and hand each recommendation to the sibling project `clickhouse_benchmark` (`chbench`) as a runnable suite
over the real column, so the prediction is validated empirically.

Two repositories:
- **turbo-parakeet:** the type mapping (§2), the ClickHouse on-disk size emulator (§3), the decisions (§4), the
  output columns and keywords (§5), streaming (§6), the `to_chbench` handoff (§7), bindings (§8).
- **clickhouse_benchmark** (`c:/Users/Alexander/clickhouse_benchmark`, its own conventions — TDD, spikes,
  measured-only vocabulary): a real-data dataset source and a vocabulary spike (§9).

Out of scope: Delta / DoubleDelta (ToDo 3); T64, Gorilla, FPC (not in any ToDo); skip indexes (sub-project B);
`chbench` validation of nested types (`Array`, `Tuple` get a type and a predicted size, but no suite);
`timestamp_with_offset` (no ClickHouse type: its ClickHouse cells are null with the reason in the candidates);
a synthetic "twin" dataset — the two projects are expected to merge, with one synthetic generator (ToDo, §12).

## 2. Arrow → ClickHouse type mapping

Applied to the **recommended** Arrow type (`datafusion_type`) and its statistics; wrapped in `Nullable(...)` when
`nullable` is true (inside `Array` / `LowCardinality`, which cannot be wrapped).

| recommended Arrow type | ClickHouse type |
|---|---|
| bool | `Bool` |
| int8 / 16 / 32 / 64, uint8 / 16 / 32 / 64 | `Int8` … `Int64`, `UInt8` … `UInt64` |
| float32 / float64 | `Float32` / `Float64` |
| decimal128(p, s) | `Decimal32(s)` if p ≤ 9, `Decimal64(s)` if p ≤ 18, else `Decimal128(s)` |
| date32 | `Date` (2 bytes) if min and max lie in 1970-01-01 … 2149-06-06, else `Date32` |
| timestamp(unit, tz) | `DateTime('tz')` if the unit is seconds and min / max lie in 1970-01-01 … 2106-02-07; else `DateTime64(p, 'tz')`, p = 0 / 3 / 6 / 9 for s / ms / µs / ns (no tz → no tz argument) |
| time32 / time64 | `Time64(p)`, p = 0 / 3 / 6 / 9 for s / ms / µs / ns — accepted without a setting (chbench R36); 8-byte signed, GCD not applied |
| string, large_string, string_view, binary, large_binary, binary_view | `String` |
| dictionary<K, string> (the recommender's string→dictionary candidate) | `LowCardinality(String)` |
| a Polars Enum kept as the original | `Enum8(...)` if ≤ 127 categories, else `Enum16(...)`, categories in order, codes 1… |
| fixed_size_binary(n) | `FixedString(n)` |
| list<T> / fixed_size_list<T> | `Array(T′)` |
| struct | `Tuple(name T′, …)` |
| timestamp_with_offset, null, anything else | null (`clickhouse_candidates` names the reason) |

`T′` is the mapping of the child type (nullable inside when the child holds nulls).  The mapping is one Rust function
(`recommenders/engine/clickhouse.rs`) with a unit test per row of this table.

## 3. ClickHouse size emulator

`common/clickhouse_sizes.rs`: the column laid out as MergeTree writes it (Wide parts), compressed block by block.
Approximate by design; calibrated against measured bytes (§10).

- **Streams.**  Fixed-width values little-endian (`Bool`/`Int8`/`UInt8`/`Enum8` 1 byte, `Int16`/`UInt16`/`Date`/
  `Enum16` 2, `Int32`/`UInt32`/`Float32`/`Date32`/`DateTime`/`Decimal32` 4, 64-bit types and `DateTime64`/
  `Decimal64` 8, `Decimal128` 16, `FixedString(n)` n); `Nullable(T)`: a `UInt8` null-map stream plus T's stream with
  the type's default in null slots; `String`: per row a varint length and the bytes; `LowCardinality(String)`: a
  dictionary stream (the distinct strings, String layout) and a keys stream of UInt8 / UInt16 / UInt32 by dictionary
  size (approximation: one dictionary per part, no per-granule headers); `Array(T)`: a `UInt64` offsets stream plus
  T's streams; `Tuple`: each field's streams.  Each stream is compressed separately with the column's codec chain;
  the column's size is the sum.
- **Blocks.**  Rows are taken in granules of 8 192 (`index_granularity`); a compression block closes after the
  granule that brings it to ≥ 65 536 bytes (`min_compress_block_size`) and never exceeds 1 048 576
  (`max_compress_block_size`; a larger granule is split).  Each block costs its compressed payload plus 25 bytes
  (16-byte checksum + 9-byte header).
- **Codecs.**  `GCD`: per block, the block's own GCD of the values (as ClickHouse applies it), the values divided,
  same width, plus one value-width header (approximation, confirmed or corrected by §9.2 / §10).  `LZ4`: liblz4
  `LZ4_compress_default` (the `lz4` crate, which binds liblz4 — the same library ClickHouse and pyarrow use).
  `ZSTD(l)`: the `zstd` crate at level l.
- **Byte entropy.**  The emulator also returns H_b: the Shannon entropy, in bits per byte, of one 256-bin histogram
  over every uncompressed byte of the column's streams (after GCD when the chain has it).  One pass, used by §4.

## 4. Decisions

Thresholds are recommender keywords (decisions belong to the recommenders; techniques are unchanged).

### 4.1 GCD (ToDo 1b) — analytic pruning, then emulation

- **Applicable** to integer, decimal, `Date`, `Date32`, `DateTime`, `DateTime64` columns (not `Bool`, floats,
  strings, enums, nested).
- **g** is the GCD of the **ClickHouse physical values** (the recommended type's: e.g. after a decimal rescale or a
  timestamp unit change, which already used part of the original GCD).  One-shot computes it on the recommended
  array; streaming derives it from the running GCD divided by the type rule's rescale factor.  In streaming, a
  numeric type derived from text (a numeric-string column) has no running GCD: its GCD candidate is not offered
  (`clickhouse_candidates` says why); one-shot computes it.  Parity tests exclude that case.
- **Pruning rule (O(1), from g and the min / max).**  With `bytes_needed(x)` the number of bytes holding the
  magnitude of x (plus a sign bit for signed types; 0 → 1 byte), GCD is a candidate only if
  `bytes_needed(max(|min|, |max|) / g) < bytes_needed(max(|min|, |max|))`.  Rationale: dividing by g shortens every
  value by ⌊log₂ g⌋ to ⌈log₂ g⌉ bits — at least two bits needs g ≥ 4, k bits needs g ≥ 2^k — but byte compressors
  gain little from sub-byte savings, and division is invertible (an ideal entropy coder already exploits "every value
  is a multiple of g").  What they do exploit is whole zero (or, for negatives, 0xFF) leading bytes.  g ≥ 256 always
  passes (≥ 8 bits removed); 2 ≤ g < 256 passes only when the maximum sits just above a byte boundary; g = 1 never.
  Column-level g and maximum are conservative: per block, ClickHouse's GCD and the block maximum can only be as or
  more favourable.
- A pruned GCD is listed in `clickhouse_candidates` with outcome `pruned` and its evidence (`g`, bytes at max before
  → after).

### 4.2 Compression (ToDo 1f) — the near-incompressible shortcut, then emulation

- **Shortcut.**  ZSTD's gain over LZ4 comes from entropy-coding literal bytes (bounded by 1 − H_b / 8) and from
  better matches (which need repeated values).  If `1 − H_b/8 < zstd_min_saving` **and** the column is mostly distinct
  (`n_unique / (n_rows − n_null) ≥ 0.5`, the project's existing "mostly distinct" boundary), ZSTD is ruled out:
  LZ4 is chosen and no ZSTD level is emulated.  Evidence: H_b and the distinct ratio.
- **Otherwise** LZ4 and ZSTD at each level of `zstd_levels` are emulated.  The ZSTD level is the lowest whose size is
  within `zstd_level_tolerance` of the smallest ZSTD size; ZSTD is chosen over LZ4 only if its size ≤
  (1 − `zstd_min_saving`) × LZ4's (LZ4 decompresses faster).
- **Chains.**  Candidates are {no GCD} ∪ {GCD if it survived §4.1}, each with the compression chosen as above.  GCD is
  chosen if its chain's size ≤ (1 − `gcd_min_saving`) × the chain without it.
- **`clickhouse_codecs`** is the chosen chain as written in DDL: `LZ4`, `ZSTD(3)`, `GCD, LZ4`, `GCD, ZSTD(3)`.
  `LowCardinality` is part of `clickhouse_type`, not a codec.
- **Arrow IPC** (DataFusion and Polars): the same shortcut and rule choose `LZ4_FRAME` or `ZSTD(level)` for the
  recommended type's IPC body, H_b taken over the body's buffers; sizes from `common/ipc_sizes.rs` extended with an
  LZ4_FRAME body (pyarrow's layout: per non-empty buffer an 8-byte uncompressed length + one LZ4 frame).  These are
  exact against pyarrow (predicted = measured), as the ZSTD sizes are.

### 4.3 Keywords (both recommenders)

| keyword | default | meaning |
|---|---|---|
| `zstd_levels` | `(1, 3, 9)` | ZSTD levels emulated; each within the zstd crate's range, distinct, non-empty |
| `zstd_level_tolerance` | `0.01` | the lowest level within this fraction of the best ZSTD size wins |
| `zstd_min_saving` | `0.10` | ZSTD must save this fraction over LZ4; also the shortcut's bound |
| `gcd_min_saving` | `0.01` | GCD must save this fraction over the chain without it |

Fractions must lie in [0, 1).  The existing `zstd_level` keeps feeding the existing `*_size_zstd_bytes` columns,
unchanged.

## 5. Output columns

Placed per the skeleton (spec 2026-10-18 §2.1), the reserved names kept:

| block | columns (types) |
|---|---|
| DataFusion | after `datafusion_size_zstd_bytes`: `datafusion_size_lz4_bytes` UInt64, `datafusion_ipc_compression` String (`LZ4_FRAME` / `ZSTD`), `datafusion_ipc_zstd_level` Int32 (null unless ZSTD), `datafusion_ipc_size_bytes` UInt64 (body size under the chosen compression) |
| Polars | the same four with `polars_` |
| ClickHouse (after the Polars block, before `candidates`) | `clickhouse_type` String, `clickhouse_codecs` String, `clickhouse_size_bytes` UInt64 (predicted), `clickhouse_candidates` `List<Struct{codecs: Utf8, predicted_bytes: UInt64, outcome: Utf8, evidence: Utf8}>` — outcome ∈ {chosen, rejected, pruned}; `clickhouse_skip_indexes` stays reserved for sub-project B |

Ineligible columns: all null.  A column whose ClickHouse type is null has null `clickhouse_codecs` /
`clickhouse_size_bytes` and one candidate row giving the reason.

## 6. Streaming

The emulator and the Arrow IPC LZ4 / level sizes run on the reservoir sample's blocks and are scaled by
`n_rows / sampled_rows`, as the ZSTD sizes are; H_b is taken over the sampled blocks; GCD and min / max come from the
running statistics (exact).  `reservoir_rows = 0` → the sizes, choices and `clickhouse_codecs` are null
(`clickhouse_type` is still given).  Parity with one-shot: exact when one batch holds every row, `block_rows` is a
multiple of 8 192 and the type is fixed-width (compression blocks then never straddle a sampled block); otherwise
within the tolerance measured by the parity tests (target 2%).
The Arrow IPC sizes are measured on the sampled blocks concatenated into one array (the single body one-shot measures), scaled by `n_rows / sampled_rows`; the ClickHouse emulator runs per sampled block (its blocks are granule-aligned) and sums. A temporal type derived from text has no exact range in streaming, so it maps to the always-valid wide type (`Date32`, `DateTime64(p)`) where one-shot may prove `Date` / `DateTime`; parity tests allow exactly this difference.

## 7. Handoff: `to_chbench`

`analytics.recommend.to_chbench(result, frame, out_dir, *, columns=None) -> list[tuple[str, str]]` (Python only — a
helper, not a recommender).  For each computed column of `result` (or the named `columns`) whose ClickHouse type and
original type are expressible as `chbench` base types, it writes:
- `<name>.arrow`: the original column, its exact Arrow type, one uncompressed record batch (Feather v2, as `chbench`
  caches datasets).  Types `chbench` cannot parse are normalised losslessly before writing: string_view /
  large_string → string, binary_view / large_binary → binary, Polars Categorical → its strings (dictionary dropped;
  the recommended `LowCardinality` is a type-axis value, not the stored type);
- `<name>.yaml`: a suite —
  ```yaml
  name: <name, sanitised to [a-z0-9_]>
  dataset: {path: <name>.arrow, column: <original column name>}   # + decimal_places for Decimal types
  clickhouse:
    type: [<recommended base type>, <original base type>]          # distinct values only
    nullable: [<nullable>]
    primary_key: [false]
    projection: [false]
    codec: [<GCD | none | LowCardinality>, none]                    # distinct
    skip_index: [none]
    compression: [<chosen compression>, LZ4]                         # distinct
  queries:
    kinds: [{kind: full_scan}, {kind: filter, target_fraction: 0.05, predicate: in, values: 1},
            {kind: filter, target_fraction: 0.10, predicate: range}]  # range only for numeric / temporal
    query_condition_cache: [false]
    cache_mode: [warm]
    samples: 10
  results: {percentiles: [50, 90, 99]}
  ```
- `<name>.predicted.json`: `{"<type>|<codec>|<compression>": bytes or null}` for every cell of the grid the suite will build.
  Recommended-type cells take the predicted bytes of the matching `clickhouse_candidates` chain (null when that chain was
  pruned or not measured); original-type cells are emulated on the original column through a private binding, which also
  gives the original's ClickHouse type.

It returns the skipped columns with reasons (nested, no ClickHouse type, ineligible).  Grid cells `chbench`'s rules
reject are skipped by `chbench` itself, with the server's reason.

## 8. Bindings

Both recommenders' Rust API params, pyo3 constructors, Python wrappers (`OneShotRecommender`,
`StreamingRecommender`), the C ABI (`_new`: `const int32_t *zstd_levels, size_t n_zstd_levels, double
zstd_level_tolerance, double zstd_min_saving, double gcd_min_saving`) and the Java params (`zstdLevels`,
`zstdLevelTolerance`, `zstdMinSaving`, `gcdMinSaving`, with the defaults above) gain the four keywords; the new
columns appear in every binding automatically.  New Rust dependency: `lz4` (liblz4 via `lz4-sys`; a C compiler is
already required by `zstd-sys`).

## 9. clickhouse_benchmark changes

Made in that repository, under its conventions (TDD, `spikes/` + `FINDINGS.md`, spec rulings, `clickhouse`-marked
integration tests), against ClickHouse 26.8 in a local Docker container (`clickhouse:26.8`, HTTP 8123, `chbench`'s
`CH_*` environment variables).

### 9.1 Real-data dataset source

A suite's `dataset` block may instead be `{path: <file>.arrow, column: <name>}` (path relative to the suite file),
optionally with `decimal_places` (else taken from a decimal column's scale).  The column is loaded with its exact
Arrow type; rows, null count and the manifest metrics are computed from the data with `chbench`'s existing metrics
functions; the dataset id is a content hash; generation, caching and tolerances do not apply.  Parse-time checks
(`_check_dataset_against_type_axis`) work from the file's type.  Implementation: a path dataset is a `DatasetSpec`
whose fields are measured from the file (`rows`, `arrow_type`, `null_count`, `decimal_places`, `value_range`), plus
`path` and `column`; `load_or_generate` loads the file instead of generating when `path` is set, and the manifest
holds the measured `cardinality_ratio` / `entropy_bits`, no targets and no warnings.  The harness, which reads
`DatasetSpec`, is otherwise unchanged.

### 9.2 Vocabulary spike

`spikes/spike_ch_vocabulary.py` (in the style of `spike_ch_roundtrip.py`): for each type this spec can emit —
`UInt8`, `UInt16`, `UInt64`, `Int8`, `Int16`, `Int64`, `Float32`, `Bool`, `Decimal128(s)`, `Date32`, `DateTime`,
`DateTime64(0 / 6 / 9)`, `FixedString(n)`, `Enum8`, `Enum16`, `Time64(p)` — and for the `GCD` codec, run the DDL and an
insert against the live server for every codec × compression × skip-index × nullable combination of the existing
matrix, recording accepted / rejected with the server's message; also record the GCD codec's on-disk layout (block
header, value width) from `system.parts_columns` on known inputs.  Findings go to `spikes/FINDINGS.md` §9 and
`chbench`'s spec §5.2; `validate.py` (`KNOWN_TYPES`, `KNOWN_CODECS`, skip rules and messages) admits what was
measured, nothing else.

## 10. Calibration (outcome to fill in)

`tests/performance/calibrate_clickhouse_sizes.py` (turbo-parakeet; needs the container and `chbench` importable):
`to_chbench` over `datagen.large_dataset_mixed()` plus GCD cases (UInt64 multiples of 1 000 up to 10¹²,
which lose a byte at the maximum; even UInt64 values in 2³²…2³³, which lose one byte with g = 2; Int64 multiples of 3
up to 2⁴⁰, which keep their byte width and so must be pruned; timestamps in whole seconds stored as µs), then `chbench run` on each suite, then compares
`<name>.predicted.json` with `chbench`'s measured compressed bytes per cell.  The measured relative errors per type
and codec, and the tolerance adopted from them, are written into §11 and CLAUDE.md.  If the emulator misses by more
than 10% on a fixed-width type, the approximation in §3 is revised before the tolerance is set.

## 11. Outcome

To be filled in by the implementation: the vocabulary spike's findings (§9.2), the calibration table and tolerance
(§10).

## 12. Testing and documentation

- **Rust unit tests:** the mapping (one test per §2 row, incl. the Date / DateTime range edges); emulator streams
  byte-exact for each layout on small hand-built inputs; block cutting (granule alignment, 64 KiB, 1 MiB split);
  GCD per block; H_b; the GCD pruning rule (g = 1, 2 just over a byte boundary, 3, 4, 255, 256; signed); the
  shortcut and the three thresholds; Arrow IPC LZ4_FRAME body.
- **Python tests:** LZ4_FRAME IPC sizes equal pyarrow's (`pa.ipc` with `compression="lz4"`) on `describe_mixed`
  and `large_dataset_mixed`; decisions with each keyword moved across its boundary; one-shot / streaming parity (§6);
  `to_chbench` output (files, YAML keys, grid values, skipped list) and, when `chbench` is importable, its suites
  parse with `chbench.harness.suite.load_suite`.
- **Java / C:** the new keywords round-trip; the new columns appear.
- **chbench:** its own TDD tests for §9.1; `clickhouse`-marked tests for the admitted vocabulary.
- **Docs:** CLAUDE.md (recommender paragraphs: columns, keywords, the handoff, the calibration tolerance); ToDo.md
  (1b, 1f done; 1.0's type mapping and handoff done; a new item: consolidate the two synthetic data generators —
  `tests/datagen.py` and `chbench.datagen` — when the projects merge); `chbench`'s CLAUDE.md status.
