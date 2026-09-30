# Streaming Recommender — design

Status: approved, not implemented.
Builds on: the Recommend technique (2026-09-26-recommend-technique-design.md, "Spec B"), the Arrow FFI
interface (2026-09-27-arrow-ffi-interface-design.md) and the Java binding
(2026-09-28-java-binding-design.md), whose `capi.rs` and `python` Cargo feature this spec extends.

## 1. Goal

`describe_and_recommend` needs every row of a column in memory at once. Data that does not fit
cannot be profiled, and its size is often unknown in advance. The `StreamingRecommender` is a
stateful Rust object, exposed to Python and to the C ABI, that:

- accepts record batches incrementally (`add`);
- keeps exact running statistics plus a bounded sample of contiguous row blocks;
- produces a recommendation on demand (`finish`), without consuming the state.

Batches carry whole rows. The column set may change between batches (§4.4).

In scope: the Rust core (`streaming.rs` + `api.rs`), the refactor of `describe.rs` / `recommend.rs`
that lets one-shot and streaming share the rules, the pyo3 class and Python wrapper, the C ABI
functions and their Rust tests, accuracy tests, a benchmark script.

Out of scope:
- the Java class (about 60 lines once the Java binding lands);
- per-batch verification (a possible `verify_batches` debugging flag later);
- type widening between batches (Int32 → Int64 and so on);
- `population_rows`, Duj1 and Enum recommendations (§3);
- LazyFrame input, checkpointing or serialising the state, merging recommenders;
- entropy, top-5 and argmin/argmax in the streaming output.

## 2. Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Proof of a recommendation | **Exact statistics over every row.** Each rule is value-preserving given its statistics. Four new scanner statistics close the gaps that today only cast-and-verify covers (§5) |
| Role of the sample | ZSTD sizes and a cross-check of the chosen type. `reservoir_rows = 0` is valid (stats-only) |
| Sample shape | A reservoir of **contiguous blocks** of `block_rows` rows (Algorithm L over blocks), not of rows. Arrow IPC compresses each buffer of each batch independently, so blocks match the file's real framing and keep row-order locality, which a random-row sample would destroy |
| Distinct values | Tracked only for text levels, and only up to `categorical_threshold`. Past it the dictionary is rejected anyway, since the estimate is floored at `n_unique` |
| `population_rows` / Duj1 | Dropped. Estimate = Schnabel → Chao1, projection factor r = 1, dictionaries → Polars `Categorical` |
| Schema | Dynamic: new columns are backfilled with nulls, absent columns count as null, a Null-typed column adopts the first concrete type, and any other type change is rejected |
| Output | A lean schema (§6). Type-specific statistics appear only in `rec_candidates.evidence` |
| Python | A standalone class, not a technique on the uniform contract. It still reports `status` |

## 3. What the recommender reads

Every Profile field the rules in `recommend.rs` read, and how the stream keeps it:

| Field | Streaming form |
|---|---|
| `n_rows`, `n_null`, `sum_len`, `n_midnight`, float counts, string/ISO counts | sums |
| float `max_frac_digits`; string digit min/max, `int_min/max`, `iso_max_sig_frac_digits`; list `min_len/max_len` | min / max |
| `int_overflow` | OR |
| ISO `offsets` | set union |
| min/max value (ints, decimal unscaled, floats, temporal physical) | typed extremes. The rules read argmin/argmax only to look up these values |
| `gcd` | `gcd(a, b)` |
| `n_unique`, `f1`, `f2`, `capture_history`, `sum_len_unique` | bounded distinct map (§4.2) |
| top-5 values (only when `n_unique ≤ 5`, for the boolean-pair rule) | the first ≤ 6 distinct raw values |
| original size | analytic, or a per-batch sum (§5.3) |

Not read by the rules, and absent from the streaming output: entropy, top-5 counts, Duj1.

`population_rows` did three things: it chose Exact or Duj1, it projected sizes (r), and it chose Enum.
Without it, estimates use `estimate(…, q = None)` (Schnabel, else Chao1), r = 1, and every dictionary
is `Categorical`. The costs:
- sizes are reported at the observed scale;
- a fully streamed column still gets an estimate. Schnabel's `est_high` can exceed the true count,
  which conservatively widens a dictionary key at the 256 / 65,536 boundaries.

The estimators assume exchangeable rows. A time-ordered prefix can bias Chao1 and Schnabel. Only the
dictionary key width and gate depend on them.

## 4. State (`src/streaming.rs`)

### 4.1 Per column

```
ColumnState
  name, dtype: Option<DataType>   None while only Null-typed batches have been seen
  eligible: bool                  Int128/UInt128 or unsupported: name, dtype and counts only
  first_row: u64                  R₀, the global row where the column first appeared
  n_null: u64                     includes the backfilled rows and absent batches
  outer: Partial
  inner: Option<Partial>          List/Array: Describe's flattened inner values
  min_len, max_len                List/Array lengths over non-null rows
  n_midnight                      naive Datetime only

Partial                                        merge
  extremes: (lo, hi) i128 or f64 (NaN excluded) min / max
  gcd: i128                                    gcd
  sum_len: u64                                 +
  floats: FloatStats (existing)                + / max
  strings: StringStats (existing merge), plus  + / min / max / ∪
    iso_instant_min, iso_instant_max (ns)      min / max   new
    n_f32_roundtrip_fail, n_f64_roundtrip_fail +           new
    lossy counters (§5.2)                      + / min / max new
  sum_len_gt12: u64                            +           new: string_view size
  distinct: Option<Distinct>                   text levels only
```

`n_rows` is the global row count, the same for every column.

### 4.2 Distinct values

```
Distinct                                       merge
  map: HashMap<u64 hash, u8>                   low 2 bits: count capped at 3 (saturating +); next 3 bits: capture mask (OR)
  sum_len_unique: u64                          + on the first insert of a key
  first6: Vec<String>                          the first ≤ 6 distinct raw values, in stream order
  overflowed: bool                             set when map.len() > categorical_threshold; the map is then freed
```

- Keys are the existing value hashes (`encode_series`): Categorical/Enum hash as their string.
- Capture subsets use `subset(seed, global_row)`.
- f1 and f2 are the keys whose count is 1 and 2. `history[mask − 1]` counts keys by mask.
- A text level therefore needs at most about 16 bytes × `categorical_threshold`.
- After overflow the level stops hashing values and stops maintaining `sum_len_unique` and `first6`.
  Only the plain counts continue. Overflow is exact, not statistical: the estimate is floored at
  `n_unique`, so the dictionary is rejected whatever follows. It is never triggered early by a
  batch's distinct ratio.
- Schnabel's validity test (`d/n < 0.5`) uses the merged totals at `finish`, never a single batch.

### 4.3 `add(batch)`

1. Validate the whole batch against the schema (§4.4). Compute every column's batch partial with the
   existing `describe.rs` kernels, in parallel across columns and in 64K chunks within a column. The
   global row offset is passed in.
2. Only once every column succeeds, merge the partials into the state, apply the null accounting and
   feed the reservoir (§4.5). A failed `add` leaves the state byte-identical.
3. A zero-row batch only registers new columns.

`describe.rs` gains a mergeable batch partial and `finish() -> Profile`. The one-shot `describe_one`
becomes "one partial, then finish", so both paths share every kernel.

### 4.4 Dynamic schema

- **New column** first seen at global row R₀: its state starts with `n_null = R₀` and `first_row = R₀`,
  and all other statistics empty. Reservoir blocks sampled before R₀ hold nulls for it.
- **Absent column** (seen before, missing from this batch): `n_null += batch rows`. Its reservoir
  values for those rows are null.
- **Null type**: a Null-typed batch only adds to `n_null`. A column whose dtype is still None adopts the
  first concrete type. Its earlier rows are simply nulls, so nothing is recomputed.
- **Any other type change** (including an Enum with different categories): `InvalidInput`.
- **Duplicate column name** within a batch: `InvalidInput`.
- **Output order**: first-seen.

### 4.5 Block reservoir

- The stream is cut into contiguous blocks of `block_rows` rows. A block may span batch boundaries,
  with a rolling buffer holding the block in progress.
- Each completed block is offered to a seeded Algorithm L (seed derived from `seed`), which keeps
  B = ⌊`reservoir_rows` / `block_rows`⌋ blocks.
- A block is stored as one `RecordBatch` of the columns present. At `finish`, each column's values
  in a block come from it, or are a null array when the column is absent.
- Memory: at most `reservoir_rows` + `block_rows` rows.
- `reservoir_rows = 0`: no blocks are kept.

## 5. `finish()`

`finish` takes `&self`, so it can be called mid-stream. It works on a snapshot, per column, in parallel.

### 5.1 From state to recommendation

1. **Level.** `recommend.rs`'s `Level` gets `n_rows`, `n_null` and `extremes` as fields, instead of
   deriving them from `values` and `argmin`/`argmax`. `values` is the column's sampled blocks, and
   its Profile is the finished Partial. The one-shot path fills the same fields from its whole
   column. This is the only structural change to `recommend.rs`.
2. **Estimate.** `estimate(n_unique, n, f1, f2, history, None)`. If distinct tracking overflowed,
   there is no estimate, and the dictionary candidate is `rejected` with the reason
   `n_unique > categorical_threshold`.
3. **Candidates.** The unchanged rules of Spec B §4 and §5.2, ordered by predicted size (r = 1), then
   rank. Rejected candidates go last.
4. **Proof by statistics.** In `first_success`, each candidate's attempt checks the statistics that
   make it value-preserving, instead of casting and verifying:
   - for most rules, the rule's own condition is the proof: exact extremes, `gcd`, `n_midnight`,
     `n_f32_inexact`, and the scanner counts equal to `n`;
   - string → Float32 / Float64 also needs `n_f32_roundtrip_fail = 0` / `n_f64_roundtrip_fail = 0`;
   - string → Timestamp(ns) also needs the ISO instant range to lie within the i64-ns range
     (1677–2262);
   - a candidate that fails is `failed`, with the statistic as its reason
     (`n_f64_roundtrip_fail=3`). The original type cannot fail.
5. **`rec_lossy_formatting`** comes from the lossy counters (§5.2), compared against the chosen type.
6. **Cross-check** (B > 0). The chosen type is cast and verified on every sampled block, with the
   existing `cast_to`, `verify` and `wrap`. A mismatch is a `Compute` error naming the column,
   the target and the first failing value. It indicates a wrong statistic, and is never a silent
   fallback.
7. **Sizes** (§5.3).
8. **Polars type.** As Spec B §5.5, but a dictionary is always `Categorical(Categories(name=<column>, physical=k′))`.

### 5.2 New scanner statistics

Added to `describe.rs`'s numeric and ISO scanners, counted over every non-null text value:

- `iso_instant_min`, `iso_instant_max`: the UTC instant (ns, i128) of ISO datetimes, both kinds.
- `n_f32_roundtrip_fail`, `n_f64_roundtrip_fail`: numeric strings whose parsed Float32 / Float64 value,
  rendered as by the one-shot `verify_text` float check, is not canonically equal (`canon`) to the text.
- Lossy counters: the text form of a string value after the recast.
  - Parameter-independent spellings are counted directly:
    - a leading `+`;
    - `-0` / `-0.0`;
    - leading zeros before a decimal point;
    - an ISO date/time separator other than `T`;
    - `Z` versus a numeric offset, and offset spellings that arrow-cast renders differently.
  - Parameter-dependent spellings keep the raw (not trailing-zero-stripped) min and max fraction-digit
    counts, numeric and ISO. `finish` compares them with the chosen decimal scale or time unit.
  - For every source type, the counters must reproduce one-shot `lossy()` (tested, §8).
- For float sources, `-0.0` is counted in `FloatStats` (the one-shot `lossy` float case).

### 5.3 Sizes

- **Recommended, uncompressed (Arrow and Polars layouts):** analytic at N with `body_size`. The Polars
  layout of a Utf8 result is string_view: `pad(16·N) + pad(sum_len_gt12)`, plus validity. `sizes.rs`
  gains the matching measured case, so that predicted = measured still holds (Spec B §5.1).
- **Original, uncompressed:** `body_size` of the classic layout where it applies (fixed width,
  LargeUtf8/LargeBinary, LargeList of those). Categorical / Enum sizes are analytic too, classic and
  native: keys from the column's shape, the dictionary from the dtype's category mapping (the
  categories Polars exports, as views it builds). Otherwise (Struct, deeper nesting) it is the sum
  of the per-batch measured sizes: exact for one batch. The original candidate's evidence names the
  method. `size_polars_bytes` follows the same rule, applied to the original's Polars layout
  (`polars_layout`).
- **A list's kept inner level with no closed form (a Struct):** its recommended Polars size is the
  per-batch sum of its classic layout rebuilt as one-shot's `to_polars_layout` rebuilds it (not the
  input's native export, whose view buffers Polars built differently): exact for one batch.
- **ZSTD (original and recommended, Arrow and Polars layouts):** Σ over the sampled blocks of
  `ipc_body_bytes(block, zstd_level)`, × N ÷ sampled rows, rounded. The recommended sizes use the
  block cast to the chosen type, or its Polars layout.
  - On the snapshot, the in-progress partial block is offered to the reservoir too, so a stream
    shorter than one block is measured completely.
  - When every row is sampled, this is exactly the size of an IPC file written in `block_rows`
    batches.
  - B = 0: all ZSTD sizes are null.

## 6. Output (lean schema)

One row per column, first-seen order.

| Group | Column | Type |
|---|---|---|
| keys | `column` | String |
| | `status` | String ∈ {computed, ineligible} |
| source | `dtype` | String: `pa_name` of the input Arrow type |
| | `first_row`, `n_rows`, `n_null` | UInt64 |
| core stats | `min`, `max` | String: the tracked extremes (numeric, decimal, temporal), rendered as Describe renders them; null for text, binary, boolean and nested columns, whose extremes the rules never read |
| | `gcd` | Decimal(38, 0) |
| | `sum_len`, `min_len`, `max_len` | UInt64 |
| cardinality | `n_unique` | UInt64 (null when overflowed or not text) |
| | `distinct_overflowed` | Boolean (null when not text) |
| | `est_cardinality`, `est_low`, `est_high` | Float64 |
| | `est_method` | String |
| original sizes | `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes` | UInt64 |
| recommendation | `rec_nullable`, `rec_arrow_type`, `rec_arrow_size_bytes`, `rec_arrow_size_zstd_bytes`, `rec_polars_type`, `rec_polars_size_bytes`, `rec_polars_size_zstd_bytes`, `rec_lossy_formatting`, `rec_candidates` | as Spec B §6 |
| sampling | `n_sampled_rows`, `n_sampled_blocks` | UInt64 |

- A null means not applicable. For example, `gcd` is null for strings, and the ZSTD columns are null when B = 0.
- `rec_polars_type` is always filled, including when the original type is kept (it comes from `pl_name`).
- Ineligible rows fill only the keys, `dtype`, `first_row`, `n_rows` and `n_null`.
- The type-specific statistics are not columns. Every value a rule tested is in `rec_candidates.evidence`.
- The output must export to native Arrow: no Polars-only types.

**Parity.** Take the batches concatenated diagonally (Polars `concat(how="diagonal")`, i.e. with the
null filling of §4.4), and compare with one-shot `describe_and_recommend(population_rows=None)` on it.
When no column's distinct tracking overflowed, these are equal:
- `rec_arrow_type`, `rec_polars_type`, `rec_nullable`, `rec_lossy_formatting`;
- `rec_arrow_size_bytes`, `rec_polars_size_bytes`, `size_bytes` (analytic cases);
- every candidate's type, rule, predicted size and outcome.

ZSTD sizes are equal when N ≤ `block_rows`. Otherwise streaming measures per block and one-shot
measures the column as one batch. Two differences are expected by design:
- a string → Float candidate that failed cast-and-verify in one-shot is `failed` here by a statistic,
  so the outcome is equal and the reason text differs;
- one-shot gives a dictionary an `Enum` only when `population_rows` equals the rows, which the parity
  runs never pass.

## 7. Bindings

### 7.1 `api.rs`

```rust
pub struct StreamingParams {
    pub reservoir_rows: u64, pub block_rows: u64, pub categorical_threshold: u64,
    pub zstd_level: i32, pub seed: u64, pub boolean_pairs: Vec<(String, String)>,
}
pub struct StreamingRecommender { /* streaming.rs state */ }
impl StreamingRecommender {
    pub fn new(p: StreamingParams) -> Result<Self, Error>;
    pub fn add(&mut self, batch: &RecordBatch) -> Result<(), Error>;
    pub fn finish(&self) -> Result<RecordBatch, Error>;
}
```

Parameter validation (`InvalidInput`):
- `block_rows ≥ 1`;
- `reservoir_rows = 0` or `reservoir_rows ≥ block_rows`;
- `zstd_level` within ZSTD's range;
- `boolean_pairs` are pairs of distinct non-empty strings, compared case-insensitively.

Errors:
- `InvalidInput`: parameters, a schema or type conflict, a duplicate column, an unimportable type.
- `Compute`: a kernel failure or a failed cross-check.

A failed `add` leaves the state unchanged.

### 7.2 `python.rs`

```rust
#[pyclass(frozen, module = "analytics.analytics")]
struct StreamingRecommender(Mutex<api::StreamingRecommender>);
// #[new] fn new(reservoir_rows, block_rows, categorical_threshold, zstd_level, seed, boolean_pairs)
// fn add(&self, py, data)   — any __arrow_c_stream__; its batches are added one at a time, in order,
//                             never concatenated
// fn finish(&self, py) -> ArrowTable
```

- The `Mutex` serialises concurrent callers, and the GIL is released around the work (`py.detach`,
  pyo3 0.29).
- Polars' private `_pli128` / `_plu128` columns are not rejected. They are recorded as ineligible
  (name and null count) and projected out before import.
- `InvalidInput` → ValueError, `Compute` → RuntimeError.

### 7.3 Python wrapper — `analytics/recommend/streaming.py`

```python
class StreamingRecommender:
    def __init__(self, *, reservoir_rows=524_288, block_rows=65_536, categorical_threshold=10_000,
                 zstd_level=1, seed=0, boolean_pairs=(("true", "false"),)): ...
    def add(self, frame) -> "StreamingRecommender": ...   # pl.DataFrame or any __arrow_c_stream__ object
    def finish(self) -> pl.DataFrame: ...
```

- It reaches Rust only through `analytics._plugin`.
- It is exported from `analytics.recommend`, but not listed in `IMPLEMENTATIONS`.
- `reservoir_rows` at its default, with 100 columns of 8-byte values, is about 400 MB. Wide or
  long-string sources should lower it.

### 7.4 `capi.rs`

These are added to the Java spec's `capi.rs` and follow its conventions: return codes 0 / 1 / 2,
and `char **error` freed with `analytics_free_error`.

```c
typedef struct AnalyticsRecommender AnalyticsRecommender;
int  analytics_recommender_new(uint64_t reservoir_rows, uint64_t block_rows,
                               uint64_t categorical_threshold, int32_t zstd_level, uint64_t seed,
                               const char *const *bool_true, const char *const *bool_false,
                               size_t n_bool_pairs, AnalyticsRecommender **out, char **error);
int  analytics_recommender_add(AnalyticsRecommender *h, struct ArrowArrayStream *batches, /* consumed */
                               char **error);
int  analytics_recommender_finish(AnalyticsRecommender *h, struct ArrowArrayStream *out, char **error);
void analytics_recommender_free(AnalyticsRecommender *h);   /* NULL is a no-op */
```

- `new` returns `Box::into_raw`, and `free` calls `Box::from_raw`.
- The handle holds `Mutex<api::StreamingRecommender>`, so it is thread-safe.
- `add` reads the stream batch by batch.
- A null handle or stream returns 1.

## 8. Testing

### 8.1 Rust (`cargo test --lib streaming::`, `capi::`)

- **Merge laws**: for every statistic in §4.1–4.2, the merged partials of any split equal the partial
  of the whole data, including 1-row and empty batches.
- **Distinct overflow** at exactly `categorical_threshold` and +1.
- **Schema evolution**:
  - a new column (`first_row`, backfilled `n_null`);
  - an absent column;
  - Null adopting a type;
  - a conflict and a duplicate rejected, with the state unchanged.
- **Reservoir**:
  - Algorithm L is deterministic for a given seed;
  - blocks are cut across batch boundaries;
  - every row is kept when N ≤ `reservoir_rows`;
  - block selection is uniform (χ² over many seeds).
- **Proof by statistics**:
  - string → Float fails on a value that does not round-trip, and passes otherwise;
  - Timestamp(ns) fails outside 1677–2262.
- **Lossy counters** equal one-shot `lossy()` on the Spec B known-answer cases.
- **C ABI**: new / add / finish / free, error codes and messages, `free(NULL)`, and a null handle or stream.

### 8.2 `tests/test_streaming_recommend.py` (accuracy only)

1. **Contract**:
   - schema and types;
   - first-seen order;
   - ineligible rows (Int128);
   - `finish().to_arrow()`;
   - `finish()` twice, and `add` after `finish`.
2. **Parity** (§6):
   - datasets: `describe_mixed`, `stringified(describe_mixed)` and `large_dataset.arrow`;
   - batch sizes: 1, 7, 1,000 and whole;
   - compared with `RecommendRust(population_rows=None)`;
   - `reservoir_rows ≥ N`.
3. **ZSTD against the real file**: pyarrow writes the recast column as IPC, with ZSTD at `zstd_level`
   and `max_chunksize=block_rows`, and its body is measured with `_sizes.py`.
   - With every row sampled, the sizes are equal.
   - With sampling (`large_dataset.arrow`, `block_rows=4_096`, `reservoir_rows=16_384`), they are within 10%.
4. **Known answers**:
   - a new column mid-stream, an absent column, Null → String adoption, and a type conflict (ValueError);
   - overflow → dictionary `rejected` and null `n_unique`;
   - `reservoir_rows=0` → null ZSTD sizes;
   - string → Float64 `failed` by `n_f64_roundtrip_fail`, and ns out of range `failed`;
   - parameter validation errors.

### 8.3 Benchmark — `tests/performance/benchmark_streaming_recommend.py`

A standalone script, since the harness assumes `IMPLEMENTATIONS`. On the Describe datasets it reports
`add` throughput (rows/s, at several batch sizes), `finish` time and peak memory, beside one-shot
`RecommendRust` on the same data. Results go to `tests/performance/results/streaming_recommend.parquet`.

## 9. Documentation

- CLAUDE.md: add **Streaming Recommend** under "1. Per-column", after Recommend. Add `streaming.rs` to
  the project structure. Add `StreamingRecommender` to the Rust extension notes, as the first
  stateful object across the three layers.
- Spec B: a "see also" link to this spec.

## 10. Done when

- `cargo test --lib streaming:: capi::` passes (with the python feature and without).
- `tests/test_streaming_recommend.py` passes, including parity on `large_dataset.arrow`.
- The existing Describe and Recommend tests pass unchanged after the `describe.rs` / `recommend.rs`
  refactor.
- The benchmark script runs and writes its results.
