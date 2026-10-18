# Recommender output schema, entropy and exemplars — design (ToDo 1.0 skeleton, 4, 2)

Status: approved 2026-10-18.  Implements the ToDo 1.0 schema skeleton with ToDos 4 (entropy) and 2 (exemplars) as
its first increment.

## 1. Goal and scope

Change the recommenders' output layout once, so that ToDos 1a–g and 3 only add columns, and add the first new
columns: each column's entropy (ToDo 4) and an exemplar row and null-exemplar row (ToDo 2).

In scope: both recommenders (`OneShotRecommender`, `StreamingRecommender`), their Python wrappers, the language-neutral
API (`bindings/api/recommenders.rs`), the C ABI and the Java binding, the tests (known answers, oracles, parity), one
benchmark, CLAUDE.md.

Out of scope (each ToDo's own spec): decision thresholds, the index cost model, the Arrow → ClickHouse type mapping,
the `CREATE TABLE` helper and IPC handoff to clickhouse_benchmark, REE / LZ4 size measurement, entropy of list inner
values.  Techniques (Describe, marginal entropy, …) are not changed.

## 2. Schema skeleton

The result stays one flat table (one row per column of the input).  Columns are grouped: a **profile** part that
does not depend on the engine, then one **block per engine**, prefixed by the engine's full name.  Full names, not
`df_` / `pl_` / `ch_`: technique results already use `df_a` for "dataframe".

### 2.1 Order and contents

| group | columns (new now in **bold**; *italic* = reserved, added by the named ToDo) |
|---|---|
| identity | `column`, `status`, `dtype`, `first_row` (streaming only) |
| profile | `n_rows`, `n_null`, Describe's value block (`n_unique` … `iso_n_midnight`), `n_midnight` |
| profile: entropy | **`entropy`, `entropy_low`, `entropy_high`, `entropy_method`** |
| profile: exemplars | **`exemplar_row`, `exemplar_id`, `exemplar_value`, `null_exemplar_row`, `null_exemplar_id`** |
| profile: order | *run columns, e.g. `n_runs`, `mean_run_length` (1a)*; *delta / double-delta statistics (3)* |
| profile: original sizes | `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes`; *LZ4 sizes (1f)* |
| profile: inner level | `inner_n_values`, `inner_n_null`, `inner_` + Describe's value block |
| profile: frequencies | `top_k`, `inner_top_k` |
| shared recommendation | `nullable`, `lossy_formatting` |
| DataFusion | `datafusion_type`, `datafusion_size_bytes`, `datafusion_size_zstd_bytes`; *`datafusion_encoding` (1a), `datafusion_size_lz4_bytes`, `datafusion_ipc_compression`, `datafusion_ipc_zstd_level` (1f)* |
| Polars | `polars_type`, `polars_size_bytes`, `polars_size_zstd_bytes`; *`polars_size_lz4_bytes`, `polars_ipc_compression`, `polars_ipc_zstd_level` (1f)* |
| ClickHouse | *`clickhouse_type`, `clickhouse_codecs`, `clickhouse_size_bytes` (1b, 1f), `clickhouse_skip_indexes` (1c–e)* |
| evidence | `candidates` |
| streaming only | `n_sampled_rows`, `n_sampled_blocks` |

Reserved columns are documented here and **not** emitted until their ToDo lands: later changes are additive.  One
definition in `recommenders/schema.rs` (`recommender_fields`) keeps serving both recommenders.

### 2.2 Renames (this increment)

| old | new |
|---|---|
| `rec_arrow_type`, `rec_arrow_size_bytes`, `rec_arrow_size_zstd_bytes` | `datafusion_type`, `datafusion_size_bytes`, `datafusion_size_zstd_bytes` |
| `rec_polars_type`, `rec_polars_size_bytes`, `rec_polars_size_zstd_bytes` | `polars_type`, `polars_size_bytes`, `polars_size_zstd_bytes` |
| `rec_nullable`, `rec_lossy_formatting`, `rec_candidates` | `nullable`, `lossy_formatting`, `candidates` |

The candidate struct's fields (`arrow_type`, `rule`, …) are unchanged.  No aliases for the old names.

### 2.3 Reserved ClickHouse columns (types fixed now, semantics in ToDo 1's specs)

- `clickhouse_type`: String, the full column type including wrappers (e.g. `LowCardinality(Nullable(String))`).
- `clickhouse_codecs`: String, the codec chain as written in DDL (e.g. `GCD, Delta, ZSTD(3)`); null = default codec.
- `clickhouse_size_bytes`: UInt64, predicted compressed size.
- `clickhouse_skip_indexes`: `List<Struct{kind: Utf8, granularity: UInt32, max_size: UInt32, fpr: Float64,
  bytes: UInt64, skip_rows: Float64, skip_values: Float64}>` — `kind` ∈ {minmax, set, bloom_filter}; fields that do
  not apply to a kind are null; `skip_rows` / `skip_values` are the predicted shares of point queries skipped (by
  row frequency / uniform over distinct values, as the skip-index techniques report them).

## 3. Entropy (ToDo 4)

### 3.1 Definition

Shannon entropy in bits of the column's top-level values, null counted as one category:
`H = log2(n) − (1/n)·Σ c·log2(c)` over the counts c of each distinct non-null value and of null (n = `n_rows`).
Values are identified as in Describe's frequency map (floats canonical, strings / nested values hashed).  `H = 0`
for zero rows.  Inner (list) levels: not now.

Columns: `entropy` (Float64), `entropy_low`, `entropy_high` (Float64; the interval), `entropy_method` (String ∈
{`exact`, `sample`, `ams`}).  Exact: `low = high = entropy`.  Every bound is clamped to `[0, log2(n)]`.
Ineligible columns: all four null.

### 3.2 One-shot

The counts come from the frequency map that Describe's statistics already build (`techniques/.../describe/frequencies.rs`,
reached through `recommenders::engine::prepare`) — the same `encode_series` keys `marginal_entropy.rs` hashes — plus
`n_null` as one more count.  They are reduced by the shared, SIMD `entropy_from_counts_iter`
(`common/entropy_math.rs`), the reduction `marginal_entropy` uses: O(distinct), no second encode / hash pass over the
rows (calling `marginal_entropy_impl` would repeat the O(rows) counting Describe has done).  Zero rows → 0 (the
reduction needs r > 0).  Method `exact`.  The streaming exact phase (§3.3) feeds the sample's counts to the same
reduction.

### 3.3 Streaming

- **While the distinct sample is exact** (it holds every distinct value's count): the same formula over the sample's
  counts plus the running `n_null`.  Method `exact`; equal to one-shot.
- **After it overflows**, keyword `entropy_estimator` ∈ {`"sample"`, `"ams"`} chooses the estimator.  The default is
  `"ams"`, decided by the benchmark (§3.4, outcome in §8).
  - `sample`: the bottom-k distinct sample holds exact counts for a uniform random sample of k′ distinct values
    (`recommenders/streaming/distinct_sample.rs`).  `T = Σ c·log2 c` over all non-null values is estimated as
    `T̂ = (D̂ / k′)·Σ_sample c·log2 c`, D̂ the HyperLogLog estimate (p = 14) of the non-null distinct values, NOT `est_cardinality` (Describe's Schnabel/Chao1 rule overshoots at d/n < 0.5: measured 17% on uniform, 44% on Zipf 1.1).  Interval: ±3σ with
    `Var(T̂) = D̂²·(1 − k′/D̂)·s²/k′`, s² the sample variance of the terms `c·log2 c`, evaluated at D̂ ± 3σ_HLL as well (floored at the distinct values proven seen, capped at the non-null rows); the interval is the envelope of the three.  The null term is exact.  No extra per-row work.
  - `ams`: the CCM sketch of `techniques/single_column/unordered/ams_entropy.rs` (`AmsEntropy`, 4 096 estimators),
    fed every row of every eligible column from the first batch (null → one reserved key); discarded when the
    sample stays exact.  The estimate is the mean of the per-estimator terms; interval ±3σ from their sample
    variance (the estimator exposes the terms; its estimate is unchanged).
- The sketch is per column and needs rows in order; batches are absorbed in order, columns in parallel, as today.

### 3.4 Benchmark and default

`tests/performance/benchmark_streaming_entropy.py` (standalone, results → `results/streaming_entropy.arrow`): both
estimators against the exact entropy on uniform (D = 10⁴, 10⁵, 10⁶), Zipf (α = 0.8, 1.1, 1.5) and near-constant
streams of 10⁶ rows, and on `datagen.large_dataset_mixed()`; reported per shape: relative error, interval width,
coverage (exact inside the interval), add() time and peak memory.  **Rule:** the default is `sample` unless its
relative error exceeds 5% on a shape where `ams` stays within 5%; then `ams`.  The outcome and the chosen default are
written into this spec (§8) and CLAUDE.md.

## 4. Exemplars (ToDo 2)

### 4.1 Columns

| column | type | meaning |
|---|---|---|
| `exemplar_row` | UInt64 | row of the first non-null value, in frame / stream order (0-based, the same counter as `first_row`) |
| `exemplar_id` | Struct of the id columns, or Null | the id columns' values at `exemplar_row` |
| `exemplar_value` | String | the value at `exemplar_row`, as arrow-rs display text (`arrow_cast::display::ArrayFormatter`) |
| `null_exemplar_row` | UInt64 | row of the first null |
| `null_exemplar_id` | as `exemplar_id` | the id columns' values at `null_exemplar_row` |

- First-seen rows make one-shot and streaming identical for any batch split, at O(1) memory per column.
- `exemplar_value` covers every type, nested included (e.g. `[1, 2]`, `{a: 1}`); binary renders as hex.  For scalar
  text and numbers it equals the `min` / `max` rendering.  Not truncated.
- A column with no nulls: `null_exemplar_*` null.  No non-null values: `exemplar_*` null.  Ineligible column: all
  five null.  Lists: only top-level nulls count.

### 4.2 Identifier columns

- Keyword `id_columns: Sequence[str] | None = None` on both recommenders (Python), `id_columns` in the API, C ABI
  and Java (`idColumns`).
- With id columns: `exemplar_id` / `null_exemplar_id` are `Struct{<id column>: <its Arrow type>, …}` in the given
  order; a null id value is a null field.  Without: both columns have Arrow type Null (all null).
- Validation (InvalidInput / ValueError): duplicate names; a name missing from the frame (one-shot) or from any added
  batch (streaming); an id column whose type changes between batches.  Id columns are also profiled like any other
  column.  Int128 / UInt128 id columns are already rejected by the Python wrapper.
- Streaming keeps, per column, the first non-null and first null row numbers and, when each is first seen, a one-row
  slice of the id columns (and the rendered value).  The result concatenates the slices.

## 5. Plumbing

- Rust: `recommenders/schema.rs` (fields, renames), `recommenders/engine/` (one-shot entropy, exemplars),
  `recommenders/streaming/` (`partial.rs` running state, `distinct_sample.rs` estimator, AMS wiring),
  `techniques/.../ams_entropy.rs` (expose the per-estimator terms for the interval — no behaviour change),
  `bindings/api/recommenders.rs`, `bindings/python.rs`, `bindings/capi.rs`.
- C ABI: `analytics_oneshot_recommender_new` and `analytics_streaming_recommender_new` gain
  `const char *const *id_columns, size_t n_id_columns`; the streaming one also `uint8_t entropy_estimator`
  (0 sample, 1 ams).
- Java: `OneShotParams` / `StreamingParams` gain `idColumns` (`List<String>`, default empty) and, streaming,
  `entropyEstimator` (enum `SAMPLE`, `AMS`).
- Python: keywords on both wrappers; `analytics.recommend.to_arrow` unchanged.

## 6. Testing

- **Entropy, one-shot:** against `marginal_entropy` (`analytics._plugin.marginal_entropy`, the independent
  cross-check: same keys and null policy, its own counting) to RTOL 1e-12 on `describe_mixed`, `large_dataset` and
  `large_dataset_mixed`; known answers (constant → 0, all distinct → log2 n,
  half null half one value → 1, zero rows → 0); ineligible → null.
- **Entropy, streaming:** equal to one-shot (method `exact`) while the sample is exact, for several batch splits; after
  overflow, for each estimator on a uniform and a Zipf(1.1) stream (≥ 10⁵ distinct values): relative error ≤ 5% and
  the exact value inside `[entropy_low, entropy_high]`; `entropy_method` names the estimator.
- **Exemplars:** known answers (first non-null / first null rows, values, id structs with mixed id types and null id
  values, no id columns → Null-typed columns, all-null and no-null columns, lists with inner nulls); one-shot ==
  streaming for several batch splits including splits that put the first null in a later batch; missing / duplicate
  id columns raise.
- **Schema:** both recommenders' column names and order equal §2.1 (contract test); `to_arrow()` exports.
- **Parity spec** (2026-10-01): its column lists use the new names and include entropy (exact phase) and exemplars.
- **Java:** existing tests on the new names; one test each recommender for `idColumns` and the exemplar columns;
  streaming `entropyEstimator`.
- **Rust unit tests:** the entropy formula, the sample-estimator variance on a hand-worked sample, the exemplar state
  across batches.

## 7. Documentation

CLAUDE.md: the OneShotRecommender / StreamingRecommender paragraphs (new keywords and columns, renames), the
benchmark, ToDo 4 and 2 marked done in ToDo.md with a pointer here; ToDo 1.0 marked "skeleton done (this spec); the
ClickHouse mapping, thresholds, cost model and handoff move to ToDo 1's specs".

## 8. Benchmark outcome

Run 2026-10-18 on 4 logical CPUs (Windows 10), `tests/performance/benchmark_streaming_entropy.py`: one column of 10^6 rows (large_dataset_mixed: 50 000 rows x 101 columns, error = max over columns), 10 batches, `reservoir_rows=0`, each estimator in its own process. Errors are against the one-shot exact entropy; coverage is the share of columns whose interval contains it.

| shape | estimator | max rel. error | coverage | add (s) | peak above data (MB) |
|---|---|---|---|---|---|
| uniform 1e5 | sample | 0.19% | 100% | 0.044 | 28.6 |
| uniform 1e5 | ams | 0.26% | 100% | 0.123 | 31.8 |
| uniform 1e6 | sample | 0.07% | 100% | 0.047 | 30.2 |
| uniform 1e6 | ams | 0.02% | 100% | 0.145 | 35.3 |
| zipf 0.8 | sample | 9.64% | 100% | 0.044 | 30.4 |
| zipf 0.8 | ams | 0.06% | 100% | 0.122 | 30.4 |
| zipf 1.1 | sample | 9.63% | 100% | 0.037 | 30.4 |
| zipf 1.1 | ams | 0.10% | 100% | 0.121 | 30.4 |
| zipf 1.5 | sample | 2.70% | 100% | 0.029 | 30.4 |
| zipf 1.5 | ams | 0.53% | 100% | 0.087 | 30.4 |
| large_dataset_mixed | sample | 0.12% | 100% | 0.159 | 150.8 |
| large_dataset_mixed | ams | 0.75% | 100% | 0.936 | 271.8 |

Never overflowed (the sample held every distinct value, method `exact`, nothing estimated): uniform 1e4 and near-constant.

**Decision: the default is `ams`.** The rule fires: `sample` exceeds 5% on Zipf 0.8 and Zipf 1.1 (9.6%) where `ams` stays below 0.2%, and `ams` is within 0.8% everywhere. Cost: `ams` adds a per-row hash and update, so `add()` is about 2.5-3x slower on single columns and ~6x on the 101-column frame (0.94 s vs 0.16 s), and uses ~120 MB more peak memory on that frame (sketches for 101 columns); `sample` stays available for callers who cannot afford it.  Transient cost not covered by the measurement (10 batches of 100 000 rows): with `ams` each `add` also materialises one 8-byte key per row per column until its batch is absorbed (≈ 800 MB for a 101-column batch of 1 000 000 rows), so wide frames should be added in smaller batches.  Feeding the sketch inside the per-column parallel absorb instead would avoid it; not done.
