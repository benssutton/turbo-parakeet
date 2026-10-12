# Heavy hitters (top-k) sketches — design

Status: approved 2026-10-12. Adds a per-column technique `analytics.top_k` on the uniform contract
(spec 2026-09-24) with two sketch implementations and an exact Polars reference.

## 1. Goal

Find the k most frequent values of a column, with their counts, in memory that does not grow with the
number of distinct values. The eventual user is the StreamingRecommender: today its `top_k` is exact
but null for every column above `categorical_threshold` (spec 2026-10-07 §3), which is exactly the
high-cardinality columns of a stream of hundreds of millions of rows. ToDo item 1 asks for the top-5
values per column.

This spec covers two sketches, the technique that exposes them and a benchmark of accuracy, memory and
speed against an exact Polars count. Integrating the winner into the recommenders is a later spec,
decided on this evidence.

## 2. Algorithm choice

Two sketches are implemented and compared at **equal memory**:

- **HeavyKeeper** (Gong et al., "HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant Flows",
  USENIX ATC 2018). Count-with-exponential-decay. The paper reports better precision than
  SpaceSaving at equal memory, especially when the top-k frequencies are small relative to n. It has
  no worst-case bound and **does not merge**: decayed counts and bucket ownership cannot be combined.
- **SpaceSaving** (Metwally, Agrawal, El Abbadi 2005). Deterministic, with a per-item bound
  (true count ∈ [count − ε, count]), natively weighted, and **mergeable** with its bounds intact
  (Agarwal et al., "Mergeable Summaries", PODS 2012; SpaceSaving is isomorphic to Misra–Gries).
  Mergeability fits the StreamingRecommender: each batch already arrives as an exact (key, count) list.

Rejected:

- **Misra–Gries** gives the same guarantees as SpaceSaving, but only as an under-estimate with one
  global error. SpaceSaving's per-item ε is the more useful output.
- **Count-Min + heap.** The sketch merges, but the candidate heap does not, and Count-Min needs
  more memory for the same accuracy.

### 2.1 Memory budget

The one sizing keyword, `memory_bytes`, defaults to **max(16 KiB, k × 1 KiB)**. The 16 KiB floor is
the per-column HLL's size (p = 14). Each implementation sizes itself to the largest structure whose
accounted bytes (`bytes()`, §3.3) fit the budget. A budget too small for k entries is an error.

| k | Budget | SpaceSaving (~48 B / counter) | HeavyKeeper (8 B / bucket, after the heap of k) |
|---|---|---|---|
| 5 | 16 KiB | ~340 counters | ~2 × 1,000 buckets |
| 256 | 256 KiB | ~5,400 counters | ~2 × 15,000 buckets |

## 3. Rust

Two stand-alone files in `src/techniques/single_column/unordered/`, beside `hll.rs`, plus a shared
`top_k.rs` (the indexed min-heap `Counters` both use, a SplitMix64 generator and the kernels' output
builder). They import only `common` and `techniques`. Keys are the u64 keys of `encode_series` (as HLL and AMS use them). Every tracked item
carries a **row** where its value occurs, so the values can be rendered later.

### 3.1 `heavy_keeper.rs`

```rust
pub(crate) struct HeavyKeeper { /* d = 2 bucket arrays, indexed min-heap of k */ }
impl HeavyKeeper {
    pub(crate) fn new(k: usize, memory_bytes: usize, seed: u64) -> Result<Self, String>;
    pub(crate) fn insert(&mut self, key: u64, row: u64);
    pub(crate) fn top(&self) -> Vec<Item>; // count desc, row asc; error = None
    pub(crate) fn bytes(&self) -> usize;
}
```

This is the paper's final version (Algorithm 1, Optimisations I and II):

- **Buckets.** d = 2 arrays of w buckets, each holding a `u32` fingerprint and a `u32` count (8 B).
  The paper uses 16-bit fingerprints; 32 bits makes collisions about 65,000× rarer for the same
  bucket size. Each array index and the fingerprint come from independent seeded mixes of the key.
  Counts saturate at `u32::MAX`.
- **Per row, for each array j,** with C the bucket's count:
  - If the fingerprint matches and (the key is in the heap, the heap has room, or C ≤ n_min), then
    C += 1. This is *selective increment*: a bucket larger than n_min cannot belong to an outsider
    (the paper's prose, Optimisation II), so it is left alone. With C < n_min an outsider could
    never reach n_min + 1.
  - If the fingerprint does not match, C −= 1 with probability 1.08^−C. When C reaches 0, the bucket
    takes the key's fingerprint with C = 1.
  - An empty bucket (C = 0) takes the key with C = 1.
  - The key's estimate is the maximum C over the buckets that hold its fingerprint after the update.
- **Heap.** An indexed min-heap of k entries `(key, count, row)` with a key → slot map. n_min is its
  minimum count once it holds k entries.
  - A key already in the heap: count ← max(estimate, count).
  - A key not in the heap enters if the heap has room or estimate = n_min + 1, evicting the minimum.
    This is *fingerprint-collision detection*: an outsider can only exceed n_min by 1.
- **Randomness.** The decay coin and the hash salts come from a seeded generator whose seed is derived
  from the user `seed`, the domain tag 4 (AMS uses 1–3) and the column name, as in AMS. An estimate is
  deterministic for a given build. String keys come from foldhash, so a foldhash upgrade re-rolls
  them.
- **Sizing.** The heap's k entries are paid for first, and w is the largest width whose 2 × w buckets
  fit the rest. w < k is an error.
- **Parallelism.** No merge: a column is fed sequentially, row order. Columns run in parallel (rayon).

### 3.2 `space_saving.rs`

```rust
pub(crate) struct SpaceSaving { /* m counters in an indexed min-heap by count */ }
impl SpaceSaving {
    pub(crate) fn new(k: usize, memory_bytes: usize) -> Result<Self, String>;
    pub(crate) fn insert(&mut self, key: u64, row: u64, weight: u64);
    pub(crate) fn merge(&mut self, other: &SpaceSaving);
    pub(crate) fn top(&self) -> Vec<Item>; // count desc, row asc; error = Some(ε)
    pub(crate) fn bytes(&self) -> usize;
}
```

- **Counters.** m counters `(key, count, ε, row)` in an indexed min-heap by count, with a key → slot
  map. m is the largest count whose accounted bytes fit the budget. m < k is an error.
- **Insert** with a weight w:
  - A tracked key: count += w.
  - An untracked key with a free counter: (count, ε) = (w, 0).
  - An untracked key with no free counter: it replaces the minimum counter, with
    (count, ε) = (min + w, min).
- **Guarantee.** For every tracked key, true count ∈ [count − ε, count]. Every key with frequency
  > n/m is tracked.
- **Merge.** Let min_i be summary i's minimum count when it is full, else 0. Every key in either
  summary gets count = c₁ + c₂ and ε = ε₁ + ε₂, where a side missing the key contributes min_i to
  both. Keep the m largest by (count desc, row asc). The bound above still holds, and the result is
  deterministic.
- **Kernel.** A column is split into fixed chunks of 2²⁰ rows. Each chunk gets its own summary, built
  with global row numbers, and the summaries merge left to right in chunk order. Rayon runs across
  columns *and* chunks. The chunk size is fixed rather than set by the thread count, so results are
  deterministic. This is the parallelism HeavyKeeper cannot have, and the benchmark measures it.

### 3.3 Shared details

- `Item { key, row, count, error: Option<u64> }`. A reported `row` is *a* row holding the value, not
  necessarily the first. Sketch ties therefore break on an approximate first row; the exact reference
  breaks them on the true first row.
- `bytes()` is the deterministic accounting of the allocated capacity of the structure's vectors and
  hash map (capacity × entry size + control bytes), not a measurement. Sizing uses the same formula, so
  `bytes() ≤ memory_bytes` always.
- Nulls are skipped and never counted. `n_non_null` is exact.

### 3.4 API and binding

In `src/bindings/api/techniques.rs`:

- `heavy_keeper_top_k(batch, k, memory_bytes, seed)`
- `space_saving_top_k(batch, k, memory_bytes)`

Both return `col_name, rank, row, count, error, n_non_null, sketch_bytes`, one row per
(column, rank). A column with no non-null value returns one row with null rank / row / count / error,
so `n_non_null` and `sketch_bytes` are still reported. `k = 0`, or a budget too small for k entries →
`InvalidInput`. They are exposed in `bindings/python.rs`, wrapped in `analytics/_plugin.py` and
private. There is no C ABI or Java binding.

## 4. Python technique — `analytics.top_k`

| Class | Role |
|---|---|
| `TopKHeavyKeeperRust` | HeavyKeeper |
| `TopKSpaceSavingRust` | SpaceSaving |
| ★`TopKPolars` | exact reference |

`REFERENCE = "TopKPolars"`. Keyword-only constructor arguments: `k=5`, `memory_bytes=None`
(→ max(16 KiB, k × 1 KiB)), `seed=0` (HeavyKeeper only), `dominance_threshold=0.5`. `k < 1` or a
negative budget → `ValueError`.

**Reference.** Per column, with a row index: drop nulls, group by value, aggregate `count = len()` and
`row = min(index)`, sort by (count desc, row asc), take k.

**Eligibility.** The scalar dtypes `encode_series` keys: integers ≤ 64-bit, floats, Boolean,
String / Categorical / Enum, Binary, Decimal, Date / Datetime / Duration / Time. List / Array /
Struct, Null, Object and Int128 / UInt128 are ineligible.

**Output**, one row per column:

| Part | Columns |
|---|---|
| keys | `df_a, col_a` |
| status | computed / ineligible |
| descriptors | `dtype` (Python `str(dtype)`) |
| metrics | `n_non_null` UInt64; `top_k` `List<Struct<value: Utf8, count: UInt64, error: UInt64>>`; `sketch_bytes` UInt64 |
| conclusions | `dominated` Boolean |

- `top_k` holds min(k, tracked items) entries, ranked by (count desc, row asc). It is an empty list
  for an all-null or zero-row column.
- `error` is SpaceSaving's ε, 0 for the reference and null for HeavyKeeper.
- `sketch_bytes` is null for the reference.
- **Rendering.** Implementations return rows and counts only. The base `TopK` gathers those rows from
  the column and renders them with `_plugin.render` (arrow-rs text, as Describe's min / max). Text is
  identical across implementations, and Categorical / Enum render as labels.
- `dominated` = top-1 count / `n_non_null` ≥ `dominance_threshold`, null when the list is empty.
- `List<Struct>` exports to native Arrow, so `result().to_arrow()` needs no special handling.

**Agreement** (`TopK.agreement`, overridden as MinHash's and Bloom's are):

- **Precision.** A reported value counts as correct when the reference lists it. This is a lower
  bound of tie-aware precision, because the reference lists only k of the values tied at its k-th
  count, and agreement has no full counts. The accuracy benchmark computes the exact tie-aware
  figure (§6.2). Aggregate precision over all computed columns must be ≥ 0.9. Agreement test data
  therefore avoids heavy ties at the k-th count.
- **SpaceSaving:** for every reported value, true count ∈ [count − error, count]. No exceptions.
- **HeavyKeeper:** aggregate ARE (mean |count − true| / true over correct values) ≤ 0.05.
- `n_non_null` exact. `dominated` is not compared.

## 5. Tests

### 5.1 Rust unit tests

- **HeavyKeeper:**
  - exact on a stream with fewer distinct keys than buckets;
  - same seed gives the same top;
  - selective increment keeps a colliding mouse flow out of the heap;
  - finds the top-5 of a Zipf(1.1) stream.
- **SpaceSaving:**
  - exact while distinct keys ≤ m;
  - the [count − ε, count] bound holds on a random stream;
  - `merge` of two halves keeps the bound and tracks every key with frequency > n/m;
  - a weighted insert equals w repeated inserts;
  - the chunked kernel is deterministic.
- **Both:** `bytes() ≤ memory_bytes`; a budget too small for k is an error.

### 5.2 Python accuracy — `tests/test_top_k.py`

The four standard blocks:

1. Contract.
2. Agreement (§4).
3. Known answers, for every implementation:
   - fewer than k values gives an exact list;
   - a Zipf column's top-5;
   - ties ranked by first row;
   - all-null gives an empty list;
   - Categorical renders labels;
   - a too-small `memory_bytes` → `ValueError`.
4. Conclusions via `with_metrics`, including an empty-list row.

## 6. Benchmarks

### 6.1 Speed — `tests/performance/benchmark_top_k.py`

A `harness.run` script over the standard shapes (large_dataset.arrow, narrow / long, wide) plus
`Dataset("zipf 10M x 4", ...)`, built from four columns of `top_k_streams`. It reports times, the
1-thread re-run of both Rust classes and the algorithmic / parallel / total speedups.

### 6.2 Accuracy and memory — `tests/performance/benchmark_top_k_accuracy.py`

This follows `benchmark_ams_entropy.py`: one fresh child process per run. The child records its
working set after loading, its peak after `result()` (the same Windows / POSIX method) and the wall
time, then scores the sketches against the reference.

**Dataset.** A new `datagen.top_k_streams(n_rows=10_000_000, seed=42)`:

| Column | Content | What it tests |
|---|---|---|
| `zipf_15` | Zipf α = 1.5 over 1M values | heavy skew |
| `zipf_11` | Zipf α = 1.1 over 1M values | typical skew |
| `zipf_08` | Zipf α = 0.8 over 1M values | light skew; top frequencies ≪ n/m (SpaceSaving's weak case) |
| `flat` | uniform over 1M values | top-k barely distinguishable |
| `skewed` | 95% one value, the rest uniform over 50K values | one dominant value plus a long tail |
| `zipf_str` | `zipf_11` cast to String | hashed keys and rendering |

**Sweep.** k ∈ {5, 256} × budget ∈ {¼, 1, 4} × the default.

**Output.** `results/top_k_accuracy.parquet` (git-ignored), one row per
(dataset, column, k, budget, implementation):

- `precision` (tie-aware);
- `are`, `aae` (the paper's average relative and absolute errors over correct values);
- `max_bound_violation` (SpaceSaving: must be 0);
- `sketch_bytes`, `peak_mb`, `time_s`.

**Scale run.** `--rows N` (for example `--rows 100_000_000`) reruns the 1× budget on `zipf_11` alone.
It shows sketch memory flat at 16 KiB while the exact count grows with the distinct values.

## 7. Out of scope

- Integrating either sketch into the StreamingRecommender or OneShotRecommender (next spec). That spec
  will need HeavyKeeper's weighted insert (geometric skipping over decay trials) or SpaceSaving's
  per-batch merge.
- A C ABI or Java binding.
- Changes to Describe.

CLAUDE.md gains a **Top-k** entry under Per-column and the two modules in the project structure. A
§8 Results is added once the benchmarks have run, as in the AMS spec.

## 8. Results (2026-10-12)

Measured on 4 CPUs with `benchmark_top_k.py` (speed) and `benchmark_top_k_accuracy.py`
(accuracy and memory, 10M rows; scale run 100M rows). Budgets are ¼×, 1× and 4× the default
max(16 KiB, k × 1 KiB).

**Accuracy.** Tie-aware precision (reported values whose true count ≥ the true k-th count):

| Column | k | Budget | HeavyKeeper | SpaceSaving |
|---|---|---|---|---|
| zipf_15, zipf_11, zipf_str | 5 | all | 1.0 | 1.0 |
| zipf_15, zipf_11, zipf_str | 256 | 64 KiB | 1.0 | 1.0 / 0.957 / 0.957 |
| zipf_08 | 5 | 4 KiB / 16 KiB / 64 KiB | 1.0 / 1.0 / 1.0 | 0.8 / 1.0 / 1.0 |
| zipf_08 | 256 | 64 KiB / 256 KiB / 1 MiB | 0.977 / 0.996 / 0.996 | 0.484 / 0.977 / 1.0 |
| flat | 5, 256 | all | 0.0–0.016 | 0.0 |
| skewed | 5 | all | 0.2 | 0.2 |
| skewed | 256 | 64 KiB / 256 KiB / 1 MiB | 0.059 / 0.254 / 0.984 | 0.023 / 0.066 / 0.414 |

- Counts on found values are near exact for both sketches where precision is 1: mean relative
  error ≤ 0.05% at the default budget.
- SpaceSaving's bound held on every reported value in all 42 runs (`max_bound_violation` = 0).
  Where it is starved (zipf_08 at ¼×, flat, skewed), its over-counts are large (mean relative
  error 1.75 at zipf_08 k = 5, 4 KiB): the bound is honest but wide.
- Neither sketch separates near-tied values. On flat (counts ≈ 10 ± 3) and on skewed's uniform
  tail, both find only the dominant value. That is expected: those top-k sets are noise.

**Memory.** `sketch_bytes` stays within budget: 16 376 B at k = 5, 262 136–262 144 B at
k = 256. Process peak above the loaded data is about 94 MB for both sketches at 10M rows and
864 MB at 100M rows, against 68–286 MB and 227 MB for Polars. The one-shot kernels encode the
whole column into keys and null flags first (about 9 B per row), and that dominates. The 16 KiB
sketch only pays off when it is fed batch by batch.

**Speed** (median, 4 threads; total = fastest non-Rust ÷ Rust):

| Dataset | Polars | HeavyKeeper | SpaceSaving |
|---|---|---|---|
| large_dataset.arrow (50K × 101) | 446 ms | 94 ms (4.8×) | 149 ms (3.0×) |
| narrow 10M × 4 (1,000 values) | 351 ms | 372 ms (0.9×) | 1,133 ms (0.3×) |
| wide 200K × 100 (1,000 values) | 299 ms | 250 ms (1.2×) | 575 ms (0.5×) |
| zipf 10M × 4 | 1,055 ms | 382 ms (2.8×) | 940 ms (1.1×) |
| zipf_11 100M × 1 (scale run) | 1.97 s | 4.48 s | 3.11 s |

- SpaceSaving's within-column chunk parallelism does not make up for its per-row cost. It is
  2–3× slower than HeavyKeeper at 1 thread, and slower overall on every shape.
- On a single 100M-row column both sketches lose to Polars' parallel group-by. HeavyKeeper runs
  that column on one thread.
- `agrees` is false on every speed dataset because of ties: agreement's precision is a lower
  bound, and the flat and uniform columns' top-k are ties. No bound or count-error problem was
  reported.

**Conclusion.** For the StreamingRecommender, HeavyKeeper is the better sketch at equal memory:

- it is at least as accurate everywhere, and clearly more accurate on light skew (Zipf 0.8) and on
  small budgets;
- it is 2–3× faster per row.

SpaceSaving's advantages are its proven per-item bounds and that it merges. Neither matters
much here: the recommender absorbs each column sequentially, so no merge is needed. The
integration spec should therefore:

- use HeavyKeeper, with a weighted insert (geometric skipping over decay trials) to feed each
  batch's (key, count) list;
- report top-k only as an estimate above `categorical_threshold`;
- not expect any sketch to rank near-tied values.

The one-shot memory result also means the integration should feed the sketch per batch rather
than encode whole columns.
