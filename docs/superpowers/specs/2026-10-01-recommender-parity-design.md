# Recommender parity — design

Status: approved design, not yet planned.
Builds on: Describe (2026-09-26-describe-technique-design.md), Recommend
(2026-09-26-recommend-technique-design.md), the Arrow FFI interface and Java binding
(2026-09-27, 2026-09-28) and the Streaming Recommender (2026-09-29-streaming-recommender-design.md).

## 1. Goal

Close the metric gaps between the one-shot recommender (`describe_and_recommend`, the Describe /
Recommend techniques) and the `StreamingRecommender`, so that both emit one shared set of columns
with the same names, types, order and rules.

Principle: the recommenders run on evolving data. A deterministic count is not a guarantee about
future data, and every recommendation is used alongside another control (a schema guarantee, a
contract, a monitor). Approximate or probabilistic statistics are therefore acceptable wherever their
error is stated. Within the streaming recommender, simple and internally consistent beats matching
one-shot exactly. One-shot keeps exact counts, which serve as the reference for comparison.

In scope:
- a standalone HyperLogLog (`src/hll.rs`) and a bottom-k distinct sample (`src/distinct_sample.rs`,
  replacing `partial::Distinct`);
- one estimator rule and one `class` rule, in Rust, shared by both recommenders;
- `min` / `max` as Arrow-rendered values in both, replacing `argmin` / `argmax`;
- exposing the float, numeric-string / ISO scanner and `n_midnight` statistics in streaming;
- removing `entropy`, top-5, `f1` / `f2`, `capture_history`, the separate Chao1 / Schnabel columns,
  `population_rows` and Duj1 from both;
- C ABI, Java and Python binding updates; tests; benchmarks.

Out of scope:
- `inner_*` (list inner-value) statistics in streaming — a separate spec;
- a public entry point for `Hll` (the module is built so one can be added later);
- Count-Min / conservative-update sketches: no gap needs a point query (§3.3).

## 2. Decisions

| # | Decision |
|---|---|
| D1 | One spec for: HLL + distinct sample, estimator rule, `class`, exposed stats, `min`/`max` values, `sum_len_unique`, estimator columns. `inner_*` deferred. |
| D2 | Streaming counts distinct values with HLL (n_unique) plus a bottom-k distinct sample (f1, f2, capture history). |
| D3 | Cardinality ratio d/n ≥ 0.5 → the estimate is the observed count (one-shot) or HLL (streaming). Below 0.5 → Schnabel or Chao1, floored by the observed / HLL count. |
| D4 | `entropy` and top-5 dropped from both recommenders. |
| D5 | Estimator columns: `n_unique`, `unique`, `est_cardinality`, `est_low`, `est_high`, `est_method`, `estimates_agree`. The `chao1*`, `schnabel*`, `f1`, `f2`, `capture_history` columns are dropped. |
| D6 | `min` / `max` rendered with Arrow's cast to string, in every implementation. |
| D7 | `population_rows` and Duj1 removed from both recommenders and every binding. |
| D8 | Sample size k = max(`categorical_threshold`, 1000); HLL precision fixed at p = 14. No new keywords. |

## 3. New Rust structures

### 3.1 `src/hll.rs` — HyperLogLog

```rust
pub struct Hll { p: u8, registers: Vec<u8> }   // 2^p one-byte registers
impl Hll {
    pub fn new(p: u8) -> Self;                  // 4 ≤ p ≤ 18
    pub fn insert(&mut self, hash: u64);        // bucket = top p bits; rank = leading zeros of the rest + 1; keep max
    pub fn merge(&mut self, other: &Hll);       // register-wise max (exact; panics on p mismatch)
    pub fn estimate(&self) -> f64;              // Ertl's improved estimator (2017): α∞·m²/z from the
                                                //   register histogram; no bias tables or range switches
    pub fn std_error(&self) -> f64;             // 1.04 / sqrt(m)
}
```

No Polars or Arrow types, so a later `api.rs` entry point or per-column technique can wrap it. With a
64-bit hash no large-range correction is needed. p = 14 → 16 KB per column, σ ≈ 0.81%.

### 3.2 `src/distinct_sample.rs` — bottom-k distinct sample

Replaces `partial::Distinct` and now covers every eligible dtype, not only text.

- Input: the per-batch `KeyStat`s that `partial.rs` already builds from
  `frequency_map(encode_series(s), seed, offset)` — the existing u64 keys (`shared::encode_series`),
  first occurrence, count, row-based capture mask and length. No new key encoding.
- Ordering hash: `h = foldhash::quality::FixedState::default().hash_one(key)`. Integer keys are raw
  values and not uniform, so they are always mixed; the same `h` feeds `Hll::insert`. Neither structure
  is persisted, so foldhash's instability across versions does not matter.
- Entry per sampled value: key → count capped at 3, capture mask (3 bits), length.
- Bound: at most k entries. When full, a max-heap on `h` evicts the largest; τ = the largest kept `h`
  normalised to [0, 1); a value with `h` above τ is rejected. A value in the final sample has been
  counted since its first occurrence (τ only falls), so its count and mask are exact.
- **Exact phase**: until the first eviction the sample holds every distinct value. During it the
  sample also keeps what the dictionary rules need — `few` (≤ 5 values, first-occurrence order),
  the `ViewSim` view blocks and an exact `sum_len_unique`. The first eviction clears and stops them
  (as `overflowed` does today); dictionaries are rejected from then on since d > k ≥ threshold.
- **Sampling phase**: f1, f2 and the capture history are the sample's values scaled by 1/τ;
  `sum_len_unique` = (mean sampled length) × the HLL estimate.
- `absorb(keys)` takes one batch's `KeyStat`s (k is fixed by `new(k)`); it keeps the k smallest `h`
  whatever the batch order (no `merge`: §13.6).
- k = max(`categorical_threshold`, 1000). Memory ≈ 45–50 bytes per entry (map slot + key, heap entry,
  hash-map overhead; ≈ 0.5 MB per column at k = 10 000, worst case).

`LevelStats` holds one `Hll` and one `DistinctSample` per eligible level, fed with each batch's
distinct keys (O(distinct per batch), not O(rows)). One-shot uses neither: it keeps exact counting
on `frequency_map`.

### 3.3 Sketches considered and not used

Count-Min and Count-Min with conservative update answer point queries ("how often did x occur?"):
they cannot enumerate values, and their additive error εN swamps the counts of 1 and 2 that f1 / f2
need. Conservative update tightens heavy-hitter counts but keeps the εN bound and merges only as a
plain sum. Space-Saving would serve top-5, which D4 drops. None is needed.

## 4. Estimator rule

One function in `cardinality_estimators.rs`, used by one-shot Rust and streaming; the Python
`describe/estimators.py` keeps the reference version.

Inputs: n = non-null count; d = distinct count — exact in one-shot and in streaming's exact phase,
the HLL estimate (floored at k + 1) in the sampling phase; floor = d when exact, and when HLL
  max(d − 3σ, seen) — seen = k + 1, the distinct values the sample proves.

1. n = 0 → `est_cardinality` 0, interval [0, 0], `est_method` `observed`.
2. d / n ≥ 0.5 → `est_cardinality` = d; `observed` with [d, d], or `hll` with [floor, d + 3σ].
3. d / n < 0.5 → Schnabel when its existing validity test passes, else Chao1 (each with its 95%
   interval as today); `est_cardinality = max(estimate, d)`, `est_low = max(estimator low, floor)`,
   `est_high = max(estimator high, est_cardinality)`. `est_method` `schnabel` or `chao1`.
4. `estimates_agree` = the Chao1 and Schnabel intervals overlap; null unless rule 3 applies and
   Schnabel is valid.
5. `unique` = d == n (one-shot, streaming exact phase); in the sampling phase, every sampled value
   has count 1.

`est_method` ∈ {observed, hll, schnabel, chao1}. Duj1 and `exact` are gone (D7). The dictionary
rules keep reading `est_high` and floor it at `n_unique` as today.

## 5. `class`

A new `classify` in recommend.rs, used by one-shot Rust and streaming; the Python Describe base keeps
the reference. First match wins:

| class | condition |
|---|---|
| null | `n_null == n_rows` |
| constant | d == 1 |
| boolean | d == 2 |
| ordinal | every non-null value is a whole number and 0 ≤ min, max ≤ 2·`n_rows` |
| categorical | `est_cardinality ≤ categorical_threshold` |
| discrete | otherwise |

Whole numbers: integers; Decimal with scale 0; floats with `n_fractional == n_nan == n_inf == 0`;
strings with `n_numeric_int == n`, using `numeric_int_min` / `numeric_int_max`. Temporal dtypes never
qualify. N in the ordinal rule is now always `n_rows` (D7).

In streaming d == 1 and d == 2 fall in the exact phase, so they are exact; the ordinal inputs are
existing running statistics (`int_range`, `float_range`, the scanner counts).

## 6. `min` / `max`

String columns rendered with Arrow's cast to string: arrow-rs `cast(…, Utf8)` in Rust,
`pyarrow.compute.cast(…, pa.string())` in the DataFusion and Polars Describe implementations.

Ordering is unchanged: numbers by value with NaN excluded; strings by bytes; booleans false < true;
Categorical by its string value, Enum by category order (§13.2). Lists have no `min` / `max`; `inner_min` / `inner_max`
cover their inner values (one-shot only).

- One-shot Rust: finds the argmin / argmax index as today, then renders that row in describe.rs.
  `argmin` / `argmax` leave the output.
- Streaming: `has_extremes` gains String, Categorical, Enum, Boolean and Binary. Text extremes are kept with
  a new `Key::S` (the bytes), compared per batch — one stored string per extreme. Rendering stays
  `render` in streaming.rs. The stored value is the one rendered value (a Categorical / Enum extreme is cast to a plain string first, so no dictionary is retained).

## 7. Output columns

Both recommenders emit these shared columns, with the same names, types and order:

| group | columns |
|---|---|
| counts | `n_rows`, `n_null` |
| cardinality | `n_unique`, `unique`, `est_cardinality`, `est_low`, `est_high`, `est_method` (String), `estimates_agree` |
| extremes / lengths | `min`, `max` (String), `min_len`, `max_len`, `sum_len`, `sum_len_unique`, `gcd` |
| floats | `n_nan`, `n_inf`, `n_fractional`, `max_frac_digits`, `n_f32_inexact` |
| text scanners | `n_numeric`, `n_numeric_int`, `n_leading_zero`, `numeric_int_min`, `numeric_int_max`, `numeric_max_int_digits`, `numeric_max_frac_digits`, `numeric_min_frac_digits`, `numeric_max_sig_digits`, `n_iso_date`, `n_iso_time`, `n_iso_datetime`, `n_iso_datetime_tz`, `iso_max_frac_digits`, `iso_max_sig_frac_digits`, `iso_n_offsets`, `iso_n_midnight` |
| temporal | `n_midnight` |
| class | `class` (String) |
| sizes | `size_bytes`, `size_zstd_bytes`, `size_polars_bytes`, `size_polars_zstd_bytes` |
| recommendation | every `rec_*` column |

Types follow today's Describe metrics; `est_method` and `class` become plain String in one-shot too
(they were Polars Enums, which cross Arrow as dictionaries). In streaming, estimated `n_unique` and
`sum_len_unique` are rounded to UInt64.

- One-shot removes: `argmin`, `argmax`, `top5_idx`, `top5_count`, `top5`, `entropy`, `f1`, `f2`,
  `capture_history`, `chao1`, `chao1_low`, `chao1_high`, `schnabel`, `schnabel_low`, `schnabel_high`,
  and their `inner_` twins. It keeps `inner_*` for lists, pruned the same way (`inner_min` /
  `inner_max` as values; inner estimator columns by the rule in §4).
- Streaming adds `unique`, `estimates_agree`, `class`, the float, scanner and `n_midnight` columns,
  and `min` / `max` for text and boolean.
- Streaming-only columns stay: `column`, `status`, `dtype`, `first_row`, `n_sampled_rows`,
  `n_sampled_blocks`. `distinct_overflowed` is removed (`est_method == "hll"` carries it).

## 8. Where conclusions are computed

Today the Rust one-shot output carries metrics only (argmin indices, f1 / f2), and the Python
Describe base renders extremes, estimates and `class` — so C and Java callers get none of them.

- Rust one-shot (`describe_and_recommend`, `describe_columns`) outputs the shared conclusion columns
  itself, using the same Rust functions as streaming (§4, §5, §6). C and Java see the same table as
  Python.
- The Python Describe base keeps its own estimator and `classify` as the reference, used by
  `DescribePolars` (★) and `DescribeDataFusion`. The Rust-backed classes pass Rust's columns through.
- One-shot Recommend's `few_distinct` (boolean pairs) changes from `top5_idx` to the distinct values
  in first-occurrence order when d ≤ 5, from `frequency_map`'s `first` — the same list, in the same
  order, as streaming's `few`.

## 9. Bindings and breaking changes

- Describe / Recommend output schema: §7.
- `population_rows` removed: Python `Describe` / `Recommend` keywords (including the dict form);
  the C ABI `analytics_describe_and_recommend` loses the parameter (signature change); Java
  `Params.populationRows` and `Analytics.describeAndRecommend`.
- Streaming output: `distinct_overflowed` removed; columns added per §7.
- No new keywords (D8); no new error kinds.

## 10. Testing

Rust unit tests:
- `hll.rs`: estimate within ±3σ of truth for 0, 1, 100, 10⁴, 10⁶ distinct values over several
  seeds; `merge` equals single-sketch insertion; the small-range (σ) and saturation (τ) corrections.
- `distinct_sample.rs`: exact-phase counts, f1, f2, capture history equal `frequency_map`'s;
  order-independent merge; sampling-phase scaled f1 / f2 / history within tolerance on a seeded 10⁶
  stream; first eviction ends `few`, `ViewSim` and exact `sum_len_unique`; k floor of 1000.
- Estimator rule: every branch, the floor on the estimate and on `est_low`, `estimates_agree`.
- `classify`: every class; integer-valued strings and floats as ordinal; temporal never ordinal.
- `partial.rs`: string, categorical and boolean extremes across batches.

Python accuracy tests:
- `test_describe.py`: contract schema; entropy tolerance and top-5 rendering removed; known answers
  for `min` / `max` as Arrow strings (datetime, duration, decimal, float, boolean); conclusions
  covering `class` and the new estimator rule; Rust conclusion columns equal the Python base's.
- `test_recommend.py`: schema; `few_distinct` from first occurrence (boolean pairs unchanged).
- `test_streaming_recommend.py` — parity against one-shot on large_dataset.arrow in batches:
  - equal: `n_rows`, `n_null`, `min`, `max`, lengths, `sum_len`, `gcd`, float and scanner columns,
    `n_midnight`, `rec_*` types;
  - equal in the exact phase: `n_unique`, `unique`, `class`, `sum_len_unique`;
  - HLL `n_unique` within ±3σ of one-shot's exact `n_unique` in the sampling phase;
  - `est_cardinality` inside its own interval;
  - a high-cardinality fixture forcing the sampling phase: `est_method == "hll"`, `unique` on a key
    column, estimated `sum_len_unique` within 5% of exact.

Java: JUnit checks of `class`, `min` and `est_method` for one-shot and streaming; `populationRows`
removed from tests.

## 11. Performance

- Streaming now feeds HLL and the sample for every dtype, from each batch's distinct keys. Re-run
  `benchmark_streaming_recommend.py`; report the change in add throughput.
- One-shot loses the top-5 sort and Duj1 and gains Rust rendering of `min` / `max`. Re-run the
  Describe and Recommend benchmarks; no regression expected.

## 12. Documentation

CLAUDE.md's Describe, Recommend and Streaming Recommend sections; amendments in the Describe,
Recommend and streaming specs pointing here.

## 13. Amendments (planning, 2026-10-01)

1. **Rendering is arrow-rs's, everywhere.** Arrow C++ (pyarrow) and arrow-rs format some
   types differently (timestamps `2024-01-02 03:04:05` vs `2024-01-02T03:04:05`, floats,
   durations), so "Arrow's cast" is not one format. The canonical rendering is arrow-rs's cast
   to Utf8 (`recommend::render_value`). The Python reference implementations render through
   a private Rust helper (`_plugin.render`), not `pyarrow.compute.cast`. This is a format
   convention, not a computation, so the references stay independent where it matters.
2. **Ordering of Enum extremes** is by category order (its physical code), as one-shot does
   today — not by string value (§6 corrected). Categorical orders by string value.
3. **Nested extremes are dropped.** List / Array / Struct columns have no `min` / `max` in either
   recommender (one-shot used to order them by Polars' row encoding).
4. **Sampling-phase scaling.** f1, f2 and the capture history are the sample's values scaled by
   (HLL estimate ÷ sample size) — the sample's fractions applied to the HLL count — rather than by
   1/τ. Both estimate the same quantity; this one is consistent with the reported `n_unique`.
5. **`n_unique` floor in the sampling phase**: max(HLL estimate, k + 1). At least k + 1 distinct
   values have been seen, so the dictionary gate always rejects (k ≥ `categorical_threshold`).
6. **No `DistinctSample::merge`.** Streaming absorbs batches in order; `absorb` keeps the k smallest
   hashes whatever the batch order, which the tests check instead.
7. **`classify` lives in `src/conclusions.rs`** with `whole_range` and `conclude`, not in recommend.rs.
8. **Python contract.** Describe implementations return private per-level `INPUTS` (`argmin`,
   `argmax`, `f1`, `f2`, `capture_history` and `inner_` twins) beside `METRICS`; the base derives
   the conclusions (`min`, `max`, `unique`, `est_*`, `estimates_agree`, `class`) and drops the
   inputs from the output. `describe_columns` (private) returns Rust's conclusions *and* the inputs,
   so a test compares Rust's conclusions with the Python base's on identical inputs (exact).
   `describe_and_recommend` returns Rust's conclusions only; `RecommendRust` passes them through.
   Agreement with the reference compares `min`, `max`, `unique`, `class` exactly and
   `est_cardinality` / `est_low` / `est_high` within the Schnabel tolerance (10%); `est_method` and
   `estimates_agree` depend on the seeded split and are not compared.
9. **`projected_population_bytes` stays** in `rec_candidates`; without `population_rows` it is the
   predicted size scaled by the estimated cardinality only.
