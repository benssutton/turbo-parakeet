# Distinct-count, population-cardinality and marginal-entropy techniques (ToDo 17)

Status: implemented 2026-10-24.  Sub-project 2 of spec 2026-10-24-benchmark-uniformity-design.md: the Rust kernels that
had no technique, and so no benchmark, become three techniques on the uniform contract, each with accuracy tests and
a benchmark on the uplifted harness.

## 1. Goal

Three per-column techniques:

| package | question | kernels |
|---|---|---|
| `distinct_count` | how many distinct non-null values does the column hold? | HyperLogLog (`hll.rs`) |
| `population_cardinality` | how many distinct values does the population behind the column hold, including unseen ones? | Chao1, Schnabel (`cardinality_estimators.rs`) over Describe's frequency pass (`describe/frequencies.rs`) |
| `marginal_entropy` | Shannon entropy of the column (null one category) | exact (`marginal_entropy.rs`), AMS (`ams_entropy.rs`) |

All three follow the uniform contract (`Impl(**params).add({"name": frame}).result()`, one row per column, ineligible
columns listed, results export to native Arrow) and report metrics only: estimates with their interval, no
conclusions, no thresholds.  Describe keeps its own path (consolidating the counting code is ToDo 16).  Techniques
have no Java binding.

## 2. `turbo_parakeet.techniques.distinct_count`

| implementation | what it is | role |
|---|---|---|
| `DistinctCountHllRust` | `hll.rs`: `hash_key` of each encoded value into a `p`-bit HyperLogLog per 64K-row chunk (rayon, across columns and chunks), registers merged by max, Ertl's estimator.  `EXACT = False`. | estimator |
| `DistinctCountPolarsApprox` | Polars `approx_n_unique()` (Polars' HyperLogLog).  `EXACT = False`. | non-Rust competitor |
| ★ `DistinctCountPolars` | Polars `drop_nulls().n_unique()` | reference |

**Metrics:**

| column | type | meaning |
|---|---|---|
| `n_non_null` | UInt64 | non-null values |
| `n_distinct` | Float64 | the estimate (the exact count for the reference) |
| `low`, `high` | Float64 | ±3σ interval: `n_distinct · (1 ∓ 3 · std_error)`, `low` floored at 0 and `high` capped at `n_non_null`; both equal `n_distinct` for the reference |
| `std_error` | Float64 | relative standard error: 1.04 / √(2^p) for HLL, 0 for the reference, null for `approx_n_unique` (not exposed) |
| `sketch_bytes` | UInt64 | 2^p for HLL; null otherwise |

**Keywords:** `precision=14` (HLL only; 4 ≤ p ≤ 18, else ValueError).  **Eligible:** the dtypes the Rust encoder
keys (`ELIGIBLE` of Top-k); List / Array / Struct / Null / Object, Int128 / UInt128 and zero-row columns are
ineligible.  Values are keyed as `encode_series` keys them: floats canonical (−0.0 = 0.0, one NaN), Categorical /
Enum by string.  The reference canonicalises floats the same way before `n_unique`.

**Agreement** (`agreement`): `n_non_null` exact; HLL `n_distinct` within 4 × 1.04 / √(2^p) relative (≈ 3.3% at
p = 14) of the reference; `approx_n_unique` within 5%.  **Accuracy** (benchmarks): the default hook over
`n_distinct`.

## 3. `turbo_parakeet.techniques.population_cardinality`

| implementation | what it is | role |
|---|---|---|
| `PopulationCardinalityRust` | one frequency pass (`describe/frequencies.rs`: counts, f1, f2, the 3-way split capture history with `subset(seed, row)`), then `chao1` and `schnabel` from `cardinality_estimators.rs` | Rust kernel |
| ★ `PopulationCardinalityNumpy` | `np.unique(..., return_counts=True)` on the encoded values for d, f1, f2; capture subsets from SplitMix64 of `seed + row` in NumPy uint64 (bit-identical to `subset`); masks OR-ed per value; formulas from `describe/estimators.py` | reference |

**Metrics:** `n_non_null` (UInt64), `n_distinct` (UInt64, the observed d), `f1`, `f2` (UInt64), `chao1`,
`chao1_low`, `chao1_high`, `schnabel`, `schnabel_low`, `schnabel_high` (Float64).  Both estimators are always
reported; Schnabel is NaN when undefined (d/n ≥ 0.5 or no recapture).  Picking one by the d/n rule remains
Describe's / the recommenders' decision.

**Keywords:** `seed=0` (the split).  **Eligible:** as `distinct_count`.  **Agreement:** counts exact, estimates
RTOL 1e-9.  The NumPy reference keys values the same way as the Rust encoder (canonical floats, strings by value);
it keys by value, not by foldhash, which the counts do not depend on.

**Accuracy** (benchmarks): the default figures over the estimates (agreement with the reference) and, when the
dataset carries `truth` (§5), against the population's distinct count D, over the computed columns:
`observed_rel_err` = max |d − D| / D (the baseline any estimator should beat), `chao1_rel_err`,
`schnabel_rel_err` = max |estimate − D| / D (NaN estimates skipped; `schnabel_n_undefined` counts them), and
`chao1_coverage`, `schnabel_coverage` = the share of columns whose [low, high] contains D.

## 4. `turbo_parakeet.techniques.marginal_entropy`

| implementation | what it is | role |
|---|---|---|
| `MarginalEntropyRust` | `_plugin.marginal_entropy` (shared encoder, SIMD reduction) | exact Rust |
| `MarginalEntropyAmsRust` | `_plugin.ams_marginal_entropy` (CCM AMS sketch).  `EXACT = False`; scored also against `MarginalEntropyRust`. | estimator |
| ★ `MarginalEntropyPolars` | Polars `value_counts()` + `entropy(base=2)` per column — `PairwiseEntropyPolars`' marginal expression, reused | reference |

**Metrics:** `h` (Float64, bits), `n_rows` (UInt32).  Null is its own category (the joint entropies' policy).
**Keywords:** AMS `num_samples=4096`, `seed=0` (`_ams.check_ams_params`).  **Eligible:** what the encoder accepts,
in a frame with at least one row.  **Agreement:** exact RTOL 1e-9 / ATOL 1e-12; AMS RTOL 0.05, ATOL 1e-9 (as the
joint entropies).  **Accuracy:** default hook, `ACCURACY_METRICS = ("h",)`.  No Rust change: both bindings exist.

## 5. Harness: known-truth datasets

`Dataset` gains `truth: Callable[[], dict[tuple[str, str], float]] | None = None` — (frame, column) → the population's
distinct count.  When set, the parent calls it once per dataset and passes the dict to
`accuracy(frames, result, reference, truth=truth)` (not to the exact-Rust comparison).  `Technique.accuracy` gains
`truth=None`; the default ignores it.  Existing benchmarks are unchanged.

`datagen.population_samples(n_rows, populations, seed)`: `populations` is a sequence of `(D, shape)` with shape
`"uniform"` or `("zipf", α)`; column `p<i>` draws `n_rows` values with replacement from D labels (Int64, randomly
permuted so frequent values are not small integers) with those weights.  Returns `(frame, {column: D})`.  D is the
support size; under Zipf, rare labels may never appear, which is exactly what the estimators must account for.

## 6. Rust and bindings

- `techniques/single_column/unordered/distinct_count.rs`: `hll_distinct_impl(columns, p)` → `col_name, n_non_null,
  estimate, std_error, sketch_bytes`; per column, per 64K-row chunk sketches merged (the `frequencies.rs`
  streaming-phase pattern), columns in parallel.
- `techniques/single_column/unordered/population_cardinality.rs`: `population_cardinality_impl(columns, seed)` →
  `col_name, n_non_null, n_distinct, f1, f2, chao1, chao1_low, chao1_high, schnabel, schnabel_low, schnabel_high`,
  reusing Describe's frequency pass and the two estimator functions (no copy of either).
- `bindings/api/techniques.rs`: `hll_distinct(batch, p)`, `population_cardinality(batch, seed)` (InvalidInput for
  p outside 4..=18); `bindings/python.rs` wraps both; `_plugin.hll_distinct`, `_plugin.population_cardinality`.
- Layering as today (`scripts/check_layering.py`).  Rust unit tests: HLL estimate on small exact cases and a
  merge-equals-single-sketch check; population cardinality equal to Describe's f1 / f2 / capture history on the
  same column.

## 7. Tests (the four-block convention)

Each package: contract (every implementation), reference agreement (every non-reference implementation, on
`mixed_dtypes` and `large_dataset`), known answers, and no conclusions block (none exist).  Known answers:

| technique | known answer |
|---|---|
| distinct_count | a column with 50 distinct values among 1 000 rows (HLL within 1% of 50 at p = 14: almost every register is distinct at this size); nulls excluded; −0.0 / 0.0 and NaNs one value each |
| population_cardinality | f1 / f2 counted by hand; Chao1 against the formula; Schnabel on a hand-checked split (seed 0, rows 0..n); NaN Schnabel when d/n ≥ 0.5 |
| marginal_entropy | uniform over k values → log₂(k); constant → 0; nulls one category |

Plus: `test_benchmark_harness.py` covers `truth` with the fake technique (the accuracy hook receives it);
`PopulationCardinality.accuracy`'s truth figures get a hand-worked test; `datagen.population_samples` returns the
D it was asked for and the requested shape (`test_datagen.py`).

## 8. Benchmarks

| script | datasets | variants |
|---|---|---|
| `benchmark_distinct_count.py` | `large_dataset`; narrow 10M × 4 (distinct 10³ / 10⁵ / 10⁶ / 10⁷); wide 1M × 100 (Int64 and String); strings 10M × 2 (high-cardinality text) | `precision` ∈ {10, 12, 14, 16} (`estimator_params`, HLL only) |
| `benchmark_population_cardinality.py` | `large_dataset` (agreement only); samples 1M × 8 (D ∈ {10³, 10⁴, 10⁵, 10⁶} × uniform / Zipf 1.1, with truth); samples 100K × 8 (same populations, with truth) | none |
| `benchmark_marginal_entropy.py` | `large_dataset`; narrow 2M × 4; wide 20K × 60; near-unique 1M × 9; near-constant 1M × 3 | none |

## 9. Docs

CLAUDE.md: three technique paragraphs (§1 per-column), the Project Structure tree, the `_plugin` private list
(`hll_distinct`, `population_cardinality`; `marginal_entropy` / `ams_marginal_entropy` now also used by the new
classes), "All thirteen techniques" → sixteen.  `tests/performance/README.md`: three rows.  ToDo.md: 17 done.

## 10. Out of scope

Describe or the recommenders calling the new techniques (ToDo 16); a streaming version (techniques are one-shot);
Apache DataSketches or scikit-bio competitors (optional libraries, not installed); a DataFusion implementation
(ToDo 15).

## 11. Deviations (as built)

- Binary columns are ineligible (the Rust encoder does not accept them).
- The population-cardinality reference is `PopulationCardinalityPolars`: Polars `group_by` counts, a NumPy SplitMix split and the Python `estimators.py`.
- A `truth` mapping reaches `accuracy()` only when the dataset sets it.
- `DistinctCountPolarsApprox` accepts and ignores `precision`, so benchmark variants can pass it to every estimator.
- `DistinctCountHllRust` and `DistinctCountPolarsApprox` use ATOL 2 beside the RTOL: register collisions at small d.
- The §7 claim "within 1% of 50 at p = 14" was wrong: one collision occurs (49.07). Tests use RTOL / ATOL.
