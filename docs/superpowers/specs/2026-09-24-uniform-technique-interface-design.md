# Uniform Technique Interface — Design

**Date:** 2026-09-24
**Status:** Approved in brainstorming; awaiting written-spec review

## 1. Purpose

The project finds patterns and relationships between columns within and between dataframes. Its Rust extensions exist to test a hypothesis: *a Rust function that runs in parallel across all eligible columns, with careful algorithm design, can outperform a well-proven single-core library such as numpy.*

Today each technique is reached, checked for accuracy, and benchmarked in a different way (see §9). This refactor gives every technique:

1. **One public access pattern:** a class per implementation, sharing a contract per technique.
2. **One accuracy-test pattern:** every implementation compared against a single exact reference.
3. **One speed-benchmark pattern:** a shared harness that separates the **algorithmic** gain from the **parallel** gain.

### Success criteria

- Every technique is used as `Impl(**params).add(frames).result()` and returns a flat `pl.DataFrame` with the same key columns.
- All implementations of a technique return identical schemas and agree with the reference within a tolerance declared for that technique.
- Each accuracy test file has the same four blocks (§6); each benchmark script is about 20 lines that call the shared harness (§7).
- Each benchmark reports the algorithmic, parallel and total speedups for every Rust implementation.

### Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Role of the classes | **The public API.** Rust functions are private mathematical primitives; classes chain them into analytical conclusions. |
| Contract | One contract for every class: `add(frames: dict[str, LazyFrame|DataFrame])` then `result() -> pl.DataFrame` |
| Output content | Metrics **plus** conclusions; the conclusion logic is shared in a base class per technique |
| Layout | One subpackage per technique, one file per implementation |
| Ineligible columns | **Always** present in the output, with `status="ineligible"` and null metrics |
| Thresholds | Keyword-only constructor arguments with defaults, overridable when the class is constructed |
| Run-length / REE | **Out of scope.** It has no implementation (`run_length.py` was deleted in `93f6dcb`); a Wald-Wolfowitz runs test is planned for a later iteration |
| LRU in the exact similarity method | **Kept deliberately**: it is part of what's being benchmarked. Moved to a per-instance cache (§5.2) |

## 2. Technique scopes

Every technique belongs to one of three scopes. The scope decides how combinations of columns are enumerated, and CLAUDE.md groups the techniques the same way.

| Scope | `SCOPE` value | Meaning | Combinations | Techniques |
|---|---|---|---|---|
| Per-column | `"per_column"` | One column at a time | Each column of each frame | GCD |
| Multi-set | `"multi_set"` | Compares **distinct-value sets**; rows need not line up, so pairs may span frames | Every unordered pair of columns across **all** frames (same-frame pairs included) | Similarity, Membership |
| Ordered | `"ordered"` | Row *i* of A is compared with row *i* of B; only meaningful inside one frame | Every combination of columns **within each frame**; pairs that span frames are never generated | χ², Joint Entropy, Adjusted Rand |

`ARITY` (1, 2 or 3) is declared separately; for example, `ThreewayEntropy` is `ordered` with arity 3.

## 3. Package layout

```
services/analytics/
├── Cargo.toml, pyproject.toml, src/*.rs        # unchanged
└── analytics/                                  # maturin mixed package
    ├── __init__.py          # __version__ only
    ├── analytics.pyd        # compiled plugin
    ├── _plugin.py           # PRIVATE register_plugin_function wrappers (moved from today's __init__.py)
    ├── base.py              # Technique ABC
    ├── gcd/
    │   ├── __init__.py      # Gcd base, REFERENCE, IMPLEMENTATIONS
    │   ├── rust.py          # GcdRust
    │   ├── numpy.py         # GcdNumpy
    │   └── math.py          # GcdMath ★
    ├── membership/          # Membership: BloomRust, BloomFastbloom, MembershipExact ★
    ├── similarity/          # Similarity: MinHashRust, MinHashDatasketch, SimilarityExactLRU ★
    ├── chi_squared/         # ChiSquared: ChiSquaredRust, ChiSquaredScipy ★, ChiSquaredPolarsDS
    ├── entropy/             # PairwiseEntropy, ThreewayEntropy: *Rust, *Polars ★
    └── adjusted_rand/       # AdjustedRand: AdjustedRandRust, AdjustedRandSklearn ★
```

★ = the technique's accuracy reference.

- Implementation files are named after the library they use (`rust.py`, `scipy.py`, `datasketch.py`, `exact.py`, …).
- A file imports its third-party library at module level.
- A technique's `__init__.py` imports the base and `*Rust` class eagerly. Other implementations are reached through a module-level `__getattr__`, so `import analytics.chi_squared` does not need scipy or polars-ds.
- `REFERENCE: str` and `IMPLEMENTATIONS: tuple[str, ...]` hold class **names**; tests and benchmarks resolve them with `getattr`.
- Resolving a class whose library is missing raises `ImportError`, which tests and benchmarks turn into a visible skip.

## 4. Core contract: `analytics/base.py`

```python
Scope = Literal["per_column", "multi_set", "ordered"]
Status = Literal["computed", "ineligible", "pruned"]

class Technique(ABC):
    SCOPE: ClassVar[Scope]
    ARITY: ClassVar[int]                           # 1, 2 or 3
    METRICS: ClassVar[dict[str, pl.DataType]]      # every implementation must produce exactly these
    CONCLUSIONS: ClassVar[dict[str, pl.DataType]]  # added by _conclude
    EXACT: ClassVar[bool] = True                   # False for Bloom / MinHash implementations

    def __init__(self, **thresholds): ...          # declared keyword-only by each technique base
    def add(self, frames: dict[str, pl.LazyFrame | pl.DataFrame]) -> Self
    def result(self) -> pl.DataFrame

    def eligible(self, name: str, series: pl.Series) -> bool         # technique base
    @abstractmethod
    def _compute(self, frames: dict[str, pl.DataFrame],
                 combos: list[tuple[tuple[str, str], ...]]) -> pl.DataFrame  # implementation
    def _conclude(self, metrics: pl.DataFrame) -> pl.DataFrame       # technique base
```

### `add(frames)`
- Accumulates frames under the names given, and returns `self` so calls can be chained.
- A name that has already been added raises `ValueError`, so frames are never silently replaced.

### `result()`
1. Raises `ValueError` if nothing has been added.
2. Collects each frame once (LazyFrame → DataFrame).
3. Enumerates the combinations according to `SCOPE`/`ARITY` (§2), in a fixed canonical order: frame insertion order, then column order.
4. Evaluates `eligible` once per column. Combinations that include an ineligible column skip `_compute` and get `status="ineligible"`.
5. Calls `_compute(frames, eligible_combos)` and checks that the returned columns are exactly the keys + `status` + `METRICS` with matching dtypes. Any mismatch raises `TypeError` naming the implementation.
6. Adds the ineligible rows with null metrics.
7. `_conclude` adds `CONCLUSIONS`. Conclusions are null where the metrics are null.
8. Sorts by the key columns and returns.

The base does the enumeration and eligibility checks, so **every implementation receives identical combinations**, and speed and accuracy comparisons are like-for-like.

### Output schema

`df_a, col_a[, df_b, col_b[, df_c, col_c]], status, <METRICS…>, <CONCLUSIONS…>`

- **null** means the value was not computed (`ineligible` or `pruned`).
- **NaN** means the value was computed but is mathematically undefined, for example ARI with no overlapping non-null rows.
- `status="pruned"` is used only by MinHash implementations, for pairs that LSH did not propose as candidates.

### Thresholds and parameters
- Each technique base declares its thresholds as **keyword-only arguments with defaults**, and stores them on the instance (`self.cramers_v_threshold`, …).
- Implementation-specific parameters (`num_perm`, `fp_rate`, `cache_size`) are added in the subclass `__init__`, which calls `super().__init__(**thresholds)`.
- Values out of range raise `ValueError` at construction.

### Not in the classes
- **No timing attributes.** Timing is done from outside by the benchmark harness.
- **No caches shared between instances** (see §5.2 for the one deliberate cache).
- Plugin errors pass through unchanged.

## 5. The six techniques

### 5.1 Per-column

**`gcd/` — `Gcd` (`per_column`, arity 1)**

| | |
|---|---|
| METRICS | `dtype: String`, `gcd: Int128` |
| CONCLUSIONS | `gcd_compressible: Boolean` = `gcd > 1` |
| eligible | Integer-backed dtypes: Int/UInt 8–64, Int128, Decimal, Date, Datetime, Duration, Time. UInt128 → ineligible (the plugin's Rust polars can't take it across the FFI boundary). |
| Implementations | `GcdRust` (`column_gcd`), `GcdNumpy` (`np.gcd.reduce`: native int64/uint64 arrays for ≤64-bit physical values; object-dtype arrays of Python ints for Int128, wide Decimal and i64::MIN magnitudes, so it covers every eligible column and must match ★ exactly), ★`GcdMath` (`math.gcd`, arbitrary precision) |

Semantics are unchanged from today (physical units, nulls skipped, all-null/all-zero → 0, magnitude 2¹²⁷ → null). The `dtype` metric is still reported for ineligible columns, because it is a property of the column rather than the result of a computation. It is the one metric exempt from the "null when not computed" rule.

### 5.2 Multi-set

**`membership/` — `Membership` (`multi_set`, arity 2)**

| | |
|---|---|
| Thresholds | `containment_threshold=0.95` |
| METRICS | `ratio_a_in_b`, `ratio_b_in_a` (fraction of A's distinct non-null values found in B, and the reverse), `n_distinct_a`, `n_distinct_b`, `n_non_null_a`, `n_non_null_b` |
| CONCLUSIONS | `unique_a`, `unique_b` (n_distinct = n_non_null); `relationship ∈ {pk_pk, fk_pk, pk_fk, mutual, a_in_b, b_in_a, none}` |
| eligible | Every dtype. Each implementation encodes values itself (`MembershipExact`/`BloomFastbloom` turn List/Array values into tuples or strings; `BloomRust` uses `encode_series`) |
| Implementations | `BloomRust` (param `fp_rate=0.01`): builds one filter per column over its distinct values, then runs `membership_ratio` in both directions. `BloomFastbloom` (same `fp_rate`). ★`MembershipExact`: exact `is_in` on distinct values. Bloom implementations have `EXACT=False`. |

`relationship` rules, with `t = containment_threshold`:
- `pk_pk`: both ratios ≥ t and both columns unique.
- `fk_pk`: `ratio_a_in_b ≥ t` and B unique and A not unique. `pk_fk` is the mirror case.
- `mutual`: both ratios ≥ t, otherwise.
- `a_in_b` / `b_in_a`: one direction ≥ t.
- `none`: otherwise.

The public sampled membership option (`membership_ratio_sample`) is **removed from the public API**. The Rust function stays. If sampling returns later, it will be a base-level `sample_frac` that every implementation honours.

**`similarity/` — `Similarity` (`multi_set`, arity 2)**

| | |
|---|---|
| Thresholds | `jaccard_threshold=0.6`, `overlap_threshold=0.95` |
| METRICS | `jaccard`, `overlap` (exact values, computed during verification) |
| CONCLUSIONS | `passes_jaccard`, `passes_overlap` |
| eligible | Same as Membership |
| Implementations | `MinHashRust` (param `num_perm=128`), `MinHashDatasketch` (`num_perm=128`), ★`SimilarityExactLRU` (param `cache_size=4096`). MinHash implementations have `EXACT=False`. |

- **Shared verification.** The `Similarity` base holds one verification routine that computes exact Jaccard/Overlap for a list of pairs from distinct-value sets. `SimilarityExactLRU` verifies every pair; the MinHash implementations verify only the LSH candidates and mark the rest `pruned`.
- **The LRU cache is kept deliberately and fixed.**
  - It is created per instance in `__init__`: `self._distinct = functools.lru_cache(maxsize=cache_size)(self._distinct_values)`.
  - It is cleared in `add()`.
  - This fixes three problems with today's `@lru_cache` on a method: the cache is shared across all instances at class level and keeps every instance alive; it returns stale values after a frame changes; and a second `result()` call on the same instance hits a warm cache.
  - The benchmark compares LRU-cached exact checking of all pairs against Rust MinHash/LSH pruning followed by the same LRU-cached checking of the candidates.
- The LSH threshold rule `min(jaccard*0.9, overlap*0.45)` and the `_optimal_lsh_params` search move to `MinHashRust`, with a comment explaining the 0.45 factor. The LSH S-curve is calibrated against Jaccard, so pairs that pass only on Overlap Coefficient need a lower effective threshold to become candidates.

### 5.3 Ordered

**`chi_squared/` — `ChiSquared` (`ordered`, arity 2)**

| | |
|---|---|
| Thresholds | `cramers_v_threshold=0.3`, `max_unique=1000` (None disables it) |
| METRICS | `chi2_stat`, `p_value`, `cramers_v`, `low_expected_count`, `n_valid` |
| CONCLUSIONS | `associated` = `cramers_v ≥ cramers_v_threshold` |
| eligible | Boolean, String, Categorical, Enum, integer types; and `n_unique ≤ max_unique` |
| Implementations | `ChiSquaredRust`, ★`ChiSquaredScipy` (`scipy.stats.chi2_contingency`, `correction=False`, after the drop-null policy), `ChiSquaredPolarsDS` (computes `low_expected_count` and `n_valid` itself) |

Null policy: rows where either column is null are dropped.

**`entropy/` — `PairwiseEntropy` (`ordered`, arity 2) and `ThreewayEntropy` (`ordered`, arity 3)**

| | PairwiseEntropy | ThreewayEntropy |
|---|---|---|
| Thresholds | `nmi_threshold=0.9`, `near_unique_margin=0.1` | `near_unique_margin=0.1` |
| METRICS | `h_a`, `h_b`, `h_ab`, `mi`, `nmi`, `n_rows` | `h_abc`, `n_rows` |
| CONCLUSIONS | `redundant` = `nmi ≥ nmi_threshold`; `near_unique` = `h_ab ≥ log₂(n_rows) − margin` | `near_unique` = `h_abc ≥ log₂(n_rows) − margin` |
| eligible | Every dtype | Every dtype |
| Implementations | `PairwiseEntropyRust` (chains `marginal_entropy` + `pairwise_joint_entropy`), ★`PairwiseEntropyPolars` (`value_counts` → `entropy(base=2)`) | `ThreewayEntropyRust`, ★`ThreewayEntropyPolars` |

- Null policy: null is its own category.
- `mi = h_a + h_b − h_ab`.
- `nmi = mi / min(h_a, h_b)`; NaN when `min(h_a, h_b) = 0`.

**`adjusted_rand/` — `AdjustedRand` (`ordered`, arity 2)**

| | |
|---|---|
| Thresholds | `ari_threshold=0.9` |
| METRICS | `ari`, `n_valid` |
| CONCLUSIONS | `same_partition` = `ari ≥ ari_threshold` |
| eligible | Every dtype except List/Array |
| Implementations | `AdjustedRandRust`, ★`AdjustedRandSklearn` (`adjusted_rand_score` on string-cast labels after the drop-null policy) |

## 6. Accuracy tests

**Layout.** `tests/test_<technique>.py`, one file per technique package. Every file has these four blocks:

| Block | Parametrized over | Checks |
|---|---|---|
| 1. Contract | Every implementation | Columns and dtypes = keys + `status` + METRICS + CONCLUSIONS; every combination present; `ineligible` rows have null metrics and conclusions; sorted by keys |
| 2. Reference agreement | Every non-reference implementation | Matches ★ on the same toy data (see below) |
| 3. Reference against oracle | ★ only | Tiny hand-worked cases with known answers |
| 4. Conclusions | The technique base, once | Hand-built metric frames → correct labels at the threshold edges, both with default and with overridden thresholds |

Edge cases specific to a technique are kept as additional tests in the same file, parametrized over implementations where they apply: GCD i128/Decimal/MIN magnitudes, UInt128, Bloom size mismatch.

**Reference agreement.**
- `EXACT` implementations: `assert_metrics_match(result, reference, rtol, atol)` in `tests/harness.py`.
  - It joins on the keys.
  - It requires identical `status` values.
  - It treats NaN as equal to NaN and null as equal to null.
  - It compares floats with the tolerance each technique declares (the same tolerances as today's tests: χ² `rtol=1e-4`, entropy `rtol=1e-5`, ARI `rtol=1e-9, atol=1e-12`, GCD exact).
- Non-exact implementations:
  - Bloom: `ratio_exact ≤ ratio_bloom ≤ ratio_exact + fp_slack`, since Bloom filters produce no false negatives.
  - MinHash: recall of `passes_jaccard | passes_overlap` against ★ is ≥ 0.85 (the medium-scale test is marked `slow`); pairs that aren't pruned have exactly the reference's metrics, since verification is exact.

**Toy data: `tests/datagen.py`** (seeded; tests use small sizes, benchmarks use large ones)
- `mixed_dtypes(n_rows, seed)`: today's 18-column frame (6 dtypes × 3 distributions) for ordered techniques.
- `related_frames(n_rows, seed)`: several frames with planted subset, equal, disjoint, PK-PK and FK-PK column pairs for multi-set techniques.
- `integer_multiples(n_rows, n_cols, g, seed)` and `integer_random(n_rows, n_cols, seed)`: for GCD.

**Imports.** Everything imports from the installed package (`from analytics.gcd import GcdRust`). The `sys.path` workaround in `conftest.py` is removed. `conftest.py` keeps the `slow` marker and exposes the datagen frames as fixtures. An implementation whose library is missing is skipped by `pytest` with the `ImportError` message as the reason.

## 7. Speed benchmarks

**Shared harness: `tests/performance/harness.py`.** Each `tests/performance/benchmark_<technique>.py` declares its datasets and parameters and calls `harness.run(technique_package, datasets, params)`.

- **What is timed:** `cls(**params).add(frames).result()`, with a fresh instance for every run. There is 1 warm-up run and then 5 timed runs; the harness reports the **median** and the **minimum**.
- **Thread split:** each Rust implementation is timed at N threads (the default) and at 1 thread. The 1-thread runs happen in a child process launched with `RAYON_NUM_THREADS=1` and `POLARS_MAX_THREADS=1`, because both thread pools are fixed at process start. Other implementations run with default settings, and the thread count they used is recorded.
- **Summary figures** for each dataset:

| Figure | Formula |
|---|---|
| Algorithmic speedup | fastest non-Rust implementation ÷ Rust at 1 thread |
| Parallel speedup | Rust at 1 thread ÷ Rust at N threads |
| Total speedup | fastest non-Rust implementation ÷ Rust at N threads |

- **Datasets** (the same three kinds for every technique):
  1. `tests/data/large_dataset.arrow` (50K × 101, realistic mixed types).
  2. **Narrow and long:** few columns, many rows. This tests parallelism **inside** a column.
  3. **Wide:** many columns, fewer rows. This tests parallelism **across** columns.
  Sizes 2 and 3 are chosen per technique in its benchmark script.
- **Time budget:** if an implementation's warm-up run exceeds `budget_s` (default 60), it is reported as `skipped: over budget` and not timed.
- **Sanity check:** where the reference completed, each result is compared with it using `assert_metrics_match` (exact implementations) or the technique's bound. The result is shown as ✓/✗ and never stops the run.
- **Output:**
  - A printed table (implementation × dataset × threads: median ms, min ms, speedups, ✓/✗).
  - `tests/performance/results/<technique>_<timestamp>.parquet`, recording CPU count and the polars, numpy and crate versions. This folder is git-ignored.
- **Removed:** the in-class phase timers (Bloom build/insert/query, MinHash MinHash/LSH/verify) and Bloom's `tracemalloc` figure, which cannot see Rust's `mimalloc` allocations.

## 8. Migration

**Moved or deleted**
- `services/analytics/{bloom_filter, chi_squared_polarsds, deterministic_similarity_filter, minhash_lsh_filter, minhash_lsh_filter_datasketch}.py` → their technique packages; the old files are deleted.
- `analytics/__init__.py` free functions and the `.analytics` expression namespace → private `_plugin.py`, which has no expression namespace; `BloomRust` calls the plugin functions directly.
- The old `tests/test_*.py` and `tests/performance/benchmark_*.py` are replaced one for one (`test_similarity_filters.py` → `test_similarity.py`, `test_bloom_filter.py` → `test_membership.py`, `benchmark_jaccard.py` → `benchmark_similarity.py`, `benchmark_bloom_filter.py` → `benchmark_membership.py`).

**Unchanged**
- Rust source: no changes to the maths or to the exported function names.
- The Rust follow-ups listed in CLAUDE.md "Next Steps" stay separate.
- `notebooks/NVI Clustering.ipynb` and the `tests/*_entropies.ipc` snapshots it reads.

**Behaviour changes for callers**
1. Results are flat tables with key columns, not a single struct column.
2. Similarity returns every pair with a `status` column. Filter with `passes_jaccard | passes_overlap` to get the old output.
3. Membership reports distinct-value ratios in both directions, instead of row-level ratios in one direction.
4. The public sampled membership option is removed.

**CLAUDE.md update**
- "Analytical Functions" is grouped by scope: **1) Per-column** (GCD), **2) Multi-set** (Similarity, Membership), **3) Ordered** (Chi-squared, Joint Entropy, Adjusted Rand).
- Run-length / REE is removed from the techniques and noted as "not implemented; Wald-Wolfowitz runs test planned".
- The project structure, the public API (the classes, not the plugin functions) and the testing/benchmark conventions are updated to match this spec.

**Order of work** (one commit per step; the full test suite passes after each):
1. Infrastructure: `base.py`, `_plugin.py` (the old public functions stay re-exported from `__init__` until step 5), `tests/harness.py`, `tests/datagen.py`, `tests/performance/harness.py`, and unit tests for the base contract and both harnesses.
2. Pilot: `gcd/` together with its test and benchmark.
3. `chi_squared/`, `adjusted_rand/`, `entropy/`.
4. `membership/`, `similarity/`.
5. Remove the re-exports from `analytics/__init__.py`; update CLAUDE.md.

The old test and benchmark files for each technique are deleted in the same commit as their replacements.

## 9. Appendix: inconsistencies found (before this refactor)

| Technique | Python access | Comparison libraries | Test data | Benchmark data and timing |
|---|---|---|---|---|
| Bloom | Class + expression namespace + free functions | fastbloom-rs, inline in the test and the benchmark | Own column list | Generated; per-phase timers |
| MinHash | 3 classes | Brute force and datasketch classes | Own synthetic data | large_dataset; timers inside the classes |
| χ² | Free function → struct column | polars-ds module; scipy inline | 18-column fixture | large_dataset; 1 run |
| Entropy | Free functions | Native Polars, inline | 18-column fixture | large_dataset; 1 run |
| Run-length | None (deleted in `93f6dcb`) | — | — | — |
| ARI | Free function | sklearn, inline | 18-column fixture | large_dataset; mean of 3 runs |
| GCD | Free function + UInt128 workaround | numpy/math, inline | Hand-built edge cases | 4 synthetic shapes; mean of 3 runs |

Other problems: three different import styles; key columns named differently (`col_a`, `col_name`, `column`); the three similarity classes repeat the same ~40-line verification loop; `DeterministicSimilarityFilter` accepts an unused `num_perm`; the class-level `lru_cache` on bound methods.

## 10. Amendments made while planning (2026-09-24)

Probing the plugin and Polars while writing the implementation plan changed these details. Where they conflict with earlier sections, **this section wins**.

1. **One file for each base class.** Each technique package puts its base class in `base.py`, and `__init__.py` re-exports it. This avoids circular imports between `__init__.py` and the implementation files.
2. **Entropy becomes two packages:** `analytics/pairwise_entropy/` and `analytics/threeway_entropy/`. Each package holds exactly one technique, so `REFERENCE` stays a single name. `ThreewayEntropyPolars` reuses `entropy_bits` from `pairwise_entropy/polars.py`.
3. **Row order.** Results come back in the canonical enumeration order: frames in insertion order, then column order within each frame, then `itertools.combinations` order. They are not sorted, so input column order is preserved, as GCD did before.
4. **`DESCRIPTORS` column group.** It sits between `status` and the metrics, and is filled on **every** row, including ineligible rows. GCD's `dtype` is its only member, and now uses Python's `str(dtype)` (for example `"Int64"`, not the Rust `"i64"`), so all implementations agree.
5. **A `compatible(dtypes)` hook.** It judges a combination as a whole. Multi-set pairs whose columns come from different **value families** are `ineligible`. The families are:
   - integers of 64 bits or fewer;
   - String, Categorical and Enum;
   - otherwise, the exact dtype.

   Without this rule, Rust (which hashes physical values) and the exact Python sets would disagree. For example, `Date` 19000 and `Int32` 19000 are the same key in Rust but different values in Python.
6. **An `agreement(result, reference) -> list[str]` method** on the technique base, using `RTOL`/`ATOL` class attributes. Accuracy tests and the benchmark sanity check both call it:
   - `BloomMembership` overrides it with the no-false-negative plus aggregate false-positive-rate bound.
   - `Similarity` overrides it with the exact-candidates plus recall bound.
7. **Zero-row frames.** Ordered techniques mark columns of zero-row frames `ineligible`, because the Rust plugins raise `"Cannot calculate … on empty columns"`.
8. **NaN metrics make boolean conclusions `False`.** Polars orders NaN above every number, so a bare `NaN >= t` would evaluate to `True`.
9. **`cache_size` lives on the `Similarity` base**, because the per-instance LRU serves verification for all three similarity implementations.
10. **`Dataset.exclude` in benchmarks.** It lists implementations a script does not run on that dataset, which are reported as `excluded`. For example, `GcdMath` is excluded from 10⁸-value datasets, where `to_list()` alone would exhaust memory.
11. **No Python test for Bloom size mismatches.** `BloomRust` always sizes its own filters, so a size mismatch can't happen through the class. The Rust unit tests keep covering it.
12. **Distinct values come from Polars `unique()`,** which treats `-0.0` and `0.0` as one value and all NaNs as one value, matching the Rust encoder. The pure-Python implementations then "freeze" them into set members: one shared NaN object, and tuples for nested values.
