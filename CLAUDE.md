# Project Objectives
Determine patterns and relationships between columns, both within and between dataframes.

# Technology
- Arrow columnar data format throughout
- Python with Rust extensions via pyo3, with Arrow at the FFI boundary for performance
- Polars as the primary dataframe library

# Python Environment
Conda environment: `C:\Users\Ben\miniconda3\envs\p312`
Python exe: `C:\Users\Ben\miniconda3\envs\p312\python.exe`

# Coding Style
Prioritise performance & simplicity.

# Analytical Functions

Every technique is used the same way — `Impl(**params).add({"name": frame, ...}).result()` —
and returns one flat table: `df_a, col_a[, df_b, col_b[, df_c, col_c]] | status | descriptors | metrics | conclusions`.
`status` ∈ {computed, ineligible, pruned}; null = not computed, NaN = computed but undefined.
Ineligible columns/pairs are always listed, never dropped. Results must export to native Arrow
(`result().to_arrow()`, checked in every contract test): never a Polars-only dtype such as Int128 —
use Decimal(38, 0) for 128-bit integers. `add()` accepts Polars DataFrames / LazyFrames and any Arrow
tabular object (pyarrow Table, RecordBatch, RecordBatchReader — anything with `__arrow_c_stream__`).
Int128 / UInt128 columns, at any depth, are ineligible in every technique: Arrow has no 128-bit integer
(`analytics._dtypes.WIDE_INTEGERS`). Thresholds are keyword-only
constructor arguments with the defaults below. Techniques are grouped by **scope** —
how column combinations are enumerated.

## 1. Per-column (one column at a time)

**GCD — `analytics.gcd`** (`GcdRust`, `GcdNumpy`, ★`GcdMath`). ClickHouse GCD-codec method: the GCD of the magnitudes of each integer-backed column's raw physical values (Int/UInt 8–64, Decimal → unscaled, Date → days, Datetime/Duration → time unit, Time → ns). Nulls skipped; all-null / all-zero / zero-row → 0; `gcd` is Decimal(38, 0); other dtypes (incl. Categorical/Enum and Int128/UInt128) → ineligible. Descriptor `dtype` (Python `str(dtype)`) on every row. Conclusion `gcd_compressible` = gcd > 1. Rust: rayon-parallel across columns and 64K-value chunks, `binary_gcd(g, v % g)` fold with early exit at 1.

**Describe — `analytics.describe`** (`DescribeRust`, `DescribeDataFusion`, ★`DescribePolars`). Profile for choosing narrower / more compressible Arrow types (spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md). Metrics: counts, entropy (null as a category), f1/f2, first-occurrence argmin/argmax and top-5 (indices; the base renders values), byte/list lengths, `gcd`, byte totals (`sum_len`, `sum_len_unique`), significant-digit counts, float stats (NaN/inf/fractional, decimal places, f32 round trip), numeric-string and ISO 8601 counts (Rust byte scanners / Rust-regex elsewhere — linear time), Datetime local-midnight count, Arrow IPC sizes (classic layout, plain + ZSTD) and Polars native-layout IPC sizes, and the same for list inner values. Conclusions: rendered min/max/top-5, `unique`, Chao1 / Schnabel (3-way seeded split) / Duj1 with 95% intervals and `est_cardinality` picked by rule (exact → Duj1 → Schnabel → Chao1), `estimates_agree`, and `class` ∈ {null, constant, boolean, ordinal, categorical, discrete}. Keywords: `population_rows=None`, `categorical_threshold=10_000`, `zstd_level=1`, `seed=0`. Agreement: exact except entropy (1e-9), ZSTD sizes (1%), Schnabel (10%).

**Recommend — `analytics.recommend`** (★`RecommendRust`, the only implementation). Narrowest value-preserving Arrow type per column (spec: docs/superpowers/specs/2026-09-26-recommend-technique-design.md): Describe's table plus `rec_*` columns. Step 1 type rules (null → boolean → uint → int → decimal → float → date → time → timestamp → timestamp_with_offset → string; lists → scalar when every list holds one item), step 2 dictionary encoding for strings (key width from `est_high`, Polars key one code narrower). Candidates carry a predicted IPC size and are tried smallest projected population size first (ties: hierarchy rank); each is cast, verified row by row and measured (Arrow and Polars layouts, plain and ZSTD); the original type is always the last resort. `rec_candidates` lists rule, evidence (the metric values tested), predicted/projected size and outcome for every candidate. Keywords: Describe's plus `boolean_pairs=(("true", "false"),)`. No Python implementation, so no reference agreement or algorithmic speedup; oracles are pyarrow/Polars casts and predicted = measured.

**Streaming Recommend — `analytics.recommend.StreamingRecommender`** (Rust only; not on the uniform contract). Recommend's dtype recommendations from batches added over time (spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md). `add(frame)` any number of times (columns may appear, disappear or start as Null; other type changes raise; each Arrow batch is atomic, one multi-batch `add` is not), `finish()` at any point → one row per column: `column, status, dtype, first_row, n_rows, n_null, min, max, gcd, sum_len, min_len, max_len, n_unique, distinct_overflowed, est_*`, the size columns, Recommend's `rec_*` columns, `n_sampled_rows, n_sampled_blocks`. Exact running statistics (src/partial.rs) prove each recommendation on every row (`prove`, `lossy_by_stats` in recommend.rs); a seeded Algorithm-L sample of contiguous row blocks (src/reservoir.rs; compact `copy_rows` pieces) gives ZSTD sizes — those of an IPC file written in `block_rows` batches — and cross-checks the chosen type. Original sizes analytic except Struct / deeper nesting (per-batch sums). Distinct values tracked only for text, up to `categorical_threshold`. Int128/UInt128, Object and nested-Null columns ineligible (marked by the Python wrapper, `n_null` null). No `population_rows` (estimates: Schnabel → Chao1). Keywords: `reservoir_rows=524_288` (0: no sample, ZSTD null), `block_rows=65_536`, `categorical_threshold=10_000`, `zstd_level=1`, `seed=0`, `boolean_pairs`.

## 2. Multi-set (distinct-value sets; pairs may span frames)

Pairs are compared only within one **value family** (ints ≤64-bit; String/Categorical/Enum; otherwise exact dtype) — other pairs are ineligible.

**Membership — `analytics.membership`** (`BloomRust`, `BloomFastbloom`, ★`MembershipExact`). Directional containment of distinct values: `ratio_a_in_b`, `ratio_b_in_a`, plus exact distinct/non-null counts. Conclusions (`containment_threshold=0.95`): `unique_a/b`, `relationship` ∈ {pk_pk, fk_pk, pk_fk, mutual, a_in_b, b_in_a, none}. PK-PK needs reciprocal containment of unique values. Bloom implementations: `fp_rate=0.01`; no false negatives.

**Similarity — `analytics.similarity`** (`MinHashRust`, `MinHashDatasketch`, ★`SimilarityExactLRU`). Jaccard `|A∩B|/|A∪B|` and Overlap Coefficient `|A∩B|/min(|A|,|B|)`; conclusions `passes_jaccard` (0.6) / `passes_overlap` (0.95). MinHash implementations prune with LSH (`num_perm=128`, status `pruned`) and verify candidates exactly; the exact reference checks every pair through a per-instance LRU cache of distinct-value sets (`cache_size=4096`) — the comparison being tested is Rust probabilistic pruning vs LRU-cached brute force.

## 3. Ordered (row i of A vs row i of B; within one frame; zero-row frames ineligible)

**Chi-squared — `analytics.chi_squared`** (`ChiSquaredRust`, ★`ChiSquaredScipy`, `ChiSquaredPolarsDS`). Independence test on Boolean/String/Categorical/Enum/integer columns with ≤ `max_unique=1000` values; null rows dropped (`n_valid`). χ² detects any association but not causality; at 50K rows p-values are vanishingly small, so **Cramér's V is the diagnostic**: `associated` = V ≥ 0.3. `low_expected_count` flags unreliable tables.

**Joint entropy — `analytics.pairwise_entropy`** (`PairwiseEntropyRust`, ★`PairwiseEntropyPolars`) and **`analytics.threeway_entropy`** (`ThreewayEntropyRust`, ★`ThreewayEntropyPolars`). Null is its own category. Pairwise reports `h_a, h_b, h_ab`, `mi = h_a + h_b − h_ab`, `nmi = mi / min(h_a, h_b)`; `redundant` = NMI ≥ 0.9 (one column predicts the other). `near_unique` = joint entropy ≥ log₂(n_rows) − 0.1 (near-unique combinations — a different diagnostic). Threeway reports `h_abc` and `near_unique`.

**Adjusted Rand Index — `analytics.adjusted_rand`** (`AdjustedRandRust`, ★`AdjustedRandSklearn`). Chance-corrected partition agreement (1 = identical, ≈0 = chance, floor −0.5); null rows dropped (`n_valid`); `same_partition` = ARI ≥ 0.9. Unlike NMI it is chance-adjusted; unlike χ² it measures partition identity.

★ = accuracy reference.

**Not implemented:** run-length / REE compression analysis (`run_length.py` was deleted in `93f6dcb`). A Wald-Wolfowitz runs test is planned as a future ordered/per-column technique on this same contract.

# Project Structure

```
turbo-parakeet/
├── services/analytics/
│   ├── pyproject.toml, Cargo.toml          # maturin build (editable install via analytics.pth)
│   ├── src/                                # Rust extension — lib.rs, shared.rs, entropy.rs, chi_squared.rs,
│   │                                       #   contingency.rs, ari.rs, gcd.rs, bloomfilter.rs, minhash.rs,
│   │                                       #   describe.rs, sizes.rs, cardinality_estimators.rs, recommend.rs,
│   │                                       #   partial.rs, reservoir.rs, streaming.rs (streaming recommender),
│   │                                       #   api.rs, arrow_io.rs, python.rs, capi.rs
│   ├── bindings/java/                      # Maven project: Panama FFM binding over capi.rs + JUnit tests
│   └── analytics/
│       ├── __init__.py                     # __version__ only
│       ├── analytics.pyd                   # compiled extension (module analytics.analytics)
│       ├── _plugin.py                      # PRIVATE wrappers over the Arrow binding (called only by *Rust classes)
│       ├── _dtypes.py, _sets.py            # dtype groupings / value families; canonical distinct values
│       ├── base.py                         # Technique contract, helpers, metric_mismatches
│       └── <technique>/                    # gcd, describe, recommend, membership, similarity, chi_squared,
│           ├── __init__.py                 #   pairwise_entropy, threeway_entropy, adjusted_rand
│           ├── base.py                     # technique base: METRICS, eligibility, conclusions, RTOL/ATOL
│           ├── rust.py, <library>.py …     # one file per implementation
│           └── streaming.py                # recommend/ only: StreamingRecommender (not a technique)
└── tests/
    ├── conftest.py                         # `slow` marker, `dataset` fixture
    ├── datagen.py                          # seeded generators shared with benchmarks
    ├── harness.py                          # implementation_params/load/reference/run/assert_contract/assert_agrees/with_metrics
    ├── test_base.py, test_datagen.py, test_benchmark_harness.py
    ├── test_<technique>.py                 # one per technique package
    ├── test_streaming_recommend.py         # StreamingRecommender: contract, parity with one-shot, ZSTD, known answers
    ├── data/large_dataset.arrow            # 50K rows, 101 columns
    └── performance/                        # never collected by pytest
        ├── harness.py                      # shared timing harness
        ├── benchmark_<technique>.py        # one per technique package
        └── benchmark_streaming_recommend.py # standalone: add throughput / finish time beside one-shot
```

# Rust Extension (analytics)
Build: `maturin develop --release` from `services/analytics/`. Python changes need no rebuild (editable install).
Cargo feature `python` (default) gates pyo3 + python.rs; the Java build is Python-free:
`cargo build --release --no-default-features --target-dir target/capi` (own target dir — maturin writes the
Python-linked library to `target/release`), then `./mvnw test` in `services/analytics/bindings/java/` (JDK 25).

Four layers (specs: docs/superpowers/specs/2026-09-27-arrow-ffi-interface-design.md, 2026-09-28-java-binding-design.md):
- `src/api.rs` — the language-neutral core: one `pub fn` per entry point, arrow-rs `RecordBatch` (+ plain parameters) in, `RecordBatch` out (Bloom: bytes). No pyo3 or Polars type in any signature; a future Java / C-ABI binding wraps exactly this file. Errors: `InvalidInput` (unknown or duplicate column names, a malformed Bloom array or zero Bloom/LSH parameters, wrong column counts, an unimportable Arrow type, or a kernel `ColumnNotFound`/`SchemaMismatch`/`InvalidOperation`/`ShapeMismatch` error — a column of the wrong type for the kernel) / `Compute`. `StreamingRecommender` (new / add / mark_ineligible / finish) is the one stateful entry point.
- `src/arrow_io.rs` — RecordBatch ↔ Polars Series, zero-copy through the C Data Interface (Polars' `_PL_CATEGORICAL2` / `_PL_ENUM_VALUES2` field metadata restores Categorical / Enum). Kernels still compute on Series; `sizes.rs` / `recommend.rs` measure layouts derived with `export_series`. Every import is checked, once (`CheckedReader` / `import_checked`; `api` entry points trust their RecordBatch: structure on the raw C structs, `ArrayData::validate` + `validate_values` + nulls at every level, FixedSizeList offsets, unsupported types such as Decimal256 refused from the schema), so malformed input is InvalidInput, not a process abort. Limit: the C Data Interface carries no buffer lengths, so offsets past the producer's real buffer go undetected.
- `src/python.rs` — pyo3 module `analytics.analytics`: reads any `__arrow_c_stream__` object into one batch, rejects Polars' private `_pli128` / `_plu128` (Int128 / UInt128) with ValueError naming the column, releases the GIL, returns `ArrowTable` (itself `__arrow_c_stream__`). InvalidInput → ValueError, Compute → RuntimeError, non-Arrow input → TypeError. `StreamingRecommender` pyclass (frozen, mutex-guarded; `add(data, ineligible)` reads its stream batch by batch, marking the wrapper's ineligible columns first).
- `src/capi.rs` — C ABI (`extern "C"`, no pyo3): `analytics_describe_and_recommend` (Arrow C Stream in, one-batch
  stream out, plain C parameters; `population_rows < 0` = None) and `analytics_free_error`. Returns 0 / 1 InvalidInput /
  2 Compute with a message in `*error`. Wrapped by `io.github.benssutton.analytics.Analytics` (Java 25 FFM + Arrow Java
  `arrow-c-data`), which maps 1 → IllegalArgumentException, 2 → RuntimeException. Also `analytics_recommender_{new,add,finish,free}`
  (opaque `void *` handle; `add` consumes the stream), wrapped by `io.github.benssutton.analytics.StreamingRecommender`
  (AutoCloseable; `StreamingParams`; a read/write lock keeps `close` from racing `add`/`finish`, a Cleaner frees an
  unclosed handle). `Native` holds the shared library lookup and error mapping.

Private — reached only through `analytics._plugin`, only by the `*Rust` classes (and the exceptions named below):
`column_gcd`, `pairwise_chi_squared`, `pairwise_adjusted_rand`, `marginal_entropy`,
`pairwise_joint_entropy`, `threeway_joint_entropy` (the classes always pass explicit triplets; `triplets=None` means every triplet, uncapped — C(101,3) = 166,650 at 101 cols, ~65 s at 50K rows),
`bloom_filter` + `membership_ratio` (the bit array crosses as `bytes`), `minhash` + `lsh_candidates`,
`describe_columns` + `column_sizes`, `describe_and_recommend`; `checked_table` (the checked Arrow import, used by
`analytics.base` before py-polars reads Arrow input — handed over as a pyarrow Table when pyarrow is installed);
`StreamingRecommender` (via `_plugin.streaming_recommender`, used only by `analytics.recommend.StreamingRecommender`).

# Testing Convention
Every technique package declares `REFERENCE` (the exact accuracy reference) and `IMPLEMENTATIONS`.
- `tests/test_<technique>.py` — **accuracy only**, never timed. Same four blocks everywhere, via `tests/harness.py`:
  1. contract (every implementation): schema, every combination in canonical order, non-computed rows null;
  2. reference agreement (every non-reference implementation): `impl.agreement(result, reference_result)` —
     exact implementations within the technique's `RTOL/ATOL`; Bloom: no false negatives + aggregate FP ≤ 3×fp_rate;
     MinHash: evaluated pairs exact + recall ≥ 0.85;
  3. known answers (hand-worked cases, reference included);
  4. conclusions, once per technique base, via `with_metrics` (including a NaN row).
  Optional libraries missing → visible skip naming the library.
- `tests/performance/benchmark_<technique>.py` — **speed only**: ~20-line scripts calling `harness.run(package, datasets)`.
  Times `Impl(**params).add(frames).result()` (fresh instance, 1 warm-up + 5 runs, median/min) on
  large_dataset.arrow + a narrow/long + a wide shape. Rust implementations are re-timed at 1 thread in a child
  process (`RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`) to report **algorithmic** (fastest non-Rust ÷ Rust@1),
  **parallel** (Rust@1 ÷ Rust@N) and **total** speedups. Results → `tests/performance/results/*.parquet` (git-ignored).
- Rust unit tests: `cargo test --lib <module>::` from `services/analytics/` with `PYO3_PYTHON` set to the env's `python.exe` and the env dir on `PATH` (else `STATUS_DLL_NOT_FOUND`).

# Current Focus
All nine techniques (GCD, Describe, Recommend; Membership, Similarity; Chi-squared, Pairwise/Threeway Entropy, ARI) share one class-based contract and one Arrow-in/Arrow-out Rust core, one accuracy-test pattern and one benchmark harness (spec: docs/superpowers/specs/2026-09-24-uniform-technique-interface-design.md). Benchmark results with the algorithmic/parallel/total split live in tests/performance/results/. Next technique: Wald-Wolfowitz runs test. The streaming recommender (above) is implemented, with Python and C bindings.

Joint entropy (2-way and 3-way) is dense-id encoded (see below): 83.5µs/pair, 170.3µs/triplet at 50K rows/101 cols — roughly 2x faster than the original hash-tuple implementation, and the 3-way/2-way speedup-vs-native-Polars gap that motivated the change is closed (3-way now outpaces 2-way's multiplier rather than trailing it). ARI (`pairwise_adjusted_rand`) is complete and validated against scikit-learn; chi² now shares the same dense contingency builder and reports `n_valid`.

## Entropy dense re-encoding (shared.rs + entropy.rs)
Columns feeding `pairwise_joint_entropy`/`threeway_joint_entropy` are dictionary-encoded to dense ids `0..card` via `densify`/`build_dense_cache_par` (nulls become their own id, so the per-row loop carries no separate null mask). Joint keys are then combined arithmetically (`(a·Kb + b)·Kc + c`) instead of hashing multi-field tuples:
- joint space ≤ 2²⁰ → flat-array counting, no hashing (thread-local scratch array + touched-slot list for O(distinct) reset)
- joint space > 2²⁰ but ≤ u64::MAX → single-u64 hash map
- joint space > u64::MAX (only reachable past ~2.6M rows) → u128-packed hash map fallback

This encoding is entropy/chi²-only — it's value-relabeling and has no cross-column/cross-frame identity, which is fine since entropy is invariant under injective relabeling. Bloom/MinHash must keep using the value-stable `encode_series`/`build_column_cache_par` path since they compare actual values across columns and frames.

# Next Steps

## Implementation cleanups

**[bloomfilter.rs](services/analytics/src/bloomfilter.rs)**
- ~~silent state discard on `existing_filter` length mismatch~~ **Fixed**: wrong-sized bit arrays are rejected by `api::membership_ratio` as `InvalidInput` (→ ValueError); `validate_bit_array` still guards the unchecked bit reads.

**[minhash.rs](services/analytics/src/minhash.rs)**
- [minhash.rs:75-92](services/analytics/src/minhash.rs#L75-L92): bucket-pair generation is O(|bucket|²) per band. Acceptable at typical scales; document as a known scaling concern for pathologically dense buckets.

**[similarity/rust.py]**
- MinHash recall on large_dataset.arrow is 0.62 (389/625 passing pairs), below MIN_RECALL 0.85: all misses are containment pairs (overlap = 1.0) whose Jaccard is below the ~0.43 LSH candidate threshold (min(j·0.9, o·0.45)). The heuristic is carried over unchanged from the old filter. Follow-ups: add a containment-heavy recall fixture to tests/test_similarity.py; derive the candidate threshold from the overlap threshold and the cardinality ratio.

**[membership/rust.py]**
- `BloomRust` loses to `MembershipExact` (plain Python `frozenset` intersection) on every benchmarked shape: 0.60x algorithmic on large_dataset.arrow, 0.98x on the narrow related-frames shape, and 0.16x total (6x slower overall, parallelism included) on the wide 100-column shape — see tests/performance/results/membership_*.parquet. Building a fresh Bloom filter per column (`bloom_filter_bits`, one binding call each) carries enough constant overhead that it doesn't pay off at these value-set sizes; exact set intersection is already near-optimal here. Follow-ups: batch filter construction across columns in one Rust call instead of one call per column; only reach for Bloom below a distinct-value-count threshold (or drop it in favour of `MembershipExact` as the default and keep Bloom for pathologically large value sets where set intersection stops being cheap).

## Cross-cutting notes (not bugs, worth documenting)
- `encode_series` (formerly `series_to_u64`) returns `EncodedColumn { values, is_null }` — nulls are out-of-band (no in-band sentinel), floats are canonicalised (`-0.0`→`0.0`, all NaN payloads→one key). Null policy per module: entropy = null is a category; chi² = null rows dropped; minhash/bloom = nulls skipped. Keep this in mind when deriving MI from entropy + chi² outputs. Entropy additionally densifies `EncodedColumn` → `DenseColumn` (`0..card` ids) via `densify` — see "Entropy dense re-encoding" above.
- `String / categorical / enum / list / decimal types route through foldhash → u64` (categorical/enum are cast to their string value first, so a categorical `"x"` hashes identically to the string `"x"` and identically across frames regardless of physical code). Collision probability at 50K rows is ~6×10⁻¹¹ per pair — negligible for entropy/χ², irrelevant for MinHash (deterministic seed across columns). foldhash `FixedState` is NOT stable across crate versions/platforms — don't persist bloom bit arrays or minhash signatures across rebuilds for hashed dtypes.
- chi² and ARI share `build_contingency` (contingency.rs): dense-id counting, drop-null policy, marginals + non-zero cells. Entropy keeps its own counting (null-as-category — different policy by design).
