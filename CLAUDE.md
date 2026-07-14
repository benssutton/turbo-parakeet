# Project Objectives
Determine patterns and relationships between columns, both within and between dataframes.

# Technology
- Arrow columnar data format throughout
- Python with Rust extensions via pyo3/pyo3-polars for performance
- Polars as the primary dataframe library

# Python Environment
Conda environment: `C:\Users\Ben\miniconda3\envs\p312`
Python exe: `C:\Users\Ben\miniconda3\envs\p312\python.exe`

# Coding Style
Prioritise performance & simplicity.

# Analytical Functions

Five techniques for surfacing column relationships within and between dataframes:

**1. Bloom filter — set membership / containment**
Tests whether values from one column are present in another. High `membership_ratio(A → B)` ⇒ A is contained in B, suggesting A is a foreign key referencing B. The test is **asymmetric**: PK-PK relationships require reciprocal containment of *unique* values in both directions (run two filters).

**2. MinHash + LSH — set similarity**
Estimates Jaccard similarity (`|A∩B| / |A∪B|`) between column value sets. Post-LSH verification also computes the Overlap Coefficient (`|A∩B| / min(|A|,|B|)`). High Jaccard ⇒ significant overlap of distinct values; high Overlap Coefficient ⇒ one set is largely contained in the other (catches asymmetric relationships such as PK-FK).

**3. Chi-squared — independence test**
Tests whether two columns are *independent*. χ² detects **any form of association**, not just monotonic correlation, and tells us nothing about causality (it cannot distinguish cause, effect, redundancy, or confounding). At 50K rows p-values are vanishingly small for almost any non-trivial pair, so **Cramér's V (effect size) is the diagnostic**, not statistical significance.

**4. Joint entropy — building block for Mutual Information**
The plugin returns joint entropy H(A,B) and three-way H(A,B,C). Combined with marginal entropies, this yields Mutual Information: `MI(A,B) = H(A) + H(B) − H(A,B)`. **Normalised MI close to 1 ⇒ columns are highly redundant** (one predicts the other); NMI close to 0 ⇒ independent. Separately, joint entropy approaching `log₂(n_rows)` indicates near-unique combinations — a different diagnostic.

**5. Run-length spans — REE compression candidacy**
Quantifies spans of consecutive identical values in physical column order. Average span > 2 (compression ratio > 2×) flags columns that benefit from Apache Arrow Run-End Encoding.

# Project Structure

```
turbo-parakeet/
├── services/
│   └── analytics/                  # Single service package
│       ├── pyproject.toml          # maturin build config
│       ├── Cargo.toml
│       ├── analytics/              # Compiled Rust plugin (.pyd)
│       │   ├── __init__.py
│       │   └── analytics.pyd
│       ├── src/                    # Rust source
│       │   ├── lib.rs
│       │   ├── shared.rs           # Shared utils (encode_series, build_column_cache_par, densify, build_dense_cache_par)
│       │   ├── entropy.rs          # Joint entropy (2-way and 3-way), dense-id encoded
│       │   ├── chi_squared.rs      # Chi-squared independence test
│       │   ├── bloomfilter.rs      # Bloom filter
│       │   └── minhash.rs          # MinHash + LSH candidates
│       ├── bloom_filter.py         # BloomFilter Python wrapper
│       ├── chi_squared.py          # polars-ds baseline (pure Python)
│       ├── DeterministicSimilarityFilter.py   # Brute-force Jaccard/Overlap (ground truth)
│       ├── MinHashLSHFilter.py     # Probabilistic filter (Rust plugin-backed)
│       ├── MinHashLSHFilter_datasketch.py     # Probabilistic filter (datasketch)
│       └── run_length.py           # Pure-Python REE compression analysis
└── tests/
    ├── conftest.py                 # pytest markers (slow)
    ├── data/
    │   └── large_dataset.arrow     # 50K rows, 101 columns
    ├── test_similarity_filters.py  # Correctness + recall tests for similarity filters
    ├── benchmark_bloom_filter.py   # Bloom filter benchmark vs fastbloom-rs
    ├── benchmark_chi_squared.py    # Chi-squared benchmark: Rust vs polars-ds vs scipy
    ├── benchmark_entropy.py        # Joint entropy benchmark: plugin vs native Polars
    └── benchmark_rle.py            # Run-length analysis benchmark (column_run_stats)
```

# Rust Plugin (analytics)
Build: `maturin develop --release` from `services/analytics/`

Exposed functions:
- `pairwise_joint_entropy(df)` — all column pairs
- `threeway_joint_entropy(df)` — all column triplets (no cap; C(101,3) = 166,650 at 101 cols, ~65s at 50K rows)
- `pairwise_chi_squared(df, pairs)` — chi-squared + p-value + Cramer's V
- `minhash(df, ...)` / `lsh_candidates(...)` — MinHash signatures and LSH buckets
- `membership_ratio(df)` — Bloom filter membership across n columns

# Current Focus
Chi-squared Rust plugin is complete and validated (~17x faster than polars-ds baseline, matches scipy at rtol=1e-4). MinHashLSH similarity filters are complete with correctness and recall tests passing. Run-length analysis (Approach A, pure Python via `polars.Expr.rle()`) is complete: ~142 ms / 101 cols / 50K rows.

Joint entropy (2-way and 3-way) is dense-id encoded (see below): 83.5µs/pair, 170.3µs/triplet at 50K rows/101 cols — roughly 2x faster than the original hash-tuple implementation, and the 3-way/2-way speedup-vs-native-Polars gap that motivated the change is closed (3-way now outpaces 2-way's multiplier rather than trailing it).

## Entropy dense re-encoding (shared.rs + entropy.rs)
Columns feeding `pairwise_joint_entropy`/`threeway_joint_entropy` are dictionary-encoded to dense ids `0..card` via `densify`/`build_dense_cache_par` (nulls become their own id, so the per-row loop carries no separate null mask). Joint keys are then combined arithmetically (`(a·Kb + b)·Kc + c`) instead of hashing multi-field tuples:
- joint space ≤ 2²⁰ → flat-array counting, no hashing (thread-local scratch array + touched-slot list for O(distinct) reset)
- joint space > 2²⁰ but ≤ u64::MAX → single-u64 hash map
- joint space > u64::MAX (only reachable past ~2.6M rows) → u128-packed hash map fallback

This encoding is entropy/chi²-only — it's value-relabeling and has no cross-column/cross-frame identity, which is fine since entropy is invariant under injective relabeling. Bloom/MinHash must keep using the value-stable `encode_series`/`build_column_cache_par` path since they compare actual values across columns and frames.

# Next Steps

## Implementation cleanups

**[chi_squared.rs](services/analytics/src/chi_squared.rs)**
- Follow-up (not yet done): surface a warning (or a `low_expected_count` boolean field) when expected cell counts fall below 5. This is the standard chi-squared assumption and ignoring it can inflate χ² on sparse contingency tables. The same dense re-encoding used for entropy could also speed up chi²'s contingency-table build, since it has the identical relabeling-invariance property — worth doing alongside the warning.

**[bloomfilter.rs](services/analytics/src/bloomfilter.rs)**
- ~~silent state discard on `existing_filter` length mismatch~~ **Fixed**: `m` is now consistently bits with `ceil(m/8)`-byte arrays; wrong-sized filters raise `ComputeError`, and `validate_bit_array` guards the unchecked bit reads.
- [bloom_filter.py:71](services/analytics/bloom_filter.py#L71), [:86](services/analytics/bloom_filter.py#L86), [:99](services/analytics/bloom_filter.py#L99): replace `list(self.bit_array)` with a direct bytes pass-through. At fp=1%, n=50K the filter is ~60KB and is being copied to a Python list-of-ints on every membership call.

**[minhash.rs](services/analytics/src/minhash.rs)**
- [minhash.rs:266](services/analytics/src/minhash.rs#L266): `compute_signature_for_series_with_coeffs` swallows `series_to_u64` errors and returns `vec![u32::MAX; num_perm]` — a useless signature that gets silently included downstream. Propagate the error instead.
- [minhash.rs:50](services/analytics/src/minhash.rs#L50): the `threshold` kwarg in `lsh_candidates` is read and discarded (`let _ = kwargs.threshold`). Either implement candidate-stage threshold filtering or remove the parameter from `LSHKwargs`.
- [minhash.rs:87-99](services/analytics/src/minhash.rs#L87-L99): bucket-pair generation is O(|bucket|²) per band. Acceptable at typical scales; document as a known scaling concern for pathologically dense buckets.

**[MinHashLSHFilter.py](services/analytics/MinHashLSHFilter.py)**
- [line 31](services/analytics/MinHashLSHFilter.py#L31): `lsh_threshold = min(jaccard*0.9, overlap*0.45)`. Add a comment explaining the 0.45 fudge factor for using a Jaccard-based LSH index to recover Overlap-Coefficient candidates (the LSH s-curve is calibrated against Jaccard, so OC-only matches need a lower effective threshold to make it through the candidate stage).

## Cross-cutting notes (not bugs, worth documenting)
- `encode_series` (formerly `series_to_u64`) returns `EncodedColumn { values, is_null }` — nulls are out-of-band (no in-band sentinel), floats are canonicalised (`-0.0`→`0.0`, all NaN payloads→one key). Null policy per module: entropy = null is a category; chi² = null rows dropped; minhash/bloom = nulls skipped. Keep this in mind when deriving MI from entropy + chi² outputs. Entropy additionally densifies `EncodedColumn` → `DenseColumn` (`0..card` ids) via `densify` — see "Entropy dense re-encoding" above.
- `String / categorical / enum / list / decimal types route through foldhash → u64` (categorical/enum are cast to their string value first, so a categorical `"x"` hashes identically to the string `"x"` and identically across frames regardless of physical code). Collision probability at 50K rows is ~6×10⁻¹¹ per pair — negligible for entropy/χ², irrelevant for MinHash (deterministic seed across columns). foldhash `FixedState` is NOT stable across crate versions/platforms — don't persist bloom bit arrays or minhash signatures across rebuilds for hashed dtypes.
