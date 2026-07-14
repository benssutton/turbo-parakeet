# Adjusted Rand Index (ARI) — Design

**Date:** 2026-07-14
**Status:** Approved (pending user spec review)

## Goal

Add pairwise Adjusted Rand Index as the sixth analytical technique in
`services/analytics`, following the existing Rust-plugin conventions
(`pairwise_chi_squared` is the closest template). As part of the work,
extract a shared contingency-table builder used by both ARI and a
refactored chi-squared module (Approach C from brainstorming), and add
an `n_valid` field to both plugins' outputs.

## Decisions (from brainstorming)

- **Null policy: drop rows** where either column is null (pairwise
  deletion, same as chi²). Matches sklearn semantics: a null row belongs
  to no cluster and must not vote on partition agreement.
- **Output fields:** `col_a`, `col_b`, `ari`, `n_valid` — where
  `n_valid` is the row count remaining after null-dropping, so consumers
  can discount scores computed over tiny overlaps.
- **Chi² output extension (user request):** `pairwise_chi_squared` also
  gains an `n_valid: UInt32` field with the same meaning.
- **Approach C:** shared contingency builder consumed by both ARI and
  chi², rather than per-module counting code.

## Architecture

### shared.rs — `DenseColumn.null_id` extension

`DenseColumn` gains `null_id: Option<u32>`; `densify` records the dense
id assigned to nulls (if the column has any). Consumers with a drop-null
policy skip rows carrying that id. Entropy ignores the field
(null-as-category policy) — no behavior change there.

### contingency.rs — new module: shared contingency-table builder

One clear purpose: given two `DenseColumn`s and the row count, drop rows
where either side is null and return the joint distribution.

```rust
pub(crate) struct ContingencyTable {
    /// Observed non-zero cells: (a_id, b_id, count).
    pub cells: Vec<(u32, u32, u64)>,
    /// Marginal counts over valid rows, indexed by dense id (0..card).
    /// Ids unseen among valid rows (including a null id) hold 0.
    pub marg_a: Vec<u64>,
    pub marg_b: Vec<u64>,
    /// Rows where both columns are non-null.
    pub n_valid: u64,
}

pub(crate) fn build_contingency(
    a: &DenseColumn,
    b: &DenseColumn,
    n_rows: usize,
) -> ContingencyTable
```

Internals (invisible to consumers): the flat-array/hash counting
strategy proven in entropy.rs — joint keys combined arithmetically
(`a_id · Kb + b_id`), flat `Vec<u32>` counting with a touched-slot list
when `Ka·Kb ≤ 2²⁰`, single-u64 hash map otherwise (the pair product
always fits u64 since ids are u32). Thread-local scratch buffers, reset
between calls, zero steady-state allocation except the compact output.
Marginals are card-sized flat arrays — trivially cheap.

### ari.rs — new module: `pairwise_adjusted_rand` plugin

Skeleton copied from `chi_squared.rs`: input validation (non-empty,
uniform length; <2 columns → empty struct), pair resolution via
`resolve_pairs` or all N-choose-2, `build_dense_cache_par`, Rayon
`par_iter` over pairs, struct-series assembly.

Per-pair body: `build_contingency`, then ARI from the table.

**Math.** With `index = Σᵢⱼ C(nᵢⱼ,2)` over cells, `A = Σᵢ C(aᵢ,2)` and
`B = Σⱼ C(bⱼ,2)` over marginals, and `total = C(n_valid, 2)`:

```
expected  = A·B / total
max_index = (A + B) / 2
ARI       = (index − expected) / (max_index − expected)
```

Sums accumulate in `u64` (each sum ≤ C(n,2); overflow only past ~6×10⁹
rows), the final expression in `f64` (A·B would overflow u64).

**Edge cases (mirror sklearn):**
- `n_valid == 0` → `ari = NaN`, `n_valid = 0`.
- Degenerate denominator, `max_index == expected` (e.g. both columns
  constant over valid rows, or both all-singletons) → `ari = 1.0`,
  sklearn's convention for trivially perfect agreement.

**Output struct** (`pairwise_adjusted_rand`):
`col_a: String, col_b: String, ari: Float64, n_valid: UInt32`.

### chi_squared.rs — refactor onto the shared builder

- `pairwise_chi_squared_impl` switches from
  `build_column_cache_par`/`EncodedColumn` to
  `build_dense_cache_par`/`DenseColumn`.
- `compute_chi_squared` consumes a `ContingencyTable` instead of
  building three ad-hoc hash maps. Downstream logic is unchanged:
  `unique_a`/`unique_b` = count of non-zero marginal entries; degenerate
  (<2 unique either side) → NaN row; `χ² = Σ O²/E − N` over observed
  cells with `E = marg_a[a]·marg_b[b]/n`; p-value via `ChiSquared::sf`;
  Cramér's V; `low_expected_count` from min non-zero marginals.
- Output struct gains `n_valid: UInt32` (0 for the all-null case, which
  currently yields the NaN row). `build_empty_result` and the output
  type function add the field.

Values must be numerically unchanged from today (validated by existing
tests; see Testing).

### lib.rs

Add `mod contingency;` and `mod ari;`.

### Python API — `analytics/__init__.py`

```python
def pairwise_adjusted_rand(
    df: pl.DataFrame | pl.LazyFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pl.DataFrame
```

Identical shape to `pairwise_chi_squared`: collects LazyFrames, passes
`pairs` kwarg through, returns a single `"pairwise_adjusted_rand"`
struct column. Docstring documents the null policy and the two edge-case
conventions. The `pairwise_chi_squared` docstring gains the `n_valid`
field description.

## Testing

### Rust unit tests

- `contingency.rs`: null-dropping (rows with either side null excluded
  from cells, marginals, and n_valid), flat vs hash path agreement on
  the same data, marginal/cell consistency (Σ cells = Σ marg_a = Σ
  marg_b = n_valid), empty/all-null inputs.
- `ari.rs`: hand-computed table with a known ARI (verified against
  sklearn offline and cited in a comment), identical columns → 1.0,
  independent columns → ≈ 0, null-drop behavior, constant column → 1.0
  convention, all-null overlap → NaN + n_valid 0, `pairs` kwargs
  resolution and error cases (same matrix as chi²'s tests).
- `chi_squared.rs`: existing tests unchanged and passing after the
  refactor; add an `n_valid` assertion to an existing null-policy test.

### Pytest — `tests/test_adjusted_rand.py`

Correctness vs `sklearn.metrics.adjusted_rand_score` on the shared
`conftest.make_dataset` frame (1000 rows). Eligible columns:
`boolean_*`, `uint32_*`, `cat_*`, `float64_*` → 12 columns → 66 pairs;
nested `list_*`/`arr_*` excluded (sklearn cannot label them). For each
pair the test drops rows where either column is null (mirroring the
plugin's policy), feeds the remaining values to sklearn, and asserts
agreement at rtol=1e-10 (the computation is exact integer counting on
both sides; only the final division is floating-point). Also asserts
`n_valid` equals the pandas/polars-computed non-null-overlap count.

Chi² regression: existing `tests/test_chi_squared.py` must still pass
(rtol=1e-4 vs polars-ds, as today).

### Speed benchmark — `tests/speed_benchmark_adjusted_rand.py`

Follows `speed_benchmark_chi_squared.py` format on
`tests/data/large_dataset.arrow` (50K rows, 101 cols):
- Rust plugin batch, all eligible pairs — 3 runs averaged.
- sklearn `adjusted_rand_score` per-pair Python loop (with identical
  null-drop preprocessing, preprocessing time included) — 3 runs
  averaged.
- Report table: avg times, speedup vs plugin, mismatch count at
  rtol=1e-9.

Chi² benchmark: re-run `speed_benchmark_chi_squared.py` after the
refactor and record before/after plugin timing in the PR/commit notes
(the shared builder is expected to speed chi² up; quantify it).

## Documentation — CLAUDE.md

- Analytical Functions: add technique **6. Adjusted Rand Index —
  partition agreement** (ARI ∈ [−0.5, 1]; 1 = identical partitions,
  ≈0 = chance-level agreement; unlike NMI it is chance-adjusted and
  unlike χ²/Cramér's V it measures *partition identity*, not just
  association).
- Exposed functions: add `pairwise_adjusted_rand(df, pairs)`; note
  chi²'s new `n_valid` field.
- Project structure: add `contingency.rs`, `ari.rs`,
  `test_adjusted_rand.py`, `speed_benchmark_adjusted_rand.py`.
- Corrections: mark the chi² `low_expected_count` item done (already
  implemented with tests); mark the "chi² could adopt dense re-encoding"
  follow-up resolved by this work.
- Cross-cutting notes: chi² and ARI share `build_contingency`
  (drop-null); entropy keeps its own counting (null-as-category).

## Out of scope

- No changes to entropy's counting internals (the contingency builder
  is not suitable: entropy needs no marginals and keeps nulls).
- No MinHash/Bloom changes (they require value-stable encodings; dense
  ids carry no cross-frame identity).
- No cardinality caps in the plugin; callers choose sensible pairs, as
  with chi².
