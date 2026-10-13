# Runs test (k-category, one-sample) — design

Status: approved 2026-10-13. Adds the first **order-dependent** per-column technique, `analytics.runs_test`,
on the uniform contract (spec 2026-09-24), with a Rust kernel in `techniques/single_column/ordered/` and an
exact Polars reference.

## 1. Goal

Find columns whose row order makes Arrow run-end encoding (REE) worthwhile. REE stores one entry per run of
equal consecutive values, so its size is driven by the run count R and the mean run length n / R. This spec
covers the technique and its speed benchmark. Calling it from the StreamingRecommender (on the reservoir's
contiguous row blocks) is a later spec.

The technique reports metrics only. Whether REE is smaller for a given dtype is the recommenders' decision:
they already measure real IPC / ZSTD sizes.

## 2. Statistic choice

**k-category runs of equal values**, not the classic binary test. The textbook one-sample runs test
(statsmodels `runstest_1samp`) first dichotomises the column at a cutoff (`x >= mean` / `median`) and counts
runs of the 0/1 indicator. That is an indirect signal for REE:

- distinct values merge, so the indicator's run count can be far below R — a sorted all-distinct column
  `1..8` has 2 indicator runs (strongly "clustered") but R = 8 (REE useless);
- on low-cardinality columns the cutoff often puts every row on one side (e.g. 90 % zeros, median 0 → all
  rows `>= 0` → one run, statistic undefined);
- strings, Categoricals and Enums — the best REE candidates — have no meaningful mean or median.

Counting runs of equal values makes R exactly the REE run-end array length. Its distribution under random
ordering of the same multiset has closed forms (Barton & David 1957). For n rows with k categories of
counts nᵢ (null is a category), S₂ = Σnᵢ², S₃ = Σnᵢ³:

- R = 1 + #{i : xᵢ ≠ xᵢ₋₁}
- E[R] = 1 + (n² − S₂) / n
- Var[R] = [S₂(S₂ + n(n+1)) − 2nS₃ − n³] / (n²(n−1))  (for k = 2 this reduces to 2n₁n₂(2n₁n₂ − n) / (n²(n−1)))
- z = (R − E[R]) / √Var[R]
- p = Φ(z)

**p is one-sided, lower tail**: P(R ≤ observed) under random ordering — "fewer runs than chance", the only
direction relevant to REE. Columns that alternate more than chance (`A B A B …`) give z > 0 and p near 1.
p uses the asymptotic normal approximation, without continuity correction.

**p is not the effect size.** At large n, p vanishes even when the mean run length is barely above 1
(n = 10⁶, R = 900 000 vs E[R] ≈ 999 000: p ≈ 0, mean run length 1.1). `mean_run_length` is the effect size;
p says only whether the clustering is real. (The same reasoning makes Cramér's V, not p, chi-squared's
diagnostic.)

Name: `runs_test`, not `wald_wolfowitz` — Wald–Wolfowitz's original test is the two-sample one
(statsmodels `runstest_2samp`).

## 3. Python package and contract

```
analytics/runs_test/
├── __init__.py   REFERENCE = "RunsTestPolars"; IMPLEMENTATIONS = ("RunsTestRust", "RunsTestPolars")
├── base.py       RunsTest(Technique): SCOPE = "per_column", ARITY = 1
├── rust.py       RunsTestRust   → _plugin.runs_test (one call per frame, columns grouped)
└── polars.py     RunsTestPolars → ★ exact reference
```

`RunsTestRust().add({"t": frame}).result()`. No keyword parameters (no conclusions, so no thresholds).

| Metric | dtype | Meaning |
|---|---|---|
| `n_rows` | UInt64 | n, nulls included |
| `n_runs` | UInt64 | R — maximal runs of equal values = REE run-ends length |
| `expected_runs` | Float64 | E[R] under random ordering of the same values |
| `z` | Float64 | (R − E[R]) / √Var[R] |
| `p_value` | Float64 | one-sided lower tail, Φ(z) |
| `mean_run_length` | Float64 | n / R |

Descriptor `dtype` (Python `str(dtype)`), as GCD and Describe. `CONCLUSIONS = {}`. All metric dtypes export to
native Arrow.

Order dependence needs no new SCOPE: `per_column` already enumerates one column at a time, and row order only
matters inside the kernel. The contract guarantees implementations see rows in the frame's own order (`add()`
never reorders). The "ordered" distinction lives in the Rust taxonomy and in a new CLAUDE.md heading,
"1b. Per-column, order-dependent".

The `RunsTest` docstring states: R's meaning for REE, the k-category generalisation with the formulas above,
the one-sided lower-tail p, the normal approximation without continuity correction, and that
`mean_run_length` — not p — is the effect size.

## 4. Rust kernel

`src/techniques/single_column/ordered/{mod.rs, runs_test.rs}`, importing only `common`;
`single_column/mod.rs` gains `pub(crate) mod ordered;`.

One pass per column, with hash-map work per **run**, not per row:

```rust
struct Partial<K> {                  // summary of one contiguous chunk
    rows: u64,
    runs: u64,
    first: Option<K>,                // Option<K>: null is a value (None)
    last:  Option<K>,
    counts: HashMap<Option<K>, u64>, // foldhash; += run length once per run
}
```

- **Scan:** compare xᵢ with xᵢ₋₁ (null-aware: `None == None` continues a run); at each run end,
  `counts[value] += run_len`, `runs += 1`.
- **Merge** (in row order): `runs = a.runs + b.runs − (a.last == b.first)`, counts summed. Associative, so
  rayon runs 2¹⁶-row chunks in parallel and folds them left to right; columns are also processed in parallel
  (as GCD). The same merge combines reservoir blocks later.
- **Keys** are compared, not only hashed, so results are exact:

| dtype | K |
|---|---|
| Int/UInt 8–64, Date, Datetime, Duration, Time | native physical integer |
| Boolean | `bool` |
| Float32/64 | canonical bits (`common::encode::canon_f32` / `canon_f64`: −0.0 = 0.0, one NaN) |
| Decimal | `i128` physical (scale fixed per column) |
| String, Binary | `&str` / `&[u8]` borrowed from the array |
| Categorical / Enum | physical code `u32` (bijective with the string within a column; no cast) |

- **Statistics:** S₂, S₃ from the merged counts. E[R] in f64. The variance numerator
  S₂(S₂ + n(n+1)) − 2nS₃ − n³ is evaluated **exactly in `i128`** with checked arithmetic (terms ~n⁴ cancel
  heavily on near-constant columns); on overflow (n ≳ 3·10⁹) it falls back to f64. p = Φ(z) via statrs
  `Normal`.
- **Entry point:** `bindings/api/techniques.rs::runs_test(batch: &RecordBatch) -> Result<RecordBatch>` →
  `column, n_rows, n_runs, expected_runs, z, p_value, mean_run_length`; registered in `bindings/python.rs`,
  wrapped by `_plugin.runs_test`. The C ABI is unchanged.
- **Chunked input:** each column is rechunked once (zero-copy if already one chunk) before the kernel splits
  it into its own work chunks, so input chunk boundaries never affect R.

**Reference** `RunsTestPolars`: R from `ne_missing(shift(1))`, counts from `value_counts` (after `+ 0.0`
and `fill_nan` to one NaN on floats, since `value_counts` separates −0.0 and 0.0), S₂, S₃ and the variance
numerator as Python ints (exact at any n); only the final division and Φ are floating point. It shares no
code with the kernel.

## 5. Eligibility and edge cases

Eligible: Int/UInt 8–64, Boolean, Float32/64, Decimal, Date, Datetime, Duration, Time, String, Binary,
Categorical, Enum. Ineligible (listed, never dropped): List, Array, Struct, Null, Object, Int128/UInt128
(`WIDE_INTEGERS`). Nested types are deferred (YAGNI); a hash-and-compare key can add them later. The base's
`ELIGIBLE` tuple mirrors the Rust match arms; a test keeps them in sync.

| Case | status | n_rows | n_runs | expected_runs | z, p_value | mean_run_length |
|---|---|---|---|---|---|---|
| zero rows | ineligible | null | null | null | null | null |
| n = 1 | computed | 1 | 1 | 1 | NaN | 1 |
| one distinct value (incl. all-null) | computed | n | 1 | 1 | NaN (Var = 0) | n |
| every value distinct | computed | n | n | n | NaN (Var = 0: R is fixed) | 1 |
| Var > 0 | computed | n | R | E[R] | finite | n / R |

Zero rows are ineligible, matching the ordered scope's rule. NaN = computed but undefined (Rust yields NaN
for 0/0, never null). Nulls are one category and consecutive nulls one run, as in Arrow REE's `values` child:
`1, null, null, 1` → R = 3. Floats: −0.0 and 0.0 share a run, every NaN joins one NaN run. String,
Categorical and Enum holding the same sequence give identical results. An unsupported dtype reaching Rust is
`InvalidInput` → ValueError (the base's eligibility check prevents it in normal use).

## 6. Testing

`tests/test_runs_test.py`, via `tests/harness.py`:

1. **Contract** (both implementations): schema, canonical order, ineligible rows null, `to_arrow()` native.
2. **Reference agreement** on `large_dataset.arrow` and seeded fixtures: `n_rows`, `n_runs` exact;
   `expected_runs`, `z`, `p_value`, `mean_run_length` within `RTOL = 1e-9`; NaN matches NaN.
3. **Known answers** (both implementations): `A A B B B C` → R = 3, E[R] = 14/3; alternating `0 1 0 1 0 1`
   → z > 0, p > 0.5 (one-sided direction); `1, null, null, 1` → R = 3; `[0.0, -0.0, NaN, NaN]` → R = 2;
   n = 1, all-null and constant columns → NaN z / p; zero-row → ineligible; String / Categorical / Enum
   identical; a multi-chunk input with a run spanning the boundary = the single-chunk result; every eligible
   dtype once.
4. **Statistical correctness** (in place of the conclusions block — there are none):
   - statsmodels cross-check: on two-valued columns, `z` equals `Runs(x).runs_test(correction=False)`'s z
     and `p_value` = Φ(z). Visible skip if statsmodels is missing; add it to the optional test
     dependencies.
   - Permutation Monte Carlo: for a seeded column (k = 4, n = 40), mean and variance of R over 20 000
     seeded shuffles match `expected_runs` and the closed-form variance within Monte Carlo tolerance.
   - Cancellation at scale: a near-constant column, n = 10⁶ — Rust's i128 variance equals the reference's
     Python-int variance.

Rust unit tests in `runs_test.rs`: merged partials equal the one-chunk result at every split point of a small
column; the i128 overflow path falls back to f64.

## 7. Benchmark

`tests/performance/benchmark_runs_test.py`, ~20 lines calling `harness.run("analytics.runs_test", …)`:

| Shape | Purpose |
|---|---|
| `large_dataset.arrow` | realistic mix |
| narrow 10M × 4, long runs (~1 000 rows), int + string | REE-friendly: per-run hashing → near memory-bound |
| narrow 10M × 4, random high cardinality | worst case: R ≈ n, one hash insert per row |
| wide 1M × 100, mixed run lengths | parallelism across columns |

The harness reports algorithmic (Polars ÷ Rust@1), parallel and total speedups. New seeded generators in
`tests/datagen.py` — `runs_int(n, cols, mean_run, cardinality)` and a string variant — with a
`tests/test_datagen.py` case. Accuracy is not benchmarked separately: both implementations are exact.

## 8. Documentation

- Docstrings on `RunsTest` (§3) and on each implementation.
- CLAUDE.md: a new section "1b. Per-column, order-dependent" describing `analytics.runs_test` (statistic,
  one-sided lower-tail p, normal approximation, `mean_run_length` as effect size); `runs_test` in the
  project-structure tree, the Rust source layout and the private `_plugin` function list; the "Not
  implemented" note and "Next technique" line updated.
- ToDo.md: the runs-test item removed.

## 9. Out of scope

Integration with the StreamingRecommender / reservoir; REE size estimation or a "beneficial" conclusion
(recommenders' concern); exact small-sample p-values; nested dtypes; the binary (dichotomised) runs test.
