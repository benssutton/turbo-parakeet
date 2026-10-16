# Skip-index techniques (minmax, set) — design

Status: approved 2026-10-16. Adds two **order-dependent** per-column techniques, `analytics.minmax_index` and
`analytics.set_index`, on the uniform contract (spec 2026-09-24), with Rust kernels in
`techniques/single_column/ordered/skip_index/` and exact Polars references.

## 1. Goal

Measure how well a column, in its current row order, suits ClickHouse's `minmax` and `set(max_size)` data-skipping
indexes. A skip index stores one summary per **index block** (`GRANULARITY` granules of `index_granularity` rows)
and lets a query skip a block whose summary excludes the predicate value. Future data from the same generation
process gives no guarantee of where granule boundaries fall, so every alignment is evaluated.

The techniques report metrics only. Choosing an index, `max_size` or `GRANULARITY` is the recommenders'
decision. Integration with the recommenders is a later spec; the kernel is shaped for it (§5).

## 2. Model

### 2.1 Windows

n rows; the **population** is the non-null values: N′ rows, d distinct values, value v occurring n_v times.
The window is w = `index_granularity × granularity` rows and slides one row at a time over start rows
0 … n − w (n − w + 1 windows). A column shorter than w has one window of all n rows. Each window is the index
block under one possible alignment, so the mean over windows is the expectation under a uniformly random
alignment, and the std is the spread of a randomly placed block.

### 2.2 Queries

Point queries `col = v`, with v drawn from the population in two ways, each reported:

- **row-weighted** (`skip_rows_*`): P(v) = n_v / N′ — frequent values are queried more (lookups);
- **distinct-weighted** (`skip_values_*`): P(v) = 1 / d — every distinct value equally likely (rare-value search).

A window's **skip** is the probability that a query of that kind can skip it. Toy example (§7, known answers):
`A A A B A A A C`, w = 4 → the set index skips 12.5 % row-weighted, 33.3 % distinct-weighted.

### 2.3 Per-window skip

**minmax.** lo, hi = min / max of the window's non-null values. The window is skipped for v iff v < lo or v > hi;
an all-null window is skipped for every v (skip 1, both weightings).

- row-weighted: (number of population rows with value outside [lo, hi]) / N′
- distinct-weighted: (number of distinct values outside [lo, hi]) / d

**set.** S_W = the window's distinct non-null values; the stored set also holds NULL when the window has a null
(ClickHouse stores NULL for Nullable columns), so the window's **entry count** is e_W = |S_W| + [window has a
null]. If `max_size` > 0 and e_W > `max_size`, the set is stored empty and the window cannot be skipped (skip 0).
Otherwise:

- row-weighted: 1 − Σ_{v ∈ S_W} n_v / N′
- distinct-weighted: 1 − |S_W| / d

NULL never helps a `col = v` query; it counts only toward `max_size`.

### 2.4 Value order (minmax)

Numbers and temporals: natural order on the physical value (Decimal by value at the column's scale). Boolean:
false < true. String and Binary: bytewise. Categorical: by its string. Enum: by category order (as ClickHouse
Enum compares by its numeric value). Floats are canonical: −0.0 = 0.0, all NaN are one value, sorting after every
number. The set index uses the same identity (equal values are equal entries).

### 2.5 Random-order baseline (closed form)

What the same n rows (nulls included) would give in uniformly random order. Building block — the probability
that a w-row window of a random ordering contains none of m specific rows:

  avoid(m) = C(n − m, w) / C(n, w), and 0 when n − m < w

evaluated in log space with `lgamma` (O(1) per value, O(d) per column). With n < w, w is n (avoid(m) = [m = 0]).

**minmax (exact).** L_v = non-null rows below v, G_v = non-null rows above v. The window skips v iff it holds no
row ≤ v or no row ≥ v; both events contain the all-null window, which is subtracted once:

  P_v = avoid(N′ − G_v) + avoid(N′ − L_v) − avoid(N′)

**set.** The window's entries are the distinct non-null values plus, when n > N′, the NULL entry (count
n − N′). q_u = avoid(n_u) is the probability entry u is absent. With M = `max_size`:

- μ = Σ_u (1 − q_u) — exact `expected_window_distinct`;
- σ² = Σ_u q_u(1 − q_u) + Σ_{u ≠ u′} [avoid(n_u + n_u′) − q_u q_u′] — the exact variance of the entry count;
- μ_v = Σ_{u ≠ v} (1 − avoid_{n − n_v}(n_u)) — the exact expected entry count given v absent (the window is
  then a uniform w-subset of the other n − n_v rows; avoid_{n − n_v} is avoid over n − n_v rows);
- `expected_overflow_fraction` = 1 − Ψ(M + ½ − μ);
- P_v ≈ q_v · Ψ(M + ½ − μ_v), where Ψ(x) = Φ(x / σ), or [x ≥ 0] when σ = 0 (a normal approximation of the
  entry count, with continuity correction and the unconditional σ);
- with M = 0 (unlimited) Ψ ≡ 1, overflow is 0 and the baseline is exact.

Entries are grouped by count: values with equal counts have equal q, so every sum runs over the C distinct
counts with multiplicities, and the pair sum and the μ_v are O(C²). Since 1 + 2 + … + C ≤ n, C² ≤ 2n: linear in
the worst case and far smaller in practice. Against permutation Monte Carlo (12 random Zipf columns, M near μ)
the approximation's worst error was 0.010 on every set baseline metric; a cheaper variant (independent
presences, μ_v = μ − (1 − q_v)) erred by up to 0.06.

**Both:** `expected_skip_rows` = Σ_v (n_v / N′)·P_v, `expected_skip_values` = (1/d)·Σ_v P_v.

Observed minus expected is the gain due to ordering; expected itself is the gain due to rarity, which any load
order keeps.

## 3. Python packages and contract

```
analytics/minmax_index/                       analytics/set_index/
├── __init__.py  REFERENCE = "MinMaxIndexPolars"   ├── __init__.py  REFERENCE = "SetIndexPolars"
│                IMPLEMENTATIONS = (Rust, Polars)  │                IMPLEMENTATIONS = (Rust, Polars)
├── base.py      MinMaxIndex(Technique)           ├── base.py      SetIndex(Technique)
├── rust.py      MinMaxIndexRust → _plugin        ├── rust.py      SetIndexRust → _plugin
└── polars.py    MinMaxIndexPolars ★              └── polars.py    SetIndexPolars ★
```

`MinMaxIndexRust(index_granularity=8192, granularity=1).add({"t": frame}).result()`;
`SetIndexRust(index_granularity=8192, granularity=1, max_size=100)`. Keywords are keyword-only;
`index_granularity ≥ 1`, `granularity ≥ 1`, `max_size ≥ 0` (0 = unlimited) — otherwise ValueError. One window
size per instance: compare GRANULARITY 1 / 2 / 4 with three instances. SCOPE `per_column`, ARITY 1, descriptor
`dtype` (Python `str(dtype)`), `CONCLUSIONS = {}`.

| Metric | dtype | Both | Meaning |
|---|---|---|---|
| `n_rows` | UInt64 | ✓ | n, nulls included |
| `n_non_null` | UInt64 | ✓ | N′ |
| `n_distinct` | UInt64 | ✓ | d (non-null) |
| `window_rows` | UInt64 | ✓ | effective w = min(n, index_granularity × granularity) |
| `n_windows` | UInt64 | ✓ | n − w + 1 (1 when n < w) |
| `skip_rows_mean`, `skip_rows_std` | Float64 | ✓ | row-weighted skip over windows (population std) |
| `skip_values_mean`, `skip_values_std` | Float64 | ✓ | distinct-weighted skip over windows |
| `expected_skip_rows`, `expected_skip_values` | Float64 | ✓ | random-order baseline (§2.5) |
| `window_distinct_mean` | Float64 | set | mean entry count e_W (NULL included) |
| `window_distinct_max` | UInt64 | set | max e_W |
| `overflow_fraction` | Float64 | set | share of windows with e_W > `max_size` (0 when unlimited) |
| `expected_window_distinct` | Float64 | set | μ (§2.5) |
| `expected_overflow_fraction` | Float64 | set | §2.5 |

All metric dtypes export to native Arrow. The `ELIGIBLE` tuple of the runs test moves to `analytics/_dtypes.py`
as `ORDERED_SCALARS` and is shared by all three ordered techniques; a test keeps it in sync with the Rust match
arms.

### 3.1 Eligibility and edge cases

Eligible: Int/UInt 8–64, Boolean, Float32/64, Decimal, Date, Datetime, Duration, Time, String, Binary,
Categorical, Enum. Ineligible (listed, never dropped): List, Array, Struct, Null, Object, Int128/UInt128
(`WIDE_INTEGERS`), zero-row columns.

| Case | skip / expected metrics | other |
|---|---|---|
| zero rows | null (ineligible) | all null |
| all-null column (d = 0) | NaN (no query value exists) | counts computed; set: e_W = 1 every window |
| n < w | one window | std 0 |
| one window | — | std 0 |

## 4. Rust kernels

`src/techniques/single_column/ordered/skip_index/`, importing only `common`:

- **`mod.rs`** —
  - `Population`: the column as dense **rank ids** `0..d` in §2.4 order (nulls out of band), `counts[d]`,
    `cum[d+1]` (prefix sums), n, N′, n_null.
  - `avoid(n, w, m)` (log space, `libm::lgamma`); Φ via `libm::erfc` (as the runs test).
  - `WindowSums`: each window's skips are integer numerators — (N′ − rows covered) over N′, (d − values
    covered) over d, 0 on overflow — summed exactly as Σa and Σa² in `u128`, plus, for set, Σ e_W, max e_W and
    the overflow count. `merge` adds, so it is exact and associative whatever the split; windows never cross
    segments. mean = Σa / (K·den), std = √(K·Σa² − (Σa)²) / (K·den), the numerator exact in i128 (f64 on
    overflow, as the runs test's variance).
- **`minmax.rs`** — `scan(&Population, segment) -> WindowSums`: two monotonic deques over the segment's rank ids
  (nulls skipped). Per window: row skip = 1 − (cum[hi+1] − cum[lo]) / N′, distinct skip = 1 − (hi − lo + 1) / d;
  all-null window → 1. Baseline per §2.5.
- **`set.rs`** — `scan`: a per-id `u32` count array, running |S_W|, null count in window, and running row mass
  Σ n_v over present ids; O(1) per row entering and leaving. Baseline per §2.5.

**Kernel contract (the streaming hook):** results are `merge(scan(population, segment) for segment in segments)`
plus `baseline(population, w)`. One-shot: one segment, the whole column, population = the column. The later
StreamingRecommender integration supplies the reservoir's contiguous blocks as segments and a population built
from the sampled rows; nothing in the reservoir changes in this spec.

**Parallelism:** columns in parallel (rayon); within a long column, window starts are split into ranges of
max(2¹⁶, 4w) starts (set: also ≥ d, so zeroing its per-id array never outweighs the scan); each range warms its
window over its first w − 1 rows then slides, and ranges merge. Each column is rechunked
once (zero-copy if already one chunk), so input chunk boundaries never affect results.

**Entry points:** `bindings/api/techniques.rs::minmax_index(batch, window_rows: u64)` and
`::set_index(batch, window_rows: u64, max_size: u64)` → `column` + the §3 metrics; registered in
`bindings/python.rs`; wrapped by `_plugin.minmax_index` / `_plugin.set_index`. The C ABI is unchanged. An
unsupported dtype reaching Rust is InvalidInput → ValueError.

**References** (share no code with the kernels; different algorithms):

- `MinMaxIndexPolars`: ids from Polars `rank("dense")` (after the float canonicalisation of §2.4), `rolling_min` /
  `rolling_max` over them (nulls ignored; all-null window → skip 1), joined to cumulative counts.
- `SetIndexPolars`: occurrence intervals — an occurrence of v at row r covers window starts [r − w + 1, r]
  (clipped to the valid starts); intervals are merged per value, then a difference array and cumulative sum give
  |S_W|, the null indicator and the row mass for every window. Vectorised, O(n log n).
- Baselines: Python `math.lgamma` / `math.erfc`.

## 5. Streaming readiness

Both techniques are a pure function of (population, segments, w[, max_size]); `WindowSums` merge by addition and
baselines need only counts. The follow-up spec wires them into both recommenders: one-shot exactly as here;
streaming over the reservoir's blocks (default `block_rows` 65 536 ≥ the default window; a window larger than
`block_rows` is then not evaluable), with the population from sampled rows and d from the running distinct
estimate.

## 6. Benchmark

`tests/performance/benchmark_minmax_index.py` and `benchmark_set_index.py`, ~20 lines each calling
`harness.run(...)`:

| Shape | Purpose |
|---|---|
| `large_dataset.arrow` | realistic mix |
| narrow 10M × 4, clustered (sorted / long runs, `runs_int` / `runs_str`) | index-friendly |
| narrow 10M × 4, random, cardinality 100 000 | worst case for set (overflow) and minmax (wide ranges) |
| wide 1M × 100 | parallelism across columns |

The harness reports algorithmic (Polars ÷ Rust@1), parallel and total speedups.

## 7. Testing

`tests/test_minmax_index.py` and `tests/test_set_index.py`, via `tests/harness.py`:

1. **Contract** (every implementation): schema, canonical order, ineligible rows null, `to_arrow()` native.
2. **Reference agreement** on `large_dataset.arrow` and seeded fixtures: counts exact; observed means and stds
   within `RTOL = 1e-9` (`ATOL = 1e-12`); the `expected_*` baselines within `BASELINE_RTOL = 1e-6` (sums of
   `lgamma` differences, ~10⁸ at 10⁷ rows, from two libms); NaN matches NaN. A seeded fuzz also checks both
   implementations against a brute-force oracle in the test file (every window evaluated directly from the
   §2.3 definitions).
3. **Known answers** (both implementations):
   - `A A A B A A A C`, w = 4 (5 windows; counts A 6, B 1, C 1):
     - set, `max_size` 0: skip_rows mean 0.125, std 0; skip_values mean 1/3, std 0; window_distinct mean 2, max 2;
     - set, `max_size` 1: every window overflows → skips 0, overflow_fraction 1;
     - minmax: skip_rows mean 0.1, std 0.05; skip_values mean 4/15, std 2/15;
   - sorted `1..8`, w = 4, minmax: skip_rows and skip_values mean 0.5, std 0;
   - an all-null window (skip 1); an all-null column (NaN skips, set e_W = 1); n < w (one window);
   - NULL counting toward `max_size` (`A null`, `max_size` 1 → overflow);
   - String / Categorical / Enum ordering (Enum by category order, Categorical by string);
   - `[0.0, -0.0, NaN]` canonicalisation; a multi-chunk input equals the single-chunk result; every eligible dtype
     once.
4. **Statistical correctness** (in place of the conclusions block — there are none):
   - Monte Carlo baseline: under a random order every window is a uniform w-subset of the rows, so the mean skip
     over 20 000 seeded w-subsets estimates `expected_*`. Minmax and unlimited set match within 5 Monte Carlo
     standard errors (+ 1e-9); set with finite `max_size` within 0.02 absolute (normal approximation),
     including `expected_overflow_fraction` and `expected_window_distinct`.
   - Hand-worked baseline: `1 2 3 4`, w = 2 → minmax `expected_skip_values` = `expected_skip_rows` = 1/3
     (6 equally likely pairs, 8 of 24 query–pair combinations skipped); unlimited set 1/2.

Rust unit tests: `WindowSums` merged at every split point of a small column equal the one-segment result; the
std from the integer sums equals a direct two-pass std; `avoid` edges (m = 0 → 1; n − m < w → 0).

## 8. Documentation

- Docstrings on both bases (model, weightings, baseline and its approximation) and each implementation.
- CLAUDE.md §1b: entries for `analytics.minmax_index` and `analytics.set_index`; the project-structure tree, the
  Rust source layout and the private `_plugin` list.
- ToDo.md item 1b: the skip-index bullet marked as delivered as techniques (recommender use pending).
- This spec: an "Outcome" section after benchmarking.

## 9. Out of scope

Recommender integration and reservoir evaluation (follow-up spec); range queries; `bloom_filter` and token /
n-gram indexes; byte-adaptive granules (`index_granularity_bytes`); several granularities per instance; any
decision or conclusion.
