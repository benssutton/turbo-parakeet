# AMS-sampling entropy estimator — design

Status: approved 2026-10-11. Adds a third implementation to `analytics.pairwise_entropy` and
`analytics.threeway_entropy` (spec 2026-09-24 uniform technique interface).

## 1. Goal

Find out whether a sketch can stand in for exact value counts when computing marginal, pairwise
and three-way entropies. The answer feeds the planned primary-key detector (ToDo item 6), which
prunes candidate keys of 1–3 columns by entropy in frames of up to 2,000 columns. Its streaming
version cannot hold every value's count.

This spec covers the estimator and its benchmark against the existing exact Rust kernels and the
Polars references, for both accuracy and speed. The detector itself is out of scope.

Facts that bound what the sketch can do:

- **It prunes. It does not prove.** With n rows, one duplicate pair lowers H by about 2/n bits.
  That is far below any sketch's error, so candidates that survive the entropy pruning still need
  exact verification.
- **Every combination needs its own sketch.** A joint sketch cannot be built from marginal
  sketches, so a streaming search must choose its combinations up front. The per-sketch memory and
  time measured here decide how many combinations that can be.
- **One-shot, AMS is not expected to beat exact counting on speed.** Both read every row once, and
  the exact path counts dense ids in a flat array (83.5 µs per pair at 50K rows). The sketch's
  advantage is memory: its state is O(`num_samples`) and does not depend on the number of distinct
  values.

## 2. Algorithm choice

The project uses the **Chakrabarti–Cormode–McGregor estimator** (CCM: "A near-optimal algorithm for
computing the entropy of a stream", SODA 2007 / TALG 6(3) 2010). It is one-pass and gives a
(1 ± ε) relative-error approximation of the Shannon entropy H.

The alternatives were rejected:

- **Plain AMS sampling (Lall et al. 2006)** gives no relative-error guarantee when H is near 0.
  On a near-constant column almost every sample lands on the dominant value. Passing a relative-error
  test would mean loosening `ATOL`, which quietly turns it into an additive test.
- **Tug-of-war F₂ (AMS 1996)** estimates Rényi-2 entropy, not Shannon entropy, so it could not be
  compared directly with the existing kernels.
- **Hybrid exact-then-sketch** still fails on a column with high cardinality *and* low entropy
  (for example 99% null plus 10K unique values), and adds a second code path.

### 2.1 Estimator

Let m be the stream length, λ(x) = x·log₂(m/x) with λ(0) = 0, and X(r) = λ(r) − λ(r−1). For a
uniformly chosen position J, let R be the number of occurrences of the token a_J at or after J.
Then E[X(R)] = H.

Each of the c = `num_samples` estimators keeps two samples:

- **primary** `(s0, t0, r0)`: s0 is the token of the row with the smallest random label t so far;
  r0 counts occurrences of s0 since that row;
- **backup** `(s1, t1, r1)`: the same for the smallest-label row whose token is not s0.

Update rule for a row with token x and label t, from CCM Algorithm 1:

- If t < t0: when x ≠ s0, the old primary becomes the backup. Then (s0, t0, r0) ← (x, t, 1).
- Else if x = s0: r0 += 1.
- Else if t < t1: (s1, t1, r1) ← (x, t, 1).
- Else if x = s1: r1 += 1.

At the end, s0 is a uniform sample of the stream and r0 ~ R. For any item a, the pair
(s0, r0) if s0 ≠ a, otherwise (s1, r1), is a uniform sample of the stream with every a removed
(CCM Lemma 2.4).

A **Misra–Gries summary with 256 counters** (internal constant `MG_COUNTERS`) gives p̂_max, the
share of the most frequent item a_max, with p_max − (1 − p_max)/k ≤ p̂_max ≤ p_max (CCM Lemma 2.3).
That under-estimate biases H by about 1/k relative: 8 counters would give ~12% on a
95%-one-value column with a 1,000-value tail, while 256 give ~0.4%. Each missed count shifts H by about (H_rest/(1 − p) + 1.44)/m bits, so a tail of singletons, which has a large H_rest/(1 − p), amplifies it (measured: +0.2% on a 95%-one-value stream with a 1,000-value tail, as predicted). The counters cost about 4 KB
and an amortised O(1) hash-map update per row.

Estimate:

- **p̂_max ≤ ½:** H = mean over the c estimators of X(r0).
- **p̂_max > ½:** H = (1 − p̂_max)·mean X(r′) + p̂_max·log₂(1/p̂_max). Here r′ comes from each
  estimator's sample of the stream without a_max. Estimators whose remainder sample is empty (the
  stream is all a_max) contribute 0.

The estimate takes the plain mean over the c estimators, with no median-of-means. The benchmark
reports the error distribution directly.

### 2.2 O(1) work per row

Comparing each row against 2c samples would cost O(c) per row. Two devices remove that:

- **Shared counters.** A hash map holds `cnt[key]`, the number of times a key has occurred since it
  started being tracked, for every key that some estimator holds as s0 or s1. An estimator stores
  `offset = cnt[key]` when it picks a sample, before that row is counted, so r = cnt[key] − offset.
  Each row costs one map
  lookup and one increment, however many estimators share its key. A key is dropped from the map
  when no estimator holds it any more (reference counts).
- **Two label clocks.** Labels are iid uniform in [0, 1), and only two outcomes matter: a label
  below t0 (event A, on any row) and a label in [t0, t1) on a row whose token is not s0 (event B;
  t1 = 1 with no backup). A label in [t0, t1) on an s0 row changes nothing, so it must not cost an
  event: a single clock with threshold t1 fires about every other row on a near-constant column
  whose primary is the dominant token. Each estimator instead runs two independent geometric
  clocks:
  - **A**, counted in rows, success probability t0 (1 with no primary). At its row, t is drawn in
    [0, t0): the token becomes the primary (renewed in place if it is s0; otherwise the old primary
    becomes the backup).
  - **B**, counted in rows whose token is not the primary's key K, success probability
    p′ = (t1 − t0)/(1 − t0). At its row, t is drawn in [t0, t1) and the token becomes the backup.
    An estimator without a backup has p′ = 1 and gets one on the first row of another token.

  Per row, A fires with probability t0 and, failing A, B with p′ on non-s0 rows, so B happens with
  probability t1 − t0, exactly as in Algorithm 1. A is processed before B on the same row, and
  every event redraws both clocks (memoryless: only "not fired yet" has been observed). A events
  sit in a global min-heap by row. B targets sit in a min-heap per tracked key, holding the
  estimators whose primary is that key, in units of "rows not of this key since tracking began"
  (tracked from the key's `start` row and `count`). A global *check* heap holds, per key, the
  earliest row its smallest target could fire if every following row were another token. A check
  that finds nothing due (rows of the key came in between) is rescheduled; each such wasted check
  is paid for by at least one row of that key since its previous check, so there are at most
  m + (number of events) checks in total. Each estimator has O(log m) A events (new minima) and
  O(log m) B events (new minima among non-s0 rows), so per-row work is O(1) amortised whatever
  the skew. Stale heap entries are skipped lazily (each estimator stores its `next_a` row and
  `next_b` target; a B entry is also stale once its key is no longer the estimator's primary).
  Heap ties order by estimator index and key, so a seed always gives the same estimate.

Randomness comes from a seeded generator per sketch (derived from the user `seed`, a domain tag and the column names of the column or
combination), so a combination's estimate depends neither on thread scheduling nor on which other
combinations were requested in the same call.

### 2.3 Memory

The state is about c × 2 samples × 32 B, plus the tracked-key map of at most 2c entries, plus the
A heap and the per-key B heaps (at most c live entries each in total; stale entries are bounded
by the events), plus the check heap (one live entry per tracked key), plus about 4 KB of
Misra–Gries counters: about 0.5 MB at c = 4096, independent of m and of the number of distinct
values.

## 3. Rust

### 3.1 Estimator module

`src/techniques/single_column/unordered/ams_entropy.rs`, stand-alone like `hll.rs`, importing only
`common`:

```rust
pub(crate) struct AmsEntropy { /* estimators, heap, tracked-key map, Misra–Gries, m */ }
impl AmsEntropy {
    pub(crate) fn new(num_samples: usize, seed: u64) -> Self;
    pub(crate) fn update(&mut self, key: u64);
    pub(crate) fn estimate(&self) -> f64; // bits; 0.0 when m = 0
}
pub(crate) fn ams_marginal_entropy_impl(inputs: &[Series], num_samples: usize, seed: u64)
    -> PolarsResult<Series>; // col_name, entropy
```

The marginal kernel keys each row by `encode_series` (via `build_column_cache_par`). Null is its
own category, as in `marginal_entropy.rs`: the key mixes in `is_null` so that a null cannot
collide with a real value. Columns run in parallel with rayon.

### 3.2 Joint kernels

`src/techniques/pairwise/ordered/ams_joint_entropy.rs` provides
`ams_pairwise_joint_entropy_impl` and `ams_threeway_joint_entropy_impl`. They take the same
arguments as their exact counterparts in `joint_entropy.rs` plus `num_samples` and `seed`, and
return the same output schemas. Each row's key combines the columns' null-folded `encode_series`
keys into one u64 with a mixing hash (foldhash). These keys are value-stable across batches, which
the streaming detector will need; dense ids are not. Combinations run in parallel with rayon, one
`AmsEntropy` each. A sketch cannot be split across row ranges, because its samples depend on
stream order.

### 3.3 API and binding

In `src/bindings/api/techniques.rs`:

- `ams_marginal_entropy(batch, num_samples, seed)` returns `col_name, entropy`;
- `ams_pairwise_joint_entropy(batch, pairs, num_samples, seed)` returns `col_a, col_b, entropy`;
- `ams_threeway_joint_entropy(batch, triplets, num_samples, seed)` returns
  `col_a, col_b, col_c, entropy`.

`num_samples = 0` → `InvalidInput`. Name checks are as in the exact entry points. They are exposed
in `bindings/python.rs` and wrapped in `analytics/_plugin.py`, and are private like the other
kernels. There is no C ABI or Java binding.

## 4. Python

- `analytics/pairwise_entropy/ams.py`: `PairwiseEntropyAmsRust(PairwiseEntropy)`;
- `analytics/threeway_entropy/ams.py`: `ThreewayEntropyAmsRust(ThreewayEntropy)`.

Each mirrors its `*Rust` sibling with the `ams_*` calls swapped in, and has the constructor
`__init__(self, *, num_samples=4096, seed=0, **thresholds)`, where `**thresholds` are the base's
keywords. In the pairwise class each column's marginal is sketched once per call and shared by its
pairs. Both
classes are added to their package's `IMPLEMENTATIONS`; the Polars reference is unchanged. The
`*Rust` suffix is deliberate: the benchmark harness re-times classes whose names end in `Rust` at
1 thread, and CLAUDE.md lets only `*Rust` classes call `_plugin`.

**Agreement.** Both classes set `RTOL = 0.05` and `ATOL = 1e-9`. The three-way technique reports
only `h_abc` and `n_rows`, so `ThreewayEntropyAmsRust` uses the inherited `agreement` unchanged and
calls only `ams_threeway_joint_entropy`. `PairwiseEntropyAmsRust` overrides `agreement`:

- entropies (`h_a`, `h_b`, `h_ab`) within RTOL of the reference;
- `mi` within |ΔMI| ≤ RTOL·(h_a + h_b + h_ab), computed from the reference's entropies;
- `nmi` within that bound ÷ min(h_a, h_b) (skipped where the reference's NMI is NaN);
- `n_rows` exact;
- conclusions (`redundant`, `near_unique`) not compared: an estimate within tolerance can still
  land on the other side of a threshold.

MI is a difference of nearly equal estimates, so its relative error is unbounded. That matters
for the detector's "low MI" criterion, and the benchmark reports it explicitly (section 5.3).

## 5. Tests and benchmarks

### 5.1 Rust unit tests (`ams_entropy.rs`)

- All-unique stream → exactly log₂ m (every r = 1, so every X = log₂ m).
- Constant stream → exactly 0.
- Over 100 seeds at the default c = 4096, the mean estimate is within 1% of the exact H on a
  Zipf(1.1) stream of 10K tokens, and on a 95%-one-value stream (the backup-sample path).
- The two-clock estimator and a naive reference implementation (test-only; CCM Algorithm 1 with
  a fresh label per row per estimator) give means within 4 combined standard errors of each other
  over 400 seeds at c = 64, on both streams. The naive version is O(c) per row, so it needs a
  statistical rather than a 1% test at that size.
- The same seed gives the same estimate.
- `num_samples = 0` → error at the API.

### 5.2 Python accuracy (`tests/test_pairwise_entropy.py`, `tests/test_threeway_entropy.py`)

The contract and agreement blocks include the new classes through `IMPLEMENTATIONS`. New known
answers, checked for every implementation:

- an all-unique column → h = log₂ n;
- a constant column → h = 0;
- a two-column composite key (`hi = i // k`, `lo = i % k`) → h_ab = log₂ n.

### 5.3 Speed, memory and error

**Dataset.** A new `datagen.near_unique(n_rows=1_000_000, seed=42)` builds eight columns:

| column | content |
|---|---|
| `u0`, `u1`, `u2` | near-unique: a permutation with ~0.1% of values duplicated |
| `key_hi`, `key_lo` | a genuine composite key: `i // 1000`, `i % 1000`, rows shuffled together |
| `skewed` | 95% one value, the rest uniform over 50K values |
| `mid` | ~1,000 distinct values |
| `low` | ~10 distinct values |
| `flag` | near-constant: 1 in every 10,000 rows, else 0 |

That gives 36 pairs and 84 triplets and one real two-column key. Most joint spaces exceed 2²⁰
(the exact kernels' hash-map path); `mid × low`, `key_hi × low` and the like stay in the flat array.

**Speed.** `benchmark_pairwise_entropy.py` and `benchmark_threeway_entropy.py` add
`Dataset("near-unique 1M x 8", lambda: {"t": near_unique()})` to their existing three shapes. The
harness reports median and min times, the 1-thread re-run and the algorithmic / parallel / total
speedups for both Rust classes, and the `agrees` flag.

**Memory and error.** A new `tests/performance/benchmark_ams_entropy.py` runs, for each dataset
and each implementation (exact Rust, AMS, Polars) of both packages, one fresh child process. The
child loads the data, records its working set, runs `result()` once and records its peak working
set (`GetProcessMemoryInfo` → `PeakWorkingSetSize` via ctypes on Windows, `resource.ru_maxrss`
elsewhere; no new dependency). The child cannot reset the peak, so the post-load working set is
the baseline. For AMS it also compares the result
with the reference. It writes one row per (package, dataset, implementation) to
`results/ams_entropy.parquet`:

- `peak_mb`: the peak working set minus the post-load working set;
- `max_rel_err` and `p95_rel_err` over all entropies;
- `max_abs_mi_err` (pairwise only; null for three-way).

In this one-shot setting both Rust kernels hold the encoded column cache (about 9 B per row per
column), so `peak_mb` differences show only the counting state. AMS's real memory advantage, not
keeping rows at all, applies to streaming and is not measured here.

## 6. Out of scope

- The primary-key detector (one-shot and streaming).
- A streaming class built on `AmsEntropy` (the struct is ready for it: `update` per key).
- A C ABI or Java binding for the AMS kernels.
- Sweeping `num_samples`; it is a parameter, and a sweep can be run ad hoc with
  `benchmark_ams_entropy.py`.

CLAUDE.md gains the new implementation names under Joint entropy, and the module in the project
structure.
