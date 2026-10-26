# ToDo

The objectives these items serve are in [CLAUDE.md, Project Objectives](CLAUDE.md#project-objectives). Items are numbered
for reference only; CLAUDE.md cites them by number.

**Suggested order:** the five tracks in order — synthetic data (1), the evaluation loop (2–3), adversarial datasets (4),
engines (5–7), real data (8) — with the analytics items (9–24) taken up as they unblock or as the loop surfaces them.
Track 1 is complete (item 1 lists its follow-ups). Dependencies: 2 before 4; 12 and 20 before 21; 13 lands piecemeal with 19–21; 14 under 24.

---

## Track 1 — synthetic data (`services/datagen`) — complete

1. **Synthetic data — complete.** `synthdata` (`services/datagen`): the column pipeline (targets, measured manifest,
   Arrow IPC cache), the frame layer (`ColumnSpec`, relations, row order, null patterns, type mapping, strict mode,
   frame cache), the pairwise covering design (`cover_small()` 2 000 rows / `cover_windows()` 73 728 rows, 60 covering +
   14 relation + 3 nested columns, all 233 allowed factor-level pairs) and the named scenarios. Every dataset the tests,
   the benchmarks and the bench use comes from it; `tests/datagen.py` is retired. Follow-ups — none blocks Track 2;
   take them up when the loop touches the code, or as one cleanup PR:
   - **Validation (`synthdata/column/spec.py`, `multiset.py`).**
     - The dispersion solve's sigma bracket `[0.01, 3.0]` (`solve_dispersion(lo, hi)`) is absolute, so on a wide
       integer range any realistic `index_of_dispersion` target clamps (correctly reported, but unreachable). Scale the
       bracket with the value range, or document that dispersion targets suit narrow ranges.
     - `array_null_count` validation and the categories-versus-rows rule ignore `null_count`: when nulls displace value
       rows a category can end with no non-null occurrence. Decide whether achieved metrics describe the sequence
       (today: before nulls) or the visible column, then make validation and the manifest agree.
     - The full-range uncapped `int64` draw likely overflows as `uint64` did before its fix (a float 2⁶³ cast to int64,
       tripping `wrap`'s assertion). Reproduce, then reject or handle it in `validate()` as the `uint64` case is.
   - **Manifest wording (`synthdata/column/solve.py`, `generate.py`).** `runs_z` shares the bisection status vocabulary
     although it never takes `invalid_bracket` / `not_converged`; `NOT_CONVERGED`'s message claims a non-monotone
     objective it has not established (a monotone step function that cannot land inside tolerance gives the same
     status) — reword the message; consider a runs-specific status set.
   - **Correctness and dead code (`multiset.py`, `wrap.py`, `sequence.py`).**
     - The uniform lattice draw on the uncapped path draws a continuous uniform and rounds, giving the two endpoints
       half weight: draw integers directly.
     - `wrap`'s spill fallback is unreachable under the inner-count rule: replace it with an assertion. Its decimal
       bound is computed as a float and loses precision for wide integer parts (pyarrow's cast still catches true
       overflow): compute it with integers / `Decimal`.
     - When `runs_z` is dropped for a single-group dichotomy, the inter-arrival CV tolerance is recorded as `None`,
       indistinguishable from the combined-target case; record why it was not applied.
   - **Speed (`multiset.py`, `solve.py`).**
     - The dispersion solve's guard and objective share a multiset build only at the bracket ends (≈ 2 of ≈ 105 builds
       saved; ≈ 1.5–2.5 min at 10M rows). A real speed-up needs `solve_monotone` to evaluate guard and objective at the
       same points, or a cheaper objective. Do not add an early exit to `solve_monotone`: it defeats `NOT_CONVERGED`
       detection (tried and rejected).
     - `powerlaw_counts` is recomputed after `solve_entropy` returns (immaterial at k = 1 000).
     - Re-time the inter-arrival-CV figures on a quiet machine (earlier timings were taken under fivefold load swings).
   - **Frame layer and covering (`synthdata/frame/`, `synthdata/cover/`).**
     - `cover(rows)` below ≈ 1 000 rows is seed-fragile in strict mode: runs-order |rho| noise reaches 0.35–0.43 at
       500 rows (2 of 10 seeds fail; `CLUSTERED_RHO` / `UNORDERED_RHO` in `cover/classify.py` cannot separate runs from
       clustered binary / heavy columns there). A better order statistic — e.g. the longest-run share or rho relative to
       the sorted arrangement, a new key in `frame/measure.column_stats` — fixes it; bump `FRAME_VERSION`
       (`frame/spec.py`) so cached frames regenerate.
     - Relations on an all-null parent (`ForeignKey`, `Subset` in `frame/relations.derive`) fail in `rng.choice(0, …)`;
       `frame/spec._validate_relation` does not catch it — reject an all-null parent there.
     - `check()` (`frame/generate.py`) splits a pipeline warning at the first ": ", so a column name containing ": "
       would be misparsed; strict mode also does not compare the measured `entropy_norm` with the spec's `entropy`
       (only the column pipeline's own entropy warning) — compare it, with the pipeline's tolerance.
     - A cached `cover_small()` costs ≈ 0.6 s per process against ≈ 0.8 s cold (IPC read, strict coverage check, nested
       columns): cache the checked frame with its nested columns, or skip the coverage re-check for a verified cache.
   - **Tests and timing.**
     - `test-py` is 80–85 s against 162 s before (52%; the goal was ≤ 50%). The one test above 10 s is
       `tests/test_benchmark_harness.py`'s fixture (≈ 15 s; it spawns real benchmark child processes — kept by design).
     - `test_benchmark_harness.py::test_a_failing_child_is_an_error_row` failed once under 4 xdist workers (not
       reproduced in 8 later local runs nor in CI): if it recurs, capture the row statuses it saw.
     - `services/datagen/tests/scenarios/frozen.json` freezes each scenario's values by a SHA-256 of `repr(df.rows())`
       (portable; Polars' `hash_rows` differs between Windows and Linux): a Polars upgrade that changes a value's Python
       representation would need it regenerated (`test_scenarios.py` documents how it was built).

## Track 2 — evaluation loop

2. **Loop tool.** One tool for steps b–d of the objectives: generate → recommend → benchmark the recommendation against
   alternatives on the same data → score. Scores: prediction error (predicted vs measured sizes, skip shares), regret
   (the measured best option vs the recommended one), and the analytics' own cost (time, memory). It absorbs
   `tests/performance/calibrate_clickhouse_sizes.py` and `calibrate_skip_indexes.py`. One results store to be chosen
   (the bench writes SQLite `results.db`; the analytics benchmarks write Arrow IPC). ClickHouse first.
3. **Bench measurement quality.**
   - Noise: single p50s of identical tables disagree by up to 2× (UInt32 Delta, filter 0.05: 58.5–168.0 ms over three
     identical tables); ten samples on a shared laptop cannot resolve differences under 2× — repeat runs on a quiet
     machine or `--samples-add`, and a noise bound in the report.
   - Three axes measured nothing in the example suites: `cache_mode` (a 25 MB column is served from the Docker VM's
     page cache in both modes — needs the page cache dropped inside the VM or a column larger than memory),
     `query_condition_cache` (the `IN` predicate is true in every granule, so nothing is cached) and `skip_index`
     (`in_predicate` picks frequent values present in every granule — needs rare predicate values or sorted data).
   - Decimal32 in `suites/float_codecs.yaml` (owner decision): one row is clipped to the bound 10.0 and `Decimal32(8)`
     tops out at 9.99999999, so every Decimal32 cell skips. Either a bound of 9.99999999 or 7 decimal places; then
     re-measure the Decimal32 vs Decimal64 compression asymmetry (ratio 0.65, not 0.5).
   - `LowCardinality(String)` with a `minmax` or `set(100)` index inserts ≈ 15× slower (45.7 / 46.5 s against 3.2 s
     with `bloom_filter`); not investigated.
   - A dataset whose solve reports `invalid_bracket` still runs (the warning is printed); decide whether `run` should
     refuse it, like an up-front validation error.
   - Add the closed form for the −79 runs z-score to FINDINGS §3 (`services/bench/spikes/FINDINGS.md`).

## Track 3 — adversarial datasets

4. **Defeat the techniques.** A catalogue of failure modes per technique and per decision, each with a datagen scenario
   run through the loop (2) and tracked over time. Starting points: heavy-tailed, heavily under-sampled populations
   against Chao1 / Schnabel; near-tied heavy hitters and flat tails against top-k; orders that fool the minmax / set
   window statistics; data sitting at the codec-pruning boundaries (GCD byte-width rule, Delta / DoubleDelta mean
   significant bytes, the near-incompressible ZSTD shortcut); containment pairs with low Jaccard against MinHash.

## Track 4 — engines

5. **Polars and DataFusion in the empirical bench.** An engine axis in the suites; query time and size for the
   in-process recommendations (Arrow / Polars types, dictionary encoding, IPC compression), over in-memory data and
   Arrow IPC. Point-equality and full-scan queries first, as for ClickHouse.
6. **Engine-upgrade watch.** Pin the ClickHouse image; after each ClickHouse, Polars or DataFusion release, re-run the
   calibrations and spikes (`tests/performance/spike_*.py`, `services/bench/spikes/`). Run-end encoding is not
   recommended because DataFusion 54 and 55 reject MIN / MAX over REE columns (every other operator probed works,
   including IPC reads): re-run `tests/performance/spike_datafusion_ree.py` after a DataFusion upgrade, and add REE
   (sizes and a recommendation) once MIN / MAX work.
7. **ClickHouse in CI** (optional): a service container so the `clickhouse`-marked bench tests run in CI, weighed
   against the Actions minutes (private repo; see the cost-control notes in `.github/workflows/ci-cd.yml`).

## Track 5 — real data

8. **Real-data mode.** Sample a real dataset (row blocks, as the streaming recommender's reservoir does), recommend,
   benchmark the recommendation on the sample, report. The bench already reads a real Arrow IPC column
   (`dataset: {path, column}`).

---

## Analytics — accuracy follow-ups

9. **ClickHouse size-emulator gaps** (`services/analytics/src/common/clickhouse_sizes.rs`).
   - String layout: LZ4 over-predicts (median +19%), small ZSTD(1) columns under-predict (up to 78%); calibrate.
   - Sparse serialization: ClickHouse stores a part with ≥ 93.75% default values sparsely
     (`ratio_of_defaults_for_sparse_serialization`); not modelled, so near-constant default columns are over-predicted
     (925 vs 46 bytes; on the server 50 000 zeros and 5% non-zeros are `Sparse`, 50% is `Default`).
   - Date32 / DateTime64 range limits (1900–2299) are not checked by the type mapping.
   - Block cutting: a granule over 1 MiB closes the open block early (ClickHouse fills it first); matters only for rows
     averaging > 128 bytes.
10. **Skip indexes.**
    - Calibration is thin: one column per kind at G ≥ 2, only bloom fpr 0.025 chosen, point queries on Date / Decimal
      literals not parsed by the calibration script.
    - Streaming: inject the running counts into the population and baselines instead of the sample's.
    - Cost: one-shot runs ≈ 4 granularities × 3 window scans plus ≈ 12 index-size emulations per column (bloom adds 4);
      add a line to `tests/performance/benchmark_recommend.py` before large frames.
    - Range queries (`col BETWEEN a AND b`), where minmax indexes earn most of their value — **on hold**: needs a design
      for how point and range queries combine in the decision, and whether ClickHouse's set index skips ranges.
11. **Recommender gaps, parity and Polars state.**
    - Entropy of list inner values: the `entropy*` columns cover the top-level values only.
    - Exemplars with `id_columns`: still to decide how identifier columns that are themselves null or ineligible
      (Int128 / UInt128, Object) are reported, and what a List column with nulls inside counts as.
    - String columns whose null slots hold bytes (built with `pl.when(...).then(None)` or `Series.set`): one-shot
      measures those bytes in its cast candidates (`datafusion_size_bytes`), streaming does not; `assert_parity`
      excuses the gap only for the original type. Decide which is right (an Arrow cast keeps the bytes; a writer may
      not) and align them.
    - Polars' process-global categorical mapping: a user's Categorical columns share it, so other Categorical data alive
      in the same process can inflate a column's measured dictionary size.
    - Block-sampled ZSTD estimates (streaming, a reservoir of 4 096-row blocks) miss ordered (sorted / clustered)
      columns by up to ±18% at 16 384 of 73 728 rows; random-order columns stay within 10%
      (`test_sampled_zstd_estimate_on_ordered_columns`, xfail).
    - Streaming's original `size_bytes` for a Categorical / dictionary column beside other dictionary columns counts
      every category of the shared mapping (`recommenders/streaming/sizing.rs` `cats_of`), e.g. 1 429 368 vs
      one-shot's 304 168 bytes; reproduces with plain pyarrow dictionary columns in one batch. The parity test
      excuses it per column (`shares_categories`).
    - Parity across a streaming block boundary is untested: at 73 728 rows with `block_rows` 65 536, LZ4 / IPC sizes
      differ by up to 2.6% on dictionary columns (ZSTD by design); the parity test streams with `block_rows` = the
      frame's height. Decide the multi-block bound and add a test.
12. **Bloom speed and MinHash containment recall** (prerequisite of 21).
    - `BloomRust` loses to `MembershipExact` (plain `frozenset` intersection) on every benchmarked shape (0.16× total on
      the wide 100-column shape): one binding call per column filter. Batch filter construction across columns in one
      Rust call, or use Bloom only above a distinct-count threshold and make `MembershipExact` the default.
    - MinHash recall on the former 50K × 101 benchmark frame was 0.65 (below `MIN_RECALL` 0.85): every miss is a
      containment pair (overlap 1.0) whose Jaccard is below the ≈ 0.43 LSH candidate threshold (min(j·0.9, o·0.45)) —
      exactly the FK → PK case. `tests/test_similarity.py` documents it as a strict xfail on
      `synthdata.scenarios.containment_pairs`; derive the candidate threshold from the overlap threshold and the
      cardinality ratio.
    - `minhash.rs` `bucket_pairs` is O(|bucket|²) per band: document it as a scaling concern for dense buckets.
    - Tiny Bloom filters overshoot `fp_rate` (0.01): at n = 1 distinct value the geometry is (m = 16, k = 12) and ≈ 25% of
      disjoint queries are false positives (BloomRust 0.247, Fastbloom 0.252; theory ≈ 0.06%); still 2–5× `fp_rate` up to
      n ≈ 10 (BloomRust n = 2 / 5 / 10: 0.045 / 0.049 / 0.031), close to 1% from n ≈ 30 (0.014). Suspected: degenerate
      double-hash probes in a few-byte filter. Fix with a minimum filter size / better probe derivation and a per-filter
      FP test at small n (found by a constant column of the covering frame; the Fastbloom agreement case is xfailed).

## Analytics — structure

13. **Techniques report metrics only.** Each conclusion (with its threshold keyword and its conclusions test block) is
    removed when its new home lands: membership `relationship` / `unique_a/b` → the join detector (21); `near_unique` /
    `redundant` → the key detector (20) and the ordering recommender (19); Describe's `class` → the recommenders;
    `gcd_compressible`, `dominated`, `passes_jaccard/overlap`, `associated`, `same_partition` → whichever recommender
    adopts each. Describe's `min` / `max` / `unique` / `est_*` are metrics and stay. `categorical_threshold` stays one
    keyword, although it then both sets the recommenders' class boundary and sizes the streaming distinct sample
    (k = max(threshold, 1000)).
14. **Native Rust / Arrow kernels instead of Polars.** Polars should be confined to measuring sizes in Polars' layout.
    Every technique file imports Polars today (~25k lines of Rust): spike first (port one kernel, e.g. GCD or the runs
    test, and benchmark it against the Polars version), then convert one kernel at a time under 24. A second benefit:
    build time (cold build ≈ 20 min).
15. **Explicit combination lists.** Done: `result(combinations=…)` and `techniques._select.stratified_combinations`; the
    benchmarks compute 50 stratified combinations per pair / triplet dataset (full run 8 115 → 2 356 s). Follow-ups:
    - The harness picks a dataset's combinations once, with the run-level `params`; a `Variant.params` or `impl_params`
      that changed `eligible` / `compatible` would leave that implementation with ineligible rows. Pick per parameter
      group if a benchmark ever varies eligibility.
    - `n_combinations` counts result rows: with a cap it holds eligible combinations only, uncapped (`None`, or per-column
      techniques) it includes the ineligible ones. Report eligible and total separately if the difference matters.
    - `stratified_combinations` runs `n_unique` on every candidate column in the parent (untimed, but slow on very wide
      or long datasets): sample or reuse Describe's counts if it becomes noticeable.
16. **Uniform interfaces** where parameters have the same meaning, and consistent optional parameters.
17. **Consolidate counting.** Frequency counts are taken separately in Describe, entropy and elsewhere; review and share.
18. **DataFusion in the technique benchmarks.** Deterministic techniques: Rust vs Python Polars vs Python DataFusion.
    Probabilistic techniques: Rust probabilistic vs Rust deterministic, and Rust probabilistic vs Python probabilistic vs
    Python deterministic.

## Analytics — new recommenders and detectors

19. **Ordering recommender** (`OneShotOrderingRecommender`, `StreamingOrderingRecommender`). An optional list of columns
    to optimise (default: the whole frame); at most ~10 sort columns returned. The joint entropies (exact and
    estimated) propose candidate orders by mutual partitioning; each candidate is scored on a sorted sample for
    (a) compressed size (runs, ZSTD) and (b) predicted skip share for point queries on the sort columns and the others —
    both reported, no single weighted score. Re-sorting changes every order-dependent metric, so the runs test, minmax,
    set and delta are re-evaluated on the candidate order. Consider NVI clustering (`notebooks/NVI Clustering.ipynb`)
    and a visual output.
20. **Key detector** (`OneShotKeyDetector`, `StreamingKeyDetector`). Groups of up to three columns that are unique, or
    near-unique, in a frame.
    - A group is a key when its joint entropy is within a margin of log₂ N. With a duplicate share f (default 1%), the
      maximum entropy is log₂ N − 2f (fN values occur twice, the rest once), so the margin is 2f — 0.02 bits at 1%,
      whatever N (today's `near_unique` margin of 0.1 bits tolerates 5% duplicated rows).
    - Both kinds are reported: exactly unique as sampled (gap 0) and within the margin; the cause of a gap is not
      identified. Only minimal keys; each key's mutual information / total correlation is reported and ranks the keys,
      with no MI threshold.
    - Pruning: a pair or triplet can be a key only if the sum of its marginal entropies is ≥ log₂ N; supersets of keys
      are skipped.
    - Streaming: the entropy estimate's error must be below the margin. AMS cannot (≈ 1 bit at 1M rows against
      0.02 bits); an HLL per candidate gives D̂ and log₂(N / D̂) bounds the gap from below, which prunes, but at p = 14
      (±0.035 bits at 3σ) cannot confirm a key within 0.02 bits. Confirmation is an open design point (higher precision,
      or an exact check of the few survivors).
    - Takes over the `near_unique` conclusions (13).
21. **Join detector** (`OneShotJoinDetector`, `StreamingJoinDetector`). Two or more frames; primary–primary and
    primary–foreign key pairs from the membership technique, heavily pruned by dtype. Needs 12 (Bloom speed, MinHash
    containment recall) and 20 (composite keys). Takes over membership's `relationship` / `unique_a/b` (13).
22. **Join optimisation research.** How ClickHouse, DataFusion and Polars execute joins, and which storage choices
    (sparse indexes, dense indexes, sort order) speed them up.
23. **Composable recommenders that return the data** (aspirational). (a) Return the data with the recommendations
    applied — the one-shot recommender already casts every candidate (`recommenders/engine/cast.rs`), so this is mostly
    exposing it; (b) compose recommenders, e.g. find that a coarse-typed frame and an all-string frame join on an
    integer column once both are optimised.

## Performance

24. **Rust performance review.** With the techniques and recommenders in place: consolidate hash maps and patterns
    where the gain is marginal, and apply the known performance techniques wherever they fit. Includes the native-Arrow
    kernel conversion (14).
