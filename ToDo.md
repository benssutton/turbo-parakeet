# ToDo - A List of future enhancements to this project

The objectives these serve (target engines, formats, row order, queries, validation, deliverables) are in
[CLAUDE.md § Project Objectives](CLAUDE.md#project-objectives).

ToDos are broken into thematic sections in a suggested priority order, but items within each section are in no priority order.  The order in which these are implemented should favour efficiency and code quality.  Item numbers are stable: CLAUDE.md and the specs refer to them.

**Agreed starting order:** 1a–f done, 1g on hold; 5c done; next 3 (1.0 skeleton, 4 and 2 done; 6 + 5g done).

---
The following ToDos 1 to 7 are largely intended as a clean up of existing code and implementation of encoding and skipping index recommendations.

1. Extend the Recommenders to extend encoding recommendations and provide skipping index recommendations.  Currently the only encoding recommendation made is dictionary encoding.  We should now have the statistics and techniques in this project to extend this to the following:
1.0. **Skeleton done** (spec 2026-10-18-recommender-schema-entropy-exemplars-design.md): the layout, the renames and the reserved columns. The ClickHouse type mapping and the chbench handoff (`to_chbench`) are **done** (spec 2026-10-19); the skip-index decision thresholds and cost model stay with 1c–e, REE size measurement with 1a.  Original text: **one output-schema spec covering ToDos 1, 2, 3 and 4**, so the schema (recommenders/schema.rs, both recommenders, the Java binding, the parity tests) changes once rather than four times.  It is written before ToDo 4, which is implemented as its first increment (with 2).  It defines:
- one block per engine (DataFusion, Polars, ClickHouse);
- the Arrow → ClickHouse type mapping (LowCardinality, Nullable, Enum8/16, Date32, DateTime64, Decimal …) and the Arrow → Polars mapping;
- the handoff to clickhouse_benchmark: the Arrow result written as IPC, plus a helper that renders it as `CREATE TABLE`;
- the decision thresholds and a cost model (index size versus bytes skipped);
- the size extensions predicted = measured needs: common/ipc_sizes.rs (and the streaming analytic sizes, recommenders/streaming/sizing.rs) measure plain and ZSTD layouts only — REE (1a) and LZ4_FRAME (1f) are missing.
a. **Closed — not recommended** (spec 2026-10-21): DataFusion 54 rejects MIN / MAX over run-end-encoded columns (every other operator probed works, incl. IPC reads); REE would break common queries. Re-run `tests/performance/spike_datafusion_ree.py` after a DataFusion upgrade and add REE once MIN / MAX work.
b. **Done** (spec 2026-10-19): GCD codec by byte-width pruning, emulated and calibrated.  GCD encoding (ClickHouse) using the GCD metric applied to the 'shrunk' dtype where applicable.  Use the GCD left over after the type rule (decimal rescale, datetime unit) has used it.  ClickHouse applies the codec per compression block (64 KiB–1 MiB), so the whole-column GCD is a lower bound.  Compression blocks are sized in bytes, not rows (64 KiB = 8 192 Int64 values), so a per-block figure needs sub-blocks sized per dtype, not the reservoir's 65 536-row blocks.
c. **Done** (spec 2026-10-20): `bloom_filter(fpr)` fallback, sized by distinct values per block, chosen only if skip × column bytes > index bytes.  Bloom Filter skipping index (ClickHouse).  Top K is the wrong statistic: under random order a value with share p is absent from a granule of w = 8 192 rows with probability (1 − p)^w (≈ 0.03% at p = 0.1%), so the top frequency only shows where a Bloom filter is useless.  The share of granules not containing v is what `set_index` reports with `max_size=0` (no limit); a Bloom filter skips that share × (1 − fpr).  So: `set_index` at `max_size=0`, the false-positive rate, and a filter-size model.  No new technique is needed.
d. **Done** (spec 2026-10-20): `minmax` candidate in both recommenders.  MinMax skipping index (ClickHouse) using the minmax sliding window technique (minmax.rs) which gives both an indication of skipping potential based on random order (i.e. just on frequency counts) and based on the sampled order.
e. **Done** (spec 2026-10-20): `set(N)` candidates, pruned by `max_overflow`.  Set skipping index (ClickHouse) using the set sliding window technique (set.rs) which gives both an indication of skipping potential based on random order (i.e. just on frequency counts) and based on the sampled order.
For d and e in the StreamingRecommender (skip-index spec 2026-10-16 §5; **done** in spec 2026-10-20 §7 over the sampled population; running-count injection remains a refinement, see ToDo 7):
- an entry point that takes several segments (each segment's window starts clipped to `seg.start..seg.end − w + 1`);
- row and distinct counts injected from the running statistics instead of `Population::counts`;
- `Population::from_ids` requires the counts to describe the same rows as the ids;
- a window larger than `block_rows` cannot be evaluated, which caps any GRANULARITY sweep at 65 536 rows by default.
f. **Done** (spec 2026-10-19): `*_ipc_compression` / level / size columns and `clickhouse_codecs`.  General compression choice: LZ4 or ZSTD plus level per column, for both the ClickHouse codec chain tail and Arrow IPC buffer compression (DataFusion / Polars).
g. **On hold** (2026-10-20: needs more design thought — how point and range queries combine in the skip-index decision, and whether ClickHouse's set index skips ranges).  Range queries (`col BETWEEN a AND b`) in the MinMax technique, where MinMax indexes earn most of their value.  Point queries are the current assumption.

2. **Done** (spec 2026-10-18): `exemplar_*` / `null_exemplar_*` columns in both recommenders, first-seen rows, optional `id_columns`.  Extend the Recommenders to provide a) Exemplars and b) Null Exemplars.  An optional parameter comprising of a list of column names is passed to the Recommenders which serve as the identifier for the Exemplars result.  For each column, the Recommenders identify 1 row where the column value is Not Null, and 1 row where the column value is Null, where applicable, and return the identifier for those rows as well as the value in the Exemplar case.
- The exemplar is the **first-seen** non-null row and the null exemplar the first-seen null row, in frame / stream order: deterministic, identical in one-shot and streaming, O(1) memory (streaming already tracks `first_row`).
- Values are rendered as text the same way as min / max (`common::text::render_value`).
- Still to decide: identifier columns that are themselves null or ineligible (Int128 / UInt128, Object); List columns with nulls inside.

3. Add techniques to identify candidates for delta and double-delta encoding, and if ToDo #1 above is already implemented, add to the Recommenders.  Calculating the delta and double-delta between consecutive rows and their value range and entropies should indicate candidates.  These are ClickHouse codec recommendations only (Arrow IPC has no delta encoding).

4. **Done** (spec 2026-10-18): `entropy*` columns, exact in one-shot and in streaming's exact phase, `sample` / `ams` (default) estimates past it; entropy of list inner values is not done.  Extend the Recommenders to return the entropy of each column.
- OneShotRecommender: compute it from Describe's existing frequency map (techniques/single_column/unordered/describe/frequencies.rs), one pass over the distinct values with nulls as one category, rather than re-hashing the column with marginal_entropy.rs.
- StreamingRecommender: exact while the distinct sample is exact (it holds every value's count), then an estimate once it overflows — the same rule as `est_cardinality`.  Two candidates, to benchmark against each other:
  - the AMS sketch (ams_entropy.rs).  AMS cannot take over at the overflow: it must see the stream from its first row, so it is fed from the first batch for every eligible column and discarded if the sample stays exact (cost: per-row work and ≈ 4 096 samples per column);
  - the bottom-k distinct sample itself.  After overflow it still holds exact counts for a uniform random sample of k distinct values (distinct_sample.rs), so Σ c·log c ≈ (D̂ / k)·Σ over the sample — no extra per-row cost.  Weak on skewed columns whose heavy values fall outside the sample.

5. Address inconsistencies in the Rust module structure:
a. Techniques return metrics only; decisions sit in the Recommenders and detectors.  Each conclusion (and its threshold keyword and conclusions test block) is removed when its new home lands: membership `relationship` / `unique_a/b` → JoinDetector (10); `near_unique` / `redundant` → KeyDetector (9) and OrderingRecommender (8); Describe's `class` → the recommenders; `gcd_compressible`, `dominated`, `passes_jaccard/overlap`, `associated`, `same_partition` → whichever recommender adopts each.  Describe's `min` / `max` / `unique` / `est_*` are metrics and stay.  `categorical_threshold` stays one keyword, although it then both sets the recommenders' class boundary and sizes the streaming distinct sample (k = max(threshold, 1000)).
b. Techniques should use native Rust and Arrow functions instead of Polars.  Polars should ideally be confined to measuring the size of Arrow columns in Polars' layout.  Every technique file imports Polars today (~25k lines of Rust), so: spike first (port one kernel, e.g. GCD or the runs test, and benchmark it against the Polars version), then convert one kernel at a time under ToDo 13.  A second benefit: build time (cold build ≈ 20 min).
c. **Done** (spec 2026-10-22): Python-only rename; the Rust crate, C library and Java binding keep `analytics`.  Rename the Python package to `turbo_parakeet`, with maturin's mixed layout: `services/analytics/python/turbo_parakeet/{techniques,recommenders,detectors}/`, compiled module `turbo_parakeet._core`.  This removes the clash with Segment's `analytics` package.  It is a public import path, so it touches every test, CI, sonar config and CLAUDE.md; it is cheapest before more packages are added, so do it just before the first new Python package (ToDo 3 or 8).
d. Today no technique accepts a column / pair / triplet list: `add()` takes frames only, and `Technique.enumerate` always runs every compatible combination (for the bloomfilter, pairs within one value family).  Add the optional explicit list; ToDos 8 and 9 need it (101 columns = 166 650 triplets).
e. Interfaces should be uniform where parameters have the same meaning, and should be consistent in optional parameters, again where applicable.
f. An audit of benchmarks should be carried out to ensure speed, accuracy and test sets are uniform.
g. **Done** (spec 2026-10-17-generated-dataset-design.md): benchmark results are Arrow IPC; polars is built without default features (no polars-parquet); tests/test_layering.py guards both.

6. **Done** (spec 2026-10-17-generated-dataset-design.md): `datagen.large_dataset(n_rows, seed, null_rate, mean_run, clustering)` and `large_dataset_mixed()` replace tests/data/large_dataset.arrow.  The git pack still holds the old blob (history not rewritten).

7. Any remaining follow-ups tracked in CLAUDE.md.  The Bloom and MinHash follow-ups are prerequisites for ToDo 10: BloomRust is ≈ 6× slower than exact sets on the wide benchmark, and MinHash misses containment pairs (overlap 1, low Jaccard) — exactly the foreign-key → primary-key case.
   Skip-index cost: one-shot runs ≈ 4 granularities × 3 window scans plus ≈ 12 index-size emulations per column (bloom adds 4); not yet benchmarked — add a line to `tests/performance/benchmark_recommend.py` before large frames.
- Polars' process-global categorical mapping: a user's Categorical columns share it, so other Categorical data alive in the same process can inflate a column's measured dictionary size (found while fixing the skip-index flake, commit 3ccc998).
- Skip-index calibration is thin: one column per kind at G ≥ 2, only bloom fpr 0.025 chosen, point queries on Date / Decimal literals not parsed by the calibration script (spec 2026-10-20 §10).
- Streaming skip indexes: inject running counts into the population / baselines instead of the sample's (spec 2026-10-20 §7).
- Consolidate the synthetic data generators (`tests/datagen.py` and `chbench.datagen`) when turbo-parakeet and clickhouse_benchmark merge.
- ClickHouse String size emulation: LZ4 over-predicts (median +19%), small ZSTD(1) columns under-predict (up to 78%) — calibrate the String layout (spec 2026-10-19 §11).
- Emulate sparse serialization (spec 2026-10-19 §11): ClickHouse stores a column part with ≥ 93.75 % default values in sparse serialization (`ratio_of_defaults_for_sparse_serialization`); the emulator does not model it, so near-constant default columns are over-predicted (e.g. 925 vs 46 bytes; confirmed on the server: 50 000 zeros and 5 % non-zeros are `Sparse`, 50 % is `Default`).
- ClickHouse Date32 / DateTime64 range limits (1900–2299) are not checked by the type mapping (spec 2026-10-19 §2).
- Emulator block cutting: a granule over 1 MiB closes the open block early (ClickHouse fills it first) — matters only for rows averaging > 128 bytes.
- Recommender parity, string columns whose null slots hold bytes (e.g. built with `pl.when(...).then(None)` or `Series.set`): one-shot measures those bytes in its cast candidates (`datafusion_size_bytes`), streaming does not; `assert_parity` excuses the gap only for the original type.  Decide which is right (an Arrow cast keeps the bytes; a writer may not) and align them.

---
The following ToDo 8 is largely aimed at identifying an optimal sort-order on the dataset to increase the efficiency of sparse indexes such as the ClickHouse primary key.

8. Introduce new Recommenders parallel to the existing Recommenders, 'OneShotOrderingRecommender', 'StreamingOrderingRecommender'.  These will take an optional list of column names which are the columns to be sorted/optimised in particular.  Using the Joint exact and estimated entropies from the existing technique, the recommender should recurse to find which columns have the greatest partitioning effect on other columns and hence the optimal sort order for those columns based on mutual partitioning of data.  If column names are not passed, the optimal sort order for the entire dataframe should be found.  The sort order should have a maximum number of returned columns, say, 10.
- The entropy heuristic proposes candidate orders.  Each candidate is then scored on a sorted sample on both (a) compressed size (runs, REE, ZSTD) and (b) predicted skip share for point queries on the sort columns and the others.  Both are reported; there is no single weighted score.
- Re-sorting changes every order-dependent metric, so the recommender re-evaluates the runs test, MinMax, Set and delta on the candidate order.
- Consider whether NVI clustering (notebooks/NVI Clustering.ipynb) and visual output would help.

---
The following ToDos 9 to 11 are largely aimed at identifying candidate optimisations to joins between two or more dataframes which can be slow in columnar storage

9. Introduce new classes parallel to the recommenders: OneShotKeyDetector, StreamingKeyDetector (ordered pair/triplet-wise) - uses the joint and marginal entropies to detect groups of up to three columns that are unique (or close to unique for a given threshold) for a dataframe.
- A group is a key when its (joint) entropy is within a margin of the maximum, log2(N) for N rows.  The margin allows for data-quality issues — a key that should be unique but is not quite, e.g. a duplicate row or an erroneous null.
- The margin is set by a duplicate share f of at most 1% (default 1%): the threshold is the maximum entropy a column can have when a share f of its rows repeat an earlier value.  With N(1 − f) distinct values the maximum has fN values occurring twice and the rest once, giving H = log2(N) − 2f.  So a key needs H ≥ log2(N) − 2f, i.e. a margin of 0.02 bits at f = 1%, whatever N is.  (Today's `near_unique` margin of 0.1 bits tolerates 5% of rows duplicated.)
- Both kinds are reported: keys that are exactly unique as sampled (gap 0) and keys that need the margin.  The cause of the gap is not identified.
- Only minimal keys are reported (no column subset is itself a key within the margin).  Each key's mutual information / total correlation is reported as a metric and keys are ranked by it, with no MI threshold, so "unique by chance" stays visible rather than filtered.
- Pruning: a pair or triplet can only be a key if the sum of its marginal entropies is at least log2(N) (joint entropy never exceeds the sum of marginals), and supersets of keys are skipped.
- Constraint for the streaming detector: the entropy estimate's error must be smaller than the margin, or exactly unique and near-unique keys cannot be told apart.  AMS cannot do this (held to 5%: ≈ 1 bit at 1M rows, against a 0.02-bit margin).  An HLL of each candidate combination gives D̂, and log2(N / D̂) is a lower bound on the gap, which prunes candidates; at p = 14 (≈ 0.8%, ≈ ±0.035 bits at 3σ) it is still too coarse to confirm a key within 0.02 bits.  Confirmation in streaming mode is an open design point (e.g. a larger HLL precision or an exact check of the few surviving candidates).
- Takes over the techniques' `near_unique` conclusions (ToDo 5a).

10. Introduce new classes parallel to the recommenders: OneShotJoinDetector and StreamingJoinDetector that take two or more dataframes and using the bloom filter technique identify primary-primary key pairs, and primary-foreign (non-unique) key pairs.  Heavy pruning can be achieved through the comparison of dtypes.  Depends on ToDo 7 (Bloom performance, MinHash containment recall) and on ToDo 9 (PK–PK and PK–FK detection on composite keys needs 9's keys).  Takes over membership's `relationship` / `unique_a/b` conclusions (ToDo 5a).

11. Research how ClickHouse, DataFusion and Polars join tables and think if there are other optimisations (sparse indexes, dense indexes, sort ordering) that would again optimise the speed of the join.

---
Aspirational!

12. Find a way to a) make the Recommenders return the data with their recommendations applied, b) make the Recommenders composable.  This would allow us, for example to take two dataframes one with coarse-grained data types, the other all cast as string and identify that they have a primary-foreign join on an integer column if both appropriately optimised.  12a is partly built: the OneShotRecommender already casts every candidate (recommenders/engine/cast.rs), so returning the data with recommendations applied mostly means exposing that.

---
Performance.

13. With the desired suite of techniques and recommenders in place perform a thorough review of the performance of the Rust libraries, consolidating on fewer hashmaps, patterns etc. where performance is marginal, and ensuring the full suite of performance techniques we have identified are applied wherever applicable.  Includes converting the kernels from Polars to native Arrow (ToDo 5b).

---
Refactoring and more aspiration!

14. This project is a sibling of another project in github named 'clickhouse_benchmark' which is the start of a framework for generating synthetic datasets and testing the effectiveness of dtype shrinking, codecs, skipping indexes and sort ordering in Clickhouse, i.e. the empirical analogue of this project.  The end goal is to create a comprehensive toolset for optimising ClickHouse through identification of candidate columns and techniques and empirical benchmarking.  The handoff is agreed (CLAUDE.md § Project Objectives, Validation): the recommender's Arrow result written as IPC, plus a `CREATE TABLE` helper (ToDo 1.0).

---
Manually added 18th Oct and to be worked into the plan above.

15. Extend the accuracy and performance benchmarks to include DataFusion. On deterministic techniques, the benchmark is Rust (Deterministic) Technique vs Python Polars vs Python DataFusion.  On probabilistic techniques the benchmarks are: a) Rust Probabilistic vs Rust Deterministic, b) Rust (Probabilistic) Technique vs Python Probabilistic Techniques vs Python Deterministic Techniques.

16. Review and consolidate counting methods: frequency counts take place in Describe, Entropy etc.