# Skip-index recommendations — design (ToDo 1c, 1d, 1e)

Status: approved 2026-10-20.  Sub-project B of ToDo 1a–g (A: ClickHouse types / codecs / compression, spec
2026-10-19, done; C: run-end encoding 1a; D: range queries 1g).  Fills the reserved `clickhouse_skip_indexes` column
of the schema skeleton (spec 2026-10-18 §2.3).

## 1. Goal and scope

Recommend, per column, at most one ClickHouse data-skipping index — `minmax`, `set(N)` or `bloom_filter(fpr)`, with
its GRANULARITY — for point queries `col = v`, in both recommenders; list every evaluated candidate with its
measured skip shares, index size and the reason it was chosen, rejected or pruned; and validate the predictions with
`clickhouse_benchmark` (`chbench`).

Measurements come from the existing techniques (spec 2026-10-16, `analytics.minmax_index`, `analytics.set_index`;
Rust `techniques/single_column/ordered/skip_index/`); this spec adds only the decisions, index sizes, the streaming
wiring and the handoff.  Row order is the stored order (CLAUDE.md § Project Objectives): for ClickHouse the caller
supplies data sorted by the intended `ORDER BY`.

Out of scope: range queries (ToDo 1g, sub-project D); the `text` / `ngrambf_v1` / `tokenbf_v1` indexes; several
indexes on one column; skip indexes on the primary-key column (the caller's `ORDER BY` is not known here).

## 2. Candidates

For each eligible column (the skip-index techniques' ordered scalar types, `analytics._dtypes.ORDERED_SCALARS`) and
each GRANULARITY G in `skip_granularities` (window w = G × 8 192 rows, `index_granularity` 8 192):
- `minmax`;
- `set(N)` for N in `set_max_sizes`;
- `bloom_filter(fpr)` for fpr in `bloom_fprs` — evaluated only when no `minmax` or `set` candidate passes (§4).

A kind ClickHouse rejects for the column's ClickHouse type (`clickhouse_type`, spec 2026-10-19 §2), as measured by
chbench's vocabulary spike (R36: `bloom_filter` on `Decimal*`, `Date32`, `Time64`), is never a candidate (a `pruned`
row names the rule).  A column without a ClickHouse type has no candidates.

## 3. Metrics (existing techniques)

Per candidate, on the stored order:
- `skip_rows`: the share of point queries skipped, the queried value drawn by row frequency — the observed amount of
  data skipped (technique `skip_rows_mean`); `skip_values`: the same, uniform over distinct values (`skip_values_mean`);
- `expected_skip_rows`: the same under a uniformly random row order (the technique's closed-form baseline);
- `set(N)`: the technique at `max_size = N`, plus `window_distinct_mean`, `window_distinct_max`, `overflow_fraction`
  (share of windows with more than N entries, values + NULL);
- `bloom_filter(fpr)`: the unlimited set's shares (`set_index` at `max_size = 0`) × (1 − fpr) — a block not holding v
  is skipped unless the filter reports a false positive.

## 4. Decision

Keywords (both recommenders) with defaults: `skip_granularities = (1, 2, 4, 8)`, `set_max_sizes = (125, 250)`,
`bloom_fprs = (0.025,)`, `min_skip = 0.3`, `min_gain = 0.1`, `max_overflow = 0.1`.

1. **Gate (every kind):** `skip_rows ≥ min_skip` and `skip_rows − expected_skip_rows ≥ min_gain`; else `rejected`
   with the two figures as evidence.
2. **Set overflow:** `set(N)` with `overflow_fraction ≥ max_overflow` is `pruned` (evidence: mean / max window
   distinct, overflow fraction) — high **local** cardinality; a globally high-cardinality but clustered column stays
   eligible.
3. **Bloom (fallback only):** evaluated when no `minmax` / `set` candidate passed 1–2; it must also pass the gate and
   the economics test `skip_rows × clickhouse_size_bytes > bloom bytes` (bytes skipped per point query vs the filter
   every query reads; `clickhouse_size_bytes` is the column's predicted compressed size, spec 2026-10-19).
4. **Granularity within a kind:** among a kind's passing candidates (per N / fpr), the largest G whose `skip_rows` is
   within 0.02 of that kind's best — a coarser index for nearly the same skipping.
5. **Across kinds:** the highest `skip_rows`; ties (within 0.02) go to the smaller index (`minmax` < `set` < `bloom`).
   At most one `chosen` per column; the others passing 1–3 are `rejected` with "lower skip than <chosen>" or "larger
   index than <chosen>".

## 5. Index sizes

Emulated, per index block of G granules (`common/clickhouse_sizes.rs` conventions, ClickHouse physical values):
- `minmax`: two values per block at the type's width (min, max; nullable: plus a flag byte), compressed as one stream
  with LZ4;
- `set(N)`: per block its distinct values (String layout for strings) or an empty set when it overflows, plus a count
  per block, compressed with LZ4;
- `bloom_filter(fpr)`: sized by the **distinct values D of each index block**, not its rows (measured, chbench R37 /
  FINDINGS §10): per block ⌈D × b / 8⌉ + v(D) bytes, b = 15 / 10 / 8 / 7 bits for fpr 0.001 / 0.01 / 0.025 / 0.05,
  v(D) = 1 byte for D < 128, 2 for D < 16 384, else 3; stored effectively uncompressed (random bits; compression
  only collapses byte-identical blocks).  An fpr outside the table → a `pruned` row.  So a bloom filter is cheap on
  clustered data and costs about a byte per row only where every row of a block is distinct.
- Measured per-block constants (R37): `minmax` 2 × value width (16 bytes for Int64); `set(N)` 8 bytes plus
  value width × D, 8 bytes when the block overflows; index marks (121–305 bytes) are not modelled.

Approximate by design; calibrated (§8.3).

## 6. Output columns

- `clickhouse_skip_indexes` (reserved, now emitted; type finalised): `List<Struct{kind: Utf8, granularity: UInt32,
  max_size: UInt32, fpr: Float64, bytes: UInt64, skip_rows: Float64, skip_values: Float64, expected_skip_rows:
  Float64, outcome: Utf8, evidence: Utf8}>` — every evaluated candidate (null fields where they do not apply;
  `outcome` ∈ {chosen, rejected, pruned}), ordered chosen first, then by `skip_rows` descending.
- `clickhouse_skip_index` (new, String, after `clickhouse_codecs`): the chosen index as written in DDL —
  `minmax GRANULARITY 1`, `set(125) GRANULARITY 2`, `bloom_filter(0.025) GRANULARITY 1` — or null.

## 7. Streaming

The sampled reservoir blocks are the window segments (spec 2026-10-16 §5): (a) a multi-segment entry point in the
techniques — window starts clipped to `seg.start .. seg.end − w + 1` per segment, `WindowSums` merged; (b) the
population — ids and counts — is that of the sampled rows, so `Population::from_ids`' counts describe exactly the
rows scanned, and the denominators and the random-order baselines are the sample's (consistent with the windows;
exact when every row is sampled).  Injecting the running counts instead is a later refinement.  A G whose window exceeds
`block_rows` cannot be evaluated: `pruned` with that reason (with the default 65 536 rows, G ≤ 8).  No sample
(`reservoir_rows = 0`) → no candidates, `clickhouse_skip_index` null.  Parity: exact when one block holds every row
(one batch, fully sampled); otherwise the shares are those of the sampled blocks (a stated tolerance in the tests).

## 8. Validation with chbench

### 8.1 chbench changes (in that repository, its conventions)

- Record each table's skip-index size from `system.data_skipping_indices` (`data_compressed_bytes`,
  `data_uncompressed_bytes`) in the store and the report CSV, alongside the existing per-query `selected_marks`.
- A spike measuring `bloom_filter(fpr)` index bytes per row for fpr ∈ {0.001, 0.01, 0.025, 0.05} on known inputs
  (FINDINGS; a ruling) — the source of §5's `bits_per_row` table.

### 8.2 `to_chbench`

The recommended suite's `skip_index` axis becomes `[<chosen>, none]` (when an index is chosen), and its queries gain
point `in` filters (`values: 1`) at target fractions 0.001, 0.01 and 0.05 (chbench picks the value).

### 8.3 Calibration

The calibration script (`tests/performance/calibrate_clickhouse_sizes.py`, extended, or a sibling) runs the suites
and compares, per query: the predicted skip for that query's actual value — computed exactly in Python from the
dataset (share of index blocks not containing v, or with v outside [min, max]) — with the measured share of marks
skipped (`1 − selected_marks / total_marks`); and per table the predicted index bytes with the measured ones.
Tolerances are adopted from the evidence and written into §10.

## 9. Testing

- **Rust:** the decision rules (gate, overflow prune, bloom fallback and economics, granularity choice, cross-kind
  choice and ties); index-size emulation (byte-level for small inputs); the multi-segment entry point (equals the
  single-segment scan on one segment; segments shorter than w contribute nothing).
- **Python (known answers):** a sorted unique column → `minmax`; a high-cardinality column clustered in groups of
  ~1 000 rows → `set(N)` at the smallest N that does not overflow; a 50-value column in random order → none (gate:
  baseline); grouped random hashes (high local cardinality, clustered, unordered within a block) → `bloom_filter`;
  each keyword moved across its boundary; ineligible types; a Decimal column → no bloom candidate.
- **Parity:** one-shot vs streaming (one batch, fully sampled: equal; several batches: within the stated tolerance).
- **Java / C ABI:** the keywords and the two columns.
- **chbench:** index sizes in the store; the bloom spike.

## 10. Outcome

Calibration run: `tests/performance/calibrate_skip_indexes.py` (ClickHouse 26.8 in `chbench-ch`, chbench branch
`skip-index-sizes`; ~3 min for 40 suites, 3 workers), result `tests/performance/results/skip_index_calibration.arrow`
(666 cells = 333 `none` controls + 333 indexed). Frames: the four known-answer shapes at 1 000 000 rows (clustered ids
in runs of 1 000, grouped hashes in runs of 10; random 50 values chooses no index, sorted unique ids chooses `minmax`)
and `large_dataset_mixed(200 000)`; 40 suites had a chosen index (37 from `large`, 3 from the shapes) (14 columns `minmax`, 10 `set`, 10 `bloom_filter`,
and one column each at G = 2 / 2 / 4). One type, the first codec's `none`, one compression. Queries: the three `in`
point queries of `to_chbench` plus six extra fractions (more distinct values), one sample each. The predicted skip is
computed exactly in Python from the suite's dataset in stored order; measured = `1 − selected_marks / total_marks`.

**Marks.** `total_marks` (the `none` table's full scan `selected_marks`) equals `ceil(rows / 8 192)` in all 333 tables
(no off-by-one; the last partial granule counts as one mark). The control is exact: with `none`, every point query
selects all marks (measured skip 0 in 333 of 333 cells).

**Bloom sizes (chbench R37; spike FINDINGS 10, 24 of 24 cells reproduced).** A block holds `ceil(D × b / 8)` bytes plus
1 / 2 / 3 bytes of length (D < 128 / < 16 384 / larger), D the block's distinct values:

| fpr | 0.001 | 0.01 | 0.025 | 0.05 |
|---|---|---|---|---|
| bits per distinct value b | 15 | 10 | 8 | 7 |

`minmax` 16 B per block; `set(N)` 8 B per block plus 8 B per value (8 B when the block overflows).

**Skip share, predicted (exact set semantics) vs measured** — |predicted − measured|, in share of marks:

| kind | G | cells | distinct (column, value) | mean abs | max abs | mean (measured − predicted) |
|---|---|---|---|---|---|---|
| minmax | 1 | 126 | 15 | 0 | 0 | 0 |
| minmax | 2 | 9 | 1 | 0 | 0 | 0 |
| set | 1 | 90 | 38 | 0 | 0 | 0 |
| set | 2 | 9 | 1 | 0 | 0 | 0 |
| bloom_filter(0.025) | 1 | 90 | 35 | 0.0053 | 0.080 | −0.0053 |
| bloom_filter(0.025) | 4 | 9 | 1 | 0.065 | 0.065 | −0.065 |

Against the fpr-adjusted prediction (`× (1 − fpr)`): G = 1 mean abs 0.0097, max 0.066; G = 4 0.041. minmax and set
(including set overflow blocks, `set(250)` and `set(125)` cells) match the model to the mark. Bloom: over the
36 distinct (column, value) points the measured skippable-block loss is 1.7 % at G = 1 (expected fpr 2.5 %; 14.04 of
14.28 predicted skip-units) and 6.7 % for the single G = 4 point (2 of 31 blocks); the cells are small
(25 marks per `large` table: one false positive is 0.04, two 0.08, which is the `max abs` above), so the data are
consistent with fpr and neither confirm nor refute a systematic offset. Only fpr 0.025 was ever chosen, so 0.001 / 0.01
/ 0.05 skip shares remain uncalibrated (sizes: R37).

**Index bytes, predicted vs measured compressed** (per table):

| kind | G | tables | median rel | max rel | max abs bytes | mean signed rel (pred − meas) |
|---|---|---|---|---|---|---|
| minmax | 1 | 14 | 0 | 0 | 0 | 0 |
| minmax | 2 | 1 | 0 | 0 | 0 | 0 |
| set | 1 | 10 | 0 | 0 | 0 | 0 |
| set | 2 | 1 | 0 | 0 | 0 | 0 |
| bloom_filter(0.025) | 1 | 10 | 0.66 % | 0.67 % | 125 | −0.66 % |
| bloom_filter(0.025) | 4 | 1 | 0.44 % | 0.44 % | 446 | −0.44 % |

minmax and set bytes are exact (the compressed file equals the model's byte count). Bloom is a consistent 0.4–0.7 %
under the measurement (e.g. 18 376 predicted vs 18 500 measured): incompressible bloom bits plus LZ4 framing
(plausibly the worst-case expansion, not confirmed separately); the model reproduces D and the per-block header exactly.

**Adopted tolerances** (for tests and for reading the recommender's predictions):
- `minmax` / `set` skip share: exact (the share of marks is a multiple of 1 / total_marks; use an absolute 1e-9 for
  float comparison); index bytes exact (0 B).
- `bloom_filter` skip share: expectation `(1 − fpr) × exact`; a single value deviates by binomial noise,
  tolerance one mark's worth per false positive, i.e. an absolute 0.08 on tables of 25 marks (≈ 2 / total_marks + 0.01
  in general), and 0.01 on the mean over many values (observed 0.005 signed, always on the pessimistic side).
- `bloom_filter` index bytes: predicted within 1 % of measured (observed −0.4 … −0.7 %, always under).

No systematic deviation found. Caveats: few distinct queried values per column (uniform data gives one value for all
target fractions, so 91 of 333 cells are distinct points), 25 marks per `large` table, one fpr, G ≥ 2 on one column
each, and 54 point queries on Date / Decimal-typed columns whose literals the script does not parse (skipped, not
counted); the `large` frame uses 200 000 rows rather than 50 000 for mark resolution.

## 11. Documentation

CLAUDE.md (recommender paragraphs: the candidates, metrics, decision, keywords, columns; streaming segments), ToDo.md
(1c, 1d, 1e done; the streaming items under 1d/1e done), chbench CLAUDE.md (index sizes, the spike).
