# First query-time scores: `query_regret` on the calibration group (2026-10-29)

> Dated record. Run `services/evaluate/runs/20261029-204940` (git-ignored), commit b168b58, ClickHouse 26.8.31.2
> (`chbench-ch`, 4 CPUs), High performance power scheme, 1 worker, 10 samples. It is the first run under chbench's
> interleaved rounds (`services/bench/docs/results/2026-10-29-interleaved-rounds.md`).

```
python -m evaluate run calibration --workers 1        # 67 min, 1 911 table loads
```

`calibration` is `synth_frame` (50 000 rows) plus the `gcd`, `delta` and `skip` frames (1M rows each). Coverage:
`columns 74 (skipped 19) · cells 1547 measured / 1768 (221 skipped by ClickHouse) · null predictions 118 · notes 0 ·
failed datasets 0`.

## Sizes and skip shares: unchanged

The sampling protocol does not touch bytes or skip shares. All 1 321 fixed-width cells are byte-exact, and String
remains the gap (median |error| 2.5%, p90 28%, max 97%; ToDo 9). Size regret: the recommended chain is the measured
best for 59 of 74 columns, and the worst regret is again `c09_dec_uniq_na_sparse_rand_mult` at 5.9% (LZ4 against
ZSTD(9)). Skip regret: the chosen index is the measured best in 15 of 21 suites. The one large loss is again
`grouped_hashes` (`bloom_filter(0.025):4` skips 0.902 against 0.992 for `minmax:1`); the others lose one mark.

## Query regret

`query_regret` has one row per (dataset, column, suite, query). Each row names the cheapest alternative to the
recommender's chosen table by paired CPU ratio (alternative / chosen, per round). That ratio's interval is
Bonferroni-adjusted over the suite's `n_alternatives`, so a row is resolved only when the best of up to 15 tables beats
the choice at the family-wise level. `regret_cpu` is 1 − ratio when resolved, else 0.

| dataset | rows | queries | resolved | median adjusted interval width | median CPU per query |
|---|---|---|---|---|---|
| gcd | 1M | 16 | 9 | 0.41 | 20.5 ms |
| delta | 1M | 88 | 16 | 0.55 | 12.3 ms |
| skip | 1M | 46 | 4 | 0.51 | 7.2 ms |
| synth_frame | 50 000 | 356 | 3 | 0.88 | 3.4 ms |
| **all** | | **506** | **32** | | |

Summary: 506 rows, 32 resolved (22 of 411 filters, 10 of 95 full scans). The median `regret_cpu` is 0, the median
among resolved rows is 0.33, and the worst is 0.51 (`delta.walk`, full scan: `LZ4` alone at 0.49× the CPU of the
chosen `Delta, ZSTD(9)`).

### Readings

- **The regret is the CPU price of a size choice.** Size suites hold 29 of the 32 resolved rows. The size recommender
  picks the smallest chain, which is usually a ZSTD level: 196 of 296 size-suite queries chose ZSTD(9). In 27 of those 29
  rows the cheaper alternative is an LZ4 chain, at 0.49–0.90× the chosen CPU. This matches the bench's
  finding that ZSTD never costs less CPU than LZ4. It is a trade-off the recommender does not yet weigh, not a
  misprediction. A recommender objective over bytes *and* query CPU would turn these rows into decisions (follow-up).
- **Skip suites hold 3 resolved rows of 210.** On `grouped_hashes` at filter 0.001, `minmax:2` costs 0.81× the chosen
  bloom filter; this is the same column where skip regret shows the bloom choice losing. On
  `c26_int_na_na_none_na_walk` at 0.2, `minmax:2` costs 0.72× of `minmax:1`. On `c34_str_uniq_na_none_sort_na`, no
  index costs 0.75× of `minmax:1` on the full scan: the index costs CPU without saving any reads.
- **50 000-row frames cannot resolve query time.** synth_frame's queries take a median 3.4 ms of CPU, and its paired
  intervals are 0.60 wide before adjustment and 0.88 after. Only 3 of its 356 rows resolve. The 1M-row frames resolve
  at about 0.3 unadjusted. Query-time scoring needs frames of ≳ 1M rows, or more samples on the small ones.
- The Bonferroni adjustment matters at this n. With 15 alternatives, each interval is taken at level 1 − 0.05/15,
  which at 10 rounds widens the paired interval from ranks 2–9 to ranks 1–10, the full range.

## Follow-ups (ToDo 2)

- Query CPU in the size recommender's objective, or a reported bytes / CPU frontier, so the LZ4-vs-ZSTD rows become
  decisions.
- Query-time scores on synth_frame need more rows or samples (`synth_frame:1000000`, or `--samples 20`).
- `--workers` defaults to 3, which times suites in parallel on one server; this run used `--workers 1` (ToDo 3
  follow-up).
