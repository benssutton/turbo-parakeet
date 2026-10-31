# The follow-ups run: query time at 1M rows, the frontier, agreement, streaming (2026-10-31)

> Dated record. Runs `services/evaluate/runs/20261030-232510` (`query_time`, one-shot) and `20261031-014618`
> (`calibration`, streaming), both git-ignored, commit 64af040 and re-scored at 57b05e6 (binary skip literals);
> ClickHouse 26.8.31.2 (`chbench-ch`, 4 CPUs), High performance power scheme, 1 worker, 10 samples. The one-shot
> reference is `20261029-204940` (`2026-10-29-query-regret.md`).

```
python -m evaluate run query_time --workers 1                          # 141 min
python -m evaluate run calibration --recommender streaming --workers 1 # 66 min
python -m evaluate report services/evaluate/runs/<run>                 # re-score, no server
python -m evaluate compare services/evaluate/runs/20261029-204940 services/evaluate/runs/20261031-014618
```

`query_time` is `calibration` with `synth_frame` at 1M rows instead of 50 000. Both runs: `columns 79 (skipped 13) ·
cells 1567 measured / 1848 (281 skipped by ClickHouse) · null predictions 118 · failed datasets 0`. The five binary
columns of the covering frame now load (79 columns against 74; 20 measured size cells and 4 skip suites each) and
every skip query has a prediction: 126 binary skip queries at 1M rows, |error| at most 2.2%.

## Query regret at 1M rows

| dataset | rows | queries | resolved (50 000-row run) | resolved (1M-row run) |
|---|---|---|---|---|
| synth_frame filter | 1M | 354 | 2 of 285 | 55 |
| synth_frame full scan | 1M | 82 | 1 of 71 | 18 |
| gcd, delta, skip | 1M | 150 | 29 | 30 |
| all | | 586 | 32 of 506 | 103 |

At 1M rows a full scan of a `synth_frame` column costs milliseconds of server CPU rather than tens of microseconds, so
the Bonferroni-adjusted paired intervals now resolve one query in six. The resolved regrets are large: median 0.31
(filter) and 0.39 (full scan), worst 0.68. They are a CPU-for-bytes trade-off, not mispredicted sizes. The worst
rows are ZSTD(9) choices (`rel_p_int_mid`: 1.95 MB at 16.1 ms) beaten by ZSTD(1) (4.00 MB at 6.8 ms), and ZSTD(1)
String choices beaten by LZ4. The other datasets are unchanged within noise (their rows did not change).

## The bytes × CPU frontier

`size_frontier` on the 79 recommended-type suites of the 1M-row run:
- The recommended chain is the measured smallest in 62 and on the bytes × full-scan CPU Pareto frontier in 66.
- The fastest cell is another chain in 63. In 33 its paired 95% interval (not adjusted) lies below 1: there it
  saves a median 34% of the CPU for a median 69% more bytes, and it is LZ4 in 26 of them (ZSTD(1) 4, ZSTD(3) 3).

The recommender minimises bytes, so this is the frequent trade-off ToDo 2 asked about: an objective with a weight on
CPU would move about two in five columns, mostly to LZ4.

## Agreement: the codec rule on measured bytes

`agrees` (the recommender's own rule re-run on the measured bytes of the chains it compared) holds for 71 of 79 columns
(streaming: 72). Of the 8 disagreements, 4 have zero size regret: the recommended ZSTD(9) is the measured best, but
within the 1% level tolerance of ZSTD(3), which the rule then prefers. Three lose 2–4% (`c25_bin_mid_…`,
`rel_fk_str_unique`, `rel_p_str_unique`). One is a real miss: `c58_bin_const_na_none_na_na`, a constant 8-byte
binary column, recommended ZSTD(1) at 263 898 B against ZSTD(9) at 12 053 B (regret 20.9×; 18.4× at 50 000 rows).
The emulator predicts 6 640 B at every ZSTD level, so the level tolerance takes ZSTD(1). ClickHouse's ZSTD(1) is 21×
its ZSTD(3) on this column. The constant String column `c21_…` measures the same and escapes only because its
recommended type differs. This is ToDo 9's small-ZSTD(1) String gap, at its extreme.

## Streaming against one-shot (calibration)

Against `20261029-204940`, on the 74 columns both runs score:
- Types are identical.
- One codec chain differs: `gcd_1000` (1M rows, over the 524 288-row reservoir), GCD + ZSTD(1) against GCD +
  ZSTD(9), costing 1.1%.
- One skip index differs: `grouped_hashes` `bloom_filter(0.025):2` against `:4`, regret 0.024 against 0.089. Streaming
  is better here.
- Mean size regret 0.0043 against 0.0041 (the streaming run's mean over all 79 columns is 0.24, all of it `c58`).
- Mean skip regret 0.0031 against 0.0062.

Cost of the streaming recommender: at 1M rows, half the one-shot wall time and a third of its peak memory (`gcd` 3.6 s
and 239 MB against 7.9 s and 785 MB). At 50 000 rows it is slower and heavier (2.1 s and 264 MB against 1.8 s and
129 MB).

The one-shot recommender at 1M rows of the 77-column `synth_frame`: 39 s wall, 122 s CPU, 1.64 GB peak above a 523 MB
frame.
