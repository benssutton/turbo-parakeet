# Parity with the calibration scripts, and the first runs (2026-10-28)

ClickHouse 26.8.31.2 (`chbench-ch`), Windows laptop, 3 workers, 3 samples. Code: commit 11d7266 (the runs' `run.json`
says 01a943c: they ran with 11d7266's `predict.py` change uncommitted, and both were re-scored with
`python -m evaluate report` at 11d7266).

## Method
1. The retired scripts, at 01a943c, server up:
   `python tests/performance/calibrate_clickhouse_sizes.py` (`synth_frame(50 000)` + the GCD and Delta frames, 1M rows)
   and `python tests/performance/calibrate_skip_indexes.py` (`synth_frame(200 000)` + the four skip frames, 1M rows).
   The skip script crashed on a literal `1061684000.0`: chbench renders a nullable integer column's queried values as
   floats (numpy has no nullable int). It was re-scored from its kept stores (`--reuse`) with a local one-line patch
   (integral float literals → int) — the same fix 11d7266 makes in `evaluate/predict.py`, where they were counted
   unparsed.
2. The loop on the same frames (the registry's `gcd` / `delta` / `skip` frames equal the scripts' frames at 1M rows):
   ```
   python -m evaluate run synth_frame gcd delta --runs-dir services/evaluate/runs      # 20261028-140014, 1 764 table loads
   python -m evaluate run synth_frame:200000 skip --runs-dir services/evaluate/runs   # 20261028-142426, 1 646 table loads
   ```
3. Join, old against new. Sizes on (column, base type, codec, compression); skip shares on (column, index, queried
   value), the value normalised (quotes, `true` → 1, `7.0` → 7). A key can occur twice (a recommended and an original
   type with the same base type, e.g. `Time64(6)` / `Time64(9)`; one value drawn by two target fractions), so an old row
   counts as matched when some new row with its key agrees on every compared field.

## Parity
| | old rows | matched | predicted differ | measured differ |
|---|---|---|---|---|
| size cells (bytes) | 369 | 369 | 0 | 0 |
| skip queries with an index (share; index bytes) | 99 (11 columns) | 99 | 0 (share), 0 (bytes) | 0 (share), 0 (bytes) |

Predicted and measured bytes are equal; skip shares within 1e-12. The old skip control rows (99, no index) are not
compared: the loop's `none` rows measure the same 0. Cells the scripts did not produce, and the loop does: the binary
columns (5 suites failed in each script; the loop records them in `suites/skipped.json`), the fixed grid on every
column (the size script widened only the Delta frames), and the rejected skip candidates (the skip script ran the
chosen index only). The loop's grid is a superset: 1 427 measured size cells and 1 044 skip queries (run 1),
1 299 and 1 134 (run 2).

## First-run figures
Coverage lines (`evaluate report`):
```
20261028-140014: columns 70 (skipped 19) · cells 1427 measured / 1648 (221 skipped by ClickHouse) · null predictions 118 · notes 0 · failed datasets 0
20261028-142426: columns 63 (skipped 20) · cells 1299 measured / 1520 (221 skipped by ClickHouse) · null predictions 118 · notes 0 · failed datasets 0
```
Skipped columns: all-null and nested columns (to_chbench), binary columns (chbench). The 118 null predictions are cells
ClickHouse measures but the emulator does not model: Delta / DoubleDelta on Float64 (44), Delta / GCD / DoubleDelta on
Bool (60), Delta / DoubleDelta on Time64 (14).

**Size** (cells with both bytes; run 1 / run 2): every fixed-width cell is byte-exact (1 201 / 1 073 cells, all GCD,
Delta and DoubleDelta chains included). `LowCardinality(String)`: median |error| 0.8% / 1.2%, p90 3.5% / 18%, max 32% /
24% (44 cells each). String: median 9.5% / 9.1%, p90 35%, max 97% (64 cells each; ToDo 9).

**Size regret** (recommended type, 70 / 63 columns): the recommended chain is the measured best in 56 / 51; the worst
regret is 5.9% / 5.7% (`c09_dec_uniq_na_sparse_rand_mult`: LZ4 against ZSTD(9)); delta `walk` loses 4.0% to
DoubleDelta, ZSTD(1); `gcd_three` 2.9% to GCD, ZSTD(9).

**Skip share** (|predicted − measured| per query): exact (≤ 1e-16) on every non-null column for minmax and set; bloom
(fpr 0.025, `grouped_hashes`) within 0.009–0.041 (the × (1 − fpr) expectation). Nullable columns are not modelled:
minmax mispredicts 27 of 144 / 85 of 180 queries, by up to 0.96, and agrees cell for cell with a block's NULL counted
as +∞ (checked on the old run's `minmax:1` queries); set skips nothing on the two nullable columns with a set
candidate (26 of 36 / 108 of 108 queries mispredicted, measured 0 against a predicted mean of 0.65 / 0.84). Index
bytes: exact on non-null minmax / set; nullable columns over-predict minmax (up to +72%) and under-predict set (−3.5%
to −33%); bloom within 0.5%.

**Skip regret** (18 / 16 skip suites): the chosen index is the measured best in 15 / 13; the others lose 0.008 (one
mark in 123: G 2 against G 1 on sorted data) except `grouped_hashes`, where the chosen `bloom_filter(0.025):4` skips
0.902 against 0.992 for the rejected `minmax:1`.

**Recommender cost** (one-shot, child process): 16.5 s wall, 49.3 s CPU, 781 MB peak above the frame for run 1's three
frames (2.05M rows); 9.9 s, 29.2 s, 416 MB for run 2 (1.2M rows).
