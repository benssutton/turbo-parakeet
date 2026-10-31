# Evaluation-loop findings

What the loop has taught us about the recommenders: read this before changing a recommender's decisions or its
objective. Each entry is the current reading, with its evidence. The dated run notes in `docs/results/` hold the full
figures; ClickHouse engine behaviour and bench gotchas are in `services/bench/spikes/FINDINGS.md`. Add an entry when
a run teaches something new, and revise an entry when a later run changes it (keep the old figure only if the change
itself is informative). Figures are ClickHouse 26.8.31.2 unless stated.

## 1. Size predictions: exact for fixed width, String is the gap
Every fixed-width cell is byte-exact (1 321 cells, every GCD / Delta / DoubleDelta chain included). `LowCardinality(String)`
has a median |error| of about 1% (max 32%). String has no tolerance yet (median 2.5–9.5%, max 97%; ToDo 9). The emulator
does not model Delta / DoubleDelta on Float64, Bool or Time64 (118 cells measured, not predicted).
Evidence: `2026-10-28-parity-and-first-run.md`, `2026-10-29-query-regret.md`.

## 2. On bytes, the size choice is close to the best
The recommended chain is the measured smallest in about four in five columns (62 of 79 at 1M rows). The worst regret
is 5.9% (`c09_dec_uniq_na_sparse_rand_mult`: LZ4 against ZSTD(9)). The rest come from the ZSTD level tolerance:
`agrees` (the codec rule re-run on measured bytes) holds for 71 of 79. In 4 of the 8 misses the recommended ZSTD(9) is
the measured best but within the 1% tolerance of ZSTD(3), which the rule then prefers.
Evidence: `2026-10-31-follow-ups.md`, Agreement.

## 3. Small or constant String / binary columns: ZSTD(1) is badly under-predicted
`c58_bin_const_na_none_na_na` (a constant 8-byte binary column) is recommended ZSTD(1) at 263 898 B against ZSTD(9)
at 12 053 B (20.9×). The emulator predicts 6 640 B at every ZSTD level, so the level tolerance picks ZSTD(1), while
ClickHouse's ZSTD(1) is 21× its ZSTD(3) here. The constant String column `c21_…` measures the same and escapes only
because its recommended type differs. This is the one real size miss (ToDo 9).
Evidence: `2026-10-31-follow-ups.md`.

## 4. Skip shares: exact on non-null columns, Nullable not modelled
minmax and set skip shares and index bytes are exact on non-null columns. Bloom skip shares match the × (1 − fpr)
expectation (0.009–0.041 per query), and bloom bytes are within 0.5%. Nullable columns are not modelled: ClickHouse's
minmax counts a block's NULL as +∞, and set skips nothing on the nullable columns tried (ToDo 10). Skip regret is one
mark in nearly every suite. The exception is `grouped_hashes`, where `bloom_filter(0.025):4` skips 0.902 against 0.992
for the rejected `minmax:1`.
Evidence: `2026-10-28-parity-and-first-run.md`.

## 5. Query time resolves only at ≳ 1M rows
At 50 000 rows a query costs about 3 ms of server CPU, and paired intervals are 0.60 wide (0.88 after Bonferroni over
15 alternatives): 3 of 356 queries resolve. At 1M rows, 103 of 586 resolve. Use the `query_time` group, `--workers 1`
and 10 samples when scoring query cost. At 10 rounds a 15-way Bonferroni widens the paired interval to the full rank
range, so more alternatives need more rounds.
Evidence: `2026-10-29-query-regret.md`, `2026-10-31-follow-ups.md`.

## 6. Bytes against CPU: the size-optimal chain is often not the fastest
The recommender minimises bytes. On the 1M-row run (`20261030-232510`), in 33 of 79 recommended-type columns a faster
chain's paired 95% full-scan CPU interval lies below 1. It saves a median 34% CPU for a median 69% more bytes, and it
is an LZ4 chain in 26 of the 33. `scripts/faster_chains.py <run>` lists them. Four groups:
- **Delta / GCD walks and steps under heavy ZSTD** (`delta/*`, `c26` / `c28` / `c49` walks, `gcd_1000`): the faster
  chain is LZ4 or Delta + LZ4, at +70–530% bytes for −12–50% CPU. This is the most expensive trade in bytes.
- **Unique or sorted columns under plain ZSTD** (String / categorical / binary / int keys, `rel_*_unique`, `gcd_even`,
  `gcd_three`): LZ4 at +30–60% bytes for −30–60% CPU. This is the cheapest trade.
- **Mid-cardinality integers at ZSTD(9)** (`rel_indep` / `rel_p` / `rel_noisy50` / `rel_subset90_int_mid`): ZSTD(1)
  or ZSTD(3) doubles the bytes but cuts CPU by 53–61%. Level choice alone is a large CPU lever (inferred: the
  ZSTD(9) output decodes slower here; not verified).
- **Poor trades**: floats (`c15`: +161% bytes for −5% CPU; `c31`: +111% for −13%), `c14` (+57% for −11%).
`c58` is in the 33 but is no trade-off: ZSTD(3) is 95% smaller *and* 68% faster (§3). Absolute savings are small: the
recommended chains' full scans take 6–37 ms per million rows. The bench agrees that ZSTD never costs less CPU than LZ4.
This bears on the objective (ToDo 25): a CPU weight would move the unique / sorted columns and the ZSTD(9) levels
first, and the floats last.
Evidence: `2026-10-31-follow-ups.md`, The bytes × CPU frontier; `2026-10-29-query-regret.md`.

## 7. Streaming matches one-shot, and is cheaper on large frames
On the calibration group, types are identical. One chain differs: `gcd_1000`, decided on the 524 288-row reservoir,
ZSTD(1) against ZSTD(9), costing 1.1%. One skip index differs: `grouped_hashes` `:2` against `:4`, where streaming is
better. At 1M rows streaming takes half the one-shot wall time and a third of its peak memory. At 50 000 rows it is
slightly slower and heavier.
Evidence: `2026-10-31-follow-ups.md`, Streaming against one-shot.

## 8. Recommender cost
One-shot on the 77-column `synth_frame` at 1M rows: 39 s wall, 122 s CPU, 1.64 GB peak above a 523 MB frame. On
`gcd` (1M rows): 7.9 s and 785 MB one-shot against 3.6 s and 239 MB streaming.
Evidence: `2026-10-31-follow-ups.md`.

## 9. Method gotchas
- `--workers N > 1` times suites in parallel on one server: sizes and skip shares are fine, query cost is not.
- `evaluate compare` aggregates across datasets and does not check rows: compare runs of the same group.
- Runs made before b9a5326 define the String full scan differently (chbench now reads the characters): do not resume
  them for String query cost.
- `size_regret` and summary sections group by the recommended chain, so when a column changes chain between runs,
  `compare` moves its counts between groups rather than changing one group's count.
