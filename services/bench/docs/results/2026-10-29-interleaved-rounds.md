# Interleaved rounds: the four example suites, 2026-10-29

> Dated record. It supersedes the timing claims of `2026-09-17-first-runs.md`; that note's size claims are re-checked
> below and all hold.

The first runs under the ToDo 3 sampling protocol: every table loaded first, one untimed warm-up round, then timed
rounds that run every (table, query) pair once in a shuffled order, with the query cache and the query condition cache
off. Time is server CPU (`cpu_us`, User + System from ProfileEvents) first and server wall (`server_ms`) beside it.
Variants are compared by paired per-round ratios with distribution-free 95% intervals (`chbench report --baseline`).
A ratio is called *resolved* when its interval excludes 1, marked ↑ (slower than the baseline) or ↓ (faster). Every
number is from the CSV exports in this directory, written with `--baseline` (replicates: `skip_index=minmax`, the
others: `codec=none`). The four runs share the store `results-2026-10-29.db`, which is not committed.

## Machine and run

- Same laptop as 2026-09-17: i5-1235U, 15.8 GB, NVMe. ClickHouse 26.8.31.2 in Docker Desktop 29.9.2: container
  `chbench-ch`, 4 CPUs and 8.6 GB for the VM. Python 3.12.13, numpy 2.5.3, pyarrow 25.0.1, clickhouse-connect 1.1.1,
  psutil 7.0.0.
- Power scheme High performance. Spotify and the Epic Games launcher were closed; VS Code and the Claude Code session
  driving the run stayed open.
- Quietness as recorded on each run row (host CPU over 3 s, before / after):

| suite | wall | host CPU % | ClickHouse LoadAverage1 | active merges |
|---|---|---|---|---|
| replicates | 0 min 46 s | 32.3 / 38.3 | 0.78 / 1.45 | 0 / 0 |
| int_encodings | 1 min 42 s | 18.5 / 24.7 | 3.05 / 3.02 | 0 / 0 |
| float_codecs | 9 min 52 s | 18.7 / 13.1 | 1.43 / 2.44 | 0 / 0 |
| decimal32_codecs | 8 min 16 s | 28.8 / 19.4 | 1.63 / 2.23 | 0 / 0 |

  Every run warned that host CPU was over 10%. That is the floor with the editor and the driving session open, so
  these runs are *quieted*, not idle.

## Protocol check: `replicates`

The setup: Int32 at 10M rows in near-random order. Three tables differ only in their skip index (`minmax`,
`set(100)`, `bloom_filter`). They are queried by a full scan and by `IN` filters on frequent values (targets 0.05 and
0.20, actual 0.049 and 0.200). No index can skip a granule here, so the three tables are replicates. There were 20
rounds.

| table vs minmax | query | cpu_ratio [95%] | server_ratio [95%] |
|---|---|---|---|
| bloom_filter | filter 0.05 | 0.97 [0.92–1.07] | 0.96 [0.93–1.07] |
| bloom_filter | filter 0.20 | 1.00 [0.87–1.05] | 0.94 [0.75–1.04] |
| bloom_filter | full scan | 0.99 [0.90–1.06] | 0.93 [0.89–1.04] |
| set(100) | filter 0.05 | 0.99 [0.94–1.07] | 0.99 [0.92–1.05] |
| set(100) | filter 0.20 | 1.01 [0.96–1.07] | 1.00 [0.86–1.17] |
| set(100) | full scan | 0.95 [0.73–1.25] | 0.98 [0.87–1.26] |

**All twelve intervals contain 1.** Four of the six CPU intervals lie within ±10%; the widest is set(100) full scan, [0.73–1.25]. On 2026-09-17 the unpaired p50s
of three identical tables differed by up to 2.9× (58.5 to 168.0 ms). Pairing by round removes the drift that a
shared laptop adds between one table's samples and the next.

The resolution depends on n. At 20 rounds the median order-statistic interval is ranks 6 and 15. At the codec suites'
10 rounds it is ranks 2 and 9, and the median width of a paired CPU-ratio interval there is 0.33. So at 10 samples
these runs resolve differences of about 20% or more, not 5%. Use `--samples-add` where a finer call matters.

No row in any suite has `clocks_disagree`: wherever both clocks resolve, they resolve in the same direction. CPU
resolves more often than server wall (float_codecs: 48 of 72 paired rows against 33).

## float_codecs (Float64, Decimal64(8); 8 dp, lognormal; 10 rounds)

**Sizes hold exactly** (MB, compressed):

| type | pk | compression | none | Delta | DoubleDelta | Gorilla | T64 |
|---|---|---|---|---|---|---|---|
| Decimal64(8) | 0 | LZ4 | 57.1 | 59.6 | 46.5 | — | 37.4 |
| Decimal64(8) | 0 | ZSTD | 39.9 | 50.3 | 46.3 | — | 37.6 |
| Float64 | 0 | LZ4 | 80.3 | 80.4 | 86.7 | 81.6 | — |
| Float64 | 0 | ZSTD | 75.4 | 75.8 | 82.3 | 78.7 | — |
| Decimal64(8) | 1 | LZ4 | 39.2 | 23.3 | 12.6 | — | 16.3 |
| Decimal64(8) | 1 | ZSTD | 10.6 | 8.7 | 12.5 | — | 14.5 |
| Float64 | 1 | LZ4 | 60.9 | 24.6 | 35.2 | 41.2 | — |
| Float64 | 1 | ZSTD | 45.3 | 15.1 | 33.3 | 38.6 | — |

**Codec vs no codec** (paired CPU ratio against `codec=none` in the same type, order and compression; full scan /
filter 0.10 / filter 0.20):

| type | pk | compression | Delta | DoubleDelta | Gorilla | T64 |
|---|---|---|---|---|---|---|
| Decimal64(8) | 0 | LZ4 | 1.25 / 1.23↑ / 1.21↑ | 1.58↑ / 1.41↑ / 1.48↑ | — | 0.98 / 0.97 / 0.97 |
| Decimal64(8) | 0 | ZSTD | 1.10 / 1.12 / 1.14↑ | 0.89↓ / 0.89↓ / 0.89↓ | — | 0.60↓ / 0.61↓ / 0.58↓ |
| Float64 | 0 | LZ4 | 1.71↑ / 1.26 / 1.32↑ | 4.43↑ / 2.21↑ / 3.03↑ | 3.92↑ / 2.03↑ / 2.48↑ | — |
| Float64 | 0 | ZSTD | 1.28↑ / 1.11 / 1.15 | 2.44↑ / 1.76↑ / 2.00↑ | 2.14↑ / 1.65↑ / 1.75↑ | — |
| Decimal64(8) | 1 | LZ4 | 0.84↓ / 0.95 / 0.82↓ | 1.18 / 1.03 / 1.00 | — | 0.94 / 0.86 / 0.89 |
| Decimal64(8) | 1 | ZSTD | 1.11 / 1.01 / 1.07 | 0.88 / 0.84↓ / 0.85↓ | — | 0.83 / 0.78 / 0.80↓ |
| Float64 | 1 | LZ4 | 0.79↓ / 0.83↓ / 0.80↓ | 2.26↑ / 1.56↑ / 1.53↑ | 1.26↑ / 1.19 / 1.19↑ | — |
| Float64 | 1 | ZSTD | 1.23↑ / 1.26↑ / 1.17 | 2.01↑ / 1.48↑ / 1.56↑ | 1.44↑ / 1.31↑ / 1.35↑ | — |

Readings, and how they stand against 2026-09-17:

- **"T64 is the fastest Decimal64 scan in generation order" holds only under ZSTD.** T64 ZSTD costs 0.58–0.61× the
  CPU of no codec + ZSTD, resolved on all three queries. Under LZ4, T64 is level with no codec (0.97–0.98, unresolved).
  The earlier 47.0 against 61.5 ms was noise. T64 is still the smallest Decimal64 column in generation order.
- **Float codecs cost CPU on Float64 and save little or no space in generation order.** Gorilla is 2.0–3.9× no codec
  under LZ4 and 1.7–2.1× under ZSTD; DoubleDelta is 1.8–4.4×. Both are also larger than no codec.
- **Delta LZ4 is the one sorted (pk=1) codec that is cheaper than no codec on both types**: 0.79–0.83× on Float64
  and 0.82–0.95× on Decimal64 (resolved in 5 of 6 rows), and 60% / 41% smaller on disk.
- **Decimal64 against Float64** (paired with `--baseline ch_type=Float64`, in the 12 codec / compression / order
  cells both types have, × 3 queries): Decimal64 is resolved slower in 20 of 24 none / Delta rows (median 1.38×) and
  resolved faster in 11 of 12 DoubleDelta rows (median 0.76×). This is the same direction as 2026-09-17, which listed
  DoubleDelta ZSTD as the only exception; DoubleDelta LZ4 is now an exception too. The full scan sums
  `toFloat64(v)`, so the Decimal64 scan includes a conversion.
- **ZSTD never costs less CPU than LZ4** (paired with `--baseline compression=LZ4`): of 48 ZSTD rows in this suite,
  28 are resolved slower and none faster (median 1.18–1.41× by type and order). The same holds in decimal32_codecs,
  27 slower of 48.

## decimal32_codecs (Decimal32(7), Decimal64(7); 7 dp, same distribution; 10 rounds)

This is the Decimal32 evidence 2026-09-17 could not take: every `Decimal32(8)` cell skipped because one value sat at
the bound 10.0. At 7 dp all 32 valid cells load; Gorilla does not apply to Decimals. The dataset warning (158 of 7.5M
support values merged onto a range bound) is a cardinality miss, not a refusal.

| type | pk | compression | none | Delta | DoubleDelta | T64 |
|---|---|---|---|---|---|---|
| Decimal32(7) | 0 | LZ4 | 40.2 | 40.2 | 46.5 | 33.5 |
| Decimal64(7) | 0 | LZ4 | 52.2 | 54.2 | 46.5 | 33.5 |
| Decimal32(7) | 0 | ZSTD | 35.6 | 36.2 | 44.0 | 33.0 |
| Decimal64(7) | 0 | ZSTD | 36.4 | 41.9 | 44.1 | 33.0 |
| Decimal32(7) | 1 | LZ4 | 30.2 | 13.9 | 8.8 | 11.3 |
| Decimal64(7) | 1 | LZ4 | 37.8 | 20.3 | 8.9 | 11.6 |
| Decimal32(7) | 1 | ZSTD | 17.1 | 6.1 | 6.9 | 10.4 |
| Decimal64(7) | 1 | ZSTD | 8.2 | 5.6 | 7.0 | 10.7 |

- **The Decimal32 / Decimal64 size ratio is not one number.**
  - Under LZ4 it is 0.68–0.80 with no codec or Delta, which brackets the 0.65 that was measured at 100 000 rows.
  - T64 and DoubleDelta store both widths in the same bytes (ratio 0.97–1.00). T64 drops the unused high bytes and
    DoubleDelta encodes differences, so the declared width stops mattering.
  - Under ZSTD with no codec the ratio is 0.98 in generation order and **2.09 sorted**: there, Decimal32 is twice
    the size of Decimal64. Sorted Delta ZSTD is also larger for Decimal32 (6.1 against 5.6 MB). The cause is not
    investigated.
  - Width is therefore a codec-dependent choice, not a free 2× saving.
- **Decimal32 scans cheaper than Decimal64** (paired with `--baseline ch_type=Decimal64(7)`): faster in 39 of 48
  rows and never resolved slower. The median ratio is 0.56 in generation order and 0.74 sorted, and 0.40–0.52 with no
  codec or Delta in generation order.
- Within Decimal32, no codec is cheaper than none except T64 ZSTD on the sorted full scan (0.94↓): DoubleDelta costs
  1.3–3.2× and T64 LZ4 1.49–1.59× in generation order. Within Decimal64 the float_codecs pattern repeats: T64 ZSTD
  runs at 0.54–0.63× of no codec + ZSTD in generation order.

## int_encodings (Int32, UInt32, String, LowCardinality(String); 1000 values; 10 rounds)

`skip_index` is now `[none]`, and `cache_mode` and `query_condition_cache` are retired, so the grid is 8 tables, not
the earlier 72. Sizes hold byte for byte (23 681 004 none, 22 161 670 Delta, 25 707 850 T64 for both integer types;
25 241 654 String; 11 558 861 LowCardinality).

- **T64 is slower than no codec on the full scan** (Int32 1.31↑ [1.21–1.39], UInt32 1.31↑ [1.09–1.53]), on the
  0.20 filter for Int32 (1.13↑) and on the 0.05 filter for UInt32 (1.19↑). Combined with being 8.6% larger, it has nothing to recommend it on this column.
- **Delta** is 6.4% smaller and 1.00–1.13× the CPU of no codec: resolved slower on the Int32 full scan and 0.20
  filter, unresolved elsewhere.
- **Signedness**: Int32 and UInt32 are the same size and their paired ratios agree within their intervals.
- **LowCardinality(String)** is the cheapest of the 8 tables on the filters. On the 0.05 filter it runs at 0.28×
  [0.26–0.31] of String and 0.51× [0.42–0.58] of Int32; on the 0.20 filter at 0.27× and 0.47×. These pairings were
  computed from the store with `harness/stats.paired_ratio`: `--baseline` pairs only within one `ch_type`.
- **The String full scan does not read the strings.** `sum(length(v))` reads 80 MB, which is the 8-byte size
  subcolumn, so the String full scan (18.6 ms CPU) is the cheapest in the suite and does not compare with the integer
  scans. LowCardinality's full scan reads 20 MB (its 2-byte keys). ToDo 3 follow-up.
- Insert, with no index on any table: integers 7.4–8.1 s, String 13.3 s, LowCardinality 13.5 s (FINDINGS §11 covers
  the indexed case).
