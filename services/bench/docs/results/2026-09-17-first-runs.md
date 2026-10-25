# First runs of the two example suites, 2026-09-17

The first end-to-end runs of `chbench` on the two suites in spec §5.1,
`suites/float_codecs.yaml` and `suites/int_encodings.yaml`, at their
specified 10 000 000 rows. Every number below is from the CSV exports
`docs/results/float_codecs.csv` and `docs/results/int_encodings.csv`
(`chbench report --suite <name> --csv`), one row per query variant, keyed by
`query_id`. The raw samples, plans and table rows are in `results.db`, and
the tables are still in the `bench` database.

## Machine and versions

- 12th Gen Intel Core i5-1235U (10 cores, 12 threads, 2.5 GHz base),
  15.8 GB RAM, Samsung MZ9L4256HCJQ NVMe SSD, Windows 11 Home 10.0.26200.
- ClickHouse 26.8.2.7 in Docker Desktop 29.5.2 on Windows (container
  `reverent_noyce`, image `clickhouse:latest`, no CPU or memory limit set on
  the container; the Docker VM sees 12 CPUs and 8.2 GB), HTTP port 8123.
  Everything below crosses the Docker Desktop network stack and its
  file-backed virtual disk; the client and the server share the same ten
  cores.
- Python 3.12.13, numpy 2.3.5, pyarrow 24.0.0, clickhouse-connect 1.8.0.
- Single run of each suite, one after the other, on an otherwise idle
  machine. There is no repeat run yet, so the run-to-run noise is not
  bounded; the within-run spread is in the CSV's `p90`/`p99`/`max` columns.

## float_codecs

Two runs, and the numbers below are their union. The first run (07:51:24
to 07:55:39, **4 min 15 s**) used the suite as spec §5.1 first wrote it,
codec axis Gorilla, Delta, DoubleDelta: dataset generation 6.0 s, then 20
tables at 1.9 to 5.4 s per load and 60 queries of 10 samples. Ruling R28
then added `none` and `T64` to the codec axis, and the rerun (08:32:19 to
08:34:52, **2 min 33 s**) reused the 20 loaded tables without reloading
them (ruling R30) and loaded the 12 new cells, 36 queries, 360 samples.
Nothing errored or had to be resumed. Every query holds `n = 10`.

**Dataset** `9272b36a3bcd968d`: 10 000 000 float64, lognormal (mean 4,
sigma 0.25, shift -1) clipped to `[-1, 10]`, 8 decimal places, cardinality
target 0.75, achieved 0.7499868 (within tolerance). One warning: `miss:
cardinality_ratio short by clip collapse: 132 of 7500000 requested support
values merged onto a value_range bound`. The achieved minimum is
0.05101115 and the maximum is exactly 10.0, held by a single row.

**Grid: 60 cells, 32 ok, 28 skipped, 0 error.** The axes are type
{Float64, Decimal32, Decimal64} x primary key {yes, no} x codec {none,
T64, Gorilla, Delta, DoubleDelta} x compression {LZ4, ZSTD}. Skips:

- All 20 `Decimal32(8)` cells: `Decimal32(8) can represent magnitude up to
  9.99999999 at scale 8, but the achieved maximum is 10.0, which
  overflows.` The one row at 10.0 (a lognormal draw clipped to the range
  bound) disqualifies the whole type. Spec §5.2 had expected this case to
  pass at the rate of the rounding boundary alone (about 4.5e-5 per
  100 000 draws) and did not allow for clipping to the bound, which at
  10 000 000 rows is reached; the spec is amended (ruling R29) and the
  range and scale stand as the owner specified them. **The suite as
  specified produces no Decimal32 evidence**, and will not until the bound
  or the scale changes, which is an owner decision.
- All 4 `Float64` x T64 cells: `T64 codec is not supported for specified
  type Float64.`
- All 4 `Decimal64(8)` x Gorilla cells, with the server's own message:
  `The combination of compression codecs Gorilla(8), LZ4 is meaningless,
  because it does not make sense to apply a floating-point time series
  codec to non-floating-point columns` (and the same for ZSTD).

Note that the `primary_key` axis changes the physical order of the table
(`ORDER BY v` against `ORDER BY row_id`), so a `pk=1` table is sorted by
value and a `pk=0` table is in generation order. The codec comparisons
below are therefore only meaningful within one `pk` value.

**Compressed bytes of the `v` column**, smallest first (`compressed_bytes`
and `compression_ratio` from the CSV; the uncompressed size is 80 000 000
bytes for every variant):

| variant | compressed bytes | ratio |
|---|---|---|
| Decimal64(8) / Delta / ZSTD / pk=1 | 8 734 118 | 9.16 |
| Decimal64(8) / none / ZSTD / pk=1 | 10 608 286 | 7.54 |
| Decimal64(8) / DoubleDelta / ZSTD / pk=1 | 12 520 614 | 6.39 |
| Decimal64(8) / DoubleDelta / LZ4 / pk=1 | 12 553 462 | 6.37 |
| Decimal64(8) / T64 / ZSTD / pk=1 | 14 513 433 | 5.51 |
| Float64 / Delta / ZSTD / pk=1 | 15 052 704 | 5.31 |
| Decimal64(8) / T64 / LZ4 / pk=1 | 16 254 867 | 4.92 |
| Decimal64(8) / Delta / LZ4 / pk=1 | 23 270 370 | 3.44 |
| Float64 / Delta / LZ4 / pk=1 | 24 631 089 | 3.25 |
| Float64 / DoubleDelta / ZSTD / pk=1 | 33 307 958 | 2.40 |
| Float64 / DoubleDelta / LZ4 / pk=1 | 35 182 657 | 2.27 |
| Decimal64(8) / T64 / LZ4 / pk=0 | 37 440 586 | 2.14 |
| Decimal64(8) / T64 / ZSTD / pk=0 | 37 589 133 | 2.13 |
| Float64 / Gorilla / ZSTD / pk=1 | 38 648 224 | 2.07 |
| Decimal64(8) / none / LZ4 / pk=1 | 39 236 721 | 2.04 |
| Decimal64(8) / none / ZSTD / pk=0 | 39 872 280 | 2.01 |
| Float64 / Gorilla / LZ4 / pk=1 | 41 250 560 | 1.94 |
| Float64 / none / ZSTD / pk=1 | 45 300 681 | 1.77 |
| Decimal64(8) / DoubleDelta / ZSTD / pk=0 | 46 333 455 | 1.73 |
| Decimal64(8) / DoubleDelta / LZ4 / pk=0 | 46 504 234 | 1.72 |
| Decimal64(8) / Delta / ZSTD / pk=0 | 50 308 044 | 1.59 |
| Decimal64(8) / none / LZ4 / pk=0 | 57 126 485 | 1.40 |
| Decimal64(8) / Delta / LZ4 / pk=0 | 59 572 803 | 1.34 |
| Float64 / none / LZ4 / pk=1 | 60 865 618 | 1.31 |
| Float64 / none / ZSTD / pk=0 | 75 353 869 | 1.06 |
| Float64 / Delta / ZSTD / pk=0 | 75 786 640 | 1.06 |
| Float64 / Gorilla / ZSTD / pk=0 | 78 679 873 | 1.02 |
| Float64 / none / LZ4 / pk=0 | 80 343 339 | 1.00 |
| Float64 / Delta / LZ4 / pk=0 | 80 372 603 | 1.00 |
| Float64 / Gorilla / LZ4 / pk=0 | 81 627 090 | 0.98 |
| Float64 / DoubleDelta / ZSTD / pk=0 | 82 314 793 | 0.97 |
| Float64 / DoubleDelta / LZ4 / pk=0 | 86 652 671 | 0.92 |

Readings on size:

- Sorting by value (`pk=1`) is worth more than any codec: the same
  Decimal64 Delta ZSTD column is 8.7 MB sorted and 50.3 MB unsorted, and
  the same Float64 column with no codec is 60.9 MB sorted and 80.3 MB
  unsorted under LZ4.
- **T64 is the best Decimal64 codec in generation order.** At `pk=0`,
  Decimal64 T64 LZ4 is 37 440 586 bytes: 34.5 percent under LZ4 alone
  (57 126 485), 37.2 percent under Delta LZ4 (59 572 803) and 19.5
  percent under DoubleDelta LZ4 (46 504 234). With ZSTD the codec matters
  much less: T64 ZSTD (37 589 133) is only 5.7 percent under ZSTD alone
  (39 872 280), and both beat Delta ZSTD (50 308 044) and DoubleDelta
  ZSTD (46 333 455). Delta is a net loss on unsorted Decimal64 under LZ4
  (59.6 MB against 57.1 MB with no codec), the same expansion Plan 1 saw
  on Decimal32 at 100 000 rows.
- **Sorted, T64 loses to Delta.** At `pk=1`, Delta ZSTD (8 734 118) is
  the smallest column in the suite and T64 ZSTD (14 513 433) is 66
  percent larger; under LZ4 T64 (16 254 867) beats Delta (23 270 370) by
  30 percent and no codec (39 236 721) by 59 percent.
- Decimal64 is smaller than Float64 under every matching codec and
  compression, in both orders. With no codec at `pk=0`: 57.1 MB against
  80.3 MB (LZ4), 39.9 against 75.4 (ZSTD). Decimal64 T64 LZ4 at 37.4 MB
  is less than half of any Float64 variant in generation order (the best
  is 75.4 MB, no codec ZSTD).
- Float64 with 8 decimal places of a lognormal is essentially
  incompressible in generation order: no Float64 codec beat ratio 1.06,
  and Gorilla, the documented float codec, gave 0.98 (LZ4) and 1.02
  (ZSTD), both within 2 percent of no codec at all (1.00, 1.06).

**p50 times per query kind**, milliseconds. `client` is the wall clock
around the client call, `server` is `query_duration_ms` from
`system.query_log`. All queries are `cache_mode = warm`,
`query_condition_cache` off. The filters are `v BETWEEN 1.84034585 AND
2.24994892` (target 0.10, actual 0.1000) and `v BETWEEN 2.96396722 AND
3.62662256` (target 0.20, actual 0.2000), identical for every variant.
Same row order as the bytes table.

| variant | full_scan client | full_scan server | filter 0.10 client | filter 0.10 server | filter 0.20 client | filter 0.20 server |
|---|---|---|---|---|---|---|
| Decimal64(8) / Delta / ZSTD / pk=1 | 128.3 | 121.0 | 65.2 | 16.5 | 60.1 | 53.0 |
| Decimal64(8) / none / ZSTD / pk=1 | 122.4 | 115.5 | 67.4 | 20.0 | 71.8 | 29.0 |
| Decimal64(8) / DoubleDelta / ZSTD / pk=1 | 134.8 | 129.5 | 72.8 | 27.0 | 70.0 | 25.0 |
| Decimal64(8) / DoubleDelta / LZ4 / pk=1 | 117.3 | 111.5 | 74.8 | 28.5 | 58.8 | 34.0 |
| Decimal64(8) / T64 / ZSTD / pk=1 | 109.9 | 102.5 | 64.2 | 18.5 | 71.8 | 25.0 |
| Float64 / Delta / ZSTD / pk=1 | 51.7 | 43.0 | 57.6 | 12.0 | 59.4 | 13.5 |
| Decimal64(8) / T64 / LZ4 / pk=1 | 83.4 | 76.5 | 61.5 | 15.5 | 69.2 | 22.5 |
| Decimal64(8) / Delta / LZ4 / pk=1 | 78.4 | 73.0 | 60.7 | 14.0 | 71.9 | 23.5 |
| Float64 / Delta / LZ4 / pk=1 | 79.2 | 34.0 | 57.3 | 10.0 | 57.7 | 12.0 |
| Float64 / DoubleDelta / ZSTD / pk=1 | 75.4 | 71.5 | 70.2 | 24.5 | 67.6 | 22.0 |
| Float64 / DoubleDelta / LZ4 / pk=1 | 60.1 | 57.0 | 61.7 | 16.5 | 70.3 | 32.5 |
| Decimal64(8) / T64 / LZ4 / pk=0 | 52.7 | 47.0 | 106.4 | 99.5 | 64.9 | 57.5 |
| Decimal64(8) / T64 / ZSTD / pk=0 | 72.6 | 66.5 | 76.0 | 69.5 | 71.2 | 64.5 |
| Float64 / Gorilla / ZSTD / pk=1 | 56.4 | 53.5 | 63.4 | 18.0 | 65.5 | 19.0 |
| Decimal64(8) / none / LZ4 / pk=1 | 44.7 | 42.0 | 64.7 | 17.0 | 72.2 | 25.0 |
| Decimal64(8) / none / ZSTD / pk=0 | 149.4 | 142.5 | 159.6 | 151.5 | 148.2 | 141.5 |
| Float64 / Gorilla / LZ4 / pk=1 | 47.1 | 40.5 | 62.2 | 17.0 | 61.8 | 17.0 |
| Float64 / none / ZSTD / pk=1 | 72.1 | 65.5 | 65.1 | 18.5 | 65.7 | 22.5 |
| Decimal64(8) / DoubleDelta / ZSTD / pk=0 | 110.0 | 103.0 | 167.7 | 161.5 | 130.3 | 123.5 |
| Decimal64(8) / DoubleDelta / LZ4 / pk=0 | 133.1 | 127.0 | 132.3 | 125.5 | 129.9 | 123.5 |
| Decimal64(8) / Delta / ZSTD / pk=0 | 133.8 | 126.5 | 180.1 | 173.5 | 172.9 | 166.0 |
| Decimal64(8) / none / LZ4 / pk=0 | 70.3 | 61.5 | 95.6 | 89.5 | 91.5 | 85.0 |
| Decimal64(8) / Delta / LZ4 / pk=0 | 91.9 | 82.5 | 147.9 | 140.0 | 103.1 | 97.0 |
| Float64 / none / LZ4 / pk=1 | 54.6 | 49.0 | 58.1 | 12.0 | 66.5 | 27.0 |
| Float64 / none / ZSTD / pk=0 | 67.7 | 62.0 | 105.1 | 98.5 | 71.1 | 66.0 |
| Float64 / Delta / ZSTD / pk=0 | 57.8 | 54.0 | 61.2 | 58.0 | 58.4 | 55.5 |
| Float64 / Gorilla / ZSTD / pk=0 | 74.9 | 72.0 | 90.5 | 88.0 | 98.1 | 92.5 |
| Float64 / none / LZ4 / pk=0 | 74.0 | 31.0 | 80.2 | 73.0 | 71.0 | 31.5 |
| Float64 / Delta / LZ4 / pk=0 | 71.2 | 44.5 | 71.6 | 34.0 | 74.0 | 36.0 |
| Float64 / Gorilla / LZ4 / pk=0 | 58.9 | 56.0 | 63.7 | 59.5 | 58.4 | 56.0 |
| Float64 / DoubleDelta / ZSTD / pk=0 | 215.4 | 208.5 | 236.2 | 229.5 | 171.9 | 156.5 |
| Float64 / DoubleDelta / LZ4 / pk=0 | 95.1 | 74.5 | 85.4 | 81.5 | 66.3 | 63.0 |

Readings on time:

- The filters read the column, as intended after the `_exact_count_projection`
  finding (FINDINGS §5): on the `pk=1` Float64 Delta LZ4 table the 0.10
  filter reads 1 007 616 rows over 123 marks and the 0.20 filter 2 007 040
  rows over 245 marks, against 10 000 000 rows and 1221 marks for the
  `pk=0` table (`samples.read_rows`, `samples.selected_marks`).
- The primary key does what it should for the filters: server p50 drops
  from 34 to 10 ms (Float64 Delta LZ4, 0.10 filter) and from 140 to 14 ms
  (Decimal64 Delta LZ4). It does not help the full scan.
- **Client time has a floor of about 50 ms** that the server time does
  not: every `pk=1` filter reports a server p50 of 10 to 53 ms and a
  client p50 of 57 to 75 ms. That is the HTTP round trip through Docker
  Desktop plus clickhouse-connect's own work, and it means `client_p50`
  cannot rank anything faster than about 50 ms; use `server_p50` for those.
- **T64 is also the fastest Decimal64 scan in generation order**: full-scan
  server p50 47.0 ms for T64 LZ4 against 61.5 (no codec LZ4), 82.5 (Delta
  LZ4), 127.0 (DoubleDelta LZ4), 142.5 (no codec ZSTD) and 66.5 (T64
  ZSTD). It is within noise of the Float64 scans under LZ4 (31.0 with no
  codec, 44.5 with Delta) while being half their size.
- Otherwise Decimal64 is slower to scan than Float64 under the same codec
  and order, even though it is smaller on disk: full-scan server p50 at
  `pk=0` is 82.5 against 44.5 ms (Delta LZ4), 126.5 against 54.0 (Delta
  ZSTD), 127.0 against 74.5 (DoubleDelta LZ4) and 61.5 against 31.0 (no
  codec LZ4). The exceptions are DoubleDelta ZSTD, where the Float64 table
  is the slowest in the suite at 208.5 ms against 103.0, and T64, which
  Float64 cannot use. The full scan sums `toFloat64(v)`, so the Decimal64
  scan includes a conversion the Float64 scan does not.
- ZSTD costs scan time on this data without buying much space in
  generation order: Float64 DoubleDelta goes from 74.5 ms (LZ4) to
  208.5 ms (ZSTD) for a 5 percent smaller column, and Decimal64 with no
  codec from 61.5 to 142.5 ms for 30 percent.
- Differences under about a factor of two between single p50s are within
  the noise this machine showed (see the int suite's repeats below):
  Float64 no-codec LZ4 gives 73.0 ms for the 0.10 filter and 31.5 for the
  0.20 filter on the same table.

## int_encodings

**Wall clock 14 min 41 s** (07:55:50 to 08:10:31): dataset generation
6.2 s, then 24 tables at 1.6 to 7.8 s per load (except the two
`LowCardinality(String)` tables with a `minmax` or `set(100)` index,
whose inserts took 45.7 and 46.5 s against 3.2 s for the `bloom_filter`
one; `tables.insert_ms`) and 288 queries of 10 samples each (2880 samples).
Nothing had to be resumed. Every query holds `n = 10`. Six tables errored
in this run (below); the rerun after ruling R27 (08:35:06 to 08:35:46,
**40 s**) reused all 24 loaded tables without reloading them (ruling
R30), ran nothing, and re-recorded the six errored cells as skips, so the
store and the CSV now describe the grid as it stands.

**Dataset** `9a4f9f52ff2e2c1c`: 10 000 000 int32 in `[1 000 000,
9 999 999]`, 1000 distinct values (cardinality ratio 0.0001, exact),
entropy 4.0 bits (achieved 3.9999992), runs z-score -0.25 (achieved
-0.2502). No warnings; both solves `ok`.

**Grid: 36 cells, 24 ok, 12 skipped, 0 error** (after the rerun; the
first run had 24 ok, 6 skipped, 6 error). The axes are type {Int32,
UInt32, String} x codec {none, T64, Delta, LowCardinality} x skip index
{minmax, bloom_filter, set(100)}, LZ4 only, no primary key. Skips, with
the server's messages:

- String x T64 (3 cells): `T64 codec is not supported for specified type
  String.`
- String x Delta (3 cells): `Codec Delta is not applicable for String
  because the data type is not of fixed size.`
- `LowCardinality(Int32)` and `LowCardinality(UInt32)` (6 cells, one per
  skip index): `Creating columns of type LowCardinality(Int32) is
  prohibited by default due to expected negative impact on performance.
  It can be enabled with the `allow_suspicious_low_cardinality_types`
  setting.`

The last six were **table errors in the first run**, not skips: the
server refused the `CREATE TABLE` with that message (code 455,
`SUSPICIOUS_TYPE_FOR_LOW_CARDINALITY`), and `validate.py` treated
`LowCardinality` as always applicable because the Plan 1 spike had
measured it on `String` only (FINDINGS §5), exactly the over-strong rule
the deferred-findings note warned about. The harness did what §6.4 says
(error recorded on the table, run continued, no rows in the CSV). Ruling
R27 made the wrapper a validation skip on every type except `String`,
with the server's messages measured by DDL for the whole type vocabulary
(spec §5.2, FINDINGS §5); the setting is not enabled, on the same
reasoning as the nullable sorting key. The rerun recorded the six cells
as skips. `LowCardinality(String)` is the only LowCardinality variant
with numbers.

**Compressed bytes of the `v` column**, smallest first. The skip index is
a separate file and does not change the column's bytes, so the 24 tables
collapse to 8 rows (each figure is identical across the three
`skip_index` values in the CSV):

| type / codec | compressed bytes | uncompressed bytes | ratio |
|---|---|---|---|
| LowCardinality(String) | 11 558 861 | 20 027 553 | 1.73 |
| Int32 / Delta | 22 161 670 | 40 000 000 | 1.80 |
| UInt32 / Delta | 22 161 670 | 40 000 000 | 1.80 |
| Int32 / none | 23 681 004 | 40 000 000 | 1.69 |
| UInt32 / none | 23 681 004 | 40 000 000 | 1.69 |
| String / none | 25 241 654 | 150 000 000 | 5.94 |
| Int32 / T64 | 25 707 850 | 40 000 000 | 1.56 |
| UInt32 / T64 | 25 707 850 | 40 000 000 | 1.56 |

On this column (1000 distinct 7-digit values, 4 bits of entropy, near
random order) **T64 is the worst of the three integer codecs: 8.6 percent
larger than LZ4 alone** (25.71 MB against 23.68 MB), while Delta is
6.4 percent smaller than LZ4 alone. Signedness makes no difference to a
byte. `LowCardinality(String)` is half the size of any integer encoding of
the same values, and a plain `String` column ends up at 25.24 MB, within
7 percent of the raw integers, because LZ4 removes most of the 3.75-fold
inflation of storing seven-digit strings.

**p50 times per query kind, warm mode, query condition cache off**,
milliseconds. The filters are `v IN (1063680)` (target 0.05, actual
0.0488) and `v IN (3971462, 1063680, 9695571, 1947451, ...)` with five
values (target 0.20, actual 0.2000); the String tables get the same
literals quoted. The other three cache/qcc combinations are in the CSV;
their aggregate effect is in the readings below.

| variant | full_scan client | full_scan server | filter 0.05 client | filter 0.05 server | filter 0.20 client | filter 0.20 server |
|---|---|---|---|---|---|---|
| LowCardinality(String) / minmax | 70.9 | 27.5 | 69.4 | 24.0 | 79.2 | 32.5 |
| LowCardinality(String) / bloom_filter | 77.3 | 31.0 | 74.8 | 33.0 | 65.4 | 51.0 |
| LowCardinality(String) / set(100) | 77.3 | 33.0 | 76.0 | 28.5 | 49.8 | 39.5 |
| Int32 / Delta / minmax | 80.6 | 74.0 | 84.0 | 76.0 | 75.0 | 69.5 |
| Int32 / Delta / bloom_filter | 65.8 | 59.0 | 92.9 | 84.5 | 71.7 | 65.5 |
| Int32 / Delta / set(100) | 67.0 | 59.5 | 75.7 | 69.0 | 70.8 | 64.5 |
| UInt32 / Delta / minmax | 64.6 | 35.0 | 71.7 | 66.0 | 69.7 | 65.0 |
| UInt32 / Delta / bloom_filter | 84.8 | 39.0 | 174.5 | 168.0 | 77.3 | 71.5 |
| UInt32 / Delta / set(100) | 83.8 | 35.5 | 63.9 | 58.5 | 71.5 | 65.5 |
| Int32 / none / minmax | 69.7 | 61.0 | 77.2 | 71.0 | 78.3 | 72.0 |
| Int32 / none / bloom_filter | 57.1 | 49.5 | 72.9 | 64.5 | 77.5 | 69.0 |
| Int32 / none / set(100) | 58.5 | 52.0 | 69.9 | 62.5 | 78.2 | 72.5 |
| UInt32 / none / minmax | 71.9 | 36.0 | 62.8 | 57.5 | 59.1 | 54.0 |
| UInt32 / none / bloom_filter | 45.8 | 37.0 | 58.0 | 52.5 | 75.8 | 70.0 |
| UInt32 / none / set(100) | 75.6 | 33.0 | 65.8 | 60.5 | 76.5 | 70.0 |
| String / none / minmax | 54.1 | 50.0 | 78.8 | 73.5 | 114.7 | 109.0 |
| String / none / bloom_filter | 87.0 | 78.5 | 118.5 | 111.5 | 133.3 | 126.5 |
| String / none / set(100) | 63.9 | 57.0 | 96.8 | 91.0 | 100.3 | 94.0 |
| Int32 / T64 / minmax | 67.9 | 60.0 | 80.0 | 73.0 | 89.7 | 83.5 |
| Int32 / T64 / bloom_filter | 103.1 | 97.0 | 70.9 | 65.0 | 84.0 | 77.0 |
| Int32 / T64 / set(100) | 54.5 | 47.0 | 76.3 | 69.5 | 89.6 | 82.0 |
| UInt32 / T64 / minmax | 62.5 | 55.0 | 154.5 | 148.0 | 77.2 | 72.0 |
| UInt32 / T64 / bloom_filter | 97.2 | 90.0 | 164.2 | 157.5 | 107.6 | 101.0 |
| UInt32 / T64 / set(100) | 72.4 | 62.0 | 93.7 | 88.0 | 74.6 | 69.0 |

Readings:

- **No skip index prunes anything.** Every filter on every table reads
  all 10 000 000 rows over all 1221 marks (`samples.read_rows`,
  `samples.selected_marks`), whichever of minmax, bloom_filter or
  set(100) is defined. `in_predicate` picks the K frequent values whose
  combined frequency is closest to the target (1063680, the 0.05 filter's
  single value, is the fourth most frequent at 4.88 percent), and a value
  that covers 5 percent of a near-randomly ordered column is present in
  every 8192-row granule, so there is nothing for a granule-level index to
  skip.
  The `skip_index` axis is inert for this suite as designed.
- That makes the three skip-index tables of each type/codec **three
  repeats of the same measurement**, which is the best noise estimate
  these runs offer: their server p50s for the same query spread from
  58.5 to 168.0 ms (UInt32 Delta, filter 0.05) and from 47.0 to 97.0 ms
  (Int32 T64, full scan). Differences under about a factor of two between
  single p50s should not be read as codec effects.
- Within that noise, the per-type/codec medians of `server_p50` across
  the three indexes, two cache modes and two qcc settings (12 CSV rows
  for the full scan, 24 for the filters) are: full scan 30.2 ms `LowCardinality(String)`, 33.5 `UInt32/none`,
  41.8 `UInt32/Delta`, 42.2 `Int32/none`, 43.2 `Int32/Delta`, 51.2
  `Int32/T64`, 56.8 `UInt32/T64`, 58.0 `String/none`; filter 33.2
  `LowCardinality(String)`, 58.8 `UInt32/none`, 65.2 `UInt32/Delta` and
  `Int32/none`, 65.5 `Int32/Delta`, 69.0 `Int32/T64`, 85.8 `UInt32/T64`,
  90.2 `String/none`. **T64 is the slowest integer encoding as well as
  the largest**, and `LowCardinality(String)` is the fastest and smallest
  variant in the suite.
- **Cold and warm do not separate**, and not because the difference is
  small: cold is *faster* server-side. Over the 144 matched (table, query,
  qcc) pairs, cold minus warm server p50 has median -8.5 ms and mean
  -12.5 ms, and cold is the faster of the pair in 103 of 144. For
  example, `Int32/none/minmax` full scan: 61.0 ms warm, 30.5 ms cold;
  `Int32/Delta/minmax`: 74.0 warm, 40.5 cold. The profile events show
  why the drops change nothing: in both modes the whole column
  (`ReadCompressedBytes` about 23.8 MB) is served from the Docker VM's
  page cache (`ThreadPoolReaderPageCacheHit` is present, `OSReadBytes`
  is absent), and `SYSTEM DROP MARK CACHE` only costs one or two
  `MarkCacheMisses` per cold sample (1 in 1080 samples, 2 in 360; 0 warm). A 25 MB column cannot be
  made cold by dropping ClickHouse's own caches; it would need the OS
  page cache dropped inside the VM or a column larger than memory. The
  client-side picture is the reverse (median `client_p50` over the CSV
  rows: 99.6 ms cold against 76.3 warm), so the two clocks disagree on the sign; neither
  measures what `cache_mode` was meant to measure here.
- **The query condition cache never hits.** All 960 samples with
  `use_query_condition_cache = 1` on a filter (96 queries, 24 tables x 2
  filters x 2 cache modes, x 10 samples) report
  `QueryConditionCacheMisses = 1` and no `QueryConditionCacheHits`, warm
  or cold, though the same predicate runs ten times in a row on the same
  table. Task 7 saw hits on a primary-key range filter that excludes
  granules; here no granule is ever excluded, and evidently the cache
  records nothing when the predicate is true somewhere in every mark. The
  `qcc` axis is inert for this suite too: qcc-on minus qcc-off server p50
  has median +0.25 ms over the 144 matched pairs.
- Client time again has a floor near 50 ms: the `LowCardinality(String)`
  full scan is 27.5 ms on the server and 70.9 ms at the client.

## What this says about the motivating question

The motivating question (CLAUDE.md, spec §1) is whether the documented
advice of Decimal64 with T64 for fixed-precision numerics holds up, and
in particular whether T64 ever beats Decimal32 or Decimal64 with
compression alone, and how Decimal32 compares with Decimal64 and Float64
under matching codecs. After ruling R28 the float suite has the cells the
first half needs (`none` as the compression-alone baseline and `T64` on
Decimal64); the second half is still half-blind, because every Decimal32
cell skips (ruling R29). Every figure below is a `float_codecs.csv` row,
named by type / codec / compression / pk; the int suite's T64 rows are
`int_encodings.csv`.

**Does T64 beat Decimal64 with compression alone?** Yes, in generation
order, and it is the best Decimal64 codec there:

- Decimal64 / T64 / LZ4 / pk=0 is 37 440 586 bytes against 57 126 485
  for Decimal64 / none / LZ4 / pk=0, a 34.5 percent saving. It also
  beats Delta (59 572 803, which is a net loss against no codec) and
  DoubleDelta (46 504 234).
- Under ZSTD the compression does most of the work on its own: Decimal64 /
  T64 / ZSTD / pk=0 is 37 589 133 against 39 872 280 for none / ZSTD,
  5.7 percent. T64 with LZ4 lands within 0.4 percent of T64 with ZSTD,
  so T64 + LZ4 gets ZSTD-level size at LZ4 cost.
- T64 does not cost scan time; it saves it. Full-scan server p50 at
  `pk=0` is 47.0 ms for T64 LZ4 against 61.5 for none LZ4, 82.5 for
  Delta LZ4 and 142.5 for none ZSTD; with ZSTD, 66.5 against 142.5. The
  0.20 filter tells the same story (57.5 against 85.0 and 141.5).

Sorted by value (`pk=1`) the answer reverses under ZSTD: Decimal64 /
Delta / ZSTD / pk=1 is the smallest column in the suite at 8 734 118
bytes, none / ZSTD is 10 608 286, and T64 / ZSTD is 14 513 433, 37 percent
larger than no codec at all. Under LZ4 T64 still wins (16 254 867 against
39 236 721 for none and 23 270 370 for Delta). So the documented pairing
holds for unsorted or LZ4 columns and fails for a value-sorted column
under ZSTD, where Delta is the codec to use.

**How does Decimal64 compare with Float64 under matching codecs?**
Smaller in every one of the twelve matched pairs (three codecs, two
compressions, two orders), and slower to scan in ten of them. With no codec at `pk=0`: 57 126 485 against 80 343 339
bytes (LZ4, 28.9 percent smaller) and 39 872 280 against 75 353 869
(ZSTD, 47.1 percent). With Delta LZ4, 59 572 803 against 80 372 603;
DoubleDelta ZSTD, 46 333 455 against 82 314 793. Decimal64 with T64, which
Float64 cannot use, is 37 440 586 bytes, less than half of the smallest
Float64 column in generation order (75 353 869, none ZSTD). Float64 at 8
decimal places is essentially incompressible in generation order: its
best ratio is 1.06, and Gorilla, the documented float codec, gives 0.98
and 1.02. On time, full-scan server p50 at `pk=0` is Float64 31.0 ms
against Decimal64 61.5 (none LZ4), 44.5 against 82.5 (Delta LZ4), 54.0
against 126.5 (Delta ZSTD), 62.0 against 142.5 (none ZSTD); the two
exceptions are DoubleDelta ZSTD at `pk=0` (Float64 208.5, Decimal64
103.0) and none LZ4 at `pk=1` (49.0 against 42.0, within noise). The two
T64 cells, which have no Float64 partner, put Decimal64 (47.0 LZ4, 66.5
ZSTD) within noise of the Float64 LZ4 scans. The full scan sums `toFloat64(v)`, so every
Decimal64 scan carries a conversion the Float64 scan does not.

**How does Decimal32 compare?** Not measured. All twenty Decimal32(8)
cells were skipped because one row of the ten million sits at exactly
10.0, the range bound, and `Decimal32(8)` tops out at 9.99999999. That is
the suite's `value_range: [-1, 10]` at `decimal_places: 8` meeting a
lognormal tail clipped to the bound, not a server limit; the validator is
right and the skip is the finding (ruling R29). The Plan 1 measurement
that Decimal32 is the cheaper width for this data (deferred-findings
note: Delta LZ4 ratio 0.65 between the widths at 100 000 rows) remains
the only Decimal32 number the project has, and it was taken before the
filter query read the column. Decimal32 evidence needs either a bound of
9.99999999 (only 459 of ten million rows exceed 9.0, so the lognormal is
essentially untouched) or 7 decimal places; both change what use case 1
is, so the decision is the owner's, and the range and scale stand.

**T64 on plain integers** (int suite) is the opposite result: on 1000
distinct 7-digit values at 4 bits of entropy, T64 is the largest and
slowest of the three integer codecs, 25 707 850 bytes against 23 681 004
for LZ4 alone and 22 161 670 for Delta, with full-scan medians of 51 to
57 ms against 34 to 43. A reading consistent with both results, not a
measurement: T64 stores a block at the bit width its values span, and the
Decimal64 column's scaled integers (0.05 to 10.0 at scale 8, so under
2^30 in a 64-bit type) leave it about half the bits to drop per value,
while the 7-digit integers in Int32 (under 2^24) leave it a quarter, and
LZ4 alone already finds the 4-byte repeats of a 1000-value column. On
that reading "T64 for fixed-precision numerics" is really "T64 for a wide
type whose values are narrow", which Decimal64 at scale 8 is and Int32
holding 7-digit values is not. Testing it needs a decimal column that
uses more of its width, which the current suites do not have.

**Sort order dominates all of it.** The same Decimal64 Delta ZSTD column
is 8.7 MB sorted by value and 50.3 MB in generation order, and the same
Float64 no-codec LZ4 column 60.9 MB against 80.3 MB, gaps bigger than any
codec produces in either order.

**What the runs say about the harness.** It ran both suites end to end
with no crash, resumption or query error, at 4 and 15 minutes for the
first runs and 2.5 minutes and 40 seconds for the reruns, which reused
every already-loaded table (ruling R30) and loaded only the 12 new float
cells; the filters read the column as intended; the skip-and-record path
handled 40 skips with verbatim server messages, six of them recorded as
table errors in the first int run before ruling R27 taught the validator
the `LowCardinality` rejection. The measurements also show three axes
that are inert for these suites on this machine (`cache_mode`,
`query_condition_cache`, `skip_index`), and a client-time floor of about
50 ms through Docker Desktop that makes `server_p50` the column to rank
by. Every comparison above rests on a single run of ten samples on a
shared laptop; the int suite's three-way repeats show single p50s
disagreeing by up to a factor of two, so the size figures are solid and
the time figures need a repeat run on a quiet machine before any gap
under 2x is trusted. The T64-versus-Delta-versus-none scan times above
are all within that band except none ZSTD (142.5 ms), which is not.
