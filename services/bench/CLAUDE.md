# Empirical benchmark (`services/bench`, package `chbench`)

Component 3 of the toolkit (objectives: the root CLAUDE.md). Empirically tests storage and query optimisations: generate
synthetic single-column datasets with controllable statistics (or read a real Arrow IPC column), load them under a grid
of table parameters (type, codec, compression, primary key, projection, skip index), run query grids repeatedly,
persist per-sample results in SQLite and report summary statistics. Three engines: ClickHouse (a server) and the
in-process engines Polars and DataFusion (worker processes, Arrow IPC or in memory). The package keeps the name
`chbench` although it is no longer ClickHouse-only; whether to refactor it into an engine-neutral core is ToDo 27.
Motivation: documented advice (e.g. Decimal64 + T64 for fixed-precision numerics) has proven unreliable in practice;
this produces evidence instead. Paths below are relative to `services/bench/`.

# Design
- The generator (`synthdata`, `services/datagen/`) is best-effort: it records *achieved* metrics in a manifest and warns on misses; only the `DatasetSpec` layer
  raises (`SpecError`: `DatasetSpec` validation, or a real-data file that cannot be read or has changed since its spec).
- Two stages: a multiset (values + frequencies, solved for cardinality ratio, entropy or index of dispersion), then a
  sequence (the order, for a runs z-score or an inter-arrival CV), then a bounded refine pass for the inter-arrival CV
  only — the runs z-score is exact by construction. When both are targeted, `runs_z` is honoured and the achieved CV is
  recorded without its tolerance.
- One data column per table, plus `row_id UInt64`. Parts are forced Wide.
- Invalid grid cells (a codec not applicable, a type that cannot hold the range) are skipped and recorded with the
  server's reason, never coerced. `LowCardinality` is a skip on every type but `String` / `FixedString`.
- Default scale for suites comparing query time: 10 000 000 rows.
- Spike scripts in `spikes/` are kept permanently; findings and gotchas go in `spikes/FINDINGS.md`.

# Environment
- Python: the project's p312 env (root CLAUDE.md). Tests: `python scripts/check.py test-bench` from the repository root,
  or `python -m pytest` from `services/bench/` (`pytest.ini` deselects `slow`; `-m slow` runs the five full-scale
  tests). Tests marked `clickhouse` need the server and skip without it.
- ClickHouse in Docker Desktop: container `chbench-ch`, image pinned in `docs/engines/versions.yaml` (tag + digest;
  `run` warns when the server's version differs), HTTP port 8123 (`docker start chbench-ch`; to recreate it:
  `docker run -d --name chbench-ch -p 8123:8123 -e CLICKHOUSE_USER=CLICKHOUSE_USER -e CLICKHOUSE_PASSWORD=CLICKHOUSE_PASSWORD -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 <image from versions.yaml>`).
  After an upgrade: `docs/engines/README.md`. Credentials are the literal strings user `CLICKHOUSE_USER`, password
  `CLICKHOUSE_PASSWORD`. The harness reads `CH_HOST`, `CH_PORT`, `CH_USER`, `CH_PASSWORD`, `CH_DATABASE`, defaulting to
  these.
- Benchmark tables live in database `bench`. Results in `results.db` (SQLite) and cached datasets in `datasets/` (Arrow
  IPC, the exact Arrow type, no re-encoding), both relative to the working directory (defaults of `--store` /
  `--cache-dir`) and git-ignored under `services/bench/`; the committed evidence is
  the CSV exports and notes under `docs/results/`.
- No Rust: hot numeric paths (metrics, refine) are pure numpy behind small boundaries; profile before proposing an
  extension (a numpy vectorisation beat the Rust case once already — FINDINGS §8).

# Running a suite
From `services/bench/`:
```
python -m chbench.cli run suites/float_codecs.yaml [--dry-run] [--force] [--samples-add N] [--explain]
        [--allow-unsolved] [--ignore-unfinished] [--store PATH] [--cache-dir PATH]
python -m chbench.cli report --suite float_codecs [--csv docs/results/float_codecs.csv]
        [--where "codec=T64,kind=filter"] [--baseline "codec=none"] [--store PATH]
python -m chbench.cli datasets [--metrics] [--cache-dir PATH]
python -m chbench.cli clean --suite float_codecs [--store PATH] [--cache-dir PATH]
```
`run` generates or loads the dataset, expands the grid (skipped cells printed with their reason), loads every table,
runs the sampling rounds (Measurement notes) and prints the report; `--dry-run` prints the grid and stops. A dataset
whose solve reports `invalid_bracket` is refused (`UnsolvedDatasetError`) unless `--allow-unsolved`. Rerunning resumes: queries already
holding their sample count are skipped, and a table whose queries are all held is neither dropped nor reloaded;
`--force` or a sample shortfall (`--samples-add`, or a new query) loads it. `report` re-reads SQLite only; `--csv`
exports the same rows. `--baseline "axis=value,..."` adds paired per-round ratios (`cpu_ratio`, `server_ratio`, each
with a 95% interval, `n_rounds`, `clocks_disagree`) against the row matching it within each group of rows equal on
every other axis and the same query shape; a baseline matching no row is an error. A baseline over several axes frees
each of them: `--baseline "ch_type=String,codec=none"` pairs `LowCardinality(String)` with `String`. `clean` drops a
suite's ClickHouse tables from `bench` and deletes its IPC files (`<cache_dir>/ipc/`; the store keeps its rows; no server
is needed without ClickHouse tables). IPC files are keyed by table id, not per suite: cleaning one suite can delete a file
another suite with the same variant uses (regenerated on that suite's next run). The `EXPLAIN` plan of every query is stored; `--explain` only
prints it. `run` refuses to start while the store holds an unfinished run of the same suite (a process that died without stamping `finished_at`); `--ignore-unfinished` overrides that. The store
carries a schema version (SQLite `user_version`, now 3: the engine columns): `run` refuses an older or newer store (`StoreVersionError`,
name another `--store`); `report` reads an older one, its missing columns NULL. A suite names at least one of
`clickhouse:`, `polars:`, `datafusion:`; `run` prints each engine's version and thread count and warns when one differs
from `docs/engines/versions.yaml`. The piped report is 600 columns wide (`orchestrate._PIPED_WIDTH`); the widest reports
(~670) still truncate: use `--csv`. `--baseline engine=…` is refused (a baseline pairs within one engine; use
`--where engine=<name>`). A suite
whose dataset type cannot feed its `type` axis (a Decimal axis without `decimal_places`, a `list<T>` dataset, a numeric
axis over a string or date dataset) is rejected at parse time.

# Suite vocabulary
- **Real-data datasets:** `dataset: {path: <file>.arrow, column: <name>}` (path relative to the suite file,
  `decimal_places` optional) reads the column as stored; nothing is generated or cached; `rows`, `arrow_type`,
  `null_count`, `decimal_places`, `value_range` are measured from the file (`synthdata/column/realdata.py`), and `dataset_id`
  follows the file's content. This is what `turbo_parakeet.recommenders.to_chbench` writes. `binary` (path datasets
  only) loads into `String` as it is; its literals are UTF-8 or `unhex`.
- **Types** (measured on ClickHouse 26.8.31.2, `spikes/spike_ch_vocabulary.py`, FINDINGS §9): `Int8/16/32/64`,
  `UInt8/16/32/64`, `Float32/64`, `Bool`, `Decimal32/64/128`, `Date`, `Date32`, `DateTime`, `DateTime64`,
  `FixedString`, `Enum8/16`, `Time64`, `String`. Parameters come from the dataset (`validate.resolve_type`: decimal
  scale, `DateTime64(p, 'tz')` from the timestamp unit and zone, `FixedString(n)`, sorted `Enum` values); `arrow_type`
  includes `timestamp[unit, tz=zone]`, `fixed_size_binary(n)` and `binary` for path datasets. Skip rules carry the server's
  messages (GCD, Delta / DoubleDelta / Gorilla sizes, T64, FPC, LowCardinality, `bloom_filter` off Decimals, Date32 and
  Time64).
- **Codecs:** `none`, `T64`, `Delta`, `DoubleDelta`, `Gorilla`, `FPC`, `GCD` and the `LowCardinality` type wrapper
  (`validate.KNOWN_CODECS`), then the `compression` axis (`LZ4`, `ZSTD` or `ZSTD(N)`, N 1–22).
  GCD on disk keeps the column width and saves 13–45% on multiples of 1000.
- **Skip indexes:** `skip_index` values are `<index>[:<G>]`, G = GRANULARITY in {1, 2, 4, 8} (default 1), index
  `minmax`, `set(N)`, `bloom_filter` (ClickHouse's default fpr) or `bloom_filter(F)` with F in {0.001, 0.01, 0.025, 0.05}; `none` leaves the table unindexed.
  `storage_stats` reads `system.data_skipping_indices`: the store and the report / CSV carry `index_compressed_bytes` /
  `index_uncompressed_bytes` (the store also `index_marks_bytes`). Sizes (FINDINGS §10): a `bloom_filter` block is sized
  by its distinct values — ceil(D × b / 8) + 1–3 bytes, b = 15 / 10 / 8 / 7 bits for fpr 0.001 / 0.01 / 0.025 / 0.05 —
  not by its rows; `minmax` is 16 B and `set` 8 B + 8 B per value per Int64 block (8 B when it overflows).

# In-process engines (Polars, DataFusion)
- **Blocks:** `polars:` / `datafusion:` with `type`, `storage` (`memory`, `ipc`), `compression` (`none`, `LZ4_FRAME`,
  `ZSTD` = `ZSTD(1)`, `ZSTD(N)`), optional `threads` (default: the Docker VM's CPU count, else the host's) and
  optional `ipc_batch_rows` (rows per IPC record batch, default 122 880, stored as `"122880"`, or `whole`, one batch; it
  crosses `storage: ipc` cells only, and memory cells store NULL) (`harness/inproc_types.py`). Polars types:
  `Int8…UInt64`, `Float32/64`, `Decimal(p, s)`, `Boolean`, `Date`, `Datetime(unit[, zone])`, `Time`, `String`, `Binary`,
  `Categorical(UInt8|UInt16|UInt32)` (bare = UInt32; always named categories, in a namespace per table), `Enum`
  (first-seen categories, or `Enum(categories=[…])`); the recommenders' `str(dtype)` spellings are accepted and labelled
  in the short form. DataFusion types: pyarrow type strings as pyarrow prints them, dictionaries of string / binary
  values included; run-end encoding is refused. An in-process table id hashes the resolved thread count (the Docker VM's
  CPU count by default): a run on a VM with another CPU count makes new tables.
- **Skipped cells:** `memory` with compression; a cast the engine refuses (a Categorical or dictionary width too narrow
  for the distinct values: Polars' UInt8 holds 255); a cast that does not round-trip row by row; a type the dataset
  cannot feed (parse time, `inproc_types.FEEDS`).
- **Execution** (`harness/engines/`): one worker process per engine and thread count (spawn; `POLARS_MAX_THREADS` set in
  the parent around the start; DataFusion `with_target_partitions`). `memory` tables stay resident (DataFusion: `threads`
  partitions, dealt round-robin in batches of min(65 536, ⌈rows / threads⌉) rows so every thread gets one); a worker
  holds the dataset array plus every memory-resident variant's table at once, so size memory cells for the machine (16 GB
  host, Docker VM 8 GB at 10M rows); `memory_bytes` and RSS are not comparable across engines. `ipc` tables are files
  `<cache_dir>/ipc/<table_id>.arrow` in record batches of `ipc_batch_rows` rows (one batch for `whole`; written by
  pyarrow's writer, which takes a ZSTD level: Polars' layout or DataFusion's Arrow type), reused on resume when their
  size matches the store's, and opened per query. Size predictions (`to_chbench`, the recommenders) describe one batch,
  so a batched file's `ipc_bytes` includes per-batch overhead (record-batch metadata, per-batch compression framing)
  they omit (ToDo 26). Why batched by default: on a 50M-row Int64 column one whole-file batch was 3–4× slower to query
  (Polars 0.44 vs 0.12 s uncompressed, 0.90 vs 0.19 s LZ4; DataFusion 0.45 vs 0.15 s, 0.74 vs 0.20 s LZ4, 1.01 vs
  0.34 s ZSTD(1)) and cost a full private copy in Polars, ~2× peak memory in DataFusion with LZ4; realistic writers
  produce batched files. Queries are chosen once from the data for every engine (`queries.choose_queries`); DataFusion
  runs SQL with typed literals, Polars lazy expressions; both aggregate `count` and a total (bytes for strings, seconds
  for timestamps) like ClickHouse's full scan. DataFusion 55 rejects `octet_length` on Binary / LargeBinary, so the
  binary total is `sum(octet_length(encode(v, 'hex')) / 2)`.
- **Measurement:** `process_time_ns` (User + System of the worker's threads) advances in 15.625 ms steps on Windows, too
  coarse for one query, so a sample is a batch of k runs back to back: k is calibrated per (table, query) in the warm-up
  (the smallest power of two whose batch lasts ≥ 250 ms of wall, at most 1024; `engines/inproc.py` `TARGET_BATCH_NS`,
  `MAX_REPEATS`) and fixed for the run's timed rounds, so per-round pairing holds; `cpu_us` needs batches ≥ 250 ms.
  Wall (`server_ms`, `perf_counter_ns`) and CPU (`cpu_us`) are taken around the whole batch, the collector paused, and
  stored per query (batch totals / k); the sample records `repeats` (k) and `repeats_capped` (k hit the cap: a warm-up
  warning). A later run (`--samples-add`) recalibrates. `memory_bytes` is the worker's RSS peak over the batch above its
  pre-batch RSS (sampled about every millisecond); it can read 0 after the warm-up (the allocator reuses pages the worker
  holds), so it is the growth of resident memory, not the query's allocation. `client_ms` adds the pipe round trip (per
  query). Each query's warm-up answer must equal a pyarrow reference (count exactly, totals within 1e-9 of the summed
  magnitude) or the query is `error` with both answers. A worker that raises, dies or outlives its call timeout
  (`engines/inproc.TIMEOUTS_S`) comes back as per-query `error` / `interrupted` statuses, not an exception: read the
  statuses in a results note. Store columns: `tables.engine`, `storage`, `threads`, `ipc_batch_rows`, `ipc_bytes` (the
  whole file), `memory_bytes` (resident; Polars' can include its categories, ToDo 11); `samples.repeats`,
  `repeats_capped`; the run row records every engine version. Report axes `engine`, `storage`, `threads`,
  `ipc_batch_rows` (also in `--where` / `--baseline`), columns `repeats` (the largest k) and `repeats_capped`; a
  `--baseline` pairs rows within one engine.

# Measurement notes
- Sampling: every table is loaded first, then one untimed warm-up round, then `samples` timed rounds; a round runs
  every (table, query) pair once in an order shuffled by `(seed, round_no)`, with `use_query_cache = 0` and
  `use_query_condition_cache = 0`, and reads `system.query_log` once. `--samples-add N` adds N rounds over every pair,
  so samples stay paired by `round_no`; a query added later is paired only over the rounds it shares (`run` warns).
  Ctrl-C marks every query short of its samples `interrupted`.
- Resolution: on replicates (near-identical tables, 6 rows per median, host CPU 24-36%, Balanced power) the median
  paired CPU-ratio interval was 0.18 / 0.08 / 0.06 wide at 10 / 20 / 30 rounds; the codec suites measured 0.33 at 10
  rounds (2026-10-29, High performance). Plan with the wider figure, and pick `--samples-add` so that half the width
  is below the smallest difference the question needs (`docs/results/2026-10-30-resolution-vs-rounds.md`).
- String full scan: it disables `optimize_functions_to_subcolumns` so `sum(length(v))` reads the characters, not just
  the size subcolumn (FINDINGS §12). A store holding String full-scan samples from before this change must be rerun
  with `--force` or a fresh `--store`, because `query_id` does not hash the SELECT list.
- Time metrics: server CPU (`cpu_us` = User + System microseconds from ProfileEvents) is primary, server wall
  (`server_ms`) beside it; client time has a floor of ≈ 50 ms through Docker Desktop's HTTP path. `report` gives each
  median a distribution-free 95% interval (order statistics; none below 6 samples, `harness/stats.py`).
- Compare variants by the paired `--baseline` ratios, not by separate medians: unpaired p50s of identical tables
  disagreed by up to 2× before interleaving. Quietness (host CPU over 3 s, Docker CPUs and memory, power scheme,
  ClickHouse `LoadAverage1`, active merges) is recorded on the run row before and after the rounds; `run` warns when
  host CPU is over 10% or merges are active. `suites/replicates.yaml` (three tables no index can tell apart) checks
  the protocol: every pairwise ratio interval should contain 1.
- Retired axes (ToDo 3): `cache_mode` (the column is served from the Docker VM's page cache in both modes) and
  `query_condition_cache`; a suite naming either fails to parse.
- The `primary_key` axis changes the physical order (`ORDER BY v` vs `ORDER BY row_id`), and order dominates codec
  (Decimal64 Delta ZSTD: 8.7 MB sorted vs 50.3 MB unsorted): compare codecs within one `primary_key` value.
- `minmax` / `set` on `LowCardinality(String)` cost ≈ 3.3 ns × rows × insert-block rows / index_granularity at insert
  (server-side; FINDINGS §11): `load_table` sends ≤ 1 048 576-row batches, leaving ≈ 4 s at 10M rows; compare insert
  times only within one batching. Inserts under 10 MiB run as async inserts (`AsyncInsertFlush` rows in `query_log`).

# Layout
```
synthdata          (services/datagen) the generator: the column pipeline (spec, multiset, sequence, refine, solve, metrics, wrap, realdata) and its cache
chbench/harness/   suite, inproc_types, validate, ddl, client, loader, queries, runner, store, report, orchestrate, stats, quiet, versions
chbench/harness/engines/  base (round protocol), clickhouse, inproc (worker proxy), worker (child loop), polars_engine, datafusion_engine
chbench/cli.py     chbench run | report | datasets | clean
suites/            YAML suites (example use cases)
spikes/            feasibility scripts + FINDINGS.md (dated record)
docs/results/      CSV exports and result notes (dated records)
tests/             pytest; ClickHouse integration tests marked `clickhouse`, full-scale ones `slow`
notebooks/         results_explorer.ipynb
```

# Conventions
- TDD: write the failing test first.
- Keep hot numeric paths (metrics, refine) as pure numpy functions behind small boundaries.

# State
- The example suites `replicates`, `int_encodings`, `float_codecs` and `decimal32_codecs` ran at 10M rows under
  interleaved rounds (`docs/results/2026-10-29-interleaved-rounds.md`; sizes as in `2026-09-17-first-runs.md`, whose
  timing claims it supersedes). The replicates check passes: every pairwise ratio interval contains 1. T64 is the
  smallest Decimal64 codec in generation order and the cheapest scan only under ZSTD; ZSTD never costs less CPU than
  LZ4; Decimal32 scans at 0.56–0.74× Decimal64 (median CPU ratio) but is not always smaller: sorted under ZSTD(1)–(13)
  with no codec it is 1.99–2.09×. Adjacent sorted values share 3 bytes in Decimal32 and 7 in Decimal64 (its constant high
  bytes); the gap closes when ZSTD's minimum match is 3 (btopt and above, level ≥ 14 at 64 KiB blocks). FINDINGS §13;
  the size emulator reproduces every LZ4 / ZSTD / Delta cell byte for byte.
- The evaluation loop (`services/evaluate`) runs its suites here: the ClickHouse size emulator (fixed-width types
  byte-exact, `LowCardinality(String)` median 0.8%, String not yet — ToDo 9) and the skip-index predictions (minmax /
  set skip shares and index bytes exact on non-null columns, Nullable not yet — ToDo 10; bloom within 0.009–0.041 per
  query and 0.5% in bytes). Binary columns load as String (UTF-8 literals quoted, others `unhex`). At 1M rows the
  paired, Bonferroni-adjusted CPU ratios resolve 103 of the loop's 586 queries
  (`services/evaluate/docs/results/2026-10-31-follow-ups.md`).
- The in-process engines are in (ToDo 5); `suites/inproc_strings.yaml` (10M rows) has not been run at scale yet. The
  first run is an open follow-up: `python -m chbench.cli run suites/inproc_strings.yaml` from `services/bench/` on a quiet
  machine (≈ 30–45 min: 45 cells × 4 queries × 20 samples at ≥ 250 ms per batch, plus calibration and IPC writes), then
  a dated results note in `docs/results/`.
- Open: ToDo 1 (generator items), ToDo 26 (the loop's in-process scores), ToDo 27 (an engine-neutral core).
