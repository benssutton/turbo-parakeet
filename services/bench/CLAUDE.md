# Empirical benchmark (`services/bench`, package `chbench`)

Component 3 of the toolkit (objectives: the root CLAUDE.md). Empirically tests storage and query optimisations: generate
synthetic single-column datasets with controllable statistics (or read a real Arrow IPC column), load them under a grid
of table parameters (type, codec, compression, primary key, projection, skip index), run query grids repeatedly,
persist per-sample results in SQLite and report summary statistics. ClickHouse today; Polars and DataFusion are ToDo 5.
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
- ClickHouse 26.8 in Docker Desktop: container `chbench-ch`, image `clickhouse/clickhouse-server:26.8`, HTTP port 8123
  (`docker start chbench-ch`). Credentials are the literal strings user `CLICKHOUSE_USER`, password
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
python -m chbench.cli clean --suite float_codecs [--store PATH]
```
`run` generates or loads the dataset, expands the grid (skipped cells printed with their reason), loads every table,
runs the sampling rounds (Measurement notes) and prints the report; `--dry-run` prints the grid and stops. A dataset
whose solve reports `invalid_bracket` is refused (`UnsolvedDatasetError`) unless `--allow-unsolved`. Rerunning resumes: queries already
holding their sample count are skipped, and a table whose queries are all held is neither dropped nor reloaded;
`--force` or a sample shortfall (`--samples-add`, or a new query) loads it. `report` re-reads SQLite only; `--csv`
exports the same rows. `--baseline "axis=value,..."` adds paired per-round ratios (`cpu_ratio`, `server_ratio`, each
with a 95% interval, `n_rounds`, `clocks_disagree`) against the row matching it within each group of rows equal on
every other axis and the same query shape; a baseline matching no row is an error. `clean` drops a suite's tables from `bench` (the store keeps its rows). The `EXPLAIN` plan of
every query is stored; `--explain` only prints it. `run` refuses to start while the store holds an unfinished run of
the same suite (a process that died without stamping `finished_at`); `--ignore-unfinished` overrides that. The store
carries a schema version (SQLite `user_version`, now 2): `run` refuses an older or newer store (`StoreVersionError`,
name another `--store`); `report` reads an older one, its missing columns NULL. A suite
whose dataset type cannot feed its `type` axis (a Decimal axis without `decimal_places`, a `list<T>` dataset, a numeric
axis over a string or date dataset) is rejected at parse time.

# Suite vocabulary
- **Real-data datasets:** `dataset: {path: <file>.arrow, column: <name>}` (path relative to the suite file,
  `decimal_places` optional) reads the column as stored; nothing is generated or cached; `rows`, `arrow_type`,
  `null_count`, `decimal_places`, `value_range` are measured from the file (`synthdata/column/realdata.py`), and `dataset_id`
  follows the file's content. This is what `turbo_parakeet.recommenders.to_chbench` writes.
- **Types** (measured on ClickHouse 26.8.31.2, `spikes/spike_ch_vocabulary.py`, FINDINGS §9): `Int8/16/32/64`,
  `UInt8/16/32/64`, `Float32/64`, `Bool`, `Decimal32/64/128`, `Date`, `Date32`, `DateTime`, `DateTime64`,
  `FixedString`, `Enum8/16`, `Time64`, `String`. Parameters come from the dataset (`validate.resolve_type`: decimal
  scale, `DateTime64(p, 'tz')` from the timestamp unit and zone, `FixedString(n)`, sorted `Enum` values); `arrow_type`
  includes `timestamp[unit, tz=zone]` and `fixed_size_binary(n)` for path datasets. Skip rules carry the server's
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

# Measurement notes
- Sampling: every table is loaded first, then one untimed warm-up round, then `samples` timed rounds; a round runs
  every (table, query) pair once in an order shuffled by `(seed, round_no)`, with `use_query_cache = 0` and
  `use_query_condition_cache = 0`, and reads `system.query_log` once. `--samples-add N` adds N rounds over every pair,
  so samples stay paired by `round_no`; a query added later is paired only over the rounds it shares (`run` warns).
  Ctrl-C marks every query short of its samples `interrupted`.
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
chbench/harness/   suite, validate, ddl, client, loader, queries, runner, store, report, orchestrate, stats, quiet
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
  LZ4; Decimal32 scans at 0.56–0.74× Decimal64 (median CPU ratio) but is not always smaller (2.09× sorted under ZSTD with no codec).
- The evaluation loop (`services/evaluate`) runs its suites here: the ClickHouse size emulator (fixed-width types
  byte-exact, `LowCardinality(String)` median 0.8%, String not yet — ToDo 9) and the skip-index predictions (minmax /
  set skip shares and index bytes exact on non-null columns, Nullable not yet — ToDo 10; bloom within 0.009–0.041 per
  query and 0.5% in bytes).
- Open: ToDo 3's follow-ups (String full scan, resolution at 10 samples), ToDo 1 (generator items), ToDo 5
  (engines).
