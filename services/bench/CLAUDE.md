# Empirical benchmark (`services/bench`, package `chbench`)

Component 3 of the toolkit (objectives: the root CLAUDE.md). Empirically tests storage and query optimisations: generate
synthetic single-column datasets with controllable statistics (or read a real Arrow IPC column), load them under a grid
of table parameters (type, codec, compression, primary key, projection, skip index), run query grids repeatedly,
persist per-sample results in SQLite and report summary statistics. ClickHouse today; Polars and DataFusion are ToDo 5.
Motivation: documented advice (e.g. Decimal64 + T64 for fixed-precision numerics) has proven unreliable in practice;
this produces evidence instead. Paths below are relative to `services/bench/`.

# Design
- The generator is best-effort: it records *achieved* metrics in a manifest and warns on misses; only the spec layer
  raises (`SpecError`: `DatasetSpec` validation, or a real-data file that cannot be read or has changed since its spec).
- Two stages: a multiset (values + frequencies, solved for cardinality ratio, entropy or index of dispersion), then a
  sequence (the order, for a runs z-score or an inter-arrival CV), then a bounded refine pass for the inter-arrival CV
  only — the runs z-score is exact by construction. When both are targeted, `runs_z` is honoured and the achieved CV is
  recorded without its tolerance.
- One data column per table, plus `row_id UInt64`. Parts are forced Wide.
- Invalid grid cells (a codec not applicable, a type that cannot hold the range) are skipped and recorded with the
  server's reason, never coerced. `LowCardinality` is a skip on every type but `String` / `FixedString`.
- Cache mode is a per-suite axis: `warm` and/or `cold`.
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
        [--ignore-unfinished] [--store PATH] [--cache-dir PATH]
python -m chbench.cli report --suite float_codecs [--csv docs/results/float_codecs.csv]
        [--where "codec=T64,cache_mode=warm"] [--store PATH]
python -m chbench.cli datasets [--metrics] [--cache-dir PATH]
python -m chbench.cli clean --suite float_codecs [--store PATH]
```
`run` generates or loads the dataset, expands the grid (skipped cells printed with their reason), loads each table,
runs every query and prints the report; `--dry-run` prints the grid and stops. Rerunning resumes: queries already
holding their sample count are skipped, and a table whose queries are all held is neither dropped nor reloaded;
`--force` or a sample shortfall (`--samples-add`, or a new query) loads it. `report` re-reads SQLite only; `--csv`
exports the same rows. `clean` drops a suite's tables from `bench` (the store keeps its rows). The `EXPLAIN` plan of
every query is stored; `--explain` only prints it. `run` refuses to start while the store holds an unfinished run of
the same suite (a process that died without stamping `finished_at`); `--ignore-unfinished` overrides that. A suite
whose dataset type cannot feed its `type` axis (a Decimal axis without `decimal_places`, a `list<T>` dataset, a numeric
axis over a string or date dataset) is rejected at parse time.

# Suite vocabulary
- **Real-data datasets:** `dataset: {path: <file>.arrow, column: <name>}` (path relative to the suite file,
  `decimal_places` optional) reads the column as stored; nothing is generated or cached; `rows`, `arrow_type`,
  `null_count`, `decimal_places`, `value_range` are measured from the file (`datagen/realdata.py`), and `dataset_id`
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
  `minmax`, `set(N)` or `bloom_filter(F)` with F in {0.001, 0.01, 0.025, 0.05}; `none` leaves the table unindexed.
  `storage_stats` reads `system.data_skipping_indices`: the store and the report / CSV carry `index_compressed_bytes` /
  `index_uncompressed_bytes` (the store also `index_marks_bytes`). Sizes (FINDINGS §10): a `bloom_filter` block is sized
  by its distinct values — ceil(D × b / 8) + 1–3 bytes, b = 15 / 10 / 8 / 7 bits for fpr 0.001 / 0.01 / 0.025 / 0.05 —
  not by its rows; `minmax` is 16 B and `set` 8 B + 8 B per value per Int64 block.

# Measurement notes
- Rank by `server_p50` under ≈ 100 ms: client time has a floor of ≈ 50 ms through Docker Desktop's HTTP path.
- Noise between identical tables reaches 2×; ten samples on a shared laptop cannot resolve smaller differences (ToDo 3).
- The `primary_key` axis changes the physical order (`ORDER BY v` vs `ORDER BY row_id`), and order dominates codec
  (Decimal64 Delta ZSTD: 8.7 MB sorted vs 50.3 MB unsorted): compare codecs within one `primary_key` value.

# Layout
```
chbench/datagen/   spec, multiset, sequence, refine, solve, metrics, wrap, cache, realdata
chbench/harness/   suite, validate, ddl, client, loader, queries, runner, store, report, orchestrate
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
- Example suites `float_codecs` and `int_encodings` ran end to end at 10M rows (`docs/results/2026-09-17-first-runs.md`):
  T64 is the best Decimal64 codec in generation order (37.4 MB vs 57.1 MB LZ4 alone), Delta + ZSTD wins sorted by value;
  Decimal64 is smaller than Float64 in all twelve matched cells and slower to scan in ten; Float64 at 8 dp is
  incompressible unsorted.
- turbo-parakeet's calibrations run here: the ClickHouse size emulator (fixed-width types within max(2% of measured,
  256 bytes), `LowCardinality(String)` within 2.5%, String not yet — ToDo 9) and the skip-index predictions (minmax /
  set skip shares and index bytes exact; bloom within 1–2 marks per query and 1% in bytes).
- Open: ToDo 3 (noise, inert axes, Decimal32), ToDo 1 (generator items), ToDo 5 (engines).
