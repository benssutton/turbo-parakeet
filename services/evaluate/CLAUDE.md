# Evaluation loop (`services/evaluate`, package `evaluate`)

Component 4 of the toolkit (objectives: the root CLAUDE.md): steps b–d of the loop — recommend, benchmark the
recommendation against alternatives on the same data, score. ClickHouse only (Polars / DataFusion: ToDo 5).

# Running
From the repository root, with the `chbench-ch` container up (services/bench/CLAUDE.md, Environment):
```
python -m evaluate run <datasets…> [--workers 3] [--samples 10] [--dry-run] [--resume RUN_DIR] [--runs-dir DIR]
python -m evaluate report RUN_DIR          # re-score, no server
python -m evaluate compare RUN_A RUN_B     # summary metric changes
```
Datasets: registry names (`evaluate/registry.py`: synth tiers, the planted `gcd` / `delta` / `skip` frames,
synthdata scenarios), `name:rows` for scalable entries, the group `calibration`, `--all`, or a path to an Arrow IPC
frame. `--dry-run` needs no server and prints the table-load count. Time a run whose query costs matter with
`--workers 1`: parallel suites share the server's CPUs.

# Design
- One directory per run, `services/evaluate/runs/<ts>/` (git-ignored): run.json, per dataset frame.arrow,
  result.arrow, cost.json, suites/, stores/ (one chbench SQLite store per suite), then the score files.
- Per column a size suite per type (the recommender's candidates + {none, GCD, Delta, DoubleDelta} × {LZ4, ZSTD(1),
  ZSTD(3)}) and a skip suite (chosen + rejected skip candidates + none, on the recommended chain). chbench expands
  axes as a product, which is why the two are separate.
- Scores (metrics only, no thresholds): size_cells, size_regret, skip_queries, skip_regret, query_cost,
  query_regret, recommend_cost (`evaluate/score.py` holds the schemas), summary.arrow and compare. CPU and memory are
  captured on every run: the recommender in a child process (wall, CPU, peak memory above the frame) and each
  ClickHouse query (memory_usage, User + System CPU from ProfileEvents). `query_cost` carries each table's CPU median
  with its 95% interval and, against the recommender's chosen table for the same query, the paired per-round
  `cpu_ratio` / `server_ratio` with intervals and `n_rounds` (chbench's `harness/stats.py`). `query_regret`, one row
  per (dataset, column, suite, query): the best alternative by `cpu_ratio`, its interval Bonferroni-adjusted over the
  `n_alternatives`; `resolved` when the interval lies below 1, `regret_cpu` = 1 − ratio when resolved, else 0.
- `bench.py` is the only module that talks to ClickHouse; `predict.py` (skip shares per queried value) and
  `score.py` need no server.
- Resume: each step skips what is on disk; chbench skips held samples; scoring always reruns. A suite's
  ClickHouse tables are dropped once it has run.
- Suites chbench cannot load (binary columns: no `arrow_type` for them) are dropped into the dataset's
  `suites/skipped.json` with to_chbench's own skips.

# Tests
`python scripts/check.py test-evaluate` (serial). Unit tests use a fake chbench store (`tests/fakes.py`); the live
test (`tests/test_live.py`, marked `clickhouse`) skips without a server. TDD.

# State
- Parity with the retired calibration scripts and the first full runs: `docs/results/`.
- First query-time scores (`docs/results/2026-10-29-query-regret.md`): 32 of 506 queries resolved, nearly all an LZ4
  chain beating the size-chosen ZSTD level on CPU; 50 000-row frames do not resolve.
