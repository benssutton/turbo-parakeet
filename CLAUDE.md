# Project Objectives
A living "DBA" toolkit for columnar storage: recommend how to store and query tabular data efficiently in three
engines, given equal weight — **ClickHouse**, **DataFusion** (pure Arrow) and **Polars** (more engines may follow) — by
determining patterns and relationships between columns, within and between dataframes, and prove the recommendations
empirically. As the engines improve, the techniques and recommendations are re-measured and refined.

## Components
1. **Synthetic data** — seeded generators with targeted statistics (cardinality, entropy, runs, inter-arrival,
   real-data columns). Today in two places: `services/bench/chbench/datagen` (target-driven, one column) and
   `tests/datagen.py` (the analytics fixtures); consolidated into `services/datagen/` by ToDo 1.
2. **Analytics** — `services/analytics/`: techniques (metrics), recommenders and detectors (decisions); Rust core,
   Python API `turbo_parakeet`, Java binding. Details: `services/analytics/CLAUDE.md`.
3. **Empirical benchmark** — `services/bench/` (`chbench`): loads a dataset under a grid of table variants and runs query
   grids; ClickHouse today, Polars and DataFusion planned (ToDo 5). Details: `services/bench/CLAUDE.md`.

## The loop
a. Design synthetic datasets that try to defeat the analytic techniques. b. Run them through the techniques and
recommenders. c. Benchmark the recommendations empirically against alternatives on the same data. d. Score the
techniques' accuracy (statistics, predicted sizes and skip shares), their efficiency (speed, memory) and the decisions'
quality; improve, and repeat b–d. Last: run the loop on (a sample of) real data (ToDo 8).

## Rules
- **Measure vs decide.** Techniques report metrics only: statistics, estimates (with interval and method) and effect
  sizes. Anything that compares a value against a tunable threshold is a decision, and decisions belong to the
  recommenders and detectors.
- **Targets.**
  - DataFusion: Arrow types and encodings (dictionary; REE is not recommended — DataFusion 54 and 55 reject MIN / MAX
    over it; ToDo 6 re-checks).
  - Polars: Polars dtypes (Categorical / Enum; no REE).
  - Both: Arrow IPC buffer compression (LZ4_FRAME / ZSTD + level).
  - ClickHouse: column type, codec chain (GCD, Delta, DoubleDelta, then LZ4 / ZSTD + level), skip indexes
    (`minmax`, `set`, `bloom_filter`, GRANULARITY) and `ORDER BY`.
- **Formats.** In-memory and Arrow IPC only. **No Parquet anywhere**: not as an analysis target, and not for the
  project's own files (fixtures, benchmark results). `tests/test_layering.py` guards every tracked Python file.
- **Row order.** The frame's row order is taken to be the stored order. Order-dependent metrics (runs, MinMax, Set,
  delta) are measured on it, beside random-order baselines. For ClickHouse the caller supplies data sorted by the
  intended `ORDER BY`. Evaluating proposed orders is the ordering recommender's job (ToDo 19).
- **Queries.** Point-equality queries `col = v`, weighted by row frequency or uniformly over distinct values. Range
  queries are on hold (ToDo 10).
- **Validation.** DataFusion and Polars recommendations are proven in process: cast, verified row by row and measured
  as IPC; their empirical query bench is ToDo 5. ClickHouse recommendations carry predicted metrics and are validated
  empirically by the bench: `turbo_parakeet.recommenders.to_chbench` writes the recommender's Arrow result as IPC plus a
  chbench suite, and `tests/performance/calibrate_*.py` compare predicted with measured (ToDo 2 turns this into the
  loop).
- **Deliverables.** Analytics: every recommender and detector ships one-shot and streaming versions (streaming tested
  for parity against one-shot) and a Java binding at parity; techniques have no Java binding (Python API only).
  Synthetic data and the bench are Python only.

# Layout
```
turbo-parakeet/
├── services/
│   ├── analytics/     # component 2: Rust crate, Python package turbo_parakeet, Java binding (its CLAUDE.md)
│   └── bench/         # component 3: chbench — datagen, harness, CLI, suites, spikes (its CLAUDE.md)
├── tests/             # turbo_parakeet's tests; datagen.py (analytics fixtures); performance/ (benchmarks, calibration)
├── docs/engines/      # engine knowledge notes (clickhouse.md)
├── scripts/           # check.py (local = CI), check_layering.py
├── notebooks/, sonar/
└── CLAUDE.md, README.md, ToDo.md
```

# Python Environment
Conda environment: `C:\Users\Alexander\miniconda3\envs\p312`
Python exe: `C:\Users\Alexander\miniconda3\envs\p312\python.exe`

# Coding Style
Prioritise performance & simplicity.

# Checks, CI and local tooling
`python scripts/check.py <step|group>` (run with the p312 Python) runs exactly what CI runs: `fast` (rustfmt, black, clippy for both
feature sets, ruff, and the `layering` step), `check` (fast + build + `cargo test --lib` + pytest + `test-bench`, the bench's own pytest run from `services/bench/`; `pytest.ini` runs it on 4 pytest-xdist workers, `-n 0` for serial), `ci` (check + Java), or single steps (`fmt`, `fmt-check`,
`lint`, `build`, `test-rust`, `test-py`, `test-bench`, `test-java`, `coverage-rust`, `coverage-rust-full`, `coverage-py`, `sonar`). Run `check` / `ci` before pushing.
Versions are pinned: `requirements-dev.txt` (black, ruff, pre-commit, maturin, pytest-xdist) and `rust-toolchain.toml` (compiler, rustfmt, clippy); `requirements.txt` caps each runtime / reference library below its next major version (a new major arrives through a Dependabot PR);
`Cargo.lock` is committed. Optional git hooks: `.pre-commit-config.yaml` (commit: rustfmt + black; push: clippy + ruff + Rust tests). The hooks call `python`: put the p312 env on `PATH` first (Git Bash: `export PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"`; the push hook's Rust tests also need `PYO3_PYTHON`), else they fail with "Python was not found" and the commit / push is refused.
`.github/workflows/ci-cd.yml`: lint, Rust + Python coverage (one job: unit tests, then pytest on an instrumented extension), Java,
CodeQL (public repos only), Semgrep, Windows Python tests, the bench tests (no ClickHouse server in CI), and a `status` gate job; actions are pinned by SHA. **Cost control** (private
repo, limited Actions minutes; Windows bills 2x): the Windows job runs only on the weekly schedule / manual dispatch, Java not on
pull requests, CodeQL not on pushes (pull requests, weekly, dispatch), the weekly run also tests the Java binding against the release (LTO) C library, docs-only changes (`*.md`, docs/, sonar/, notebooks/) skip the pipeline, superseded runs are cancelled; a cold Rust
cache (any `Cargo.lock` change) makes a run ~3x dearer. `.github/dependabot.yml`: Rust / Python weekly, Java / Actions monthly,
grouped, at most 2 open PRs per ecosystem.
Rust coverage (`coverage-rust-full`, what CI uploads to Codecov) merges the unit tests with the Python tests run against an
instrumented build of the extension (Cargo profile `coverage`: no LTO; dedicated target dir `target/llvm-cov-target`), because
bindings/python.rs is reachable only from Python; the step restores the normal extension afterwards. Cold build ≈ 20 min (dependencies,
cached in CI), warm ≈ 2.5 min. `coverage-rust` is the unit-tests-only variant.
Local SonarQube Community Build + the official SonarQube MCP server (`.mcp.json`): see `sonar/README.md`.

# Git
- Branch from `main`; run `python scripts/check.py ci` before pushing.
- `docs/superpowers/` (brainstorming specs and implementation plans) is git-ignored on purpose and must not be pushed:
  never `git add -f` it. Code, tests and CLAUDE.md files do not cite it.

# State
Open work is tracked in [ToDo.md](ToDo.md); its header gives the suggested order.
