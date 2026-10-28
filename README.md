# turbo-parakeet

A toolkit for storing and querying columnar data efficiently in ClickHouse, DataFusion and Polars. It recommends column
types, encodings, compression, skip indexes and sort orders from patterns and relationships between columns, and tests
those recommendations empirically. Four components:

1. **Synthetic data** — `services/datagen/` (`synthdata`): seeded generators with targeted statistics, frames with
   planted relations, pairwise covering datasets and named scenarios.
2. **Analytics** — `services/analytics/`: techniques and recommenders; Python API over a Rust extension (pyo3, Arrow at
   the FFI boundary), with a Java binding over the same C ABI.
3. **Empirical benchmark** — `services/bench/` (`chbench`): loads a dataset under a grid of ClickHouse table variants,
   runs query grids and reports sizes and timings.
4. **Evaluation loop** — `services/evaluate/` (`evaluate`): recommends, benchmarks the ClickHouse recommendations
   against alternatives with chbench and scores prediction error, regret, CPU and memory.

Objectives and architecture: [CLAUDE.md](CLAUDE.md); planned work: [ToDo.md](ToDo.md).

## Building and testing locally

### One-time setup

Use the conda env `p312` (Python 3.12).

1. `pip install -r requirements.txt -r requirements-dev.txt` — pytest, pytest-xdist, black,
   ruff, maturin; the accuracy references scipy (chi-squared) and scikit-learn (adjusted Rand),
   which the tests and benchmarks need; and the optional competing implementations
   (datasketch, polars-ds, fastbloom-rs) and statsmodels, whose tests are skipped when the
   library is missing. It also installs the bench's `clickhouse-connect` and `rich`.
2. Build the extension into the env: `python scripts/check.py build` (`maturin develop
   --release` in `services/analytics/`; the Python package is an editable install, so Python
   changes need no rebuild).
3. Rust: the pinned toolchain (1.96.0 with rustfmt, clippy, llvm-tools) installs itself from
   `rust-toolchain.toml` on first cargo use.
4. Optional:
   - `cargo install cargo-llvm-cov` — the coverage steps.
   - JDK 25 — the Java tests.
   - Docker — local SonarQube, see [sonar/README.md](sonar/README.md).
   - Docker with the `chbench-ch` ClickHouse container — the bench's integration tests and the evaluation loop (see [services/bench/CLAUDE.md](services/bench/CLAUDE.md)).
   - `pre-commit install` — git hooks (commit: rustfmt + black; push: clippy + ruff + Rust tests).
     The hooks call `python`, so the p312 env must be on `PATH` when you commit or push
     (activate it, or in Git Bash
     `export PATH="$HOME/miniconda3/envs/p312:$HOME/miniconda3/envs/p312/Scripts:$PATH"`);
     otherwise they fail with "Python was not found" and the commit or push is refused.

## Using it

```python
import polars as pl
from turbo_parakeet.recommenders import OneShotRecommender
from turbo_parakeet.techniques.distinct_count import DistinctCountHllRust

frame = pl.read_ipc("data.arrow")

# Recommendations per column: narrowest Arrow / Polars type, IPC compression, ClickHouse type,
# codec chain and skip index, each with predicted sizes.
recommendations = OneShotRecommender().add(frame).result()

# Any technique: the same three calls, one flat table of metrics per column / pair / triplet.
counts = DistinctCountHllRust(precision=14).add({"data": frame}).result()
```

What is available:

- **Recommenders:** `OneShotRecommender` (one frame) and `StreamingRecommender` (batches over
  time), with the same output schema; both also have a Java binding.
- **Techniques** (`turbo_parakeet.techniques.<name>`): per column — `gcd`, `describe`, `top_k`,
  `distinct_count`, `population_cardinality`, `marginal_entropy`; per column, order-dependent —
  `runs_test`, `delta`, `minmax_index`, `set_index`; between distinct-value sets — `membership`,
  `similarity`; between rows of column pairs / triplets — `chi_squared`, `pairwise_entropy`,
  `threeway_entropy`, `adjusted_rand`. Each has a Rust implementation and an exact reference;
  details in [services/analytics/CLAUDE.md](services/analytics/CLAUDE.md).
- **Benchmarks:** [tests/performance/README.md](tests/performance/README.md).

Run a benchmark suite (needs the ClickHouse container):

```
cd services/bench
python -m chbench.cli run suites/int_encodings.yaml --dry-run   # the grid, without running it
python -m chbench.cli run suites/int_encodings.yaml
python -m chbench.cli report --suite int_encodings
```

### Before pushing

Run `python scripts/check.py ci` with the p312 Python. It runs these steps, in order:

| Step | What it does |
|---|---|
| `fmt-check` | `cargo fmt --check` and `black --check` |
| `lint` | clippy for both feature sets (`-D warnings`) and ruff |
| `layering` | `scripts/check_layering.py`: the Rust crate's one-way module dependencies (bindings → recommenders → techniques → common) |
| `build` | `maturin develop --release` |
| `test-rust` | `cargo test --lib` |
| `test-py` | pytest on 4 pytest-xdist workers (`-n 4` in `pytest.ini`; `-n 0` for serial) |
| `test-bench` | pytest for `services/bench` (its own run; tests needing ClickHouse skip without the server) |
| `test-datagen` | pytest for `services/datagen` (its own run) |
| `test-evaluate` | pytest for `services/evaluate` (its own run; the live test skips without the server) |
| `test-java` | builds the C library with Cargo profile `ci` (release without fat LTO), then `./mvnw test` with JaCoCo |

Groups: `fast` = `fmt-check` + `lint` + `layering`; `check` = everything except `test-java`; `ci` = all of
the above. Any step can also be run on its own (`python scripts/check.py test-py`).

On demand:

- `coverage-rust-full` — Rust coverage from the unit tests merged with the Python tests run
  against an instrumented extension (what CI uploads to Codecov). The normal extension is
  restored afterwards.
- `coverage-py` — Python coverage (`coverage.xml`, `htmlcov/`; the bench's in `services/bench/coverage.xml`).
- `sonar` — Python and Java coverage, then the SonarQube scanner (needs Docker,
  `SONARQUBE_TOKEN` and the compose stack running).

`CAPI_PROFILE=release` makes `test-java` build and test the shipped (fat LTO) C library.

## CI

Workflow: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml). It runs on pushes and
pull requests to `main` (not for docs-only changes: `*.md`, `docs/`, `sonar/`, `notebooks/`),
weekly on Mondays, and on manual dispatch. A newer push cancels a superseded run.

| Job | When | What it runs |
|---|---|---|
| Linting & Formatting | always | `check.py fast` |
| Rust Tests & Coverage | always | `check.py coverage-rust-full`, then `check.py test-bench`, `test-datagen`, `test-evaluate`, then Codecov upload |
| Java Tests | not on pull requests | `check.py test-java` (profile `ci`); weekly and dispatch also run it with `CAPI_PROFILE=release` |
| Python Tests (Windows) | weekly and dispatch only | `check.py build` (release), then `check.py test-py` |
| CodeQL | pull requests, weekly, dispatch (not pushes; public repos only) | CodeQL analysis |
| Semgrep | always | `semgrep scan --config p/default`, SARIF to the Security tab |
| Pipeline Status | always | gate job: fails if any job above failed or was cancelled |

`coverage-rust-full` runs `cargo test --lib --profile coverage`, then
`maturin develop --profile coverage`, then pytest with coverage against that instrumented
extension, then the llvm-cov report.

Versions are pinned (`requirements-dev.txt`, `rust-toolchain.toml`, committed `Cargo.lock`,
actions pinned by SHA); `requirements.txt` caps each library below its next major version, so a
new major arrives through a Dependabot PR. Dependabot: Rust and Python weekly, Java and Actions monthly, grouped,
with a 7-day cooldown. A cold Rust cache (any `Cargo.lock` change) makes a run about three
times slower.

## Where local and CI differ

| | Local (`check.py ci`) | CI |
|---|---|---|
| Python tests run against | the **release** extension | Linux: the **instrumented non-LTO** extension (inside the coverage job); the release build only on the weekly Windows run |
| Rust unit tests | their own step (`test-rust`) | inside the coverage job |
| Coverage, CodeQL, Semgrep | not part of `ci` (coverage on demand) | run in CI |
| Java on pull requests | always | skipped |
| Operating system | yours (Windows) | Linux every run; Windows weekly |

## Licence

MIT — see [LICENSE](LICENSE).
