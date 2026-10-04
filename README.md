# turbo-parakeet

Finds patterns and relationships between columns, within and between dataframes. Python API
over a Rust extension (pyo3, Arrow at the FFI boundary), with a Java binding over the same
C ABI. Technique and architecture details: [CLAUDE.md](CLAUDE.md).

## Building and testing locally

### One-time setup

Use the conda env `p312` (Python 3.12).

1. `pip install -r requirements.txt -r requirements-dev.txt` — pytest, pytest-xdist, black,
   ruff, maturin and the optional reference libraries (polars-ds, fastbloom-rs, datasketch;
   their tests are skipped when a library is missing).
2. Rust: the pinned toolchain (1.96.0 with rustfmt, clippy, llvm-tools) installs itself from
   `rust-toolchain.toml` on first cargo use.
3. Optional:
   - `cargo install cargo-llvm-cov` — the coverage steps.
   - JDK 25 — the Java tests.
   - Docker — local SonarQube, see [sonar/README.md](sonar/README.md).
   - `pre-commit install` — git hooks (commit: rustfmt + black; push: clippy + ruff + Rust tests).

### Before pushing

Run `python scripts/check.py ci` with the p312 Python. It runs these steps, in order:

| Step | What it does |
|---|---|
| `fmt-check` | `cargo fmt --check` and `black --check` |
| `lint` | clippy for both feature sets (`-D warnings`) and ruff |
| `build` | `maturin develop --release` |
| `test-rust` | `cargo test --lib` |
| `test-py` | pytest on 4 pytest-xdist workers (`-n 4` in `pytest.ini`; `-n 0` for serial) |
| `test-java` | builds the C library with Cargo profile `ci` (release without fat LTO), then `./mvnw test` with JaCoCo |

Groups: `fast` = `fmt-check` + `lint`; `check` = everything except `test-java`; `ci` = all of
the above. Any step can also be run on its own (`python scripts/check.py test-py`).

On demand:

- `coverage-rust-full` — Rust coverage from the unit tests merged with the Python tests run
  against an instrumented extension (what CI uploads to Codecov). The normal extension is
  restored afterwards.
- `coverage-py` — Python coverage (`coverage.xml`, `htmlcov/`).
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
| Rust Tests & Coverage | always | `check.py coverage-rust-full`, then Codecov upload |
| Java Tests | not on pull requests | `check.py test-java` (profile `ci`); weekly and dispatch also run it with `CAPI_PROFILE=release` |
| Python Tests (Windows) | weekly and dispatch only | `check.py build` (release), then `check.py test-py` |
| CodeQL | pull requests, weekly, dispatch (not pushes; public repos only) | CodeQL analysis |
| Semgrep | always | `semgrep scan --config p/default`, SARIF to the Security tab |
| Pipeline Status | always | gate job: fails if any job above failed or was cancelled |

`coverage-rust-full` runs `cargo test --lib --profile coverage`, then
`maturin develop --profile coverage`, then pytest with coverage against that instrumented
extension, then the llvm-cov report.

Versions are pinned (`requirements-dev.txt`, `rust-toolchain.toml`, committed `Cargo.lock`,
actions pinned by SHA). Dependabot: Rust and Python weekly, Java and Actions monthly, grouped,
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
