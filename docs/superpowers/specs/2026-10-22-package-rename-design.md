# Python package rename to `turbo_parakeet` (ToDo 5c)

Status: approved 2026-10-22.  Done just before the first new Python package (ToDo 3).

## 1. Goal

The Python package `analytics` clashes with Segment's `analytics` package on PyPI.  Rename it to `turbo_parakeet`
with maturin's mixed layout, group the subpackages by kind (techniques, recommenders), and make a clean break: no
`analytics` shim.  The package is unpublished and chbench does not import it, so this repository is its only user.

**Out of scope:** the Rust crate and its C library keep the name `analytics` (`[lib] name = "analytics"`,
`analytics.dll` / `libanalytics.so`), so the Java binding (`io.github.benssutton.analytics`, the
`analytics.library.dir` property) and the C ABI are unchanged; the `services/analytics/` directory keeps its name (to
be revisited when turbo-parakeet and clickhouse_benchmark merge, ToDo 14).

## 2. Layout

Every move is a `git mv`, so history follows the files.

```
services/analytics/python/turbo_parakeet/
├── __init__.py              # __version__; docstring rewritten with the new paths
├── _core.pyd / _core*.so    # compiled extension (was analytics/analytics.pyd; git-ignored)
├── _plugin.py, _dtypes.py   # private, shared by techniques and recommenders
├── techniques/
│   ├── __init__.py          # docstring only (moved from the package docstring)
│   ├── base.py, _sets.py, _skip_index.py, _ams.py
│   └── gcd/, describe/, top_k/, runs_test/, minmax_index/, set_index/, membership/, similarity/,
│       chi_squared/, pairwise_entropy/, threeway_entropy/, adjusted_rand/
└── recommenders/            # was analytics/recommend/: __init__, oneshot, streaming, _input, _maps, chbench
```

| old | new |
|---|---|
| `analytics.<technique>` | `turbo_parakeet.techniques.<technique>` |
| `analytics.base`, `._sets`, `._skip_index`, `._ams` | `turbo_parakeet.techniques.base`, `._sets`, `._skip_index`, `._ams` |
| `analytics._plugin`, `analytics._dtypes` | `turbo_parakeet._plugin`, `turbo_parakeet._dtypes` |
| `analytics.recommend` (and its modules) | `turbo_parakeet.recommenders` |
| `analytics.analytics` (compiled) | `turbo_parakeet._core` |

`detectors/` is not created now; the first detector (ToDo 9) creates it.  Technique package names are unchanged.

## 3. Build

- `services/analytics/pyproject.toml`: `[project] name = "turbo-parakeet"`; new `[tool.maturin]` with
  `python-source = "python"` and `module-name = "turbo_parakeet._core"`.
- `src/bindings/python.rs`: `#[pymodule] fn _core` (was `fn analytics`).
- `Cargo.toml` unchanged (lib name `analytics`): no dependency change, a warm rebuild.
- `.gitignore`: the compiled extension and its `.pdb` at the new path.

## 4. References updated

- Every import in `services/analytics/python/`, `tests/` and `tests/performance/`, including strings that name
  modules (e.g. `importlib`, `monkeypatch` targets, subprocess `-c` snippets in the benchmark harness).
- `scripts/check.py`: the coverage step's extension backup / restore globs `_core*.pyd` / `_core*.so` in
  `python/turbo_parakeet/`; any other path naming `services/analytics/analytics`.
- `tests/test_layering.py` / `scripts/check_layering.py` where they name Python paths.
- `.github/workflows/ci-cd.yml`: the Windows import smoke test (`import turbo_parakeet`).
- `sonar-project.properties`: `sonar.sources` → `services/analytics/python`.
- `notebooks/Examples.ipynb`, `sonar/false-positives.md`.
- `CLAUDE.md`: every `analytics.<x>` Python path, the Project Structure tree, the private-binding list; ToDo.md 5c
  marked **Done** (spec 2026-10-22).
- Not changed: the historical specs and plans under `docs/superpowers/` (they record what was true when written), and
  Rust / Java / C names (§1).

## 5. Environment

`pip uninstall -y analytics` in p312 (removes the stale `analytics.pth` and `analytics-0.1.dist-info`), then
`maturin develop --release` from `services/analytics/`, which installs the editable `turbo_parakeet` pointing at
`python/`.

## 6. Verification

- `python scripts/check.py ci` passes (same test count as before the rename).
- `git grep -nE "(from|import) analytics\b|analytics\.(recommend|base|_plugin|_dtypes)"` finds nothing outside
  `docs/superpowers/`.
- In p312, `python -c "import analytics"` raises `ModuleNotFoundError`; `python -c "import turbo_parakeet._core,
  turbo_parakeet.techniques.gcd, turbo_parakeet.recommenders"` succeeds.
- `python scripts/check.py coverage-rust-full` restores `_core` after the instrumented run (run once by hand).
