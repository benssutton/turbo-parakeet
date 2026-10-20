# Python Package Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Python package `analytics` to `turbo_parakeet` (maturin mixed layout, `techniques/` and `recommenders/` subpackages, compiled module `turbo_parakeet._core`), clean break.

**Architecture:** `git mv` the package to `services/analytics/python/turbo_parakeet/`, point maturin at it, rename the pyo3 module, then rewrite every reference with one ordered regex script (run once from the scratchpad, not committed). Rust crate, C library, Java and the `services/analytics/` directory keep their names.

**Tech Stack:** maturin 1.15 (mixed layout), pyo3 0.29, Python 3.12, git.

**Spec:** `docs/superpowers/specs/2026-10-22-package-rename-design.md`.

Conventions: repo `c:/Users/Alexander/turbo-parakeet`; `E=/c/Users/Alexander/miniconda3/envs/p312; export PATH="$E:$E/Scripts:$E/Library/bin:/c/Users/Alexander/.cargo/bin:$PATH" PYO3_PYTHON="$E/python.exe"; PY=$E/python.exe` in every shell; commit as `git -c user.email=benssutton@hotmail.com -c user.name=benssutton commit …` without `--no-verify` (the hook needs the PATH above); messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `docs/superpowers/` needs `git add -f`; ToDo.md items 15–16 stay byte-for-byte intact; never push. Do not touch `docs/superpowers/` except this plan's checkboxes (historical documents keep the old names).

---

### Task 0: Branch and baseline

- [ ] `git switch -c package-rename`
- [ ] Record the baseline test count: `$PY -m pytest --collect-only -q -n 0 | tail -1` (e.g. "N tests collected"); keep N for Task 3.

### Task 1: Move, build configuration, rewrite, rebuild

**Files:** everything under `services/analytics/analytics/` (moved), `services/analytics/pyproject.toml`, `services/analytics/src/bindings/python.rs`, and every text file that references the old paths.

- [ ] **Step 1: Move the package** (from `services/analytics/`):

```bash
cd /c/Users/Alexander/turbo-parakeet/services/analytics
mkdir -p python/turbo_parakeet/techniques
git mv analytics/__init__.py analytics/_plugin.py analytics/_dtypes.py python/turbo_parakeet/
git mv analytics/base.py analytics/_sets.py analytics/_skip_index.py analytics/_ams.py python/turbo_parakeet/techniques/
for p in gcd describe top_k runs_test minmax_index set_index membership similarity chi_squared pairwise_entropy threeway_entropy adjusted_rand; do
  git mv analytics/$p python/turbo_parakeet/techniques/$p
done
git mv analytics/recommend python/turbo_parakeet/recommenders
git ls-files analytics   # must print nothing
rm -rf analytics         # only untracked leftovers remain: analytics.pyd, analytics.pdb, __pycache__
```

- [ ] **Step 2: `techniques/__init__.py`** — create `services/analytics/python/turbo_parakeet/techniques/__init__.py`:

```python
"""
Techniques: metrics only, every one used the same way:

    from turbo_parakeet.techniques.chi_squared import ChiSquaredRust
    result = ChiSquaredRust().add({"sales": df}).result()

Per-column:            gcd, describe, top_k
Per-column, ordered:   runs_test, minmax_index, set_index
Multi-set:             membership, similarity
Ordered (row-aligned): chi_squared, pairwise_entropy, threeway_entropy, adjusted_rand
"""
```

- [ ] **Step 3: package docstring** — replace the docstring of `services/analytics/python/turbo_parakeet/__init__.py` (keep `__version__ = "0.1.0"`) with:

```python
"""
turbo_parakeet: storage recommendations for ClickHouse, DataFusion and Polars, from
column patterns and relationships, with a Rust core.

    turbo_parakeet.techniques     metrics (one subpackage per technique)
    turbo_parakeet.recommenders   OneShotRecommender, StreamingRecommender, to_arrow, to_chbench

The compiled extension turbo_parakeet._core and its wrappers in turbo_parakeet._plugin
are private.
"""
```

- [ ] **Step 4: maturin** — in `services/analytics/pyproject.toml` change `name = "analytics"` to `name = "turbo-parakeet"` and append:

```toml

[tool.maturin]
python-source = "python"
module-name = "turbo_parakeet._core"
```

- [ ] **Step 5: pyo3 module** — in `services/analytics/src/bindings/python.rs` (line ~586) replace

```rust
#[pymodule]
fn analytics(m: &Bound<'_, PyModule>) -> PyResult<()> {
```
with
```rust
#[pymodule(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
```
(The three `#[pyclass(frozen, module = "analytics.analytics")]` strings are handled by the script in Step 6.) Leave `Cargo.toml` (`[lib] name = "analytics"`) unchanged.

- [ ] **Step 6: rewrite references** — write this script to the scratchpad (not the repo), e.g. `$TMP/rename_refs.py`, and run it from the repo root: `$PY $TMP/rename_refs.py`.

```python
"""One-off: rewrite references to the old `analytics` Python package (spec 2026-10-22)."""

import re
import subprocess
from pathlib import Path

TECH = (
    "gcd|describe|top_k|runs_test|minmax_index|set_index|membership|similarity|"
    "chi_squared|pairwise_entropy|threeway_entropy|adjusted_rand"
)
TECH_PRIVATE = "base|_sets|_skip_index|_ams"
NOT_AFTER = r"(?<![\w./-])"  # not part of a longer name, path or Java package
RULES = [
    (r"services/analytics/analytics/", "services/analytics/python/turbo_parakeet/"),
    (r"services/analytics/analytics\b", "services/analytics/python/turbo_parakeet"),
    (r"from services\.analytics\.analytics import recommend\b", "from turbo_parakeet import recommenders as recommend"),
    (NOT_AFTER + r"analytics\.analytics\b", "turbo_parakeet._core"),
    (NOT_AFTER + r"analytics\.recommend\b", "turbo_parakeet.recommenders"),
    (NOT_AFTER + r"analytics\.(_plugin|_dtypes)\b", r"turbo_parakeet.\1"),
    (NOT_AFTER + rf"analytics\.({TECH_PRIVATE}|{TECH})\b", r"turbo_parakeet.techniques.\1"),
    (NOT_AFTER + r"analytics\.__version__\b", "turbo_parakeet.__version__"),
    (NOT_AFTER + r"analytics\.<technique>", "turbo_parakeet.techniques.<technique>"),
    (NOT_AFTER + r"analytics/recommend/", "turbo_parakeet/recommenders/"),
    (NOT_AFTER + r"analytics/(_plugin|_dtypes)\.py", r"turbo_parakeet/\1.py"),
    (NOT_AFTER + rf"analytics/({TECH_PRIVATE}|{TECH})\b", r"turbo_parakeet/techniques/\1"),
    (r"\bfrom analytics import analytics as\b", "from turbo_parakeet import _core as"),
    (r"\bfrom analytics import _plugin\b", "from turbo_parakeet import _plugin"),
    (rf"\bfrom analytics import ({TECH})\b", r"from turbo_parakeet.techniques import \1"),
    (r"^(\s*)import analytics(\s|$)", r"\1import turbo_parakeet\2"),
]
SUFFIXES = {".py", ".rs", ".java", ".md", ".toml", ".yml", ".yaml", ".ini", ".properties", ".ipynb", ".cfg", ".txt"}
SKIP = ("docs/superpowers/", "ToDo.md", "CLAUDE.md")  # CLAUDE.md and ToDo.md are edited by hand (Task 2)

files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
for name in files:
    path = Path(name)
    if path.suffix not in SUFFIXES or name.startswith(SKIP) or not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    new = text
    for pattern, repl in RULES:
        new = re.sub(pattern, repl, new, flags=re.MULTILINE)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="")
        print("rewrote", name)
```

Then fix by hand what the rules cannot:
  - `tests/performance/harness.py`: after the script it reads `import turbo_parakeet` and `analytics=pl.lit(turbo_parakeet.__version__)` — keep the result column key `analytics` (results schema unchanged).
  - `scripts/check.py` (~line 173–178): the coverage extension backup must read
    ```python
    ext = [
        f
        for pattern in ("_core*.pyd", "_core*.so")
        for f in (CRATE / "python" / "turbo_parakeet").glob(pattern)
    ]
    backup = Path(tempfile.mkdtemp(prefix="turbo-parakeet-ext-"))
    ```
    and wherever it restores the files it must copy them back into `CRATE / "python" / "turbo_parakeet"`.
  - `.github/workflows/ci-cd.yml` line ~191: `run: python -c "import turbo_parakeet; print('turbo_parakeet module loaded')"`.
  - `sonar-project.properties`: `sonar.sources=services/analytics/python,services/analytics/src,services/analytics/bindings/java/src/main` (check the script made exactly this).
  - `notebooks/Examples.ipynb`: the import became `from turbo_parakeet import recommenders as recommend`; confirm the notebook's other cells still read (`recommend.X`) and contain no other `analytics` package reference.
  - Java javadoc in `OneShotParams.java` / `StreamingParams.java` now names `turbo_parakeet.recommenders.*Recommender`; `analytics.library.dir` and `io.github.benssutton.analytics` must be unchanged.
  - `services/analytics/python/turbo_parakeet/_plugin.py` docstring line 1: "(turbo_parakeet._core)" instead of "(analytics.pyd)".

- [ ] **Step 7: leftovers** — run

```bash
git grep -nP "(^|[^\w./-])analytics(\.[a-z_]|/[a-z_]|\b)" -- ':!docs/superpowers' ':!CLAUDE.md' ':!ToDo.md' ':!*.lock' \
  | grep -vE "services/analytics|analytics\.library\.dir|benssutton\.analytics|benssutton/analytics|analytics_(oneshot|streaming|free)|lib(analytics)|analytics\.(dll|so|dylib)|ANALYTICS_|\[lib\]"
```
Inspect every hit. Fix the ones that name the Python package or its files; leave Rust crate / C library / Java / directory names and plain prose ("analytics" as a word). Also `git grep -n "import analytics\|from analytics" -- ':!docs/superpowers'` must print nothing.

- [ ] **Step 8: environment and build**

```bash
$PY -m pip uninstall -y analytics
cd /c/Users/Alexander/turbo-parakeet/services/analytics && $PY -m maturin develop --release
cd /c/Users/Alexander/turbo-parakeet
$PY -c "import turbo_parakeet._core, turbo_parakeet.techniques.gcd, turbo_parakeet.recommenders; print('ok')"
$PY -c "import analytics" ; echo "exit=$?"   # expect ModuleNotFoundError, exit=1
```
Expected: `ok`; then `ModuleNotFoundError: No module named 'analytics'`. The compiled file is `services/analytics/python/turbo_parakeet/_core.pyd` (covered by `.gitignore`'s `*.pyd` / `*.pdb`; `git status` must not list it).

- [ ] **Step 9: checks** — `$PY scripts/check.py fmt && $PY scripts/check.py check`. Expected: all pass; pytest's count equals Task 0's N. Fix failures (typically a missed string reference, e.g. a module name in `importlib.import_module`, a `monkeypatch.setattr("analytics....")` target or a benchmark child-process `-c` snippet).

- [ ] **Step 10: commit**

```bash
git add -A services/analytics tests scripts .github sonar-project.properties sonar notebooks
git status --short   # review: renames (R) plus the edited files; no .pyd/.pdb
git -c user.email=benssutton@hotmail.com -c user.name=benssutton commit -m "Rename the Python package to turbo_parakeet (ToDo 5c)

Maturin mixed layout: services/analytics/python/turbo_parakeet/{techniques,recommenders}/,
compiled module turbo_parakeet._core. Rust crate, C library and Java binding unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: CLAUDE.md and ToDo.md

**Files:** `CLAUDE.md`, `ToDo.md`.

- [ ] **Step 1: CLAUDE.md paths** — run the Task 1 script's `RULES` over `CLAUDE.md` only (copy the script, set the file list to `["CLAUDE.md"]`, drop `SKIP`). Then check: `io.github.benssutton.analytics.*`, `analytics.library.dir`, `analytics_*_recommender_*` C names, `services/analytics/` paths and the heading "Rust Extension (analytics)" are unchanged; `analytics.recommend.to_arrow` / `analytics.recommend.to_chbench` / `analytics._dtypes.WIDE_INTEGERS` / `analytics._plugin` etc. now read `turbo_parakeet.…`.
- [ ] **Step 2: Project Structure tree** — replace the block from `│   └── analytics/` through the `oneshot.py, streaming.py, …` line with:

```
│   └── python/turbo_parakeet/              # the Python package (maturin mixed layout; spec 2026-10-22)
│       ├── __init__.py                     # __version__ only
│       ├── _core.pyd                       # compiled extension (module turbo_parakeet._core)
│       ├── _plugin.py                      # PRIVATE wrappers over the Arrow binding (called only by *Rust classes)
│       ├── _dtypes.py                      # dtype groupings / value families
│       ├── techniques/
│       │   ├── base.py                     # Technique contract, helpers, metric_mismatches
│       │   ├── _sets.py, _skip_index.py, _ams.py  # canonical distinct values; the skip-index base; AMS parameter checks
│       │   └── <technique>/                # gcd, describe, top_k, runs_test, minmax_index, set_index, membership, similarity,
│       │       ├── __init__.py             #   chi_squared, pairwise_entropy, threeway_entropy, adjusted_rand
│       │       ├── base.py                 # technique base: METRICS, eligibility, conclusions, RTOL/ATOL
│       │       └── rust.py, <library>.py … # one file per implementation
│       └── recommenders/                   # oneshot.py, streaming.py, _input.py, _maps.py (to_arrow), chbench.py (to_chbench)
```
and change the `pyproject.toml, Cargo.toml` comment to `# maturin build (mixed layout, python-source = "python"; editable install)`. In "Rust Extension": `pyo3 module \`turbo_parakeet._core\`` for the python.rs bullet.
- [ ] **Step 3: Current Focus** — replace "Next (ToDo.md): 5c (package rename), then 3 (delta / double-delta codecs)." with "The Python package is `turbo_parakeet` (ToDo 5c, spec 2026-10-22). Next (ToDo.md): 3 (delta / double-delta codecs)."
- [ ] **Step 4: ToDo.md** — item 5c: insert `**Done** (spec 2026-10-22): Python-only rename; the Rust crate, C library and Java binding keep \`analytics\`.  ` right after `c. `; in the "Agreed starting order" line replace "next 5c then 3" with "5c done; next 3" and delete the sentence "Do 5c (rename) just before the first new Python package (ToDo 3 or 8)." Confirm with `git diff ToDo.md` that items 15–16 are untouched.
- [ ] **Step 5:** `git grep -nP "(^|[^\w./-])analytics\.(recommend|base|_plugin|_dtypes|_sets|gcd|describe)\b" CLAUDE.md` prints nothing. Commit `"Docs: turbo_parakeet package paths (CLAUDE.md, ToDo 5c)"`.

### Task 3: Verification

- [ ] `$PY scripts/check.py ci` → "all checks passed"; pytest count = N.
- [ ] `$PY scripts/check.py coverage-rust-full` (≈ 20 min cold for the instrumented build) → passes, and afterwards `services/analytics/python/turbo_parakeet/_core.pyd` is the normal (restored) extension: `$PY -c "import turbo_parakeet._core"` works and `$PY -m pytest tests/test_gcd.py -q` passes.
- [ ] `git grep -nE "(from|import) analytics\b|analytics\.(recommend|base|_plugin|_dtypes)" -- ':!docs/superpowers'` prints nothing.
- [ ] Report.
