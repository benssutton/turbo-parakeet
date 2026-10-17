# Generated Large Dataset and Parquet Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the checked-in `tests/data/large_dataset.arrow` with a seeded generator (plus null / run / clustering settings and a mixed preset), move benchmark results to Arrow IPC, and stop the polars crate pulling in `polars-parquet`.

**Architecture:** The generator lives in `tests/datagen.py` beside the other seeded generators, cached with `functools.cache`, and every consumer (6 test files, the benchmark harness) calls it instead of reading the file. Benchmarks write `.arrow` (`write_ipc`, ZSTD). `Cargo.toml` sets `default-features = false` on `polars` and re-adds only the features the crate needs; a guard test keeps Parquet out.

**Tech Stack:** Python 3.12 (numpy, Polars, pytest + xdist), Rust (polars 0.51, maturin), git.

**Spec:** `docs/superpowers/specs/2026-10-17-generated-dataset-design.md`

---

## Conventions for every task

- Shell: Git Bash. Python: `/c/Users/Alexander/miniconda3/envs/p312/python.exe` (written `PY` below; `PY=/c/Users/Alexander/miniconda3/envs/p312/python.exe`).
- Run everything from the repo root `c:/Users/Alexander/turbo-parakeet` unless a step says otherwise.
- pytest runs on 4 xdist workers by default (`pytest.ini`); add `-n 0` for a serial run.
- **Commits:** the pre-commit hook needs the conda env on `PATH`, else it fails with "Python was not found". Commit with:
  ```bash
  PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH" git commit -m "..."
  ```
  Never use `--no-verify`. End every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (after a blank line).
- `docs/superpowers/` is git-ignored but tracked: add files there with `git add -f`.
- Formatting: `$PY scripts/check.py fmt` (rustfmt + black) before each commit that touches code; `$PY scripts/check.py fast` must pass.

## File map

| File | Change |
|---|---|
| `tests/datagen.py` | `_run_lengths` helper (shared with `runs_int`), `large_dataset`, `large_dataset_mixed`, `LARGE_DATASET_REGIMES` |
| `tests/test_datagen.py` | generator tests |
| `tests/test_runs_test.py`, `test_minmax_index.py`, `test_set_index.py`, `test_describe.py`, `test_recommend.py`, `test_streaming_recommend.py` | read the generator; agreement / parity also on the mixed preset |
| `tests/performance/harness.py` | `large_dataset()` wraps the generator; results as `.arrow` |
| `tests/performance/benchmark_recommend.py`, `benchmark_top_k_accuracy.py`, `benchmark_ams_entropy.py` | results and child hand-off files as `.arrow` |
| `services/analytics/Cargo.toml`, `Cargo.lock` | `polars` without default features |
| `tests/test_layering.py` | Parquet guard |
| `tests/data/large_dataset.arrow` | deleted |
| `services/analytics/analytics/similarity/rust.py`, `CLAUDE.md`, `ToDo.md`, the spec | numbers and docs |

---

### Task 0: Record the spec amendments found while planning

Probing the prototype generator against the real tests showed three things the spec must say. No code.

**Files:**
- Modify: `docs/superpowers/specs/2026-10-17-generated-dataset-design.md`

- [ ] **Step 1: Amend §2.2 step 2** — replace the sentence `1 → sorted, 0 → a random permutation (today's order).` with:

```markdown
1 → sorted.  `clustering = 0` skips this step entirely and keeps the drawn order, so runs from step 1 survive
(sorting and then shuffling every position would destroy them).
```

- [ ] **Step 2: Amend §2.2 step 3** — replace `3. **Nulls — `null_rate` ∈ [0, 1].**  An independent Bernoulli(`null_rate`) mask per column.` with:

```markdown
3. **Nulls — `null_rate` ∈ [0, 1].**  An independent Bernoulli(`null_rate`) mask per column, applied with
   `Series.zip_with` so null slots hold no bytes (as an Arrow builder leaves them).  `pl.when(...).then(None)` and
   `Series.set(mask, None)` keep the old bytes under the nulls; the one-shot recommender then measures those bytes
   in the cast candidates while the streaming one does not, and the parity test fails (e.g. `col_6`
   `rec_arrow_size_bytes` 341 376 vs 356 264).  That gap is recorded as a follow-up in ToDo 7; this spec does not
   fix it.
```

- [ ] **Step 3: Amend §4.2** — replace the paragraph starting `` `polars` gets `default-features = false`. `` up to the acceptance list with:

```markdown
`polars` gets `default-features = false`.  The defaults (`docs, zip_with, csv, polars-parquet, temporal, fmt,
dtype-slim`) also switched on, through polars-io, `dtype-i8` / `dtype-i16` / `dtype-time` and the whole cloud stack
(tokio, reqwest, object_store).  The library still compiles without them, but losing a dtype feature would change
runtime behaviour silently, so the feature list is:
`lazy, performant, dtype-slim, dtype-i8, dtype-i16, dtype-time, dtype-decimal, dtype-struct, dtype-array,
dtype-categorical, temporal, timezones, fmt, zip_with`.  Acceptance adds: the features enabled on polars-core,
polars-ops, polars-io, polars-time and polars-lazy (`cargo tree -e features`) lose nothing but the Parquet, CSV,
IPC, serde, cloud / async, new-streaming and unused-ops features listed in the plan (Task 5).
```

- [ ] **Step 4: Amend §3 Consumers, "Measured numbers"** — replace the sentence `Re-measure and update every figure taken from the old file: ... labelled as measured on the old file.` with:

```markdown
The MinHash recall is re-measured (on the generated frame: 0.65, 407/625, 218 misses, all containment pairs with
Jaccard 0.00002–0.33, median 0.10) and updated in `analytics/similarity/rust.py` and CLAUDE.md.  Benchmark speedups
quoted in CLAUDE.md (runs test 16.9x; Membership 0.60x) are not re-run: they are labelled as measured on the former
large_dataset.arrow.
```

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-10-17-generated-dataset-design.md
git commit -m "Spec: generated dataset — clustering 0 keeps drawn order, zip_with nulls, polars feature list"
```

---

### Task 1: The generator and its tests

**Files:**
- Modify: `tests/datagen.py` (module docstring; `runs_int` at ~line 356; new section after `runs_str`)
- Test: `tests/test_datagen.py`

- [ ] **Step 1: Write the failing tests** — in `tests/test_datagen.py`, add `from datetime import date` at the top of the imports, add `LARGE_DATASET_REGIMES`, `large_dataset`, `large_dataset_mixed` to the `from datagen import (...)` list (alphabetical order), and append:

```python
LARGE_DTYPES = (pl.Int32, pl.Float64, pl.String, pl.Date)


def test_large_dataset_has_the_former_files_schema():
    df = large_dataset()
    assert df.shape == (50_000, 101) and df.n_chunks() == 1
    assert df.columns == ["identity"] + [f"col_{j}" for j in range(100)]
    assert df.schema["identity"] == pl.Int64
    assert df["identity"].to_numpy().tolist() == list(range(50_000))
    for j in range(100):
        assert df.schema[f"col_{j}"] == LARGE_DTYPES[j % 4], j


def test_large_dataset_defaults():
    df = large_dataset()
    assert sum(df.null_count().row(0)) == 0
    counts = [df[f"col_{4 * g}"].n_unique() for g in range(25)]
    assert counts[:9] == [100, 20, 10, 7, 5, 4, 3, 3, 2] and counts[-1] == 1
    for g in range(25):
        assert df[f"col_{4 * g + 3}"].n_unique() == counts[g]
        assert df[f"col_{4 * g + 1}"].n_unique() == 50_000
        strings = df[f"col_{4 * g + 2}"]
        assert strings.n_unique() > 40_000
        assert strings.str.len_bytes().unique().to_list() == [3]
    assert df["col_3"].min() == date(2017, 1, 1)
    assert not df["col_1"].is_sorted()


def test_large_dataset_is_seeded_and_settings_reroll_only_their_step():
    assert large_dataset.__wrapped__().equals(large_dataset())  # uncached rebuild
    assert not large_dataset(seed=1).equals(large_dataset())
    plain, nulls = large_dataset(), large_dataset(null_rate=0.1)
    for c in plain.columns:
        keep = nulls[c].is_not_null()
        assert plain[c].filter(keep).equals(nulls[c].filter(keep)), c


def test_large_dataset_null_rate():
    df = large_dataset(null_rate=0.1)
    assert df["identity"].null_count() == 0
    three_sigma = 3 * (0.1 * 0.9 / 50_000) ** 0.5
    for c in df.columns[1:]:
        assert abs(df[c].null_count() / 50_000 - 0.1) < three_sigma, c
    # Null slots hold no bytes (zip_with), as an Arrow builder would leave them.
    s = df["col_2"]
    data = s.to_arrow(compat_level=pl.CompatLevel.oldest()).buffers()[2]
    assert data.size == s.str.len_bytes().sum()


def test_large_dataset_mean_run():
    df = large_dataset(mean_run=50)
    for c in ("col_1", "col_2"):  # high cardinality: adjacent runs rarely merge
        n_runs = df[c].ne_missing(df[c].shift(1)).sum()
        assert 50_000 / n_runs == pytest.approx(50, rel=0.1), c


def test_large_dataset_clustering():
    df = large_dataset(clustering=1.0)
    assert all(df[c].is_sorted() for c in df.columns)
    half = large_dataset(clustering=0.5)["col_1"]
    assert not half.is_sorted()
    assert (half == half.sort()).sum() >= 25_000


@pytest.mark.parametrize(
    "bad",
    [{"mean_run": 0}, {"null_rate": -0.1}, {"null_rate": 1.5}, {"clustering": 2.0}],
)
def test_large_dataset_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        large_dataset(n_rows=10, **bad)


def test_large_dataset_mixed_cycles_the_regimes():
    mixed = large_dataset_mixed()
    assert mixed["identity"].equals(large_dataset()["identity"])
    for g in range(25):
        regime = LARGE_DATASET_REGIMES[g % 5]
        cols = [f"col_{4 * g + k}" for k in range(4)]
        assert mixed.select(cols).equals(large_dataset(**regime).select(cols)), g
    assert mixed["col_5"].null_count() > 0 and mixed["col_1"].null_count() == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_datagen.py -n 0 -q`
Expected: collection error `ImportError: cannot import name 'LARGE_DATASET_REGIMES' from 'datagen'`.

- [ ] **Step 3: Factor `_run_lengths` out of `runs_int`** — in `tests/datagen.py`, insert above `def runs_int(`:

```python
def _run_lengths(n_rows: int, mean_run: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric run lengths with mean `mean_run`, summing to at least `n_rows`."""
    lengths = rng.geometric(1 / mean_run, size=int(n_rows // mean_run) + 64)
    while lengths.sum() < n_rows:
        more = rng.geometric(1 / mean_run, size=int(n_rows // mean_run) + 64)
        lengths = np.concatenate([lengths, more])
    return lengths
```

and replace the body of the `for` loop in `runs_int` (the `lengths = ...` / `while` / `values` / `columns[...]` lines) with the same draws in the same order:

```python
    for i in range(n_cols):
        lengths = _run_lengths(n_rows, mean_run, rng)
        values = rng.integers(0, cardinality, lengths.size, dtype=np.int64)
        columns[f"{prefix}{i}"] = np.repeat(values, lengths)[:n_rows]
```

- [ ] **Step 4: Add the generator** — in `tests/datagen.py`, add `import string` and `from functools import cache` to the imports (stdlib block, alphabetical), and insert this section after `runs_str` (before `# ── per-column (describe)`):

```python
# ── large dataset (agreement / parity fixture, benchmarks) ────────────────────

LARGE_DATASET_GROUPS = 25
LARGE_DATASET_REGIMES = (  # large_dataset_mixed: group g uses regime g mod 5
    {"null_rate": 0.0, "mean_run": 1, "clustering": 0.0},  # plain
    {"null_rate": 0.1, "mean_run": 1, "clustering": 0.0},  # nulls
    {"null_rate": 0.0, "mean_run": 50, "clustering": 0.0},  # runs
    {"null_rate": 0.0, "mean_run": 1, "clustering": 1.0},  # sorted
    {"null_rate": 0.1, "mean_run": 50, "clustering": 0.9},  # combined
)
_LETTERS = np.frombuffer(string.ascii_letters.encode(), dtype=np.uint8)
_DAYS_TO_2017 = 17_167  # 1970-01-01 -> 2017-01-01


def _group_cardinality(g: int) -> int:
    """Int32 / Date distinct values of group g: 100, 20, 10, 7, 5, 4, 3, 3, 2, …, 1."""
    return 100 if g == 0 else max(1, round(20 / g))


def _draw(kind: int, card: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k values for a group's column `kind`: 0 Int32, 1 Float64, 2 String (as S3 bytes),
    3 Date (as Int32 days since the epoch)."""
    if kind == 0:
        return rng.integers(0, card, k, dtype=np.int32)
    if kind == 1:
        return rng.random(k)
    if kind == 2:
        return _LETTERS[rng.integers(0, _LETTERS.size, (k, 3))].view("S3").ravel()
    return rng.integers(0, card, k, dtype=np.int32) + np.int32(_DAYS_TO_2017)


def _large_column(
    name: str,
    kind: int,
    card: int,
    n_rows: int,
    rng: np.random.Generator,
    null_rate: float,
    mean_run: float,
    clustering: float,
) -> pl.Series:
    """Values in geometric runs, then the clustering shuffle, then nulls — drawn in that
    order, so changing a setting re-rolls only its own step and the later ones."""
    lengths = _run_lengths(n_rows, mean_run, rng)
    values = np.repeat(_draw(kind, card, lengths.size, rng), lengths)[:n_rows]
    if clustering > 0:  # 0 keeps the drawn order, runs included
        values = np.sort(values)
        moved = rng.choice(n_rows, round((1 - clustering) * n_rows), replace=False)
        values[moved] = values[rng.permutation(moved)]
    s = pl.Series(name, values)
    if kind == 2:
        s = s.cast(pl.String)
    elif kind == 3:
        s = s.cast(pl.Date)
    if null_rate > 0:
        # zip_with leaves null slots empty; pl.when and Series.set keep the old bytes,
        # which the one-shot recommender measures and the streaming one does not.
        keep = pl.Series(rng.random(n_rows) >= null_rate)
        s = s.zip_with(keep, pl.repeat(None, n_rows, dtype=s.dtype, eager=True))
    return s


def _check_large_settings(null_rate: float, mean_run: float, clustering: float) -> None:
    if mean_run < 1 or not 0 <= null_rate <= 1 or not 0 <= clustering <= 1:
        raise ValueError(
            "need mean_run >= 1 and null_rate, clustering in [0, 1]; got "
            f"{mean_run=}, {null_rate=}, {clustering=}"
        )


def _large(n_rows: int, seed: int, regimes: tuple[dict, ...]) -> pl.DataFrame:
    for regime in regimes:
        _check_large_settings(**regime)
    seeds = np.random.SeedSequence(seed).spawn(4 * LARGE_DATASET_GROUPS)
    columns = [pl.Series("identity", np.arange(n_rows, dtype=np.int64))]
    for g in range(LARGE_DATASET_GROUPS):
        regime = regimes[g % len(regimes)]
        for kind in range(4):
            j = 4 * g + kind
            rng = np.random.default_rng(seeds[j])
            card = _group_cardinality(g)
            columns.append(
                _large_column(f"col_{j}", kind, card, n_rows, rng, **regime)
            )
    return pl.DataFrame(columns)


@cache
def large_dataset(
    n_rows: int = 50_000,
    seed: int = 0,
    null_rate: float = 0.0,
    mean_run: float = 1,
    clustering: float = 0.0,
) -> pl.DataFrame:
    """The shared large frame (spec 2026-10-17-generated-dataset-design.md): `identity`
    (Int64 0..n) then 25 groups of (Int32, Float64, 3-letter String, Date), Int32 / Date
    distinct counts stepping down from 100 to 1. Settings apply to every column but
    `identity`: `mean_run` (geometric run length), `clustering` (1 sorted, 0 drawn
    order, in between: a share 1 − clustering of positions shuffled after sorting),
    `null_rate` (Bernoulli). Cached: callers must not modify the frame in place."""
    regime = {"null_rate": null_rate, "mean_run": mean_run, "clustering": clustering}
    return _large(n_rows, seed, (regime,))


@cache
def large_dataset_mixed(n_rows: int = 50_000, seed: int = 0) -> pl.DataFrame:
    """large_dataset with LARGE_DATASET_REGIMES cycled over the 25 groups: group g equals
    group g of large_dataset(n_rows, seed, **LARGE_DATASET_REGIMES[g % 5])."""
    return _large(n_rows, seed, LARGE_DATASET_REGIMES)
```

- [ ] **Step 5: Update the module docstring** — in the list at the top of `tests/datagen.py`, after the `Per-column, ordered (runs test): runs_int, runs_str` line add:

```
Shared large frame (agreement / parity tests, benchmarks): large_dataset, large_dataset_mixed
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_datagen.py -n 0 -q`
Expected: all pass (the 11 new ones included; `test_runs_generators_plant_mean_run_length` still passes — `runs_int` draws are unchanged).

- [ ] **Step 7: Format, lint, commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add tests/datagen.py tests/test_datagen.py
git commit -m "datagen: large_dataset / large_dataset_mixed generator (ToDo 6)"
```

---

### Task 2: Tests read the generator; agreement and parity also on the mixed preset

The prototype of this task was run before the plan was written: every implementation agrees with its reference on both frames (0.5–6.5 s per check) and streaming parity holds on both. If anything fails here, it is a real finding — stop and use superpowers:systematic-debugging; do not loosen a tolerance.

**Files:**
- Modify: `tests/test_runs_test.py`, `tests/test_minmax_index.py`, `tests/test_set_index.py`, `tests/test_describe.py`, `tests/test_recommend.py`, `tests/test_streaming_recommend.py`

- [ ] **Step 1: `tests/test_runs_test.py`** — delete the `LARGE = Path(...)` line and `from pathlib import Path` (its only other use); change the datagen import to `from datagen import large_dataset, large_dataset_mixed, mixed_dtypes, runs_int, runs_str`; replace the large-dataset test with:

```python
@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_with_reference_large_dataset(impl, make):
    frames = {"large": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))
```

- [ ] **Step 2: `tests/test_minmax_index.py`** — same import / `LARGE` / `Path` changes; replace the test with:

```python
@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("params", [{}, {"index_granularity": 1_024, "granularity": 2}])
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_with_reference_large_dataset(impl, params, make):
    frames = {"large": make()}
    cls = load(impl)
    assert_agrees(
        cls(**params), run(cls, frames, **params), run(reference(PKG), frames, **params)
    )
```

- [ ] **Step 3: `tests/test_set_index.py`** — same import / `LARGE` / `Path` changes; keep the test's existing `params` parametrize decorator(s) above it unchanged, add the `make` parametrize below them, and change the signature and body to:

```python
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_with_reference_large_dataset(impl, params, make):
    agree(impl, {"large": make()}, **params)
```

- [ ] **Step 4: `tests/test_describe.py`** — delete the `LARGE = ...` line (keep `from pathlib import Path`: it has another use); import `describe_mixed, large_dataset, large_dataset_mixed, stringified` from datagen; replace the slow agreement test with:

```python
@pytest.mark.slow
@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_agrees_with_reference_on_large_dataset(impl, make):
    frames = {"large": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))
```

and in `test_stringified_large_dataset_columns` replace `large = pl.read_ipc(LARGE).head(5_000)` with `large = large_dataset().head(5_000)`.

- [ ] **Step 5: `tests/test_recommend.py`** — delete `LARGE = ...` and `from pathlib import Path`; import `describe_mixed, large_dataset, stringified` from datagen; in `ORACLE_FRAMES` replace `lambda: {"large": pl.read_ipc(LARGE)}` with `lambda: {"large": large_dataset()}`.

- [ ] **Step 6: `tests/test_streaming_recommend.py`** — delete `LARGE = ...` and `from pathlib import Path`; import `describe_mixed, large_dataset, large_dataset_mixed, stringified` from datagen; replace the parity test with:

```python
@pytest.mark.slow
@pytest.mark.parametrize("batch_rows", [7_919, 50_000])
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_parity_on_the_large_dataset(batch_rows, make):
    frame = make()
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)
```

and in `test_sampled_zstd_estimate_is_within_ten_percent` replace `frame = pl.read_ipc(LARGE)` with `frame = large_dataset()`. (Not the mixed preset: sampled blocks of sorted / run columns are not representative — 22% error measured — and the test does not claim they are.)

- [ ] **Step 7: Confirm no test reads the file**

Run: `grep -rn "large_dataset.arrow\|LARGE\b" tests/*.py`
Expected: no output.

- [ ] **Step 8: Run the touched tests, slow ones included**

Run: `$PY -m pytest tests/test_runs_test.py tests/test_minmax_index.py tests/test_set_index.py tests/test_describe.py tests/test_recommend.py tests/test_streaming_recommend.py -q --durations=10`
Expected: all pass. Note the slowest durations; the new `mixed` cases should each be under ~10 s.

- [ ] **Step 9: Lint, commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add tests/test_runs_test.py tests/test_minmax_index.py tests/test_set_index.py tests/test_describe.py tests/test_recommend.py tests/test_streaming_recommend.py
git commit -m "tests: large-dataset fixtures from datagen; agreement and parity on the mixed preset"
```

---

### Task 3: Benchmark harness reads the generator

**Files:**
- Modify: `tests/performance/harness.py:41-71`

- [ ] **Step 1: Replace the file constant and `large_dataset`** — delete the `LARGE_DATASET = ...` line; after `import analytics  # noqa: E402` add:

```python
from datagen import large_dataset as generated_large_dataset  # noqa: E402
```

and replace `large_dataset` with:

```python
def large_dataset(columns: int | None = None, exclude: tuple[str, ...] = ()) -> Dataset:
    """datagen.large_dataset() (50K rows × 101 cols, generated), optionally only its first `columns`."""
    label = "large_dataset" + (f" (first {columns} cols)" if columns else "")

    def make() -> dict[str, pl.DataFrame]:
        df = generated_large_dataset()
        return {"large": df.select(df.columns[:columns]) if columns else df}

    return Dataset(label, make, exclude)
```

- [ ] **Step 2: Smoke-run one benchmark** (child processes import the harness too)

Run: `$PY tests/performance/benchmark_runs_test.py`
Expected: a table with a `large_dataset` row for each implementation, `agrees` yes, and `saved ...runs_test_<stamp>.parquet` (Parquet until Task 4).

- [ ] **Step 3: Local, git-ignored script** — if `tests/performance/results/boundary_timing.py` exists, replace its `DATA = ...` line and `df = pl.read_ipc(DATA)` with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tests/ -> datagen
from datagen import large_dataset  # noqa: E402
```

and `df = large_dataset()`. Nothing to commit for this file.

- [ ] **Step 4: Commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add tests/performance/harness.py
git commit -m "bench: harness large_dataset() from datagen"
```

---

### Task 4: Benchmark results as Arrow IPC

`.gitignore` already ignores all of `tests/performance/results/`; no change needed there.

**Files:**
- Modify: `tests/performance/harness.py` (`_report`), `tests/performance/benchmark_recommend.py`, `tests/performance/benchmark_top_k_accuracy.py`, `tests/performance/benchmark_ams_entropy.py`

- [ ] **Step 1: `harness._report`** — replace

```python
    path = RESULTS_DIR / f"{technique}_{stamp}.parquet"
    frame.write_parquet(path)
```

with

```python
    path = RESULTS_DIR / f"{technique}_{stamp}.arrow"
    frame.write_ipc(path, compression="zstd")
```

- [ ] **Step 2: `benchmark_recommend.py`** — replace the two writes with

```python
    pl.DataFrame(rows).write_ipc(out / "recommend.arrow", compression="zstd")
    if orders:
        pl.DataFrame(orders).write_ipc(
            out / "recommend_dictionary_order.arrow", compression="zstd"
        )
```

and in the docstring change `results/recommend_dictionary_order.parquet` → `results/recommend_dictionary_order.arrow` and `results/recommend.parquet` → `results/recommend.arrow`.

- [ ] **Step 3: `benchmark_top_k_accuracy.py`** — in `child`: `result.select("top_k", "sketch_bytes").write_ipc(out)`; in `main`: `out = str(Path(tmp) / "out.arrow")`, `result = pl.read_ipc(out, memory_map=False)`, and

```python
    df.write_ipc(
        RESULTS_DIR / ("top_k_scale.arrow" if scale else "top_k_accuracy.arrow"),
        compression="zstd",
    )
```

Docstring: `top_k_scale.parquet` → `top_k_scale.arrow`, `top_k_accuracy.parquet` → `top_k_accuracy.arrow`. (`memory_map=False` so the temporary directory can be deleted on Windows while the frame is alive.)

- [ ] **Step 4: `benchmark_ams_entropy.py`** — in `child`: `result.write_ipc(out)`; in `main`: `out = str(Path(tmp) / f"{name}.arrow")`, `results[name] = pl.read_ipc(out, memory_map=False)`, and `df.write_ipc(RESULTS_DIR / "ams_entropy.arrow", compression="zstd")`. Docstring: `ams_entropy.parquet` → `ams_entropy.arrow`.

- [ ] **Step 5: Verify nothing writes Parquet**

Run: `grep -rn "parquet" tests/ --include=*.py`
Expected: no output.

- [ ] **Step 6: Smoke-run**

Run: `$PY tests/performance/benchmark_gcd.py`
Expected: ends with `saved ...gcd_<stamp>.arrow`; `$PY -c "import polars as pl, glob; print(pl.read_ipc(sorted(glob.glob('tests/performance/results/gcd_*.arrow'))[-1]).shape)"` prints a shape.

- [ ] **Step 7: Convert the existing local results once** (not committed; the directory is git-ignored)

```bash
$PY - <<'EOF'
from pathlib import Path
import polars as pl
for p in Path("tests/performance/results").glob("*.parquet"):
    pl.read_parquet(p).write_ipc(p.with_suffix(".arrow"), compression="zstd")
    p.unlink()
    print("converted", p.name)
EOF
```

Expected: one `converted` line per former `.parquet` file; `ls tests/performance/results/*.parquet` then finds nothing.

- [ ] **Step 8: Commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add tests/performance/harness.py tests/performance/benchmark_recommend.py tests/performance/benchmark_top_k_accuracy.py tests/performance/benchmark_ams_entropy.py
git commit -m "bench: results and child hand-off as Arrow IPC (ToDo 5g)"
```

---

### Task 5: polars without default features

Warning: changing `Cargo.toml` changes `Cargo.lock`, so the next CI run starts from a cold Rust cache (~3x the usual minutes). That is expected once.

**Files:**
- Modify: `services/analytics/Cargo.toml:19`, `services/analytics/Cargo.lock`

- [ ] **Step 1: Record the current feature set and a baseline build time**

```bash
cd services/analytics
export PATH="/c/Users/Alexander/.cargo/bin:$PATH"
feats() { for c in polars-core polars-arrow polars-compute polars-ops polars-io polars-time polars-lazy; do cargo tree -e features -i $c 2>/dev/null | grep -o "$c feature \"[^\"]*\"" | sort -u; done; }
feats > target/polars_features_before.txt
wc -l target/polars_features_before.txt        # expected: 147
cargo tree -i polars-parquet | head -1       # expected: polars-parquet v0.51.0
```

Baseline cold build of the Python-free C library, in a throwaway target dir (run it in the background; ~10–20 min):

```bash
time cargo build --profile ci --no-default-features --target-dir target/tb-before 2>&1 | tail -1
```

Record the `real` time.

- [ ] **Step 2: Edit the dependency** — replace line 19 of `services/analytics/Cargo.toml` with:

```toml
polars = { version = "0.51.0", default-features = false, features = ["lazy", "performant", "dtype-slim", "dtype-i8", "dtype-i16", "dtype-time", "dtype-decimal", "dtype-struct", "dtype-array", "dtype-categorical", "temporal", "timezones", "fmt", "zip_with"] }
```

- [ ] **Step 3: Check Parquet is gone and no needed feature was lost**

```bash
cargo tree -i polars-parquet 2>&1 | head -1    # expected: "warning: nothing to print."
feats > target/polars_features_after.txt
diff target/polars_features_before.txt target/polars_features_after.txt
```

Expected diff — exactly these removals (`<`) and additions (`>`), nothing else (grouped by crate here; `diff` prints one line per feature, e.g. `< polars-core feature "docs"`):

```
< polars-core: docs, partition_by, serde, serde-lazy, serde_json
< polars-arrow: arrow-format, io_ipc, io_ipc_compression, lz4, serde, zstd
< polars-compute: approx_unique, serde
< polars-ops: peaks, rle, unique_counts
< polars-io: async, async-trait, cloud, file_cache, futures, http, object_store, polars-parquet, reqwest, serde, serde_json, tokio, tokio-util, url
< polars-lazy: csv, new_streaming, polars-stream
> polars-ops: dtype-i16, dtype-i8, dtype-time
> polars-io: dtype-i16, dtype-i8
> polars-time: dtype-i16, dtype-i8
```

Any other `<` line (in particular a `dtype-*`, `temporal` or `timezones` feature) means a feature must be added back to the list in Step 2.

- [ ] **Step 4: Rust unit tests**

```bash
export PYO3_PYTHON=/c/Users/Alexander/miniconda3/envs/p312/python.exe
export PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH"
cargo test --lib 2>&1 | tail -3
```

Expected: `test result: ok.` (Without `dtype-i8` / `dtype-i16` the gcd and joint_entropy tests would not compile — that is how the gap was found.)

- [ ] **Step 5: Rebuild the extension and run the Python suite**

```bash
cd ../.. && $PY scripts/check.py build && $PY scripts/check.py test-py
```

Expected: build succeeds; pytest all pass.

- [ ] **Step 6: Java binding (C library build + JUnit)**

Run: `$PY scripts/check.py test-java`
Expected: `BUILD SUCCESS`.

- [ ] **Step 7: Cold build time after**

```bash
cd services/analytics && time cargo build --profile ci --no-default-features --target-dir target/tb-after 2>&1 | tail -1
rm -rf target/tb-before target/tb-after target/polars_features_*.txt
```

Record the `real` time; Task 7 writes both figures into CLAUDE.md.

- [ ] **Step 8: Commit** (`Cargo.lock` shrinks: the cloud stack and polars-parquet drop out)

```bash
cd ../.. && git add services/analytics/Cargo.toml services/analytics/Cargo.lock
git commit -m "build: polars without default features — no polars-parquet, no cloud stack (ToDo 5g)"
```

---

### Task 6: Parquet guard

**Files:**
- Modify: `tests/test_layering.py`

- [ ] **Step 1: Write the guard tests** — add `import re` and `import subprocess` to the imports of `tests/test_layering.py`, extend its module docstring with `Also: no Parquet anywhere (CLAUDE.md § Project Objectives, Formats).`, and append:

```python
ROOT = HERE.parent
PARQUET = re.compile(
    r"read_parquet|write_parquet|scan_parquet|sink_parquet|pyarrow\.parquet|\.parquet[\"']"
)


def parquet_uses(paths: list[Path]) -> list[str]:
    """`file:line` for every line that reads or writes Parquet."""
    return [
        f"{p.name}:{i}"
        for p in paths
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if PARQUET.search(line)
    ]


def test_parquet_uses_are_found(tmp_path):
    files = tree(
        tmp_path,
        {
            "a.py": 'df.write_parquet(out)\nprint("no Parquet here")\n',
            "b.py": 'import pyarrow.parquet as pq\nx = "r.parquet"\n',
        },
    )
    assert parquet_uses(sorted(files.glob("*.py"))) == ["a.py:1", "b.py:1", "b.py:2"]


def test_no_python_file_uses_parquet():
    tracked = subprocess.run(
        ["git", "ls-files", "--", "tests/*.py", "services/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = [ROOT / t for t in tracked if ROOT / t != Path(__file__).resolve()]
    assert paths and parquet_uses(paths) == []


def test_polars_crate_builds_without_parquet():
    cargo = (ROOT / "services" / "analytics" / "Cargo.toml").read_text(encoding="utf-8")
    polars = next(line for line in cargo.splitlines() if line.startswith("polars ="))
    assert "default-features = false" in polars
    assert "parquet" not in cargo
```

(`git ls-files` limits the scan to tracked files, so git-ignored local scripts under `tests/performance/results/` and build output are not scanned; this file is excluded because it contains the pattern.)

- [ ] **Step 2: Run them**

Run: `$PY -m pytest tests/test_layering.py -n 0 -q`
Expected: all pass (Tasks 4 and 5 already removed every use). To see the guard bite, temporarily add `# df.write_parquet("x")` to `tests/datagen.py`, rerun (expected: `test_no_python_file_uses_parquet` FAILS listing `datagen.py:<line>`), then remove the line again.

- [ ] **Step 3: Commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add tests/test_layering.py
git commit -m "test: guard against Parquet in Python sources and the polars features"
```

---

### Task 7: Delete the data file; docs and numbers

**Files:**
- Delete: `tests/data/large_dataset.arrow`
- Modify: `services/analytics/analytics/similarity/rust.py:45-47`, `CLAUDE.md`, `ToDo.md`

- [ ] **Step 1: Nothing references the file any more**

Run: `grep -rn "large_dataset.arrow\|tests/data" --include=*.py --include=*.rs --include=*.java --include=*.yml --include=*.toml --include=*.ini --include=*.cfg . | grep -v "^./target\|^./docs/"`
Expected: only the `similarity/rust.py` comment (fixed in Step 3).

- [ ] **Step 2: Delete it**

```bash
git rm tests/data/large_dataset.arrow
```

(`tests/data/` disappears with its only file.)

- [ ] **Step 3: Re-measure MinHash recall**

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0, "tests")
import polars as pl
from datagen import large_dataset
from analytics.similarity import MinHashRust, SimilarityExactLRU
frames = {"large": large_dataset()}
got, ref = MinHashRust().add(frames).result(), SimilarityExactLRU().add(frames).result()
def passing(df):
    flags = pl.col("passes_jaccard").fill_null(False) | pl.col("passes_overlap").fill_null(False)
    return set(df.with_row_index().filter(flags)["index"].to_list())
t, p = passing(ref), passing(got)
missed = ref.with_row_index().filter(pl.col("index").is_in(list(t - p)))
print(f"recall {len(t & p) / len(t):.2f} ({len(t & p)}/{len(t)}), misses {len(t - p)}, "
      f"overlap == 1: {(missed['overlap'] == 1.0).sum()}, jaccard {missed['jaccard'].min():.5f}"
      f"..{missed['jaccard'].max():.2f} median {missed['jaccard'].median():.2f}")
EOF
```

Expected (measured while planning): `recall 0.65 (407/625), misses 218, overlap == 1: 218, jaccard 0.00002..0.33 median 0.10`. Use the printed figures below if they differ.

- [ ] **Step 4: `similarity/rust.py`** — replace the comment's last sentence

```python
        # cardinality differences. Measured on large_dataset.arrow: recall 0.62
        # (389/625), all 236 misses are containment pairs with Jaccard from 0.00002 to
        # 0.36 (median 0.10) that never became LSH candidates.
```

with

```python
        # cardinality differences. Measured on datagen.large_dataset(): recall 0.65
        # (407/625), all 218 misses are containment pairs with Jaccard from 0.00002 to
        # 0.33 (median 0.10) that never became LSH candidates.
```

- [ ] **Step 5: CLAUDE.md** — make these edits:

1. Runs test outcome (§1b): `total speedup 16.9x on large_dataset.arrow` → `total speedup 16.9x on the former large_dataset.arrow`.
2. Project Structure, tests tree: delete the line `├── data/large_dataset.arrow            # 50K rows, 101 columns`, and change the datagen line to
   `├── datagen.py                          # seeded generators shared with benchmarks; large_dataset / large_dataset_mixed (50K × 101)`.
3. Testing Convention, benchmarks: `on\n  large_dataset.arrow + a narrow/long + a wide shape` → `on\n  `datagen.large_dataset()` (50K × 101, generated) + a narrow/long + a wide shape`, and `Results → `tests/performance/results/*.parquet` (git-ignored; to move to Arrow IPC, ToDo 5g).` → `Results → `tests/performance/results/*.arrow` (Arrow IPC, ZSTD; git-ignored).`
4. Testing Convention, accuracy tests: after item 4 of the four blocks, add the line
   `  Large-frame agreement / parity tests run on `datagen.large_dataset()` and `large_dataset_mixed()` (nulls, runs, sorted groups; spec 2026-10-17).`
5. Checks, CI: `Cold build ≈ 20 min` → the measured figure from Task 5 Step 7 if it moved by more than a minute (e.g. `Cold build ≈ N min`), and add after the Cargo sentence in "Rust Extension": `polars is built without default features (no polars-parquet, no cloud stack); the guard is tests/test_layering.py.`
6. Current Focus: `Next, in order (ToDo.md): ToDo 6 + 5g (generate large_dataset.arrow with null / run / clustering settings; benchmark results to Arrow IPC) → ToDo 1.0` → `Next, in order (ToDo.md): ToDo 1.0`.
7. MinHash follow-up: `MinHash recall on large_dataset.arrow is 0.62 (389/625 passing pairs)` → `MinHash recall on datagen.large_dataset() is 0.65 (407/625 passing pairs)`.
8. Bloom follow-up: `0.60x algorithmic on large_dataset.arrow` → `0.60x algorithmic on the former large_dataset.arrow`.

- [ ] **Step 6: ToDo.md**

1. Agreed starting order line → `**Agreed starting order:** 1.0 (spec) → 4 + 2 (the spec's first increment) → 1a–g (6 + 5g done).  Do 5c (rename) just before the first new Python package (ToDo 3 or 8).`
2. Replace the body of item 5g with `g. **Done** (spec 2026-10-17-generated-dataset-design.md): benchmark results are Arrow IPC; polars is built without default features (no polars-parquet); tests/test_layering.py guards both.`
3. Replace item 6 and its bullets with:
   ```markdown
   6. **Done** (spec 2026-10-17-generated-dataset-design.md): `datagen.large_dataset(n_rows, seed, null_rate, mean_run, clustering)` and `large_dataset_mixed()` replace tests/data/large_dataset.arrow.  The git pack still holds the old blob (history not rewritten).
   ```
4. Append to item 7:
   ```markdown
   - Recommender parity, string columns whose null slots hold bytes (e.g. built with `pl.when(...).then(None)` or `Series.set`): one-shot measures those bytes in its cast candidates (`rec_arrow_size_bytes`), streaming does not; `assert_parity` excuses the gap only for the original type.  Decide which is right (an Arrow cast keeps the bytes; a writer may not) and align them.
   ```

- [ ] **Step 7: Commit**

```bash
$PY scripts/check.py fmt && $PY scripts/check.py lint
git add -A tests/data services/analytics/analytics/similarity/rust.py CLAUDE.md ToDo.md
git commit -m "Remove tests/data/large_dataset.arrow; docs and re-measured MinHash recall (ToDo 6)"
```

- [ ] **Step 8: Repository housekeeping**

```bash
git gc
git count-objects -vH
```

Expected: no `no corresponding .pack` warnings; `size-pack` still ≈ 392 MiB (history not rewritten).

---

### Task 8: Full verification

- [ ] **Step 1: Everything CI runs**

Run: `$PY scripts/check.py ci`
Expected: fast, build, Rust unit tests, pytest and Java all pass.

- [ ] **Step 2: Slow tests explicitly** (if `check` deselects `slow`)

Run: `$PY -m pytest -m slow -q`
Expected: all pass, `mixed` variants included.

- [ ] **Step 3: Final spec check** — confirm against the spec: §2 (generator, settings, determinism, mixed), §3 (consumers, numbers, removal), §4 (IPC results, Cargo, guard), §5 (gc, no rewrite), §6 (tests), §7 (docs). Report any deviation to the user rather than fixing silently.
