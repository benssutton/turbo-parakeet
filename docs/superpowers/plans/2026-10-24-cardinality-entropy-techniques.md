# Distinct-count, Population-cardinality and Marginal-entropy Techniques Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three new per-column techniques on the uniform contract (`distinct_count`, `population_cardinality`, `marginal_entropy`), each with accuracy tests and a benchmark on the uplifted harness; the harness learns known-truth datasets.

**Architecture:** `marginal_entropy` is Python-only (both Rust bindings exist). `distinct_count` and `population_cardinality` each get a thin Rust kernel file wrapping existing code (`hll.rs`; Describe's `frequencies.rs` + `cardinality_estimators.rs`), an `api` entry point, a pyo3 function and a private `_plugin` wrapper. Population estimates are scored against the reference and, via a new `Dataset.truth`, against the population's real distinct count.

**Tech Stack:** Python 3.12, Polars, NumPy, pytest; Rust (pyo3, polars, rayon) built with maturin.

**Spec:** `docs/superpowers/specs/2026-10-24-cardinality-entropy-techniques-design.md`

---

## Conventions for every task

- Repo root `C:/Users/Alexander/turbo-parakeet`, branch `cardinality-entropy-techniques` (do not switch).
- Python: `C:/Users/Alexander/miniconda3/envs/p312/python.exe` (written `PY`). pytest: `-n 0` for targeted runs.
- Python-only changes need no rebuild. **Rust changes** need `maturin develop --release` from `services/analytics/` (≈ 2.5 min warm): `cd services/analytics && PY -m maturin develop --release`.
- Rust unit tests: from `services/analytics/`, with `PYO3_PYTHON=C:/Users/Alexander/miniconda3/envs/p312/python.exe` and the env dir on PATH: `export PATH="/c/Users/Alexander/miniconda3/envs/p312:$PATH"; PYO3_PYTHON=... cargo test --lib <module>::`.
- Commits (Git Bash): `export PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"` first (the pre-commit hook needs it). End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Files under `docs/superpowers/` are git-ignored: `git add -f`.
- Format: `black` on touched Python, `cargo fmt` on touched Rust.

## Spec deviations decided while planning (apply them; Task 9 records them in the spec)

1. **Eligibility** for `distinct_count` and `population_cardinality`: `encodable(dtype) and not is_nested(dtype) and series.len() > 0` (`turbo_parakeet._dtypes`). Binary is not eligible: the Rust encoder (`encode_series`) does not key it (spec §2 said "as Top-k", whose Rust kernels have their own encoding).
2. The population-cardinality reference is **`PopulationCardinalityPolars`**: counts and capture masks by a Polars `group_by` (with `bitwise_or`), the split subsets by SplitMix64 in NumPy uint64, the formulas from `describe/estimators.py`. Polars does the grouping because `np.unique` does not handle strings / Decimal / temporal types uniformly.
3. `Dataset.truth` is passed to `accuracy(..., truth=...)` **only when set** (keyword), so existing `accuracy` overrides (pairwise entropy, top-k) need no change.

## File map

| File | Change |
|---|---|
| `tests/performance/harness.py` | `Dataset.truth`; pass it to the reference comparison |
| `tests/performance/fake_technique.py`, `tests/test_benchmark_harness.py` | truth test |
| `services/analytics/python/turbo_parakeet/techniques/base.py` | `accuracy(..., truth=None)` |
| `tests/datagen.py`, `tests/test_datagen.py` | `population_samples` |
| `services/analytics/python/turbo_parakeet/techniques/marginal_entropy/{__init__,base,rust,ams,polars}.py` | new package |
| `services/analytics/src/techniques/single_column/unordered/{distinct_count,population_cardinality}.rs` | new kernels |
| `services/analytics/src/techniques/single_column/unordered/{mod.rs,describe/mod.rs}`, `src/techniques/mod.rs` | module wiring (`frequencies` becomes `pub(crate)`) |
| `services/analytics/src/bindings/api/techniques.rs`, `src/bindings/python.rs`, `python/turbo_parakeet/_plugin.py` | entry points |
| `services/analytics/python/turbo_parakeet/techniques/distinct_count/{__init__,base,hll,polars}.py` | new package |
| `services/analytics/python/turbo_parakeet/techniques/population_cardinality/{__init__,base,rust,polars}.py` | new package |
| `tests/test_{marginal_entropy,distinct_count,population_cardinality}.py` | accuracy tests |
| `tests/performance/benchmark_{marginal_entropy,distinct_count,population_cardinality}.py` | benchmarks |
| `CLAUDE.md`, `ToDo.md`, `tests/performance/README.md`, the spec | docs |

---

### Task 1: Known-truth datasets in the harness

**Files:** Modify `tests/performance/harness.py`, `services/analytics/python/turbo_parakeet/techniques/base.py`, `tests/performance/fake_technique.py`, `tests/test_benchmark_harness.py`.

- [ ] **Step 1: Failing test.** In `tests/performance/fake_technique.py`, add to class `FakeSum` (all implementations inherit it):

```python
    def accuracy(self, frames, result, reference, truth=None):
        """The default figures, plus `truth_rel_err` (max |total − truth| / truth) when the
        dataset carries a truth."""
        out = super().accuracy(frames, result, reference)
        if truth is not None:
            got = dict(zip(result["col_a"], result["total"]))
            out["truth_rel_err"] = max(abs(got[c] - t) / t for (_, c), t in truth.items())
        return out
```

In `tests/test_benchmark_harness.py`, change the `bench` fixture's dataset to carry a truth (the exact column sums of the fixture frame: a = 6.0, b = 60.0):

```python
            [
                Dataset(
                    "tiny",
                    lambda: {"t": pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})},
                    truth=lambda: {("t", "a"): 6.0, ("t", "b"): 60.0},
                )
            ],
```

and append:

```python
@pytest.mark.slow
def test_truth_reaches_the_accuracy_hook(bench):
    assert _one(bench, "FakeSumRust", "lo", "N")["truth_rel_err"] == 0.0
    assert _one(bench, "FakeSumSketchRust", "hi", "N")["truth_rel_err"] == pytest.approx(0.2)
    # truth goes to the reference comparison only
    assert "exact_rust_truth_rel_err" not in bench.columns
```

- [ ] **Step 2:** `PY -m pytest tests/test_benchmark_harness.py -n 0 -q` → FAIL (`Dataset.__init__() got an unexpected keyword argument 'truth'`).

- [ ] **Step 3: Implement.**
  - `base.py`: `Technique.accuracy` signature becomes `accuracy(self, frames, result, reference, truth=None)`; docstring: "`truth` ((frame, column) → a known true value, from benchmark datasets that have one) is ignored by the default; overrides that can use it do."
  - `harness.py`:
    - `Dataset` gains, after `exclude`, `truth: Callable[[], dict[tuple[str, str], float]] | None = None` with the comment `# (frame, column) → known true value, passed to accuracy()`.
    - In `run()`, per dataset, `truth = ds.truth() if ds.truth else None`, passed to `_score(..., truth)`.
    - In `_score` / `_compare`: the reference comparison calls `impl.accuracy(frames, result, ref, **({"truth": truth} if truth is not None else {}))`; the exact-Rust comparison never passes truth.
    - Add one docstring sentence: "A `Dataset.truth` reaches the reference comparison's `accuracy(..., truth=...)`."

- [ ] **Step 4:** `PY -m pytest tests/test_benchmark_harness.py tests/test_base.py -n 0 -q` → PASS.

- [ ] **Step 5: Commit** — `harness: known-truth datasets (Dataset.truth → accuracy(truth=...))`.

---

### Task 2: `datagen.population_samples`

**Files:** Modify `tests/datagen.py` (add after `top_k_streams`; add `population_samples` to the module docstring's list as "Per-column (population cardinality): population_samples"), `tests/test_datagen.py`.

- [ ] **Step 1: Failing test** (append to `tests/test_datagen.py`, add `population_samples` to its import list):

```python
def test_population_samples():
    frame, truth = population_samples(
        200_000, [(1_000, "uniform"), (50_000, ("zipf", 1.1))], seed=3
    )
    assert frame.columns == ["p0", "p1"] and frame.height == 200_000
    assert truth == {"p0": 1_000, "p1": 50_000}
    assert frame["p0"].n_unique() == 1_000  # uniform, 200 draws per label: all seen
    assert frame["p1"].n_unique() < 50_000  # Zipf: rare labels unseen
    again, _ = population_samples(
        200_000, [(1_000, "uniform"), (50_000, ("zipf", 1.1))], seed=3
    )
    assert frame.equals(again)
```

- [ ] **Step 2:** `PY -m pytest tests/test_datagen.py -n 0 -q -k population` → FAIL (ImportError).

- [ ] **Step 3: Implement** in `tests/datagen.py`:

```python
def population_samples(
    n_rows: int, populations: list[tuple[int, str | tuple[str, float]]], seed: int = 42
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Int64 columns p0… each drawing `n_rows` values with replacement from a population
    of exactly D labels (`populations`: (D, "uniform") or (D, ("zipf", alpha))), labels
    randomly permuted so frequent values are not small integers. Returns the frame and
    each column's D — the support size, the truth Chao1 / Schnabel estimate."""
    rng = np.random.default_rng(seed)
    cols, truth = {}, {}
    for i, (d, shape) in enumerate(populations):
        if shape == "uniform":
            ranks = rng.integers(0, d, n_rows)
        else:
            ranks = _bounded_zipf(n_rows, d, shape[1], rng)
        labels = rng.permutation(d).astype(np.int64)
        cols[f"p{i}"] = labels[ranks]
        truth[f"p{i}"] = d
    return pl.DataFrame(cols), truth
```

Check `_bounded_zipf`'s signature/return in `datagen.py` (`_bounded_zipf(n_rows, n_values, alpha, rng)`, ranks 0-based after its own permutation — read it; if its values are not 0-based ranks in [0, d), adapt by indexing accordingly so every value is one of the D labels).

- [ ] **Step 4:** `PY -m pytest tests/test_datagen.py -n 0 -q` → PASS.
- [ ] **Step 5: Commit** — `datagen: population_samples (columns with a known population distinct count)`.

---

### Task 3: `marginal_entropy` package (Python only)

**Files:** Create `services/analytics/python/turbo_parakeet/techniques/marginal_entropy/{__init__,base,rust,ams,polars}.py`; create `tests/test_marginal_entropy.py`.

- [ ] **Step 1: Failing tests** — `tests/test_marginal_entropy.py`:

```python
"""
Marginal-entropy accuracy tests — every implementation in
turbo_parakeet.techniques.marginal_entropy. Oracles: hand-worked entropies and
MarginalEntropyPolars (the reference). Null is its own category.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math

import polars as pl
import pytest

from datagen import large_dataset, low_cardinality, mixed_dtypes, near_constant
from harness import (
    assert_agrees,
    assert_contract,
    implementation_params,
    load,
    reference,
    run,
)

PKG = "turbo_parakeet.techniques.marginal_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def h(impl: str, s: pl.Series) -> float:
    return run(load(impl), {"t": s.to_frame()})["h"][0]


# 1. Contract


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# 2. Reference agreement


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [lambda: large_dataset(), lambda: mixed_dtypes(2_000), lambda: low_cardinality(50_000, 6)],
    ids=["large", "mixed", "lowcard"],
)
def test_agrees_with_reference(impl, make):
    frames = {"t": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers (exact implementations; AMS within its RTOL)


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("k", [1, 2, 4, 16])
def test_uniform_is_log2_k(impl, k):
    got = h(impl, pl.Series("x", list(range(k)) * 250))
    assert got == pytest.approx(math.log2(k), rel=load(impl).RTOL, abs=1e-9)


@pytest.mark.parametrize("impl", ALL)
def test_null_is_one_category(impl):
    got = h(impl, pl.Series("x", [1, None, 1, None] * 100))
    assert got == pytest.approx(1.0, rel=load(impl).RTOL, abs=1e-9)


@pytest.mark.parametrize("impl", ALL)
def test_near_constant(impl):
    frames = {"t": near_constant(100_000, 1)}
    want = run(reference(PKG), frames)["h"][0]
    assert run(load(impl), frames)["h"][0] == pytest.approx(want, rel=load(impl).RTOL, abs=1e-9)


def test_ams_is_an_estimator():
    from turbo_parakeet.techniques.marginal_entropy import MarginalEntropyAmsRust

    assert MarginalEntropyAmsRust.EXACT is False
    with pytest.raises(ValueError):
        MarginalEntropyAmsRust(num_samples=0)
```

- [ ] **Step 2:** `PY -m pytest tests/test_marginal_entropy.py -n 0 -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Read `techniques/pairwise_entropy/{base,rust,ams,polars}.py` and `techniques/_ams.py` first and mirror them.

`base.py`:

```python
"""Shannon entropy of each column, null one category (per-column scope)."""

import polars as pl

from turbo_parakeet._dtypes import encodable
from turbo_parakeet.techniques.base import Technique


class MarginalEntropy(Technique):
    """H(col) in bits; null is its own category (the joint entropies' policy). Eligible:
    any column the Rust encoder accepts, in a frame with at least one row. Metrics only;
    `n_rows` counts nulls."""

    SCOPE = "per_column"
    ARITY = 1
    METRICS = {"h": pl.Float64, "n_rows": pl.UInt32}
    RTOL = 1e-9
    ATOL = 1e-12
    ACCURACY_METRICS = ("h",)

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)
```

`rust.py` (`MarginalEntropyRust`): per frame (`group_by_frame`), `_plugin.marginal_entropy(df)` → `col_name, entropy`; build metrics with `h` from it and `n_rows = df.height`; return `self.metrics_frame(combos, {"h": [...], "n_rows": [...]})` in `combos` order (look the entropy up by column name).

`ams.py` (`MarginalEntropyAmsRust`): `EXACT = False`; `RTOL = 0.05`; `ATOL = 1e-9`; `__init__(self, *, num_samples: int = 4096, seed: int = 0)` calling `check_ams_params(num_samples, seed)`; compute as `rust.py` but with `_plugin.ams_marginal_entropy(df, self.num_samples, self.seed)`.

`polars.py` (`MarginalEntropyPolars`), docstring "Reference: Polars value_counts + entropy per column (PairwiseEntropyPolars' marginal expression)": `h` = `entropy_bits(frames[n].select(c), [c])` imported from `turbo_parakeet.techniques.pairwise_entropy.polars`; `n_rows` = frame height.

`__init__.py`:

```python
"""Shannon entropy per column (per-column scope). See MarginalEntropy."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.marginal_entropy.ams import MarginalEntropyAmsRust
from turbo_parakeet.techniques.marginal_entropy.base import MarginalEntropy
from turbo_parakeet.techniques.marginal_entropy.rust import MarginalEntropyRust

REFERENCE = "MarginalEntropyPolars"
IMPLEMENTATIONS = ("MarginalEntropyRust", "MarginalEntropyAmsRust", "MarginalEntropyPolars")

__getattr__ = lazy_attributes(__name__, {"MarginalEntropyPolars": ".polars"})
__all__ = [
    "MarginalEntropy",
    "MarginalEntropyRust",
    "MarginalEntropyAmsRust",
    "MarginalEntropyPolars",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
```

- [ ] **Step 4:** `PY -m pytest tests/test_marginal_entropy.py -q` → PASS. If AMS misses RTOL 0.05 on a known-answer case (tiny columns), report the measured error rather than loosening silently. Also run `PY -m pytest tests/test_layering.py tests/test_base.py -q` (package discovery tests, if any, may enumerate technique packages).
- [ ] **Step 5: Commit** — `techniques: marginal_entropy (exact Rust, AMS, Polars reference)`.

---

### Task 4: Rust `distinct_count` kernel and binding

**Files:** Create `services/analytics/src/techniques/single_column/unordered/distinct_count.rs`; modify `.../unordered/mod.rs`, `src/techniques/mod.rs`, `src/bindings/api/techniques.rs`, `src/bindings/python.rs`, `python/turbo_parakeet/_plugin.py`.

- [ ] **Step 1: Kernel with failing unit tests** — `distinct_count.rs`:

```rust
//! HyperLogLog distinct count of every column (the `distinct_count` technique's Rust
//! implementation, spec 2026-10-24-cardinality-entropy-techniques §2): one sketch per
//! 64K-row chunk, merged by register max, columns and chunks in parallel.

use polars::prelude::*;
use rayon::prelude::*;

use crate::common::encode_series;
use crate::techniques::hll::{hash_key, Hll};

const CHUNK: usize = 1 << 16;

/// `col_name`, `n_non_null`, `estimate`, `std_error`, `sketch_bytes` per column.
pub(crate) fn hll_distinct_impl(inputs: &[Series], p: u8) -> PolarsResult<Series> {
    let rows = inputs
        .par_iter()
        .map(|s| {
            let col = encode_series(s)?;
            let sketch = col
                .values
                .par_chunks(CHUNK)
                .zip(col.is_null.par_chunks(CHUNK))
                .map(|(values, nulls)| {
                    let mut h = Hll::new(p);
                    for (&v, &null) in values.iter().zip(nulls) {
                        if !null {
                            h.insert(hash_key(v));
                        }
                    }
                    h
                })
                .reduce(
                    || Hll::new(p),
                    |mut a, b| {
                        a.merge(&b);
                        a
                    },
                );
            let n_non_null = col.is_null.iter().filter(|&&null| !null).count() as u64;
            Ok((n_non_null, sketch.estimate(), sketch.std_error()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let n = inputs.len();
    let names = StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
        .into_series()
        .with_name("col_name".into());
    let series = [
        names,
        UInt64Chunked::from_vec("n_non_null".into(), rows.iter().map(|r| r.0).collect()).into_series(),
        Float64Chunked::from_vec("estimate".into(), rows.iter().map(|r| r.1).collect()).into_series(),
        Float64Chunked::from_vec("std_error".into(), rows.iter().map(|r| r.2).collect()).into_series(),
        UInt64Chunked::from_vec("sketch_bytes".into(), vec![1u64 << p; n]).into_series(),
    ];
    Ok(StructChunked::from_series("distinct_count".into(), n, series.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(s: Series, p: u8) -> (u64, f64, f64, u64) {
        let df = hll_distinct_impl(&[s], p).unwrap().into_frame().unnest(["distinct_count"]).unwrap();
        let get_u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        let get_f = |c: &str| df.column(c).unwrap().f64().unwrap().get(0).unwrap();
        (get_u("n_non_null"), get_f("estimate"), get_f("std_error"), get_u("sketch_bytes"))
    }

    #[test]
    fn small_column_is_near_exact_and_skips_nulls() {
        let values: Vec<Option<i64>> = (0..1_000).map(|i| (i % 7 != 0).then_some(i % 50)).collect();
        let (n, est, se, bytes) = row(Series::new("a".into(), values), 14);
        assert_eq!(n, 1_000 - 143);
        assert!((est - 50.0).abs() <= 0.5, "estimate {est}");
        assert!((se - 1.04 / 128.0).abs() < 1e-12);
        assert_eq!(bytes, 1 << 14);
    }

    #[test]
    fn chunks_merge_to_the_single_sketch() {
        // 3 chunks of 2^16 rows; the merged estimate equals one sketch over every value.
        let v: Vec<i64> = (0..(3 * CHUNK as i64)).map(|i| i % 100_000).collect();
        let (_, est, _, _) = row(Series::new("a".into(), v.clone()), 12);
        let mut one = Hll::new(12);
        for x in v {
            one.insert(hash_key(x as u64));
        }
        assert_eq!(est, one.estimate());
    }

    #[test]
    fn canonical_floats_are_one_value() {
        let (_, est, _, _) = row(Series::new("f".into(), [0.0f64, -0.0, f64::NAN, f64::NAN]), 14);
        assert!((est - 2.0).abs() <= 0.01, "estimate {est}");
    }
}
```

Check that `encode_series` keys an `i64` value `x` as `x as u64` (read `common/encode.rs`); if not, build the expected sketch from `encode_series(&series)?.values` instead. Wire the module: add `pub(crate) mod distinct_count;` to `src/techniques/single_column/unordered/mod.rs` and `distinct_count` to the `single_column::unordered::{...}` re-export in `src/techniques/mod.rs`.

- [ ] **Step 2:** From `services/analytics/`: `cargo test --lib distinct_count::` (env as in Conventions) → PASS (3 tests). If a test fails, fix the kernel, not the expected values, unless the expectation is demonstrably wrong (explain).

- [ ] **Step 3: Entry points.**
  - `src/bindings/api/techniques.rs`, after `set_index`:

```rust
/// `col_name`, `n_non_null`, `estimate`, `std_error`, `sketch_bytes` per column: a
/// HyperLogLog of 2^precision registers; `precision` in 4..=18, else InvalidInput.
pub fn hll_distinct(batch: &RecordBatch, precision: u8) -> Result<RecordBatch> {
    if !(4..=18).contains(&precision) {
        return Err(Error::InvalidInput(format!(
            "precision must be in 4..=18, got {precision}"
        )));
    }
    table(crate::techniques::distinct_count::hll_distinct_impl(
        &columns(batch)?,
        precision,
    ))
}
```

  - `src/bindings/python.rs`, after `set_index`'s pyfunction, and register it beside `set_index` in the module init (`m.add_function(wrap_pyfunction!(hll_distinct, m)?)?;`):

```rust
#[pyfunction]
#[pyo3(signature = (data, precision))]
fn hll_distinct(py: Python<'_>, data: &Bound<'_, PyAny>, precision: u8) -> PyResult<ArrowTable> {
    let batch = read_batch(data)?;
    run(py, || api::hll_distinct(&batch, precision)).map(ArrowTable)
}
```

  - `_plugin.py`, after `set_index`:

```python
def hll_distinct(df: Frame, precision: int) -> pl.DataFrame:
    """col_name, n_non_null, estimate, std_error, sketch_bytes per column (HyperLogLog,
    2^precision registers)."""
    return pl.DataFrame(_rs.hll_distinct(_collected(df), precision))
```

- [ ] **Step 4: Build and smoke-check.** `cd services/analytics && PY -m maturin develop --release`; then `PY -c "import polars as pl; from turbo_parakeet import _plugin; print(_plugin.hll_distinct(pl.DataFrame({'a': list(range(1000))}), 14))"` → one row, estimate ≈ 1000. And `_plugin.hll_distinct(df, 3)` raises ValueError. Run `PY scripts/check_layering.py` → OK.
- [ ] **Step 5: Commit** — `techniques: hll_distinct kernel and binding (distinct_count, spec 2026-10-24 §6)`.

---

### Task 5: `distinct_count` package

**Files:** Create `services/analytics/python/turbo_parakeet/techniques/distinct_count/{__init__,base,hll,polars}.py`; create `tests/test_distinct_count.py`.

- [ ] **Step 1: Failing tests** — `tests/test_distinct_count.py` (same header pattern as `tests/test_marginal_entropy.py`; PKG `turbo_parakeet.techniques.distinct_count`):

```python
def count(impl: str, s: pl.Series, **params) -> dict:
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {
        "mixed": mixed_dtypes(300),
        "empty": pl.DataFrame(schema={"x": pl.Int64}),
    }
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [lambda: large_dataset(), lambda: mixed_dtypes(2_000), lambda: low_cardinality(200_000, 4, cardinality=100_000)],
    ids=["large", "mixed", "highcard"],
)
def test_agrees_with_reference(impl, make):
    frames = {"t": make()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", ALL)
def test_small_column_counts_non_null_values(impl):
    s = pl.Series("x", [(i % 50) if i % 7 else None for i in range(1_000)])
    r = count(impl, s)
    assert r["n_non_null"] == 857
    assert r["n_distinct"] == pytest.approx(50, rel=0.01)
    assert r["low"] <= r["n_distinct"] <= r["high"] <= 857


@pytest.mark.parametrize("impl", ALL)
def test_floats_are_canonical(impl):
    r = count(impl, pl.Series("f", [0.0, -0.0, float("nan"), float("nan"), None]))
    assert r["n_distinct"] == pytest.approx(2, rel=0.01)


@pytest.mark.parametrize("impl", ALL)
def test_categorical_counts_like_string(impl):
    s = pl.Series("s", ["a", "b", "a", "c"] * 50)
    assert count(impl, s)["n_distinct"] == pytest.approx(
        count(impl, s.cast(pl.Categorical))["n_distinct"]
    )


def test_reference_is_exact_and_hll_reports_its_error():
    from turbo_parakeet.techniques.distinct_count import DistinctCountHllRust, DistinctCountPolars

    s = pl.Series("x", list(range(10_000)))
    exact = run(DistinctCountPolars, {"t": s.to_frame()}).row(0, named=True)
    assert exact["n_distinct"] == exact["low"] == exact["high"] == 10_000
    assert exact["std_error"] == 0.0 and exact["sketch_bytes"] is None
    hll = run(DistinctCountHllRust, {"t": s.to_frame()}, precision=10).row(0, named=True)
    assert hll["std_error"] == pytest.approx(1.04 / 32)
    assert hll["sketch_bytes"] == 1024
    assert DistinctCountHllRust.EXACT is False
    with pytest.raises(ValueError):
        DistinctCountHllRust(precision=3)
```

Imports: `pytest`, `polars as pl`, `from datagen import large_dataset, low_cardinality, mixed_dtypes`, the harness helpers.

- [ ] **Step 2:** `PY -m pytest tests/test_distinct_count.py -n 0 -q` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement.**

`base.py`:

```python
"""How many distinct non-null values each column holds (per-column scope; spec
docs/superpowers/specs/2026-10-24-cardinality-entropy-techniques-design.md §2)."""

import polars as pl

from turbo_parakeet._dtypes import encodable, is_nested
from turbo_parakeet.techniques.base import Technique, metric_mismatches

FLOATS = (pl.Float32, pl.Float64)


def canonical(s: pl.Series) -> pl.Series:
    """Values as the Rust encoder keys them: −0.0 = 0.0, one NaN, Categorical / Enum by
    string; nulls dropped."""
    s = s.drop_nulls()
    if s.dtype in FLOATS:
        return (s + 0.0).fill_nan(float("nan"))
    if isinstance(s.dtype, (pl.Categorical, pl.Enum)):
        return s.cast(pl.String)
    return s


class DistinctCount(Technique):
    """`n_distinct`: the (estimated) number of distinct non-null values, with a ±3σ
    interval [`low`, `high`] (floored at 0, capped at `n_non_null`), the relative
    `std_error` (0 when exact, null when not exposed) and the sketch's `sketch_bytes`.
    Floats canonical, Categorical / Enum by string. Eligible: the Rust encoder's scalar
    dtypes in a non-empty column; nested, Binary, Struct, Null, Object and 128-bit
    integers are not. Metrics only."""

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_non_null": pl.UInt64,
        "n_distinct": pl.Float64,
        "low": pl.Float64,
        "high": pl.Float64,
        "std_error": pl.Float64,
        "sketch_bytes": pl.UInt64,
    }
    ACCURACY_METRICS = ("n_distinct",)

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)

    def describe(self, frames, combos):
        dtypes = {(n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()}
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def interval_rows(self, combos, n_non_null, n_distinct, std_error, sketch_bytes):
        """metrics_frame with the ±3σ interval derived once (std_error null → no interval)."""
        low, high = [], []
        for n, d, se in zip(n_non_null, n_distinct, std_error):
            spread = 3 * d * (se or 0.0)
            low.append(max(d - spread, 0.0) if se is not None else None)
            high.append(min(d + spread, float(n)) if se is not None else None)
        return self.metrics_frame(
            combos,
            {
                "n_non_null": n_non_null,
                "n_distinct": n_distinct,
                "low": low,
                "high": high,
                "std_error": std_error,
                "sketch_bytes": sketch_bytes,
            },
        )

    def agreement(self, result, reference):
        """`n_non_null` exact; `n_distinct` within RTOL (the estimators' class / instance
        tolerance); interval columns are not compared."""
        keys = self.key_columns()
        return metric_mismatches(result, reference, keys, ["n_non_null"], 0.0, 0.0) + metric_mismatches(
            result, reference, keys, ["n_distinct"], self.RTOL, self.ATOL
        )
```

`polars.py`: `DistinctCountPolars` (docstring "Reference: exact Polars n_unique of the canonical non-null values") — per combo `v = canonical(frames[n][c])`, `n_distinct = float(v.n_unique())`, `std_error = 0.0`, `sketch_bytes = None`, `n_non_null = v.len()`; via `interval_rows` (the interval then equals the count). And `DistinctCountPolarsApprox` (`EXACT = False`, `RTOL = 0.05`, docstring "Polars approx_n_unique (Polars' HyperLogLog) of the canonical non-null values; no precision setting, std_error not exposed (null)") — `n_distinct = float(v.approx_n_unique())` (returns a Series / scalar depending on the Polars version: use `v.to_frame().select(pl.first().approx_n_unique()).item()`), `std_error = None`, `sketch_bytes = None`.

`hll.py`: `DistinctCountHllRust` (`EXACT = False`, docstring "Rust extension `hll_distinct`: one 2^precision-register HyperLogLog per 64K-row chunk, merged by register max, Ertl's estimator; rayon across columns and chunks"):

```python
    def __init__(self, *, precision: int = 14):
        super().__init__()
        if type(precision) is not int or not 4 <= precision <= 18:
            raise ValueError(f"precision must be an integer in 4..=18, got {precision!r}")
        self.precision = precision
        # agreement: within 4 standard errors of the exact count
        self.RTOL = 4 * 1.04 / math.sqrt(2**precision)

    def _compute(self, frames, combos):
        rows = {}  # (frame, column) → plugin row; one binding call per frame
        for frame, group in group_by_frame(combos).items():
            out = _plugin.hll_distinct(
                frames[frame].select([c for ((_, c),) in group]), self.precision
            )
            rows.update(((frame, r["col_name"]), r) for r in out.iter_rows(named=True))
        cols = [rows[k] for (k,) in combos]  # canonical order
        return self.interval_rows(
            combos,
            [r["n_non_null"] for r in cols],
            [r["estimate"] for r in cols],
            [r["std_error"] for r in cols],
            [r["sketch_bytes"] for r in cols],
        )
```

(`import math`, `from turbo_parakeet import _plugin`, `from turbo_parakeet.techniques.base import group_by_frame`.) Rows are emitted once, in `combos` order. Use the same pattern in `PopulationCardinalityRust` and `MarginalEntropy*Rust`. The Rust kernel sees Categorical columns through the binding and keys them by string (as the other kernels do); `test_categorical_counts_like_string` verifies it.

`DistinctCountPolarsApprox.__init__(self, *, precision: int | None = None)` accepts and ignores `precision`, documented as "accepted so benchmark variants can pass `precision` to every estimator; Polars' sketch has no precision setting". The harness gives `Variant.estimator_params` to every `EXACT = False` implementation, and both sketches are estimators. Add `test_polars_approx_accepts_and_ignores_precision` (same `n_distinct` with and without `precision=10`).

`__init__.py`: REFERENCE `"DistinctCountPolars"`; IMPLEMENTATIONS `("DistinctCountHllRust", "DistinctCountPolarsApprox", "DistinctCountPolars")`; `__getattr__ = lazy_attributes(__name__, {"DistinctCountPolars": ".polars", "DistinctCountPolarsApprox": ".polars"})`; `__all__` with the base `DistinctCount` too.

- [ ] **Step 4:** `PY -m pytest tests/test_distinct_count.py -q` → PASS. If `DistinctCountPolarsApprox` misses 5% on a case, measure and report its actual error before changing RTOL.
- [ ] **Step 5: Commit** — `techniques: distinct_count (HyperLogLog Rust, Polars approx_n_unique, exact Polars reference)`.

---

### Task 6: Rust `population_cardinality` kernel and binding

**Files:** Create `services/analytics/src/techniques/single_column/unordered/population_cardinality.rs`; modify `.../unordered/describe/mod.rs` (`mod frequencies;` → `pub(crate) mod frequencies;`), `.../unordered/mod.rs`, `src/techniques/mod.rs`, `src/bindings/api/techniques.rs`, `src/bindings/python.rs`, `python/turbo_parakeet/_plugin.py`.

- [ ] **Step 1: Kernel with failing unit tests:**

```rust
//! Population cardinality of every column (spec 2026-10-24-cardinality-entropy-techniques
//! §3): Describe's frequency pass (counts, f1, f2, the 3-way split capture history), then
//! the Chao1 and Schnabel estimators. Both are always reported; Schnabel is NaN when
//! undefined (d/n ≥ 0.5 or no recapture).

use polars::prelude::*;
use rayon::prelude::*;

use crate::common::encode_series;
use crate::techniques::cardinality_estimators::{chao1, schnabel};
use crate::techniques::describe::frequencies::frequencies;

/// `col_name`, `n_non_null`, `n_distinct`, `f1`, `f2`, `chao1`, `chao1_low`, `chao1_high`,
/// `schnabel`, `schnabel_low`, `schnabel_high` per column.
pub(crate) fn population_cardinality_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows = inputs
        .par_iter()
        .map(|s| {
            let col = encode_series(s)?;
            let f = frequencies(&col, seed, None, None);
            let n = col.is_null.iter().filter(|&&null| !null).count() as u64;
            let c = chao1(f.n_unique, f.f1, f.f2);
            let sc = schnabel(&f.capture_history, f.n_unique, n).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            Ok((n, f.n_unique, f.f1, f.f2, c, sc))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let u = |name: &str, get: fn(&(u64, u64, u64, u64, (f64, f64, f64), (f64, f64, f64))) -> u64| {
        UInt64Chunked::from_vec(name.into(), rows.iter().map(get).collect()).into_series()
    };
    let x = |name: &str, get: fn(&(u64, u64, u64, u64, (f64, f64, f64), (f64, f64, f64))) -> f64| {
        Float64Chunked::from_vec(name.into(), rows.iter().map(get).collect()).into_series()
    };
    let series = [
        StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str()))
            .into_series()
            .with_name("col_name".into()),
        u("n_non_null", |r| r.0),
        u("n_distinct", |r| r.1),
        u("f1", |r| r.2),
        u("f2", |r| r.3),
        x("chao1", |r| r.4 .0),
        x("chao1_low", |r| r.4 .1),
        x("chao1_high", |r| r.4 .2),
        x("schnabel", |r| r.5 .0),
        x("schnabel_low", |r| r.5 .1),
        x("schnabel_high", |r| r.5 .2),
    ];
    Ok(StructChunked::from_series("population_cardinality".into(), inputs.len(), series.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(s: Series, seed: u64) -> DataFrame {
        population_cardinality_impl(&[s], seed)
            .unwrap()
            .into_frame()
            .unnest(["population_cardinality"])
            .unwrap()
    }

    #[test]
    fn counts_and_chao1_by_hand() {
        // a×3, b×2, c×1, d×1, null×2: d = 4, f1 = 2, f2 = 1, n = 7
        let s = Series::new("x".into(), [Some("a"), Some("a"), Some("a"), Some("b"), Some("b"), Some("c"), Some("d"), None, None]);
        let df = frame(s, 0);
        let u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        assert_eq!((u("n_non_null"), u("n_distinct"), u("f1"), u("f2")), (7, 4, 2, 1));
        let chao = df.column("chao1").unwrap().f64().unwrap().get(0).unwrap();
        assert!((chao - (4.0 + 2.0 * 1.0 / (2.0 * 2.0))).abs() < 1e-12); // d + f1(f1−1)/(2(f2+1))
        // d/n = 4/7 ≥ 0.5: Schnabel undefined
        assert!(df.column("schnabel").unwrap().f64().unwrap().get(0).unwrap().is_nan());
    }

    #[test]
    fn matches_describe_frequencies() {
        let v: Vec<i64> = (0..200_000i64).map(|i| (i * 7919) % 30_000).collect();
        let s = Series::new("x".into(), v);
        let f = frequencies(&encode_series(&s).unwrap(), 5, None, None);
        let df = frame(s, 5);
        let u = |c: &str| df.column(c).unwrap().u64().unwrap().get(0).unwrap();
        assert_eq!((u("n_distinct"), u("f1"), u("f2")), (f.n_unique, f.f1, f.f2));
        let want = schnabel(&f.capture_history, f.n_unique, 200_000).unwrap().0;
        let got = df.column("schnabel").unwrap().f64().unwrap().get(0).unwrap();
        assert_eq!(got, want);
    }
}
```

Make `frequencies` importable: in `describe/mod.rs`, `mod frequencies;` → `pub(crate) mod frequencies;`. Wire `pub(crate) mod population_cardinality;` and the re-export as in Task 4. If the tuple-closure helpers do not compile neatly (closures vs `fn` pointers), restructure into a small `struct Row` with named fields — keep it simple.

- [ ] **Step 2:** `cargo test --lib population_cardinality::` → PASS (2 tests).

- [ ] **Step 3: Entry points.**
  - `api/techniques.rs`:

```rust
/// `col_name` + PopulationCardinality's metrics per column; `seed` sets the 3-way split.
pub fn population_cardinality(batch: &RecordBatch, seed: u64) -> Result<RecordBatch> {
    table(crate::techniques::population_cardinality::population_cardinality_impl(
        &columns(batch)?,
        seed,
    ))
}
```

  - `python.rs`: `#[pyfunction] #[pyo3(signature = (data, seed))] fn population_cardinality(py, data, seed: u64)` — same body pattern as `hll_distinct`; register it.
  - `_plugin.py`:

```python
def population_cardinality(df: Frame, seed: int) -> pl.DataFrame:
    """col_name, n_non_null, n_distinct, f1, f2, chao1[_low, _high], schnabel[_low, _high]
    per column (Schnabel NaN when undefined)."""
    return pl.DataFrame(_rs.population_cardinality(_collected(df), seed))
```

- [ ] **Step 4:** rebuild (`maturin develop --release`); smoke: `_plugin.population_cardinality(pl.DataFrame({"a": [1, 1, 2, 3, 3, 3, 4] * 1000}), 0)` prints one row with `n_distinct` 4. `PY scripts/check_layering.py` → OK. `cargo test --lib describe::` still passes (visibility change only).
- [ ] **Step 5: Commit** — `techniques: population_cardinality kernel and binding (Chao1 + Schnabel, spec 2026-10-24 §6)`.

---

### Task 7: `population_cardinality` package

**Files:** Create `services/analytics/python/turbo_parakeet/techniques/population_cardinality/{__init__,base,rust,polars}.py`; create `tests/test_population_cardinality.py`.

- [ ] **Step 1: Failing tests** — `tests/test_population_cardinality.py` (header pattern as before; PKG `turbo_parakeet.techniques.population_cardinality`; imports `math`, `numpy as np`, `polars as pl`, `pytest`, `from datagen import large_dataset, mixed_dtypes, population_samples`, the harness helpers):

```python
def stats(impl, s, **params):
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"mixed": mixed_dtypes(300), "empty": pl.DataFrame(schema={"x": pl.Int64})}
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", [0, 7])
def test_agrees_with_reference(impl, seed):
    sample, _ = population_samples(
        50_000, [(1_000, "uniform"), (20_000, "uniform"), (20_000, ("zipf", 1.1))], seed=seed
    )
    for frames in ({"large": large_dataset()}, {"mixed": mixed_dtypes(2_000)}, {"s": sample}):
        cls = load(impl)
        assert_agrees(cls(seed=seed), run(cls, frames, seed=seed), run(reference(PKG), frames, seed=seed))


@pytest.mark.parametrize("impl", ALL)
def test_counts_and_chao1_by_hand(impl):
    r = stats(impl, pl.Series("x", ["a", "a", "a", "b", "b", "c", "d", None, None]))
    assert (r["n_non_null"], r["n_distinct"], r["f1"], r["f2"]) == (7, 4, 2, 1)
    assert r["chao1"] == pytest.approx(4.5)
    assert math.isnan(r["schnabel"])  # d/n = 4/7 ≥ 0.5


@pytest.mark.parametrize("impl", ALL)
def test_schnabel_defined_on_a_dense_sample(impl):
    r = stats(impl, pl.Series("x", np.arange(100_000) % 1_000))  # every value seen 100×
    assert r["n_distinct"] == 1_000
    assert r["schnabel"] == pytest.approx(1_000, rel=0.02)
    assert r["schnabel_low"] <= r["schnabel"] <= r["schnabel_high"]
    assert r["chao1"] == 1_000.0  # no singletons


def test_split_subsets_match_the_rust_splitmix():
    from turbo_parakeet.techniques.population_cardinality.polars import subsets

    # The Rust `subset(0, row)` for rows 0..8, computed with Python ints from the SplitMix64
    # finaliser in describe/frequencies.rs (independently of the code under test).
    assert subsets(0, 8).tolist() == [1, 2, 1, 0, 1, 2, 2, 0]


def test_truth_figures_by_hand():
    from turbo_parakeet.techniques.population_cardinality import PopulationCardinalityPolars

    frames = {"t": pl.DataFrame({"x": np.arange(100_000) % 1_000})}
    result = run(PopulationCardinalityPolars, frames)
    out = PopulationCardinalityPolars().accuracy(frames, result, result, truth={("t", "x"): 2_000})
    assert out["observed_rel_err"] == pytest.approx(0.5)  # d = 1 000 of D = 2 000
    assert out["chao1_rel_err"] == pytest.approx(0.5)
    assert out["chao1_coverage"] == 0.0
    assert out["schnabel_n_undefined"] == 0
    assert "max_rel_err" in out  # the default agreement figures are kept
```

Also add the same 8 values as a Rust assertion in `frequencies.rs`'s tests (`assert_eq!((0..8).map(|r| subset(0, r)).collect::<Vec<_>>(), [1, 2, 1, 0, 1, 2, 2, 0])`), so both sides are pinned to one independent computation.

- [ ] **Step 2:** `PY -m pytest tests/test_population_cardinality.py -n 0 -q` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement.**

`base.py`:

```python
"""Distinct values in the population behind each column, unseen ones included
(per-column scope; spec docs/superpowers/specs/2026-10-24-cardinality-entropy-techniques-design.md §3)."""

import math

import polars as pl

from turbo_parakeet._dtypes import encodable, is_nested
from turbo_parakeet.techniques.base import Technique

ESTIMATORS = ("chao1", "schnabel")


class PopulationCardinality(Technique):
    """Observed `n_distinct` (d of `n_non_null`), singletons `f1`, doubletons `f2`, and two
    estimates of the population's distinct count with 95% intervals: bias-corrected
    Chao1 (log-normal interval) and Schnabel over a seeded 3-way row split (Byar's
    interval; NaN when d/n ≥ 0.5 or no recapture). Both are always reported: choosing one
    is the recommenders' decision. Values keyed as the Rust encoder keys them (floats
    canonical, Categorical / Enum by string). Eligible: as DistinctCount."""

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {
        "n_non_null": pl.UInt64,
        "n_distinct": pl.UInt64,
        "f1": pl.UInt64,
        "f2": pl.UInt64,
        **{f"{e}{s}": pl.Float64 for e in ESTIMATORS for s in ("", "_low", "_high")},
    }
    RTOL = 1e-9
    ATOL = 1e-9

    def __init__(self, *, seed: int = 0):
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed

    def eligible(self, series):
        return series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)

    def describe(self, frames, combos):
        dtypes = {(n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()}
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def accuracy(self, frames, result, reference, truth=None):
        """The default figures (agreement with the reference) and, with `truth` ((frame,
        column) → the population's distinct count D), over the computed columns:
        observed_rel_err = max |d − D| / D; <est>_rel_err = max |estimate − D| / D (NaN
        estimates skipped, counted in <est>_n_undefined); <est>_coverage = share of
        columns whose [low, high] contains D."""
        out = super().accuracy(frames, result, reference)
        if truth is None:
            return out
        rows = [
            r
            for r in result.iter_rows(named=True)
            if r["status"] == "computed" and (r["df_a"], r["col_a"]) in truth
        ]
        if not rows:
            return out
        d_true = [truth[r["df_a"], r["col_a"]] for r in rows]
        out["observed_rel_err"] = max(abs(r["n_distinct"] - t) / t for r, t in zip(rows, d_true))
        for e in ESTIMATORS:
            pairs = [(r, t) for r, t in zip(rows, d_true) if not math.isnan(r[e])]
            out[f"{e}_n_undefined"] = len(rows) - len(pairs)
            out[f"{e}_rel_err"] = max((abs(r[e] - t) / t for r, t in pairs), default=None)
            out[f"{e}_coverage"] = (
                sum(r[f"{e}_low"] <= t <= r[f"{e}_high"] for r, t in pairs) / len(pairs)
                if pairs
                else None
            )
        return out
```

(Black will reformat; keep the logic.)

`rust.py` — `PopulationCardinalityRust`: per frame, `_plugin.population_cardinality(df, self.seed)`; return a `metrics_frame(combos, {...})` in `combos` order, looking rows up by `col_name` (same approach as Task 5).

`polars.py` — `PopulationCardinalityPolars` (docstring "Reference: counts and capture masks by a Polars group_by, the split by SplitMix64 in NumPy uint64 (bit-identical to the Rust `subset`), the formulas of describe/estimators.py. Shares no counting code with the Rust kernel."):

```python
import numpy as np
import polars as pl

from turbo_parakeet.techniques.describe import estimators
from turbo_parakeet.techniques.distinct_count.base import canonical
from turbo_parakeet.techniques.population_cardinality.base import PopulationCardinality

_U = np.uint64


def subsets(seed: int, n: int) -> np.ndarray:
    """SplitMix64 finaliser of seed + row, mod 3, for rows 0..n (describe/frequencies.rs
    `subset`)."""
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=_U) + _U(seed) + _U(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
        return ((z ^ (z >> _U(31))) % _U(3)).astype(np.uint8)


def statistics(s: pl.Series, seed: int) -> dict:
    keep = s.is_not_null().to_numpy()
    bits = (np.uint8(1) << subsets(seed, s.len()))[keep]
    groups = (
        pl.DataFrame({"v": canonical(s), "bit": bits})
        .group_by("v")
        .agg(count=pl.len(), mask=pl.col("bit").bitwise_or())
    )
    counts = groups["count"].to_numpy()
    history = np.bincount(groups["mask"].to_numpy(), minlength=8)[1:].tolist()
    n, d = int(keep.sum()), groups.height
    f1, f2 = int((counts == 1).sum()), int((counts == 2).sum())
    chao = estimators.chao1(d, f1, f2)
    sch = estimators.schnabel(history, d, n) or (float("nan"),) * 3
    return {
        "n_non_null": n, "n_distinct": d, "f1": f1, "f2": f2,
        "chao1": chao[0], "chao1_low": chao[1], "chao1_high": chao[2],
        "schnabel": sch[0], "schnabel_low": sch[1], "schnabel_high": sch[2],
    }


class PopulationCardinalityPolars(PopulationCardinality):
    def _compute(self, frames, combos):
        rows = [statistics(frames[f][c], self.seed) for ((f, c),) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})
```

`canonical` drops nulls, which keeps it aligned with `bits[keep]`. Check `pl.Expr.bitwise_or` exists as an aggregation in the installed Polars; if not, aggregate `pl.col("bit").unique()` and OR in Python, or `(pl.col("bit") == 1).any()` etc. per subset — any correct equivalent.

`__init__.py`: REFERENCE `"PopulationCardinalityPolars"`; IMPLEMENTATIONS `("PopulationCardinalityRust", "PopulationCardinalityPolars")`; lazy-load `.polars`.

- [ ] **Step 4:** `PY -m pytest tests/test_population_cardinality.py -q` → PASS. Agreement must be exact on the counts and within 1e-9 on the estimates; a mismatch means the split or keying differs — debug it, never loosen RTOL.
- [ ] **Step 5: Commit** — `techniques: population_cardinality (Chao1 + Schnabel, Rust vs Polars/NumPy reference, truth accuracy)`.

---

### Task 8: Benchmarks

**Files:** Create `tests/performance/benchmark_{marginal_entropy,distinct_count,population_cardinality}.py`.

- [ ] **Step 1: Write the three scripts** (style of `benchmark_pairwise_entropy.py`: module docstring with purpose / shapes / run line, then `run(...)`).

`benchmark_marginal_entropy.py`:

```python
"""
Marginal entropy benchmark (speed, peak memory, accuracy; harness.py): MarginalEntropyRust
and MarginalEntropyAmsRust (estimator, also scored against MarginalEntropyRust) vs
MarginalEntropyPolars (reference). The pairwise-entropy shapes.

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_marginal_entropy.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality, near_constant, near_unique

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.marginal_entropy",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
            Dataset("near-unique 1M x 9", lambda: {"t": near_unique()}),
            Dataset("near-constant 1M x 3", lambda: {"t": near_constant()}),
        ],
    )
```

`benchmark_distinct_count.py`: datasets — `large_dataset()`; `narrow 10M x 4` = four Int64 columns with 10³ / 10⁵ / 10⁶ / 10⁷ distinct values (`pl.DataFrame({f"d{e}": np.random.default_rng(e).integers(0, 10**e, 10_000_000) for e in (3, 5, 6, 7)})`; note uniform draws of 10⁷ labels over 10⁷ rows see ≈ 63% of them — the reference is exact either way); `wide 1M x 100` = `describe_wide(1_000_000, 100)` (Int64 / Float64 / String / Date); `strings 10M x 2` = `("item_" + integers(0, 10**6)).cast(String)` and `integers(0, 10**7)` as strings. Variants: `[Variant(f"p={p}", estimator_params={"precision": p}) for p in (10, 12, 14, 16)]`. `DistinctCountPolarsApprox` accepts and ignores `precision` (Task 5), so it runs once per variant with identical results; that is expected.

`benchmark_population_cardinality.py`:

```python
POPULATIONS = [(d, shape) for d in (1_000, 10_000, 100_000, 1_000_000) for shape in ("uniform", ("zipf", 1.1))]


def samples(n_rows: int) -> Dataset:
    @cache
    def make():
        return population_samples(n_rows, POPULATIONS, seed=1)

    return Dataset(
        f"samples {n_rows:,} x 8",
        lambda: {"t": make()[0]},
        truth=lambda: {("t", c): float(d) for c, d in make()[1].items()},
    )

# run(..., [large_dataset(), samples(1_000_000), samples(100_000)])
```

- [ ] **Step 2: Smoke runs** (each a few minutes at most): run `benchmark_marginal_entropy.py` and `benchmark_population_cardinality.py` fully; run `benchmark_distinct_count.py` once fully if it finishes within ~20 minutes, else report its progress and timings per dataset. Check in the printed tables: every `ok` row has `peak_mb`; exact implementations `agrees` True with `max_rel_err` ≈ 0; `MarginalEntropyAmsRust` has `exact_rust_*` columns; population rows on the sample datasets carry `chao1_rel_err`, `chao1_coverage`, `schnabel_rel_err`, `schnabel_n_undefined`, `observed_rel_err` (Chao1/Schnabel should beat `observed_rel_err` on the Zipf columns). Report the key numbers.
- [ ] **Step 3:** `ruff check tests/performance` and `black --check tests/performance` → clean.
- [ ] **Step 4: Commit** — `benchmarks: marginal_entropy, distinct_count, population_cardinality`.

---

### Task 9: Docs and full check

**Files:** `CLAUDE.md`, `tests/performance/README.md` (currently untracked — add it in this commit), `ToDo.md`, the spec.

- [ ] **Step 1: CLAUDE.md** (LF; Edit, not rewrite; concise, the file's style):
  - §1 Per-column: add three paragraphs after **Top-k** — **Distinct count — `turbo_parakeet.techniques.distinct_count`** (`DistinctCountHllRust`, `DistinctCountPolarsApprox`, ★`DistinctCountPolars`), **Population cardinality — `turbo_parakeet.techniques.population_cardinality`** (`PopulationCardinalityRust`, ★`PopulationCardinalityPolars`), **Marginal entropy — `turbo_parakeet.techniques.marginal_entropy`** (`MarginalEntropyRust`, `MarginalEntropyAmsRust`, ★`MarginalEntropyPolars`): metrics, keywords, eligibility, agreement tolerances, kernel files, benchmark script — each 3–6 lines, from the spec and the code as built.
  - Project Structure: the three packages in the `techniques/<technique>/` list, the two new `.rs` files in the `techniques/` line, the three test files and benchmark scripts.
  - `_plugin` private list: add `hll_distinct`, `population_cardinality`; note `marginal_entropy` / `ams_marginal_entropy` are also used by `MarginalEntropy*`.
  - Current Focus: "All thirteen techniques" → "All sixteen techniques" with the three names added; the harness's `Dataset.truth` in the Testing Convention benchmark bullet (one clause).
- [ ] **Step 2: README** — add three rows to the technique table in `tests/performance/README.md` (after Top-k), in its column format (purpose, module, precise reference, datasets).
- [ ] **Step 3: ToDo / spec** — ToDo 17 `**Designed**` → `**Done**`; spec `Status: draft` → `Status: implemented 2026-10-24`, and add a short "Deviations" note listing the three planning deviations at the top of this plan plus the `precision`-accepting `DistinctCountPolarsApprox`.
- [ ] **Step 4: Full check** — `PY scripts/check.py check` (background; can take 10–30 min). All steps must pass; investigate any failure (own changes first; report unrelated failures with evidence).
- [ ] **Step 5: Commit** — `Docs: distinct-count, population-cardinality, marginal-entropy techniques (CLAUDE.md, README, ToDo 17)`; `git add -f` the spec and this plan.
