# Uniform Technique Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `tests/performance/benchmark_<technique>.py` reports speed, peak memory and accuracy for every implementation in one run, on one harness; the two bespoke accuracy/memory scripts are folded in and deleted.

**Architecture:** The harness parent writes each dataset to Arrow IPC once, then runs one child process per (dataset, implementation, variant, threads); the child (the harness itself, `harness.py --child spec.json`) measures the warm-up's peak working set, times the repeats and writes its result as IPC. The parent scores each result with `Technique.agreement` and a new `Technique.accuracy` hook against the reference and, for estimators (`EXACT = False`), against the exact Rust sibling.

**Tech Stack:** Python 3.12, Polars, pytest (+ xdist), the `turbo_parakeet` extension (no Rust changes).

**Spec:** `docs/superpowers/specs/2026-10-24-benchmark-uniformity-design.md`

---

## Conventions for every task

- Python: `C:/Users/Alexander/miniconda3/envs/p312/python.exe` (written `PY` below). Run from the repo root `C:/Users/Alexander/turbo-parakeet`.
- No rebuild needed: only Python changes (editable install).
- pytest runs 4 xdist workers by default; use `-n 0` for single tests.
- Commits: the pre-commit hook needs the env on PATH. In Git Bash:
  `export PATH="/c/Users/Alexander/miniconda3/envs/p312:/c/Users/Alexander/miniconda3/envs/p312/Scripts:$PATH"` before `git commit`.
  End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File map

| File | Change | Responsibility |
|---|---|---|
| `services/analytics/python/turbo_parakeet/techniques/base.py` | modify | `ACCURACY_METRICS`, `Technique.accuracy`, `numeric_errors` |
| `services/analytics/python/turbo_parakeet/techniques/pairwise_entropy/base.py` | modify | entropy-only accuracy + `max_abs_mi_err` |
| `services/analytics/python/turbo_parakeet/techniques/threeway_entropy/base.py` | modify | `ACCURACY_METRICS = ("h_abc",)` |
| `services/analytics/python/turbo_parakeet/techniques/{pairwise,threeway}_entropy/ams.py` | modify | `EXACT = False` |
| `services/analytics/python/turbo_parakeet/techniques/top_k/base.py` | modify | `true_counts`, `score_top_k`, `TopK.accuracy` (moved from the accuracy script) |
| `tests/performance/harness.py` | rewrite | child-per-measurement harness |
| `tests/performance/fake_technique.py` | create | tiny technique for the harness tests |
| `tests/test_benchmark_harness.py` | modify | unit + integration tests of the harness |
| `tests/test_base.py`, `tests/test_top_k.py`, `tests/test_pairwise_entropy.py`, `tests/test_threeway_entropy.py` | modify | accuracy-hook tests |
| `tests/performance/benchmark_top_k.py` | rewrite | absorbs the accuracy script |
| `tests/performance/benchmark_streaming_entropy.py` | modify | `memory_mb` from the harness |
| `tests/performance/benchmark_pairwise_entropy.py` | modify | docstring only |
| `tests/performance/benchmark_top_k_accuracy.py`, `tests/performance/benchmark_ams_entropy.py` | delete | folded in |
| `CLAUDE.md`, `ToDo.md`, the spec | modify | docs |

---

### Task 1: Default accuracy hook on `Technique`

**Files:**
- Modify: `services/analytics/python/turbo_parakeet/techniques/base.py` (class attributes near line 51; `agreement` near line 259; module function after `metric_mismatches`, near line 409)
- Test: `tests/test_base.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_base.py`, add `import math` to the imports, and add `numeric_errors` to the `from turbo_parakeet.techniques.base import (...)` list. Append:

```python
def _error_frame(m, i, status):
    return pl.DataFrame(
        {
            "df_a": ["t"] * len(m),
            "col_a": [f"c{n}" for n in range(len(m))],
            "status": pl.Series(status, dtype=STATUS),
            "m": pl.Series(m, dtype=pl.Float64),
            "i": pl.Series(i, dtype=pl.UInt64),
        }
    )


def test_numeric_errors_relative_error_and_missing():
    status = ["computed", "computed", "computed", "ineligible"]
    want = _error_frame([1.0, 0.0, 3.0, None], [10, 20, 30, None], status)
    got = _error_frame([1.1, 0.0, None, None], [10, 20, 30, None], status)
    out = numeric_errors(got, want, ["df_a", "col_a"], ["m", "i"])
    # cells: m 0.1, 0 (0/0), missing; i 0, 0, 0
    assert out["max_rel_err"] == pytest.approx(0.1)
    assert out["p95_rel_err"] == pytest.approx(0.1)
    assert out["n_missing"] == 1


def test_numeric_errors_nan_is_kept_but_nan_nan_is_no_error():
    status = ["computed"] * 3
    want = _error_frame([1.0, math.nan, 2.0], [1, 1, 1], status)
    got = _error_frame([1.0, math.nan, math.nan], [1, 1, 1], status)
    out = numeric_errors(got, want, ["df_a", "col_a"], ["m"])
    assert math.isnan(out["max_rel_err"])
    assert out["n_missing"] == 0


def test_numeric_errors_rejects_misaligned_rows():
    status = ["computed"] * 2
    want = _error_frame([1.0, 2.0], [1, 2], status)
    got = want.reverse()
    with pytest.raises(ValueError, match="key columns"):
        numeric_errors(got, want, ["df_a", "col_a"], ["m"])


def test_accuracy_defaults_to_numeric_metrics():
    frames = {"t": pl.DataFrame({"ab": [1, 2], "c": ["x", "y"]})}
    reference = PerColumnToy().add(frames).result()
    result = reference.with_columns(score=pl.col("score") * 1.5)
    out = PerColumnToy().accuracy(frames, result, reference)
    assert out == {
        "max_rel_err": pytest.approx(0.5),
        "p95_rel_err": pytest.approx(0.5),
        "n_missing": 0,
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PY -m pytest tests/test_base.py -n 0 -k "numeric_errors or accuracy_defaults" -v`
Expected: collection ERROR, `ImportError: cannot import name 'numeric_errors'`.

- [ ] **Step 3: Implement**

In `base.py`, after `ATOL: ClassVar[float] = 0.0` add:

```python
    # Metrics the default `accuracy` compares (None: every numeric METRICS column).
    ACCURACY_METRICS: ClassVar[tuple[str, ...] | None] = None
```

After the `agreement` method add:

```python
    def accuracy(
        self, frames: dict[str, pl.DataFrame], result: pl.DataFrame, reference: pl.DataFrame
    ) -> dict[str, float | int | None]:
        """Error figures of `result` against `reference` for the benchmarks (metrics
        only: never in a result, never a decision): `numeric_errors` over
        ACCURACY_METRICS. `frames` serves overrides that rebuild ground truth (Top-k)."""
        metrics = self.ACCURACY_METRICS
        if metrics is None:
            metrics = tuple(c for c, dt in self.METRICS.items() if dt.is_numeric())
        return numeric_errors(result, reference, self.key_columns(), list(metrics))
```

After `metric_mismatches` add:

```python
def numeric_errors(
    result: pl.DataFrame,
    reference: pl.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> dict[str, float | int | None]:
    """Relative error |got − want| / |want| of every `metrics` cell the reference computed
    (equal values, 0/0 and NaN/NaN → 0; NaN elsewhere is kept, so it shows in the max):
    `max_rel_err`, `p95_rel_err` and `n_missing` (nulls in `result` where the reference has
    a value). Both frames in canonical order; misaligned keys → ValueError."""
    if result.select(keys).rows() != reference.select(keys).rows():
        raise ValueError("key columns differ from the reference")
    computed = reference["status"] == "computed"
    got, want = result.filter(computed), reference.filter(computed)
    g, w = pl.col("g"), pl.col("w")
    rels, missing = [], 0
    for c in metrics:
        pair = pl.DataFrame(
            {"g": got[c].cast(pl.Float64), "w": want[c].cast(pl.Float64)}
        )
        missing += pair.filter(g.is_null() & w.is_not_null()).height
        rels.append(
            pair.select(
                pl.when((g == w) | (g.is_nan() & w.is_nan()))
                .then(0.0)
                .otherwise((g - w).abs() / w.abs())
            ).to_series()
        )
    rel = pl.concat(rels).drop_nulls() if rels else pl.Series([], dtype=pl.Float64)
    if rel.is_empty():
        return {"max_rel_err": None, "p95_rel_err": None, "n_missing": missing}
    return {
        "max_rel_err": rel.nan_max(),
        "p95_rel_err": rel.quantile(0.95),
        "n_missing": missing,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/test_base.py -n 0 -v`
Expected: all PASS (old and new).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/python/turbo_parakeet/techniques/base.py tests/test_base.py
git commit -m "techniques: Technique.accuracy hook and numeric_errors (spec 2026-10-24 §6)"
```

---

### Task 2: Entropy accuracy figures; AMS classes are estimators

**Files:**
- Modify: `services/analytics/python/turbo_parakeet/techniques/pairwise_entropy/base.py` (class `PairwiseEntropy`, after `ATOL = 1e-12`)
- Modify: `services/analytics/python/turbo_parakeet/techniques/threeway_entropy/base.py` (class `ThreewayEntropy`, after `METRICS`)
- Modify: `services/analytics/python/turbo_parakeet/techniques/pairwise_entropy/ams.py`, `.../threeway_entropy/ams.py` (class bodies, beside `RTOL`)
- Test: `tests/test_pairwise_entropy.py`, `tests/test_threeway_entropy.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pairwise_entropy.py` (add `import pytest` / `import polars as pl` to the imports if they are not already there):

```python
def test_accuracy_compares_entropies_and_mi_in_bits():
    from turbo_parakeet.techniques.pairwise_entropy import (
        PairwiseEntropyAmsRust,
        PairwiseEntropyPolars,
    )

    frames = {"t": pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]})}
    reference = PairwiseEntropyPolars().add(frames).result()
    result = reference.with_columns(h_ab=pl.col("h_ab") * 1.01, mi=pl.col("mi") + 0.05)
    out = PairwiseEntropyPolars().accuracy(frames, result, reference)
    assert out["max_rel_err"] == pytest.approx(0.01)
    assert out["max_abs_mi_err"] == pytest.approx(0.05)
    assert out["n_missing"] == 0
    assert PairwiseEntropyAmsRust.EXACT is False
```

Append to `tests/test_threeway_entropy.py` (same imports):

```python
def test_accuracy_compares_the_joint_entropy_only():
    from turbo_parakeet.techniques.threeway_entropy import (
        ThreewayEntropyAmsRust,
        ThreewayEntropyPolars,
    )

    frames = {"t": pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [1, 1, 1, 2]})}
    reference = ThreewayEntropyPolars().add(frames).result()
    result = reference.with_columns(h_abc=pl.col("h_abc") * 1.02)
    out = ThreewayEntropyPolars().accuracy(frames, result, reference)
    assert out["max_rel_err"] == pytest.approx(0.02)
    assert ThreewayEntropyAmsRust.EXACT is False
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PY -m pytest tests/test_pairwise_entropy.py tests/test_threeway_entropy.py -n 0 -k accuracy -v`
Expected: FAIL. Pairwise fails with `KeyError: 'max_abs_mi_err'`, or with a `max_rel_err` mismatch because the default compares `mi` / `nmi` / `n_rows` too. Threeway fails on `EXACT`.

- [ ] **Step 3: Implement**

In `pairwise_entropy/base.py`, class `PairwiseEntropy`, after `ATOL = 1e-12`:

```python
    # Relative MI / NMI error is meaningless where MI ≈ 0: MI is compared in bits instead.
    ACCURACY_METRICS = ("h_a", "h_b", "h_ab")
```

and add the method (e.g. after `__init__`):

```python
    def accuracy(self, frames, result, reference):
        """The default figures over the entropies, plus `max_abs_mi_err` (bits)."""
        out = super().accuracy(frames, result, reference)
        computed = reference["status"] == "computed"
        mi = (result.filter(computed)["mi"] - reference.filter(computed)["mi"]).abs()
        out["max_abs_mi_err"] = mi.max()
        return out
```

In `threeway_entropy/base.py`, class `ThreewayEntropy`, after `METRICS = {...}`:

```python
    ACCURACY_METRICS = ("h_abc",)
```

In `pairwise_entropy/ams.py`, class `PairwiseEntropyAmsRust`, before `RTOL = 0.05`:

```python
    EXACT = False
```

In `threeway_entropy/ams.py`, class `ThreewayEntropyAmsRust`, add the same line `EXACT = False` beside its `RTOL`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/test_pairwise_entropy.py tests/test_threeway_entropy.py -v`
Expected: all PASS. The existing agreement tests are unaffected: no entropy `agreement` reads `EXACT`.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/python/turbo_parakeet/techniques/pairwise_entropy services/analytics/python/turbo_parakeet/techniques/threeway_entropy tests/test_pairwise_entropy.py tests/test_threeway_entropy.py
git commit -m "entropy: accuracy figures for the benchmarks; AMS implementations are estimators (EXACT = False)"
```

---

### Task 3: Top-k accuracy against exact counts

**Files:**
- Modify: `services/analytics/python/turbo_parakeet/techniques/top_k/base.py` (module functions after `_binary_as_text`; method on `TopK` after `agreement`)
- Test: `tests/test_top_k.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_top_k.py` (ensure `import polars as pl` and `import pytest` are in its imports):

```python
from turbo_parakeet.techniques.top_k.base import score_top_k, true_counts  # noqa: E402

STREAM = pl.Series("x", ["a"] * 10 + ["b"] * 5 + ["c"] * 5 + ["d"])


def _item(value, count, error):
    return {"value": value, "count": count, "error": error}


def test_true_counts_every_value_count_desc():
    truth = true_counts(STREAM)
    assert truth.schema == {"value": pl.String, "true": pl.UInt64}
    assert dict(truth.rows()) == {"a": 10, "b": 5, "c": 5, "d": 1}
    assert truth["true"].to_list() == [10, 5, 5, 1]


def test_score_top_k_is_tie_aware():
    # b and c tie at the k-th count (5): reporting c instead of b is still exact.
    s = score_top_k([_item("a", 10, 0), _item("c", 6, 2)], true_counts(STREAM), k=2)
    assert s["precision"] == 1.0
    assert s["are"] == pytest.approx(0.1)  # (0 + 1/5) / 2
    assert s["aae"] == pytest.approx(0.5)
    assert s["max_bound_violation"] == 0  # 5 ∈ [6 − 2, 6]


def test_score_top_k_bound_violation_and_wrong_value():
    truth = true_counts(STREAM)
    s = score_top_k([_item("a", 10, 0), _item("c", 6, 0)], truth, k=2)
    assert s["max_bound_violation"] == 1  # true 5 < count − error = 6
    s = score_top_k([_item("a", 10, 0), _item("d", 1, None)], truth, k=2)
    assert s["precision"] == 0.5  # d's true count 1 < the k-th count 5
    assert s["max_bound_violation"] is None  # no bound reported


def test_accuracy_of_the_reference_is_perfect():
    from turbo_parakeet.techniques.top_k import TopKPolars

    frames = {"t": STREAM.to_frame()}
    reference = TopKPolars(k=2).add(frames).result()
    out = TopKPolars(k=2).accuracy(frames, reference, reference)
    assert out == {
        "precision": 1.0,
        "are": 0.0,
        "aae": 0.0,
        "max_bound_violation": 0,
        "sketch_bytes": None,
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PY -m pytest tests/test_top_k.py -n 0 -k "true_counts or score_top_k or accuracy_of" -v`
Expected: collection ERROR, `ImportError: cannot import name 'score_top_k'`.

- [ ] **Step 3: Implement**

In `top_k/base.py`, after `_binary_as_text`, add these functions. The scoring formulas are moved unchanged from `tests/performance/benchmark_top_k_accuracy.py` (`truth_of`, `score`):

```python
def true_counts(series: pl.Series) -> pl.DataFrame:
    """Exact count of every distinct non-null value, count desc; values rendered as `top_k`
    renders them (arrow-rs text; binary as text, non-UTF-8 → null)."""
    from turbo_parakeet import _plugin

    counts = series.drop_nulls().value_counts(sort=True, name="__true")
    values = counts[series.name]
    if values.dtype == pl.Binary:
        values = _binary_as_text(values)
    return pl.DataFrame(
        {
            "value": _plugin.render(values) if counts.height else [],
            "true": counts["__true"].cast(pl.UInt64),
        },
        schema={"value": pl.String, "true": pl.UInt64},
    )


def score_top_k(top: list[dict], truth: pl.DataFrame, k: int) -> dict:
    """Benchmark scores of one column's reported items against `true_counts`:
    precision (tie-aware: reported values whose true count ≥ the true k-th count, ÷ min(k,
    distinct values)), are / aae (mean relative / absolute count error over reported
    values) and max_bound_violation (how far the true count falls outside [count − error,
    count]; null when no bound is reported)."""
    floor = truth["true"][k - 1] if truth.height >= k else 0
    reported = pl.DataFrame(
        top, schema={"value": pl.String, "count": pl.UInt64, "error": pl.UInt64}
    ).join(truth, on="value", how="left")
    count = reported["count"].cast(pl.Int64)
    true = reported["true"].cast(pl.Int64)
    err = (count - true).abs()
    violation = None
    if reported["error"].null_count() == 0 and reported.height:
        lower = count - reported["error"].cast(pl.Int64)
        gaps = [(lower - true).max(), (true - count).max()]
        violation = max(0, *(g for g in gaps if g is not None))
    return {
        "precision": (true >= floor).sum() / min(k, truth.height),
        "are": (err / true).mean(),
        "aae": err.mean(),
        "max_bound_violation": violation,
    }
```

On `TopK`, after `agreement`, add:

```python
    def accuracy(self, frames, result, reference):
        """Scores against exact counts rebuilt from `frames` (the reference's list holds only
        k values, so it cannot score counts outside it): `score_top_k` per computed column,
        then the worst — min precision, max are / aae / max_bound_violation — and the max
        sketch_bytes. `reference` is not used."""
        scores, sizes = [], []
        rows = zip(
            result.select(self.key_columns()).rows(),
            result["status"],
            result["top_k"].to_list(),
            result["sketch_bytes"],
        )
        for (df, col), status, top, size in rows:
            if status != "computed":
                continue
            truth = true_counts(frames[df][col])
            if truth.height:
                scores.append(score_top_k(top, truth, self.k))
                sizes.append(size)

        def worst(key, pick):
            values = [s[key] for s in scores if s[key] is not None]
            return pick(values) if values else None

        return {
            "precision": worst("precision", min),
            "are": worst("are", max),
            "aae": worst("aae", max),
            "max_bound_violation": worst("max_bound_violation", max),
            "sketch_bytes": max((s for s in sizes if s is not None), default=None),
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/test_top_k.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/python/turbo_parakeet/techniques/top_k/base.py tests/test_top_k.py
git commit -m "top_k: accuracy against exact counts (moved from benchmark_top_k_accuracy.py)"
```

---

### Task 4: Fake technique and failing harness tests

**Files:**
- Create: `tests/performance/fake_technique.py`
- Modify: `tests/test_benchmark_harness.py`

- [ ] **Step 1: Create the fake technique**

`tests/performance/fake_technique.py`. pytest never collects `tests/performance/`. Children import it as `performance.fake_technique`, because the harness puts `tests/` on `sys.path`.

```python
"""A tiny technique for tests/test_benchmark_harness.py: the sum of each numeric column.
FakeSumPolars is the reference, FakeSumRust its exact Rust-named sibling (with a ballast
allocation, to check the memory figure), FakeSumSketchRust an estimator off by `noise`, and
FakeSumBroken raises."""

import polars as pl

from turbo_parakeet.techniques.base import Technique


class FakeSum(Technique):
    SCOPE = "per_column"
    ARITY = 1
    METRICS = {"total": pl.Float64}

    def __init__(self, *, noise: float = 0.0, ballast_mb: int = 0):
        super().__init__()
        self.noise = noise
        self.ballast_mb = ballast_mb

    def eligible(self, series: pl.Series) -> bool:
        return series.dtype.is_numeric()

    def _compute(self, frames, combos):
        ballast = b"x" * (self.ballast_mb * 2**20)  # written, so it is in the working set
        totals = [
            float(frames[n][c].sum()) * (1 + self.noise) for ((n, c),) in combos
        ]
        del ballast
        return self.metrics_frame(combos, {"total": totals})


class FakeSumPolars(FakeSum):
    pass


class FakeSumRust(FakeSum):
    pass


class FakeSumSketchRust(FakeSum):
    EXACT = False


class FakeSumBroken(FakeSum):
    def _compute(self, frames, combos):
        raise RuntimeError("broken on purpose")


REFERENCE = "FakeSumPolars"
IMPLEMENTATIONS = ("FakeSumRust", "FakeSumSketchRust", "FakeSumBroken", "FakeSumPolars")
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_benchmark_harness.py`, replace the imports block at the top with:

```python
"""The benchmark harness: pure parts (timing loop, speedup arithmetic) and, on a fake
technique, one end-to-end run (child processes, memory, accuracy, variants, failures)."""

import time

import polars as pl
import pytest

from performance import harness
from performance.harness import Dataset, Variant, memory_mb, speedups, time_call
```

In both existing `speedups` tests, add `"variant": None,` to the expected dict, after `"dataset": "d",`. The output now names the variant.

Append:

```python
def test_time_call_reports_after_the_warmup():
    seen = []
    time_call(lambda: seen.append("run"), runs=2, budget_s=10.0, on_warmup=lambda: seen.append("w"))
    assert seen == ["run", "w", "run", "run"]


def test_memory_mb_current_and_peak():
    current, peak = memory_mb()
    assert 0 < current <= peak


def test_speedups_per_variant_share_the_exact_rows_of_their_group():
    rows = [
        {**_row("XPolars", "default", 100.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "N", 10.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "1", 20.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "N", 25.0), "variant": "hi", "group": "lo"},
        {**_row("XSketchRust", "1", 50.0), "variant": "hi", "group": "lo"},
    ]
    got = {s["variant"]: s for s in speedups(rows)}
    assert got["lo"]["total"] == 10.0
    assert got["hi"]["total"] == 4.0
    assert got["hi"]["algorithmic"] == 2.0


PKG = "performance.fake_technique"


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness, "RESULTS_DIR", tmp_path_factory.mktemp("results"))
        return harness.run(
            PKG,
            [Dataset("tiny", lambda: {"t": pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})})],
            impl_params={"FakeSumRust": {"ballast_mb": 64}},
            variants=[
                Variant("lo", estimator_params={"noise": 0.1}),
                Variant("hi", estimator_params={"noise": 0.2}),
            ],
            runs=1,
        )


def _one(bench, implementation, variant, threads) -> dict:
    rows = bench.filter(
        implementation=implementation, variant=variant, threads=threads
    ).to_dicts()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.slow
def test_one_row_per_measurement(bench):
    assert sorted(bench.select("implementation", "variant", "threads").rows()) == sorted(
        [
            ("FakeSumPolars", "lo", "default"),
            ("FakeSumBroken", "lo", "default"),
            ("FakeSumRust", "lo", "N"),
            ("FakeSumRust", "lo", "1"),
            ("FakeSumSketchRust", "lo", "N"),
            ("FakeSumSketchRust", "lo", "1"),
            ("FakeSumSketchRust", "hi", "N"),
            ("FakeSumSketchRust", "hi", "1"),
        ]
    )


@pytest.mark.slow
def test_a_failing_child_is_an_error_row(bench):
    assert _one(bench, "FakeSumBroken", "lo", "default")["status"] == "error"
    assert set(bench.filter(pl.col("implementation") != "FakeSumBroken")["status"]) == {"ok"}


@pytest.mark.slow
def test_accuracy_against_the_reference_and_the_exact_sibling(bench):
    exact = _one(bench, "FakeSumRust", "lo", "N")
    assert exact["agrees"] is True
    assert exact["max_rel_err"] == 0.0
    assert exact["exact_rust_agrees"] is None
    for variant, noise in (("lo", 0.1), ("hi", 0.2)):
        sketch = _one(bench, "FakeSumSketchRust", variant, "N")
        assert sketch["agrees"] is False
        assert sketch["max_rel_err"] == pytest.approx(noise)
        assert sketch["exact_rust_max_rel_err"] == pytest.approx(noise)
        assert sketch["speedup_vs_exact_rust"] > 0
    assert _one(bench, "FakeSumPolars", "lo", "default")["agrees"] is None


@pytest.mark.slow
def test_memory_and_speedups(bench):
    exact = _one(bench, "FakeSumRust", "lo", "N")
    assert exact["peak_mb"] >= 60  # the 64 MiB ballast
    assert exact["load_mb"] > 0
    assert exact["algorithmic"] is not None and exact["parallel"] is not None
    assert _one(bench, "FakeSumRust", "lo", "1")["algorithmic"] is None
    assert bench["technique"].unique().to_list() == ["fake_technique"]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `PY -m pytest tests/test_benchmark_harness.py -n 0 -v`
Expected: collection ERROR, `ImportError: cannot import name 'Variant'`.

- [ ] **Step 4: Commit the tests with the fake**

```bash
git add tests/performance/fake_technique.py tests/test_benchmark_harness.py
git commit -m "test: harness tests on a fake technique (failing until the harness rewrite)"
```

---

### Task 5: Rewrite the harness

**Files:**
- Rewrite: `tests/performance/harness.py`

- [ ] **Step 1: Replace the whole file**

```python
"""
Shared benchmark harness: speed, peak memory and accuracy of every implementation of one
technique (spec docs/superpowers/specs/2026-10-24-benchmark-uniformity-design.md). Every
tests/performance/benchmark_<technique>.py is:

    from harness import Dataset, large_dataset, run
    from datagen import ...
    if __name__ == "__main__":
        run("turbo_parakeet.techniques.<technique>", [large_dataset(), Dataset("narrow …", …), Dataset("wide …", …)])

One child process per measurement, i.e. per (dataset, implementation, variant, threads),
reading the dataset from Arrow IPC files the parent wrote once. The child (this file,
`--child spec.json`) collects garbage, notes its working set (`load_mb`), runs the warm-up
Impl(**params).add(frames).result() and reports `peak_mb`, the peak working set above
`load_mb` (the OS peak never resets, so only a fresh process gives a clean figure;
`peak_is_lower_bound` when the warm-up never exceeded the load-time peak), then times `runs`
runs (fresh instance each) → median / min. A warm-up over `budget_s` is "skipped: over
budget". Params cross as JSON (tuples arrive as lists).

Rust implementations run at N threads and at 1 (RAYON_NUM_THREADS=1, POLARS_MAX_THREADS=1 —
both pools are fixed at process start):
    algorithmic = fastest non-Rust ÷ Rust@1 thread
    parallel    = Rust@1 thread   ÷ Rust@N threads
    total       = fastest non-Rust ÷ Rust@N threads

Accuracy (never a gate; the pytest suite is the correctness gate): each result is compared
with the reference's (`agrees` = impl.agreement() finds nothing, plus impl.accuracy()'s
figures) and an estimator's (EXACT = False) also with its exact Rust sibling's — the one
Rust implementation with EXACT = True, if there is exactly one — as `exact_rust_*` and
`speedup_vs_exact_rust`. Variants: `Variant.estimator_params` go to estimators only, so the
exact implementations run once per distinct `Variant.params` (their rows carry the first
such variant's label).
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

sys.path.insert(
    0, str(Path(__file__).parents[1])
)  # tests/ -> `import datagen` in benchmark scripts

import turbo_parakeet  # noqa: E402
from datagen import large_dataset as generated_large_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
SINGLE_THREAD_ENV = {"RAYON_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
CHILD_TIMEOUT_S = 1800
COLUMNS = [
    "dataset",
    "implementation",
    "variant",
    "threads",
    "status",
    "median_ms",
    "min_ms",
    "load_mb",
    "peak_mb",
    "peak_is_lower_bound",
    "agrees",
    "algorithmic",
    "parallel",
    "total",
    "speedup_vs_exact_rust",
]


@dataclass(frozen=True)
class Dataset:
    name: str
    make: Callable[[], dict[str, pl.DataFrame]]
    exclude: tuple[
        str, ...
    ] = ()  # implementations not run on this dataset (e.g. would exhaust memory)


@dataclass(frozen=True)
class Variant:
    label: str = "default"
    params: dict = field(default_factory=dict)  # every implementation
    estimator_params: dict = field(default_factory=dict)  # estimators (EXACT = False) only


def large_dataset(columns: int | None = None, exclude: tuple[str, ...] = ()) -> Dataset:
    """datagen.large_dataset() (50K rows × 101 cols, generated), optionally only its first `columns`."""
    label = "large_dataset" + (f" (first {columns} cols)" if columns else "")

    def make() -> dict[str, pl.DataFrame]:
        df = generated_large_dataset()
        return {"large": df.select(df.columns[:columns]) if columns else df}

    return Dataset(label, make, exclude)


def memory_mb() -> tuple[float, float]:
    """This process's (current, peak) working set (resident set) so far, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        assert ok, "GetProcessMemoryInfo failed"
        return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # bytes
        return maxrss / 2**20, maxrss / 2**20
    with open("/proc/self/statm") as f:  # pages
        current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return current / 2**20, maxrss / 2**10  # ru_maxrss is KiB on Linux


def time_call(
    fn: Callable[[], object],
    runs: int,
    budget_s: float,
    on_warmup: Callable[[], None] | None = None,
) -> tuple[list[float] | None, object]:
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    if on_warmup is not None:
        on_warmup()
    if elapsed > budget_s:
        return None, None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def speedups(rows: list[dict]) -> list[dict]:
    """Per (dataset, variant, Rust implementation); the non-Rust competitors are the rows of
    the same dataset and params group (rows without a group form one)."""
    out = []
    ok = [r for r in rows if r["status"] == "ok"]
    for dataset, group in dict.fromkeys((r["dataset"], r.get("group")) for r in ok):
        same = [r for r in ok if r["dataset"] == dataset and r.get("group") == group]
        others = [
            r["median_ms"] for r in same if not r["implementation"].endswith("Rust")
        ]
        best = min(others) if others else None
        rust = dict.fromkeys(
            (r["implementation"], r.get("variant"))
            for r in same
            if r["implementation"].endswith("Rust")
        )
        for impl, variant in rust:

            def median(threads: str) -> float | None:
                return next(
                    (
                        r["median_ms"]
                        for r in same
                        if r["implementation"] == impl
                        and r.get("variant") == variant
                        and r["threads"] == threads
                    ),
                    None,
                )

            n, one = median("N"), median("1")
            out.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "implementation": impl,
                    "algorithmic": best / one if best and one else None,
                    "parallel": one / n if one and n else None,
                    "total": best / n if best and n else None,
                }
            )
    return out


def _load(module, name: str):
    try:
        return getattr(module, name)
    except ImportError as exc:
        print(f"  {name}: unavailable ({exc})", file=sys.stderr)
        return None


def _threads(name: str) -> tuple[str, ...]:
    return ("N", "1") if name.endswith("Rust") else ("default",)


def _plan(
    classes: dict, variants: list[Variant], params: dict, impl_params: dict
) -> list[tuple[str, str, str, str, dict]]:
    """(implementation, variant, group, threads, kwargs) per measurement. The group is the
    first variant with the same `params`; non-estimators run once per group."""
    first: dict[str, str] = {}
    group = {
        v.label: first.setdefault(json.dumps(v.params, sort_keys=True), v.label)
        for v in variants
    }
    plan = []
    for name, cls in classes.items():
        for v in variants:
            estimator = not cls.EXACT
            if not estimator and group[v.label] != v.label:
                continue
            kwargs = {
                **params,
                **v.params,
                **(v.estimator_params if estimator else {}),
                **impl_params.get(name, {}),
            }
            for threads in _threads(name):
                plan.append((name, v.label, group[v.label], threads, kwargs))
    return plan


def _measure(
    package: str,
    name: str,
    kwargs: dict,
    frames: dict[str, str],
    threads: str,
    runs: int,
    budget_s: float,
    out: str,
) -> dict:
    spec = Path(out).with_suffix(".json")
    spec.write_text(
        json.dumps(
            {
                "package": package,
                "implementation": name,
                "params": kwargs,
                "frames": frames,
                "runs": runs,
                "budget_s": budget_s,
                "out": out,
            }
        )
    )
    env = {**os.environ, **(SINGLE_THREAD_ENV if threads == "1" else {})}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(spec)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").splitlines()[-20:])
        print(f"    error:\n{tail}", file=sys.stderr, flush=True)
        return {"status": "error"}
    except subprocess.TimeoutExpired:
        print("    timeout", file=sys.stderr, flush=True)
        return {"status": "timeout"}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _child(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    frames = {
        n: pl.read_ipc(p, memory_map=False) for n, p in spec["frames"].items()
    }
    cls = getattr(importlib.import_module(spec["package"]), spec["implementation"])
    gc.collect()
    load_mb, peak_before = memory_mb()
    peak: list[float] = []
    times, result = time_call(
        lambda: cls(**spec["params"]).add(frames).result(),
        spec["runs"],
        spec["budget_s"],
        on_warmup=lambda: peak.append(memory_mb()[1]),
    )
    info = {
        "load_mb": load_mb,
        "peak_mb": peak[0] - load_mb,
        "peak_is_lower_bound": peak[0] == peak_before,
    }
    if times is None:
        info.update(
            status=f"skipped: over budget ({spec['budget_s']:.0f}s)",
            median_ms=None,
            min_ms=None,
        )
    else:
        result.write_ipc(spec["out"])
        info.update(
            status="ok",
            median_ms=statistics.median(times) * 1e3,
            min_ms=min(times) * 1e3,
        )
    print(json.dumps(info))


def _compare(impl, frames: dict, result: pl.DataFrame, other: pl.DataFrame, prefix: str) -> dict:
    try:
        out = {
            "agrees": not impl.agreement(result, other),
            **impl.accuracy(frames, result, other),
        }
    except Exception as exc:  # a failed comparison must not lose the run's other rows
        print(f"    comparison failed: {exc!r}", file=sys.stderr, flush=True)
        out = {"agrees": False}
    return {prefix + k: v for k, v in out.items()}


def _score(
    rows: list[dict],
    results: dict,
    kwargs_of: dict,
    frames: dict,
    classes: dict,
    reference: str,
    sibling: str | None,
) -> None:
    """Fill one dataset's agreement / accuracy columns in place."""
    row_of = {(r["implementation"], r["variant"], r["threads"]): r for r in rows}
    for row in rows:
        key = (row["implementation"], row["variant"], row["threads"])
        result = results.get(key)
        if result is None or row["implementation"] == reference:
            continue
        cls = classes[row["implementation"]]
        impl = cls(**kwargs_of[key])
        ref = results.get((reference, row["group"], _threads(reference)[0]))
        if ref is not None:
            row.update(_compare(impl, frames, result, ref, ""))
        if sibling is None or cls.EXACT:
            continue
        sib = results.get((sibling, row["group"], row["threads"]))
        if sib is not None:
            row.update(_compare(impl, frames, result, sib, "exact_rust_"))
            sib_ms = row_of[sibling, row["group"], row["threads"]]["median_ms"]
            row["speedup_vs_exact_rust"] = sib_ms / row["median_ms"]


def run(
    package: str,
    datasets: list[Dataset],
    params: dict | None = None,
    impl_params: dict[str, dict] | None = None,
    variants: list[Variant] | None = None,
    runs: int = 5,
    budget_s: float = 60.0,
    results_name: str | None = None,
) -> pl.DataFrame:
    params, impl_params = params or {}, impl_params or {}
    variants = variants or [Variant()]
    if len({v.label for v in variants}) != len(variants):
        raise ValueError("variant labels must be unique")
    module = importlib.import_module(package)
    names = [module.REFERENCE] + [
        n for n in module.IMPLEMENTATIONS if n != module.REFERENCE
    ]
    classes = {n: c for n in names if (c := _load(module, n)) is not None}
    exact_rust = [n for n, c in classes.items() if n.endswith("Rust") and c.EXACT]
    sibling = exact_rust[0] if len(exact_rust) == 1 else None
    plan = _plan(classes, variants, params, impl_params)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ds in enumerate(datasets):
            frames = {
                n: f.collect() if isinstance(f, pl.LazyFrame) else f
                for n, f in ds.make().items()
            }
            folder = Path(tmp) / f"dataset{i}"
            folder.mkdir()
            paths = {}
            for j, (n, f) in enumerate(frames.items()):
                paths[n] = str(folder / f"frame{j}.arrow")
                f.write_ipc(paths[n])
            ds_rows, results, kwargs_of = [], {}, {}
            for j, (name, label, group, threads, kwargs) in enumerate(plan):
                row = {
                    "dataset": ds.name,
                    "implementation": name,
                    "variant": label,
                    "threads": threads,
                    "group": group,
                }
                ds_rows.append(row)
                if name in ds.exclude:
                    row["status"] = "excluded"
                    continue
                print(
                    f"  {ds.name} | {label}: {name} @{threads} …",
                    file=sys.stderr,
                    flush=True,
                )
                out = str(folder / f"result{j}.arrow")
                row.update(
                    _measure(package, name, kwargs, paths, threads, runs, budget_s, out)
                )
                if row["status"] == "ok":
                    results[name, label, threads] = pl.read_ipc(out, memory_map=False)
                    kwargs_of[name, label, threads] = kwargs
            _score(ds_rows, results, kwargs_of, frames, classes, module.REFERENCE, sibling)
            rows += ds_rows
            del frames, results

    by_key = {(s["dataset"], s["variant"], s["implementation"]): s for s in speedups(rows)}
    for r in rows:
        s = by_key.get((r["dataset"], r["variant"], r["implementation"]))
        if s is not None and r["threads"] == "N":
            r.update(algorithmic=s["algorithmic"], parallel=s["parallel"], total=s["total"])
        del r["group"]
    return _report(results_name or package.rsplit(".", 1)[-1], package, rows)


def _report(name: str, package: str, rows: list[dict]) -> pl.DataFrame:
    rows = [{**dict.fromkeys(COLUMNS), **r} for r in rows]
    frame = pl.from_dicts(rows, infer_schema_length=None)
    extras = [c for c in frame.columns if c not in COLUMNS]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame = frame.select(*COLUMNS, *extras).with_columns(
        technique=pl.lit(package.rsplit(".", 1)[-1]),
        cpu_count=pl.lit(os.cpu_count()),
        processor=pl.lit(platform.processor()),
        polars=pl.lit(pl.__version__),
        numpy=pl.lit(np.__version__),
        analytics=pl.lit(turbo_parakeet.__version__),
        timestamp=pl.lit(stamp),
    )
    print(
        f"\n{name} — median / min of timed runs, one process per measurement, {os.cpu_count()} CPUs"
    )
    keys = ["dataset", "implementation", "variant", "threads"]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=40):
        print(
            frame.select(
                *keys, "median_ms", "min_ms", "peak_mb", "peak_is_lower_bound", "agrees", "status"
            )
        )
        print(
            frame.filter(pl.col("parallel").is_not_null()).select(
                *keys[:3], "algorithmic", "parallel", "total"
            )
        )
        print(
            frame.filter(pl.col("agrees").is_not_null(), pl.col("threads") != "1").select(
                *keys[:3], "agrees", *extras, "speedup_vs_exact_rust"
            )
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_{stamp}.arrow"
    frame.write_ipc(path, compression="zstd")
    print(f"\nsaved {path}")
    return frame


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    _child(sys.argv[2])
```

- [ ] **Step 2: Run the harness tests**

Run: `PY -m pytest tests/test_benchmark_harness.py -n 0 -v`
Expected: all PASS. The integration tests start 8 child processes, which takes about 20–40 s.

If `test_memory_and_speedups` fails on `peak_mb >= 60`, print `bench.select("implementation","threads","load_mb","peak_mb")`. Then check that the ballast is really written: `b"x" * n` touches every page, whereas `bytearray(n)` would not.

- [ ] **Step 3: Format and lint**

Run: `PY scripts/check.py fmt` then `PY scripts/check.py lint`
Expected: lint clean. If ruff flags the `except Exception` in `_compare`, keep it: the comment states why. Add `# noqa: BLE001` only if that rule is enabled.

- [ ] **Step 4: Commit**

```bash
git add tests/performance/harness.py tests/test_benchmark_harness.py
git commit -m "harness: one child per measurement — speed, peak memory, accuracy vs reference and exact Rust (spec 2026-10-24 §3-5)"
```

---

### Task 6: Fold the top-k and AMS scripts in

**Files:**
- Rewrite: `tests/performance/benchmark_top_k.py`
- Modify: `tests/performance/benchmark_streaming_entropy.py:29-31`
- Modify: `tests/performance/benchmark_pairwise_entropy.py:1-6` (docstring)
- Delete: `tests/performance/benchmark_top_k_accuracy.py`, `tests/performance/benchmark_ams_entropy.py`

- [ ] **Step 1: Rewrite `benchmark_top_k.py`**

```python
"""
Top-k benchmark: TopKHeavyKeeperRust and TopKSpaceSavingRust (estimators) vs TopKPolars
(reference): speed, peak memory and accuracy (harness.py).

Run: C:/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_top_k.py [--rows N]

Shapes: large_dataset (realistic mix); narrow / long (parallelism inside a column:
SpaceSaving merges 2^20-row chunks, HeavyKeeper cannot); wide (parallelism across columns);
one 10M x 1 dataset per stream of datagen.TOP_K_COLUMNS (Zipf α = 1.5 / 1.1 / 0.8, flat,
skewed, string), one column each so precision and memory are per stream.
Variants: k ∈ {5, 256} × memory_bytes ∈ {¼, 1, 4} × max(16 KiB, k × 1 KiB); the reference
runs once per k. Accuracy is against exact counts (TopK.accuracy): tie-aware precision,
are / aae, SpaceSaving's max_bound_violation and sketch_bytes. `agrees` on large_dataset can
be false from ties at the k-th count (agreement precision is a lower bound there).

--rows N: the scale run instead: zipf_11 alone at N rows, k = 5 at 1× budget
(results/top_k_scale_<stamp>.arrow).
"""

import argparse

from harness import Dataset, Variant, large_dataset, run

from datagen import TOP_K_COLUMNS, low_cardinality, top_k_streams
from turbo_parakeet.techniques.top_k.base import default_budget

PKG = "turbo_parakeet.techniques.top_k"


def variant(k: int, multiple: float) -> Variant:
    return Variant(
        f"k={k} mem={multiple:g}x",
        params={"k": k},
        estimator_params={"memory_bytes": int(default_budget(k) * multiple)},
    )


def stream(column: str, n_rows: int) -> Dataset:
    return Dataset(
        f"{column} {n_rows:,} x 1",
        lambda: {"t": top_k_streams(n_rows, columns=[column])},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()
    if args.rows:
        run(
            PKG,
            [stream("zipf_11", args.rows)],
            variants=[variant(5, 1.0)],
            runs=3,
            results_name="top_k_scale",
        )
    else:
        run(
            PKG,
            [
                large_dataset(),
                Dataset(
                    "narrow 10M x 4",
                    lambda: {"t": low_cardinality(10_000_000, 4, cardinality=1_000)},
                ),
                Dataset(
                    "wide 200K x 100",
                    lambda: {"t": low_cardinality(200_000, 100, cardinality=1_000)},
                ),
                *(stream(c, 10_000_000) for c in TOP_K_COLUMNS),
            ],
            variants=[variant(k, m) for k in (5, 256) for m in (0.25, 1.0, 4.0)],
            runs=3,
        )
```

- [ ] **Step 2: Point the streaming-entropy script at the harness**

In `tests/performance/benchmark_streaming_entropy.py` replace

```python
sys.path.insert(0, str(HERE))  # memory_mb

from benchmark_ams_entropy import memory_mb  # noqa: E402
```

with

```python
sys.path.insert(0, str(HERE))  # harness

from harness import memory_mb  # noqa: E402
```

- [ ] **Step 3: Update the pairwise docstring**

In `tests/performance/benchmark_pairwise_entropy.py`, replace the first docstring line

```
Pairwise joint entropy speed benchmark: PairwiseEntropyRust and PairwiseEntropyAmsRust vs
PairwiseEntropyPolars (reference).
```

with

```
Pairwise joint entropy benchmark (speed, peak memory, accuracy; harness.py): PairwiseEntropyRust
and PairwiseEntropyAmsRust (estimator, also scored against PairwiseEntropyRust) vs
PairwiseEntropyPolars (reference).
```

- [ ] **Step 4: Delete the folded scripts**

```bash
git rm tests/performance/benchmark_top_k_accuracy.py tests/performance/benchmark_ams_entropy.py
```

Then confirm nothing else imports them:
Run: `git grep -n "benchmark_ams_entropy\|benchmark_top_k_accuracy" -- tests services scripts`
Expected: no output.

- [ ] **Step 5: Smoke-run the cheapest real benchmark and the top-k scale path**

Run: `PY tests/performance/benchmark_adjusted_rand.py`
Expected: three tables, then `saved …adjusted_rand_<stamp>.arrow`. Every `ok` row has `peak_mb`. `AdjustedRandRust` rows (threads N) show `agrees` true and `max_rel_err` ≈ 0, with algorithmic, parallel and total filled.

Run: `PY tests/performance/benchmark_top_k.py --rows 1000000`
Expected: rows for TopKPolars (default), and for HeavyKeeper and SpaceSaving at N and 1. The sketches have `precision` ≈ 1.0 on zipf_11, and SpaceSaving has `max_bound_violation` 0. The run saves `top_k_scale_<stamp>.arrow`.

Run: `PY tests/performance/benchmark_streaming_entropy.py --help 2>&1 | head -5`. If the script takes no arguments, run `PY -c "import sys; sys.path[:0]=['tests/performance','tests']; import benchmark_streaming_entropy"` instead.
Expected: no ImportError.

- [ ] **Step 6: Commit**

```bash
git add tests/performance
git commit -m "benchmarks: top-k accuracy and AMS memory/error folded into the technique benchmarks"
```

---

### Task 7: Docs and full check

**Files:**
- Modify: `CLAUDE.md`, `ToDo.md`, `docs/superpowers/specs/2026-10-24-benchmark-uniformity-design.md`

- [ ] **Step 1: CLAUDE.md, Testing Convention**

Replace the bullet that starts with ``- `tests/performance/benchmark_<technique>.py` — **speed only**`` and ends with `(Arrow IPC, ZSTD; git-ignored).` with:

```markdown
- `tests/performance/benchmark_<technique>.py` — **speed, peak memory and accuracy** (spec 2026-10-24): ~20-line scripts calling
  `harness.run(package, datasets[, variants])`. One child process per (dataset, implementation, variant, threads), reading the dataset
  from Arrow IPC written once: the warm-up `Impl(**params).add(frames).result()` gives `peak_mb` (working set above the loaded data;
  `peak_is_lower_bound`), then 5 timed runs (fresh instance) give median/min. Rust implementations run at N and 1 threads
  (`RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`): **algorithmic** (fastest non-Rust ÷ Rust@1), **parallel** (Rust@1 ÷ Rust@N),
  **total**. Every result is scored against the reference — `agrees` (`agreement()`) and `Technique.accuracy()` (default: max / p95
  relative error over `ACCURACY_METRICS` and `n_missing`; Top-k: tie-aware precision, ARE / AAE, bound violations against exact
  counts) — and an estimator's (`EXACT = False`) also against its exact Rust sibling (`exact_rust_*`, `speedup_vs_exact_rust`).
  `Variant.estimator_params` (e.g. `memory_bytes`) go to estimators only. Never a gate. Results → `tests/performance/results/*.arrow`
  (Arrow IPC, ZSTD; git-ignored).
```

- [ ] **Step 2: CLAUDE.md, Top-k and AMS notes**

Replace

```
Benchmarks: tests/performance/benchmark_top_k.py (speed), benchmark_top_k_accuracy.py (tie-aware precision, ARE/AAE, bounds, peak memory; `--rows N` scale run).
```

with

```
Benchmark: tests/performance/benchmark_top_k.py (speed, peak memory, tie-aware precision, ARE/AAE, bounds; per stream and k × budget variant; `--rows N` scale run).
```

Replace

```
Benchmarks: tests/performance/benchmark_ams_entropy.py (memory + error) and the two entropy speed benchmarks (near-unique 1M x 9, near-constant 1M x 3).
```

with

```
Benchmarks: the two entropy benchmarks (speed, peak memory, error against the reference and the exact Rust kernel; near-unique 1M x 9, near-constant 1M x 3).
```

In the Project Structure tree, change `# shared timing harness` to `# shared harness: speed, memory, accuracy (one child per measurement)`. Then check that no other CLAUDE.md line names either deleted script:
Run: `git grep -n "benchmark_ams_entropy\|benchmark_top_k_accuracy" -- CLAUDE.md`
Expected: no output.

- [ ] **Step 3: ToDo.md and the spec status**

In `ToDo.md`, item 5f: replace `**Designed** (spec 2026-10-24-benchmark-uniformity-design.md)` with `**Done** (spec 2026-10-24-benchmark-uniformity-design.md)`.

In the spec, replace `Status: draft 2026-10-24.` with `Status: implemented 2026-10-24.`

- [ ] **Step 4: Run the full local CI check**

Run: `PY scripts/check.py check`
Expected: every step passes: fmt, clippy, ruff, layering, build, `cargo test --lib` and pytest. The harness integration tests run under xdist inside their own module.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md ToDo.md
git add -f docs/superpowers/specs/2026-10-24-benchmark-uniformity-design.md docs/superpowers/plans/2026-10-24-benchmark-uniformity.md
git commit -m "Docs: uniform technique benchmarks (CLAUDE.md, ToDo 5f)"
```
