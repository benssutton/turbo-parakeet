# Uniform Technique Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every analytical technique behind one public contract, `Impl(**params).add(frames).result()`, which returns a flat keyed table. Every implementation is then checked for accuracy against one exact reference, and timed by one shared harness that separates the algorithmic speedup from the parallel speedup.

**Architecture:**
- An abstract `Technique` base in `analytics/base.py` owns:
  - enumerating column combinations, according to its scope (`per_column`, `multi_set` or `ordered`);
  - deciding eligibility, including recording ineligible rows explicitly;
  - checking that each result matches the schema;
  - comparing results with the reference (`agreement`).
- Each technique is a subpackage (`analytics/<technique>/`) containing:
  - a technique base (`base.py`) that adds its metric schema, eligibility rules and threshold conclusions;
  - one file per implementation, named after the library it uses (`rust.py`, `scipy.py`, …).
- The Rust plugin wrappers become private (`analytics/_plugin.py`).
- Tests and benchmarks work only through the classes, using `tests/harness.py`, `tests/datagen.py` and `tests/performance/harness.py`.

**Tech Stack:** Python 3.12, polars 1.41.2, numpy 2.3.5, scipy 1.18.0, scikit-learn, polars-ds, datasketch 2.0.0, fastbloom-rs, pyarrow, pytest. The Rust plugin (pyo3-polars 0.24 / polars 0.51) is **unchanged**.

**Spec:** [docs/superpowers/specs/2026-09-24-uniform-technique-interface-design.md](../specs/2026-09-24-uniform-technique-interface-design.md). Read §10, "Amendments made while planning": it overrides earlier sections.

## Global Constraints

- The Python interpreter is `/c/Users/Ben/miniconda3/envs/p312/python.exe`. Run every command from the repo root `C:\Users\Ben\turbo-parakeet` in Git Bash, and write the interpreter path in full, because shell variables don't persist between tool calls.
- **No Rust source changes and no `maturin` rebuild.** The package is installed in editable mode (`site-packages/analytics.pth` points at `services/analytics`), so new `.py` files under `services/analytics/analytics/` can be imported immediately.
- Technique scopes: `per_column` (GCD), `multi_set` (Membership, Similarity), `ordered` (Chi-squared, Pairwise/Threeway Entropy, Adjusted Rand).
- Result columns, in order: `df_a, col_a[, df_b, col_b[, df_c, col_c]]`, `status`, DESCRIPTORS, METRICS, CONCLUSIONS.
- `status` is `pl.Enum(["computed", "ineligible", "pruned"])`. **null** = not computed; **NaN** = computed but undefined.
- Default thresholds, copied from the spec:
  - Membership: `containment_threshold=0.95`
  - Similarity: `jaccard_threshold=0.6`, `overlap_threshold=0.95`
  - Chi-squared: `cramers_v_threshold=0.3`, `max_unique=1000`
  - Pairwise entropy: `nmi_threshold=0.9`, `near_unique_margin=0.1`
  - Threeway entropy: `near_unique_margin=0.1`
  - ARI: `ari_threshold=0.9`
  - GCD: `gcd > 1`

  All are keyword-only constructor arguments that can be overridden.
- Implementation-specific parameters: `fp_rate=0.01` (Bloom), `num_perm=128` (MinHash), `cache_size=4096` (Similarity LRU).
- Accuracy tolerances (`RTOL`/`ATOL` on each technique base), copied from today's tests:

  | Technique | Tolerance |
  |---|---|
  | χ² | `1e-4 / 1e-12` |
  | Entropy | `1e-5 / 1e-12` |
  | ARI | `1e-9 / 1e-12` |
  | GCD | exact |
- `tests/test_*.py` check accuracy only and never time anything. `tests/performance/benchmark_*.py` are standalone scripts and are never collected by pytest.
- Every commit message ends with a blank line followed by `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **The full non-slow suite must pass at the end of every task:** `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"`.

## Review Focus

These inputs aren't exercised by the spec's main flows, but they are the most likely to cause problems. Each one has a pinned test in the task named at the end of its line.

1. **Zero-row frame given to an ordered technique.** The Rust plugins raise `Cannot calculate … on empty columns`. Expected behaviour: every combination from that frame is `ineligible`, and no exception is raised. Tested in Tasks 5, 6, 7 and 8 (`test_zero_row_frame_is_ineligible`).
2. **NaN metric at a threshold.** In Polars, `NaN >= 0.6` is `True`, so an undefined Cramér's V, NMI, ARI or ratio would produce a positive verdict. Expected behaviour: the conclusion is `False`. Tested in every `test_conclusions_*` (a NaN row) and in `test_base.py::test_at_least_treats_nan_as_false`.
3. **Multi-set pairs from different value families** (Int vs String, Date vs Int32, Float vs Int). Rust hashes physical values while the exact sets compare Python values. Expected behaviour: the pair is `ineligible`, and Int32 vs Int64 **is** compared. Tested in Tasks 9 and 10 (`test_value_families`).
4. **Column or frame names containing `|`, or column names shared between frames.** MinHash's qualified names are `"<prefix>|<column>"`. Expected behaviour: correct keys and exact metrics. Tested in Task 10 (`test_pipe_and_shared_names`).
5. **Multi-set columns with no non-null values.** Bloom sizing would divide by n = 0, and the MinHash signature is null. Expected behaviour: NaN for ratios and Jaccard/Overlap measured from the empty side, `relationship="none"`, `passes_*=False`, and `status="computed"`. Tested in Tasks 9 and 10 (`test_empty_column`).

---

## File Structure

```
services/analytics/analytics/
  __init__.py            MODIFY  → __version__ (+ transitional re-exports until Task 11)
  _plugin.py             CREATE  (git mv of today's __init__.py) private plugin wrappers
  _dtypes.py             CREATE  dtype groupings: encodable, is_nested, value_family
  _sets.py               CREATE  freeze/distinct_values for pure-Python multi-set implementations
  base.py                CREATE  Technique ABC + helpers (computed, at_least, check_unit, metric_mismatches, …)
  gcd/{__init__,base,rust,numpy,math}.py
  chi_squared/{__init__,base,rust,scipy,polars_ds}.py
  adjusted_rand/{__init__,base,rust,sklearn}.py
  pairwise_entropy/{__init__,base,rust,polars}.py
  threeway_entropy/{__init__,base,rust,polars}.py
  membership/{__init__,base,rust,fastbloom,exact}.py
  similarity/{__init__,base,rust,datasketch,exact}.py
services/analytics/{bloom_filter,chi_squared_polarsds,deterministic_similarity_filter,
                    minhash_lsh_filter,minhash_lsh_filter_datasketch}.py   DELETE (Tasks 5, 9, 10)
tests/
  conftest.py            MODIFY  dataset fixture delegates to datagen; sys.path hack removed in Task 11
  datagen.py             CREATE  seeded generators shared by tests and benchmarks
  harness.py             CREATE  implementation_params, load, reference, run, assert_contract, assert_agrees, with_metrics
  test_base.py           CREATE
  test_datagen.py        CREATE
  test_benchmark_harness.py CREATE
  test_gcd.py, test_chi_squared.py, test_adjusted_rand.py      REWRITE
  test_pairwise_entropy.py, test_threeway_entropy.py           CREATE (test_entropy.py DELETE)
  test_membership.py     CREATE (test_bloom_filter.py DELETE)
  test_similarity.py     CREATE (test_similarity_filters.py DELETE)
  performance/
    harness.py           CREATE  shared timing harness (1-thread child process, speedups, parquet)
    benchmark_gcd.py, benchmark_chi_squared.py, benchmark_adjusted_rand.py        REWRITE
    benchmark_pairwise_entropy.py, benchmark_threeway_entropy.py                 CREATE (benchmark_entropy.py DELETE)
    benchmark_membership.py (benchmark_bloom_filter.py DELETE)
    benchmark_similarity.py (benchmark_jaccard.py DELETE)
.gitignore               MODIFY  tests/performance/results/
CLAUDE.md                MODIFY  (Task 11)
```

Implementation modules named after libraries (`numpy.py`, `math.py`, `scipy.py`, `polars.py`) are safe to use. Python 3 imports are absolute, so `import numpy` inside `analytics/gcd/numpy.py` still loads the real numpy. Never add a bare `import rust`-style relative import.

---

### Task 1: Core contract (`analytics/base.py`), dtype groupings, private plugin module

**Files:**
- Create: `services/analytics/analytics/_dtypes.py`, `services/analytics/analytics/base.py`
- Move: `services/analytics/analytics/__init__.py` → `services/analytics/analytics/_plugin.py`, then create a new `__init__.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Produces (used by every later task):
  - `analytics.base`: `STATUS`, `Column = tuple[str, str]`, `Combo = tuple[Column, ...]`, `class Technique` (class attributes `SCOPE, ARITY, METRICS, DESCRIPTORS={}, CONCLUSIONS={}, EXACT=True, RTOL=0.0, ATOL=0.0`; methods `add(frames) -> Technique`, `result() -> pl.DataFrame`; classmethods `key_columns()`, `enumerate(frames)`, `keys_frame(combos)`, `metrics_frame(combos, metrics, status="computed")`, `null_frame(combos, status)`, `rows_from_plugin(frame, plugin_rows)`; hooks `eligible(series)`, `compatible(dtypes)`, `describe(frames, combos)`, `_compute(frames, combos)` (abstract), `_conclude(out)`, `_on_add()`, `agreement(result, reference) -> list[str]`; attribute `self._collected` (the collected frames while `result()` runs)).
  - Module functions: `computed(expr)`, `at_least(col, threshold)`, `check_unit(name, value)`, `columns_of(combos)`, `group_by_frame(combos)`, `same_value(got, want, rtol, atol)`, `metric_mismatches(result, reference, keys, metrics, rtol, atol)`, `lazy_attributes(package, modules)`.
  - `analytics._dtypes`: `INTEGERS_64`, `STRING_LIKE`, `NESTED`, `encodable(dtype)`, `is_nested(dtype)`, `value_family(dtype)`.
  - `analytics._plugin`: every function currently in `analytics/__init__.py`, unchanged.

- [ ] **Step 1: Move the plugin wrappers to a private module and keep them re-exported**

```bash
git mv services/analytics/analytics/__init__.py services/analytics/analytics/_plugin.py
```

In `services/analytics/analytics/_plugin.py`, delete the line `__version__ = "0.1.0"`. `PLUGIN_PATH = Path(__file__).parent` stays correct, because the file is still next to `analytics.pyd`.

Create `services/analytics/analytics/__init__.py`:

```python
__version__ = "0.1.0"

# Transitional: the plugin wrappers stay importable from the top-level package
# until every technique has moved to its class (removed in Task 11). Importing
# _plugin also registers the `.analytics` expression namespace used by the old
# bloom_filter.py until Task 9.
from analytics._plugin import (  # noqa: E402,F401
    column_gcd,
    lsh_candidates,
    marginal_entropy,
    membership_ratio,
    membership_ratio_sample,
    minhash,
    pairwise_adjusted_rand,
    pairwise_chi_squared,
    pairwise_joint_entropy,
    threeway_joint_entropy,
)
```

- [ ] **Step 2: Confirm nothing broke**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"`
Expected: PASS (same count as before the move).

- [ ] **Step 3: Write the failing tests for the base contract**

Create `tests/test_base.py`:

```python
"""
Unit tests for the Technique contract (analytics/base.py) using a toy technique.

The toy's metric is the total length of the column names in a combination / 10,
so every expected value below can be worked out by hand.
"""

import math
import weakref

import polars as pl
import pytest

from analytics._dtypes import encodable, is_nested, value_family
from analytics.base import (
    STATUS,
    Technique,
    at_least,
    check_unit,
    columns_of,
    computed,
    group_by_frame,
    lazy_attributes,
    metric_mismatches,
    same_value,
)


class Toy(Technique):
    SCOPE = "ordered"
    ARITY = 2
    METRICS = {"score": pl.Float64}
    CONCLUSIONS = {"high": pl.Boolean}

    def __init__(self, *, threshold: float = 0.5):
        super().__init__()
        check_unit("threshold", threshold)
        self.threshold = threshold

    def eligible(self, series: pl.Series) -> bool:
        return series.dtype == pl.Int64

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, {"score": [sum(len(c) for _, c in k) / 10 for k in combos]})

    def _conclude(self, out):
        return out.with_columns(high=computed(at_least("score", self.threshold)))


class PerColumnToy(Toy):
    SCOPE = "per_column"
    ARITY = 1


class MultiSetToy(Toy):
    SCOPE = "multi_set"

    def eligible(self, series):
        return True

    def compatible(self, dtypes):
        return len({value_family(d) for d in dtypes}) == 1


class TripletToy(Toy):
    ARITY = 3

    def eligible(self, series):
        return True


FRAMES = {
    "f": pl.DataFrame({"a": [1, 2], "bb": [3, 4], "s": ["x", "y"]}),
    "g": pl.DataFrame({"a": [5, 6], "ccc": [7, 8]}),
}
F_A, F_BB, F_S, G_A, G_CCC = ("f", "a"), ("f", "bb"), ("f", "s"), ("g", "a"), ("g", "ccc")


# ── enumeration ──────────────────────────────────────────────────────────────

def test_enumerate_ordered_stays_within_each_frame():
    assert Toy.enumerate(FRAMES) == [(F_A, F_BB), (F_A, F_S), (F_BB, F_S), (G_A, G_CCC)]


def test_enumerate_per_column():
    assert PerColumnToy.enumerate(FRAMES) == [(F_A,), (F_BB,), (F_S,), (G_A,), (G_CCC,)]


def test_enumerate_multi_set_spans_frames_and_includes_same_frame_pairs():
    combos = MultiSetToy.enumerate(FRAMES)
    assert len(combos) == 10
    assert (F_A, G_A) in combos and (F_A, F_BB) in combos


def test_enumerate_ordered_triplets():
    assert TripletToy.enumerate(FRAMES) == [(F_A, F_BB, F_S)]


def test_key_columns():
    assert PerColumnToy.key_columns() == ["df_a", "col_a"]
    assert TripletToy.key_columns() == ["df_a", "col_a", "df_b", "col_b", "df_c", "col_c"]


# ── result() ─────────────────────────────────────────────────────────────────

def test_result_columns_statuses_and_conclusions():
    out = Toy(threshold=0.35).add(FRAMES).result()
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score", "high"]
    assert out.schema["status"] == STATUS
    assert out.rows() == [
        ("f", "a", "f", "bb", "computed", 0.3, False),
        ("f", "a", "f", "s", "ineligible", None, None),
        ("f", "bb", "f", "s", "ineligible", None, None),
        ("g", "a", "g", "ccc", "computed", 0.4, True),
    ]


def test_lazyframes_give_the_same_result():
    lazy = {n: f.lazy() for n, f in FRAMES.items()}
    assert Toy().add(lazy).result().equals(Toy().add(FRAMES).result())


def test_add_is_chainable_and_accumulates():
    t = Toy()
    assert t.add({"f": FRAMES["f"]}) is t
    t.add({"g": FRAMES["g"]})
    assert t.result().height == 4


def test_compatible_hook_marks_pairs_ineligible():
    out = MultiSetToy().add(FRAMES).result()
    statuses = dict(zip(zip(out["col_a"], out["col_b"], out["df_b"]), out["status"]))
    assert statuses[("a", "s", "f")] == "ineligible"   # Int64 vs String
    assert statuses[("a", "a", "g")] == "computed"


def test_descriptors_are_filled_on_every_row():
    class Described(PerColumnToy):
        DESCRIPTORS = {"dtype": pl.String}

        def describe(self, frames, combos):
            return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    out = Described().add(FRAMES).result()
    assert out.columns == ["df_a", "col_a", "status", "dtype", "score", "high"]
    assert out["dtype"].to_list() == ["Int64", "Int64", "String", "Int64", "Int64"]
    assert out.filter(pl.col("col_a") == "s")["status"].item() == "ineligible"


def test_frame_with_no_columns_gives_empty_result_with_schema():
    out = Toy().add({"e": pl.DataFrame()}).result()
    assert out.height == 0
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score", "high"]


def test_collected_frames_are_available_during_compute():
    seen = {}

    class Peek(Toy):
        def _compute(self, frames, combos):
            seen["same"] = self._collected is frames
            return super()._compute(frames, combos)

    Peek().add({n: f.lazy() for n, f in FRAMES.items()}).result()
    assert seen["same"]


# ── errors ───────────────────────────────────────────────────────────────────

def test_result_before_add_raises():
    with pytest.raises(ValueError, match="before add"):
        Toy().result()


def test_duplicate_frame_name_raises():
    t = Toy().add({"f": FRAMES["f"]})
    with pytest.raises(ValueError, match="already added"):
        t.add({"f": FRAMES["g"]})


@pytest.mark.parametrize("name", ["", 3])
def test_bad_frame_name_raises(name):
    with pytest.raises(ValueError, match="non-empty strings"):
        Toy().add({name: FRAMES["f"]})


def test_non_frame_raises():
    with pytest.raises(TypeError, match="DataFrame or LazyFrame"):
        Toy().add({"f": {"a": [1]}})


def test_wrong_schema_from_compute_raises():
    class Wrong(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, combos).rename({"score": "oops"})

    with pytest.raises(TypeError, match="Wrong._compute returned schema"):
        Wrong().add(FRAMES).result()


def test_missing_row_from_compute_raises():
    class Short(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, combos[:-1])

    with pytest.raises(ValueError, match="rows for"):
        Short().add(FRAMES).result()


def test_row_for_wrong_combination_raises():
    class Swapped(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, [(b, a) for a, b in combos])

    with pytest.raises(ValueError, match="exactly one row per eligible combination"):
        Swapped().add(FRAMES).result()


def test_compute_may_not_claim_ineligible():
    class Claims(Toy):
        def _compute(self, frames, combos):
            return self.null_frame(combos, "ineligible")

    with pytest.raises(ValueError, match="'computed' or 'pruned'"):
        Claims().add(FRAMES).result()


def test_pruned_rows_are_allowed_and_null():
    class Prunes(Toy):
        def _compute(self, frames, combos):
            return self.null_frame(combos, "pruned")

    out = Prunes().add(FRAMES).result()
    assert out["status"].to_list() == ["pruned", "ineligible", "ineligible", "pruned"]
    assert out["high"].null_count() == 4


def test_threshold_validation():
    with pytest.raises(ValueError, match="threshold must be in"):
        Toy(threshold=1.5)


def test_instances_are_not_kept_alive():
    t = Toy().add(FRAMES)
    t.result()
    ref = weakref.ref(t)
    del t
    assert ref() is None


# ── helpers ──────────────────────────────────────────────────────────────────

def test_at_least_treats_nan_as_false():
    df = pl.DataFrame({"v": [0.2, 0.3, float("nan")]})
    assert df.select(at_least("v", 0.3)).to_series().to_list() == [False, True, False]


def test_computed_nulls_non_computed_rows():
    df = pl.DataFrame({"status": pl.Series(["computed", "pruned", "ineligible"], dtype=STATUS)})
    assert df.select(computed(pl.lit(True))).to_series().to_list() == [True, None, None]


def test_columns_of_and_group_by_frame():
    combos = [(F_A, F_BB), (F_A, F_S), (G_A, G_CCC)]
    assert columns_of(combos) == [F_A, F_BB, F_S, G_A, G_CCC]
    assert group_by_frame(combos) == {"f": combos[:2], "g": combos[2:]}


def test_rows_from_plugin_attaches_keys_and_status():
    plugin_rows = pl.DataFrame({"col_a": ["a"], "col_b": ["bb"], "score": [0.3]})
    out = Toy.rows_from_plugin("f", plugin_rows)
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score"]
    assert out.row(0) == ("f", "a", "f", "bb", "computed", 0.3)


@pytest.mark.parametrize(
    "got, want, rtol, atol, same",
    [
        (None, None, 0, 0, True),
        (None, 1.0, 0, 0, False),
        (float("nan"), float("nan"), 0, 0, True),
        (float("nan"), 1.0, 0, 0, False),
        (1.0, 1.00001, 1e-4, 0, True),
        (1.0, 1.001, 1e-4, 0, False),
        (0.0, 1e-13, 0, 1e-12, True),
        (2**100, 2**100, 0, 0, True),
        (True, False, 0, 0, False),
    ],
)
def test_same_value(got, want, rtol, atol, same):
    assert same_value(got, want, rtol, atol) is same


def test_metric_mismatches_reports_status_and_value_differences():
    a = Toy(threshold=0.35).add(FRAMES).result()
    b = a.with_columns(score=pl.when(pl.col("col_b") == "ccc").then(0.5).otherwise(pl.col("score")))
    problems = metric_mismatches(b, a, Toy.key_columns(), ["score"], 0.0, 0.0)
    assert problems == ["g.a ~ g.ccc: score 0.5 != 0.4"]
    assert metric_mismatches(a.head(1), a, Toy.key_columns(), ["score"], 0, 0) == [
        "key columns differ from the reference"
    ]


def test_default_agreement_uses_rtol_atol():
    a = Toy().add(FRAMES).result()
    assert Toy().agreement(a, a) == []


def test_lazy_attributes_imports_on_first_use():
    getter = lazy_attributes("json", {"JSONDecoder": ".decoder"})
    import json.decoder

    assert getter("JSONDecoder") is json.decoder.JSONDecoder
    with pytest.raises(AttributeError):
        getter("Nope")


# ── dtype groupings ──────────────────────────────────────────────────────────

def test_value_family():
    assert value_family(pl.Int32()) == value_family(pl.UInt64()) == "int"
    assert value_family(pl.String()) == value_family(pl.Categorical()) == "str"
    assert value_family(pl.Date()) != value_family(pl.Int32())
    assert value_family(pl.Float32()) != value_family(pl.Float64())
    assert value_family(pl.Int128()) != "int"


def test_encodable_and_nested():
    assert encodable(pl.List(pl.Int32)) and is_nested(pl.Array(pl.Int32, 2))
    assert not encodable(pl.Struct({"a": pl.Int64})) and not encodable(pl.Binary()) and not encodable(pl.Null())
    assert not is_nested(pl.Int64())
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_base.py -q`
Expected: collection ERROR `ModuleNotFoundError: No module named 'analytics._dtypes'`.

- [ ] **Step 5: Write `_dtypes.py`**

Create `services/analytics/analytics/_dtypes.py`:

```python
"""Dtype groupings shared by the technique bases."""

import polars as pl

INTEGERS_64 = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)
NESTED = (pl.List, pl.Array)

# Dtypes the Rust encoder (src/shared.rs::encode_series) accepts. Anything else
# (Struct, Binary, Null, Object, UInt128) makes the plugin raise.
ENCODABLE = (
    *INTEGERS_64, pl.Int128, pl.Boolean, pl.Float32, pl.Float64,
    pl.Date, pl.Datetime, pl.Duration, pl.Time, *STRING_LIKE, pl.Decimal, *NESTED,
)


def encodable(dtype: pl.DataType) -> bool:
    return isinstance(dtype, ENCODABLE)


def is_nested(dtype: pl.DataType) -> bool:
    return isinstance(dtype, NESTED)


def value_family(dtype: pl.DataType) -> str:
    """Values of two columns are only comparable (multi-set techniques) within one family.

    Integers up to 64 bits share a family because the Rust encoder widens them to
    the same u64 key; String/Categorical/Enum share one because categoricals hash
    by label. Everything else must match its dtype exactly: a Date 19000 and an
    Int32 19000 share a physical key in Rust but are different values.
    """
    if isinstance(dtype, INTEGERS_64):
        return "int"
    if isinstance(dtype, STRING_LIKE):
        return "str"
    return str(dtype)
```

- [ ] **Step 6: Write `base.py`**

Create `services/analytics/analytics/base.py`:

```python
"""Uniform contract shared by every analytical technique.

Every technique is used the same way:

    result = Impl(**params).add({"name": frame, ...}).result()

and returns one flat table, in canonical combination order:

    df_a, col_a[, df_b, col_b[, df_c, col_c]] | status | DESCRIPTORS | METRICS | CONCLUSIONS

The base enumerates column combinations and decides eligibility, so every
implementation of a technique computes metrics for exactly the same rows. An
implementation supplies only `_compute`; the technique base supplies
`eligible` / `compatible` / `describe` / `_conclude`.

status: "computed"   metrics filled in
        "ineligible" a column's dtype/content (or the pair's value families) do not
                     qualify - metrics and conclusions null
        "pruned"     a probabilistic implementation chose not to evaluate the
                     combination - metrics and conclusions null
Null means "not computed"; NaN means "computed but mathematically undefined".
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Callable, ClassVar, Literal, Sequence

import polars as pl

Scope = Literal["per_column", "multi_set", "ordered"]
Column = tuple[str, str]  # (frame name, column name)
Combo = tuple[Column, ...]  # ARITY columns
STATUS = pl.Enum(["computed", "ineligible", "pruned"])
_SUFFIXES = ("a", "b", "c")


class Technique(ABC):
    SCOPE: ClassVar[Scope]
    ARITY: ClassVar[int]
    METRICS: ClassVar[dict[str, pl.DataType]]
    DESCRIPTORS: ClassVar[dict[str, pl.DataType]] = {}
    CONCLUSIONS: ClassVar[dict[str, pl.DataType]] = {}
    EXACT: ClassVar[bool] = True
    RTOL: ClassVar[float] = 0.0
    ATOL: ClassVar[float] = 0.0

    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame | pl.LazyFrame] = {}
        self._collected: dict[str, pl.DataFrame] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, frames: dict[str, pl.DataFrame | pl.LazyFrame]) -> Technique:
        """Register named frames. Names are unique for the life of the instance."""
        for name, frame in frames.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"frame names must be non-empty strings, got {name!r}")
            if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
                raise TypeError(
                    f"frame {name!r} must be a polars DataFrame or LazyFrame, got {type(frame).__name__}"
                )
            if name in self._frames:
                raise ValueError(f"frame {name!r} already added")
        self._frames.update(frames)
        self._on_add()
        return self

    def result(self) -> pl.DataFrame:
        if not self._frames:
            raise ValueError(f"{type(self).__name__}.result() called before add()")
        frames = {n: f.collect() if isinstance(f, pl.LazyFrame) else f for n, f in self._frames.items()}
        self._collected = frames
        try:
            ok = {(n, c): self.eligible(f[c]) for n, f in frames.items() for c in f.columns}
            combos = self.enumerate(frames)
            usable = [
                all(ok[col] for col in k) and self.compatible([frames[n].schema[c] for n, c in k])
                for k in combos
            ]
            good = [k for k, u in zip(combos, usable) if u]
            bad = [k for k, u in zip(combos, usable) if not u]
            rows = self._compute(frames, good) if good else self.null_frame([], "computed")
            self._check(rows, len(good))
            keys = self.key_columns()
            columns = [*keys, "status", *self.METRICS]
            out = self.keys_frame(combos).join(
                pl.concat([rows.select(columns), self.null_frame(bad, "ineligible")]),
                on=keys,
                how="left",
                maintain_order="left",
            )
            if out.height != len(combos) or out["status"].null_count():
                raise ValueError(
                    f"{type(self).__name__}._compute must return exactly one row per eligible combination"
                )
            described = self.describe(frames, combos)
            out = out.with_columns([pl.Series(k, v, dtype=self.DESCRIPTORS[k]) for k, v in described.items()])
            out = self._conclude(out)
            return out.select(*keys, "status", *self.DESCRIPTORS, *self.METRICS, *self.CONCLUSIONS)
        finally:
            self._collected = {}

    # ── row builders (used by implementations) ────────────────────────────────

    @classmethod
    def key_columns(cls) -> list[str]:
        return [f"{p}_{s}" for s in _SUFFIXES[: cls.ARITY] for p in ("df", "col")]

    @classmethod
    def enumerate(cls, frames: dict[str, pl.DataFrame]) -> list[Combo]:
        """Canonical combinations: frame insertion order, then column order."""
        if cls.SCOPE == "per_column":
            return [((n, c),) for n, f in frames.items() for c in f.columns]
        if cls.SCOPE == "ordered":
            return [
                k
                for n, f in frames.items()
                for k in combinations([(n, c) for c in f.columns], cls.ARITY)
            ]
        return list(combinations([(n, c) for n, f in frames.items() for c in f.columns], cls.ARITY))

    @classmethod
    def keys_frame(cls, combos: Sequence[Combo]) -> pl.DataFrame:
        data: dict[str, list[str]] = {}
        for i, s in enumerate(_SUFFIXES[: cls.ARITY]):
            data[f"df_{s}"] = [k[i][0] for k in combos]
            data[f"col_{s}"] = [k[i][1] for k in combos]
        return pl.DataFrame(data, schema={k: pl.String for k in cls.key_columns()})

    @classmethod
    def metrics_frame(
        cls,
        combos: Sequence[Combo],
        metrics: dict[str, Sequence],
        status: str | Sequence[str] = "computed",
    ) -> pl.DataFrame:
        """keys + status + METRICS, one row per combo, values in combo order."""
        statuses = [status] * len(combos) if isinstance(status, str) else list(status)
        return cls.keys_frame(combos).with_columns(
            pl.Series("status", statuses, dtype=STATUS),
            *(pl.Series(name, list(metrics[name]), dtype=dtype, strict=False) for name, dtype in cls.METRICS.items()),
        )

    @classmethod
    def null_frame(cls, combos: Sequence[Combo], status: str) -> pl.DataFrame:
        return cls.metrics_frame(combos, {m: [None] * len(combos) for m in cls.METRICS}, status)

    @classmethod
    def rows_from_plugin(cls, frame: str, plugin_rows: pl.DataFrame) -> pl.DataFrame:
        """Plugin output keyed by col_a[, col_b[, col_c]] -> keys + computed status + METRICS."""
        return plugin_rows.with_columns(
            *(pl.lit(frame).alias(f"df_{s}") for s in _SUFFIXES[: cls.ARITY]),
            pl.lit("computed", dtype=STATUS).alias("status"),
        ).select(*cls.key_columns(), "status", *(pl.col(m).cast(dt) for m, dt in cls.METRICS.items()))

    # ── hooks ─────────────────────────────────────────────────────────────────

    def eligible(self, series: pl.Series) -> bool:
        """Technique base: may this column take part at all?"""
        return True

    def compatible(self, dtypes: Sequence[pl.DataType]) -> bool:
        """Technique base: may these columns be compared with each other?"""
        return True

    def describe(self, frames: dict[str, pl.DataFrame], combos: list[Combo]) -> dict[str, list]:
        """Technique base: DESCRIPTORS values for every combo (eligible or not)."""
        return {}

    @abstractmethod
    def _compute(self, frames: dict[str, pl.DataFrame], combos: list[Combo]) -> pl.DataFrame:
        """Implementation: keys + status ("computed"/"pruned") + METRICS for exactly `combos`."""

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        """Technique base: add CONCLUSIONS (null unless status == computed)."""
        return out

    def _on_add(self) -> None:
        """Called after every add(); e.g. clears per-instance caches."""

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """Problems found comparing `result` with the reference implementation's result."""
        return metric_mismatches(result, reference, self.key_columns(), list(self.METRICS), self.RTOL, self.ATOL)

    # ── internal ──────────────────────────────────────────────────────────────

    def _check(self, rows: pl.DataFrame, expected_rows: int) -> None:
        name = type(self).__name__
        expected = {**{k: pl.String for k in self.key_columns()}, "status": STATUS, **self.METRICS}
        if dict(rows.schema) != expected:
            raise TypeError(f"{name}._compute returned schema {dict(rows.schema)}, expected {expected}")
        if rows.height != expected_rows:
            raise ValueError(f"{name}._compute returned {rows.height} rows for {expected_rows} combinations")
        if (rows["status"] == "ineligible").any():
            raise ValueError(f"{name}._compute may only return status 'computed' or 'pruned'")


# ── helpers for technique bases and implementations ─────────────────────────────

def computed(expr: pl.Expr) -> pl.Expr:
    """`expr` where status == computed, null elsewhere."""
    return pl.when(pl.col("status") == "computed").then(expr)


def at_least(column: str, threshold: float) -> pl.Expr:
    """column >= threshold, with NaN -> False (Polars orders NaN above every number)."""
    return pl.col(column).is_not_nan() & (pl.col(column) >= threshold)


def check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def columns_of(combos: Sequence[Combo]) -> list[Column]:
    """Distinct columns used by `combos`, in first-seen order."""
    return list(dict.fromkeys(col for k in combos for col in k))


def group_by_frame(combos: Sequence[Combo]) -> dict[str, list[Combo]]:
    """Combos grouped by the frame of their first column (ordered/per-column scopes)."""
    groups: dict[str, list[Combo]] = {}
    for k in combos:
        groups.setdefault(k[0][0], []).append(k)
    return groups


def same_value(got, want, rtol: float, atol: float) -> bool:
    if got is None or want is None:
        return got is None and want is None
    if isinstance(got, float) or isinstance(want, float):
        if math.isnan(got) or math.isnan(want):
            return math.isnan(got) and math.isnan(want)
        return math.isclose(got, want, rel_tol=rtol, abs_tol=atol)
    return got == want


def metric_mismatches(
    result: pl.DataFrame,
    reference: pl.DataFrame,
    keys: list[str],
    metrics: list[str],
    rtol: float,
    atol: float,
) -> list[str]:
    """Row-by-row differences in status and `metrics`; both frames in canonical order."""
    key_rows = result.select(keys).rows()
    if key_rows != reference.select(keys).rows():
        return ["key columns differ from the reference"]
    labels = [" ~ ".join(f"{r[i]}.{r[i + 1]}" for i in range(0, len(r), 2)) for r in key_rows]
    problems = [
        f"{label}: status {got} != {want}"
        for label, got, want in zip(labels, result["status"], reference["status"])
        if got != want
    ]
    for m in metrics:
        for label, got, want in zip(labels, result[m].to_list(), reference[m].to_list()):
            if not same_value(got, want, rtol, atol):
                problems.append(f"{label}: {m} {got!r} != {want!r}")
    return problems


def lazy_attributes(package: str, modules: dict[str, str]) -> Callable[[str], type]:
    """Module-level __getattr__ that imports optional implementations on first use,
    so `import analytics.<technique>` never needs third-party reference libraries."""

    def __getattr__(name: str) -> type:
        if name in modules:
            return getattr(importlib.import_module(modules[name], package), name)
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__
```

- [ ] **Step 7: Run the new tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_base.py -q`
Expected: all PASS.

- [ ] **Step 8: Run the full suite**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add services/analytics/analytics/ tests/test_base.py
git commit -m "feat: Technique base contract; plugin wrappers moved to private _plugin

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Shared test data and test helpers

**Files:**
- Create: `tests/datagen.py`, `tests/harness.py`, `tests/test_datagen.py`
- Modify: `tests/conftest.py` (the `dataset` fixture delegates to `datagen.mixed_dtypes`; generator code moves out)

**Interfaces:**
- Consumes: `analytics.base.STATUS`, `Technique`.
- Produces:
  - `datagen.mixed_dtypes(n_rows=1_000, seed=42) -> pl.DataFrame` (18 columns named `{float64|uint32|boolean|categorical|list|arr}_{skewed|high_unique|sparse}`)
  - `datagen.low_cardinality(n_rows, n_cols, cardinality=20, null_rate=0.05, seed=42) -> pl.DataFrame` (columns `c000…`)
  - `datagen.related_frames(n_rows=1_000, seed=42) -> dict[str, pl.DataFrame]` (`customers`, `orders`, `archive`)
  - `datagen.similar_frames(n_similar=8, n_independent=12, col_size=200, n_elements=2_000, seed=42) -> dict[str, pl.DataFrame]` (`df0`, `df1`)
  - `datagen.integer_multiples(n_rows, n_cols, g, seed=42)`, `datagen.integer_random(n_rows, n_cols, seed=42)` (columns `c0…`)
  - `harness.implementation_params(package, include_reference=True) -> list[pytest.param]` (values `"package:ClassName"`)
  - `harness.load(spec) -> type` (skips if an optional library is missing), `harness.reference(package) -> type`
  - `harness.run(cls, frames, **params) -> pl.DataFrame`
  - `harness.assert_contract(cls, result, frames)`, `harness.assert_agrees(impl, result, reference_result)`
  - `harness.with_metrics(base, **metrics) -> type`

- [ ] **Step 1: Write the failing datagen tests**

Create `tests/test_datagen.py`:

```python
"""The planted structure that other tests and benchmarks rely on."""

import polars as pl

from datagen import integer_multiples, integer_random, low_cardinality, mixed_dtypes, related_frames, similar_frames


def test_mixed_dtypes_layout():
    df = mixed_dtypes(200)
    assert df.shape == (200, 18)
    assert df.columns[:3] == ["float64_skewed", "float64_high_unique", "float64_sparse"]
    assert df.schema["categorical_skewed"] == pl.Categorical


def test_mixed_dtypes_is_deterministic_per_seed():
    assert mixed_dtypes(100).equals(mixed_dtypes(100))
    assert not mixed_dtypes(100).equals(mixed_dtypes(100, seed=7))


def test_low_cardinality():
    df = low_cardinality(1_000, 10, cardinality=5)
    assert df.shape == (1_000, 10) and df.columns[0] == "c000"
    assert df["c000"].drop_nulls().n_unique() <= 5
    assert 0 < df["c000"].null_count() < 200


def test_related_frames_planted_relationships():
    f = related_frames(1_000)
    ids = set(f["customers"]["id"].to_list())
    assert f["customers"]["id"].n_unique() == 1_000
    assert set(f["orders"]["customer_id"].drop_nulls().to_list()) <= ids
    assert f["orders"]["customer_id"].null_count() > 0
    assert set(f["archive"]["id"].to_list()) == ids
    assert set(f["orders"]["region"].to_list()) < set(f["customers"]["region"].to_list())
    shared = set(f["archive"]["name"].to_list()) & set(f["customers"]["name"].to_list())
    assert 0.9 < len(shared) / 1_000 < 0.95


def test_similar_frames():
    f = similar_frames()
    assert set(f) == {"df0", "df1"}
    a, b = set(f["df0"]["sim_00"].drop_nulls().to_list()), set(f["df1"]["sim_00"].drop_nulls().to_list())
    assert len(a & b) / min(len(a), len(b)) > 0.9


def test_integer_generators():
    m = integer_multiples(1_000, 3, 12)
    assert m.columns == ["c0", "c1", "c2"] and (m["c0"] % 12 == 0).all()
    assert integer_random(1_000, 2).shape == (1_000, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_datagen.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'datagen'`.

- [ ] **Step 3: Create `tests/datagen.py`**

`_skewed_tokens`, `_high_unique_tokens`, `_sparse_tokens` and `make_column` are moved **unchanged** from `tests/conftest.py`. `make_dataset` becomes `mixed_dtypes`.

```python
"""
Seeded toy-data generators shared by accuracy tests (small sizes) and speed
benchmarks (large sizes). Every generator is deterministic for a given seed.

Ordered techniques:   mixed_dtypes, low_cardinality
Multi-set techniques: related_frames, similar_frames
Per-column (GCD):     integer_multiples, integer_random
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa


def _with_nulls(name: str, values: np.ndarray, null_mask: np.ndarray) -> pl.Series:
    return pl.from_arrow(pa.array(values, mask=null_mask)).alias(name)


# ── ordered techniques ────────────────────────────────────────────────────────

def _skewed_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """30 tokens drawn with power-law weights (heavy skew, ~30 distinct values)."""
    k = np.arange(1, 31)
    w = 1.0 / k ** 0.8
    w /= w.sum()
    return rng.choice(30, size=n, p=w)


def _high_unique_tokens(n: int, rng: np.random.Generator) -> np.ndarray:
    """All n tokens distinct (random permutation)."""
    return rng.permutation(n)


def _sparse_tokens(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """200 unique tokens at random positions; remaining positions forced null.
       Returns (tokens, null_mask) — tokens[i] is only valid when null_mask[i]=False.
    """
    tokens = np.empty(n, dtype=np.int64)
    positions = rng.choice(n, size=min(200, n), replace=False)
    tokens[positions] = np.arange(len(positions))
    null_mask = np.ones(n, dtype=bool)
    null_mask[positions] = False
    return tokens, null_mask


def make_column(
    name: str,
    dtype_spec: str,
    shape: str,
    n_rows: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Build one pl.Series with the given dtype and distribution shape.

    dtype_spec : "float64" | "uint32" | "boolean" | "categorical" | "list" | "arr"
    shape      : "skewed" | "high_unique" | "sparse"
    """
    if shape == "sparse":
        tokens, null_mask = _sparse_tokens(n_rows, rng)
    else:
        tokens = _skewed_tokens(n_rows, rng) if shape == "skewed" else _high_unique_tokens(n_rows, rng)
        null_rate = 0.10 if shape == "skewed" else 0.05
        null_mask = rng.random(n_rows) < null_rate

    values: list = []
    for i in range(n_rows):
        if null_mask[i]:
            values.append(None)
            continue
        t = int(tokens[i])
        if dtype_spec == "float64":
            values.append(float(t))
        elif dtype_spec == "uint32":
            values.append(t)
        elif dtype_spec == "boolean":
            values.append(bool(t % 2))
        elif dtype_spec == "categorical":
            values.append(f"cat_{t}")
        elif dtype_spec == "list":
            values.append([t, t + 1])
        else:  # arr
            values.append([t, t + 1, t + 2])

    if dtype_spec == "float64":
        return pl.Series(name, values, dtype=pl.Float64)
    if dtype_spec == "uint32":
        return pl.Series(name, values, dtype=pl.UInt32)
    if dtype_spec == "boolean":
        return pl.Series(name, values, dtype=pl.Boolean)
    if dtype_spec == "categorical":
        return pl.Series(name, values, dtype=pl.String).cast(pl.Categorical)
    if dtype_spec == "list":
        return pl.Series(name, values, dtype=pl.List(pl.Int32))
    return pl.Series(name, values, dtype=pl.Array(pl.Int32, 3))


def mixed_dtypes(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame:
    """18 columns (6 dtypes × 3 shapes) named {dtype_spec}_{shape}, e.g. categorical_sparse."""
    rng = np.random.default_rng(seed)
    cols = []
    for dtype_spec in ("float64", "uint32", "boolean", "categorical", "list", "arr"):
        for shape in ("skewed", "high_unique", "sparse"):
            cols.append(make_column(f"{dtype_spec}_{shape}", dtype_spec, shape, n_rows, rng))
    return pl.DataFrame(cols)


def low_cardinality(
    n_rows: int, n_cols: int, cardinality: int = 20, null_rate: float = 0.05, seed: int = 42
) -> pl.DataFrame:
    """Int64 columns c000… with `cardinality` values each. Every 5th column is a noisy
    copy of the previous one, so some pairs are genuinely associated."""
    rng = np.random.default_rng(seed)
    cols, prev = [], None
    for i in range(n_cols):
        if prev is not None and i % 5 == 4:
            values = (prev + rng.integers(0, 2, n_rows)) % cardinality
        else:
            values = rng.integers(0, cardinality, n_rows)
        prev = values
        cols.append(_with_nulls(f"c{i:03d}", values, rng.random(n_rows) < null_rate))
    return pl.DataFrame(cols)


# ── multi-set techniques ──────────────────────────────────────────────────────

def related_frames(n_rows: int = 1_000, seed: int = 42) -> dict[str, pl.DataFrame]:
    """Three frames with planted relationships (enumeration order customers, orders, archive):

        customers.id     ~ orders.customer_id    pk_fk   (FK, 10% null)
        customers.id     ~ archive.id            pk_pk   (same id set, shuffled)
        customers.name   ~ orders.customer_name  pk_fk
        customers.name   ~ archive.name          ~93% shared - similar, not contained
        customers.region ~ orders.region         b_in_a  (orders uses 2 of 4 regions)
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_rows) + 1_000_000
    names = [f"cust_{i}" for i in ids]
    regions = np.array(["north", "south", "east", "west"])
    customers = pl.DataFrame(
        {
            "id": ids,
            "name": names,
            "region": rng.choice(regions, n_rows),
            "score": rng.integers(0, 100, n_rows),
        }
    )
    pick = rng.integers(0, n_rows, n_rows)
    orders = pl.DataFrame(
        [
            pl.Series("order_id", np.arange(n_rows) + 5_000_000),
            _with_nulls("customer_id", ids[pick], rng.random(n_rows) < 0.10),
            pl.Series("customer_name", [names[i] for i in pick]),
            pl.Series("region", rng.choice(regions[:2], n_rows)),
            pl.Series("amount", rng.integers(1, 500, n_rows)),
        ]
    )
    archive_names = [names[i] for i in rng.permutation(n_rows)]
    for i in np.flatnonzero(rng.random(n_rows) < 0.07):
        archive_names[i] = f"new_{i}"
    archive = pl.DataFrame({"id": rng.permutation(ids), "name": archive_names})
    return {"customers": customers, "orders": orders, "archive": archive}


def similar_frames(
    n_similar: int = 8,
    n_independent: int = 12,
    col_size: int = 200,
    n_elements: int = 2_000,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Two frames each with (n_similar + n_independent) columns. Each of the n_similar
    cross-frame pairs sim_ii shares >= 93% of elements (OC above 0.9). Independent
    columns come from disjoint regions of the element space to minimise accidental
    similarity. Shorter columns are null-padded."""
    rng = np.random.default_rng(seed)
    df0: dict[str, list] = {}
    df1: dict[str, list] = {}
    for i in range(n_similar):
        base = rng.choice(n_elements, size=col_size, replace=False).tolist()
        shared = int(col_size * 0.93)
        extra = rng.integers(0, n_elements, size=col_size - shared).tolist()
        df0[f"sim_{i:02d}"] = base
        df1[f"sim_{i:02d}"] = base[:shared] + extra
    region = n_elements // max(n_independent, 1)
    for i in range(n_independent):
        lo = (i * region) % n_elements
        pool = list(range(lo, min(lo + region, n_elements))) or list(range(n_elements))
        size = int(min(len(pool), rng.integers(50, col_size)))
        df0[f"ind0_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()
        df1[f"ind1_{i:02d}"] = rng.choice(pool, size=size, replace=False).tolist()

    def to_frame(cols: dict[str, list]) -> pl.DataFrame:
        longest = max(len(v) for v in cols.values())
        return pl.DataFrame({k: v + [None] * (longest - len(v)) for k, v in cols.items()})

    return {"df0": to_frame(df0), "df1": to_frame(df1)}


# ── per-column (GCD) ──────────────────────────────────────────────────────────

def integer_multiples(n_rows: int, n_cols: int, g: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… whose values are k·g, k uniform in [-2**40, 2**40)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({f"c{i}": rng.integers(-(2**40), 2**40, n_rows, dtype=np.int64) * g for i in range(n_cols)})


def integer_random(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """Int64 columns c0… of random values; whole-column GCD is 1 almost surely (early exit)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({f"c{i}": rng.integers(-(2**62), 2**62, n_rows, dtype=np.int64) for i in range(n_cols)})
```

In `_sparse_tokens`, `min(200, n)` replaces the original `200` so that small `n_rows` values (tests use 100–200) stay valid. For `n_rows ≥ 200` the output is identical to before.

- [ ] **Step 4: Point conftest at datagen**

Replace the whole body of `tests/conftest.py` after the imports. Keep the `sys.path.insert(...)` line until Task 11, because the old test modules still import top-level service modules.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analytics"))

import polars as pl
import pytest

from datagen import mixed_dtypes


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow — deselect with '-m \"not slow\"'",
    )


N_ROWS: int = 1_000  # Change here to scale the shared test dataset


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """18-column DataFrame shared across all test modules in the session."""
    return mixed_dtypes(N_ROWS)
```

- [ ] **Step 5: Create `tests/harness.py`**

```python
"""
Shared accuracy-test helpers. Every technique's test file uses them the same way:

    PKG = "analytics.<technique>"
    ALL = implementation_params(PKG)                            # contract block
    OTHERS = implementation_params(PKG, include_reference=False)  # agreement block

    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))
"""

from __future__ import annotations

import importlib

import polars as pl
import pytest

from analytics.base import STATUS, Technique


def implementation_params(package: str, include_reference: bool = True) -> list:
    module = importlib.import_module(package)
    return [
        pytest.param(f"{package}:{name}", id=name)
        for name in module.IMPLEMENTATIONS
        if include_reference or name != module.REFERENCE
    ]


def load(spec: str) -> type[Technique]:
    """'package:ClassName' -> class; skips the test (visibly) if its library is missing."""
    package, name = spec.split(":")
    try:
        return getattr(importlib.import_module(package), name)
    except ImportError as exc:
        pytest.skip(f"{name} unavailable: {exc}")


def reference(package: str) -> type[Technique]:
    return load(f"{package}:{importlib.import_module(package).REFERENCE}")


def run(cls: type[Technique], frames: dict, **params) -> pl.DataFrame:
    return cls(**params).add(frames).result()


def assert_contract(cls: type[Technique], result: pl.DataFrame, frames: dict) -> None:
    keys = cls.key_columns()
    expected = {
        **{k: pl.String for k in keys},
        "status": STATUS,
        **cls.DESCRIPTORS,
        **cls.METRICS,
        **cls.CONCLUSIONS,
    }
    assert list(result.schema.items()) == list(expected.items())
    collected = {n: f.collect() if isinstance(f, pl.LazyFrame) else f for n, f in frames.items()}
    assert result.select(keys).equals(cls.keys_frame(cls.enumerate(collected))), "rows must be every combination, in canonical order"
    idle = result.filter(pl.col("status") != "computed")
    for col in [*cls.METRICS, *cls.CONCLUSIONS]:
        assert idle[col].null_count() == idle.height, f"{col} must be null where status != computed"
    for col in cls.DESCRIPTORS:
        assert result[col].null_count() == 0, f"descriptor {col} must be filled on every row"


def assert_agrees(impl: Technique, result: pl.DataFrame, reference_result: pl.DataFrame) -> None:
    problems = impl.agreement(result, reference_result)
    assert not problems, f"{type(impl).__name__} disagrees with the reference:\n" + "\n".join(problems[:20])


def with_metrics(base: type[Technique], **metrics: list) -> type[Technique]:
    """Subclass of a technique base whose _compute returns `metrics` verbatim, so
    conclusion logic can be tested once, independently of any implementation."""

    class Fixed(base):
        def _compute(self, frames, combos):
            return self.metrics_frame(combos, metrics)

    Fixed.__name__ = f"Fixed{base.__name__}"
    return Fixed
```

- [ ] **Step 6: Run the datagen tests and the full suite**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_datagen.py -q` → all PASS.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS. The old tests still use the `dataset` fixture, which now comes from `mixed_dtypes`, and give identical data at `N_ROWS=1_000`.

- [ ] **Step 7: Commit**

```bash
git add tests/datagen.py tests/harness.py tests/test_datagen.py tests/conftest.py
git commit -m "test: shared seeded datagen and accuracy-test harness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Shared speed-benchmark harness

**Files:**
- Create: `tests/performance/harness.py`, `tests/test_benchmark_harness.py`
- Modify: `.gitignore` (add `tests/performance/results/`)

**Interfaces:**
- Consumes: the technique package protocol, meaning a module with `REFERENCE: str`, `IMPLEMENTATIONS: tuple[str, ...]` and implementation classes that have `.agreement()`.
- Produces:
  - `Dataset(name: str, make: Callable[[], dict[str, pl.DataFrame]], exclude: tuple[str, ...] = ())`
  - `large_dataset(columns: int | None = None) -> Dataset`
  - `time_call(fn, runs, budget_s) -> tuple[list[float] | None, object]`
  - `speedups(rows) -> list[dict]`
  - `run(package, datasets, params=None, impl_params=None, runs=5, budget_s=60.0) -> pl.DataFrame | None`
  - Every benchmark script starts with `from harness import …` before `from datagen import …`, because importing `harness` puts `tests/` on `sys.path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_harness.py`:

```python
"""Pure parts of the benchmark harness (timing loop and speedup arithmetic). Nothing is benchmarked here."""

import time

from performance.harness import speedups, time_call


def test_time_call_warms_up_then_times_every_run():
    calls = []
    times, result = time_call(lambda: calls.append(1) or len(calls), runs=3, budget_s=10.0)
    assert len(times) == 3
    assert len(calls) == 4  # 1 warm-up + 3 timed
    assert result == 4


def test_time_call_skips_when_warmup_exceeds_budget():
    times, result = time_call(lambda: time.sleep(0.05), runs=3, budget_s=0.01)
    assert times is None and result is None


def _row(impl, threads, median, status="ok"):
    return {"dataset": "d", "implementation": impl, "threads": threads, "status": status, "median_ms": median}


def test_speedups_separate_algorithm_from_parallelism():
    rows = [
        _row("XRust", "N", 10.0),
        _row("XRust", "1", 40.0),
        _row("XNumpy", "default", 200.0),
        _row("XMath", "default", 400.0),
    ]
    assert speedups(rows) == [
        {"dataset": "d", "implementation": "XRust", "algorithmic": 5.0, "parallel": 4.0, "total": 20.0}
    ]


def test_speedups_without_a_finished_competitor():
    rows = [
        _row("XRust", "N", 10.0),
        _row("XRust", "1", 40.0),
        _row("XMath", "default", None, status="skipped: over budget (60s)"),
    ]
    assert speedups(rows) == [
        {"dataset": "d", "implementation": "XRust", "algorithmic": None, "parallel": 4.0, "total": None}
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_benchmark_harness.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'performance.harness'`.

- [ ] **Step 3: Create `tests/performance/harness.py`**

```python
"""
Shared speed-benchmark harness. Every tests/performance/benchmark_<technique>.py is:

    from harness import Dataset, large_dataset, run
    from datagen import ...
    if __name__ == "__main__":
        run("analytics.<technique>", [large_dataset(), Dataset("narrow …", …), Dataset("wide …", …)])

What is timed: Impl(**params).add(frames).result() — the whole analytical question,
fresh instance per run (so caches never carry over), 1 warm-up + `runs` timed runs,
median and min reported. An implementation whose warm-up exceeds `budget_s` is
reported "skipped: over budget" and not timed.

Hypothesis split: each *Rust implementation is also timed at 1 thread in a child
process (RAYON_NUM_THREADS=1, POLARS_MAX_THREADS=1 — both pools are fixed at process
start), giving
    algorithmic = fastest non-Rust ÷ Rust@1 thread
    parallel    = Rust@1 thread   ÷ Rust@N threads
    total       = fastest non-Rust ÷ Rust@N threads

Sanity check (never a gate): each result is compared with the reference via
impl.agreement(); the pytest suite is the correctness gate.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1]))  # tests/ -> `import datagen` in benchmark scripts

import analytics  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
LARGE_DATASET = Path(__file__).parents[1] / "data" / "large_dataset.arrow"
SINGLE_THREAD_ENV = {"RAYON_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
_CHILD_ENV = "ANALYTICS_BENCH_CHILD"


@dataclass(frozen=True)
class Dataset:
    name: str
    make: Callable[[], dict[str, pl.DataFrame]]
    exclude: tuple[str, ...] = ()  # implementations not run on this dataset (e.g. would exhaust memory)


def large_dataset(columns: int | None = None) -> Dataset:
    """tests/data/large_dataset.arrow (50K rows × 101 cols), optionally only its first `columns`."""
    label = "large_dataset.arrow" + (f" (first {columns} cols)" if columns else "")

    def make() -> dict[str, pl.DataFrame]:
        df = pl.read_ipc(LARGE_DATASET)
        return {"large": df.select(df.columns[:columns]) if columns else df}

    return Dataset(label, make)


def time_call(fn: Callable[[], object], runs: int, budget_s: float) -> tuple[list[float] | None, object]:
    t0 = time.perf_counter()
    result = fn()
    if time.perf_counter() - t0 > budget_s:
        return None, None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def speedups(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        ok = [r for r in rows if r["dataset"] == dataset and r["status"] == "ok"]
        others = [r["median_ms"] for r in ok if not r["implementation"].endswith("Rust")]
        best = min(others) if others else None
        for impl in dict.fromkeys(r["implementation"] for r in ok if r["implementation"].endswith("Rust")):
            n = next((r["median_ms"] for r in ok if r["implementation"] == impl and r["threads"] == "N"), None)
            one = next((r["median_ms"] for r in ok if r["implementation"] == impl and r["threads"] == "1"), None)
            out.append(
                {
                    "dataset": dataset,
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


def _measure(cls, frames, params: dict, runs: int, budget_s: float) -> tuple[dict, object]:
    times, result = time_call(lambda: cls(**params).add(frames).result(), runs, budget_s)
    if times is None:
        return {"status": f"skipped: over budget ({budget_s:.0f}s)", "median_ms": None, "min_ms": None}, None
    return {"status": "ok", "median_ms": statistics.median(times) * 1e3, "min_ms": min(times) * 1e3}, result


def run(
    package: str,
    datasets: list[Dataset],
    params: dict | None = None,
    impl_params: dict[str, dict] | None = None,
    runs: int = 5,
    budget_s: float = 60.0,
) -> pl.DataFrame | None:
    params, impl_params = params or {}, impl_params or {}
    module = importlib.import_module(package)
    names = [module.REFERENCE] + [n for n in module.IMPLEMENTATIONS if n != module.REFERENCE]
    classes = {n: c for n in names if (c := _load(module, n)) is not None}
    rust = [n for n in classes if n.endswith("Rust")]
    child = os.environ.get(_CHILD_ENV) == "1"

    rows: list[dict] = []
    for ds in datasets:
        frames = ds.make()
        ref_result = None
        for name in rust if child else classes:
            base = {"dataset": ds.name, "implementation": name, "threads": "N" if name in rust else "default"}
            if name in ds.exclude:
                rows.append({**base, "status": "excluded", "median_ms": None, "min_ms": None, "agrees": None})
                continue
            print(f"  {ds.name}: {name} …", file=sys.stderr, flush=True)
            p = {**params, **impl_params.get(name, {})}
            measured, result = _measure(classes[name], frames, p, runs, budget_s)
            agrees = None
            if not child:
                if name == module.REFERENCE:
                    ref_result = result
                elif result is not None and ref_result is not None:
                    agrees = not classes[name](**p).agreement(result, ref_result)
            rows.append({**base, **measured, "agrees": agrees})

    if child:
        print(json.dumps(rows))
        return None
    if rust:
        rows += _single_thread_rows()
    return _report(package, rows)


def _single_thread_rows() -> list[dict]:
    print("  re-running Rust implementations at 1 thread …", file=sys.stderr, flush=True)
    proc = subprocess.run(
        [sys.executable, *sys.argv],
        env={**os.environ, **SINGLE_THREAD_ENV, _CHILD_ENV: "1"},
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    rows = json.loads(proc.stdout.strip().splitlines()[-1])
    return [{**r, "threads": "1", "agrees": None} for r in rows]


def _fmt(x, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def _report(package: str, rows: list[dict]) -> pl.DataFrame:
    technique = package.rsplit(".", 1)[-1]
    print(f"\n{technique} — median / min of timed runs, fresh instance per run, {os.cpu_count()} CPUs")
    print(f"{'dataset':<36} {'implementation':<28} {'threads':>7} {'median ms':>11} {'min ms':>11} {'agrees':>7}  status")
    for r in rows:
        agrees = {True: "yes", False: "NO", None: "—"}[r["agrees"]]
        print(
            f"{r['dataset']:<36} {r['implementation']:<28} {r['threads']:>7} "
            f"{_fmt(r['median_ms'], '11.2f')} {_fmt(r['min_ms'], '11.2f')} {agrees:>7}  {r['status']}"
        )
    print(f"\n{'dataset':<36} {'Rust implementation':<28} {'algorithmic':>12} {'parallel':>9} {'total':>9}")
    for s in speedups(rows):
        print(
            f"{s['dataset']:<36} {s['implementation']:<28} "
            f"{_fmt(s['algorithmic'], '11.1f')}x {_fmt(s['parallel'], '8.1f')}x {_fmt(s['total'], '8.1f')}x"
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        technique=pl.lit(technique),
        cpu_count=pl.lit(os.cpu_count()),
        processor=pl.lit(platform.processor()),
        polars=pl.lit(pl.__version__),
        numpy=pl.lit(np.__version__),
        analytics=pl.lit(analytics.__version__),
        timestamp=pl.lit(stamp),
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{technique}_{stamp}.parquet"
    frame.write_parquet(path)
    print(f"\nsaved {path}")
    return frame
```

- [ ] **Step 4: Ignore the results folder**

Append to `.gitignore`:

```
tests/performance/results/
```

- [ ] **Step 5: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_benchmark_harness.py -q` → PASS.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/performance/harness.py tests/test_benchmark_harness.py .gitignore
git commit -m "test: shared benchmark harness with 1-thread vs N-thread speedup split

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: GCD technique (pilot, per-column scope)

**Files:**
- Create: `services/analytics/analytics/gcd/__init__.py`, `gcd/base.py`, `gcd/rust.py`, `gcd/numpy.py`, `gcd/math.py`
- Rewrite: `tests/test_gcd.py`, `tests/performance/benchmark_gcd.py`

**Interfaces:**
- Consumes: `Technique`, `computed`, `group_by_frame`, `lazy_attributes` (Task 1); `harness.*` and `datagen.integer_multiples`, `datagen.integer_random`, `datagen.mixed_dtypes` (Task 2); `performance.harness.run`, `Dataset`, `large_dataset` (Task 3); `analytics._plugin.column_gcd` (unchanged).
- Produces: `analytics.gcd` exporting `Gcd`, `GcdRust`, `GcdNumpy`, `GcdMath`, with `REFERENCE = "GcdMath"` and `IMPLEMENTATIONS = ("GcdRust", "GcdNumpy", "GcdMath")`. Result columns: `df_a, col_a, status, dtype, gcd, gcd_compressible`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_gcd.py` with:

```python
"""
Whole-column GCD accuracy tests — every implementation in analytics.gcd.

Oracles: known answers (values built as k·g) and GcdMath (math.gcd, arbitrary
precision), the technique's reference. Semantics (ClickHouse GCD-codec method):
GCD of the magnitudes of the raw physical integer values; nulls skipped;
all-null / all-zero / zero-row → 0; non-integer-backed dtypes → status
"ineligible" with gcd null; a magnitude of 2**127 (only i128::MIN) → null.

Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import decimal
import random
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from analytics.gcd import Gcd
from datagen import integer_multiples, mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.gcd"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)

CHUNK = 1 << 16  # gcd.rs parallel chunk size
_DEC_CTX = decimal.Context(prec=80)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def gcds(impl: str, df: pl.DataFrame | pl.LazyFrame) -> dict[str, int | None]:
    out = run(load(impl), {"t": df})
    return dict(zip(out["col_a"].to_list(), out["gcd"].to_list()))


def gcd_of(impl: str, s: pl.Series) -> int | None:
    return gcds(impl, s.to_frame())[s.name]


def math_gcd(s: pl.Series) -> int | None:
    return gcd_of(f"{PKG}:GcdMath", s)


def physical_dtype(dtype: pl.DataType) -> pl.DataType:
    if dtype == pl.Date:
        return pl.Int32()
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.Int64()
    return dtype


def from_physical(name: str, ints: list[int | None], dtype: pl.DataType) -> pl.Series:
    """Series of `dtype` whose physical values are exactly `ints` (None = null)."""
    if isinstance(dtype, pl.Decimal):
        vals = [None if v is None else Decimal(v).scaleb(-dtype.scale, context=_DEC_CTX) for v in ints]
        return pl.Series(name, vals, dtype=dtype)
    return pl.Series(name, ints, dtype=physical_dtype(dtype)).cast(dtype)


UNSIGNED = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def is_signed(dtype: pl.DataType) -> bool:
    return not (isinstance(dtype, UNSIGNED) or dtype == pl.Time)


# (dtype, g, k_lo, k_hi): physical values are k·g, k ∈ [k_lo, k_hi], always incl. k=1.
CASES = [
    pytest.param(pl.Int8(), 4, -32, 31, id="Int8"),
    pytest.param(pl.Int16(), 12, -2_000, 2_000, id="Int16"),
    pytest.param(pl.Int32(), 1_000, -2_000_000, 2_000_000, id="Int32"),
    pytest.param(pl.Int64(), 3_600, -(2**40), 2**40, id="Int64"),
    pytest.param(pl.Int128(), 10**20, -(10**15), 10**15, id="Int128"),
    pytest.param(pl.UInt8(), 5, 0, 51, id="UInt8"),
    pytest.param(pl.UInt16(), 12, 0, 5_000, id="UInt16"),
    pytest.param(pl.UInt32(), 1_000, 0, 4_000_000, id="UInt32"),
    pytest.param(pl.UInt64(), 2**40, 0, 2**23, id="UInt64"),
    pytest.param(pl.Decimal(38, 4), 25, -(10**30), 10**30, id="Decimal38_4"),
    pytest.param(pl.Date(), 7, -5_000, 5_000, id="Date"),
    pytest.param(pl.Datetime("ms"), 60_000, -(10**6), 10**6, id="Datetime_ms"),
    pytest.param(pl.Datetime("us"), 3_600_000_000, -(10**5), 10**5, id="Datetime_us"),
    pytest.param(pl.Datetime("ns", "UTC"), 86_400 * 10**9, -(10**4), 10**4, id="Datetime_ns_UTC"),
    pytest.param(pl.Duration("us"), 250, -(10**9), 10**9, id="Duration_us"),
    pytest.param(pl.Time(), 15 * 60 * 10**9, 0, 95, id="Time"),
]
ALL_DTYPES = [pytest.param(p.values[0], id=p.id) for p in CASES]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract

@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"mixed": mixed_dtypes(200), "ints": integer_multiples(500, 3, 12)}
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement

@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"ints": integer_multiples(5_000, 4, 3_600), "mixed": mixed_dtypes(500)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)

@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype, g, k_lo, k_hi", CASES)
def test_multiples_of_known_gcd(impl, dtype, g, k_lo, k_hi):
    rng = random.Random(1234)
    ints = [g] + [None if rng.random() < 0.1 else rng.randint(k_lo, k_hi) * g for _ in range(1_000)]
    assert gcd_of(impl, from_physical("x", ints, dtype)) == g


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_coprime_values_give_one(impl, dtype):
    assert gcd_of(impl, from_physical("x", [6, 10, 15], dtype)) == 1


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_single_value_is_its_magnitude(impl, dtype):
    v = -42 if is_signed(dtype) else 42
    assert gcd_of(impl, from_physical("x", [v], dtype)) == 42


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_zeros_and_nulls(impl, dtype):
    assert gcd_of(impl, from_physical("x", [0, 0, 12, None, 18, 0], dtype)) == 6
    assert gcd_of(impl, from_physical("x", [0, 0, 0], dtype)) == 0
    assert gcd_of(impl, from_physical("x", [None, None], dtype)) == 0
    assert gcd_of(impl, from_physical("x", [None, 12, None, 18, None], dtype)) == 6


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize(
    "dtype, ints, expected",
    [
        pytest.param(pl.Int8(), [-128], 128, id="i8_min"),
        pytest.param(pl.Int8(), [-128, 64], 64, id="i8_min_and_64"),
        pytest.param(pl.UInt64(), [2**64 - 1], 2**64 - 1, id="u64_max"),
        pytest.param(pl.Int64(), [-(2**63)], 2**63, id="i64_min"),
        pytest.param(pl.Int64(), [-(2**63), 2**62], 2**62, id="i64_min_and_2^62"),
        pytest.param(pl.Int128(), [2**127 - 1], 2**127 - 1, id="i128_max"),
        pytest.param(pl.Int128(), [-(2**127), 2**126], 2**126, id="i128_min_and_2^126"),
        pytest.param(pl.Int128(), [-(2**127)], None, id="i128_min_unrepresentable"),
        pytest.param(pl.Int128(), [-(2**127), None, 0], None, id="i128_min_with_null_zero"),
    ],
)
def test_dtype_extremes(impl, dtype, ints, expected):
    assert gcd_of(impl, from_physical("x", ints, dtype)) == expected


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame(impl):
    schema = {f"c{i}": p.values[0] for i, p in enumerate(CASES)} | {"s": pl.String()}
    assert gcds(impl, pl.DataFrame(schema=schema)) == {**{f"c{i}": 0 for i in range(len(CASES))}, "s": None}


@pytest.mark.parametrize("impl", ALL)
def test_zero_column_frame(impl):
    cls = load(impl)
    out = run(cls, {"t": pl.DataFrame()})
    assert out.height == 0
    assert_contract(cls, out, {"t": pl.DataFrame()})


@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_series(impl):
    parts = [pl.Series("x", [12, 24]), pl.Series("x", [None, 36]), pl.Series("x", [18])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    assert gcd_of(impl, s) == 6


@pytest.mark.parametrize("impl", ALL)
def test_long_series_crosses_parallel_chunks(impl):
    n = 3 * CHUNK + 17
    values = np.full(n, 12, dtype=np.int64)
    values[2 * CHUNK + 5] = 18
    mask = np.arange(n) % 7 == 3  # scattered nulls
    assert not mask[2 * CHUNK + 5]  # the spoiler stays valid
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask))) == 6


@pytest.mark.parametrize("impl", ALL)
def test_null_payloads_ignored(impl):
    # 1s live under null slots; a correct null mask ignores them (otherwise the
    # GCD collapses to 1, and a masked 1 would also trigger the early exit).
    values = np.tile(np.array([12, 1, 18, 1], dtype=np.int64), 50_000)
    arr = pa.array(values, mask=values == 1)
    assert np.frombuffer(arr.buffers()[1], dtype=np.int64)[1] == 1  # payload really is there
    assert gcd_of(impl, pl.from_arrow(arr)) == 6


@pytest.mark.parametrize("impl", ALL)
def test_sliced_series(impl):
    # Leading 7s are sliced away; offsets are not multiples of 8, so the
    # validity bitmap has a sub-byte offset.
    n = 2 * CHUNK + 100
    values = np.full(n, 12, dtype=np.int64)
    values[:5] = 7
    values[CHUNK + 1] = 18
    mask = np.zeros(n, dtype=bool)
    mask[CHUNK + 2 :: 11] = True
    values[mask] = 7  # payloads under nulls
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask)).slice(5, 2 * CHUNK + 50)) == 6
    assert gcd_of(impl, pl.from_arrow(pa.array(values, mask=mask)).slice(3)) == 1  # keeps two leading 7s


@pytest.mark.parametrize("impl", ALL)
def test_physical_unit_results(impl):
    hourly = pl.datetime_range(datetime(2024, 1, 1, 7), datetime(2024, 1, 3), "1h", time_unit="us", eager=True)
    assert gcd_of(impl, hourly) == 3_600_000_000
    quarters = pl.Series("p", [Decimal("1.25"), Decimal("0.50"), Decimal("0.75"), Decimal("-2.00")], dtype=pl.Decimal(10, 2))
    assert gcd_of(impl, quarters) == 25
    weekly = pl.date_range(date(2024, 1, 1), date(2024, 6, 30), "1w", eager=True)
    assert gcd_of(impl, weekly) == math_gcd(weekly)  # raw epoch days, not the 7-day step


_FUZZ_INT_RANGES = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.Int64: (-(2**63), 2**63 - 1),
    pl.Int128: (-(2**127), 2**127 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
    pl.UInt64: (0, 2**64 - 1),
}


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("seed", range(40))
def test_seeded_fuzz_matches_reference(impl, seed):
    rng = random.Random(seed)
    dtype, (lo, hi) = rng.choice(list(_FUZZ_INT_RANGES.items()))
    g = rng.choice([1, 2, 3, 6, 7, 12, 1_000, 2**20, rng.randint(1, hi)])
    k_lo, k_hi = -(-lo // g), hi // g  # ceil(lo/g), floor(hi/g): k·g stays in range
    n = rng.choice([0, 1, 17, 1_000, CHUNK + rng.randint(1, 5_000)])
    null_rate = rng.choice([0.0, 0.05, 0.5, 1.0])
    ints = [None if rng.random() < null_rate else rng.randint(k_lo, k_hi) * g for _ in range(n)]
    cut = rng.randint(0, n)  # optional second Arrow chunk
    s = pl.concat([pl.Series("x", ints[:cut], dtype=dtype), pl.Series("x", ints[cut:], dtype=dtype)], rechunk=False)
    assert gcd_of(impl, s) == math_gcd(s), f"seed={seed} dtype={dtype} g={g} n={n}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions (technique base, once)

def test_conclusions():
    Fixed = with_metrics(Gcd, gcd=[0, 1, 2, None])
    df = pl.DataFrame({"zero": [0], "one": [1], "two": [2], "big": pl.Series([0], dtype=pl.Int128)})
    out = Fixed().add({"t": df}).result()
    assert out["gcd_compressible"].to_list() == [False, False, True, None]


# ─────────────────────────────────────────────────────────────────────────────
# Output contract details

@pytest.mark.parametrize("impl", ALL)
def test_ineligible_dtypes_are_reported_with_their_dtype(impl):
    cols = {
        "f64": pl.Series([2.0, 4.0], dtype=pl.Float64),
        "str": ["a", "b"],
        "bool": [True, False],
        "cat": pl.Series(["x", "y"], dtype=pl.Categorical),
        "enum": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
        "list": [[2, 4], [6]],
        "arr": pl.Series([[2, 4], [6, 8]], dtype=pl.Array(pl.Int64, 2)),
        "struct": [{"a": 2}, {"a": 4}],
        "bin": [b"\x02", b"\x04"],
        "null": pl.Series([None, None], dtype=pl.Null),
    }
    if hasattr(pl, "UInt128"):  # the plugin's Rust polars cannot receive UInt128
        cols["u128"] = pl.Series([12, 18], dtype=pl.UInt128)
    df = pl.DataFrame(cols)
    out = run(load(impl), {"t": df})
    assert out["status"].to_list() == ["ineligible"] * df.width
    assert out["gcd"].null_count() == df.width
    assert out["gcd_compressible"].null_count() == df.width
    assert out["dtype"].to_list() == [str(dt) for dt in df.dtypes]


@pytest.mark.parametrize("impl", ALL)
def test_order_lazyframes_and_multiple_frames(impl):
    df = pl.DataFrame({"z": [4, 8], "a": ["x", "y"], "m": [9, 6]})
    out = run(load(impl), {"first": df.lazy(), "second": df})
    assert out.select("df_a", "col_a").rows() == [
        ("first", "z"), ("first", "a"), ("first", "m"), ("second", "z"), ("second", "a"), ("second", "m")
    ]
    assert out["gcd"].to_list() == [4, None, 3, 4, None, 3]
    assert out["dtype"].to_list() == ["Int64", "String", "Int64"] * 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_gcd.py -q`
Expected: collection ERROR `ModuleNotFoundError: No module named 'analytics.gcd'`.

- [ ] **Step 3: Write the technique base**

Create `services/analytics/analytics/gcd/base.py`:

```python
"""Whole-column GCD — the quantity ClickHouse's GCD codec divides by."""

import polars as pl

from analytics.base import Technique, computed

I128_LIMIT = 2**127
INTEGER_BACKED = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Decimal, pl.Date, pl.Datetime, pl.Duration, pl.Time,
)


class Gcd(Technique):
    """GCD of the magnitudes of each column's raw physical integer values.

    Results are in physical units: Decimal → unscaled integer, Date → days,
    Datetime/Duration → their time unit, Time → ns. Nulls are skipped; all-null,
    all-zero and zero-row columns → 0; a magnitude of 2**127 (only i128::MIN
    values) is not representable as Int128 → null. Other dtypes — including
    Categorical/Enum and UInt128, which the plugin's Rust polars cannot receive —
    are ineligible. `dtype` (Python's str(dtype)) is reported on every row.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = {"gcd": pl.Int128}
    CONCLUSIONS = {"gcd_compressible": pl.Boolean}

    def eligible(self, series: pl.Series) -> bool:
        return isinstance(series.dtype, INTEGER_BACKED)

    def describe(self, frames, combos):
        return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(gcd_compressible=computed(pl.col("gcd") > 1))
```

- [ ] **Step 4: Write the three implementations and the package `__init__`**

Create `services/analytics/analytics/gcd/rust.py`:

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.gcd.base import Gcd


class GcdRust(Gcd):
    """Rust plugin `column_gcd`: rayon-parallel across columns and 64K-value chunks;
    binary GCD after one hardware remainder per value; early exit once the GCD is 1."""

    def _compute(self, frames, combos):
        gcd = {}
        for frame, group in group_by_frame(combos).items():
            columns = [c for ((_, c),) in group]
            out = _plugin.column_gcd(frames[frame].select(columns)).unnest("column_gcd")
            gcd.update(((frame, c), g) for c, g in zip(out["column"].to_list(), out["gcd"].to_list()))
        return self.metrics_frame(combos, {"gcd": [gcd[k[0]] for k in combos]})
```

Create `services/analytics/analytics/gcd/math.py`:

```python
import math

import polars as pl

from analytics.gcd.base import I128_LIMIT, Gcd


def math_gcd(series: pl.Series) -> int | None:
    g = math.gcd(*series.to_physical().drop_nulls().to_list())
    return None if g >= I128_LIMIT else g


class GcdMath(Gcd):
    """Reference: math.gcd over each column's physical values (arbitrary precision,
    so Int128, wide Decimal and MIN magnitudes are exact). Single core."""

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, {"gcd": [math_gcd(frames[n][c]) for ((n, c),) in combos]})
```

Create `services/analytics/analytics/gcd/numpy.py`:

```python
import numpy as np
import polars as pl

from analytics.gcd.base import I128_LIMIT, Gcd

_NATIVE = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
_SIGNED_MIN = {pl.Int8: -(2**7), pl.Int16: -(2**15), pl.Int32: -(2**31), pl.Int64: -(2**63)}


def numpy_gcd(series: pl.Series) -> int | None:
    phys = series.to_physical().drop_nulls()
    if phys.len() == 0:
        return 0
    base = phys.dtype.base_type()
    if base in _NATIVE and phys.min() != _SIGNED_MIN.get(base):
        values = phys.to_numpy()
    else:
        # 128-bit physical values and |MIN| of a signed type do not fit numpy's
        # native ints: fall back to an object array of Python ints.
        values = np.array(phys.to_list(), dtype=object)
    # reduce() of one element returns it unchanged (possibly negative): take abs.
    g = abs(int(np.gcd.reduce(values)))
    return None if g >= I128_LIMIT else g


class GcdNumpy(Gcd):
    """numpy.gcd.reduce per column (vectorised, single core); object arrays for
    values outside numpy's native integer range."""

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, {"gcd": [numpy_gcd(frames[n][c]) for ((n, c),) in combos]})
```

Create `services/analytics/analytics/gcd/__init__.py`:

```python
"""Whole-column GCD (per-column scope). See Gcd for semantics."""

from analytics.base import lazy_attributes
from analytics.gcd.base import Gcd
from analytics.gcd.rust import GcdRust

REFERENCE = "GcdMath"
IMPLEMENTATIONS = ("GcdRust", "GcdNumpy", "GcdMath")

__getattr__ = lazy_attributes(__name__, {"GcdNumpy": ".numpy", "GcdMath": ".math"})
__all__ = ["Gcd", "GcdRust", "GcdNumpy", "GcdMath", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 5: Run the GCD tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_gcd.py -q`
Expected: all PASS. If `test_dtype_extremes[i8_min-GcdNumpy]` fails, check the `_SIGNED_MIN` lookup. The object-array fallback must be taken whenever a signed column contains its type's MIN.

- [ ] **Step 6: Rewrite the benchmark**

Replace the entire contents of `tests/performance/benchmark_gcd.py` with:

```python
"""
Whole-column GCD speed benchmark: GcdRust vs GcdNumpy vs GcdMath (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_gcd.py

Shapes: large_dataset.arrow (realistic mix); narrow/long (parallelism inside a
column); wide (parallelism across columns); early exit (random values, GCD 1).
GcdMath is excluded from the 10^7–10^8-value shapes: to_list() alone would
exhaust memory.
"""

from harness import Dataset, large_dataset, run

from datagen import integer_multiples, integer_random

G = 3_600

if __name__ == "__main__":
    run(
        "analytics.gcd",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", lambda: {"t": integer_multiples(10_000_000, 4, G)}, exclude=("GcdMath",)),
            Dataset("wide 1M x 100", lambda: {"t": integer_multiples(1_000_000, 100, G)}, exclude=("GcdMath",)),
            Dataset("early exit 10M x 4", lambda: {"t": integer_random(10_000_000, 4)}, exclude=("GcdMath",)),
        ],
    )
```

- [ ] **Step 7: Smoke-run the benchmark**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_gcd.py`
Expected output:
- a table with `GcdMath`, `GcdRust` (threads `N` and `1`) and `GcdNumpy` rows;
- `agrees` shows `yes` wherever the reference ran;
- a speedup table with an algorithmic, parallel and total figure for each dataset;
- a `saved tests/performance/results/gcd_<stamp>.parquet` line.

This takes a few minutes. Paste the speedup table into the task report.

- [ ] **Step 8: Full suite and commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics/gcd tests/test_gcd.py tests/performance/benchmark_gcd.py
git commit -m "feat: GCD technique classes (GcdRust/GcdNumpy/GcdMath) on the uniform contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Chi-squared technique (ordered scope)

**Files:**
- Create: `services/analytics/analytics/chi_squared/{__init__,base,rust,scipy,polars_ds}.py`
- Rewrite: `tests/test_chi_squared.py`, `tests/performance/benchmark_chi_squared.py`
- Delete: `services/analytics/chi_squared_polarsds.py` (its logic moves to `chi_squared/polars_ds.py`)

**Interfaces:**
- Consumes: `Technique`, `computed`, `at_least`, `check_unit`, `group_by_frame`, `lazy_attributes`, `INTEGERS_64`, `STRING_LIKE`; `_plugin.pairwise_chi_squared(df, pairs)`, which returns the struct column `pairwise_chi_squared` with fields `col_a, col_b, chi2_stat, p_value, cramers_v, low_expected_count, n_valid`.
- Produces: `analytics.chi_squared` exporting `ChiSquared`, `ChiSquaredRust`, `ChiSquaredScipy`, `ChiSquaredPolarsDS`, with `REFERENCE = "ChiSquaredScipy"` and `IMPLEMENTATIONS = ("ChiSquaredRust", "ChiSquaredScipy", "ChiSquaredPolarsDS")`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_chi_squared.py` with:

```python
"""
Pairwise chi-squared accuracy tests — every implementation in analytics.chi_squared.

Reference: ChiSquaredScipy (scipy.stats.chi2_contingency, correction=False, after
dropping rows where either column is null). Eligible columns in mixed_dtypes:
boolean_*, uint32_*, categorical_* (9) → C(9,2) = 36 computed pairs; float64_*,
list_*, arr_* are ineligible.
"""

import math

import polars as pl
import pytest

from analytics.chi_squared import ChiSquared
from datagen import mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.chi_squared"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


# 1. Contract
@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 36


# 2. Reference agreement
@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# 3. Known answers
@pytest.mark.parametrize("impl", ALL)
def test_known_2x2_table(impl):
    # [[20, 30], [30, 20]], N=100: every expected count is 25 → χ² = 4 × (5²/25) = 4.
    a = ["x"] * 50 + ["y"] * 50
    b = ["p"] * 20 + ["q"] * 30 + ["p"] * 30 + ["q"] * 20
    row = run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)
    assert row["chi2_stat"] == pytest.approx(4.0)
    assert row["p_value"] == pytest.approx(0.04550026389635842)
    assert row["cramers_v"] == pytest.approx(0.2)
    assert row["low_expected_count"] is False
    assert row["n_valid"] == 100


@pytest.mark.parametrize("impl", ALL)
def test_nulls_dropped_and_constant_column_undefined(impl):
    df = pl.DataFrame({"a": ["x", "y", "x", "y", None], "b": ["p", "p", "q", "q", "p"], "c": ["k"] * 5})
    rows = {(r["col_a"], r["col_b"]): r for r in run(load(impl), {"t": df}).iter_rows(named=True)}
    assert rows[("a", "b")]["n_valid"] == 4
    assert rows[("a", "b")]["low_expected_count"] is True
    for pair in [("a", "c"), ("b", "c")]:
        r = rows[pair]
        assert math.isnan(r["chi2_stat"]) and math.isnan(r["p_value"]) and math.isnan(r["cramers_v"])
        assert r["low_expected_count"] is False
        assert r["associated"] is False
    assert rows[("a", "c")]["n_valid"] == 4


# 4. Conclusions
def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        ChiSquared,
        chi2_stat=[1.0] * 3,
        p_value=[0.5] * 3,
        cramers_v=[0.29, 0.3, NAN],
        low_expected_count=[False] * 3,
        n_valid=[10] * 3,
    )
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    assert Fixed().add({"t": df}).result()["associated"].to_list() == [False, True, False]
    assert Fixed(cramers_v_threshold=0.2).add({"t": df}).result()["associated"].to_list() == [True, True, False]


def test_threshold_validation():
    Fixed = with_metrics(ChiSquared)
    with pytest.raises(ValueError):
        Fixed(cramers_v_threshold=1.5)
    with pytest.raises(ValueError):
        Fixed(max_unique=1)


# Eligibility
@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.String), "b": pl.Series([], dtype=pl.Int64)})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]


@pytest.mark.parametrize("impl", ALL)
def test_cardinality_cap_and_dtype_eligibility(impl):
    df = pl.DataFrame({"a": [1, 2, 3, 1], "b": [1, 1, 2, 2], "f": [1.0, 2.0, 3.0, 4.0]})
    out = run(load(impl), {"t": df}, max_unique=2)
    assert out.select("col_a", "col_b", "status").rows() == [
        ("a", "b", "ineligible"),  # a has 3 distinct values > max_unique
        ("a", "f", "ineligible"),
        ("b", "f", "ineligible"),  # Float64 is not categorical
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_chi_squared.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.chi_squared'`.

- [ ] **Step 3: Write the technique base**

Create `services/analytics/analytics/chi_squared/base.py`:

```python
"""Chi-squared independence test between pairs of categorical columns."""

import polars as pl

from analytics._dtypes import INTEGERS_64, STRING_LIKE
from analytics.base import Technique, at_least, check_unit, computed

CATEGORICAL = (pl.Boolean, *STRING_LIKE, *INTEGERS_64, pl.Int128)


class ChiSquared(Technique):
    """χ² test of independence per column pair within a frame.

    Null policy: rows where either column is null are dropped (n_valid counts the
    rest). Constant columns and empty overlaps → NaN statistics. At realistic N,
    p-values are vanishingly small for almost any pair, so Cramér's V (effect size)
    drives the `associated` verdict. low_expected_count: rarest row marginal ×
    rarest column marginal / N < 5.

    Eligible: Boolean, String, Categorical, Enum and integer columns with at least
    one row and at most `max_unique` distinct values (None disables the cap).
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {
        "chi2_stat": pl.Float64,
        "p_value": pl.Float64,
        "cramers_v": pl.Float64,
        "low_expected_count": pl.Boolean,
        "n_valid": pl.UInt32,
    }
    CONCLUSIONS = {"associated": pl.Boolean}
    RTOL = 1e-4
    ATOL = 1e-12

    def __init__(self, *, cramers_v_threshold: float = 0.3, max_unique: int | None = 1000):
        super().__init__()
        check_unit("cramers_v_threshold", cramers_v_threshold)
        if max_unique is not None and max_unique < 2:
            raise ValueError(f"max_unique must be >= 2 or None, got {max_unique}")
        self.cramers_v_threshold = cramers_v_threshold
        self.max_unique = max_unique

    def eligible(self, series: pl.Series) -> bool:
        return (
            series.len() > 0
            and isinstance(series.dtype, CATEGORICAL)
            and (self.max_unique is None or series.n_unique() <= self.max_unique)
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(associated=computed(at_least("cramers_v", self.cramers_v_threshold)))
```

- [ ] **Step 4: Write the implementations and `__init__`**

Create `services/analytics/analytics/chi_squared/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.chi_squared.base import ChiSquared


class ChiSquaredRust(ChiSquared):
    """Rust plugin `pairwise_chi_squared`: dense-id contingency tables, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            pairs = [(a, b) for (_, a), (_, b) in group]
            df = frames[frame].select(list(dict.fromkeys(c for p in pairs for c in p)))
            out = _plugin.pairwise_chi_squared(df, pairs).unnest("pairwise_chi_squared")
            parts.append(self.rows_from_plugin(frame, out))
        return pl.concat(parts)
```

Create `services/analytics/analytics/chi_squared/scipy.py`:

```python
import math

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

from analytics.chi_squared.base import ChiSquared


def _table(pair: pl.DataFrame) -> np.ndarray:
    """Dense contingency table (rows: values of column 0, cols: values of column 1)."""
    a, b = pair.columns
    counts = pair.group_by(a, b).len().with_columns(
        (pl.col(a).rank("dense").cast(pl.Int64) - 1),
        (pl.col(b).rank("dense").cast(pl.Int64) - 1),
    )
    table = np.zeros((counts[a].max() + 1, counts[b].max() + 1))
    table[counts[a].to_numpy(), counts[b].to_numpy()] = counts["len"].to_numpy()
    return table


def chi_squared_pair(df: pl.DataFrame, a: str, b: str) -> dict:
    pair = df.select(pl.col(a).cast(pl.String), pl.col(b).cast(pl.String)).drop_nulls()
    n = pair.height
    undefined = {"chi2_stat": math.nan, "p_value": math.nan, "cramers_v": math.nan, "low_expected_count": False, "n_valid": n}
    if n == 0:
        return undefined
    table = _table(pair)
    rows, cols = table.shape
    if rows < 2 or cols < 2:
        return undefined
    res = chi2_contingency(table, correction=False)
    stat = float(res.statistic)
    return {
        "chi2_stat": stat,
        "p_value": float(res.pvalue),
        "cramers_v": math.sqrt(stat / (n * (min(rows, cols) - 1))),
        "low_expected_count": bool(table.sum(axis=1).min() * table.sum(axis=0).min() / n < 5),
        "n_valid": n,
    }


class ChiSquaredScipy(ChiSquared):
    """Reference: scipy.stats.chi2_contingency (correction=False) on a Polars-built table, one pair at a time."""

    def _compute(self, frames, combos):
        rows = [chi_squared_pair(frames[f], a, b) for (f, a), (_, b) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})
```

Create `services/analytics/analytics/chi_squared/polars_ds.py`:

```python
import math

import polars as pl
import polars_ds as pds

from analytics.chi_squared.base import ChiSquared


def chi_squared_pair(df: pl.DataFrame, a: str, b: str) -> dict:
    pair = df.select(a, b).drop_nulls()
    n = pair.height
    undefined = {"chi2_stat": math.nan, "p_value": math.nan, "cramers_v": math.nan, "low_expected_count": False, "n_valid": n}
    if n == 0:
        return undefined
    unique_a, unique_b = pair[a].n_unique(), pair[b].n_unique()
    if unique_a < 2 or unique_b < 2:
        return undefined
    result = pair.select(pds.chi2(a, b).alias("r")).unnest("r")
    if "statistic" not in result.columns or "pvalue" not in result.columns:
        raise RuntimeError(f"polars-ds chi2 returned unexpected struct fields: {result.columns}")
    stat = float(result["statistic"][0])
    low = pair[a].value_counts()["count"].min() * pair[b].value_counts()["count"].min() / n < 5
    return {
        "chi2_stat": stat,
        "p_value": float(result["pvalue"][0]),
        "cramers_v": math.sqrt(stat / (n * (min(unique_a, unique_b) - 1))),
        "low_expected_count": bool(low),
        "n_valid": n,
    }


class ChiSquaredPolarsDS(ChiSquared):
    """polars-ds `chi2` expression, one pair at a time (the original pure-Python baseline)."""

    def _compute(self, frames, combos):
        rows = [chi_squared_pair(frames[f], a, b) for (f, a), (_, b) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})
```

Create `services/analytics/analytics/chi_squared/__init__.py`:

```python
"""Chi-squared independence test (ordered scope). See ChiSquared for semantics."""

from analytics.base import lazy_attributes
from analytics.chi_squared.base import ChiSquared
from analytics.chi_squared.rust import ChiSquaredRust

REFERENCE = "ChiSquaredScipy"
IMPLEMENTATIONS = ("ChiSquaredRust", "ChiSquaredScipy", "ChiSquaredPolarsDS")

__getattr__ = lazy_attributes(__name__, {"ChiSquaredScipy": ".scipy", "ChiSquaredPolarsDS": ".polars_ds"})
__all__ = ["ChiSquared", "ChiSquaredRust", "ChiSquaredScipy", "ChiSquaredPolarsDS", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 5: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_chi_squared.py -q`
Expected: all PASS. If `test_agrees_with_reference[ChiSquaredPolarsDS]` fails **only** on `p_value`, that is a real difference in how polars-ds computes p-values. Report it with the mismatch lines, and don't loosen `RTOL` without asking. χ² and Cramér's V have always agreed to 1e-4.

- [ ] **Step 6: Rewrite the benchmark and delete the old baseline module**

Replace the entire contents of `tests/performance/benchmark_chi_squared.py` with:

```python
"""
Chi-squared speed benchmark: ChiSquaredRust vs ChiSquaredScipy (reference) vs ChiSquaredPolarsDS.

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_chi_squared.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.chi_squared",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        ],
    )
```

```bash
git rm services/analytics/chi_squared_polarsds.py
```

- [ ] **Step 7: Smoke-run the benchmark, then run the full suite**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_chi_squared.py`. It should complete, with `agrees` = `yes` for `ChiSquaredRust`. Paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

- [ ] **Step 8: Commit**

```bash
git add services/analytics/analytics/chi_squared tests/test_chi_squared.py tests/performance/benchmark_chi_squared.py
git commit -m "feat: chi-squared technique classes (Rust/scipy/polars-ds) on the uniform contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Adjusted Rand Index technique (ordered scope)

**Files:**
- Create: `services/analytics/analytics/adjusted_rand/{__init__,base,rust,sklearn}.py`
- Rewrite: `tests/test_adjusted_rand.py`, `tests/performance/benchmark_adjusted_rand.py`

**Interfaces:**
- Consumes: `Technique`, `computed`, `at_least`, `check_unit`, `group_by_frame`, `lazy_attributes`, `encodable`, `is_nested`; `_plugin.pairwise_adjusted_rand(df, pairs)`, which returns the struct column `pairwise_adjusted_rand` with fields `col_a, col_b, ari, n_valid`.
- Produces: `analytics.adjusted_rand` exporting `AdjustedRand`, `AdjustedRandRust`, `AdjustedRandSklearn`, with `REFERENCE = "AdjustedRandSklearn"` and `IMPLEMENTATIONS = ("AdjustedRandRust", "AdjustedRandSklearn")`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_adjusted_rand.py` with:

```python
"""
Pairwise Adjusted Rand Index accuracy tests — every implementation in analytics.adjusted_rand.

Reference: AdjustedRandSklearn (sklearn.metrics.adjusted_rand_score on string
labels after dropping rows where either column is null). Eligible columns in
mixed_dtypes: everything except list_* / arr_* (12) → C(12,2) = 66 computed pairs.
"""

import math

import polars as pl
import pytest

from analytics.adjusted_rand import AdjustedRand
from datagen import mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.adjusted_rand"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 66


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def _ari(impl, a, b):
    return run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_known_answers(impl):
    assert _ari(impl, [0, 0, 1, 1], [5, 5, 7, 7])["ari"] == pytest.approx(1.0)  # identical partitions, relabelled
    assert _ari(impl, [0, 0, 1, 1], [0, 1, 0, 1])["ari"] == pytest.approx(-0.5)  # maximally crossed
    assert _ari(impl, [1, 1, 1, 1], [2, 2, 2, 2])["ari"] == pytest.approx(1.0)  # both constant (sklearn convention)


@pytest.mark.parametrize("impl", ALL)
def test_no_overlap_is_nan(impl):
    row = _ari(impl, [1, 2, None, None], [None, None, 1, 2])
    assert math.isnan(row["ari"]) and row["n_valid"] == 0 and row["same_partition"] is False


@pytest.mark.parametrize("impl", ALL)
def test_negative_zero_and_zero_are_one_label(impl):
    row = _ari(impl, [-0.0, 0.0, 1.0, 1.0], [3, 3, 4, 4])
    assert row["ari"] == pytest.approx(1.0)


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(AdjustedRand, ari=[0.89, 0.9, float("nan")], n_valid=[4] * 3)
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    assert Fixed().add({"t": df}).result()["same_partition"].to_list() == [False, True, False]
    assert Fixed(ari_threshold=0.5).add({"t": df}).result()["same_partition"].to_list() == [True, True, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Int64)})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]


@pytest.mark.parametrize("impl", ALL)
def test_nested_columns_are_ineligible(impl):
    df = pl.DataFrame({"a": [1, 2], "l": [[1], [2]]})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_adjusted_rand.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.adjusted_rand'`.

- [ ] **Step 3: Write the base, the implementations and `__init__`**

Create `services/analytics/analytics/adjusted_rand/base.py`:

```python
"""Adjusted Rand Index between the row partitions induced by two columns."""

import polars as pl

from analytics._dtypes import encodable, is_nested
from analytics.base import Technique, at_least, check_unit, computed


class AdjustedRand(Technique):
    """Chance-corrected agreement between two partitions (rows sharing a value form
    a cluster). 1.0 = identical partitions, ≈0 = chance, floor -0.5.

    Null policy: rows where either column is null are dropped (n_valid counts the
    rest). No overlapping rows → NaN; degenerate denominator (e.g. both columns
    constant) → 1.0, matching sklearn. Eligible: any non-nested column the Rust
    encoder accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {"ari": pl.Float64, "n_valid": pl.UInt32}
    CONCLUSIONS = {"same_partition": pl.Boolean}
    RTOL = 1e-9
    ATOL = 1e-12

    def __init__(self, *, ari_threshold: float = 0.9):
        super().__init__()
        check_unit("ari_threshold", ari_threshold)
        self.ari_threshold = ari_threshold

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype) and not is_nested(series.dtype)

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(same_partition=computed(at_least("ari", self.ari_threshold)))
```

Create `services/analytics/analytics/adjusted_rand/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.adjusted_rand.base import AdjustedRand
from analytics.base import group_by_frame


class AdjustedRandRust(AdjustedRand):
    """Rust plugin `pairwise_adjusted_rand`: shared dense contingency builder, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            pairs = [(a, b) for (_, a), (_, b) in group]
            df = frames[frame].select(list(dict.fromkeys(c for p in pairs for c in p)))
            out = _plugin.pairwise_adjusted_rand(df, pairs).unnest("pairwise_adjusted_rand")
            parts.append(self.rows_from_plugin(frame, out))
        return pl.concat(parts)
```

Create `services/analytics/analytics/adjusted_rand/sklearn.py`:

```python
import math

import polars as pl
from sklearn.metrics import adjusted_rand_score

from analytics.adjusted_rand.base import AdjustedRand


def _labels(series: pl.Series) -> list[str]:
    """String labels; -0.0 and 0.0 are one label, as in the Rust encoder."""
    if series.dtype.is_float():
        series = series.to_frame().select(pl.when(pl.first() == 0).then(0.0).otherwise(pl.first())).to_series()
    return series.cast(pl.String).to_list()


class AdjustedRandSklearn(AdjustedRand):
    """Reference: sklearn.metrics.adjusted_rand_score, one pair at a time."""

    def _compute(self, frames, combos):
        ari, n_valid = [], []
        for (f, a), (_, b) in combos:
            pair = frames[f].select(a, b).drop_nulls()
            n_valid.append(pair.height)
            ari.append(adjusted_rand_score(_labels(pair[a]), _labels(pair[b])) if pair.height else math.nan)
        return self.metrics_frame(combos, {"ari": ari, "n_valid": n_valid})
```

Create `services/analytics/analytics/adjusted_rand/__init__.py`:

```python
"""Adjusted Rand Index (ordered scope). See AdjustedRand for semantics."""

from analytics.adjusted_rand.base import AdjustedRand
from analytics.adjusted_rand.rust import AdjustedRandRust
from analytics.base import lazy_attributes

REFERENCE = "AdjustedRandSklearn"
IMPLEMENTATIONS = ("AdjustedRandRust", "AdjustedRandSklearn")

__getattr__ = lazy_attributes(__name__, {"AdjustedRandSklearn": ".sklearn"})
__all__ = ["AdjustedRand", "AdjustedRandRust", "AdjustedRandSklearn", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 4: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_adjusted_rand.py -q` → all PASS.

- [ ] **Step 5: Rewrite the benchmark**

Replace the entire contents of `tests/performance/benchmark_adjusted_rand.py` with:

```python
"""
Adjusted Rand Index speed benchmark: AdjustedRandRust vs AdjustedRandSklearn (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_adjusted_rand.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.adjusted_rand",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        ],
    )
```

- [ ] **Step 6: Smoke-run the benchmark, run the full suite, commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_adjusted_rand.py`. It should complete. `AdjustedRandSklearn` may show `skipped: over budget` on `large_dataset.arrow`, which is acceptable. Paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics/adjusted_rand tests/test_adjusted_rand.py tests/performance/benchmark_adjusted_rand.py
git commit -m "feat: Adjusted Rand technique classes (Rust/sklearn) on the uniform contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Pairwise entropy technique (ordered scope)

**Files:**
- Create: `services/analytics/analytics/pairwise_entropy/{__init__,base,rust,polars}.py`
- Create: `tests/test_pairwise_entropy.py`, `tests/performance/benchmark_pairwise_entropy.py`
- Modify: `tests/test_entropy.py`, where you delete the pairwise and marginal tests and keep only the threeway test until Task 8.

**Interfaces:**
- Consumes: `Technique`, `computed`, `at_least`, `check_unit`, `group_by_frame`, `lazy_attributes`, `encodable`, `is_nested`. From the plugin:
  - `_plugin.marginal_entropy(df)` returns the struct `marginal_entropy` with fields `col_name, entropy`.
  - `_plugin.pairwise_joint_entropy(df, pairs)` returns the struct `pairwise_entropy` with fields `col_a, col_b, entropy`.
- Produces:
  - `analytics.pairwise_entropy` exporting `PairwiseEntropy`, `PairwiseEntropyRust`, `PairwiseEntropyPolars`, with `REFERENCE = "PairwiseEntropyPolars"` and `IMPLEMENTATIONS = ("PairwiseEntropyRust", "PairwiseEntropyPolars")`.
  - `analytics.pairwise_entropy.polars.entropy_bits(df, columns) -> float`, reused in Task 8.
  - `analytics.pairwise_entropy.base.near_unique(h_column, margin) -> pl.Expr`, reused in Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pairwise_entropy.py`:

```python
"""
Pairwise joint entropy / mutual information accuracy tests — every implementation
in analytics.pairwise_entropy.

Reference: PairwiseEntropyPolars (value_counts → entropy(base=2)); null is its own
category. mixed_dtypes: all 18 columns eligible → C(18,2) = 153 pairs.
"""

import polars as pl
import pytest

from analytics.pairwise_entropy import PairwiseEntropy
from datagen import mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.pairwise_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert (out["status"] == "computed").sum() == 153


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def _row(impl, a, b):
    return run(load(impl), {"t": pl.DataFrame({"a": a, "b": b})}).row(0, named=True)


@pytest.mark.parametrize("impl", ALL)
def test_independent_columns(impl):
    r = _row(impl, [0, 0, 1, 1], [0, 1, 0, 1])
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 2.0))
    assert r["mi"] == pytest.approx(0.0, abs=1e-12) and r["nmi"] == pytest.approx(0.0, abs=1e-12)
    assert r["redundant"] is False and r["near_unique"] is True  # 4 rows, 4 distinct pairs
    assert r["n_rows"] == 4


@pytest.mark.parametrize("impl", ALL)
def test_redundant_columns_and_null_category(impl):
    r = _row(impl, [None, None, 1, 1], ["x", "x", "y", "y"])  # null is a category: a determines b
    assert (r["h_a"], r["h_b"], r["h_ab"]) == pytest.approx((1.0, 1.0, 1.0))
    assert r["nmi"] == pytest.approx(1.0) and r["redundant"] is True and r["near_unique"] is False


@pytest.mark.parametrize("impl", ALL)
def test_constant_column_nmi_is_nan(impl):
    r = _row(impl, [1, 1, 1, 1], [0, 1, 0, 1])
    assert r["h_a"] == pytest.approx(0.0, abs=1e-12)
    assert r["nmi"] != r["nmi"] and r["redundant"] is False  # NaN


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        PairwiseEntropy,
        h_a=[1.0] * 3,
        h_b=[1.0] * 3,
        h_ab=[1.95, 1.5, 1.0],
        mi=[0.05, 0.5, 1.0],
        nmi=[0.95, 0.9, NAN],
        n_rows=[4, 4, 1],
    )
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = Fixed().add({"t": df}).result()
    assert out["redundant"].to_list() == [True, True, False]
    assert out["near_unique"].to_list() == [True, False, False]  # log2(4) - 0.1 = 1.9; n_rows 1 → False
    assert Fixed(nmi_threshold=0.92).add({"t": df}).result()["redundant"].to_list() == [True, False, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.String)})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_pairwise_entropy.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.pairwise_entropy'`.

- [ ] **Step 3: Write the base, the implementations and `__init__`**

Create `services/analytics/analytics/pairwise_entropy/base.py`:

```python
"""Pairwise joint entropy and (normalised) mutual information."""

import math

import polars as pl

from analytics._dtypes import encodable
from analytics.base import Technique, at_least, check_unit, computed


def near_unique(h_column: str, margin: float) -> pl.Expr:
    """Joint entropy within `margin` bits of log2(n_rows): combinations are (almost) all distinct."""
    return (pl.col("n_rows") > 1) & (pl.col(h_column) >= pl.col("n_rows").cast(pl.Float64).log(2) - margin)


class PairwiseEntropy(Technique):
    """H(A), H(B), H(A,B) in bits; MI = H(A) + H(B) − H(A,B); NMI = MI / min(H(A), H(B))
    (NaN when either marginal entropy is 0). NMI ≈ 1 ⇒ one column predicts the other.

    Null policy: null is its own category. Eligible: any column the Rust encoder
    accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 2
    METRICS = {
        "h_a": pl.Float64,
        "h_b": pl.Float64,
        "h_ab": pl.Float64,
        "mi": pl.Float64,
        "nmi": pl.Float64,
        "n_rows": pl.UInt32,
    }
    CONCLUSIONS = {"redundant": pl.Boolean, "near_unique": pl.Boolean}
    RTOL = 1e-5
    ATOL = 1e-12

    def __init__(self, *, nmi_threshold: float = 0.9, near_unique_margin: float = 0.1):
        super().__init__()
        check_unit("nmi_threshold", nmi_threshold)
        if near_unique_margin < 0:
            raise ValueError(f"near_unique_margin must be >= 0, got {near_unique_margin}")
        self.nmi_threshold = nmi_threshold
        self.near_unique_margin = near_unique_margin

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)

    def entropy_rows(self, combos, h_a, h_b, h_ab, n_rows) -> pl.DataFrame:
        """Metrics frame from marginal and joint entropies (MI/NMI derived here, once)."""
        mi = [a + b - ab for a, b, ab in zip(h_a, h_b, h_ab)]
        nmi = [m / min(a, b) if min(a, b) > 0 else math.nan for m, a, b in zip(mi, h_a, h_b)]
        return self.metrics_frame(
            combos, {"h_a": h_a, "h_b": h_b, "h_ab": h_ab, "mi": mi, "nmi": nmi, "n_rows": n_rows}
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            redundant=computed(at_least("nmi", self.nmi_threshold)),
            near_unique=computed(near_unique("h_ab", self.near_unique_margin)),
        )
```

Create `services/analytics/analytics/pairwise_entropy/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.pairwise_entropy.base import PairwiseEntropy


class PairwiseEntropyRust(PairwiseEntropy):
    """Rust plugins `marginal_entropy` + `pairwise_joint_entropy`: dense-id encoding,
    flat-array counting, rayon-parallel across pairs."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(list(dict.fromkeys(c for k in group for _, c in k)))
            marginal = dict(_plugin.marginal_entropy(df).unnest("marginal_entropy").iter_rows())
            joint = _plugin.pairwise_joint_entropy(df, [(a, b) for (_, a), (_, b) in group])
            h_ab = {frozenset((a, b)): h for a, b, h in joint.unnest("pairwise_entropy").iter_rows()}
            parts.append(
                self.entropy_rows(
                    group,
                    [marginal[a] for (_, a), _ in group],
                    [marginal[b] for _, (_, b) in group],
                    [h_ab[frozenset((a, b))] for (_, a), (_, b) in group],
                    [df.height] * len(group),
                )
            )
        return pl.concat(parts)
```

Create `services/analytics/analytics/pairwise_entropy/polars.py`:

```python
import polars as pl

from analytics._dtypes import is_nested
from analytics.pairwise_entropy.base import PairwiseEntropy


def entropy_bits(df: pl.DataFrame, columns: list[str]) -> float:
    """H(columns) in bits via Polars value_counts; null is its own category.

    Falls back to casting List/Array columns to String if Polars cannot group a
    struct containing nested fields (behaviour is version-dependent).
    """

    def h(frame: pl.DataFrame) -> float:
        key = frame.select(pl.struct(columns).alias("k")).to_series()
        return key.value_counts().get_column("count").entropy(base=2, normalize=True)

    try:
        return h(df)
    except Exception:
        nested = [c for c in columns if is_nested(df.schema[c])]
        if not nested:
            raise
        return h(df.with_columns(pl.col(c).cast(pl.String) for c in nested))


class PairwiseEntropyPolars(PairwiseEntropy):
    """Reference: native Polars value_counts + entropy, one pair at a time (marginals memoised per call)."""

    def _compute(self, frames, combos):
        marginal: dict = {}

        def h1(col):
            if col not in marginal:
                marginal[col] = entropy_bits(frames[col[0]], [col[1]])
            return marginal[col]

        return self.entropy_rows(
            combos,
            [h1(a) for a, _ in combos],
            [h1(b) for _, b in combos],
            [entropy_bits(frames[f], [a, b]) for (f, a), (_, b) in combos],
            [frames[f].height for (f, _), _ in combos],
        )
```

Create `services/analytics/analytics/pairwise_entropy/__init__.py`:

```python
"""Pairwise joint entropy / mutual information (ordered scope). See PairwiseEntropy."""

from analytics.base import lazy_attributes
from analytics.pairwise_entropy.base import PairwiseEntropy
from analytics.pairwise_entropy.rust import PairwiseEntropyRust

REFERENCE = "PairwiseEntropyPolars"
IMPLEMENTATIONS = ("PairwiseEntropyRust", "PairwiseEntropyPolars")

__getattr__ = lazy_attributes(__name__, {"PairwiseEntropyPolars": ".polars"})
__all__ = ["PairwiseEntropy", "PairwiseEntropyRust", "PairwiseEntropyPolars", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 4: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_pairwise_entropy.py -q` → all PASS.

- [ ] **Step 5: Trim the old entropy tests, add the benchmark**

In `tests/test_entropy.py`, delete `assert_all_pairs`, `assert_all_marginals`, `test_pairwise_entropy`, `test_pairwise_entropy_subadditivity`, `test_pairwise_entropy_subadditivity_rust_marginal` and `test_marginal_entropy`, along with the now-unused imports `pairwise_joint_entropy` and `marginal_entropy`. Keep `reference_entropy`, `assert_all_triplets` and `test_threeway_entropy`; Task 8 deletes the file.

Create `tests/performance/benchmark_pairwise_entropy.py`:

```python
"""
Pairwise joint entropy speed benchmark: PairwiseEntropyRust vs PairwiseEntropyPolars (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_pairwise_entropy.py
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.pairwise_entropy",
        [
            large_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        ],
    )
```

- [ ] **Step 6: Smoke-run the benchmark, run the full suite, commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_pairwise_entropy.py`. It should complete; paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics/pairwise_entropy tests/test_pairwise_entropy.py tests/test_entropy.py tests/performance/benchmark_pairwise_entropy.py
git commit -m "feat: pairwise entropy / NMI technique classes (Rust/Polars) on the uniform contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Threeway entropy technique (ordered scope, arity 3)

**Files:**
- Create: `services/analytics/analytics/threeway_entropy/{__init__,base,rust,polars}.py`
- Create: `tests/test_threeway_entropy.py`, `tests/performance/benchmark_threeway_entropy.py`
- Delete: `tests/test_entropy.py`, `tests/performance/benchmark_entropy.py`

**Interfaces:**
- Consumes: `Technique`, `computed`, `group_by_frame`, `lazy_attributes`, `encodable`; `analytics.pairwise_entropy.polars.entropy_bits` and `analytics.pairwise_entropy.base.near_unique` (Task 7). From the plugin, `_plugin.threeway_joint_entropy(df, triplets)` returns the struct `threeway_entropy` with fields `col_a, col_b, col_c, entropy`.
- Produces: `analytics.threeway_entropy` exporting `ThreewayEntropy`, `ThreewayEntropyRust`, `ThreewayEntropyPolars`, with `REFERENCE = "ThreewayEntropyPolars"` and `IMPLEMENTATIONS = ("ThreewayEntropyRust", "ThreewayEntropyPolars")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_threeway_entropy.py`:

```python
"""
Three-way joint entropy accuracy tests — every implementation in analytics.threeway_entropy.

Reference: ThreewayEntropyPolars. mixed_dtypes: 18 eligible columns → C(18,3) = 816 triplets.
"""

import polars as pl
import pytest

from analytics.threeway_entropy import ThreewayEntropy
from datagen import mixed_dtypes
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.threeway_entropy"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"t": mixed_dtypes(200)}
    cls = load(impl)
    out = run(cls, frames)
    assert_contract(cls, out, frames)
    assert out.height == 816


@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"t": mixed_dtypes(1_000)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.parametrize("impl", ALL)
def test_known_answers(impl):
    df = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "c": [0, 0, 0, None]})
    r = run(load(impl), {"t": df}).row(0, named=True)
    assert r["h_abc"] == pytest.approx(2.0)  # 4 distinct triples
    assert r["n_rows"] == 4 and r["near_unique"] is True


def test_conclusions():
    Fixed = with_metrics(ThreewayEntropy, h_abc=[1.95, 1.5, 0.0, 1.95], n_rows=[4, 4, 1, 4])
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})  # C(4,3) = 4 triplets
    assert Fixed().add({"t": df}).result()["near_unique"].to_list() == [True, False, False, True]
    assert Fixed(near_unique_margin=0.01).add({"t": df}).result()["near_unique"].to_list() == [False, False, False, False]


@pytest.mark.parametrize("impl", ALL)
def test_zero_row_frame_is_ineligible(impl):
    df = pl.DataFrame({c: pl.Series([], dtype=pl.Int64) for c in "abc"})
    assert run(load(impl), {"t": df})["status"].to_list() == ["ineligible"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_threeway_entropy.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.threeway_entropy'`.

- [ ] **Step 3: Write the base, the implementations and `__init__`**

Create `services/analytics/analytics/threeway_entropy/base.py`:

```python
"""Three-way joint entropy H(A,B,C)."""

import polars as pl

from analytics._dtypes import encodable
from analytics.base import Technique, computed
from analytics.pairwise_entropy.base import near_unique


class ThreewayEntropy(Technique):
    """H(A,B,C) in bits for every column triplet within a frame. Joint entropy near
    log2(n_rows) ⇒ the triplet (almost) identifies rows.

    Null policy: null is its own category. Eligible: any column the Rust encoder
    accepts, in a frame with at least one row.
    """

    SCOPE = "ordered"
    ARITY = 3
    METRICS = {"h_abc": pl.Float64, "n_rows": pl.UInt32}
    CONCLUSIONS = {"near_unique": pl.Boolean}
    RTOL = 1e-5
    ATOL = 1e-12

    def __init__(self, *, near_unique_margin: float = 0.1):
        super().__init__()
        if near_unique_margin < 0:
            raise ValueError(f"near_unique_margin must be >= 0, got {near_unique_margin}")
        self.near_unique_margin = near_unique_margin

    def eligible(self, series: pl.Series) -> bool:
        return series.len() > 0 and encodable(series.dtype)

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(near_unique=computed(near_unique("h_abc", self.near_unique_margin)))
```

Create `services/analytics/analytics/threeway_entropy/rust.py`:

```python
import polars as pl

from analytics import _plugin
from analytics.base import group_by_frame
from analytics.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyRust(ThreewayEntropy):
    """Rust plugin `threeway_joint_entropy`: dense-id arithmetic keys, rayon-parallel across triplets."""

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(list(dict.fromkeys(c for k in group for _, c in k)))
            triplets = [tuple(c for _, c in k) for k in group]
            out = _plugin.threeway_joint_entropy(df, triplets).unnest("threeway_entropy")
            h = {frozenset((a, b, c)): e for a, b, c, e in out.iter_rows()}
            parts.append(
                self.metrics_frame(
                    group,
                    {"h_abc": [h[frozenset(t)] for t in triplets], "n_rows": [df.height] * len(group)},
                )
            )
        return pl.concat(parts)
```

Create `services/analytics/analytics/threeway_entropy/polars.py`:

```python
from analytics.pairwise_entropy.polars import entropy_bits
from analytics.threeway_entropy.base import ThreewayEntropy


class ThreewayEntropyPolars(ThreewayEntropy):
    """Reference: native Polars value_counts + entropy over a 3-field struct, one triplet at a time."""

    def _compute(self, frames, combos):
        return self.metrics_frame(
            combos,
            {
                "h_abc": [entropy_bits(frames[k[0][0]], [c for _, c in k]) for k in combos],
                "n_rows": [frames[k[0][0]].height for k in combos],
            },
        )
```

Create `services/analytics/analytics/threeway_entropy/__init__.py`:

```python
"""Three-way joint entropy (ordered scope, arity 3). See ThreewayEntropy."""

from analytics.base import lazy_attributes
from analytics.threeway_entropy.base import ThreewayEntropy
from analytics.threeway_entropy.rust import ThreewayEntropyRust

REFERENCE = "ThreewayEntropyPolars"
IMPLEMENTATIONS = ("ThreewayEntropyRust", "ThreewayEntropyPolars")

__getattr__ = lazy_attributes(__name__, {"ThreewayEntropyPolars": ".polars"})
__all__ = ["ThreewayEntropy", "ThreewayEntropyRust", "ThreewayEntropyPolars", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 4: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_threeway_entropy.py -q` → all PASS.

- [ ] **Step 5: Replace the old entropy test and benchmark**

```bash
git rm tests/test_entropy.py tests/performance/benchmark_entropy.py
```

Create `tests/performance/benchmark_threeway_entropy.py`:

```python
"""
Three-way joint entropy speed benchmark: ThreewayEntropyRust vs ThreewayEntropyPolars (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_threeway_entropy.py

large_dataset.arrow is cut to its first 40 columns (9,880 triplets): all 101 columns
give 166,650 triplets (~65 s per Rust run), far past the reference's budget.
"""

from harness import Dataset, large_dataset, run

from datagen import low_cardinality

if __name__ == "__main__":
    run(
        "analytics.threeway_entropy",
        [
            large_dataset(columns=40),
            Dataset("narrow 1M x 5", lambda: {"t": low_cardinality(1_000_000, 5)}),
            Dataset("wide 20K x 30", lambda: {"t": low_cardinality(20_000, 30)}),
        ],
        runs=3,
    )
```

- [ ] **Step 6: Smoke-run the benchmark, run the full suite, commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_threeway_entropy.py`. It should complete; paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics/threeway_entropy tests/test_threeway_entropy.py tests/performance/benchmark_threeway_entropy.py
git commit -m "feat: three-way entropy technique classes (Rust/Polars); retire old entropy test and benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Membership technique (multi-set scope; Bloom filters)

**Files:**
- Create: `services/analytics/analytics/_sets.py`
- Create: `services/analytics/analytics/membership/{__init__,base,rust,fastbloom,exact}.py`
- Modify: `services/analytics/analytics/_plugin.py` (add `bloom_filter_bits`; remove the `.analytics` expression namespace class `AnalyticFunctions` and `membership_ratio_sample`), `services/analytics/analytics/__init__.py` (drop `membership_ratio_sample` from the re-exports)
- Create: `tests/test_membership.py`, `tests/performance/benchmark_membership.py`
- Delete: `services/analytics/bloom_filter.py`, `tests/test_bloom_filter.py`, `tests/performance/benchmark_bloom_filter.py`

**Interfaces:**
- Consumes: `Technique`, `Column`, `columns_of`, `computed`, `at_least`, `check_unit`, `same_value`, `metric_mismatches`, `lazy_attributes`, `encodable`, `value_family`. From the plugin, `_plugin.membership_ratio(df, bit_array_bytes, k, m)` returns the struct `membership_ratio` with fields `col_name, ratio_all, ratio_non_null`.
- Produces:
  - `analytics._sets.distinct_values(series) -> list` and `analytics._sets.freeze(value)`, both reused in Task 10.
  - `analytics._plugin.bloom_filter_bits(series, k, m) -> list[int]`.
  - `analytics.membership` exporting `Membership`, `BloomMembership`, `BloomRust`, `BloomFastbloom`, `MembershipExact`, with `REFERENCE = "MembershipExact"` and `IMPLEMENTATIONS = ("BloomRust", "BloomFastbloom", "MembershipExact")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_membership.py`:

```python
"""
Set-membership / containment accuracy tests — every implementation in analytics.membership.

Reference: MembershipExact (Python sets of distinct non-null values). Bloom
implementations are probabilistic (EXACT=False): no false negatives, exact
distinct/non-null counts, and an aggregate false-positive rate within
FP_TOLERANCE × fp_rate of exact containment.
"""

import math

import polars as pl
import pytest

from analytics.membership import Membership
from datagen import mixed_dtypes, related_frames
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.membership"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")


@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = related_frames(200)
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: related_frames(1_000), id="related"),
        pytest.param(lambda: {"x": mixed_dtypes(300), "y": mixed_dtypes(300, seed=7)}, id="mixed"),
    ],
)
def test_agrees_with_reference(impl, make):
    frames = make()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def test_reference_known_containment():
    frames = {"p": pl.DataFrame({"id": [1, 2, 3, 4]}), "c": pl.DataFrame({"pid": [1, 1, 2, None]})}
    r = run(reference(PKG), frames).row(0, named=True)
    assert (r["ratio_a_in_b"], r["ratio_b_in_a"]) == (0.5, 1.0)
    assert (r["n_distinct_a"], r["n_distinct_b"], r["n_non_null_a"], r["n_non_null_b"]) == (4, 2, 4, 3)
    assert (r["unique_a"], r["unique_b"], r["relationship"]) == (True, False, "pk_fk")


@pytest.mark.parametrize("impl", ALL)
def test_planted_relationships(impl):
    out = run(load(impl), related_frames(1_000))
    rel = {(r["df_a"], r["col_a"], r["df_b"], r["col_b"]): r["relationship"] for r in out.iter_rows(named=True)}
    assert rel[("customers", "id", "orders", "customer_id")] == "pk_fk"
    assert rel[("customers", "id", "archive", "id")] == "pk_pk"
    assert rel[("customers", "name", "orders", "customer_name")] == "pk_fk"
    assert rel[("customers", "region", "orders", "region")] == "b_in_a"
    assert rel[("customers", "name", "archive", "name")] == "none"  # ~93% shared: similar, not contained
    assert rel[("customers", "id", "customers", "name")] is None  # Int64 vs String: ineligible


@pytest.mark.parametrize("impl", ALL)
def test_value_families(impl):
    df = pl.DataFrame(
        {
            "i32": pl.Series([1, 2], dtype=pl.Int32),
            "i64": pl.Series([1, 2], dtype=pl.Int64),
            "d": pl.Series([1, 2], dtype=pl.Int32).cast(pl.Date),
            "s": ["1", "2"],
            "f": [1.0, 2.0],
        }
    )
    out = run(load(impl), {"t": df})
    done = out.filter(pl.col("status") == "computed")
    assert done.select("col_a", "col_b").rows() == [("i32", "i64")]
    assert done.row(0, named=True)["ratio_a_in_b"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_empty_column(impl):
    df = pl.DataFrame({"e": pl.Series([None, None], dtype=pl.Int64), "v": [1, 2]})
    r = run(load(impl), {"t": df}).row(0, named=True)
    assert r["status"] == "computed"
    assert math.isnan(r["ratio_a_in_b"]) and r["ratio_b_in_a"] == 0.0
    assert r["unique_a"] is False and r["relationship"] == "none"


@pytest.mark.parametrize("impl", ALL)
def test_categorical_values_match_by_label_across_frames(impl):
    shared = [f"tok_{i}" for i in range(50)]
    pad = [f"pad_{i}" for i in range(50)]
    frames = {
        "f": pl.DataFrame({"c": pl.Series(shared).cast(pl.Categorical)}),
        "q": pl.DataFrame({"c": pl.Series(pad + shared).cast(pl.Categorical)}),
    }
    assert run(load(impl), frames).row(0, named=True)["ratio_a_in_b"] == 1.0


@pytest.mark.parametrize("impl", OTHERS)
def test_false_positive_rate_on_disjoint_sets(impl):
    frames = {"a": pl.DataFrame({"v": list(range(2_000))}), "b": pl.DataFrame({"w": list(range(10_000, 12_000))})}
    r = run(load(impl), frames).row(0, named=True)
    assert r["ratio_a_in_b"] <= 0.03 and r["ratio_b_in_a"] <= 0.03


@pytest.mark.parametrize("impl", OTHERS)
def test_fp_rate_validation(impl):
    with pytest.raises(ValueError):
        load(impl)(fp_rate=0.0)


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(
        Membership,
        ratio_a_in_b=[1.0, 1.0, 0.4, 1.0, 0.96, 0.5, 0.5, NAN, 0.0, 0.0],
        ratio_b_in_a=[1.0, 0.4, 1.0, 1.0, 0.5, 0.95, 0.5, NAN, 0.0, 0.0],
        n_distinct_a=[5, 3, 5, 3, 3, 3, 3, 0, 1, 1],
        n_distinct_b=[5, 5, 3, 3, 5, 3, 3, 0, 1, 1],
        n_non_null_a=[5, 6, 5, 6, 6, 6, 3, 0, 1, 1],
        n_non_null_b=[5, 5, 6, 6, 6, 6, 3, 0, 1, 1],
    )
    df = pl.DataFrame({c: [1] for c in "abcde"})  # C(5,2) = 10 pairs
    out = Fixed().add({"t": df}).result()
    assert out["relationship"].to_list() == [
        "pk_pk", "fk_pk", "pk_fk", "mutual", "a_in_b", "b_in_a", "none", "none", "none", "none"
    ]
    assert out["unique_a"].to_list() == [True, False, True, False, False, False, True, False, True, True]
    assert out["unique_b"].to_list() == [True, True, False, False, False, False, True, False, True, True]
    relaxed = Fixed(containment_threshold=0.5).add({"t": df}).result()
    assert relaxed["relationship"][6] == "pk_pk"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_membership.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.membership'`.

- [ ] **Step 3: Write `_sets.py` and the plugin wrapper**

Create `services/analytics/analytics/_sets.py`:

```python
"""Canonical distinct values for the pure-Python multi-set implementations.

Mirrors how the Rust encoder identifies values: Polars `unique()` already treats
-0.0 == 0.0 and every NaN as one value; freezing keeps that true inside Python
sets (one shared NaN object, since nan != nan) and makes nested values hashable.
"""

import math

import polars as pl

NAN = float("nan")


def freeze(value):
    if isinstance(value, float):
        if math.isnan(value):
            return NAN
        return 0.0 if value == 0 else value
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    return value


def distinct_values(series: pl.Series) -> list:
    """Distinct non-null values of `series`, frozen for use as set members."""
    return [freeze(v) for v in series.drop_nulls().unique().to_list()]
```

In `services/analytics/analytics/_plugin.py`:
- delete the whole `@pl.api.register_expr_namespace("analytics") class AnalyticFunctions` block (including its TODO comment);
- delete the whole `membership_ratio_sample` function;
- add this function in the same `Bloom Filter` section:

```python
def bloom_filter_bits(series: pl.Series, k: int, m: int) -> list[int]:
    """Build a fresh k-hash, m-bit Bloom filter over `series`; returns its ceil(m/8)
    bytes as ints. (pyo3-polars 0.24 pins pyo3 < 0.27, so kwargs cannot carry bytes.)"""
    out = series.to_frame().select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="bloom_filter",
            args=pl.col(series.name),
            kwargs={"bit_array_bytes": [], "k": k, "m": m},
            is_elementwise=True,
        )
    )
    return list(out.to_series()[0])
```

In `services/analytics/analytics/__init__.py`, remove `membership_ratio_sample,` from the re-export list.

- [ ] **Step 4: Write the base, the implementations and `__init__`**

Create `services/analytics/analytics/membership/base.py`:

```python
"""Set membership / containment between the distinct-value sets of two columns."""

import math

import polars as pl

from analytics._dtypes import encodable, value_family
from analytics.base import Technique, at_least, check_unit, computed, metric_mismatches, same_value

RELATIONSHIPS = ["pk_pk", "fk_pk", "pk_fk", "mutual", "a_in_b", "b_in_a", "none"]
_COUNTS = ["n_distinct_a", "n_distinct_b", "n_non_null_a", "n_non_null_b"]


class Membership(Technique):
    """Directional containment of distinct non-null values, both ways.

    ratio_a_in_b = |distinct(A) ∩ distinct(B)| / |distinct(A)| (NaN if A has no
    values; 0.0 if only B is empty). A column is unique when every non-null value
    is distinct. With t = containment_threshold:
        pk_pk   both ratios ≥ t, both unique
        fk_pk   A ⊂ B (ratio_a_in_b ≥ t), B unique, A not      (pk_fk: the mirror)
        mutual  both ratios ≥ t otherwise
        a_in_b / b_in_a   one direction ≥ t;  none   otherwise
    Pairs are compared only within one value family (analytics._dtypes.value_family).
    """

    SCOPE = "multi_set"
    ARITY = 2
    METRICS = {
        "ratio_a_in_b": pl.Float64,
        "ratio_b_in_a": pl.Float64,
        "n_distinct_a": pl.UInt32,
        "n_distinct_b": pl.UInt32,
        "n_non_null_a": pl.UInt32,
        "n_non_null_b": pl.UInt32,
    }
    CONCLUSIONS = {"unique_a": pl.Boolean, "unique_b": pl.Boolean, "relationship": pl.Enum(RELATIONSHIPS)}

    def __init__(self, *, containment_threshold: float = 0.95):
        super().__init__()
        check_unit("containment_threshold", containment_threshold)
        self.containment_threshold = containment_threshold

    def eligible(self, series: pl.Series) -> bool:
        return encodable(series.dtype)

    def compatible(self, dtypes) -> bool:
        return len({value_family(d) for d in dtypes}) == 1

    def membership_rows(self, frames, combos, n_distinct: dict, contained: dict) -> pl.DataFrame:
        """Metrics from exact distinct counts and directional ratios
        contained[(x, y)] = fraction of x's distinct values found in y."""

        def ratio(x, y):
            return contained.get((x, y), 0.0) if n_distinct[x] else math.nan

        non_null = {c: frames[c[0]][c[1]].len() - frames[c[0]][c[1]].null_count() for c in n_distinct}
        return self.metrics_frame(
            combos,
            {
                "ratio_a_in_b": [ratio(a, b) for a, b in combos],
                "ratio_b_in_a": [ratio(b, a) for a, b in combos],
                "n_distinct_a": [n_distinct[a] for a, _ in combos],
                "n_distinct_b": [n_distinct[b] for _, b in combos],
                "n_non_null_a": [non_null[a] for a, _ in combos],
                "n_non_null_b": [non_null[b] for _, b in combos],
            },
        )

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        t = self.containment_threshold
        a_in_b, b_in_a = at_least("ratio_a_in_b", t), at_least("ratio_b_in_a", t)
        ua = (pl.col("n_distinct_a") == pl.col("n_non_null_a")) & (pl.col("n_non_null_a") > 0)
        ub = (pl.col("n_distinct_b") == pl.col("n_non_null_b")) & (pl.col("n_non_null_b") > 0)
        rel = (
            pl.when(a_in_b & b_in_a & ua & ub).then(pl.lit("pk_pk"))
            .when(a_in_b & ub & ~ua).then(pl.lit("fk_pk"))
            .when(b_in_a & ua & ~ub).then(pl.lit("pk_fk"))
            .when(a_in_b & b_in_a).then(pl.lit("mutual"))
            .when(a_in_b).then(pl.lit("a_in_b"))
            .when(b_in_a).then(pl.lit("b_in_a"))
            .otherwise(pl.lit("none"))
        )
        return out.with_columns(
            unique_a=computed(ua),
            unique_b=computed(ub),
            relationship=computed(rel).cast(pl.Enum(RELATIONSHIPS)),
        )


class BloomMembership(Membership):
    """Shared by the Bloom-filter implementations: `fp_rate` and the probabilistic
    agreement bound (no false negatives; aggregate FP rate ≤ FP_TOLERANCE × fp_rate)."""

    EXACT = False
    FP_TOLERANCE = 3.0

    def __init__(self, *, fp_rate: float = 0.01, **thresholds):
        super().__init__(**thresholds)
        if not 0.0 < fp_rate < 1.0:
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")
        self.fp_rate = fp_rate

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        problems = metric_mismatches(result, reference, self.key_columns(), _COUNTS, 0.0, 0.0)
        if problems[:1] == ["key columns differ from the reference"]:
            return problems
        false_pos = negatives = 0.0
        for side, n_col in (("ratio_a_in_b", "n_distinct_a"), ("ratio_b_in_a", "n_distinct_b")):
            for got, want, n in zip(result[side].to_list(), reference[side].to_list(), reference[n_col].to_list()):
                if want is None or math.isnan(want):
                    if not same_value(got, want, 0.0, 0.0):
                        problems.append(f"{side}: {got!r} where the reference has {want!r}")
                    continue
                if got is None or got < want - 1e-12:
                    problems.append(f"{side}: false negative ({got!r} < exact {want!r})")
                    continue
                false_pos += (got - want) * n
                negatives += (1.0 - want) * n
        if negatives and false_pos / negatives > self.FP_TOLERANCE * self.fp_rate:
            problems.append(
                f"false-positive rate {false_pos / negatives:.4f} > {self.FP_TOLERANCE} × fp_rate {self.fp_rate}"
            )
        return problems
```

Create `services/analytics/analytics/membership/rust.py`:

```python
import math

import polars as pl

from analytics import _plugin
from analytics.base import columns_of
from analytics.membership.base import BloomMembership


def bloom_geometry(n: int, fp_rate: float) -> tuple[int, int]:
    """Optimal (m bits, k hashes) for n items: m = -n·ln(p)/ln(2)², rounded up to
    whole bytes (the plugin expects m in bits and ceil(m/8) bytes); k = (m/n)·ln 2."""
    m = -(n * math.log(fp_rate)) / (math.log(2) ** 2)
    m = (int(math.ceil(m)) + 7) // 8 * 8
    return m, max(1, int(math.ceil((m / n) * math.log(2))))


class BloomRust(BloomMembership):
    """Rust plugin: one Bloom filter per column over its distinct values (`bloom_filter`),
    then `membership_ratio` checks all partner columns against it in one rayon-parallel
    call (partners' distinct values null-padded to one frame; ratio_non_null is the
    distinct-value ratio)."""

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        distinct = {c: frames[c[0]][c[1]].drop_nulls().unique() for c in columns}
        partners: dict = {c: [] for c in columns}
        for a, b in combos:
            partners[a].append(b)
            partners[b].append(a)
        contained = {}
        for y in columns:
            queries = [x for x in partners[y] if distinct[x].len()]
            if not distinct[y].len() or not queries:
                continue
            m, k = bloom_geometry(distinct[y].len(), self.fp_rate)
            bits = _plugin.bloom_filter_bits(distinct[y], k=k, m=m)
            longest = max(distinct[x].len() for x in queries)
            padded = pl.DataFrame(
                [distinct[x].extend_constant(None, longest - distinct[x].len()).alias(f"q{i}") for i, x in enumerate(queries)]
            )
            ratios = _plugin.membership_ratio(padded, bit_array_bytes=bits, k=k, m=m).unnest("membership_ratio")
            contained.update(((x, y), r) for x, r in zip(queries, ratios["ratio_non_null"].to_list()))
        return self.membership_rows(frames, combos, {c: distinct[c].len() for c in columns}, contained)
```

Create `services/analytics/analytics/membership/fastbloom.py`:

```python
from fastbloom_rs import BloomFilter as FastBloomFilter

from analytics._sets import distinct_values
from analytics.base import columns_of
from analytics.membership.base import BloomMembership


class BloomFastbloom(BloomMembership):
    """fastbloom-rs (Rust-backed Bloom filter, one column at a time from Python);
    values keyed by their string form."""

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        keys = {c: [str(v) for v in distinct_values(frames[c[0]][c[1]])] for c in columns}
        partners: dict = {c: [] for c in columns}
        for a, b in combos:
            partners[a].append(b)
            partners[b].append(a)
        contained = {}
        for y in columns:
            if not keys[y]:
                continue
            bloom = FastBloomFilter(len(keys[y]), self.fp_rate)
            bloom.add_str_batch(keys[y])
            for x in partners[y]:
                if keys[x]:
                    contained[(x, y)] = sum(bloom.contains_str_batch(keys[x])) / len(keys[x])
        return self.membership_rows(frames, combos, {c: len(k) for c, k in keys.items()}, contained)
```

Create `services/analytics/analytics/membership/exact.py`:

```python
from analytics._sets import distinct_values
from analytics.base import columns_of
from analytics.membership.base import Membership


class MembershipExact(Membership):
    """Reference: exact containment from Python sets of distinct non-null values."""

    def _compute(self, frames, combos):
        sets = {c: frozenset(distinct_values(frames[c[0]][c[1]])) for c in columns_of(combos)}
        contained = {}
        for a, b in combos:
            shared = len(sets[a] & sets[b])
            if sets[a]:
                contained[(a, b)] = shared / len(sets[a])
            if sets[b]:
                contained[(b, a)] = shared / len(sets[b])
        return self.membership_rows(frames, combos, {c: len(s) for c, s in sets.items()}, contained)
```

Create `services/analytics/analytics/membership/__init__.py`:

```python
"""Set membership / containment (multi-set scope). See Membership for semantics."""

from analytics.base import lazy_attributes
from analytics.membership.base import BloomMembership, Membership
from analytics.membership.exact import MembershipExact
from analytics.membership.rust import BloomRust

REFERENCE = "MembershipExact"
IMPLEMENTATIONS = ("BloomRust", "BloomFastbloom", "MembershipExact")

__getattr__ = lazy_attributes(__name__, {"BloomFastbloom": ".fastbloom"})
__all__ = [
    "Membership", "BloomMembership", "BloomRust", "BloomFastbloom", "MembershipExact",
    "REFERENCE", "IMPLEMENTATIONS",
]
```

- [ ] **Step 5: Run the tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_membership.py -q` → all PASS.

- [ ] **Step 6: Replace the old Bloom files, add the benchmark**

```bash
git rm services/analytics/bloom_filter.py tests/test_bloom_filter.py tests/performance/benchmark_bloom_filter.py
```

Create `tests/performance/benchmark_membership.py`:

```python
"""
Membership (containment) speed benchmark: BloomRust vs BloomFastbloom vs MembershipExact (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_membership.py
"""

from harness import Dataset, large_dataset, run

from datagen import related_frames, similar_frames

if __name__ == "__main__":
    run(
        "analytics.membership",
        [
            large_dataset(),
            Dataset("narrow related 3 frames x 1M rows", lambda: related_frames(1_000_000)),
            Dataset(
                "wide 2 frames x 100 cols",
                lambda: similar_frames(n_similar=50, n_independent=50, col_size=5_000, n_elements=100_000),
            ),
        ],
    )
```

- [ ] **Step 7: Smoke-run the benchmark, run the full suite, commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_membership.py`. It should complete, with `agrees` = `yes` for both Bloom implementations. Paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics tests/test_membership.py tests/performance/benchmark_membership.py
git commit -m "feat: membership technique classes (BloomRust/BloomFastbloom/MembershipExact); retire BloomFilter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Similarity technique (multi-set scope; MinHash + LSH)

**Files:**
- Create: `services/analytics/analytics/similarity/{__init__,base,rust,datasketch,exact}.py`
- Create: `tests/test_similarity.py`, `tests/performance/benchmark_similarity.py`
- Delete: `services/analytics/deterministic_similarity_filter.py`, `services/analytics/minhash_lsh_filter.py`, `services/analytics/minhash_lsh_filter_datasketch.py`, `tests/test_similarity_filters.py`, `tests/performance/benchmark_jaccard.py`

**Interfaces:**
- Consumes: `Technique`, `columns_of`, `computed`, `at_least`, `check_unit`, `metric_mismatches`, `lazy_attributes`, `encodable`, `value_family`; `analytics._sets.distinct_values` (Task 9). From the plugin:
  - `_plugin.minhash(df, name, num_perm)` returns columns `qualified_name` (`"{name}|{column}"`) and `minhash` (`List[UInt32]`, null for a column with no values).
  - `_plugin.lsh_candidates(names, signatures, threshold, num_bands, rows_per_band)` is an Expr giving the struct `col_a, col_b`.
- Produces: `analytics.similarity` exporting `Similarity`, `MinHashRust`, `MinHashDatasketch`, `SimilarityExactLRU`, `optimal_lsh_params`, with `REFERENCE = "SimilarityExactLRU"` and `IMPLEMENTATIONS = ("MinHashRust", "MinHashDatasketch", "SimilarityExactLRU")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_similarity.py`:

```python
"""
Set-similarity (Jaccard / Overlap Coefficient) accuracy tests — every implementation
in analytics.similarity.

Reference: SimilarityExactLRU (brute force over every pair; distinct-value sets
memoised by a per-instance LRU cache). MinHash implementations are probabilistic
(EXACT=False): pairs they evaluate carry exact metrics (verification is exact), the
rest are "pruned", and recall of passing pairs must be >= Similarity.MIN_RECALL.

Run slow tests:   pytest -m slow
"""

import gc
import math
import weakref

import polars as pl
import pytest

from analytics.similarity import Similarity, SimilarityExactLRU
from datagen import related_frames, similar_frames
from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics

PKG = "analytics.similarity"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)
NAN = float("nan")

SMALL = {
    "df1": pl.DataFrame({"A": list(range(1, 11)), "B": list(range(3, 13)), "C": list(range(10, 20))}),
    "df2": pl.DataFrame({"A": list(range(1, 6)), "B": list(range(4, 9)), "C": list(range(10, 15))}),
}


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("make", [pytest.param(lambda: SMALL, id="small"), pytest.param(lambda: related_frames(200), id="related")])
def test_contract(impl, make):
    frames = make()
    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)


@pytest.mark.parametrize("impl", OTHERS)
@pytest.mark.parametrize("make", [pytest.param(lambda: SMALL, id="small"), pytest.param(lambda: related_frames(1_000), id="related")])
def test_agrees_with_reference(impl, make):
    frames = make()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.slow
@pytest.mark.parametrize("impl", OTHERS)
def test_recall_medium_scale(impl):
    frames = similar_frames()
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


def test_reference_known_pairs():
    out = run(SimilarityExactLRU, SMALL, overlap_threshold=0.9)
    passing = out.filter(pl.col("passes_jaccard") | pl.col("passes_overlap"))
    assert set(passing.select("df_a", "col_a", "df_b", "col_b").rows()) == {
        ("df1", "A", "df1", "B"),  # Jaccard 8/12
        ("df1", "A", "df2", "A"),  # df2.A ⊂ df1.A
        ("df1", "A", "df2", "B"),
        ("df1", "B", "df2", "B"),
        ("df1", "C", "df2", "C"),
    }
    aa = out.filter((pl.col("col_a") == "A") & (pl.col("df_b") == "df2") & (pl.col("col_b") == "A")).row(0, named=True)
    assert (aa["jaccard"], aa["overlap"]) == (0.5, 1.0)


@pytest.mark.parametrize("impl", ALL)
def test_value_families(impl):
    df = pl.DataFrame({"i32": pl.Series([1, 2], dtype=pl.Int32), "i64": pl.Series([1, 2], dtype=pl.Int64), "s": ["1", "2"]})
    out = run(load(impl), {"t": df})
    assert out.select("col_a", "col_b", "status").rows() == [
        ("i32", "i64", "computed"),
        ("i32", "s", "ineligible"),
        ("i64", "s", "ineligible"),
    ]
    assert out.row(0, named=True)["jaccard"] == 1.0


@pytest.mark.parametrize("impl", ALL)
def test_pipe_and_shared_names(impl):
    frames = {"x|y": pl.DataFrame({"a|b": [1, 2, 3]}), "z": pl.DataFrame({"a|b": [1, 2, 3]})}
    r = run(load(impl), frames).row(0, named=True)
    assert (r["df_a"], r["col_a"], r["df_b"], r["col_b"]) == ("x|y", "a|b", "z", "a|b")
    assert r["status"] == "computed" and (r["jaccard"], r["overlap"]) == (1.0, 1.0)


@pytest.mark.parametrize("impl", ALL)
def test_empty_column(impl):
    df = pl.DataFrame(
        {"e": pl.Series([None, None], dtype=pl.Int64), "v": [1, 2], "e2": pl.Series([None, None], dtype=pl.Int64)}
    )
    rows = {(r["col_a"], r["col_b"]): r for r in run(load(impl), {"t": df}).iter_rows(named=True)}
    ev, ee = rows[("e", "v")], rows[("e", "e2")]
    assert ev["status"] == ee["status"] == "computed"
    assert ev["jaccard"] == 0.0 and math.isnan(ev["overlap"])
    assert math.isnan(ee["jaccard"]) and math.isnan(ee["overlap"])
    assert not any(r["passes_jaccard"] or r["passes_overlap"] for r in rows.values())


def test_conclusions_default_override_and_nan():
    Fixed = with_metrics(Similarity, jaccard=[0.6, 0.59, NAN], overlap=[0.5, 0.95, NAN])
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1]})
    out = Fixed().add({"t": df}).result()
    assert out["passes_jaccard"].to_list() == [True, False, False]
    assert out["passes_overlap"].to_list() == [False, True, False]
    out = Fixed(jaccard_threshold=0.5, overlap_threshold=0.99).add({"t": df}).result()
    assert out["passes_jaccard"].to_list() == [True, True, False]
    assert out["passes_overlap"].to_list() == [False, False, False]


# ── the deliberate LRU cache ────────────────────────────────────────────────

def test_lru_is_per_instance_and_cleared_on_add():
    a, b = SimilarityExactLRU(), SimilarityExactLRU()
    assert a._distinct is not b._distinct
    a.add(SMALL).result()
    info = a._distinct.cache_info()
    assert info.currsize == 6 and info.hits == 24  # 15 pairs × 2 lookups, 6 misses
    a.add({"df3": pl.DataFrame({"A": [1]})})
    assert a._distinct.cache_info().currsize == 0


def test_cache_size_bounds_the_lru():
    t = SimilarityExactLRU(cache_size=2)
    t.add(SMALL).result()
    assert t._distinct.cache_info().maxsize == 2


def test_instances_are_not_kept_alive_by_the_cache():
    t = SimilarityExactLRU()
    t.add(SMALL).result()
    ref = weakref.ref(t)
    del t
    gc.collect()
    assert ref() is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_similarity.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'analytics.similarity'`.

- [ ] **Step 3: Write the base**

Create `services/analytics/analytics/similarity/base.py`:

```python
"""Set similarity (Jaccard index, Overlap Coefficient) between distinct-value sets."""

import functools
import math

import polars as pl

from analytics._dtypes import encodable, value_family
from analytics._sets import distinct_values
from analytics.base import Technique, at_least, check_unit, computed, metric_mismatches


class Similarity(Technique):
    """Jaccard = |A∩B| / |A∪B| (NaN when both sets are empty); Overlap = |A∩B| / min(|A|, |B|)
    (NaN when either set is empty). High Jaccard ⇒ heavy overlap of distinct values;
    high Overlap ⇒ one set is (nearly) contained in the other (catches PK-FK-like pairs).

    Sets are distinct non-null values; pairs from different value families are
    ineligible. Verification is always exact, memoised by a per-instance LRU cache of
    distinct-value sets (`cache_size`) that add() clears and no other instance shares.
    Probabilistic implementations mark pairs they never evaluate "pruned".
    """

    SCOPE = "multi_set"
    ARITY = 2
    METRICS = {"jaccard": pl.Float64, "overlap": pl.Float64}
    CONCLUSIONS = {"passes_jaccard": pl.Boolean, "passes_overlap": pl.Boolean}
    MIN_RECALL = 0.85

    def __init__(self, *, jaccard_threshold: float = 0.6, overlap_threshold: float = 0.95, cache_size: int = 4096):
        super().__init__()
        check_unit("jaccard_threshold", jaccard_threshold)
        check_unit("overlap_threshold", overlap_threshold)
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self.jaccard_threshold = jaccard_threshold
        self.overlap_threshold = overlap_threshold
        self._distinct = functools.lru_cache(maxsize=cache_size)(self._distinct_values)

    def eligible(self, series: pl.Series) -> bool:
        return encodable(series.dtype)

    def compatible(self, dtypes) -> bool:
        return len({value_family(d) for d in dtypes}) == 1

    def _on_add(self) -> None:
        self._distinct.cache_clear()

    def _distinct_values(self, frame: str, column: str) -> frozenset:
        return frozenset(distinct_values(self._collected[frame][column]))

    def verify(self, combos) -> dict[str, list[float]]:
        """Exact Jaccard / Overlap for `combos` from (cached) distinct-value sets."""
        jaccard, overlap = [], []
        for (fa, ca), (fb, cb) in combos:
            a, b = self._distinct(fa, ca), self._distinct(fb, cb)
            shared = len(a & b)
            union = len(a) + len(b) - shared
            smaller = min(len(a), len(b))
            jaccard.append(shared / union if union else math.nan)
            overlap.append(shared / smaller if smaller else math.nan)
        return {"jaccard": jaccard, "overlap": overlap}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        return out.with_columns(
            passes_jaccard=computed(at_least("jaccard", self.jaccard_threshold)),
            passes_overlap=computed(at_least("overlap", self.overlap_threshold)),
        )

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        if self.EXACT:
            return super().agreement(result, reference)
        keys = self.key_columns()
        if result.select(keys).rows() != reference.select(keys).rows():
            return ["key columns differ from the reference"]
        got, want = result["status"].to_list(), reference["status"].to_list()
        problems = [
            f"row {i}: status {g} vs reference {w}"
            for i, (g, w) in enumerate(zip(got, want))
            if (g == "ineligible") != (w == "ineligible")
        ]
        evaluated = [i for i, g in enumerate(got) if g == "computed"]
        problems += metric_mismatches(result[evaluated], reference[evaluated], keys, list(self.METRICS), 0.0, 0.0)

        def passing(df: pl.DataFrame) -> set[int]:
            flags = (df["passes_jaccard"].fill_null(False) | df["passes_overlap"].fill_null(False)).to_list()
            return {i for i, p in enumerate(flags) if p}

        truth = passing(reference)
        if truth:
            recall = len(truth & passing(result)) / len(truth)
            if recall < self.MIN_RECALL:
                problems.append(f"recall {recall:.3f} < {self.MIN_RECALL} (missed rows {sorted(truth - passing(result))[:10]})")
        return problems
```

- [ ] **Step 4: Write the implementations and `__init__`**

Create `services/analytics/analytics/similarity/exact.py`:

```python
from analytics.similarity.base import Similarity


class SimilarityExactLRU(Similarity):
    """Reference: brute-force exact Jaccard / Overlap for every pair. Distinct-value sets
    are memoised by the per-instance LRU cache — part of what the MinHash
    implementations are benchmarked against, not an incidental detail."""

    def _compute(self, frames, combos):
        return self.metrics_frame(combos, self.verify(combos))
```

Create `services/analytics/analytics/similarity/rust.py`. `optimal_lsh_params` is copied verbatim from the old `MinHashLSHFilter._optimal_lsh_params`.

```python
import polars as pl

from analytics import _plugin
from analytics.base import columns_of
from analytics.similarity.base import Similarity


def optimal_lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """(bands, rows_per_band) whose LSH S-curve best separates pairs around `threshold`."""
    min_error = float("inf")
    best_b, best_r = 1, num_perm
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            # P(candidate | similarity = s) = 1 - (1 - s^r)^b, evaluated at the threshold
            fp = 1 - (1 - threshold**r) ** b
            fn = (1 - threshold**r) ** b
            error = abs(fp - 0.9) + abs(fn - 0.1)
            if error < min_error:
                min_error = error
                best_b, best_r = b, r
    return best_b, best_r


class MinHashRust(Similarity):
    """Rust plugins `minhash` (signatures for all columns, rayon-parallel) and
    `lsh_candidates` (banded LSH); candidates then verified exactly, the rest pruned."""

    EXACT = False

    def __init__(self, *, num_perm: int = 128, **thresholds):
        super().__init__(**thresholds)
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        self.num_perm = num_perm
        # The LSH S-curve is calibrated against Jaccard. A pair that passes only on
        # the Overlap Coefficient (a small set inside a large one) can have a low
        # Jaccard, so the candidate stage uses 0.45 × the overlap threshold to let
        # such pairs through; exact verification then applies the real thresholds.
        self.lsh_threshold = min(self.jaccard_threshold * 0.9, self.overlap_threshold * 0.45)
        self.bands, self.rows_per_band = optimal_lsh_params(self.lsh_threshold, num_perm)

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        empty = {c for c in columns if frames[c[0]][c[1]].null_count() == frames[c[0]][c[1]].len()}
        names = list(frames)
        # Frame *index* prefixes the qualified name, so frame names containing "|"
        # are safe; split on the first "|" only, so column names may contain it.
        signatures = [
            _plugin.minhash(frames[name].select(cols), str(i), self.num_perm)
            for i, name in enumerate(names)
            if (cols := [c for f, c in columns if f == name and (f, c) not in empty])
        ]
        found: set = set()
        if signatures:
            sigs = pl.concat(signatures)
            if sigs.height > 1:
                pairs = sigs.select(
                    _plugin.lsh_candidates(
                        pl.col("qualified_name"),
                        pl.col("minhash"),
                        threshold=self.lsh_threshold,
                        num_bands=self.bands,
                        rows_per_band=self.rows_per_band,
                    ).alias("c")
                ).unnest("c")

                def parse(qualified: str):
                    i, column = qualified.split("|", 1)
                    return names[int(i)], column

                found = {frozenset((parse(a), parse(b))) for a, b in pairs.iter_rows()}
        chosen = [k for k in combos if frozenset(k) in found or k[0] in empty or k[1] in empty]
        chosen_set = set(chosen)
        pruned = [k for k in combos if k not in chosen_set]
        return pl.concat([self.metrics_frame(chosen, self.verify(chosen)), self.null_frame(pruned, "pruned")])
```

Create `services/analytics/analytics/similarity/datasketch.py`:

```python
import polars as pl
from datasketch import MinHash, MinHashLSH

from analytics.base import columns_of
from analytics.similarity.base import Similarity


class MinHashDatasketch(Similarity):
    """datasketch MinHash + MinHashLSH (pure Python, one column at a time); candidates
    then verified exactly, the rest pruned."""

    EXACT = False

    def __init__(self, *, num_perm: int = 128, **thresholds):
        super().__init__(**thresholds)
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        self.num_perm = num_perm
        # Kept from the original datasketch filter: 0.45 × the lower threshold (see MinHashRust).
        self.lsh_threshold = min(self.jaccard_threshold, self.overlap_threshold) * 0.45

    def _compute(self, frames, combos):
        columns = columns_of(combos)
        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm, weights=(0.9, 0.1))
        sketches = {}
        for i, (f, c) in enumerate(columns):
            values = [str(v).encode("utf8") for v in self._distinct(f, c)]
            if values:
                sketch = MinHash(num_perm=self.num_perm)
                sketch.update_batch(values)
                sketches[i] = sketch
                lsh.insert(i, sketch)
        found = {frozenset((columns[i], columns[j])) for i, s in sketches.items() for j in lsh.query(s) if j != i}
        empty = {c for i, c in enumerate(columns) if i not in sketches}
        chosen = [k for k in combos if frozenset(k) in found or k[0] in empty or k[1] in empty]
        chosen_set = set(chosen)
        pruned = [k for k in combos if k not in chosen_set]
        return pl.concat([self.metrics_frame(chosen, self.verify(chosen)), self.null_frame(pruned, "pruned")])
```

Create `services/analytics/analytics/similarity/__init__.py`:

```python
"""Set similarity — Jaccard / Overlap Coefficient (multi-set scope). See Similarity."""

from analytics.base import lazy_attributes
from analytics.similarity.base import Similarity
from analytics.similarity.exact import SimilarityExactLRU
from analytics.similarity.rust import MinHashRust, optimal_lsh_params

REFERENCE = "SimilarityExactLRU"
IMPLEMENTATIONS = ("MinHashRust", "MinHashDatasketch", "SimilarityExactLRU")

__getattr__ = lazy_attributes(__name__, {"MinHashDatasketch": ".datasketch"})
__all__ = [
    "Similarity", "MinHashRust", "MinHashDatasketch", "SimilarityExactLRU", "optimal_lsh_params",
    "REFERENCE", "IMPLEMENTATIONS",
]
```

- [ ] **Step 5: Run the tests, including the slow recall test**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_similarity.py -q` → all PASS.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_similarity.py -q -m slow` → PASS (recall ≥ 0.85 for both MinHash implementations).

- [ ] **Step 6: Replace the old similarity files, add the benchmark**

```bash
git rm services/analytics/deterministic_similarity_filter.py services/analytics/minhash_lsh_filter.py \
       services/analytics/minhash_lsh_filter_datasketch.py tests/test_similarity_filters.py \
       tests/performance/benchmark_jaccard.py
```

Create `tests/performance/benchmark_similarity.py`:

```python
"""
Set-similarity speed benchmark: MinHashRust vs MinHashDatasketch vs SimilarityExactLRU (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_similarity.py

The hypothesis here is "probabilistic pruning in Rust beats LRU-cached exact
checking of every pair"; `agrees` also enforces recall ≥ Similarity.MIN_RECALL.
"""

from harness import Dataset, large_dataset, run

from datagen import related_frames, similar_frames

if __name__ == "__main__":
    run(
        "analytics.similarity",
        [
            large_dataset(),
            Dataset("narrow related 3 frames x 1M rows", lambda: related_frames(1_000_000)),
            Dataset("medium 2 frames x 20 cols", lambda: similar_frames()),
            Dataset(
                "wide 2 frames x 100 cols",
                lambda: similar_frames(n_similar=50, n_independent=50, col_size=5_000, n_elements=100_000),
            ),
        ],
    )
```

- [ ] **Step 7: Smoke-run the benchmark, run the full suite, commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_similarity.py`. It should complete, with `agrees` = `yes` for both MinHash implementations. Paste the speedup table into the task report.
Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q -m "not slow"` → PASS.

```bash
git add services/analytics/analytics/similarity tests/test_similarity.py tests/performance/benchmark_similarity.py
git commit -m "feat: similarity technique classes (MinHashRust/MinHashDatasketch/SimilarityExactLRU); retire old filters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Remove transitional re-exports, finish test plumbing, update CLAUDE.md

**Files:**
- Modify: `services/analytics/analytics/__init__.py`, `services/analytics/analytics/_plugin.py` (docstring only), `tests/conftest.py`, `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the finished public API. `import analytics` exposes only `__version__`, and techniques are imported from their subpackages.

- [ ] **Step 1: Confirm that nothing outside the package uses the top-level plugin functions**

Run: `grep -rnE "from analytics import [a-z]|analytics\.(pairwise|threeway|marginal|column_gcd|minhash|lsh_|membership)|_analytics\." tests services --include=*.py`
Expected: only lines inside `services/analytics/analytics/` of the form `from analytics import _plugin`. Anything else must be migrated first.

- [ ] **Step 2: Shrink the package `__init__` and label `_plugin` private**

Replace the entire contents of `services/analytics/analytics/__init__.py` with:

```python
"""
Column-relationship analytics. Every technique is a subpackage with one class per
implementation, all used the same way:

    from analytics.chi_squared import ChiSquaredRust
    result = ChiSquaredRust(cramers_v_threshold=0.3).add({"sales": df}).result()

Per-column: analytics.gcd
Multi-set:  analytics.membership, analytics.similarity
Ordered:    analytics.chi_squared, analytics.pairwise_entropy,
            analytics.threeway_entropy, analytics.adjusted_rand

The Rust plugin wrappers in analytics._plugin are private.
"""

__version__ = "0.1.0"
```

At the top of `services/analytics/analytics/_plugin.py`, add this module docstring above the imports:

```python
"""Private thin wrappers around the compiled Rust plugin (analytics.pyd).

Only the *Rust implementation classes call these; the public API is the technique
classes in the analytics.<technique> subpackages.
"""
```

- [ ] **Step 3: Remove the `sys.path` workaround from conftest**

In `tests/conftest.py`, delete these lines:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analytics"))
```

- [ ] **Step 4: Run everything, including the slow tests**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q`
Expected: every test PASSes, including those marked `slow`. There should be no skips, except `fastbloom-rs`, `datasketch` or `polars-ds` if one of them is not installed; each such skip names the missing library.

- [ ] **Step 5: Update CLAUDE.md**

Make these edits to `CLAUDE.md`:

**(a)** Replace the whole `# Analytical Functions` section, from its heading up to (not including) `# Project Structure`, with:

```markdown
# Analytical Functions

Every technique is used the same way — `Impl(**params).add({"name": frame, ...}).result()` —
and returns one flat table: `df_a, col_a[, df_b, col_b[, df_c, col_c]] | status | descriptors | metrics | conclusions`.
`status` ∈ {computed, ineligible, pruned}; null = not computed, NaN = computed but undefined.
Ineligible columns/pairs are always listed, never dropped. Thresholds are keyword-only
constructor arguments with the defaults below. Techniques are grouped by **scope** —
how column combinations are enumerated.

## 1. Per-column (one column at a time)

**GCD — `analytics.gcd`** (`GcdRust`, `GcdNumpy`, ★`GcdMath`). ClickHouse GCD-codec method: the GCD of the magnitudes of each integer-backed column's raw physical values (Int/UInt 8–64, Int128, Decimal → unscaled, Date → days, Datetime/Duration → time unit, Time → ns). Nulls skipped; all-null / all-zero / zero-row → 0; magnitude 2¹²⁷ → null; other dtypes (incl. Categorical/Enum, UInt128) → ineligible. Descriptor `dtype` (Python `str(dtype)`) on every row. Conclusion `gcd_compressible` = gcd > 1. Rust: rayon-parallel across columns and 64K-value chunks, `binary_gcd(g, v % g)` fold with early exit at 1.

## 2. Multi-set (distinct-value sets; pairs may span frames)

Pairs are compared only within one **value family** (ints ≤64-bit; String/Categorical/Enum; otherwise exact dtype) — other pairs are ineligible.

**Membership — `analytics.membership`** (`BloomRust`, `BloomFastbloom`, ★`MembershipExact`). Directional containment of distinct values: `ratio_a_in_b`, `ratio_b_in_a`, plus exact distinct/non-null counts. Conclusions (`containment_threshold=0.95`): `unique_a/b`, `relationship` ∈ {pk_pk, fk_pk, pk_fk, mutual, a_in_b, b_in_a, none}. PK-PK needs reciprocal containment of unique values. Bloom implementations: `fp_rate=0.01`; no false negatives.

**Similarity — `analytics.similarity`** (`MinHashRust`, `MinHashDatasketch`, ★`SimilarityExactLRU`). Jaccard `|A∩B|/|A∪B|` and Overlap Coefficient `|A∩B|/min(|A|,|B|)`; conclusions `passes_jaccard` (0.6) / `passes_overlap` (0.95). MinHash implementations prune with LSH (`num_perm=128`, status `pruned`) and verify candidates exactly; the exact reference checks every pair through a per-instance LRU cache of distinct-value sets (`cache_size=4096`) — the comparison being tested is Rust probabilistic pruning vs LRU-cached brute force.

## 3. Ordered (row i of A vs row i of B; within one frame; zero-row frames ineligible)

**Chi-squared — `analytics.chi_squared`** (`ChiSquaredRust`, ★`ChiSquaredScipy`, `ChiSquaredPolarsDS`). Independence test on Boolean/String/Categorical/Enum/integer columns with ≤ `max_unique=1000` values; null rows dropped (`n_valid`). χ² detects any association but not causality; at 50K rows p-values are vanishingly small, so **Cramér's V is the diagnostic**: `associated` = V ≥ 0.3. `low_expected_count` flags unreliable tables.

**Joint entropy — `analytics.pairwise_entropy`** (`PairwiseEntropyRust`, ★`PairwiseEntropyPolars`) and **`analytics.threeway_entropy`** (`ThreewayEntropyRust`, ★`ThreewayEntropyPolars`). Null is its own category. Pairwise reports `h_a, h_b, h_ab`, `mi = h_a + h_b − h_ab`, `nmi = mi / min(h_a, h_b)`; `redundant` = NMI ≥ 0.9 (one column predicts the other). `near_unique` = joint entropy ≥ log₂(n_rows) − 0.1 (near-unique combinations — a different diagnostic). Threeway reports `h_abc` and `near_unique`.

**Adjusted Rand Index — `analytics.adjusted_rand`** (`AdjustedRandRust`, ★`AdjustedRandSklearn`). Chance-corrected partition agreement (1 = identical, ≈0 = chance, floor −0.5); null rows dropped (`n_valid`); `same_partition` = ARI ≥ 0.9. Unlike NMI it is chance-adjusted; unlike χ² it measures partition identity.

★ = accuracy reference.

**Not implemented:** run-length / REE compression analysis (`run_length.py` was deleted in `93f6dcb`). A Wald-Wolfowitz runs test is planned as a future ordered/per-column technique on this same contract.
```

**(b)** Replace the whole `# Project Structure` section with:

````markdown
# Project Structure

```
turbo-parakeet/
├── services/analytics/
│   ├── pyproject.toml, Cargo.toml          # maturin build (editable install via analytics.pth)
│   ├── src/                                # Rust plugin — lib.rs, shared.rs, entropy.rs, chi_squared.rs,
│   │                                       #   contingency.rs, ari.rs, gcd.rs, bloomfilter.rs, minhash.rs
│   └── analytics/
│       ├── __init__.py                     # __version__ only
│       ├── analytics.pyd                   # compiled plugin
│       ├── _plugin.py                      # PRIVATE plugin wrappers (called only by *Rust classes)
│       ├── _dtypes.py, _sets.py            # dtype groupings / value families; canonical distinct values
│       ├── base.py                         # Technique contract, helpers, metric_mismatches
│       └── <technique>/                    # gcd, membership, similarity, chi_squared,
│           ├── __init__.py                 #   pairwise_entropy, threeway_entropy, adjusted_rand
│           ├── base.py                     # technique base: METRICS, eligibility, conclusions, RTOL/ATOL
│           └── rust.py, <library>.py …     # one file per implementation
└── tests/
    ├── conftest.py                         # `slow` marker, `dataset` fixture
    ├── datagen.py                          # seeded generators shared with benchmarks
    ├── harness.py                          # implementation_params/load/reference/run/assert_contract/assert_agrees/with_metrics
    ├── test_base.py, test_datagen.py, test_benchmark_harness.py
    ├── test_<technique>.py                 # one per technique package
    ├── data/large_dataset.arrow            # 50K rows, 101 columns
    └── performance/                        # never collected by pytest
        ├── harness.py                      # shared timing harness
        └── benchmark_<technique>.py        # one per technique package
```
````

**(c)** Replace the whole `# Rust Plugin (analytics)` section with:

```markdown
# Rust Plugin (analytics)
Build: `maturin develop --release` from `services/analytics/`. Python changes need no rebuild (editable install).

Private — reached only through `analytics._plugin`, only by the `*Rust` classes:
`column_gcd`, `pairwise_chi_squared`, `pairwise_adjusted_rand`, `marginal_entropy`,
`pairwise_joint_entropy`, `threeway_joint_entropy` (no cap; C(101,3) = 166,650 at 101 cols, ~65 s at 50K rows),
`bloom_filter_bits` + `membership_ratio`, `minhash` + `lsh_candidates`.
(`membership`, `membership_ratio_sample` remain compiled but unused.)
```

**(d)** Replace the whole `# Testing Convention` section with:

```markdown
# Testing Convention
Every technique package declares `REFERENCE` (the exact accuracy reference) and `IMPLEMENTATIONS`.
- `tests/test_<technique>.py` — **accuracy only**, never timed. Same four blocks everywhere, via `tests/harness.py`:
  1. contract (every implementation): schema, every combination in canonical order, non-computed rows null;
  2. reference agreement (every non-reference implementation): `impl.agreement(result, reference_result)` —
     exact implementations within the technique's `RTOL/ATOL`; Bloom: no false negatives + aggregate FP ≤ 3×fp_rate;
     MinHash: evaluated pairs exact + recall ≥ 0.85;
  3. known answers (hand-worked cases, reference included);
  4. conclusions, once per technique base, via `with_metrics` (including a NaN row).
  Optional libraries missing → visible skip naming the library.
- `tests/performance/benchmark_<technique>.py` — **speed only**: ~20-line scripts calling `harness.run(package, datasets)`.
  Times `Impl(**params).add(frames).result()` (fresh instance, 1 warm-up + 5 runs, median/min) on
  large_dataset.arrow + a narrow/long + a wide shape. Rust implementations are re-timed at 1 thread in a child
  process (`RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`) to report **algorithmic** (fastest non-Rust ÷ Rust@1),
  **parallel** (Rust@1 ÷ Rust@N) and **total** speedups. Results → `tests/performance/results/*.parquet` (git-ignored).
- Rust unit tests: `cargo test --lib <module>::` from `services/analytics/` with `PYO3_PYTHON` set to the env's `python.exe` and the env dir on `PATH` (else `STATUS_DLL_NOT_FOUND`).
```

**(e)** In `# Current Focus`, replace the first paragraph with:

```markdown
All seven techniques (GCD; Membership, Similarity; Chi-squared, Pairwise/Threeway Entropy, ARI) share one class-based contract, one accuracy-test pattern and one benchmark harness (spec: docs/superpowers/specs/2026-09-24-uniform-technique-interface-design.md). Benchmark results with the algorithmic/parallel/total split live in tests/performance/results/. Next technique: Wald-Wolfowitz runs test.
```

Leave the "Entropy dense re-encoding" subsection as it is.

**(f)** In `# Next Steps`:
- Replace the `bloom_filter.py:71/86/99` bullet with: "`membership/rust.py`: bit arrays cross the FFI as `list[int]` (~60KB at fp=1%, n=50K) because pyo3-polars 0.24 pins pyo3 < 0.27 (no bytes kwargs). Pass bytes directly once pyo3-polars upgrades."
- Delete the whole `**[MinHashLSHFilter.py]…**` item. The 0.45 comment now lives in `similarity/rust.py`.

- [ ] **Step 6: Final full run and commit**

Run: `/c/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -q` → PASS.

```bash
git add services/analytics/analytics/__init__.py services/analytics/analytics/_plugin.py tests/conftest.py CLAUDE.md
git commit -m "refactor: technique classes are the public API; plugin wrappers private; CLAUDE.md grouped by scope

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
