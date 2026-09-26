# Describe Technique Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `analytics.describe`, a per-column profiling technique (counts, entropy, cardinality estimates, extremes, lengths, float/string/ISO scanners, Arrow and Polars sizes, inner-list values, classification) with a single-pass Rust implementation benchmarked against a Polars/pyarrow reference and a DataFusion implementation.

**Architecture:** A `Describe` technique base on the existing uniform contract (`analytics/base.py`) owns the metric schema, eligibility, rendering, cardinality estimators and classification. Three implementations fill the metrics: `DescribePolars` ★ (Polars expressions + pyarrow IPC sizes), `DescribeDataFusion` (SQL aggregates) and `DescribeRust` (two plugin entry points: `describe_columns` — one pass per column, rayon across columns and 64K-row chunks — and `column_sizes`). Implementations report min/max/top-5 as first-occurrence row indices; the base renders values once for all of them.

**Tech Stack:** Python 3.12, Polars 1.41, pyarrow 24, numpy, DataFusion 54 (Python); Rust with polars 0.51, pyo3-polars 0.24, rayon, foldhash, zstd 0.13, ryu, chrono/chrono-tz, bytemuck; maturin; pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-describe-technique-design.md`

## Global Constraints

- Contract: `Impl(**params).add({"name": frame}).result()`; output `df_a, col_a | status | dtype | METRICS | CONCLUSIONS`; `status` ∈ {computed, ineligible}; null = not computed / not applicable, NaN = computed but undefined.
- Constructor keyword-only: `population_rows: int | dict[str, int] | None = None`, `categorical_threshold: int = 10_000`, `zstd_level: int = 1` (1–22), `seed: int = 0`.
- Eligible: every dtype except `Object`, `Null`, `UInt128` (and any dtype nesting them).
- Ordering for argmin/argmax matches Polars `sort()`; ties keep the lowest row index; NaN excluded.
- Top-5 order: count descending, then first-occurrence index ascending.
- Split for Schnabel: Rust `splitmix64(seed + row) % 3`; Python `numpy.random.default_rng(seed).integers(0, 3, n)`.
- Sizes: Arrow = IPC body bytes of `to_arrow(CompatLevel.oldest())`; Polars = `estimated_size()` and ZSTD IPC body bytes of `to_arrow(CompatLevel.newest())`; columns nesting Int128 inside List/Array/Struct get null sizes.
- Numeric-string grammar `-?[0-9]+(\.[0-9]+)?`; ISO grammar: date `YYYY-MM-DD`, time `HH:MM[:SS[.f{1,9}]]`, datetime = date + `T`/space + time, offset `Z`/`±HH:MM`; uppercase `T`/`Z` only; no regex in Rust; regex elsewhere only via the Rust `regex` engine (Polars, DataFusion) — never Python `re` in library code.
- Agreement tolerances: all metrics exact except `entropy`/`inner_entropy` RTOL 1e-9, `size_zstd_bytes`/`size_polars_zstd_bytes` RTOL 0.01; `capture_history`/`inner_capture_history` not compared — `schnabel`/`inner_schnabel` within RTOL 0.10.
- The leading-zero rationale (spec §5.1) must appear verbatim as a comment in `patterns.rs`.
- Python: `C:\Users\Ben\miniconda3\envs\p312\python.exe` (below: `$PY` = `/c/Users/Ben/miniconda3/envs/p312/python.exe`). Rust tests: from `services/analytics/`, `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib <module>::`. Build: from `services/analytics/`, `$PY -m maturin develop --release`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Categorical/Enum sizes** — py-polars (pyarrow oracle) and the plugin's polars 0.51 may export different dictionaries for the same Categorical (global category mapping); a reasonable user expects all implementations to report the same size. Pinned by `test_categorical_and_enum_sizes_agree` in Task 11.
2. **DataFusion on nested values** — GROUP BY / ORDER BY over List, Array, Struct and dictionary columns may be unsupported or order nulls inside lists differently from Polars; a user expects identical argmin/n_unique. Pinned by `test_nested_ordering_with_null_elements` in Task 12.
3. **Multi-chunk and sliced input** — frames built with `pl.concat(..., rechunk=False)` or `.slice()` reach Rust as several chunks or with non-zero offsets; per-chunk code (first-occurrence indices, list offsets, IPC offsets) must still be exact. Pinned by `test_multi_chunk_and_sliced_input` in Task 11.
4. **Chunk-boundary merges** — a value whose first occurrence and minimum lie in different 64K-row chunks; a user expects the global first index. Pinned by `test_first_occurrence_across_parallel_chunks` in Task 11.
5. **Float32 digits** — `max_frac_digits` of a Float32 column must use the f32 shortest representation (`0.1f32` → 1, not 8 from its f64 widening). Pinned by `test_float32_uses_its_own_shortest_repr` in Task 5.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tests/datagen.py` | + `describe_mixed`, `stringified` (tests); `describe_narrow`, `describe_wide`, `describe_nested` (benchmarks) | 1, 13 |
| `services/analytics/analytics/describe/__init__.py` | exports, `REFERENCE`, `IMPLEMENTATIONS`, lazy imports | 2, 5, 11, 12 |
| `.../describe/estimators.py` | Chao1, Schnabel, Duj1, `estimate` (pure functions) | 2 |
| `.../describe/base.py` | `Describe`: metric schema, eligibility, validation, rendering, estimates, classification, agreement | 3 |
| `.../describe/_values.py` | shared Python helpers: dtype groups, `flatten`, split, frequency summary, frac digits, `n_midnight`, regex grammar | 3, 5, 6 |
| `.../describe/_sizes.py` | pyarrow IPC body sizes (Arrow + Polars) | 4 |
| `.../describe/polars.py` | `DescribePolars` ★ | 5, 6 |
| `services/analytics/src/shared.rs` | `encode_series` + Binary, Struct | 7 |
| `services/analytics/src/describe/frequency.rs` | counts, entropy, f1/f2, top-5, capture history | 7 |
| `services/analytics/src/describe/patterns.rs` | numeric and ISO byte scanners, `StringStats` | 8 |
| `services/analytics/src/describe/numeric.rs` | float stats | 9 |
| `services/analytics/src/describe/range.rs` | argmin/argmax, lengths, list ranges | 9 |
| `services/analytics/src/describe/sizes.rs` | IPC body walk + `column_sizes` plugin | 10 |
| `services/analytics/src/describe/mod.rs` | `describe_columns` plugin: composition, inner values, `n_midnight` | 11 |
| `services/analytics/analytics/_plugin.py` | + `describe_columns`, `column_sizes` wrappers | 11 |
| `.../describe/rust.py` | `DescribeRust` | 11 |
| `.../describe/datafusion.py` | `DescribeDataFusion` | 12 |
| `tests/test_describe.py` | contract, agreement, known answers, conclusions | 2–12 |
| `tests/performance/benchmark_describe.py` | speed benchmark | 13 |
| `CLAUDE.md` | technique entry, structure, plugin list | 13 |

---

### Task 1: Test data generators

**Files:**
- Modify: `tests/datagen.py` (append a new section; add imports)
- Test: `tests/test_datagen.py`

**Interfaces:**
- Produces: `describe_mixed(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame` (columns: `i8 i16 i32 i64 i128 u8 u16 u32 u64 codes f32 f64 f64_price f64_whole dec bool date dt_naive dt_tz dur time str_free str_int str_lead str_dec str_date str_dt str_dt_tz cat enum bin list_i64 list_str arr_i32 struct all_null`); `stringified(frame: pl.DataFrame) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_datagen.py`:

```python
from polars.testing import assert_frame_equal

from datagen import describe_mixed, stringified


def test_describe_mixed_is_seeded_and_covers_every_family():
    a, b = describe_mixed(300), describe_mixed(300)
    assert_frame_equal(a, b)
    kinds = {type(dt) for dt in a.dtypes}
    for kind in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128, pl.UInt8, pl.UInt16, pl.UInt32,
                 pl.UInt64, pl.Float32, pl.Float64, pl.Decimal, pl.Boolean, pl.Date, pl.Datetime,
                 pl.Duration, pl.Time, pl.String, pl.Categorical, pl.Enum, pl.Binary, pl.List,
                 pl.Array, pl.Struct):
        assert kind in kinds, kind
    assert a["all_null"].null_count() == 300
    assert a["dt_tz"].dtype.time_zone == "Europe/London"


def test_stringified_casts_only_castable_columns():
    s = stringified(describe_mixed(100))
    assert all(dt == pl.String or dt == pl.List(pl.String) for dt in s.dtypes)
    assert {"i32", "f64", "dec", "date", "dt_tz", "time", "bool", "list_i64"} <= set(s.columns)
    assert not {"dur", "bin", "struct", "arr_i32", "str_int", "cat", "all_null"} & set(s.columns)
```

(`pl` is already imported in `tests/test_datagen.py`; if not, add `import polars as pl`.)

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_datagen.py -k "describe_mixed or stringified" -v`
Expected: FAIL — `ImportError: cannot import name 'describe_mixed'`.

- [ ] **Step 3: Implement** — in `tests/datagen.py` add to the imports `from datetime import date, datetime, time, timedelta` and `from decimal import Decimal`, update the module docstring's list with `Per-column (describe): describe_mixed, stringified`, and append:

```python
# ── per-column (describe) ─────────────────────────────────────────────────────

_WORDS = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])


def describe_mixed(n_rows: int = 1_000, seed: int = 42) -> pl.DataFrame:
    """One column per dtype family `describe` profiles, with ~10% nulls, float
    specials (±0, NaN, ±inf, 0.1, 1e-7, 1.5e20), strings for the numeric and ISO
    scanners (incl. leading zeros and mixed offsets), nested and all-null columns."""
    rng = np.random.default_rng(seed)
    n = n_rows

    def nulls(values, rate: float = 0.1) -> list:
        mask = rng.random(n) < rate
        return [None if m else v for v, m in zip(values, mask)]

    ints = rng.integers(-1_000, 1_000, n)
    floats = np.round(rng.normal(100.0, 15.0, n), 2)
    specials = rng.choice(np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 0.1, 1e-7, 1.5e20]), n)
    days = rng.integers(0, 366, n)
    micros = rng.integers(0, 86_400, n) * 1_000_000 * (rng.random(n) < 0.5)  # ~half at midnight
    datetimes = [datetime(2024, 1, 1) + timedelta(days=int(d), microseconds=int(u)) for d, u in zip(days, micros)]
    offsets = np.array(["Z", "+02:00", "-05:30"])
    lists = [
        None if i % 11 == 0 else [] if i % 13 == 0 else [int(x) for x in rng.integers(0, 20, rng.integers(1, 4))]
        for i in range(n)
    ]
    words = lambda: _WORDS[rng.integers(0, 5, n)]
    return pl.DataFrame(
        [
            pl.Series("i8", nulls(rng.integers(-128, 128, n).tolist()), dtype=pl.Int8),
            pl.Series("i16", nulls(ints.tolist()), dtype=pl.Int16),
            pl.Series("i32", nulls((ints * 1_000).tolist()), dtype=pl.Int32),
            pl.Series("i64", nulls((ints.astype(np.int64) * 10**12).tolist()), dtype=pl.Int64),
            pl.Series("i128", nulls([int(v) * 10**25 for v in ints]), dtype=pl.Int128),
            pl.Series("u8", nulls(rng.integers(0, 256, n).tolist()), dtype=pl.UInt8),
            pl.Series("u16", rng.integers(0, 65_536, n).tolist(), dtype=pl.UInt16),
            pl.Series("u32", nulls(rng.integers(0, 2**32, n).tolist()), dtype=pl.UInt32),
            pl.Series("u64", nulls(([2**64 - 1] + rng.integers(0, 2**63, n).tolist())[:n]), dtype=pl.UInt64),
            pl.Series("codes", rng.integers(0, 5, n).tolist(), dtype=pl.Int64),
            pl.Series("f32", nulls(floats.tolist()), dtype=pl.Float32),
            pl.Series("f64", nulls(specials.tolist()), dtype=pl.Float64),
            pl.Series("f64_price", nulls(floats.tolist()), dtype=pl.Float64),
            pl.Series("f64_whole", np.arange(n, dtype=np.float64)),
            pl.Series("dec", nulls([Decimal(f"{v:.2f}") for v in floats]), dtype=pl.Decimal(10, 2)),
            pl.Series("bool", nulls((ints > 0).tolist()), dtype=pl.Boolean),
            pl.Series("date", nulls([date(2024, 1, 1) + timedelta(days=int(d)) for d in days]), dtype=pl.Date),
            pl.Series("dt_naive", nulls(datetimes), dtype=pl.Datetime("us")),
            pl.Series("dt_tz", nulls(datetimes), dtype=pl.Datetime("us")).dt.replace_time_zone(
                "Europe/London", ambiguous="earliest", non_existent="null"
            ),
            pl.Series("dur", nulls([timedelta(seconds=int(v)) for v in ints]), dtype=pl.Duration("us")),
            pl.Series("time", nulls([time(int(v) // 3600, int(v) // 60 % 60, int(v) % 60) for v in rng.integers(0, 86_400, n)]), dtype=pl.Time),
            pl.Series("str_free", nulls([f"{w} {v}" for w, v in zip(words(), ints)])),
            pl.Series("str_int", nulls([str(v) for v in ints])),
            pl.Series("str_lead", nulls([f"{abs(v):05d}" for v in ints])),
            pl.Series("str_dec", nulls([f"{v:.2f}" for v in floats])),
            pl.Series("str_date", nulls([str(date(2024, 1, 1) + timedelta(days=int(d))) for d in days])),
            pl.Series("str_dt", nulls([d.strftime("%Y-%m-%dT%H:%M:%S.%f") for d in datetimes])),
            pl.Series("str_dt_tz", nulls([d.strftime("%Y-%m-%dT%H:%M:%S") + o for d, o in zip(datetimes, offsets[rng.integers(0, 3, n)])])),
            pl.Series("cat", nulls(words().tolist()), dtype=pl.Categorical),
            pl.Series("enum", nulls(words().tolist()), dtype=pl.Enum(_WORDS.tolist())),
            pl.Series("bin", nulls([w.encode() for w in words()]), dtype=pl.Binary),
            pl.Series("list_i64", lists, dtype=pl.List(pl.Int64)),
            pl.Series("list_str", [None if v is None else [str(x) for x in v] for v in lists], dtype=pl.List(pl.String)),
            pl.Series("arr_i32", nulls([[int(x) for x in rng.integers(0, 10, 3)] for _ in range(n)]), dtype=pl.Array(pl.Int32, 3)),
            pl.Series("struct", nulls([{"a": int(a), "b": str(b)} for a, b in zip(rng.integers(0, 3, n), _WORDS[rng.integers(0, 2, n)])])),
            pl.Series("all_null", [None] * n, dtype=pl.String),
        ]
    )


def _castable(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or isinstance(dtype, (pl.Date, pl.Datetime, pl.Time, pl.Boolean))


def stringified(frame: pl.DataFrame) -> pl.DataFrame:
    """Every non-string column Polars can cast to String, cast to String
    (List(<castable>) → List(String)). Other columns are dropped: string-like,
    Duration (Polars cannot cast it to String), Binary, Array, Struct, all-null."""
    out = []
    for name, dtype in frame.schema.items():
        if _castable(dtype):
            out.append(frame[name].cast(pl.String))
        elif isinstance(dtype, pl.List) and _castable(dtype.inner):
            out.append(frame[name].cast(pl.List(pl.String)))
    return pl.DataFrame(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_datagen.py -v`
Expected: PASS (all datagen tests, old and new).

- [ ] **Step 5: Commit**

```bash
git add tests/datagen.py tests/test_datagen.py
git commit -m "test: add describe_mixed and stringified data generators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Cardinality estimators

**Files:**
- Create: `services/analytics/analytics/describe/__init__.py`, `services/analytics/analytics/describe/estimators.py`
- Create: `tests/test_describe.py`

**Interfaces:**
- Produces (in `analytics.describe.estimators`): `Z = 1.96`; `chao1(d: int, f1: int, f2: int) -> tuple[float, float, float]`; `schnabel(history: Sequence[int], d: int, n: int) -> tuple[float, float, float] | None`; `duj1(d: int, f1: int, n: int, q: float) -> float`; `estimate(d, n, f1, f2, history, q: float | None) -> dict` with keys `unique, chao1, chao1_low, chao1_high, schnabel, schnabel_low, schnabel_high, est_cardinality, est_method, est_low, est_high, estimates_agree`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_describe.py`:

```python
"""
describe accuracy tests — every implementation in analytics.describe.

Oracles: hand-worked known answers and DescribePolars, the technique's reference.
Accuracy only — nothing here is timed. Benchmarks live in tests/performance/.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pytest import approx

from analytics.describe import estimators

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — estimators (pure functions)

def test_chao1_with_doubletons():
    assert estimators.chao1(10, 4, 2) == approx((12.0, 10.249903382590167, 26.00618590489368))


def test_chao1_without_doubletons():
    assert estimators.chao1(5, 3, 0) == approx((8.0, 5.369121767830802, 29.38219792045809))


def test_chao1_without_unseen_mass_is_exact():
    assert estimators.chao1(7, 0, 3) == (7.0, 7.0, 7.0)
    assert estimators.chao1(7, 1, 0) == (7.0, 7.0, 7.0)


def test_schnabel_when_every_value_is_in_every_subset():
    assert estimators.schnabel([0, 0, 0, 0, 0, 0, 10], 10, 30_000) == approx(
        (9.523809523809524, 6.474567576908709, 16.378255262343956)
    )


def test_schnabel_mixed_history():
    # |S1| = 13, |S2| = 13, |S3| = 7, |S1 ∪ S2| = 20, R = 6 + 5 = 11, A = 13·13 + 7·20 = 309
    assert estimators.schnabel([5, 5, 5, 2, 2, 2, 1], 22, 1_000) == approx(
        (25.75, 15.69849888622768, 56.349688010901595)
    )


def test_schnabel_invalid_cases():
    assert estimators.schnabel([5, 5, 5, 2, 2, 2, 1], 22, 44) is None  # d/n = 0.5
    assert estimators.schnabel([3, 3, 3, 0, 0, 0, 0], 9, 100) is None  # no recaptures
    assert estimators.schnabel([0] * 7, 0, 0) is None


def test_duj1():
    assert estimators.duj1(10, 4, 40, 0.5) == approx(10.526315789473685)
    assert estimators.duj1(0, 0, 0, 0.5) == 0.0


HISTORY_10 = [0, 0, 0, 0, 0, 0, 10]


def test_estimate_picks_exact_then_duj1_then_schnabel_then_chao1():
    exact = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=1.0)
    assert (exact["est_method"], exact["est_cardinality"], exact["est_low"], exact["est_high"]) == ("exact", 10.0, 10.0, 10.0)
    duj = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=0.5)
    assert duj["est_method"] == "duj1" and duj["est_cardinality"] == approx(10.526315789473685)
    assert duj["est_low"] is None and duj["est_high"] is None
    sch = estimators.estimate(10, 40, 4, 2, HISTORY_10, q=None)
    assert sch["est_method"] == "schnabel" and sch["est_cardinality"] == approx(9.523809523809524)
    assert sch["estimates_agree"] is True  # [10.25, 26.0] overlaps [6.47, 16.38]
    chao = estimators.estimate(10, 15, 4, 2, HISTORY_10, q=None)  # d/n ≥ 0.5 → Schnabel invalid
    assert chao["est_method"] == "chao1" and chao["est_cardinality"] == 12.0
    assert chao["schnabel"] is None and chao["estimates_agree"] is None


def test_estimates_disagree_under_heavy_skew():
    e = estimators.estimate(10, 40, 8, 0, HISTORY_10, q=None)  # Chao1 [17.47, 114.95] vs Schnabel [6.47, 16.38]
    assert e["estimates_agree"] is False


def test_unique_flag():
    assert estimators.estimate(5, 5, 5, 0, [5, 0, 0, 0, 0, 0, 0], None)["unique"] is True
    assert estimators.estimate(0, 0, 0, 0, [0] * 7, None)["unique"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analytics.describe'`.

- [ ] **Step 3: Implement**

`services/analytics/analytics/describe/__init__.py`:

```python
"""Per-column profile for choosing narrower / more compressible Arrow types. See Describe."""

from analytics.base import lazy_attributes

REFERENCE = "DescribePolars"
IMPLEMENTATIONS = ("DescribePolars",)

__getattr__ = lazy_attributes(__name__, {"DescribePolars": ".polars"})
__all__ = ["REFERENCE", "IMPLEMENTATIONS"]
```

`services/analytics/analytics/describe/estimators.py`:

```python
"""Number-of-distinct-values estimators behind Describe's cardinality conclusions.

Pure functions of frequency counts, shared by every implementation. Intervals are
95% (z = 1.96) and closed-form. Estimators are picked by rule, never averaged:
Chao1 is a lower bound and Lincoln–Petersen/Schnabel is biased low when some
values are far more common than others, so an average has no meaningful interval.
"""

from __future__ import annotations

import math
from typing import Sequence

Z = 1.96


def chao1(d: int, f1: int, f2: int) -> tuple[float, float, float]:
    """Bias-corrected Chao1 with Chao's (1987) log-normal interval; variance as in
    the EstimateS user guide. Returns (estimate, low, high)."""
    s = d + f1 * (f1 - 1) / (2 * (f2 + 1))
    t = s - d
    if t <= 0:
        return float(s), float(d), float(d)
    if f2 > 0:
        var = (
            f1 * (f1 - 1) / (2 * (f2 + 1))
            + f1 * (2 * f1 - 1) ** 2 / (4 * (f2 + 1) ** 2)
            + f1**2 * f2 * (f1 - 1) ** 2 / (4 * (f2 + 1) ** 4)
        )
    else:
        var = f1 * (f1 - 1) / 2 + f1 * (2 * f1 - 1) ** 2 / 4 - f1**4 / (4 * s)
    k = math.exp(Z * math.sqrt(math.log(1 + max(var, 0.0) / t**2)))
    return float(s), d + t / k, d + t * k


def schnabel(history: Sequence[int], d: int, n: int) -> tuple[float, float, float] | None:
    """Schnabel (multi-sample Lincoln–Petersen) over the three split subsets.

    history[k-1] = distinct values whose subset mask is k (bit i = seen in subset i).
    Occasions in subset order: catches C_t = |S_t|, marked M_2 = |S1|, M_3 = |S1 ∪ S2|,
    recaptures R_2 = |S1 ∩ S2|, R_3 = |S3 ∩ (S1 ∪ S2)|. Interval: Byar's closed-form
    Poisson limits on R = R_2 + R_3. None unless n > 0, d/n < 0.5 and R ≥ 1.
    """
    if n == 0 or d / n >= 0.5:
        return None
    h = lambda *masks: sum(history[m - 1] for m in masks)
    s1, s2, s3 = h(1, 3, 5, 7), h(2, 3, 6, 7), h(4, 5, 6, 7)
    union12 = h(1, 2, 3, 5, 6, 7)
    r = h(3, 7) + h(5, 6, 7)
    if r < 1:
        return None
    a = s2 * s1 + s3 * union12
    r_lo = r * (1 - 1 / (9 * r) - Z / (3 * math.sqrt(r))) ** 3
    r_hi = (r + 1) * (1 - 1 / (9 * (r + 1)) + Z / (3 * math.sqrt(r + 1))) ** 3
    return a / (r + 1), a / r_hi, a / r_lo


def duj1(d: int, f1: int, n: int, q: float) -> float:
    """Haas–Stokes Duj1: sample of n non-null values (sampling fraction q) → population NDV."""
    return 0.0 if n == 0 else d / (1 - (1 - q) * f1 / n)


def estimate(d: int, n: int, f1: int, f2: int, history: Sequence[int], q: float | None) -> dict:
    """Every estimate plus the one picked by rule: q == 1 → exact; q < 1 → Duj1;
    Schnabel valid → Schnabel; else Chao1. q = frame rows / population rows (None: unknown)."""
    c, c_lo, c_hi = chao1(d, f1, f2)
    sch = schnabel(history, d, n)
    s, s_lo, s_hi = sch if sch else (None, None, None)
    if q == 1.0:
        method, est, lo, hi = "exact", float(d), float(d), float(d)
    elif q is not None:
        method, est, lo, hi = "duj1", duj1(d, f1, n, q), None, None
    elif sch:
        method, est, lo, hi = "schnabel", s, s_lo, s_hi
    else:
        method, est, lo, hi = "chao1", c, c_lo, c_hi
    return {
        "unique": d == n and n > 0,
        "chao1": c, "chao1_low": c_lo, "chao1_high": c_hi,
        "schnabel": s, "schnabel_low": s_lo, "schnabel_high": s_hi,
        "est_cardinality": est, "est_method": method, "est_low": lo, "est_high": hi,
        "estimates_agree": None if sch is None else (c_lo <= s_hi and s_lo <= c_hi),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe tests/test_describe.py
git commit -m "feat: describe cardinality estimators (Chao1, Schnabel, Duj1) with intervals

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Describe technique base

**Files:**
- Create: `services/analytics/analytics/describe/base.py`, `services/analytics/analytics/describe/_values.py`
- Modify: `services/analytics/analytics/describe/__init__.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `estimators.estimate` (Task 2); `Technique`, `metric_mismatches` (`analytics/base.py`).
- Produces: `analytics.describe.base`: `GROUP_A`, `GROUP_B`, `GROUP_C`, `VALUE_METRICS`, `SIZE_METRICS`, `METRICS`, `ESTIMATES`, `CONCLUSIONS`, `CLASS`, `METHOD`, `TOP5` (dicts / dtypes), `TOLERANCES`, `class Describe(Technique)` with attributes `population_rows, categorical_threshold, zstd_level, seed` and `_population(frame: str) -> int | None`. `analytics.describe._values`: `FLOATS`, `STRING_LIKE`, `INTEGERS`, `flatten(s: pl.Series) -> pl.Series`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_describe.py` (and add `from analytics.describe import Describe` and `from harness import assert_agrees, assert_contract, implementation_params, load, reference, run, with_metrics` to its imports):

```python
# ─────────────────────────────────────────────────────────────────────────────
# 4. Conclusions — rendering, estimates, classification (via with_metrics)

DEFAULTS = {
    "n_rows": 4, "n_null": 0, "n_unique": 4, "entropy": 2.0, "f1": 4, "f2": 0,
    "argmin": 0, "argmax": 3, "top5_idx": [0, 1, 2, 3], "top5_count": [1, 1, 1, 1],
    "capture_history": [1, 1, 0, 1, 0, 0, 1],
    "size_bytes": 32, "size_zstd_bytes": 32, "size_polars_bytes": 32, "size_polars_zstd_bytes": 32,
}


def conclude(s: pl.Series, params: dict | None = None, **overrides) -> dict:
    """Describe's conclusions for one column whose metrics are DEFAULTS | overrides."""
    metrics = {m: None for m in Describe.METRICS} | DEFAULTS | overrides
    cls = with_metrics(Describe, **{k: [v] for k, v in metrics.items()})
    return run(cls, {"t": s.to_frame()}, **(params or {})).row(0, named=True)


def test_renders_min_max_and_top5_from_indices():
    r = conclude(pl.Series("x", [3.5, 1.25, None, 3.5]), argmin=1, argmax=0, top5_idx=[0, 1], top5_count=[2, 1])
    assert (r["min"], r["max"]) == ("1.25", "3.5")
    assert r["top5"] == [{"value": "3.5", "count": 2}, {"value": "1.25", "count": 1}]


def test_renders_inner_values_from_flattened_indices():
    s = pl.Series("x", [[1, 2], None, [], [3]], dtype=pl.List(pl.Int64))
    r = conclude(
        s, n_null=1, inner_n_values=3, inner_n_null=0, inner_n_unique=3, inner_f1=3, inner_f2=0,
        inner_argmin=0, inner_argmax=2, inner_top5_idx=[2], inner_top5_count=[1],
        inner_capture_history=[1, 1, 1, 0, 0, 0, 0],
    )
    assert (r["inner_min"], r["inner_max"]) == ("1", "3")
    assert r["inner_top5"] == [{"value": "3", "count": 1}]


def test_scalar_columns_have_null_inner_conclusions():
    r = conclude(pl.Series("x", [1, 2, 3, 4]))
    assert r["inner_min"] is None and r["inner_class"] is None and r["inner_est_method"] is None


def test_estimate_branches_follow_population_rows():
    s = pl.Series("x", [1, 2, 3, 4])
    assert conclude(s)["est_method"] == "chao1" and conclude(s)["est_cardinality"] == 10.0
    assert conclude(s, {"population_rows": 4})["est_method"] == "exact"
    duj = conclude(s, {"population_rows": 8})
    assert duj["est_method"] == "duj1" and duj["est_cardinality"] == approx(8.0)
    assert conclude(s, {"population_rows": {"t": 8}})["est_method"] == "duj1"
    with pytest.raises(ValueError, match="population_rows"):
        conclude(s, {"population_rows": 3})


def test_schnabel_branch_and_agreement_flag():
    s = pl.Series("x", list(range(10)) * 4)
    agree = conclude(s, n_rows=40, n_unique=10, f1=0, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10])
    assert agree["est_method"] == "schnabel" and agree["estimates_agree"] is True
    skew = conclude(s, n_rows=40, n_unique=10, f1=8, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10])
    assert skew["estimates_agree"] is False


D = datetime(2024, 1, 1)
CLASS_CASES = [
    pytest.param(pl.Series("x", [None] * 4, dtype=pl.Int64), dict(n_null=4, n_unique=0, argmin=None, argmax=None), {}, "null", id="null"),
    pytest.param(pl.Series("x", [7, 7, None, 7]), dict(n_null=1, n_unique=1), {}, "constant", id="constant"),
    pytest.param(pl.Series("x", [1, 5, 1, 5]), dict(n_unique=2, argmax=1), {}, "boolean", id="boolean"),
    pytest.param(pl.Series("x", [0, 4, 1, 2, 3]), dict(n_rows=5, n_unique=5, argmin=0, argmax=1), {}, "ordinal", id="ordinal_before_categorical"),
    pytest.param(pl.Series("x", [-3, 5, 1, 2]), dict(argmin=0, argmax=1), {}, "categorical", id="negative_not_ordinal"),
    pytest.param(pl.Series("x", [-3, 5, 1, 2]), dict(argmin=0, argmax=1), {"categorical_threshold": 5}, "discrete", id="over_threshold"),
    pytest.param(pl.Series("x", [0, 9, 1, 2]), dict(argmin=0, argmax=1), {}, "categorical", id="max_over_2N"),
    pytest.param(pl.Series("x", [0, 9, 1, 2]), dict(argmin=0, argmax=1), {"population_rows": 100}, "ordinal", id="2N_uses_population"),
    pytest.param(pl.Series("x", [0.0, 2.0, 1.0, 3.0]), dict(argmax=3, n_nan=0, n_inf=0, n_fractional=0), {}, "ordinal", id="whole_floats"),
    pytest.param(pl.Series("x", [0.0, 2.5, 1.0, 3.0]), dict(argmax=3, n_nan=0, n_inf=0, n_fractional=1), {}, "categorical", id="fractional_floats"),
    pytest.param(pl.Series("x", ["3", "1", "2", "0"]), dict(argmin=3, argmax=0, n_numeric_int=4, n_leading_zero=0, numeric_int_min=0, numeric_int_max=3), {}, "ordinal", id="integer_strings"),
    pytest.param(pl.Series("x", ["3", "01", "2", "0"]), dict(argmin=3, argmax=0, n_numeric_int=4, n_leading_zero=1, numeric_int_min=0, numeric_int_max=3), {}, "categorical", id="leading_zero_strings"),
    pytest.param(pl.Series("x", [D, D, D, D]).dt.date(), dict(), {}, "categorical", id="temporal_never_ordinal"),
]


@pytest.mark.parametrize("s, overrides, params, expected", CLASS_CASES)
def test_classification(s, overrides, params, expected):
    assert conclude(s, params, **overrides)["class"] == expected


def test_zero_row_column_conclusions():
    r = conclude(
        pl.Series("x", [], dtype=pl.Int64), n_rows=0, n_unique=0, entropy=float("nan"), f1=0, f2=0,
        argmin=None, argmax=None, top5_idx=[], top5_count=[], capture_history=[0] * 7,
    )
    assert r["class"] == "null" and r["top5"] == [] and r["min"] is None
    assert r["est_method"] == "chao1" and r["est_cardinality"] == 0.0 and r["unique"] is False


@pytest.mark.parametrize(
    "params",
    [{"categorical_threshold": -1}, {"zstd_level": 0}, {"zstd_level": 23}, {"seed": -1}, {"population_rows": -5}, {"population_rows": {"t": -1}}],
)
def test_constructor_validates(params):
    cls = with_metrics(Describe, **{m: [None] for m in Describe.METRICS})
    with pytest.raises(ValueError):
        cls(**params)


def test_agreement_tolerances():
    s = pl.Series("x", list(range(10)) * 4)
    base = dict(n_rows=40, n_unique=10, f1=0, f2=0, capture_history=[0, 0, 0, 0, 0, 0, 10], entropy=3.0, size_zstd_bytes=1_000)
    ref = run(with_metrics(Describe, **{k: [v] for k, v in ({m: None for m in Describe.METRICS} | DEFAULTS | base).items()}), {"t": s.to_frame()})

    def compare(**changes) -> list[str]:
        metrics = {m: None for m in Describe.METRICS} | DEFAULTS | base | changes
        cls = with_metrics(Describe, **{k: [v] for k, v in metrics.items()})
        return cls().agreement(run(cls, {"t": s.to_frame()}), ref)

    assert compare(entropy=3.0 + 1e-12, size_zstd_bytes=1_005) == []
    assert compare(capture_history=[0, 0, 0, 0, 0, 1, 9]) == []  # Schnabel moves < 10%
    assert any("size_zstd_bytes" in p for p in compare(size_zstd_bytes=1_100))
    assert any("n_unique" in p for p in compare(n_unique=11))
    # [5, 4, 0, 0, 0, 0, 1]: |S1| = 6, |S2| = 5, |S3| = 1, R = 2 → Schnabel 40/3 ≈ 13.3 vs 9.52 (> 10%)
    assert any("schnabel" in p for p in compare(capture_history=[5, 4, 0, 0, 0, 0, 1]))
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: FAIL — `ImportError: cannot import name 'Describe'`.

- [ ] **Step 3: Implement**

`services/analytics/analytics/describe/_values.py`:

```python
"""Helpers shared by Describe's base and its Python implementations."""

from __future__ import annotations

import polars as pl

FLOATS = (pl.Float32, pl.Float64)
STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)
INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def flatten(s: pl.Series) -> pl.Series:
    """Values one nesting level down, in order, skipping null lists.

    Element i of the result is what `inner_argmin` / `inner_top5_idx` index into.
    Empty lists are filtered before exploding because explode turns them into a
    null row. The Rust kernel (describe/mod.rs::flatten) uses the same definition.
    """
    valid = s.drop_nulls()
    lengths = valid.list.len() if isinstance(s.dtype, pl.List) else valid.arr.len()
    return valid.filter(lengths > 0).explode()
```

`services/analytics/analytics/describe/base.py`:

```python
"""Per-column profile for choosing narrower / more compressible Arrow types (Spec A:
docs/superpowers/specs/2026-09-26-describe-technique-design.md).

Implementations fill METRICS. Min, max and top-5 values are reported as row indices
of their first occurrence; this base renders them, estimates cardinality and
classifies each column identically for every implementation.
"""

from __future__ import annotations

import polars as pl

from analytics.base import Technique, metric_mismatches
from analytics.describe import estimators
from analytics.describe._values import FLOATS, INTEGERS, STRING_LIKE, flatten

U64, U32, I128, F64 = pl.UInt64, pl.UInt32, pl.Int128, pl.Float64
LU64 = pl.List(pl.UInt64)

GROUP_A = {  # whole values — every eligible dtype
    "n_unique": U64, "entropy": F64, "f1": U64, "f2": U64, "argmin": U64, "argmax": U64,
    "min_len": U64, "max_len": U64, "top5_idx": LU64, "top5_count": LU64, "capture_history": LU64,
}
GROUP_B = {  # Float32 / Float64 only
    "n_nan": U64, "n_inf": U64, "n_fractional": U64, "max_frac_digits": U32, "n_f32_inexact": U64,
}
GROUP_C = {  # String / Categorical / Enum only
    "n_numeric": U64, "n_numeric_int": U64, "n_leading_zero": U64,
    "numeric_int_min": I128, "numeric_int_max": I128,
    "numeric_max_int_digits": U32, "numeric_max_frac_digits": U32,
    "n_iso_date": U64, "n_iso_time": U64, "n_iso_datetime": U64, "n_iso_datetime_tz": U64,
    "iso_max_frac_digits": U32, "iso_n_offsets": U64, "iso_n_midnight": U64,
}
VALUE_METRICS = {**GROUP_A, **GROUP_B, **GROUP_C}  # computed on outer values and on inner values
SIZE_METRICS = {"size_bytes": U64, "size_zstd_bytes": U64, "size_polars_bytes": U64, "size_polars_zstd_bytes": U64}
METRICS = {
    "n_rows": U64, "n_null": U64, **VALUE_METRICS, "n_midnight": U64, **SIZE_METRICS,
    "inner_n_values": U64, "inner_n_null": U64, **{f"inner_{k}": v for k, v in VALUE_METRICS.items()},
}

CLASS = pl.Enum(["null", "constant", "boolean", "ordinal", "categorical", "discrete"])
METHOD = pl.Enum(["exact", "duj1", "schnabel", "chao1"])
TOP5 = pl.List(pl.Struct({"value": pl.String, "count": pl.UInt64}))
ESTIMATES = {
    "unique": pl.Boolean, "chao1": F64, "chao1_low": F64, "chao1_high": F64,
    "schnabel": F64, "schnabel_low": F64, "schnabel_high": F64,
    "est_cardinality": F64, "est_method": METHOD, "est_low": F64, "est_high": F64, "estimates_agree": pl.Boolean,
}
_RENDERED = {"min": pl.String, "max": pl.String, "top5": TOP5}
CONCLUSIONS = {
    **_RENDERED, **ESTIMATES, "class": CLASS,
    **{f"inner_{k}": v for k, v in _RENDERED.items()}, **{f"inner_{k}": v for k, v in ESTIMATES.items()}, "inner_class": CLASS,
}

# Agreement: exact unless listed. capture_history depends on each implementation's
# own seeded split, so it is compared through the Schnabel estimate instead.
TOLERANCES = {"entropy": 1e-9, "inner_entropy": 1e-9, "size_zstd_bytes": 0.01, "size_polars_zstd_bytes": 0.01}
SPLIT_DEPENDENT = ("capture_history", "inner_capture_history")
SCHNABEL_RTOL = 0.10

_UINT128 = getattr(pl, "UInt128", None)


def _unsupported(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.Object, pl.Null)) or (_UINT128 is not None and dtype == _UINT128):
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _unsupported(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_unsupported(f.dtype) for f in dtype.fields)
    return False


class Describe(Technique):
    """Per-column profile: counts, entropy, cardinality estimates, extremes, lengths,
    float / numeric-string / ISO-datetime scanners, Arrow and Polars sizes, the same
    for list inner values, and a classification (first match wins):
    null → constant → boolean → ordinal → categorical → discrete.

    `population_rows` (int, or dict frame → int) is the size of the population the
    frame samples; it selects the estimator (exact / Duj1) and N in the ordinal rule
    (0 ≤ min, max ≤ 2N). `seed` fixes the 3-way split behind the Schnabel estimate.
    """

    SCOPE = "per_column"
    ARITY = 1
    DESCRIPTORS = {"dtype": pl.String}
    METRICS = METRICS
    CONCLUSIONS = CONCLUSIONS

    def __init__(
        self,
        *,
        population_rows: int | dict[str, int] | None = None,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        pops = [] if population_rows is None else population_rows.values() if isinstance(population_rows, dict) else [population_rows]
        if any(not isinstance(p, int) or p < 0 for p in pops):
            raise ValueError(f"population_rows must be non-negative ints, got {population_rows!r}")
        if categorical_threshold < 0:
            raise ValueError(f"categorical_threshold must be >= 0, got {categorical_threshold}")
        if not 1 <= zstd_level <= 22:
            raise ValueError(f"zstd_level must be in 1..22, got {zstd_level}")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed}")
        self.population_rows = population_rows
        self.categorical_threshold = categorical_threshold
        self.zstd_level = zstd_level
        self.seed = seed

    def _population(self, frame: str) -> int | None:
        if isinstance(self.population_rows, dict):
            return self.population_rows.get(frame)
        return self.population_rows

    # ── technique hooks ───────────────────────────────────────────────────────

    def eligible(self, series: pl.Series) -> bool:
        return not _unsupported(series.dtype)

    def describe(self, frames, combos):
        dtypes = {(n, c): str(dt) for n, f in frames.items() for c, dt in f.schema.items()}
        return {"dtype": [dtypes[n, c] for ((n, c),) in combos]}

    def _conclude(self, out: pl.DataFrame) -> pl.DataFrame:
        columns: dict[str, list] = {k: [] for k in self.CONCLUSIONS}
        for row in out.iter_rows(named=True):
            values = self._conclusions(row) if row["status"] == "computed" else {}
            for k, v in columns.items():
                v.append(values.get(k))
        return out.with_columns(pl.Series(k, v, dtype=self.CONCLUSIONS[k]) for k, v in columns.items())

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        keys = self.key_columns()
        exact = [m for m in self.METRICS if m not in TOLERANCES and m not in SPLIT_DEPENDENT]
        problems = metric_mismatches(result, reference, keys, exact, 0.0, 0.0)
        for metric, rtol in TOLERANCES.items():
            problems += metric_mismatches(result, reference, keys, [metric], rtol, 0.0)
        problems += metric_mismatches(result, reference, keys, ["schnabel", "inner_schnabel"], SCHNABEL_RTOL, 0.0)
        return list(dict.fromkeys(problems))

    # ── conclusions for one computed row ─────────────────────────────────────

    def _conclusions(self, r: dict) -> dict:
        s = self._collected[r["df_a"]][r["col_a"]]
        n_rows = r["n_rows"]
        pop = self._population(r["df_a"])
        if pop is not None and pop < n_rows:
            raise ValueError(f"population_rows {pop} < {n_rows} rows in frame {r['df_a']!r}")
        q = None if pop is None else 1.0 if pop == n_rows else n_rows / pop
        out = self._one_level(s, r, "", n_rows, q, pop if pop is not None else n_rows)
        if r["inner_n_values"] is not None:
            inner_n = r["inner_n_values"]
            out |= self._one_level(flatten(s), r, "inner_", inner_n, q, inner_n / q if q else inner_n)
        return out

    def _one_level(self, s: pl.Series, r: dict, p: str, n_values: int, q: float | None, big_n: float) -> dict:
        n = n_values - r[f"{p}n_null"]
        est = estimators.estimate(r[f"{p}n_unique"], n, r[f"{p}f1"], r[f"{p}f2"], r[f"{p}capture_history"], q)
        at = lambda i: None if i is None else str(s[i])
        return {
            f"{p}min": at(r[f"{p}argmin"]),
            f"{p}max": at(r[f"{p}argmax"]),
            f"{p}top5": [{"value": str(s[i]), "count": c} for i, c in zip(r[f"{p}top5_idx"], r[f"{p}top5_count"])],
            **{f"{p}{k}": v for k, v in est.items()},
            f"{p}class": self._classify(s, r, p, n_values, n, big_n, est["est_cardinality"]),
        }

    def _classify(self, s: pl.Series, r: dict, p: str, n_values: int, n: int, big_n: float, est: float) -> str:
        if r[f"{p}n_null"] == n_values:
            return "null"
        if r[f"{p}n_unique"] == 1:
            return "constant"
        if r[f"{p}n_unique"] == 2:
            return "boolean"
        bounds = _whole_range(s, r, p, n)
        if bounds is not None and 0 <= bounds[0] and bounds[1] <= 2 * big_n:
            return "ordinal"
        if est <= self.categorical_threshold:
            return "categorical"
        return "discrete"


def _whole_range(s: pl.Series, r: dict, p: str, n: int) -> tuple[float, float] | None:
    """(min, max) when every non-null value is a whole number, else None. Temporal
    dtypes never qualify (their physical integers are not quantities)."""
    lo, hi = r[f"{p}argmin"], r[f"{p}argmax"]
    if lo is None:
        return None
    dtype = s.dtype
    if isinstance(dtype, INTEGERS) or (isinstance(dtype, pl.Decimal) and dtype.scale == 0):
        return int(s[lo]), int(s[hi])
    if isinstance(dtype, FLOATS) and r[f"{p}n_fractional"] == r[f"{p}n_nan"] == r[f"{p}n_inf"] == 0:
        return s[lo], s[hi]
    if (
        isinstance(dtype, STRING_LIKE)
        and r[f"{p}n_numeric_int"] == n
        and r[f"{p}n_leading_zero"] == 0
        and r[f"{p}numeric_int_min"] is not None
    ):
        return r[f"{p}numeric_int_min"], r[f"{p}numeric_int_max"]
    return None
```

In `services/analytics/analytics/describe/__init__.py`, add `from analytics.describe.base import Describe` after the `lazy_attributes` import and add `"Describe"` to `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe tests/test_describe.py
git commit -m "feat: Describe technique base - schema, rendering, estimates, classification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: pyarrow IPC size helper

**Files:**
- Create: `services/analytics/analytics/describe/_sizes.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Produces: `ipc_body_bytes(arr: pa.Array, zstd_level: int | None) -> int`; `column_sizes(s: pl.Series, zstd_level: int) -> dict[str, int | None]` with keys `size_bytes, size_zstd_bytes, size_polars_bytes, size_polars_zstd_bytes`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_describe.py` (add `import pyarrow as pa` and `from analytics.describe import _sizes` to the imports):

```python
# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers — the pyarrow size oracle itself

def test_ipc_body_bytes_framing():
    seq = pa.array(np.arange(1_000, dtype=np.int32))
    assert _sizes.ipc_body_bytes(seq, None) == 4_000
    assert _sizes.ipc_body_bytes(seq, 1) == 1_912  # 8-byte prefix + ZSTD frame, padded to 8
    with_nulls = pa.array([None if i % 3 == 0 else i for i in range(1_000)], pa.int32())
    assert _sizes.ipc_body_bytes(with_nulls, None) == 4_128  # + 125-byte validity padded to 128
    assert _sizes.ipc_body_bytes(pa.array([], pa.int32()), None) == 0
    assert _sizes.ipc_body_bytes(pa.array([1], pa.int32()), 1) == 24
    assert _sizes.ipc_body_bytes(pa.array(["ab", None], pa.large_string()), None) == 40
    assert _sizes.ipc_body_bytes(pa.array([], pa.large_string()), None) == 8  # offsets [0]


def test_column_sizes_arrow_and_polars():
    s = pl.Series("x", np.arange(1_000, dtype=np.int32))
    assert _sizes.column_sizes(s, 1) == {
        "size_bytes": 4_000, "size_zstd_bytes": 1_912, "size_polars_bytes": 4_000, "size_polars_zstd_bytes": 1_912,
    }


def test_column_sizes_int128_and_nested_int128():
    assert _sizes.column_sizes(pl.Series("x", [1, None, 3], dtype=pl.Int128), 1)["size_bytes"] == 56  # 8 + 48
    nested = pl.Series("x", [[1], None], dtype=pl.List(pl.Int128))
    assert _sizes.column_sizes(nested, 1) == dict.fromkeys(_sizes.SIZE_KEYS)


def test_categorical_size_includes_its_dictionary():
    s = pl.Series("x", ["a", "b", "a"], dtype=pl.Categorical)
    assert _sizes.column_sizes(s, 1)["size_bytes"] == 48  # dictionary batch 32 + keys 16
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -k "ipc or column_sizes or categorical_size" -v`
Expected: FAIL — `ImportError: cannot import name '_sizes'`.

- [ ] **Step 3: Implement** — `services/analytics/analytics/describe/_sizes.py`:

```python
"""Arrow IPC body sizes shared by the Python implementations; pyarrow is the oracle.

A column's size is the body length of the IPC messages that carry it (dictionary
batches + record batch). pyarrow pads every buffer to 8 bytes, writes a validity
buffer only when the array has nulls, and gives an empty buffer no space. With
ZSTD, each non-empty buffer is an 8-byte uncompressed-length prefix plus one ZSTD
frame — pyarrow never falls back to raw bytes. src/describe/sizes.rs mirrors this.

Arrow sizes use the classic layout (CompatLevel.oldest(): LargeUtf8, LargeList);
Polars sizes are `estimated_size()` and the ZSTD body of its native layout
(CompatLevel.newest(): Utf8View/BinaryView) — what `write_ipc(compression="zstd")`
writes. pyarrow cannot import Polars Int128, so a top-level Int128 column is passed
as decimal128(38, 0) over the same 16-byte values; columns nesting Int128 inside
List/Array/Struct get null sizes (in every implementation).
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

SIZE_KEYS = ("size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes")


def ipc_body_bytes(arr: pa.Array, zstd_level: int | None) -> int:
    batch = pa.record_batch([arr], names=["x"])
    codec = None if zstd_level is None else pa.Codec("zstd", compression_level=zstd_level)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema, options=pa.ipc.IpcWriteOptions(compression=codec)) as writer:
        writer.write_batch(batch)
    return sum(m.body.size for m in pa.ipc.MessageReader.open_stream(sink.getvalue()) if m.type != "schema")


def _nests_int128(dtype: pl.DataType) -> bool:
    if isinstance(dtype, (pl.List, pl.Array)):
        return dtype.inner == pl.Int128 or _nests_int128(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(f.dtype == pl.Int128 or _nests_int128(f.dtype) for f in dtype.fields)
    return False


def _to_arrow(s: pl.Series, compat: pl.CompatLevel) -> pa.Array:
    if s.dtype == pl.Int128:
        values = b"".join((v or 0).to_bytes(16, "little", signed=True) for v in s.to_list())
        validity = pa.array(s.is_not_null().to_list()).buffers()[1] if s.null_count() else None
        return pa.Array.from_buffers(pa.decimal128(38, 0), s.len(), [validity, pa.py_buffer(values)], null_count=s.null_count())
    return s.to_arrow(compat_level=compat)


def column_sizes(s: pl.Series, zstd_level: int) -> dict[str, int | None]:
    if _nests_int128(s.dtype):
        return dict.fromkeys(SIZE_KEYS)
    s = s.rechunk()
    classic = _to_arrow(s, pl.CompatLevel.oldest())
    native = _to_arrow(s, pl.CompatLevel.newest())
    return {
        "size_bytes": ipc_body_bytes(classic, None),
        "size_zstd_bytes": ipc_body_bytes(classic, zstd_level),
        "size_polars_bytes": s.estimated_size(),
        "size_polars_zstd_bytes": ipc_body_bytes(native, zstd_level),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe/_sizes.py tests/test_describe.py
git commit -m "feat: pyarrow IPC body-size oracle for describe (Arrow and Polars layouts)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: DescribePolars — frequencies, extremes, lengths, floats, datetimes, sizes, inner values

**Files:**
- Create: `services/analytics/analytics/describe/polars.py`
- Modify: `services/analytics/analytics/describe/_values.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `Describe`, `VALUE_METRICS`, `GROUP_B`, `GROUP_C` (Task 3); `column_sizes` (Task 4); `flatten`, `FLOATS`, `STRING_LIKE` (Task 3).
- Produces: `_values.subsets(n: int, seed: int) -> np.ndarray`; `_values.frequency_summary(count, first, mask, n_rows: int, n_null: int) -> dict` (keys: `n_unique, entropy, f1, f2, top5_idx, top5_count, capture_history`); `_values.frac_digits(reprs: pl.Series) -> int | None`; `_values.n_midnight(s: pl.Series) -> int | None`; `class DescribePolars(Describe)`; module functions `profile(s, seed) -> dict` (VALUE_METRICS keys), `string_stats(s) -> dict` (GROUP_C keys; returns all-None until Task 6).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_describe.py`:

```python
from datagen import describe_mixed, stringified

PKG = "analytics.describe"
ALL = implementation_params(PKG)
OTHERS = implementation_params(PKG, include_reference=False)


def profile(impl: str, s: pl.Series, **params) -> dict:
    """describe one Series; its result row as a dict."""
    return run(load(impl), {"t": s.to_frame()}, **params).row(0, named=True)


def ineligible_frame() -> pl.DataFrame:
    cols = [pl.Series("obj", [object(), object()], dtype=pl.Object), pl.Series("nul", [None, None], dtype=pl.Null)]
    if hasattr(pl, "UInt128"):
        cols.append(pl.Series("u128", [1, 2], dtype=pl.UInt128))
    return pl.DataFrame(cols)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract

@pytest.mark.parametrize("impl", ALL)
def test_contract(impl):
    frames = {"mixed": describe_mixed(300), "empty": describe_mixed(50).clear(), "bad": ineligible_frame()}
    cls = load(impl)
    result = run(cls, frames)
    assert_contract(cls, result, frames)
    assert set(result.filter(pl.col("df_a") == "bad")["status"]) == {"ineligible"}
    assert set(result.filter(pl.col("df_a") != "bad")["status"]) == {"computed"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reference agreement

@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference(impl):
    frames = {"mixed": describe_mixed(2_000), "strings": stringified(describe_mixed(500)), "empty": describe_mixed(50).clear()}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


@pytest.mark.slow
@pytest.mark.parametrize("impl", OTHERS)
def test_agrees_with_reference_on_large_dataset(impl):
    frames = {"large": pl.read_ipc(LARGE)}
    cls = load(impl)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known answers (the reference is included, so these are its oracle tests)

@pytest.mark.parametrize("impl", ALL)
def test_frequencies_entropy_and_top5(impl):
    r = profile(impl, pl.Series("x", ["a", "a", "b", None]))
    assert (r["n_rows"], r["n_null"], r["n_unique"], r["f1"], r["f2"]) == (4, 1, 2, 1, 1)
    assert r["entropy"] == approx(1.5)
    assert (r["top5_idx"], r["top5_count"]) == ([0, 2], [2, 1])
    assert sum(r["capture_history"]) == 2


@pytest.mark.parametrize("impl", ALL)
def test_top5_ties_break_by_first_occurrence(impl):
    r = profile(impl, pl.Series("x", [3, 1, 2, 1, 2, 3, 4, 5, 6]))
    assert (r["top5_idx"], r["top5_count"]) == ([0, 1, 2, 6, 7], [2, 2, 2, 1, 1])


@pytest.mark.parametrize("impl", ALL)
def test_extremes_are_first_occurrences(impl):
    r = profile(impl, pl.Series("x", [5, 1, 3, 1, 5]))
    assert (r["argmin"], r["argmax"], r["min"], r["max"]) == (1, 0, "1", "5")


@pytest.mark.parametrize("impl", ALL)
def test_float_zero_and_nan_are_one_value_each(impl):
    r = profile(impl, pl.Series("x", [0.0, -0.0, float("nan"), float("nan"), 1.5]))
    assert (r["n_unique"], r["n_nan"], r["argmin"], r["argmax"], r["n_fractional"], r["max_frac_digits"]) == (3, 2, 0, 4, 1, 1)


@pytest.mark.parametrize("impl", ALL)
def test_float_decimal_places_and_specials(impl):
    r = profile(impl, pl.Series("x", [0.1, 1e-7, 1.5e20, 3.0, float("inf"), None]))
    assert (r["max_frac_digits"], r["n_fractional"], r["n_inf"], r["n_nan"]) == (7, 2, 1, 0)


@pytest.mark.parametrize("impl", ALL)
def test_f32_round_trip(impl):
    assert profile(impl, pl.Series("x", [0.1]))["n_f32_inexact"] == 1
    assert profile(impl, pl.Series("x", [0.5, 3.0, None]))["n_f32_inexact"] == 0


@pytest.mark.parametrize("impl", ALL)
def test_float32_uses_its_own_shortest_repr(impl):
    r = profile(impl, pl.Series("x", [0.1, 0.25], dtype=pl.Float32))
    assert (r["max_frac_digits"], r["n_f32_inexact"]) == (2, None)


@pytest.mark.parametrize("impl", ALL)
def test_non_float_columns_have_null_float_stats(impl):
    r = profile(impl, pl.Series("x", [1, 2]))
    assert all(r[k] is None for k in ("n_nan", "n_inf", "n_fractional", "max_frac_digits", "n_f32_inexact"))


@pytest.mark.parametrize("impl", ALL)
def test_string_and_list_lengths(impl):
    s = profile(impl, pl.Series("x", ["ab", "", None, "héllo"]))
    assert (s["min_len"], s["max_len"]) == (0, 6)  # UTF-8 bytes
    lst = profile(impl, pl.Series("x", [[1, 2], [], None], dtype=pl.List(pl.Int64)))
    assert (lst["min_len"], lst["max_len"]) == (0, 2)
    assert profile(impl, pl.Series("x", [1, 2]))["min_len"] is None


@pytest.mark.parametrize("impl", ALL)
def test_enum_orders_by_category_and_categorical_by_string(impl):
    e = profile(impl, pl.Series("x", ["a", "z", "a"], dtype=pl.Enum(["z", "a"])))
    assert (e["argmin"], e["argmax"]) == (1, 0)
    c = profile(impl, pl.Series("x", ["z", "a", "z"], dtype=pl.Categorical))
    assert (c["argmin"], c["argmax"]) == (1, 0)


@pytest.mark.parametrize("impl", ALL)
def test_equal_struct_values_count_once(impl):
    r = profile(impl, pl.Series("x", [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 1, "b": None}, None]))
    assert (r["n_unique"], r["n_null"]) == (2, 1)


@pytest.mark.parametrize("impl", ALL)
def test_list_whole_and_inner_values(impl):
    r = profile(impl, pl.Series("x", [[1, None], None, [], [3, 1]], dtype=pl.List(pl.Int64)))
    assert (r["n_rows"], r["n_null"], r["n_unique"], r["argmin"], r["argmax"]) == (4, 1, 3, 2, 3)
    assert (r["inner_n_values"], r["inner_n_null"], r["inner_n_unique"]) == (4, 1, 2)  # [1, None, 3, 1]
    assert (r["inner_argmin"], r["inner_argmax"], r["inner_f1"], r["inner_f2"]) == (0, 2, 1, 1)
    assert (r["inner_top5_idx"], r["inner_top5_count"]) == ([0, 2], [2, 1])
    assert (r["inner_min"], r["inner_max"]) == ("1", "3")


@pytest.mark.parametrize("impl", ALL)
def test_n_midnight_uses_local_time(impl):
    s = pl.Series(
        "x", [datetime(2024, 3, 31, 0, 0), datetime(2024, 3, 31, 12, 0), datetime(2024, 7, 1, 0, 0), datetime(2024, 10, 28, 0, 0, 0, 1)]
    ).dt.replace_time_zone("Europe/London")
    # local midnight, noon, local midnight in BST (23:00 UTC — a UTC check would miss it), 1 µs past midnight
    assert profile(impl, s)["n_midnight"] == 2
    assert profile(impl, pl.Series("x", [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)]))["n_midnight"] == 1
    assert profile(impl, pl.Series("x", [1]))["n_midnight"] is None



@pytest.mark.parametrize("impl", ALL)
def test_zero_row_and_all_null_columns(impl):
    z = profile(impl, pl.Series("x", [], dtype=pl.Int32))
    assert (z["n_rows"], z["n_unique"], z["argmin"], z["top5_idx"], z["capture_history"]) == (0, 0, None, [], [0] * 7)
    assert math.isnan(z["entropy"]) and z["size_bytes"] == 0 and z["min_len"] is None
    a = profile(impl, pl.Series("x", [None, None], dtype=pl.String))
    assert (a["n_unique"], a["entropy"], a["argmin"]) == (0, 0.0, None)


@pytest.mark.parametrize("impl", ALL)
def test_capture_history(impl):
    everywhere = profile(impl, pl.Series("x", np.repeat(np.arange(10), 1_000)))
    assert everywhere["capture_history"] == [0, 0, 0, 0, 0, 0, 10]
    rng = np.random.default_rng(7)
    r = profile(impl, pl.Series("x", rng.integers(0, 5_000, 20_000)))
    assert sum(r["capture_history"]) == r["n_unique"]


@pytest.mark.parametrize("impl", ALL)
def test_sizes_through_the_technique(impl):
    r = profile(impl, pl.Series("x", np.arange(1_000, dtype=np.int32)))
    assert (r["size_bytes"], r["size_polars_bytes"]) == (4_000, 4_000)
    assert r["size_zstd_bytes"] == approx(1_912, rel=0.01)
    assert profile(impl, pl.Series("x", [None if i % 3 == 0 else i for i in range(1_000)], dtype=pl.Int32))["size_bytes"] == 4_128
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: the new tests are SKIPPED with `DescribePolars unavailable: No module named 'analytics.describe.polars'` (`harness.load` turns ImportError into a visible skip) — not yet passing.

- [ ] **Step 3: Implement**

Append to `services/analytics/analytics/describe/_values.py` (add `import numpy as np` and `from datetime import time` at the top):

```python
def subsets(n: int, seed: int) -> np.ndarray:
    """Seeded 3-way split for the Schnabel estimate (Python implementations).
    Rust uses splitmix64(seed + row) % 3, so capture histories differ between them
    and are compared through the Schnabel estimate (within 10%)."""
    return np.random.default_rng(seed).integers(0, 3, n)


def frequency_summary(count, first, mask, n_rows: int, n_null: int) -> dict:
    """Group A metrics from a frequency table of distinct non-null values: their
    counts, first-occurrence row indices and OR-ed split masks (numpy arrays)."""
    count = np.asarray(count, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    cats = np.append(count, n_null) if n_null else count
    p = cats / n_rows if n_rows else cats.astype(float)
    top = np.lexsort((first, -count))[:5]  # count desc, then first occurrence asc
    return {
        "n_unique": len(count),
        "entropy": float(-(p * np.log2(p)).sum()) + 0.0 if n_rows else float("nan"),
        "f1": int((count == 1).sum()),
        "f2": int((count == 2).sum()),
        "top5_idx": first[top].tolist(),
        "top5_count": count[top].tolist(),
        "capture_history": np.bincount(np.asarray(mask, dtype=np.int64), minlength=8)[1:8].tolist(),
    }


def frac_digits(reprs: pl.Series) -> int | None:
    """Max decimal places over shortest round-trip float strings ("0.1", "1e-7",
    "1.5e+20", "3.0"): max(0, fraction digits without trailing zeros − exponent).
    None when there are none."""
    if reprs.len() == 0:
        return None
    parts = reprs.str.extract_groups(r"^-?[0-9]+(?:\.([0-9]*?)0*)?(?:[eE]\+?(-?[0-9]+))?$")
    frac = parts.struct.field("1").str.len_bytes().fill_null(0).cast(pl.Int64)
    exp = parts.struct.field("2").cast(pl.Int64).fill_null(0)
    return int((frac - exp).clip(lower_bound=0).max())


def n_midnight(s: pl.Series) -> int | None:
    """Datetime values at exactly 00:00:00 local time (column time zone, else naive)."""
    if not isinstance(s.dtype, pl.Datetime):
        return None
    return int((s.dt.time() == time(0)).sum())
```

`services/analytics/analytics/describe/polars.py`:

```python
"""DescribePolars ★ — the accuracy reference: Polars expressions per column
(group_by frequency table, sort for extremes, str.* for the scanners) and pyarrow
for sizes. Columns are processed one after another."""

from __future__ import annotations

import numpy as np
import polars as pl

from analytics.describe._sizes import column_sizes
from analytics.describe._values import FLOATS, STRING_LIKE, flatten, frac_digits, frequency_summary, n_midnight, subsets
from analytics.describe.base import GROUP_B, GROUP_C, VALUE_METRICS, Describe


class DescribePolars(Describe):
    """Reference: one Polars pipeline per column; pyarrow IPC writer for sizes."""

    def _compute(self, frames, combos):
        rows = [self._row(frames[n][c]) for ((n, c),) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})

    def _row(self, s: pl.Series) -> dict:
        row = {
            "n_rows": s.len(), "n_null": s.null_count(), **profile(s, self.seed),
            "n_midnight": n_midnight(s), **column_sizes(s, self.zstd_level),
        }
        inner = flatten(s) if isinstance(s.dtype, (pl.List, pl.Array)) else None
        row["inner_n_values"] = None if inner is None else inner.len()
        row["inner_n_null"] = None if inner is None else inner.null_count()
        inner_profile = dict.fromkeys(VALUE_METRICS) if inner is None else profile(inner, self.seed)
        return row | {f"inner_{k}": v for k, v in inner_profile.items()}


def profile(s: pl.Series, seed: int) -> dict:
    """Every VALUE_METRICS entry for one series (outer column or flattened inner values)."""
    freq = frequencies(s, seed)
    summary = frequency_summary(freq["count"].to_numpy(), freq["first"].to_numpy(), freq["mask"].to_numpy(), s.len(), s.null_count())
    return {**summary, **extremes(s, freq), **lengths(s), **float_stats(s), **string_stats(s)}


def frequencies(s: pl.Series, seed: int) -> pl.DataFrame:
    """Distinct non-null values with count, first row index and OR of 1 << split subset.
    Polars groups -0.0 with 0.0 and all NaNs together."""
    return (
        pl.DataFrame({"v": s, "i": np.arange(s.len(), dtype=np.uint64), "m": (1 << subsets(s.len(), seed)).astype(np.uint8)})
        .filter(pl.col("v").is_not_null())
        .group_by("v")
        .agg(pl.len().cast(pl.UInt64).alias("count"), pl.col("i").min().alias("first"), pl.col("m").bitwise_or().alias("mask"))
    )


def extremes(s: pl.Series, freq: pl.DataFrame) -> dict:
    """First occurrence of the min and max: sort the distinct values (Polars order;
    NaN excluded) and take their first-occurrence indices."""
    values = freq.select("v", "first")
    if isinstance(s.dtype, FLOATS):
        values = values.filter(pl.col("v").is_not_nan())
    if values.height == 0:
        return {"argmin": None, "argmax": None}
    values = values.sort("v")
    return {"argmin": values["first"][0], "argmax": values["first"][-1]}


def lengths(s: pl.Series) -> dict:
    dtype = s.dtype
    if isinstance(dtype, STRING_LIKE):
        lens = s.cast(pl.String).str.len_bytes()
    elif dtype == pl.Binary:
        lens = s.bin.size()
    elif isinstance(dtype, pl.List):
        lens = s.list.len()
    elif isinstance(dtype, pl.Array):
        lens = s.arr.len()
    else:
        return {"min_len": None, "max_len": None}
    return {"min_len": lens.min(), "max_len": lens.max()}


def float_stats(s: pl.Series) -> dict:
    if not isinstance(s.dtype, FLOATS):
        return dict.fromkeys(GROUP_B)
    v = s.drop_nulls()
    finite = v.filter(v.is_finite())
    return {
        "n_nan": int(v.is_nan().sum()),
        "n_inf": int(v.is_infinite().sum()),
        "n_fractional": int((finite != finite.floor()).sum()),
        # Shortest round-trip strings in the column's own width (f32 digits for Float32).
        "max_frac_digits": frac_digits(finite.unique().cast(pl.String)),
        "n_f32_inexact": None if s.dtype == pl.Float32 else int((finite.cast(pl.Float32).cast(pl.Float64) != finite).sum()),
    }


def string_stats(s: pl.Series) -> dict:
    """Group C — the numeric-string and ISO scanners (implemented in Task 6)."""
    return dict.fromkeys(GROUP_C)
```

Update `services/analytics/analytics/describe/__init__.py`: no change needed (`IMPLEMENTATIONS = ("DescribePolars",)` and the lazy `.polars` entry are already there).

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: PASS (agreement tests are skipped as an empty parameter set — only the reference exists).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe tests/test_describe.py
git commit -m "feat: DescribePolars reference - frequencies, extremes, floats, sizes, inner values

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: DescribePolars — numeric-string and ISO scanners

**Files:**
- Modify: `services/analytics/analytics/describe/_values.py`, `services/analytics/analytics/describe/polars.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Produces: `_values.NUMERIC, NUMERIC_INT, LEADING_ZERO, ISO_DATE, ISO_TIME, ISO_DATETIME, ISO_DATETIME_TZ, ISO_FRACTION, ISO_OFFSET, ISO_MIDNIGHT, INT_DIGITS, FRAC_DIGITS` (regex strings); `polars.string_stats(s) -> dict` (GROUP_C keys).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_describe.py`:

```python
NUMERIC_CASES = [
    pytest.param(["5.", ".5", "1.2.3", "+5", "1e5", " 5", "5 ", "\u0663"], dict(n_numeric=0, n_numeric_int=0, n_leading_zero=0, numeric_max_int_digits=None), id="rejected"),
    pytest.param(["007", "-012"], dict(n_numeric=2, n_numeric_int=2, n_leading_zero=2, numeric_int_min=-12, numeric_int_max=7), id="leading_zero"),
    pytest.param(["0", "-0"], dict(n_numeric=2, n_numeric_int=2, n_leading_zero=0, numeric_max_int_digits=0), id="zero_is_not_leading"),
    pytest.param(["007.50"], dict(n_numeric=1, n_numeric_int=0, n_leading_zero=0, numeric_max_int_digits=1, numeric_max_frac_digits=1), id="decimal_ignores_zeros"),
    pytest.param(["12", "-7", "300", "0.25"], dict(numeric_int_min=-7, numeric_int_max=300, numeric_max_int_digits=3, numeric_max_frac_digits=2), id="ranges"),
    pytest.param(["1" + "0" * 38], dict(n_numeric_int=1, numeric_int_min=None, numeric_int_max=None, numeric_max_int_digits=39), id="39_digits"),
]


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("values, expected", NUMERIC_CASES)
def test_numeric_scanner(impl, values, expected):
    r = profile(impl, pl.Series("x", values))
    assert {k: r[k] for k in expected} == expected


ISO_CASES = [
    pytest.param(["2024-02-29", "2023-02-29", "2024-13-01", "2024-1-05"], dict(n_iso_date=1, iso_max_frac_digits=None), id="calendar"),
    pytest.param(["23:59", "24:00", "23:59:60", "10:00:00.123456789", "10:00:00.1234567890", "10:00:00."], dict(n_iso_time=2, iso_max_frac_digits=9), id="times"),
    pytest.param(["2024-01-05T10:00", "2024-01-05 10:00:00", "2024-01-05t10:00"], dict(n_iso_datetime=2, n_iso_datetime_tz=0, iso_max_frac_digits=0), id="separator"),
    pytest.param(
        ["2024-01-05T10:00:00Z", "2024-01-05 10:00:00+00:00", "2024-01-05T10:00-00:00", "2024-01-05T10:00+02:00", "2024-01-05T10:00+0200"],
        dict(n_iso_datetime_tz=4, iso_n_offsets=2), id="offsets",
    ),
    pytest.param(["2024-01-05T00:00:00.000", "2024-01-05T00:00", "2024-01-05T00:00:01", "2024-01-05T00:00Z"], dict(n_iso_datetime=3, n_iso_datetime_tz=1, iso_n_midnight=3), id="midnight"),
]


@pytest.mark.parametrize("impl", ALL)
@pytest.mark.parametrize("values, expected", ISO_CASES)
def test_iso_scanner(impl, values, expected):
    r = profile(impl, pl.Series("x", values))
    assert {k: r[k] for k in expected} == expected


@pytest.mark.parametrize("impl", ALL)
def test_scanners_on_categorical_and_non_strings(impl):
    c = profile(impl, pl.Series("x", ["12", "007", "x"], dtype=pl.Categorical))
    assert (c["n_numeric"], c["n_leading_zero"]) == (2, 1)
    assert profile(impl, pl.Series("x", [1, 2]))["n_numeric"] is None


@pytest.mark.parametrize("impl", ALL)
def test_adversarial_long_strings(impl):
    s = pl.Series("x", ["0" * 10**6 + "." + "0" * 10**6 + ".", "0" * 10**6, "2024-01-05T" + "0" * 10**6, "9" * 39])
    r = profile(impl, s)
    assert (r["n_numeric"], r["n_numeric_int"], r["n_leading_zero"]) == (2, 2, 1)
    assert r["numeric_int_min"] is None  # "9"*39 has more than 38 significant digits
    assert r["n_iso_date"] == r["n_iso_datetime"] == r["n_iso_datetime_tz"] == 0


def _non_null(frame: pl.DataFrame, col: str) -> int:
    return frame[col].len() - frame[col].null_count()


@pytest.mark.parametrize("impl", ALL)
def test_stringified_describe_mixed(impl):
    source = describe_mixed(500)
    text = stringified(source)
    out = {r["col_a"]: r for r in run(load(impl), {"s": text}).iter_rows(named=True)}
    for c in ("i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "codes"):
        assert out[c]["n_numeric_int"] == _non_null(source, c), c
        assert out[c]["n_leading_zero"] == 0, c
    assert out["date"]["n_iso_date"] == _non_null(source, "date")
    assert out["dt_naive"]["n_iso_datetime"] == _non_null(source, "dt_naive")  # "2024-01-05 10:00:00.000000"
    assert out["dt_tz"]["n_iso_datetime_tz"] == _non_null(source, "dt_tz")  # "…+00:00" / "…+01:00"
    assert out["dt_tz"]["iso_n_offsets"] == 2  # GMT and BST across 2024
    assert out["time"]["n_iso_time"] == _non_null(source, "time")
    assert out["f64_price"]["n_numeric"] == _non_null(source, "f64_price")
    assert out["dec"]["n_numeric"] == _non_null(source, "dec") and out["dec"]["numeric_max_frac_digits"] <= 2
    # Polars writes "1e-7", "1.5e+20", "inf", "NaN": exponent and special forms are deliberately not numeric.
    assert out["f64"]["n_numeric"] == int(text["f64"].is_in(["0.0", "-0.0", "0.1"]).sum())
    assert out["bool"]["n_numeric"] == 0 and out["bool"]["n_iso_date"] == 0  # "true" / "false"
    lst = out["list_i64"]
    assert lst["inner_n_numeric_int"] == lst["inner_n_values"] - lst["inner_n_null"]


@pytest.mark.parametrize("impl", ALL)
def test_stringified_large_dataset_columns(impl):
    large = pl.read_ipc(LARGE).head(5_000)
    cols = [c for c, dt in large.schema.items() if dt in (pl.Int32, pl.Int64, pl.Date, pl.Float64)][:25]
    source = large.select(cols)
    out = {r["col_a"]: r for r in run(load(impl), {"s": stringified(source)}).iter_rows(named=True)}
    for c in cols:
        n = _non_null(source, c)
        if source[c].dtype in (pl.Int32, pl.Int64):
            assert out[c]["n_numeric_int"] == n, c
        elif source[c].dtype == pl.Date:
            assert out[c]["n_iso_date"] == n, c
        else:
            assert out[c]["n_numeric"] <= n, c
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -k "scanner or stringified or adversarial" -v`
Expected: FAIL — group C metrics are all `None`.

- [ ] **Step 3: Implement**

Append to `services/analytics/analytics/describe/_values.py`:

```python
# ── string grammar ───────────────────────────────────────────────────────────
# Anchored and ASCII-only. Polars' str.* and DataFusion's regexp_* run these on the
# Rust `regex` engine (finite automata, no backtracking), so run time is linear in
# the input for any string — hostile input cannot trigger catastrophic
# backtracking. Never evaluate them with Python's `re`. The Rust kernel
# (src/describe/patterns.rs) implements the same grammar with byte scanners.
#
# Leading zeros: an integer-looking string with a leading zero ("007") must stay a
# String; a value with a decimal point is judged on numeric equality only. See
# the rationale in src/describe/patterns.rs.

NUMERIC = r"^-?[0-9]+(\.[0-9]+)?$"
NUMERIC_INT = r"^-?[0-9]+$"
LEADING_ZERO = r"^-?0[0-9]+$"
INT_DIGITS = r"^-?0*([0-9]*)"          # significant integer-part digits
FRAC_DIGITS = r"\.([0-9]*?)0*$"        # fraction digits without trailing zeros

_DATE = r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"  # days-per-month checked by parsing
_TIME = r"([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9](\.[0-9]{1,9})?)?"
_OFFSET = r"(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"
ISO_DATE = f"^{_DATE}$"
ISO_TIME = f"^{_TIME}$"
ISO_DATETIME = f"^{_DATE}[T ]{_TIME}$"
ISO_DATETIME_TZ = f"^{_DATE}[T ]{_TIME}{_OFFSET}$"
ISO_FRACTION = r":[0-5][0-9]\.([0-9]+)"        # fractional-second digits as written
ISO_OFFSET = r"(Z|[+-][0-9]{2}:[0-9]{2})$"
ISO_MIDNIGHT = r"[T ]00:00(:00(\.0+)?)?(Z|[+-][0-9]{2}:[0-9]{2})?$"
```

In `services/analytics/analytics/describe/polars.py`, extend the `_values` import with `ISO_DATE, ISO_DATETIME, ISO_DATETIME_TZ, ISO_FRACTION, ISO_MIDNIGHT, ISO_OFFSET, ISO_TIME, INT_DIGITS, FRAC_DIGITS, LEADING_ZERO, NUMERIC, NUMERIC_INT` and replace `string_stats` with:

```python
def string_stats(s: pl.Series) -> dict:
    """Group C: numeric-string and ISO 8601 counts over non-null values."""
    if not isinstance(s.dtype, STRING_LIKE):
        return dict.fromkeys(GROUP_C)
    v = s.cast(pl.String).drop_nulls()
    numeric = v.filter(v.str.contains(NUMERIC))
    ints = v.filter(v.str.contains(NUMERIC_INT))
    int_digits = ints.str.extract(INT_DIGITS, 1).str.len_bytes()
    in_range = ints.len() > 0 and int_digits.max() <= 38
    parsed = ints.str.to_integer(dtype=pl.Int128) if in_range else None

    date_ok = v.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False).is_not_null()
    is_date = v.str.contains(ISO_DATE) & date_ok
    is_time = v.str.contains(ISO_TIME)
    is_dt = v.str.contains(ISO_DATETIME) & date_ok
    is_tz = v.str.contains(ISO_DATETIME_TZ) & date_ok
    timed = v.filter(is_time | is_dt | is_tz)
    offsets = v.filter(is_tz).str.extract(ISO_OFFSET, 1).replace({"Z": "+00:00", "-00:00": "+00:00"})
    stamped = v.filter(is_dt | is_tz)

    return {
        "n_numeric": numeric.len(),
        "n_numeric_int": ints.len(),
        "n_leading_zero": int(v.str.contains(LEADING_ZERO).sum()),
        "numeric_int_min": parsed.min() if parsed is not None else None,
        "numeric_int_max": parsed.max() if parsed is not None else None,
        "numeric_max_int_digits": numeric.str.extract(INT_DIGITS, 1).str.len_bytes().max(),
        "numeric_max_frac_digits": numeric.str.extract(FRAC_DIGITS, 1).str.len_bytes().fill_null(0).max(),
        "n_iso_date": int(is_date.sum()),
        "n_iso_time": int(is_time.sum()),
        "n_iso_datetime": int(is_dt.sum()),
        "n_iso_datetime_tz": int(is_tz.sum()),
        "iso_max_frac_digits": timed.str.extract(ISO_FRACTION, 1).str.len_bytes().fill_null(0).max(),
        "iso_n_offsets": offsets.n_unique(),
        "iso_n_midnight": int(stamped.str.contains(ISO_MIDNIGHT).sum()),
    }
```

(`Series.max()` of an empty series returns `None`, which is the required value when no string qualifies.)

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe tests/test_describe.py
git commit -m "feat: DescribePolars numeric-string and ISO 8601 scanners; stringified tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Rust — Struct/Binary encoding and the frequency kernel

**Files:**
- Modify: `services/analytics/Cargo.toml`, `services/analytics/src/shared.rs`, `services/analytics/src/lib.rs`
- Create: `services/analytics/src/describe/mod.rs` (module declarations only for now), `services/analytics/src/describe/frequency.rs`

**Interfaces:**
- Consumes: `crate::shared::{encode_series, EncodedColumn}`.
- Produces: `describe::frequency::{frequencies(col: &EncodedColumn, seed: u64) -> Frequencies, subset(seed: u64, row: u64) -> u8, CHUNK: usize}`; `Frequencies { n_unique: u64, entropy: f64, f1: u64, f2: u64, top5_idx: Vec<u64>, top5_count: Vec<u64>, capture_history: [u64; 7] }`; `encode_series` accepts `Binary` and `Struct`.

- [ ] **Step 1: Dependencies** — in `services/analytics/Cargo.toml`, change the polars line and add crates under `[dependencies]`:

```toml
polars = { version = "0.51.0", features = ["lazy", "performant", "dtype-decimal", "dtype-struct", "dtype-array", "dtype-categorical", "timezones"] }

#describe: IPC ZSTD sizes, shortest float repr, time zones, buffer casts
zstd = "0.13"
ryu = "1"
chrono = { version = "0.4", default-features = false, features = ["std"] }
chrono-tz = "0.10"
bytemuck = "1"
```

`services/analytics/src/describe/mod.rs` (this task only):

```rust
// `describe`: per-column profile (docs/superpowers/specs/2026-09-26-describe-technique-design.md).
// Entry points are added in Tasks 10–11.

pub(crate) mod frequency;
```

In `services/analytics/src/lib.rs` add `mod describe;` after `mod gcd;`.

- [ ] **Step 2: Write the failing tests** — create `services/analytics/src/describe/frequency.rs` containing only the tests module and stubs that do not compile yet:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::encode_series;
    use polars::prelude::*;

    fn freq(s: Series) -> Frequencies {
        frequencies(&encode_series(&s).unwrap(), 0)
    }

    #[test]
    fn counts_entropy_and_top5() {
        let f = freq(Series::new("a".into(), &[Some("a"), Some("a"), Some("b"), None]));
        assert_eq!((f.n_unique, f.f1, f.f2), (2, 1, 1));
        assert!((f.entropy - 1.5).abs() < 1e-12);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 2], vec![2, 1]));
    }

    #[test]
    fn ties_break_by_first_occurrence() {
        let f = freq(Series::new("a".into(), &[3i64, 1, 2, 1, 2, 3, 4, 5, 6]));
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, 1, 2, 6, 7], vec![2, 2, 2, 1, 1]));
    }

    #[test]
    fn merges_across_parallel_chunks() {
        let n = 3 * CHUNK + 5;
        // value 9 first appears in the third chunk and again in the fourth; every other value is 0.
        let mut v = vec![0i64; n];
        v[2 * CHUNK + 1] = 9;
        v[3 * CHUNK + 2] = 9;
        let f = freq(Series::new("a".into(), v));
        assert_eq!(f.n_unique, 2);
        assert_eq!((f.top5_idx, f.top5_count), (vec![0, (2 * CHUNK + 1) as u64], vec![(n - 2) as u64, 2]));
    }

    #[test]
    fn capture_history_sums_to_n_unique_and_fills_all_subsets() {
        let f = freq(Series::new("a".into(), (0..10_000i64).map(|i| i % 10).collect::<Vec<_>>()));
        assert_eq!(f.capture_history, [0, 0, 0, 0, 0, 0, 10]);
        let g = freq(Series::new("a".into(), (0..20_000i64).map(|i| (i * 7919) % 5_003).collect::<Vec<_>>()));
        assert_eq!(g.capture_history.iter().sum::<u64>(), g.n_unique);
    }

    #[test]
    fn subsets_are_roughly_uniform() {
        let mut counts = [0u64; 3];
        for row in 0..30_000 {
            counts[subset(0, row) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9_500..10_500).contains(&c)), "{counts:?}");
    }

    #[test]
    fn zero_rows_and_all_null() {
        let z = freq(Series::new_empty("a".into(), &DataType::Int32));
        assert!(z.entropy.is_nan());
        assert_eq!((z.n_unique, z.capture_history), (0, [0; 7]));
        let a = freq(Series::new("a".into(), &[None::<i32>, None]));
        assert_eq!((a.n_unique, a.entropy), (0, 0.0));
    }

    #[test]
    fn struct_values_hash_whole_and_binary_encodes() {
        let a = Series::new("a".into(), &[1i32, 1, 1]);
        let b = Series::new("b".into(), &[Some("x"), Some("x"), None]);
        let s = StructChunked::from_series("s".into(), 3, [a, b].iter()).unwrap().into_series();
        assert_eq!(freq(s).n_unique, 2);
        let bin = Series::new("b".into(), &[Some(b"ab".as_ref()), Some(b"ab".as_ref()), None]);
        assert_eq!(freq(bin).n_unique, 1);
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run (from `services/analytics/`): `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::frequency::`
Expected: FAIL to compile — `cannot find function frequencies`, `cannot find type Frequencies`.

- [ ] **Step 4: Implement**

In `services/analytics/src/shared.rs`, add two arms to `encode_series` before the final `_ =>` arm:

```rust
        DataType::Binary => {
            let build_hasher = FoldHashFixed::default();
            series
                .binary()?
                .iter()
                .map(|v| v.map_or((0, true), |b| (hash_one(&build_hasher, b), false)))
                .unzip()
        }
        // Struct: the whole value is one key — a hash of every field's key and
        // null-ness — so equal structs match and a null struct never aliases a
        // struct whose fields are all null (the outer validity is checked first).
        DataType::Struct(_) => {
            let build_hasher = FoldHashFixed::default();
            let ca = series.struct_()?;
            let fields = ca
                .fields_as_series()
                .iter()
                .map(encode_series)
                .collect::<PolarsResult<Vec<_>>>()?;
            let validity = ca.rechunk_validity();
            (0..ca.len())
                .map(|i| {
                    if validity.as_ref().is_some_and(|bm| !bm.get_bit(i)) {
                        return (0, true);
                    }
                    let mut h = build_hasher.build_hasher();
                    fields.len().hash(&mut h);
                    for f in &fields {
                        f.is_null[i].hash(&mut h);
                        f.values[i].hash(&mut h);
                    }
                    (h.finish(), false)
                })
                .unzip()
        }
```

Prepend to `services/analytics/src/describe/frequency.rs` (above the tests module):

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — value frequencies (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// One foldhash map `key → (count, first row, split-subset mask)` per 64K-row
// chunk, built in parallel and merged (counts summed, first = min, masks OR-ed).
// One O(distinct) sweep of the merged map yields n_unique, entropy (null as its
// own category), f1/f2, the capture history and the top 5 (count desc, then
// first occurrence asc). Keys come from `encode_series` (floats canonicalised;
// strings, nested and struct values hashed — collisions ~6e-11 per pair at 50K
// rows, accepted as documented in CLAUDE.md).

use crate::shared::EncodedColumn;
use foldhash::fast::FixedState;
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Entry {
    count: u64,
    first: u64,
    mask: u8,
}

type Map = HashMap<u64, Entry, FixedState>;

pub(crate) struct Frequencies {
    pub n_unique: u64,
    pub entropy: f64,
    pub f1: u64,
    pub f2: u64,
    pub top5_idx: Vec<u64>,
    pub top5_count: Vec<u64>,
    pub capture_history: [u64; 7],
}

/// Split subset (0, 1 or 2) of `row`: the SplitMix64 finaliser of `seed + row`.
/// Seeded and language-independent; the Python implementations use numpy's
/// generator instead, which is why capture histories are compared only through
/// the Schnabel estimate.
#[inline]
pub(crate) fn subset(seed: u64, row: u64) -> u8 {
    let mut z = seed.wrapping_add(row).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % 3) as u8
}

fn count_chunk(values: &[u64], is_null: &[bool], start: usize, seed: u64) -> Map {
    let mut map = Map::with_capacity_and_hasher(1024, FixedState::default());
    for (j, (&key, &null)) in values.iter().zip(is_null).enumerate() {
        if null {
            continue;
        }
        let row = (start + j) as u64;
        let e = map.entry(key).or_insert(Entry { count: 0, first: row, mask: 0 });
        e.count += 1;
        e.mask |= 1 << subset(seed, row);
    }
    map
}

fn merge(mut a: Map, mut b: Map) -> Map {
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    for (key, e) in b {
        a.entry(key)
            .and_modify(|x| {
                x.count += e.count;
                x.first = x.first.min(e.first);
                x.mask |= e.mask;
            })
            .or_insert(e);
    }
    a
}

pub(crate) fn frequencies(col: &EncodedColumn, seed: u64) -> Frequencies {
    let n = col.len();
    let map = col
        .values
        .par_chunks(CHUNK)
        .zip(col.is_null.par_chunks(CHUNK))
        .enumerate()
        .map(|(i, (values, nulls))| count_chunk(values, nulls, i * CHUNK, seed))
        .reduce(|| Map::with_hasher(FixedState::default()), merge);

    let n_null = col.is_null.iter().filter(|&&x| x).count();
    let nf = n as f64;
    let mut entropy = if n == 0 { f64::NAN } else { 0.0 };
    let (mut f1, mut f2, mut history) = (0u64, 0u64, [0u64; 7]);
    let n_unique = map.len() as u64;
    let mut entries: Vec<Entry> = Vec::with_capacity(map.len());
    for e in map.into_values() {
        let p = e.count as f64 / nf;
        entropy -= p * p.log2();
        f1 += (e.count == 1) as u64;
        f2 += (e.count == 2) as u64;
        history[e.mask as usize - 1] += 1;
        entries.push(e);
    }
    if n_null > 0 {
        let p = n_null as f64 / nf;
        entropy -= p * p.log2();
    }
    let order = |a: &Entry, b: &Entry| b.count.cmp(&a.count).then(a.first.cmp(&b.first));
    if entries.len() > 5 {
        entries.select_nth_unstable_by(4, order);
        entries.truncate(5);
    }
    entries.sort_unstable_by(order);
    Frequencies {
        n_unique,
        entropy: entropy + 0.0, // -0.0 → 0.0 for an all-null column
        f1,
        f2,
        top5_idx: entries.iter().map(|e| e.first).collect(),
        top5_count: entries.iter().map(|e| e.count).collect(),
        capture_history: history,
    }
}
```

- [ ] **Step 5: Run to verify pass**

Run (from `services/analytics/`): `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::frequency:: && PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib`
Expected: PASS (new tests and every existing Rust test).

- [ ] **Step 6: Commit**

```bash
git add services/analytics/Cargo.toml services/analytics/Cargo.lock services/analytics/src
git commit -m "feat: describe frequency kernel; encode_series supports Binary and Struct

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Rust — numeric and ISO byte scanners

**Files:**
- Create: `services/analytics/src/describe/patterns.rs`
- Modify: `services/analytics/src/describe/mod.rs` (add `pub(crate) mod patterns;`)

**Interfaces:**
- Produces: `patterns::{scan_numeric(b: &[u8]) -> Option<Numeric>, parse_i128(b: &[u8]) -> Option<i128>, scan_iso(b: &[u8]) -> Option<Iso>, StringStats}`; `StringStats::default()`, `.add(&mut self, b: &[u8])`, `.merge(self, other) -> Self`, public fields `n_numeric, n_numeric_int, n_leading_zero: u64; int_min, int_max: Option<i128>; int_overflow: bool; max_int_digits, max_frac_digits: Option<u32>; n_iso_date, n_iso_time, n_iso_datetime, n_iso_datetime_tz: u64; iso_max_frac_digits: Option<u32>; offsets: HashSet<i32>; iso_n_midnight: u64`.

- [ ] **Step 1: Write the failing tests** — create `services/analytics/src/describe/patterns.rs` with only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn num(s: &str) -> Option<(bool, bool, u32, u32)> {
        scan_numeric(s.as_bytes()).map(|n| (n.is_int, n.leading_zero, n.int_digits, n.frac_digits))
    }

    #[test]
    fn numeric_grammar() {
        for bad in ["5.", ".5", "1.2.3", "+5", "1e5", " 5", "5 ", "", "-", "\u{0663}"] {
            assert_eq!(num(bad), None, "{bad:?}");
        }
        assert_eq!(num("007"), Some((true, true, 1, 0)));
        assert_eq!(num("-012"), Some((true, true, 2, 0)));
        assert_eq!(num("0"), Some((true, false, 0, 0)));
        assert_eq!(num("-0"), Some((true, false, 0, 0)));
        assert_eq!(num("007.50"), Some((false, false, 1, 1)));
        assert_eq!(num("0.25"), Some((false, false, 0, 2)));
    }

    #[test]
    fn i128_parse_limit() {
        assert_eq!(parse_i128(b"-012"), Some(-12));
        assert_eq!(parse_i128(format!("9{}", "0".repeat(37)).as_bytes()), Some(9 * 10i128.pow(37)));
        assert_eq!(parse_i128(format!("1{}", "0".repeat(38)).as_bytes()), None);
        assert_eq!(parse_i128(format!("{}1", "0".repeat(50)).as_bytes()), Some(1));
    }

    fn iso(s: &str) -> Option<Iso> {
        scan_iso(s.as_bytes())
    }

    #[test]
    fn iso_grammar() {
        assert_eq!(iso("2024-02-29"), Some(Iso::Date));
        for bad in ["2023-02-29", "2024-13-01", "2024-1-05", "24:00", "23:59:60", "10:00:00.1234567890", "10:00:00.", "2024-01-05t10:00", "2024-01-05T10:00+0200", "2024-01-05T10:00Zx"] {
            assert_eq!(iso(bad), None, "{bad:?}");
        }
        assert_eq!(iso("23:59"), Some(Iso::Time { frac: 0 }));
        assert_eq!(iso("10:00:00.123456789"), Some(Iso::Time { frac: 9 }));
        assert_eq!(iso("2024-01-05 10:00:00"), Some(Iso::DateTime { frac: 0, midnight: false }));
        assert_eq!(iso("2024-01-05T00:00:00.000"), Some(Iso::DateTime { frac: 3, midnight: true }));
        assert_eq!(iso("2024-01-05T00:00Z"), Some(Iso::DateTimeTz { frac: 0, midnight: true, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-00:00"), Some(Iso::DateTimeTz { frac: 0, midnight: false, offset_minutes: 0 }));
        assert_eq!(iso("2024-01-05T10:00-05:30"), Some(Iso::DateTimeTz { frac: 0, midnight: false, offset_minutes: -330 }));
    }

    #[test]
    fn stats_accumulate_and_merge() {
        let mut a = StringStats::default();
        for s in ["007", "12", "0.25", "2024-01-05T00:00Z", "2024-01-05T10:00+02:00"] {
            a.add(s.as_bytes());
        }
        let mut b = StringStats::default();
        b.add(format!("1{}", "0".repeat(38)).as_bytes());
        let m = a.merge(b);
        assert_eq!((m.n_numeric, m.n_numeric_int, m.n_leading_zero), (4, 3, 1));
        assert!(m.int_overflow);
        assert_eq!((m.max_int_digits, m.max_frac_digits), (Some(39), Some(2)));
        assert_eq!((m.n_iso_datetime_tz, m.offsets.len(), m.iso_n_midnight), (2, 2, 1));
    }

    #[test]
    fn adversarial_inputs_are_linear_and_correct() {
        let long = format!("{}.{}.", "0".repeat(1_000_000), "0".repeat(1_000_000));
        assert_eq!(num(&long), None);
        assert_eq!(iso(&format!("2024-01-05T{}", "0".repeat(1_000_000))), None);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::patterns::`
Expected: FAIL to compile — `cannot find function scan_numeric`.

- [ ] **Step 3: Implement** — prepend to `patterns.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — numeric-string and ISO 8601 scanners (group C)
// ─────────────────────────────────────────────────────────────────────────────
//
// Hand-written byte state machines: one forward pass per value, no backtracking
// and no regex, so run time is linear in the input for any string — adversarial
// input cannot trigger catastrophic backtracking. The Python implementations use
// the same grammar as anchored regexes on the Rust `regex` engine
// (analytics/describe/_values.py).
//
// Numeric grammar: -?[0-9]+(\.[0-9]+)?   (ASCII digits only)
// ISO grammar:     date     YYYY-MM-DD (calendar-valid, Gregorian leap years)
//                  time     HH:MM[:SS[.f{1,9}]]   (00–23, 00–59, 00–59; no leap second)
//                  datetime date ('T' | ' ') time
//                  offset   'Z' | ±HH:MM          (uppercase T and Z only)
//
// Leading-zero rule:
// An integer-looking string with a leading zero ("007") must stay a String:
// identifiers such as UUID fragments, account numbers or zip codes can be all
// digits with significant leading zeros, and casting to an integer would lose
// them. A value with a single decimal point ("007.50") is unlikely to be an
// identifier, so only numeric equivalence matters for it — differing leading or
// trailing zeros are acceptable.

use std::collections::HashSet;

pub(crate) struct Numeric {
    pub is_int: bool,
    /// Integer-looking with a leading zero ("007", "-012"; not "0" or "-0").
    pub leading_zero: bool,
    /// Significant integer-part digits (leading zeros ignored).
    pub int_digits: u32,
    /// Fraction digits with trailing zeros removed.
    pub frac_digits: u32,
}

pub(crate) fn scan_numeric(b: &[u8]) -> Option<Numeric> {
    let body = b.strip_prefix(b"-").unwrap_or(b);
    let int_len = body.iter().take_while(|c| c.is_ascii_digit()).count();
    if int_len == 0 {
        return None;
    }
    let (int, rest) = body.split_at(int_len);
    let frac = match rest {
        [] => None,
        [b'.', frac @ ..] if !frac.is_empty() && frac.iter().all(u8::is_ascii_digit) => Some(frac),
        _ => return None,
    };
    let significant = int.iter().position(|&c| c != b'0').map_or(0, |p| int.len() - p);
    Some(Numeric {
        is_int: frac.is_none(),
        leading_zero: frac.is_none() && int.len() > 1 && int[0] == b'0',
        int_digits: significant as u32,
        frac_digits: frac.map_or(0, |f| f.iter().rposition(|&c| c != b'0').map_or(0, |p| p + 1)) as u32,
    })
}

/// Value of an integer-looking string with at most 38 significant digits
/// (every such value fits i128); None beyond that.
pub(crate) fn parse_i128(b: &[u8]) -> Option<i128> {
    let (neg, digits) = match b.strip_prefix(b"-") {
        Some(d) => (true, d),
        None => (false, b),
    };
    let digits = &digits[digits.iter().position(|&c| c != b'0').unwrap_or(digits.len())..];
    if digits.len() > 38 {
        return None;
    }
    let v = digits.iter().fold(0i128, |acc, &c| acc * 10 + (c - b'0') as i128);
    Some(if neg { -v } else { v })
}

#[derive(Debug, PartialEq)]
pub(crate) enum Iso {
    Date,
    Time { frac: u32 },
    DateTime { frac: u32, midnight: bool },
    DateTimeTz { frac: u32, midnight: bool, offset_minutes: i32 },
}

#[inline]
fn two(b: &[u8], i: usize) -> Option<u32> {
    let (x, y) = (*b.get(i)?, *b.get(i + 1)?);
    (x.is_ascii_digit() && y.is_ascii_digit()).then(|| ((x - b'0') * 10 + (y - b'0')) as u32)
}

fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    }
}

/// `YYYY-MM-DD` at the start of `b`; Some(10) when present and calendar-valid.
fn date(b: &[u8]) -> Option<usize> {
    let y = two(b, 0)? * 100 + two(b, 2)?;
    if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') {
        return None;
    }
    let (m, d) = (two(b, 5)?, two(b, 8)?);
    (1..=days_in_month(y, m)).contains(&d).then_some(10)
}

/// `HH:MM[:SS[.f{1,9}]]` from `i`: (end, fraction digits as written, all-zero time).
fn time(b: &[u8], i: usize) -> Option<(usize, u32, bool)> {
    let (h, m) = (two(b, i)?, two(b, i + 3)?);
    if b.get(i + 2) != Some(&b':') || h > 23 || m > 59 {
        return None;
    }
    let (mut end, mut frac, mut zero) = (i + 5, 0, h == 0 && m == 0);
    if b.get(end) == Some(&b':') {
        let s = two(b, end + 1)?;
        if s > 59 {
            return None;
        }
        zero &= s == 0;
        end += 3;
        if b.get(end) == Some(&b'.') {
            let digits = b[end + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            if !(1..=9).contains(&digits) {
                return None;
            }
            zero &= b[end + 1..end + 1 + digits].iter().all(|&c| c == b'0');
            frac = digits as u32;
            end += 1 + digits;
        }
    }
    Some((end, frac, zero))
}

/// `Z` or `±HH:MM` from `i` to the end of `b`, in minutes east of UTC.
fn offset(b: &[u8], i: usize) -> Option<i32> {
    match *b.get(i)? {
        b'Z' => (b.len() == i + 1).then_some(0),
        sign @ (b'+' | b'-') => {
            let (h, m) = (two(b, i + 1)?, two(b, i + 4)?);
            let ok = b.get(i + 3) == Some(&b':') && h <= 23 && m <= 59 && b.len() == i + 6;
            ok.then(|| (if sign == b'-' { -1 } else { 1 }) * (h * 60 + m) as i32)
        }
        _ => None,
    }
}

pub(crate) fn scan_iso(b: &[u8]) -> Option<Iso> {
    if let Some(d) = date(b) {
        if b.len() == d {
            return Some(Iso::Date);
        }
        if !matches!(b[d], b'T' | b' ') {
            return None;
        }
        let (end, frac, midnight) = time(b, d + 1)?;
        if end == b.len() {
            return Some(Iso::DateTime { frac, midnight });
        }
        return offset(b, end).map(|offset_minutes| Iso::DateTimeTz { frac, midnight, offset_minutes });
    }
    let (end, frac, _) = time(b, 0)?;
    (end == b.len()).then_some(Iso::Time { frac })
}

/// Group C accumulator over the non-null string values of one column.
#[derive(Default)]
pub(crate) struct StringStats {
    pub n_numeric: u64,
    pub n_numeric_int: u64,
    pub n_leading_zero: u64,
    pub int_min: Option<i128>,
    pub int_max: Option<i128>,
    /// Some integer-looking value has more than 38 significant digits → min/max null.
    pub int_overflow: bool,
    pub max_int_digits: Option<u32>,
    pub max_frac_digits: Option<u32>,
    pub n_iso_date: u64,
    pub n_iso_time: u64,
    pub n_iso_datetime: u64,
    pub n_iso_datetime_tz: u64,
    pub iso_max_frac_digits: Option<u32>,
    pub offsets: HashSet<i32>,
    pub iso_n_midnight: u64,
}

fn opt_min<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl StringStats {
    pub fn add(&mut self, b: &[u8]) {
        if let Some(n) = scan_numeric(b) {
            self.n_numeric += 1;
            self.max_int_digits = self.max_int_digits.max(Some(n.int_digits));
            self.max_frac_digits = self.max_frac_digits.max(Some(n.frac_digits));
            if n.is_int {
                self.n_numeric_int += 1;
                self.n_leading_zero += n.leading_zero as u64;
                match parse_i128(b) {
                    Some(v) => {
                        self.int_min = opt_min(self.int_min, Some(v));
                        self.int_max = self.int_max.max(Some(v));
                    }
                    None => self.int_overflow = true,
                }
            }
            return; // a numeric string is never an ISO value
        }
        match scan_iso(b) {
            Some(Iso::Date) => self.n_iso_date += 1,
            Some(Iso::Time { frac }) => {
                self.n_iso_time += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
            }
            Some(Iso::DateTime { frac, midnight }) => {
                self.n_iso_datetime += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
                self.iso_n_midnight += midnight as u64;
            }
            Some(Iso::DateTimeTz { frac, midnight, offset_minutes }) => {
                self.n_iso_datetime_tz += 1;
                self.iso_max_frac_digits = self.iso_max_frac_digits.max(Some(frac));
                self.iso_n_midnight += midnight as u64;
                self.offsets.insert(offset_minutes);
            }
            None => {}
        }
    }

    pub fn merge(mut self, o: Self) -> Self {
        self.n_numeric += o.n_numeric;
        self.n_numeric_int += o.n_numeric_int;
        self.n_leading_zero += o.n_leading_zero;
        self.int_min = opt_min(self.int_min, o.int_min);
        self.int_max = self.int_max.max(o.int_max);
        self.int_overflow |= o.int_overflow;
        self.max_int_digits = self.max_int_digits.max(o.max_int_digits);
        self.max_frac_digits = self.max_frac_digits.max(o.max_frac_digits);
        self.n_iso_date += o.n_iso_date;
        self.n_iso_time += o.n_iso_time;
        self.n_iso_datetime += o.n_iso_datetime;
        self.n_iso_datetime_tz += o.n_iso_datetime_tz;
        self.iso_max_frac_digits = self.iso_max_frac_digits.max(o.iso_max_frac_digits);
        self.offsets.extend(o.offsets);
        self.iso_n_midnight += o.iso_n_midnight;
        self
    }
}
```

Add `pub(crate) mod patterns;` to `src/describe/mod.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::patterns::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe
git commit -m "feat: describe numeric and ISO 8601 byte scanners (linear time, no regex)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Rust — float stats, extremes and lengths

**Files:**
- Create: `services/analytics/src/describe/numeric.rs`, `services/analytics/src/describe/range.rs`
- Modify: `services/analytics/src/describe/mod.rs` (add `pub(crate) mod numeric; pub(crate) mod range;`)

**Interfaces:**
- Produces: `numeric::{frac_digits(repr: &str) -> u32, float_stats(s: &Series) -> PolarsResult<Option<FloatStats>>, FloatStats { n_nan, n_inf, n_fractional, n_f32_inexact: u64, max_frac_digits: Option<u32> }}`; `range::{range(s: &Series) -> PolarsResult<Range>, list_ranges(ca: &ListChunked) -> Vec<Option<(usize, usize)>>, Range { argmin, argmax, min_len, max_len: Option<u64> }}`.

- [ ] **Step 1: Write the failing tests** — `numeric.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn decimal_places_of_shortest_reprs() {
        for (repr, want) in [("0.1", 1), ("1e-7", 7), ("1.5e20", 0), ("1.5e+20", 0), ("3.0", 0), ("-0.0", 0), ("123.45", 2), ("1.25e-3", 5)] {
            assert_eq!(frac_digits(repr), want, "{repr}");
        }
    }

    #[test]
    fn f64_stats() {
        let s = Series::new("x".into(), &[Some(0.1), Some(1e-7), Some(1.5e20), Some(3.0), Some(f64::INFINITY), Some(f64::NAN), None]);
        let st = float_stats(&s).unwrap().unwrap();
        assert_eq!((st.n_nan, st.n_inf, st.n_fractional, st.max_frac_digits), (1, 1, 2, Some(7)));
        assert_eq!(st.n_f32_inexact, 3); // 0.1, 1e-7, 1.5e20
    }

    #[test]
    fn f32_uses_its_own_repr_and_non_floats_are_none() {
        let st = float_stats(&Series::new("x".into(), &[0.1f32, 0.25])).unwrap().unwrap();
        assert_eq!(st.max_frac_digits, Some(2));
        assert!(float_stats(&Series::new("x".into(), &[1i32])).unwrap().is_none());
        assert_eq!(float_stats(&Series::new("x".into(), &[f64::NAN])).unwrap().unwrap().max_frac_digits, None);
    }
}
```

`range.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn r(s: Series) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
        let x = range(&s).unwrap();
        (x.argmin, x.argmax, x.min_len, x.max_len)
    }

    #[test]
    fn first_occurrence_extremes() {
        assert_eq!(r(Series::new("x".into(), &[5i64, 1, 3, 1, 5])), (Some(1), Some(0), None, None));
        assert_eq!(r(Series::new("x".into(), &[0.0f64, -0.0, f64::NAN, 1.5])), (Some(0), Some(3), None, None));
        assert_eq!(r(Series::new("x".into(), &[None::<i32>, None])), (None, None, None, None));
    }

    #[test]
    fn strings_bytes_and_lengths() {
        assert_eq!(r(Series::new("x".into(), &[Some("ab"), Some(""), None, Some("héllo")])), (Some(1), Some(3), Some(0), Some(6)));
    }

    #[test]
    fn lists_use_polars_sort_order() {
        let s = Series::new("x".into(), [Some(Series::new("".into(), &[1i64, 5])), Some(Series::new("".into(), &[2i64, 1])), None, Some(Series::new_empty("".into(), &DataType::Int64))]);
        assert_eq!(r(s), (Some(3), Some(1), Some(0), Some(2)));
    }
}
```

Enum and Categorical ordering are pinned by the Python known-answer test `test_enum_orders_by_category_and_categorical_by_string`, which runs through the plugin from Task 11 (building those dtypes in Rust tests is version-fragile).

- [ ] **Step 2: Run to verify failure**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::numeric:: describe::range::` (run the two filters as two commands)
Expected: FAIL to compile — `cannot find function float_stats` / `range`.

- [ ] **Step 3: Implement** — prepend to `numeric.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — float statistics (group B)
// ─────────────────────────────────────────────────────────────────────────────
//
// Decimal places come from the shortest round-trip representation (ryu) in the
// column's own width — f32 digits for Float32 — exponent-aware:
// max(0, fraction digits without trailing zeros − exponent).

use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use rayon::prelude::*;

const CHUNK: usize = 1 << 16;

#[derive(Default, Clone, Copy)]
pub(crate) struct FloatStats {
    pub n_nan: u64,
    pub n_inf: u64,
    pub n_fractional: u64,
    pub max_frac_digits: Option<u32>,
    /// Finite values that change under f64 → f32 → f64 (meaningless for Float32).
    pub n_f32_inexact: u64,
}

pub(crate) fn frac_digits(repr: &str) -> u32 {
    let repr = repr.trim_start_matches('-');
    let (mantissa, exp) = match repr.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (repr, 0),
    };
    let frac = mantissa.split_once('.').map_or("", |(_, f)| f).trim_end_matches('0');
    (frac.len() as i64 - exp).max(0) as u32
}

impl FloatStats {
    fn merge(self, o: Self) -> Self {
        Self {
            n_nan: self.n_nan + o.n_nan,
            n_inf: self.n_inf + o.n_inf,
            n_fractional: self.n_fractional + o.n_fractional,
            max_frac_digits: self.max_frac_digits.max(o.max_frac_digits),
            n_f32_inexact: self.n_f32_inexact + o.n_f32_inexact,
        }
    }

    fn add_f64(&mut self, x: f64, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self.max_frac_digits.max(Some(frac_digits(buf.format_finite(x))));
            self.n_f32_inexact += ((x as f32) as f64 != x) as u64;
        }
    }

    fn add_f32(&mut self, x: f32, buf: &mut ryu::Buffer) {
        if x.is_nan() {
            self.n_nan += 1;
        } else if x.is_infinite() {
            self.n_inf += 1;
        } else {
            self.n_fractional += (x != x.trunc()) as u64;
            self.max_frac_digits = self.max_frac_digits.max(Some(frac_digits(buf.format_finite(x))));
        }
    }
}

/// Fold `add` over the valid values of one Arrow chunk, CHUNK values per task.
fn fold<T: Copy + Sync>(values: &[T], validity: Option<&Bitmap>, add: impl Fn(&mut FloatStats, T, &mut ryu::Buffer) + Sync) -> FloatStats {
    let validity = validity.filter(|bm| bm.unset_bits() > 0);
    values
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            let (mut st, mut buf) = (FloatStats::default(), ryu::Buffer::new());
            for (j, &x) in chunk.iter().enumerate() {
                if validity.is_some_and(|bm| !bm.get_bit(i * CHUNK + j)) {
                    continue;
                }
                add(&mut st, x, &mut buf);
            }
            st
        })
        .reduce(FloatStats::default, FloatStats::merge)
}

pub(crate) fn float_stats(s: &Series) -> PolarsResult<Option<FloatStats>> {
    Ok(Some(match s.dtype() {
        DataType::Float64 => s
            .f64()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f64))
            .fold(FloatStats::default(), FloatStats::merge),
        DataType::Float32 => s
            .f32()?
            .downcast_iter()
            .map(|arr| fold(arr.values().as_slice(), arr.validity(), FloatStats::add_f32))
            .fold(FloatStats::default(), FloatStats::merge),
        _ => return Ok(None),
    }))
}
```

Prepend to `range.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — first-occurrence extremes and value lengths (group A)
// ─────────────────────────────────────────────────────────────────────────────
//
// Ordering matches Polars `sort()`: integers, Decimal and temporals by physical
// value; floats numerically with NaN excluded (-0.0 ties 0.0); strings and
// binary by bytes; Categorical by string value; Enum by category order (its
// physical code); List, Array and Struct by Polars' row encoding — the encoding
// its sort uses. Ties keep the lowest row index.

use polars::chunked_array::ops::row_encode::_get_rows_encoded_arr;
use polars::prelude::*;

pub(crate) struct Range {
    pub argmin: Option<u64>,
    pub argmax: Option<u64>,
    pub min_len: Option<u64>,
    pub max_len: Option<u64>,
}

fn lt<T: PartialOrd>(a: T, b: T) -> bool {
    a < b
}

fn extremes<T: Copy>(values: impl Iterator<Item = Option<T>>, lt: impl Fn(T, T) -> bool) -> (Option<u64>, Option<u64>) {
    let (mut lo, mut hi): (Option<(u64, T)>, Option<(u64, T)>) = (None, None);
    for (i, v) in values.enumerate() {
        let Some(v) = v else { continue };
        if lo.is_none_or(|(_, m)| lt(v, m)) {
            lo = Some((i as u64, v));
        }
        if hi.is_none_or(|(_, m)| lt(m, v)) {
            hi = Some((i as u64, v));
        }
    }
    (lo.map(|x| x.0), hi.map(|x| x.0))
}

fn arg_extremes(s: &Series) -> PolarsResult<(Option<u64>, Option<u64>)> {
    Ok(match s.dtype() {
        DataType::Float32 => extremes(s.f32()?.iter().map(|v| v.filter(|x| !x.is_nan())), lt),
        DataType::Float64 => extremes(s.f64()?.iter().map(|v| v.filter(|x| !x.is_nan())), lt),
        DataType::String => extremes(s.str()?.iter().map(|v| v.map(str::as_bytes)), lt),
        DataType::Binary => extremes(s.binary()?.iter(), lt),
        DataType::Categorical(_, _) => return arg_extremes(&s.cast(&DataType::String)?),
        DataType::Boolean => extremes(s.bool()?.iter(), lt),
        DataType::List(_) | DataType::Array(_, _) | DataType::Struct(_) => {
            let rows = _get_rows_encoded_arr(&[s.clone().into_column()], &[false], &[false])?;
            let valid = s.is_not_null();
            extremes(rows.values_iter().zip(valid.iter()).map(|(r, ok)| (ok == Some(true)).then_some(r)), lt)
        }
        _ => {
            let p = s.to_physical_repr();
            match p.dtype() {
                DataType::Int8 => extremes(p.i8()?.iter(), lt),
                DataType::Int16 => extremes(p.i16()?.iter(), lt),
                DataType::Int32 => extremes(p.i32()?.iter(), lt),
                DataType::Int64 => extremes(p.i64()?.iter(), lt),
                DataType::Int128 => extremes(p.i128()?.iter(), lt),
                DataType::UInt8 => extremes(p.u8()?.iter(), lt),
                DataType::UInt16 => extremes(p.u16()?.iter(), lt),
                DataType::UInt32 => extremes(p.u32()?.iter(), lt),
                DataType::UInt64 => extremes(p.u64()?.iter(), lt),
                dt => polars_bail!(ComputeError: "describe: no ordering for {dt}"),
            }
        }
    })
}

fn min_max(values: impl Iterator<Item = Option<u64>>) -> (Option<u64>, Option<u64>) {
    values.flatten().fold((None, None), |(lo, hi), v| {
        (Some(lo.map_or(v, |x: u64| x.min(v))), Some(hi.map_or(v, |x: u64| x.max(v))))
    })
}

/// Per row of a **rechunked** ListChunked: `Some((start, len))` into its child
/// values (`get_inner()`), `None` for a null list.
pub(crate) fn list_ranges(ca: &ListChunked) -> Vec<Option<(usize, usize)>> {
    ca.downcast_iter()
        .flat_map(|arr| {
            let offsets = arr.offsets().as_slice();
            (0..arr.len()).map(move |i| arr.is_valid(i).then(|| (offsets[i] as usize, (offsets[i + 1] - offsets[i]) as usize)))
        })
        .collect()
}

fn lengths(s: &Series) -> PolarsResult<(Option<u64>, Option<u64>)> {
    Ok(match s.dtype() {
        DataType::String => min_max(s.str()?.iter().map(|v| v.map(|x| x.len() as u64))),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => return lengths(&s.cast(&DataType::String)?),
        DataType::Binary => min_max(s.binary()?.iter().map(|v| v.map(|x| x.len() as u64))),
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            min_max(list_ranges(&ca).into_iter().map(|r| r.map(|(_, len)| len as u64)))
        }
        DataType::Array(_, width) => min_max(s.is_not_null().iter().map(|ok| (ok == Some(true)).then_some(*width as u64))),
        _ => (None, None),
    })
}

pub(crate) fn range(s: &Series) -> PolarsResult<Range> {
    let (argmin, argmax) = arg_extremes(s)?;
    let (min_len, max_len) = lengths(s)?;
    Ok(Range { argmin, argmax, min_len, max_len })
}
```

(`Option::is_none_or` needs Rust ≥ 1.82; if the toolchain is older use `lo.map_or(true, |(_, m)| lt(v, m))`.)

Add `pub(crate) mod numeric;` and `pub(crate) mod range;` to `src/describe/mod.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe
git commit -m "feat: describe float stats, first-occurrence extremes and lengths (Rust)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Rust — IPC sizes and the `column_sizes` plugin

**Files:**
- Create: `services/analytics/src/describe/sizes.rs`
- Modify: `services/analytics/src/describe/mod.rs` (add `pub(crate) mod sizes;`)

**Interfaces:**
- Produces: `sizes::ipc_body_bytes(arr: &dyn Array, level: Option<i32>) -> PolarsResult<u64>`; plugin `column_sizes(inputs, kwargs {zstd_level: i32})` → struct `{column: String, size_bytes, size_zstd_bytes, size_polars_bytes, size_polars_zstd_bytes: UInt64}` one row per input.

- [ ] **Step 1: Write the failing tests** — `sizes.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn arrow(s: &Series) -> ArrayRef {
        s.rechunk().to_arrow(0, CompatLevel::oldest())
    }

    #[test]
    fn primitive_framing_matches_pyarrow() {
        let s = Series::new("x".into(), (0..1_000).collect::<Vec<i32>>());
        assert_eq!(ipc_body_bytes(arrow(&s).as_ref(), None).unwrap(), 4_000);
        let zstd = ipc_body_bytes(arrow(&s).as_ref(), Some(1)).unwrap();
        assert!((1_890..=1_935).contains(&zstd), "{zstd}"); // pyarrow: 1912
        let one = Series::new("x".into(), &[1i32]);
        assert_eq!(ipc_body_bytes(arrow(&one).as_ref(), Some(1)).unwrap(), 24);
    }

    #[test]
    fn validity_only_with_nulls() {
        let s = Series::new("x".into(), (0..1_000).map(|i| (i % 3 != 0).then_some(i)).collect::<Vec<Option<i32>>>());
        assert_eq!(ipc_body_bytes(arrow(&s).as_ref(), None).unwrap(), 4_128);
        assert_eq!(ipc_body_bytes(arrow(&Series::new_empty("x".into(), &DataType::Int32)).as_ref(), None).unwrap(), 0);
    }

    #[test]
    fn large_utf8_offsets_values_validity() {
        assert_eq!(ipc_body_bytes(arrow(&Series::new("x".into(), &[Some("ab"), None])).as_ref(), None).unwrap(), 40);
        assert_eq!(ipc_body_bytes(arrow(&Series::new_empty("x".into(), &DataType::String)).as_ref(), None).unwrap(), 8);
    }

    #[test]
    fn nested_int128_sizes_are_null() {
        let s = Series::new("x".into(), [Some(Series::new("".into(), &[1i128]))]);
        assert_eq!(sizes(&s, 1).unwrap(), [None; 4]);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::sizes::`
Expected: FAIL to compile — `cannot find function ipc_body_bytes`.

- [ ] **Step 3: Implement** — prepend to `sizes.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — Arrow IPC body sizes (group E) and the `column_sizes` plugin
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors pyarrow's IPC writer (checked against pyarrow 24; the Python oracle is
// analytics/describe/_sizes.py): a column's size is the body length of the IPC
// messages that carry it (dictionary batches + record batch). Every buffer is
// padded to 8 bytes; a validity buffer is written only when the array has nulls;
// an empty buffer takes no space; List/Utf8 offsets are rebased to 0 and their
// child/values sliced to the referenced range. With ZSTD each non-empty buffer is
// an 8-byte uncompressed-length prefix plus one ZSTD frame (content size
// included) — pyarrow never falls back to raw bytes.
//
// Arrow sizes use CompatLevel::oldest() (LargeUtf8, LargeList); Polars sizes are
// `estimated_size()` and the ZSTD body of CompatLevel::newest() (view types).
// Columns nesting Int128 inside List/Array/Struct get null sizes: pyarrow cannot
// import them, so there is no oracle to agree with.

use bytemuck::Pod;
use polars::prelude::*;
use polars_arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray, ListArray, PrimitiveArray,
    StructArray, Utf8Array, Utf8ViewArray, View,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::buffer::Buffer;
use polars_arrow::datatypes::PhysicalType;
use polars_arrow::offset::{Offset, OffsetsBuffer};
use polars_arrow::types::f16;
use polars_arrow::with_match_primitive_type_full;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

struct Body {
    level: Option<i32>,
    bytes: u64,
}

impl Body {
    fn buffer(&mut self, data: &[u8]) -> PolarsResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let len = match self.level {
            None => data.len(),
            Some(level) => 8 + zstd::bulk::compress(data, level).map_err(|e| polars_err!(ComputeError: "zstd: {e}"))?.len(),
        };
        self.bytes += len.next_multiple_of(8) as u64;
        Ok(())
    }

    fn bits(&mut self, bm: &Bitmap) -> PolarsResult<()> {
        let (bytes, offset, len) = bm.as_slice();
        if offset == 0 {
            return self.buffer(&bytes[..len.div_ceil(8)]);
        }
        let packed: Bitmap = bm.iter().collect(); // re-align to bit 0, as pyarrow does
        self.bits(&packed)
    }

    fn validity(&mut self, arr: &dyn Array) -> PolarsResult<()> {
        match arr.validity() {
            Some(bm) if arr.null_count() > 0 => self.bits(bm),
            _ => Ok(()),
        }
    }

    fn slice<T: Pod>(&mut self, values: &[T]) -> PolarsResult<()> {
        self.buffer(bytemuck::cast_slice(values))
    }

    /// Writes the offsets (rebased to 0); returns the referenced values range.
    fn offsets<O: Offset + Pod>(&mut self, offsets: &OffsetsBuffer<O>) -> PolarsResult<(usize, usize)> {
        let (first, last) = (offsets.first().to_usize(), offsets.last().to_usize());
        if first == 0 {
            self.slice(offsets.as_slice())?;
        } else {
            let rebased: Vec<O> = offsets.as_slice().iter().map(|o| O::from_as_usize(o.to_usize() - first)).collect();
            self.slice(&rebased)?;
        }
        Ok((first, last))
    }

    fn var_size<O: Offset + Pod>(&mut self, arr: &dyn Array, offsets: &OffsetsBuffer<O>, values: &Buffer<u8>) -> PolarsResult<()> {
        self.validity(arr)?;
        let (first, last) = self.offsets(offsets)?;
        self.buffer(&values.as_slice()[first..last])
    }

    fn views(&mut self, arr: &dyn Array, views: &Buffer<View>, data: &[Buffer<u8>]) -> PolarsResult<()> {
        self.validity(arr)?;
        self.slice(views.as_slice())?;
        data.iter().try_for_each(|b| self.buffer(b.as_slice()))
    }

    fn list<O: Offset + Pod>(&mut self, arr: &dyn Array, a: &ListArray<O>) -> PolarsResult<()> {
        self.validity(arr)?;
        let (first, last) = self.offsets(a.offsets())?;
        self.array(a.values().sliced(first, last - first).as_ref())
    }

    fn array(&mut self, arr: &dyn Array) -> PolarsResult<()> {
        let any = arr.as_any();
        match arr.dtype().to_physical_type() {
            PhysicalType::Null => Ok(()),
            PhysicalType::Boolean => {
                self.validity(arr)?;
                self.bits(any.downcast_ref::<BooleanArray>().unwrap().values())
            }
            PhysicalType::Primitive(p) => with_match_primitive_type_full!(p, |$T| {
                self.validity(arr)?;
                self.slice(any.downcast_ref::<PrimitiveArray<$T>>().unwrap().values().as_slice())
            }),
            PhysicalType::Utf8 => { let a = any.downcast_ref::<Utf8Array<i32>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::LargeUtf8 => { let a = any.downcast_ref::<Utf8Array<i64>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::Binary => { let a = any.downcast_ref::<BinaryArray<i32>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::LargeBinary => { let a = any.downcast_ref::<BinaryArray<i64>>().unwrap(); self.var_size(arr, a.offsets(), a.values()) }
            PhysicalType::Utf8View => { let a = any.downcast_ref::<Utf8ViewArray>().unwrap(); self.views(arr, a.views(), a.data_buffers()) }
            PhysicalType::BinaryView => { let a = any.downcast_ref::<BinaryViewArray>().unwrap(); self.views(arr, a.views(), a.data_buffers()) }
            PhysicalType::List => self.list(arr, any.downcast_ref::<ListArray<i32>>().unwrap()),
            PhysicalType::LargeList => self.list(arr, any.downcast_ref::<ListArray<i64>>().unwrap()),
            PhysicalType::FixedSizeList => {
                let a = any.downcast_ref::<FixedSizeListArray>().unwrap();
                self.validity(arr)?;
                self.array(a.values().sliced(0, a.len() * a.size()).as_ref())
            }
            PhysicalType::Struct => {
                self.validity(arr)?;
                any.downcast_ref::<StructArray>().unwrap().values().iter().try_for_each(|f| self.array(f.as_ref()))
            }
            PhysicalType::Dictionary(_) => {
                macro_rules! dict {
                    ($($k:ty),*) => {$(
                        if let Some(d) = any.downcast_ref::<DictionaryArray<$k>>() {
                            self.array(d.keys())?;
                            return self.array(d.values().as_ref()); // the dictionary batch
                        }
                    )*};
                }
                dict!(u8, u16, u32, u64, i8, i16, i32, i64);
                polars_bail!(ComputeError: "describe sizes: unexpected dictionary key type")
            }
            other => polars_bail!(ComputeError: "describe sizes: unsupported Arrow type {other:?}"),
        }
    }
}

pub(crate) fn ipc_body_bytes(arr: &dyn Array, level: Option<i32>) -> PolarsResult<u64> {
    let mut body = Body { level, bytes: 0 };
    body.array(arr)?;
    Ok(body.bytes)
}

fn nests_int128(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(inner) | DataType::Array(inner, _) => **inner == DataType::Int128 || nests_int128(inner),
        DataType::Struct(fields) => fields.iter().any(|f| *f.dtype() == DataType::Int128 || nests_int128(f.dtype())),
        _ => false,
    }
}

type Sizes = [Option<u64>; 4];
const SIZE_FIELDS: [&str; 4] = ["size_bytes", "size_zstd_bytes", "size_polars_bytes", "size_polars_zstd_bytes"];

fn sizes(s: &Series, level: i32) -> PolarsResult<Sizes> {
    if nests_int128(s.dtype()) {
        return Ok([None; 4]);
    }
    let s = s.rechunk();
    let polars_bytes = Some(s.estimated_size() as u64);
    if s.n_chunks() == 0 {
        return Ok([Some(0), Some(0), polars_bytes, Some(0)]);
    }
    let classic = s.to_arrow(0, CompatLevel::oldest());
    let native = s.to_arrow(0, CompatLevel::newest());
    Ok([
        Some(ipc_body_bytes(classic.as_ref(), None)?),
        Some(ipc_body_bytes(classic.as_ref(), Some(level))?),
        polars_bytes,
        Some(ipc_body_bytes(native.as_ref(), Some(level))?),
    ])
}

fn sizes_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let mut fields = vec![Field::new("column".into(), DataType::String)];
    fields.extend(SIZE_FIELDS.iter().map(|n| Field::new((*n).into(), DataType::UInt64)));
    Ok(Field::new("column_sizes".into(), DataType::Struct(fields)))
}

#[derive(Deserialize)]
struct SizesKwargs {
    zstd_level: i32,
}

pub(crate) fn column_sizes_impl(inputs: &[Series], level: i32) -> PolarsResult<Series> {
    let rows: Vec<Sizes> = inputs.par_iter().map(|s| sizes(s, level)).collect::<PolarsResult<_>>()?;
    let mut columns = vec![StringChunked::from_iter(inputs.iter().map(|s| s.name().as_str())).into_series().with_name("column".into())];
    for (j, name) in SIZE_FIELDS.iter().enumerate() {
        columns.push(UInt64Chunked::from_iter_options((*name).into(), rows.iter().map(|r| r[j])).into_series());
    }
    Ok(StructChunked::from_series("column_sizes".into(), inputs.len(), columns.iter())?.into_series())
}

#[polars_expr(output_type_func=sizes_output_type)]
fn column_sizes(inputs: &[Series], kwargs: SizesKwargs) -> PolarsResult<Series> {
    column_sizes_impl(inputs, kwargs.zstd_level)
}
```

Add `pub(crate) mod sizes;` to `src/describe/mod.rs`. If a downcast type name differs in polars-arrow 0.51 (e.g. `Utf8ViewArray` data buffers accessor), fix the name only — the framing rules above are fixed by the pyarrow oracle.

- [ ] **Step 4: Run to verify pass**

Run: `PYO3_PYTHON=$PY PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" cargo test --lib describe::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/describe
git commit -m "feat: column_sizes plugin - Arrow IPC body sizes mirroring pyarrow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Rust — `describe_columns` plugin and DescribeRust

**Files:**
- Modify: `services/analytics/src/describe/mod.rs`, `services/analytics/analytics/_plugin.py`, `services/analytics/analytics/describe/__init__.py`
- Create: `services/analytics/analytics/describe/rust.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `frequency::frequencies`, `patterns::StringStats`, `numeric::float_stats`, `range::{range, list_ranges}`, `sizes::column_sizes` (Tasks 7–10); `Describe`, `group_by_frame` (Python).
- Produces: plugin `describe_columns(inputs, kwargs {seed: u64})` → struct `{column, n_rows, n_null, <VALUE_METRICS>, n_midnight, inner_n_values, inner_n_null, inner_<VALUE_METRICS>}`; `_plugin.describe_columns(df: pl.DataFrame, seed: int) -> pl.DataFrame`; `_plugin.column_sizes(df: pl.DataFrame, zstd_level: int) -> pl.DataFrame`; `class DescribeRust(Describe)`; `IMPLEMENTATIONS = ("DescribeRust", "DescribePolars")`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_describe.py`:

```python
@pytest.mark.parametrize("impl", ALL)
def test_multi_chunk_and_sliced_input(impl):
    parts = [pl.Series("x", ["b", None]), pl.Series("x", ["a", "c"]), pl.Series("x", ["a"])]
    s = pl.concat(parts, rechunk=False)
    assert s.n_chunks() == 3
    r = profile(impl, s)
    assert (r["n_unique"], r["argmin"], r["argmax"], r["top5_idx"]) == (3, 2, 3, [2, 0, 3])
    sliced = pl.Series("x", [[9], [1, 2], None, [3]], dtype=pl.List(pl.Int64)).slice(1, 3)
    q = profile(impl, sliced)
    assert (q["inner_n_values"], q["inner_argmin"], q["size_bytes"]) == (3, 0, profile(impl, pl.Series("x", [[1, 2], None, [3]], dtype=pl.List(pl.Int64)))["size_bytes"])


@pytest.mark.parametrize("impl", ALL)
def test_first_occurrence_across_parallel_chunks(impl):
    chunk = 1 << 16
    values = np.full(3 * chunk + 17, 5, dtype=np.int64)
    values[2 * chunk + 3] = 1   # first minimum, third chunk
    values[3 * chunk + 1] = 1   # later minimum, fourth chunk
    values[chunk + 7] = 9       # maximum, second chunk
    r = profile(impl, pl.Series("x", values))
    assert (r["argmin"], r["argmax"]) == (2 * chunk + 3, chunk + 7)
    assert (r["top5_idx"], r["top5_count"]) == ([0, 2 * chunk + 3, chunk + 7], [len(values) - 3, 2, 1])


@pytest.mark.parametrize("impl", OTHERS)
def test_categorical_and_enum_sizes_agree(impl):
    frame = describe_mixed(1_000).select("cat", "enum", "str_free")
    exact = ["col_a", "size_bytes", "size_polars_bytes"]
    assert run(load(impl), {"t": frame}).select(exact).equals(run(reference(PKG), {"t": frame}).select(exact))
```

Then replace `services/analytics/analytics/describe/__init__.py` with its final Rust-enabled form:

```python
"""Per-column profile for choosing narrower / more compressible Arrow types. See Describe."""

from analytics.base import lazy_attributes
from analytics.describe.base import Describe
from analytics.describe.rust import DescribeRust

REFERENCE = "DescribePolars"
IMPLEMENTATIONS = ("DescribeRust", "DescribePolars")

__getattr__ = lazy_attributes(__name__, {"DescribePolars": ".polars"})
__all__ = ["Describe", "DescribeRust", "DescribePolars", "REFERENCE", "IMPLEMENTATIONS"]
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -v`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'analytics.describe.rust'`.

- [ ] **Step 3: Implement**

Replace `services/analytics/src/describe/mod.rs` with:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// describe — per-column profile (plugin entry points `describe_columns`, `column_sizes`)
// ─────────────────────────────────────────────────────────────────────────────
//
// Spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md. One pass
// per column, columns in parallel (rayon); inside a column the frequency, float
// and string work runs over 64K-row chunks in parallel. Every value metric is
// computed on the column and again on its inner values (List/Array flattened one
// level, null lists skipped). Field names and order match
// analytics/describe/base.py (DescribeRust maps them by name).

pub(crate) mod frequency;
pub(crate) mod numeric;
pub(crate) mod patterns;
pub(crate) mod range;
pub(crate) mod sizes;

use crate::shared::encode_series;
use patterns::StringStats;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

type Row = Vec<AnyValue<'static>>;

/// Metrics computed on any value series, in output order (base.py VALUE_METRICS).
fn value_fields() -> Vec<(&'static str, DataType)> {
    use DataType::{Float64 as F64, Int128 as I128, UInt32 as U32, UInt64 as U64};
    let list = DataType::List(Box::new(U64));
    vec![
        ("n_unique", U64), ("entropy", F64), ("f1", U64), ("f2", U64), ("argmin", U64), ("argmax", U64),
        ("min_len", U64), ("max_len", U64), ("top5_idx", list.clone()), ("top5_count", list.clone()), ("capture_history", list),
        ("n_nan", U64), ("n_inf", U64), ("n_fractional", U64), ("max_frac_digits", U32), ("n_f32_inexact", U64),
        ("n_numeric", U64), ("n_numeric_int", U64), ("n_leading_zero", U64), ("numeric_int_min", I128), ("numeric_int_max", I128),
        ("numeric_max_int_digits", U32), ("numeric_max_frac_digits", U32),
        ("n_iso_date", U64), ("n_iso_time", U64), ("n_iso_datetime", U64), ("n_iso_datetime_tz", U64),
        ("iso_max_frac_digits", U32), ("iso_n_offsets", U64), ("iso_n_midnight", U64),
    ]
}

fn fields() -> Vec<(String, DataType)> {
    let mut f = vec![("column".to_string(), DataType::String), ("n_rows".into(), DataType::UInt64), ("n_null".into(), DataType::UInt64)];
    f.extend(value_fields().into_iter().map(|(n, d)| (n.to_string(), d)));
    f.push(("n_midnight".into(), DataType::UInt64));
    f.push(("inner_n_values".into(), DataType::UInt64));
    f.push(("inner_n_null".into(), DataType::UInt64));
    f.extend(value_fields().into_iter().map(|(n, d)| (format!("inner_{n}"), d)));
    f
}

fn describe_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = fields().into_iter().map(|(n, d)| Field::new(n.into(), d)).collect();
    Ok(Field::new("describe".into(), DataType::Struct(fields)))
}

fn u64v(v: Option<u64>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt64) }
fn u32v(v: Option<u32>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::UInt32) }
fn i128v(v: Option<i128>) -> AnyValue<'static> { v.map_or(AnyValue::Null, AnyValue::Int128) }
fn listv(v: &[u64]) -> AnyValue<'static> { AnyValue::List(Series::new(PlSmallStr::EMPTY, v)) }
fn nulls(n: usize) -> Row { vec![AnyValue::Null; n] }

fn strings(s: &Series) -> PolarsResult<Option<StringStats>> {
    let st = match s.dtype() {
        DataType::String => s.clone(),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => s.cast(&DataType::String)?,
        _ => return Ok(None),
    };
    Ok(Some(
        st.str()?
            .downcast_iter()
            .map(|arr| {
                (0..arr.len())
                    .into_par_iter()
                    .with_min_len(frequency::CHUNK)
                    .fold(StringStats::default, |mut acc, i| {
                        if arr.is_valid(i) {
                            acc.add(arr.value(i).as_bytes());
                        }
                        acc
                    })
                    .reduce(StringStats::default, StringStats::merge)
            })
            .fold(StringStats::default(), StringStats::merge),
    ))
}

/// Every value metric for one series, in `value_fields()` order.
fn profile(s: &Series, seed: u64) -> PolarsResult<Row> {
    let f = frequency::frequencies(&encode_series(s)?, seed);
    let r = range::range(s)?;
    let mut row: Row = vec![
        AnyValue::UInt64(f.n_unique), AnyValue::Float64(f.entropy), AnyValue::UInt64(f.f1), AnyValue::UInt64(f.f2),
        u64v(r.argmin), u64v(r.argmax), u64v(r.min_len), u64v(r.max_len),
        listv(&f.top5_idx), listv(&f.top5_count), listv(&f.capture_history),
    ];
    match numeric::float_stats(s)? {
        Some(fl) => row.extend([
            AnyValue::UInt64(fl.n_nan), AnyValue::UInt64(fl.n_inf), AnyValue::UInt64(fl.n_fractional), u32v(fl.max_frac_digits),
            if s.dtype() == &DataType::Float32 { AnyValue::Null } else { AnyValue::UInt64(fl.n_f32_inexact) },
        ]),
        None => row.extend(nulls(5)),
    }
    match strings(s)? {
        Some(st) => {
            let (lo, hi) = if st.int_overflow { (None, None) } else { (st.int_min, st.int_max) };
            row.extend([
                AnyValue::UInt64(st.n_numeric), AnyValue::UInt64(st.n_numeric_int), AnyValue::UInt64(st.n_leading_zero),
                i128v(lo), i128v(hi), u32v(st.max_int_digits), u32v(st.max_frac_digits),
                AnyValue::UInt64(st.n_iso_date), AnyValue::UInt64(st.n_iso_time), AnyValue::UInt64(st.n_iso_datetime),
                AnyValue::UInt64(st.n_iso_datetime_tz), u32v(st.iso_max_frac_digits),
                AnyValue::UInt64(st.offsets.len() as u64), AnyValue::UInt64(st.iso_n_midnight),
            ])
        }
        None => row.extend(nulls(14)),
    }
    Ok(row)
}

/// Datetime values at exactly 00:00:00 local time (column time zone, else naive).
fn n_midnight(s: &Series) -> PolarsResult<Option<u64>> {
    let DataType::Datetime(unit, tz) = s.dtype() else { return Ok(None) };
    let per_day: i64 = match unit {
        TimeUnit::Nanoseconds => 86_400_000_000_000,
        TimeUnit::Microseconds => 86_400_000_000,
        TimeUnit::Milliseconds => 86_400_000,
    };
    let phys = s.to_physical_repr();
    let values = phys.i64()?;
    let count = match tz {
        None => values.into_iter().flatten().filter(|v| v.rem_euclid(per_day) == 0).count(),
        Some(tz) => {
            let zone: chrono_tz::Tz = tz.as_str().parse().map_err(|_| polars_err!(ComputeError: "describe: unknown time zone {tz}"))?;
            let per_sec = per_day / 86_400;
            values
                .into_iter()
                .flatten()
                .filter(|&v| {
                    v.rem_euclid(per_sec) == 0
                        && chrono::DateTime::from_timestamp(v.div_euclid(per_sec), 0)
                            .is_some_and(|t| t.with_timezone(&zone).time() == chrono::NaiveTime::MIN)
                })
                .count()
        }
    };
    Ok(Some(count as u64))
}

/// Values one level down, skipping null lists — the same definition as the
/// Python `flatten` (drop_nulls, then explode the non-empty lists). Element i is
/// what `inner_argmin` / `inner_top5_idx` index into.
fn flatten(s: &Series) -> PolarsResult<Option<Series>> {
    let (inner, ranges): (Series, Vec<Option<(usize, usize)>>) = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list()?.rechunk();
            (ca.get_inner(), range::list_ranges(&ca))
        }
        DataType::Array(_, width) => {
            let ca = s.array()?.rechunk();
            let valid = ca.is_not_null();
            let ranges = valid.iter().enumerate().map(|(i, ok)| (ok == Some(true)).then_some((i * width, *width))).collect();
            (ca.get_inner(), ranges)
        }
        _ => return Ok(None),
    };
    let idx: Vec<IdxSize> = ranges.into_iter().flatten().flat_map(|(start, len)| (start..start + len).map(|i| i as IdxSize)).collect();
    Ok(Some(inner.take_slice(&idx)?))
}

fn describe_one(s: &Series, seed: u64) -> PolarsResult<Row> {
    let mut row: Row = vec![
        AnyValue::StringOwned(s.name().clone()),
        AnyValue::UInt64(s.len() as u64),
        AnyValue::UInt64(s.null_count() as u64),
    ];
    row.extend(profile(s, seed)?);
    row.push(u64v(n_midnight(s)?));
    match flatten(s)? {
        Some(inner) => {
            row.push(AnyValue::UInt64(inner.len() as u64));
            row.push(AnyValue::UInt64(inner.null_count() as u64));
            row.extend(profile(&inner, seed)?);
        }
        None => row.extend(nulls(2 + value_fields().len())),
    }
    Ok(row)
}

pub(crate) fn describe_columns_impl(inputs: &[Series], seed: u64) -> PolarsResult<Series> {
    let rows: Vec<Row> = inputs.par_iter().map(|s| describe_one(s, seed)).collect::<PolarsResult<_>>()?;
    let columns = fields()
        .iter()
        .enumerate()
        .map(|(j, (name, dtype))| {
            let values: Vec<AnyValue> = rows.iter().map(|r| r[j].clone()).collect();
            Series::from_any_values_and_dtype(name.as_str().into(), &values, dtype, true)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructChunked::from_series("describe".into(), inputs.len(), columns.iter())?.into_series())
}

#[derive(Deserialize)]
struct DescribeKwargs {
    seed: u64,
}

#[polars_expr(output_type_func=describe_output_type)]
fn describe_columns(inputs: &[Series], kwargs: DescribeKwargs) -> PolarsResult<Series> {
    describe_columns_impl(inputs, kwargs.seed)
}
```

Append to `services/analytics/analytics/_plugin.py`:

```python
"""
Describe
"""


def describe_columns(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """One row per column of `df`: `column` + describe's value metrics for the column
    and its inner values (see analytics.describe.base). `seed` fixes the 3-way split
    behind the capture history. Private — called only by DescribeRust."""
    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="describe_columns",
            args=df.get_columns(),
            kwargs={"seed": seed},
            is_elementwise=False,
            changes_length=True,
        ).alias("describe")
    ).unnest("describe")


def column_sizes(df: pl.DataFrame, zstd_level: int) -> pl.DataFrame:
    """One row per column of `df`: Arrow IPC body bytes (classic layout, plain and
    ZSTD) and Polars sizes (estimated_size, ZSTD IPC of the native layout).
    Private — called only by DescribeRust."""
    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="column_sizes",
            args=df.get_columns(),
            kwargs={"zstd_level": zstd_level},
            is_elementwise=False,
            changes_length=True,
        ).alias("column_sizes")
    ).unnest("column_sizes")
```

`services/analytics/analytics/describe/rust.py`:

```python
from analytics import _plugin
from analytics.base import group_by_frame
from analytics.describe.base import Describe


class DescribeRust(Describe):
    """Rust plugin: `describe_columns` (one pass per column; rayon across columns and
    64K-row chunks; hash-map frequencies, byte scanners, row-encoded extremes) and
    `column_sizes` (Arrow buffer walk + zstd, mirroring pyarrow's IPC writer)."""

    def _compute(self, frames, combos):
        rows: dict[tuple[str, str], dict] = {}
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select([c for ((_, c),) in group])
            stats = _plugin.describe_columns(df, self.seed).join(_plugin.column_sizes(df, self.zstd_level), on="column")
            rows.update(((frame, r["column"]), r) for r in stats.iter_rows(named=True))
        return self.metrics_frame(combos, {m: [rows[k[0]][m] for k in combos] for m in self.METRICS})
```

- [ ] **Step 4: Build and run**

Run (from `services/analytics/`): `$PY -m maturin develop --release`
Then (from repo root): `$PY -m pytest tests/test_describe.py -v`
Expected: PASS for both implementations, including `test_agrees_with_reference[DescribeRust]`. If agreement fails, read the first mismatch lines: they name the column and metric. Fix the Rust side unless the Python oracle contradicts a known-answer test.

Then: `$PY -m pytest tests/test_describe.py -m slow -v` (large-dataset agreement) and `$PY -m pytest tests -m "not slow"` (whole suite — the `encode_series` change must not break other techniques).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src services/analytics/analytics tests/test_describe.py
git commit -m "feat: DescribeRust - describe_columns plugin composing the describe kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: DescribeDataFusion

**Files:**
- Create: `services/analytics/analytics/describe/datafusion.py`
- Modify: `services/analytics/analytics/describe/__init__.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `column_sizes`; `_values.{FLOATS, STRING_LIKE, flatten, frac_digits, frequency_summary, n_midnight, subsets}` and the regex constants; `Describe`, `VALUE_METRICS`, `GROUP_B`, `GROUP_C`.
- Produces: `class DescribeDataFusion(Describe)`; `IMPLEMENTATIONS = ("DescribeRust", "DescribeDataFusion", "DescribePolars")`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_describe.py`:

```python
@pytest.mark.parametrize("impl", ALL)
def test_nested_ordering_with_null_elements(impl):
    s = pl.Series("x", [[2], [None], [1, None], [], [1]], dtype=pl.List(pl.Int64))
    want = s.arg_sort(nulls_last=False)  # Polars order is the definition
    r = profile(impl, s)
    assert (r["argmin"], r["argmax"], r["n_unique"]) == (want[0], want[-1], 5)
```

Add `"DescribeDataFusion"` to `IMPLEMENTATIONS` (between Rust and Polars) and `"DescribeDataFusion": ".datafusion"` to the lazy map in `describe/__init__.py`, and `"DescribeDataFusion"` to `__all__`.

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_describe.py -k DataFusion -v -rs`
Expected: every `DescribeDataFusion` test is SKIPPED with `DescribeDataFusion unavailable: No module named 'analytics.describe.datafusion'` (`harness.load` turns any ImportError into a skip). Once the module exists, a skip naming the `datafusion` library would mean the library is missing — it is installed (54.0.0), so after Step 3 there must be no skips.

- [ ] **Step 3: Implement** — `services/analytics/analytics/describe/datafusion.py`:

```python
"""DescribeDataFusion — Apache DataFusion SQL over each column's Arrow data."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
from datafusion import SessionContext

from analytics.describe._sizes import column_sizes
from analytics.describe._values import (
    FLOATS, FRAC_DIGITS, INT_DIGITS, ISO_DATE, ISO_DATETIME, ISO_DATETIME_TZ, ISO_FRACTION, ISO_MIDNIGHT,
    ISO_OFFSET, ISO_TIME, LEADING_ZERO, NUMERIC, NUMERIC_INT, STRING_LIKE,
    flatten, frac_digits, frequency_summary, n_midnight, subsets,
)
from analytics.describe.base import GROUP_B, GROUP_C, VALUE_METRICS, Describe


class DescribeDataFusion(Describe):
    """Each column (and its flattened inner values) is registered as an Arrow table
    `t(row, sub, v, o)` and profiled with DataFusion SQL: GROUP BY for frequencies
    (COUNT, MIN(row), BIT_OR of the split subset), ORDER BY … LIMIT 1 for
    first-occurrence extremes, regexp_like (Rust regex: linear time) and
    try_cast(… AS DATE) for the string scanners. Enum columns order by their
    physical codes (`o`); Categorical/Enum values are scanned as strings (`v`).

    Computed outside SQL — exactly, never approximated:
      - sizes: the shared pyarrow helper (_sizes.column_sizes);
      - inner values: the shared Polars `flatten` (DataFusion's unnest drops null elements);
      - n_midnight: Polars dt.time() (timezone-aware date_trunc is not relied on);
      - entropy / f1 / f2 / top-5 / capture history: numpy over the GROUP BY result;
      - max_frac_digits: the shared parser over CAST(v AS VARCHAR) of distinct finite values.
    Int128 is registered as Decimal(38, 0); values beyond 38 digits are not supported.
    """

    def _compute(self, frames, combos):
        ctx = SessionContext()
        rows = [self._row(ctx, frames[n][c]) for ((n, c),) in combos]
        return self.metrics_frame(combos, {m: [r[m] for r in rows] for m in self.METRICS})

    def _row(self, ctx: SessionContext, s: pl.Series) -> dict:
        row = {
            "n_rows": s.len(), "n_null": s.null_count(), **self._profile(ctx, s),
            "n_midnight": n_midnight(s), **column_sizes(s, self.zstd_level),
        }
        inner = flatten(s) if isinstance(s.dtype, (pl.List, pl.Array)) else None
        row["inner_n_values"] = None if inner is None else inner.len()
        row["inner_n_null"] = None if inner is None else inner.null_count()
        inner_profile = dict.fromkeys(VALUE_METRICS) if inner is None else self._profile(ctx, inner)
        return row | {f"inner_{k}": v for k, v in inner_profile.items()}

    def _profile(self, ctx: SessionContext, s: pl.Series) -> dict:
        n = s.len()
        v = s.cast(pl.String) if isinstance(s.dtype, (pl.Categorical, pl.Enum)) else s
        o = s.to_physical() if isinstance(s.dtype, pl.Enum) else v
        table = pa.table({"row": np.arange(n, dtype=np.uint64), "sub": subsets(n, self.seed), "v": _arrow(v), "o": _arrow(o)})
        ctx.register_record_batches("t", [table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]])
        try:
            return {**_frequencies(ctx, s), **_extremes(ctx, s), **_lengths(ctx, s), **_floats(ctx, s), **_strings(ctx, s)}
        finally:
            ctx.deregister_table("t")


def _arrow(s: pl.Series) -> pa.Array:
    if s.dtype == pl.Int128:
        s = s.cast(pl.Decimal(38, 0))
    return s.rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())


def _one(ctx: SessionContext, sql: str) -> dict:
    return ctx.sql(sql).to_arrow_table().to_pylist()[0]


def _key(dtype: pl.DataType, col: str) -> str:
    """Canonical grouping/ordering key: one NaN, and -0.0 → 0.0 for floats."""
    if not isinstance(dtype, FLOATS):
        return col
    t = "REAL" if dtype == pl.Float32 else "DOUBLE"
    return f"CASE WHEN isnan({col}) THEN CAST('NaN' AS {t}) WHEN {col} = 0 THEN abs({col}) ELSE {col} END"


def _frequencies(ctx: SessionContext, s: pl.Series) -> dict:
    key = _key(s.dtype, "v")
    freq = ctx.sql(
        f"SELECT {key} AS k, COUNT(*) AS c, MIN(row) AS f, BIT_OR(CAST(1 AS BIGINT) << sub) AS m "
        f"FROM t WHERE v IS NOT NULL GROUP BY {key}"
    ).to_arrow_table()
    return frequency_summary(freq["c"].to_numpy(), freq["f"].to_numpy(), freq["m"].to_numpy(), s.len(), s.null_count())


def _extremes(ctx: SessionContext, s: pl.Series) -> dict:
    ok = "o IS NOT NULL" + (" AND NOT isnan(o)" if isinstance(s.dtype, FLOATS) else "")
    key = _key(s.dtype, "o")
    r = _one(
        ctx,
        f"SELECT (SELECT row FROM t WHERE {ok} ORDER BY {key} ASC NULLS FIRST, row ASC LIMIT 1) AS lo, "
        f"(SELECT row FROM t WHERE {ok} ORDER BY {key} DESC NULLS LAST, row ASC LIMIT 1) AS hi",
    )
    return {"argmin": r["lo"], "argmax": r["hi"]}


def _lengths(ctx: SessionContext, s: pl.Series) -> dict:
    if isinstance(s.dtype, STRING_LIKE) or s.dtype == pl.Binary:
        expr = "octet_length(v)"
    elif isinstance(s.dtype, (pl.List, pl.Array)):
        expr = "array_length(v)"
    else:
        return {"min_len": None, "max_len": None}
    r = _one(ctx, f"SELECT MIN({expr}) AS lo, MAX({expr}) AS hi FROM t WHERE v IS NOT NULL")
    return {"min_len": r["lo"], "max_len": r["hi"]}


def _floats(ctx: SessionContext, s: pl.Series) -> dict:
    if not isinstance(s.dtype, FLOATS):
        return dict.fromkeys(GROUP_B)
    t = "REAL" if s.dtype == pl.Float32 else "DOUBLE"
    finite = f"(NOT isnan(v) AND abs(v) <> CAST('Infinity' AS {t}))"
    r = _one(
        ctx,
        f"""SELECT
              SUM(CASE WHEN isnan(v) THEN 1 ELSE 0 END) AS nan,
              SUM(CASE WHEN NOT isnan(v) AND abs(v) = CAST('Infinity' AS {t}) THEN 1 ELSE 0 END) AS inf,
              SUM(CASE WHEN {finite} AND v <> trunc(v) THEN 1 ELSE 0 END) AS frac,
              SUM(CASE WHEN {finite} AND CAST(CAST(v AS REAL) AS DOUBLE) <> CAST(v AS DOUBLE) THEN 1 ELSE 0 END) AS f32
            FROM t WHERE v IS NOT NULL""",
    )
    reprs = ctx.sql(f"SELECT DISTINCT CAST(v AS VARCHAR) AS r FROM t WHERE v IS NOT NULL AND {finite}").to_arrow_table()
    return {
        "n_nan": r["nan"] or 0,
        "n_inf": r["inf"] or 0,
        "n_fractional": r["frac"] or 0,
        "max_frac_digits": frac_digits(pl.Series(reprs["r"].to_pylist(), dtype=pl.String)),
        "n_f32_inexact": None if s.dtype == pl.Float32 else r["f32"] or 0,
    }


def _strings(ctx: SessionContext, s: pl.Series) -> dict:
    if not isinstance(s.dtype, STRING_LIKE):
        return dict.fromkeys(GROUP_C)
    sig = "length(regexp_replace(v, '^-?0*', ''))"
    off = f"regexp_match(v, '{ISO_OFFSET}')[1]"
    r = _one(
        ctx,
        f"""WITH s AS (
              SELECT v,
                regexp_like(v, '{NUMERIC}') AS num, regexp_like(v, '{NUMERIC_INT}') AS nint,
                try_cast(substr(v, 1, 10) AS DATE) IS NOT NULL AS dok,
                regexp_like(v, '{ISO_DATE}') AS d, regexp_like(v, '{ISO_TIME}') AS tm,
                regexp_like(v, '{ISO_DATETIME}') AS dt, regexp_like(v, '{ISO_DATETIME_TZ}') AS tz
              FROM t WHERE v IS NOT NULL)
            SELECT
              SUM(CAST(num AS BIGINT)) AS n_numeric,
              SUM(CAST(nint AS BIGINT)) AS n_numeric_int,
              SUM(CAST(regexp_like(v, '{LEADING_ZERO}') AS BIGINT)) AS n_leading_zero,
              MAX(CASE WHEN nint THEN {sig} END) AS int_sig,
              MIN(CASE WHEN nint AND {sig} <= 38 THEN CAST(v AS DECIMAL(38, 0)) END) AS int_min,
              MAX(CASE WHEN nint AND {sig} <= 38 THEN CAST(v AS DECIMAL(38, 0)) END) AS int_max,
              MAX(CASE WHEN num THEN length(regexp_match(v, '{INT_DIGITS}')[1]) END) AS int_digits,
              MAX(CASE WHEN num THEN COALESCE(length(regexp_match(v, '{FRAC_DIGITS}')[1]), 0) END) AS frac_digits,
              SUM(CAST(d AND dok AS BIGINT)) AS n_iso_date,
              SUM(CAST(tm AS BIGINT)) AS n_iso_time,
              SUM(CAST(dt AND dok AS BIGINT)) AS n_iso_datetime,
              SUM(CAST(tz AND dok AS BIGINT)) AS n_iso_datetime_tz,
              MAX(CASE WHEN tm OR ((dt OR tz) AND dok) THEN COALESCE(length(regexp_match(v, '{ISO_FRACTION}')[1]), 0) END) AS iso_frac,
              COUNT(DISTINCT CASE WHEN tz AND dok THEN (CASE WHEN {off} IN ('Z', '-00:00') THEN '+00:00' ELSE {off} END) END) AS iso_n_offsets,
              SUM(CAST((dt OR tz) AND dok AND regexp_like(v, '{ISO_MIDNIGHT}') AS BIGINT)) AS iso_n_midnight
            FROM s""",
    )
    in_range = r["int_sig"] is not None and r["int_sig"] <= 38
    return {
        "n_numeric": r["n_numeric"] or 0,
        "n_numeric_int": r["n_numeric_int"] or 0,
        "n_leading_zero": r["n_leading_zero"] or 0,
        "numeric_int_min": int(r["int_min"]) if in_range else None,
        "numeric_int_max": int(r["int_max"]) if in_range else None,
        "numeric_max_int_digits": r["int_digits"],
        "numeric_max_frac_digits": r["frac_digits"],
        "n_iso_date": r["n_iso_date"] or 0,
        "n_iso_time": r["n_iso_time"] or 0,
        "n_iso_datetime": r["n_iso_datetime"] or 0,
        "n_iso_datetime_tz": r["n_iso_datetime_tz"] or 0,
        "iso_max_frac_digits": r["iso_frac"],
        "iso_n_offsets": r["iso_n_offsets"] or 0,
        "iso_n_midnight": r["iso_n_midnight"] or 0,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_describe.py -v` then `$PY -m pytest tests/test_describe.py -m slow -v`
Expected: PASS for all three implementations. Where DataFusion cannot express a metric exactly for some dtype (e.g. GROUP BY on Struct, `array_length` on fixed-size lists, nested-null ordering), compute that metric for that dtype with Polars/pyarrow inside this class, add it to the "Computed outside SQL" list in the class docstring, and keep the test unchanged.

- [ ] **Step 5: Commit**

```bash
git add services/analytics/analytics/describe tests/test_describe.py
git commit -m "feat: DescribeDataFusion - SQL aggregates over each column's Arrow data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Benchmark and documentation

**Files:**
- Modify: `tests/datagen.py`, `tests/test_datagen.py`, `CLAUDE.md`
- Create: `tests/performance/benchmark_describe.py`

**Interfaces:**
- Produces: `describe_narrow(n_rows: int, seed: int = 42) -> pl.DataFrame` (columns `int, float, num_str, iso_str`); `describe_wide(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame` (`c000…`, cycling Int64 / Float64 / String / Date); `describe_nested(n_rows: int, seed: int = 42) -> pl.DataFrame` (`list_i64, list_str, struct`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_datagen.py`:

```python
from datagen import describe_narrow, describe_nested, describe_wide


def test_describe_benchmark_shapes():
    narrow = describe_narrow(1_000)
    assert narrow.columns == ["int", "float", "num_str", "iso_str"] and narrow.height == 1_000
    wide = describe_wide(100, 8)
    assert wide.width == 8 and [str(dt) for dt in wide.dtypes[:4]] == ["Int64", "Float64", "String", "Date"]
    nested = describe_nested(500)
    assert nested.schema == {"list_i64": pl.List(pl.Int64), "list_str": pl.List(pl.String), "struct": pl.Struct({"a": pl.Int64, "b": pl.String})}
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_datagen.py -k benchmark_shapes -v`
Expected: FAIL — `ImportError: cannot import name 'describe_narrow'`.

- [ ] **Step 3: Implement** — append to `tests/datagen.py` (vectorised; benchmark sizes reach 10M rows):

```python
def describe_narrow(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """int, float, numeric-string and ISO-datetime-string columns (benchmarks)."""
    rng = np.random.default_rng(seed)
    ints = rng.integers(0, 1_000_000, n_rows)
    return pl.DataFrame({"int": ints, "float": np.round(rng.normal(100.0, 15.0, n_rows), 2)}).with_columns(
        pl.col("int").cast(pl.String).alias("num_str"),
        (pl.datetime(2024, 1, 1) + pl.duration(seconds=pl.col("int") * 30)).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("iso_str"),
    )


def describe_wide(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """n_cols columns c000… cycling Int64 (low cardinality) / Float64 / String / Date."""
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(n_cols):
        kind = i % 4
        if kind == 0:
            cols.append(pl.Series(f"c{i:03d}", rng.integers(0, 50, n_rows)))
        elif kind == 1:
            cols.append(pl.Series(f"c{i:03d}", np.round(rng.normal(0.0, 1.0, n_rows), 3)))
        elif kind == 2:
            cols.append(pl.Series(f"c{i:03d}", _WORDS[rng.integers(0, 5, n_rows)]))
        else:
            cols.append(pl.Series(f"c{i:03d}", rng.integers(19_000, 20_000, n_rows)).cast(pl.Int32).cast(pl.Date))
    return pl.DataFrame(cols)


def describe_nested(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """List(Int64), List(String) (0–4 elements) and Struct{a: Int64, b: String} columns."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 5, n_rows)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = rng.integers(0, 100, int(offsets[-1]))
    list_i64 = pl.from_arrow(pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(values)))
    return pl.DataFrame(
        [
            list_i64.alias("list_i64"),
            list_i64.cast(pl.List(pl.String)).alias("list_str"),
            pl.DataFrame({"a": rng.integers(0, 3, n_rows), "b": _WORDS[rng.integers(0, 2, n_rows)]}).to_struct("struct"),
        ]
    )
```

`tests/performance/benchmark_describe.py`:

```python
"""
describe speed benchmark: DescribeRust vs DescribeDataFusion vs DescribePolars (reference).

Run: /c/Users/Ben/miniconda3/envs/p312/python.exe tests/performance/benchmark_describe.py

Shapes: large_dataset.arrow (realistic mix); narrow 10M x 4 (parallelism inside a
column — chunked frequency maps and scanners); wide 1M x 100 (parallelism across
columns); nested 1M x 3 (inner values, row-encoded extremes, struct hashing).
"""

from harness import Dataset, large_dataset, run

from datagen import describe_narrow, describe_nested, describe_wide

if __name__ == "__main__":
    run(
        "analytics.describe",
        [
            large_dataset(),
            Dataset("narrow 10M x 4", lambda: {"t": describe_narrow(10_000_000)}),
            Dataset("wide 1M x 100", lambda: {"t": describe_wide(1_000_000, 100)}),
            Dataset("nested 1M x 3", lambda: {"t": describe_nested(1_000_000)}),
        ],
    )
```

`CLAUDE.md` edits:
1. Under **Analytical Functions → 1. Per-column**, after the GCD entry, add:

```markdown
**Describe — `analytics.describe`** (`DescribeRust`, `DescribeDataFusion`, ★`DescribePolars`). Profile for choosing narrower / more compressible Arrow types (spec: docs/superpowers/specs/2026-09-26-describe-technique-design.md). Metrics: counts, entropy (null as a category), f1/f2, first-occurrence argmin/argmax and top-5 (indices; the base renders values), byte/list lengths, float stats (NaN/inf/fractional, decimal places, f32 round trip), numeric-string and ISO 8601 counts (Rust byte scanners / Rust-regex elsewhere — linear time), Datetime local-midnight count, Arrow IPC sizes (classic layout, plain + ZSTD) and Polars sizes, and the same for list inner values. Conclusions: rendered min/max/top-5, `unique`, Chao1 / Schnabel (3-way seeded split) / Duj1 with 95% intervals and `est_cardinality` picked by rule (exact → Duj1 → Schnabel → Chao1), `estimates_agree`, and `class` ∈ {null, constant, boolean, ordinal, categorical, discrete}. Keywords: `population_rows=None`, `categorical_threshold=10_000`, `zstd_level=1`, `seed=0`. Agreement: exact except entropy (1e-9), ZSTD sizes (1%), Schnabel (10%).
```

2. In **Project Structure**, add `describe` to the `<technique>/` comment list and `describe/` to the Rust `src/` list.
3. In **Rust Plugin (analytics)**, add `describe_columns` and `column_sizes` to the private function list.

- [ ] **Step 4: Run to verify**

Run: `$PY -m pytest tests -m "not slow"` then `$PY tests/performance/benchmark_describe.py`
Expected: tests PASS; the benchmark prints a table per dataset with median/min times and algorithmic / parallel / total speedups for `DescribeRust`, and writes `tests/performance/results/describe_*.parquet`.

- [ ] **Step 5: Commit**

```bash
git add tests/datagen.py tests/test_datagen.py tests/performance/benchmark_describe.py CLAUDE.md
git commit -m "test: describe benchmark (large, narrow, wide, nested); document the technique

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
