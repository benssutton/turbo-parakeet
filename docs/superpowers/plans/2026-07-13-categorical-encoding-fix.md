# Categorical Encoding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode `Categorical`/`Enum` columns by their string value (via foldhash) instead of their physical u32 code, fixing cross-frame membership correctness and false-positive-rate hash clustering in one change.

**Architecture:** A single arm of `encode_series` in [shared.rs](../../../services/analytics/src/shared.rs) is changed to cast the categorical Series to `String` and hash each value with `foldhash::fast::FixedState` — the exact logic the existing `String` arm uses. Because the fix lives in the shared encoding layer, all four consumers (bloom, minhash, entropy, chi²) inherit it. Equivalence classes are unchanged (distinct category → distinct key), so entropy/chi² counts are unaffected; only cross-column set operations change, and they become correct.

**Tech Stack:** Rust (polars 0.51.0, pyo3-polars 0.24, foldhash), maturin, Python 3.12, pytest, fastbloom-rs.

## Global Constraints

- Python interpreter: `C:\Users\Ben\miniconda3\envs\p312\python.exe` (conda env `p312`).
- Rust plugin build command: `maturin develop --release`, run from `services/analytics/`.
- Polars version is pinned at 0.51.0 — categorical is a generic `CategoricalChunked<T>` (u8/u16/u32 physical widths). Do **not** use `get_rev_map()` or assume a u32 physical repr; those do not exist / do not hold in 0.51. Resolve categories via `series.cast(&DataType::String)`.
- foldhash `FixedState` is the project-standard hash for value-hashed dtypes (String, List, Array, Int128, Decimal). Categorical joins that set.
- Run the Python suite with the `not slow` marker unless explicitly testing recall: `-m "not slow"`.

---

## File Structure

- **Modify:** [services/analytics/src/shared.rs](../../../services/analytics/src/shared.rs) — the `Categorical(_, _) | Enum(_, _)` arm of `encode_series` (lines 190-196). Sole behavioural change.
- **Modify:** [tests/test_bloom_filter.py](../../../tests/test_bloom_filter.py) — add a cross-frame regression test; re-include categorical in the FP-rate and cross-implementation parametrizations; remove the now-obsolete categorical caveat.
- **Modify:** [CLAUDE.md](../../../CLAUDE.md) — extend the "String / list / decimal types route through foldhash" cross-cutting note to include categorical/enum.

---

### Task 1: Cross-frame categorical regression test (RED)

This test proves the bug is present against the current build. It builds a bloom filter from one categorical Series, then queries with a second, independently constructed categorical Series in which the same string values are assigned **different** physical codes. Under the current code-based encoding those lookups miss (false negatives); the test must fail now and pass after Task 2.

**Files:**
- Modify: `tests/test_bloom_filter.py` (add one test function at the end of the "No false negatives" section, after `test_no_false_negatives_fastbloom`)

**Interfaces:**
- Consumes: `BloomFilter` (already imported), `FP_RATE` (already defined), `pl` (already imported).
- Produces: `test_cross_frame_categorical_membership` — no downstream consumers.

- [ ] **Step 1: Write the failing test**

Add this function immediately after `test_no_false_negatives_fastbloom` in `tests/test_bloom_filter.py`:

```python
def test_cross_frame_categorical_membership() -> None:
    """
    A categorical value must be found by its string identity, not its physical
    code. Build the filter on one Series, then query with a separately built
    Series where the SAME strings get DIFFERENT physical codes (achieved by
    prepending distinct padding categories). Under code-based encoding the
    shared values map to codes the filter never saw → false negatives. Under
    string-based encoding they hash identically → all found.
    """
    shared = [f"tok_{i}" for i in range(50)]
    filter_series = pl.Series("c", shared, dtype=pl.String).cast(pl.Categorical)
    bf = BloomFilter(len(shared), FP_RATE)
    bf.add(pl.DataFrame({"c": filter_series}))

    # Prepend 50 distinct padding categories so tok_i is assigned code 50+i
    # in the query Series instead of code i.
    pad = [f"pad_{i}" for i in range(50)]
    query_series = pl.Series("c", pad + shared, dtype=pl.String).cast(pl.Categorical)
    result = bf.membership(pl.DataFrame({"c": query_series})).to_series()

    # Only the shared tail must be fully found (the pad prefix should not be).
    shared_found = int(result.slice(len(pad)).sum())
    assert shared_found == len(shared), (
        f"cross-frame categorical: {shared_found}/{len(shared)} shared values "
        "found; a categorical must be identified by string value, not code"
    )
```

- [ ] **Step 2: Run the test to verify it FAILS against the current build**

Run:
```bash
C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_bloom_filter.py::test_cross_frame_categorical_membership -v
```
Expected: **FAIL** — `shared_found` is ~0 (not 50), because the current `.pyd` encodes categoricals by physical code and the query's shared values carry codes 50-99 that the filter (codes 0-49) never set.

> If this test unexpectedly PASSES against the current build, stop: the bug reproduction is wrong and Task 2 would not be validated. Re-check that the plugin has not already been rebuilt with the fix.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_bloom_filter.py
git commit -m "test: add cross-frame categorical membership regression (currently failing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Fix `encode_series` categorical arm + rebuild + doc note (GREEN)

**Files:**
- Modify: `services/analytics/src/shared.rs:190-196` (the Categorical/Enum arm)
- Modify: `CLAUDE.md` (cross-cutting foldhash note)
- Test: `tests/test_bloom_filter.py::test_cross_frame_categorical_membership` (from Task 1)

**Interfaces:**
- Consumes: `FoldHashFixed` and `hash_one` — both already in scope in `shared.rs` ([import at line 5](../../../services/analytics/src/shared.rs#L5), [`hash_one` at line 257](../../../services/analytics/src/shared.rs#L257)).
- Produces: unchanged `encode_series` signature — `EncodedColumn { values: Vec<u64>, is_null: Vec<bool> }`. Categorical values now equal `foldhash(string_value)` instead of `code as u64`.

- [ ] **Step 1: Replace the Categorical/Enum arm**

In `services/analytics/src/shared.rs`, replace exactly this block:

```rust
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let phys = series.to_physical_repr();
            phys.u32()?
                .iter()
                .map(|v| v.map_or((0, true), |x| (x as u64, false)))
                .unzip()
        }
```

with:

```rust
        // Resolve categories to their string values so a categorical "x" encodes
        // identically to the string "x" — and identically across frames, whatever
        // per-Series physical code each frame assigned. Casting to String is
        // version-stable across the Polars 0.51 generic-categorical rework and
        // preserves nulls; from here the logic is identical to the String arm.
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let build_hasher = FoldHashFixed::default();
            let str_series = series.cast(&DataType::String)?;
            str_series
                .str()?
                .iter()
                .map(|v| v.map_or((0, true), |s| (hash_one(&build_hasher, s), false)))
                .unzip()
        }
```

- [ ] **Step 2: Rebuild the plugin**

Run (from the analytics package directory):
```bash
cd services/analytics && maturin develop --release
```
Expected: `Finished` / `Installed analytics-…` with no compile errors. If the compiler rejects `series.cast(&DataType::String)` or `.str()?`, do not guess alternatives — re-read `encode_series`'s existing `String` arm ([shared.rs:182-189](../../../services/analytics/src/shared.rs#L182-L189)) and mirror its accessor usage.

- [ ] **Step 3: Run the regression test to verify it now PASSES**

Run:
```bash
C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_bloom_filter.py::test_cross_frame_categorical_membership -v
```
Expected: **PASS** — all 50 shared values found.

- [ ] **Step 4: Run the Rust unit tests to confirm no regression in the plugin**

Run (from `services/analytics/`):
```bash
cd services/analytics && cargo test
```
Expected: all tests pass (bloomfilter.rs, minhash.rs suites). These do not exercise categoricals directly, so this is a guard that the shared-layer edit did not break compilation or other dtype paths.

- [ ] **Step 5: Update the CLAUDE.md cross-cutting note**

In `CLAUDE.md`, find this line in the "Cross-cutting notes" section:

```
- `encode_series` (formerly `series_to_u64`) returns `EncodedColumn { values, is_null }` — nulls are out-of-band (no in-band sentinel), floats are canonicalised (`-0.0`→`0.0`, all NaN payloads→one key). Null policy per module: entropy = null is a category; chi² = null rows dropped; minhash/bloom = nulls skipped. Keep this in mind when deriving MI from entropy + chi² outputs.
- `String / list / decimal types route through foldhash → u64. Collision probability at 50K rows is ~6×10⁻¹¹ per pair — negligible for entropy/χ², irrelevant for MinHash (deterministic seed across columns). foldhash `FixedState` is NOT stable across crate versions/platforms — don't persist bloom bit arrays or minhash signatures across rebuilds for hashed dtypes.
```

Change the second bullet's opening so categorical/enum are listed, and note they are resolved by string value:

```
- `String / categorical / enum / list / decimal types route through foldhash → u64` (categorical/enum are cast to their string value first, so a categorical `"x"` hashes identically to the string `"x"` and identically across frames regardless of physical code). Collision probability at 50K rows is ~6×10⁻¹¹ per pair — negligible for entropy/χ², irrelevant for MinHash (deterministic seed across columns). foldhash `FixedState` is NOT stable across crate versions/platforms — don't persist bloom bit arrays or minhash signatures across rebuilds for hashed dtypes.
```

- [ ] **Step 6: Commit**

```bash
git add services/analytics/src/shared.rs CLAUDE.md
git commit -m "fix: encode categorical/enum by string value, not physical code

Casts categorical to String and hashes with foldhash in encode_series,
matching the String arm. Fixes cross-frame bloom/minhash correctness
(same string -> same key) and removes the small-integer FP-rate clustering.
Equivalence classes unchanged, so entropy/chi-squared counts are unaffected.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Bring categorical inside the FP-rate assertions (GREEN)

Categorical was excluded from the FP-rate and cross-implementation tests solely because of the clustering bug. With the fix, categorical strings hash through foldhash and achieve the configured FP rate, so it rejoins those assertions and the two work-around constants become redundant.

**Files:**
- Modify: `tests/test_bloom_filter.py` (docstring, two parametrization constants, two docstrings inside tests)

**Interfaces:**
- Consumes: `NON_BOOLEAN_COLS` and `SCALAR_NON_BOOLEAN_COLS` (already defined) become the parametrization sources.
- Produces: no new symbols; removes `CUSTOM_FP_RATE_COLS` and `CROSS_IMPL_FP_COLS`.

- [ ] **Step 1: Remove the module-docstring caveat**

In `tests/test_bloom_filter.py`, delete the entire "Note on Categorical FP rate testing" section. Replace this block:

```python
Cross-implementation comparison is at the FP-rate level only. The two
implementations use incompatible hash functions, so per-item membership
decisions on false positives are expected to differ and are not compared.

Note on Categorical FP rate testing
------------------------------------
The custom Rust plugin encodes Categorical values by their physical u32 code
(see encode_series in shared.rs), NOT by their string value. Categorical codes
are small consecutive integers (0, 1, 2, ...). Feeding small sequential u64
keys into xxh3_128 and a small bloom filter produces correlated bit patterns
that inflate the empirical FP rate well above the configured target — this is a
hash-clustering artefact, not a correctness bug (the no-false-negatives
guarantee still holds). FP rate tests for the custom filter therefore exclude
Categorical; the no-false-negatives tests still exercise that dtype.

fastbloom-rs hashes the string representation of each value, bypassing the code
issue entirely, so its categorical FP rate tests are unaffected.
"""
```

with:

```python
Cross-implementation comparison is at the FP-rate level only. The two
implementations use incompatible hash functions, so per-item membership
decisions on false positives are expected to differ and are not compared.

Categorical/enum columns encode by string value (encode_series casts to String
and hashes with foldhash), so they behave like string columns here and are
covered by every test group below.
"""
```

- [ ] **Step 2: Delete the `CUSTOM_FP_RATE_COLS` constant and its comment**

Remove this block:

```python
# FP rate testing excludes Categorical for the custom filter: physical u32 codes
# are small sequential integers, and xxh3_128 on those keys in a small filter
# produces clustered bit patterns that inflate the empirical FP rate.
CUSTOM_FP_RATE_COLS = [
    f"{dtype}_{shape}"
    for dtype in ("float64", "uint32", "list", "arr")
    for shape in ("skewed", "high_unique", "sparse")
]
```

(No replacement — `NON_BOOLEAN_COLS` is used in its place in Step 4.)

- [ ] **Step 3: Delete the `CROSS_IMPL_FP_COLS` constant and its comment**

Remove this block:

```python
# Cross-implementation FP rate comparison excludes Categorical: the custom
# filter hashes physical codes while fastbloom-rs hashes string representations,
# making a rate comparison meaningless.
CROSS_IMPL_FP_COLS = [
    f"{dtype}_{shape}"
    for dtype in ("float64", "uint32")
    for shape in ("skewed", "high_unique", "sparse")
]
```

(No replacement — `SCALAR_NON_BOOLEAN_COLS` is used in its place in Step 5.)

- [ ] **Step 4: Repoint `test_false_positive_rate_custom` to `NON_BOOLEAN_COLS`**

Change its parametrize decorator from:

```python
@pytest.mark.parametrize("col", CUSTOM_FP_RATE_COLS)
def test_false_positive_rate_custom(dataset: pl.DataFrame, col: str) -> None:
```

to:

```python
@pytest.mark.parametrize("col", NON_BOOLEAN_COLS)
def test_false_positive_rate_custom(dataset: pl.DataFrame, col: str) -> None:
```

- [ ] **Step 5: Repoint `test_fp_rate_agreement_vs_fastbloom` to `SCALAR_NON_BOOLEAN_COLS` and fix its docstring**

Change the decorator from:

```python
@pytest.mark.parametrize("col", CROSS_IMPL_FP_COLS)
```

to:

```python
@pytest.mark.parametrize("col", SCALAR_NON_BOOLEAN_COLS)
```

Then replace the test's docstring:

```python
    """
    Custom and fastbloom-rs must both achieve a FP rate within tolerance.

    Categorical is excluded: the custom filter hashes physical u32 codes while
    fastbloom-rs hashes string representations, making a rate comparison
    meaningless (see module docstring).

    Per-item membership results on negatives are deliberately NOT compared — the
    two implementations use different hash functions, so per-item disagreement on
    false positives is expected and correct.
    """
```

with:

```python
    """
    Custom and fastbloom-rs must both achieve a FP rate within tolerance.

    Categorical is included: both implementations now hash the string value
    (the custom filter casts categorical to String in encode_series), so the
    rate comparison is meaningful.

    Per-item membership results on negatives are deliberately NOT compared — the
    two implementations use different hash functions, so per-item disagreement on
    false positives is expected and correct.
    """
```

- [ ] **Step 6: Run the full non-slow bloom suite**

Run:
```bash
C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_bloom_filter.py -v
```
Expected: **all pass**, including the three `test_false_positive_rate_custom[categorical_*]` cases and the three `test_fp_rate_agreement_vs_fastbloom[categorical_*]` cases that are now parametrized in.

- [ ] **Step 7: Run the whole non-slow suite to confirm no cross-module regression**

Run:
```bash
C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest -m "not slow" -q
```
Expected: **all pass** (entropy, chi-squared, similarity, bloom). This confirms the shared-encoding change did not perturb entropy/chi² counts (equivalence classes preserved).

- [ ] **Step 8: Commit**

```bash
git add tests/test_bloom_filter.py
git commit -m "test: cover categorical in bloom FP-rate and cross-impl tests

Categorical now hashes by string value, so it achieves the configured FP
rate and can be compared against fastbloom-rs. Removes the two work-around
constants and the obsolete clustering caveat.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Core change (encode_series Categorical/Enum → string+foldhash) → Task 2, Step 1. ✓
- Global scope / all consumers inherit → single shared-layer edit; verified by Task 3 Step 7 (full suite). ✓
- Re-include categorical in `CUSTOM_FP_RATE_COLS`/`CROSS_IMPL_FP_COLS` → Task 3, Steps 2-5 (constants removed; tests repointed to `NON_BOOLEAN_COLS` / `SCALAR_NON_BOOLEAN_COLS`, which include categorical). ✓
- Delete "Note on Categorical FP rate testing" caveat → Task 3, Step 1. ✓
- CLAUDE.md foldhash note update → Task 2, Step 5. ✓
- Cross-frame regression test → Task 1. ✓
- Validation: `maturin develop --release` (Task 2 Step 2), full non-slow suite green with categorical inside assertions (Task 3 Steps 6-7), regression test passes (Task 2 Step 3). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" — all steps carry literal code and exact commands. ✓

**Type consistency:** `encode_series` return type unchanged (`EncodedColumn { values: Vec<u64>, is_null: Vec<bool> }`); `hash_one(&FoldHashFixed, &str) -> u64` matches the existing String arm's usage; `NON_BOOLEAN_COLS` and `SCALAR_NON_BOOLEAN_COLS` are pre-existing constants reused, not redefined. ✓
