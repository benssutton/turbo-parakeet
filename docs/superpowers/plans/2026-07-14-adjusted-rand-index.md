# Adjusted Rand Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pairwise_adjusted_rand` Rust Polars plugin computing pairwise ARI across dataframe columns, built on a new shared contingency-table builder that a refactored chi² plugin also consumes; both plugins gain an `n_valid` output field.

**Architecture:** A new `contingency.rs` module builds drop-null contingency tables from `DenseColumn`s (dense dictionary ids, extended with `null_id`) using the flat-array/hash counting strategy proven in `entropy.rs`. `ari.rs` and a refactored `chi_squared.rs` are thin consumers. The Python API mirrors `pairwise_chi_squared` exactly.

**Tech Stack:** Rust (polars 0.51, pyo3-polars, rayon, foldhash), maturin, Python 3.12 (polars, pytest, scikit-learn 1.9).

**Spec:** `docs/superpowers/specs/2026-07-14-adjusted-rand-index-design.md`

## Global Constraints

- Python exe: `C:/Users/Ben/miniconda3/envs/p312/python.exe` (conda env `p312`).
- Build plugin: `cd services/analytics && C:/Users/Ben/miniconda3/envs/p312/python.exe -m maturin develop --release` (~5 min).
- Rust tests (Git Bash syntax; the env vars are REQUIRED — without them the build fails to find Python and the test exe fails with STATUS_DLL_NOT_FOUND):
  `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release <filter>`
- Pytest from repo root: `C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/ -q`.
- No new Cargo or Python dependencies.
- Null policy for ARI and chi²: rows where either column is null are dropped.
- ARI edge conventions (must match sklearn): `n_valid == 0` → NaN; degenerate denominator (`max_index == expected`, incl. `n_valid == 1`) → 1.0.
- `n_valid` output dtype: UInt32 on both plugins.
- Commit after every task; keep the existing comment style (banner sections, explain *why* not *what*).

---

### Task 1: `DenseColumn.null_id` extension

**Files:**
- Modify: `services/analytics/src/shared.rs` (DenseColumn struct ~line 306, `densify` ~line 301, `build_dense_cache_par` placeholder ~line 337)
- Test: `services/analytics/src/entropy.rs` (tests module, `test_densify_nulls_get_own_id`)

**Interfaces:**
- Produces: `DenseColumn { ids: Vec<u32>, card: u32, null_id: Option<u32> }` — `null_id` is the dense id assigned to nulls, `None` if the column has no nulls. Task 2 consumes it to skip null rows.

- [ ] **Step 1: Extend the existing densify test to assert null_id (failing test)**

In `services/analytics/src/entropy.rs`, replace the body of `test_densify_nulls_get_own_id` with:

```rust
    #[test]
    fn test_densify_nulls_get_own_id() {
        // [1, null, 1, 2] → 3 distinct categories (null is one of them).
        let s = Series::new("test".into(), &[Some(1i32), None, Some(1), Some(2)]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.card, 3);
        assert_eq!(dense.ids[0], dense.ids[2]); // 1 == 1
        assert_ne!(dense.ids[0], dense.ids[1]); // 1 != null
        assert_ne!(dense.ids[1], dense.ids[3]); // null != 2
        assert_eq!(dense.null_id, Some(dense.ids[1])); // null id recorded
    }

    #[test]
    fn test_densify_no_nulls_has_no_null_id() {
        let s = Series::new("test".into(), &[1i32, 2, 3]);
        let enc = encode_series(&s).unwrap();
        let dense = densify(&enc);
        assert_eq!(dense.null_id, None);
    }
```

- [ ] **Step 2: Verify it fails to compile**

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release densify 2>&1 | tail -20`
Expected: compile error — `no field null_id on type DenseColumn`.

- [ ] **Step 3: Implement**

In `services/analytics/src/shared.rs`, add the field to `DenseColumn` (keep the existing doc comment, extend it):

```rust
pub(crate) struct DenseColumn {
    pub ids: Vec<u32>,
    pub card: u32,
    /// Dense id assigned to nulls, if the column has any. Consumers with a
    /// drop-null policy (contingency-based stats) skip rows carrying this id;
    /// consumers treating null as a category (entropy) ignore it.
    pub null_id: Option<u32>,
}
```

Update `densify` to record it:

```rust
pub(crate) fn densify(col: &EncodedColumn) -> DenseColumn {
    let mut map: HashMap<(u64, bool), u32, FoldHashFixed> =
        HashMap::with_capacity_and_hasher(col.len(), FoldHashFixed::default());
    let mut ids = Vec::with_capacity(col.len());
    let mut null_id: Option<u32> = None;
    for i in 0..col.len() {
        let next = map.len() as u32;
        let id = *map.entry((col.values[i], col.is_null[i])).or_insert(next);
        if col.is_null[i] && null_id.is_none() {
            null_id = Some(id);
        }
        ids.push(id);
    }
    DenseColumn {
        ids,
        card: map.len() as u32,
        null_id,
    }
}
```

In `build_dense_cache_par`, the placeholder constructor gains the field:

```rust
    let mut cache: Vec<DenseColumn> = (0..inputs.len())
        .map(|_| DenseColumn {
            ids: Vec::new(),
            card: 0,
            null_id: None,
        })
        .collect();
```

- [ ] **Step 4: Run tests**

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release 2>&1 | tail -5`
Expected: all tests pass (76 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add services/analytics/src/shared.rs services/analytics/src/entropy.rs
git commit -m "feat: record null dense id on DenseColumn for drop-null consumers"
```

---

### Task 2: `contingency.rs` — shared contingency-table builder

**Files:**
- Create: `services/analytics/src/contingency.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod contingency;`)

**Interfaces:**
- Consumes: `DenseColumn` (Task 1), incl. `null_id`.
- Produces (Tasks 3 and 4 rely on these exact names):

```rust
pub(crate) struct ContingencyTable {
    pub cells: Vec<(u32, u32, u64)>, // observed non-zero cells (a_id, b_id, count)
    pub marg_a: Vec<u64>,            // indexed by dense id 0..card_a; zero for unseen/null ids
    pub marg_b: Vec<u64>,
    pub n_valid: u64,                // rows where both columns are non-null
}
pub(crate) fn build_contingency(a: &DenseColumn, b: &DenseColumn, n_rows: usize) -> ContingencyTable
```

- [ ] **Step 1: Create the module with failing tests**

Create `services/analytics/src/contingency.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// Shared contingency-table builder — used by chi_squared.rs and ari.rs
// ─────────────────────────────────────────────────────────────────────────────
//
// Both chi-squared and ARI reduce a column pair to the same object: the joint
// distribution over rows where BOTH columns are non-null (pairwise deletion),
// plus the two marginals. This module builds that table once so the two
// consumers cannot drift apart on counting or null policy.
//
// Counting reuses the dense-id strategy from entropy.rs: the joint key is
// a_id·Kb + b_id (a single integer, no tuple hashing). Joint spaces ≤ FLAT_MAX
// are counted in a flat thread-local array with a touched-slot list for
// O(distinct) reset; larger spaces fall back to a u64-keyed hash map. The pair
// product always fits u64 because dense ids are u32.

use foldhash::fast::RandomState as FoldHashFast;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::shared::DenseColumn;

pub(crate) struct ContingencyTable {
    /// Observed non-zero cells: (a_id, b_id, count).
    pub cells: Vec<(u32, u32, u64)>,
    /// Marginal counts over valid rows, indexed by dense id (0..card).
    /// Ids unseen among valid rows (including a null id) hold 0.
    pub marg_a: Vec<u64>,
    pub marg_b: Vec<u64>,
    /// Rows where both columns are non-null.
    pub n_valid: u64,
}

const FLAT_MAX: u64 = 1 << 20; // 4 MB of u32 counts per worker thread

thread_local! {
    static FLAT_COUNTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static TOUCHED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static FREQ: RefCell<HashMap<u64, u64, FoldHashFast>> =
        RefCell::new(HashMap::with_hasher(FoldHashFast::default()));
}

/// Build the drop-null contingency table for one column pair.
///
/// Rows where either column carries its null id are excluded from cells,
/// marginals, and `n_valid`.
pub(crate) fn build_contingency(
    a: &DenseColumn,
    b: &DenseColumn,
    n_rows: usize,
) -> ContingencyTable {
    let kb = b.card as u64;
    let mut marg_a = vec![0u64; a.card as usize];
    let mut marg_b = vec![0u64; b.card as usize];
    let mut n_valid = 0u64;

    // Dense ids run 0..card-1, so u32::MAX can never be a real id — it is a
    // safe "no null id" sentinel that lets the hot loop use a plain compare.
    let a_null = a.null_id.unwrap_or(u32::MAX);
    let b_null = b.null_id.unwrap_or(u32::MAX);

    let space = (a.card as u64) * kb; // fits u64: both factors are u32

    if space <= FLAT_MAX {
        FLAT_COUNTS.with(|counts_cell| {
            TOUCHED.with(|touched_cell| {
                let mut counts = counts_cell.borrow_mut();
                let mut touched = touched_cell.borrow_mut();
                if counts.len() < space as usize {
                    counts.resize(space as usize, 0);
                }
                for i in 0..n_rows {
                    let (ai, bi) = (a.ids[i], b.ids[i]);
                    if ai == a_null || bi == b_null {
                        continue;
                    }
                    marg_a[ai as usize] += 1;
                    marg_b[bi as usize] += 1;
                    n_valid += 1;
                    let key = (ai as u64 * kb + bi as u64) as usize;
                    let slot = &mut counts[key];
                    if *slot == 0 {
                        touched.push(key as u32);
                    }
                    *slot += 1;
                }
                let mut cells = Vec::with_capacity(touched.len());
                for &t in touched.iter() {
                    let cnt = counts[t as usize] as u64;
                    cells.push(((t as u64 / kb) as u32, (t as u64 % kb) as u32, cnt));
                    counts[t as usize] = 0;
                }
                touched.clear();
                ContingencyTable {
                    cells,
                    marg_a,
                    marg_b,
                    n_valid,
                }
            })
        })
    } else {
        FREQ.with(|freq_cell| {
            let mut freq = freq_cell.borrow_mut();
            freq.clear();
            for i in 0..n_rows {
                let (ai, bi) = (a.ids[i], b.ids[i]);
                if ai == a_null || bi == b_null {
                    continue;
                }
                marg_a[ai as usize] += 1;
                marg_b[bi as usize] += 1;
                n_valid += 1;
                *freq.entry(ai as u64 * kb + bi as u64).or_insert(0) += 1;
            }
            let cells = freq
                .iter()
                .map(|(&k, &c)| ((k / kb) as u32, (k % kb) as u32, c))
                .collect();
            ContingencyTable {
                cells,
                marg_a,
                marg_b,
                n_valid,
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{densify, encode_series};
    use polars::prelude::*;
    use std::collections::HashMap as StdHashMap;

    fn dense(series: &Series) -> DenseColumn {
        densify(&encode_series(series).unwrap())
    }

    /// Collect cells into a map keyed by (a_id, b_id) since cell order is unspecified.
    fn cell_map(t: &ContingencyTable) -> StdHashMap<(u32, u32), u64> {
        t.cells.iter().map(|&(a, b, c)| ((a, b), c)).collect()
    }

    #[test]
    fn test_basic_counts() {
        // a=[0,0,1,2], b=[0,0,1,1] → cells {(0,0):2, (1,1):1, (2,1):1}, n=4.
        let a = dense(&Series::new("a".into(), &[0i32, 0, 1, 2]));
        let b = dense(&Series::new("b".into(), &[0i32, 0, 1, 1]));
        let t = build_contingency(&a, &b, 4);
        assert_eq!(t.n_valid, 4);
        assert_eq!(t.cells.len(), 3);
        let cells = cell_map(&t);
        // Dense ids follow first-appearance order: a → {0:0, 1:1, 2:2}, b → {0:0, 1:1}.
        assert_eq!(cells[&(0, 0)], 2);
        assert_eq!(cells[&(1, 1)], 1);
        assert_eq!(cells[&(2, 1)], 1);
        assert_eq!(t.marg_a, vec![2, 1, 1]);
        assert_eq!(t.marg_b, vec![2, 2]);
    }

    #[test]
    fn test_null_rows_dropped() {
        // Row 2 (a null) and row 3 (b null) are excluded everywhere.
        let a = dense(&Series::new("a".into(), &[Some(1i32), Some(1), None, Some(2), Some(2)]));
        let b = dense(&Series::new("b".into(), &[Some(7i32), Some(7), Some(7), None, Some(8)]));
        let t = build_contingency(&a, &b, 5);
        assert_eq!(t.n_valid, 3);
        let total_cells: u64 = t.cells.iter().map(|&(_, _, c)| c).sum();
        assert_eq!(total_cells, 3);
        // Null ids hold zero marginal counts.
        let a_null = a.null_id.unwrap() as usize;
        let b_null = b.null_id.unwrap() as usize;
        assert_eq!(t.marg_a[a_null], 0);
        assert_eq!(t.marg_b[b_null], 0);
    }

    #[test]
    fn test_sums_consistent() {
        // Σ cells == Σ marg_a == Σ marg_b == n_valid, on data with nulls.
        let a = dense(&Series::new(
            "a".into(),
            &(0..100).map(|i| if i % 7 == 0 { None } else { Some(i % 5) }).collect::<Vec<Option<i32>>>(),
        ));
        let b = dense(&Series::new(
            "b".into(),
            &(0..100).map(|i| if i % 11 == 0 { None } else { Some(i % 3) }).collect::<Vec<Option<i32>>>(),
        ));
        let t = build_contingency(&a, &b, 100);
        let s_cells: u64 = t.cells.iter().map(|&(_, _, c)| c).sum();
        let s_a: u64 = t.marg_a.iter().sum();
        let s_b: u64 = t.marg_b.iter().sum();
        assert_eq!(s_cells, t.n_valid);
        assert_eq!(s_a, t.n_valid);
        assert_eq!(s_b, t.n_valid);
    }

    #[test]
    fn test_flat_and_hash_paths_agree() {
        // 1500 distinct ids per column → space 2.25M > FLAT_MAX → hash path.
        // Verify against a brute-force reference; also run a truncated (small)
        // version of the same data through the flat path and cross-check.
        let va: Vec<i32> = (0..1500).collect();
        let vb: Vec<i32> = (0..1500).map(|i| (i * 7) % 1500).collect();
        let a = dense(&Series::new("a".into(), &va));
        let b = dense(&Series::new("b".into(), &vb));
        let t = build_contingency(&a, &b, 1500);
        assert_eq!(t.n_valid, 1500);

        let mut reference: StdHashMap<(u32, u32), u64> = StdHashMap::new();
        for i in 0..1500 {
            *reference.entry((a.ids[i], b.ids[i])).or_insert(0) += 1;
        }
        assert_eq!(cell_map(&t), reference);
    }

    #[test]
    fn test_all_null_column() {
        let a = dense(&Series::new("a".into(), &[None::<i32>, None, None]));
        let b = dense(&Series::new("b".into(), &[1i32, 2, 3]));
        let t = build_contingency(&a, &b, 3);
        assert_eq!(t.n_valid, 0);
        assert!(t.cells.is_empty());
        assert!(t.marg_b.iter().all(|&c| c == 0));
    }
}
```

- [ ] **Step 2: Register the module and verify tests fail before, pass after**

In `services/analytics/src/lib.rs` add `mod contingency;` after `mod chi_squared;`:

```rust
mod bloomfilter;
mod minhash;
mod shared;
mod entropy;
mod chi_squared;
mod contingency;
```

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release contingency 2>&1 | tail -10`
Expected: 5 tests pass. (A `dead_code` warning for `build_contingency` is expected until Task 3 consumes it.)

- [ ] **Step 3: Commit**

```bash
git add services/analytics/src/contingency.rs services/analytics/src/lib.rs
git commit -m "feat: add shared drop-null contingency-table builder"
```

---

### Task 3: chi² refactor onto `ContingencyTable` + `n_valid` output field

**Files:**
- Modify: `services/analytics/src/chi_squared.rs`

**Interfaces:**
- Consumes: `build_contingency`/`ContingencyTable` (Task 2), `build_dense_cache_par` (shared.rs).
- Produces: `pairwise_chi_squared` output struct gains field `n_valid: UInt32` (after `low_expected_count`). Python-side docstring updated in Task 5. All numeric outputs unchanged.

- [ ] **Step 1: Add a failing test for n_valid**

In the tests module of `services/analytics/src/chi_squared.rs`:

```rust
    #[test]
    fn test_n_valid_counts_non_null_overlap() {
        // Rows 2 (a null) and 3 (b null) dropped → n_valid = 4.
        let s1 = Series::new("a".into(), &[Some(0i32), Some(0), None, Some(1), Some(1), Some(0)]);
        let s2 = Series::new("b".into(), &[Some(0i32), Some(1), Some(0), None, Some(1), Some(0)]);
        let result = pairwise_chi_squared_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_chi_squared"]).unwrap();
        let n_valid = df.column("n_valid").unwrap().u32().unwrap().get(0).unwrap();
        assert_eq!(n_valid, 4);
    }
```

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release chi_squared 2>&1 | tail -10`
Expected: FAIL — `n_valid` column not found.

- [ ] **Step 2: Refactor**

In `services/analytics/src/chi_squared.rs`:

**(a)** Replace the imports at the top:

```rust
use crate::contingency::{ContingencyTable, build_contingency};
use crate::shared::{PairwiseKwargs, build_dense_cache_par, resolve_pairs};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{HashMap, HashSet};
```

(`foldhash` and `EncodedColumn` imports go away — the ad-hoc maps are gone.)

**(b)** In `chi_squared_output_type`, add the field after `low_expected_count`:

```rust
        Field::new("n_valid".into(), DataType::UInt32),
```

**(c)** In `pairwise_chi_squared_impl`, switch the cache and the per-pair body:

```rust
    // Build parallel dense cache (encode + dictionary-encode needed columns).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Compute chi-squared stats in parallel across all pairs.
    let results: Vec<(String, String, f64, f64, f64, bool, u32)> = pairs
        .par_iter()
        .map(|(i, j)| {
            let table = build_contingency(&cache[*i], &cache[*j], r);
            let (chi2, p, v, low_exp) = compute_chi_squared(&table);
            (
                col_names[*i].clone(),
                col_names[*j].clone(),
                chi2,
                p,
                v,
                low_exp,
                table.n_valid as u32,
            )
        })
        .collect();
```

**(d)** Extend the output assembly (add alongside the existing vectors):

```rust
    let mut n_valid_values = Vec::with_capacity(n_pairs);

    for (a, b, chi2, p, v, low_exp, n_valid) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        chi2_values.push(chi2);
        p_values.push(p);
        cramers_v_values.push(v);
        low_exp_values.push(low_exp);
        n_valid_values.push(n_valid);
    }
    // ...
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), n_valid_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        n_pairs,
        [col_a_s, col_b_s, chi2_s, p_s, v_s, low_exp_s, n_valid_s].iter(),
    )?;
```

**(e)** Replace `compute_chi_squared` with the table-consuming version. All downstream math is line-for-line the old logic; only the counting source changed:

```rust
/// Compute chi-squared statistic, p-value, Cramer's V, and low-expected-count flag
/// from a drop-null contingency table.
///
/// Null policy: rows where either column is null are dropped (pairwise deletion)
/// by build_contingency. Note this differs from the entropy plugin, which treats
/// null as its own category — keep that in mind when deriving mutual information
/// from the two outputs.
///
/// Returns (chi2_stat, p_value, cramers_v, low_expected_count); NaN for degenerate
/// inputs. `low_expected_count` is true when the minimum expected cell count (rarest
/// row marginal × rarest col marginal / N) is below 5, the standard threshold above
/// which the chi-squared approximation is reliable.
fn compute_chi_squared(t: &ContingencyTable) -> (f64, f64, f64, bool) {
    if t.n_valid == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, false);
    }

    // Marginal entries can be zero (ids seen only in dropped rows, or a null id);
    // uniqueness and minima consider non-zero entries only.
    let unique_a = t.marg_a.iter().filter(|&&c| c > 0).count();
    let unique_b = t.marg_b.iter().filter(|&&c| c > 0).count();

    // Degenerate: constant column — chi-squared is undefined.
    if unique_a < 2 || unique_b < 2 {
        return (f64::NAN, f64::NAN, f64::NAN, false);
    }

    let n_f = t.n_valid as f64;

    // χ² = Σ O²/E − N, summing only over observed (non-zero) cells since zero-obs
    // cells contribute 0 to Σ O²/E. This is O(observed pairs) rather than the
    // O(unique_a × unique_b) nested-loop form, which hangs on high-cardinality columns.
    let mut chi2_stat = 0.0f64;
    for &(ai, bi, obs) in &t.cells {
        let expected = t.marg_a[ai as usize] as f64 * t.marg_b[bi as usize] as f64 / n_f;
        chi2_stat += (obs as f64 * obs as f64) / expected;
    }
    chi2_stat -= n_f;

    // low_expected_count: true when the minimum possible expected cell count falls
    // below 5, the standard chi-squared validity threshold.
    let min_row = *t.marg_a.iter().filter(|&&c| c > 0).min().unwrap() as f64;
    let min_col = *t.marg_b.iter().filter(|&&c| c > 0).min().unwrap() as f64;
    let low_expected_count = min_row * min_col / n_f < 5.0;

    // Degrees of freedom = (unique_a - 1) * (unique_b - 1).
    let df_val = ((unique_a - 1) * (unique_b - 1)) as f64;

    // Use sf (survival function) rather than 1 - cdf to avoid catastrophic
    // cancellation in the far tail where χ² is most significant.
    let p_value = match ChiSquared::new(df_val) {
        Ok(dist) => dist.sf(chi2_stat),
        Err(_) => f64::NAN,
    };

    let min_dim = (unique_a - 1).min(unique_b - 1) as f64;
    let cramers_v = (chi2_stat / (n_f * min_dim)).sqrt();

    (chi2_stat, p_value, cramers_v, low_expected_count)
}
```

**(f)** In `build_empty_result`, add the empty field and include it in the struct:

```rust
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), vec![]).into_series();
    let struct_ca = StructChunked::from_series(
        "pairwise_chi_squared".into(),
        0,
        [col_a_s, col_b_s, chi2_s, p_s, v_s, low_exp_s, n_valid_s].iter(),
    )?;
```

- [ ] **Step 3: Run the full Rust suite**

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release 2>&1 | tail -5`
Expected: all pass, including every pre-existing chi² test unchanged (numeric behavior identical) and the new `test_n_valid_counts_non_null_overlap`.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/src/chi_squared.rs
git commit -m "refactor: chi-squared consumes shared contingency table, adds n_valid field"
```

---

### Task 4: `ari.rs` — pairwise Adjusted Rand Index plugin

**Files:**
- Create: `services/analytics/src/ari.rs`
- Modify: `services/analytics/src/lib.rs` (add `mod ari;`)

**Interfaces:**
- Consumes: `build_contingency`/`ContingencyTable` (Task 2), `PairwiseKwargs`, `build_dense_cache_par`, `resolve_pairs` (shared.rs).
- Produces: plugin function `pairwise_adjusted_rand` returning struct `pairwise_adjusted_rand { col_a: String, col_b: String, ari: Float64, n_valid: UInt32 }`. Task 5's Python wrapper registers it by this exact name.

- [ ] **Step 1: Create the module (tests included — they fail to compile until the impl exists, which is the failing state)**

Create `services/analytics/src/ari.rs`:

```rust
// ─────────────────────────────────────────────────────────────────────────────
// Pairwise Adjusted Rand Index
// ─────────────────────────────────────────────────────────────────────────────
//
// ARI measures agreement between two partitions of the same rows, corrected
// for chance. Each column is a partition (rows sharing a value are one
// cluster). ARI = 1 → identical partitions; ≈ 0 → chance-level agreement;
// can go negative (floor −0.5) for worse-than-chance.
//
// Null policy: rows where either column is null are dropped (pairwise
// deletion, via the shared contingency builder) — a null row belongs to no
// cluster, so it must not vote on partition agreement.

use crate::contingency::{ContingencyTable, build_contingency};
use crate::shared::{PairwiseKwargs, build_dense_cache_par, resolve_pairs};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Output type
// ─────────────────────────────────────────────────────────────────────────────

fn ari_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("col_a".into(), DataType::String),
        Field::new("col_b".into(), DataType::String),
        Field::new("ari".into(), DataType::Float64),
        Field::new("n_valid".into(), DataType::UInt32),
    ];
    Ok(Field::new(
        "pairwise_adjusted_rand".into(),
        DataType::Struct(fields),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// ARI from a drop-null contingency table.
///
/// With index = Σᵢⱼ C(nᵢⱼ,2), A = Σᵢ C(aᵢ,2), B = Σⱼ C(bⱼ,2), total = C(n,2):
///   ARI = (index − A·B/total) / ((A+B)/2 − A·B/total)
///
/// Sums accumulate in u64 (each is ≤ C(n,2); the C(x,2) multiply overflows
/// only past ~4×10⁹ rows). The final expression is f64 because A·B overflows
/// u64. Edge conventions match sklearn.metrics.adjusted_rand_score:
///   - n_valid == 0 → NaN (no data, undefined)
///   - degenerate denominator (max_index == expected, e.g. both columns
///     constant, both all-singletons, or n_valid == 1) → 1.0: sklearn's
///     "trivially perfect agreement" convention (its fn == 0 && fp == 0 branch).
fn compute_ari(t: &ContingencyTable) -> f64 {
    if t.n_valid == 0 {
        return f64::NAN;
    }

    #[inline]
    fn comb2(x: u64) -> u64 {
        x * (x - 1) / 2
    }

    let index: u64 = t.cells.iter().map(|&(_, _, c)| comb2(c)).sum();
    let a_sum: u64 = t.marg_a.iter().map(|&c| comb2(c)).sum();
    let b_sum: u64 = t.marg_b.iter().map(|&c| comb2(c)).sum();
    let total = comb2(t.n_valid) as f64;

    // n_valid == 1: no pairs exist, both partitions trivially identical.
    if total == 0.0 {
        return 1.0;
    }

    let expected = a_sum as f64 * b_sum as f64 / total;
    let max_index = (a_sum as f64 + b_sum as f64) / 2.0;

    if max_index == expected {
        return 1.0;
    }

    (index as f64 - expected) / (max_index - expected)
}

pub(crate) fn pairwise_adjusted_rand_impl(
    inputs: &[Series],
    kwargs: PairwiseKwargs,
) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "pairwise_adjusted_rand requires at least one column".into(),
        ));
    }

    let n_cols = inputs.len();

    // Fewer than 2 columns → no pairs, return empty struct.
    if n_cols < 2 {
        return build_empty_result();
    }

    let r = inputs[0].len();
    if r == 0 {
        return Err(PolarsError::ComputeError(
            "Cannot calculate ARI on empty columns".into(),
        ));
    }

    // Validate uniform length.
    for (idx, series) in inputs.iter().enumerate() {
        if series.len() != r {
            return Err(PolarsError::ShapeMismatch(
                format!(
                    "All columns must have the same length: column {} has length {} but expected {}",
                    idx,
                    series.len(),
                    r
                )
                .into(),
            ));
        }
    }

    // Resolve pairs from kwargs or generate all N-choose-2.
    let pairs: Vec<(usize, usize)> = match &kwargs.pairs {
        Some(raw_pairs) => {
            let name_map: HashMap<String, usize> = inputs
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name().to_string(), i))
                .collect();
            resolve_pairs(raw_pairs, &name_map)?
        }
        None => (0..n_cols)
            .flat_map(|i| ((i + 1)..n_cols).map(move |j| (i, j)))
            .collect(),
    };

    let n_pairs = pairs.len();

    // Build parallel dense cache (encode + dictionary-encode needed columns).
    let needed: HashSet<usize> = pairs.iter().flat_map(|(i, j)| [*i, *j]).collect();
    let cache = build_dense_cache_par(inputs, &needed)?;

    let col_names: Vec<String> = inputs.iter().map(|s| s.name().to_string()).collect();

    // Compute ARI in parallel across all pairs.
    let results: Vec<(String, String, f64, u32)> = pairs
        .par_iter()
        .map(|(i, j)| {
            let table = build_contingency(&cache[*i], &cache[*j], r);
            let ari = compute_ari(&table);
            (
                col_names[*i].clone(),
                col_names[*j].clone(),
                ari,
                table.n_valid as u32,
            )
        })
        .collect();

    // Build output struct series.
    let mut col_a_names = Vec::with_capacity(n_pairs);
    let mut col_b_names = Vec::with_capacity(n_pairs);
    let mut ari_values = Vec::with_capacity(n_pairs);
    let mut n_valid_values = Vec::with_capacity(n_pairs);

    for (a, b, ari, n_valid) in results {
        col_a_names.push(a);
        col_b_names.push(b);
        ari_values.push(ari);
        n_valid_values.push(n_valid);
    }

    let col_a_s = StringChunked::from_iter(col_a_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(col_b_names.iter().map(|s| s.as_str()))
        .into_series()
        .with_name("col_b".into());
    let ari_s = Float64Chunked::from_vec("ari".into(), ari_values).into_series();
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), n_valid_values).into_series();

    let struct_ca = StructChunked::from_series(
        "pairwise_adjusted_rand".into(),
        n_pairs,
        [col_a_s, col_b_s, ari_s, n_valid_s].iter(),
    )?;

    Ok(struct_ca.into_series())
}

fn build_empty_result() -> PolarsResult<Series> {
    let col_a_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_a".into());
    let col_b_s = StringChunked::from_iter(std::iter::empty::<&str>())
        .into_series()
        .with_name("col_b".into());
    let ari_s = Float64Chunked::from_vec("ari".into(), vec![]).into_series();
    let n_valid_s = UInt32Chunked::from_vec("n_valid".into(), vec![]).into_series();
    let struct_ca = StructChunked::from_series(
        "pairwise_adjusted_rand".into(),
        0,
        [col_a_s, col_b_s, ari_s, n_valid_s].iter(),
    )?;
    Ok(struct_ca.into_series())
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin entry point
// ─────────────────────────────────────────────────────────────────────────────

#[polars_expr(output_type_func=ari_output_type)]
fn pairwise_adjusted_rand(inputs: &[Series], kwargs: PairwiseKwargs) -> PolarsResult<Series> {
    pairwise_adjusted_rand_impl(inputs, kwargs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn no_pairs() -> PairwiseKwargs {
        PairwiseKwargs { pairs: None }
    }

    fn ari_of(s1: Series, s2: Series) -> (f64, u32) {
        let result = pairwise_adjusted_rand_impl(&[s1, s2], no_pairs()).unwrap();
        let df = result.into_frame().unnest(["pairwise_adjusted_rand"]).unwrap();
        let ari = df.column("ari").unwrap().f64().unwrap().get(0).unwrap();
        let n_valid = df.column("n_valid").unwrap().u32().unwrap().get(0).unwrap();
        (ari, n_valid)
    }

    // sklearn doc example: adjusted_rand_score([0,0,1,2], [0,0,1,1]) = 4/7 ≈ 0.5714.
    #[test]
    fn test_sklearn_doc_example() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 2]);
        let s2 = Series::new("b".into(), &[0i32, 0, 1, 1]);
        let (ari, n_valid) = ari_of(s1, s2);
        let expected = 4.0 / 7.0;
        assert!((ari - expected).abs() < 1e-12, "Expected {}, got {}", expected, ari);
        assert_eq!(n_valid, 4);
    }

    // Identical partitions → 1.0 (label values need not match, only the grouping).
    #[test]
    fn test_identical_partitions() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[5i32, 5, 9, 9, 7, 7]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari - 1.0).abs() < 1e-12, "Expected 1.0, got {}", ari);
    }

    // Maximal disagreement on a 2x2 crossing: sklearn gives exactly -0.5.
    #[test]
    fn test_crossed_partitions_negative() {
        let s1 = Series::new("a".into(), &[0i32, 0, 1, 1]);
        let s2 = Series::new("b".into(), &[0i32, 1, 0, 1]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari + 0.5).abs() < 1e-12, "Expected -0.5, got {}", ari);
    }

    // Null rows dropped: appending a null-containing row to the sklearn doc
    // example must not change the score.
    #[test]
    fn test_null_rows_dropped() {
        let s1 = Series::new("a".into(), &[Some(0i32), Some(0), Some(1), Some(2), None]);
        let s2 = Series::new("b".into(), &[Some(0i32), Some(0), Some(1), Some(1), Some(9)]);
        let (ari, n_valid) = ari_of(s1, s2);
        let expected = 4.0 / 7.0;
        assert!((ari - expected).abs() < 1e-12, "Expected {}, got {}", expected, ari);
        assert_eq!(n_valid, 4);
    }

    // Both columns constant → degenerate denominator → 1.0 (sklearn convention).
    #[test]
    fn test_both_constant_is_one() {
        let s1 = Series::new("a".into(), &[1i32, 1, 1, 1]);
        let s2 = Series::new("b".into(), &[2i32, 2, 2, 2]);
        let (ari, _) = ari_of(s1, s2);
        assert!((ari - 1.0).abs() < 1e-12, "Expected 1.0, got {}", ari);
    }

    // No non-null overlap → NaN, n_valid 0.
    #[test]
    fn test_disjoint_nulls_nan() {
        let s1 = Series::new("a".into(), &[Some(1i32), Some(2), None, None]);
        let s2 = Series::new("b".into(), &[None::<i32>, None, Some(1), Some(2)]);
        let (ari, n_valid) = ari_of(s1, s2);
        assert!(ari.is_nan(), "Expected NaN, got {}", ari);
        assert_eq!(n_valid, 0);
    }

    #[test]
    fn test_three_columns_three_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let result = pairwise_adjusted_rand_impl(&[s1, s2, s3], no_pairs()).unwrap();
        assert_eq!(result.len(), 3); // 3-choose-2
    }

    #[test]
    fn test_specific_pairs() {
        let s1 = Series::new("a".into(), &[1i32, 1, 2, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2, 1, 2]);
        let s3 = Series::new("c".into(), &[1i32, 1, 1, 2]);
        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "c".to_string()]]),
        };
        let result = pairwise_adjusted_rand_impl(&[s1, s2, s3], kwargs).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_invalid_pair_column_errors() {
        let s1 = Series::new("a".into(), &[1i32, 2]);
        let s2 = Series::new("b".into(), &[1i32, 2]);
        let kwargs = PairwiseKwargs {
            pairs: Some(vec![vec!["a".to_string(), "nope".to_string()]]),
        };
        assert!(pairwise_adjusted_rand_impl(&[s1, s2], kwargs).is_err());
    }

    #[test]
    fn test_empty_inputs_error() {
        assert!(pairwise_adjusted_rand_impl(&[], no_pairs()).is_err());
    }

    #[test]
    fn test_single_column_empty_result() {
        let s1 = Series::new("a".into(), &[1i32, 2, 3]);
        let result = pairwise_adjusted_rand_impl(&[s1], no_pairs()).unwrap();
        assert_eq!(result.len(), 0);
    }
}
```

- [ ] **Step 2: Register and run**

In `services/analytics/src/lib.rs`:

```rust
mod bloomfilter;
mod minhash;
mod shared;
mod entropy;
mod chi_squared;
mod contingency;
mod ari;
```

Run: `cd services/analytics && PATH="/c/Users/Ben/miniconda3/envs/p312:$PATH" PYO3_PYTHON="C:/Users/Ben/miniconda3/envs/p312/python.exe" cargo test --release ari 2>&1 | tail -15`
Expected: 11 ARI tests pass. Then run the full suite (`cargo test --release`) — everything passes.

- [ ] **Step 3: Commit**

```bash
git add services/analytics/src/ari.rs services/analytics/src/lib.rs
git commit -m "feat: add pairwise adjusted rand index plugin"
```

---

### Task 5: Python API + plugin build

**Files:**
- Modify: `services/analytics/analytics/__init__.py` (new function after `pairwise_chi_squared`, ~line 268; chi² docstring fields list ~line 246-251)

**Interfaces:**
- Consumes: Rust plugin function name `pairwise_adjusted_rand` (Task 4).
- Produces: `analytics.pairwise_adjusted_rand(df, pairs=None) -> pl.DataFrame` — Tasks 6 and 7 import this.

- [ ] **Step 1: Add the wrapper**

In `services/analytics/analytics/__init__.py`, after the `pairwise_chi_squared` function add:

```python
"""
Adjusted Rand Index
"""

def pairwise_adjusted_rand(
    df: pl.DataFrame | pl.LazyFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """
    Calculate pairwise Adjusted Rand Index using the Rust plugin.

    Each column is treated as a partition of the rows (rows sharing a value
    form one cluster). ARI measures chance-corrected agreement between two
    partitions: 1.0 = identical partitions, ~0 = chance-level agreement,
    negative (floor -0.5) = worse than chance.

    Null policy: rows where either column is null are dropped (pairwise
    deletion). Edge conventions match sklearn.metrics.adjusted_rand_score:
    no overlapping non-null rows -> NaN; degenerate denominator (e.g. both
    columns constant) -> 1.0.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.
    pairs : list of (str, str) tuples, optional
        Specific column pairs to score.
        When None (default), computes all N-choose-2 combinations.

    Returns
    -------
    pl.DataFrame
        Single column "pairwise_adjusted_rand" containing structs with:
        - col_a: String - First column name
        - col_b: String - Second column name
        - ari: f64 - Adjusted Rand Index
        - n_valid: u32 - Rows remaining after null-dropping
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs_dict = {
        "pairs": [list(p) for p in pairs] if pairs is not None else None
    }

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="pairwise_adjusted_rand",
            args=df.get_columns(),
            kwargs=kwargs_dict,
            is_elementwise=False,
        ).alias("pairwise_adjusted_rand")
    )
```

Also extend the `pairwise_chi_squared` docstring's returns list with:

```python
        - low_expected_count: bool - True when min expected cell count < 5
        - n_valid: u32 - Rows remaining after null-dropping
```

(The `low_expected_count` line is currently missing from the docstring too — add both.)

- [ ] **Step 2: Build the plugin**

Run: `cd services/analytics && C:/Users/Ben/miniconda3/envs/p312/python.exe -m maturin develop --release 2>&1 | tail -3`
Expected: `Installed analytics-0.1` (~5 min; the `PyInit_analytics` warning is pre-existing and harmless).

- [ ] **Step 3: Smoke test**

Run:
```bash
C:/Users/Ben/miniconda3/envs/p312/python.exe -c "
import polars as pl
from analytics import pairwise_adjusted_rand, pairwise_chi_squared
df = pl.DataFrame({'a': [0, 0, 1, 2], 'b': [0, 0, 1, 1]})
print(pairwise_adjusted_rand(df).unnest('pairwise_adjusted_rand'))
print(pairwise_chi_squared(df).unnest('pairwise_chi_squared').columns)
"
```
Expected: one row with `ari ≈ 0.571429`, `n_valid = 4`; chi² columns include `n_valid`.

- [ ] **Step 4: Commit**

```bash
git add services/analytics/analytics/__init__.py
git commit -m "feat: expose pairwise_adjusted_rand Python API, document chi2 n_valid"
```

---

### Task 6: Pytest correctness vs scikit-learn + chi² regression

**Files:**
- Create: `tests/test_adjusted_rand.py`

**Interfaces:**
- Consumes: `analytics.pairwise_adjusted_rand` (Task 5), `conftest.dataset` fixture (18 columns, 1000 rows).

- [ ] **Step 1: Write the test file**

Create `tests/test_adjusted_rand.py`:

```python
"""
Pairwise Adjusted Rand Index correctness tests: Rust plugin vs scikit-learn.

Eligible columns from the shared 18-column dataset (see conftest.make_dataset):
    boolean_*   (3 columns) — Boolean
    uint32_*    (3 columns) — UInt32
    cat_*       (3 columns) — Categorical
    float64_*   (3 columns) — Float64 (discrete labels; ARI is defined for any labelling)

Excluded: list_* and arr_* (nested types; sklearn cannot label them).

12 eligible columns → C(12,2) = 66 pairs tested.

The reference applies the plugin's null policy (drop rows where either column
is null) before calling sklearn.metrics.adjusted_rand_score. Labels are cast
to String so every dtype becomes uniform hashable labels — an injective
relabelling, which ARI is invariant to.

Skipped automatically if scikit-learn is not installed.
"""

import math
from itertools import combinations

import polars as pl
import pytest

sk_metrics = pytest.importorskip("sklearn.metrics")

_analytics = pytest.importorskip("analytics")
pairwise_adjusted_rand = _analytics.pairwise_adjusted_rand


_ARI_ELIGIBLE_PREFIXES = ("boolean_", "uint32_", "cat_", "float64_")


def ari_eligible_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(_ARI_ELIGIBLE_PREFIXES)]


def sklearn_ari(df: pl.DataFrame, col_a: str, col_b: str) -> tuple[float, int]:
    """Reference ARI with the plugin's drop-null policy. Returns (ari, n_valid)."""
    sub = df.select([col_a, col_b]).drop_nulls()
    n_valid = sub.height
    if n_valid == 0:
        return float("nan"), 0
    labels_a = sub[col_a].cast(pl.String).to_list()
    labels_b = sub[col_b].cast(pl.String).to_list()
    return sk_metrics.adjusted_rand_score(labels_a, labels_b), n_valid


def test_matches_sklearn_on_shared_dataset(dataset: pl.DataFrame) -> None:
    cols = ari_eligible_columns(dataset)
    assert len(cols) == 12
    sub_df = dataset.select(cols)

    result = pairwise_adjusted_rand(sub_df)
    rows = result.unnest(result.columns[0])
    assert rows.height == 66  # C(12,2)

    failures = []
    for row in rows.iter_rows(named=True):
        col_a, col_b = row["col_a"], row["col_b"]
        expected_ari, expected_n = sklearn_ari(sub_df, col_a, col_b)

        if row["n_valid"] != expected_n:
            failures.append(
                f"  ({col_a}, {col_b}): n_valid {row['n_valid']} != {expected_n}"
            )
            continue
        if math.isnan(expected_ari) != math.isnan(row["ari"]):
            failures.append(
                f"  ({col_a}, {col_b}): NaN mismatch rust={row['ari']} sklearn={expected_ari}"
            )
            continue
        if math.isnan(expected_ari):
            continue
        # Both sides count exact integers; only the final division is float.
        # abs_tol covers ARI values at/near zero where rel_tol is meaningless.
        if not math.isclose(row["ari"], expected_ari, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(
                f"  ({col_a}, {col_b}): ari rust={row['ari']} sklearn={expected_ari}"
            )

    assert not failures, "ARI mismatches vs sklearn:\n" + "\n".join(failures)


def test_identical_column_is_one(dataset: pl.DataFrame) -> None:
    df = dataset.select(
        pl.col("uint32_skewed").alias("x"),
        pl.col("uint32_skewed").alias("y"),
    )
    result = pairwise_adjusted_rand(df)
    row = result.unnest(result.columns[0]).row(0, named=True)
    assert math.isclose(row["ari"], 1.0, rel_tol=1e-12)


def test_specific_pairs_subset(dataset: pl.DataFrame) -> None:
    cols = ari_eligible_columns(dataset)
    wanted = [(cols[0], cols[1]), (cols[0], cols[2])]
    result = pairwise_adjusted_rand(dataset.select(cols), pairs=wanted)
    rows = result.unnest(result.columns[0])
    assert rows.height == 2
    assert set(zip(rows["col_a"], rows["col_b"])) == set(wanted)


def test_constant_columns_convention() -> None:
    # Both constant → sklearn returns 1.0; plugin must agree.
    df = pl.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2]})
    row = pairwise_adjusted_rand(df).unnest("pairwise_adjusted_rand").row(0, named=True)
    assert math.isclose(row["ari"], 1.0, rel_tol=1e-12)
    assert sk_metrics.adjusted_rand_score(df["a"].to_list(), df["b"].to_list()) == 1.0


def test_no_overlap_is_nan() -> None:
    df = pl.DataFrame({"a": [1, 2, None, None], "b": [None, None, 1, 2]})
    row = pairwise_adjusted_rand(df).unnest("pairwise_adjusted_rand").row(0, named=True)
    assert math.isnan(row["ari"])
    assert row["n_valid"] == 0
```

- [ ] **Step 2: Run the new tests**

Run: `cd <repo root> && C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/test_adjusted_rand.py -q`
Expected: 5 passed.

- [ ] **Step 3: Chi² and full-suite regression**

Run: `C:/Users/Ben/miniconda3/envs/p312/python.exe -m pytest tests/ -q`
Expected: all pass (66 pre-existing + 5 new), specifically `tests/test_chi_squared.py` unchanged and green — this validates the chi² refactor end-to-end against polars-ds at rtol=1e-4.

- [ ] **Step 4: Commit**

```bash
git add tests/test_adjusted_rand.py
git commit -m "test: ARI correctness vs scikit-learn, chi2 regression green"
```

---

### Task 7: Speed benchmark vs scikit-learn + chi² benchmark regression

**Files:**
- Create: `tests/speed_benchmark_adjusted_rand.py`

**Interfaces:**
- Consumes: `analytics.pairwise_adjusted_rand` (Task 5), `tests/data/large_dataset.arrow` (50K rows, 101 cols).

- [ ] **Step 1: Write the benchmark**

Create `tests/speed_benchmark_adjusted_rand.py`:

```python
"""
Benchmark comparing the Rust plugin vs scikit-learn for pairwise Adjusted
Rand Index.

This script:
1. Loads large_dataset.arrow (50K rows, 101 columns; String columns are
   near-unique, so scores there are ~1 by construction — still valid timing)
2. Runs Rust plugin pairwise_adjusted_rand over all C(101,2) = 5050 pairs —
   3 runs averaged
3. Runs sklearn.metrics.adjusted_rand_score per pair with the same drop-null
   preprocessing (preprocessing time included — it is part of the sklearn
   workflow) — 3 runs averaged
4. Reports timing and validates ARI values match (rel_tol=1e-9, abs_tol=1e-12)
"""

import math
import sys
import time
from pathlib import Path

import polars as pl
from sklearn.metrics import adjusted_rand_score

_ANALYTICS_ROOT = Path(__file__).parent.parent / "services" / "analytics"
sys.path.insert(0, str(_ANALYTICS_ROOT))

from analytics import pairwise_adjusted_rand

DATA_PATH = Path(__file__).parent / "data" / "large_dataset.arrow"


def load_data() -> pl.DataFrame:
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    return pl.read_ipc(DATA_PATH)


def _run_plugin(df: pl.DataFrame) -> tuple[dict, float]:
    """Run the Rust plugin; returns ({(a, b): (ari, n_valid)}, seconds)."""
    start = time.perf_counter()
    result = pairwise_adjusted_rand(df)
    duration = time.perf_counter() - start
    rows = result.unnest(result.columns[0])
    out = {
        (row["col_a"], row["col_b"]): (row["ari"], row["n_valid"])
        for row in rows.iter_rows(named=True)
    }
    return out, duration


def _sklearn_pair(df: pl.DataFrame, col_a: str, col_b: str) -> float:
    """sklearn ARI with the plugin's drop-null policy (preprocessing included)."""
    sub = df.select([col_a, col_b]).drop_nulls()
    if sub.height == 0:
        return float("nan")
    # to_physical: Date -> Int32 etc.; String stays String. sklearn accepts both.
    x = sub[col_a].to_physical().to_numpy()
    y = sub[col_b].to_physical().to_numpy()
    return adjusted_rand_score(x, y)


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"{df.width} columns -> {df.width * (df.width - 1) // 2} pairs")

    # -- Rust plugin (3 runs) --------------------------------------------------
    print("Running Rust plugin (3 runs)...")
    plugin_result: dict = {}
    plugin_times = []
    for run in range(3):
        plugin_result, t = _run_plugin(df)
        plugin_times.append(t)
        print(f"  run {run + 1}/3 done in {t:.2f}s")
    time_plugin = sum(plugin_times) / len(plugin_times)

    # -- sklearn per-pair (3 runs) ----------------------------------------------
    print("Running sklearn per-pair (3 runs)...")
    sklearn_result: dict = {}
    sklearn_times = []
    for run in range(3):
        start = time.perf_counter()
        sklearn_result = {
            pair: _sklearn_pair(df, pair[0], pair[1]) for pair in plugin_result
        }
        t = time.perf_counter() - start
        sklearn_times.append(t)
        print(f"  run {run + 1}/3 done in {t:.1f}s")
    time_sklearn = sum(sklearn_times) / len(sklearn_times)

    # -- Correctness -------------------------------------------------------------
    mismatches = 0
    for pair, (ari, _n_valid) in plugin_result.items():
        sk = sklearn_result[pair]
        if math.isnan(ari) and math.isnan(sk):
            continue
        if math.isnan(ari) != math.isnan(sk) or not math.isclose(
            ari, sk, rel_tol=1e-9, abs_tol=1e-12
        ):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH {pair}: plugin={ari} sklearn={sk}")

    # -- Summary -------------------------------------------------------------------
    W = 32
    print()
    print("ADJUSTED RAND INDEX BENCHMARK")
    print(f"{len(plugin_result)} pairs | {df.height} rows")
    print()
    print(f"{'Method':<{W}}  {'Avg time (3 runs)':>18}  {'vs plugin':>10}")
    print("-" * (W + 32))
    print(f"{'Rust plugin batch':<{W}}  {time_plugin * 1000:>17.1f}ms  {'1.0x':>10}")
    print(f"{'sklearn (per-pair loop)':<{W}}  {time_sklearn * 1000:>17.1f}ms  {time_sklearn / time_plugin:>9.1f}x")
    print()
    print(f"Correctness vs sklearn (rel_tol=1e-9): {'yes' if mismatches == 0 else f'no ({mismatches} mismatches)'}")
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `C:/Users/Ben/miniconda3/envs/p312/python.exe tests/speed_benchmark_adjusted_rand.py`
Expected: 0 mismatches; plugin substantially faster (record the multiplier for the commit message). The sklearn loop takes minutes — that is the point being measured.

- [ ] **Step 3: Chi² benchmark regression**

Run: `C:/Users/Ben/miniconda3/envs/p312/python.exe tests/speed_benchmark_chi_squared.py`
Expected: correctness still "yes" vs polars-ds and scipy; note the plugin timing (the shared dense-id builder is expected to be at least as fast as the old hash-map counting — record before/after in the commit message; the pre-refactor baseline is in the CLAUDE.md line "~17x faster than polars-ds").

- [ ] **Step 4: Commit**

```bash
git add tests/speed_benchmark_adjusted_rand.py
git commit -m "test: ARI speed benchmark vs scikit-learn (<Nx> speedup); chi2 benchmark regression green"
```

---

### Task 8: CLAUDE.md documentation updates

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Apply the edits**

1. **Analytical Functions** — change the intro line "Five techniques" to "Six techniques" and append after technique 5:

```markdown
**6. Adjusted Rand Index — partition agreement**
Treats each column as a partition of the rows and measures chance-corrected agreement between two partitions (ARI ∈ [−0.5, 1]; 1 = identical partitions, ≈0 = chance-level). Unlike NMI it is chance-adjusted, and unlike χ²/Cramér's V it measures *partition identity*, not just association. Null policy: rows where either column is null are dropped; `n_valid` reports the surviving row count so scores over tiny overlaps can be discounted.
```

2. **Project Structure** — in the `src/` tree add after `chi_squared.rs`:

```markdown
│       │   ├── contingency.rs      # Shared drop-null contingency-table builder (chi², ARI)
│       │   ├── ari.rs              # Pairwise Adjusted Rand Index
```

and in the `tests/` tree add:

```markdown
    ├── test_adjusted_rand.py       # ARI correctness vs scikit-learn
    ├── speed_benchmark_adjusted_rand.py  # ARI benchmark vs scikit-learn
```

3. **Exposed functions** — add and amend:

```markdown
- `pairwise_chi_squared(df, pairs)` — chi-squared + p-value + Cramer's V + low_expected_count + n_valid
- `pairwise_adjusted_rand(df, pairs)` — Adjusted Rand Index + n_valid
```

4. **Next Steps → chi_squared.rs item** — delete the whole item (the `low_expected_count` warning already exists in the code with tests, and the dense re-encoding adoption is delivered by this work). If that leaves the `**[chi_squared.rs]**` heading empty, delete the heading too.

5. **Cross-cutting notes** — append a bullet:

```markdown
- chi² and ARI share `build_contingency` (contingency.rs): dense-id counting, drop-null policy, marginals + non-zero cells. Entropy keeps its own counting (null-as-category — different policy by design).
```

6. **Current Focus** — append a sentence to the entropy paragraph:

```markdown
ARI (`pairwise_adjusted_rand`) is complete and validated against scikit-learn; chi² now shares the same dense contingency builder and reports `n_valid`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document ARI technique, shared contingency builder, chi2 n_valid"
```
