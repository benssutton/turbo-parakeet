# Generated large dataset and Parquet removal — design (ToDo 6 + 5g)

Status: approved 2026-10-17.  Next spec: ToDo 1.0 (output schema) with ToDos 4 + 2 as its first increment.

## 1. Goal

Replace the checked-in `tests/data/large_dataset.arrow` (40 MB, 50 000 × 101, one batch) with a seeded generator
that reproduces its shape and adds settings for null rate, run length and clustering, so that null exemplars (ToDo 2)
and the order-dependent recommendations (ToDos 1a, 1d, 1e, 3) can be tested at scale.  In the same change, remove
Parquet from the project: benchmark results move to Arrow IPC and the polars crate stops pulling in
`polars-parquet`.

Out of scope: rewriting git history (the 392 MiB pack stays; see §5), any change to techniques or recommenders.

## 2. Generator

In `tests/datagen.py`, cached with `functools.cache` (Polars frames are immutable; each pytest-xdist worker builds
the frame once, in well under a second):

```python
def large_dataset(n_rows=50_000, seed=0, null_rate=0.0, mean_run=1, clustering=0.0) -> pl.DataFrame
def large_dataset_mixed(n_rows=50_000, seed=0) -> pl.DataFrame
```

### 2.1 Shape

- `identity`: Int64 `0..n_rows`, never null, never reordered (the exemplar identifier for ToDo 2).
- 25 groups g = 0..24 of four columns each, named and typed as today:

  | column | dtype | values |
  |---|---|---|
  | `col_{4g}` | Int32 | uniform in `[0, card_g)` |
  | `col_{4g+1}` | Float64 | uniform in `[0, 1)` |
  | `col_{4g+2}` | String | 3 characters, each uniform over `[A-Za-z]` |
  | `col_{4g+3}` | Date | 2017-01-01 + uniform `[0, card_g)` days |

- `card_0 = 100`, `card_g = max(1, round(20 / g))` for g ≥ 1 (Python `round`): 100, 20, 10, 7, 5, 4, 3, 3, 2, …, 1 —
  the step-down of the current file, not its exact counts.  Bit-for-bit reproduction is neither possible nor
  needed: every current use is an agreement or parity fixture (§3).
- One chunk; dtypes exactly as above (Date, not Datetime).

### 2.2 Settings

Applied to every non-`identity` column, in this order:

1. **Values.**  Run lengths are geometric with mean `mean_run` (the logic of `runs_int`, factored into a private
   helper both use); each run draws one value.  `mean_run=1` → every row an independent draw.  Adjacent runs
   drawing the same value merge, as in `runs_int`.
2. **Order — `clustering` ∈ [0, 1].**  Sort the column ascending, then pick a uniformly random subset of
   `round((1 − clustering) · n_rows)` positions and shuffle their values among themselves.  1 → sorted, 0 → a
   random permutation (today's order).
3. **Nulls — `null_rate` ∈ [0, 1].**  An independent Bernoulli(`null_rate`) mask per column.

Out-of-range settings (`mean_run < 1`, `clustering` or `null_rate` outside [0, 1]) raise `ValueError`.

### 2.3 Determinism

Each column draws from its own `np.random.Generator`, spawned from `np.random.SeedSequence(seed)` by column index.
Within a column the three steps above draw in that order, so changing one setting changes only that step's draws
and later ones — never another column.

### 2.4 Mixed preset

`large_dataset_mixed` builds the same 25 groups but cycles five regimes over them (group g uses regime g mod 5):

| regime | null_rate | mean_run | clustering |
|---|---|---|---|
| 0 plain | 0 | 1 | 0 |
| 1 nulls | 0.1 | 1 | 0 |
| 2 runs | 0 | 50 | 0 |
| 3 sorted | 0 | 1 | 1 |
| 4 combined | 0.1 | 50 | 0.9 |

Implementation: one private builder takes a per-group `(null_rate, mean_run, clustering)` list; both public functions
call it.  A column's draws depend only on `seed`, its index and its own regime, so group g of
`large_dataset_mixed()` equals group g of `large_dataset(**regime)`.

## 3. Consumers

- **Tests.**  `test_describe.py`, `test_minmax_index.py`, `test_recommend.py`, `test_runs_test.py`,
  `test_set_index.py` and `test_streaming_recommend.py` replace `pl.read_ipc(LARGE)` with `datagen.large_dataset()`
  (keeping their `slow` marks and existing slicing, e.g. `head(5_000)`).  The large-dataset agreement tests of
  runs_test, minmax_index, set_index and describe, and the streaming parity test, are additionally parametrised over
  `large_dataset_mixed()`.
- **Benchmarks.**  `tests/performance/harness.py`: `large_dataset(columns=None, exclude=())` keeps its signature and
  wraps the generator; the label becomes `large_dataset` (generated).  `LARGE_DATASET` is removed.
- **Measured numbers.**  Re-measure and update every figure taken from the old file: the MinHash recall 0.62
  (389/625, 236 misses) in `analytics/similarity/rust.py` and CLAUDE.md, and the large_dataset.arrow speedups
  quoted in CLAUDE.md (runs test 16.9x; Membership 0.60x).  Figures not re-measured are labelled as measured on the
  old file.
- **Removal.**  `tests/data/large_dataset.arrow` and `tests/data/` are deleted in an ordinary commit.

## 4. Parquet removal (ToDo 5g)

### 4.1 Benchmark results

- `harness._report` writes `results/{technique}_{stamp}.arrow` with `write_ipc(path, compression="zstd")`.
- `benchmark_recommend.py` (`recommend.arrow`, `recommend_dictionary_order.arrow`),
  `benchmark_top_k_accuracy.py` (`top_k_accuracy.arrow`, `top_k_scale.arrow`) and `benchmark_ams_entropy.py`
  (`ams_entropy.arrow`) likewise; the temporary files the last two use to pass results from child processes become
  `.arrow` (`write_ipc` / `read_ipc`).  Docstrings naming `.parquet` are updated.
- `.gitignore`: results are ignored as `*.arrow` (whatever pattern currently covers `*.parquet` is replaced).
- The existing local `.parquet` results are converted once to `.arrow` by hand (not a committed script) and deleted.

### 4.2 Cargo

`polars` gets `default-features = false`.  Its defaults are `docs, zip_with, csv, polars-parquet, temporal, fmt,
dtype-slim`; add back only those the crate needs (expected: `fmt`, `temporal`, `zip_with`, `dtype-slim`; the build
decides — `csv` and `polars-parquet` must not return).  Acceptance:

- `cargo tree -i polars-parquet` reports no match;
- `python scripts/check.py ci` passes (Python extension, C-API build, Java tests);
- cold `maturin develop --release` time before and after is recorded in the plan's final task and in CLAUDE.md if it
  changed materially.

### 4.3 Guard

`tests/test_layering.py` gains a test: no `.py` file under `tests/` or `services/` matches
`read_parquet|write_parquet|scan_parquet|sink_parquet|pyarrow\.parquet|\.parquet["']` (so prose may still name the
format), and `services/analytics/Cargo.toml` sets `default-features = false` on `polars` and names neither `parquet`
nor `polars-parquet`.

## 5. Repository housekeeping

Run `git gc` locally after the deletion (clears the 11 orphaned pack `.idx` files).  History is not rewritten: no
force-push, SHAs referenced in CLAUDE.md and the specs (e.g. `93f6dcb`, `1bb7ca4`) stay valid.  ToDo.md records
that the pack keeps the old blob.

## 6. Tests for the generator

In `tests/test_datagen.py`:

- schema: column names and dtypes equal the old file's (`identity` Int64; Int32, Float64, String, Date per group);
- determinism: same arguments → equal frames; changing `null_rate` leaves every column's non-null values at the
  same rows unchanged;
- defaults: no nulls; `identity` sorted; Int32 / Date distinct counts equal `card_g` at 50 000 rows; String
  distinct count > 40 000;
- `null_rate=0.1`: each column's null share within 3σ (≈ ±0.004 at 50 000 rows);
- `mean_run=50`: on a high-cardinality column (Float64), `n_rows / n_runs` within 10% of 50;
- `clustering=1`: every non-identity column sorted (nulls aside); `clustering=0.5`: at least half the positions hold
  their sorted value, and the column is not sorted;
- invalid settings raise `ValueError`;
- `large_dataset_mixed`: group g equals the corresponding group of `large_dataset(**regime)`.

## 7. Documentation

CLAUDE.md: Project Structure (no `tests/data/`), Testing Convention (benchmark datasets, results as `.arrow`),
`datagen` generator list, re-measured numbers (§3).  ToDo.md: 6 and 5g done; the agreed starting order moves on to
1.0.
