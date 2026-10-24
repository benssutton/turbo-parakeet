# Uniform technique benchmarks: speed, memory and accuracy (ToDo 5f)

Status: draft 2026-10-24.  Sub-project 1 of 2; sub-project 2 (new techniques for the kernels that have no
benchmark: HyperLogLog, Schnabel / Chao1, marginal entropy) is ToDo 17 and gets its own spec.

## 1. Goal

Every `tests/performance/benchmark_<technique>.py` reports, for every implementation on every dataset, **speed,
peak memory and accuracy** in one run, on one harness, in one results table.  Today the speed harness reports
times and a boolean `agrees` sanity check; memory and error magnitudes live in two bespoke scripts
(`benchmark_top_k_accuracy.py`, `benchmark_ams_entropy.py`) with their own child-process plumbing.  Those two
scripts are folded into the technique benchmarks and deleted.

Probabilistic implementations (sketches) are compared the same way as Rust against Python: against the exact
reference, and also against the exact Rust implementation where one exists.

Unchanged: the benchmarks are never a gate (pytest is the correctness gate); `benchmark_streaming_entropy.py`,
`benchmark_recommend.py`, `calibrate_*.py` and `spike_*.py` keep their own scripts (they are not
implementation-by-implementation comparisons of one technique).  DataFusion implementations (ToDo 15) are not
added here; the baseline columns of §5 are what ToDo 15 will report against.

## 2. Package declarations

An **estimator** is an implementation class with `EXACT = False` — the existing `Technique.EXACT` class
attribute, already `False` on `BloomRust`, `BloomFastbloom`, `MinHashRust`, `MinHashDatasketch`,
`TopKHeavyKeeperRust` and `TopKSpaceSavingRust`.  `PairwiseEntropyAmsRust` and `ThreewayEntropyAmsRust` gain
`EXACT = False` (only Similarity's `agreement` reads the attribute today, so nothing else changes).

The **exact Rust sibling** of an estimator is the package's Rust implementation (name ending `Rust`) with
`EXACT = True`, if there is exactly one (pairwise / threeway entropy); otherwise none (top-k, membership,
similarity).

## 3. Harness architecture

`tests/performance/harness.py` keeps its public entry point:

```python
run(package, datasets, params=None, impl_params=None, variants=None, runs=5, budget_s=60.0)
```

and benchmark scripts stay ~20 lines.

**Parent, per dataset.**  Calls `Dataset.make()` once (LazyFrames collected) and writes each frame to
`<tmp>/<dataset>/<frame>.arrow` (Arrow IPC, uncompressed).

**Child, one per measurement** — one per (dataset, implementation, variant, threads).  Rust implementations run
at `threads = N` and `threads = 1` (`RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`, set in the child's environment
— both pools are fixed at process start); others once, `threads = default`.  The child is the harness itself,
`python harness.py --child <spec.json>`, independent of the benchmark script; the spec names the package, class,
params, frame files and output path.  The child:

1. reads its frames (`pl.read_ipc(..., memory_map=False)`), imports the class;
2. `gc.collect()`, records `load_mb` (current working set) and the peak so far;
3. runs the warm-up `cls(**params).add(frames).result()`; `peak_mb` = peak working set after it − `load_mb`;
   `peak_is_upper_bound` = the peak did not move (the warm-up never exceeded the load-time peak, so `peak_mb` only bounds the figure from above);
4. if the warm-up took longer than `budget_s`: reports `skipped: over budget` and stops;
5. runs `runs` timed repeats (fresh instance each) → `median_ms`, `min_ms`;
6. writes the last result to the output path (Arrow IPC) and prints one JSON line.

Memory therefore comes from the warm-up of a fresh process: the OS peak (Windows `PeakWorkingSetSize`) never
resets, so a peak measured later in a process, or after another implementation, would be polluted.
`memory_mb()` moves from `benchmark_ams_entropy.py` into the harness, unchanged (Windows / Linux / macOS).

**Parent, scoring** (§5), then the report (console table + `results/<technique>_<stamp>.arrow`, ZSTD, with the
existing environment columns: cpu_count, processor, versions, timestamp).

## 4. Variants

```python
@dataclass(frozen=True)
class Variant:
    label: str
    params: dict = {}            # every implementation
    estimator_params: dict = {}  # estimators (EXACT = False) only, e.g. memory_bytes, num_samples
```

`variants` defaults to `[Variant("default")]`, so existing scripts are unchanged.  Variants apply to every
dataset.  Non-estimators (the reference and exact implementations) run once per distinct `params`, not once per
variant; their rows carry the label of the first variant with those params, and every estimator row is scored
against the non-estimator rows with the same `params`.

## 5. Results row and scoring

One row per (dataset, implementation, variant, threads):

| group | columns |
|---|---|
| keys | `dataset`, `implementation`, `variant`, `threads`, `status` |
| speed | `median_ms`, `min_ms` |
| memory | `load_mb`, `peak_mb`, `peak_is_upper_bound` |
| vs reference | `agrees`, accuracy columns (§6) |
| vs exact Rust | `exact_rust_agrees`, accuracy columns prefixed `exact_rust_`, `speedup_vs_exact_rust` |
| speedups (Rust rows, `threads = N`) | `algorithmic`, `parallel`, `total` |

- `agrees` = `not impl.agreement(result, reference_result)` with the technique's existing tolerances (as today);
  null for the reference itself.
- Accuracy columns come from `impl.accuracy(frames, result, reference_result)` (§6), computed for every row
  except the reference's own (an exact implementation should score ≈ 0, which is itself a check).
- `exact_rust_*`: the same two calls with the exact Rust sibling's result (same dataset, params and threads) in
  place of the reference's; `speedup_vs_exact_rust` = sibling `median_ms` ÷ estimator `median_ms` at the same
  `threads`.  Null where there is no sibling.
- `algorithmic` = fastest non-Rust ÷ Rust@1, `parallel` = Rust@1 ÷ Rust@N, `total` = fastest non-Rust ÷ Rust@N,
  as today, now within a (dataset, variant).  Estimators end in `Rust`, so they get these too.

The console report prints speed + memory + `agrees` per row, then a speedup table, then an accuracy table
(estimators and any row whose `agrees` is false).

## 6. The accuracy hook

On `turbo_parakeet.techniques.base.Technique`:

```python
def accuracy(self, frames: dict, result: pl.DataFrame, reference: pl.DataFrame) -> dict[str, float | int | None]
```

Metrics only, never in a technique's result table and never a decision.

**Default.**  Rows aligned on the key columns (misaligned → ValueError).  Over the rows the reference computed,
for every column of `ACCURACY_METRICS` (a new class attribute; `None`, the default, means every numeric column
of `METRICS` — integer, float, decimal): relative error |got − want| / |want|, with 0/0
counted as 0 and NaN elsewhere kept (it shows up as NaN in the max).  Returns `max_rel_err`, `p95_rel_err`
(over all those cells) and `n_missing` (nulls in `result` where the reference computed).  Non-numeric metrics are
ignored.  This reproduces `benchmark_ams_entropy.py`'s `max_rel_err` / `p95_rel_err` / `n_missing`.

**Pairwise entropy** sets `ACCURACY_METRICS = ("h_a", "h_b", "h_ab")` (relative MI / NMI error is meaningless
where MI ≈ 0) and adds `max_abs_mi_err` (as the AMS script did); **threeway entropy** sets `("h_abc",)`.

**Top-k** overrides it.  Ground truth is rebuilt from `frames` (exact counts of every distinct non-null value per
column, values as text), not from the reference's list, which holds only k values.  Per column, as
`benchmark_top_k_accuracy.py` computes them (code moved, not rewritten): `precision` (tie-aware: reported values
whose true count ≥ the true k-th count, ÷ min(k, distinct)), `are`, `aae` and `max_bound_violation` (SpaceSaving
only; null when `error` is null).  Over a dataset's columns the hook returns the minimum `precision` and the
maximum of the others, plus `sketch_bytes` (max).  Datasets that need per-column figures are one column wide (§7).

## 7. Script changes

**`benchmark_top_k.py`** absorbs `benchmark_top_k_accuracy.py`:
- datasets: `large_dataset`, `narrow 10M x 4`, `wide 200K x 100`, and one 10M × 1 dataset per stream of
  `datagen.TOP_K_COLUMNS` (Zipf α 1.5 / 1.1 / 0.8, flat, skewed, string), replacing `zipf 10M x 4` — one column
  per dataset keeps the per-stream precision and memory the outcome notes rely on;
- variants: k ∈ {5, 256} × `estimator_params={"memory_bytes": m × max(16 KiB, k × 1 KiB)}`, m ∈ {¼, 1, 4};
- `--rows N`: the scale run — datasets replaced by `zipf_11` alone at N rows, variants by k = 5 at 1×, results to
  `results/top_k_scale_<stamp>.arrow`.

**`benchmark_pairwise_entropy.py` / `benchmark_threeway_entropy.py`** already use the AMS script's datasets
(threeway: `large_dataset(columns=40)`, narrow 1M × 5, wide 20K × 30, near-unique, near-constant); they gain the
memory and accuracy columns with no script change beyond any `exclude` the AMS script carried.

**Deleted:** `benchmark_top_k_accuracy.py`, `benchmark_ams_entropy.py`, and their results files
(`top_k_accuracy.arrow`, `ams_entropy.arrow`, git-ignored).  `benchmark_streaming_entropy.py` imports
`memory_mb` from the harness instead.  The top-k grid runs with `runs=3` (as threeway entropy already does).

**All other benchmark scripts:** unchanged unless a dataset needs a narrower `exclude` because children now run
it at 1 thread with memory measured (none expected).

## 8. Failure handling

Each failure produces a row; nothing aborts the run.
- child timeout (1800 s) → `status = timeout`; crash → `status = error` (last 20 stderr lines printed);
- over budget → `skipped: over budget (Ns)`; `exclude` → `excluded`; library missing → `unavailable` (printed,
  as today);
- reference (or exact sibling) failed → the columns that depend on it are null for that dataset / variant.

## 9. Tests

`tests/test_benchmark_harness.py` (pytest, fast) keeps the `time_call` / `speedups` tests and adds, against a
small fake technique package in `tests/performance/` (an exact implementation, a deterministic noisy "estimator",
an exact Rust-named sibling, and one that raises):
- rows: expected columns; one row per (implementation, variant, threads); Rust-named rows at 1 and N;
- accuracy: the exact implementation scores 0, the noisy one a known nonzero error; `exact_rust_*` filled for the
  estimator only;
- memory: a child that allocates a known buffer reports `peak_mb` ≥ its size;
- variants: non-estimators run once per distinct `params`;
- failure: the raising implementation gives an `error` row and the run completes.

`tests/test_top_k.py` adds a known-answer test of `accuracy()`: hand-made counts with a tie at the k-th count and
a SpaceSaving bound that holds and one that fails.  `tests/test_base.py` adds one for the default hook (0/0, NaN,
missing, misaligned rows).

## 10. Docs

CLAUDE.md: the Testing Convention's benchmark paragraph (speed + memory + accuracy, one child per measurement,
`ESTIMATORS`, the exact-Rust columns, variants); the Project Structure tree (two scripts gone); the Top-k and
AMS outcome notes point at `benchmark_top_k.py` / the entropy benchmarks.  ToDo.md: 5f marked done (this spec);
17 added for sub-project 2.

## 11. Costs and risks

- One process per measurement: ≈ 1 s start-up (Polars + extension import) per child, plus reading the frames.
  Top-k has the largest grid: per 10M-row stream ≈ 2 reference + 24 estimator children.
- `peak_mb` includes allocator caching and is a working-set figure, not a heap figure; `peak_is_upper_bound`
  flags the runs whose figure only bounds the truth from above.  Comparable between implementations on one
  machine, not across machines.
- Timing in a fresh child excludes nothing the in-process loop included (warm-up is still discarded), so speed
  figures stay comparable with earlier results.
