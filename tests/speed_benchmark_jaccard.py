"""
Benchmark comparing three column-similarity filter implementations.

Measures time to find all column pairs with Jaccard Index or Overlap
Coefficient above threshold across all columns of large_dataset.arrow.

Implementations:
  - DeterministicSimilarityFilter  -- brute-force exact, used as ground truth (1 run)
  - MinHashLSHFilter               -- Rust plugin MinHash + LSH (3 runs, averaged)
  - MinHashLSHFilter_datasketch    -- datasketch library (3 runs, averaged)
    Install: pip install datasketch
"""

import sys
import time
from pathlib import Path

import polars as pl

_ANALYTICS_ROOT = Path(__file__).parent.parent / "services" / "analytics"
sys.path.insert(0, str(_ANALYTICS_ROOT))

from deterministic_similarity_filter import DeterministicSimilarityFilter
from minhash_lsh_filter import MinHashLSHFilter

DATASKETCH_AVAILABLE = False
try:
    from minhash_lsh_filter_datasketch import MinHashLSHFilter_datasketch
    DATASKETCH_AVAILABLE = True
except ImportError:
    print("WARNING: datasketch not installed. Install with: pip install datasketch")

DATA_PATH = Path(__file__).parent / "data" / "large_dataset.arrow"

JACCARD_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.95
NUM_PERM = 128


def load_data() -> pl.LazyFrame:
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    return pl.scan_ipc(DATA_PATH)


def _pair_set(df: pl.DataFrame) -> set[tuple[str, str]]:
    """Return normalised (sorted) column-pair set from a similarity result DataFrame."""
    if df.is_empty():
        return set()
    return {tuple(sorted([row["col_a"], row["col_b"]])) for row in df.iter_rows(named=True)}


def _run_deterministic(lf: pl.LazyFrame) -> tuple[set, float]:
    """Return (pair_set, verify_seconds). Runs once — used as ground truth."""
    f = DeterministicSimilarityFilter(
        jaccard_threshold=JACCARD_THRESHOLD,
        overlap_threshold=OVERLAP_THRESHOLD,
    )
    f.add({"df": lf})
    result = f.get_similar_pairs()
    return _pair_set(result), f.verification_time


def _run_rust(lf: pl.LazyFrame) -> tuple[set, int, float, float, float, float]:
    """Return (pair_set, n_candidates, minhash_s, lsh_s, verify_s, total_s)."""
    f = MinHashLSHFilter(
        jaccard_threshold=JACCARD_THRESHOLD,
        overlap_threshold=OVERLAP_THRESHOLD,
        num_perm=NUM_PERM,
    )
    f.add({"df": lf})
    t0 = time.perf_counter()
    result = f.get_similar_pairs()
    total = time.perf_counter() - t0
    return _pair_set(result), len(f.candidates), f.minhash_time, f.lsh_time, f.verification_time, total


def _run_datasketch(lf: pl.LazyFrame) -> tuple[set, int, float, float, float, float]:
    """Return (pair_set, n_candidates, minhash_s, lsh_s, verify_s, total_s)."""
    f = MinHashLSHFilter_datasketch(
        jaccard_threshold=JACCARD_THRESHOLD,
        overlap_threshold=OVERLAP_THRESHOLD,
        num_perm=NUM_PERM,
    )
    f.add({"df": lf})
    t0 = time.perf_counter()
    result = f.get_similar_pairs()
    total = time.perf_counter() - t0
    return _pair_set(result), len(f.candidates), f.minhash_time, f.lsh_time, f.verification_time, total


def main() -> None:
    print("Loading dataset...")
    lf = load_data()
    n_cols = len(lf.collect_schema())
    n_pairs = n_cols * (n_cols - 1) // 2
    n_rows = lf.select(pl.len()).collect().item()

    print(f"Running deterministic filter ({n_pairs} pairs, ground truth)...")
    det_pairs, det_verify_t = _run_deterministic(lf)
    n_true = len(det_pairs)

    print("Running MinHashLSH Rust filter (3 runs)...")
    rust_runs = [_run_rust(lf) for _ in range(3)]
    rust_pairs, rust_candidates, rust_minhash_t, rust_lsh_t, rust_verify_t, _ = rust_runs[-1]
    time_rust = sum(r[5] for r in rust_runs) / 3

    ds_pairs: set = set()
    ds_candidates = 0
    ds_minhash_t = ds_lsh_t = ds_verify_t = time_ds = 0.0
    if DATASKETCH_AVAILABLE:
        print("Running MinHashLSH datasketch filter (3 runs)...")
        ds_runs = [_run_datasketch(lf) for _ in range(3)]
        ds_pairs, ds_candidates, ds_minhash_t, ds_lsh_t, ds_verify_t, _ = ds_runs[-1]
        time_ds = sum(r[5] for r in ds_runs) / 3

    # Column width: header right-aligned to C chars; value f"{val*1000:>{C-2}.1f}ms" = C chars
    W = 28
    C = 11

    print()
    print("SIMILARITY FILTER BENCHMARK")
    print(f"{n_pairs} pairs | {n_rows:,} rows | {n_cols} columns")
    print(f"Thresholds: jaccard={JACCARD_THRESHOLD}, overlap={OVERLAP_THRESHOLD}, num_perm={NUM_PERM}")
    print()
    print(f"{'Method':<{W}}  {'MinHash':>{C}}  {'LSH':>{C}}  {'Verify':>{C}}  {'Total (avg)':>{C}}  {'Pairs':>6}")
    print("-" * (W + 4 * C + 5 * 2 + 6))
    print(
        f"{'Deterministic (brute force)':<{W}}"
        f"  {'n/a':>{C}}"
        f"  {'n/a':>{C}}"
        f"  {det_verify_t * 1000:>{C-2}.1f}ms"
        f"  {det_verify_t * 1000:>{C-2}.1f}ms"
        f"  {n_true:>6}"
    )
    print(
        f"{'MinHashLSH (Rust)':<{W}}"
        f"  {rust_minhash_t * 1000:>{C-2}.1f}ms"
        f"  {rust_lsh_t * 1000:>{C-2}.1f}ms"
        f"  {rust_verify_t * 1000:>{C-2}.1f}ms"
        f"  {time_rust * 1000:>{C-2}.1f}ms"
        f"  {len(rust_pairs):>6}"
    )
    if DATASKETCH_AVAILABLE:
        print(
            f"{'MinHashLSH (datasketch)':<{W}}"
            f"  {ds_minhash_t * 1000:>{C-2}.1f}ms"
            f"  {ds_lsh_t * 1000:>{C-2}.1f}ms"
            f"  {ds_verify_t * 1000:>{C-2}.1f}ms"
            f"  {time_ds * 1000:>{C-2}.1f}ms"
            f"  {len(ds_pairs):>6}"
        )
    print()
    print(f"Speedup vs deterministic")
    print(f"  MinHashLSH (Rust):       {det_verify_t / time_rust:.1f}x")
    if DATASKETCH_AVAILABLE:
        print(f"  MinHashLSH (datasketch): {det_verify_t / time_ds:.1f}x")
    print()
    print(f"Candidates examined  (of {n_pairs} pairs total)")
    print(f"  MinHashLSH (Rust):       {rust_candidates}")
    if DATASKETCH_AVAILABLE:
        print(f"  MinHashLSH (datasketch): {ds_candidates}")
    print()
    if n_true == 0:
        print("Recall: no pairs above threshold (deterministic found none)")
    else:
        rust_found = len(rust_pairs & det_pairs)
        rust_recall = rust_found / n_true
        print(f"Recall vs deterministic")
        print(f"  MinHashLSH (Rust):       {'yes' if rust_recall == 1.0 else f'no  ({rust_recall*100:.1f}%  {rust_found}/{n_true} pairs)'}")
        if DATASKETCH_AVAILABLE:
            ds_found = len(ds_pairs & det_pairs)
            ds_recall = ds_found / n_true
            print(f"  MinHashLSH (datasketch): {'yes' if ds_recall == 1.0 else f'no  ({ds_recall*100:.1f}%  {ds_found}/{n_true} pairs)'}")
    print()


if __name__ == "__main__":
    main()
