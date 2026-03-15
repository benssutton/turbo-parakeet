"""
Benchmark comparing Rust MinHash+LSH plugin vs datasketch vs brute-force for
pairwise column similarity detection (Jaccard Index and Overlap Coefficient).

This script:
1. Generates synthetic test data with known similar column pairs
2. Runs MinHashLSHFilter (Rust plugin) — 3 runs averaged
3. Runs MinHashLSHFilter_datasketch (datasketch library) — 3 runs averaged
4. Runs DeterministicSimilarityFilter (brute-force ground truth) — 1 run
5. Reports timing, speedup ratios, recall, and false negative/positive counts
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

_SERVICES_ROOT = Path(__file__).parent.parent.parent / "services"
sys.path.insert(0, str(_SERVICES_ROOT))
sys.path.insert(0, str(_SERVICES_ROOT / "analytics"))

from MinHashLSHFilter import MinHashLSHFilter
from comparitors.DeterministicSimilarityFilter import DeterministicSimilarityFilter
from comparitors.MinHashLSHFilter_datasketch import MinHashLSHFilter_datasketch


# ============================================================================
# Test data generation
# ============================================================================

def _create_similar_pair(
    n_elements: int,
    size_a: int,
    size_b: int,
    intersection_size: int,
    rng: np.random.Generator,
) -> Tuple[Set[int], Set[int]]:
    """Create two sets with specified sizes and intersection."""
    all_elements = np.arange(n_elements)
    elements_a = set(rng.choice(all_elements, size=min(size_a, n_elements), replace=False))

    intersection = set(list(elements_a)[:intersection_size])

    remaining = [x for x in range(n_elements) if x not in elements_a]
    unique_b_size = size_b - intersection_size
    if unique_b_size > 0 and len(remaining) >= unique_b_size:
        unique_b = set(rng.choice(remaining, size=unique_b_size, replace=False))
    else:
        unique_b = set(remaining[:max(0, unique_b_size)]) if unique_b_size > 0 else set()

    elements_b = intersection | unique_b
    return elements_a, elements_b


def _assign_pair_to_dataframes(
    n_dataframes: int,
    cross_df: bool,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Assign a pair of columns to dataframe indices."""
    if cross_df and n_dataframes > 1:
        df_idx_a = int(rng.integers(0, n_dataframes))
        df_idx_b = (df_idx_a + int(rng.integers(1, n_dataframes))) % n_dataframes
    else:
        df_idx_a = df_idx_b = int(rng.integers(0, n_dataframes))
    return df_idx_a, df_idx_b


def generate_test_data(
    n_columns: int,
    n_elements: int,
    density: float = 0.1,
    n_dataframes: int = 2,
    seed: int = 42,
    similar_pair_ratio: float = 0.15,
    jaccard_threshold: float = 0.7,
    overlap_threshold: float = 0.8,
    dataframe_densities: Optional[List[float]] = None,
    cross_df_ratio: float = 0.5,
) -> Tuple[Dict[str, pl.LazyFrame], Dict[str, List[Tuple[str, str]]]]:
    """Generate synthetic test data with known similar column pairs.

    Creates pairs in three categories:
    - jaccard_only: meets Jaccard threshold but not Overlap threshold
    - overlap_only: meets Overlap threshold but not Jaccard threshold
    - both: meets both thresholds
    """
    rng = np.random.default_rng(seed)

    if dataframe_densities is None:
        dataframe_densities = [1.0] * n_dataframes
    assert len(dataframe_densities) == n_dataframes

    n_total_pairs = max(3, int(n_columns * similar_pair_ratio / 2))
    n_jaccard_only = n_total_pairs // 3
    n_overlap_only = n_total_pairs // 3
    n_both = n_total_pairs - n_jaccard_only - n_overlap_only

    col_to_dataframe: Dict[str, int] = {}
    all_columns: Dict[str, List] = {}
    col_idx = 0

    expected_pairs: Dict[str, List[Tuple[str, str]]] = {
        "jaccard_only": [],
        "overlap_only": [],
        "both": [],
    }

    # Phase 1: Jaccard-only pairs (J >= J_th, O < O_th)
    for _ in range(n_jaccard_only):
        cross_df = (rng.random() < cross_df_ratio) and (n_dataframes > 1)
        df_idx_a, df_idx_b = _assign_pair_to_dataframes(n_dataframes, cross_df, rng)

        base_size = int(n_elements * density * dataframe_densities[df_idx_a])
        size_a = size_b = max(50, base_size)

        target_j = jaccard_threshold + 0.05
        intersection_size = int(target_j * 2 * size_a / (1 + target_j)) + 1

        expected_overlap = intersection_size / size_a
        if expected_overlap >= overlap_threshold:
            intersection_size = int(overlap_threshold * size_a * 0.85)

        intersection_size = max(1, min(intersection_size, size_a - 1))

        set_a, set_b = _create_similar_pair(n_elements, size_a, size_b, intersection_size, rng)

        col_a_name = f"col_{col_idx:03d}"
        all_columns[col_a_name] = list(set_a)
        col_to_dataframe[col_a_name] = df_idx_a
        col_idx += 1

        col_b_name = f"col_{col_idx:03d}"
        all_columns[col_b_name] = list(set_b)
        col_to_dataframe[col_b_name] = df_idx_b
        col_idx += 1

        qualified_a = f"df{df_idx_a}:{col_a_name}"
        qualified_b = f"df{df_idx_b}:{col_b_name}"
        sorted_pair = sorted([qualified_a, qualified_b])
        expected_pairs["jaccard_only"].append((sorted_pair[0], sorted_pair[1]))

    # Phase 2: Overlap-only pairs (J < J_th, O >= O_th)
    for _ in range(n_overlap_only):
        cross_df = (rng.random() < cross_df_ratio) and (n_dataframes > 1)
        df_idx_a, df_idx_b = _assign_pair_to_dataframes(n_dataframes, cross_df, rng)

        base_size = int(n_elements * density * dataframe_densities[df_idx_a])
        size_a = max(30, int(base_size * 0.3))

        ratio = (overlap_threshold * (1 + jaccard_threshold) / jaccard_threshold - 1) * 1.5
        size_b = int(size_a * ratio)
        size_b = min(size_b, n_elements - size_a)
        size_b = max(size_b, size_a + 10)

        target_overlap = overlap_threshold + 0.03
        intersection_size = int(target_overlap * size_a)
        intersection_size = max(1, min(intersection_size, size_a))

        set_a, set_b = _create_similar_pair(n_elements, size_a, size_b, intersection_size, rng)

        col_a_name = f"col_{col_idx:03d}"
        all_columns[col_a_name] = list(set_a)
        col_to_dataframe[col_a_name] = df_idx_a
        col_idx += 1

        col_b_name = f"col_{col_idx:03d}"
        all_columns[col_b_name] = list(set_b)
        col_to_dataframe[col_b_name] = df_idx_b
        col_idx += 1

        qualified_a = f"df{df_idx_a}:{col_a_name}"
        qualified_b = f"df{df_idx_b}:{col_b_name}"
        sorted_pair = sorted([qualified_a, qualified_b])
        expected_pairs["overlap_only"].append((sorted_pair[0], sorted_pair[1]))

    # Phase 3: Both-threshold pairs (J >= J_th, O >= O_th)
    for _ in range(n_both):
        cross_df = (rng.random() < cross_df_ratio) and (n_dataframes > 1)
        df_idx_a, df_idx_b = _assign_pair_to_dataframes(n_dataframes, cross_df, rng)

        base_col_size = int(n_elements * density * dataframe_densities[df_idx_a])
        base_col_size = max(50, base_col_size)

        size_a = base_col_size
        target_overlap = overlap_threshold + 0.02
        intersection_size = int(target_overlap * size_a)

        size_b = int(size_a * (target_overlap * (1 + jaccard_threshold) - jaccard_threshold) / jaccard_threshold)

        if size_b < size_a:
            size_a, size_b = size_b, size_a
            intersection_size = int(target_overlap * size_a)

        size_a = min(size_a, n_elements)
        size_b = min(size_b, n_elements)
        intersection_size = max(1, min(intersection_size, size_a))

        set_a, set_b = _create_similar_pair(n_elements, size_a, size_b, intersection_size, rng)

        col_a_name = f"col_{col_idx:03d}"
        all_columns[col_a_name] = list(set_a)
        col_to_dataframe[col_a_name] = df_idx_a
        col_idx += 1

        col_b_name = f"col_{col_idx:03d}"
        all_columns[col_b_name] = list(set_b)
        col_to_dataframe[col_b_name] = df_idx_b
        col_idx += 1

        qualified_a = f"df{df_idx_a}:{col_a_name}"
        qualified_b = f"df{df_idx_b}:{col_b_name}"
        sorted_pair = sorted([qualified_a, qualified_b])
        expected_pairs["both"].append((sorted_pair[0], sorted_pair[1]))

    # Phase 4: Remaining independent columns
    while col_idx < n_columns:
        df_idx = col_idx % n_dataframes
        col_size = int(n_elements * density * dataframe_densities[df_idx])

        region_start = (col_idx * n_elements // n_columns) % n_elements
        available_elements = [(region_start + i) % n_elements for i in range(n_elements)]

        col_elements = set(rng.choice(
            available_elements,
            size=min(col_size, len(available_elements)),
            replace=False,
        ))

        col_name = f"col_{col_idx:03d}"
        all_columns[col_name] = list(col_elements)
        col_to_dataframe[col_name] = df_idx
        col_idx += 1

    # Build LazyFrames
    col_names = list(all_columns.keys())
    lazyframes_dict: Dict[str, pl.LazyFrame] = {}

    for df_idx in range(n_dataframes):
        df_name = f"df{df_idx}"
        df_col_names = [cn for cn in col_names if col_to_dataframe[cn] == df_idx]

        if not df_col_names:
            continue

        max_col_len = max(len(all_columns[cn]) for cn in df_col_names)

        df_columns = {}
        for col_name in df_col_names:
            col_values = all_columns[col_name].copy()
            if len(col_values) < max_col_len:
                col_values += [None] * (max_col_len - len(col_values))
            df_columns[col_name] = col_values

        lazyframes_dict[df_name] = pl.DataFrame(df_columns).lazy()

    return lazyframes_dict, expected_pairs


# ============================================================================
# Result extraction
# ============================================================================

def _extract_pair_set(results_df: pl.DataFrame) -> Set[Tuple[str, str]]:
    """Extract normalised (sorted) pairs from a results DataFrame."""
    pairs: Set[Tuple[str, str]] = set()
    for row in results_df.iter_rows(named=True):
        qualified_a = f"{row['df_a']}:{row['col_a']}"
        qualified_b = f"{row['df_b']}:{row['col_b']}"
        pair = tuple(sorted([qualified_a, qualified_b]))
        pairs.add(pair)
    return pairs


# ============================================================================
# Per-method run helpers
# ============================================================================

def _run_brute_force(
    lazyframes: Dict[str, pl.LazyFrame],
    jaccard_threshold: float,
    overlap_threshold: float,
) -> Tuple[pl.DataFrame, float]:
    f = DeterministicSimilarityFilter(
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
    )
    f.add(lazyframes)
    start = time.perf_counter()
    result_df = f.get_similar_pairs()
    duration = time.perf_counter() - start
    return result_df, duration


def _run_datasketch(
    lazyframes: Dict[str, pl.LazyFrame],
    jaccard_threshold: float,
    overlap_threshold: float,
    num_perm: int,
) -> Tuple[pl.DataFrame, float]:
    f = MinHashLSHFilter_datasketch(
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
        num_perm=num_perm,
    )
    f.add(lazyframes)
    start = time.perf_counter()
    result_df = f.get_similar_pairs()
    duration = time.perf_counter() - start
    return result_df, duration


def _run_rust(
    lazyframes: Dict[str, pl.LazyFrame],
    jaccard_threshold: float,
    overlap_threshold: float,
    num_perm: int,
) -> Tuple[pl.DataFrame, float]:
    f = MinHashLSHFilter(
        jaccard_threshold=jaccard_threshold,
        overlap_threshold=overlap_threshold,
        num_perm=num_perm,
    )
    f.add(lazyframes)
    start = time.perf_counter()
    result_df = f.get_similar_pairs()
    duration = time.perf_counter() - start
    return result_df, duration


# ============================================================================
# Main benchmark
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("MINHASH + LSH SIMILARITY FILTER BENCHMARK")
    print()
    print("  1. Rust plugin       (MinHashLSHFilter)")
    print("  2. datasketch        (MinHashLSHFilter_datasketch)")
    print("  3. Brute force       (DeterministicSimilarityFilter — ground truth)")
    print("=" * 70)
    print()

    # Configuration
    # (n_columns, n_elements, jaccard_threshold, overlap_threshold, n_dataframes, densities)
    configs = [
        (200,  5_000, 0.6, 0.95, 2, [1.0, 0.7]),
        (500, 20_000, 0.6, 0.95, 2, [1.0, 0.5]),
    ]
    num_perm = 128
    n_runs = 3

    for n_cols, n_elem, j_th, oc_th, n_dfs, densities in configs:
        config_label = f"{n_cols} columns / {n_elem:,} elements / J≥{j_th} / O≥{oc_th}"
        print(f"--- {config_label} ---")
        print()

        print("Generating test data...")
        lazyframes, expected_pairs = generate_test_data(
            n_cols, n_elem,
            density=0.15,
            n_dataframes=n_dfs,
            jaccard_threshold=j_th,
            overlap_threshold=oc_th,
            dataframe_densities=densities,
        )
        n_expected = sum(len(v) for v in expected_pairs.values())
        print(f"  Expected similar pairs: {n_expected} "
              f"(jaccard_only={len(expected_pairs['jaccard_only'])}, "
              f"overlap_only={len(expected_pairs['overlap_only'])}, "
              f"both={len(expected_pairs['both'])})")
        print()

        # ── Brute force (ground truth, single run) ────────────────────────
        print("Running brute force (1 run — ground truth)...")
        brute_df, brute_time = _run_brute_force(lazyframes, j_th, oc_th)
        brute_pairs = _extract_pair_set(brute_df)
        print(f"  Time: {brute_time * 1000:.1f}ms  |  Found: {len(brute_pairs)} pairs")
        print()

        # ── datasketch (3 runs) ───────────────────────────────────────────
        print(f"Running datasketch ({n_runs} runs)...")
        ds_times: List[float] = []
        ds_pairs: Set[Tuple[str, str]] = set()
        for run in range(n_runs):
            df, t = _run_datasketch(lazyframes, j_th, oc_th, num_perm)
            ds_times.append(t)
            ds_pairs = _extract_pair_set(df)
            print(f"  Run {run + 1}: {t * 1000:.1f}ms")
        ds_avg = sum(ds_times) / len(ds_times)
        ds_fn = len(brute_pairs - ds_pairs)
        ds_fp = len(ds_pairs - brute_pairs)
        ds_recall = 1.0 - ds_fn / len(brute_pairs) if brute_pairs else 1.0
        print(f"  Average: {ds_avg * 1000:.1f}ms  |  "
              f"Found: {len(ds_pairs)}  |  FN: {ds_fn}  FP: {ds_fp}  |  Recall: {ds_recall:.3f}")
        print()

        # ── Rust plugin (3 runs) ──────────────────────────────────────────
        print(f"Running Rust plugin ({n_runs} runs)...")
        rust_times: List[float] = []
        rust_pairs: Set[Tuple[str, str]] = set()
        for run in range(n_runs):
            df, t = _run_rust(lazyframes, j_th, oc_th, num_perm)
            rust_times.append(t)
            rust_pairs = _extract_pair_set(df)
            print(f"  Run {run + 1}: {t * 1000:.1f}ms")
        rust_avg = sum(rust_times) / len(rust_times)
        rust_fn = len(brute_pairs - rust_pairs)
        rust_fp = len(rust_pairs - brute_pairs)
        rust_recall = 1.0 - rust_fn / len(brute_pairs) if brute_pairs else 1.0
        print(f"  Average: {rust_avg * 1000:.1f}ms  |  "
              f"Found: {len(rust_pairs)}  |  FN: {rust_fn}  FP: {rust_fp}  |  Recall: {rust_recall:.3f}")
        print()

        # ── Summary ───────────────────────────────────────────────────────
        print("Summary:")
        print(f"  Brute force:   {brute_time * 1000:.1f}ms  (ground truth)")
        print(f"  datasketch:    {ds_avg * 1000:.1f}ms  "
              f"speedup vs brute: {brute_time / ds_avg:.1f}x  recall: {ds_recall:.3f}")
        if rust_avg > 0:
            print(f"  Rust plugin:   {rust_avg * 1000:.1f}ms  "
                  f"speedup vs brute: {brute_time / rust_avg:.1f}x  "
                  f"speedup vs datasketch: {ds_avg / rust_avg:.1f}x  "
                  f"recall: {rust_recall:.3f}")
        print()
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
