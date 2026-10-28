"""
Top-k benchmark: TopKHeavyKeeperRust and TopKSpaceSavingRust (estimators) vs TopKPolars
(reference): speed, peak memory and accuracy (harness.py).

Run: python tests/performance/benchmark_top_k.py [--rows N]

Shapes: synth_frame (50 000 rows, the covering mix); narrow / long (parallelism inside a column:
SpaceSaving merges 2^20-row chunks, HeavyKeeper cannot); wide (parallelism across columns);
one 10M x 1 dataset per stream of synthdata.scenarios.TOP_K_COLUMNS (Zipf α = 1.5 / 1.1 / 0.8, flat,
skewed, string), one column each so precision and memory are per stream.
Variants: k ∈ {5, 256} × memory_bytes ∈ {¼, 1, 4} × max(16 KiB, k × 1 KiB); the reference
runs once per k. Accuracy is against exact counts (TopK.accuracy): tie-aware precision,
are / aae, SpaceSaving's max_bound_violation and sketch_bytes. `agrees` on synth_frame can
be false from ties at the k-th count (agreement precision is a lower bound there).

--rows N: the scale run instead: zipf_11 alone at N rows, k = 5 at 1× budget
(results/top_k_scale_<stamp>.arrow).
"""

import argparse

from harness import Dataset, Variant, synth_dataset, run

from synthdata.scenarios import TOP_K_COLUMNS, low_cardinality, top_k_streams
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
                synth_dataset(),
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
