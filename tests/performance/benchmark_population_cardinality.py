"""
Population-cardinality benchmark (speed, peak memory, accuracy; harness.py): the Chao1 /
Schnabel estimators, Rust vs the Polars / NumPy reference. Besides speed, the sample datasets
(8 columns drawn from populations of 10^3 .. 10^6 labels, uniform and Zipf 1.1) score every
estimate against the known truth (`observed_rel_err`, `chao1_rel_err`, `schnabel_rel_err`, ...).

Run: python tests/performance/benchmark_population_cardinality.py
"""

from functools import cache

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import population_samples

POPULATIONS = [
    (d, shape)
    for d in (1_000, 10_000, 100_000, 1_000_000)
    for shape in ("uniform", ("zipf", 1.1))
]


def samples(n_rows: int) -> Dataset:
    @cache
    def make():
        return population_samples(n_rows, POPULATIONS, seed=1)

    return Dataset(
        f"samples {n_rows:,} x 8",
        lambda: {"t": make()[0]},
        truth=lambda: {("t", c): float(d) for c, d in make()[1].items()},
    )


if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.population_cardinality",
        [cover_dataset(), samples(1_000_000), samples(100_000)],
    )
