"""
Whole-column GCD speed benchmark: GcdRust vs GcdNumpy vs GcdMath (reference).

Run: python tests/performance/benchmark_gcd.py

Shapes: cover (50 000 rows, the covering mix); narrow/long (parallelism inside a
column); wide (parallelism across columns); early exit (random values, GCD 1).
GcdMath is excluded from the 10^7–10^8-value shapes: to_list() alone would
exhaust memory.
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import integer_multiples, integer_random

G = 3_600

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.gcd",
        [
            cover_dataset(),
            Dataset(
                "narrow 10M x 4",
                lambda: {"t": integer_multiples(10_000_000, 4, G)},
                exclude=("GcdMath",),
            ),
            Dataset(
                "wide 1M x 100",
                lambda: {"t": integer_multiples(1_000_000, 100, G)},
                exclude=("GcdMath",),
            ),
            Dataset(
                "early exit 10M x 4",
                lambda: {"t": integer_random(10_000_000, 4)},
                exclude=("GcdMath",),
            ),
        ],
    )
