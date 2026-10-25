"""
Chi-squared speed benchmark: ChiSquaredRust vs ChiSquaredScipy (reference) vs ChiSquaredPolarsDS.

Run: python tests/performance/benchmark_chi_squared.py
"""

from harness import Dataset, cover_dataset, run

from synthdata.scenarios import low_cardinality

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.chi_squared",
        [
            cover_dataset(),
            Dataset("narrow 2M x 4", lambda: {"t": low_cardinality(2_000_000, 4)}),
            Dataset("wide 20K x 60", lambda: {"t": low_cardinality(20_000, 60)}),
        ],
    )
