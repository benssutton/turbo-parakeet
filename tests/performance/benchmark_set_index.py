"""
Set skip-index speed benchmark: SetIndexRust vs SetIndexPolars (reference).

Run: python tests/performance/benchmark_set_index.py

Default window (8 192 rows) and max_size (100). The datasets of benchmark_minmax_index.py:
clustered columns stay under max_size; random ones overflow every window. Results: spec
2026-10-16-skip-index-techniques §10.
"""

from benchmark_minmax_index import DATASETS
from harness import run

if __name__ == "__main__":
    run("turbo_parakeet.techniques.set_index", DATASETS)
