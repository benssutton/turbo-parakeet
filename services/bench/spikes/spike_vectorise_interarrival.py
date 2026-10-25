"""Spike 8: can pure numpy remove the inter-arrival layout's Python loop?

A profile of `generate()` with an `interarrival_cv` target at
10 000 000 rows and cardinality 0.75 put 96 percent of the runtime in
`sequence.layout_interarrival`, whose `for j in range(m)` loop does one
`np.repeat` and two list appends per repeat-run, 2.5 million times per
layout, 55 layouts per generate. The deferred-findings note names this as the
first place a Rust extension would earn its keep. Before recommending Rust,
this spike asks whether the same layout can be expressed in numpy alone.

The vectorised function below makes exactly the same random draws in the
same order as the package function, so for the same generator state it must
produce the same array; the spike checks that with `np.array_equal`, times
both on the same full-scale multiset, and prints all three. The vectorised
function lives here only; it could be lifted into `chbench/` if the owner
agrees.

Run from the repository root or from inside spikes/: this file puts the
repository root on sys.path itself, resolved from its own __file__ rather
than the current directory, so `chbench` (installed in editable mode)
imports correctly either way.
"""

from __future__ import annotations

import pathlib
import sys
import time

# Resolve the repository root from this file, not the cwd, so the script runs
# correctly from any working directory (including spikes/ itself).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from synthdata.column.metrics import largest_remainder
from synthdata.column.multiset import build_multiset
from synthdata.column.sequence import layout_interarrival
from synthdata.column.spec import DatasetSpec

ROWS = 10_000_000


def layout_interarrival_vectorised(
    support: np.ndarray, counts: np.ndarray, cv: float, rng: np.random.Generator
) -> np.ndarray:
    """`sequence.layout_interarrival` with the per-run loop replaced.

    Everything up to `free_int` is copied verbatim so the rng is consumed
    identically. The loop built `seg0, run0, seg1, run1, ..., run(m-1), seg m`
    by appending 2m + 1 slices; here run j's start index is the sum of the
    segments 0..j and the runs 0..j-1, so all run positions come from one
    `np.repeat` over the starts plus a within-run offset, the runs are
    scattered into a preallocated array, and the singles fill the remaining
    positions in order.
    """
    support, counts = np.asarray(support), np.asarray(counts)
    is_run = counts >= 2
    run_vals, run_len = support[is_run], counts[is_run]
    singles = rng.permutation(support[~is_run])
    m, S = len(run_vals), len(singles)
    if m == 0:
        return singles
    order = rng.permutation(m)
    run_vals, run_len = run_vals[order], run_len[order]
    shape = 1.0 / cv**2
    mean_gap = S / (m + 1) + (run_len[:-1].mean() if m > 1 else 0.0)
    draws = rng.gamma(shape, mean_gap / shape, size=m + 1)
    interior = np.maximum(draws[: m - 1] - run_len[:-1], 0.0)
    free = np.concatenate(([draws[m - 1]], interior, [draws[m]]))
    free_int = largest_remainder(free, S)

    run_before = np.concatenate(([0], np.cumsum(run_len[:-1])))
    run_start = np.cumsum(free_int[:m]) + run_before
    total_run = int(run_len.sum())
    n = S + total_run
    within = np.arange(total_run) - np.repeat(run_before, run_len)
    run_pos = np.repeat(run_start, run_len) + within

    out = np.empty(n, dtype=support.dtype)
    is_run_pos = np.zeros(n, dtype=bool)
    is_run_pos[run_pos] = True
    out[run_pos] = np.repeat(run_vals, run_len)
    out[~is_run_pos] = singles
    return out


def main() -> None:
    spec = DatasetSpec(
        rows=ROWS,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=42,
        cardinality_ratio=0.75,
        interarrival_cv=1.0,
    )
    t0 = time.perf_counter()
    ms = build_multiset(spec, np.random.default_rng(spec.seed))
    print(
        f"multiset at {ROWS:,} rows: {time.perf_counter() - t0:.2f}s; "
        f"support {len(ms.support):,}, repeat-runs {(ms.counts >= 2).sum():,}, "
        f"singles {(ms.counts == 1).sum():,}"
    )

    t0 = time.perf_counter()
    a = layout_interarrival(ms.support, ms.counts, 1.0, np.random.default_rng(7))
    t_loop = time.perf_counter() - t0
    print(f"package layout_interarrival (Python loop): {t_loop:.2f}s")

    t0 = time.perf_counter()
    b = layout_interarrival_vectorised(
        ms.support, ms.counts, 1.0, np.random.default_rng(7)
    )
    t_vec = time.perf_counter() - t0
    print(f"vectorised layout (numpy only):            {t_vec:.2f}s")

    same = np.array_equal(a, b)
    print(
        f"np.array_equal(package, vectorised): {same}; "
        f"speedup {t_loop / t_vec:.1f}x"
    )
    if not same:
        raise SystemExit("vectorised layout differs from the package layout")


if __name__ == "__main__":
    main()
