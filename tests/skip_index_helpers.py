"""Shared by tests/test_minmax_index.py and tests/test_set_index.py: dtypes to cover, a
brute-force oracle evaluating every window directly from the spec's definitions
(docs/superpowers/specs/2026-10-16-skip-index-techniques-design.md §2.3), and a Monte Carlo of
the random-order baselines (under a random order every window is a uniform w-subset of the
rows)."""

import math
from collections import Counter

import numpy as np
import polars as pl
import pytest

ELIGIBLE_DTYPES = [
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
    pl.Decimal(10, 2),
    pl.Date(),
    pl.Datetime("us"),
    pl.Datetime("ns", "UTC"),
    pl.Duration("ms"),
    pl.Time(),
    pl.String(),
    pl.Binary(),
    pl.Categorical(),
    pl.Enum(["0", "1", "2"]),
]


def as_dtype(values: list[int], dtype: pl.DataType) -> pl.Series:
    """Single-digit ints keep their order under every cast (strings: "0" < "1" < "2")."""
    s = pl.Series("x", values)
    if dtype in (pl.String, pl.Binary, pl.Categorical) or isinstance(dtype, pl.Enum):
        return s.cast(pl.String).cast(dtype)
    return s.cast(dtype)


def zipf_column(seed: int, k: int, n: int, alpha: float, null_rate: float) -> list:
    rng = np.random.default_rng(seed)
    p = np.arange(1, k + 1) ** -alpha
    p /= p.sum()
    return [
        None if rng.random() < null_rate else int(v) for v in rng.choice(k, size=n, p=p)
    ]


def _mean_std(a: list[int], den: int) -> tuple[float, float]:
    if not den:
        return math.nan, math.nan
    x = np.array(a, dtype=float) / den
    return float(x.mean()), float(x.std())


def oracle(values: list, window_rows: int, max_size: int = 0) -> tuple[dict, dict]:
    """(minmax, set) observed metrics of `values` (comparable Python values, None = null),
    every window evaluated directly."""
    n = len(values)
    w = min(n, window_rows)
    k = n - w + 1
    counts = Counter(v for v in values if v is not None)
    dist = sorted(counts)
    nn, d = sum(counts.values()), len(dist)
    mm_rows, mm_values, set_rows, set_values, entries = [], [], [], [], []
    for s in range(k):
        win = values[s : s + w]
        held = {v for v in win if v is not None}
        if held:
            lo, hi = min(held), max(held)
            outside = [v for v in dist if v < lo or v > hi]
            mm_rows.append(sum(counts[v] for v in outside))
            mm_values.append(len(outside))
        else:
            mm_rows.append(nn)
            mm_values.append(d)
        e = len(held) + (None in win)
        entries.append(e)
        if max_size and e > max_size:
            set_rows.append(0)
            set_values.append(0)
        else:
            set_rows.append(nn - sum(counts[v] for v in held))
            set_values.append(d - len(held))
    common = dict(n_rows=n, n_non_null=nn, n_distinct=d, window_rows=w, n_windows=k)
    mm, st = dict(common), dict(common)
    for out, rows, vals in ((mm, mm_rows, mm_values), (st, set_rows, set_values)):
        out["skip_rows_mean"], out["skip_rows_std"] = _mean_std(rows, nn)
        out["skip_values_mean"], out["skip_values_std"] = _mean_std(vals, d)
    st.update(
        window_distinct_mean=float(np.mean(entries)),
        window_distinct_max=max(entries),
        overflow_fraction=float(
            np.mean([max_size > 0 and e > max_size for e in entries])
        ),
    )
    return mm, st


def sampled(
    values: list, w: int, max_size: int = 0, samples: int = 20_000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """(mean, standard error) of each skip over uniformly random w-row subsets: keys
    mm_rows, mm_values, set_rows, set_values, entries, overflow."""
    rng = np.random.default_rng(seed)
    n = len(values)
    counts = Counter(v for v in values if v is not None)
    dist = sorted(counts)
    nn, d = sum(counts.values()), len(dist)
    rank = {v: i for i, v in enumerate(dist)}
    codes = np.array([-1 if v is None else rank[v] for v in values])
    cnt = np.array([counts[v] for v in dist])
    cum = np.concatenate([[0], np.cumsum(cnt)])
    keys = ("mm_rows", "mm_values", "set_rows", "set_values", "entries", "overflow")
    acc = {key: np.empty(samples) for key in keys}
    for i in range(samples):
        win = codes[rng.choice(n, w, replace=False)]
        held = np.unique(win[win >= 0])
        if held.size:
            lo, hi = held[0], held[-1]
            acc["mm_rows"][i] = (nn - (cum[hi + 1] - cum[lo])) / nn
            acc["mm_values"][i] = (d - (hi - lo + 1)) / d
        else:
            acc["mm_rows"][i] = acc["mm_values"][i] = 1.0
        e = held.size + bool((win < 0).any())
        over = max_size > 0 and e > max_size
        acc["entries"][i] = e
        acc["overflow"][i] = over
        acc["set_rows"][i] = 0 if over else (nn - cnt[held].sum()) / nn
        acc["set_values"][i] = 0 if over else (d - held.size) / d
    return {key: (a.mean(), a.std() / math.sqrt(samples)) for key, a in acc.items()}


def assert_metrics(
    got: dict, want: dict, rtol: float = 1e-9, atol: float = 1e-12
) -> None:
    for m, v in want.items():
        g = got[m]
        if isinstance(v, float) and math.isnan(v):
            assert isinstance(g, float) and math.isnan(g), f"{m}: {g!r} != nan"
        else:
            assert g == pytest.approx(v, rel=rtol, abs=atol), f"{m}: {g!r} != {v!r}"
