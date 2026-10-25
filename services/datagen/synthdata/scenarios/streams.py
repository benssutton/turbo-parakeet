"""Streams for top-k and population-cardinality estimation."""

from __future__ import annotations

import numpy as np
import polars as pl

TOP_K_COLUMNS = ("zipf_15", "zipf_11", "zipf_08", "flat", "skewed", "zipf_str")


def _bounded_zipf(n_rows: int, n_values: int, alpha: float, rng) -> np.ndarray:
    """Zipf(alpha) truncated to n_values ranks (any alpha > 0, unlike numpy's zipf), mapped
    through a random permutation so the most frequent values are not 0, 1, 2, …"""
    cdf = np.cumsum(np.arange(1, n_values + 1, dtype=np.float64) ** -alpha)
    cdf /= cdf[-1]
    ranks = np.minimum(
        np.searchsorted(cdf, rng.random(n_rows), side="right"), n_values - 1
    )
    return rng.permutation(n_values)[ranks]


def top_k_streams(
    n_rows: int = 10_000_000, seed: int = 42, columns: list[str] | None = None
) -> pl.DataFrame:
    """Int64 streams for top-k sketches over 1M possible values: zipf_15 / zipf_11 / zipf_08
    (Zipf α = 1.5 / 1.1 / 0.8), flat (uniform), skewed (95% one value, the rest over 50,000
    others) and zipf_str (zipf_11 as "item_<n>" strings). Each column has its own seeded
    generator, so a column is the same whichever `columns` are asked for."""
    columns = list(TOP_K_COLUMNS) if columns is None else columns

    def rng(name: str):
        return np.random.default_rng([seed, TOP_K_COLUMNS.index(name)])

    def make(name: str) -> pl.Series:
        if name == "zipf_str":
            return ("item_" + make("zipf_11").cast(pl.String)).alias(name)
        r = rng(name)
        if name == "flat":
            values = r.integers(0, 1_000_000, n_rows)
        elif name == "skewed":
            values = np.where(r.random(n_rows) < 0.95, 0, r.integers(1, 50_001, n_rows))
        else:
            alpha = {"zipf_15": 1.5, "zipf_11": 1.1, "zipf_08": 0.8}[name]
            values = _bounded_zipf(n_rows, 1_000_000, alpha, r)
        return pl.Series(name, values.astype(np.int64))

    return pl.DataFrame([make(c) for c in columns])


def population_samples(
    n_rows: int, populations: list[tuple[int, str | tuple[str, float]]], seed: int = 42
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Int64 columns p0… each drawing `n_rows` values with replacement from a population
    of exactly D labels (`populations`: (D, "uniform") or (D, ("zipf", alpha))), labels
    randomly permuted so frequent values are not small integers. Returns the frame and
    each column's D — the support size, the truth Chao1 / Schnabel estimate."""
    rng = np.random.default_rng(seed)
    cols, truth = {}, {}
    for i, (d, shape) in enumerate(populations):
        if shape == "uniform":
            ranks = rng.integers(0, d, n_rows)
        else:
            # `_bounded_zipf` returns values in [0, d) (ranks already permuted)
            ranks = _bounded_zipf(n_rows, d, shape[1], rng)
        labels = rng.permutation(d).astype(np.int64)
        cols[f"p{i}"] = labels[ranks]
        truth[f"p{i}"] = d
    return pl.DataFrame(cols), truth
