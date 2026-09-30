"""Private thin wrappers around the compiled Rust extension (analytics.pyd).

Tables cross the boundary as Arrow: the Polars frames given here are passed as they
are (they implement the Arrow PyCapsule interface, zero-copy) and results come back
as Arrow tables, read into flat Polars frames. Only the *Rust implementation classes
call these; the public API is the technique classes in analytics.<technique>.
"""

from collections.abc import Sequence

import polars as pl

from analytics import analytics as _rs

Frame = pl.DataFrame | pl.LazyFrame


def _collected(df: Frame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _tuples(
    combos: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str]] | None,
) -> list[tuple[str, str]] | list[tuple[str, str, str]] | None:
    return None if combos is None else [tuple(c) for c in combos]


def bloom_filter_bits(series: pl.Series, k: int, m: int) -> bytes:
    """A fresh k-hash, m-bit Bloom filter over `series`: its ceil(m/8) bytes."""
    return _rs.bloom_filter(series.to_frame(), k, m)


def membership_ratio(df: Frame, bits: bytes, k: int, m: int) -> pl.DataFrame:
    """Per column: col_name, ratio_all (nulls in the denominator), ratio_non_null."""
    return pl.DataFrame(_rs.membership_ratio(_collected(df), bits, k, m))


def pairwise_joint_entropy(
    df: Frame, pairs: Sequence[tuple[str, str]] | None = None
) -> pl.DataFrame:
    """col_a, col_b, entropy (H(A,B) in bits); every pair when `pairs` is None."""
    return pl.DataFrame(_rs.pairwise_joint_entropy(_collected(df), _tuples(pairs)))


def threeway_joint_entropy(
    df: Frame, triplets: Sequence[tuple[str, str, str]] | None = None
) -> pl.DataFrame:
    """col_a, col_b, col_c, entropy (H(A,B,C) in bits); every triplet when None —
    no cap (C(101, 3) = 166,650 at 101 columns, ~65 s at 50K rows)."""
    return pl.DataFrame(_rs.threeway_joint_entropy(_collected(df), _tuples(triplets)))


def marginal_entropy(df: Frame) -> pl.DataFrame:
    """col_name, entropy per column: the shared encoder and SIMD reduction, not the
    dense-id counting the joint entropies use (an independent cross-check)."""
    return pl.DataFrame(_rs.marginal_entropy(_collected(df)))


def pairwise_chi_squared(
    df: Frame, pairs: Sequence[tuple[str, str]] | None = None
) -> pl.DataFrame:
    """col_a, col_b, chi2_stat, p_value, cramers_v, low_expected_count, n_valid."""
    return pl.DataFrame(_rs.pairwise_chi_squared(_collected(df), _tuples(pairs)))


def pairwise_adjusted_rand(
    df: Frame, pairs: Sequence[tuple[str, str]] | None = None
) -> pl.DataFrame:
    """col_a, col_b, ari, n_valid; null rows dropped pairwise (sklearn conventions)."""
    return pl.DataFrame(_rs.pairwise_adjusted_rand(_collected(df), _tuples(pairs)))


def column_gcd(df: Frame) -> pl.DataFrame:
    """column, dtype (Rust display form), gcd: Decimal(38, 0) per column."""
    return pl.DataFrame(_rs.column_gcd(_collected(df)))


def minhash(df: Frame, name: str, num_perm: int = 128) -> pl.DataFrame:
    """qualified_name ("{name}|{column}"), minhash: List(UInt32) per column."""
    return pl.DataFrame(_rs.minhash(_collected(df), name, num_perm))


def lsh_candidates(
    signatures: pl.DataFrame, num_bands: int, rows_per_band: int
) -> pl.DataFrame:
    """col_a, col_b per LSH candidate pair of `signatures` (as `minhash` returns them)."""
    return pl.DataFrame(
        _rs.lsh_candidates(
            signatures.select("qualified_name", "minhash"), num_bands, rows_per_band
        )
    )


def describe_columns(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """column + Describe's value metrics per column (see analytics.describe.base)."""
    return pl.DataFrame(_rs.describe_columns(df, seed))


def column_sizes(df: pl.DataFrame, zstd_level: int) -> pl.DataFrame:
    """column + Arrow (classic layout) and Polars (native layout) IPC body sizes."""
    return pl.DataFrame(_rs.column_sizes(df, zstd_level))


def describe_and_recommend(
    df: pl.DataFrame,
    *,
    seed: int,
    zstd_level: int,
    population_rows: int | None,
    categorical_threshold: int,
    boolean_pairs: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    """column, every Describe metric, the size metrics and the rec_* columns."""
    return pl.DataFrame(
        _rs.describe_and_recommend(
            df,
            seed=seed,
            zstd_level=zstd_level,
            population_rows=population_rows,
            categorical_threshold=categorical_threshold,
            boolean_pairs=[tuple(p) for p in boolean_pairs],
        )
    )


def streaming_recommender(
    *,
    reservoir_rows: int,
    block_rows: int,
    categorical_threshold: int,
    zstd_level: int,
    seed: int,
    boolean_pairs: tuple[tuple[str, str], ...],
):
    """The Rust streaming recommender (src/streaming.rs): .add(data, ineligible), .finish()."""
    return _rs.StreamingRecommender(
        reservoir_rows=reservoir_rows,
        block_rows=block_rows,
        categorical_threshold=categorical_threshold,
        zstd_level=zstd_level,
        seed=seed,
        boolean_pairs=[tuple(p) for p in boolean_pairs],
    )
