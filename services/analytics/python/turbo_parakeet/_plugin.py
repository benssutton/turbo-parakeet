"""Private thin wrappers around the compiled Rust extension (turbo_parakeet._core).

Tables cross the boundary as Arrow: the Polars frames given here are passed as they
are (they implement the Arrow PyCapsule interface, zero-copy) and results come back
as Arrow tables, read into flat Polars frames. Callers: the *Rust implementation
classes, turbo_parakeet.techniques.base (`checked_table`, imported lazily when Arrow input is added)
and turbo_parakeet.recommenders's recommenders (`streaming_recommender`,
`oneshot_recommender`); the public API is the technique classes in
turbo_parakeet.techniques.<technique> and the two recommenders.
"""

from collections.abc import Sequence

import polars as pl

from turbo_parakeet import _core as _rs

Frame = pl.DataFrame | pl.LazyFrame


def _collected(df: Frame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _tuples(
    combos: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str]] | None,
) -> list[tuple[str, str]] | list[tuple[str, str, str]] | None:
    return None if combos is None else [tuple(c) for c in combos]


def checked_table(data):
    """Arrow tabular data (`__arrow_c_stream__`) read into one table after the Rust
    boundary's checks (structure, values, types Polars cannot import): ValueError
    naming the column otherwise. For turbo_parakeet.techniques.base, before Polars reads the data.
    """
    return _rs.checked_table(data)


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


def ams_marginal_entropy(df: Frame, num_samples: int, seed: int) -> pl.DataFrame:
    """col_name, entropy per column, estimated by the AMS (CCM) sketch."""
    return pl.DataFrame(_rs.ams_marginal_entropy(_collected(df), num_samples, seed))


def ams_pairwise_joint_entropy(
    df: Frame,
    pairs: Sequence[tuple[str, str]] | None,
    num_samples: int,
    seed: int,
) -> pl.DataFrame:
    """col_a, col_b, entropy (H(A,B) in bits, AMS estimate); every pair when None."""
    return pl.DataFrame(
        _rs.ams_pairwise_joint_entropy(
            _collected(df), _tuples(pairs), num_samples, seed
        )
    )


def ams_threeway_joint_entropy(
    df: Frame,
    triplets: Sequence[tuple[str, str, str]] | None,
    num_samples: int,
    seed: int,
) -> pl.DataFrame:
    """col_a, col_b, col_c, entropy (H(A,B,C) in bits, AMS estimate); every triplet when None."""
    return pl.DataFrame(
        _rs.ams_threeway_joint_entropy(
            _collected(df), _tuples(triplets), num_samples, seed
        )
    )


def space_saving_top_k(df: Frame, k: int, memory_bytes: int) -> pl.DataFrame:
    """col_name, rank, row, count, error, n_non_null, sketch_bytes: each column's top k by
    SpaceSaving (one row per rank; one null-rank row for a column without values)."""
    return pl.DataFrame(_rs.space_saving_top_k(_collected(df), k, memory_bytes))


def heavy_keeper_top_k(df: Frame, k: int, memory_bytes: int, seed: int) -> pl.DataFrame:
    """As space_saving_top_k, by HeavyKeeper (error null)."""
    return pl.DataFrame(_rs.heavy_keeper_top_k(_collected(df), k, memory_bytes, seed))


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


def runs_test(df: Frame) -> pl.DataFrame:
    """column, n_rows, n_runs, expected_runs, z, p_value (one-sided, lower tail),
    mean_run_length per column — rows in the frame's order."""
    return pl.DataFrame(_rs.runs_test(_collected(df)))


def delta(df: Frame) -> pl.DataFrame:
    """column + Delta's metrics per column — rows in the frame's order."""
    return pl.DataFrame(_rs.delta(_collected(df)))


def minmax_index(df: Frame, window_rows: int) -> pl.DataFrame:
    """column + MinMaxIndex's metrics per column — windows of `window_rows` rows sliding
    over the frame's row order."""
    return pl.DataFrame(_rs.minmax_index(_collected(df), window_rows))


def set_index(df: Frame, window_rows: int, max_size: int) -> pl.DataFrame:
    """column + SetIndex's metrics per column — windows of `window_rows` rows sliding over
    the frame's row order; `max_size` 0 = unlimited."""
    return pl.DataFrame(_rs.set_index(_collected(df), window_rows, max_size))


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


def describe_columns(
    df: pl.DataFrame, seed: int, categorical_threshold: int
) -> pl.DataFrame:
    """column + Describe's value metrics and Rust's conclusions per column, then the
    private estimator inputs (argmin / argmax, f1, f2, capture history; spec
    2026-10-01 §13.8)."""
    return pl.DataFrame(_rs.describe_columns(df, seed, categorical_threshold))


def column_sizes(df: pl.DataFrame, zstd_level: int) -> pl.DataFrame:
    """column + Arrow (classic layout) and Polars (native layout) IPC body sizes."""
    return pl.DataFrame(_rs.column_sizes(df, zstd_level))


def ipc_body_sizes(table, codec: str) -> dict[str, int]:
    """Arrow IPC body bytes per column of an Arrow table under `codec` ("none", "lz4",
    "zstd:<level>"). Private: tests and the calibration script only."""
    out = pl.DataFrame(_rs.ipc_body_sizes(table, codec))
    return dict(zip(out["column"], out["bytes"]))


def clickhouse_profile(table, column: str, codecs: str) -> dict:
    """ClickHouse type of `column`'s own Arrow type (None: unmappable) and its emulated size
    under the codec chain `codecs` ("LZ4", "ZSTD(3)", "GCD, LZ4", "GCD, ZSTD(3)"; None when
    unmappable or GCD does not apply). Private: tests and `to_chbench` only."""
    row = pl.DataFrame(_rs.clickhouse_profile(table, column, codecs)).row(0, named=True)
    return {"clickhouse_type": row["clickhouse_type"], "bytes": row["bytes"]}


def render(s: pl.Series) -> list[str | None]:
    """`s`'s values as text, rendered by arrow-rs's cast to Utf8 — the one format of
    Describe's / Recommend's min and max (recommend.rs `render_value`)."""
    return pl.DataFrame(_rs.render(s.to_frame())).to_series().to_list()


def streaming_recommender(
    *,
    reservoir_rows: int,
    block_rows: int,
    categorical_threshold: int,
    top_k: int | None,
    zstd_level: int,
    seed: int,
    boolean_pairs: tuple[tuple[str, str], ...],
    entropy_estimator: str,
    id_columns: list[str],
    zstd_levels: list[int],
    zstd_level_tolerance: float,
    zstd_min_saving: float,
    gcd_min_saving: float,
    skip_granularities: list[int],
    set_max_sizes: list[int],
    bloom_fprs: list[float],
    min_skip: float,
    min_gain: float,
    max_overflow: float,
):
    """The Rust streaming recommender (src/recommenders/streaming/): .add(data, ineligible), .result()."""
    return _rs.StreamingRecommender(
        reservoir_rows=reservoir_rows,
        block_rows=block_rows,
        categorical_threshold=categorical_threshold,
        top_k=top_k,
        zstd_level=zstd_level,
        seed=seed,
        boolean_pairs=[tuple(p) for p in boolean_pairs],
        entropy_estimator=entropy_estimator,
        id_columns=id_columns,
        zstd_levels=zstd_levels,
        zstd_level_tolerance=zstd_level_tolerance,
        zstd_min_saving=zstd_min_saving,
        gcd_min_saving=gcd_min_saving,
        skip_granularities=skip_granularities,
        set_max_sizes=set_max_sizes,
        bloom_fprs=bloom_fprs,
        min_skip=min_skip,
        min_gain=min_gain,
        max_overflow=max_overflow,
    )


def oneshot_recommender(
    *,
    categorical_threshold: int,
    top_k: int | None,
    zstd_level: int,
    seed: int,
    boolean_pairs: tuple[tuple[str, str], ...],
    id_columns: list[str],
    zstd_levels: list[int],
    zstd_level_tolerance: float,
    zstd_min_saving: float,
    gcd_min_saving: float,
    skip_granularities: list[int],
    set_max_sizes: list[int],
    bloom_fprs: list[float],
    min_skip: float,
    min_gain: float,
    max_overflow: float,
):
    """The Rust one-shot recommender (src/recommenders/oneshot.rs): .add(data, ineligible), .result()."""
    return _rs.OneShotRecommender(
        id_columns=id_columns,
        categorical_threshold=categorical_threshold,
        top_k=top_k,
        zstd_level=zstd_level,
        seed=seed,
        boolean_pairs=[tuple(p) for p in boolean_pairs],
        zstd_levels=zstd_levels,
        zstd_level_tolerance=zstd_level_tolerance,
        zstd_min_saving=zstd_min_saving,
        gcd_min_saving=gcd_min_saving,
        skip_granularities=skip_granularities,
        set_max_sizes=set_max_sizes,
        bloom_fprs=bloom_fprs,
        min_skip=min_skip,
        min_gain=min_gain,
        max_overflow=max_overflow,
    )
