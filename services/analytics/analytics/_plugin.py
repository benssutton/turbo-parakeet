from typing import TYPE_CHECKING, Any, Sequence
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

"""
Bloom Filter
"""

@pl.api.register_expr_namespace("analytics")
class AnalyticFunctions:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr
    #TODO: pyo3 supports bytes in version 27 and above, however pyo3-polars currently requires version 26.  When pyo3-polars upgrades to pyo3 v27 or above, pass bytes between python and the rust plugin
    def bloom_filter(self,
                        existing_filter: list[int],
                        k: int,
                        m: int) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="bloom_filter",
            args=self._expr,
            kwargs={
                "bit_array_bytes": existing_filter or [],
                "k": k,
                "m": m
            },
            is_elementwise=True,
        )

    def membership(self,
                        bit_array_bytes: list[int],
                        k: int,
                        m: int) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="membership",
            args=self._expr,
            kwargs={
                "bit_array_bytes": bit_array_bytes,
                "k": k,
                "m": m
            },
            is_elementwise=True,
        )

def membership_ratio(
    df: pl.DataFrame | pl.LazyFrame,
    bit_array_bytes: list[int],
    k: int,
    m: int,
) -> pl.DataFrame:
    """
    Calculate membership ratio for each column independently against a bloom filter.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.
    bit_array_bytes : list[int]
        The bloom filter bit array as a list of bytes.
    k : int
        Number of hash functions used in the bloom filter.
    m : int
        Size of the bloom filter bit array in bytes.

    Returns
    -------
    pl.DataFrame
        Single column "membership_ratio" containing structs with:
        - col_name: String - Column name
        - ratio_all: f64 - Fraction found (nulls counted in denominator)
        - ratio_non_null: f64 - Fraction found (nulls excluded from denominator)
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="membership_ratio",
            args=df.get_columns(),
            kwargs={"bit_array_bytes": bit_array_bytes, "k": k, "m": m},
            is_elementwise=False,
        ).alias("membership_ratio")
    )

def membership_ratio_sample(
    df: pl.DataFrame | pl.LazyFrame,
    bit_array_bytes: list[int],
    k: int,
    m: int,
    sample_frac: float = 0.05,
) -> pl.DataFrame:
    """
    Calculate membership ratio for each column independently, using a random sample.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.
    bit_array_bytes : list[int]
        The bloom filter bit array as a list of bytes.
    k : int
        Number of hash functions used in the bloom filter.
    m : int
        Size of the bloom filter bit array in bytes.
    sample_frac : float, optional
        Fraction of each column to sample (default 0.05 = 5%).

    Returns
    -------
    pl.DataFrame
        Single column "membership_ratio_sample" containing structs with:
        - col_name: String - Column name
        - ratio_all: f64 - Fraction found in sample (nulls in denominator)
        - ratio_non_null: f64 - Fraction found in sample (nulls excluded)
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="membership_ratio_sample",
            args=df.get_columns(),
            kwargs={"bit_array_bytes": bit_array_bytes, "k": k, "m": m, "sample_frac": sample_frac},
            is_elementwise=False,
        ).alias("membership_ratio_sample")
    )


"""
Entropy Calculation
"""

def pairwise_joint_entropy(
    df: pl.DataFrame | pl.LazyFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """
    Calculate joint entropy for pairwise column combinations.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data with columns to analyze. LazyFrames will be collected.
    pairs : list of (str, str) tuples, optional
        Specific column triplets to compute entropy for.
        When None (default), computes all N-choose-2 combinations.

    Returns
    -------
    pl.DataFrame
        Single column "pairwise_entropy" containing structs with:
        - col_a: String - First column name
        - col_b: String - Second column name
        - entropy: f64 - Joint entropy H(A,B) in bits
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs_dict = {
        "pairs": [list(p) for p in pairs] if pairs is not None else None
    }

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="pairwise_joint_entropy",
            args=df.get_columns(),
            kwargs=kwargs_dict,
            is_elementwise=False,
        ).alias("pairwise_entropy")
    )

def threeway_joint_entropy(
    df: pl.DataFrame | pl.LazyFrame,
    triplets: list[tuple[str, str, str]] | None = None,
) -> pl.DataFrame:
    """
    Calculate 3-way joint entropy for column triplet combinations (v2).

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data with columns to analyze. LazyFrames will be collected.
    triplets : list of (str, str, str) tuples, optional
        Specific column triplets to compute entropy for.
        When None (default), computes all N-choose-3 combinations (up to 5000).

    Returns
    -------
    pl.DataFrame
        Single column "threeway_entropy" containing structs with:
        - col_a: String - First column name
        - col_b: String - Second column name
        - col_c: String - Third column name
        - entropy: f64 - Joint entropy H(A,B,C) in bits
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs_dict = {
        "triplets": [list(t) for t in triplets] if triplets is not None else None
    }

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="threeway_joint_entropy",
            args=df.get_columns(),
            kwargs=kwargs_dict,
            is_elementwise=False,
        ).alias("threeway_entropy")
    )

def marginal_entropy(
    df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """
    Calculate marginal (single-column) entropy for every column independently.

    Diagnostic/independent cross-check: computed via the shared null-safe
    encoder and the SIMD entropy reduction, but deliberately NOT the dense-id
    counting path pairwise/threeway joint entropy use — useful for isolating
    whether a discrepancy originates in per-column encoding/SIMD reduction
    versus the joint-entropy-specific counting machinery.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data with columns to analyze. LazyFrames will be collected.

    Returns
    -------
    pl.DataFrame
        Single column "marginal_entropy" containing structs with:
        - col_name: String - Column name
        - entropy: f64 - Marginal entropy H(col) in bits
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="marginal_entropy",
            args=df.get_columns(),
            kwargs={},
            is_elementwise=False,
        ).alias("marginal_entropy")
    )

"""
Chi Squared Independence Test
"""

def pairwise_chi_squared(
    df: pl.DataFrame | pl.LazyFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """
    Calculate pairwise chi-squared independence statistics using the Rust plugin.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.
    pairs : list of (str, str) tuples, optional
        Specific column pairs to test.
        When None (default), computes all N-choose-2 combinations.

    Returns
    -------
    pl.DataFrame
        Single column "pairwise_chi_squared" containing structs with:
        - col_a: String - First column name
        - col_b: String - Second column name
        - chi2_stat: f64 - Chi-squared test statistic
        - p_value: f64 - P-value for the independence test
        - cramers_v: f64 - Cramer's V effect size (0=no association, 1=perfect)
        - low_expected_count: bool - True when min expected cell count < 5
        - n_valid: u32 - Rows remaining after null-dropping
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs_dict = {
        "pairs": [list(p) for p in pairs] if pairs is not None else None
    }

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="pairwise_chi_squared",
            args=df.get_columns(),
            kwargs=kwargs_dict,
            is_elementwise=False,
        ).alias("pairwise_chi_squared")
    )

"""
Adjusted Rand Index
"""

def pairwise_adjusted_rand(
    df: pl.DataFrame | pl.LazyFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """
    Calculate pairwise Adjusted Rand Index using the Rust plugin.

    Each column is treated as a partition of the rows (rows sharing a value
    form one cluster). ARI measures chance-corrected agreement between two
    partitions: 1.0 = identical partitions, ~0 = chance-level agreement,
    negative (floor -0.5) = worse than chance.

    Null policy: rows where either column is null are dropped (pairwise
    deletion). Edge conventions match sklearn.metrics.adjusted_rand_score:
    no overlapping non-null rows -> NaN; degenerate denominator (e.g. both
    columns constant) -> 1.0.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.
    pairs : list of (str, str) tuples, optional
        Specific column pairs to score.
        When None (default), computes all N-choose-2 combinations.

    Returns
    -------
    pl.DataFrame
        Single column "pairwise_adjusted_rand" containing structs with:
        - col_a: String - First column name
        - col_b: String - Second column name
        - ari: f64 - Adjusted Rand Index
        - n_valid: u32 - Rows remaining after null-dropping
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs_dict = {
        "pairs": [list(p) for p in pairs] if pairs is not None else None
    }

    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="pairwise_adjusted_rand",
            args=df.get_columns(),
            kwargs=kwargs_dict,
            is_elementwise=False,
        ).alias("pairwise_adjusted_rand")
    )

"""
Column GCD
"""

_COLUMN_GCD_SCHEMA = pl.Struct({"column": pl.String, "dtype": pl.String, "gcd": pl.Int128})
_UINT128 = getattr(pl, "UInt128", None)


def _contains_uint128(dtype: pl.DataType) -> bool:
    """True if `dtype` is, or nests, UInt128 (unknown to the plugin's Rust polars)."""
    if _UINT128 is not None and dtype == _UINT128:
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _contains_uint128(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_contains_uint128(f.dtype) for f in dtype.fields)
    return False


def column_gcd(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Whole-column greatest common divisor, using the Rust plugin.

    Follows ClickHouse's GCD codec: the GCD of the magnitudes of each column's
    raw physical integer values. Results are in physical units — Decimal uses
    the unscaled integer (Decimal(10,2) in 0.25 steps -> 25), Date uses days,
    Datetime/Duration use their time unit, Time uses nanoseconds.

    Null policy: nulls are skipped. All-null, all-zero and zero-row columns
    -> 0. Non-integer-backed dtypes (float, string, boolean, categorical,
    nested, ...) -> null. A magnitude of 2**127 (only i128::MIN values) is not
    representable as Int128 -> null.

    UInt128 columns (and nested types containing UInt128) are not passed to the
    plugin — its Rust polars has no UInt128, and crossing the FFI with one
    aborts the interpreter. They get gcd = null and their Python dtype name
    (e.g. "UInt128") as the dtype label.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input data. LazyFrames will be collected.

    Returns
    -------
    pl.DataFrame
        Single column "column_gcd" containing one struct per input column,
        in input order:
        - column: String - Column name
        - dtype: String - Polars dtype (Rust Display form, e.g. "datetime[μs]")
        - gcd: Int128 - Whole-column GCD, or null if not applicable
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if df.width == 0:
        return pl.DataFrame(schema={"column_gcd": _COLUMN_GCD_SCHEMA})
    skipped = {c: str(dt) for c, dt in df.schema.items() if _contains_uint128(dt)}
    if skipped:
        computed = column_gcd(df.drop(list(skipped))).unnest("column_gcd")
        by_name = {r["column"]: r for r in computed.iter_rows(named=True)}
        rows = [
            by_name[c] if c in by_name else {"column": c, "dtype": skipped[c], "gcd": None}
            for c in df.columns
        ]
        return pl.DataFrame({"column_gcd": rows}, schema={"column_gcd": _COLUMN_GCD_SCHEMA})
    return df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="column_gcd",
            args=df.get_columns(),
            is_elementwise=False,
            changes_length=True,
        ).alias("column_gcd")
    )

"""
Min Hash LSH
"""

def minhash(
    df: pl.DataFrame | pl.LazyFrame,
    name: str,
    num_perm: int = 128,
) -> pl.DataFrame:
    """
    Compute MinHash signatures for all columns in a DataFrame.

    This function processes all columns in parallel using a Rust implementation,
    returning a DataFrame with qualified names and MinHash signatures that is
    directly compatible with find_lsh_candidates().

    Args:
        df: DataFrame with columns to compute MinHash signatures for.
            Each column should contain the unique values to hash.
        name: Name prefix for qualified column names (e.g., "df0").
              Column names will be formatted as "{name}|{column_name}".
        num_perm: Number of hash permutations (default: 128).
                 Higher values give more accurate similarity estimates.

    Returns:
        DataFrame with columns:
        - qualified_name: String column with "{name}|{column_name}" format
        - minhash: List[UInt32] column with MinHash signatures

    Example:
        >>> from analytics import compute_minhash_batch, find_lsh_candidates
        >>> df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> minhashes = compute_minhash_batch(df, "df0", num_perm=64)
        >>> # minhashes has columns: qualified_name, minhash
        >>> # Can be directly used with find_lsh_candidates
    """
    # Pack all columns into a struct expression
    struct_expr = pl.struct(pl.all())

    # Call the plugin - returns M rows (one per column), using changes_length=True
    # to allow output length to differ from input length
    result = df.select(
        register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="minhash",
            args=struct_expr,
            kwargs={"df_name": name, "num_perm": num_perm},
            is_elementwise=False,
            changes_length=True,
        )
    )

    # Unnest the struct to get qualified_name and minhash columns
    return result.unnest(result.columns[0])

def lsh_candidates(
    names: IntoExpr,
    signatures: IntoExpr,
    threshold: float = 0.5,
    num_bands: int | None = None,
    rows_per_band: int | None = None
) -> pl.Expr:
    """
    Find candidate pairs using Locality Sensitive Hashing on MinHash signatures.

    LSH partitions MinHash signatures into bands and finds items that
    hash to the same bucket in at least one band.

    Args:
        names: Expression for the qualified names column (Utf8/String)
        signatures: Expression for the MinHash signatures column (List[UInt32])
        threshold: Jaccard similarity threshold (default: 0.5)
        num_bands: Number of bands to partition signature into.
        rows_per_band: Rows per band. If None, automatically computed.
        num_perm: Number of permutations used for MinHash (default: 128).

    Returns:
        Polars expression returning candidate pairs as a Struct with
        fields (col_a: Utf8, col_b: Utf8).

    Example:
        >>> from analytics import find_lsh_candidates
        >>> df.select(find_lsh_candidates(
        ...     pl.col("name"),
        ...     pl.col("minhash"),
        ...     threshold=0.6
        ... ))
    """

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="lsh_candidates",
        args=[names, signatures],
        kwargs={
            "threshold": threshold,
            "num_bands": num_bands,
            "rows_per_band": rows_per_band,
        },
        is_elementwise=False,
    )