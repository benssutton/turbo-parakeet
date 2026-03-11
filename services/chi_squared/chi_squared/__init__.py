"""
Pairwise chi-squared independence tests on Polars DataFrames using polars-ds.

This is the pure-Python baseline implementation using polars-ds as the
statistical engine. It serves as a reference for correctness and a performance
baseline before any Rust-backed optimisation.
"""

from __future__ import annotations

import math
from itertools import combinations

import polars as pl
import polars_ds as pds

__version__ = "0.1.0"

# Dtypes eligible for chi-squared independence testing by default.
# These represent categorical / nominal data where the test is meaningful.
_SUITABLE_DTYPE_CLASSES = (
    pl.Boolean,
    pl.String,
    pl.Categorical,
    pl.Enum,
)

_INTEGER_DTYPE_CLASSES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)


def _is_suitable(dtype: pl.DataType, include_integer: bool) -> bool:
    if isinstance(dtype, _SUITABLE_DTYPE_CLASSES):
        return True
    if include_integer and isinstance(dtype, _INTEGER_DTYPE_CLASSES):
        return True
    return False


def _get_suitable_columns(
    df: pl.DataFrame,
    include_integer: bool = False,
    min_unique: int = 2,
    max_unique: int | None = 1000,
) -> list[str]:
    """Return column names with dtypes suitable for chi-squared testing.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to inspect.
    include_integer : bool, default False
        When True, integer columns are also considered.
    min_unique : int, default 2
        Exclude columns with fewer unique values than this threshold.
        A column with only 1 unique value is constant — chi-squared is
        undefined for it.
    max_unique : int or None, default 1000
        Exclude columns with more unique values than this threshold.
        Chi-squared on near-unique columns produces meaningless results and
        can cause memory exhaustion. Pass None to disable the check.
    """
    result = []
    for name, dtype in df.schema.items():
        if not _is_suitable(dtype, include_integer):
            continue
        n_unique = df[name].n_unique()
        if n_unique < min_unique:
            continue
        if max_unique is not None and n_unique > max_unique:
            continue
        result.append(name)
    return result


def pairwise_chi_squared(
    lf: pl.LazyFrame | pl.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    include_integer: bool = False,
    min_unique: int = 2,
    max_unique: int | None = 1000,
) -> pl.DataFrame:
    """
    Compute chi-squared independence tests for column pairs.

    Parameters
    ----------
    lf : pl.LazyFrame or pl.DataFrame
        Input data. LazyFrames are collected internally.
    pairs : list of (str, str), optional
        Specific column pairs to test. When None, all combinations of
        suitable columns (Boolean, String, Categorical, Enum) are tested.
    include_integer : bool, default False
        When True, integer columns are also included in automatic pair
        detection. Only recommended for low-cardinality integer columns.
    min_unique : int, default 2
        When auto-detecting columns, exclude those with fewer unique values.
        A constant column makes the test undefined.
    max_unique : int or None, default 1000
        When auto-detecting pairs (pairs=None), columns with more unique
        values than this are excluded. Also guards each pair individually
        to prevent memory exhaustion on high-cardinality data. Pass None
        to disable the check entirely.

    Returns
    -------
    pl.DataFrame
        Single column "pairwise_chi_squared" containing structs with fields:
        - col_a: String - first column name
        - col_b: String - second column name
        - chi2_stat: f64 - chi-squared test statistic
        - p_value: f64 - p-value for the independence test
        - cramers_v: f64 - Cramer's V effect size (0 = no association, 1 = perfect)
    """
    if isinstance(lf, pl.LazyFrame):
        df = lf.collect()
    else:
        df = lf

    if pairs is None:
        cols = _get_suitable_columns(
            df,
            include_integer=include_integer,
            min_unique=min_unique,
            max_unique=max_unique,
        )
        pairs = list(combinations(cols, 2))

    _fields_verified = False
    records: list[dict] = []

    for col_a, col_b in pairs:
        pair_df = df.select([col_a, col_b]).drop_nulls()
        n = len(pair_df)

        # Guard: skip degenerate high-cardinality pairs even when passed explicitly.
        if max_unique is not None:
            if pair_df[col_a].n_unique() > max_unique or pair_df[col_b].n_unique() > max_unique:
                records.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "chi2_stat": float("nan"),
                    "p_value": float("nan"),
                    "cramers_v": float("nan"),
                })
                continue

        if n == 0:
            records.append({
                "col_a": col_a,
                "col_b": col_b,
                "chi2_stat": float("nan"),
                "p_value": float("nan"),
                "cramers_v": float("nan"),
            })
            continue

        try:
            result_df = pair_df.select(
                pds.chi2(col_a, col_b).alias("_r")
            ).unnest("_r")

            if not _fields_verified:
                actual = result_df.columns
                if "statistic" not in actual or "pvalue" not in actual:
                    raise RuntimeError(
                        f"polars-ds chi2 returned unexpected struct fields: {actual}. "
                        "Update field name references in chi_squared/__init__.py "
                        "(look for 'statistic' and 'pvalue')."
                    )
                _fields_verified = True

            chi2_stat: float = result_df["statistic"][0]
            p_value: float = result_df["pvalue"][0]

        except RuntimeError:
            raise
        except Exception:
            records.append({
                "col_a": col_a,
                "col_b": col_b,
                "chi2_stat": float("nan"),
                "p_value": float("nan"),
                "cramers_v": float("nan"),
            })
            continue

        n_unique_a = pair_df[col_a].n_unique()
        n_unique_b = pair_df[col_b].n_unique()
        min_dim = min(n_unique_a - 1, n_unique_b - 1)

        if min_dim <= 0 or n == 0:
            cramers_v = float("nan")
        else:
            cramers_v = math.sqrt(chi2_stat / (n * min_dim))

        records.append({
            "col_a": col_a,
            "col_b": col_b,
            "chi2_stat": chi2_stat,
            "p_value": p_value,
            "cramers_v": cramers_v,
        })

    _RESULT_SCHEMA = {
        "col_a": pl.String,
        "col_b": pl.String,
        "chi2_stat": pl.Float64,
        "p_value": pl.Float64,
        "cramers_v": pl.Float64,
    }

    if not records:
        return pl.DataFrame({
            "pairwise_chi_squared": pl.Series(
                [],
                dtype=pl.Struct(_RESULT_SCHEMA),
            )
        })

    inner_df = pl.DataFrame(records, schema=_RESULT_SCHEMA)
    return pl.DataFrame({
        "pairwise_chi_squared": inner_df.to_struct("pairwise_chi_squared")
    })
