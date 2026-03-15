from typing import TYPE_CHECKING, Any, Sequence
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

__version__ = "0.1.0"

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

    def membership_ratio(self,
                        bit_array_bytes: list[int],
                        k: int,
                        m: int) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="membership_ratio",
            args=self._expr,
            kwargs={
                "bit_array_bytes": bit_array_bytes,
                "k": k,
                "m": m
            },
            is_elementwise=True,
        )

    def membership_ratio_sample(self,
                        bit_array_bytes: list[int],
                        k: int,
                        m: int,
                        sample_frac: float) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="membership_ratio_sample",
            args=self._expr,
            kwargs={
                "bit_array_bytes": bit_array_bytes,
                "k": k,
                "m": m,
                "sample_frac": sample_frac,
            },
            is_elementwise=True,
        )

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
