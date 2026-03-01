from typing import TYPE_CHECKING, Any, Sequence
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

__version__ = "0.1.0"

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
            function_name="pairwise_joint_entropy_v2",
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
