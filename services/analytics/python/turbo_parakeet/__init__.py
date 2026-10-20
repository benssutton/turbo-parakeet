"""
turbo_parakeet: storage recommendations for ClickHouse, DataFusion and Polars, from
column patterns and relationships, with a Rust core.

    turbo_parakeet.techniques     metrics (one subpackage per technique)
    turbo_parakeet.recommenders   OneShotRecommender, StreamingRecommender, to_arrow, to_chbench

The compiled extension turbo_parakeet._core and its wrappers in turbo_parakeet._plugin
are private.
"""

__version__ = "0.1.0"
