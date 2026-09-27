"""
Column-relationship analytics. Every technique is a subpackage with one class per
implementation, all used the same way:

    from analytics.chi_squared import ChiSquaredRust
    result = ChiSquaredRust(cramers_v_threshold=0.3).add({"sales": df}).result()

Per-column: analytics.gcd
Multi-set:  analytics.membership, analytics.similarity
Ordered:    analytics.chi_squared, analytics.pairwise_entropy,
            analytics.threeway_entropy, analytics.adjusted_rand

The Rust extension wrappers in analytics._plugin are private.
"""

__version__ = "0.1.0"
