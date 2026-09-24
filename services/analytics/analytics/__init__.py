__version__ = "0.1.0"

# Transitional: the plugin wrappers stay importable from the top-level package
# until every technique has moved to its class (removed in Task 11).
from analytics._plugin import (  # noqa: E402,F401
    column_gcd,
    lsh_candidates,
    marginal_entropy,
    membership_ratio,
    minhash,
    pairwise_adjusted_rand,
    pairwise_chi_squared,
    pairwise_joint_entropy,
    threeway_joint_entropy,
)
