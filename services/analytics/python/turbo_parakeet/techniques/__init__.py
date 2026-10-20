"""
Techniques: metrics only, every one used the same way:

    from turbo_parakeet.techniques.chi_squared import ChiSquaredRust
    result = ChiSquaredRust().add({"sales": df}).result()

Per-column:            gcd, describe, top_k
Per-column, ordered:   runs_test, minmax_index, set_index
Multi-set:             membership, similarity
Ordered (row-aligned): chi_squared, pairwise_entropy, threeway_entropy, adjusted_rand
"""
