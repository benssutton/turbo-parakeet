from typing import List, Set, Dict
from functools import lru_cache
import time

import polars as pl

class DeterministicSimilarityFilter:
    """
    Class for identifying column pairs among multiple dataframes
    with Jaccard Index or Overlap Coefficient above given thresholds
    using brute-force iteration.

    Args:
        jaccard_threshold = the Jaccard Index value above which to return a similarity result
        overlap_threshold = the Overlap Coefficient value above which to return a similarity result
    """
    def __init__(self,
                 jaccard_threshold: float = 0.6,
                 overlap_threshold: float = 0.95,
                 num_perm: int = 0):
        self.jaccard_threshold = jaccard_threshold
        self.overlap_threshold = overlap_threshold
        #num_perm is unused but kept for interface consistency
        self.lfs = {}
        self.minhash_time: float = 0.0
        self.lsh_time: float = 0.0
        self.verification_time: float = 0.0

    def add(self, lfs: Dict[str, pl.LazyFrame]) -> None:
        """
        Add dataframes to the filter
        """
        self.lfs.update(lfs)

    def find_candidate_pairs(self) -> pl.DataFrame:
        """
        Added for interface consistency with probabilistic filter methods.
        Simply returns the same result as get_similar_pairs
        """
        return self.get_similar_pairs()
    
    def get_similar_pairs(self) -> pl.DataFrame:
        """
        Brute force search of columns with a Jaccard Index or 
        Overlap Coefficient over given thresholds
        """
        results = []
        all_cols = []
        for name, lf in self.lfs.items():
            all_cols.extend([(name, col) for col in lf.collect_schema().names()])
        
        exact_computations = 0
        start_time = time.perf_counter_ns()
        for i, col_a_details in enumerate (all_cols):
            lfa = col_a_details[0]
            col_a = col_a_details[1]
            for col_b_details in all_cols[i+1:]:
                
                exact_computations += 1

                lfb = col_b_details[0]
                col_b = col_b_details[1]

                A = self._get_unique_values_polars(lfa, col_a)
                B = self._get_unique_values_polars(lfb, col_b)
                len_A = len(A)
                len_B = len(B)
                len_AnB = len(A.intersection(B))
                len_AuB = len_A + len_B - len_AnB

                JI = len_AnB / len_AuB
                OC = len_AnB / min(len_A, len_B)

                if JI >= self.jaccard_threshold or OC >= self.overlap_threshold:

                    results.append({
                        "df_a": lfa,
                        "df_b": lfb,
                        'col_a': col_a,
                        'col_b': col_b,
                        'jaccard': JI,
                        'overlap': OC,
                        'passes_jaccard': JI >= self.jaccard_threshold,
                        'passes_overlap': OC >= self.overlap_threshold,
                    })
        self.verification_time = (time.perf_counter_ns() - start_time) / 1000 ** 3

        return pl.DataFrame(results)

    @lru_cache(maxsize=4096)
    def _get_unique_values_polars(self, df_name, col_name)-> Set:
        """
        Get unique values for each column in a Polars LazyFrame.
        Cached for performance.
        """
        unique_values = self.lfs[df_name]\
                        .select(pl.col(col_name).drop_nulls().unique())\
                        .collect()\
                        .to_series()\
                        .to_list()
        return set(unique_values)    