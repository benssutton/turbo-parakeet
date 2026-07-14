from typing import Optional, Set, Dict, Tuple, List
import time

import polars as pl
from functools import lru_cache

from analytics import minhash, lsh_candidates

class MinHashLSHFilter:
    """
    Class for identifying column pairs among multiple dataframes
    with Jaccard Index or Overlap Coefficient above given thresholds
    using MinHash + Locality Sensitive Hashing implemented as extensions to Polars

    Args:
        jaccard_threshold = the Jaccard Index value above which to return a similarity result
        overlap_threshold = the Overlap Coefficient value above which to return a similarity result
        num_perm = the number of hashes to create per item
    """
    def __init__(
        self,
        jaccard_threshold: float = 0.6,
        overlap_threshold: float = 0.95,
        num_perm: int = 128,
    ):
        self.jaccard_threshold = jaccard_threshold
        self.overlap_threshold = overlap_threshold
        self.num_perm = num_perm
        # for the overlap co-efficient, the LSH threshold needs to be a minimum of
        # half the value of the desired threshold 
        self.lsh_threshold = min(jaccard_threshold * 0.9, overlap_threshold * 0.45) 
        self.lfs = {}
        self.min_hashes: Optional[pl.DataFrame] = None
        self.candidates: List[Tuple[str, str]] = []
        self.minhash_time: float = 0.0
        self.lsh_time: float = 0.0
        self.verification_time: float = 0.0
        self.b, self.r = self._optimal_lsh_params(self.lsh_threshold, self.num_perm)

    @staticmethod
    def _optimal_lsh_params(threshold: float, num_perm: int) -> Tuple[int, int]:
        min_error = float("inf")
        best_b, best_r = 1, num_perm

        for b in range(1, num_perm + 1):
            max_r = num_perm // b
            for r in range(1, max_r + 1):
                # Calculate the probability of becoming a candidate at threshold
                # P(candidate | similarity=s) = 1 - (1 - s^r)^b

                # False positive rate: probability of becoming candidate when s < threshold
                # Approximated by the probability at s = threshold (simplified)
                fp = 1 - (1 - threshold ** r) ** b

                # False negative rate: probability of NOT becoming candidate when s >= threshold
                # This is (1 - threshold^r)^b at the threshold
                fn = (1 - threshold ** r) ** b

                # We want fp to be low (don't match dissimilar items)
                # and fn to be low (don't miss similar items)
                # The optimal point is where the S-curve crosses 0.5 at threshold
                # Error = distance from ideal: at threshold, P should be ~0.5
                error = abs(fp - 0.9) + abs(fn - 0.1)

                if error < min_error:
                    min_error = error
                    best_b, best_r = b, r

        return best_b, best_r

    def add(self, lfs: Dict[str, pl.LazyFrame]) -> None:
        """
        Add dataframes to the filter
        """
        self.lfs.update(lfs)

    def find_candidate_pairs(self):
        """
        Create a MinHash signature for all columns and
        subsequently use LSH to find candidate matches
        """
        # Phase 1: Create MinHashes for all columns in batch
        start_time = time.perf_counter_ns()
        minhash_dfs = []
        for name, lf in self.lfs.items():
            # Compute MinHash for all columns at once
            minhash_dfs.append(minhash(lf, name, self.num_perm))
        self.min_hashes = pl.concat(minhash_dfs)
        self.minhash_time = (time.perf_counter_ns() - start_time) / 1000 ** 3

        # Phase 2: Build LSH index and find candidate pairs
        start_time = time.perf_counter_ns()
        candidates_df = self.min_hashes.select(
            lsh_candidates(
                names=pl.col("qualified_name"),
                signatures=pl.col("minhash"),
                threshold=self.lsh_threshold,
                num_bands=self.b,
                rows_per_band=self.r,
            ).alias("candidates")
        ).unnest("candidates").collect()
        self.lsh_time = (time.perf_counter_ns() - start_time) / 1000 ** 3
        # Extract candidate pairs as list of tuples
        self.candidates = list(zip(
            candidates_df["col_a"].to_list(),
            candidates_df["col_b"].to_list()
        ))

    def get_similar_pairs(self):
        """
        Search short list of candidate pairs of columns for those 
        with a Jaccard Index or Overlap Coefficient over given thresholds
        """
        if len(self.candidates) == 0:
            self.find_candidate_pairs()
        results = []
        start_time = time.perf_counter_ns()
        for lfa_col_a, lfb_col_b in self.candidates:
            lfa_split = lfa_col_a.split("|")
            lfb_split = lfb_col_b.split("|")
            lfa = lfa_split[0]
            col_a = lfa_split[1]
            lfb = lfb_split[0]
            col_b = lfb_split[1]

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