from typing import Set, Dict, Tuple, List
import time

import polars as pl
from datasketch import MinHash, MinHashLSH
from functools import lru_cache

class MinHashLSHFilter_datasketch:
    """
    Class for identifying column pairs among multiple dataframes
    with Jaccard Index or Overlap Coefficient above given thresholds
    using MinHash + Locality Sensitive Hashing from the datasketch library

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
        self.lsh_threshold = min(jaccard_threshold, overlap_threshold) * 0.45
        self.lfs = {}
        self.min_hashes: List[Tuple[str, str, MinHash]] = []
        self.candidates: Set[Tuple[str, str]] = set()
        self.minhash_time: float = 0.0
        self.lsh_time: float = 0.0
        self.verification_time: float = 0.0

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

        # Create MinHashes for all columns
        start_time = time.perf_counter_ns()
        for name, lf in self.lfs.items():
            for col in lf.collect_schema().names():
                min_hash = MinHash(num_perm=self.num_perm)
                values = lf.select(pl.col(col).drop_nulls().unique().cast(pl.String)).collect().to_series().to_list()
                values = [v.encode('utf8') for v in values]
                min_hash.update_batch(values)
                self.min_hashes.append((name, col, min_hash))
        self.minhash_time = (time.perf_counter_ns() - start_time) / 1000 ** 3

        # Phase 2: Build LSH index and find candidate pairs
        start = time.perf_counter()
        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm, weights=(0.9, 0.1))
        self.lsh_b = lsh.b
        self.lsh_r = lsh.r

        # Insert all minhashes
        for name, col, min_hash in self.min_hashes:
            lsh.insert(name + "|" + col, min_hash)

        # Find candidate pairs by querying each column
        self.candidates = set()
        for name, col, min_hash in self.min_hashes:
            neighbors = lsh.query(min_hash)
            for neighbor in neighbors:
                if neighbor != name + "|" + col:
                    pair = tuple(sorted([name + "|" + col, neighbor]))
                    self.candidates.add(pair)
        self.lsh_time = time.perf_counter() - start


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