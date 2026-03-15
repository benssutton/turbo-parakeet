import math
from typing import NamedTuple

import polars as pl

import analytics

class BloomFilter:
    """
    Bloom filter backed by a Rust/Polars plugin for high-performance membership testing.
    Operates on a single column; the caller is responsible for selecting the column to test.
    """
    def __init__(self, expected_element_count: int, false_positive_rate: float = 0.01):
        self.expected_element_count = expected_element_count
        self.false_positive_rate = false_positive_rate
        self.bit_array = None
        self.num_elements_added = 0

        # m = -(n * ln(p)) / (ln(2)^2)
        self.bit_array_size = self._calculate_bit_array_size(
            expected_element_count, false_positive_rate)

        # k = (m/n) * ln(2)
        self.num_hash_functions = self._calculate_num_hash_functions(
            self.bit_array_size, expected_element_count)

    @staticmethod
    def _calculate_bit_array_size(n: int, p: float) -> int:
        """Calculate optimal bit array size in *bytes*."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return (int(math.ceil(m)) + 7) // 8 * 8

    @staticmethod
    def _calculate_num_hash_functions(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def add(self, df: pl.LazyFrame | pl.DataFrame) -> None:
        """Add all items in the first column to the bloom filter."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        col = df.columns[0]
        result = df.select(
            pl.col(col).analytics.bloom_filter(
                k=self.num_hash_functions,
                m=self.bit_array_size,
                existing_filter=list(self.bit_array) if self.bit_array else [],
            )
        )
        self.bit_array = result.to_series()[0]
        self.num_elements_added += len(df)

    def membership(self, data: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        """Check membership for each item in the first column. Returns a Boolean Series."""
        df = self._collect(data)
        col = df.columns[0]
        return df.select(
            pl.col(col).analytics.membership(
                bit_array_bytes=list(self.bit_array) if self.bit_array else [],
                k=self.num_hash_functions,
                m=self.bit_array_size,
            )
        )

    def membership_ratio(self, df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Return the fraction of items in the first column found in the bloom filter.
        """
        result = analytics.membership_ratio(
            df,
            bit_array_bytes=list(self.bit_array) if self.bit_array else [],
            k=self.num_hash_functions,
            m=self.bit_array_size,
        )
        return result.unnest("membership_ratio")

    def membership_ratio_sample(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        sample_frac: float = 0.05,
    ) -> pl.DataFrame:
        """
        Return the membership ratio for a random sample of items.

        Used for early-exit optimisation: if a sample shows low membership,
        skip checking the full dataset.

        Parameters
        ----------
        data : LazyFrame or DataFrame
        sample_frac : float
            Fraction of items to sample (default 0.05 = 5%).
        """
        result = analytics.membership_ratio_sample(
            df,
            bit_array_bytes=list(self.bit_array) if self.bit_array else [],
            k=self.num_hash_functions,
            m=self.bit_array_size,
            sample_frac=sample_frac,
        )
        return result.unnest("membership_ratio_sample")

    def __len__(self) -> int:
        """Return the number of elements added to the filter."""
        return self.num_elements_added
