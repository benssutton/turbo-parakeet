import math

import polars as pl

import analytics

class BloomFilter:
    """
    Class to create and check membership against a Bloom filter

    A note on Bloom filters...

    Bloom filters are performant mechanisms for checking membership between two sets of data.
    They achieve this by hashing values and permitting hash collisions with a given probability.

    This means that Bloom filters may return false positives, but will never return false negatives,
    i.e. if the membership function of a Bloom filter says an item is not present, then it is
    deterministically not present.  However, if a Bloom filter is created with a 1% false positive
    rate, if the Bloom filter membership function says and item is present, then it is 99% likely
    to be present.

    It is this ability to reason mathematically about the probability of hash collisions that
    makes Bloom filters useful whilc still being probabilistic and performant.

    See the following for more details:
    https://www.geeksforgeeks.org/python/bloom-filters-introduction-and-python-implementation/
    https://en.wikipedia.org/wiki/Bloom_filter

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
        """
        Calculate optimal bit array size m in *bits*, where
        m = -(n * ln(p)) / (ln(2)^2), rounded up to a multiple of 8 so the
        backing byte array is exactly m/8 bytes. The Rust plugin expects m in
        bits and a byte array of ceil(m/8) bytes.
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return (int(math.ceil(m)) + 7) // 8 * 8

    @staticmethod
    def _calculate_num_hash_functions(m: int, n: int) -> int:
        """
        Calculate optimal number of hash functions, k, where 
        k = (m/n) * lg(2) where m is the number of bits
        """
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
                existing_filter=self.bit_array if self.bit_array is not None else [],
            )
        )
        # Convert bytes → list[int] once here so membership calls pass it directly
        # without a per-call list() copy (~60KB at fp=1%, n=50K).
        self.bit_array = list(result.to_series()[0])
        self.num_elements_added += len(df)

    def membership(self, data: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Check for membership for each item in the first column. Returns a Series of
        True/False indicating membership.
        """
        df = data.collect() if isinstance(data, pl.LazyFrame) else data
        col = df.columns[0]
        return df.select(
            pl.col(col).analytics.membership(
                bit_array_bytes=self.bit_array if self.bit_array is not None else [],
                k=self.num_hash_functions,
                m=self.bit_array_size,
            )
        )

    def membership_ratio(self, df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Return the fraction of items found in the bloom filter for all columns
        in the given LazyFrame.
        """
        result = analytics.membership_ratio(
            df,
            bit_array_bytes=self.bit_array if self.bit_array is not None else [],
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
        Return the membership ratio for a random sample of items for all
        columns in the LazyFrame.
        """
        result = analytics.membership_ratio_sample(
            df,
            bit_array_bytes=self.bit_array if self.bit_array is not None else [],
            k=self.num_hash_functions,
            m=self.bit_array_size,
            sample_frac=sample_frac,
        )
        return result.unnest("membership_ratio_sample")

    def __len__(self) -> int:
        """
        Return the number of elements added to the filter.
        """
        return self.num_elements_added
