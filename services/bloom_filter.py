import math

import polars as pl

import analytics

class BloomFilter:
    """
    Bloom filter using the murmur3 hash algorithm and polars expression plug ins
    in Rust for optimisation
    """
    def __init__(self, expected_element_count: int, false_positive_rate: float = 0.01):
        self.expected_element_count = expected_element_count
        self.false_positive_rate = false_positive_rate
        self.bit_array = None
        self.num_elements_added = 0

        # Calculate optimal bit array size (m)
        # m = -(n * ln(p)) / (ln(2)^2)
        self.bit_array_size = self._calculate_bit_array_size(
            expected_element_count, false_positive_rate)

        # Calculate optimal number of hash functions (k)
        # k = (m/n) * ln(2)
        self.num_hash_functions = self._calculate_num_hash_functions(
            self.bit_array_size, expected_element_count)

    @staticmethod
    def _calculate_bit_array_size(n: int, p: float) -> int:
        """Calculate optimal bit array size in *bytes*."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return (int(math.ceil(m)) + 7 ) // 8 * 8

    @staticmethod
    def _calculate_num_hash_functions(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _normalize_input(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(data, pl.LazyFrame):
            cols = data.columns
            num_cols = len(cols)
            if num_cols == 0:
                raise ValueError("DataFrame must have at least one column")
            elif num_cols == 1:
                return data.filter(~pl.all_horizontal(pl.all().is_null()))\
                           .select(pl.col(cols[0]).cast(pl.Utf8).str.to_lowercase().alias("items"))
            else:
                return data.filter(~pl.all_horizontal(pl.all().is_null()))\
                           .select(pl.concat_str([pl.col(c).cast(pl.Utf8).str.to_lowercase() for c in cols], separator="\x00").alias("items"))
        else:
            raise TypeError(f"Expected LazyFrame, got {type(data)}")

    def add(self, data: pl.LazyFrame) -> None:
        df = self._normalize_input(data)
        result = df.select(pl.col("items").analytics.bloom_filter(k=self.num_hash_functions,
                                                                  m=self.bit_array_size,
                                                                  existing_filter=list(self.bit_array) if self.bit_array else []))\
                   .collect()
        self.bit_array = result["items"][0]
        self.num_elements_added += df.select(pl.len()).collect()["len"][0]

    def membership(self, data: pl.LazyFrame) -> pl.DataFrame:
        """Check membership for each item, returning a boolean Series."""
        df = self._normalize_input(data)
        result = df.select(pl.col("items").analytics.membership(bit_array_bytes=list(self.bit_array) if self.bit_array else [],
                                                                  k=self.num_hash_functions,
                                                                  m=self.bit_array_size,
                                                                  ))\
                   .collect()
        return result

    def membership_ratio(self, data: pl.LazyFrame) -> float:
        """Return the ratio of items found in the bloom filter (0.0 to 1.0)."""
        df = self._normalize_input(data)
        result = df.select(pl.col("items").analytics.membership_ratio(
                                bit_array_bytes=list(self.bit_array) if self.bit_array else [],
                                k=self.num_hash_functions,
                                m=self.bit_array_size,
                          ))\
                   .collect()
        return result["items"][0]

    def membership_ratio_sample(self, data: pl.LazyFrame, sample_frac: float = 0.05) -> float:
        """Return the membership ratio for a random sample of items (0.0 to 1.0).

        Used for early-exit optimization: if a sample shows low membership,
        skip checking the full dataset.

        Args:
            data: LazyFrame containing items to check
            sample_frac: Fraction of items to sample (0.0 to 1.0), default 0.05 (5%)

        Returns:
            Ratio of sampled items found in the bloom filter
        """
        df = self._normalize_input(data)
        result = df.select(pl.col("items").analytics.membership_ratio_sample(
                                bit_array_bytes=list(self.bit_array) if self.bit_array else [],
                                k=self.num_hash_functions,
                                m=self.bit_array_size,
                                sample_frac=sample_frac,
                          ))\
                   .collect()
        return result["items"][0]

    def __len__(self) -> int:
        """Return the number of elements added to the filter."""
        return self.num_elements_added
