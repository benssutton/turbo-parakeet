import polars as pl

from services.MinHashLSHFilter import MinHashLSHFilter
from services.comparitors.DeterministicSimilarityFilter import DeterministicSimilarityFilter
from services.comparitors.MinHashLSHFilter_datasketch import MinHashLSHFilter_datasketch


def test_deterministic_similarity_filter():
    data1 = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'C': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }
    df1 = pl.DataFrame(data1).lazy()

    data2 = {
        'A': [1, 2, 3, 4, 5],
        'B': [4, 5, 6, 7, 8],
        'C': [10, 11, 12, 13, 14],
    }
    df2 = pl.DataFrame(data2).lazy()

    filter = DeterministicSimilarityFilter(jaccard_threshold=0.6, overlap_threshold=0.9)
    filter.add({"df1": df1, "df2": df2})
    results_df = filter.get_similar_pairs()
    assert results_df.shape == (5, 8)
    assert results_df.filter(pl.col("df_a") == "df1", pl.col("df_b") == "df2").shape == (4, 8)


def test_minhash_datasketch_similarity_filter():
    data1 = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'C': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }
    df1 = pl.DataFrame(data1).lazy()

    data2 = {
        'A': [1, 2, 3, 4, 5],
        'B': [4, 5, 6, 7, 8],
        'C': [10, 11, 12, 13, 14],
    }
    df2 = pl.DataFrame(data2).lazy()

    filter = MinHashLSHFilter_datasketch(jaccard_threshold=0.6, overlap_threshold=0.9)
    filter.add({"df1": df1, "df2": df2})
    results_df = filter.get_similar_pairs()
    assert results_df.shape == (5, 8)
    assert results_df.filter(pl.col("df_a") == "df1", pl.col("df_b") == "df2").shape == (4, 8)


def test_minhash_similarity_filter():
    data1 = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'C': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }
    df1 = pl.DataFrame(data1).lazy()

    data2 = {
        'A': [1, 2, 3, 4, 5],
        'B': [4, 5, 6, 7, 8],
        'C': [10, 11, 12, 13, 14],
    }
    df2 = pl.DataFrame(data2).lazy()

    filter = MinHashLSHFilter(jaccard_threshold=0.6, overlap_threshold=0.9)
    filter.add({"df1": df1, "df2": df2})
    results_df = filter.get_similar_pairs()
    assert results_df.shape == (5, 8)
    assert results_df.filter(pl.col("df_a") == "df1", pl.col("df_b") == "df2").shape == (4, 8)
