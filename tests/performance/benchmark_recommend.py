"""
Recommender speed: OneShotRecommender (add, result) and StreamingRecommender (add
throughput in rows/s, result time) on the same data. Standalone — the shared harness
assumes IMPLEMENTATIONS.
Also, per dataset: one-shot at top_k=0 (the cost of building the top_k maps; the ranking and frequency-ordered dictionary run regardless), the Polars top-k oracle's time
(the deterministic baseline), and per dictionary column the Arrow ZSTD size of the
frequency-ordered dictionary against pyarrow's first-seen dictionary_encode
(→ results/recommend_dictionary_order.parquet).

Run: /c/Users/Alexander/miniconda3/envs/p312/python.exe tests/performance/benchmark_recommend.py

Results → tests/performance/results/recommend.parquet (git-ignored).
`peak_mb` is the process's peak memory so far (dataset included), where the OS reports it.
"""

import statistics
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow as pa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/: datagen

from datagen import describe_narrow, describe_wide  # noqa: E402

from analytics.describe import _sizes  # noqa: E402

from analytics.recommend import OneShotRecommender, StreamingRecommender  # noqa: E402

DATASETS = {
    "large_dataset": lambda: pl.read_ipc(HERE.parent / "data" / "large_dataset.arrow"),
    "narrow 1M x 4": lambda: describe_narrow(1_000_000),
    "wide 50K x 100": lambda: describe_wide(50_000, 100),
    "zipf 1M x 4": lambda: zipf_frame(),
}
BATCH_ROWS = (10_000, 100_000)
RUNS = 3


def peak_mb() -> float | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # bytes / KiB
    except ImportError:
        try:
            import psutil
        except ImportError:
            return None
        return getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20


def stream(frame: pl.DataFrame, batch_rows: int) -> tuple[float, float]:
    rec = StreamingRecommender()
    t0 = time.perf_counter()
    for off in range(0, frame.height, batch_rows):
        rec.add(frame.slice(off, batch_rows))
    t1 = time.perf_counter()
    rec.result()
    return t1 - t0, time.perf_counter() - t1


def one_shot_total(frame: pl.DataFrame, **params) -> float:
    t0 = time.perf_counter()
    OneShotRecommender(**params).add(frame).result()
    return time.perf_counter() - t0


def polars_top_k_s(frame: pl.DataFrame, k: int = 256) -> float:
    """The Polars oracle over every String / Categorical / Enum column."""
    t0 = time.perf_counter()
    for name, dtype in frame.schema.items():
        if isinstance(dtype, (pl.String, pl.Categorical, pl.Enum)):
            (
                frame.select(pl.col(name).cast(pl.String).alias("v"))
                .with_row_index("row")
                .drop_nulls("v")
                .group_by("v")
                .agg(n=pl.len(), first=pl.col("row").min())
                .sort(["n", "first"], descending=[True, False])
                .head(k)
            )
    return time.perf_counter() - t0


def zipf_frame(n: int = 1_000_000, seed: int = 0) -> pl.DataFrame:
    """Skewed low-cardinality text columns: dictionary candidates."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cols = {}
    for distinct in (50, 500, 5_000):
        draws = np.minimum(rng.zipf(1.3, n), distinct) - 1
        # shuffle the id -> label map so first-seen order differs from frequency order
        labels = np.array([f"value_{i:05d}" for i in rng.permutation(distinct)])
        cols[f"zipf_{distinct}"] = pl.Series(labels[draws])
    cols["zipf_cat"] = cols["zipf_500"].cast(pl.Categorical)
    return pl.DataFrame(cols)


def dictionary_order(name: str, frame: pl.DataFrame) -> list[dict]:
    """Arrow ZSTD bytes of each chosen dictionary: the recommender's (frequency order)
    against pyarrow's dictionary_encode of the same column (first-seen order)."""
    out = []
    result = OneShotRecommender().add(frame).result()
    chosen = result.filter(pl.col("rec_arrow_type").str.starts_with("dictionary"))
    for r in chosen.iter_rows(named=True):
        first_seen = (
            frame[r["column"]]
            .cast(pl.String)
            .to_arrow()
            .cast(pa.string())
            .dictionary_encode()
        )
        index_type = (
            r["rec_arrow_type"].split("<values=string, indices=")[1].split(",")[0]
        )
        first_seen = pa.DictionaryArray.from_arrays(
            first_seen.indices.cast(getattr(pa, index_type)()), first_seen.dictionary
        )
        first_seen_zstd = _sizes.ipc_body_bytes(first_seen, 1)
        out.append(
            {
                "dataset": name,
                "column": r["column"],
                "n_unique": r["n_unique"],
                "frequency_order_zstd": r["rec_arrow_size_zstd_bytes"],
                "first_seen_zstd": first_seen_zstd,
                "gain": 1 - r["rec_arrow_size_zstd_bytes"] / first_seen_zstd,
            }
        )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    orders = []
    for name, make in DATASETS.items():
        frame = make()
        one_shot_add, one_shot_result = [], []
        for _ in range(RUNS):
            rec = OneShotRecommender()
            t0 = time.perf_counter()
            rec.add(frame)
            t1 = time.perf_counter()
            rec.result()
            one_shot_add.append(t1 - t0)
            one_shot_result.append(time.perf_counter() - t1)
        no_top_k = statistics.median(
            one_shot_total(frame, top_k=0) for _ in range(RUNS)
        )
        polars_s = statistics.median(polars_top_k_s(frame) for _ in range(RUNS))
        orders.extend(dictionary_order(name, frame))
        for batch_rows in BATCH_ROWS:
            runs = [stream(frame, batch_rows) for _ in range(RUNS)]
            add_s = statistics.median(r[0] for r in runs)
            rows.append(
                {
                    "dataset": name,
                    "rows": frame.height,
                    "cols": frame.width,
                    "batch_rows": batch_rows,
                    "add_s": add_s,
                    "add_rows_per_s": frame.height / add_s,
                    "result_s": statistics.median(r[1] for r in runs),
                    "one_shot_add_s": statistics.median(one_shot_add),
                    "one_shot_result_s": statistics.median(one_shot_result),
                    "one_shot_no_maps_s": no_top_k,
                    "polars_top_k_s": polars_s,
                    "peak_mb": peak_mb(),
                }
            )
            print(rows[-1])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "recommend.parquet")
    if orders:
        pl.DataFrame(orders).write_parquet(out / "recommend_dictionary_order.parquet")
        with pl.Config(tbl_rows=-1):
            print(pl.DataFrame(orders).sort("gain"))


if __name__ == "__main__":
    main()
