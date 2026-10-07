"""StreamingRecommender: Recommend's dtype recommendations from batches added over time.

Spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md. All state lives
in Rust (src/recommenders/streaming/): exact running statistics prove each recommendation on every
row; a sample of contiguous row blocks (`reservoir_rows`, in blocks of `block_rows`)
gives ZSTD sizes — those of an IPC file written in `block_rows` batches — and a
cross-check. Not a technique on the uniform contract: add batches, result at any time.

Distinct counts (spec docs/superpowers/specs/2026-10-01-recommender-parity-design.md):
every eligible level, every dtype, keeps a HyperLogLog and a bottom-k distinct sample
(k = max(categorical_threshold, 1000)); exact while the sample holds every value.
"""

from __future__ import annotations

import polars as pl

from analytics import _plugin
from analytics.recommend import _input


class StreamingRecommender:
    """Narrowest value-preserving Arrow type per column, from a stream of batches.

    `add(frame)` takes a Polars DataFrame or any Arrow tabular object
    (`__arrow_c_stream__`); its batches are processed one at a time. Columns may
    appear, disappear (their rows count as null) or start as the Null type. Int128 /
    UInt128, Object and nested-Null (List(Null), a Struct with a Null field, ...)
    columns are listed as ineligible. `result()` returns one row
    per column and keeps the state, so adding can continue. `dtype` names the Arrow type
    the recommender received, so it can differ by input form (e.g. string_view from
    Polars, large_string from pyarrow).

    `top_k` / `inner_top_k` (dictionary candidates only, else null): the level's
    values by count, highest first, ties to the value seen first — at most `top_k`
    entries (None: all; 0: off — always null); key them 0, 1, 2, … in that order for the most compressible
    dictionary. The dictionary candidate is measured with keys in that order. Polars
    holds them as lists of key / value structs; `analytics.recommend.to_arrow` gives
    Arrow maps.

    Memory per eligible level (a column, or a list's inner values), every dtype:
    ≈ 16 KB of HyperLogLog plus a distinct sample of up to ≈ 50 bytes × k, where
    k = max(categorical_threshold, 1000) — ≈ 0.5 MB at the default 10 000, so
    ≈ 0.5 GB for 1 000 high-cardinality columns.
    """

    def __init__(
        self,
        *,
        reservoir_rows: int = 524_288,
        block_rows: int = 65_536,
        categorical_threshold: int = 10_000,
        top_k: int | None = 256,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
    ) -> None:
        self._rs = _plugin.streaming_recommender(
            reservoir_rows=reservoir_rows,
            block_rows=block_rows,
            categorical_threshold=categorical_threshold,
            top_k=_input.top_k(top_k),
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=_input.boolean_pairs(boolean_pairs),
        )

    def add(self, frame) -> StreamingRecommender:
        """Adds every batch of `frame`, in order.

        Each Arrow batch is atomic (on error the state is as before that batch), but
        one `add` over a multi-batch stream is not: if a later batch or the producer
        fails, the earlier batches and the ineligible marks stay applied.
        """
        if isinstance(frame, pl.LazyFrame):
            raise TypeError(
                "a LazyFrame is not supported: collect it, or add its batches"
            )
        frame, ineligible = _input.prepare(frame)
        self._rs.add(frame, ineligible)
        return self

    def result(self) -> pl.DataFrame:
        return pl.DataFrame(self._rs.result())
