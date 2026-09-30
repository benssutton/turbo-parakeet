"""StreamingRecommender: Recommend's dtype recommendations from batches added over time.

Spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md. All state lives
in Rust (src/streaming.rs): exact running statistics prove each recommendation on every
row; a sample of contiguous row blocks (`reservoir_rows`, in blocks of `block_rows`)
gives ZSTD sizes — those of an IPC file written in `block_rows` batches — and a
cross-check. Not a technique on the uniform contract: add batches, finish at any time.
"""

from __future__ import annotations

import polars as pl

from analytics import _plugin
from analytics._dtypes import holds_wide_integer


class StreamingRecommender:
    """Narrowest value-preserving Arrow type per column, from a stream of batches.

    `add(frame)` takes a Polars DataFrame or any Arrow tabular object
    (`__arrow_c_stream__`); its batches are processed one at a time. Columns may
    appear, disappear (their rows count as null) or start as the Null type. Int128 /
    UInt128 and Object columns are listed as ineligible. `finish()` returns one row
    per column and keeps the state, so adding can continue.
    """

    def __init__(
        self,
        *,
        reservoir_rows: int = 524_288,
        block_rows: int = 65_536,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
    ) -> None:
        self._rs = _plugin.streaming_recommender(
            reservoir_rows=reservoir_rows,
            block_rows=block_rows,
            categorical_threshold=categorical_threshold,
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=tuple(tuple(p) for p in boolean_pairs),
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
        ineligible: list[tuple[str, str]] = []
        if isinstance(frame, pl.DataFrame):
            ineligible = [
                (name, str(dtype))
                for name, dtype in frame.schema.items()
                if holds_wide_integer(dtype) or isinstance(dtype, pl.Object)
            ]
            if len(ineligible) == frame.width:
                # Dropping every column would lose the height; the rows still count.
                frame = pl.DataFrame(height=frame.height)
            else:
                frame = frame.drop([name for name, _ in ineligible])
        self._rs.add(frame, ineligible)
        return self

    def finish(self) -> pl.DataFrame:
        return pl.DataFrame(self._rs.finish())
