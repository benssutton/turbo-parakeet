"""OneShotRecommender: the narrowest value-preserving Arrow type per column of one frame.

Spec: docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md. All state lives
in Rust (src/recommenders/oneshot.rs): `add` collects each column's exact statistics and sizes;
`result` casts every candidate type, verifies it row by row and measures it (Arrow and
Polars layouts, plain and ZSTD) — once; later calls return the same table.
"""

from __future__ import annotations

import polars as pl

from analytics import _plugin
from analytics.recommend import _input


class OneShotRecommender:
    """Narrowest value-preserving Arrow type per column, from one frame.

    `add(frame)` takes a Polars DataFrame or LazyFrame (collected), or any Arrow
    tabular object (`__arrow_c_stream__`, read whole); a second `add` raises
    ValueError. Int128 / UInt128, Object, Null and nested-Null columns are listed as
    ineligible. `result()` returns one row per column — the streaming recommender's
    columns without `first_row`, `n_sampled_rows` and `n_sampled_blocks` — and may be
    called any number of times. `dtype` names the Arrow type the recommender received,
    so it can differ by input form (e.g. string_view from Polars, large_string from
    pyarrow).

    `rec_arrow_type` is pyarrow's spelling, `rec_polars_type` Python's `str(dtype)` of
    the equivalent Polars type; sizes are Arrow IPC bodies (Polars: its native layout),
    plain and ZSTD. `rec_candidates` lists every candidate with its rule, the metric
    values it tested, its predicted / projected size and its outcome.

    `boolean_pairs`: (true text, false text) pairs a two-valued string column may map
    to Boolean, compared case-insensitively.
    """

    def __init__(
        self,
        *,
        categorical_threshold: int = 10_000,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
    ) -> None:
        self._rs = _plugin.oneshot_recommender(
            categorical_threshold=categorical_threshold,
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=_input.boolean_pairs(boolean_pairs),
        )

    def add(self, frame) -> OneShotRecommender:
        """Adds `frame` and collects its statistics; a second call raises ValueError."""
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        frame, ineligible = _input.prepare(frame)
        self._rs.add(frame, ineligible)
        return self

    def result(self) -> pl.DataFrame:
        return pl.DataFrame(self._rs.result())
