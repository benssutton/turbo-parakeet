"""OneShotRecommender: the narrowest value-preserving Arrow type per column of one frame.

Spec: docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md. All state lives
in Rust (src/recommenders/oneshot.rs): `add` collects each column's exact statistics and sizes;
`result` casts every candidate type, verifies it row by row and measures it (Arrow and
Polars layouts, plain and ZSTD) — once; later calls return the same table.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from analytics import _plugin
from analytics.recommend import _input
from analytics.recommend._maps import from_arrow


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

    `datafusion_type` is pyarrow's spelling, `polars_type` Python's `str(dtype)` of
    the equivalent Polars type; sizes are Arrow IPC bodies (Polars: its native layout),
    plain and ZSTD. `candidates` lists every candidate with its rule, the metric
    values it tested, its predicted / projected size and its outcome.

    Exemplars (spec docs/superpowers/specs/2026-10-18-recommender-schema-entropy-exemplars-design.md):
    `exemplar_row` is the first non-null row of the column, `exemplar_value` its text,
    `null_exemplar_row` the first null row; `exemplar_id` / `null_exemplar_id` hold the
    values of the `id_columns` at those rows as a Struct with one field per id column, in
    the given order and with the id column's own type (a null id value is a null field).
    The struct is null when there is no such row, and for ineligible columns. Without
    `id_columns` both columns are Null-typed. Id columns must be in the input (every
    batch, when streaming), unique, and eligible (not Int128 / UInt128 / Object); they
    are profiled like any other column.

    `top_k` / `inner_top_k` (dictionary candidates only, else null): the level's
    values by count, highest first, ties to the value seen first — at most `top_k`
    entries (None: all; 0: off — always null); key them 0, 1, 2, … in that order for the most compressible
    dictionary. The dictionary candidate is measured with keys in that order — every
    value in that order, so pass `top_k=None` to get the full order. Polars
    holds them as lists of key / value structs; `analytics.recommend.to_arrow` gives
    Arrow maps.

    `boolean_pairs`: (true text, false text) pairs a two-valued string column may map
    to Boolean, compared case-insensitively.

    Compression (spec docs/superpowers/specs/2026-10-19-clickhouse-recommendations-design.md
    §4): `zstd_levels` are the ZSTD levels tried (distinct, each in 1…22);
    `zstd_level_tolerance` the relative size slack within which the lowest level wins;
    `zstd_min_saving` the smallest relative saving that makes ZSTD preferable to LZ4;
    `gcd_min_saving` the smallest relative saving that keeps a GCD codec. The three
    fractions lie in [0, 1).

    Skip indexes (spec docs/superpowers/specs/2026-10-20-skip-index-recommendations-design.md
    §4): `skip_granularities` are the GRANULARITY values tried, `set_max_sizes` the
    `set(max_size)` sizes and `bloom_fprs` the `bloom_filter` false-positive rates
    (each non-empty and distinct; fprs in (0, 1)). `min_skip` is the smallest share of
    rows a point query must skip, `min_gain` the smallest gain over a random row order,
    and `max_overflow` the largest share of windows whose distinct values may exceed a
    set's size; the three lie in [0, 1].

    Run-end encoding: never recommended. DataFusion 54 rejects MIN / MAX aggregates over a
    run-end-encoded column ("Min/Max accumulator not implemented"), while every other operator
    probed works and returns identical results, and Arrow IPC files with REE columns read fine; an
    REE recommendation would break common queries. Re-run tests/performance/spike_datafusion_ree.py
    after a DataFusion upgrade; REE is added (spec 2026-10-21) once MIN / MAX work.
    """

    def __init__(
        self,
        *,
        categorical_threshold: int = 10_000,
        top_k: int | None = 256,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
        id_columns: Sequence[str] | None = None,
        zstd_levels: Sequence[int] = (1, 3, 9),
        zstd_level_tolerance: float = 0.01,
        zstd_min_saving: float = 0.10,
        gcd_min_saving: float = 0.01,
        skip_granularities: Sequence[int] = (1, 2, 4, 8),
        set_max_sizes: Sequence[int] = (125, 250),
        bloom_fprs: Sequence[float] = (0.025,),
        min_skip: float = 0.3,
        min_gain: float = 0.1,
        max_overflow: float = 0.1,
    ) -> None:
        self._ids = _input.id_columns(id_columns)
        self._rs = _plugin.oneshot_recommender(
            categorical_threshold=categorical_threshold,
            top_k=_input.top_k(top_k),
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=_input.boolean_pairs(boolean_pairs),
            id_columns=self._ids,
            zstd_levels=list(zstd_levels),
            zstd_level_tolerance=zstd_level_tolerance,
            zstd_min_saving=zstd_min_saving,
            gcd_min_saving=gcd_min_saving,
            skip_granularities=list(skip_granularities),
            set_max_sizes=list(set_max_sizes),
            bloom_fprs=list(bloom_fprs),
            min_skip=min_skip,
            min_gain=min_gain,
            max_overflow=max_overflow,
        )

    def add(self, frame) -> OneShotRecommender:
        """Adds `frame` and collects its statistics; a second call raises ValueError."""
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        frame, ineligible = _input.prepare(frame)
        _input.check_ids(self._ids, ineligible)
        self._rs.add(frame, ineligible)
        return self

    def result(self) -> pl.DataFrame:
        return from_arrow(self._rs.result())
