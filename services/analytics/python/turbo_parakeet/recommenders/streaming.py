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

from collections.abc import Sequence

import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.recommenders import _input
from turbo_parakeet.recommenders._maps import from_arrow


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
    holds them as lists of key / value structs; `turbo_parakeet.recommenders.to_arrow` gives
    Arrow maps.

    Memory per eligible level (a column, or a list's inner values), every dtype:
    ≈ 16 KB of HyperLogLog plus a distinct sample of up to ≈ 65 bytes × k, where
    k = max(categorical_threshold, 1000) — ≈ 0.65 MB at the default 10 000, so
    ≈ 0.65 GB for 1 000 high-cardinality columns; text levels also keep their
    distinct values' text while exact. `entropy_estimator="ams"` adds an AMS sketch per
    column (≈ 4 096 estimators, a few hundred KB, plus a per-row hash and update), and
    each `add` holds one 8-byte key per row per column until its batch is absorbed
    (≈ 8 MB per column per million-row batch): add smaller batches for wide frames.

    `entropy_estimator` (default `"ams"`, spec §8): past the exact phase, `"sample"` (the bottom-k distinct
    sample with the HyperLogLog distinct count: no extra per-row cost, weak on skewed columns whose heavy values fall
    outside the sample) or `"ams"` (an AMS sketch fed every row of every eligible
    column from the first batch: ≈ 4 096 estimators per column, accurate on skew).

    Compression (spec docs/superpowers/specs/2026-10-19-clickhouse-recommendations-design.md
    §4): `zstd_levels` are the ZSTD levels tried (distinct, each in 1…22);
    `zstd_level_tolerance` the relative size slack within which the lowest level wins;
    `zstd_min_saving` the smallest relative saving that makes ZSTD preferable to LZ4;
    `transform_min_saving` the smallest relative saving that keeps a transform codec (GCD, Delta, DoubleDelta) over no transform. The three
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
        reservoir_rows: int = 524_288,
        block_rows: int = 65_536,
        categorical_threshold: int = 10_000,
        top_k: int | None = 256,
        zstd_level: int = 1,
        seed: int = 0,
        boolean_pairs: tuple[tuple[str, str], ...] = (("true", "false"),),
        entropy_estimator: str = "ams",
        id_columns: Sequence[str] | None = None,
        zstd_levels: Sequence[int] = (1, 3, 9),
        zstd_level_tolerance: float = 0.01,
        zstd_min_saving: float = 0.10,
        transform_min_saving: float = 0.01,
        skip_granularities: Sequence[int] = (1, 2, 4, 8),
        set_max_sizes: Sequence[int] = (125, 250),
        bloom_fprs: Sequence[float] = (0.025,),
        min_skip: float = 0.3,
        min_gain: float = 0.1,
        max_overflow: float = 0.1,
    ) -> None:
        self._ids = _input.id_columns(id_columns)
        self._rs = _plugin.streaming_recommender(
            reservoir_rows=reservoir_rows,
            block_rows=block_rows,
            categorical_threshold=categorical_threshold,
            top_k=_input.top_k(top_k),
            zstd_level=zstd_level,
            seed=seed,
            boolean_pairs=_input.boolean_pairs(boolean_pairs),
            entropy_estimator=entropy_estimator,
            id_columns=self._ids,
            zstd_levels=list(zstd_levels),
            zstd_level_tolerance=zstd_level_tolerance,
            zstd_min_saving=zstd_min_saving,
            transform_min_saving=transform_min_saving,
            skip_granularities=list(skip_granularities),
            set_max_sizes=list(set_max_sizes),
            bloom_fprs=list(bloom_fprs),
            min_skip=min_skip,
            min_gain=min_gain,
            max_overflow=max_overflow,
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
        _input.check_ids(self._ids, ineligible)
        self._rs.add(frame, ineligible)
        return self

    def result(self) -> pl.DataFrame:
        return from_arrow(self._rs.result())
