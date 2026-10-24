"""
StreamingRecommender accuracy tests. Oracles: OneShotRecommender on the batches
concatenated diagonally (parity, spec §6); the pyarrow IPC oracle (_sizes) for ZSTD
sizes; a complete sample for the sampled ZSTD estimate; hand-worked known answers.
Accuracy only — nothing here is timed.
"""

import struct

import polars as pl
import pyarrow as pa
import pytest

from turbo_parakeet.techniques.describe import _sizes
from turbo_parakeet.techniques.gcd import GcdRust
from turbo_parakeet.recommenders import OneShotRecommender, StreamingRecommender
from datagen import describe_mixed, large_dataset, large_dataset_mixed, stringified

CANDIDATE = pl.Struct(
    {
        "arrow_type": pl.String,
        "rule": pl.String,
        "evidence": pl.String,
        "predicted_bytes": pl.UInt64,
        "projected_population_bytes": pl.Float64,
        "outcome": pl.String,
        "reason": pl.String,
    }
)
VALUE_BLOCK = {
    "n_unique": pl.UInt64,
    "unique": pl.Boolean,
    "est_cardinality": pl.Float64,
    "est_low": pl.Float64,
    "est_high": pl.Float64,
    "est_method": pl.String,
    "estimates_agree": pl.Boolean,
    "class": pl.String,
    "min": pl.String,
    "max": pl.String,
    "min_len": pl.UInt64,
    "max_len": pl.UInt64,
    "gcd": pl.Decimal(38, 0),
    "sum_len": pl.UInt64,
    "sum_len_unique": pl.UInt64,
    "n_nan": pl.UInt64,
    "n_inf": pl.UInt64,
    "n_fractional": pl.UInt64,
    "max_frac_digits": pl.UInt32,
    "n_f32_inexact": pl.UInt64,
    "n_numeric": pl.UInt64,
    "n_numeric_int": pl.UInt64,
    "n_leading_zero": pl.UInt64,
    "numeric_int_min": pl.Decimal(38, 0),
    "numeric_int_max": pl.Decimal(38, 0),
    "numeric_max_int_digits": pl.UInt32,
    "numeric_max_frac_digits": pl.UInt32,
    "numeric_min_frac_digits": pl.UInt32,
    "numeric_max_sig_digits": pl.UInt32,
    "n_iso_date": pl.UInt64,
    "n_iso_time": pl.UInt64,
    "n_iso_datetime": pl.UInt64,
    "n_iso_datetime_tz": pl.UInt64,
    "iso_max_frac_digits": pl.UInt32,
    "iso_max_sig_frac_digits": pl.UInt32,
    "iso_n_offsets": pl.UInt64,
    "iso_n_midnight": pl.UInt64,
}
SCHEMA = {
    "column": pl.String,
    "status": pl.String,
    "dtype": pl.String,
    "first_row": pl.UInt64,
    "n_rows": pl.UInt64,
    "n_null": pl.UInt64,
    **VALUE_BLOCK,
    "n_midnight": pl.UInt64,
    "entropy": pl.Float64,
    "entropy_low": pl.Float64,
    "entropy_high": pl.Float64,
    "entropy_method": pl.String,
    "exemplar_row": pl.UInt64,
    "exemplar_id": pl.Null,
    "exemplar_value": pl.String,
    "null_exemplar_row": pl.UInt64,
    "null_exemplar_id": pl.Null,
    "size_bytes": pl.UInt64,
    "size_zstd_bytes": pl.UInt64,
    "size_polars_bytes": pl.UInt64,
    "size_polars_zstd_bytes": pl.UInt64,
    "inner_n_values": pl.UInt64,
    "inner_n_null": pl.UInt64,
    **{f"inner_{k}": v for k, v in VALUE_BLOCK.items()},
    "top_k": pl.List(pl.Struct({"key": pl.String, "value": pl.UInt64})),
    "inner_top_k": pl.List(pl.Struct({"key": pl.String, "value": pl.UInt64})),
    "nullable": pl.Boolean,
    "lossy_formatting": pl.Boolean,
    "datafusion_type": pl.String,
    "datafusion_size_bytes": pl.UInt64,
    "datafusion_size_zstd_bytes": pl.UInt64,
    "datafusion_size_lz4_bytes": pl.UInt64,
    "datafusion_ipc_compression": pl.String,
    "datafusion_ipc_zstd_level": pl.Int32,
    "datafusion_ipc_size_bytes": pl.UInt64,
    "polars_type": pl.String,
    "polars_size_bytes": pl.UInt64,
    "polars_size_zstd_bytes": pl.UInt64,
    "polars_size_lz4_bytes": pl.UInt64,
    "polars_ipc_compression": pl.String,
    "polars_ipc_zstd_level": pl.Int32,
    "polars_ipc_size_bytes": pl.UInt64,
    "clickhouse_type": pl.String,
    "clickhouse_codecs": pl.String,
    "clickhouse_skip_index": pl.String,
    "clickhouse_size_bytes": pl.UInt64,
    "clickhouse_candidates": pl.List(
        pl.Struct(
            {
                "codecs": pl.String,
                "predicted_bytes": pl.UInt64,
                "outcome": pl.String,
                "evidence": pl.String,
            }
        )
    ),
    "clickhouse_skip_indexes": pl.List(
        pl.Struct(
            {
                "kind": pl.String,
                "granularity": pl.UInt32,
                "max_size": pl.UInt32,
                "fpr": pl.Float64,
                "bytes": pl.UInt64,
                "skip_rows": pl.Float64,
                "skip_values": pl.Float64,
                "expected_skip_rows": pl.Float64,
                "outcome": pl.String,
                "evidence": pl.String,
            }
        )
    ),
    "candidates": pl.List(CANDIDATE),
    "n_sampled_rows": pl.UInt64,
    "n_sampled_blocks": pl.UInt64,
}


def stream(frame: pl.DataFrame, batch_rows: int, **params) -> pl.DataFrame:
    rec = StreamingRecommender(**params)
    for off in range(0, max(frame.height, 1), batch_rows):
        rec.add(frame.slice(off, batch_rows))
    return rec.result()


def one_shot(frame: pl.DataFrame) -> dict[str, dict]:
    out = OneShotRecommender().add(frame).result()
    return {r["column"]: r for r in out.iter_rows(named=True)}


def kept_original(r: dict) -> bool:
    return any(
        c["outcome"] == "chosen" and c["rule"] == "original" for c in r["candidates"]
    )


def candidates(
    r: dict, with_original_sizes: bool, dictionary_sizes: bool = True
) -> list:
    return [
        (
            (
                c["arrow_type"]
                if dictionary_sizes or c["rule"] != "string→dictionary"
                else None
            ),
            c["rule"],
            str(c["outcome"]),
            (
                None
                if (c["rule"] == "original" and not with_original_sizes)
                or (c["rule"] == "string→dictionary" and not dictionary_sizes)
                else c["predicted_bytes"]
            ),
        )
        for c in r["candidates"]
    ]


def null_slots_hold_bytes(s: pl.Series) -> bool:
    """A string column whose null rows hold bytes (Polars' int → String cast writes the
    null slots' values): one-shot measures them in the original's buffers, the
    analytic size counts values only."""
    if s.dtype != pl.String:
        return False
    data = s.to_arrow(compat_level=pl.CompatLevel.oldest()).buffers()[2]
    return (data.size if data else 0) > (s.str.len_bytes().sum() or 0)


def per_batch_sum(r: dict) -> bool:
    """The streamed original's size is a sum of per-batch measurements (Struct and
    deeper nesting): exact for one batch only."""
    return any(
        c["rule"] == "original" and "per-batch sum" in (c["evidence"] or "")
        for c in r["candidates"]
    )


def wide_temporal(r: dict, o: dict) -> bool:
    """A temporal type derived from text has no exact range in streaming: it maps to the
    wide type (Date32 / DateTime64(0)) where one-shot may prove Date / DateTime."""
    if r["dtype"] not in ("string", "large_string", "string_view"):
        return False
    a, b = r["clickhouse_type"], o["clickhouse_type"]
    if a is None or b is None:
        return False
    for pre, post in (("", ""), ("Nullable(", ")")):
        if not (a.startswith(pre) and a.endswith(post) and b.startswith(pre)):
            continue
        ia, ib = a[len(pre) : len(a) - len(post)], b[len(pre) : len(b) - len(post)]
        if ia == "Date32" and ib == "Date":
            return True
        if (
            ia.startswith("DateTime64(0")
            and ib.startswith("DateTime")
            and not ib.startswith("DateTime64")
        ):
            return True
    return False


def assert_compression_parity(name: str, r: dict, o: dict):
    """ClickHouse and Arrow IPC compression columns (spec 2026-10-19 §6): equal when the
    sampled block is the whole column; else sizes within 2% and the same codecs unless
    the best chains are within 2%."""
    whole = r["n_sampled_rows"] == r["n_rows"] and r["n_sampled_blocks"] == 1
    # Streaming cannot offer GCD for numbers derived from text.
    gcd_unknown = wide_temporal(r, o) or any(
        "GCD unknown" in (c["evidence"] or "")
        for c in (r["clickhouse_candidates"] or [])
    )
    pairs = [
        ("datafusion_ipc_compression", "datafusion_ipc_zstd_level"),
        ("polars_ipc_compression", "polars_ipc_zstd_level"),
    ]
    ipc = [
        "datafusion_size_lz4_bytes",
        "polars_size_lz4_bytes",
        "datafusion_ipc_size_bytes",
        "polars_ipc_size_bytes",
    ]
    if r["n_sampled_rows"] == 0:
        return
    # With GCD unknown, one-shot's GCD chain may win: ClickHouse sizes and codecs differ.
    ch = [] if gcd_unknown else ["clickhouse_size_bytes"]
    if whole:
        exact = ipc + [x for p in pairs for x in p]
        if not gcd_unknown:
            exact += ["clickhouse_codecs", *ch]
        for k in exact:
            assert r[k] == o[k], (name, k, r[k], o[k])
        return
    for k in ipc + ch:
        if o[k] is None:
            assert r[k] is None, (name, k, r[k])
        else:
            assert r[k] == pytest.approx(o[k], rel=0.02), (name, k, r[k], o[k])
    if not gcd_unknown:
        best = sorted(
            c["predicted_bytes"]
            for c in r["clickhouse_candidates"]
            if c["predicted_bytes"] is not None
        )
        if r["clickhouse_codecs"] != o["clickhouse_codecs"]:
            assert len(best) > 1, (
                name,
                r["clickhouse_codecs"],
                o["clickhouse_codecs"],
                r["clickhouse_candidates"],
            )
            assert best[1] <= best[0] * 1.02, (
                name,
                r["clickhouse_codecs"],
                o["clickhouse_codecs"],
                r["clickhouse_candidates"],
            )


def skip_key(c: dict):
    return (c["kind"], c["granularity"], c["max_size"], c["fpr"])


def assert_skip_index_parity(name: str, r: dict, o: dict):
    """Skip indexes (spec 2026-10-20 §7): with the whole column in one sampled block the
    index and every candidate row equal one-shot's; else the index is equal unless
    one-shot's two best measured candidates are within 0.05 skip_rows."""
    if r["n_sampled_rows"] == 0:
        assert r["clickhouse_skip_index"] is None
        assert not r["clickhouse_skip_indexes"]
        return
    if wide_temporal(r, o):
        return
    if r["n_sampled_rows"] == r["n_rows"] and r["n_sampled_blocks"] == 1:
        assert r["clickhouse_skip_index"] == o["clickhouse_skip_index"], name
        got, want = r["clickhouse_skip_indexes"], o["clickhouse_skip_indexes"]
        assert len(got) == len(want), name
        for g, w in zip(got, want):
            assert skip_key(g) == skip_key(w), (
                name,
                g,
                w,
            )
            assert g["outcome"] == w["outcome"], (
                name,
                g,
                w,
            )
            assert g["bytes"] == w["bytes"], (name, g, w)
            for k in ("skip_rows", "skip_values", "expected_skip_rows"):
                assert g[k] == pytest.approx(w[k], rel=1e-9), (name, k, g, w)
        return
    if r["clickhouse_skip_index"] != o["clickhouse_skip_index"]:
        best = sorted(
            (
                c["skip_rows"]
                for c in o["clickhouse_skip_indexes"] or []
                if c["skip_rows"] is not None and c["outcome"] != "pruned"
            ),
            reverse=True,
        )
        assert len(best) > 1, (
            name,
            r["clickhouse_skip_index"],
            o["clickhouse_skip_index"],
            o["clickhouse_skip_indexes"],
        )
        assert best[0] - best[1] <= 0.05, (
            name,
            r["clickhouse_skip_index"],
            o["clickhouse_skip_index"],
            o["clickhouse_skip_indexes"],
        )


def assert_parity(streamed: pl.DataFrame, frame: pl.DataFrame, single_batch: bool):
    """Parity spec (docs/superpowers/specs/2026-10-01-recommender-parity-design.md)
    §7 / §10: equal recommendations, sizes, candidates and shared value columns; ZSTD
    when N ≤ block_rows.

    Known, accepted differences, narrowed per column:
    - the distinct sample in its sampling phase (`est_method` "hll"): the counts are
      HyperLogLog estimates, so the rejected dictionary's key width and size follow the
      estimated est_high, not one-shot's exact one; n_unique is within 3 standard
      errors (p = 14) and sum_len_unique within 5% rather than exact;
    - the original's size, in several batches, where it is a per-batch sum;
    - string source whose null slots hold bytes: one-shot measures them in the
      original's size.
    """
    ref = one_shot(frame)
    checked = 0
    for r in streamed.filter(pl.col("status") == "computed").iter_rows(named=True):
        name = r["column"]
        o = ref[name]
        overflowed = r["est_method"] == "hll"
        original_sizes = (
            single_batch or not per_batch_sum(r)
        ) and not null_slots_hold_bytes(frame[name])
        keys = ["nullable", "lossy_formatting", "datafusion_type"]
        if original_sizes or not kept_original(o):
            keys += ["datafusion_size_bytes", "polars_size_bytes"]
            keys += ["datafusion_size_zstd_bytes", "polars_size_zstd_bytes"]
        if original_sizes:
            keys += ["size_bytes", "size_polars_bytes"]
        assert r["polars_type"] == o["polars_type"], name
        for k in keys:
            assert r[k] == o[k], (name, k, r[k], o[k])
        got = candidates(r, original_sizes, not overflowed)
        want = candidates(o, original_sizes, not overflowed)
        assert got == want, name
        assert r["clickhouse_type"] == o["clickhouse_type"] or wide_temporal(r, o), (
            name,
            r["clickhouse_type"],
            o["clickhouse_type"],
        )
        assert_compression_parity(name, r, o)
        assert_skip_index_parity(name, r, o)
        # Both rank by exact counts while the dictionary is a candidate (spec 2026-10-07 §4).
        for k in ("top_k", "inner_top_k"):
            assert r[k] == o[k], (name, k, r[k], o[k])
        exact_cols = [
            "n_rows",
            "n_null",
            "min",
            "max",
            "min_len",
            "max_len",
            "sum_len",
            "gcd",
            "n_midnight",
            "n_nan",
            "n_inf",
            "n_fractional",
            "max_frac_digits",
            "n_f32_inexact",
            *[
                c
                for c in VALUE_BLOCK
                if c.startswith(("n_numeric", "numeric_", "n_leading", "n_iso", "iso_"))
            ],
        ]
        for k in exact_cols:
            assert r[k] == o[k], (name, k, r[k], o[k])
        for k in ["exemplar_row", "exemplar_value", "null_exemplar_row"]:
            assert r[k] == o[k], (name, k, r[k], o[k])
        if not overflowed:
            assert r["entropy_method"] == "exact", name
            assert r["entropy"] == pytest.approx(
                o["entropy"], rel=1e-12, abs=1e-12
            ), name
        else:
            assert r["entropy_method"] in ("sample", "ams"), name
            assert (
                r["entropy_low"] - 1e-9 <= o["entropy"] <= r["entropy_high"] + 1e-9
            ), (name, r["entropy_low"], o["entropy"], r["entropy_high"])
        if overflowed:
            sigma = 3 * 1.04 / 128  # p = 14
            assert abs(r["n_unique"] - o["n_unique"]) <= sigma * o["n_unique"] + 1, name
            for k in ["unique", "class"]:
                assert r[k] == o[k], (name, k, r[k], o[k])
            if o["sum_len_unique"] is not None:
                assert r["sum_len_unique"] == pytest.approx(
                    o["sum_len_unique"], rel=0.05
                ), (name, r["sum_len_unique"], o["sum_len_unique"])
        else:
            for k in ["n_unique", "unique", "class", "sum_len_unique"]:
                assert r[k] == o[k], (name, k, r[k], o[k])
        if r["inner_n_values"] is not None:
            for k in [
                "inner_n_values",
                "inner_n_null",
                "inner_min",
                "inner_max",
                "inner_min_len",
                "inner_max_len",
                "inner_sum_len",
                "inner_gcd",
            ]:
                assert r[k] == o[k], (name, k, r[k], o[k])
            if r["inner_est_method"] == "hll":
                sigma = 3 * 1.04 / 128  # p = 14
                assert (
                    abs(r["inner_n_unique"] - o["inner_n_unique"])
                    <= sigma * o["inner_n_unique"] + 1
                ), name
            else:
                for k in ["inner_n_unique", "inner_unique", "inner_class"]:
                    assert r[k] == o[k], (name, k, r[k], o[k])
        assert r["est_low"] <= r["est_cardinality"] <= r["est_high"], name
        checked += 1
    assert checked > 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


def test_contract():
    out = stream(describe_mixed(200), 50)
    assert list(out.schema.items()) == list(SCHEMA.items())
    assert out.to_arrow().num_rows == out.height
    assert out["column"].to_list() == describe_mixed(10).columns
    empty = StreamingRecommender().result()
    assert empty.height == 0
    assert list(empty.schema) == list(SCHEMA)


def test_ineligible_columns_are_listed():
    frame = pl.DataFrame(
        {"w": pl.Series([1, 2], dtype=pl.Int128), "a": [1, 2], "n": [None, None]}
    )
    out = StreamingRecommender().add(frame).result()
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert rows["w"]["status"] == "ineligible"
    assert rows["w"]["dtype"] == "Int128"
    assert rows["w"]["n_null"] is None
    assert rows["w"]["datafusion_type"] is None
    assert rows["n"]["status"] == "ineligible"  # still the Null type
    assert rows["a"]["status"] == "computed"


def test_result_keeps_the_state():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    assert rec.result()["n_rows"].to_list() == [2]
    assert rec.add(pl.DataFrame({"a": [3]})).result()["n_rows"].to_list() == [3]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parity with OneShotRecommender (spec §6)


@pytest.mark.parametrize("batch_rows", [1, 7, 60, 200])
@pytest.mark.parametrize(
    "make",
    [lambda: describe_mixed(200), lambda: stringified(describe_mixed(200))],
    ids=["mixed", "stringified"],
)
def test_parity_with_one_shot(make, batch_rows):
    frame = make()
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)


@pytest.mark.parametrize("batch_rows", [7, 100])
def test_parity_with_list_columns(batch_rows):
    frame = pl.DataFrame(
        {
            "l": [[1, 2], None, [3], [None, 4], [5, 5, 6]] * 20,
            "a": pl.Series(
                [[1.5, 2.0], [0.5, None]] * 50, dtype=pl.Array(pl.Float64, 2)
            ),
        }
    )
    streamed = stream(frame, batch_rows)
    assert streamed["inner_n_values"].null_count() == 0
    assert_parity(streamed, frame, batch_rows >= frame.height)


@pytest.mark.slow
@pytest.mark.parametrize("batch_rows", [7_919, 50_000])
@pytest.mark.parametrize(
    "make", [large_dataset, large_dataset_mixed], ids=["large", "mixed"]
)
def test_parity_on_the_large_dataset(batch_rows, make):
    frame = make()
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)


def test_parity_with_columns_appearing_and_disappearing():
    parts = [
        pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}),
        pl.DataFrame({"a": [4], "c": [0.5]}),
        pl.DataFrame({"b": ["z", None], "c": [1.25, None]}),
    ]
    rec = StreamingRecommender()
    for p in parts:
        rec.add(p)
    assert_parity(rec.result(), pl.concat(parts, how="diagonal"), single_batch=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZSTD sizes against the IPC oracle, and the sampled estimate


def test_zstd_sizes_are_those_of_a_file_written_in_blocks():
    frame = describe_mixed(3_000).select("i32", "u16", "f64_price", "str_free", "date")
    block = 1_000
    out = stream(frame, 700, block_rows=block, reservoir_rows=4 * block)
    for r in out.iter_rows(named=True):
        s = frame[r["column"]]
        want = sum(
            _sizes.ipc_body_bytes(
                s.slice(o, block).to_arrow(compat_level=pl.CompatLevel.oldest()), 1
            )
            for o in range(0, frame.height, block)
        )
        assert r["size_zstd_bytes"] == want, r["column"]
        assert r["n_sampled_rows"] == frame.height


@pytest.mark.slow
def test_sampled_zstd_estimate_is_within_ten_percent():
    frame = large_dataset()
    full = stream(frame, 10_000, block_rows=4_096, reservoir_rows=frame.height + 4_096)
    sampled = stream(frame, 10_000, block_rows=4_096, reservoir_rows=16_384)
    for a, b in zip(full.iter_rows(named=True), sampled.iter_rows(named=True)):
        if a["status"] != "computed" or (a["datafusion_size_zstd_bytes"] or 0) < 1_000:
            continue
        assert b["datafusion_size_zstd_bytes"] == pytest.approx(
            a["datafusion_size_zstd_bytes"], rel=0.10
        ), a["column"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Known answers


def row(out: pl.DataFrame, column: str) -> dict:
    return out.filter(pl.col("column") == column).row(0, named=True)


def by_rule(r: dict) -> dict:
    return {c["rule"]: c for c in r["candidates"]}


def test_a_new_column_is_backfilled():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    out = rec.add(pl.DataFrame({"a": [3], "b": ["x"]})).result()
    b = row(out, "b")
    assert (b["first_row"], b["n_rows"], b["n_null"], b["nullable"]) == (
        2,
        3,
        2,
        True,
    )


def test_an_absent_column_counts_as_null():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1], "b": [1]}))
    out = rec.add(pl.DataFrame({"a": [2, 3]})).result()
    assert row(out, "b")["n_null"] == 2


def test_an_all_ineligible_frame_keeps_its_rows():
    out = (
        StreamingRecommender()
        .add(pl.DataFrame({"k": [1]}))
        .add(pl.DataFrame({"w": pl.Series([1, 2, 3], dtype=pl.Int128)}))
        .add(pl.DataFrame({"k": [2]}))
        .result()
    )
    k = row(out, "k")
    assert (k["n_rows"], k["n_null"]) == (5, 3)
    assert row(out, "w")["status"] == "ineligible"


def test_a_null_typed_column_adopts_a_type():
    rec = StreamingRecommender().add(
        pl.DataFrame({"b": pl.Series([None, None], dtype=pl.Null)})
    )
    b = row(rec.add(pl.DataFrame({"b": ["x", "y"]})).result(), "b")
    assert b["status"] == "computed"
    assert b["n_null"] == 2


def test_a_type_change_is_rejected():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1]}))
    changed = pl.DataFrame({"a": ["x"]})
    with pytest.raises(ValueError, match="type changed"):
        rec.add(changed)
    assert row(rec.result(), "a")["n_rows"] == 1


def test_exact_count_past_the_threshold_rejects_the_dictionary():
    frame = pl.DataFrame({"s": ["a", "b", "c", "d", "e", "a"]})
    s = row(stream(frame, 2, categorical_threshold=3), "s")
    assert s["n_unique"] == 5
    assert s["est_method"] == "observed"
    assert by_rule(s)["string→dictionary"]["outcome"] == "rejected"


def test_overflow_rejects_the_dictionary():
    # More distinct values than the sample holds (k = max(threshold, 1000)).
    frame = pl.DataFrame({"s": [f"v{i}" for i in range(2_000)]})
    s = row(stream(frame, 500, categorical_threshold=3), "s")
    assert s["n_unique"] >= 1_000 + 1
    assert s["est_method"] == "hll"
    assert by_rule(s)["string→dictionary"]["outcome"] == "rejected"


def test_sampling_phase_estimates():
    n = 60_000
    frame = pl.DataFrame(
        {
            "key": pl.Series([f"id-{i:08d}" for i in range(n)]),
            "x": [i % 7 for i in range(n)],
        }
    )
    out = stream(frame, 8_192)
    r = {row["column"]: row for row in out.iter_rows(named=True)}
    key = r["key"]
    assert key["est_method"] == "hll"
    assert key["unique"] is True
    assert abs(key["n_unique"] - n) <= 3 * 1.04 / 128 * n
    assert key["n_unique"] <= n
    assert key["est_high"] <= n
    assert abs(key["sum_len_unique"] - 11 * n) <= 0.05 * 11 * n
    assert key["class"] == "discrete"
    assert r["x"]["class"] == "ordinal"
    assert r["x"]["n_unique"] == 7


def test_no_sample_means_no_zstd_sizes():
    out = stream(pl.DataFrame({"a": [1, 2, 3]}), 2, reservoir_rows=0)
    a = row(out, "a")
    assert a["size_zstd_bytes"] is None
    assert a["datafusion_size_zstd_bytes"] is None
    assert (a["n_sampled_rows"], a["datafusion_type"]) == (0, "uint8")


def test_a_float_that_does_not_round_trip_fails_by_statistic():
    tiny = "0." + "0" * 400 + "1"
    s = row(stream(pl.DataFrame({"s": [tiny, "1"]}), 1), "s")
    c = by_rule(s)["string→float64"]
    assert c["outcome"] == "failed"
    assert c["reason"].startswith("n_f64_roundtrip_fail=")


def test_nanoseconds_out_of_range_fail_by_statistic():
    frame = pl.DataFrame(
        {"s": ["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"]}
    )
    c = by_rule(row(stream(frame, 1), "s"))["string→timestamp"]
    assert c["outcome"] == "failed"
    assert "iso_instant" in c["reason"]


@pytest.mark.parametrize(
    "params",
    [
        {"block_rows": 0},
        {"reservoir_rows": 10, "block_rows": 100},
        {"boolean_pairs": (("a", "A"),)},
        {"zstd_level": 99},
    ],
)
def test_parameters_are_validated(params):
    with pytest.raises(ValueError):
        StreamingRecommender(**params)


def test_lazyframe_is_refused():
    tech, data = StreamingRecommender(), pl.LazyFrame({"a": [1]})
    with pytest.raises(TypeError, match="LazyFrame"):
        tech.add(data)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Ordinary Polars frames: nested Null, sliced nested columns


@pytest.mark.parametrize(
    "dtype, value",
    [
        (pl.List(pl.Null), [None]),
        (pl.Struct({"n": pl.Null}), {"n": None}),
        (pl.Array(pl.Null, 2), [None, None]),
        (pl.List(pl.Struct({"n": pl.Null})), [{"n": None}]),
    ],
    ids=["list", "struct", "array", "list_struct"],
)
def test_a_nested_null_column_is_ineligible_like_one_shot(dtype, value):
    frame = pl.DataFrame({"c": pl.Series([value, None], dtype=dtype), "k": [1, 2]})
    out = StreamingRecommender().add(frame).result()
    ref = one_shot(frame)
    for r in out.iter_rows(named=True):
        assert r["status"] == ref[r["column"]]["status"], r["column"]
    c = row(out, "c")
    assert (c["status"], c["dtype"]) == ("ineligible", str(dtype))
    assert row(out, "k")["n_rows"] == 2


def test_a_nested_null_arrow_column_is_ineligible():
    table = pa.table({"c": pa.array([[None], None], pa.list_(pa.null())), "k": [1, 2]})
    out = StreamingRecommender().add(table).result()
    assert row(out, "c")["status"] == "ineligible"
    assert row(out, "k")["status"] == "computed"


SLICED = {
    "list_array": pl.Series(
        [[[1, 2]], None, [[3, 4], None], [[5, 6]]], dtype=pl.List(pl.Array(pl.Int64, 2))
    ),
    "list_struct": pl.Series([[{"a": 1}], None, [{"a": 3}, None], [{"a": 5}]]),
    "list_list_struct": pl.Series(
        [[[{"a": 1}]], None, [[{"a": 3}, None], None], [[{"a": 5}]]]
    ),
    "array": pl.Series([[1, 2], None, [3, None], [5, 6]], dtype=pl.Array(pl.Int64, 2)),
    "struct": pl.Series([{"a": 1}, None, {"a": 3}, {"a": None}]),
    "struct_list": pl.Series([{"a": [1]}, None, {"a": [3, None]}, {"a": None}]),
    "array_array": pl.Series(
        [[[1, 2], [3, 4]], None, [[5, 6], None], [[7, 8], [9, 0]]],
        dtype=pl.Array(pl.Array(pl.Int64, 2), 2),
    ),
}


@pytest.mark.parametrize("name", list(SLICED))
@pytest.mark.parametrize("offset", [1, 2, 8, 16])
def test_a_sliced_nested_column_streams_like_one_shot(name, offset):
    # Repeated so the slice at offsets 8 and 16 (validity offset a multiple of 8) is in range.
    frame = pl.DataFrame({"c": pl.concat([SLICED[name]] * 6)}).slice(offset, 2)
    out = StreamingRecommender().add(frame).result()
    assert row(out, "c")["n_null"] == frame["c"].null_count()
    assert_parity(out, frame, single_batch=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Malformed or unsupported Arrow input is refused, never a crash


def dictionary_keys_out_of_range() -> pa.Table:
    keys = pa.array([0, 1, 5, 1], pa.int32())
    return pa.table(
        {"c": pa.DictionaryArray.from_arrays(keys, pa.array(["x", "y"]), safe=False)}
    )


def invalid_utf8() -> pa.Table:
    offsets = pa.py_buffer(struct.pack("<3i", 0, 2, 4))
    data = pa.py_buffer(bytes([0xFF, 0xFE, 0xC3, 0x28]))
    return pa.table({"c": pa.Array.from_buffers(pa.string(), 2, [None, offsets, data])})


def decimal256() -> pa.Table:
    return pa.table({"c": pa.array([1, 2], pa.decimal256(10, 2))})


MALFORMED = [
    (dictionary_keys_out_of_range, "out of bounds"),
    (invalid_utf8, "(?i)utf-?8"),
    (lambda: pl.DataFrame(invalid_utf8()), "(?i)utf-?8"),  # Polars does not check it
    (decimal256, "Decimal256"),
]


@pytest.mark.parametrize(
    "make, match",
    MALFORMED,
    ids=["dictionary_keys", "utf8", "utf8_polars", "decimal256"],
)
def test_malformed_input_is_a_value_error(make, match):
    data = make()
    tech = StreamingRecommender()
    with pytest.raises(ValueError, match=match):
        tech.add(data)
    # One-shot refuses the same input in add() or, for Polars-built data, in result().
    tech = OneShotRecommender()
    with pytest.raises(ValueError, match=match):
        tech.add(data).result()


def test_an_unsupported_type_is_refused_even_with_no_rows():
    # A zero-row table has no batches, so only the schema can be checked.
    import decimal

    def empty():
        col = pa.array([decimal.Decimal("1")], pa.decimal256(40, 2))
        return pa.table({"c": col}).slice(1, 0)

    data = empty()
    for tech in (StreamingRecommender(), OneShotRecommender(), GcdRust()):
        frames = data if not isinstance(tech, GcdRust) else {"t": data}
        with pytest.raises(ValueError, match="Decimal256"):
            tech.add(frames)


@pytest.mark.parametrize("pairs", [(("y",),), ((1, 2),), (("a", "b", "c"),)])
def test_boolean_pairs_must_be_pairs(pairs):
    with pytest.raises(ValueError, match="boolean_pairs must be pairs of two strings"):
        StreamingRecommender(boolean_pairs=pairs)


def test_top_k_over_batches():
    frame = pl.DataFrame({"s": ["b", "a", "b", None, "c", "a", "b"]})
    out = stream(frame, 3, top_k=1)
    assert out["top_k"].to_list() == [[{"key": "b", "value": 3}]]


def test_top_k_streams_exactly_like_one_shot():
    frame = pl.DataFrame(
        {
            "s": ["b", "a", "b", None, "c", "a", "b"] * 30,
            "l": pl.Series([["b", "a"], None, ["c"], [], ["a"], None, ["b"]] * 30),
            "c": pl.Series(
                ["q", "p", "q", None, "r", "p", "q"] * 30, dtype=pl.Categorical
            ),
        }
    )
    want = (
        OneShotRecommender()
        .add(frame)
        .result()
        .select("column", "top_k", "inner_top_k")
    )
    assert want["top_k"][0].to_list()[0] == {"key": "b", "value": 90}
    for batch_rows in (1, 7, 210):
        got = stream(frame, batch_rows).select("column", "top_k", "inner_top_k")
        assert got.equals(want), batch_rows


def test_top_k_is_null_past_the_sample():
    frame = pl.DataFrame({"s": [f"v{i}" for i in range(12_000)]})
    assert stream(frame, 4_000)["top_k"].to_list() == [None]
