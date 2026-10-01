"""
StreamingRecommender accuracy tests. Oracles: one-shot RecommendRust on the batches
concatenated diagonally (parity, spec §6); the pyarrow IPC oracle (_sizes) for ZSTD
sizes; a complete sample for the sampled ZSTD estimate; hand-worked known answers.
Accuracy only — nothing here is timed.
"""

import struct
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from analytics.describe import _sizes
from analytics.gcd import GcdRust
from analytics.recommend import RecommendRust, StreamingRecommender
from datagen import describe_mixed, stringified

LARGE = Path(__file__).parent / "data" / "large_dataset.arrow"

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
SCHEMA = {
    "column": pl.String,
    "status": pl.String,
    "dtype": pl.String,
    "first_row": pl.UInt64,
    "n_rows": pl.UInt64,
    "n_null": pl.UInt64,
    "min": pl.String,
    "max": pl.String,
    "gcd": pl.Decimal(38, 0),
    "sum_len": pl.UInt64,
    "min_len": pl.UInt64,
    "max_len": pl.UInt64,
    "n_unique": pl.UInt64,
    "distinct_overflowed": pl.Boolean,
    "est_cardinality": pl.Float64,
    "est_low": pl.Float64,
    "est_high": pl.Float64,
    "est_method": pl.String,
    "size_bytes": pl.UInt64,
    "size_zstd_bytes": pl.UInt64,
    "size_polars_bytes": pl.UInt64,
    "size_polars_zstd_bytes": pl.UInt64,
    "rec_nullable": pl.Boolean,
    "rec_arrow_type": pl.String,
    "rec_arrow_size_bytes": pl.UInt64,
    "rec_arrow_size_zstd_bytes": pl.UInt64,
    "rec_polars_type": pl.String,
    "rec_polars_size_bytes": pl.UInt64,
    "rec_polars_size_zstd_bytes": pl.UInt64,
    "rec_lossy_formatting": pl.Boolean,
    "rec_candidates": pl.List(CANDIDATE),
    "n_sampled_rows": pl.UInt64,
    "n_sampled_blocks": pl.UInt64,
}


def stream(frame: pl.DataFrame, batch_rows: int, **params) -> pl.DataFrame:
    rec = StreamingRecommender(**params)
    for off in range(0, max(frame.height, 1), batch_rows):
        rec.add(frame.slice(off, batch_rows))
    return rec.finish()


def one_shot(frame: pl.DataFrame) -> dict[str, dict]:
    out = RecommendRust().add({"t": frame}).result()
    return {r["col_a"]: r for r in out.iter_rows(named=True)}


def kept_original(r: dict) -> bool:
    return any(
        c["outcome"] == "chosen" and c["rule"] == "original"
        for c in r["rec_candidates"]
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
        for c in r["rec_candidates"]
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
        for c in r["rec_candidates"]
    )


def assert_parity(streamed: pl.DataFrame, frame: pl.DataFrame, single_batch: bool):
    """Spec §6: equal recommendations, sizes and candidates; ZSTD when N ≤ block_rows.

    Known, accepted differences, narrowed per column:
    - distinct tracking overflowed: parity is not defined (spec §6); the rejected
      dictionary's key width follows `categorical_threshold + 1`, not one-shot's est_high;
    - the original's size, in several batches, where it is a per-batch sum;
    - string source whose null slots hold bytes: one-shot measures them in the
      original's size.
    """
    ref = one_shot(frame)
    checked = 0
    for r in streamed.filter(pl.col("status") == "computed").iter_rows(named=True):
        name = r["column"]
        o = ref[name]
        overflowed = bool(r["distinct_overflowed"])
        original_sizes = (
            single_batch or not per_batch_sum(r)
        ) and not null_slots_hold_bytes(frame[name])
        keys = ["rec_nullable", "rec_lossy_formatting", "rec_arrow_type"]
        if original_sizes or not kept_original(o):
            keys += ["rec_arrow_size_bytes", "rec_polars_size_bytes"]
            keys += ["rec_arrow_size_zstd_bytes", "rec_polars_size_zstd_bytes"]
        if original_sizes:
            keys += ["size_bytes", "size_polars_bytes"]
        if not kept_original(o):
            assert r["rec_polars_type"] == o["rec_polars_type"], name
        for k in keys:
            assert r[k] == o[k], (name, k, r[k], o[k])
        got = candidates(r, original_sizes, not overflowed)
        want = candidates(o, original_sizes, not overflowed)
        assert got == want, name
        checked += 1
    assert checked > 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Contract


def test_contract():
    out = stream(describe_mixed(200), 50)
    assert list(out.schema.items()) == list(SCHEMA.items())
    assert out.to_arrow().num_rows == out.height
    assert out["column"].to_list() == describe_mixed(10).columns
    empty = StreamingRecommender().finish()
    assert empty.height == 0 and list(empty.schema) == list(SCHEMA)


def test_ineligible_columns_are_listed():
    frame = pl.DataFrame(
        {"w": pl.Series([1, 2], dtype=pl.Int128), "a": [1, 2], "n": [None, None]}
    )
    out = StreamingRecommender().add(frame).finish()
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert rows["w"]["status"] == "ineligible" and rows["w"]["dtype"] == "Int128"
    assert rows["w"]["n_null"] is None and rows["w"]["rec_arrow_type"] is None
    assert rows["n"]["status"] == "ineligible"  # still the Null type
    assert rows["a"]["status"] == "computed"


def test_finish_keeps_the_state():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    assert rec.finish()["n_rows"].to_list() == [2]
    assert rec.add(pl.DataFrame({"a": [3]})).finish()["n_rows"].to_list() == [3]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parity with one-shot RecommendRust (spec §6)


@pytest.mark.parametrize("batch_rows", [1, 7, 60, 200])
@pytest.mark.parametrize(
    "make",
    [lambda: describe_mixed(200), lambda: stringified(describe_mixed(200))],
    ids=["mixed", "stringified"],
)
def test_parity_with_one_shot(make, batch_rows):
    frame = make()
    assert_parity(stream(frame, batch_rows), frame, batch_rows >= frame.height)


@pytest.mark.slow
@pytest.mark.parametrize("batch_rows", [7_919, 50_000])
def test_parity_on_the_large_dataset(batch_rows):
    frame = pl.read_ipc(LARGE)
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
    assert_parity(rec.finish(), pl.concat(parts, how="diagonal"), single_batch=False)


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
    frame = pl.read_ipc(LARGE)
    full = stream(frame, 10_000, block_rows=4_096, reservoir_rows=frame.height + 4_096)
    sampled = stream(frame, 10_000, block_rows=4_096, reservoir_rows=16_384)
    for a, b in zip(full.iter_rows(named=True), sampled.iter_rows(named=True)):
        if a["status"] != "computed" or (a["rec_arrow_size_zstd_bytes"] or 0) < 1_000:
            continue
        assert b["rec_arrow_size_zstd_bytes"] == pytest.approx(
            a["rec_arrow_size_zstd_bytes"], rel=0.10
        ), a["column"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Known answers


def row(out: pl.DataFrame, column: str) -> dict:
    return out.filter(pl.col("column") == column).row(0, named=True)


def by_rule(r: dict) -> dict:
    return {c["rule"]: c for c in r["rec_candidates"]}


def test_a_new_column_is_backfilled():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1, 2]}))
    out = rec.add(pl.DataFrame({"a": [3], "b": ["x"]})).finish()
    b = row(out, "b")
    assert (b["first_row"], b["n_rows"], b["n_null"], b["rec_nullable"]) == (
        2,
        3,
        2,
        True,
    )


def test_an_absent_column_counts_as_null():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1], "b": [1]}))
    out = rec.add(pl.DataFrame({"a": [2, 3]})).finish()
    assert row(out, "b")["n_null"] == 2


def test_an_all_ineligible_frame_keeps_its_rows():
    out = (
        StreamingRecommender()
        .add(pl.DataFrame({"k": [1]}))
        .add(pl.DataFrame({"w": pl.Series([1, 2, 3], dtype=pl.Int128)}))
        .add(pl.DataFrame({"k": [2]}))
        .finish()
    )
    k = row(out, "k")
    assert (k["n_rows"], k["n_null"]) == (5, 3)
    assert row(out, "w")["status"] == "ineligible"


def test_a_null_typed_column_adopts_a_type():
    rec = StreamingRecommender().add(
        pl.DataFrame({"b": pl.Series([None, None], dtype=pl.Null)})
    )
    b = row(rec.add(pl.DataFrame({"b": ["x", "y"]})).finish(), "b")
    assert b["status"] == "computed" and b["n_null"] == 2


def test_a_type_change_is_rejected():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="type changed"):
        rec.add(pl.DataFrame({"a": ["x"]}))
    assert row(rec.finish(), "a")["n_rows"] == 1


def test_overflow_rejects_the_dictionary():
    # More distinct values than the sample holds (k = max(threshold, 1000)).
    frame = pl.DataFrame({"s": [f"v{i}" for i in range(2_000)]})
    s = row(stream(frame, 500, categorical_threshold=3), "s")
    assert s["n_unique"] >= 1_000 + 1 and s["distinct_overflowed"] is True
    assert s["est_method"] == "hll"
    assert by_rule(s)["string→dictionary"]["outcome"] == "rejected"


def test_no_sample_means_no_zstd_sizes():
    out = stream(pl.DataFrame({"a": [1, 2, 3]}), 2, reservoir_rows=0)
    a = row(out, "a")
    assert a["size_zstd_bytes"] is None and a["rec_arrow_size_zstd_bytes"] is None
    assert (a["n_sampled_rows"], a["rec_arrow_type"]) == (0, "uint8")


def test_a_float_that_does_not_round_trip_fails_by_statistic():
    tiny = "0." + "0" * 400 + "1"
    s = row(stream(pl.DataFrame({"s": [tiny, "1"]}), 1), "s")
    c = by_rule(s)["string→float64"]
    assert c["outcome"] == "failed" and c["reason"].startswith("n_f64_roundtrip_fail=")


def test_nanoseconds_out_of_range_fail_by_statistic():
    frame = pl.DataFrame(
        {"s": ["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"]}
    )
    c = by_rule(row(stream(frame, 1), "s"))["string→timestamp"]
    assert c["outcome"] == "failed" and "iso_instant" in c["reason"]


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
    with pytest.raises(TypeError, match="LazyFrame"):
        StreamingRecommender().add(pl.LazyFrame({"a": [1]}))


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
    out = StreamingRecommender().add(frame).finish()
    ref = one_shot(frame)
    for r in out.iter_rows(named=True):
        assert r["status"] == ref[r["column"]]["status"], r["column"]
    c = row(out, "c")
    assert (c["status"], c["dtype"]) == ("ineligible", str(dtype))
    assert row(out, "k")["n_rows"] == 2


def test_a_nested_null_arrow_column_is_ineligible():
    table = pa.table({"c": pa.array([[None], None], pa.list_(pa.null())), "k": [1, 2]})
    out = StreamingRecommender().add(table).finish()
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
    out = StreamingRecommender().add(frame).finish()
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
    with pytest.raises(ValueError, match=match):
        StreamingRecommender().add(make())
    with pytest.raises(ValueError, match=match):
        RecommendRust().add({"t": make()}).result()


def test_an_unsupported_type_is_refused_even_with_no_rows():
    # A zero-row table has no batches, so only the schema can be checked.
    import decimal

    def empty():
        col = pa.array([decimal.Decimal("1")], pa.decimal256(40, 2))
        return pa.table({"c": col}).slice(1, 0)

    with pytest.raises(ValueError, match="Decimal256"):
        StreamingRecommender().add(empty())
    with pytest.raises(ValueError, match="Decimal256"):
        RecommendRust().add({"t": empty()}).result()
    with pytest.raises(ValueError, match="Decimal256"):
        GcdRust().add({"t": empty()}).result()
