"""
StreamingRecommender accuracy tests. Oracles: one-shot RecommendRust on the batches
concatenated diagonally (parity, spec §6); the pyarrow IPC oracle (_sizes) for ZSTD
sizes; a complete sample for the sampled ZSTD estimate; hand-worked known answers.
Accuracy only — nothing here is timed.
"""

from pathlib import Path

import polars as pl
import pytest

from analytics.describe import _sizes
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
        c["outcome"] == "chosen" and c["rule"] == "original" for c in r["rec_candidates"]
    )


def candidates(r: dict, with_original_sizes: bool, dictionary_sizes: bool = True) -> list:
    return [
        (
            c["arrow_type"] if dictionary_sizes or c["rule"] != "string→dictionary" else None,
            c["rule"],
            str(c["outcome"]),
            None
            if (c["rule"] == "original" and not with_original_sizes)
            or (c["rule"] == "string→dictionary" and not dictionary_sizes)
            else c["predicted_bytes"],
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


def assert_parity(streamed: pl.DataFrame, frame: pl.DataFrame, single_batch: bool):
    """Spec §6: equal recommendations, sizes and candidates; ZSTD when N ≤ block_rows.

    Known, accepted differences, narrowed per column:
    - distinct tracking overflowed: parity is not defined (spec §6); the rejected
      dictionary's key width follows `categorical_threshold + 1`, not one-shot's est_high;
    - Categorical / Enum source, several batches: the original's size is a per-batch
      sum (every batch carries the dictionary), so the original candidate can move in
      the order and even win or lose; the other candidates are compared without it;
    - string source whose null slots hold bytes: one-shot measures them in the
      original's size.
    """
    ref = one_shot(frame)
    checked = 0
    for r in streamed.filter(pl.col("status") == "computed").iter_rows(named=True):
        name = r["column"]
        o = ref[name]
        overflowed = bool(r["distinct_overflowed"])
        per_batch_original = not single_batch and isinstance(
            frame.schema[name], (pl.Categorical, pl.Enum)
        )
        original_sizes = single_batch and not null_slots_hold_bytes(frame[name])
        kept = kept_original(o) or kept_original(r)
        keys = ["rec_nullable", "rec_lossy_formatting"]
        if not (per_batch_original and kept):
            keys += ["rec_arrow_type"]
            if single_batch or not kept_original(o):
                keys += ["rec_arrow_size_bytes", "rec_polars_size_bytes"]
                keys += ["rec_arrow_size_zstd_bytes", "rec_polars_size_zstd_bytes"]
            if not kept_original(o):
                assert r["rec_polars_type"] == o["rec_polars_type"], name
        for k in keys:
            assert r[k] == o[k], (name, k, r[k], o[k])
        got = candidates(r, original_sizes, not overflowed)
        want = candidates(o, original_sizes, not overflowed)
        if per_batch_original:
            got = [c for c in got if c[1] != "original"]
            want = [c for c in want if c[1] != "original"]
            if kept:
                got = [c[:2] + c[3:] for c in got]
                want = [c[:2] + c[3:] for c in want]
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
    assert (b["first_row"], b["n_rows"], b["n_null"], b["rec_nullable"]) == (2, 3, 2, True)


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
    rec = StreamingRecommender().add(pl.DataFrame({"b": pl.Series([None, None], dtype=pl.Null)}))
    b = row(rec.add(pl.DataFrame({"b": ["x", "y"]})).finish(), "b")
    assert b["status"] == "computed" and b["n_null"] == 2


def test_a_type_change_is_rejected():
    rec = StreamingRecommender().add(pl.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="type changed"):
        rec.add(pl.DataFrame({"a": ["x"]}))
    assert row(rec.finish(), "a")["n_rows"] == 1


def test_overflow_rejects_the_dictionary():
    frame = pl.DataFrame({"s": ["a", "b", "c", "d", "e", "a"]})
    s = row(stream(frame, 2, categorical_threshold=3), "s")
    assert s["n_unique"] is None and s["distinct_overflowed"] is True
    assert s["est_method"] == "overflowed"
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
    frame = pl.DataFrame({"s": ["2300-01-01T00:00:00.123456789", "2024-01-01T00:00:00"]})
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
