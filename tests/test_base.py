"""
Unit tests for the Technique contract (analytics/base.py) using a toy technique.

The toy's metric is the total length of the column names in a combination / 10,
so every expected value below can be worked out by hand.
"""

import weakref

import polars as pl
import pyarrow as pa
import pytest

from analytics._dtypes import encodable, is_nested, value_family
from analytics.base import (
    STATUS,
    Technique,
    at_least,
    check_unit,
    columns_of,
    computed,
    group_by_frame,
    lazy_attributes,
    metric_mismatches,
    same_value,
)


class Toy(Technique):
    SCOPE = "ordered"
    ARITY = 2
    METRICS = {"score": pl.Float64}
    CONCLUSIONS = {"high": pl.Boolean}

    def __init__(self, *, threshold: float = 0.5):
        super().__init__()
        check_unit("threshold", threshold)
        self.threshold = threshold

    def eligible(self, series: pl.Series) -> bool:
        return series.dtype == pl.Int64

    def _compute(self, frames, combos):
        return self.metrics_frame(
            combos, {"score": [sum(len(c) for _, c in k) / 10 for k in combos]}
        )

    def _conclude(self, out):
        return out.with_columns(high=computed(at_least("score", self.threshold)))


class PerColumnToy(Toy):
    SCOPE = "per_column"
    ARITY = 1


class MultiSetToy(Toy):
    SCOPE = "multi_set"

    def eligible(self, series):
        return True

    def compatible(self, dtypes):
        return len({value_family(d) for d in dtypes}) == 1


class TripletToy(Toy):
    ARITY = 3

    def eligible(self, series):
        return True


FRAMES = {
    "f": pl.DataFrame({"a": [1, 2], "bb": [3, 4], "s": ["x", "y"]}),
    "g": pl.DataFrame({"a": [5, 6], "ccc": [7, 8]}),
}
F_A, F_BB, F_S, G_A, G_CCC = (
    ("f", "a"),
    ("f", "bb"),
    ("f", "s"),
    ("g", "a"),
    ("g", "ccc"),
)


# ── enumeration ──────────────────────────────────────────────────────────────


def test_enumerate_ordered_stays_within_each_frame():
    assert Toy.enumerate(FRAMES) == [(F_A, F_BB), (F_A, F_S), (F_BB, F_S), (G_A, G_CCC)]


def test_enumerate_per_column():
    assert PerColumnToy.enumerate(FRAMES) == [(F_A,), (F_BB,), (F_S,), (G_A,), (G_CCC,)]


def test_enumerate_multi_set_spans_frames_and_includes_same_frame_pairs():
    combos = MultiSetToy.enumerate(FRAMES)
    assert len(combos) == 10
    assert (F_A, G_A) in combos and (F_A, F_BB) in combos


def test_enumerate_ordered_triplets():
    assert TripletToy.enumerate(FRAMES) == [(F_A, F_BB, F_S)]


def test_enumerate_unknown_scope_raises():
    class BadScope(Toy):
        SCOPE = "bogus"

    with pytest.raises(ValueError, match="unknown SCOPE"):
        BadScope.enumerate(FRAMES)


def test_key_columns():
    assert PerColumnToy.key_columns() == ["df_a", "col_a"]
    assert TripletToy.key_columns() == [
        "df_a",
        "col_a",
        "df_b",
        "col_b",
        "df_c",
        "col_c",
    ]


# ── result() ─────────────────────────────────────────────────────────────────


def test_result_columns_statuses_and_conclusions():
    out = Toy(threshold=0.35).add(FRAMES).result()
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score", "high"]
    assert out.schema["status"] == STATUS
    assert out.rows() == [
        ("f", "a", "f", "bb", "computed", 0.3, False),
        ("f", "a", "f", "s", "ineligible", None, None),
        ("f", "bb", "f", "s", "ineligible", None, None),
        ("g", "a", "g", "ccc", "computed", 0.4, True),
    ]


def test_lazyframes_give_the_same_result():
    lazy = {n: f.lazy() for n, f in FRAMES.items()}
    assert Toy().add(lazy).result().equals(Toy().add(FRAMES).result())


def test_add_is_chainable_and_accumulates():
    t = Toy()
    assert t.add({"f": FRAMES["f"]}) is t
    t.add({"g": FRAMES["g"]})
    assert t.result().height == 4


def test_compatible_hook_marks_pairs_ineligible():
    out = MultiSetToy().add(FRAMES).result()
    statuses = dict(zip(zip(out["col_a"], out["col_b"], out["df_b"]), out["status"]))
    assert statuses[("a", "s", "f")] == "ineligible"  # Int64 vs String
    assert statuses[("a", "a", "g")] == "computed"


def test_descriptors_are_filled_on_every_row():
    class Described(PerColumnToy):
        DESCRIPTORS = {"dtype": pl.String}

        def describe(self, frames, combos):
            return {"dtype": [str(frames[n].schema[c]) for ((n, c),) in combos]}

    out = Described().add(FRAMES).result()
    assert out.columns == ["df_a", "col_a", "status", "dtype", "score", "high"]
    assert out["dtype"].to_list() == ["Int64", "Int64", "String", "Int64", "Int64"]
    assert out.filter(pl.col("col_a") == "s")["status"].item() == "ineligible"


def test_schema_is_hoisted_once_per_frame_not_per_combination():
    """I1: result() must build the dtype lookup once per frame, not rebuild the
    frame's whole Schema for every column of every combination (a real cost: ~70us
    per Schema build at 101 columns, 29s of overhead on ThreewayEntropy)."""

    class CountingFrame(pl.DataFrame):
        count = 0

        @property
        def schema(self):
            CountingFrame.count += 1
            return super().schema

    class Wide(MultiSetToy):
        pass

    n_cols = 30
    frame = CountingFrame({f"c{i}": [1, 2] for i in range(n_cols)})
    combos = Wide.enumerate({"f": frame})
    assert (
        len(combos) == n_cols * (n_cols - 1) // 2
    )  # 435 — old code touched schema ~2x per combo

    Wide().add({"f": frame}).result()
    assert (
        CountingFrame.count <= 3
    ), f"schema was rebuilt {CountingFrame.count} times for {len(combos)} combinations"


def test_compatible_receives_the_actual_column_dtypes():
    seen = []

    class RecordingCompatible(MultiSetToy):
        def compatible(self, dtypes):
            seen.append(list(dtypes))
            return True

    frame = pl.DataFrame(
        {"a": pl.Series([1], dtype=pl.Int32), "b": pl.Series([1], dtype=pl.Int64)}
    )
    RecordingCompatible().add({"f": frame}).result()
    assert seen == [[pl.Int32(), pl.Int64()]]


def test_frame_with_no_columns_gives_empty_result_with_schema():
    out = Toy().add({"e": pl.DataFrame()}).result()
    assert out.height == 0
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score", "high"]


def test_collected_frames_are_available_during_compute():
    seen = {}

    class Peek(Toy):
        def _compute(self, frames, combos):
            seen["same"] = self._collected is frames
            return super()._compute(frames, combos)

    Peek().add({n: f.lazy() for n, f in FRAMES.items()}).result()
    assert seen["same"]


# ── errors ───────────────────────────────────────────────────────────────────


def test_result_before_add_raises():
    with pytest.raises(ValueError, match="before add"):
        Toy().result()


def test_duplicate_frame_name_raises():
    t = Toy().add({"f": FRAMES["f"]})
    with pytest.raises(ValueError, match="already added"):
        t.add({"f": FRAMES["g"]})


@pytest.mark.parametrize("name", ["", 3])
def test_bad_frame_name_raises(name):
    with pytest.raises(ValueError, match="non-empty strings"):
        Toy().add({name: FRAMES["f"]})


def test_non_frame_raises():
    with pytest.raises(
        TypeError, match="DataFrame or LazyFrame, or Arrow tabular data"
    ):
        Toy().add({"f": {"a": [1]}})


def test_non_tabular_arrow_raises():
    with pytest.raises(TypeError, match="not Arrow tabular data"):
        Toy().add({"f": pa.array([1, 2, 3])})


@pytest.mark.parametrize("dtype", [pa.decimal32(5, 2), pa.decimal64(12, 2)])
def test_arrow_input_reads_as_polars_reads_pyarrow(dtype):
    # Checked at the Rust boundary first, then read as py-polars reads a pyarrow
    # Table (its Arrow C stream import aborts on these decimals).
    t = Toy().add({"f": pa.table({"d": pa.array([1, 2], dtype)})})
    frame = t._frames["f"]
    assert frame["d"].dtype == pl.Decimal(dtype.precision, 2)
    assert pa.table(frame).num_rows == 2  # a frame py-polars can export again


def test_arrow_input_polars_cannot_read_is_a_type_error():
    union = pa.UnionArray.from_sparse(
        pa.array([0, 1], pa.int8()), [pa.array([1, 2]), pa.array(["a", "b"])]
    )
    with pytest.raises(TypeError, match="not Arrow tabular data"):
        Toy().add({"f": pa.table({"u": union})})


def test_malformed_arrow_input_is_a_value_error():
    keys = pa.array([0, 1, 5, 1], pa.int32())
    bad = pa.DictionaryArray.from_arrays(keys, pa.array(["x", "y"]), safe=False)
    with pytest.raises(ValueError, match="out of bounds"):
        Toy().add({"f": pa.table({"c": bad})})


def test_wrong_schema_from_compute_raises():
    class Wrong(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, combos).rename({"score": "oops"})

    with pytest.raises(TypeError, match="Wrong._compute returned schema"):
        Wrong().add(FRAMES).result()


def test_missing_row_from_compute_raises():
    class Short(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, combos[:-1])

    with pytest.raises(ValueError, match="rows for"):
        Short().add(FRAMES).result()


def test_row_for_wrong_combination_raises():
    class Swapped(Toy):
        def _compute(self, frames, combos):
            return super()._compute(frames, [(b, a) for a, b in combos])

    with pytest.raises(ValueError, match="exactly one row per eligible combination"):
        Swapped().add(FRAMES).result()


def test_compute_may_not_claim_ineligible():
    class Claims(Toy):
        def _compute(self, frames, combos):
            return self.null_frame(combos, "ineligible")

    with pytest.raises(ValueError, match="'computed' or 'pruned'"):
        Claims().add(FRAMES).result()


def test_pruned_rows_are_allowed_and_null():
    class Prunes(Toy):
        def _compute(self, frames, combos):
            return self.null_frame(combos, "pruned")

    out = Prunes().add(FRAMES).result()
    assert out["status"].to_list() == ["pruned", "ineligible", "ineligible", "pruned"]
    assert out["high"].null_count() == 4


def test_threshold_validation():
    with pytest.raises(ValueError, match="threshold must be in"):
        Toy(threshold=1.5)


def test_instances_are_not_kept_alive():
    t = Toy().add(FRAMES)
    t.result()
    ref = weakref.ref(t)
    del t
    assert ref() is None


# ── helpers ──────────────────────────────────────────────────────────────────


def test_at_least_treats_nan_as_false():
    df = pl.DataFrame({"v": [0.2, 0.3, float("nan")]})
    assert df.select(at_least("v", 0.3)).to_series().to_list() == [False, True, False]


def test_computed_nulls_non_computed_rows():
    df = pl.DataFrame(
        {"status": pl.Series(["computed", "pruned", "ineligible"], dtype=STATUS)}
    )
    assert df.select(computed(pl.lit(True))).to_series().to_list() == [True, None, None]


def test_columns_of_and_group_by_frame():
    combos = [(F_A, F_BB), (F_A, F_S), (G_A, G_CCC)]
    assert columns_of(combos) == [F_A, F_BB, F_S, G_A, G_CCC]
    assert group_by_frame(combos) == {"f": combos[:2], "g": combos[2:]}


def test_rows_from_plugin_attaches_keys_and_status():
    plugin_rows = pl.DataFrame({"col_a": ["a"], "col_b": ["bb"], "score": [0.3]})
    out = Toy.rows_from_plugin("f", plugin_rows)
    assert out.columns == ["df_a", "col_a", "df_b", "col_b", "status", "score"]
    assert out.row(0) == ("f", "a", "f", "bb", "computed", 0.3)


@pytest.mark.parametrize(
    "got, want, rtol, atol, same",
    [
        (None, None, 0, 0, True),
        (None, 1.0, 0, 0, False),
        (float("nan"), float("nan"), 0, 0, True),
        (float("nan"), 1.0, 0, 0, False),
        (1.0, 1.00001, 1e-4, 0, True),
        (1.0, 1.001, 1e-4, 0, False),
        (0.0, 1e-13, 0, 1e-12, True),
        (2**100, 2**100, 0, 0, True),
        (True, False, 0, 0, False),
    ],
)
def test_same_value(got, want, rtol, atol, same):
    assert same_value(got, want, rtol, atol) is same


def test_metric_mismatches_reports_status_and_value_differences():
    a = Toy(threshold=0.35).add(FRAMES).result()
    b = a.with_columns(
        score=pl.when(pl.col("col_b") == "ccc").then(0.5).otherwise(pl.col("score"))
    )
    problems = metric_mismatches(b, a, Toy.key_columns(), ["score"], 0.0, 0.0)
    assert problems == ["g.a ~ g.ccc: score 0.5 != 0.4"]
    assert metric_mismatches(a.head(1), a, Toy.key_columns(), ["score"], 0, 0) == [
        "key columns differ from the reference"
    ]


def test_default_agreement_uses_rtol_atol():
    a = Toy().add(FRAMES).result()
    assert Toy().agreement(a, a) == []


def test_lazy_attributes_imports_on_first_use():
    getter = lazy_attributes("json", {"JSONDecoder": ".decoder"})
    import json.decoder

    assert getter("JSONDecoder") is json.decoder.JSONDecoder
    with pytest.raises(AttributeError):
        getter("Nope")


# ── dtype groupings ──────────────────────────────────────────────────────────


def test_value_family():
    assert value_family(pl.Int32()) == value_family(pl.UInt64()) == "int"
    assert value_family(pl.String()) == value_family(pl.Categorical()) == "str"
    assert value_family(pl.Date()) != value_family(pl.Int32())
    assert value_family(pl.Float32()) != value_family(pl.Float64())
    assert value_family(pl.Int128()) != "int"


def test_encodable_and_nested():
    assert encodable(pl.List(pl.Int32)) and is_nested(pl.Array(pl.Int32, 2))
    assert (
        not encodable(pl.Struct({"a": pl.Int64}))
        and not encodable(pl.Binary())
        and not encodable(pl.Null())
    )
    assert not is_nested(pl.Int64())
