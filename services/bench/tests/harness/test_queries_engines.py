"""The engine-neutral half of a filter (`ValueChoice`) and the per-engine renderers. The ClickHouse
output is pinned by test_queries.py, unchanged."""

import dataclasses
import decimal
import itertools

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datafusion import SessionConfig, SessionContext

from chbench.harness.inproc_types import arrow_family, parse_datafusion_type
from chbench.harness.queries import (
    Reference,
    ValueChoice,
    choice_values_arrow,
    choose_in,
    choose_queries,
    choose_range,
    datafusion_sql,
    datafusion_table,
    expand_queries,
    in_predicate,
    query_values,
    range_predicate,
    reference_result,
    render_clickhouse,
    render_datafusion,
    render_polars_text,
    render_predicate,
    render_queries,
    results_agree,
)
from chbench.harness.suite import QueryAxes, QueryKind
from tests.harness.test_queries import variant

AXES = QueryAxes(
    kinds=[
        QueryKind("full_scan", None, None, None),
        QueryKind("filter", 0.1, "range", None),
        QueryKind("filter", 0.05, "in", 2),
    ],
    samples=1,
)


def _vals():
    return np.round(np.random.default_rng(0).uniform(-1, 10, 10_000), 2)


def test_one_choice_renders_the_old_clickhouse_predicates():
    vals = _vals()
    c = choose_range(vals, 0.1, np.random.default_rng(3))
    assert c.predicate_kind == "range" and len(c.values) == 2
    assert range_predicate(vals, 0.1, np.random.default_rng(3), ch_type="Float64") == (
        render_clickhouse(c, "Float64"),
        c.actual_fraction,
    )
    i = choose_in(vals, 0.01, 3)
    assert i.predicate_kind == "in" and len(i.values) == 3
    assert in_predicate(vals, 0.01, 3) == (render_clickhouse(i), i.actual_fraction)


def test_choose_queries_is_what_expand_queries_renders():
    vals = _vals()
    choices = choose_queries(AXES, vals, np.random.default_rng(7))
    assert choices[0] is None
    assert [c.predicate_kind for c in choices[1:]] == ["range", "in"]
    assert render_queries(AXES, choices, variant()) == expand_queries(
        AXES, vals, variant(), np.random.default_rng(7)
    )


def test_a_value_choice_is_frozen():
    c = ValueChoice("in", np.array([1.0]), 0.5)
    with pytest.raises(AttributeError):
        c.actual_fraction = 0.1


_TABLES = itertools.count()
D = decimal.Decimal
# (dataset column, DataFusion column types it feeds); repeated values so an IN choice exists.
DF_CASES = [
    (pa.array([1, 2, 2, None, 3] * 20, pa.int64()), ["int16", "uint64", "double"]),
    (pa.array([0.5, 1.25, 1.25, None, -3.0] * 20), ["double", "float"]),
    (
        pa.array(
            [D("0.50"), D("1.25"), D("1.25"), None, D("-3.00")] * 20,
            pa.decimal128(10, 2),
        ),
        ["decimal32(5, 2)", "decimal128(18, 2)", "double"],
    ),
    (pa.array([True, False, True, None] * 20), ["bool"]),
    (
        pa.array(["a", "b'q", "b'q", None, "ccc"] * 20),
        [
            "string",
            "large_string",
            "string_view",
            "binary",
            "dictionary<values=string, indices=int16, ordered=0>",
            "dictionary<values=large_string, indices=uint8, ordered=0>",
        ],
    ),
    (
        pa.array([b"\x00\xff", b"ab", b"ab", None] * 20),
        [
            "binary",
            "large_binary",
            "dictionary<values=binary, indices=int32, ordered=0>",
        ],
    ),
    (
        pa.array([b"\x00\xff", b"ab", b"ab", None] * 20, pa.binary(2)),
        ["fixed_size_binary[2]"],
    ),
    (pa.array([0, 1, 1, None, 400] * 20, pa.date32()), ["date32[day]", "timestamp[s]"]),
    (
        pa.array(
            [0, 1_500_000, 1_500_000, None, 9_000_000] * 20,
            pa.timestamp("us", tz="Europe/London"),
        ),
        ["timestamp[us, tz=Europe/London]", "timestamp[ns, tz=UTC]"],
    ),
    (
        pa.array([0, 1_500, 1_500, None, 90_000] * 20, pa.timestamp("ms")),
        ["timestamp[ms]", "timestamp[us]"],
    ),
    (
        pa.array([0, 1_500, 1_500, None, 90_000] * 20, pa.time32("ms")),
        ["time32[ms]", "time64[us]"],
    ),
    (
        pa.array([0, 1_500_000_001, 1_500_000_001, None, 3] * 20, pa.time64("ns")),
        ["time64[ns]"],
    ),
    (pa.array([1, 1, 2, None] * 20, pa.time32("s")), ["time32[s]"]),
]
_IDS = [f"{a.type}->{t}" for a, ts in DF_CASES for t in ts]


@pytest.fixture(scope="module")
def ctx():
    return SessionContext(SessionConfig().with_target_partitions(2))


@pytest.mark.parametrize(
    "array, name", [(a, t) for a, ts in DF_CASES for t in ts], ids=_IDS
)
def test_datafusion_sql_answers_the_pyarrow_reference(ctx, array, name):
    column_type = parse_datafusion_type(name)
    table = datafusion_table(f"x{next(_TABLES)}")
    ctx.register_record_batches(
        table, [[pa.record_batch({"v": pc.cast(array, column_type)})]]
    )
    values = query_values(array)
    family = arrow_family(column_type)
    choices = [None, choose_in(values, 0.4, 1)]
    if family not in ("boolean", "binary"):
        choices.append(choose_range(values, 0.5, np.random.default_rng(1)))
    for choice in choices:
        arrow_values = (
            None if choice is None else choice_values_arrow(choice, array.type)
        )
        predicate = (
            None
            if choice is None
            else render_datafusion(choice.predicate_kind, arrow_values, column_type)
        )
        got = ctx.sql(datafusion_sql(table, column_type, predicate)).to_arrow_table()
        ref = reference_result(
            array,
            None if choice is None else choice.predicate_kind,
            arrow_values,
            family,
        )
        count, total = got.column(0)[0].as_py(), got.column(1)[0].as_py()
        assert results_agree(ref, count, total), (predicate, count, total, ref)


def test_a_quote_in_a_string_literal_is_doubled():
    v = pa.array(["b'q"])
    assert render_datafusion("in", v, pa.string()) == "v IN ('b''q')"


def test_datafusion_literals_are_typed():
    assert render_datafusion(
        "in", pa.array([D("1.25")], pa.decimal128(10, 2)), pa.decimal32(5, 2)
    ) == ("v IN (CAST('1.25' AS DECIMAL(5, 2)))")
    assert (
        render_datafusion("in", pa.array([b"\x00\xff"]), pa.binary())
        == "v IN (X'00ff')"
    )
    assert render_datafusion("range", pa.array([0, 1], pa.date32()), pa.date32()) == (
        "v BETWEEN DATE '1970-01-01' AND DATE '1970-01-02'"
    )
    assert render_datafusion(
        "in",
        pa.array([1_500_000], pa.timestamp("us", tz="UTC")),
        pa.timestamp("us", tz="UTC"),
    ) == (
        "v IN (arrow_cast('1970-01-01T00:00:01.500000Z', 'Timestamp(Microsecond, Some(\"UTC\"))'))"
    )
    assert render_datafusion("in", pa.array([1.5], pa.float32()), pa.float32()) == (
        "v IN (CAST(1.5 AS FLOAT))"
    )


def test_polars_text_names_the_values():
    c = ValueChoice("in", np.array(["a", "b'c"], dtype=object), 0.2)
    assert render_polars_text("in", choice_values_arrow(c, pa.string())) == (
        'col("v").is_in([\'a\', "b\'c"])'
    )
    r = ValueChoice("range", np.array([1.5, 2.5]), 0.5)
    assert render_polars_text("range", choice_values_arrow(r, pa.float64())) == (
        'col("v").is_between(1.5, 2.5)'
    )
    s = ValueChoice("range", np.array(["a", "c"], dtype=object), 0.5)
    assert render_polars_text("range", choice_values_arrow(s, pa.string())) == (
        "col(\"v\").is_between(lit('a'), lit('c'))"
    )


def test_choice_values_come_back_in_the_dataset_type():
    for array in (
        pa.array([0, 1500, None, 1500], pa.time32("ms")),
        pa.array([1, 2, None, 2], pa.int32()),
        pa.array([2**40, None, 2**40], pa.int64()),
        pa.array([0, 5000, None, 5000], pa.timestamp("ns", tz="Europe/London")),
        pa.array([D("1.25"), None, D("1.25")], pa.decimal32(5, 2)),
    ):
        choice = choose_in(query_values(array), 0.5, 1)
        out = choice_values_arrow(choice, array.type)
        assert out.type == array.type and out[0].as_py() in array.to_pylist()


def test_reference_counts_and_totals():
    a = pa.array([1.0, 2.0, None, 2.0])
    assert reference_result(a, None, None) == Reference(4, 5.0, 5.0)
    assert reference_result(a, "in", pa.array([2.0])) == Reference(2, 4.0, 4.0)
    assert reference_result(a, "range", pa.array([1.0, 1.5])) == Reference(1, 1.0, 1.0)
    assert reference_result(pa.array(["ab", None, "c"]), None, None) == Reference(
        3, 3, 3.0
    )
    ts = pa.array([1_500_000, None], pa.timestamp("us", tz="UTC"))
    assert reference_result(ts, None, None).total == 1.5
    assert reference_result(pa.array([2, None], pa.date32()), None, None).total == 2.0
    assert reference_result(
        pa.array([2, None], pa.date32()), None, None, "timestamp"
    ).total == (2 * 86_400.0)
    t = pa.array([2_000_000_000], pa.time64("ns"))
    assert reference_result(t, None, None).total == 2.0
    dec = pa.array([D("1.25"), D("1.25")], pa.decimal32(5, 2))
    assert (
        reference_result(dec, "in", pa.array([D("1.25")], pa.decimal32(5, 2))).total
        == 2.5
    )
    assert (
        reference_result(pa.array([b"ab", None], pa.binary(2)), None, None).total == 2
    )
    assert reference_result(pa.array([], pa.float64()), None, None) == Reference(
        0, 0.0, 0.0
    )
    assert reference_result(pa.array([1, 22]), None, None, "string") == Reference(
        2, None, 0.0
    )


def test_results_agree():
    r = Reference(3, 10.0, 10.0)
    assert results_agree(r, 3, 10.0 + 1e-12)
    assert not results_agree(r, 4, 10.0) and not results_agree(r, 3, 10.1)
    assert results_agree(Reference(2, 0.0, 4.0), 2, 1e-15)
    assert results_agree(Reference(0, 0.0, 0.0), 0, None)
    assert results_agree(Reference(2, 7, 7.0), 2, 7) and not results_agree(
        Reference(2, 7, 7.0), 2, 8
    )
    assert results_agree(Reference(2, None, 0.0), 2, 123) and not results_agree(
        Reference(2, None, 0.0), 1, 123
    )


BIG_INTS = [
    pa.array([2**64 - 1, 2**64 - 2, 2**64 - 2, None, 2**53 + 1] * 20, pa.uint64()),
    pa.array([2**62 + 1, -(2**62) - 3, -(2**62) - 3, None, 7] * 20, pa.int64()),
    pa.array(
        [1_700_000_000_123_456_789, 3, 3, None, -1_700_000_000_987_654_321] * 20,
        pa.timestamp("ns"),
    ),
]


def test_reference_totals_integers_beyond_two_to_the_53():
    """Hash / id columns and nanosecond timestamps do not fit a double exactly: the reference
    rounds them as the engines' own CAST(v AS DOUBLE) does, rather than refusing."""
    assert reference_result(BIG_INTS[0][:2], None, None).total == pytest.approx(
        2.0 * 2**64
    )
    assert reference_result(BIG_INTS[1][:1], None, None).total == float(2**62)
    assert reference_result(BIG_INTS[2][:1], None, None).total == pytest.approx(
        1_700_000_000.123_456_789
    )


@pytest.mark.parametrize("array", BIG_INTS, ids=["uint64", "int64", "timestamp_ns"])
def test_datafusion_agrees_on_integers_beyond_two_to_the_53(ctx, array):
    """DataFusion's lossy CAST(v AS DOUBLE) and pyarrow's sum in a different order: the relative
    tolerance absorbs the rounding."""
    column_type = array.type
    table = datafusion_table(f"x{next(_TABLES)}")
    ctx.register_record_batches(table, [[pa.record_batch({"v": array})]])
    values = query_values(array)
    for choice in (None, choose_in(values, 0.4, 1)):
        arrow_values = (
            None if choice is None else choice_values_arrow(choice, array.type)
        )
        predicate = (
            None
            if choice is None
            else render_datafusion(choice.predicate_kind, arrow_values, column_type)
        )
        got = ctx.sql(datafusion_sql(table, column_type, predicate)).to_arrow_table()
        ref = reference_result(
            array, None if choice is None else choice.predicate_kind, arrow_values
        )
        count, total = got.column(0)[0].as_py(), got.column(1)[0].as_py()
        assert results_agree(ref, count, total), (predicate, count, total, ref)


def test_non_finite_totals_agree_with_the_same_non_finite_result():
    nan, inf = float("nan"), float("inf")
    for values in ([nan, 1.0, inf], [1e308, 1e308], [-1e308, -1e308], [inf, 2.0]):
        ref = reference_result(pa.array(values), None, None)
        assert not np.isfinite(ref.total)
        assert results_agree(ref, len(values), float(sum(values)))
    ref_nan = reference_result(pa.array([nan, nan, 1.0]), "in", pa.array([nan]))
    assert results_agree(ref_nan, ref_nan.count, nan)
    # a finite result disagrees
    assert not results_agree(Reference(1, inf, inf), 1, 1e308)
    assert not results_agree(Reference(1, nan, nan), 1, 0.0)
    assert not results_agree(Reference(1, nan, nan), 2, nan)
    # a finite reference never accepts a non-finite result
    assert not results_agree(Reference(1, 1.0, 1.0), 1, nan)
    assert not results_agree(Reference(1, 1.0, 1.0), 1, inf)


def test_a_non_finite_reference_accepts_any_non_finite_result():
    """With overflow or both infinities in the data, summation order alone decides between
    inf, -inf and nan (inf + -inf = nan; 1e308 + 1e308 + -inf = nan or -inf; 1e308, 1e308,
    -1e308, -1e308 = inf or -inf), so a non-finite reference accepts any non-finite result.
    """
    nan, inf = float("nan"), float("inf")
    ref = reference_result(pa.array([inf, -inf]), None, None)
    assert np.isnan(ref.total)
    assert results_agree(ref, 2, inf) and results_agree(ref, 2, -inf)
    assert results_agree(Reference(3, inf, inf), 3, nan)
    assert results_agree(Reference(3, -inf, inf), 3, nan)
    assert results_agree(Reference(4, inf, inf), 4, -inf)
    assert not results_agree(ref, 2, 0.0)


def test_an_integer_total_is_not_matched_by_truncation():
    r = Reference(2, 7, 7.0)
    assert results_agree(r, 2, 7) and results_agree(r, 2, 7.0)
    assert not results_agree(r, 2, 7.9) and not results_agree(r, 2, 6.99)
    assert not results_agree(r, 2, float("nan")) and not results_agree(
        r, 2, float("inf")
    )
    assert results_agree(Reference(0, 0, 0.0), 0, None)


def test_render_predicate_refuses_an_unknown_engine():
    v = dataclasses.replace(variant(), engine="sqlite")
    with pytest.raises(ValueError, match="sqlite"):
        render_predicate(ValueChoice("in", np.array([1.0]), 0.5), v)


def test_render_clickhouse_exact_sql():
    dec = np.array(
        [
            D("1.25"),
            D("1.25"),
            D("3.5"),
            D("0.1"),
            D("7"),
            D("-2.5"),
            D("9.0001"),
            D("4"),
        ],
        dtype=object,
    )
    text = np.array(["a", "b'q", "b'q", "ccc", "d", "e", "gg", "h"], dtype=object)
    for ch_type, vals, rng_sql, in_sql in [
        (
            "Decimal64(4)",
            dec,
            "v BETWEEN '1.2500' AND '4.0000'",
            "v IN ('1.2500', '-2.5000')",
        ),
        ("String", text, r"v BETWEEN 'b\'q' AND 'e'", r"v IN ('b\'q', 'a')"),
    ]:
        r = choose_range(vals, 0.5, np.random.default_rng(1))
        assert render_clickhouse(r, ch_type) == rng_sql
        assert render_clickhouse(choose_in(vals, 0.4, 2), ch_type) == in_sql


@pytest.mark.parametrize(
    "array",
    [
        pa.array([1, 2, 2, None, 3] * 20, pa.int64()),
        pa.array([0.5, 1.25, 1.25, None, -3.0] * 20),
        pa.array(["a", "b'q", "b'q", None, "ccc"] * 20),
        pa.array([b"\x00\xff", b"ab", b"ab", None] * 20),
    ],
    ids=str,
)
def test_polars_text_evaluates_to_the_reference_count(array):
    """The stored text is a Polars expression: evaluated on a frame, its count is the reference's."""
    pl = pytest.importorskip("polars")
    values = query_values(array)
    frame = pl.from_arrow(pa.table({"v": array}))
    choices = [choose_in(values, 0.4, 1)]
    if not pa.types.is_binary(array.type):
        choices.append(choose_range(values, 0.5, np.random.default_rng(1)))
    for choice in choices:
        arrow_values = choice_values_arrow(choice, array.type)
        text = render_polars_text(choice.predicate_kind, arrow_values)
        got = frame.filter(
            eval(text, {"col": pl.col, "lit": pl.lit})
        ).height  # noqa: S307
        ref = reference_result(array, choice.predicate_kind, arrow_values)
        assert got == ref.count, text
