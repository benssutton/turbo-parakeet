import dataclasses
import hashlib

import pytest

from chbench.harness.suite import ClickHouseAxes, parse_suite
from chbench.harness.validate import (
    TableVariant,
    engine_key,
    expand_grid,
    expand_inproc_grid,
    resolve_type,
    table_id,
)
from tests.harness.test_suite import INPROC, MINIMAL, drop_block


def suite_with(**axes):
    s = parse_suite(MINIMAL)
    ck = {k: list(v) for k, v in vars(s.clickhouse).items()}
    ck.update({k: list(v) for k, v in axes.items()})
    return type(s)(
        name=s.name,
        dataset=s.dataset,
        clickhouse=ClickHouseAxes(**ck),
        queries=s.queries,
        results=s.results,
        source_yaml=s.source_yaml,
    )


def test_cartesian_product_size():
    s = suite_with(
        type=["Float64", "Decimal32"],
        codec=["none", "Delta"],
        compression=["LZ4", "ZSTD"],
    )
    assert (
        len(
            expand_grid(
                s,
                null_count=0,
                value_min=-1.0,
                value_max=9.99999999,
                clickhouse_version="26.8.2.7",
            )
        )
        == 8
    )


def test_resolve_type_uses_decimal_places_for_scale():
    s = parse_suite(MINIMAL)
    assert resolve_type("Decimal32", s.dataset) == "Decimal32(8)"
    assert resolve_type("Decimal64", s.dataset) == "Decimal64(8)"
    assert resolve_type("Float64", s.dataset) == "Float64"
    assert resolve_type("DateTime64", s.dataset) == "DateTime64(3)"


@pytest.mark.parametrize(
    "ch_type,codec,accepted",
    [
        ("Float64", "Gorilla", True),
        ("Int32", "Gorilla", False),
        ("Decimal32", "Gorilla", False),
        ("Float64", "T64", False),
        ("Int32", "T64", True),
        ("Decimal64", "T64", True),
        ("String", "Delta", False),
        ("String", "DoubleDelta", False),
        ("Int32", "Delta", True),
        ("String", "none", True),
        ("String", "LowCardinality", True),
        ("Int32", "LowCardinality", False),
        ("UInt32", "LowCardinality", False),
        ("Float64", "LowCardinality", False),
        ("Decimal64", "LowCardinality", False),
        ("Date", "LowCardinality", False),
    ],
)
def test_codec_applicability_matches_the_measured_matrix(ch_type, codec, accepted):
    s = suite_with(type=[ch_type], codec=[codec])
    [v] = expand_grid(
        s,
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert (v.status == "ok") == accepted


def test_skip_reasons_are_clickhouses_own_messages():
    [t64] = expand_grid(
        suite_with(type=["Float64"], codec=["T64"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert t64.skip_reason == "T64 codec is not supported for specified type Float64."
    [gor] = expand_grid(
        suite_with(type=["Int32"], codec=["Gorilla"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert "Gorilla(4), LZ4 is meaningless" in gor.skip_reason
    [dd] = expand_grid(
        suite_with(type=["String"], codec=["DoubleDelta"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert (
        dd.skip_reason
        == "Codec DoubleDelta is not applicable for String because the data type is not numeric."
    )
    [d] = expand_grid(
        suite_with(type=["String"], codec=["Delta"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert (
        d.skip_reason
        == "Codec Delta is not applicable for String because the data type is not of fixed size."
    )


def test_gorilla_width_in_the_message_follows_the_type():
    for t, n in (("Date", 2), ("Decimal32", 4), ("Decimal64", 8), ("DateTime64", 8)):
        [v] = expand_grid(
            suite_with(type=[t], codec=["Gorilla"]),
            null_count=0,
            value_min=-1.0,
            value_max=9.99999999,
            clickhouse_version="26.8.2.7",
        )
        assert f"Gorilla({n}), LZ4" in v.skip_reason


def test_nullable_keys_off_the_base_type():
    [v] = expand_grid(
        suite_with(type=["Int32"], nullable=[True], codec=["T64"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "ok"
    [g] = expand_grid(
        suite_with(type=["Int32"], nullable=[True], codec=["Gorilla"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert g.status == "skipped" and "Gorilla(4)" in g.skip_reason


@pytest.mark.parametrize(
    "ch_type,index,accepted",
    [
        ("Int32", "bloom_filter", True),
        ("Decimal32", "bloom_filter", False),
        ("Decimal64", "bloom_filter", False),
        ("Decimal32", "minmax", True),
        ("String", "set(100)", True),
    ],
)
def test_skip_index_applicability(ch_type, index, accepted):
    [v] = expand_grid(
        suite_with(type=[ch_type], skip_index=[index]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert (v.status == "ok") == accepted
    if not accepted:
        assert "bloom filter index" in v.skip_reason and "Decimal(" in v.skip_reason


def test_nulls_in_data_require_nullable():
    [v] = expand_grid(
        suite_with(type=["Float64"], nullable=[False]),
        null_count=5,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "skipped" and "nullable" in v.skip_reason.lower()


def test_unsigned_type_requires_non_negative_min():
    [v] = expand_grid(
        suite_with(type=["UInt32"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "skipped" and "unsigned" in v.skip_reason.lower()


def test_projection_with_primary_key_warns_redundant_but_runs():
    [v] = expand_grid(
        suite_with(primary_key=[True], projection=[True]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "ok"
    assert any("redundant" in w for w in v.warnings)


def test_table_id_is_stable_and_includes_the_server_version():
    s = suite_with()
    [a] = expand_grid(
        s,
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    [b] = expand_grid(
        s,
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    [c] = expand_grid(
        s,
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.9.0.1",
    )
    assert table_id(a, "d", "26.8.2.7") == table_id(b, "d", "26.8.2.7")
    assert table_id(a, "d", "26.8.2.7") != table_id(c, "d", "26.9.0.1")
    assert len(table_id(a, "d", "26.8.2.7")) == 16


def test_low_cardinality_wraps_nullable_not_the_reverse():
    # Nullable(LowCardinality(X)) is rejected by ClickHouse (code 43); only
    # LowCardinality(Nullable(X)) is legal. Nullable must wrap first.
    [v] = expand_grid(
        suite_with(type=["String"], nullable=[True], codec=["LowCardinality"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.ch_type == "LowCardinality(Nullable(String))"


def test_unmeasured_type_is_a_hard_error_not_a_silent_string_bucket():
    # A type outside the measured vocabulary must never fall through to the
    # String applicability branch and be skipped with a fabricated reason.
    s = suite_with(type=["UUID"])
    with pytest.raises(ValueError, match="measured"):
        expand_grid(
            s,
            null_count=0,
            value_min=-1.0,
            value_max=9.99999999,
            clickhouse_version="26.8.2.7",
        )


def test_gorilla_message_names_the_actual_compression_not_a_hardcoded_lz4():
    [v] = expand_grid(
        suite_with(type=["Int32"], codec=["Gorilla"], compression=["ZSTD"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert "Gorilla(4), ZSTD is meaningless" in v.skip_reason
    [lz4] = expand_grid(
        suite_with(type=["Int32"], codec=["Gorilla"], compression=["LZ4"]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert "Gorilla(4), LZ4 is meaningless" in lz4.skip_reason


def test_decimal_range_fits_use_case_1_for_both_widths():
    for t in ("Decimal32", "Decimal64"):
        [v] = expand_grid(
            suite_with(type=[t]),
            null_count=0,
            value_min=-1.0,
            value_max=9.99999999,
            clickhouse_version="26.8.2.7",
        )
        assert v.status == "ok"


def test_decimal_range_overflow_skips_decimal32_but_not_decimal64():
    [d32] = expand_grid(
        suite_with(type=["Decimal32"]),
        null_count=0,
        value_min=-1.0,
        value_max=10.0,
        clickhouse_version="26.8.2.7",
    )
    assert d32.status == "skipped" and "Decimal32" in d32.skip_reason
    [d64] = expand_grid(
        suite_with(type=["Decimal64"]),
        null_count=0,
        value_min=-1.0,
        value_max=10.0,
        clickhouse_version="26.8.2.7",
    )
    assert d64.status == "ok"


def test_nullable_primary_key_is_a_measured_skip():
    [v] = expand_grid(
        suite_with(nullable=[True], primary_key=[True]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "skipped"
    assert v.skip_reason == (
        "Sorting key contains nullable columns, but merge tree setting "
        "`allow_nullable_key` is disabled"
    )


def test_nullable_without_primary_key_stays_ok():
    [v] = expand_grid(
        suite_with(nullable=[True], primary_key=[False], projection=[True]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert v.status == "ok"


def _one(**axes):
    # value_min 0.0 so the unsigned-minimum rule, which runs first, does not
    # pre-empt the codec rule under test for UInt32.
    [v] = expand_grid(
        suite_with(**axes),
        null_count=0,
        value_min=0.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    return v


@pytest.mark.parametrize("ch_type", ["Int32", "UInt32"])
def test_low_cardinality_on_integers_is_a_measured_skip(ch_type):
    # Measured on 26.8.2.7 (first runs, 2026-09-17): the server
    # refuses the DDL with code 455 SUSPICIOUS_TYPE_FOR_LOW_CARDINALITY. The
    # harness records the skip; it does not enable the setting.
    v = _one(type=[ch_type], codec=["LowCardinality"])
    assert v.status == "skipped"
    assert v.skip_reason == (
        f"Creating columns of type LowCardinality({ch_type}) is prohibited by "
        "default due to expected negative impact on performance. It can be "
        "enabled with the `allow_suspicious_low_cardinality_types` setting."
    )


def test_low_cardinality_skip_message_names_the_nullable_wrapper():
    v = _one(type=["Int32"], nullable=[True], codec=["LowCardinality"])
    assert v.status == "skipped"
    assert v.skip_reason.startswith(
        "Creating columns of type LowCardinality(Nullable(Int32)) is prohibited"
    )


def test_low_cardinality_on_decimal_and_datetime64_is_the_code_43_skip():
    # A different rejection (code 43 ILLEGAL_TYPE_OF_ARGUMENT) naming the
    # resolved Decimal(P, S) spelling, measured the same day.
    v = _one(type=["Decimal64"], codec=["LowCardinality"])
    assert v.skip_reason == (
        "DataTypeLowCardinality is supported only for numbers, strings, Date "
        "or DateTime, but got Decimal(18, 8)."
    )
    v = _one(type=["DateTime64"], nullable=[True], codec=["LowCardinality"])
    assert v.skip_reason == (
        "DataTypeLowCardinality is supported only for numbers, strings, Date "
        "or DateTime, but got Nullable(DateTime64(3))."
    )


def test_low_cardinality_on_string_stays_ok_nullable_or_not():
    assert _one(type=["String"], codec=["LowCardinality"]).status == "ok"
    v = _one(type=["String"], nullable=[True], codec=["LowCardinality"])
    assert v.status == "ok"
    assert v.ch_type == "LowCardinality(Nullable(String))"


@pytest.mark.parametrize(
    "index,accepted",
    [
        ("bloom_filter(0.01):4", False),
        ("bloom_filter(0.05)", False),
        ("minmax:2", True),
        ("set(125):8", True),
    ],
)
def test_applicability_ignores_fpr_and_granularity(index, accepted):
    [v] = expand_grid(
        suite_with(type=["Decimal32"], skip_index=[index]),
        null_count=0,
        value_min=-1.0,
        value_max=9.99999999,
        clickhouse_version="26.8.2.7",
    )
    assert (v.status == "ok") == accepted


def test_in_process_grid_crosses_type_storage_compression_threads():
    s = parse_suite(INPROC)
    v = expand_inproc_grid(s, "polars", default_threads=8)
    assert len(v) == 2 * 2 * 3
    assert {x.engine for x in v} == {"polars"} and {x.threads for x in v} == {8}
    mem = {(x.compression, x.status) for x in v if x.storage == "memory"}
    assert mem == {("none", "ok"), ("ZSTD(1)", "skipped"), ("LZ4_FRAME", "skipped")}
    assert all("ipc only" in x.skip_reason for x in v if x.status == "skipped")
    assert all(x.status == "ok" for x in v if x.storage == "ipc")
    assert all(
        x.nullable is None and x.codec is None and x.skip_index is None for x in v
    )
    assert [x.ch_type for x in v[:1]] == ["Float64"] and v[0].type_spec == "Float64"
    d = expand_inproc_grid(s, "datafusion", default_threads=8)
    assert [x.threads for x in d] == [2, 4] and d[0].ch_type == "double"


def test_no_block_no_variants():
    s = parse_suite(MINIMAL)
    assert expand_inproc_grid(s, "polars", default_threads=4) == []
    no_ch = parse_suite(drop_block(INPROC, "clickhouse"))
    assert (
        expand_grid(
            no_ch, null_count=0, value_min=0, value_max=1, clickhouse_version="x"
        )
        == []
    )


def test_engine_key():
    s = parse_suite(INPROC)
    [ch] = expand_grid(
        s, null_count=0, value_min=-1.0, value_max=9.0, clickhouse_version="v"
    )
    assert engine_key(ch) == "clickhouse"
    assert {
        engine_key(x) for x in expand_inproc_grid(s, "datafusion", default_threads=1)
    } == {
        "datafusion:2",
        "datafusion:4",
    }


def test_the_id_of_a_clickhouse_variant_is_unchanged():
    v = TableVariant(
        ch_type="Float64",
        nullable=False,
        primary_key=False,
        projection=False,
        codec="none",
        skip_index="none",
        compression="LZ4",
        status="ok",
        skip_reason=None,
        warnings=(),
    )
    key = "\x1f".join(
        ["Float64", "False", "False", "False", "none", "none", "LZ4", "d1", "26.8.31.2"]
    )
    assert (
        table_id(v, "d1", "26.8.31.2") == hashlib.sha256(key.encode()).hexdigest()[:16]
    )
    # Literal id computed with table_id as of 01c4ca0 (before engines existed).
    assert table_id(v, "d1", "26.8.31.2") == "4780922b48788bf5"


def test_in_process_ids_hash_engine_type_storage_compression_threads_version():
    grid = expand_inproc_grid(parse_suite(INPROC), "polars", default_threads=4)
    assert len({table_id(v, "d1", "2.0.0") for v in grid}) == len(grid)
    v = next(x for x in grid if x.storage == "ipc")
    same = table_id(v, "d1", "2.0.0")
    assert same == table_id(v, "d1", "2.0.0")
    assert same != table_id(dataclasses.replace(v, threads=2), "d1", "2.0.0")
    assert same != table_id(v, "d1", "2.0.1")
    assert same != table_id(dataclasses.replace(v, engine="datafusion"), "d1", "2.0.0")
    assert same != table_id(
        dataclasses.replace(v, type_spec="Enum(categories=['a'])"), "d1", "2.0.0"
    )
    assert same != table_id(
        dataclasses.replace(v, ipc_batch_rows="65536"), "d1", "2.0.0"
    )
    # A memory cell's id has no batch in it: the axis does not apply to it.
    mem = grid[0]
    assert mem.storage == "memory" and mem.ipc_batch_rows is None
    key = "\x1f".join(
        ["polars", "Float64", "Float64", "memory", "none", "4", "d1", "2.0.0"]
    )
    assert table_id(mem, "d1", "2.0.0") == hashlib.sha256(key.encode()).hexdigest()[:16]


def test_ipc_batch_rows_crosses_ipc_cells_only():
    # The default is Polars' own chunking, 122 880 rows per record batch.
    d = expand_inproc_grid(parse_suite(INPROC), "datafusion", default_threads=8)
    assert {x.ipc_batch_rows for x in d} == {"122880"}
    y = INPROC.replace(
        "  threads: [2, 4]\n", "  threads: [2]\n  ipc_batch_rows: [whole, 1000]\n"
    )
    d = expand_inproc_grid(parse_suite(y), "datafusion", default_threads=8)
    assert [(x.storage, x.ipc_batch_rows, x.status) for x in d] == [
        ("ipc", "whole", "ok"),
        ("ipc", "1000", "ok"),
    ]
    m = parse_suite(
        y.replace("  storage: [ipc]\n", "  storage: [memory, ipc]\n", 1).replace(
            "  compression: [ZSTD(3)]\n", "  compression: [none]\n", 1
        )
    )
    cells = expand_inproc_grid(m, "datafusion", default_threads=8)
    assert [(x.storage, x.ipc_batch_rows, x.status) for x in cells] == [
        ("memory", None, "ok"),
        ("ipc", "whole", "ok"),
        ("ipc", "1000", "ok"),
    ]
    assert all(
        x.ipc_batch_rows is None
        for x in expand_grid(
            m, null_count=0, value_min=-1.0, value_max=9.0, clickhouse_version="v"
        )
    )
