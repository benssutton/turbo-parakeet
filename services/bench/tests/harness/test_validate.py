import pytest

from chbench.harness.suite import ClickHouseAxes, parse_suite
from chbench.harness.validate import expand_grid, resolve_type, table_id
from tests.harness.test_suite import MINIMAL


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
    # Measured on 26.8.2.7 (first runs, 2026-09-17; ruling R27): the server
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
