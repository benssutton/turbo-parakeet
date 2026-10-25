import textwrap
import pytest

from chbench.harness.suite import (
    ClickHouseAxes, QueryAxes, QueryKind, ResultsConfig, Suite, SuiteError,
    load_suite, parse_suite,
)

MINIMAL = textwrap.dedent("""
    name: t
    dataset:
      rows: 1000
      arrow_type: float64
      value_range: [-1, 10]
      decimal_places: 8
      null_count: 0
      seed: 1
    clickhouse:
      type: [Float64]
      nullable: [false]
      primary_key: [false]
      projection: [false]
      codec: [none]
      skip_index: [none]
      compression: [LZ4]
    queries:
      kinds:
        - {kind: full_scan}
      query_condition_cache: [false]
      cache_mode: [warm]
      samples: 3
    results:
      percentiles: [50, 90]
""")


def test_parses_the_minimal_suite():
    s = parse_suite(MINIMAL)
    assert s.name == "t"
    assert s.dataset.rows == 1000 and s.dataset.value_range == (-1.0, 10.0)
    assert s.clickhouse.type == ["Float64"] and s.clickhouse.compression == ["LZ4"]
    assert s.queries.kinds == [QueryKind(kind="full_scan", target_fraction=None,
                                         predicate=None, values=None)]
    assert s.queries.samples == 3
    assert s.results.percentiles == [50, 90]
    assert s.source_yaml == MINIMAL


def test_distribution_block_maps_onto_the_dataclass():
    y = MINIMAL.replace("seed: 1", "seed: 1\n  distribution: {kind: lognormal, mean: 4, sigma: 0.25, shift: -1}\n  cardinality_ratio: 0.75")
    s = parse_suite(y)
    assert s.dataset.distribution.kind == "lognormal"
    assert s.dataset.distribution.mean == 4.0 and s.dataset.distribution.shift == -1.0
    assert s.dataset.cardinality_ratio == 0.75


def test_filter_kind_carries_its_parameters():
    y = MINIMAL.replace("- {kind: full_scan}",
                        "- {kind: filter, target_fraction: 0.1, predicate: range}\n"
                        "    - {kind: filter, target_fraction: 0.2, predicate: in, values: 5}")
    s = parse_suite(y)
    assert s.queries.kinds[0] == QueryKind("filter", 0.1, "range", None)
    assert s.queries.kinds[1] == QueryKind("filter", 0.2, "in", 5)


def test_invalid_dataset_fails_at_parse_time():
    with pytest.raises(SuiteError, match="null_count"):
        parse_suite(MINIMAL.replace("null_count: 0", "null_count: 5000"))


def test_missing_block_names_the_block():
    with pytest.raises(SuiteError, match="queries"):
        parse_suite(MINIMAL.replace("queries:", "queriez:"))


def test_unknown_query_kind_is_rejected():
    with pytest.raises(SuiteError, match="kind"):
        parse_suite(MINIMAL.replace("kind: full_scan", "kind: everything"))


def test_filter_requires_target_and_predicate():
    with pytest.raises(SuiteError, match="target_fraction"):
        parse_suite(MINIMAL.replace("- {kind: full_scan}", "- {kind: filter}"))


def test_in_predicate_requires_values():
    with pytest.raises(SuiteError, match="values"):
        parse_suite(MINIMAL.replace("- {kind: full_scan}",
                                    "- {kind: filter, target_fraction: 0.1, predicate: in}"))


def test_cache_mode_must_be_warm_or_cold():
    with pytest.raises(SuiteError, match="cache_mode"):
        parse_suite(MINIMAL.replace("cache_mode: [warm]", "cache_mode: [tepid]"))


def test_every_axis_must_be_a_non_empty_list():
    with pytest.raises(SuiteError, match="compression"):
        parse_suite(MINIMAL.replace("compression: [LZ4]", "compression: []"))


@pytest.mark.parametrize("path,rows", [
    ("suites/float_codecs.yaml", 10_000_000), ("suites/int_encodings.yaml", 10_000_000),
])
def test_the_two_example_suites_load(path, rows):
    s = load_suite(path)
    assert s.dataset.rows == rows
    assert len(s.clickhouse.type) == 3


def test_int_encodings_suite_matches_the_settled_decisions():
    s = load_suite("suites/int_encodings.yaml")
    assert s.dataset.cardinality_ratio == 0.0001      # 1000 distinct at 10M
    assert s.dataset.entropy_bits == 4.0 and s.dataset.runs_z == -0.25
    assert s.queries.cache_mode == ["warm", "cold"]


def test_distribution_typo_key_is_rejected():
    y = MINIMAL.replace(
        "seed: 1",
        "seed: 1\n  distribution: {kind: lognormal, mean: 4, sigmaa: 0.25, shift: -1}",
    )
    with pytest.raises(SuiteError, match="sigmaa"):
        parse_suite(y)


def test_value_range_must_be_a_list():
    y = MINIMAL.replace("value_range: [-1, 10]", "value_range: 5")
    with pytest.raises(SuiteError, match="value_range"):
        parse_suite(y)


def test_target_fraction_must_be_between_zero_and_one():
    y = MINIMAL.replace("- {kind: full_scan}",
                        "- {kind: filter, target_fraction: 1.5, predicate: range}")
    with pytest.raises(SuiteError, match="target_fraction"):
        parse_suite(y)


def test_percentile_must_be_between_zero_and_100():
    y = MINIMAL.replace("percentiles: [50, 90]", "percentiles: [50, 190]")
    with pytest.raises(SuiteError, match="percentiles"):
        parse_suite(y)


def test_samples_must_be_a_positive_integer():
    y = MINIMAL.replace("samples: 3", "samples: 0")
    with pytest.raises(SuiteError, match="samples"):
        parse_suite(y)


def test_stray_top_level_key_is_rejected():
    y = MINIMAL + "extra_key: 1\n"
    with pytest.raises(SuiteError, match="extra_key"):
        parse_suite(y)


def test_stray_key_inside_queries_is_rejected():
    y = MINIMAL.replace("samples: 3", "samples: 3\n  bogus: true")
    with pytest.raises(SuiteError, match="bogus"):
        parse_suite(y)


def test_unmeasured_clickhouse_type_is_rejected():
    y = MINIMAL.replace("type: [Float64]", "type: [UUID]")
    with pytest.raises(SuiteError, match="UUID"):
        parse_suite(y)


def test_unknown_codec_is_rejected():
    y = MINIMAL.replace("codec: [none]", "codec: [Zigzag]")
    with pytest.raises(SuiteError, match="Zigzag"):
        parse_suite(y)


def test_unknown_skip_index_is_rejected():
    y = MINIMAL.replace("skip_index: [none]", "skip_index: [bogus_index]")
    with pytest.raises(SuiteError, match="bogus_index"):
        parse_suite(y)


def test_unknown_compression_is_rejected():
    y = MINIMAL.replace("compression: [LZ4]", "compression: [Brotli]")
    with pytest.raises(SuiteError, match="Brotli"):
        parse_suite(y)


def test_set_n_skip_index_is_accepted():
    y = MINIMAL.replace("skip_index: [none]", 'skip_index: ["set(100)"]')
    s = parse_suite(y)
    assert s.clickhouse.skip_index == ["set(100)"]


def test_zstd_level_compression_is_accepted():
    y = MINIMAL.replace("compression: [LZ4]", 'compression: ["ZSTD(3)"]')
    s = parse_suite(y)
    assert s.clickhouse.compression == ["ZSTD(3)"]


# --- Final fix wave, ruling R31(c): cross-block checks that used to crash
# later, in `validate` or the renderer, instead of failing at parse time.


def test_decimal_axis_requires_decimal_places():
    # An int32 dataset has no `decimal_places`; `resolve_type` rendered
    # `Decimal64(None)` and `validate` then raised TypeError on `int - None`.
    y = MINIMAL.replace("arrow_type: float64", "arrow_type: int32") \
               .replace("value_range: [-1, 10]", "value_range: [-100, 100]") \
               .replace("  decimal_places: 8\n", "") \
               .replace("type: [Float64]", "type: [Int32, Decimal64]")
    with pytest.raises(SuiteError, match=r"clickhouse\.type.*Decimal64.*dataset\.decimal_places"):
        parse_suite(y)


def test_list_dataset_is_rejected():
    # No `type` axis value can hold a `list<T>` column.
    y = MINIMAL.replace("arrow_type: float64", "arrow_type: list<float64>\n  array_len_range: [1, 3]")
    with pytest.raises(SuiteError, match=r"dataset\.arrow_type.*list<float64>.*clickhouse\.type"):
        parse_suite(y)


@pytest.mark.parametrize("arrow_type, extra, axis", [
    ("string", "\n  string_length: 8", "UInt32"),
    ("date32", "\n  value_range: [0, 1000]", "UInt32"),
    ("string", "\n  string_length: 8", "Float64"),
    ("date32", "\n  value_range: [0, 1000]", "Decimal64"),
])
def test_numeric_axis_requires_a_numeric_dataset(arrow_type, extra, axis):
    # `validate._build_variant` compares `value_min < 0` for an unsigned
    # axis and `abs(value_min)` for a decimal one; on a string or a date
    # dataset that was a TypeError mid-run.
    y = MINIMAL.replace("arrow_type: float64", f"arrow_type: {arrow_type}{extra}") \
               .replace("  value_range: [-1, 10]\n", "") \
               .replace("  decimal_places: 8\n", "") \
               .replace("type: [Float64]", f"type: [String, {axis}]")
    with pytest.raises(SuiteError, match=rf"clickhouse\.type.*{axis}.*numeric.*{arrow_type}"):
        parse_suite(y)


def test_string_axis_on_a_numeric_dataset_stays_legal():
    # That is the int suite: an int32 dataset cast to String by the loader.
    y = MINIMAL.replace("arrow_type: float64", "arrow_type: int32") \
               .replace("value_range: [-1, 10]", "value_range: [1, 100]") \
               .replace("  decimal_places: 8\n", "") \
               .replace("type: [Float64]", "type: [Int32, UInt32, String]")
    assert parse_suite(y).clickhouse.type == ["Int32", "UInt32", "String"]


def test_both_example_suites_still_parse():
    from chbench.harness.suite import load_suite
    for name in ("float_codecs", "int_encodings"):
        assert load_suite(f"suites/{name}.yaml").name == name


@pytest.mark.parametrize("value", ["minmax:4", "set(125):2", "bloom_filter(0.025):1",
                                   "bloom_filter(0.001)", "bloom_filter:8"])
def test_granularity_and_fpr_skip_indexes_are_accepted(value):
    y = MINIMAL.replace("skip_index: [none]", f'skip_index: ["{value}"]')
    assert parse_suite(y).clickhouse.skip_index == [value]


@pytest.mark.parametrize("value", ["minmax:3", "minmax:0", "minmax:", "none:2", "bloom_filter(0.5)",
                                   "bloom_filter(0.01", "set(0):1", "minmax:4:2", "minmax:²",
                                   "minmax:٤"])
def test_bad_skip_index_grammar_is_rejected(value):
    y = MINIMAL.replace("skip_index: [none]", f'skip_index: ["{value}"]')
    with pytest.raises(SuiteError, match="skip_index"):
        parse_suite(y)
