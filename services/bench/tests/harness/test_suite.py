import textwrap
import pytest

from chbench.harness.inproc_types import type_label
from chbench.harness.suite import (
    InprocAxes,
    QueryKind,
    SuiteError,
    load_suite,
    parse_suite,
)
from chbench.harness.validate import expand_inproc_grid

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
      samples: 3
    results:
      percentiles: [50, 90]
""")


def test_parses_the_minimal_suite():
    s = parse_suite(MINIMAL)
    assert s.name == "t"
    assert s.dataset.rows == 1000
    assert s.dataset.value_range == (-1.0, 10.0)
    assert s.clickhouse.type == ["Float64"]
    assert s.clickhouse.compression == ["LZ4"]
    assert s.queries.kinds == [
        QueryKind(kind="full_scan", target_fraction=None, predicate=None, values=None)
    ]
    assert s.queries.samples == 3
    assert s.results.percentiles == [50, 90]
    assert s.source_yaml == MINIMAL


def test_distribution_block_maps_onto_the_dataclass():
    y = MINIMAL.replace(
        "seed: 1",
        "seed: 1\n  distribution: {kind: lognormal, mean: 4, sigma: 0.25, shift: -1}\n  cardinality_ratio: 0.75",
    )
    s = parse_suite(y)
    assert s.dataset.distribution.kind == "lognormal"
    assert s.dataset.distribution.mean == 4.0
    assert s.dataset.distribution.shift == -1.0
    assert s.dataset.cardinality_ratio == 0.75


def test_filter_kind_carries_its_parameters():
    y = MINIMAL.replace(
        "- {kind: full_scan}",
        "- {kind: filter, target_fraction: 0.1, predicate: range}\n"
        "    - {kind: filter, target_fraction: 0.2, predicate: in, values: 5}",
    )
    s = parse_suite(y)
    assert s.queries.kinds[0] == QueryKind("filter", 0.1, "range", None)
    assert s.queries.kinds[1] == QueryKind("filter", 0.2, "in", 5)


def test_invalid_dataset_fails_at_parse_time():
    y = MINIMAL.replace("null_count: 0", "null_count: 5000")
    with pytest.raises(SuiteError, match="null_count"):
        parse_suite(y)


def test_missing_block_names_the_block():
    y = MINIMAL.replace("queries:", "queriez:")
    with pytest.raises(SuiteError, match="queries"):
        parse_suite(y)


def test_unknown_query_kind_is_rejected():
    y = MINIMAL.replace("kind: full_scan", "kind: everything")
    with pytest.raises(SuiteError, match="kind"):
        parse_suite(y)


def test_filter_requires_target_and_predicate():
    y = MINIMAL.replace("- {kind: full_scan}", "- {kind: filter}")
    with pytest.raises(SuiteError, match="target_fraction"):
        parse_suite(y)


def test_in_predicate_requires_values():
    y = MINIMAL.replace(
        "- {kind: full_scan}",
        "- {kind: filter, target_fraction: 0.1, predicate: in}",
    )
    with pytest.raises(SuiteError, match="values"):
        parse_suite(y)


@pytest.mark.parametrize("key", ["cache_mode", "query_condition_cache"])
def test_retired_query_keys_are_rejected_with_the_reason(key):
    y = MINIMAL.replace("  samples: 3", f"  {key}: [false]\n  samples: 3")
    with pytest.raises(SuiteError, match=rf"queries.{key} was retired \(ToDo 3\)"):
        parse_suite(y)


def test_every_axis_must_be_a_non_empty_list():
    y = MINIMAL.replace("compression: [LZ4]", "compression: []")
    with pytest.raises(SuiteError, match="compression"):
        parse_suite(y)


SUITES = ("float_codecs", "int_encodings", "decimal32_codecs", "replicates")


@pytest.mark.parametrize("name", SUITES)
def test_the_example_suites_load(name):
    s = load_suite(f"suites/{name}.yaml")
    assert s.name == name
    assert s.dataset.rows == 10_000_000


def test_decimal32_has_its_own_suite_at_seven_places():
    s = load_suite("suites/decimal32_codecs.yaml")
    assert s.clickhouse.type == ["Decimal32", "Decimal64"]
    assert s.dataset.decimal_places == 7
    assert "Decimal32" not in load_suite("suites/float_codecs.yaml").clickhouse.type


def test_replicates_are_three_inert_indexes_on_one_table_shape():
    s = load_suite("suites/replicates.yaml")
    assert s.clickhouse.skip_index == ["minmax", "bloom_filter", "set(100)"]
    assert s.clickhouse.codec == ["none"]
    assert s.clickhouse.type == ["Int32"]
    assert load_suite("suites/int_encodings.yaml").clickhouse.skip_index == ["none"]


def test_int_encodings_suite_matches_the_settled_decisions():
    s = load_suite("suites/int_encodings.yaml")
    assert s.dataset.cardinality_ratio == 0.0001  # 1000 distinct at 10M
    assert s.dataset.entropy_bits == 4.0
    assert s.dataset.runs_z == -0.25


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
    y = MINIMAL.replace(
        "- {kind: full_scan}",
        "- {kind: filter, target_fraction: 1.5, predicate: range}",
    )
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


# --- Final fix wave: cross-block checks that used to crash
# later, in `validate` or the renderer, instead of failing at parse time.


def test_decimal_axis_requires_decimal_places():
    # An int32 dataset has no `decimal_places`; `resolve_type` rendered
    # `Decimal64(None)` and `validate` then raised TypeError on `int - None`.
    y = (
        MINIMAL.replace("arrow_type: float64", "arrow_type: int32")
        .replace("value_range: [-1, 10]", "value_range: [-100, 100]")
        .replace("  decimal_places: 8\n", "")
        .replace("type: [Float64]", "type: [Int32, Decimal64]")
    )
    with pytest.raises(
        SuiteError, match=r"clickhouse\.type.*Decimal64.*dataset\.decimal_places"
    ):
        parse_suite(y)


def test_list_dataset_is_rejected():
    # No `type` axis value can hold a `list<T>` column.
    y = MINIMAL.replace(
        "arrow_type: float64", "arrow_type: list<float64>\n  array_len_range: [1, 3]"
    )
    with pytest.raises(
        SuiteError, match=r"dataset\.arrow_type.*list<float64>.*clickhouse\.type"
    ):
        parse_suite(y)


@pytest.mark.parametrize(
    "arrow_type, extra, axis",
    [
        ("string", "\n  string_length: 8", "UInt32"),
        ("date32", "\n  value_range: [0, 1000]", "UInt32"),
        ("string", "\n  string_length: 8", "Float64"),
        ("date32", "\n  value_range: [0, 1000]", "Decimal64"),
    ],
)
def test_numeric_axis_requires_a_numeric_dataset(arrow_type, extra, axis):
    # `validate._build_variant` compares `value_min < 0` for an unsigned
    # axis and `abs(value_min)` for a decimal one; on a string or a date
    # dataset that was a TypeError mid-run.
    y = (
        MINIMAL.replace("arrow_type: float64", f"arrow_type: {arrow_type}{extra}")
        .replace("  value_range: [-1, 10]\n", "")
        .replace("  decimal_places: 8\n", "")
        .replace("type: [Float64]", f"type: [String, {axis}]")
    )
    with pytest.raises(
        SuiteError, match=rf"clickhouse\.type.*{axis}.*numeric.*{arrow_type}"
    ):
        parse_suite(y)


def test_string_axis_on_a_numeric_dataset_stays_legal():
    # That is the int suite: an int32 dataset cast to String by the loader.
    y = (
        MINIMAL.replace("arrow_type: float64", "arrow_type: int32")
        .replace("value_range: [-1, 10]", "value_range: [1, 100]")
        .replace("  decimal_places: 8\n", "")
        .replace("type: [Float64]", "type: [Int32, UInt32, String]")
    )
    assert parse_suite(y).clickhouse.type == ["Int32", "UInt32", "String"]


@pytest.mark.parametrize(
    "value",
    [
        "minmax:4",
        "set(125):2",
        "bloom_filter(0.025):1",
        "bloom_filter(0.001)",
        "bloom_filter:8",
    ],
)
def test_granularity_and_fpr_skip_indexes_are_accepted(value):
    y = MINIMAL.replace("skip_index: [none]", f'skip_index: ["{value}"]')
    assert parse_suite(y).clickhouse.skip_index == [value]


@pytest.mark.parametrize(
    "value",
    [
        "minmax:3",
        "minmax:0",
        "minmax:",
        "none:2",
        "bloom_filter(0.5)",
        "bloom_filter(0.01",
        "set(0):1",
        "minmax:4:2",
        "minmax:²",
        "minmax:٤",
    ],
)
def test_bad_skip_index_grammar_is_rejected(value):
    y = MINIMAL.replace("skip_index: [none]", f'skip_index: ["{value}"]')
    with pytest.raises(SuiteError, match="skip_index"):
        parse_suite(y)


INPROC = MINIMAL.replace(
    "queries:",
    "polars:\n"
    "  type: [Float64, Float32]\n"
    "  storage: [memory, ipc]\n"
    "  compression: [none, ZSTD, LZ4_FRAME]\n"
    "datafusion:\n"
    "  type: [double]\n"
    "  storage: [ipc]\n"
    "  compression: [ZSTD(3)]\n"
    "  threads: [2, 4]\n"
    "queries:",
)


def drop_block(text: str, block: str) -> str:
    """`text` without the top-level `block` (its header line and indented body)."""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith(f"{block}:"):
            skipping = True
            continue
        if skipping and not line.startswith(" "):
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out)


def test_in_process_blocks_parse_with_their_defaults():
    s = parse_suite(INPROC)
    assert s.polars == InprocAxes(
        type=["Float64", "Float32"],
        storage=["memory", "ipc"],
        compression=["none", "ZSTD(1)", "LZ4_FRAME"],
        threads=None,
        ipc_batch_rows=["122880"],
    )
    assert s.datafusion.threads == [2, 4]
    assert s.datafusion.compression == ["ZSTD(3)"]
    assert s.source_yaml == INPROC


def test_ipc_batch_rows_is_an_axis():
    y = INPROC.replace(
        "  threads: [2, 4]\n",
        "  threads: [2, 4]\n  ipc_batch_rows: [whole, 65536, whole]\n",
    )
    assert parse_suite(y).datafusion.ipc_batch_rows == ["whole", "65536"]
    for bad in ("[0]", "[big]", "[]"):
        z = INPROC.replace(
            "  threads: [2, 4]\n", f"  threads: [2, 4]\n  ipc_batch_rows: {bad}\n"
        )
        with pytest.raises(SuiteError, match=r"datafusion\.ipc_batch_rows"):
            parse_suite(z)


def test_a_suite_may_omit_clickhouse():
    s = parse_suite(drop_block(INPROC, "clickhouse"))
    assert s.clickhouse is None
    assert s.polars is not None


def test_a_clickhouse_only_suite_has_no_in_process_blocks():
    s = parse_suite(MINIMAL)
    assert s.polars is None
    assert s.datafusion is None


def test_a_suite_names_at_least_one_engine():
    y = drop_block(MINIMAL, "clickhouse")
    with pytest.raises(SuiteError, match="names no engine"):
        parse_suite(y)


@pytest.mark.parametrize(
    "old, new, match",
    [
        (
            "storage: [memory, ipc]",
            "storage: [memory, ipc]\n  codec: [none]",
            r"polars has unknown key",
        ),
        (
            "  storage: [memory, ipc]\n",
            "",
            r"polars block is missing the 'storage' axis",
        ),
        ("storage: [memory, ipc]", "storage: [disk]", r"polars\.storage"),
        (
            "compression: [none, ZSTD, LZ4_FRAME]",
            "compression: [LZ4]",
            r"polars\.compression",
        ),
        ("type: [Float64, Float32]", "type: [Float16]", r"polars\.type"),
        (
            "type: [Float64, Float32]",
            "type: [Float64, Float64]",
            r"names one type twice",
        ),
        ("threads: [2, 4]", "threads: [0]", r"datafusion\.threads"),
        ("threads: [2, 4]", "threads: ['4']", r"datafusion\.threads"),
        ("type: [double]", "type: [float64]", r"datafusion\.type.*'double'"),
    ],
)
def test_in_process_block_errors_name_the_field(old, new, match):
    y = INPROC.replace(old, new)
    with pytest.raises(SuiteError, match=match):
        parse_suite(y)


def test_two_spellings_of_one_categorical_are_one_type():
    y = INPROC.replace(
        "type: [Float64, Float32]", "type: [Categorical, 'Categorical(UInt32)']"
    )
    with pytest.raises(SuiteError, match="names one type twice"):
        parse_suite(y)


def test_a_type_the_dataset_cannot_feed_is_refused_at_parse_time():
    y = INPROC.replace("type: [Float64, Float32]", "type: ['Categorical(UInt8)']")
    with pytest.raises(
        SuiteError, match=r"polars\.type value 'Categorical\(UInt8\)'.*numeric"
    ):
        parse_suite(y)


@pytest.mark.parametrize(
    "old,new,field,expected",
    [
        (
            "storage: [memory, ipc]",
            "storage: [ipc, ipc, memory, ipc]",
            "storage",
            ["ipc", "memory"],
        ),
        ("storage: [ipc]", "storage: [ipc, ipc]", "storage", ["ipc"]),
        ("threads: [2, 4]", "threads: [4, 2, 4, 2]", "threads", [4, 2]),
    ],
)
def test_storage_and_threads_are_deduplicated_in_order(old, new, field, expected):
    s = parse_suite(INPROC.replace(old, new))
    block = s.polars if field == "storage" and "memory" in expected else s.datafusion
    assert getattr(block, field) == expected


def test_the_inproc_strings_example_parses_and_expands():
    s = load_suite("suites/inproc_strings.yaml")
    assert s.clickhouse is None
    assert s.dataset.rows == 10_000_000
    assert [type_label("polars", t) for t in s.polars.type] == [
        "String",
        "Categorical(UInt8)",
        "Categorical(UInt16)",
        "Categorical(UInt32)",
        "Enum",
    ]
    assert s.datafusion.type[1] == "dictionary<values=string, indices=uint8, ordered=0>"
    assert s.polars.threads is None
    assert s.datafusion.threads is None
    polars = expand_inproc_grid(s, "polars", default_threads=4)
    datafusion = expand_inproc_grid(s, "datafusion", default_threads=4)
    assert (len(polars), sum(v.status == "ok" for v in polars)) == (40, 25)
    assert (len(datafusion), sum(v.status == "ok" for v in datafusion)) == (32, 20)
