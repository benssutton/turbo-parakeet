"""A suite's `dataset: {path, column}` (spec section 5.1, ruling R35)."""
import decimal

import pyarrow as pa
import pytest

from chbench.harness.suite import SuiteError, load_suite, parse_suite

SUITE = """
name: real
dataset:
  {dataset}
clickhouse:
  type: [{axis}]
  nullable: [false]
  primary_key: [false]
  projection: [false]
  codec: [none]
  skip_index: [none]
  compression: [LZ4]
queries:
  kinds:
    - {{kind: full_scan}}
  query_condition_cache: [false]
  cache_mode: [warm]
  samples: 3
results:
  percentiles: [50]
"""


def write(directory, array, name="v", file="x.arrow"):
    p = directory / file
    batch = pa.record_batch([array], names=[name])
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    return p


def suite_file(directory, dataset="{path: x.arrow, column: v}", axis="Int32"):
    p = directory / "s.yaml"
    p.write_text(SUITE.format(dataset=dataset, axis=axis), encoding="utf-8")
    return p


def test_path_resolves_relative_to_the_suite_file_and_fields_are_measured(tmp_path, monkeypatch):
    sub = tmp_path / "suites"
    sub.mkdir()
    write(sub, pa.array([5, None, 2, 9], pa.int32()))
    monkeypatch.chdir(tmp_path)                       # not the suite's directory
    d = load_suite(suite_file(sub)).dataset
    assert d.path == str(sub / "x.arrow") and d.column == "v"
    assert (d.rows, d.arrow_type, d.null_count, d.seed) == (4, "int32", 1, 0)
    assert d.value_range == (2.0, 9.0) and d.decimal_places is None


def test_a_seed_may_be_given(tmp_path):
    write(tmp_path, pa.array([1, 2], pa.int32()))
    s = load_suite(suite_file(tmp_path, "{path: x.arrow, column: v, seed: 7}"))
    assert s.dataset.seed == 7


def test_decimal_scale_is_measured_and_explicit_decimal_places_apply_to_a_float(tmp_path):
    write(tmp_path, pa.array([decimal.Decimal("1.50")], pa.decimal128(9, 2)))
    d = load_suite(suite_file(tmp_path, axis="Decimal64")).dataset
    assert d.decimal_places == 2 and d.arrow_type == "decimal128(9,2)"
    write(tmp_path, pa.array([1.5, 2.25], pa.float64()))
    d = load_suite(suite_file(tmp_path, "{path: x.arrow, column: v, decimal_places: 2}",
                              axis="Decimal64")).dataset
    assert d.decimal_places == 2


def test_a_decimal_axis_over_a_float_file_without_decimal_places_is_rejected(tmp_path):
    write(tmp_path, pa.array([1.5, 2.25], pa.float64()))
    with pytest.raises(SuiteError, match="decimal_places"):
        load_suite(suite_file(tmp_path, axis="Decimal64"))
    load_suite(suite_file(tmp_path, axis="Float64"))          # fine without


def test_a_numeric_axis_over_a_string_file_is_rejected_from_the_file_type(tmp_path):
    write(tmp_path, pa.array(["a", "b"]))
    with pytest.raises(SuiteError, match="numeric"):
        load_suite(suite_file(tmp_path, axis="Int32"))
    load_suite(suite_file(tmp_path, axis="String"))


@pytest.mark.parametrize("extra", ["distribution: {kind: uniform}", "cardinality_ratio: 0.5",
                                   "entropy_bits: 2.0", "rows: 4", "null_count: 0",
                                   "value_range: [0, 1]", "arrow_type: int32"])
def test_generator_and_measured_keys_beside_a_path_are_rejected_by_name(tmp_path, extra):
    write(tmp_path, pa.array([1, 2], pa.int32()))
    with pytest.raises(SuiteError, match=extra.split(":")[0]):
        load_suite(suite_file(tmp_path, "{path: x.arrow, column: v, %s}" % extra))


def test_path_without_column_and_column_without_path(tmp_path):
    write(tmp_path, pa.array([1, 2], pa.int32()))
    with pytest.raises(SuiteError, match="column"):
        load_suite(suite_file(tmp_path, "{path: x.arrow}"))
    with pytest.raises(SuiteError, match="path"):
        load_suite(suite_file(tmp_path, "{column: v, rows: 3, arrow_type: int32, "
                                        "value_range: [0, 1], null_count: 0, seed: 1}"))


def test_missing_file_and_missing_column_are_named(tmp_path):
    with pytest.raises(SuiteError, match="x.arrow"):
        load_suite(suite_file(tmp_path))
    write(tmp_path, pa.array([1, 2], pa.int32()))
    with pytest.raises(SuiteError, match="'zzz'"):
        load_suite(suite_file(tmp_path, "{path: x.arrow, column: zzz}"))


def test_an_unexpressible_arrow_type_is_named(tmp_path):
    write(tmp_path, pa.array(["a"], pa.large_string()))
    with pytest.raises(SuiteError, match="large_string"):
        load_suite(suite_file(tmp_path, axis="String"))


def test_parse_suite_text_resolves_against_base_dir_or_the_working_directory(tmp_path, monkeypatch):
    write(tmp_path, pa.array([1, 2], pa.int32()))
    text = SUITE.format(dataset="{path: x.arrow, column: v}", axis="Int32")
    assert parse_suite(text, base_dir=tmp_path).dataset.rows == 2
    monkeypatch.chdir(tmp_path)
    assert parse_suite(text).dataset.rows == 2
