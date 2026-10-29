"""Real-data datasets (`dataset: {path, column}`): a `DatasetSpec` with
`path` / `column` is measured from an existing Arrow IPC file, never generated."""

import datetime as dt
import decimal
import json
from dataclasses import replace

import pyarrow as pa
import pytest

from synthdata import Manifest, load_or_generate
from synthdata.column.realdata import path_spec
from synthdata.column.spec import DatasetSpec, Distribution, SpecError

COLUMNS = {
    "int32": pa.array([3, None, 3, 7, 9, None, 1, 1], pa.int32()),
    "dec": pa.array(
        [
            decimal.Decimal("1.25"),
            decimal.Decimal("-3.50"),
            None,
            decimal.Decimal("1.25"),
        ],
        pa.decimal128(9, 2),
    ),
    "text": pa.array(["a", "b", "a", None, "c", "a"]),
    "day": pa.array(
        [dt.date(2020, 1, 1), dt.date(2021, 5, 6), None, dt.date(2020, 1, 1)],
        pa.date32(),
    ),
    "ts": pa.array(
        [dt.datetime(2020, 1, 1, 0, 0, 1), None, dt.datetime(2020, 1, 1, 0, 0, 2)],
        pa.timestamp("ms"),
    ),
}


def write(tmp_path, array, name="v", file="x.arrow"):
    p = tmp_path / file
    batch = pa.record_batch([array], names=[name])
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    return p


@pytest.mark.parametrize("key", list(COLUMNS))
def test_loading_a_path_dataset_returns_the_file_column_unchanged(tmp_path, key):
    array = COLUMNS[key]
    spec = path_spec(write(tmp_path, array), "v")
    ds = load_or_generate(spec, tmp_path / "cache")
    assert ds.array.type == array.type
    assert ds.array.equals(array)
    assert not (tmp_path / "cache").exists()  # no generation, no cache copy


def test_the_spec_fields_are_measured_from_the_file(tmp_path):
    spec = path_spec(write(tmp_path, COLUMNS["int32"]), "v")
    assert (spec.rows, spec.arrow_type, spec.null_count, spec.seed) == (
        8,
        "int32",
        2,
        0,
    )
    assert spec.value_range == (1.0, 9.0)
    assert spec.decimal_places is None
    assert spec.column == "v" and spec.path.endswith("x.arrow")
    spec.validate()


def test_decimal_scale_and_explicit_decimal_places(tmp_path):
    p = write(tmp_path, COLUMNS["dec"])
    spec = path_spec(p, "v")
    assert spec.arrow_type == "decimal128(9,2)" and spec.decimal_places == 2
    assert spec.value_range == (-3.5, 1.25)
    with pytest.raises(SpecError, match="decimal_places"):
        path_spec(p, "v", decimal_places=3)  # must equal the scale


def test_a_float_file_takes_the_explicit_decimal_places(tmp_path):
    p = write(tmp_path, pa.array([1.5, 2.25, None], pa.float64()))
    assert path_spec(p, "v").decimal_places is None
    assert path_spec(p, "v", decimal_places=2).decimal_places == 2


def test_string_and_temporal_columns(tmp_path):
    assert path_spec(write(tmp_path, COLUMNS["text"]), "v").value_range is None
    assert path_spec(write(tmp_path, COLUMNS["day"]), "v").arrow_type == "date32"
    spec = path_spec(write(tmp_path, COLUMNS["ts"]), "v")
    assert spec.arrow_type == "timestamp[ms]" and spec.value_range is not None


def test_manifest_has_measured_metrics_and_no_targets(tmp_path):
    ds = load_or_generate(
        path_spec(write(tmp_path, COLUMNS["text"]), "v"), tmp_path / "c"
    )
    m = ds.manifest
    # a, b, a, c, a: 3 distinct of 5 non-null values; counts 3, 1, 1
    assert m.achieved["cardinality_ratio"] == pytest.approx(3 / 5)
    assert m.achieved["entropy_bits"] == pytest.approx(1.3709505944546687)
    assert m.targets == {} and m.within_tolerance == {} and m.solves == {}
    assert m.warnings == []
    assert m.dataset_id == m.spec.dataset_id


def test_dataset_id_is_stable_for_a_file_and_changes_with_its_content(tmp_path):
    p = write(tmp_path, COLUMNS["int32"])
    a, b = path_spec(p, "v"), path_spec(p, "v")
    assert a.dataset_id == b.dataset_id
    write(tmp_path, pa.array([3, None, 3, 7, 9, None, 1, 2], pa.int32()))
    assert path_spec(p, "v").dataset_id != a.dataset_id


def test_dataset_id_depends_on_the_column_name(tmp_path):
    batch = pa.record_batch([COLUMNS["int32"], COLUMNS["int32"]], names=["v", "w"])
    p = tmp_path / "two.arrow"
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as w:
            w.write_batch(batch)
    assert path_spec(p, "v").dataset_id != path_spec(p, "w").dataset_id


def test_a_file_changed_after_the_spec_was_measured_is_refused(tmp_path):
    p = write(tmp_path, COLUMNS["int32"])
    spec = path_spec(p, "v")
    write(tmp_path, pa.array([1, 2, 3], pa.int32()))
    with pytest.raises(SpecError, match="changed"):
        load_or_generate(spec, tmp_path / "c")


def test_a_multi_batch_file_is_combined(tmp_path):
    p = tmp_path / "m.arrow"
    schema = pa.schema([("v", pa.int64())])
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as w:
            w.write_batch(
                pa.record_batch([pa.array([1, 2], pa.int64())], schema=schema)
            )
            w.write_batch(pa.record_batch([pa.array([3], pa.int64())], schema=schema))
    ds = load_or_generate(path_spec(p, "v"), tmp_path / "c")
    assert ds.array.to_pylist() == [1, 2, 3]


def test_spec_and_manifest_round_trip_through_json(tmp_path):
    ds = load_or_generate(
        path_spec(write(tmp_path, COLUMNS["dec"]), "v"), tmp_path / "c"
    )
    spec = ds.manifest.spec
    assert DatasetSpec.from_dict(json.loads(spec.to_json())) == spec
    again = Manifest.from_dict(json.loads(ds.manifest.to_json()))
    assert again.spec == spec and again.dataset_id == ds.manifest.dataset_id
    assert again.achieved == ds.manifest.achieved


@pytest.mark.parametrize(
    "bad",
    [
        dict(distribution=Distribution(kind="normal", sigma=1.0)),
        dict(cardinality_ratio=0.5),
        dict(entropy_bits=1.0),
        dict(categories=[1]),
        dict(string_length=3),
        dict(runs_z=0.0),
        dict(array_len_range=(1, 2)),
    ],
)
def test_validate_rejects_generator_fields_beside_a_path(tmp_path, bad):
    spec = path_spec(write(tmp_path, COLUMNS["int32"]), "v")
    with pytest.raises(SpecError, match="path"):
        replace(spec, **bad).validate()


def test_column_without_path_and_path_without_column_are_rejected(tmp_path):
    spec = path_spec(write(tmp_path, COLUMNS["int32"]), "v")
    with pytest.raises(SpecError, match="column"):
        replace(spec, column=None).validate()
    with pytest.raises(SpecError, match="path"):
        replace(spec, path=None).validate()


def test_missing_file_column_and_unexpressible_type(tmp_path):
    with pytest.raises(SpecError, match="nope.arrow"):
        path_spec(tmp_path / "nope.arrow", "v")
    p = write(tmp_path, COLUMNS["int32"])
    with pytest.raises(SpecError, match="'zzz'"):
        path_spec(p, "zzz")
    q = write(tmp_path, pa.array(["a"], pa.large_string()), file="l.arrow")
    with pytest.raises(SpecError, match="large_string"):
        path_spec(q, "v")
    r = write(tmp_path, pa.array([1], pa.decimal256(40, 2)), file="d.arrow")
    with pytest.raises(SpecError, match="decimal256"):
        path_spec(r, "v")
    s = write(tmp_path, pa.array([], pa.int32()), file="e.arrow")
    with pytest.raises(SpecError, match="rows"):
        path_spec(s, "v")


# The arrow_type vocabulary gains time zones and fixed_size_binary,
# which the recommended ClickHouse types (DateTime('tz'), FixedString(n)) need.
@pytest.mark.parametrize(
    "dtype,text",
    [
        (pa.timestamp("s", tz="UTC"), "timestamp[s, tz=UTC]"),
        (pa.timestamp("ns", tz="Europe/Paris"), "timestamp[ns, tz=Europe/Paris]"),
        (pa.timestamp("us"), "timestamp[us]"),
        (pa.binary(4), "fixed_size_binary(4)"),
    ],
)
def test_arrow_type_strings_round_trip_zones_and_fixed_size_binary(dtype, text):
    from synthdata.column.realdata import arrow_type_string
    from synthdata.column.spec import parse_arrow_type

    assert arrow_type_string(dtype) == text
    assert parse_arrow_type(text) == dtype


def test_zoned_timestamp_and_fixed_size_binary_files_are_measured(tmp_path):
    zoned = pa.array([1, None, 3], pa.timestamp("s", tz="UTC"))
    spec = path_spec(write(tmp_path, zoned), "v")
    assert spec.arrow_type == "timestamp[s, tz=UTC]" and spec.value_range == (1.0, 3.0)
    fixed = pa.array([b"abcd", None, b"wxyz"], pa.binary(4))
    spec = path_spec(write(tmp_path, fixed, file="f.arrow"), "v")
    assert spec.arrow_type == "fixed_size_binary(4)" and spec.value_range is None
    assert load_or_generate(spec, tmp_path / "c").array.equals(fixed)


def test_a_binary_file_is_measured_and_loaded(tmp_path):
    binary = pa.array([b"\xff\x00", None, b"ab", b"\xff\x00"], pa.binary())
    spec = path_spec(write(tmp_path, binary), "v")
    assert spec.arrow_type == "binary" and spec.value_range is None
    ds = load_or_generate(spec, tmp_path / "c")
    assert ds.array.equals(binary)
    assert ds.manifest.achieved["cardinality_ratio"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "bad",
    [
        "timestamp[s, tz=]",
        "timestamp[xx, tz=UTC]",
        "fixed_size_binary(0)",
        "fixed_size_binary(x)",
    ],
)
def test_malformed_zone_and_fixed_size_binary_strings_are_rejected(bad):
    from synthdata.column.spec import parse_arrow_type

    with pytest.raises(SpecError):
        parse_arrow_type(bad)


@pytest.mark.parametrize("arrow_type", ["timestamp[s, tz=UTC]", "fixed_size_binary(4)"])
def test_a_generated_dataset_cannot_use_zones_or_fixed_size_binary(arrow_type):
    spec = DatasetSpec(
        rows=10, arrow_type=arrow_type, null_count=0, seed=0, value_range=(0, 5)
    )
    with pytest.raises(SpecError, match="path dataset"):
        spec.validate()


@pytest.mark.parametrize(
    "dtype, scale", [(pa.decimal64(18, 18), 18), (pa.decimal32(9, 2), 2)]
)
def test_a_narrow_decimal_file_is_loaded_and_measured(tmp_path, dtype, scale):
    # dictionary_encode has no kernel for decimal32/64; the metrics go through decimal128
    values = [
        decimal.Decimal(1) / 4 if scale == 18 else decimal.Decimal("1.25"),
        decimal.Decimal(0),
        None,
        decimal.Decimal(1) / 4 if scale == 18 else decimal.Decimal("1.25"),
    ]
    array = pa.array(values, dtype)
    spec = path_spec(write(tmp_path, array), "v")
    ds = load_or_generate(spec, tmp_path / "cache")
    assert ds.array.type == dtype
    assert ds.manifest.achieved["cardinality_ratio"] == pytest.approx(2 / 3)
    assert ds.manifest.achieved["entropy_bits"] == pytest.approx(0.9182958340544896)
