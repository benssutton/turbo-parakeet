import json
import math

import pyarrow as pa
import pytest

from synthdata import generate, load_or_generate, save
from synthdata.column.solve import WARNING_PREFIXES
from synthdata.generate import Manifest, SolveRecord
from synthdata.column.spec import DatasetSpec, Distribution, SpecError


def uc1(rows=20_000, **kw) -> DatasetSpec:
    d = dict(
        rows=rows,
        arrow_type="float64",
        value_range=(-1.0, 10.0),
        decimal_places=8,
        null_count=0,
        seed=42,
        cardinality_ratio=0.75,
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=0.25, shift=-1.0),
    )
    d.update(kw)
    return DatasetSpec(**d)


def test_generate_returns_the_requested_row_count_and_type():
    d = generate(uc1())
    assert len(d.array) == 20_000
    assert d.array.type == pa.float64()


def test_manifest_records_achieved_metrics():
    d = generate(uc1())
    assert d.manifest.achieved["cardinality_ratio"] == pytest.approx(0.75, abs=0.01)
    assert "entropy_bits" in d.manifest.achieved
    assert d.manifest.dataset_id == uc1().dataset_id


def test_manifest_flags_tolerance_per_metric():
    d = generate(uc1(runs_z=-0.25))
    assert d.manifest.within_tolerance["runs_z"] is True
    assert d.manifest.targets["runs_z"] == -0.25


def test_a_missed_tolerance_warns_and_does_not_raise():
    """Best effort, never fails."""
    d = generate(uc1(interarrival_cv=0.01))  # unreachable by construction
    assert d.manifest.within_tolerance["interarrival_cv"] is False
    assert d.manifest.warnings


def test_an_unreachable_target_is_recorded_as_a_clamp_not_a_near_miss():
    d = generate(uc1(rows=5_000, cardinality_ratio=0.5, entropy_bits=0.1))
    rec = d.manifest.solves["entropy_bits"]
    assert rec.status == "clamped_low"
    assert "clamped" in rec.message


def test_generation_is_deterministic():
    a, b = generate(uc1()), generate(uc1())
    assert a.array.equals(b.array)
    assert a.manifest.achieved == b.manifest.achieved


def test_a_different_seed_gives_a_different_column():
    assert not generate(uc1()).array.equals(generate(uc1(seed=43)).array)


def test_round_trips_through_the_arrow_ipc_cache(tmp_path):
    d = generate(uc1())
    path = save(d, tmp_path)
    assert path.suffix == ".arrow"
    back = load_or_generate(uc1(), tmp_path)
    assert back.array.equals(d.array)
    assert back.manifest.dataset_id == d.manifest.dataset_id


def test_cached_column_is_bit_identical_after_reload(tmp_path):
    """Arrow IPC keeps the exact type and buffers, with no encode/decode step."""
    d = generate(uc1())
    save(d, tmp_path)
    back = load_or_generate(uc1(), tmp_path)
    assert back.array.type == d.array.type
    assert back.array.to_pylist() == d.array.to_pylist()


def test_cache_miss_regenerates_and_cache_hit_does_not(tmp_path):
    d = generate(uc1())
    save(d, tmp_path)
    hit = load_or_generate(uc1(), tmp_path)
    assert hit.manifest.generation_seconds == d.manifest.generation_seconds


def test_manifest_is_written_as_readable_json(tmp_path):
    save(generate(uc1()), tmp_path)
    mpath = tmp_path / f"{uc1().dataset_id}.json"
    loaded = json.loads(mpath.read_text())
    assert loaded["dataset_id"] == uc1().dataset_id
    assert "achieved" in loaded and "warnings" in loaded


def test_use_case_2_shape_generates(tmp_path):
    """The int_encodings example suite's dataset, at a row count fast enough for
    a test."""
    s = DatasetSpec(
        rows=100_000,
        arrow_type="int32",
        value_range=(1_000_000, 9_999_999),
        null_count=0,
        seed=42,
        cardinality_ratio=0.01,
        entropy_bits=4.0,
        runs_z=-0.25,
    )
    d = generate(s)
    assert len(d.array) == 100_000
    assert d.manifest.achieved["entropy_bits"] == pytest.approx(4.0, abs=0.01)
    assert d.manifest.achieved["runs_z"] == pytest.approx(-0.25, abs=0.01)


def test_list_of_integer_generates_end_to_end():
    s = DatasetSpec(
        rows=1_000,
        arrow_type="list<int32>",
        value_range=(0, 99_999),
        null_count=0,
        seed=1,
        cardinality_ratio=0.5,
        array_len_range=(2, 6),
    )
    d = generate(s)
    assert d.array.type == pa.list_(pa.int32())
    assert len(d.array) == 1_000
    assert len(d.array.values) == 1_000 * 4  # rows * round(mean(2, 6))


def test_a_spread_distribution_with_no_sigma_is_a_spec_error_not_a_value_error():
    """The generator raises only from validation."""
    s = uc1(distribution=Distribution(kind="lognormal", mean=4.0, shift=-1.0))
    with pytest.raises(SpecError, match="sigma"):
        generate(s)


@pytest.mark.parametrize(
    "arrow_type,lo,hi,pt",
    [
        ("date32", 0, 20_000, pa.date32()),
        ("timestamp[ms]", 0, 10**12, pa.timestamp("ms")),
    ],
)
def test_dates_and_timestamps_generate_end_to_end(arrow_type, lo, hi, pt):
    s = DatasetSpec(
        rows=5_000,
        arrow_type=arrow_type,
        value_range=(lo, hi),
        null_count=0,
        seed=1,
        cardinality_ratio=0.1,
        runs_z=0.0,
    )
    d = generate(s)
    assert d.array.type == pt
    assert len(d.array) == 5_000


def test_random_strings_generate_end_to_end():
    s = DatasetSpec(
        rows=5_000,
        arrow_type="string",
        string_length=8,
        null_count=0,
        seed=1,
        cardinality_ratio=0.1,
        entropy_bits=5.0,
    )
    d = generate(s)
    assert d.array.type == pa.string()
    assert all(len(v) == 8 for v in d.array.to_pylist())
    assert d.manifest.achieved["entropy_bits"] == pytest.approx(5.0, abs=0.01)
    assert d.manifest.within_tolerance["entropy_bits"] is True


def test_uncapped_cardinality_generates_end_to_end():
    d = generate(uc1(cardinality_ratio=None))
    assert len(d.array) == 20_000
    assert 0 < d.manifest.achieved["cardinality_ratio"] <= 1
    assert "cardinality_ratio" not in d.manifest.targets


REPRESENTATIVE_SPECS = [
    ("int32", dict(value_range=(-1_000, 1_000))),
    ("uint16", dict(value_range=(0, 60_000))),
    ("float64", dict(value_range=(-1.0, 10.0), decimal_places=8)),
    ("float32", dict(value_range=(0.0, 1.0), decimal_places=3)),
    ("decimal32(9,4)", dict(value_range=(0.0, 1_000.0), decimal_places=4)),
    ("decimal64(18,2)", dict(value_range=(0.0, 1_000.0), decimal_places=2)),
    ("decimal128(38,10)", dict(value_range=(0.0, 1_000_000.0), decimal_places=10)),
    ("date32", dict(value_range=(0, 20_000))),
    ("timestamp[ms]", dict(value_range=(0, 10**12))),
    ("time32[s]", dict(value_range=(0, 86_399))),
    ("time64[us]", dict(value_range=(0, 86_399_999_999))),
    ("bool", dict(value_range=(0, 1), cardinality_ratio=0.001)),
    ("string", dict(string_length=6)),
    ("string", dict(categories=["red", "green", "blue"])),
    ("list<int32>", dict(value_range=(0, 1_000), array_len_range=(1, 4))),
    (
        "list<float64>",
        dict(value_range=(0.0, 1.0), decimal_places=4, array_len_range=(1, 4)),
    ),
]


@pytest.mark.parametrize(
    "arrow_type,kw",
    REPRESENTATIVE_SPECS,
    ids=[f"{t}:{'/'.join(k)}" for t, k in REPRESENTATIVE_SPECS],
)
def test_every_type_family_generates_without_raising(arrow_type, kw):
    """One representative spec per supported `arrow_type` family.
    Validation is the only place the generator may raise."""
    d = dict(
        rows=2_000,
        arrow_type=arrow_type,
        null_count=5,
        seed=7,
        cardinality_ratio=0.05,
        runs_z=0.0,
    )
    d.update(kw)
    if "categories" in kw:
        d.pop("cardinality_ratio")
    s = DatasetSpec(**d)
    out = generate(s)
    assert out.array.type == s.arrow_dtype
    assert len(out.array) == 2_000
    assert out.manifest.achieved["cardinality_ratio"] > 0


def test_a_dispersion_target_on_strings_is_a_spec_error_not_a_type_error():
    s = DatasetSpec(
        rows=1_000,
        arrow_type="string",
        string_length=4,
        null_count=0,
        seed=1,
        cardinality_ratio=0.1,
        index_of_dispersion=1.0,
    )
    with pytest.raises(SpecError, match="index_of_dispersion"):
        generate(s)


@pytest.mark.parametrize(
    "arrow_type,value_range",
    [
        ("int64", (-(2**63), 2**63 - 1)),
        ("uint64", (0, 2**64 - 1)),
    ],
)
def test_int64_and_uint64_generate_over_their_full_range(arrow_type, value_range):
    """A span beyond int64 overflowed `rng.choice(span)`; the range is
    legitimate, so the draw must handle it rather than validation reject it."""
    s = DatasetSpec(
        rows=2_000,
        arrow_type=arrow_type,
        value_range=value_range,
        null_count=0,
        seed=1,
        cardinality_ratio=0.5,
    )
    d = generate(s)
    assert d.array.type == s.arrow_dtype
    assert len(d.array) == 2_000
    assert d.manifest.achieved["cardinality_ratio"] == pytest.approx(0.5, abs=0.001)


def test_an_undefined_interarrival_cv_is_recorded_as_undefined_not_a_miss():
    """Every value distinct: no repeat-runs, so the metric is nan by
    definition. The manifest must say the metric is undefined for this data,
    not that the achieved value is "outside the tolerance"."""
    d = generate(uc1(rows=2_000, cardinality_ratio=1.0, interarrival_cv=0.5))
    m = d.manifest
    assert math.isnan(m.achieved["interarrival_cv"])
    assert m.within_tolerance["interarrival_cv"] is False
    assert any(w.startswith("undefined: interarrival_cv") for w in m.warnings)
    assert not any("outside the" in w for w in m.warnings)
    assert "interarrival_cv" not in m.solves


def test_runs_z_on_a_constant_column_is_recorded_as_undefined_not_a_miss():
    d = generate(
        uc1(rows=2_000, value_range=(3.0, 3.0), cardinality_ratio=None, runs_z=-0.25)
    )
    m = d.manifest
    assert math.isnan(m.achieved["runs_z"])
    assert m.within_tolerance["runs_z"] is False
    assert any(w.startswith("undefined: runs_z") for w in m.warnings)
    assert not any("outside the" in w for w in m.warnings)


def test_manifest_records_the_support_shortfall_by_mechanism():
    d = generate(
        uc1(
            rows=5_000,
            distribution=Distribution(
                kind="lognormal", mean=4.0, sigma=1.0, shift=-1.0
            ),
        )
    )
    m = d.manifest
    assert m.clip_collapsed > 0
    assert m.round_collapsed >= 0
    assert m.to_dict()["clip_collapsed"] == m.clip_collapsed
    clean = generate(uc1(rows=5_000)).manifest
    assert clean.clip_collapsed == 0


def test_solves_are_records_with_a_status_string_and_a_message():
    """A harness reads `status`, never parses `message`."""
    d = generate(uc1(rows=5_000, cardinality_ratio=0.02, entropy_bits=4.0))
    rec = d.manifest.solves["entropy_bits"]
    assert isinstance(rec, SolveRecord)
    assert rec.status == "ok" and rec.message == "ok"
    as_json = json.loads(d.manifest.to_json())
    assert as_json["solves"]["entropy_bits"] == {"status": "ok", "message": "ok"}
    back = Manifest.from_dict(as_json)
    assert back.solves["entropy_bits"] == rec


def test_an_out_of_band_runs_target_is_recorded_as_a_clamp_under_solves():
    low = generate(uc1(rows=5_000, runs_z=-10_000.0)).manifest
    assert low.solves["runs_z"].status == "clamped_low"
    assert any(w.startswith("clamp: runs_z") for w in low.warnings)
    high = generate(uc1(rows=5_000, runs_z=10_000.0)).manifest
    assert high.solves["runs_z"].status == "clamped_high"
    in_band = generate(uc1(rows=5_000, runs_z=-0.25)).manifest
    assert in_band.solves["runs_z"].status == "ok"


MISS_INDUCING = [
    dict(rows=5_000, cardinality_ratio=0.5, entropy_bits=0.1),  # clamp
    dict(
        rows=5_000,
        distribution=Distribution(kind="lognormal", mean=4.0, sigma=1.0, shift=-1.0),
    ),  # clip
    dict(rows=2_000, cardinality_ratio=1.0, interarrival_cv=0.5),  # undefined
    dict(
        rows=2_000, value_range=(3.0, 3.0), cardinality_ratio=None, runs_z=-0.25
    ),  # undefined
    dict(rows=5_000, runs_z=-10_000.0),  # clamp
    dict(rows=5_000, runs_z=-0.25, interarrival_cv=0.5),  # combined
    dict(rows=5_000, interarrival_cv=0.01),  # clamp
    dict(
        rows=1_000,
        value_range=(0.0, 1.0),
        decimal_places=2,
        cardinality_ratio=0.05,
        distribution=Distribution(kind="normal", mean=0.5, sigma=0.03),
    ),  # rounding
    dict(
        rows=20_000, value_range=(0.0, 1.0), decimal_places=2, cardinality_ratio=0.5
    ),  # capacity
]


@pytest.mark.parametrize(
    "kw", MISS_INDUCING, ids=[str(i) for i in range(len(MISS_INDUCING))]
)
def test_every_warning_starts_with_one_of_the_four_prefixes_and_a_metric_name(kw):
    """The vocabulary a harness author depends on: `clamp:`, `miss:`,
    `skipped:` or `undefined:`, then the metric name."""
    metrics = (
        "cardinality_ratio",
        "entropy_bits",
        "index_of_dispersion",
        "runs_z",
        "interarrival_cv",
    )
    m = generate(uc1(**kw)).manifest
    assert m.warnings, "this spec was chosen to warn"
    for w in m.warnings:
        prefix, _, rest = w.partition(" ")
        assert prefix in WARNING_PREFIXES, w
        assert rest.split(":")[0].split("=")[0].split(" ")[0] in metrics, w


def test_dispersion_solve_warnings_use_the_vocabulary():
    s = DatasetSpec(
        rows=5_000,
        arrow_type="float64",
        value_range=(-20.0, 20.0),
        decimal_places=4,
        null_count=0,
        seed=0,
        cardinality_ratio=0.5,
        distribution=Distribution(kind="lognormal", mean=4.0, shift=-10.0),
        index_of_dispersion=1.0,
    )
    m = generate(s).manifest
    assert m.solves["index_of_dispersion"].status == "invalid_bracket"
    assert any(w.startswith("miss: index_of_dispersion") for w in m.warnings)
    for w in m.warnings:
        assert w.partition(" ")[0] in WARNING_PREFIXES, w
