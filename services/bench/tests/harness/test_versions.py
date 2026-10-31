"""chbench.harness.versions: the pinned engine versions (docs/engines/versions.yaml) and drift
warnings. No server."""

from chbench.harness import versions

PIN = """\
clickhouse:
  image: "clickhouse/clickhouse-server:26.8.31.2@sha256:abc"
  version: "26.8.31.2"
polars: "2.0.0"
datafusion: "55.0.0"
pyarrow: "25.0.1"
"""


def test_load_pinned_reads_the_file(tmp_path):
    p = tmp_path / "versions.yaml"
    p.write_text(PIN, encoding="utf-8")
    pinned = versions.load_pinned(p)
    assert pinned["clickhouse"]["version"] == "26.8.31.2"
    assert pinned["polars"] == "2.0.0"


def test_load_pinned_of_a_missing_file_is_empty(tmp_path):
    assert versions.load_pinned(tmp_path / "absent.yaml") == {}


def test_the_repository_file_pins_clickhouse_by_digest():
    pinned = versions.load_pinned()
    image = pinned["clickhouse"]["image"]
    assert "@sha256:" in image
    assert image.split("@")[0].endswith(":" + pinned["clickhouse"]["version"])
    for engine in ("polars", "datafusion", "pyarrow"):
        assert isinstance(pinned[engine], str)


def test_no_drift_when_versions_match(tmp_path):
    p = tmp_path / "versions.yaml"
    p.write_text(PIN, encoding="utf-8")
    assert versions.drift(versions.load_pinned(p), {"clickhouse": "26.8.31.2"}) == []


def test_drift_names_the_engine_and_both_versions(tmp_path):
    p = tmp_path / "versions.yaml"
    p.write_text(PIN, encoding="utf-8")
    out = versions.drift(
        versions.load_pinned(p), {"clickhouse": "26.9.1.1", "polars": "2.1.0"}
    )
    assert out == [
        "clickhouse 26.9.1.1 differs from the pinned 26.8.31.2 "
        "(docs/engines/versions.yaml; after an upgrade see docs/engines/README.md)",
        "polars 2.1.0 differs from the pinned 2.0.0 "
        "(docs/engines/versions.yaml; after an upgrade see docs/engines/README.md)",
    ]


def test_drift_ignores_engines_without_a_pin_or_an_actual_version():
    assert versions.drift({}, {"clickhouse": "26.9.1.1"}) == []
    assert versions.drift({"polars": "2.0.0"}, {"clickhouse": None}) == []
