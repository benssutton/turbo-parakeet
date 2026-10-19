"""ClickHouse recommendations, Arrow IPC compression choice and the chbench handoff (spec 2026-10-19)."""

import polars as pl
import pyarrow as pa
import pytest
from datagen import describe_mixed, large_dataset_mixed

from analytics import _plugin
from analytics.describe._sizes import ipc_body_bytes
from analytics.recommend import OneShotRecommender, StreamingRecommender


def pyarrow_body(arr: pa.Array, compression: str | int | None) -> int:
    """Body length of the record-batch IPC messages pyarrow writes for one column
    (`compression`: None, a ZSTD level, or "lz4" = LZ4_FRAME)."""
    return ipc_body_bytes(arr, compression)


@pytest.mark.parametrize(
    "make_frame",
    [lambda: describe_mixed(2_000), large_dataset_mixed],
    ids=["describe", "large"],
)
def test_lz4_frame_body_sizes_equal_pyarrow(make_frame):
    frame = make_frame()
    for name in frame.columns:
        arr = frame[name].rechunk().to_arrow(compat_level=pl.CompatLevel.oldest())
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if pa.types.is_dictionary(arr.type) or pa.types.is_null(arr.type):
            continue
        t = pa.table({"x": arr})
        assert _plugin.ipc_body_sizes(t, "lz4")["x"] == pyarrow_body(arr, "lz4"), name
        assert _plugin.ipc_body_sizes(t, "none")["x"] == pyarrow_body(arr, None), name
        assert _plugin.ipc_body_sizes(t, "zstd:1")["x"] == pyarrow_body(arr, 1), name


@pytest.mark.parametrize("cls", [OneShotRecommender, StreamingRecommender])
@pytest.mark.parametrize(
    "kw, word",
    [
        ({"zstd_levels": ()}, "zstd_levels"),
        ({"zstd_levels": (1, 1)}, "zstd_levels"),
        ({"zstd_levels": (99,)}, "zstd_levels"),
        ({"zstd_min_saving": 1.0}, "zstd_min_saving"),
        ({"gcd_min_saving": -0.1}, "gcd_min_saving"),
        ({"zstd_level_tolerance": float("nan")}, "zstd_level_tolerance"),
    ],
)
def test_compression_keywords_are_validated(cls, kw, word):
    with pytest.raises(ValueError, match=word):
        cls(**kw)


def one_shot(frame: pl.DataFrame, **kw) -> dict[str, dict]:
    out = OneShotRecommender(**kw).add(frame).result()
    return {r["column"]: r for r in out.iter_rows(named=True)}


def ch_frame(n: int = 200_000) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "small": rng.integers(0, 200, n).astype(np.int64),  # -> UInt8
            "thousands": (rng.integers(1, 10**9, n) * 1_000).astype(
                np.int64
            ),  # GCD 1000
            "threes": (rng.integers(1, 2**38, n) * 3).astype(np.int64),  # GCD 3: pruned
            "noise": rng.random(n),  # near-incompressible
            "text": rng.choice(["alpha", "beta", "gamma"], n),  # LowCardinality(String)
            "maybe": pl.Series(([None, 1, 2] * (n // 3 + 1))[:n], dtype=pl.Int64),
        }
    )


def test_clickhouse_types_and_codecs():
    rows = one_shot(ch_frame())
    assert rows["small"]["clickhouse_type"] == "UInt8"
    assert rows["text"]["clickhouse_type"] == "LowCardinality(String)"
    assert rows["maybe"]["clickhouse_type"] == "Nullable(UInt8)"
    assert rows["thousands"]["clickhouse_codecs"].startswith("GCD")
    gcd_rows = [
        c for c in rows["threes"]["clickhouse_candidates"] if c["codecs"] == "GCD"
    ]
    assert gcd_rows and gcd_rows[0]["outcome"] == "pruned"
    assert rows["noise"]["clickhouse_codecs"] == "LZ4"
    assert any(
        c["outcome"] == "pruned" and c["codecs"].startswith("ZSTD")
        for c in rows["noise"]["clickhouse_candidates"]
    )
    for r in rows.values():
        assert r["clickhouse_size_bytes"] > 0
        assert sum(c["outcome"] == "chosen" for c in r["clickhouse_candidates"]) == 1
        assert r["datafusion_ipc_compression"] in ("LZ4_FRAME", "ZSTD")
        assert r["polars_ipc_compression"] in ("LZ4_FRAME", "ZSTD")
        assert (r["datafusion_ipc_zstd_level"] is None) == (
            r["datafusion_ipc_compression"] == "LZ4_FRAME"
        )


def test_compression_keywords_change_choices():
    base = one_shot(ch_frame())
    assert not one_shot(ch_frame(), gcd_min_saving=0.99)["thousands"][
        "clickhouse_codecs"
    ].startswith("GCD")
    levels = one_shot(ch_frame(), zstd_levels=(3,))
    for r in levels.values():
        if r["clickhouse_codecs"] and "ZSTD" in r["clickhouse_codecs"]:
            assert r["clickhouse_codecs"].endswith("ZSTD(3)")
    eager = one_shot(ch_frame(), zstd_min_saving=0.0)
    assert "ZSTD" in eager["small"]["clickhouse_codecs"]
    assert base["small"]["clickhouse_type"] == eager["small"]["clickhouse_type"]


def test_unmappable_type_has_a_reason():
    from datetime import timedelta

    frame = pl.DataFrame({"d": [timedelta(seconds=1), timedelta(seconds=2)]})
    r = one_shot(frame)["d"]
    assert r["clickhouse_type"] is None and r["clickhouse_codecs"] is None
    assert r["clickhouse_candidates"][0]["outcome"] == "pruned"


def test_streaming_matches_one_shot_on_ch_frame():
    frame = ch_frame()
    streamed = StreamingRecommender(reservoir_rows=frame.height + 65_536)
    for batch in frame.iter_slices(50_000):
        streamed.add(batch)
    got = {r["column"]: r for r in streamed.result().iter_rows(named=True)}
    want = one_shot(frame)
    for name, r in got.items():
        o = want[name]
        assert r["n_sampled_rows"] == frame.height, name
        assert r["clickhouse_type"] == o["clickhouse_type"], name
        for k in (
            "clickhouse_size_bytes",
            "datafusion_size_lz4_bytes",
            "polars_size_lz4_bytes",
            "datafusion_ipc_size_bytes",
            "polars_ipc_size_bytes",
        ):
            assert r[k] == pytest.approx(o[k], rel=0.02), (name, k)
    gcd_rows = [
        c for c in got["threes"]["clickhouse_candidates"] if c["codecs"] == "GCD"
    ]
    assert gcd_rows and gcd_rows[0]["outcome"] == "pruned"
    assert got["thousands"]["clickhouse_codecs"].startswith("GCD")
    assert got["noise"]["clickhouse_codecs"] == "LZ4"


def test_streaming_without_sample():
    frame = ch_frame(1_000)
    out = StreamingRecommender(reservoir_rows=0).add(frame).result()
    for r in out.iter_rows(named=True):
        assert r["clickhouse_type"] is not None, r["column"]
        for k in (
            "clickhouse_codecs",
            "clickhouse_size_bytes",
            "clickhouse_candidates",
            "datafusion_ipc_compression",
            "datafusion_size_lz4_bytes",
            "polars_ipc_compression",
        ):
            assert not r[k], (r["column"], k)


# ---- to_chbench (spec 2026-10-19 §7) ----------------------------------------------------


def chbench_frame(n: int = 50_000) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {
            "id": rng.integers(0, 200, n).astype(np.int64),
            "price": np.round(rng.integers(100, 100_000, n) / 100, 2),
            "tag": rng.choice(["alpha", "beta", "gamma"], n),
            "maybe": pl.Series(([None, 1, 2] * (n // 3 + 1))[:n], dtype=pl.Int64),
            "items": pl.Series([[1, 2]] * n),
            "2 Col!": rng.integers(0, 50, n).astype(np.int64),
            "2 col?": rng.integers(0, 50, n).astype(np.int64),
        }
    )


@pytest.fixture(scope="module")
def chbench_out(tmp_path_factory):
    import yaml

    from analytics.recommend import to_chbench

    frame = chbench_frame()
    result = OneShotRecommender().add(frame).result()
    out = tmp_path_factory.mktemp("chbench")
    skipped = to_chbench(result, frame, out)
    suites = {p.stem: yaml.safe_load(p.read_text()) for p in out.glob("*.yaml")}
    return out, skipped, suites


def test_to_chbench_files_and_skips(chbench_out):
    out, skipped, suites = chbench_out
    assert [c for c, _ in skipped] == ["items"]
    assert set(suites) == {"id", "price", "tag", "maybe", "c_2_col_", "c_2_col__2"}
    for name in suites:
        assert (out / f"{name}.arrow").exists()
        assert (out / f"{name}.predicted.json").exists()
    assert suites["c_2_col_"]["dataset"] == {
        "path": "c_2_col_.arrow",
        "column": "2 Col!",
    }


def test_to_chbench_yaml_axes(chbench_out):
    _, _, suites = chbench_out
    tag = suites["tag"]
    assert list(tag) == [
        "name",
        "dataset",
        "clickhouse",
        "queries",
        "results",
    ]
    assert tag["clickhouse"]["type"] == ["String"]
    assert tag["clickhouse"]["codec"] == ["LowCardinality", "none"]
    assert tag["clickhouse"]["nullable"] == [False]
    assert [k["kind"] for k in tag["queries"]["kinds"]] == ["full_scan", "filter"]
    price = suites["price"]
    assert price["clickhouse"]["type"][0] in ("Decimal32", "Decimal64")
    assert price["clickhouse"]["type"][1] == "Float64"
    assert price["dataset"]["decimal_places"] == 2
    assert price["queries"]["kinds"][-1]["predicate"] == "range"
    assert suites["maybe"]["clickhouse"]["nullable"] == [True]
    assert set(suites["id"]["clickhouse"]) == {
        "type",
        "nullable",
        "primary_key",
        "projection",
        "codec",
        "skip_index",
        "compression",
    }
    assert suites["id"]["clickhouse"]["type"] == ["UInt8", "Int64"]
    assert suites["id"]["clickhouse"]["compression"][-1] == "LZ4" or suites["id"][
        "clickhouse"
    ]["compression"] == ["LZ4"]


def test_to_chbench_predicted_and_arrow(chbench_out):
    import json

    out, _, suites = chbench_out
    for name, suite in suites.items():
        ch = suite["clickhouse"]
        predicted = json.loads((out / f"{name}.predicted.json").read_text())
        cells = {
            f"{t}|{k}|{c}"
            for t in ch["type"]
            for k in ch["codec"]
            for c in ch["compression"]
        }
        assert set(predicted) == cells
    ident = json.loads((out / "id.predicted.json").read_text())
    comp = suites["id"]["clickhouse"]["compression"][0]
    assert ident[f"UInt8|none|{comp}"] > 0
    assert ident["Int64|none|LZ4"] > 0
    assert json.loads((out / "tag.predicted.json").read_text())["String|none|LZ4"] > 0
    with pa.ipc.open_file(out / "tag.arrow") as f:
        assert f.num_record_batches == 1
        t = f.get_batch(0)
        assert t.schema.field("tag").type == pa.string()
        assert t.num_rows == 50_000


def test_to_chbench_normalises_categorical_and_columns(tmp_path):
    from analytics.recommend import to_chbench

    frame = pl.DataFrame(
        {"c": pl.Series(["a", "b"] * 500, dtype=pl.Categorical), "x": [1] * 1000}
    )
    result = OneShotRecommender().add(frame).result()
    to_chbench(result, frame, tmp_path, columns=["c"])
    assert sorted(p.name for p in tmp_path.glob("*.yaml")) == ["c.yaml"]
    with pa.ipc.open_file(tmp_path / "c.arrow") as f:
        assert f.get_batch(0).schema.field("c").type == pa.string()


def test_to_chbench_suites_parse_in_chbench(chbench_out):
    import sys
    from pathlib import Path

    root = Path("c:/Users/Alexander/clickhouse_benchmark")
    if not root.exists():
        pytest.skip("clickhouse_benchmark not found")
    sys.path.insert(0, str(root))
    pytest.importorskip("chbench")
    from chbench.harness.suite import load_suite

    out, _, suites = chbench_out
    for name in suites:
        load_suite(out / f"{name}.yaml")


def test_clickhouse_profile_binding():
    t = pa.table({"a": pa.array([3, 6, 9, None] * 100, pa.int64()), "f": [0.5] * 400})
    p = _plugin.clickhouse_profile(t, "a", "GCD, LZ4")
    assert p["clickhouse_type"] == "Nullable(Int64)" and p["bytes"] > 0
    f = _plugin.clickhouse_profile(t, "f", "GCD, LZ4")
    assert f["clickhouse_type"] == "Float64" and f["bytes"] is None
    assert _plugin.clickhouse_profile(t, "f", "ZSTD(3)")["bytes"] > 0
    with pytest.raises(ValueError):
        _plugin.clickhouse_profile(t, "nope", "LZ4")
    with pytest.raises(ValueError):
        _plugin.clickhouse_profile(t, "a", "Delta")
