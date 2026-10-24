"""ClickHouse recommendations, Arrow IPC compression choice and the chbench handoff (spec 2026-10-19)."""

from decimal import Decimal

import polars as pl
import pyarrow as pa
import pytest
from datagen import describe_mixed, large_dataset_mixed
from harness import pyarrow_body

from turbo_parakeet import _plugin
from turbo_parakeet.recommenders import OneShotRecommender, StreamingRecommender


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
        ({"zstd_levels": (0,)}, "zstd_levels"),
        ({"zstd_levels": (23,)}, "zstd_levels"),
        ({"zstd_min_saving": 1.0}, "zstd_min_saving"),
        ({"transform_min_saving": -0.1}, "transform_min_saving"),
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
    assert gcd_rows
    assert gcd_rows[0]["outcome"] == "pruned"
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
    assert not one_shot(ch_frame(), transform_min_saving=0.99)["thousands"][
        "clickhouse_codecs"
    ].startswith("GCD")
    levels = one_shot(ch_frame(), zstd_levels=(3,))
    for r in levels.values():
        if r["clickhouse_codecs"] and "ZSTD" in r["clickhouse_codecs"]:
            assert r["clickhouse_codecs"].endswith("ZSTD(3)")
    eager = one_shot(ch_frame(), zstd_min_saving=0.0)
    assert "ZSTD" in eager["small"]["clickhouse_codecs"]
    assert base["small"]["clickhouse_type"] == eager["small"]["clickhouse_type"]


def delta_frame(n: int = 200_000) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(1)
    steps = rng.choice([-100, 100], n).astype(np.int64)
    return pl.DataFrame(
        {
            "regular": pl.Series(
                np.arange(n, dtype=np.int64) * 1_000 + 1_700_000_000_123
            ).cast(pl.Datetime("ms")),
            "walk": 10**12 + np.cumsum(steps),
            "random_int": rng.integers(-(2**62), 2**62, n).astype(np.int64),
            "float": rng.random(n),
            "text": rng.choice(["alpha", "beta", "gamma"], n),
            "dec": pl.Series(
                [Decimal(int(v) * 10**8) / 100 for v in rng.integers(1, 10**18, n)],
                dtype=pl.Decimal(30, 2),
            ),
        }
    )


def _delta_rows(rows, col):
    return [
        c
        for c in rows[col]["clickhouse_candidates"]
        if c["codecs"].startswith(("Delta", "DoubleDelta"))
    ]


def check_delta_choices(rows):
    assert rows["regular"]["clickhouse_codecs"].startswith("DoubleDelta")
    assert any(c["codecs"].startswith("Delta") for c in _delta_rows(rows, "regular"))
    assert rows["walk"]["clickhouse_codecs"].startswith("Delta")
    ri = _delta_rows(rows, "random_int")
    assert ri
    assert all(c["outcome"] == "pruned" for c in ri)
    assert all("delta" in c["evidence"] for c in ri)
    for col in ("float", "text", "dec"):
        assert not _delta_rows(rows, col), col


def test_delta_families_one_shot():
    rows = one_shot(delta_frame())
    check_delta_choices(rows)


def test_transform_min_saving_disables_delta():
    rows = one_shot(delta_frame(), transform_min_saving=0.9999)
    for col in ("regular", "walk", "random_int"):
        codec = rows[col]["clickhouse_codecs"]
        assert not codec.startswith(("GCD", "Delta", "DoubleDelta")), (col, codec)


def test_delta_families_streaming_parity():
    frame = delta_frame()
    one = one_shot(frame)
    streamed = StreamingRecommender(reservoir_rows=frame.height + 65_536)
    streamed.add(frame)
    got = {r["column"]: r for r in streamed.result().iter_rows(named=True)}
    check_delta_choices(got)
    for name, r in got.items():
        assert r["clickhouse_codecs"] == one[name]["clickhouse_codecs"], name
        assert r["clickhouse_size_bytes"] == one[name]["clickhouse_size_bytes"], name


def test_unmappable_type_has_a_reason():
    from datetime import timedelta

    frame = pl.DataFrame({"d": [timedelta(seconds=1), timedelta(seconds=2)]})
    r = one_shot(frame)["d"]
    assert r["clickhouse_type"] is None
    assert r["clickhouse_codecs"] is None
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
    assert gcd_rows
    assert gcd_rows[0]["outcome"] == "pruned"
    assert got["thousands"]["clickhouse_codecs"].startswith("GCD")
    assert got["noise"]["clickhouse_codecs"] == "LZ4"


def test_streaming_low_cardinality_counts_one_dictionary():
    import numpy as np

    rng = np.random.default_rng(3)
    vals = np.array([f"value-{i:05d}" for i in range(5_000)])
    frame = pl.DataFrame({"s": rng.choice(vals, 200_000)})
    streamed = StreamingRecommender(reservoir_rows=frame.height + 65_536)
    for batch in frame.iter_slices(50_000):
        streamed.add(batch)
    got = streamed.result().row(0, named=True)
    want = one_shot(frame)["s"]
    assert "LowCardinality" in want["clickhouse_type"]
    assert got["clickhouse_type"] == want["clickhouse_type"]
    assert got["clickhouse_size_bytes"] == pytest.approx(
        want["clickhouse_size_bytes"], rel=0.02
    )


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

    from turbo_parakeet.recommenders import to_chbench

    frame = chbench_frame()
    result = OneShotRecommender().add(frame).result()
    out = tmp_path_factory.mktemp("chbench")
    skipped = to_chbench(result, frame, out)
    suites = {p.stem: yaml.safe_load(p.read_text()) for p in out.glob("*.yaml")}
    return out, skipped, suites


def test_to_chbench_files_and_skips(chbench_out):
    out, skipped, suites = chbench_out
    assert [c for c, _ in skipped] == ["items"]
    assert set(suites) == {
        "id",
        "id_original",
        "price",
        "price_original",
        "tag",
        "maybe",
        "maybe_original",
        "c_2_col_",
        "c_2_col__original",
        "c_2_col__2",
        "c_2_col__2_original",
    }
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
    assert [k["kind"] for k in tag["queries"]["kinds"]] == ["full_scan"] + [
        "filter"
    ] * 3
    assert [k["target_fraction"] for k in tag["queries"]["kinds"][1:]] == [
        0.001,
        0.01,
        0.05,
    ]
    assert {k["predicate"] for k in tag["queries"]["kinds"][1:]} == {"in"}
    price = suites["price"]
    assert price["clickhouse"]["type"] in (["Decimal32"], ["Decimal64"])
    assert suites["price_original"]["clickhouse"]["type"] == ["Float64"]
    assert price["dataset"]["decimal_places"] == 2
    assert "decimal_places" not in suites["price_original"]["dataset"]
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
    assert suites["id"]["clickhouse"]["type"] == ["UInt8"]
    assert suites["id_original"]["clickhouse"]["type"] == ["Int64"]
    assert suites["id_original"]["dataset"]["path"] == "id_original.arrow"
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
    assert (
        json.loads((out / "id_original.predicted.json").read_text())["Int64|none|LZ4"]
        > 0
    )
    assert json.loads((out / "tag.predicted.json").read_text())["String|none|LZ4"] > 0
    with pa.ipc.open_file(out / "tag.arrow") as f:
        assert f.num_record_batches == 1
        t = f.get_batch(0)
        assert t.schema.field("tag").type == pa.string()
        assert t.num_rows == 50_000


def test_to_chbench_normalises_categorical_and_columns(tmp_path):
    from turbo_parakeet.recommenders import to_chbench

    frame = pl.DataFrame(
        {"c": pl.Series(["a", "b"] * 500, dtype=pl.Categorical), "x": [1] * 1000}
    )
    result = OneShotRecommender().add(frame).result()
    to_chbench(result, frame, tmp_path, columns=["c"])
    assert sorted(p.name for p in tmp_path.glob("*.yaml")) == ["c.yaml"]
    with pa.ipc.open_file(tmp_path / "c.arrow") as f:
        assert f.get_batch(0).schema.field("c").type == pa.string()


def test_to_chbench_suites_parse_in_chbench(chbench_out, monkeypatch):
    import os
    from pathlib import Path

    root = Path(
        os.environ.get("CHBENCH_DIR")
        or Path(__file__).resolve().parents[1].parent / "clickhouse_benchmark"
    )
    if not root.exists():
        pytest.skip("clickhouse_benchmark not found")
    monkeypatch.syspath_prepend(str(root))
    pytest.importorskip("chbench")
    from chbench.harness.suite import load_suite

    out, _, suites = chbench_out
    for name in suites:
        load_suite(out / f"{name}.yaml")


def test_clickhouse_profile_binding():
    t = pa.table({"a": pa.array([3, 6, 9, None] * 100, pa.int64()), "f": [0.5] * 400})
    p = _plugin.clickhouse_profile(t, "a", "GCD, LZ4")
    assert p["clickhouse_type"] == "Nullable(Int64)"
    assert p["bytes"] > 0
    f = _plugin.clickhouse_profile(t, "f", "GCD, LZ4")
    assert f["clickhouse_type"] == "Float64"
    assert f["bytes"] is None
    assert _plugin.clickhouse_profile(t, "f", "ZSTD(3)")["bytes"] > 0
    with pytest.raises(ValueError):
        _plugin.clickhouse_profile(t, "nope", "LZ4")
    with pytest.raises(ValueError):
        _plugin.clickhouse_profile(t, "a", "Gorilla, LZ4")


def _suite(path, name):
    import yaml

    return yaml.safe_load((path / f"{name}.yaml").read_text())


def _arrow_type(path, name, col):
    with pa.ipc.open_file(path / f"{name}.arrow") as f:
        return f.get_batch(0).schema.field(col).type


def test_to_chbench_bool_over_integers_uses_boolean_dataset(tmp_path):
    import json

    from turbo_parakeet.recommenders import to_chbench

    frame = pl.DataFrame({"flag": pl.Series([0, 1] * 5_000, dtype=pl.Int32)})
    result = OneShotRecommender().add(frame).result()
    to_chbench(result, frame, tmp_path)
    assert _suite(tmp_path, "flag")["clickhouse"]["type"] == ["Bool"]
    assert _suite(tmp_path, "flag_original")["clickhouse"]["type"] == ["Int32"]
    assert _arrow_type(tmp_path, "flag", "flag") == pa.bool_()
    assert _arrow_type(tmp_path, "flag_original", "flag") == pa.int32()
    assert all(
        k.startswith("Bool|")
        for k in json.loads((tmp_path / "flag.predicted.json").read_text())
    )


def test_to_chbench_recommended_dataset_carries_recommended_type(tmp_path):
    from datetime import date, datetime, timedelta
    from decimal import Decimal

    from turbo_parakeet.recommenders import to_chbench

    n = 6_000
    frame = pl.DataFrame(
        {
            "dec": pl.Series(
                [Decimal(i % 500) / 100 for i in range(n)], dtype=pl.Decimal(10, 4)
            ),
            "ts": pl.Series(
                [
                    datetime(2024, 1, 1 + i % 20, 1, 2, 3, (i % 900) * 1000)
                    for i in range(n)
                ],
                dtype=pl.Datetime("ns"),
            ),
            "small": pl.Series([i % 200 for i in range(n)], dtype=pl.Int64),
            "cat": pl.Series([f"v{i % 40}" for i in range(n)]),
            "dates": pl.Series(
                [(date(2000, 1, 1) + timedelta(i)).isoformat() for i in range(n)]
            ),
        }
    )
    result = OneShotRecommender().add(frame).result()
    skipped = dict(to_chbench(result, frame, tmp_path))
    rec = {r["column"]: r for r in result.iter_rows(named=True)}
    assert rec["dec"]["datafusion_type"] == "decimal32(3, 2)"
    assert _arrow_type(tmp_path, "dec", "dec") == pa.decimal32(3, 2)
    assert _suite(tmp_path, "dec")["dataset"]["decimal_places"] == 2
    assert _arrow_type(tmp_path, "dec_original", "dec") == pa.decimal128(10, 4)
    assert _suite(tmp_path, "dec_original")["dataset"]["decimal_places"] == 4
    assert rec["ts"]["datafusion_type"] == "timestamp[ms]"
    assert _arrow_type(tmp_path, "ts", "ts") == pa.timestamp("ms")
    assert _arrow_type(tmp_path, "ts_original", "ts") == pa.timestamp("ns")
    assert rec["small"]["datafusion_type"] == "uint8"
    assert _arrow_type(tmp_path, "small", "small") == pa.uint8()
    assert _arrow_type(tmp_path, "small_original", "small") == pa.int64()
    assert "LowCardinality" in rec["cat"]["clickhouse_type"]
    assert rec["cat"]["datafusion_type"].startswith("dictionary<")
    assert _arrow_type(tmp_path, "cat", "cat") == pa.string()
    assert _suite(tmp_path, "cat")["clickhouse"]["codec"][0] == "LowCardinality"
    assert not (tmp_path / "cat_original.yaml").exists()
    # pyarrow parses ISO text to a date, so the text -> date recommendation is reproducible.
    assert rec["dates"]["datafusion_type"] == "date32[day]"
    assert "dates" not in skipped
    assert _arrow_type(tmp_path, "dates", "dates") == pa.date32()
    assert _arrow_type(tmp_path, "dates_original", "dates") == pa.string()


def test_to_chbench_skips_unreproducible_recommended_type(tmp_path, monkeypatch):
    from turbo_parakeet.recommenders import chbench, to_chbench

    frame = pl.DataFrame({"small": pl.Series([i % 200 for i in range(6_000)])})
    result = OneShotRecommender().add(frame).result()
    monkeypatch.setattr(chbench, "_arrow_type", lambda name: pa.date32())
    skipped = to_chbench(result, frame, tmp_path)
    assert len(skipped) == 1
    assert skipped[0][1].startswith(
        "recommended type not reproducible by a lossless pyarrow cast: "
    )
    assert not (tmp_path / "small.yaml").exists()
    assert (tmp_path / "small_original.yaml").exists()


def test_to_chbench_skips_a_row_without_a_sample(tmp_path):
    from turbo_parakeet.recommenders import to_chbench

    frame = pl.DataFrame({"a": [i % 7 for i in range(2_000)]})
    rec = StreamingRecommender(reservoir_rows=0)
    result = rec.add(frame).result()
    assert result["clickhouse_codecs"].null_count() == 1
    assert to_chbench(result, frame, tmp_path) == [("a", "no codec chosen (no sample)")]
    assert not list(tmp_path.glob("*.yaml"))


def test_clickhouse_profile_delta_chains():
    t = pa.table(
        {"a": pa.array(range(1000, 101_000, 7), pa.int64()), "f": [0.5] * 14286}
    )
    for codecs in (
        "Delta",
        "DoubleDelta",
        "Delta, LZ4",
        "DoubleDelta, ZSTD(1)",
        "NONE",
    ):
        assert _plugin.clickhouse_profile(t, "a", codecs)["bytes"] > 0
    assert _plugin.clickhouse_profile(t, "f", "Delta")["bytes"] is None
    dd = _plugin.clickhouse_profile(t, "a", "DoubleDelta")["bytes"]
    assert dd < _plugin.clickhouse_profile(t, "a", "NONE")["bytes"] / 10


def test_to_chbench_delta_transform(tmp_path):
    import json

    import yaml

    from turbo_parakeet.recommenders import to_chbench

    n = 200_000
    ts = pl.Series(range(n), dtype=pl.Int64) * 1_000 + 1_700_000_000_123
    frame = pl.DataFrame({"ts": ts.cast(pl.Datetime("ms"))})
    result = OneShotRecommender().add(frame).result()
    assert result["clickhouse_codecs"][0].startswith("DoubleDelta")
    to_chbench(result, frame, tmp_path)
    suite = yaml.safe_load((tmp_path / "ts.yaml").read_text())
    assert suite["clickhouse"]["codec"] == ["DoubleDelta", "none"]
    predicted = json.loads((tmp_path / "ts.predicted.json").read_text())
    comp = result["clickhouse_codecs"][0].split(", ")[-1]
    assert predicted[f"DateTime64|DoubleDelta|{comp}"] is not None
    assert f"DateTime64|none|{comp}" in predicted
