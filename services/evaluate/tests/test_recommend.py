import json

import numpy as np
import polars as pl

from evaluate.recommend import recommend


def _frame(tmp_path):
    p = tmp_path / "frame.arrow"
    pl.DataFrame({"a": np.arange(20_000), "b": np.arange(20_000) % 7}).write_ipc(p)
    return p


def test_child_writes_result_and_cost(tmp_path):
    cost = recommend(_frame(tmp_path), tmp_path)
    assert cost["status"] == "ok", cost
    assert cost["rows"] == 20_000 and cost["cols"] == 2
    assert cost["wall_s"] > 0 and cost["cpu_s"] > 0
    assert cost["peak_mb"] is not None and cost["peak_mb"] >= 0
    assert cost["frame_mb"] > 0
    result = pl.read_ipc(tmp_path / "result.arrow")
    assert result["column"].to_list() == ["a", "b"]
    assert json.loads((tmp_path / "cost.json").read_text()) == cost


def test_a_crashing_child_is_failed(tmp_path):
    bad = tmp_path / "frame.arrow"
    bad.write_bytes(b"not arrow")
    cost = recommend(bad, tmp_path)
    assert cost["status"] == "failed"
    assert cost["exit_code"] != 0 and cost["error"]
    assert (tmp_path / "cost.json").is_file()
    assert not (tmp_path / "result.arrow").exists()


def test_the_streaming_child_writes_a_result(tmp_path):
    frame = pl.DataFrame({"a": list(range(200_000)), "b": ["x", "y"] * 100_000})
    frame_path = tmp_path / "frame.arrow"
    frame.write_ipc(frame_path)
    cost = recommend(frame_path, tmp_path, kind="streaming")
    assert cost["status"] == "ok" and cost["recommender"] == "streaming"
    assert set(pl.read_ipc(tmp_path / "result.arrow")["column"]) == {"a", "b"}
