"""The benchmark harness: pure parts (timing loop, speedup arithmetic) and, on a fake
technique, one end-to-end run (child processes, memory, accuracy, variants, failures).
"""

import time

import polars as pl
import pytest

from performance import harness
from performance.harness import Dataset, Variant, memory_mb, speedups, time_call


def test_time_call_warms_up_then_times_every_run():
    calls = []
    times, result = time_call(
        lambda: calls.append(1) or len(calls), runs=3, budget_s=10.0
    )
    assert len(times) == 3
    assert len(calls) == 4  # 1 warm-up + 3 timed
    assert result == 4


def test_time_call_skips_when_warmup_exceeds_budget():
    times, result = time_call(lambda: time.sleep(0.05), runs=3, budget_s=0.01)
    assert times is None
    assert result is None


def _row(impl, threads, median, status="ok"):
    return {
        "dataset": "d",
        "implementation": impl,
        "threads": threads,
        "status": status,
        "median_ms": median,
    }


def test_speedups_separate_algorithm_from_parallelism():
    rows = [
        _row("XRust", "N", 10.0),
        _row("XRust", "1", 40.0),
        _row("XNumpy", "default", 200.0),
        _row("XMath", "default", 400.0),
    ]
    assert speedups(rows) == [
        {
            "dataset": "d",
            "variant": None,
            "implementation": "XRust",
            "algorithmic": 5.0,
            "parallel": 4.0,
            "total": 20.0,
        }
    ]


def test_speedups_without_a_finished_competitor():
    rows = [
        _row("XRust", "N", 10.0),
        _row("XRust", "1", 40.0),
        _row("XMath", "default", None, status="skipped: over budget (60s)"),
    ]
    assert speedups(rows) == [
        {
            "dataset": "d",
            "variant": None,
            "implementation": "XRust",
            "algorithmic": None,
            "parallel": 4.0,
            "total": None,
        }
    ]


def test_time_call_reports_after_the_warmup():
    seen = []
    time_call(
        lambda: seen.append("run"),
        runs=2,
        budget_s=10.0,
        on_warmup=lambda: seen.append("w"),
    )
    assert seen == ["run", "w", "run", "run"]


def test_memory_mb_current_and_peak():
    current, peak = memory_mb()
    assert 0 < current <= peak


def test_speedups_per_variant_share_the_exact_rows_of_their_group():
    rows = [
        {**_row("XPolars", "default", 100.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "N", 10.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "1", 20.0), "variant": "lo", "group": "lo"},
        {**_row("XSketchRust", "N", 25.0), "variant": "hi", "group": "lo"},
        {**_row("XSketchRust", "1", 50.0), "variant": "hi", "group": "lo"},
    ]
    got = {s["variant"]: s for s in speedups(rows)}
    assert got["lo"]["total"] == 10.0
    assert got["hi"]["total"] == 4.0
    assert got["hi"]["algorithmic"] == 2.0


PKG = "performance.fake_technique"


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness, "RESULTS_DIR", tmp_path_factory.mktemp("results"))
        return harness.run(
            PKG,
            [
                Dataset(
                    "tiny",
                    lambda: {
                        "t": pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
                    },
                )
            ],
            impl_params={"FakeSumRust": {"ballast_mb": 64}},
            variants=[
                Variant("lo", estimator_params={"noise": 0.1}),
                Variant("hi", estimator_params={"noise": 0.2}),
            ],
            runs=1,
        )


def _one(bench, implementation, variant, threads) -> dict:
    rows = bench.filter(
        implementation=implementation, variant=variant, threads=threads
    ).to_dicts()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.slow
def test_one_row_per_measurement(bench):
    assert sorted(
        bench.select("implementation", "variant", "threads").rows()
    ) == sorted(
        [
            ("FakeSumPolars", "lo", "default"),
            ("FakeSumBroken", "lo", "default"),
            ("FakeSumRust", "lo", "N"),
            ("FakeSumRust", "lo", "1"),
            ("FakeSumSketchRust", "lo", "N"),
            ("FakeSumSketchRust", "lo", "1"),
            ("FakeSumSketchRust", "hi", "N"),
            ("FakeSumSketchRust", "hi", "1"),
        ]
    )


@pytest.mark.slow
def test_a_failing_child_is_an_error_row(bench):
    assert _one(bench, "FakeSumBroken", "lo", "default")["status"] == "error"
    assert set(bench.filter(pl.col("implementation") != "FakeSumBroken")["status"]) == {
        "ok"
    }


@pytest.mark.slow
def test_accuracy_against_the_reference_and_the_exact_sibling(bench):
    exact = _one(bench, "FakeSumRust", "lo", "N")
    assert exact["agrees"] is True
    assert exact["max_rel_err"] == 0.0
    assert exact["exact_rust_agrees"] is None
    for variant, noise in (("lo", 0.1), ("hi", 0.2)):
        sketch = _one(bench, "FakeSumSketchRust", variant, "N")
        assert sketch["agrees"] is False
        assert sketch["max_rel_err"] == pytest.approx(noise)
        assert sketch["exact_rust_max_rel_err"] == pytest.approx(noise)
        assert sketch["speedup_vs_exact_rust"] > 0
    assert _one(bench, "FakeSumPolars", "lo", "default")["agrees"] is None


@pytest.mark.slow
def test_memory_and_speedups(bench):
    exact = _one(bench, "FakeSumRust", "lo", "N")
    assert exact["peak_mb"] >= 60  # the 64 MiB ballast
    assert exact["load_mb"] > 0
    assert exact["algorithmic"] is not None and exact["parallel"] is not None
    assert _one(bench, "FakeSumRust", "lo", "1")["algorithmic"] is None
    assert bench["technique"].unique().to_list() == ["fake_technique"]
