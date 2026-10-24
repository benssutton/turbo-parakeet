"""The benchmark harness: pure parts (timing loop, speedup arithmetic) and, on a fake
technique, one end-to-end run (child processes, memory, accuracy, variants, failures).
"""

import time

import polars as pl
import pytest

from performance import harness
from performance.harness import (
    Dataset,
    Variant,
    memory_mb,
    peak_extra,
    speedups,
    time_call,
)


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


def test_speedups_estimator_variants_do_not_cross():
    rows = [
        {**_row("XSketchPolars", "default", 5.0), "variant": "tiny", "group": "g"},
        {**_row("XSketchPolars", "default", 90.0), "variant": "big", "group": "g"},
        {**_row("XSketchRust", "N", 10.0), "variant": "big", "group": "g"},
        {**_row("XSketchRust", "1", 20.0), "variant": "big", "group": "g"},
    ]
    got = speedups(rows, {"XSketchPolars"})
    assert got[0]["total"] == 9.0  # not 0.5: the tiny variant's time is no competitor
    assert got[0]["algorithmic"] == 4.5
    assert (
        speedups(rows)[0]["total"] == 0.5
    )  # without the estimator set: the old minimum


def test_peak_extra_is_an_upper_bound_when_the_peak_did_not_move():
    assert peak_extra(100.0, 150.0, 150.0) == (50.0, True)
    assert peak_extra(100.0, 150.0, 150.5) == (50.5, True)  # within the tolerance
    assert peak_extra(100.0, 150.0, 220.0) == (120.0, False)  # a real measurement


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
                    truth=lambda: {("t", "a"): 6.0, ("t", "b"): 60.0},
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
            ("FakeSumSilent", "lo", "default"),
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
    failing = ["FakeSumBroken", "FakeSumSilent"]
    for name in failing:
        assert _one(bench, name, "lo", "default")["status"] == "error"
    assert set(bench.filter(~pl.col("implementation").is_in(failing))["status"]) == {
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
    assert exact["peak_mb"] >= 50  # the 64 MiB ballast
    assert exact["load_mb"] > 0
    assert exact["algorithmic"] is not None
    assert exact["parallel"] is not None
    assert _one(bench, "FakeSumRust", "lo", "1")["algorithmic"] is None
    assert bench["technique"].unique().to_list() == ["fake_technique"]


@pytest.mark.slow
def test_fixed_columns_have_their_dtypes(bench):
    for column, dtype in harness.FIXED_DTYPES.items():
        assert bench.schema[column] == dtype, column


def test_report_without_an_estimator_keeps_typed_fixed_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    rows = [
        {
            "dataset": "d",
            "implementation": "XPolars",
            "variant": "default",
            "threads": "default",
            "status": "error",
        }
    ]
    frame = harness._report("x", "pkg.x", rows)
    assert list(frame.columns[: len(harness.COLUMNS)]) == harness.COLUMNS
    for column, dtype in harness.FIXED_DTYPES.items():
        assert frame.schema[column] == dtype, column


def test_report_third_table_shows_truth_figures_of_exact_rows(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    base = {"dataset": "d", "variant": "default", "threads": "N", "status": "ok"}
    rows = [
        {**base, "implementation": "XPlainRust", "agrees": True, "max_rel_err": 0.0},
        {
            **base,
            "implementation": "XTruthRust",
            "agrees": True,
            "max_rel_err": 0.0,
            "zz_truth_rel_err": 0.123,
        },
    ]
    harness._report("x", "pkg.x", rows)
    third = capsys.readouterr().out.split("zz_truth_rel_err")
    assert len(third) == 2  # the column appears in the third table only
    assert "XTruthRust" in third[1]
    assert "XPlainRust" not in third[1]


def test_duplicate_dataset_names_are_rejected():
    ds = Dataset("same", lambda: {})
    with pytest.raises(ValueError, match="unique"):
        harness.run(PKG, [ds, ds])


@pytest.mark.slow
def test_truth_reaches_the_accuracy_hook(bench):
    assert _one(bench, "FakeSumRust", "lo", "N")["truth_rel_err"] == 0.0
    assert _one(bench, "FakeSumSketchRust", "hi", "N")[
        "truth_rel_err"
    ] == pytest.approx(0.2)
    # truth goes to the reference comparison only
    assert "exact_rust_truth_rel_err" not in bench.columns
