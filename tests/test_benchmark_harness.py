"""Pure parts of the benchmark harness (timing loop and speedup arithmetic). Nothing is benchmarked here."""

import time

from performance.harness import speedups, time_call


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
            "implementation": "XRust",
            "algorithmic": None,
            "parallel": 4.0,
            "total": None,
        }
    ]
