"""Runs test over k categories (per-column, order-dependent). See RunsTest for semantics."""

from turbo_parakeet.techniques.runs_test.base import RunsTest
from turbo_parakeet.techniques.runs_test.polars import RunsTestPolars
from turbo_parakeet.techniques.runs_test.rust import RunsTestRust

REFERENCE = "RunsTestPolars"
IMPLEMENTATIONS = ("RunsTestRust", "RunsTestPolars")

__all__ = ["RunsTest", "RunsTestRust", "RunsTestPolars", "REFERENCE", "IMPLEMENTATIONS"]
