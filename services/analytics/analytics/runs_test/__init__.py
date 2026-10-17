"""Runs test over k categories (per-column, order-dependent). See RunsTest for semantics."""

from analytics.runs_test.base import RunsTest
from analytics.runs_test.polars import RunsTestPolars
from analytics.runs_test.rust import RunsTestRust

REFERENCE = "RunsTestPolars"
IMPLEMENTATIONS = ("RunsTestRust", "RunsTestPolars")

__all__ = ["RunsTest", "RunsTestRust", "RunsTestPolars", "REFERENCE", "IMPLEMENTATIONS"]
