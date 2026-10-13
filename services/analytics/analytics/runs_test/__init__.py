"""Runs test over k categories (per-column, order-dependent). See RunsTest for semantics."""

from analytics.runs_test.base import RunsTest
from analytics.runs_test.polars import RunsTestPolars

REFERENCE = "RunsTestPolars"
IMPLEMENTATIONS = ("RunsTestPolars",)

__all__ = ["RunsTest", "RunsTestPolars", "REFERENCE", "IMPLEMENTATIONS"]
