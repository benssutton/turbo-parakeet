"""
Shared accuracy-test helpers. Every technique's test file uses them the same way:

    PKG = "analytics.<technique>"
    ALL = implementation_params(PKG)                            # contract block
    OTHERS = implementation_params(PKG, include_reference=False)  # agreement block

    cls = load(impl)
    assert_contract(cls, run(cls, frames), frames)
    assert_agrees(cls(), run(cls, frames), run(reference(PKG), frames))
"""

from __future__ import annotations

import importlib

import polars as pl
import pytest

from analytics.base import STATUS, Technique


def implementation_params(package: str, include_reference: bool = True) -> list:
    module = importlib.import_module(package)
    return [
        pytest.param(f"{package}:{name}", id=name)
        for name in module.IMPLEMENTATIONS
        if include_reference or name != module.REFERENCE
    ]


def load(spec: str) -> type[Technique]:
    """'package:ClassName' -> class; skips the test (visibly) if its library is missing."""
    package, name = spec.split(":")
    try:
        return getattr(importlib.import_module(package), name)
    except ImportError as exc:
        pytest.skip(f"{name} unavailable: {exc}")


def reference(package: str) -> type[Technique]:
    return load(f"{package}:{importlib.import_module(package).REFERENCE}")


def run(cls: type[Technique], frames: dict, **params) -> pl.DataFrame:
    return cls(**params).add(frames).result()


def assert_contract(cls: type[Technique], result: pl.DataFrame, frames: dict) -> None:
    keys = cls.key_columns()
    expected = {
        **{k: pl.String for k in keys},
        "status": STATUS,
        **cls.DESCRIPTORS,
        **cls.METRICS,
        **cls.CONCLUSIONS,
    }
    assert list(result.schema.items()) == list(expected.items())
    assert result.to_arrow().num_rows == result.height, "result must export to native Arrow"
    collected = {n: f.collect() if isinstance(f, pl.LazyFrame) else f for n, f in frames.items()}
    assert result.select(keys).equals(cls.keys_frame(cls.enumerate(collected))), "rows must be every combination, in canonical order"
    idle = result.filter(pl.col("status") != "computed")
    for col in [*cls.METRICS, *cls.CONCLUSIONS]:
        assert idle[col].null_count() == idle.height, f"{col} must be null where status != computed"
    for col in cls.DESCRIPTORS:
        assert result[col].null_count() == 0, f"descriptor {col} must be filled on every row"


def assert_agrees(impl: Technique, result: pl.DataFrame, reference_result: pl.DataFrame) -> None:
    problems = impl.agreement(result, reference_result)
    assert not problems, f"{type(impl).__name__} disagrees with the reference:\n" + "\n".join(problems[:20])


def with_metrics(base: type[Technique], **metrics: list) -> type[Technique]:
    """Subclass of a technique base whose _compute returns `metrics` verbatim, so
    conclusion logic can be tested once, independently of any implementation."""

    class Fixed(base):
        def _compute(self, frames, combos):
            return self.metrics_frame(combos, metrics)

    Fixed.__name__ = f"Fixed{base.__name__}"
    return Fixed
