"""Shared by tests/test_describe.py and tests/test_recommend.py: `pytest.approx` over a column
that may hold nulls. pytest 9.2 raises a TypeError for a None expected value once a tolerance
is set, so the nulls are compared exactly (same places) and the other values with `approx`.
"""

import pytest


def assert_close(got: list, want: list, msg: str = "", **tolerance) -> None:
    assert [g is None for g in got] == [w is None for w in want], msg
    assert [g for g in got if g is not None] == pytest.approx(
        [w for w in want if w is not None], **tolerance
    ), msg
