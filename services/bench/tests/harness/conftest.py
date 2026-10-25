"""Shared fixtures for the harness tests.

`ch` is session-scoped: a later task's module-scoped fixture depends on it,
and pytest rejects a broader fixture depending on a narrower one. It skips,
rather than fails, whenever the container is not reachable so the fast suite
stays green with ClickHouse stopped.
"""

from __future__ import annotations

import pytest

from chbench.harness.client import CHClient, Connection


@pytest.fixture(scope="session")
def ch():
    try:
        client = CHClient(Connection.from_env())
        client.version()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ClickHouse not reachable: {type(e).__name__}: {e}")
    return client
