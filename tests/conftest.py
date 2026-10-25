import polars as pl
import pytest

from synthdata.cover import cover_small, spread


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow — deselect with '-m \"not slow\"'",
    )


# Load the covering frame once per process, at collection, so no test pays for it.
cover_small()

N_ROWS: int = 1_000  # Change here to scale the shared test dataset


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """18 covering-frame columns shared across all test modules in the session."""
    return spread(cover_small(), 18).head(N_ROWS)
