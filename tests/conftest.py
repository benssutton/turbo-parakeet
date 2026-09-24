import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analytics"))

import polars as pl
import pytest

from datagen import mixed_dtypes


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow — deselect with '-m \"not slow\"'",
    )


N_ROWS: int = 1_000  # Change here to scale the shared test dataset


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """18-column DataFrame shared across all test modules in the session."""
    return mixed_dtypes(N_ROWS)
