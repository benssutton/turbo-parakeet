"""A FrameManifest for any frame (scenarios are explicit constructions)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from synthdata.synth.classify import base_values_of, family_of
from synthdata.frame.generate import FrameManifest
from synthdata.frame.measure import column_stats


def measure_frame(frame: pl.DataFrame, seed: int = 0) -> FrameManifest:
    """The measured statistics of every supported column of `frame`."""
    columns = {}
    for name in frame.columns:
        if family_of(frame.schema[name]) is None:
            continue
        values, integer = base_values_of(frame[name])
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        columns[name] = {
            "type": str(frame.schema[name]),
            "spec": None,
            **column_stats(values, frame.height, integer),
        }
    return FrameManifest("", frame.height, seed, np.__version__, columns)
