"""Seeded synthetic columnar datasets with targeted, measured statistics.

`generate(spec)` runs the single-column pipeline (`synthdata.column`) and returns a
`Dataset` (an Arrow array plus its `Manifest`); `save` and `load_or_generate` are its
Arrow IPC cache keyed by `spec.dataset_id`.
"""

from synthdata.cache import load_or_generate, load_path, save
from synthdata.column.spec import DatasetSpec, Distribution, SpecError, Tolerances
from synthdata.generate import Dataset, Manifest, SolveRecord, generate
from synthdata.cover import cover, cover_small, cover_windows, spread

__all__ = [
    "Dataset",
    "DatasetSpec",
    "Distribution",
    "Manifest",
    "SolveRecord",
    "SpecError",
    "Tolerances",
    "cover",
    "cover_small",
    "cover_windows",
    "generate",
    "load_or_generate",
    "load_path",
    "save",
    "spread",
]
