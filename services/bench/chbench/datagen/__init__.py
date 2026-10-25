"""Synthetic single-column dataset generator.

Public API: `generate(spec)` runs the pipeline and returns a `Dataset` (an
Arrow array plus its `Manifest`); `save` and `load_or_generate` are the Arrow
IPC cache keyed by `spec.dataset_id`.
"""

from chbench.datagen.cache import Dataset, Manifest, generate, load_or_generate, save
from chbench.datagen.spec import DatasetSpec, Distribution, SpecError, Tolerances

__all__ = [
    "Dataset",
    "Manifest",
    "generate",
    "load_or_generate",
    "save",
    "DatasetSpec",
    "Distribution",
    "SpecError",
    "Tolerances",
]
